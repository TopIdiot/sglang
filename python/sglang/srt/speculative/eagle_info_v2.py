from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.utils import get_alloc_len_per_decode
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    generate_simulated_accept_index,
)
from sglang.srt.speculative.welmv4_mtp_sampling import (
    welm_mtp_batch_base_positions,
    welm_mtp_deterministic_uniforms,
    welm_mtp_sample_from_weights_with_uniform,
    welmv4_mtp_apply_root_only_sampling,
    welmv4_mtp_fused_topk_softmax_sample,
    welmv4_mtp_fused_verify_top1_dense_target,
    welmv4_mtp_fused_verify_top1_sparse,
)
from sglang.srt.utils.common import is_cuda, is_hip, is_musa, is_npu, next_power_of_2

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

if is_cuda() or is_musa():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )


_WELM_MTP_VERIFY_SAMPLING_TOPK_ENV = "SGLANG_WELM_MTP_VERIFY_SAMPLING_TOPK"
_WELM_MTP_VERIFY_FUSED_REJECT_ENV = "SGLANG_WELM_MTP_VERIFY_FUSED_REJECT"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _welm_mtp_verify_sampling_topk() -> int:
    value = os.environ.get(_WELM_MTP_VERIFY_SAMPLING_TOPK_ENV, "0")
    try:
        return max(0, int(value))
    except ValueError:
        return 0


def _welm_mtp_verify_fused_reject_enabled() -> bool:
    return _env_flag(_WELM_MTP_VERIFY_FUSED_REJECT_ENV, "1")


def _welm_mtp_root_only_random_uniforms(
    root_only_mask: torch.Tensor, width: int
) -> torch.Tensor:
    ordinary_rows = torch.nonzero(~root_only_mask, as_tuple=False).flatten()
    root_rows = torch.nonzero(root_only_mask, as_tuple=False).flatten()
    uniforms = torch.zeros(
        (root_only_mask.numel(), width),
        dtype=torch.float32,
        device=root_only_mask.device,
    )
    if ordinary_rows.numel() > 0:
        uniforms.index_copy_(
            0,
            ordinary_rows,
            torch.rand(
                (ordinary_rows.numel(), width),
                dtype=torch.float32,
                device=root_only_mask.device,
            ),
        )
    uniforms[root_rows, -1] = torch.rand(
        (root_rows.numel(),), dtype=torch.float32, device=root_only_mask.device
    )
    return uniforms


def verify_top1_chain_with_draft_probs_v2(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor | None = None,
    draft_topk_indices: torch.Tensor | None = None,
    draft_topk_values: torch.Tensor | None = None,
    uniforms: torch.Tensor | None = None,
) -> None:
    bs = candidates.shape[0]
    spec_steps = accept_index.shape[1]
    eps = torch.finfo(target_probs.dtype).tiny

    def get_q(b: int, local_idx: int, token_id: int) -> torch.Tensor:
        if draft_probs is not None:
            return draft_probs[b, local_idx, token_id]
        indices = draft_topk_indices[b, local_idx]
        values = draft_topk_values[b, local_idx]
        return torch.where(indices == token_id, values, torch.zeros_like(values)).sum()

    def get_residual_probs(b: int, local_idx: int) -> torch.Tensor:
        if draft_probs is not None:
            return torch.clamp(
                target_probs[b, local_idx] - draft_probs[b, local_idx],
                min=0,
            )
        residual_probs = target_probs[b, local_idx].clone()
        residual_probs[draft_topk_indices[b, local_idx]] -= draft_topk_values[
            b, local_idx
        ]
        return torch.clamp(residual_probs, min=0)

    for b in range(bs):
        local_parent_idx = 0
        parent_predict_idx = int(retrieve_index[b, local_parent_idx].item())
        accept_index[b, 0] = parent_predict_idx
        accepted = 0

        for step in range(1, spec_steps):
            next_local_idx = int(retrieve_next_token[b, local_parent_idx].item())
            if next_local_idx < 0:
                break

            draft_token_id = int(candidates[b, next_local_idx].item())
            p = target_probs[b, local_parent_idx, draft_token_id]
            q = get_q(b, local_parent_idx, draft_token_id)
            accept_prob = torch.clamp(p / torch.clamp(q, min=eps), max=1.0)
            accept_coin = (
                uniforms[b, step - 1]
                if uniforms is not None
                else torch.rand((), dtype=torch.float32, device=target_probs.device)
            )
            if (accept_coin <= accept_prob).item():
                predicts[parent_predict_idx] = draft_token_id
                accepted += 1
                child_predict_idx = int(retrieve_index[b, next_local_idx].item())
                accept_index[b, accepted] = child_predict_idx
                local_parent_idx = next_local_idx
                parent_predict_idx = child_predict_idx
            else:
                residual_probs = get_residual_probs(b, local_parent_idx)
                if (residual_probs.sum() <= 0).item():
                    residual_probs = target_probs[b, local_parent_idx]
                predicts[parent_predict_idx] = (
                    welm_mtp_sample_from_weights_with_uniform(
                        residual_probs, uniforms[b, spec_steps - 1]
                    ).to(dtype=torch.int32)
                    if uniforms is not None
                    else torch.multinomial(residual_probs, num_samples=1)
                    .to(dtype=torch.int32)
                    .squeeze(0)
                )
                break
        else:
            predicts[parent_predict_idx] = (
                welm_mtp_sample_from_weights_with_uniform(
                    target_probs[b, local_parent_idx], uniforms[b, spec_steps - 1]
                ).to(dtype=torch.int32)
                if uniforms is not None
                else torch.multinomial(
                    target_probs[b, local_parent_idx], num_samples=1
                )
                .to(dtype=torch.int32)
                .squeeze(0)
            )

        accept_token_num[b] = accepted


def verify_top1_chain_with_sparse_target_probs_v2_vectorized(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    target_topk_indices: torch.Tensor,
    target_topk_values: torch.Tensor,
    draft_topk_indices: torch.Tensor,
    draft_topk_values: torch.Tensor,
    uniforms: torch.Tensor | None = None,
) -> None:
    bs, draft_token_num = candidates.shape
    num_draft_steps = min(accept_index.shape[1] - 1, draft_token_num - 1)
    device = candidates.device

    def gather_2d(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return values.gather(1, index.view(bs, 1)).squeeze(1)

    def gather_3d(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return values.gather(
            1, index.view(bs, 1, 1).expand(-1, 1, values.shape[-1])
        ).squeeze(1)

    chain_nodes = [torch.zeros((bs,), dtype=torch.long, device=device)]
    parent_nodes = []
    child_nodes = []
    draft_token_ids = []
    accept_probs = []

    for _ in range(num_draft_steps):
        parent = chain_nodes[-1]
        child = gather_2d(retrieve_next_token, parent).to(dtype=torch.long)
        valid_child = child >= 0
        safe_child = child.clamp(min=0)
        parent_nodes.append(parent)
        child_nodes.append(safe_child)
        chain_nodes.append(safe_child)

        token_id = gather_2d(candidates, safe_child).to(dtype=torch.long)
        draft_token_ids.append(token_id)

        target_indices = gather_3d(target_topk_indices, parent)
        target_values = gather_3d(target_topk_values, parent)
        draft_indices = gather_3d(draft_topk_indices, parent)
        draft_values = gather_3d(draft_topk_values, parent)

        p = torch.where(
            target_indices == token_id.view(bs, 1),
            target_values,
            torch.zeros_like(target_values),
        ).sum(dim=1)
        q = torch.where(
            draft_indices == token_id.view(bs, 1),
            draft_values,
            torch.zeros_like(draft_values),
        ).sum(dim=1)
        eps = torch.finfo(target_values.dtype).tiny
        accept_prob = torch.clamp(p / torch.clamp(q, min=eps), max=1.0)
        accept_probs.append(
            torch.where(valid_child, accept_prob, torch.zeros_like(accept_prob))
        )
        chain_nodes[-1] = torch.where(valid_child, safe_child, parent)

    if not accept_probs:
        accept_token_num.zero_()
        return

    accept_prob_tensor = torch.stack(accept_probs, dim=1)
    accept_coins = (
        uniforms[:, :num_draft_steps]
        if uniforms is not None
        else torch.rand_like(accept_prob_tensor, dtype=torch.float32)
    )
    accept_each = accept_coins <= accept_prob_tensor
    prefix_accept = torch.cumprod(accept_each.to(dtype=torch.int32), dim=1).bool()
    accepted_count = prefix_accept.to(dtype=torch.int32).sum(dim=1)

    accept_index[:, 0] = retrieve_index[:, 0].to(dtype=accept_index.dtype)
    neg_one = torch.full((bs,), -1, dtype=accept_index.dtype, device=device)
    for step, child in enumerate(child_nodes):
        child_predict_idx = gather_2d(retrieve_index, child).to(
            dtype=accept_index.dtype
        )
        accept_index[:, step + 1] = torch.where(
            prefix_accept[:, step], child_predict_idx, neg_one
        )

    predict_indices_to_write = []
    predict_values_to_write = []
    for step, parent in enumerate(parent_nodes):
        parent_predict_idx = gather_2d(retrieve_index, parent).to(dtype=torch.long)
        mask = prefix_accept[:, step]
        predict_indices_to_write.append(parent_predict_idx[mask])
        predict_values_to_write.append(
            draft_token_ids[step][mask].to(dtype=predicts.dtype)
        )

    chain_tensor = torch.stack(chain_nodes, dim=1)
    sample_parent = chain_tensor.gather(
        1, accepted_count.to(dtype=torch.long).view(bs, 1)
    ).squeeze(1)
    sample_predict_idx = gather_2d(retrieve_index, sample_parent).to(dtype=torch.long)

    target_indices = gather_3d(target_topk_indices, sample_parent)
    target_values = gather_3d(target_topk_values, sample_parent)
    draft_indices = gather_3d(draft_topk_indices, sample_parent)
    draft_values = gather_3d(draft_topk_values, sample_parent)
    q_on_target = (
        (target_indices.unsqueeze(2) == draft_indices.unsqueeze(1))
        .to(dtype=draft_values.dtype)
        .mul(draft_values.unsqueeze(1))
        .sum(dim=2)
    )
    residual_values = torch.clamp(target_values - q_on_target, min=0)
    residual_sum = residual_values.sum(dim=1, keepdim=True)
    use_residual = (accepted_count < num_draft_steps) & (residual_sum.squeeze(1) > 0)
    sample_values = torch.where(
        use_residual.view(bs, 1), residual_values, target_values
    )
    sample_total = torch.clamp(sample_values.sum(dim=1, keepdim=True), min=1.0e-30)
    sample_threshold = (
        uniforms[:, num_draft_steps : num_draft_steps + 1]
        if uniforms is not None
        else torch.rand((bs, 1), dtype=torch.float32, device=device)
    ) * sample_total
    cdf = torch.cumsum(sample_values, dim=1)
    sample_pos = torch.sum(cdf < sample_threshold, dim=1, keepdim=True).clamp(
        max=sample_values.shape[1] - 1
    )
    sampled_token = (
        target_indices.gather(1, sample_pos).squeeze(1).to(dtype=predicts.dtype)
    )
    predict_indices_to_write.append(sample_predict_idx)
    predict_values_to_write.append(sampled_token)

    write_indices = torch.cat(predict_indices_to_write, dim=0)
    write_values = torch.cat(predict_values_to_write, dim=0)
    predicts[write_indices] = write_values
    accept_token_num.copy_(accepted_count.to(dtype=accept_token_num.dtype))


@triton.jit
def assign_draft_cache_locs_page_size_1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # Copy from req_to_token to out_cache_loc
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)


@dataclass
class EagleDraftInputV2Mixin:
    def prepare_for_decode(
        self: EagleDraftInput,
        batch: ScheduleBatch,
        row_active_mask: Optional[torch.Tensor] = None,
    ):
        batch.maybe_evict_swa()

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # Now seq_lens is correct
        batch.maybe_wait_verify_done()

        # Accumulate penalty
        # This is a relaxed version of penalties for speculative decoding.
        if batch.sampling_info.penalizer_orchestrator.is_required:
            output_ids = torch.tensor(
                [
                    (
                        req.output_ids[-1]
                        if len(req.output_ids)
                        else req.origin_input_ids[-1]
                    )
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                output_ids, row_active_mask=row_active_mask
            )

        page_size = batch.token_to_kv_pool_allocator.page_size
        cur_kv_lens_cpu = []
        nxt_kv_lens_cpu = []
        num_needed_tokens = 0
        alloc_len_per_decode = get_alloc_len_per_decode()
        for r in batch.reqs:
            # Over-allocation happens here
            x = max(
                0,
                r.kv_committed_len + 2 * alloc_len_per_decode - r.kv_allocated_len,
            )
            cur_kv_lens_cpu.append(r.kv_allocated_len)
            nxt_kv_lens_cpu.append(r.kv_allocated_len + x)
            num_needed_tokens += x
            r.kv_allocated_len += x
            r.decode_batch_idx += 1
            # Pre-claim bonus slot here (like normal decode); resolve subtracts 1.
            r.kv_committed_len += 1

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens_cpu, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens_cpu, dtype=torch.int32, device="cpu")

        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
            if (get_global_server_args().speculative_eagle_topk or 1) > 1:
                out_cache_loc = out_cache_loc.clone()
        else:
            cur_kv_lens = cur_kv_lens_cpu.to(device=batch.device)
            nxt_kv_lens = nxt_kv_lens_cpu.to(device=batch.device)
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens,
            )
            out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens,
                cur_kv_lens_cpu,
                nxt_kv_lens,
                nxt_kv_lens_cpu,
                last_loc,
                num_needed_tokens,
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens_cpu.to(device=batch.device),
            nxt_kv_lens_cpu.to(device=batch.device),
            out_cache_loc,
            bs,
        )

        # FIXME(lsyin): make this sync optional
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

    def prepare_for_v2_draft(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        if not batch.forward_mode.is_idle():
            bs = len(batch.seq_lens)

            # Assign cache locations
            batch.out_cache_loc = torch.empty(
                (bs * topk * num_steps,),
                dtype=torch.int64,
                device=batch.input_ids.device,
            )
            # FIXME(lsyin): align with the default code path
            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices,
                req_to_token_pool.req_to_token,
                batch.seq_lens,
                batch.out_cache_loc,
                req_to_token_pool.req_to_token.shape[1],
                topk,
                num_steps,
            )

        # Get a forward batch
        self.num_tokens_per_req = topk
        self.num_tokens_for_logprob_per_req = topk
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        self.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph

    def prepare_for_extend_to_fill_draft_kvcache(
        self,
        batch: ModelWorkerBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        seq_lens_cpu_ = batch.seq_lens_cpu
        extend_num_tokens = len(batch.seq_lens) * num_draft_tokens

        batch.spec_info = self
        batch.input_ids = predict
        batch.seq_lens = batch.seq_lens + num_draft_tokens
        batch.seq_lens_cpu = batch.seq_lens_cpu + num_draft_tokens
        batch.seq_lens_sum += extend_num_tokens
        batch.extend_seq_lens = [num_draft_tokens for _ in range(len(batch.seq_lens))]
        batch.extend_prefix_lens = seq_lens_cpu_.tolist()
        batch.extend_num_tokens = extend_num_tokens
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DRAFT_EXTEND_V2
        )
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        return forward_batch


@dataclass
class EagleVerifyInputV2Mixin:
    def prepare_for_v2_verify(
        self: EagleVerifyInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        target_worker: TpModelWorker,
        prepare_attn_backend: bool = True,
    ):
        if not batch.forward_mode.is_idle():
            # Assign cache locations
            bs = len(batch.req_pool_indices)
            batch.input_ids = self.draft_token
            device = batch.input_ids.device
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                draft_token_num=self.draft_token_num,
                device=device,
            )
            # Set mamba_track_indices for mamba prefix-cache state tracking
            if get_global_server_args().enable_mamba_extra_buffer():
                mapping = (
                    req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping
                )
                req_pool_idx_tensor = batch.req_pool_indices.to(
                    device=mapping.device, dtype=torch.int64
                )
                track_col_idx = torch.tensor(
                    [req.mamba_next_track_idx for req in batch.reqs],
                    dtype=torch.int64,
                    pin_memory=True,
                ).to(mapping.device, non_blocking=True)
                batch.mamba_track_indices = mapping[
                    req_pool_idx_tensor, track_col_idx
                ].to(dtype=torch.int64)
                batch.mamba_track_mask = None
                batch.mamba_track_seqlens = None

        # Get a forward batch
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)
        cached_welm_oe_hash = getattr(self, "welm_mtp_oe_hashed_inputs", None)
        if cached_welm_oe_hash is not None:
            verify_forward_batch.welm_oe_decode_hashed_inputs = cached_welm_oe_hash

        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        if prepare_attn_backend:
            if can_run_cuda_graph:
                target_worker.model_runner.graph_runner.replay_prepare(
                    verify_forward_batch
                )
            else:
                if not batch.forward_mode.is_idle():
                    target_worker.model_runner.attn_backend.init_forward_metadata(
                        verify_forward_batch
                    )

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self: EagleVerifyInput,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: torch.Tensor = None,
    ):
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).
        """
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=batch.input_ids.device)
            num_correct_drafts = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            accept_index = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            return predict, num_correct_drafts, accept_index

        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        # Apply penalty
        # This is a relaxed version of penalties for speculative decoding.
        if sampling_info.acc_additive_penalties is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.acc_additive_penalties, self.draft_token_num, dim=0
                )
            )
        if sampling_info.acc_scaling_penalties is not None:
            apply_scaling_penalties(
                next_token_logits,
                torch.repeat_interleave(
                    sampling_info.acc_scaling_penalties, self.draft_token_num, dim=0
                ),
            )
        if sampling_info.logit_bias is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.logit_bias, self.draft_token_num, dim=0
                )
            )

        # Apply grammar mask if provided
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=next_token_logits, vocab_mask=vocab_mask
            )

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=device
        )
        num_correct_drafts = torch.empty((bs,), dtype=torch.int32, device=device)
        verify_base_positions = None
        verify_uniforms = None
        root_only_verify_mask = getattr(
            self, "welm_mtp_root_only_verify_mask", None
        )
        if root_only_verify_mask is not None:
            if (
                self.topk != 1
                or root_only_verify_mask.dtype != torch.bool
                or root_only_verify_mask.ndim != 1
                or root_only_verify_mask.numel() != bs
                or root_only_verify_mask.device != device
            ):
                raise RuntimeError(
                    "WeLM MTP root-only sampling requires a batch-aligned bool "
                    "mask and topk=1."
                )
        root_target_predict = None
        root_target_probs = None
        root_target_topk_indices = None
        root_target_topk_values = None
        root_sampling_uniforms = None
        root_sampled_tokens = None
        ordinary_rows = (
            None
            if root_only_verify_mask is None
            else torch.nonzero(~root_only_verify_mask, as_tuple=False).flatten()
        )
        root_rows = (
            None
            if root_only_verify_mask is None
            else torch.nonzero(root_only_verify_mask, as_tuple=False).flatten()
        )

        # Sample tokens
        if sampling_info.is_all_greedy or _is_npu or _is_hip:
            if ordinary_rows is None:
                target_predict = torch.argmax(next_token_logits, dim=-1).reshape(
                    bs, self.draft_token_num
                )
                root_target_predict = target_predict
                predict, accept_index, num_correct_drafts = verify_tree_greedy_func(
                    predicts=predict,  # mutable
                    accept_index=accept_index,  # mutable
                    accept_token_num=num_correct_drafts,  # mutable
                    candidates=candidates,
                    retrieve_index=self.retrieve_index,
                    retrieve_next_token=self.retrieve_next_token,
                    retrieve_next_sibling=self.retrieve_next_sibling,
                    target_predict=target_predict,
                    topk=self.topk,
                )
            else:
                token_offsets = torch.arange(
                    self.draft_token_num, device=device, dtype=torch.long
                )
                ordinary_flat_rows = (
                    ordinary_rows[:, None] * self.draft_token_num
                    + token_offsets[None, :]
                ).reshape(-1)
                ordinary_target_predict = torch.argmax(
                    next_token_logits[ordinary_flat_rows], dim=-1
                ).reshape(ordinary_rows.numel(), self.draft_token_num)
                root_target_predict = torch.argmax(
                    next_token_logits[root_rows * self.draft_token_num], dim=-1
                ).reshape(root_rows.numel(), 1)
                if ordinary_rows.numel() > 0:
                    ordinary_accept_index = accept_index[ordinary_rows].contiguous()
                    ordinary_num_correct_drafts = num_correct_drafts[
                        ordinary_rows
                    ].contiguous()
                    predict, ordinary_accept_index, ordinary_num_correct_drafts = (
                        verify_tree_greedy_func(
                            predicts=predict,
                            accept_index=ordinary_accept_index,
                            accept_token_num=ordinary_num_correct_drafts,
                            candidates=candidates[ordinary_rows],
                            retrieve_index=self.retrieve_index[ordinary_rows],
                            retrieve_next_token=self.retrieve_next_token[ordinary_rows],
                            retrieve_next_sibling=self.retrieve_next_sibling[
                                ordinary_rows
                            ],
                            target_predict=ordinary_target_predict,
                            topk=self.topk,
                        )
                    )
                    accept_index[ordinary_rows] = ordinary_accept_index
                    num_correct_drafts[ordinary_rows] = ordinary_num_correct_drafts
        else:
            if getattr(sampling_info, "sampling_seed", None) is not None:
                verify_base_positions = welm_mtp_batch_base_positions(
                    self.positions,
                    batch_size=bs,
                    draft_token_num=self.draft_token_num,
                )
                verify_uniforms = welm_mtp_deterministic_uniforms(
                    sampling_info=sampling_info,
                    positions=verify_base_positions,
                    batch_size=bs,
                    width=accept_index.shape[1],
                    salt=1000,
                )
            verify_sampling_topk = _welm_mtp_verify_sampling_topk()
            use_sparse_verify_topk = (
                verify_sampling_topk > 0
                and self.topk == 1
                and not getattr(sampling_info, "need_top_p_sampling", False)
                and self.draft_topk_indices is not None
                and self.draft_topk_values is not None
            )
            use_top1_draft_representation = self.topk == 1 and (
                self.draft_probs is not None or self.draft_topk_indices is not None
            )
            target_probs = None
            target_topk_indices = None
            target_topk_values = None
            if (
                root_only_verify_mask is not None
                and verify_uniforms is None
                and use_top1_draft_representation
            ):
                verify_uniforms = _welm_mtp_root_only_random_uniforms(
                    root_only_verify_mask, accept_index.shape[1]
                )

            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)
            target_logits = next_token_logits
            target_row_indices = None
            ordinary_target_row_count = bs * self.draft_token_num
            if ordinary_rows is not None:
                token_offsets = torch.arange(
                    self.draft_token_num, device=device, dtype=torch.long
                )
                ordinary_target_row_count = (
                    ordinary_rows.numel() * self.draft_token_num
                )
                ordinary_flat_rows = (
                    ordinary_rows[:, None] * self.draft_token_num
                    + token_offsets[None, :]
                ).reshape(-1)
                root_flat_rows = root_rows * self.draft_token_num
                target_row_indices = torch.cat(
                    (ordinary_flat_rows, root_flat_rows)
                )
                target_logits = next_token_logits[target_row_indices]
                expanded_temperature = expanded_temperature[target_row_indices]

            if use_sparse_verify_topk:
                k = min(int(verify_sampling_topk), int(target_logits.shape[-1]))
                target_topk = None
                if _is_cuda and not getattr(
                    sampling_info, "need_top_p_sampling", False
                ):
                    uniform = (
                        torch.full(
                            (target_logits.shape[0], 1),
                            0.5,
                            dtype=torch.float32,
                            device=next_token_logits.device,
                        )
                        if verify_uniforms is not None
                        else torch.empty(
                            (target_logits.shape[0], 1),
                            dtype=torch.float32,
                            device=next_token_logits.device,
                        )
                    )
                    if verify_uniforms is None:
                        uniform.uniform_()
                    try:
                        target_topk = welmv4_mtp_fused_topk_softmax_sample(
                            target_logits,
                            expanded_temperature,
                            uniform,
                            k,
                        )
                    except Exception as exc:
                        if root_only_verify_mask is not None:
                            raise RuntimeError(
                                "WeLM MTP root-only Target top-k fast path failed."
                            ) from exc
                        target_topk = None

                if target_topk is not None:
                    _, _, target_topk_indices, target_topk_values = target_topk
                else:
                    target_topk_logits, target_topk_indices = torch.topk(
                        target_logits / expanded_temperature,
                        k=k,
                        dim=-1,
                        sorted=False,
                    )
                    target_topk_values = F.softmax(target_topk_logits, dim=-1)
                if ordinary_rows is None:
                    target_topk_indices = target_topk_indices.reshape(
                        bs, self.draft_token_num, k
                    )
                    target_topk_values = target_topk_values.reshape(
                        bs, self.draft_token_num, k
                    )
                else:
                    root_target_topk_indices = target_topk_indices[
                        ordinary_target_row_count:
                    ].reshape(root_rows.numel(), 1, k)
                    root_target_topk_values = target_topk_values[
                        ordinary_target_row_count:
                    ].reshape(root_rows.numel(), 1, k)
                    target_topk_indices = target_topk_indices[
                        :ordinary_target_row_count
                    ].reshape(ordinary_rows.numel(), self.draft_token_num, k)
                    target_topk_values = target_topk_values[
                        :ordinary_target_row_count
                    ].reshape(ordinary_rows.numel(), self.draft_token_num, k)
                target_probs = None
            else:
                target_probs = F.softmax(
                    target_logits / expanded_temperature, dim=-1
                )  # (bs * num_draft_tokens, vocab_size)
                if getattr(sampling_info, "need_top_k_sampling", True):
                    expanded_top_ks = torch.repeat_interleave(
                        sampling_info.top_ks, self.draft_token_num, dim=0
                    )
                    if target_row_indices is not None:
                        expanded_top_ks = expanded_top_ks[target_row_indices]
                    target_probs = top_k_renorm_prob(
                        target_probs,
                        expanded_top_ks,
                    )  # (bs * num_draft_tokens, vocab_size)
                if getattr(sampling_info, "need_top_p_sampling", False):
                    expanded_top_ps = torch.repeat_interleave(
                        sampling_info.top_ps, self.draft_token_num, dim=0
                    )
                    if target_row_indices is not None:
                        expanded_top_ps = expanded_top_ps[target_row_indices]
                    target_probs = top_p_renorm_prob(
                        target_probs,
                        expanded_top_ps,
                    )
                if ordinary_rows is None:
                    target_probs = target_probs.reshape(
                        bs, self.draft_token_num, -1
                    )
                else:
                    vocab_size = target_probs.shape[-1]
                    root_target_probs = target_probs[
                        ordinary_target_row_count:
                    ].reshape(root_rows.numel(), 1, vocab_size)
                    target_probs = target_probs[:ordinary_target_row_count].reshape(
                        ordinary_rows.numel(), self.draft_token_num, vocab_size
                    )

            # Sync sampling results across TP ranks: different GPUs may
            # produce slightly different target_probs due to floating-point
            # non-determinism in softmax/top_k/top_p, causing different
            # sampled tokens. Broadcast from rank 0 to ensure consistency.
            tp_group = (
                get_attention_tp_group()
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if use_top1_draft_representation:
                draft_probs = (
                    self.draft_probs.to(dtype=torch.float32, device=device).contiguous()
                    if self.draft_probs is not None
                    else None
                )
                draft_topk_indices = (
                    self.draft_topk_indices.to(device=device).contiguous()
                    if self.draft_topk_indices is not None
                    else None
                )
                draft_topk_values = (
                    self.draft_topk_values.to(
                        dtype=torch.float32, device=device
                    ).contiguous()
                    if self.draft_topk_values is not None
                    else None
                )
                ordinary_accept_index = (
                    accept_index
                    if ordinary_rows is None
                    else accept_index[ordinary_rows].contiguous()
                )
                ordinary_num_correct_drafts = (
                    num_correct_drafts
                    if ordinary_rows is None
                    else num_correct_drafts[ordinary_rows].contiguous()
                )
                ordinary_candidates = (
                    candidates if ordinary_rows is None else candidates[ordinary_rows]
                )
                ordinary_retrieve_index = (
                    self.retrieve_index
                    if ordinary_rows is None
                    else self.retrieve_index[ordinary_rows]
                )
                ordinary_retrieve_next_token = (
                    self.retrieve_next_token
                    if ordinary_rows is None
                    else self.retrieve_next_token[ordinary_rows]
                )
                ordinary_target_probs = target_probs
                ordinary_target_topk_indices = target_topk_indices
                ordinary_target_topk_values = target_topk_values
                ordinary_draft_probs = (
                    draft_probs
                    if ordinary_rows is None or draft_probs is None
                    else draft_probs[ordinary_rows]
                )
                ordinary_draft_topk_indices = (
                    draft_topk_indices
                    if ordinary_rows is None or draft_topk_indices is None
                    else draft_topk_indices[ordinary_rows]
                )
                ordinary_draft_topk_values = (
                    draft_topk_values
                    if ordinary_rows is None or draft_topk_values is None
                    else draft_topk_values[ordinary_rows]
                )
                ordinary_uniforms = (
                    verify_uniforms
                    if ordinary_rows is None or verify_uniforms is None
                    else verify_uniforms[ordinary_rows]
                )
                run_ordinary_verify = (
                    ordinary_rows is None or ordinary_rows.numel() > 0
                )
                if use_sparse_verify_topk and run_ordinary_verify:
                    fused_verify_done = False
                    if _is_cuda and _welm_mtp_verify_fused_reject_enabled():
                        try:
                            fused_verify_done = welmv4_mtp_fused_verify_top1_sparse(
                                predicts=predict,
                                accept_index=ordinary_accept_index,
                                accept_token_num=ordinary_num_correct_drafts,
                                candidates=ordinary_candidates,
                                retrieve_index=ordinary_retrieve_index,
                                retrieve_next_token=ordinary_retrieve_next_token,
                                target_topk_indices=ordinary_target_topk_indices,
                                target_topk_values=ordinary_target_topk_values,
                                draft_topk_indices=ordinary_draft_topk_indices,
                                draft_topk_values=ordinary_draft_topk_values,
                                uniforms=ordinary_uniforms,
                            )
                        except Exception as exc:
                            if root_only_verify_mask is not None:
                                raise RuntimeError(
                                    "WeLM MTP root-only sparse verify fast path failed."
                                ) from exc
                            fused_verify_done = False
                    if not fused_verify_done:
                        verify_top1_chain_with_sparse_target_probs_v2_vectorized(
                            predicts=predict,
                            accept_index=ordinary_accept_index,
                            accept_token_num=ordinary_num_correct_drafts,
                            candidates=ordinary_candidates,
                            retrieve_index=ordinary_retrieve_index,
                            retrieve_next_token=ordinary_retrieve_next_token,
                            target_topk_indices=ordinary_target_topk_indices,
                            target_topk_values=ordinary_target_topk_values,
                            draft_topk_indices=ordinary_draft_topk_indices,
                            draft_topk_values=ordinary_draft_topk_values,
                            uniforms=ordinary_uniforms,
                        )
                elif run_ordinary_verify:
                    fused_verify_done = False
                    if (
                        draft_probs is None
                        and _is_cuda
                        and _welm_mtp_verify_fused_reject_enabled()
                    ):
                        try:
                            fused_verify_done = (
                                welmv4_mtp_fused_verify_top1_dense_target(
                                    predicts=predict,
                                    accept_index=ordinary_accept_index,
                                    accept_token_num=ordinary_num_correct_drafts,
                                    candidates=ordinary_candidates,
                                    retrieve_index=ordinary_retrieve_index,
                                    retrieve_next_token=ordinary_retrieve_next_token,
                                    target_probs=ordinary_target_probs,
                                    draft_topk_indices=ordinary_draft_topk_indices,
                                    draft_topk_values=ordinary_draft_topk_values,
                                    uniforms=ordinary_uniforms,
                                )
                            )
                        except Exception as exc:
                            if root_only_verify_mask is not None:
                                raise RuntimeError(
                                    "WeLM MTP root-only dense verify fast path failed."
                                ) from exc
                            fused_verify_done = False
                    if not fused_verify_done:
                        verify_top1_chain_with_draft_probs_v2(
                            predicts=predict,
                            accept_index=ordinary_accept_index,
                            accept_token_num=ordinary_num_correct_drafts,
                            candidates=ordinary_candidates,
                            retrieve_index=ordinary_retrieve_index,
                            retrieve_next_token=ordinary_retrieve_next_token,
                            target_probs=ordinary_target_probs,
                            draft_probs=ordinary_draft_probs,
                            draft_topk_indices=ordinary_draft_topk_indices,
                            draft_topk_values=ordinary_draft_topk_values,
                            uniforms=ordinary_uniforms,
                        )
                if ordinary_rows is not None and ordinary_rows.numel() > 0:
                    accept_index[ordinary_rows] = ordinary_accept_index
                    num_correct_drafts[ordinary_rows] = ordinary_num_correct_drafts
            else:
                draft_probs = torch.zeros_like(target_probs)

                # coins for rejection sampling
                tree_uniforms = welm_mtp_deterministic_uniforms(
                    sampling_info=sampling_info,
                    positions=verify_base_positions,
                    batch_size=bs,
                    width=self.draft_token_num + 1,
                    salt=2000,
                )
                if root_only_verify_mask is not None and tree_uniforms is None:
                    tree_uniforms = _welm_mtp_root_only_random_uniforms(
                        root_only_verify_mask, self.draft_token_num + 1
                    )
                coins = (
                    tree_uniforms[:, : self.draft_token_num].contiguous()
                    if tree_uniforms is not None
                    else torch.rand_like(candidates, dtype=torch.float32, device=device)
                )
                # coins for final sampling
                coins_for_final_sampling = (
                    tree_uniforms[:, self.draft_token_num].contiguous()
                    if tree_uniforms is not None
                    else torch.rand((bs,), dtype=torch.float32, device=device)
                )
                ordinary_accept_index = (
                    accept_index
                    if ordinary_rows is None
                    else accept_index[ordinary_rows].contiguous()
                )
                ordinary_num_correct_drafts = (
                    num_correct_drafts
                    if ordinary_rows is None
                    else num_correct_drafts[ordinary_rows].contiguous()
                )
                if ordinary_rows is None or ordinary_rows.numel() > 0:
                    tree_speculative_sampling_target_only(
                        predicts=predict,
                        accept_index=ordinary_accept_index,
                        accept_token_num=ordinary_num_correct_drafts,
                        candidates=(
                            candidates
                            if ordinary_rows is None
                            else candidates[ordinary_rows]
                        ),
                        # kwarg LHS retained as `retrive_*` to match op schema.
                        retrive_index=(
                            self.retrieve_index
                            if ordinary_rows is None
                            else self.retrieve_index[ordinary_rows]
                        ),
                        retrive_next_token=(
                            self.retrieve_next_token
                            if ordinary_rows is None
                            else self.retrieve_next_token[ordinary_rows]
                        ),
                        retrive_next_sibling=(
                            self.retrieve_next_sibling
                            if ordinary_rows is None
                            else self.retrieve_next_sibling[ordinary_rows]
                        ),
                        uniform_samples=(
                            coins if ordinary_rows is None else coins[ordinary_rows]
                        ),
                        uniform_samples_for_final_sampling=(
                            coins_for_final_sampling
                            if ordinary_rows is None
                            else coins_for_final_sampling[ordinary_rows]
                        ),
                        target_probs=target_probs,
                        draft_probs=draft_probs,
                        threshold_single=(
                            get_global_server_args().speculative_accept_threshold_single
                        ),
                        threshold_acc=(
                            get_global_server_args().speculative_accept_threshold_acc
                        ),
                        deterministic=True,
                    )
                if ordinary_rows is not None and ordinary_rows.numel() > 0:
                    accept_index[ordinary_rows] = ordinary_accept_index
                    num_correct_drafts[ordinary_rows] = ordinary_num_correct_drafts
                if root_only_verify_mask is not None:
                    root_sampling_uniforms = tree_uniforms[:, -1]

            if root_only_verify_mask is not None:
                if root_sampling_uniforms is None:
                    root_sampling_uniforms = verify_uniforms[:, -1]
                welmv4_mtp_apply_root_only_sampling(
                    predicts=predict,
                    accept_index=accept_index,
                    accept_token_num=num_correct_drafts,
                    retrieve_index=self.retrieve_index,
                    root_only_verify_mask=root_only_verify_mask,
                    target_probs=root_target_probs,
                    target_topk_indices=root_target_topk_indices,
                    target_topk_values=root_target_topk_values,
                    root_uniforms=root_sampling_uniforms,
                )

            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(num_correct_drafts, src=0)
            if root_only_verify_mask is not None:
                root_predict_indices = self.retrieve_index[
                    root_only_verify_mask, 0
                ].to(dtype=torch.long)
                root_sampled_tokens = predict[root_predict_indices].clone()

        if SIMULATE_ACC_LEN > 0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                num_correct_drafts=num_correct_drafts,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        if root_only_verify_mask is not None:
            if root_target_predict is not None:
                welmv4_mtp_apply_root_only_sampling(
                    predicts=predict,
                    accept_index=accept_index,
                    accept_token_num=num_correct_drafts,
                    retrieve_index=self.retrieve_index,
                    root_only_verify_mask=root_only_verify_mask,
                    target_predict=root_target_predict,
                )
            elif SIMULATE_ACC_LEN > 0:
                root_rows = torch.nonzero(
                    root_only_verify_mask, as_tuple=False
                ).flatten()
                root_predict_indices = self.retrieve_index[root_rows, 0].to(
                    dtype=torch.long
                )
                accept_index[root_rows] = -1
                accept_index[root_rows, 0] = root_predict_indices.to(
                    dtype=accept_index.dtype
                )
                num_correct_drafts[root_rows] = 0
                predict[root_predict_indices] = root_sampled_tokens
            torch._assert_async(
                torch.all(num_correct_drafts[root_only_verify_mask] == 0),
                "WeLM MTP root-only verify accepted a non-root draft token.",
            )

        # `num_correct_drafts` stays drafts-only inside this function; the returned
        # tensor includes the trailing/bonus token via out-of-place +1 so the
        # name no longer flips semantics mid-function (naming doc C2).
        return predict, num_correct_drafts + 1, accept_index


@triton.jit
def fill_new_bonus_tokens(
    bonus_tokens,
    accept_lens,
    new_bonus_tokens,
    num_draft_tokens: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    # `accept_lens` includes the bonus token; the last accepted slot is at -1.
    accept_len = tl.load(accept_lens + pid)

    bonus_tokens_idx = num_draft_tokens * pid + accept_len - 1
    bonus_tokens_data = tl.load(bonus_tokens + bonus_tokens_idx)
    tl.store(new_bonus_tokens + pid, bonus_tokens_data)


@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    if _is_cuda or _is_hip or _is_musa:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )

        return out_cache_loc

    elif _is_npu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int32,
            device=device,
        )
        torch.ops.npu.cache_loc_update(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
        )

        return out_cache_loc
