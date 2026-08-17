from __future__ import annotations

import bisect
import contextlib
import logging
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_cp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
)
from sglang.srt.layers.moe.utils import get_speculative_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.models.welm_perf_opt import get_welm_oe_hash_config
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.welmv4_mtp_sampling import (
    welm_mtp_deterministic_uniforms,
    welmv4_mtp_fused_topk_softmax_sample,
)
from sglang.srt.utils import (
    get_bool_env_var,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)
from sglang.srt.utils.common import fast_topk, is_cuda, is_musa
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

_is_cuda = is_cuda()
_is_musa = is_musa()
logger = logging.getLogger(__name__)

if _is_cuda or _is_musa:
    from sgl_kernel import top_p_renorm_prob

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker


def _validate_dp_proposal_graph_consensus_config(
    *, enable_dp_attention: bool, dp_size: int
) -> None:
    if (
        enable_dp_attention
        and dp_size > 1
        and envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
    ):
        raise ValueError(
            "WeLM MTP draft proposal CUDA graph with DP attention requires the "
            "scheduler all-gather for rank-consistent graph selection; unset "
            "SGLANG_SCHEDULER_SKIP_ALL_GATHER or disable "
            "SGLANG_WELM_MTP_DRAFT_CUDA_GRAPH."
        )


def _compute_capture_sampling_modes(
    eagle_worker, *, topk: int, supports_draft_top_p: bool
) -> list[tuple[bool, bool]]:
    """Return every (sample_draft, use_top_p) mode the launch policy can require.

    _required_draft_sampling_mode picks one of these per batch at replay time;
    a mode missing from the captured family degrades every such batch to the
    eager fallback.
    """
    if not eagle_worker._is_welmv4_mtp_draft_sampling_enabled() or topk != 1:
        return [(False, False)]
    fixed_top_p = eagle_worker.welmv4_mtp_draft_fixed_top_p
    if eagle_worker._has_welmv4_mtp_fixed_draft_sampling_params():
        # Fixed draft params force sampling for every batch, so no greedy
        # graph is reachable; top-p is static only when fixed top-p sets it.
        if fixed_top_p is not None:
            return [(True, fixed_top_p < 1.0 and supports_draft_top_p)]
        if supports_draft_top_p:
            return [(True, False), (True, True)]
        return [(True, False)]
    modes = [(False, False), (True, False)]
    if supports_draft_top_p:
        modes.append((True, True))
    return modes


@dataclass
class WelmMTPDraftProposalInputBuffers(ForwardInputBuffers):
    input_ids: torch.Tensor
    first_input_ids: torch.Tensor
    req_pool_indices: torch.Tensor
    out_cache_loc: torch.Tensor
    out_cache_loc_swa: Optional[torch.Tensor]
    positions: torch.Tensor
    mrope_positions: torch.Tensor
    hidden_states: torch.Tensor
    mirrored_kv_indices: Optional[torch.Tensor]
    identity_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    num_correct_drafts: torch.Tensor
    num_accept_tokens: torch.Tensor
    custom_last_index: torch.Tensor
    custom_last_cache_loc: torch.Tensor
    base_positions: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    temperature: torch.Tensor
    top_p: torch.Tensor
    uniform_samples: torch.Tensor
    welm_mtp_oe_hash_out: Optional[torch.Tensor]
    welm_mtp_query_hash_inputs: Optional[torch.Tensor]
    welm_mtp_oe_entry_history: Optional[torch.Tensor]
    welm_mtp_oe_work_history_a: Optional[torch.Tensor]
    welm_mtp_oe_work_history_b: Optional[torch.Tensor]
    welm_mtp_oe_parent_indices: Optional[torch.Tensor]
    welm_mtp_oe_prev_input_ids: Optional[torch.Tensor]
    welm_mtp_oe_prev_prev_input_ids: Optional[torch.Tensor]
    welm_mtp_oe_output_prev_input_ids: Optional[torch.Tensor]
    welm_mtp_oe_hash_out_batch_major: Optional[torch.Tensor]
    welm_mtp_draft_input_ids: Optional[torch.Tensor]
    welm_mtp_candidate_indices: Optional[torch.Tensor]
    linear_verify_tokens: torch.Tensor
    linear_verify_positions: torch.Tensor
    linear_verify_mask: torch.Tensor
    linear_verify_retrieve_index: torch.Tensor
    linear_verify_retrieve_next_token: torch.Tensor
    linear_verify_retrieve_next_sibling: torch.Tensor
    linear_verify_hash_seq_lens: torch.Tensor
    linear_proposal_parent_list: torch.Tensor
    linear_proposal_top_scores_index: torch.Tensor
    linear_proposal_topk_p: torch.Tensor
    linear_proposal_topk_index: torch.Tensor
    linear_proposal_tokens: torch.Tensor
    linear_proposal_draft_topk_indices: Optional[torch.Tensor]
    linear_proposal_draft_topk_values: Optional[torch.Tensor]
    welm_mtp_branch_step_cache_locs: Optional[torch.Tensor]
    welm_mtp_branch_flat_cache_locs: Optional[torch.Tensor]
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]
    num_token_non_padded: Optional[torch.Tensor]


class WelmMTPDraftProposalCudaGraphRunner:
    """CUDA graph runner for WeLM MTP merged extend-draft proposal generation.

    The graph layer captures the merged extend-draft callable supplied by
    EagleDraftWorker. It owns static buffers, graph capture, replay, and output
    copying.
    """

    def __init__(
        self,
        eagle_worker: "EagleDraftWorker",
        *,
        draft_extend_attn_backend=None,
    ):
        self.eagle_worker = eagle_worker
        self.model_runner = model_runner = eagle_worker.draft_runner
        self.graphs = {}
        self.output_buffers = {}
        self.forward_batches = {}
        self.contracted_rows_by_bs = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.use_token_owner = model_runner.server_args.enable_token_owner
        self.use_token_owner_deepep = (
            self.use_token_owner and get_speculative_moe_a2a_backend().is_deepep()
        )
        self.tp_size = self.model_runner.tp_size
        self.dp_size = self.model_runner.dp_size
        self.use_dp_sampling_consensus = (
            model_runner.server_args.enable_dp_attention and self.dp_size > 1
        )
        _validate_dp_proposal_graph_consensus_config(
            enable_dp_attention=model_runner.server_args.enable_dp_attention,
            dp_size=self.dp_size,
        )
        self.supports_draft_top_p = _is_cuda or _is_musa
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.draft_extend_attn_backend = (
            draft_extend_attn_backend or eagle_worker.draft_extend_attn_backend
        )
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()
        self.welm_mtp_mirror_kv_states = None
        self.welm_mtp_mirror_padding_index = 0
        self.welm_mtp_mirror_kv_len = 0
        eagle_topk = int(model_runner.server_args.speculative_eagle_topk)
        self.capture_sampling_modes = _compute_capture_sampling_modes(
            eagle_worker,
            topk=eagle_topk,
            supports_draft_top_p=self.supports_draft_top_p,
        )
        self.family_has_sampling = any(
            sample for sample, _ in self.capture_sampling_modes
        )
        self.sampling_topk = (
            eagle_worker._get_welmv4_mtp_draft_sampling_topk()
            if self.family_has_sampling
            else 0
        )
        # Active sampling mode; capture and replay re-point it per graph family.
        self.sample_draft, self.use_top_p = self.capture_sampling_modes[0]
        self.graphs_by_mode = {}
        self.output_buffers_by_mode = {}
        if len(self.capture_sampling_modes) > 1:
            logger.info(
                "WeLM MTP draft proposal graph captures one family per "
                "(sample_draft, use_top_p) mode %s with sampling top-k %d.",
                self.capture_sampling_modes,
                self.sampling_topk,
            )
        self.fused_topk_sample = get_bool_env_var(
            "SGLANG_WELM_MTP_FUSED_DRAFT_TOPK_SAMPLE"
        )
        if self.fused_topk_sample:
            logger.info(
                "SGLANG_WELM_MTP_FUSED_DRAFT_TOPK_SAMPLE=1: use the exact "
                "small-K Triton top-k/top-p/sample path in the MTP draft graph."
            )
        self.distributed_topk = (
            self.fused_topk_sample
            and self.tp_size > 1
            and self.dp_size == 1
            and get_bool_env_var("SGLANG_WELM_MTP_DISTRIBUTED_TOPK")
        )
        if self.distributed_topk:
            logger.info(
                "SGLANG_WELM_MTP_DISTRIBUTED_TOPK=1: replace full-vocab TP "
                "all-gather with local top-k plus compact candidate gather."
            )
        self.fused_prepare = get_bool_env_var("SGLANG_WELM_MTP_FUSED_PREPARE")
        if self.fused_prepare:
            logger.info(
                "SGLANG_WELM_MTP_FUSED_PREPARE=1: pack ragged accepted tokens "
                "into CUDA-graph buffers with one fused Triton launch."
            )
        self.persistent_handoff = self.fused_prepare and get_bool_env_var(
            "SGLANG_WELM_MTP_PERSISTENT_HANDOFF"
        )
        if self.persistent_handoff:
            logger.info(
                "SGLANG_WELM_MTP_PERSISTENT_HANDOFF=1: write target-verify "
                "outputs directly into persistent MTP proposal graph buffers."
            )
        self.linear_verify_prepare = (
            self.persistent_handoff
            and int(model_runner.server_args.speculative_eagle_topk) == 1
            and get_bool_env_var("SGLANG_WELM_MTP_LINEAR_VERIFY_PREPARE")
        )
        self._fused_linear_graph_outputs_base = (
            self.linear_verify_prepare
            and self.speculative_num_steps == 3
            and self.speculative_num_draft_tokens == 4
            and self.family_has_sampling
            and self.sampling_topk > 0
            and get_bool_env_var("SGLANG_WELM_MTP_FUSED_LINEAR_GRAPH_OUTPUTS")
        )
        if self._fused_linear_graph_outputs_base:
            logger.info(
                "SGLANG_WELM_MTP_FUSED_LINEAR_GRAPH_OUTPUTS=1: fuse MTP3 "
                "proposal output assembly into one proposal-graph kernel."
            )
        self.synced_draft_generator = None
        if self.fused_prepare and self.family_has_sampling:
            # TP workers receive the same server seed before model-runner init.
            # A dedicated generator therefore produces identical draft samples
            # on every TP rank without a per-replay NCCL broadcast, while being
            # isolated from unrelated model/runtime RNG consumption.
            self.synced_draft_generator = torch.Generator(device=model_runner.device)
            self.synced_draft_generator.manual_seed(
                int(model_runner.server_args.random_seed)
            )

        self.topk = int(model_runner.server_args.speculative_eagle_topk)
        self.branch_tokens_per_bs = int(self.topk * self.speculative_num_steps)
        self.num_tokens_per_bs = int(
            max(self.speculative_num_draft_tokens, self.branch_tokens_per_bs)
            if self.topk > 1
            else self.speculative_num_draft_tokens
        )
        capture_bs, compile_bs = get_batch_sizes_to_capture(
            model_runner, num_tokens_per_bs=self.num_tokens_per_bs
        )
        self.capture_bs = self._filter_contracted_dp_capture_bs(capture_bs)
        self.compile_bs = [bs for bs in compile_bs if bs in self.capture_bs]
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.max_branch_num_token = self.max_bs * self.branch_tokens_per_bs
        self._welm_mtp_mirror_cu_seqlens_q = {
            bs: torch.arange(
                0,
                bs + 1,
                dtype=torch.int32,
                device=model_runner.device,
            )
            for bs in self.capture_bs
        }

        self.draft_extend_attn_backend.init_cuda_graph_state(
            self.max_bs, self.max_num_token
        )
        if self.topk > 1:
            self.eagle_worker.draft_attn_backend.init_cuda_graph_state(
                self.max_bs, self.max_branch_num_token
            )
        self.seq_len_fill_value = (
            self.draft_extend_attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        # MR !151 added seq_len bucketing to CudaGraphRunner.capture(); mirror
        # the interface here so CudaGraphRunner.capture(self) (duck-typed) works.
        # The draft proposal graph is DRAFT_EXTEND (not decode) and does not use
        # AttnCP sharded-KV bucketing, so each bs maps to a single fill value and
        # the graph key carries no seq_len bucket suffix.
        self.seq_len_fill_values_by_bs = {
            int(bs): [self.seq_len_fill_value] for bs in self.capture_bs
        }
        self.enable_seq_len_graph_buckets = False
        self.extend_seq_lens_cpu = [self.num_tokens_per_bs] * self.max_bs

        if self.enable_torch_compile:
            set_torch_compile_config()

        with torch.device(model_runner.device):
            input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)
            first_input_ids = torch.zeros((self.max_bs,), dtype=torch.int64)
            req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int64)
            out_cache_loc = torch.zeros((self.max_num_token,), dtype=torch.int64)
            out_cache_loc_swa = (
                torch.zeros((self.max_num_token,), dtype=torch.int32)
                if self.model_runner.is_hybrid_swa
                else None
            )
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)
            hidden_states = torch.zeros(
                (self.max_num_token, self.model_runner.model_config.spec_hidden_size),
                dtype=self.model_runner.dtype,
            )
            mirrored_kv_indices = torch.arange(self.max_num_token, dtype=torch.int64)
            identity_indices = torch.arange(self.max_num_token, dtype=torch.int64)
            seq_lens = torch.full(
                (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            extend_seq_lens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            num_correct_drafts = torch.full(
                (self.max_bs,), self.num_tokens_per_bs - 1, dtype=torch.int32
            )
            num_accept_tokens = torch.full(
                (self.max_bs,), self.num_tokens_per_bs, dtype=torch.int32
            )
            custom_last_index = (
                torch.arange(self.max_bs, dtype=torch.long) * self.num_tokens_per_bs
                + self.num_tokens_per_bs
                - 1
            )
            custom_last_cache_loc = torch.zeros((self.max_bs,), dtype=torch.int64)
            base_positions = torch.zeros((self.max_bs,), dtype=torch.int64)
            if hasattr(self.model_runner.model_config.hf_config, "draft_vocab_size"):
                vocab_size = self.model_runner.model_config.hf_config.draft_vocab_size
            elif hasattr(self.model_runner.model_config.hf_config, "hot_vocab_size"):
                vocab_size = self.model_runner.model_config.hf_config.hot_vocab_size
            else:
                vocab_size = self.model_runner.model_config.vocab_size
            next_token_logits_buffer = torch.zeros(
                (self.max_num_token, vocab_size),
                dtype=torch.float,
            )
            temperature = torch.full(
                (self.max_bs, 1),
                (
                    1.0
                    if eagle_worker.welmv4_mtp_draft_fixed_temperature is None
                    else eagle_worker.welmv4_mtp_draft_fixed_temperature
                ),
                dtype=torch.float32,
            )
            top_p = torch.full(
                (self.max_bs,),
                (
                    1.0
                    if eagle_worker.welmv4_mtp_draft_fixed_top_p is None
                    else eagle_worker.welmv4_mtp_draft_fixed_top_p
                ),
                dtype=torch.float32,
            )
            uniform_samples = torch.full(
                (self.speculative_num_steps, self.max_bs, 1),
                0.5,
                dtype=torch.float32,
            )
            welm_mtp_candidate_indices = (
                torch.zeros(
                    (
                        self.speculative_num_steps,
                        self.max_bs,
                        self.sampling_topk * self.tp_size,
                    ),
                    dtype=torch.int64,
                )
                if self.distributed_topk
                else None
            )
            linear_verify_tokens = torch.zeros((self.max_num_token,), dtype=torch.int64)
            linear_verify_positions = torch.zeros(
                (self.max_num_token,), dtype=torch.int64
            )
            linear_verify_mask = (
                torch.ones(
                    (self.max_bs, self.num_tokens_per_bs, self.num_tokens_per_bs),
                    dtype=torch.bool,
                )
                .tril()
                .reshape(-1)
            )
            linear_verify_retrieve_index = torch.arange(
                self.max_num_token, dtype=torch.int64
            ).reshape(self.max_bs, self.num_tokens_per_bs)
            linear_verify_retrieve_next_token = torch.arange(
                1, self.num_tokens_per_bs + 1, dtype=torch.int64
            ).repeat(self.max_bs, 1)
            linear_verify_retrieve_next_token[:, -1].fill_(-1)
            linear_verify_retrieve_next_sibling = torch.full(
                (self.max_bs, self.num_tokens_per_bs), -1, dtype=torch.int64
            )
            linear_verify_hash_seq_lens = torch.zeros((self.max_bs,), dtype=torch.int32)
            # A top-k=1, three-step proposal is always the same chain.  The
            # generic proposal builder derives these two tensors with
            # cumulative products, topk, sort and gather on every replay even
            # though the result is invariant.  Keep exact static equivalents
            # for both the fast verify path and its generic-tree rollback.
            linear_proposal_parent_list = torch.tensor(
                [-1, *range(self.speculative_num_steps - 1)],
                dtype=torch.int64,
            ).repeat(self.max_bs, 1)
            linear_proposal_top_scores_index = torch.arange(
                self.speculative_num_steps, dtype=torch.int64
            ).repeat(self.max_bs, 1)
            linear_proposal_topk_p = torch.zeros(
                (self.max_bs, self.speculative_num_steps), dtype=torch.float32
            )
            linear_proposal_topk_index = torch.zeros(
                (self.max_bs, self.speculative_num_steps), dtype=torch.int64
            )
            linear_proposal_tokens = torch.zeros(
                (self.max_bs, self.speculative_num_steps), dtype=torch.int64
            )
            if self.family_has_sampling:
                linear_proposal_draft_topk_indices = torch.zeros(
                    (self.max_bs, self.num_tokens_per_bs, self.sampling_topk),
                    dtype=torch.int64,
                )
                linear_proposal_draft_topk_values = torch.zeros(
                    (self.max_bs, self.num_tokens_per_bs, self.sampling_topk),
                    dtype=torch.float32,
                )
            else:
                linear_proposal_draft_topk_indices = None
                linear_proposal_draft_topk_values = None
            if self.topk > 1:
                welm_mtp_branch_step_cache_locs = torch.zeros(
                    (self.speculative_num_steps, self.max_bs * self.topk),
                    dtype=torch.int64,
                )
                welm_mtp_branch_flat_cache_locs = torch.zeros(
                    (self.max_branch_num_token,),
                    dtype=torch.int64,
                )
            else:
                welm_mtp_branch_step_cache_locs = None
                welm_mtp_branch_flat_cache_locs = None

            if eagle_worker._should_use_welmv4_mtp_oe_hash_kernel():
                oe_grams, oe_vocab_sizes = get_welm_oe_hash_config(
                    self.model_runner.model_config
                )
                history_width = max(int(g) for g in oe_grams)
                welm_mtp_oe_hash_out = torch.zeros(
                    (len(oe_vocab_sizes), self.max_num_token), dtype=torch.int64
                )
                welm_mtp_query_hash_inputs = torch.zeros(
                    (len(oe_vocab_sizes), self.max_bs), dtype=torch.int64
                )
                welm_mtp_oe_entry_history = torch.zeros(
                    (self.max_bs, history_width), dtype=torch.int64
                )
                welm_mtp_oe_work_history_a = torch.zeros(
                    (self.max_num_token, history_width), dtype=torch.int64
                )
                welm_mtp_oe_work_history_b = torch.zeros(
                    (self.max_num_token, history_width), dtype=torch.int64
                )
                welm_mtp_oe_parent_indices = torch.zeros(
                    (self.max_num_token,), dtype=torch.int64
                )
                welm_mtp_oe_prev_input_ids = torch.zeros(
                    (self.max_num_token,), dtype=torch.int64
                )
                welm_mtp_oe_prev_prev_input_ids = torch.zeros(
                    (self.max_num_token,), dtype=torch.int64
                )
                welm_mtp_oe_output_prev_input_ids = torch.zeros(
                    (self.max_num_token,), dtype=torch.int64
                )
                welm_mtp_oe_hash_out_batch_major = torch.zeros(
                    (self.max_num_token, len(oe_vocab_sizes)), dtype=torch.int64
                )
                welm_mtp_draft_input_ids = torch.zeros(
                    (self.speculative_num_steps, self.max_num_token),
                    dtype=torch.int64,
                )
            else:
                welm_mtp_oe_hash_out = None
                welm_mtp_query_hash_inputs = None
                welm_mtp_oe_entry_history = None
                welm_mtp_oe_work_history_a = None
                welm_mtp_oe_work_history_b = None
                welm_mtp_oe_parent_indices = None
                welm_mtp_oe_prev_input_ids = None
                welm_mtp_oe_prev_prev_input_ids = None
                welm_mtp_oe_output_prev_input_ids = None
                welm_mtp_oe_hash_out_batch_major = None
                welm_mtp_draft_input_ids = None

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None
            num_token_non_padded = (
                torch.zeros((1,), dtype=torch.int32) if self.use_token_owner else None
            )

            self.welm_mtp_mirror_padding_index = self.max_num_token
            self.welm_mtp_mirror_kv_len = self.max_num_token + 1
            self.welm_mtp_mirror_kv_states = {}
            # These mirror-KV scratch buffers hold a copy of the target's KV
            # for the draft proposal graph. Tag them GPU_MEMORY_TYPE_KV_CACHE
            # so release_memory_occupation / pause(GPU_MEMORY_TYPE_KV_CACHE)
            # can release them alongside the draft's own KV pool. Stale data
            # after resume is harmless: _copy_welmv4_mtp_mirror_kv_states
            # refreshes them before graph.replay() on the next replay.
            with self.model_runner.memory_saver_adapter.region(
                GPU_MEMORY_TYPE_KV_CACHE
            ):
                for layer_idx, kv_size in self._welmv4_mtp_mirror_kv_specs():
                    packed_kv = torch.zeros(
                        (self.welm_mtp_mirror_kv_len, 2 * kv_size),
                        dtype=self.model_runner.dtype,
                    )
                    self.welm_mtp_mirror_kv_states[layer_idx] = (
                        packed_kv[:, :kv_size],
                        packed_kv[:, kv_size:],
                    )

        seq_lens_cpu = torch.full(
            (self.max_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        self.buffers = WelmMTPDraftProposalInputBuffers(
            input_ids=input_ids,
            first_input_ids=first_input_ids,
            req_pool_indices=req_pool_indices,
            out_cache_loc=out_cache_loc,
            out_cache_loc_swa=out_cache_loc_swa,
            positions=positions,
            mrope_positions=mrope_positions,
            hidden_states=hidden_states,
            mirrored_kv_indices=mirrored_kv_indices,
            identity_indices=identity_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            num_correct_drafts=num_correct_drafts,
            num_accept_tokens=num_accept_tokens,
            custom_last_index=custom_last_index,
            custom_last_cache_loc=custom_last_cache_loc,
            base_positions=base_positions,
            next_token_logits_buffer=next_token_logits_buffer,
            temperature=temperature,
            top_p=top_p,
            uniform_samples=uniform_samples,
            welm_mtp_oe_hash_out=welm_mtp_oe_hash_out,
            welm_mtp_query_hash_inputs=welm_mtp_query_hash_inputs,
            welm_mtp_oe_entry_history=welm_mtp_oe_entry_history,
            welm_mtp_oe_work_history_a=welm_mtp_oe_work_history_a,
            welm_mtp_oe_work_history_b=welm_mtp_oe_work_history_b,
            welm_mtp_oe_parent_indices=welm_mtp_oe_parent_indices,
            welm_mtp_oe_prev_input_ids=welm_mtp_oe_prev_input_ids,
            welm_mtp_oe_prev_prev_input_ids=welm_mtp_oe_prev_prev_input_ids,
            welm_mtp_oe_output_prev_input_ids=welm_mtp_oe_output_prev_input_ids,
            welm_mtp_oe_hash_out_batch_major=welm_mtp_oe_hash_out_batch_major,
            welm_mtp_draft_input_ids=welm_mtp_draft_input_ids,
            welm_mtp_candidate_indices=welm_mtp_candidate_indices,
            linear_verify_tokens=linear_verify_tokens,
            linear_verify_positions=linear_verify_positions,
            linear_verify_mask=linear_verify_mask,
            linear_verify_retrieve_index=linear_verify_retrieve_index,
            linear_verify_retrieve_next_token=linear_verify_retrieve_next_token,
            linear_verify_retrieve_next_sibling=linear_verify_retrieve_next_sibling,
            linear_verify_hash_seq_lens=linear_verify_hash_seq_lens,
            linear_proposal_parent_list=linear_proposal_parent_list,
            linear_proposal_top_scores_index=linear_proposal_top_scores_index,
            linear_proposal_topk_p=linear_proposal_topk_p,
            linear_proposal_topk_index=linear_proposal_topk_index,
            linear_proposal_tokens=linear_proposal_tokens,
            linear_proposal_draft_topk_indices=linear_proposal_draft_topk_indices,
            linear_proposal_draft_topk_values=linear_proposal_draft_topk_values,
            welm_mtp_branch_step_cache_locs=welm_mtp_branch_step_cache_locs,
            welm_mtp_branch_flat_cache_locs=welm_mtp_branch_flat_cache_locs,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
            num_token_non_padded=num_token_non_padded,
        )
        self.buffers.share_buffers()

        from sglang.srt.models.welm_v45_80a3_h2048_hd256_pre_attn_v2_config import (
            welm_v45_80a3_h2048_hd256_pre_attn_v2_enabled,
        )

        welm_qkv_prepared = 0
        if welm_v45_80a3_h2048_hd256_pre_attn_v2_enabled():
            from sglang.srt.models.welm_v45_80a3_h2048_hd256_pre_attn_v2 import (
                prepare_welm_v45_80a3_h2048_hd256_nextn_pre_attn_v2_cuda_graphs,
            )

            welm_qkv_prepared = (
                prepare_welm_v45_80a3_h2048_hd256_nextn_pre_attn_v2_cuda_graphs(
                    self.model_runner,
                    self.buffers,
                    self.capture_bs,
                    self.num_tokens_per_bs,
                    self.welm_mtp_mirror_kv_states,
                )
            )
        if welm_qkv_prepared:
            logger.info(
                "Prepared %d WeLM v4.5 80A3 H2048/HD256 NextN Pre-Attn V2 "
                "CUDA graph operations.",
                welm_qkv_prepared,
            )

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture WeLM MTP draft proposal cuda graph failed: {e}\n"
                f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    @property
    def fused_linear_graph_outputs(self) -> bool:
        # Follows the active mode: fused output assembly is captured only into
        # the sampling graphs of a family.
        return self._fused_linear_graph_outputs_base and self.sample_draft

    def build_linear_verify_inputs(
        self, draft_input: EagleDraftInput, seq_lens: torch.Tensor
    ):
        """Return persistent verify tensors for the fixed top-k=1 MTP chain."""
        if not self.linear_verify_prepare:
            return None
        proposal_tokens = getattr(draft_input, "draft_proposal_tokens", None)
        bonus_tokens = getattr(draft_input, "bonus_tokens", None)
        if proposal_tokens is None or bonus_tokens is None:
            return None
        bs = int(seq_lens.numel())
        if bs <= 0 or bs > self.max_bs:
            return None
        from sglang.srt.speculative.welmv4_mtp_staging import (
            build_welm_mtp_linear_verify_inputs,
        )

        num_tokens = bs * self.num_tokens_per_bs
        graph_verify_ready = bool(
            self.fused_linear_graph_outputs
            and getattr(draft_input, "welm_mtp_linear_verify_ready", False)
            and getattr(draft_input, "future_indices", None) is None
        )
        if not graph_verify_ready:
            build_welm_mtp_linear_verify_inputs(
                bonus_tokens=bonus_tokens,
                proposal_tokens=proposal_tokens,
                seq_lens=seq_lens,
                output_tokens=self.buffers.linear_verify_tokens,
                output_positions=self.buffers.linear_verify_positions,
                batch_size=bs,
                tokens_per_bs=self.num_tokens_per_bs,
            )
        mask_tokens = bs * self.num_tokens_per_bs * self.num_tokens_per_bs
        return (
            self.buffers.linear_verify_mask[:mask_tokens],
            self.buffers.linear_verify_positions[:num_tokens],
            self.buffers.linear_verify_retrieve_index[:bs],
            self.buffers.linear_verify_retrieve_next_token[:bs],
            self.buffers.linear_verify_retrieve_next_sibling[:bs],
            self.buffers.linear_verify_tokens[:num_tokens],
            self.buffers.linear_verify_hash_seq_lens[:bs],
        )

    def _filter_contracted_dp_capture_bs(self, capture_bs: list[int]) -> list[int]:
        if (
            self.use_token_owner
            or not self.require_gathered_buffer
            or not DpPaddingMode.get_default_mode_in_cuda_graph().is_max_len()
        ):
            return capture_bs

        mul_base = get_attention_tp_size()
        attn_cp_size = get_attention_cp_size()
        if mul_base % attn_cp_size != 0:
            mul_base *= attn_cp_size
        if mul_base <= 1:
            return capture_bs

        num_max_requests = self.model_runner.req_to_token_pool.size
        num_max_requests = (num_max_requests + mul_base - 1) // mul_base * mul_base
        candidates = list(capture_bs)
        if num_max_requests >= mul_base and not any(
            bs % mul_base == 0 for bs in candidates
        ):
            candidates.append(mul_base)
        filtered = sorted(
            {bs for bs in candidates if bs % mul_base == 0 and bs <= num_max_requests}
        )
        if not filtered:
            raise RuntimeError(
                "WeLM MTP draft proposal cuda graph requires a capture batch "
                "size divisible by the attention TP size when DP attention is "
                f"enabled, got capture_bs={capture_bs}, align={mul_base}."
            )
        return filtered

    def _welmv4_mtp_mirror_kv_specs(self) -> list[tuple[int, int]]:
        specs = {}
        model = getattr(self.model_runner.model, "model", None)
        for layer in getattr(model, "decoder_layers", []):
            attn = getattr(layer, "self_attn", None)
            qkv_proj = getattr(attn, "qkv_proj", None)
            imitated_layer_idx = getattr(qkv_proj, "imitated_layer_idx", None)
            mirror_layer_idx = getattr(qkv_proj, "mirror_layer_idx", None)
            kv_size = getattr(attn, "kv_size", None)
            if imitated_layer_idx is not None and kv_size is not None:
                layer_idx = (
                    int(mirror_layer_idx)
                    if mirror_layer_idx is not None
                    else int(imitated_layer_idx)
                )
                specs[layer_idx] = int(kv_size)
        return sorted(specs.items())

    def _welmv4_mtp_graph_model_specific_states(self):
        if not self.welm_mtp_mirror_kv_states:
            return None
        return {
            "welm_kv_mirror_states": {
                layer_idx: (k, v)
                for layer_idx, (k, v) in self.welm_mtp_mirror_kv_states.items()
            }
        }

    def _copy_welmv4_mtp_mirror_kv_states(
        self,
        forward_batch: Optional[ForwardBatch],
        graph_num_tokens: int,
        *,
        model_specific_states=None,
        mirrored_kv_indices=None,
    ) -> None:
        if not self.welm_mtp_mirror_kv_states:
            return
        if forward_batch is not None:
            mirrored_kv_indices = getattr(
                forward_batch.spec_info, "mirrored_kv_indices", None
            )
            model_specific_states = forward_batch.model_specific_states
        if mirrored_kv_indices is not None and mirrored_kv_indices.numel() == 0:
            for dst_k, dst_v in self.welm_mtp_mirror_kv_states.values():
                dst_k[self.welm_mtp_mirror_padding_index].zero_()
                dst_v[self.welm_mtp_mirror_padding_index].zero_()
            return
        model_specific_states = model_specific_states or {}
        kv_states = model_specific_states.get("welm_kv_mirror_states")
        if not isinstance(kv_states, dict):
            raise RuntimeError(
                "Missing WeLM MTP mirrored KV states for draft proposal graph replay."
            )
        # Target verify materializes a dense KV tensor for all verify slots.  Copy
        # that tensor's statically known first dimension instead of synchronizing
        # the GPU every replay with mirrored_kv_indices.max().item().  The graph
        # consumes only the rows selected by mirrored_kv_indices.
        required_kv_len = None
        for layer_idx in self.welm_mtp_mirror_kv_states:
            src_pair = kv_states.get(layer_idx)
            if src_pair is None:
                raise RuntimeError(
                    "Missing WeLM MTP mirrored KV state for draft proposal graph "
                    f"replay: {layer_idx=}."
                )
            src_k, src_v = src_pair
            if src_k.shape[0] != src_v.shape[0]:
                raise RuntimeError(
                    "WeLM MTP mirrored K/V row counts differ: "
                    f"{layer_idx=} k_len={src_k.shape[0]} v_len={src_v.shape[0]}."
                )
            if required_kv_len is None:
                required_kv_len = int(src_k.shape[0])
            elif required_kv_len != int(src_k.shape[0]):
                raise RuntimeError(
                    "WeLM MTP mirrored KV layers have inconsistent row counts: "
                    f"expected={required_kv_len} {layer_idx=} got={src_k.shape[0]}."
                )
        required_kv_len = int(required_kv_len or 0)
        if required_kv_len > self.welm_mtp_mirror_padding_index:
            raise RuntimeError(
                "WeLM MTP mirrored KV source exceeds draft proposal graph capacity: "
                f"{required_kv_len=} capacity={self.welm_mtp_mirror_padding_index}"
            )
        real_dsts = []
        real_srcs = []
        padding_dsts = []
        padding_srcs = []
        for layer_idx, (dst_k, dst_v) in self.welm_mtp_mirror_kv_states.items():
            src_pair = kv_states.get(layer_idx)
            if src_pair is None:
                raise RuntimeError(
                    "Missing WeLM MTP mirrored KV state for draft proposal graph "
                    f"replay: {layer_idx=}"
                )
            src_k, src_v = src_pair
            if src_k.shape[0] < required_kv_len or src_v.shape[0] < required_kv_len:
                raise RuntimeError(
                    "WeLM MTP mirrored KV state is shorter than mirrored_kv_indices "
                    f"require: {layer_idx=} {required_kv_len=} "
                    f"k_len={src_k.shape[0]} v_len={src_v.shape[0]}"
                )
            if required_kv_len > 0:
                real_dsts.extend([dst_k[:required_kv_len], dst_v[:required_kv_len]])
                real_srcs.extend([src_k[:required_kv_len], src_v[:required_kv_len]])
                padding_dsts.extend(
                    [
                        dst_k[self.welm_mtp_mirror_padding_index],
                        dst_v[self.welm_mtp_mirror_padding_index],
                    ]
                )
                padding_srcs.extend(
                    [src_k[required_kv_len - 1], src_v[required_kv_len - 1]]
                )
            else:
                dst_k[self.welm_mtp_mirror_padding_index].zero_()
                dst_v[self.welm_mtp_mirror_padding_index].zero_()
        # One multi-tensor launch per copy shape replaces two launches for every
        # mirrored layer.  All tensors share device/dtype and foreach-copy keeps
        # exactly the same asynchronous stream semantics as Tensor.copy_.
        if real_dsts:
            torch._foreach_copy_(real_dsts, real_srcs)
            torch._foreach_copy_(padding_dsts, padding_srcs)

    def _copy_welmv4_mtp_mirrored_kv_indices(
        self, forward_batch: ForwardBatch, graph_num_tokens: int
    ) -> None:
        buffers = self.buffers
        if buffers.mirrored_kv_indices is None:
            return
        buffers.mirrored_kv_indices[:graph_num_tokens].fill_(
            self.welm_mtp_mirror_padding_index
        )
        mirrored_kv_indices = getattr(
            forward_batch.spec_info, "mirrored_kv_indices", None
        )
        if mirrored_kv_indices is None:
            raw_num_tokens = int(forward_batch.input_ids.numel())
            buffers.mirrored_kv_indices[:raw_num_tokens].copy_(
                torch.arange(
                    raw_num_tokens,
                    dtype=buffers.mirrored_kv_indices.dtype,
                    device=buffers.mirrored_kv_indices.device,
                )
            )
            return
        if mirrored_kv_indices.numel() > graph_num_tokens:
            raise RuntimeError(
                "WeLM MTP mirrored_kv_indices is longer than draft proposal graph "
                f"tokens: {mirrored_kv_indices.numel()} > {graph_num_tokens}"
            )
        buffers.mirrored_kv_indices[: mirrored_kv_indices.numel()].copy_(
            mirrored_kv_indices
        )

    def _make_padded_accept_lens(
        self, accepted_lens_cpu: list[int], graph_bs: int
    ) -> Optional[list[int]]:
        if not 0 <= len(accepted_lens_cpu) <= graph_bs or graph_bs <= 0:
            return None
        if any(
            accepted_len < 0 or accepted_len > self.num_tokens_per_bs
            for accepted_len in accepted_lens_cpu
        ):
            return None

        padded_lens_cpu = list(accepted_lens_cpu)
        padded_lens_cpu.extend([1] * (graph_bs - len(padded_lens_cpu)))

        graph_num_tokens = graph_bs * self.num_tokens_per_bs
        real_total = sum(padded_lens_cpu)
        if real_total > graph_num_tokens:
            return None

        pad_tokens = graph_num_tokens - real_total
        for i in range(graph_bs - 1, -1, -1):
            if pad_tokens <= 0:
                break
            room = self.num_tokens_per_bs - padded_lens_cpu[i]
            if room <= 0:
                continue
            add = min(room, pad_tokens)
            padded_lens_cpu[i] += add
            pad_tokens -= add

        if pad_tokens != 0 or sum(padded_lens_cpu) != graph_num_tokens:
            return None
        return padded_lens_cpu

    def _copy_runtime_flat_layout(
        self,
        *,
        input_ids: torch.Tensor,
        out_cache_loc: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        mirrored_kv_indices: Optional[torch.Tensor],
        cached_hash: Optional[torch.Tensor],
        accept_lens: torch.Tensor,
        extend_start_loc: torch.Tensor,
        accepted_lens_cpu: list[int],
        padded_lens_cpu: list[int],
        raw_bs: int,
        graph_num_tokens: int,
    ) -> None:
        buffers = self.buffers
        if self.fused_prepare:
            if buffers.mirrored_kv_indices is None:
                raise RuntimeError(
                    "Fused WeLM MTP graph staging requires a mirror-index buffer."
                )
            from sglang.srt.speculative.welmv4_mtp_staging import (
                pack_welm_mtp_graph_inputs,
            )

            pack_welm_mtp_graph_inputs(
                input_ids=input_ids,
                out_cache_loc=out_cache_loc,
                positions=positions,
                hidden_states=hidden_states,
                mirrored_kv_indices=mirrored_kv_indices,
                cached_hash=cached_hash,
                accept_lens=accept_lens,
                extend_start_loc=extend_start_loc,
                output_input_ids=buffers.input_ids,
                output_cache_loc=buffers.out_cache_loc,
                output_positions=buffers.positions,
                output_hidden_states=buffers.hidden_states,
                output_mirrored_kv_indices=buffers.mirrored_kv_indices,
                output_hash=buffers.welm_mtp_oe_hash_out,
                custom_last_index=buffers.custom_last_index,
                custom_last_cache_loc=buffers.custom_last_cache_loc,
                raw_bs=raw_bs,
                graph_num_tokens=graph_num_tokens,
                tokens_per_bs=self.num_tokens_per_bs,
                mirror_padding_index=self.welm_mtp_mirror_padding_index,
            )
            return
        buffers.input_ids[:graph_num_tokens].zero_()
        buffers.out_cache_loc[:graph_num_tokens].zero_()
        buffers.positions[:graph_num_tokens].zero_()
        buffers.hidden_states[:graph_num_tokens].zero_()
        if buffers.mirrored_kv_indices is not None:
            buffers.mirrored_kv_indices[:graph_num_tokens].fill_(
                self.welm_mtp_mirror_padding_index
            )
        if buffers.welm_mtp_oe_hash_out is not None:
            buffers.welm_mtp_oe_hash_out[:, :graph_num_tokens].zero_()

        real_last_indices = []
        src_start = 0
        dst_start = 0
        for row, padded_len in enumerate(padded_lens_cpu):
            if padded_len <= 0:
                raise RuntimeError(
                    "WeLM MTP draft proposal graph padding produced an empty "
                    f"graph row: row={row}, padded_lens={padded_lens_cpu}."
                )
            if row < raw_bs:
                real_len = accepted_lens_cpu[row]
                src_end = src_start + real_len
                dst_end = dst_start + real_len
                pad_len = padded_len - real_len

                if real_len > 0:
                    buffers.input_ids[dst_start:dst_end].copy_(
                        input_ids[src_start:src_end]
                    )
                    buffers.out_cache_loc[dst_start:dst_end].copy_(
                        out_cache_loc[src_start:src_end]
                    )
                    buffers.positions[dst_start:dst_end].copy_(
                        positions[src_start:src_end]
                    )
                    buffers.hidden_states[dst_start:dst_end].copy_(
                        hidden_states[src_start:src_end]
                    )
                    if buffers.mirrored_kv_indices is not None:
                        if mirrored_kv_indices is None:
                            buffers.mirrored_kv_indices[dst_start:dst_end].copy_(
                                torch.arange(
                                    src_start,
                                    src_end,
                                    dtype=buffers.mirrored_kv_indices.dtype,
                                    device=buffers.mirrored_kv_indices.device,
                                )
                            )
                        else:
                            buffers.mirrored_kv_indices[dst_start:dst_end].copy_(
                                mirrored_kv_indices[src_start:src_end]
                            )
                    if (
                        cached_hash is not None
                        and buffers.welm_mtp_oe_hash_out is not None
                    ):
                        buffers.welm_mtp_oe_hash_out[:, dst_start:dst_end].copy_(
                            cached_hash[:, src_start:src_end]
                        )

                real_last = dst_end - 1 if real_len > 0 else dst_start + padded_len - 1
                real_last_indices.append(real_last)
                if pad_len > 0 and real_len > 0:
                    pad_start = dst_end
                    pad_end = dst_start + padded_len
                    buffers.input_ids[pad_start:pad_end].copy_(
                        input_ids[src_end - 1].expand(pad_len)
                    )
                    buffers.positions[pad_start:pad_end].copy_(
                        positions[src_end - 1]
                        + torch.arange(
                            1,
                            pad_len + 1,
                            dtype=buffers.positions.dtype,
                            device=buffers.positions.device,
                        )
                    )
                src_start = src_end
            else:
                real_last_indices.append(dst_start + padded_len - 1)

            dst_start += padded_len

        if src_start != input_ids.numel() or dst_start != graph_num_tokens:
            raise RuntimeError(
                "WeLM MTP draft proposal graph padding layout mismatch: "
                f"src={src_start}/{input_ids.numel()} "
                f"dst={dst_start}/{graph_num_tokens}."
            )
        buffers.custom_last_index[: len(padded_lens_cpu)].copy_(
            torch.tensor(
                real_last_indices,
                dtype=buffers.custom_last_index.dtype,
                device=buffers.custom_last_index.device,
            )
        )
        real_rows = min(raw_bs, len(real_last_indices))
        if real_rows > 0:
            buffers.custom_last_cache_loc[:real_rows].copy_(
                buffers.out_cache_loc[buffers.custom_last_index[:real_rows]]
            )
        if len(real_last_indices) > real_rows:
            buffers.custom_last_cache_loc[real_rows : len(real_last_indices)].zero_()

    def _set_welmv4_mtp_mirror_metadata(self, bs: int) -> None:
        if not self.model_runner.server_args.enable_welm_kv_mirror_opt:
            return
        mirror_cu_seqlens_q = self._welm_mtp_mirror_cu_seqlens_q.get(bs)
        if mirror_cu_seqlens_q is None:
            mirror_cu_seqlens_q = torch.arange(
                0,
                bs + 1,
                dtype=torch.int32,
                device=self.model_runner.device,
            )
            self._welm_mtp_mirror_cu_seqlens_q[bs] = mirror_cu_seqlens_q

        set_metadata = getattr(
            self.draft_extend_attn_backend,
            "set_welm_mtp_mirror_cuda_graph_metadata",
            None,
        )
        if set_metadata is not None:
            set_metadata(bs, mirror_cu_seqlens_q)
            return

        draft_extend_metadata = getattr(
            self.draft_extend_attn_backend, "draft_extend_metadata", None
        )
        if draft_extend_metadata is None:
            raise RuntimeError(
                "WeLM KV mirror is not supported by speculative draft attention "
                f"backend {type(self.draft_extend_attn_backend).__name__}."
            )
        metadata = draft_extend_metadata.get(bs)
        if metadata is None:
            raise RuntimeError(
                "WeLM KV mirror draft attention metadata was not initialized "
                f"for CUDA graph batch size {bs}."
            )
        metadata.mirror_cu_seqlens_q = mirror_cu_seqlens_q
        metadata.mirror_max_seq_len_q = 1

    def _copy_sampling_params(self, forward_batch: ForwardBatch, raw_bs: int, bs: int):
        self._copy_sampling_params_from_info(forward_batch.sampling_info, raw_bs, bs)

    def _copy_sampling_params_from_info(self, sampling_info, raw_bs: int, bs: int):
        buffers = self.buffers
        fixed_temperature = self.eagle_worker.welmv4_mtp_draft_fixed_temperature
        fixed_top_p = self.eagle_worker.welmv4_mtp_draft_fixed_top_p

        if sampling_info is None:
            return
        if fixed_temperature is None:
            buffers.temperature[:bs].fill_(1.0)
            temperatures = getattr(sampling_info, "temperatures", None)
            if temperatures is not None:
                buffers.temperature[:raw_bs].copy_(
                    temperatures[:raw_bs].to(dtype=torch.float32)
                )
        if fixed_top_p is None:
            buffers.top_p[:bs].fill_(1.0)
            top_ps = getattr(sampling_info, "top_ps", None)
            if top_ps is not None:
                buffers.top_p[:raw_bs].copy_(top_ps[:raw_bs].to(dtype=torch.float32))

    def _copy_sampling_randomness(
        self, forward_batch: ForwardBatch, raw_bs: int, bs: int
    ):
        self._copy_sampling_randomness_from_info(
            forward_batch.sampling_info,
            forward_batch.seq_lens,
            raw_bs,
            bs,
        )

    def _copy_sampling_randomness_from_info(
        self, sampling_info, seq_lens: torch.Tensor, raw_bs: int, bs: int
    ):
        if not self.sample_draft:
            return
        if (
            sampling_info is not None
            and getattr(sampling_info, "sampling_seed", None) is not None
        ):
            base_positions = (seq_lens[:raw_bs] - 1).to(
                dtype=self.buffers.positions.dtype
            )
            deterministic_uniforms = welm_mtp_deterministic_uniforms(
                sampling_info=sampling_info,
                positions=base_positions,
                batch_size=raw_bs,
                width=self.speculative_num_steps,
                salt=3000,
            )
            if deterministic_uniforms is not None:
                self.buffers.uniform_samples[:, :bs].fill_(0.5)
                self.buffers.uniform_samples[:, :raw_bs, 0].copy_(
                    deterministic_uniforms.transpose(0, 1).contiguous()
                )
                # The hash is stateless and its seed/position inputs are the
                # same on all TP ranks, so fused prepare needs no collective.
                if (
                    self.synced_draft_generator is None
                    and self.model_runner.tp_group.world_size > 1
                ):
                    uniform_samples = self.buffers.uniform_samples[:, :bs]
                    broadcast_samples = uniform_samples.contiguous()
                    self.model_runner.tp_group.broadcast(broadcast_samples, src=0)
                    uniform_samples.copy_(broadcast_samples)
                return
        uniform_samples = self.buffers.uniform_samples[:, :bs]
        uniform_samples.uniform_(generator=self.synced_draft_generator)
        if (
            self.synced_draft_generator is None
            and self.model_runner.tp_group.world_size > 1
        ):
            broadcast_samples = uniform_samples.contiguous()
            self.model_runner.tp_group.broadcast(broadcast_samples, src=0)
            uniform_samples.copy_(broadcast_samples)

    def _copy_branch_cache_locs(self, raw_bs: int, bs: int) -> None:
        if self.topk <= 1:
            return
        buffers = self.buffers
        if (
            buffers.welm_mtp_branch_step_cache_locs is None
            or buffers.welm_mtp_branch_flat_cache_locs is None
        ):
            raise RuntimeError("Missing WeLM MTP branch cache loc graph buffers.")

        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        req_pool_indices = buffers.req_pool_indices[:bs].to(
            device=req_to_token.device, dtype=torch.long
        )
        seq_lens = buffers.seq_lens[:bs].to(
            device=req_to_token.device, dtype=torch.long
        )
        if raw_bs < bs:
            seq_lens = seq_lens.clone()
            seq_lens[raw_bs:bs].zero_()
        offsets = torch.arange(
            self.branch_tokens_per_bs,
            dtype=torch.long,
            device=req_to_token.device,
        )
        flat_cache_locs = req_to_token[
            req_pool_indices[:, None],
            seq_lens[:, None] + offsets[None, :],
        ].contiguous()
        branch_flat = buffers.welm_mtp_branch_flat_cache_locs[
            : bs * self.branch_tokens_per_bs
        ]
        branch_flat.copy_(flat_cache_locs.flatten())
        branch_step = (
            flat_cache_locs.reshape(bs, self.topk, self.speculative_num_steps)
            .permute(2, 0, 1)
            .reshape(self.speculative_num_steps, bs * self.topk)
            .contiguous()
        )
        buffers.welm_mtp_branch_step_cache_locs[:, : bs * self.topk].copy_(branch_step)

    def _init_tree_draft_attn_graph_metadata_capture(
        self,
        forward_batch: ForwardBatch,
        bs: int,
    ) -> None:
        if self.topk <= 1:
            return
        if self.eagle_worker.draft_attn_backend is None:
            raise RuntimeError("WeLM MTP topk>1 requires a draft attention backend.")
        buffers = self.buffers
        assert buffers.welm_mtp_branch_flat_cache_locs is not None
        branch_flat = buffers.welm_mtp_branch_flat_cache_locs[
            : bs * self.branch_tokens_per_bs
        ]
        original_forward_mode = forward_batch.forward_mode
        original_is_extend = forward_batch.is_extend_in_batch
        original_out_cache_loc = forward_batch.out_cache_loc
        original_num_tokens = forward_batch.spec_info.num_tokens_per_req
        original_num_logprob_tokens = (
            forward_batch.spec_info.num_tokens_for_logprob_per_req
        )
        try:
            forward_batch.forward_mode = ForwardMode.DECODE
            forward_batch.is_extend_in_batch = False
            forward_batch.out_cache_loc = branch_flat
            forward_batch.spec_info.num_tokens_per_req = self.topk
            forward_batch.spec_info.num_tokens_for_logprob_per_req = self.topk
            self.eagle_worker.draft_attn_backend.init_forward_metadata_capture_cuda_graph(
                forward_batch
            )
        finally:
            forward_batch.forward_mode = original_forward_mode
            forward_batch.is_extend_in_batch = original_is_extend
            forward_batch.out_cache_loc = original_out_cache_loc
            forward_batch.spec_info.num_tokens_per_req = original_num_tokens
            forward_batch.spec_info.num_tokens_for_logprob_per_req = (
                original_num_logprob_tokens
            )
        forward_batch.welm_mtp_branch_step_cache_locs = (
            buffers.welm_mtp_branch_step_cache_locs[:, : bs * self.topk]
        )
        forward_batch.welm_mtp_branch_flat_cache_locs = branch_flat
        forward_batch.welm_mtp_draft_tree_graph_metadata_ready = True

    def _init_tree_draft_attn_graph_metadata_replay(
        self,
        forward_batch: ForwardBatch,
        bs: int,
        raw_bs: int,
    ) -> None:
        if self.topk <= 1:
            return
        if self.eagle_worker.draft_attn_backend is None:
            raise RuntimeError("WeLM MTP topk>1 requires a draft attention backend.")
        self._copy_branch_cache_locs(raw_bs, bs)
        buffers = self.buffers
        assert buffers.welm_mtp_branch_flat_cache_locs is not None
        branch_flat = buffers.welm_mtp_branch_flat_cache_locs[
            : bs * self.branch_tokens_per_bs
        ]
        original_forward_mode = forward_batch.forward_mode
        original_is_extend = forward_batch.is_extend_in_batch
        original_out_cache_loc = forward_batch.out_cache_loc
        original_num_tokens = forward_batch.spec_info.num_tokens_per_req
        original_num_logprob_tokens = (
            forward_batch.spec_info.num_tokens_for_logprob_per_req
        )
        try:
            forward_batch.forward_mode = ForwardMode.DECODE
            forward_batch.is_extend_in_batch = False
            forward_batch.out_cache_loc = branch_flat
            forward_batch.spec_info.num_tokens_per_req = self.topk
            forward_batch.spec_info.num_tokens_for_logprob_per_req = self.topk
            self.eagle_worker.draft_attn_backend.init_forward_metadata_replay_cuda_graph(
                forward_batch, bs
            )
        finally:
            forward_batch.forward_mode = original_forward_mode
            forward_batch.is_extend_in_batch = original_is_extend
            forward_batch.out_cache_loc = original_out_cache_loc
            forward_batch.spec_info.num_tokens_per_req = original_num_tokens
            forward_batch.spec_info.num_tokens_for_logprob_per_req = (
                original_num_logprob_tokens
            )
        forward_batch.welm_mtp_branch_step_cache_locs = (
            buffers.welm_mtp_branch_step_cache_locs[:, : bs * self.topk]
        )
        forward_batch.welm_mtp_branch_flat_cache_locs = branch_flat
        forward_batch.welm_mtp_draft_tree_graph_metadata_ready = True

    def _select_topk(
        self,
        logits: torch.Tensor,
        step: int,
        bs: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if not self.sample_draft:
            if self.topk == 1:
                topk_index = torch.argmax(logits, dim=-1, keepdim=True)
                topk_p = torch.ones(
                    topk_index.shape,
                    dtype=logits.dtype,
                    device=logits.device,
                )
            else:
                probs = torch.softmax(logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            return topk_p, topk_index, None, None

        if self.fused_topk_sample and _is_cuda:
            fused = welmv4_mtp_fused_topk_softmax_sample(
                logits,
                self.buffers.temperature[:bs],
                self.buffers.uniform_samples[step, :bs],
                self.sampling_topk,
                top_p=self.buffers.top_p[:bs] if self.use_top_p else None,
            )
            if fused is not None:
                if self.distributed_topk:
                    sampled_p, sampled_pos, top_pos, top_probs = fused
                    candidate_ids = self.buffers.welm_mtp_candidate_indices[step, :bs]
                    return (
                        sampled_p,
                        torch.gather(candidate_ids, 1, sampled_pos),
                        torch.gather(candidate_ids, 1, top_pos),
                        top_probs,
                    )
                return fused

        top_logits, top_indices = torch.topk(
            logits.float(),
            k=self.sampling_topk,
            dim=-1,
            sorted=self.use_top_p,
        )
        temperature = self.buffers.temperature[:bs]
        probs = F.softmax(top_logits / temperature, dim=-1)
        if self.use_top_p:
            probs = top_p_renorm_prob(probs, self.buffers.top_p[:bs])
        cdf = torch.cumsum(probs, dim=-1)
        uniform = self.buffers.uniform_samples[step, :bs]
        sample_pos = torch.sum(cdf < uniform, dim=-1, keepdim=True).to(torch.int64)
        sample_pos = torch.clamp(sample_pos, max=self.sampling_topk - 1)
        topk_p = torch.gather(probs, dim=-1, index=sample_pos)
        topk_index = torch.gather(top_indices, dim=-1, index=sample_pos)
        if self.distributed_topk:
            candidate_ids = self.buffers.welm_mtp_candidate_indices[step, :bs]
            topk_index = torch.gather(candidate_ids, 1, topk_index)
            top_indices = torch.gather(candidate_ids, 1, top_indices)
        return topk_p, topk_index, top_indices, probs

    def _get_dp_cuda_graph_request_bs(
        self,
        forward_batch: ForwardBatch,
    ) -> Optional[int]:
        if not (self.require_mlp_tp_gather or self.use_token_owner):
            return int(forward_batch.batch_size)

        global_num_reqs = getattr(forward_batch, "global_num_reqs_cpu", None)
        if global_num_reqs:
            return max(int(count) for count in global_num_reqs)
        if self.use_token_owner:
            raise RuntimeError(
                "Token-owner MTP draft graph requires non-empty global request counts"
            )

        # Topk tree proposals can have a different row width from scheduler
        # token counts, so DP graph replay must use synchronized request counts.
        if self.topk > 1:
            return None

        global_num_tokens = getattr(forward_batch, "global_num_tokens_cpu", None)
        if global_num_tokens is None:
            return None
        max_num_tokens = max(int(count) for count in global_num_tokens)
        return (max_num_tokens + self.num_tokens_per_bs - 1) // self.num_tokens_per_bs

    def _update_num_token_non_padded(self, *, graph_bs: int, raw_bs: int) -> None:
        if not self.use_token_owner:
            return

        contracted_rows = self.contracted_rows_by_bs.get(graph_bs)
        if contracted_rows is None:
            valid_rows = padded_rows = graph_bs * self.num_tokens_per_bs
        else:
            valid_rows, padded_rows = raw_bs, contracted_rows

        attn_tp_size = get_attention_tp_size()
        attn_tp_rank = get_attention_tp_rank()
        rows_per_rank, remainder = divmod(padded_rows, attn_tp_size)
        local_start = rows_per_rank * attn_tp_rank + min(attn_tp_rank, remainder)
        local_size = rows_per_rank + int(attn_tp_rank < remainder)
        local_valid_rows = min(max(valid_rows - local_start, 0), local_size)
        self.buffers.num_token_non_padded.fill_(local_valid_rows)

    def _reject_token_owner_graph_miss(self, reason: str) -> bool:
        if self.use_token_owner:
            raise RuntimeError(
                f"Token-owner MTP draft proposal CUDA graph {reason}; "
                "eager fallback is disabled"
            )
        return False

    def _required_draft_sampling_mode(
        self, forward_batch: ForwardBatch
    ) -> tuple[bool, bool]:
        """Return the rank-consistent (sample, use_top_p) proposal mode."""
        if not self.use_dp_sampling_consensus:
            should_sample = self.eagle_worker._should_sample_welmv4_mtp_draft(
                forward_batch
            )
            should_use_top_p = (
                should_sample
                and self.eagle_worker._should_use_welmv4_mtp_draft_top_p(forward_batch)
            )
            return should_sample, should_use_top_p

        should_sample = (
            self.eagle_worker._is_welmv4_mtp_draft_sampling_enabled()
            and self.topk == 1
            and (
                self.eagle_worker._has_welmv4_mtp_fixed_draft_sampling_params()
                or forward_batch.global_has_non_greedy_sampling
            )
        )
        if not should_sample:
            return False, False

        fixed_top_p = self.eagle_worker.welmv4_mtp_draft_fixed_top_p
        should_use_top_p = self.supports_draft_top_p and (
            fixed_top_p < 1.0
            if fixed_top_p is not None
            else forward_batch.global_needs_top_p_sampling
        )
        return True, should_use_top_p

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        if not forward_batch.forward_mode.is_draft_extend(include_v2=True):
            return False
        required_sampling_mode = self._required_draft_sampling_mode(forward_batch)
        mode_graphs = self.graphs_by_mode.get(required_sampling_mode)
        if mode_graphs is None:
            return self._reject_token_owner_graph_miss(
                "sampling mode mismatch "
                f"(required={required_sampling_mode}, "
                f"captured={sorted(self.graphs_by_mode)})"
            )
        if forward_batch.input_ids is None or forward_batch.out_cache_loc is None:
            return False
        raw_bs = int(forward_batch.batch_size)
        raw_num_tokens = int(forward_batch.input_ids.numel())
        if not (0 <= raw_num_tokens <= raw_bs * self.num_tokens_per_bs):
            return False
        if raw_bs == 0 and not self.require_mlp_sync:
            return False
        if self.require_mlp_tp_gather or self.use_token_owner:
            cuda_graph_bs = self._get_dp_cuda_graph_request_bs(forward_batch)
            if cuda_graph_bs is None or cuda_graph_bs <= 0:
                return False
        else:
            cuda_graph_bs = raw_bs
        has_capture_bucket = (
            cuda_graph_bs in mode_graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )
        if not has_capture_bucket:
            return self._reject_token_owner_graph_miss(
                f"has no capture bucket for request batch size {cuda_graph_bs}"
            )
        if self.require_mlp_sync and not forward_batch.can_run_dp_cuda_graph:
            return self._reject_token_owner_graph_miss(
                "was scheduler disabled for this DP batch"
            )
        spec_info = forward_batch.spec_info
        if not isinstance(spec_info, EagleDraftInput):
            return False
        if (
            spec_info.num_accept_tokens_cpu is None
            or spec_info.num_accept_tokens is None
            or spec_info.num_correct_drafts is None
        ):
            return False
        accepted_lens_cpu = [int(x) for x in spec_info.num_accept_tokens_cpu]
        if len(accepted_lens_cpu) != raw_bs:
            return False
        index = bisect.bisect_left(self.capture_bs, cuda_graph_bs)
        if index >= len(self.capture_bs):
            return self._reject_token_owner_graph_miss(
                f"has no capture bucket for request batch size {cuda_graph_bs}"
            )
        graph_bs = self.capture_bs[index]
        if sum(accepted_lens_cpu) != raw_num_tokens:
            return False
        return self._make_padded_accept_lens(accepted_lens_cpu, graph_bs) is not None

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        # Mirror CudaGraphRunner._capture_graph: tag the captured graph's
        # private memory pool with GPU_MEMORY_TYPE_CUDA_GRAPH when
        # torch_memory_saver is enabled (enable_memory_saver and
        # SGLANG_MEMORY_SAVER_CUDA_GRAPH), so release_memory_occupation /
        # pause(GPU_MEMORY_TYPE_CUDA_GRAPH) can release it. Otherwise fall
        # back to torch.cuda.graph (previous behavior).
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.model_runner.server_args.enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        graph_ctx = (
            partial(memory_saver_adapter.cuda_graph, tag=GPU_MEMORY_TYPE_CUDA_GRAPH)
            if memory_saver_adapter.enabled
            else torch.cuda.graph
        )
        with graph_ctx(cuda_graph=graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _replay(self):
        ctx = (
            self.model_runner.device_timer.wrap(
                metadata={"category": "welm_mtp_draft_proposal"}
            )
            if self.model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            self.graphs[self.bs].replay()

    def _seq_len_fill_values_for_bs(self, bs: int) -> List[int]:
        return self.seq_len_fill_values_by_bs.get(int(bs), [self.seq_len_fill_value])

    def _graph_seq_len_key(self, seq_len_fill_value: Optional[int]):
        # Draft proposal graphs are not bucketed by seq_len (no AttnCP sharded
        # KV, non-decode mode). Returning None keeps replay's
        # self.graphs[self.bs] lookup consistent with capture's graph key
        # (no "_s{N}" suffix is appended).
        return None

    def _post_process_after_profile(self, prof):
        # Reuse CudaGraphRunner's implementation for parity when
        # enable_profile_cuda_graph is set.
        CudaGraphRunner._post_process_after_profile(self, prof)

    def capture(self):
        for mode in self.capture_sampling_modes:
            self.sample_draft, self.use_top_p = mode
            self.graphs = self.graphs_by_mode.setdefault(mode, {})
            self.output_buffers = self.output_buffers_by_mode.setdefault(mode, {})
            CudaGraphRunner.capture(self)
        self._activate_sampling_mode(self.capture_sampling_modes[0])

    def _activate_sampling_mode(self, mode: tuple[bool, bool]) -> None:
        self.sample_draft, self.use_top_p = mode
        self.graphs = self.graphs_by_mode[mode]
        self.output_buffers = self.output_buffers_by_mode[mode]

    def capture_one_batch_size(
        self,
        bs: int,
        forward: Callable,
        stream_idx: int = 0,
        seq_len_fill_value: Optional[int] = None,
    ):
        fill_value = (
            int(seq_len_fill_value)
            if seq_len_fill_value is not None
            else self.seq_len_fill_value
        )
        buffers = self.buffers
        graph = self._create_graph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs
        forward_mode = ForwardMode.DRAFT_EXTEND

        input_ids = buffers.input_ids[:num_tokens]
        first_input_ids = buffers.first_input_ids[:bs]
        req_pool_indices = buffers.req_pool_indices[:bs]
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        out_cache_loc_swa = (
            None
            if buffers.out_cache_loc_swa is None
            else buffers.out_cache_loc_swa[:num_tokens]
        )
        positions = buffers.positions[:num_tokens]
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        hidden_states = buffers.hidden_states[:num_tokens]
        mirrored_kv_indices = buffers.mirrored_kv_indices[:num_tokens]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        extend_seq_lens = buffers.extend_seq_lens[:bs]
        extend_seq_lens_cpu = self.extend_seq_lens_cpu[:bs]
        num_correct_drafts = buffers.num_correct_drafts[:bs]
        num_accept_tokens = buffers.num_accept_tokens[:bs]
        custom_last_index = buffers.custom_last_index[:bs]
        custom_last_cache_loc = buffers.custom_last_cache_loc[:bs]
        next_token_logits_buffer = buffers.next_token_logits_buffer[:bs]
        capture_token_counts = (
            [num_tokens] * self.dp_size if self.use_token_owner else None
        )
        capture_request_counts = [bs] * self.dp_size if self.use_token_owner else None

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor([num_tokens], dtype=torch.int32, device=input_ids.device)
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor([num_tokens], dtype=torch.int32, device=input_ids.device)
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
            global_dp_buffer_len = num_tokens
        else:
            global_num_tokens = None
            global_num_tokens_for_logprob = None
            global_dp_buffer_len = None

        capture_hidden_mode = CaptureHiddenMode.LAST
        self._update_num_token_non_padded(graph_bs=bs, raw_bs=bs)

        spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            mirrored_kv_indices=mirrored_kv_indices,
            num_correct_drafts=num_correct_drafts,
            num_accept_tokens=num_accept_tokens,
            num_correct_drafts_cpu=[self.num_tokens_per_bs - 1] * bs,
            num_accept_tokens_cpu=[self.num_tokens_per_bs] * bs,
            capture_hidden_mode=capture_hidden_mode,
            num_tokens_per_req=self.num_tokens_per_bs,
            num_tokens_for_logprob_per_req=self.num_tokens_per_bs,
            model_specific_states=self._welmv4_mtp_graph_model_specific_states(),
        )
        spec_info.extend_seq_lens_cpu = list(extend_seq_lens_cpu)
        spec_info.extend_seq_lens_tensor = extend_seq_lens
        forward_batch = ForwardBatch(
            forward_mode=forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            out_cache_loc=out_cache_loc,
            out_cache_loc_swa=out_cache_loc_swa,
            seq_lens_sum=int(fill_value) * bs,
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            original_global_num_tokens_cpu=capture_token_counts,
            global_num_tokens_cpu=capture_token_counts,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            global_num_reqs_cpu=capture_request_counts,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            model_specific_states=spec_info.model_specific_states,
            capture_hidden_mode=capture_hidden_mode,
            attn_backend=self.draft_extend_attn_backend,
            oe_context=None,
            num_token_non_padded=buffers.num_token_non_padded,
        )
        forward_batch.custom_last_index = custom_last_index
        forward_batch.custom_last_cache_loc = custom_last_cache_loc
        forward_batch.welm_mtp_identity_indices = buffers.identity_indices
        forward_batch._welm_force_low_latency_deepep = self.use_token_owner_deepep
        if self.distributed_topk and self.sample_draft:
            forward_batch.welm_mtp_distributed_topk = self.sampling_topk
            forward_batch.welm_mtp_candidate_indices_buffer = (
                buffers.welm_mtp_candidate_indices[:, :bs]
            )
        forward_batch.welm_mtp_draft_extend_metadata_bs = bs
        if self.topk == 1:
            forward_batch.welm_mtp_linear_proposal_fast_path = True
            forward_batch.welm_mtp_linear_proposal_parent_list = (
                buffers.linear_proposal_parent_list[:bs]
            )
            forward_batch.welm_mtp_linear_proposal_top_scores_index = (
                buffers.linear_proposal_top_scores_index[:bs]
            )
            if self.fused_linear_graph_outputs:
                forward_batch.welm_mtp_linear_graph_output_buffers = {
                    "bonus_tokens": first_input_ids,
                    "base_positions": buffers.base_positions[:bs],
                    "topk_p": buffers.linear_proposal_topk_p[:bs],
                    "topk_index": buffers.linear_proposal_topk_index[:bs],
                    "proposal_tokens": buffers.linear_proposal_tokens[:bs],
                    "draft_topk_indices": buffers.linear_proposal_draft_topk_indices[
                        :bs
                    ],
                    "draft_topk_values": buffers.linear_proposal_draft_topk_values[:bs],
                    "verify_tokens": buffers.linear_verify_tokens[
                        : bs * self.num_tokens_per_bs
                    ],
                    "verify_positions": buffers.linear_verify_positions[
                        : bs * self.num_tokens_per_bs
                    ],
                }
        if self.topk > 1:
            forward_batch.welm_mtp_first_next_token_logits_buffer = (
                buffers.next_token_logits_buffer[:bs]
            )
            forward_batch.welm_mtp_branch_next_token_logits_buffer = (
                buffers.next_token_logits_buffer[: bs * self.topk]
            )
        forward_batch.enable_welm_kv_mirror_opt = (
            self.model_runner.server_args.enable_welm_kv_mirror_opt
        )
        if buffers.welm_mtp_oe_hash_out is not None:
            oe_work_history_size = num_tokens if self.topk == 1 else bs
            forward_batch.welm_oe_decode_hashed_inputs = buffers.welm_mtp_oe_hash_out[
                :, :num_tokens
            ]
            forward_batch.welm_mtp_oe_work_history = [
                buffers.welm_mtp_oe_work_history_a[:oe_work_history_size],
                buffers.welm_mtp_oe_work_history_b[:oe_work_history_size],
            ]
            forward_batch.welm_mtp_oe_entry_history_state = (
                buffers.welm_mtp_oe_entry_history[:bs]
            )
            forward_batch.welm_mtp_oe_parent_scratch = (
                buffers.welm_mtp_oe_parent_indices[:num_tokens]
            )
            forward_batch.welm_mtp_oe_prev_input_ids = (
                buffers.welm_mtp_oe_prev_input_ids[:num_tokens]
            )
            forward_batch.welm_mtp_oe_prev_prev_input_ids = (
                buffers.welm_mtp_oe_prev_prev_input_ids[:num_tokens]
            )
            forward_batch.welm_mtp_oe_output_prev_input_ids = (
                buffers.welm_mtp_oe_output_prev_input_ids[:num_tokens]
            )
            forward_batch.welm_mtp_oe_hash_out_batch_major = (
                buffers.welm_mtp_oe_hash_out_batch_major[:num_tokens]
            )
            forward_batch.welm_mtp_draft_input_ids = buffers.welm_mtp_draft_input_ids[
                :, :num_tokens
            ]
        if self.topk > 1:
            forward_batch.welm_mtp_branch_step_cache_locs = (
                buffers.welm_mtp_branch_step_cache_locs[:, : bs * self.topk]
            )
            forward_batch.welm_mtp_branch_flat_cache_locs = (
                buffers.welm_mtp_branch_flat_cache_locs[
                    : bs * self.branch_tokens_per_bs
                ]
            )

        self.draft_extend_attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=None,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )
        self._set_welmv4_mtp_mirror_metadata(bs)
        self._init_tree_draft_attn_graph_metadata_capture(
            forward_batch,
            bs,
        )

        def run_once():
            if out_cache_loc_swa is not None:
                from sglang.srt.speculative.welmv4_mtp_staging import (
                    translate_welm_mtp_cache_loc,
                )

                translate_welm_mtp_cache_loc(
                    full_cache_loc=out_cache_loc,
                    full_to_swa=(
                        self.model_runner.token_to_kv_pool_allocator.full_to_swa_index_mapping
                    ),
                    swa_cache_loc=out_cache_loc_swa,
                )
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            if global_num_tokens is not None:
                global_num_tokens.fill_(num_tokens)
            if global_num_tokens_for_logprob is not None:
                global_num_tokens_for_logprob.fill_(num_tokens)
            forward_batch.global_dp_buffer_len = global_dp_buffer_len
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            hidden_states_backup = forward_batch.spec_info.hidden_states
            forward_batch.welm_mtp_draft_graph_select_fn = partial(
                self._select_topk, bs=bs
            )
            forward_batch.welm_mtp_skip_draft_proposal_build = True
            try:
                self.eagle_worker._run_welmv4_mtp_merged_extend_draft(
                    forward_batch,
                    first_input_ids,
                    skip_attn_backend_init=True,
                    first_query_hashed_inputs=(
                        None
                        if buffers.welm_mtp_query_hash_inputs is None
                        else buffers.welm_mtp_query_hash_inputs[:, :bs]
                    ),
                    first_query_history_state=(
                        None
                        if buffers.welm_mtp_oe_entry_history is None
                        else buffers.welm_mtp_oe_entry_history[:bs]
                    ),
                    draft_path="decode_graph",
                )
            finally:
                forward_batch.welm_mtp_draft_graph_select_fn = None
                forward_batch.welm_mtp_skip_draft_proposal_build = False
            out = (
                forward_batch.spec_info.topk_p,
                forward_batch.spec_info.topk_index,
                forward_batch.spec_info.hidden_states,
                forward_batch.spec_info.draft_probs,
                forward_batch.spec_info.welm_mtp_draft_topk_indices,
                forward_batch.spec_info.welm_mtp_draft_topk_values,
                forward_batch.spec_info.draft_proposal_parent_list,
                forward_batch.spec_info.draft_proposal_top_scores_index,
                forward_batch.spec_info.draft_proposal_tokens,
            )
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return out

        self.deepep_adapter.capture(is_extend_in_batch=not self.use_token_owner_deepep)
        if out_cache_loc_swa is not None:
            self.model_runner.token_to_kv_pool.set_swa_loc(out_cache_loc_swa)
        self._capture_init(run_once)
        contracted_rows = getattr(
            forward_batch, "_welm_mtp_contracted_dp_metadata_rows", None
        )
        self.contracted_rows_by_bs[bs] = (
            None if contracted_rows is None else int(contracted_rows)
        )
        self._update_num_token_non_padded(graph_bs=bs, raw_bs=bs)
        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )
        set_global_graph_memory_pool(graph.pool())
        self.forward_batches[bs] = forward_batch
        return graph, out

    def can_replay_from_verify(self, batch, batch_result) -> bool:
        """Whether the fixed top-k=1 verify result can bypass ragged staging."""
        if not self.persistent_handoff or self.topk != 1 or self.dp_size != 1:
            return False
        required_sampling_mode = self._required_draft_sampling_mode(batch)
        if required_sampling_mode not in self.graphs_by_mode:
            return False
        if batch.forward_mode.is_idle() or batch.seq_lens_cpu is None:
            return False
        raw_bs = int(batch.seq_lens.shape[0])
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        if raw_bs <= 0 or index >= len(self.capture_bs):
            return False
        accept_index = getattr(batch_result, "spec_accept_index", None)
        accept_lens = getattr(batch_result, "accept_lens", None)
        logits_output = getattr(batch_result, "logits_output", None)
        return bool(
            accept_index is not None
            and accept_index.ndim == 2
            and int(accept_index.shape[0]) == raw_bs
            and int(accept_index.shape[1]) >= self.num_tokens_per_bs
            and accept_lens is not None
            and int(accept_lens.numel()) == raw_bs
            and getattr(logits_output, "hidden_states", None) is not None
            and batch.out_cache_loc is not None
            and batch.req_pool_indices is not None
        )

    def replay_from_verify(
        self,
        batch,
        batch_result,
        next_draft_input: EagleDraftInput,
        *,
        entry_history_state: Optional[torch.Tensor],
    ) -> None:
        """Replay proposal graph from target outputs without a temporary batch.

        This is the latency-critical top-k=1 path.  All graph inputs are
        filled in their final fixed-width layout on device.  There is no
        accept-lens D2H synchronization, boolean compaction, temporary
        ``ModelWorkerBatch`` mutation, or ``ForwardBatch.init_new``.
        """
        if not self.can_replay_from_verify(batch, batch_result):
            raise RuntimeError("Persistent WeLM MTP handoff preconditions are not met.")

        self._activate_sampling_mode(self._required_draft_sampling_mode(batch))
        self.deepep_adapter.replay()
        buffers = self.buffers
        raw_bs = int(batch.seq_lens.shape[0])
        index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        graph_num_tokens = bs * self.num_tokens_per_bs
        accept_lens = batch_result.accept_lens
        accept_index = batch_result.spec_accept_index
        target_hidden_states = batch_result.logits_output.hidden_states

        from sglang.srt.speculative.welmv4_mtp_staging import (
            gather_welm_mtp_last_hash,
            pack_welm_mtp_verify_handoff,
        )

        pack_welm_mtp_verify_handoff(
            predict=batch_result.next_token_ids,
            accept_index=accept_index,
            accept_lens=accept_lens,
            target_cache_loc=batch.out_cache_loc,
            target_hidden_states=target_hidden_states,
            old_seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
            bonus_tokens=next_draft_input.bonus_tokens,
            output_input_ids=buffers.input_ids,
            output_cache_loc=buffers.out_cache_loc,
            output_positions=buffers.positions,
            output_hidden_states=buffers.hidden_states,
            output_mirrored_kv_indices=buffers.mirrored_kv_indices,
            output_seq_lens=buffers.seq_lens,
            output_req_pool_indices=buffers.req_pool_indices,
            output_first_input_ids=buffers.first_input_ids,
            output_base_positions=buffers.base_positions,
            custom_last_index=buffers.custom_last_index,
            custom_last_cache_loc=buffers.custom_last_cache_loc,
            raw_bs=raw_bs,
            graph_bs=bs,
            tokens_per_bs=self.num_tokens_per_bs,
            mirror_padding_index=self.welm_mtp_mirror_padding_index,
            seq_len_fill_value=self.seq_len_fill_value,
        )
        if buffers.welm_mtp_oe_hash_out is not None:
            accepted_draft_token_ids = getattr(
                batch_result, "welm_mtp_accepted_draft_token_ids", None
            )
            if accepted_draft_token_ids is None or entry_history_state is None:
                raise RuntimeError(
                    "Persistent WeLM MTP handoff requires accepted draft tokens "
                    "and the compact OE entry history."
                )
            from sglang.jit_kernel.welm_oe import (
                welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda,
            )

            oe_grams, oe_vocab_sizes = get_welm_oe_hash_config(
                self.model_runner.model_config
            )
            next_history = buffers.welm_mtp_oe_work_history_a[:raw_bs]
            welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda(
                buffers.input_ids[: raw_bs * self.num_tokens_per_bs],
                accepted_draft_token_ids.reshape(raw_bs, -1),
                accept_lens,
                entry_history_state[:raw_bs],
                oe_grams,
                oe_vocab_sizes,
                buffers.welm_mtp_oe_hash_out[:, : raw_bs * self.num_tokens_per_bs],
                next_history,
                self.model_runner.model_config.vocab_size,
                self.num_tokens_per_bs,
                use_entry_history_for_extend_hash_prefix=True,
            )
            buffers.welm_mtp_oe_entry_history[:raw_bs].copy_(next_history)
            gather_welm_mtp_last_hash(
                buffers.welm_mtp_oe_hash_out,
                accept_lens,
                buffers.welm_mtp_query_hash_inputs,
                raw_bs=raw_bs,
                tokens_per_bs=self.num_tokens_per_bs,
            )
            if raw_bs < bs:
                buffers.welm_mtp_query_hash_inputs[:, raw_bs:bs].zero_()
                buffers.welm_mtp_oe_entry_history[raw_bs:bs].zero_()
            next_draft_input.welm_mtp_oe_history_state = (
                buffers.welm_mtp_oe_entry_history[:raw_bs]
            )
        self._copy_welmv4_mtp_mirror_kv_states(
            None,
            graph_num_tokens,
            model_specific_states=batch_result.logits_output.model_specific_states,
            mirrored_kv_indices=buffers.mirrored_kv_indices[:graph_num_tokens],
        )
        self._copy_sampling_params_from_info(batch.sampling_info, raw_bs, bs)
        self._copy_sampling_randomness_from_info(
            batch.sampling_info, buffers.seq_lens, raw_bs, bs
        )
        # The GPU sequence lengths are exact.  CPU lengths are used only to set
        # a conservative attention maximum; avoiding accept_lens.cpu() is the key
        # synchronization win here.
        buffers.seq_lens_cpu[:raw_bs].copy_(
            batch.seq_lens_cpu[:raw_bs] + self.num_tokens_per_bs
        )
        if raw_bs < bs:
            buffers.seq_lens_cpu[raw_bs:bs].fill_(self.seq_len_fill_value)
        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(graph_num_tokens)
            buffers.global_num_tokens_for_logprob_gpu.fill_(graph_num_tokens)
        self._update_num_token_non_padded(graph_bs=bs, raw_bs=raw_bs)
        replay_forward_batch = self.forward_batches[bs]
        replay_forward_batch.spec_info.extend_seq_lens_cpu = list(
            self.extend_seq_lens_cpu[:bs]
        )
        replay_forward_batch.spec_info.extend_seq_lens_tensor = buffers.extend_seq_lens[
            :bs
        ]
        replay_forward_batch.seq_lens_sum = (
            int(batch.seq_lens_sum)
            + (raw_bs * self.num_tokens_per_bs)
            + (bs - raw_bs) * int(self.seq_len_fill_value)
        )
        self.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_sum=replay_forward_batch.seq_lens_sum,
            encoder_lens=None,
            forward_mode=replay_forward_batch.forward_mode,
            spec_info=replay_forward_batch.spec_info,
            seq_lens_cpu=buffers.seq_lens_cpu,
        )
        self._set_welmv4_mtp_mirror_metadata(bs)
        self._init_tree_draft_attn_graph_metadata_replay(
            replay_forward_batch, bs, raw_bs
        )
        self.raw_bs = raw_bs
        self.bs = bs
        self._replay()
        (
            topk_p,
            topk_index,
            hidden_states,
            draft_probs,
            draft_topk_indices,
            draft_topk_values,
            proposal_parent_list,
            proposal_top_scores_index,
            proposal_tokens,
        ) = self.output_buffers[bs]
        next_draft_input.topk_p = topk_p[:raw_bs]
        next_draft_input.topk_index = topk_index[:raw_bs]
        next_draft_input.hidden_states = hidden_states[:raw_bs]
        next_draft_input.draft_probs = (
            None if draft_probs is None else draft_probs[:raw_bs]
        )
        next_draft_input.welm_mtp_draft_topk_indices = (
            None if draft_topk_indices is None else draft_topk_indices[:raw_bs]
        )
        next_draft_input.welm_mtp_draft_topk_values = (
            None if draft_topk_values is None else draft_topk_values[:raw_bs]
        )
        next_draft_input.draft_proposal_parent_list = (
            None if proposal_parent_list is None else proposal_parent_list[:raw_bs]
        )
        next_draft_input.draft_proposal_top_scores_index = (
            None
            if proposal_top_scores_index is None
            else proposal_top_scores_index[:raw_bs]
        )
        next_draft_input.draft_proposal_tokens = (
            None if proposal_tokens is None else proposal_tokens[:raw_bs]
        )
        next_draft_input.welm_mtp_linear_verify_ready = bool(
            self.fused_linear_graph_outputs
        )
        next_draft_input.welm_mtp_base_positions = buffers.base_positions[:raw_bs]
        if self.fused_linear_graph_outputs:
            num_verify_tokens = raw_bs * self.num_tokens_per_bs
            prebuilt_verify = EagleVerifyInput(
                draft_token=buffers.linear_verify_tokens[:num_verify_tokens],
                custom_mask=buffers.linear_verify_mask[
                    : num_verify_tokens * self.num_tokens_per_bs
                ],
                positions=buffers.linear_verify_positions[:num_verify_tokens],
                retrieve_index=buffers.linear_verify_retrieve_index[:raw_bs],
                retrieve_next_token=buffers.linear_verify_retrieve_next_token[:raw_bs],
                retrieve_next_sibling=buffers.linear_verify_retrieve_next_sibling[
                    :raw_bs
                ],
                retrieve_cum_len=None,
                spec_steps=self.speculative_num_steps,
                topk=self.topk,
                draft_token_num=self.speculative_num_draft_tokens,
                capture_hidden_mode=None,
                seq_lens_sum=None,
                seq_lens_cpu=None,
                draft_probs=None,
                draft_topk_indices=(
                    None if draft_topk_indices is None else draft_topk_indices[:raw_bs]
                ),
                draft_topk_values=(
                    None if draft_topk_values is None else draft_topk_values[:raw_bs]
                ),
            )
            next_history_state = getattr(
                next_draft_input, "welm_mtp_oe_history_state", None
            )
            if next_history_state is not None:
                prebuilt_verify.welm_mtp_oe_history_state = next_history_state
            prebuilt_verify.welm_mtp_oe_hash_seq_lens = (
                buffers.linear_verify_hash_seq_lens[:raw_bs]
            )
            next_draft_input.welm_mtp_prebuilt_verify_input = prebuilt_verify
            next_draft_input.welm_mtp_prebuilt_verify_bs = raw_bs

    def replay(
        self,
        forward_batch: ForwardBatch,
        first_input_ids: torch.Tensor,
        *,
        first_query_hashed_inputs: Optional[torch.Tensor],
        first_query_history_state: Optional[torch.Tensor],
    ) -> None:
        self._activate_sampling_mode(self._required_draft_sampling_mode(forward_batch))
        self.deepep_adapter.replay()
        buffers = self.buffers
        raw_bs = int(forward_batch.batch_size)

        if self.require_mlp_tp_gather or self.use_token_owner:
            cuda_graph_bs = self._get_dp_cuda_graph_request_bs(forward_batch)
            if cuda_graph_bs is None:
                raise RuntimeError(
                    "WeLM MTP draft proposal graph replay requires synchronized "
                    "request counts when topk tree capacity differs from draft tokens."
                )
            if cuda_graph_bs <= 0:
                raise RuntimeError(
                    "WeLM MTP draft proposal graph replay got an empty global "
                    "request count."
                )
            index = bisect.bisect_left(self.capture_bs, cuda_graph_bs)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        graph_num_tokens = bs * self.num_tokens_per_bs
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        accepted_lens_cpu = [int(x) for x in spec_info.num_accept_tokens_cpu]
        padded_lens_cpu = self._make_padded_accept_lens(accepted_lens_cpu, bs)
        if spec_info.hidden_states is None:
            raise RuntimeError(
                "WeLM MTP draft proposal graph replay requires hidden states."
            )
        cached_hash = getattr(forward_batch, "welm_oe_decode_hashed_inputs", None)
        if (
            cached_hash is None
            and raw_bs == 0
            and buffers.welm_mtp_oe_hash_out is not None
        ):
            # Idle dp-attention cycles replay this graph for collective
            # alignment without OE hashes (idle skips draft-extend hash prep).
            # An empty view keeps the fused pack invariant and its kernel
            # zero-fills every padding row's output hash.
            cached_hash = buffers.welm_mtp_oe_hash_out[:, :0]
        if (
            buffers.welm_mtp_oe_hash_out is not None
            and cached_hash is None
            and raw_bs > 0
        ):
            raise RuntimeError(
                "WeLM MTP draft proposal graph replay is missing dense OE hashes."
            )

        if self.fused_prepare:
            # Real rows are overwritten below and OE work buffers are outputs of
            # captured kernels.  Initialize only graph-padding rows; clearing all
            # scratch tensors here launched nine redundant kernels per replay.
            if raw_bs < bs:
                buffers.seq_lens[raw_bs:bs].fill_(self.seq_len_fill_value)
                buffers.seq_lens_cpu[raw_bs:bs].fill_(self.seq_len_fill_value)
                buffers.req_pool_indices[raw_bs:bs].zero_()
                buffers.first_input_ids[raw_bs:bs].zero_()
                if buffers.welm_mtp_oe_hash_out is not None:
                    buffers.welm_mtp_query_hash_inputs[:, raw_bs:bs].zero_()
                    buffers.welm_mtp_oe_entry_history[raw_bs:bs].zero_()
        else:
            buffers.seq_lens[:bs].fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:bs].fill_(self.seq_len_fill_value)
            buffers.req_pool_indices[:bs].zero_()
            buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
            buffers.first_input_ids[:bs].zero_()
            if buffers.welm_mtp_oe_hash_out is not None:
                buffers.welm_mtp_query_hash_inputs[:, :bs].zero_()
                buffers.welm_mtp_oe_entry_history[:bs].zero_()
                buffers.welm_mtp_oe_work_history_a[:graph_num_tokens].zero_()
                buffers.welm_mtp_oe_work_history_b[:graph_num_tokens].zero_()
                buffers.welm_mtp_oe_parent_indices[:graph_num_tokens].zero_()
                buffers.welm_mtp_oe_prev_input_ids[:graph_num_tokens].zero_()
                buffers.welm_mtp_oe_prev_prev_input_ids[:graph_num_tokens].zero_()
                buffers.welm_mtp_oe_output_prev_input_ids[:graph_num_tokens].zero_()
                buffers.welm_mtp_oe_hash_out_batch_major[:graph_num_tokens].zero_()
        if padded_lens_cpu is None:
            raise RuntimeError(
                "Cannot fit WeLM MTP draft proposal accepted lengths into graph "
                f"shape: bs={bs}, accepted_lens={accepted_lens_cpu}."
            )
        self._copy_runtime_flat_layout(
            input_ids=forward_batch.input_ids,
            out_cache_loc=forward_batch.out_cache_loc,
            positions=forward_batch.positions,
            hidden_states=spec_info.hidden_states,
            mirrored_kv_indices=getattr(spec_info, "mirrored_kv_indices", None),
            cached_hash=cached_hash,
            accept_lens=spec_info.num_accept_tokens,
            extend_start_loc=forward_batch.extend_start_loc,
            accepted_lens_cpu=accepted_lens_cpu,
            padded_lens_cpu=padded_lens_cpu,
            raw_bs=raw_bs,
            graph_num_tokens=graph_num_tokens,
        )
        self._copy_welmv4_mtp_mirror_kv_states(forward_batch, graph_num_tokens)
        self._copy_sampling_params(forward_batch, raw_bs, bs)
        self._copy_sampling_randomness(forward_batch, raw_bs, bs)
        buffers.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        if forward_batch.seq_lens_cpu is not None:
            buffers.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
        buffers.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        buffers.first_input_ids[:raw_bs].copy_(first_input_ids)
        if self.fused_prepare:
            if any(x != self.num_tokens_per_bs for x in padded_lens_cpu):
                raise RuntimeError(
                    "Fused WeLM MTP staging requires full fixed graph rows: "
                    f"padded_lens={padded_lens_cpu}."
                )
        else:
            padded_lens = torch.tensor(
                padded_lens_cpu,
                dtype=torch.int32,
                device=buffers.extend_seq_lens.device,
            )
            buffers.extend_seq_lens[:bs].copy_(padded_lens)
            buffers.num_accept_tokens[:bs].copy_(padded_lens)
            buffers.num_correct_drafts[:bs].copy_(padded_lens - 1)
        self.extend_seq_lens_cpu[:bs] = padded_lens_cpu

        if buffers.welm_mtp_oe_hash_out is not None and raw_bs > 0:
            if first_query_hashed_inputs is None or first_query_history_state is None:
                raise RuntimeError(
                    "WeLM MTP draft proposal graph replay is missing query OE state."
                )
            buffers.welm_mtp_query_hash_inputs[:, :raw_bs].copy_(
                first_query_hashed_inputs
            )
            buffers.welm_mtp_oe_entry_history[:raw_bs].copy_(first_query_history_state)

        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(graph_num_tokens)
            buffers.global_num_tokens_for_logprob_gpu.fill_(graph_num_tokens)
        self._update_num_token_non_padded(graph_bs=bs, raw_bs=raw_bs)
        replay_forward_batch = self.forward_batches[bs]
        replay_forward_batch.spec_info.extend_seq_lens_cpu = list(
            self.extend_seq_lens_cpu[:bs]
        )
        replay_forward_batch.spec_info.extend_seq_lens_tensor = buffers.extend_seq_lens[
            :bs
        ]
        replay_forward_batch.seq_lens_sum = forward_batch.seq_lens_sum + (
            bs - raw_bs
        ) * int(self.seq_len_fill_value)
        self.draft_extend_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_sum=replay_forward_batch.seq_lens_sum,
            encoder_lens=None,
            forward_mode=replay_forward_batch.forward_mode,
            spec_info=replay_forward_batch.spec_info,
            seq_lens_cpu=buffers.seq_lens_cpu,
        )
        self._set_welmv4_mtp_mirror_metadata(bs)
        self._init_tree_draft_attn_graph_metadata_replay(
            replay_forward_batch,
            bs,
            raw_bs,
        )
        self.raw_bs = raw_bs
        self.bs = bs
        self._replay()
        (
            topk_p,
            topk_index,
            hidden_states,
            draft_probs,
            draft_topk_indices,
            draft_topk_values,
            proposal_parent_list,
            proposal_top_scores_index,
            proposal_tokens,
        ) = self.output_buffers[bs]

        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        # CUDA-graph outputs are persistent runner-owned buffers.  The proposal
        # is consumed by draft() before the next replay, so views are sufficient
        # and avoid eight tiny device-to-device clone launches per cycle.
        spec_info.topk_p = topk_p[:raw_bs]
        spec_info.topk_index = topk_index[:raw_bs]
        spec_info.hidden_states = hidden_states[:raw_bs]
        spec_info.draft_probs = None if draft_probs is None else draft_probs[:raw_bs]
        spec_info.welm_mtp_draft_topk_indices = (
            None if draft_topk_indices is None else draft_topk_indices[:raw_bs]
        )
        spec_info.welm_mtp_draft_topk_values = (
            None if draft_topk_values is None else draft_topk_values[:raw_bs]
        )
        spec_info.draft_proposal_parent_list = (
            None if proposal_parent_list is None else proposal_parent_list[:raw_bs]
        )
        spec_info.draft_proposal_top_scores_index = (
            None
            if proposal_top_scores_index is None
            else proposal_top_scores_index[:raw_bs]
        )
        spec_info.draft_proposal_tokens = (
            None if proposal_tokens is None else proposal_tokens[:raw_bs]
        )
        base_position_index = buffers.custom_last_index[:raw_bs]
        spec_info.welm_mtp_base_positions = buffers.positions[base_position_index]
        forward_batch.mtp_step_idx = 0
