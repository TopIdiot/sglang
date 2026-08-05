import contextlib
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import DraftContinuationState
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.models.welm_perf_opt import (
    get_welm_oe_hash_config,
    should_use_welm_oe_hash_kernel,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_info_v2 import (
    assign_extend_cache_locs,
    fill_accepted_out_cache_loc,
    fill_new_bonus_tokens,
)
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_req_to_token_pool_func,
    build_welmv4_mtp_model_specific_states,
    copy_welmv4_mtp_kv_mirror_states_to_buffers,
    draft_tp_context,
    generate_token_bitmask,
    load_token_map,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.speculative.welmv4_mtp_draft_proposal_cuda_graph_runner import (
    WelmMTPDraftProposalCudaGraphRunner,
)
from sglang.srt.speculative.welmv4_mtp_sampling import welm_mtp_deterministic_uniforms
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

if TYPE_CHECKING:
    from sglang.srt.speculative.welm_mtp_draft_ngram_hash import (
        WelmMTPDraftNGramHistoryState,
    )

_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()

if _is_cuda or _is_musa:
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob

logger = logging.getLogger(__name__)
_WELM_MTP_DUMP_ENABLED = os.environ.get(
    "SGLANG_DUMP_MTP_ACTIVATIONS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_WELM_VERIFY_AFTER_DUMP_ENABLED = os.environ.get(
    "SGLANG_DUMP_VERIFY_AFTER_MTP_METADATA", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_WELM_TRUE_VALUES = {"1", "true", "yes", "on"}
_WELM_FALSE_VALUES = {"0", "false", "no", "off"}
_WELM_MTP_DRAFT_CUDA_GRAPH_ENABLED = (
    os.environ.get("SGLANG_WELM_MTP_DRAFT_CUDA_GRAPH", "1").strip().lower()
    not in _WELM_FALSE_VALUES
)
_WELM_MTP_DRAFT_FIXED_TEMPERATURE_ENV = "SGLANG_WELM_MTP_DRAFT_FIXED_TEMPERATURE"
_WELM_MTP_DRAFT_FIXED_TOP_P_ENV = "SGLANG_WELM_MTP_DRAFT_FIXED_TOP_P"
_WELM_MTP_DRAFT_SAMPLING_TOPK_ENV = "SGLANG_WELM_MTP_DRAFT_SAMPLING_TOPK"
_WELM_DISABLE_TARGET_VERIFY_GRAPH_FOR_DUMP = (
    os.environ.get("SGLANG_WELMV4_DISABLE_TARGET_VERIFY_GRAPH_FOR_DUMP", "0")
    .strip()
    .lower()
    in _WELM_TRUE_VALUES
)
_WELM_VERIFY_AFTER_EVENT_COUNTERS = {}


def _welm_mtp_trace(message: str) -> None:
    if os.environ.get("SGLANG_WELM_MTP_TRACE", "0") == "1":
        print(f"[WELM_MTP_TRACE pid={os.getpid()}] {message}", flush=True)


def _welm_mtp_env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _WELM_TRUE_VALUES


def _parse_optional_float_env(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return float(raw)


def _to_cpu_payload(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu_payload(v) for v in value)
    return value


def _dump_verify_after_event(name: str, payload: dict) -> None:
    if not _WELM_VERIFY_AFTER_DUMP_ENABLED:
        return
    root = os.environ.get(
        "SGLANG_DUMP_VERIFY_AFTER_MTP_DIR",
        os.environ.get("SGLANG_DUMP_MTP_ACTIVATIONS_DIR", "./sglang_mtp_dump"),
    )
    rank = os.environ.get("RANK", "0")
    dump_dir = Path(root) / f"Rank{rank}_pid{os.getpid()}" / "verify_after_events"
    dump_dir.mkdir(parents=True, exist_ok=True)
    idx = _WELM_VERIFY_AFTER_EVENT_COUNTERS.get(name, 0)
    _WELM_VERIFY_AFTER_EVENT_COUNTERS[name] = idx + 1
    event = {"name": name, "index": idx, **payload}
    torch.save(_to_cpu_payload(event), dump_dir / f"{name}_{idx:05d}.pt")


@contextlib.contextmanager
def _welmv4_mtp_dump_context(context: str):
    if not _WELM_MTP_DUMP_ENABLED:
        yield
        return
    env_key = "SGLANG_DUMP_MTP_ACTIVATIONS_CONTEXT"
    previous = os.environ.get(env_key)
    os.environ[env_key] = context
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous


def _flush_welmv4_mtp_graph_dump(
    context: str, first_dim_limit: Optional[int] = None
) -> list[Path]:
    if not _WELM_MTP_DUMP_ENABLED:
        return []
    from sglang.srt.models import welmv4_nextn as welmv4_nextn_module

    return welmv4_nextn_module._flush_mtp_graph_dump_pass(
        context, first_dim_limit=first_dim_limit
    )


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class EagleDraftWorker(BaseDraftWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.welmv4_mtp_sample_draft = _welm_mtp_env_flag(
            "SGLANG_WELM_MTP_SAMPLE_DRAFT"
        )
        self.welmv4_mtp_draft_fixed_temperature = _parse_optional_float_env(
            _WELM_MTP_DRAFT_FIXED_TEMPERATURE_ENV
        )
        self.welmv4_mtp_draft_fixed_top_p = _parse_optional_float_env(
            _WELM_MTP_DRAFT_FIXED_TOP_P_ENV
        )
        self.welm_mtp_kv_mirror_state_buffers = None
        if (
            self.welmv4_mtp_draft_fixed_temperature is not None
            and self.welmv4_mtp_draft_fixed_temperature <= 0
        ):
            raise ValueError(
                f"{_WELM_MTP_DRAFT_FIXED_TEMPERATURE_ENV} must be positive, "
                f"got {self.welmv4_mtp_draft_fixed_temperature}."
            )
        if self.welmv4_mtp_draft_fixed_top_p is not None and not (
            0 < self.welmv4_mtp_draft_fixed_top_p <= 1
        ):
            raise ValueError(
                f"{_WELM_MTP_DRAFT_FIXED_TOP_P_ENV} must be in (0, 1], "
                f"got {self.welmv4_mtp_draft_fixed_top_p}."
            )

        # Do not capture cuda graph in `TpModelWorker` init,
        # will capture later with init_cuda_graphs()
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Init draft worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx,
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            # Init draft worker
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            eagle_config = getattr(
                self.draft_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        self.init_token_map()
        self.init_lm_head()

        # Init attention backend and cuda graphs
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.init_attention_backend()
            self.init_cuda_graphs()

        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)

        if self._is_welmv4_mtp_draft_model():
            num_nextn_layers = getattr(
                self.draft_runner.model_config.hf_config,
                "num_nextn_predict_layers",
                None,
            )
            if num_nextn_layers is None:
                raise ValueError(
                    "WeLM MTP draft model requires num_nextn_predict_layers."
                )
            num_nextn_layers = int(num_nextn_layers)
            if num_nextn_layers <= 0:
                raise ValueError(
                    "WeLM MTP draft model requires a positive "
                    f"num_nextn_predict_layers, got {num_nextn_layers}."
                )
            if self.topk > 1 and self.welmv4_mtp_sample_draft:
                raise ValueError(
                    "WeLM MTP topk>1 does not support draft sampling yet. "
                    "Unset SGLANG_WELM_MTP_SAMPLE_DRAFT or use "
                    "speculative_eagle_topk=1."
                )
            if (
                self.topk == 1
                and self.speculative_num_draft_tokens != self.speculative_num_steps + 1
            ):
                raise ValueError(
                    "WeLM MTP requires speculative_num_draft_tokens to equal "
                    "speculative_num_steps + 1, got "
                    f"steps={self.speculative_num_steps}, "
                    f"draft_tokens={self.speculative_num_draft_tokens}."
                )
            if self.topk > 1:
                max_tree_tokens = (
                    1
                    + self.topk
                    + (self.speculative_num_steps - 1) * self.topk * self.topk
                )
                if self.speculative_num_draft_tokens > max_tree_tokens:
                    raise ValueError(
                        "WeLM MTP topk>1 requires speculative_num_draft_tokens "
                        "to fit the draft tree, got "
                        f"draft_tokens={self.speculative_num_draft_tokens}, "
                        f"max_tree_tokens={max_tree_tokens}, topk={self.topk}, "
                        f"steps={self.speculative_num_steps}."
                    )
            if num_nextn_layers != 1 and num_nextn_layers < self.speculative_num_steps:
                raise ValueError(
                    "WeLM MTP requires num_nextn_predict_layers to be either 1 "
                    "or at least speculative_num_steps, got "
                    f"layers={num_nextn_layers}, steps={self.speculative_num_steps}."
                )

    def _is_welmv4_mtp_draft_model(self) -> bool:
        architectures = getattr(
            self.draft_runner.model_config.hf_config, "architectures", []
        )
        return bool(architectures and architectures[0] == "WeLMV4MoeForCausalLMNextN")

    def _maybe_map_hot_token_id(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.hot_token_id is None:
            return token_ids
        return self.hot_token_id[token_ids]

    def _get_welmv4_mtp_base_positions(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        custom_last_index = getattr(forward_batch, "custom_last_index", None)
        if custom_last_index is not None:
            return forward_batch.positions[custom_last_index].clone()

        if forward_batch.extend_seq_lens is not None:
            last_query_indices = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            return forward_batch.positions[last_query_indices].clone()

        return (forward_batch.seq_lens - 1).to(forward_batch.positions.dtype)

    def _get_welmv4_mtp_topk_cs_indices(
        self,
        i: int,
        tree_info: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        forward_batch: Optional[ForwardBatch] = None,
    ) -> Optional[torch.Tensor]:
        if i == 0:
            return None
        offset = self.topk**2 * (i - 1) + self.topk
        topk_cs_idx = tree_info[2]
        scratch = (
            None
            if forward_batch is None
            else getattr(forward_batch, "welm_mtp_oe_parent_scratch", None)
        )
        if scratch is not None and scratch.numel() >= topk_cs_idx.numel():
            out = scratch[: topk_cs_idx.numel()].view_as(topk_cs_idx)
            torch.sub(topk_cs_idx, offset, out=out)
            return out
        return topk_cs_idx - offset

    def _should_use_welmv4_mtp_oe_hash_kernel(self) -> bool:
        return self._is_welmv4_mtp_draft_model() and should_use_welm_oe_hash_kernel(
            self.draft_runner.model_config
        )

    def _welmv4_mtp_oe_hash_config(
        self,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        oe_grams, oe_vocab_sizes = get_welm_oe_hash_config(
            self.draft_runner.model_config
        )
        if not oe_grams:
            return oe_grams, oe_vocab_sizes, 0
        # Target-verify sees the current verified token both in draft_token_ids
        # and in the rolling history state. Keep one extra column so the target
        # verify hash kernel can skip that current-token column and still read
        # max_gram - 1 tokens before it.
        return oe_grams, oe_vocab_sizes, max(int(g) for g in oe_grams)

    def _welmv4_mtp_oe_prefix_width(self) -> int:
        oe_grams, _, _ = self._welmv4_mtp_oe_hash_config()
        return max((int(g) for g in oe_grams), default=1) - 1

    @staticmethod
    def _flatten_welmv4_mtp_hash_prefixes(prefix_rows) -> list[int]:
        return [int(token) for row in prefix_rows for token in row]

    @staticmethod
    def _welmv4_mtp_hash_prefix_rows_from_context(
        oe_context,
        history_width: int,
    ) -> list[list[int]]:
        prefix_rows = getattr(oe_context, "hash_prefixes", None)
        if prefix_rows is None:
            raise RuntimeError(
                "WeLMV4 MTP fused OE hash path requires CPU prefix rows at "
                "stage entry."
            )
        rows = [list(row) for row in prefix_rows[:history_width]]
        if not rows:
            return rows
        while len(rows) < history_width:
            rows.append([0] * len(rows[0]))
        return rows

    @staticmethod
    def _welmv4_mtp_prefix_rows_from_reqs(
        reqs,
        prefix_width: int,
        skip_latest_output: bool,
    ) -> list[list[int]]:
        if prefix_width <= 0:
            return []
        rows = []
        for lag in range(1, prefix_width + 1):
            row = []
            for req in reqs:
                ids = req.origin_input_ids + req.output_ids
                pos = len(ids) - lag - (1 if skip_latest_output else 0)
                row.append(ids[pos] if pos >= 0 else 0)
            rows.append(row)
        return rows

    def _init_welmv4_mtp_oe_history_from_context(
        self,
        oe_context,
        *,
        device: torch.device,
        first_token_ids: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        prefix_rows: Optional[list[list[int]]] = None,
    ) -> Optional[torch.Tensor]:
        if not self._should_use_welmv4_mtp_oe_hash_kernel() or oe_context is None:
            return None
        _, _, history_width = self._welmv4_mtp_oe_hash_config()
        if history_width <= 0:
            return None
        if prefix_rows is None:
            prefix_rows = self._welmv4_mtp_hash_prefix_rows_from_context(
                oe_context,
                history_width,
            )
        else:
            prefix_rows = [list(row) for row in prefix_rows[:history_width]]
            if prefix_rows:
                while len(prefix_rows) < history_width:
                    prefix_rows.append([0] * len(prefix_rows[0]))
        batch_size = len(prefix_rows[0]) if prefix_rows else 0
        if batch_size == 0:
            return None
        if out is None:
            out = torch.empty(
                (batch_size, history_width), device=device, dtype=torch.int64
            )
        elif out.shape[0] < batch_size or out.shape[1] != history_width:
            raise RuntimeError(
                "WeLMV4 MTP OE history output has incompatible shape: "
                f"{tuple(out.shape)} vs ({batch_size}, {history_width})."
            )
        history_out = out[:batch_size]
        flat_prefixes = self._flatten_welmv4_mtp_hash_prefixes(prefix_rows)
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_mtp_init_history_from_prefixes_cuda,
        )

        if first_token_ids is not None:
            if first_token_ids.numel() != batch_size:
                raise RuntimeError(
                    "WeLMV4 MTP OE first-token count mismatch: "
                    f"{first_token_ids.numel()} vs {batch_size}."
                )
        welm_oe_hash_mtp_init_history_from_prefixes_cuda(
            flat_prefixes,
            history_out,
            first_token_ids=first_token_ids,
        )
        return history_out

    def _init_welmv4_mtp_oe_history_from_extend(
        self,
        forward_batch: ForwardBatch,
        *,
        first_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if not self._should_use_welmv4_mtp_oe_hash_kernel():
            return None
        _, _, history_width = self._welmv4_mtp_oe_hash_config()
        if history_width <= 0:
            return None
        input_ids = forward_batch.input_ids
        extend_seq_lens = forward_batch.extend_seq_lens
        extend_start_loc = forward_batch.extend_start_loc
        device = (
            first_token_ids.device
            if first_token_ids is not None
            else input_ids.device if input_ids is not None else self.device
        )
        if input_ids is None or extend_seq_lens is None or extend_start_loc is None:
            return self._init_welmv4_mtp_oe_history_from_context(
                forward_batch.oe_context,
                device=device,
                first_token_ids=first_token_ids,
            )

        batch_size = int(
            first_token_ids.numel()
            if first_token_ids is not None
            else extend_seq_lens.numel()
        )
        if int(extend_seq_lens.numel()) != batch_size:
            raise RuntimeError(
                "WeLMV4 MTP prefill OE history batch mismatch: "
                f"extend_seq_lens={extend_seq_lens.numel()} vs rows={batch_size}."
            )
        history = torch.empty(
            (batch_size, history_width),
            device=device,
            dtype=torch.int64,
        )

        has_first_token = first_token_ids is not None
        if has_first_token:
            history[:, -1] = first_token_ids.to(
                device=history.device, dtype=torch.int64
            )

        prefix_width = max(history_width - (1 if has_first_token else 0), 0)
        if prefix_width == 0:
            return history
        prefix_rows = self._welmv4_mtp_hash_prefix_rows_from_context(
            forward_batch.oe_context,
            prefix_width,
        )
        prefix_tensor = torch.tensor(
            prefix_rows,
            device=history.device,
            dtype=torch.int64,
        )
        extend_seq_lens = extend_seq_lens.to(device=history.device, dtype=torch.long)
        extend_start_loc = extend_start_loc.to(device=history.device, dtype=torch.long)
        input_ids = input_ids.to(device=history.device, dtype=torch.int64)

        for col in range(prefix_width):
            lag = prefix_width - col
            in_segment = extend_seq_lens >= lag
            segment_pos = extend_start_loc + (extend_seq_lens - lag).clamp_min(0)
            segment_values = input_ids[segment_pos]

            prefix_lag = (lag - extend_seq_lens).clamp_min(1)
            prefix_idx = (prefix_lag - 1).clamp_max(prefix_width - 1)
            prefix_values = prefix_tensor.gather(0, prefix_idx.view(1, -1)).squeeze(0)
            history[:, col] = torch.where(in_segment, segment_values, prefix_values)

        return history

    def _compute_welmv4_mtp_first_query_hash_from_entry_history(
        self,
        forward_batch: ForwardBatch,
        first_input_ids: torch.Tensor,
        entry_history_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, "WelmMTPDraftNGramHistoryState", torch.Tensor]:
        from sglang.srt.speculative.welm_mtp_draft_ngram_hash import (
            WelmMTPDraftNGramEntryHistory,
            materialize_welm_mtp_draft_ngram_history,
            should_use_mk_welm_mtp_draft_ngram_hash,
        )

        entry_hash_history: "WelmMTPDraftNGramHistoryState" = entry_history_state
        if (
            should_use_mk_welm_mtp_draft_ngram_hash()
            and entry_history_state.shape[1] >= 2
        ):
            entry_hash_history = WelmMTPDraftNGramEntryHistory(
                prev_input_ids=entry_history_state[:, -1],
                prev_prev_input_ids=[
                    int(x) for x in entry_history_state[:, -2].detach().cpu().tolist()
                ],
            )
        first_query_hashed_inputs, first_query_draft_history_state = (
            self._compute_welmv4_mtp_draft_decode_hash_inputs(
                forward_batch,
                first_input_ids.to(dtype=torch.int64),
                entry_hash_history,
                draft_history_state=None,
                base_query_count=int(entry_hash_history.shape[0]),
                step_idx=0,
                use_forward_hash_buffer=False,
            )
        )
        _, _, history_width = self._welmv4_mtp_oe_hash_config()
        first_query_verify_history_state = materialize_welm_mtp_draft_ngram_history(
            first_query_draft_history_state,
            history_width=history_width,
        )
        return (
            first_query_hashed_inputs,
            first_query_draft_history_state,
            first_query_verify_history_state,
        )

    def _get_welmv4_mtp_hash_out(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        num_branches: int,
    ) -> torch.Tensor:
        num_tokens = int(input_ids.numel())
        buffer = getattr(forward_batch, "welm_mtp_oe_hash_out", None)
        if (
            buffer is not None
            and buffer.shape[0] == num_branches
            and buffer.shape[1] >= num_tokens
        ):
            return buffer[:, :num_tokens]
        return torch.empty(
            (num_branches, num_tokens), device=input_ids.device, dtype=torch.int64
        )

    def _get_welmv4_mtp_next_history_out(
        self,
        forward_batch: ForwardBatch,
        rows: int,
        history_width: int,
        *,
        step_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if step_idx is not None:
            work_buffers = getattr(forward_batch, "welm_mtp_oe_work_history", None)
            if work_buffers:
                buffer = work_buffers[step_idx % len(work_buffers)]
                if buffer.shape[0] >= rows and buffer.shape[1] == history_width:
                    return buffer[:rows]
        buffer = getattr(forward_batch, "welm_mtp_oe_next_history", None)
        if (
            buffer is not None
            and buffer.shape[0] >= rows
            and buffer.shape[1] == history_width
        ):
            return buffer[:rows]
        return torch.empty((rows, history_width), device=self.device, dtype=torch.int64)

    def _prepare_welmv4_mtp_draft_decode_entry_history(
        self,
        forward_batch: ForwardBatch,
        draft_input: EagleDraftInput,
    ) -> Optional[torch.Tensor]:
        if (
            not self._should_use_welmv4_mtp_oe_hash_kernel()
            or forward_batch.forward_mode.is_idle()
        ):
            return None
        history_state = getattr(draft_input, "welm_mtp_oe_history_state", None)
        if history_state is not None:
            forward_batch.spec_info.welm_mtp_oe_history_state = history_state
            return history_state
        first_token_ids = getattr(draft_input, "bonus_tokens", None)
        if first_token_ids is None:
            raise RuntimeError(
                "WeLMV4 MTP fused OE hash draft-decode path requires bonus_tokens."
            )
        prefix_rows = getattr(draft_input, "welm_mtp_oe_prefix_rows", None)
        history_state = self._init_welmv4_mtp_oe_history_from_context(
            forward_batch.oe_context,
            device=forward_batch.input_ids.device,
            first_token_ids=first_token_ids,
            prefix_rows=prefix_rows,
        )
        if history_state is None:
            raise RuntimeError(
                "WeLMV4 MTP fused OE hash draft-decode path is missing CPU "
                "prefix state."
            )
        draft_input.welm_mtp_oe_history_state = history_state
        forward_batch.spec_info.welm_mtp_oe_history_state = history_state
        return history_state

    def _prepare_welmv4_mtp_segment_hash_inputs_from_prefixes(
        self,
        forward_batch: ForwardBatch,
    ) -> None:
        from sglang.jit_kernel.welm_oe import welm_oe_hash_segments_from_prefixes_cuda

        input_ids = forward_batch.input_ids
        oe_grams, oe_vocab_sizes, _ = self._welmv4_mtp_oe_hash_config()
        hashed_out = self._get_welmv4_mtp_hash_out(
            forward_batch, input_ids, len(oe_vocab_sizes)
        )
        num_segments = int(forward_batch.extend_seq_lens.numel())
        prefix_width = self._welmv4_mtp_oe_prefix_width()
        prefix_rows = self._welmv4_mtp_hash_prefix_rows_from_context(
            forward_batch.oe_context,
            prefix_width,
        )
        prefixes = self._flatten_welmv4_mtp_hash_prefixes(prefix_rows)
        extend_seq_lens_cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_seq_lens_cpu is not None:
            real_num_tokens = sum(int(x) for x in extend_seq_lens_cpu)
            if real_num_tokens < input_ids.numel():
                hashed_out[:, real_num_tokens:].zero_()
        welm_oe_hash_segments_from_prefixes_cuda(
            input_ids,
            forward_batch.extend_start_loc,
            forward_batch.extend_seq_lens,
            prefixes,
            oe_grams,
            oe_vocab_sizes,
            hashed_out,
            self.draft_runner.model_config.vocab_size,
        )
        forward_batch.welm_oe_decode_hashed_inputs = hashed_out

    def _compute_welmv4_mtp_target_verify_hash_inputs(
        self,
        input_ids: torch.Tensor,
        tree_mask: torch.Tensor,
        seq_lens: torch.Tensor,
        history_state: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_mtp_target_verify_from_history_cuda,
        )

        oe_grams, oe_vocab_sizes, _ = self._welmv4_mtp_oe_hash_config()
        batch_size = int(seq_lens.numel())
        real_num_tokens = batch_size * int(draft_token_num)
        hashed_out = torch.empty(
            (len(oe_vocab_sizes), int(input_ids.numel())),
            device=input_ids.device,
            dtype=torch.int64,
        )
        if real_num_tokens < input_ids.numel():
            hashed_out[:, real_num_tokens:].zero_()
        welm_oe_hash_mtp_target_verify_from_history_cuda(
            input_ids[:real_num_tokens],
            tree_mask,
            seq_lens,
            history_state[:batch_size],
            oe_grams,
            oe_vocab_sizes,
            hashed_out[:, :real_num_tokens],
            self.target_worker.model_config.vocab_size,
            int(draft_token_num),
        )
        return hashed_out

    def _prepare_welmv4_mtp_target_verify_hash_inputs(
        self,
        forward_batch: ForwardBatch,
        history_state: Optional[torch.Tensor],
    ) -> None:
        if (
            not self._should_use_welmv4_mtp_oe_hash_kernel()
            or history_state is None
            or forward_batch.forward_mode.is_idle()
        ):
            return
        spec_info = forward_batch.spec_info
        hash_seq_lens = getattr(
            spec_info, "welm_mtp_oe_hash_seq_lens", forward_batch.seq_lens
        )
        forward_batch.welm_oe_decode_hashed_inputs = (
            self._compute_welmv4_mtp_target_verify_hash_inputs(
                forward_batch.input_ids,
                spec_info.custom_mask,
                hash_seq_lens,
                history_state,
                int(spec_info.draft_token_num),
            )
        )

    def _precompute_welmv4_mtp_target_verify_hash_inputs(
        self,
        verify_input: EagleVerifyInput,
        batch: ModelWorkerBatch,
    ) -> None:
        history_state = getattr(verify_input, "welm_mtp_oe_history_state", None)
        if (
            not self._should_use_welmv4_mtp_oe_hash_kernel()
            or history_state is None
            or batch.forward_mode.is_idle()
        ):
            return
        verify_input.welm_mtp_oe_hashed_inputs = (
            self._compute_welmv4_mtp_target_verify_hash_inputs(
                verify_input.draft_token,
                verify_input.custom_mask,
                getattr(verify_input, "welm_mtp_oe_hash_seq_lens", batch.seq_lens),
                history_state,
                int(verify_input.draft_token_num),
            )
        )

    def _prepare_welmv4_mtp_draft_extend_hash_inputs(
        self,
        forward_batch: ForwardBatch,
        accept_lens: torch.Tensor,
        accepted_draft_token_ids: Optional[torch.Tensor],
        entry_history_state: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            not self._should_use_welmv4_mtp_oe_hash_kernel()
            or forward_batch.forward_mode.is_idle()
        ):
            return None
        if accepted_draft_token_ids is None:
            raise RuntimeError(
                "WeLMV4 MTP fused OE hash draft-extend path requires accepted "
                "draft token ids from target verify."
            )
        if entry_history_state is None:
            raise RuntimeError(
                "WeLMV4 MTP fused OE hash draft-extend path is missing entry "
                "history state."
            )
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda,
        )

        input_ids = forward_batch.input_ids
        oe_grams, oe_vocab_sizes, history_width = self._welmv4_mtp_oe_hash_config()
        batch_size = int(accept_lens.numel())
        draft_token_num = self.speculative_num_draft_tokens
        hashed_out = self._get_welmv4_mtp_hash_out(
            forward_batch, input_ids, len(oe_vocab_sizes)
        )
        next_history = self._get_welmv4_mtp_next_history_out(
            forward_batch,
            batch_size,
            history_width,
        )
        dense_token_num = batch_size * draft_token_num
        if int(input_ids.numel()) > dense_token_num:
            raise RuntimeError(
                "WeLMV4 MTP accepted-only draft extend input is larger than "
                f"the dense capacity: input_tokens={input_ids.numel()} "
                f"capacity={dense_token_num}."
            )
        fused_prepare = _welm_mtp_env_flag("SGLANG_WELM_MTP_FUSED_PREPARE")
        if fused_prepare and envs.SGLANG_SPEC_OOB_DETECTION.get():
            torch._assert_async(
                accept_lens.sum() == input_ids.numel(),
                "WeLMV4 MTP accepted-only draft extend input length mismatch.",
            )
        kernel_input_ids = input_ids
        kernel_hashed_out = hashed_out
        accepted_lens_cpu = None
        if not fused_prepare and int(input_ids.numel()) != dense_token_num:
            accepted_lens_cpu = [int(x) for x in accept_lens.detach().cpu().tolist()]
            if sum(accepted_lens_cpu) != int(input_ids.numel()):
                raise RuntimeError(
                    "WeLMV4 MTP accepted-only draft extend input length mismatch: "
                    f"sum(accept_lens)={sum(accepted_lens_cpu)} "
                    f"input_tokens={input_ids.numel()}."
                )
            kernel_input_ids = torch.empty(
                (dense_token_num,), device=input_ids.device, dtype=input_ids.dtype
            )
            src_start = 0
            for row, accepted_len in enumerate(accepted_lens_cpu):
                src_end = src_start + accepted_len
                dst_start = row * draft_token_num
                dst_end = dst_start + accepted_len
                if accepted_len > 0:
                    kernel_input_ids[dst_start:dst_end].copy_(
                        input_ids[src_start:src_end]
                    )
                if accepted_len == 0:
                    kernel_input_ids[dst_start : dst_start + draft_token_num].zero_()
                elif accepted_len < draft_token_num:
                    kernel_input_ids[dst_end : dst_start + draft_token_num].copy_(
                        input_ids[src_end - 1].expand(draft_token_num - accepted_len)
                    )
                src_start = src_end
            kernel_hashed_out = torch.empty(
                (len(oe_vocab_sizes), dense_token_num),
                device=input_ids.device,
                dtype=torch.int64,
            )
        welm_oe_hash_mtp_draft_extend_after_verify_from_history_cuda(
            kernel_input_ids,
            accepted_draft_token_ids.reshape(batch_size, -1),
            accept_lens,
            entry_history_state[:batch_size],
            oe_grams,
            oe_vocab_sizes,
            kernel_hashed_out,
            next_history,
            self.draft_runner.model_config.vocab_size,
            draft_token_num,
            use_entry_history_for_extend_hash_prefix=True,
        )
        if kernel_hashed_out is not hashed_out:
            assert accepted_lens_cpu is not None
            src_start = 0
            for row, accepted_len in enumerate(accepted_lens_cpu):
                dense_start = row * draft_token_num
                dense_end = dense_start + accepted_len
                src_end = src_start + accepted_len
                hashed_out[:, src_start:src_end].copy_(
                    kernel_hashed_out[:, dense_start:dense_end]
                )
                src_start = src_end
        forward_batch.welm_oe_decode_hashed_inputs = hashed_out
        return next_history

    def _prepare_welmv4_mtp_draft_decode_hash_inputs(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        entry_history_state: "WelmMTPDraftNGramHistoryState",
        draft_history_state: Optional["WelmMTPDraftNGramHistoryState"],
        base_query_count: int,
        step_idx: int,
        selected_parent_indices: Optional[torch.Tensor] = None,
        topk_cs_idx: Optional[torch.Tensor] = None,
    ) -> "WelmMTPDraftNGramHistoryState":
        hashed_out, next_history = self._compute_welmv4_mtp_draft_decode_hash_inputs(
            forward_batch,
            input_ids,
            entry_history_state,
            draft_history_state,
            base_query_count,
            step_idx,
            selected_parent_indices=selected_parent_indices,
            topk_cs_idx=topk_cs_idx,
            use_forward_hash_buffer=True,
        )
        forward_batch.welm_oe_decode_hashed_inputs = hashed_out
        return next_history

    def _compute_welmv4_mtp_draft_decode_hash_inputs(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        entry_history_state: "WelmMTPDraftNGramHistoryState",
        draft_history_state: Optional["WelmMTPDraftNGramHistoryState"],
        base_query_count: int,
        step_idx: int,
        *,
        selected_parent_indices: Optional[torch.Tensor] = None,
        topk_cs_idx: Optional[torch.Tensor] = None,
        use_forward_hash_buffer: bool = False,
    ) -> Tuple[torch.Tensor, "WelmMTPDraftNGramHistoryState"]:
        from sglang.jit_kernel.welm_oe import (
            welm_oe_hash_mtp_draft_decode_from_history_cuda,
        )
        from sglang.srt.speculative.welm_mtp_draft_ngram_hash import (
            should_use_mk_welm_mtp_draft_ngram_hash,
            welm_mtp_draft_ngram_hash_from_history,
        )

        oe_grams, oe_vocab_sizes, history_width = self._welmv4_mtp_oe_hash_config()
        use_mk_draft_ngram_hash = should_use_mk_welm_mtp_draft_ngram_hash()
        source_history = (
            entry_history_state if draft_history_state is None else draft_history_state
        )
        use_parent = selected_parent_indices is not None
        if selected_parent_indices is None:
            parent_indices = getattr(forward_batch, "welm_mtp_oe_parent_scratch", None)
            if parent_indices is None or parent_indices.numel() < input_ids.numel():
                parent_indices = torch.empty(
                    (input_ids.numel(),), device=input_ids.device, dtype=torch.int64
                )
        else:
            parent_indices = selected_parent_indices.to(
                device=input_ids.device, dtype=torch.int64
            )
        if use_forward_hash_buffer:
            hashed_out = self._get_welmv4_mtp_hash_out(
                forward_batch, input_ids, len(oe_vocab_sizes)
            )
        else:
            batch_major_hash_out = getattr(
                forward_batch, "welm_mtp_oe_hash_out_batch_major", None
            )
            if (
                use_mk_draft_ngram_hash
                and batch_major_hash_out is not None
                and getattr(forward_batch, "welm_mtp_skip_draft_proposal_build", False)
                and batch_major_hash_out.shape[0] >= int(input_ids.numel())
                and batch_major_hash_out.shape[1] == len(oe_vocab_sizes)
            ):
                hashed_out = batch_major_hash_out[: int(input_ids.numel())].t()
            else:
                hashed_out = torch.empty(
                    (len(oe_vocab_sizes), int(input_ids.numel())),
                    device=input_ids.device,
                    dtype=torch.int64,
                )
        mk_history_state = welm_mtp_draft_ngram_hash_from_history(
            forward_batch=forward_batch,
            input_ids=input_ids,
            history_state=source_history,
            parent_indices=parent_indices,
            oe_grams=oe_grams,
            oe_vocab_sizes=oe_vocab_sizes,
            hashed_out=hashed_out,
            next_history_state=None,
            vocab_size=self.draft_runner.model_config.vocab_size,
            base_query_count=base_query_count,
            use_parent=use_parent,
            topk=self.topk,
            topk_cs_idx=topk_cs_idx,
            prev_input_ids_scratch=getattr(
                forward_batch, "welm_mtp_oe_prev_input_ids", None
            ),
            prev_prev_input_ids_scratch=getattr(
                forward_batch, "welm_mtp_oe_prev_prev_input_ids", None
            ),
            output_ids_scratch=getattr(
                forward_batch, "welm_mtp_oe_hash_out_batch_major", None
            ),
            output_prev_input_ids_scratch=getattr(
                forward_batch, "welm_mtp_oe_output_prev_input_ids", None
            ),
            source_indices_scratch=getattr(
                forward_batch, "welm_mtp_oe_parent_scratch", None
            ),
        )
        if mk_history_state is not None:
            return hashed_out, mk_history_state
        if use_mk_draft_ngram_hash:
            raise RuntimeError(
                "SGLANG_WELM_MTP_DRAFT_NGRAM_HASH is enabled, but draft decode "
                "ngram hash did not run through mk."
            )

        next_history = self._get_welmv4_mtp_next_history_out(
            forward_batch,
            int(input_ids.numel()),
            history_width,
            step_idx=step_idx,
        )
        welm_oe_hash_mtp_draft_decode_from_history_cuda(
            input_ids,
            source_history,
            parent_indices,
            oe_grams,
            oe_vocab_sizes,
            hashed_out,
            next_history,
            self.draft_runner.model_config.vocab_size,
            base_query_count,
            use_parent,
        )
        return hashed_out, next_history

    def _capture_for_decode(
        self,
        logits_output,
        draft_input: EagleDraftInput,
        forward_batch: Optional[ForwardBatch] = None,
        welmv4_mtp_base_positions: Optional[torch.Tensor] = None,
    ) -> None:
        probs = torch.softmax(logits_output.next_token_logits, dim=-1)
        draft_input.topk_p, draft_input.topk_index = fast_topk(probs, self.topk, dim=-1)
        draft_input.hidden_states = logits_output.hidden_states
        if self._is_welmv4_mtp_draft_model() and forward_batch is not None:
            if welmv4_mtp_base_positions is None:
                welmv4_mtp_base_positions = self._get_welmv4_mtp_base_positions(
                    forward_batch
                )
            draft_input.welm_mtp_base_positions = welmv4_mtp_base_positions

    def _check_welmv4_mtp_token_range(
        self,
        token_ids: Optional[torch.Tensor],
        high: int,
        label: str,
    ) -> None:
        if token_ids is None or token_ids.numel() == 0:
            return
        maybe_detect_oob(
            token_ids,
            0,
            high,
            f"WeLMV4 MTP {label} token id OOB vs vocab_size={high}",
        )

    def _should_use_welmv4_mtp_greedy_draft(self, forward_batch: ForwardBatch) -> bool:
        if (
            self.welmv4_mtp_sample_draft
            and self._has_welmv4_mtp_fixed_draft_sampling_params()
        ):
            return False
        sampling_info = forward_batch.sampling_info
        return (
            self.topk == 1
            and (sampling_info is None or sampling_info.is_all_greedy)
            and not forward_batch.forward_mode.is_idle()
        )

    def _has_welmv4_mtp_fixed_draft_sampling_params(self) -> bool:
        return (
            self.welmv4_mtp_draft_fixed_temperature is not None
            or self.welmv4_mtp_draft_fixed_top_p is not None
        )

    def _should_sample_welmv4_mtp_draft(self, forward_batch: ForwardBatch) -> bool:
        should_sample = (
            self.welmv4_mtp_sample_draft
            and self.topk == 1
            and not forward_batch.forward_mode.is_idle()
        )
        if not should_sample:
            return False
        if self._has_welmv4_mtp_fixed_draft_sampling_params():
            return True
        sampling_info = forward_batch.sampling_info
        return sampling_info is not None and not sampling_info.is_all_greedy

    def _get_welmv4_mtp_draft_sampling_topk(self) -> int:
        value = int(os.environ.get(_WELM_MTP_DRAFT_SAMPLING_TOPK_ENV, "0"))
        if value <= 0:
            return 0
        vocab_size = int(self.draft_runner.model_config.vocab_size)
        if value >= vocab_size:
            return 0
        return value

    @staticmethod
    def _expand_sampling_tensor_for_logits(
        values: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape[0] == logits.shape[0]:
            return values
        if logits.shape[0] % values.shape[0] == 0:
            return torch.repeat_interleave(
                values, logits.shape[0] // values.shape[0], dim=0
            )
        return values[: logits.shape[0]]

    def _get_welmv4_mtp_draft_temperature(
        self, forward_batch: ForwardBatch, logits: torch.Tensor
    ) -> torch.Tensor:
        if self.welmv4_mtp_draft_fixed_temperature is not None:
            return torch.full(
                (logits.shape[0], 1),
                self.welmv4_mtp_draft_fixed_temperature,
                dtype=torch.float32,
                device=logits.device,
            )

        sampling_info = forward_batch.sampling_info
        temperatures = (
            getattr(sampling_info, "temperatures", None)
            if sampling_info is not None
            else None
        )
        if temperatures is None:
            return torch.ones(
                (logits.shape[0], 1), dtype=torch.float32, device=logits.device
            )
        return self._expand_sampling_tensor_for_logits(temperatures, logits).to(
            device=logits.device, dtype=torch.float32
        )

    def _should_use_welmv4_mtp_draft_top_p(self, forward_batch: ForwardBatch) -> bool:
        if self.welmv4_mtp_draft_fixed_top_p is not None:
            return self.welmv4_mtp_draft_fixed_top_p < 1.0 and (_is_cuda or _is_musa)
        sampling_info = forward_batch.sampling_info
        return (
            sampling_info is not None
            and bool(getattr(sampling_info, "need_top_p_sampling", False))
            and (_is_cuda or _is_musa)
        )

    def _get_welmv4_mtp_draft_top_p(
        self, forward_batch: ForwardBatch, logits: torch.Tensor
    ) -> torch.Tensor:
        if self.welmv4_mtp_draft_fixed_top_p is not None:
            return torch.full(
                (logits.shape[0],),
                self.welmv4_mtp_draft_fixed_top_p,
                dtype=torch.float32,
                device=logits.device,
            )

        sampling_info = forward_batch.sampling_info
        top_ps = (
            getattr(sampling_info, "top_ps", None)
            if sampling_info is not None
            else None
        )
        if top_ps is None:
            return torch.ones(
                (logits.shape[0],), dtype=torch.float32, device=logits.device
            )
        return self._expand_sampling_tensor_for_logits(top_ps, logits).to(
            device=logits.device, dtype=torch.float32
        )

    def _get_welmv4_mtp_draft_uniform(
        self,
        forward_batch: ForwardBatch,
        *,
        device: torch.device,
        batch_size: int,
        base_positions: Optional[torch.Tensor],
        step: int,
    ) -> Optional[torch.Tensor]:
        sampling_info = forward_batch.sampling_info
        if (
            sampling_info is None
            or getattr(sampling_info, "sampling_seed", None) is None
        ):
            return None
        if base_positions is None:
            base_positions = self._get_welmv4_mtp_base_positions(forward_batch)
        deterministic_uniforms = welm_mtp_deterministic_uniforms(
            sampling_info=sampling_info,
            positions=base_positions,
            batch_size=batch_size,
            width=1,
            salt=3000 + int(step),
        )
        if deterministic_uniforms is None:
            return None
        return deterministic_uniforms[:, :1].to(device=device, dtype=torch.float32)

    def _sample_welmv4_mtp_probs_top1(
        self,
        probs: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        base_positions: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        uniform = self._get_welmv4_mtp_draft_uniform(
            forward_batch,
            device=probs.device,
            batch_size=int(probs.shape[0]),
            base_positions=base_positions,
            step=step,
        )
        if uniform is None:
            topk_index = torch.multinomial(probs, num_samples=1)
        else:
            cdf = torch.cumsum(probs.to(torch.float32), dim=-1)
            threshold = uniform * cdf[:, -1:]
            topk_index = torch.sum(cdf < threshold, dim=-1, keepdim=True).to(torch.long)
            topk_index = torch.clamp(topk_index, max=probs.shape[-1] - 1)
        topk_p = torch.gather(probs, dim=1, index=topk_index)

        tp_group = (
            get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
        )
        if tp_group.world_size > 1:
            tp_group.broadcast(topk_index, src=0)
            topk_p = torch.gather(probs, dim=1, index=topk_index)
            tp_group.broadcast(topk_p, src=0)

        return topk_p, topk_index

    def _sample_welmv4_mtp_draft_top1(
        self,
        logits: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        base_positions: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        sampling_info = forward_batch.sampling_info
        if (
            sampling_info is None
            and not self._has_welmv4_mtp_fixed_draft_sampling_params()
        ):
            raise RuntimeError(
                "WeLMV4 MTP stochastic draft sampling requires sampling_info."
            )

        temperature = self._get_welmv4_mtp_draft_temperature(forward_batch, logits)
        scaled_logits = logits.float() / temperature
        use_top_p = self._should_use_welmv4_mtp_draft_top_p(forward_batch)
        sampling_topk = self._get_welmv4_mtp_draft_sampling_topk()
        vocab_size = int(scaled_logits.shape[-1])

        if 0 < sampling_topk < vocab_size:
            top_logits, top_indices = torch.topk(
                scaled_logits,
                k=sampling_topk,
                dim=-1,
                sorted=use_top_p,
            )
            probs = F.softmax(top_logits, dim=-1)
            if use_top_p:
                top_ps = self._get_welmv4_mtp_draft_top_p(forward_batch, top_logits)
                probs = top_p_renorm_prob(probs, top_ps)

            topk_p, topk_pos = self._sample_welmv4_mtp_probs_top1(
                probs,
                forward_batch,
                base_positions=base_positions,
                step=step,
            )
            topk_index = torch.gather(top_indices, dim=-1, index=topk_pos)
            tp_group = (
                get_attention_tp_group()
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(topk_index, src=0)
                topk_p = torch.gather(probs, dim=1, index=topk_pos)
                tp_group.broadcast(topk_p, src=0)
            return topk_p, topk_index, None, top_indices, probs

        probs = F.softmax(scaled_logits, dim=-1)
        if (
            not self._has_welmv4_mtp_fixed_draft_sampling_params()
            and getattr(sampling_info, "need_top_k_sampling", False)
            and (_is_cuda or _is_musa)
        ):
            top_ks = self._expand_sampling_tensor_for_logits(
                sampling_info.top_ks, logits
            )
            probs = top_k_renorm_prob(probs, top_ks)
        if use_top_p:
            top_ps = self._get_welmv4_mtp_draft_top_p(forward_batch, logits)
            probs = top_p_renorm_prob(probs, top_ps)

        topk_p, topk_index = self._sample_welmv4_mtp_probs_top1(
            probs,
            forward_batch,
            base_positions=base_positions,
            step=step,
        )
        return topk_p, topk_index, probs.contiguous(), None, None

    def _select_or_sample_welmv4_mtp_draft_topk(
        self,
        logits: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        base_positions: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        if self._should_sample_welmv4_mtp_draft(forward_batch):
            return self._sample_welmv4_mtp_draft_top1(
                logits,
                forward_batch,
                base_positions=base_positions,
                step=step,
            )

        topk_p, topk_index = self._select_welmv4_mtp_draft_topk(logits, forward_batch)
        return topk_p, topk_index, None, None, None

    def _select_welmv4_mtp_draft_topk(
        self,
        logits: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._should_use_welmv4_mtp_greedy_draft(forward_batch):
            topk_index = torch.argmax(logits, dim=-1, keepdim=True)
            topk_p = torch.ones(
                topk_index.shape,
                dtype=logits.dtype,
                device=logits.device,
            )
            return topk_p, topk_index

        probs = torch.softmax(logits, dim=-1)
        return fast_topk(probs, self.topk, dim=-1)

    def _pad_welmv4_mtp_draft_probs_for_verify(
        self, draft_probs_list: List[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if not draft_probs_list:
            return None
        draft_probs = torch.cat(draft_probs_list, dim=1).contiguous()
        pad_len = self.speculative_num_draft_tokens - draft_probs.shape[1]
        if pad_len <= 0:
            return draft_probs
        return torch.cat(
            [
                draft_probs,
                torch.zeros(
                    (draft_probs.shape[0], pad_len, draft_probs.shape[2]),
                    dtype=draft_probs.dtype,
                    device=draft_probs.device,
                ),
            ],
            dim=1,
        )

    def _pad_welmv4_mtp_draft_topk_for_verify(
        self,
        values: List[torch.Tensor],
        *,
        pad_value: int | float,
    ) -> Optional[torch.Tensor]:
        if not values:
            return None
        out = torch.cat(values, dim=1).contiguous()
        pad_len = self.speculative_num_draft_tokens - out.shape[1]
        if pad_len <= 0:
            return out
        return torch.cat(
            [
                out,
                torch.full(
                    (out.shape[0], pad_len, out.shape[2]),
                    pad_value,
                    dtype=out.dtype,
                    device=out.device,
                ),
            ],
            dim=1,
        )

    def _select_welmv4_mtp_last_extend_hash_inputs(
        self, forward_batch: ForwardBatch
    ) -> Optional[torch.Tensor]:
        if not self._should_use_welmv4_mtp_oe_hash_kernel():
            return None
        hashed_inputs = getattr(forward_batch, "welm_oe_decode_hashed_inputs", None)
        if hashed_inputs is None:
            raise RuntimeError(
                "WeLMV4 MTP merged extend-draft requires OE hashes for the "
                "extend/fill input."
            )
        custom_last_index = getattr(forward_batch, "custom_last_index", None)
        if custom_last_index is None:
            if forward_batch.extend_seq_lens is None:
                return hashed_inputs
            custom_last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
        return hashed_inputs[:, custom_last_index.to(torch.long)]

    def _set_welmv4_mtp_merged_query(
        self,
        forward_batch: ForwardBatch,
        input_ids: torch.Tensor,
        query_hashed_inputs: Optional[torch.Tensor],
        query_positions: Optional[torch.Tensor],
    ) -> None:
        forward_batch.welm_mtp_merge_kv_fill_draft = True
        forward_batch.welm_mtp_kv_fill_positions = forward_batch.positions
        forward_batch.welm_mtp_kv_fill_cache_loc = forward_batch.out_cache_loc
        forward_batch.welm_mtp_query_input_ids = input_ids.to(torch.int64)
        forward_batch.welm_mtp_query_oe_hashed_inputs = query_hashed_inputs
        forward_batch.welm_mtp_query_positions = query_positions

    def _clear_welmv4_mtp_merged_query(self, forward_batch: ForwardBatch) -> None:
        forward_batch.welm_mtp_merge_kv_fill_draft = False
        forward_batch.welm_mtp_kv_fill_positions = None
        forward_batch.welm_mtp_kv_fill_cache_loc = None
        forward_batch.welm_mtp_query_input_ids = None
        forward_batch.welm_mtp_query_oe_hashed_inputs = None
        forward_batch.welm_mtp_query_positions = None

    def _forward_welmv4_mtp_merged_extend_draft_step(
        self,
        forward_batch: ForwardBatch,
        step: int,
        input_ids: torch.Tensor,
        main_hidden_states: Optional[torch.Tensor],
        query_hashed_inputs: Optional[torch.Tensor],
        query_positions: Optional[torch.Tensor],
        *,
        skip_attn_backend_init: bool,
    ):
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        self._clear_welmv4_mtp_kv_mirror_contract_metadata(forward_batch)
        forward_batch.spec_info.hidden_states = main_hidden_states
        forward_batch._welm_mtp_contracted_dp_metadata_rows = None
        forward_batch.mtp_step_idx = step
        self._set_welmv4_mtp_merged_query(
            forward_batch,
            input_ids,
            query_hashed_inputs,
            query_positions,
        )
        hash_attr_was_present = "welm_oe_decode_hashed_inputs" in vars(forward_batch)
        previous_hashed_inputs = getattr(
            forward_batch, "welm_oe_decode_hashed_inputs", None
        )
        uses_branch_hash_as_extend_hash = (
            self.topk > 1 and step > 0 and query_hashed_inputs is not None
        )
        if uses_branch_hash_as_extend_hash:
            forward_batch.welm_oe_decode_hashed_inputs = query_hashed_inputs
        try:
            logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=skip_attn_backend_init
            ).logits_output
        finally:
            if uses_branch_hash_as_extend_hash:
                if hash_attr_was_present:
                    forward_batch.welm_oe_decode_hashed_inputs = previous_hashed_inputs
                else:
                    with contextlib.suppress(AttributeError):
                        delattr(forward_batch, "welm_oe_decode_hashed_inputs")
            self._clear_welmv4_mtp_merged_query(forward_batch)
            forward_batch.mtp_step_idx = 0
        maybe_detect_nan(
            logits_output.next_token_logits,
            f"welmv4_mtp_merged_extend_draft_step{step}",
        )
        return logits_output

    def _select_welmv4_mtp_tree_step(
        self,
        step: int,
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        scores: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Optional[torch.Tensor],
    ]:
        batch_size = int(topk_p.shape[0]) if step == 0 else int(scores.shape[0])
        if step == 0:
            input_ids = topk_index.flatten()
            next_hidden_states = (
                None
                if hidden_states is None
                else hidden_states.repeat_interleave(self.topk, dim=0)
            )
            next_scores = topk_p
            parent_indices = torch.arange(
                batch_size, dtype=torch.long, device=input_ids.device
            ).repeat_interleave(self.topk)
            tree_info = (
                topk_p.unsqueeze(1),
                topk_index,
                torch.arange(-1, self.topk, dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .repeat(batch_size, 1),
            )
            return input_ids, next_hidden_states, next_scores, tree_info, parent_indices

        assert scores is not None
        if hidden_states is None:
            raise RuntimeError("WeLM MTP topk>1 tree step requires hidden states.")

        expand_scores = scores.unsqueeze(2) * topk_p.reshape(
            batch_size, self.topk, self.topk
        )
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), self.topk, dim=-1
        )
        next_scores = topk_cs_p

        flat_topk_index = topk_index.reshape(batch_size, self.topk**2)
        input_ids = torch.gather(flat_topk_index, index=topk_cs_index, dim=1).flatten()
        parent_indices = topk_cs_index.flatten() // self.topk + torch.arange(
            0,
            hidden_states.shape[0],
            step=self.topk,
            device=topk_index.device,
        ).repeat_interleave(self.topk)
        next_hidden_states = hidden_states[parent_indices, :]
        tree_info = (
            expand_scores,
            flat_topk_index,
            topk_cs_index + (self.topk**2 * (step - 1) + self.topk),
        )
        return input_ids, next_hidden_states, next_scores, tree_info, parent_indices

    def _get_welmv4_mtp_branch_cache_locs(
        self,
        forward_batch: ForwardBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        step_cache_locs = getattr(
            forward_batch, "welm_mtp_branch_step_cache_locs", None
        )
        flat_cache_locs = getattr(
            forward_batch, "welm_mtp_branch_flat_cache_locs", None
        )
        if step_cache_locs is not None and flat_cache_locs is not None:
            return step_cache_locs, flat_cache_locs

        if self.topk <= 1 or self.speculative_num_steps <= 1:
            empty = torch.empty(
                (self.speculative_num_steps, 0),
                dtype=torch.int64,
                device=forward_batch.seq_lens.device,
            )
            return empty, empty.flatten()

        req_to_token_pool = getattr(forward_batch, "req_to_token_pool", None)
        if req_to_token_pool is None:
            raise RuntimeError("WeLM MTP topk>1 branch draft requires req_to_token.")
        req_to_token = req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices.to(
            device=req_to_token.device, dtype=torch.long
        )
        seq_lens = forward_batch.seq_lens.to(
            device=req_to_token.device, dtype=torch.long
        )
        batch_size = int(req_pool_indices.numel())
        width = self.topk * self.speculative_num_steps
        offsets = torch.arange(width, dtype=torch.long, device=req_to_token.device)
        flat_cache_locs = req_to_token[
            req_pool_indices[:, None],
            seq_lens[:, None] + offsets[None, :],
        ].contiguous()
        step_major_cache_locs = (
            flat_cache_locs.reshape(batch_size, self.topk, self.speculative_num_steps)
            .permute(2, 0, 1)
            .reshape(self.speculative_num_steps, batch_size * self.topk)
            .contiguous()
        )
        return step_major_cache_locs, flat_cache_locs.flatten()

    def _reserve_welmv4_mtp_prefill_branch_cache_locs(
        self,
        batch: ModelWorkerBatch,
    ) -> Optional[torch.Tensor]:
        if self.topk <= 1 or self.speculative_num_steps <= 1:
            return None

        page_size = getattr(self.token_to_kv_pool_allocator, "page_size", 1)
        if page_size != 1:
            raise RuntimeError(
                "WeLM MTP topk>1 prefill proposal currently requires "
                f"page_size=1, got {page_size}."
            )
        if batch.reqs is None:
            raise RuntimeError(
                "WeLM MTP topk>1 prefill proposal requires request metadata "
                "to reserve branch KV locations."
            )
        if batch.req_pool_indices is None:
            raise RuntimeError(
                "WeLM MTP topk>1 prefill proposal requires req_pool_indices."
            )

        batch_size = len(batch.seq_lens)
        if batch_size == 0:
            return None
        if len(batch.reqs) != batch_size:
            raise RuntimeError(
                "WeLM MTP topk>1 prefill proposal batch/request mismatch: "
                f"batch_size={batch_size}, reqs={len(batch.reqs)}."
            )

        width = self.topk * self.speculative_num_steps
        seq_lens_cpu = batch.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = batch.seq_lens.detach().cpu()
        seq_lens_list = [int(x) for x in seq_lens_cpu[:batch_size].tolist()]

        cur_kv_lens = []
        nxt_kv_lens = []
        num_needed_tokens = 0
        for req, seq_len in zip(batch.reqs, seq_lens_list):
            cur_len = max(int(req.kv_allocated_len), seq_len)
            nxt_len = max(cur_len, seq_len + width)
            cur_kv_lens.append(cur_len)
            nxt_kv_lens.append(nxt_len)
            num_needed_tokens += nxt_len - cur_len

        if num_needed_tokens == 0:
            return None

        branch_cache_locs = self.token_to_kv_pool_allocator.alloc(num_needed_tokens)
        if branch_cache_locs is None:
            raise RuntimeError(
                "WeLM MTP topk>1 prefill proposal failed to reserve "
                f"{num_needed_tokens} branch KV slots; "
                f"available={self.token_to_kv_pool_allocator.available_size()}."
            )

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens, dtype=torch.int32, device="cpu")
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            cur_kv_lens_cpu.to(device=batch.seq_lens.device),
            nxt_kv_lens_cpu.to(device=batch.seq_lens.device),
            branch_cache_locs.clone(),
            batch_size,
        )
        return branch_cache_locs

    def _make_welmv4_mtp_step_major_branch_cache_locs(
        self,
        flat_cache_locs: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        return (
            flat_cache_locs.reshape(batch_size, self.topk, self.speculative_num_steps)
            .permute(2, 0, 1)
            .reshape(self.speculative_num_steps, batch_size * self.topk)
            .contiguous()
        )

    @staticmethod
    def _clear_welmv4_mtp_kv_mirror_contract_metadata(
        forward_batch: ForwardBatch,
    ) -> None:
        for name in (
            "welm_kv_mirror_last_q_indices",
            "welm_kv_mirror_active_batch_indices",
            "welm_kv_mirror_output_size",
            "kv_mirror_active_batch_indices",
            "kv_mirror_output_size",
            "welm_kv_mirror_full_q_attention",
        ):
            setattr(forward_batch, name, None)
        forward_batch.welm_kv_mirror_contracted = False

    def _select_welmv4_mtp_request_hidden_states(
        self,
        forward_batch: ForwardBatch,
        hidden_states: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if hidden_states is None:
            return None

        batch_size = int(forward_batch.batch_size)
        if batch_size == 0:
            return hidden_states[:0]
        if hidden_states.shape[0] == batch_size:
            return hidden_states

        custom_last_index = getattr(forward_batch, "custom_last_index", None)
        if (
            custom_last_index is not None
            and int(custom_last_index.numel()) == batch_size
        ):
            return hidden_states[
                custom_last_index.to(device=hidden_states.device).long()
            ]

        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is not None:
            if isinstance(extend_seq_lens, torch.Tensor):
                lens = extend_seq_lens.to(device=hidden_states.device, dtype=torch.long)
            else:
                lens = torch.tensor(
                    [int(x) for x in extend_seq_lens],
                    dtype=torch.long,
                    device=hidden_states.device,
                )
            if int(lens.numel()) == batch_size:
                last_indices = torch.cumsum(lens, dim=0) - 1
                if (
                    bool((last_indices >= 0).all().item())
                    and int(last_indices[-1].item()) < hidden_states.shape[0]
                ):
                    return hidden_states[last_indices]

        if hidden_states.shape[0] % batch_size == 0:
            block_size = hidden_states.shape[0] // batch_size
            last_indices = (
                torch.arange(batch_size, device=hidden_states.device, dtype=torch.long)
                * block_size
                + block_size
                - 1
            )
            return hidden_states[last_indices]

        raise RuntimeError(
            "WeLM MTP merged draft hidden states are not request-aligned: "
            f"hidden_rows={hidden_states.shape[0]}, batch_size={batch_size}, "
            f"extend_seq_lens={extend_seq_lens}."
        )

    @staticmethod
    def _get_welmv4_mtp_int_list_attr(
        obj,
        name: str,
        num_dp_slots: int,
    ) -> Optional[List[int]]:
        values = getattr(obj, name, None)
        if values is None:
            return None
        values = [int(x) for x in values]
        if len(values) != num_dp_slots:
            raise RuntimeError(
                f"WeLM MTP DP metadata got mismatched {name}: "
                f"values={values}, dp_slots={num_dp_slots}."
            )
        return values

    @staticmethod
    def _is_welmv4_mtp_prefill_forward_mode(mode_value: Optional[int]) -> bool:
        if mode_value is None:
            return False
        try:
            return ForwardMode(int(mode_value)).is_extend_without_speculative()
        except ValueError:
            return False

    def _get_welmv4_mtp_base_request_dp_counts(
        self,
        forward_batch: ForwardBatch,
        local_rows: int,
    ) -> Optional[List[int]]:
        if (
            not is_dp_attention_enabled()
            or getattr(forward_batch, "global_num_tokens_gpu", None) is None
        ):
            return None

        num_dp_slots = int(forward_batch.global_num_tokens_gpu.numel())
        if num_dp_slots <= 1:
            return [int(local_rows)]

        counts = self._get_welmv4_mtp_int_list_attr(
            forward_batch, "global_num_reqs_cpu", num_dp_slots
        )
        if counts is None and self.topk == 1:
            counts = self._get_welmv4_mtp_int_list_attr(
                forward_batch, "global_num_tokens_for_logprob_cpu", num_dp_slots
            )
        if counts is None:
            if get_is_capture_mode():
                return [int(local_rows)] * num_dp_slots
            return None

        prefill_counts = self._get_welmv4_mtp_int_list_attr(
            forward_batch, "welm_mtp_global_prefill_num_tokens", num_dp_slots
        )
        forward_modes = self._get_welmv4_mtp_int_list_attr(
            forward_batch, "global_forward_modes", num_dp_slots
        )
        if prefill_counts is not None and forward_modes is not None:
            for i, (prefill_count, mode) in enumerate(
                zip(prefill_counts, forward_modes)
            ):
                if prefill_count == 0 and (
                    self._is_welmv4_mtp_prefill_forward_mode(mode)
                    or mode == ForwardMode.IDLE.value
                ):
                    counts[i] = 0

        dp_rank = int(get_attention_dp_rank())
        if not 0 <= dp_rank < num_dp_slots:
            raise RuntimeError(
                "WeLM MTP DP request-count inference got an invalid attention DP "
                f"rank: dp_rank={dp_rank}, dp_slots={num_dp_slots}."
            )
        if counts[dp_rank] != int(local_rows):
            raise RuntimeError(
                "WeLM MTP DP request-count inference mismatch: "
                f"dp_rank={dp_rank}, local_rows={local_rows}, counts={counts}."
            )
        return counts

    def _get_welmv4_mtp_local_row_dp_counts(
        self,
        forward_batch: ForwardBatch,
        local_rows: int,
    ) -> Optional[List[int]]:
        if (
            not is_dp_attention_enabled()
            or getattr(forward_batch, "global_num_tokens_gpu", None) is None
        ):
            return None

        num_dp_slots = int(forward_batch.global_num_tokens_gpu.numel())
        if num_dp_slots <= 1:
            return [int(local_rows)]
        if get_is_capture_mode():
            return None

        tp_group = get_tp_group()
        if tp_group.world_size % num_dp_slots != 0:
            raise RuntimeError(
                "WeLM MTP DP row-count gather expected the TP group size to be "
                f"divisible by DP slots: tp_group={tp_group.world_size}, "
                f"dp_slots={num_dp_slots}."
            )

        local_count = torch.tensor(
            [int(local_rows)],
            dtype=torch.int64,
            device=forward_batch.global_num_tokens_gpu.device,
        )
        gathered_counts = torch.empty(
            (tp_group.world_size,),
            dtype=torch.int64,
            device=local_count.device,
        )
        tp_group.all_gather_into_tensor(gathered_counts, local_count)
        ranks_per_dp = tp_group.world_size // num_dp_slots
        counts = (
            gathered_counts.view(num_dp_slots, ranks_per_dp)
            .max(dim=1)
            .values.detach()
            .cpu()
            .tolist()
        )

        dp_rank = int(get_attention_dp_rank())
        if not 0 <= dp_rank < num_dp_slots:
            raise RuntimeError(
                "WeLM MTP DP row-count gather got an invalid attention DP rank: "
                f"dp_rank={dp_rank}, dp_slots={num_dp_slots}."
            )
        if counts[dp_rank] != int(local_rows):
            raise RuntimeError(
                "WeLM MTP DP row-count gather mismatch: "
                f"dp_rank={dp_rank}, local_rows={local_rows}, counts={counts}."
            )
        return counts

    def _get_welmv4_mtp_step0_dp_token_counts(
        self,
        forward_batch: ForwardBatch,
        token_counts: Optional[List[int]],
        base_request_counts: Optional[List[int]],
    ) -> Optional[List[int]]:
        if (
            not is_dp_attention_enabled()
            or getattr(forward_batch, "global_num_tokens_gpu", None) is None
            or token_counts is None
        ):
            return token_counts

        num_dp_slots = int(forward_batch.global_num_tokens_gpu.numel())
        counts = [int(x) for x in token_counts]
        if len(counts) != num_dp_slots:
            raise RuntimeError(
                "WeLM MTP DP token-count inference got mismatched counts: "
                f"counts={counts}, dp_slots={num_dp_slots}."
            )

        prefill_counts = self._get_welmv4_mtp_int_list_attr(
            forward_batch, "welm_mtp_global_prefill_num_tokens", num_dp_slots
        )
        forward_modes = self._get_welmv4_mtp_int_list_attr(
            forward_batch, "global_forward_modes", num_dp_slots
        )
        if prefill_counts is None or forward_modes is None:
            return counts

        for i, (prefill_count, mode) in enumerate(zip(prefill_counts, forward_modes)):
            if prefill_count > 0:
                counts[i] = prefill_count
            elif (
                self._is_welmv4_mtp_prefill_forward_mode(mode)
                or mode == ForwardMode.IDLE.value
            ):
                counts[i] = 0
            elif base_request_counts is not None and ForwardMode(int(mode)).is_decode():
                counts[i] = (
                    int(base_request_counts[i]) * self.speculative_num_draft_tokens
                )
        return counts

    def _get_welmv4_mtp_step_dp_counts_from_base_requests(
        self,
        forward_batch: ForwardBatch,
        base_request_counts: Optional[List[int]],
        *,
        step: int,
        local_rows: int,
        use_tree_proposal: bool,
    ) -> Optional[List[int]]:
        if (
            not is_dp_attention_enabled()
            or getattr(forward_batch, "global_num_tokens_gpu", None) is None
        ):
            return None

        num_dp_slots = int(forward_batch.global_num_tokens_gpu.numel())
        if base_request_counts is None:
            raise RuntimeError(
                "WeLM MTP DP step-count inference requires synchronized request "
                "counts."
            )
        if len(base_request_counts) != num_dp_slots:
            raise RuntimeError(
                "WeLM MTP DP step-count inference got mismatched row counts: "
                f"counts={base_request_counts}, dp_slots={num_dp_slots}."
            )
        if num_dp_slots <= 1:
            return [int(local_rows)]

        dp_rank = int(get_attention_dp_rank())
        if not 0 <= dp_rank < num_dp_slots:
            raise RuntimeError(
                "WeLM MTP DP step-count inference got an invalid attention DP rank: "
                f"dp_rank={dp_rank}, dp_slots={num_dp_slots}."
            )

        # The step 0 forward consumes one input row per request. Its tree
        # selection expands the *next* step to topk rows per request.
        rows_per_request = self.topk if use_tree_proposal and step > 0 else 1
        counts = [int(count) * rows_per_request for count in base_request_counts]
        expected_local_rows = counts[dp_rank]
        if expected_local_rows != local_rows:
            raise RuntimeError(
                "WeLM MTP DP step-count inference mismatch: "
                f"step={step}, local_rows={local_rows}, "
                f"expected_local_rows={expected_local_rows}, "
                f"rows_per_request={rows_per_request}, "
                f"base_request_counts={base_request_counts}."
            )

        return counts

    def _set_welmv4_mtp_step_dp_counts(
        self,
        forward_batch: ForwardBatch,
        token_counts: Optional[List[int]],
        logprob_counts: Optional[List[int]] = None,
    ) -> None:
        if token_counts is None and logprob_counts is None:
            return
        if token_counts is not None:
            token_counts = [int(x) for x in token_counts]
            forward_batch.global_num_tokens_cpu = token_counts
        if logprob_counts is None:
            logprob_counts = token_counts
        else:
            logprob_counts = [int(x) for x in logprob_counts]
        if logprob_counts is not None:
            forward_batch.global_num_tokens_for_logprob_cpu = logprob_counts
        forward_batch._welm_mtp_contract_global_num_tokens_cpu = logprob_counts
        forward_batch.dp_local_start_pos = None
        forward_batch.dp_local_num_tokens = None

    @staticmethod
    def _is_welmv4_mtp_intermediate_prefill_chunk(req) -> bool:
        if getattr(req, "is_chunked", 0) <= 0:
            return False

        fill_ids = getattr(req, "fill_ids", None)
        origin_input_ids = getattr(req, "origin_input_ids", None)
        if fill_ids is None or origin_input_ids is None:
            return True

        output_ids = getattr(req, "output_ids", ())
        return len(fill_ids) < len(origin_input_ids) + len(output_ids)

    def _should_defer_welmv4_mtp_prefill_draft(self, batch: ModelWorkerBatch) -> bool:
        """Return True for intermediate chunked prefill chunks."""

        deferred_mask = self._get_welmv4_mtp_deferred_prefill_mask(batch)
        return (
            deferred_mask is not None
            and deferred_mask.numel() > 0
            and bool(deferred_mask.all().item())
        )

    def _get_welmv4_mtp_deferred_prefill_mask(
        self,
        batch: ModelWorkerBatch,
        device: Optional[torch.device] = None,
    ) -> Optional[torch.Tensor]:
        reqs = getattr(batch, "reqs", None)
        if not reqs:
            return None
        intermediate_chunks = [
            self._is_welmv4_mtp_intermediate_prefill_chunk(req) for req in reqs
        ]
        if not any(intermediate_chunks):
            return None
        return torch.tensor(
            intermediate_chunks,
            dtype=torch.bool,
            device=device,
        )

    @staticmethod
    def _get_welmv4_mtp_global_prefill_num_tokens(
        batch: ModelWorkerBatch,
    ) -> Optional[List[int]]:
        counts = getattr(batch, "welm_mtp_global_prefill_num_tokens", None)
        if counts is None:
            return None
        return [int(x) for x in counts]

    def _prepare_welmv4_mtp_empty_prefill_collective_batch(
        self,
        batch: ModelWorkerBatch,
    ) -> None:
        device = self.device
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        empty_i64_cpu = torch.empty((0,), dtype=torch.int64)

        batch.forward_mode = ForwardMode.IDLE
        batch.input_ids = empty_i64
        batch.req_pool_indices = empty_i64
        batch.seq_lens = empty_i32
        batch.seq_lens_cpu = empty_i64_cpu
        batch.orig_seq_lens = empty_i32
        batch.out_cache_loc = empty_i64
        batch.seq_lens_sum = 0
        batch.extend_seq_lens = []
        batch.extend_prefix_lens = []
        batch.extend_logprob_start_lens = []
        batch.extend_num_tokens = 0
        batch.extend_input_logprob_token_ids = None
        batch.welm_kv_mirror_last_q_indices = None
        batch.welm_kv_mirror_active_batch_indices = None
        batch.welm_kv_mirror_output_size = None
        batch.return_logprob = False
        batch.return_hidden_states = False
        batch.input_embeds = None
        batch.multimodal_inputs = None
        batch.reqs = []
        batch.lora_ids = []
        batch.oe_context = None

    @staticmethod
    def _has_welmv4_mtp_deferred_prefill_rows(
        draft_input: EagleDraftInput,
    ) -> bool:
        deferred_mask = getattr(
            draft_input, "welm_mtp_deferred_prefill_draft_mask", None
        )
        if deferred_mask is not None:
            return (
                bool(deferred_mask.any().item()) if deferred_mask.numel() > 0 else False
            )
        return bool(getattr(draft_input, "welm_mtp_deferred_prefill_draft", False))

    def _get_welmv4_mtp_last_prefill_hidden_states(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(batch.seq_lens)
        if target_hidden_states.shape[0] == batch_size:
            return target_hidden_states

        if batch.extend_seq_lens is None:
            raise RuntimeError(
                "WeLMV4 MTP deferred prefill requires extend_seq_lens when "
                "target hidden states contain more than one row per request."
            )

        last_indices_cpu = []
        offset = 0
        for extend_len in batch.extend_seq_lens:
            extend_len = int(extend_len)
            if extend_len <= 0:
                raise RuntimeError(
                    "WeLMV4 MTP deferred prefill saw a non-positive extend "
                    f"length: {extend_len}."
                )
            last_indices_cpu.append(offset + extend_len - 1)
            offset += extend_len
        if offset != target_hidden_states.shape[0]:
            raise RuntimeError(
                "WeLMV4 MTP deferred prefill hidden-state layout mismatch: "
                f"sum(extend_lens)={offset}, hidden_rows={target_hidden_states.shape[0]}."
            )

        last_indices = torch.tensor(
            last_indices_cpu,
            dtype=torch.long,
            device=target_hidden_states.device,
        )
        return target_hidden_states[last_indices]

    def _prepare_welmv4_mtp_deferred_prefill_draft_input(
        self,
        draft_input: EagleDraftInput,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
    ) -> None:
        batch_size = len(batch.seq_lens)
        draft_input.hidden_states = self._get_welmv4_mtp_last_prefill_hidden_states(
            batch, target_hidden_states
        )

        dummy_ids = next_token_ids.to(dtype=torch.int64).view(-1, 1)
        if dummy_ids.shape[0] != batch_size:
            raise RuntimeError(
                "WeLMV4 MTP deferred prefill next-token layout mismatch: "
                f"next_token_rows={dummy_ids.shape[0]}, batch_size={batch_size}."
            )
        proposal_width = self.speculative_num_steps if self.topk == 1 else self.topk
        draft_input.topk_index = dummy_ids.expand(-1, proposal_width).contiguous()
        draft_input.topk_p = torch.ones(
            (batch_size, proposal_width),
            dtype=torch.float32,
            device=next_token_ids.device,
        )
        draft_input.draft_probs = None
        draft_input.welm_mtp_draft_topk_indices = None
        draft_input.welm_mtp_draft_topk_values = None
        draft_input.draft_proposal_parent_list = None
        draft_input.draft_proposal_top_scores_index = None
        draft_input.draft_proposal_tokens = None
        draft_input.welm_mtp_base_positions = None
        draft_input.welm_mtp_deferred_prefill_draft = True
        draft_input.welm_mtp_deferred_prefill_draft_mask = torch.ones(
            (batch_size,),
            dtype=torch.bool,
            device=next_token_ids.device,
        )

    def _copy_welmv4_mtp_pd_prefill_mirror_states(
        self,
        batch: ModelWorkerBatch,
        model_specific_states,
    ) -> None:
        if not self.server_args.enable_welm_kv_mirror_opt:
            return
        buffers = getattr(self, "welm_mtp_kv_mirror_state_buffers", None)
        if not buffers:
            return
        out_cache_loc = getattr(batch, "out_cache_loc", None)
        if out_cache_loc is None or int(out_cache_loc.numel()) == 0:
            return
        copy_welmv4_mtp_kv_mirror_states_to_buffers(
            buffers,
            model_specific_states,
            indices=out_cache_loc,
        )

    def _build_welmv4_mtp_pd_prefill_mirror_states(
        self,
        batch: ModelWorkerBatch,
    ):
        if not self.server_args.enable_welm_kv_mirror_opt:
            return None
        buffers = getattr(self, "welm_mtp_kv_mirror_state_buffers", None)
        if not buffers:
            return None
        out_cache_loc = getattr(batch, "out_cache_loc", None)
        if out_cache_loc is None or int(out_cache_loc.numel()) == 0:
            return None
        return build_welmv4_mtp_model_specific_states(
            buffers,
            indices=out_cache_loc,
        )

    def _materialize_welmv4_mtp_deferred_prefill_draft(
        self,
        model_worker_batch: ModelWorkerBatch,
        draft_input: EagleDraftInput,
    ) -> None:
        deferred_mask = getattr(
            draft_input, "welm_mtp_deferred_prefill_draft_mask", None
        )
        if deferred_mask is not None and (
            deferred_mask.numel() == 0 or not bool(deferred_mask.all().item())
        ):
            raise RuntimeError(
                "Mixed WeLM MTP deferred prefill rows reached draft(); PD decode "
                "currently isolates deferred prebuilt batches before running them."
            )

        if model_worker_batch.extend_seq_lens is None:
            raise RuntimeError(
                "WeLM MTP PD decode bootstrap requires prebuilt extend_seq_lens."
            )
        if model_worker_batch.extend_prefix_lens is None:
            raise RuntimeError(
                "WeLM MTP PD decode bootstrap requires prebuilt extend_prefix_lens."
            )

        model_specific_states = self._build_welmv4_mtp_pd_prefill_mirror_states(
            model_worker_batch
        )
        if self.server_args.enable_welm_kv_mirror_opt and model_specific_states is None:
            raise RuntimeError(
                "WeLM MTP PD decode bootstrap is missing mirrored KV states."
            )

        original_attrs = {
            name: getattr(model_worker_batch, name, None)
            for name in (
                "forward_mode",
                "is_extend_in_batch",
                "capture_hidden_mode",
                "input_ids",
            )
        }
        original_model_specific_states = draft_input.model_specific_states
        first_query_hashed_inputs = None
        first_query_history_state = None
        use_oe_hash_kernel = self._should_use_welmv4_mtp_oe_hash_kernel()
        try:
            model_worker_batch.forward_mode = ForwardMode.EXTEND
            model_worker_batch.is_extend_in_batch = False
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            model_worker_batch.input_ids = model_worker_batch.input_ids.to(
                dtype=torch.int64
            )
            draft_input.model_specific_states = model_specific_states

            forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_runner)
            forward_batch.return_logprob = False
            if use_oe_hash_kernel and model_worker_batch.oe_context is not None:
                self._prepare_welmv4_mtp_segment_hash_inputs_from_prefixes(
                    forward_batch
                )
            if use_oe_hash_kernel:
                entry_history_state = self._init_welmv4_mtp_oe_history_from_extend(
                    forward_batch,
                    first_token_ids=None,
                )
                if entry_history_state is None:
                    raise RuntimeError(
                        "WeLMV4 MTP PD decode bootstrap is missing entry OE "
                        "history for the first query."
                    )
                (
                    first_query_hashed_inputs,
                    first_query_history_state,
                    first_query_verify_history_state,
                ) = self._compute_welmv4_mtp_first_query_hash_from_entry_history(
                    forward_batch,
                    draft_input.bonus_tokens,
                    entry_history_state,
                )
                draft_input.welm_mtp_oe_history_state = (
                    first_query_verify_history_state
                )

            self._run_welmv4_mtp_merged_extend_draft(
                forward_batch,
                draft_input.bonus_tokens,
                skip_attn_backend_init=False,
                first_query_hashed_inputs=first_query_hashed_inputs,
                first_query_history_state=first_query_history_state,
                draft_path="pd_decode_prefill",
            )
        finally:
            for name, value in original_attrs.items():
                setattr(model_worker_batch, name, value)
            draft_input.model_specific_states = original_model_specific_states

        draft_input.welm_mtp_deferred_prefill_draft = False
        draft_input.welm_mtp_deferred_prefill_draft_mask = None

    def _run_welmv4_mtp_merged_extend_draft(
        self,
        forward_batch: ForwardBatch,
        first_input_ids: torch.Tensor,
        *,
        skip_attn_backend_init: bool,
        first_query_hashed_inputs: Optional[torch.Tensor] = None,
        first_query_history_state: Optional["WelmMTPDraftNGramHistoryState"] = None,
        draft_path: str = "merged_extend_draft",
    ) -> None:
        use_tree_proposal = self.topk > 1
        if use_tree_proposal and self.welmv4_mtp_sample_draft:
            raise RuntimeError("WeLM MTP topk>1 does not support draft sampling yet.")

        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        input_ids = first_input_ids.to(dtype=torch.int64)
        main_hidden_states = spec_info.hidden_states
        base_positions = self._get_welmv4_mtp_base_positions(forward_batch)
        query_positions = base_positions
        draft_history_state = None
        query_hashed_inputs = first_query_hashed_inputs
        topk_p_list: List[torch.Tensor] = []
        topk_index_list: List[torch.Tensor] = []
        draft_probs_list: List[torch.Tensor] = []
        draft_topk_indices_list: List[torch.Tensor] = []
        draft_topk_values_list: List[torch.Tensor] = []
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        scores = None
        selected_parent_indices = None
        topk_cs_idx = None
        current_mirrored_kv_indices = getattr(spec_info, "mirrored_kv_indices", None)
        branch_step_cache_locs = None
        branch_flat_cache_locs = None
        branch_metadata_initialized = False
        last_logits_output = None
        forward_batch._welm_mtp_contracted_dp_metadata_rows = None
        is_idle_batch = forward_batch.forward_mode.is_idle()
        original_batch_attrs = {
            name: getattr(forward_batch, name, None)
            for name in (
                "input_ids",
                "positions",
                "mrope_positions",
                "out_cache_loc",
                "custom_last_index",
                "custom_last_cache_loc",
                "extend_num_tokens",
                "extend_seq_lens",
                "extend_prefix_lens",
                "extend_start_loc",
                "extend_seq_lens_cpu",
                "extend_prefix_lens_cpu",
                "global_num_tokens_cpu",
                "global_num_tokens_for_logprob_cpu",
                "_welm_mtp_contract_global_num_tokens_cpu",
                "num_token_non_padded",
                "num_token_non_padded_cpu",
                "lora_ids",
                "attn_backend",
                "forward_mode",
                "is_extend_in_batch",
                "next_token_logits_buffer",
            )
        }
        original_spec_token_attrs = {
            name: getattr(spec_info, name, None)
            for name in (
                "num_tokens_per_req",
                "num_tokens_for_logprob_per_req",
                "mirrored_kv_indices",
            )
        }
        prefill_token_counts = (
            None
            if forward_batch.global_num_tokens_cpu is None
            else [int(x) for x in forward_batch.global_num_tokens_cpu]
        )
        base_request_counts = self._get_welmv4_mtp_base_request_dp_counts(
            forward_batch,
            int(input_ids.numel()),
        )
        variable_decode_token_counts = None
        if getattr(forward_batch, "welm_mtp_variable_decode_extend", False):
            variable_decode_token_counts = self._get_welmv4_mtp_local_row_dp_counts(
                forward_batch,
                int(forward_batch.extend_num_tokens),
            )
        step0_token_counts = (
            variable_decode_token_counts
            if variable_decode_token_counts is not None
            else self._get_welmv4_mtp_step0_dp_token_counts(
                forward_batch,
                prefill_token_counts,
                base_request_counts,
            )
        )

        try:
            for step in range(self.speculative_num_steps):
                request_token_counts = (
                    self._get_welmv4_mtp_step_dp_counts_from_base_requests(
                        forward_batch,
                        base_request_counts,
                        step=step,
                        local_rows=int(input_ids.numel()),
                        use_tree_proposal=use_tree_proposal,
                    )
                )
                self._set_welmv4_mtp_step_dp_counts(
                    forward_batch,
                    (
                        request_token_counts
                        if use_tree_proposal and step > 0 and not get_is_capture_mode()
                        else step0_token_counts
                    ),
                    logprob_counts=request_token_counts,
                )

                _welm_mtp_trace(
                    "merged_extend_draft_step_start "
                    f"path={draft_path} step={step} mode={forward_batch.forward_mode} "
                    f"is_idle={is_idle_batch} batch_size={forward_batch.batch_size} "
                    f"input_shape={tuple(input_ids.shape)} "
                    f"hidden_shape={None if main_hidden_states is None else tuple(main_hidden_states.shape)} "
                    f"global_tokens={getattr(forward_batch, 'global_num_tokens_cpu', None)}"
                )
                self._check_welmv4_mtp_token_range(
                    input_ids,
                    self.draft_runner.model_config.vocab_size,
                    f"{draft_path} step{step} input",
                )
                if (
                    step > 0
                    and self._should_use_welmv4_mtp_oe_hash_kernel()
                    and not is_idle_batch
                ):
                    if first_query_history_state is None:
                        raise RuntimeError(
                            "WeLMV4 MTP merged_extend_draft is missing OE history "
                            f"for {draft_path} step{step}."
                        )
                    query_hashed_inputs, draft_history_state = (
                        self._compute_welmv4_mtp_draft_decode_hash_inputs(
                            forward_batch,
                            input_ids,
                            first_query_history_state,
                            draft_history_state,
                            base_query_count=int(first_query_history_state.shape[0]),
                            step_idx=step,
                            selected_parent_indices=selected_parent_indices,
                            topk_cs_idx=topk_cs_idx,
                            use_forward_hash_buffer=False,
                        )
                    )
                elif (
                    step == 0
                    and self._should_use_welmv4_mtp_oe_hash_kernel()
                    and not is_idle_batch
                ):
                    if query_hashed_inputs is None:
                        raise RuntimeError(
                            "WeLMV4 MTP merged_extend_draft step0 is missing query "
                            f"OE hashes for {draft_path}."
                        )

                if use_tree_proposal and step > 0:
                    if branch_step_cache_locs is None or branch_flat_cache_locs is None:
                        branch_step_cache_locs, branch_flat_cache_locs = (
                            self._get_welmv4_mtp_branch_cache_locs(forward_batch)
                        )
                    if not branch_metadata_initialized:
                        if self.draft_attn_backend is None:
                            raise RuntimeError(
                                "WeLM MTP topk>1 requires a draft attention backend."
                            )
                        forward_batch.out_cache_loc = branch_flat_cache_locs
                        forward_batch.positions = base_positions.repeat_interleave(
                            self.topk
                        ).to(dtype=forward_batch.positions.dtype)
                        spec_info.num_tokens_per_req = self.topk
                        spec_info.num_tokens_for_logprob_per_req = self.topk
                        if not is_idle_batch and not getattr(
                            forward_batch,
                            "welm_mtp_draft_tree_graph_metadata_ready",
                            False,
                        ):
                            forward_batch.forward_mode = ForwardMode.DECODE
                            forward_batch.is_extend_in_batch = True
                            self.draft_attn_backend.init_forward_metadata(forward_batch)
                        branch_metadata_initialized = True

                    branch_rows = int(input_ids.numel())
                    if selected_parent_indices is not None:
                        if current_mirrored_kv_indices is None:
                            current_mirrored_kv_indices = selected_parent_indices.to(
                                dtype=torch.long
                            )
                        else:
                            maybe_detect_oob(
                                selected_parent_indices,
                                0,
                                current_mirrored_kv_indices.shape[0],
                                "WeLM MTP topk>1 branch mirrored_kv_indices parent",
                            )
                            current_mirrored_kv_indices = current_mirrored_kv_indices[
                                selected_parent_indices.to(
                                    device=current_mirrored_kv_indices.device,
                                    dtype=torch.long,
                                )
                            ]
                        if current_mirrored_kv_indices.numel() != branch_rows:
                            raise RuntimeError(
                                "WeLM MTP topk>1 branch mirror indices shape "
                                "mismatch: "
                                f"{current_mirrored_kv_indices.numel()} vs "
                                f"{branch_rows}."
                            )
                        spec_info.mirrored_kv_indices = current_mirrored_kv_indices
                    forward_batch.input_ids = input_ids
                    forward_batch.out_cache_loc = branch_step_cache_locs[
                        step - 1, :branch_rows
                    ]
                    query_positions = (base_positions + step).repeat_interleave(
                        self.topk
                    )[:branch_rows]
                    forward_batch.positions = query_positions.to(
                        dtype=forward_batch.positions.dtype
                    )
                    forward_batch.mrope_positions = None
                    forward_batch.custom_last_index = torch.arange(
                        branch_rows,
                        dtype=torch.long,
                        device=input_ids.device,
                    )
                    forward_batch.custom_last_cache_loc = forward_batch.out_cache_loc
                    forward_batch.attn_backend = self.draft_attn_backend.attn_backends[
                        step - 1
                    ]
                    forward_batch.forward_mode = (
                        ForwardMode.IDLE if is_idle_batch else ForwardMode.DECODE
                    )
                    forward_batch.is_extend_in_batch = True
                    forward_batch.next_token_logits_buffer = getattr(
                        forward_batch,
                        "welm_mtp_branch_next_token_logits_buffer",
                        None,
                    )
                elif use_tree_proposal:
                    forward_batch.next_token_logits_buffer = getattr(
                        forward_batch,
                        "welm_mtp_first_next_token_logits_buffer",
                        original_batch_attrs.get("next_token_logits_buffer"),
                    )

                logits_output = self._forward_welmv4_mtp_merged_extend_draft_step(
                    forward_batch,
                    step,
                    input_ids,
                    main_hidden_states,
                    query_hashed_inputs,
                    query_positions,
                    skip_attn_backend_init=skip_attn_backend_init
                    or (use_tree_proposal and step > 0),
                )
                _welm_mtp_trace(
                    "merged_extend_draft_step_after_forward "
                    f"path={draft_path} step={step} mode={forward_batch.forward_mode} "
                    f"logits_shape={tuple(logits_output.next_token_logits.shape)} "
                    f"global_tokens={getattr(forward_batch, 'global_num_tokens_cpu', None)}"
                )
                graph_select_fn = getattr(
                    forward_batch, "welm_mtp_draft_graph_select_fn", None
                )
                if is_idle_batch:
                    proposal_width = self.topk
                    topk_p = logits_output.next_token_logits.new_zeros(
                        (int(logits_output.next_token_logits.shape[0]), proposal_width)
                    )
                    topk_index = torch.zeros(
                        topk_p.shape,
                        dtype=torch.long,
                        device=logits_output.next_token_logits.device,
                    )
                    draft_probs = None
                    draft_topk_indices = None
                    draft_topk_values = None
                elif graph_select_fn is None:
                    (
                        topk_p,
                        topk_index,
                        draft_probs,
                        draft_topk_indices,
                        draft_topk_values,
                    ) = self._select_or_sample_welmv4_mtp_draft_topk(
                        logits_output.next_token_logits,
                        forward_batch,
                        base_positions=base_positions,
                        step=step,
                    )
                else:
                    (
                        topk_p,
                        topk_index,
                        draft_topk_indices,
                        draft_topk_values,
                    ) = graph_select_fn(logits_output.next_token_logits, step)
                    draft_probs = None
                maybe_detect_oob(
                    topk_index,
                    0,
                    logits_output.next_token_logits.shape[-1],
                    f"{draft_path} step{step}: topk_index OOB vs "
                    f"vocab_size={logits_output.next_token_logits.shape[-1]}",
                )
                mapped_topk_index = self._maybe_map_hot_token_id(topk_index)
                self._check_welmv4_mtp_token_range(
                    mapped_topk_index,
                    self.target_worker.model_config.vocab_size,
                    f"{draft_path} step{step} topk",
                )
                if use_tree_proposal:
                    step_hidden_states = logits_output.hidden_states
                    if step == 0:
                        step_hidden_states = (
                            self._select_welmv4_mtp_request_hidden_states(
                                forward_batch,
                                step_hidden_states,
                            )
                        )
                    (
                        input_ids,
                        main_hidden_states,
                        scores,
                        tree_info,
                        selected_parent_indices,
                    ) = self._select_welmv4_mtp_tree_step(
                        step,
                        topk_p,
                        mapped_topk_index,
                        step_hidden_states,
                        scores,
                    )
                    draft_input_id_buffers = getattr(
                        forward_batch, "welm_mtp_draft_input_ids", None
                    )
                    if draft_input_id_buffers is not None:
                        stable_input_ids = draft_input_id_buffers[
                            step, : input_ids.numel()
                        ]
                        stable_input_ids.copy_(input_ids)
                        input_ids = stable_input_ids
                    topk_cs_idx = self._get_welmv4_mtp_topk_cs_indices(
                        step,
                        tree_info,
                        forward_batch,
                    )
                    score_list.append(tree_info[0])
                    token_list.append(tree_info[1])
                    parents_list.append(tree_info[2])
                    if step == 0:
                        topk_p_list.append(topk_p)
                        topk_index_list.append(topk_index)
                else:
                    topk_p_list.append(topk_p)
                    topk_index_list.append(topk_index)
                if draft_probs is not None:
                    draft_probs_list.append(draft_probs.unsqueeze(1))
                if draft_topk_indices is not None:
                    draft_topk_indices_list.append(draft_topk_indices.unsqueeze(1))
                    draft_topk_values_list.append(draft_topk_values.unsqueeze(1))

                if not use_tree_proposal:
                    main_hidden_states = logits_output.hidden_states
                    next_input_ids = mapped_topk_index.flatten().to(dtype=torch.int64)
                    draft_input_id_buffers = getattr(
                        forward_batch, "welm_mtp_draft_input_ids", None
                    )
                    if getattr(
                        forward_batch, "welm_mtp_linear_proposal_fast_path", False
                    ):
                        # CUDA graph allocations have stable addresses.  For
                        # top-k=1 the sampled-id output can feed the next MTP
                        # step directly; copying it through one runner-owned
                        # buffer per step only adds three tiny graph nodes.
                        input_ids = next_input_ids
                    elif draft_input_id_buffers is not None:
                        input_ids = draft_input_id_buffers[
                            step, : next_input_ids.numel()
                        ]
                        input_ids.copy_(next_input_ids)
                    else:
                        input_ids = next_input_ids
                last_logits_output = logits_output
        finally:
            for name, value in original_batch_attrs.items():
                setattr(forward_batch, name, value)
            for name, value in original_spec_token_attrs.items():
                setattr(spec_info, name, value)

        linear_graph_outputs = getattr(
            forward_batch, "welm_mtp_linear_graph_output_buffers", None
        )
        fused_linear_graph_outputs = bool(
            linear_graph_outputs is not None
            and not use_tree_proposal
            and not draft_probs_list
            and len(topk_p_list) == 3
            and len(topk_index_list) == 3
            and len(draft_topk_indices_list) == 3
            and len(draft_topk_values_list) == 3
        )
        if fused_linear_graph_outputs:
            from sglang.srt.speculative.welmv4_mtp_staging import (
                pack_welm_mtp_linear_graph_outputs,
            )

            pack_welm_mtp_linear_graph_outputs(
                sampled_probs=topk_p_list,
                sampled_indices=topk_index_list,
                draft_topk_indices=draft_topk_indices_list,
                draft_topk_values=draft_topk_values_list,
                bonus_tokens=linear_graph_outputs["bonus_tokens"],
                base_positions=linear_graph_outputs["base_positions"],
                hot_token_map=self.hot_token_id,
                output_topk_p=linear_graph_outputs["topk_p"],
                output_topk_index=linear_graph_outputs["topk_index"],
                output_proposal_tokens=linear_graph_outputs["proposal_tokens"],
                output_draft_indices=linear_graph_outputs["draft_topk_indices"],
                output_draft_values=linear_graph_outputs["draft_topk_values"],
                output_verify_tokens=linear_graph_outputs["verify_tokens"],
                output_verify_positions=linear_graph_outputs["verify_positions"],
                batch_size=int(forward_batch.batch_size),
            )
            spec_info.topk_p = linear_graph_outputs["topk_p"]
            spec_info.topk_index = linear_graph_outputs["topk_index"]
            spec_info.draft_probs = None
            spec_info.welm_mtp_draft_topk_indices = linear_graph_outputs[
                "draft_topk_indices"
            ]
            spec_info.welm_mtp_draft_topk_values = linear_graph_outputs[
                "draft_topk_values"
            ]
        else:
            spec_info.topk_p = torch.cat(topk_p_list, dim=1)
            spec_info.topk_index = torch.cat(topk_index_list, dim=1)
            spec_info.draft_probs = self._pad_welmv4_mtp_draft_probs_for_verify(
                draft_probs_list
            )
            spec_info.welm_mtp_draft_topk_indices = (
                self._pad_welmv4_mtp_draft_topk_for_verify(
                    draft_topk_indices_list,
                    pad_value=0,
                )
            )
            spec_info.welm_mtp_draft_topk_values = (
                self._pad_welmv4_mtp_draft_topk_for_verify(
                    draft_topk_values_list,
                    pad_value=0.0,
                )
            )
        if use_tree_proposal and main_hidden_states is not None:
            final_hidden_states = main_hidden_states.reshape(
                int(base_positions.numel()), self.topk, main_hidden_states.shape[-1]
            )[:, 0, :]
        else:
            final_hidden_states = self._select_welmv4_mtp_request_hidden_states(
                forward_batch,
                (
                    None
                    if last_logits_output is None
                    else last_logits_output.hidden_states
                ),
            )
        spec_info.hidden_states = final_hidden_states
        spec_info.welm_mtp_deferred_prefill_draft = (
            self._has_welmv4_mtp_deferred_prefill_rows(spec_info)
        )
        spec_info.welm_mtp_base_positions = base_positions
        if use_tree_proposal:
            (
                spec_info.draft_proposal_parent_list,
                spec_info.draft_proposal_top_scores_index,
                spec_info.draft_proposal_tokens,
            ) = organize_draft_results(
                score_list,
                token_list,
                parents_list,
                self.speculative_num_draft_tokens,
            )
        elif getattr(forward_batch, "welm_mtp_linear_proposal_fast_path", False):
            # With top-k=1 every sampled step is retained in order.  Therefore
            # proposal_tokens is exactly the concatenated sampled-token tensor,
            # parent_list is [-1, 0, 1, ...], and top_scores_index is
            # [0, 1, 2, ...].  Avoid the generic cumulative-score/topk/sort/
            # gather construction inside the proposal CUDA graph.
            spec_info.draft_proposal_parent_list = (
                forward_batch.welm_mtp_linear_proposal_parent_list
            )
            spec_info.draft_proposal_top_scores_index = (
                forward_batch.welm_mtp_linear_proposal_top_scores_index
            )
            spec_info.draft_proposal_tokens = (
                linear_graph_outputs["proposal_tokens"]
                if fused_linear_graph_outputs
                else self._maybe_map_hot_token_id(spec_info.topk_index)
            )
        else:
            (
                spec_info.draft_proposal_parent_list,
                spec_info.draft_proposal_top_scores_index,
                spec_info.draft_proposal_tokens,
            ) = self._build_welmv4_mtp_linear_draft_proposal(spec_info)
        forward_batch.mtp_step_idx = 0

    @staticmethod
    def _copy_welmv4_mtp_draft_state(
        src: EagleDraftInput,
        dst: EagleDraftInput,
    ) -> None:
        dst.topk_p = src.topk_p
        dst.topk_index = src.topk_index
        dst.hidden_states = src.hidden_states
        dst.draft_probs = src.draft_probs
        dst.welm_mtp_draft_topk_indices = src.welm_mtp_draft_topk_indices
        dst.welm_mtp_draft_topk_values = src.welm_mtp_draft_topk_values
        dst.draft_proposal_parent_list = src.draft_proposal_parent_list
        dst.draft_proposal_top_scores_index = src.draft_proposal_top_scores_index
        dst.draft_proposal_tokens = src.draft_proposal_tokens
        dst.welm_mtp_base_positions = src.welm_mtp_base_positions
        dst.welm_mtp_deferred_prefill_draft = (
            EagleDraftWorker._has_welmv4_mtp_deferred_prefill_rows(src)
        )
        dst.welm_mtp_deferred_prefill_draft_mask = getattr(
            src, "welm_mtp_deferred_prefill_draft_mask", None
        )

    def _build_welmv4_mtp_linear_draft_proposal(
        self,
        draft_input: EagleDraftInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_p = draft_input.topk_p
        topk_index = self._maybe_map_hot_token_id(draft_input.topk_index)
        if topk_p is None or topk_index is None:
            raise RuntimeError("WeLM MTP draft proposal is missing topk tensors.")
        if (
            topk_p.ndim != 2
            or topk_index.ndim != 2
            or topk_p.shape != topk_index.shape
            or topk_p.shape[1] != self.speculative_num_steps
        ):
            raise RuntimeError(
                "Linear WeLM MTP draft proposal expects one topk entry per "
                f"step: topk_p={None if topk_p is None else tuple(topk_p.shape)}, "
                f"topk_index={None if topk_index is None else tuple(topk_index.shape)}, "
                f"steps={self.speculative_num_steps}."
            )

        batch_size = int(topk_p.shape[0])
        first_parent = (
            torch.arange(-1, self.topk, dtype=torch.long, device=topk_p.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        cumulative_score = None
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        for step in range(self.speculative_num_steps):
            step_score = topk_p[:, step].view(batch_size, 1)
            cumulative_score = (
                step_score
                if cumulative_score is None
                else cumulative_score * step_score
            )
            score_list.append(cumulative_score.view(batch_size, 1, 1))
            token_list.append(topk_index[:, step].view(batch_size, 1))
            if step == 0:
                parents_list.append(first_parent)
            else:
                parents_list.append(
                    torch.full(
                        (batch_size, 1),
                        step,
                        dtype=torch.long,
                        device=topk_p.device,
                    )
                )

        return organize_draft_results(
            score_list,
            token_list,
            parents_list,
            self.speculative_num_draft_tokens,
        )

    def _build_welmv4_mtp_draft_proposal_results(
        self,
        draft_input: EagleDraftInput,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parent_list = getattr(draft_input, "draft_proposal_parent_list", None)
        top_scores_index = getattr(draft_input, "draft_proposal_top_scores_index", None)
        draft_tokens = getattr(draft_input, "draft_proposal_tokens", None)
        if (
            parent_list is not None
            and top_scores_index is not None
            and draft_tokens is not None
        ):
            return parent_list, top_scores_index, draft_tokens
        return self._build_welmv4_mtp_linear_draft_proposal(draft_input)

    def _is_welmv4_mtp_linear_draft_proposal_input(
        self,
        draft_input: EagleDraftInput,
    ) -> bool:
        topk_p = draft_input.topk_p
        topk_index = draft_input.topk_index
        return (
            self.topk == 1
            and topk_p is not None
            and topk_index is not None
            and topk_p.ndim == 2
            and topk_index.ndim == 2
            and topk_p.shape == topk_index.shape
            and topk_p.shape[1] == self.speculative_num_steps
        )

    def _has_welmv4_mtp_draft_proposal(
        self,
        draft_input: EagleDraftInput,
    ) -> bool:
        return (
            getattr(draft_input, "draft_proposal_parent_list", None) is not None
            and getattr(draft_input, "draft_proposal_top_scores_index", None)
            is not None
            and getattr(draft_input, "draft_proposal_tokens", None) is not None
        ) or self._is_welmv4_mtp_linear_draft_proposal_input(draft_input)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.has_prefill_wrapper_verify = False
        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None
        self.cuda_graph_runner_for_draft_proposal = None

        if self.server_args.disable_cuda_graph:
            return

        if self.server_args.model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        use_welmv4_mtp_draft_proposal_graph = self._is_welmv4_mtp_draft_model()
        skip_generic_welmv4_mtp_graph = use_welmv4_mtp_draft_proposal_graph
        # Capture draft
        if self.speculative_num_steps > 1 and not skip_generic_welmv4_mtp_graph:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )
        elif self.speculative_num_steps > 1 and skip_generic_welmv4_mtp_graph:
            logger.info(
                "Skip generic WeLM MTP draft cuda graph; WeLM MTP uses "
                "merged draft proposal generation."
            )

        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_aiter_draft_extend_graph = False
        if _is_hip:
            # Keep import local so non-HIP environments do not require aiter.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        supports_cuda_draft_extend_graph = (
            _is_cuda
            and isinstance(self.draft_extend_attn_backend, FlashAttentionBackend)
        ) or (
            (_is_cuda or _is_musa)
            and (
                isinstance(self.draft_extend_attn_backend, TritonAttnBackend)
                or isinstance(
                    self.draft_extend_attn_backend, TRTLLMHAAttnBackend
                )
                or isinstance(self.draft_extend_attn_backend, TRTLLMMLABackend)
            )
        )
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        supports_draft_extend_graph_backend = (
            _is_npu
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        )
        if (
            self.draft_extend_attn_backend
            and not skip_generic_welmv4_mtp_graph
            and supports_draft_extend_graph_backend
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )
        elif self.draft_extend_attn_backend and skip_generic_welmv4_mtp_graph:
            logger.info(
                "Skip generic WeLM MTP draft-extend cuda graph; WeLM MTP uses "
                "merged draft proposal generation."
            )

        if (
            self.draft_extend_attn_backend
            and use_welmv4_mtp_draft_proposal_graph
            and supports_draft_extend_graph_backend
            and _WELM_MTP_DRAFT_CUDA_GRAPH_ENABLED
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                "Capture WeLM MTP draft proposal cuda graph begin. "
                f"avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_proposal = (
                WelmMTPDraftProposalCudaGraphRunner(self)
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                "Capture WeLM MTP draft proposal cuda graph end. "
                f"Time elapsed: {time.perf_counter() - tic:.2f} s. "
                f"mem usage={(before_mem - after_mem):.2f} GB. "
                f"avail mem={after_mem:.2f} GB."
            )
            logger.info(
                "Capture draft cuda graph end. WeLM MTP uses unified draft "
                "proposal cuda graph."
            )
        elif (
            self.draft_extend_attn_backend
            and use_welmv4_mtp_draft_proposal_graph
            and not _WELM_MTP_DRAFT_CUDA_GRAPH_ENABLED
        ):
            logger.info(
                "Skip WeLM MTP unified draft proposal cuda graph because "
                "SGLANG_WELM_MTP_DRAFT_CUDA_GRAPH is disabled."
            )

        if (
            use_welmv4_mtp_draft_proposal_graph
            and self.topk > 1
            and self.cuda_graph_runner_for_draft_proposal is None
        ):
            logger.warning(
                "WeLM MTP topk>1 is running without the unified draft proposal "
                "CUDA graph. This is supported for debugging/fallback, but can "
                "significantly reduce performance."
            )

    def draft(self, model_worker_batch: ModelWorkerBatch):
        draft_input: EagleDraftInput = model_worker_batch.spec_info
        prebuilt_verify = getattr(draft_input, "welm_mtp_prebuilt_verify_input", None)
        if (
            prebuilt_verify is not None
            and not model_worker_batch.forward_mode.is_idle()
            and getattr(draft_input, "future_indices", None) is None
            and int(getattr(draft_input, "welm_mtp_prebuilt_verify_bs", -1))
            == int(model_worker_batch.seq_lens.shape[0])
        ):
            # The previous verify->proposal phase assembled this immutable
            # top-k=1 verify object while its CUDA graph was executing.  All
            # tensors are persistent runner-owned views, so non-overlap
            # scheduling can avoid tree bookkeeping and Python dataclass
            # construction. Overlap scheduling resolves proposal rows through
            # FutureMap, so it must rebuild verify inputs from those rows.
            return prebuilt_verify
        is_welmv4_mtp = self._is_welmv4_mtp_draft_model()
        use_welmv4_mtp_draft_proposal = (
            is_welmv4_mtp and self._has_welmv4_mtp_draft_proposal(draft_input)
        )
        if (
            is_welmv4_mtp
            and not model_worker_batch.forward_mode.is_idle()
            and self._has_welmv4_mtp_deferred_prefill_rows(draft_input)
        ):
            self._materialize_welmv4_mtp_deferred_prefill_draft(
                model_worker_batch,
                draft_input,
            )
            use_welmv4_mtp_draft_proposal = self._has_welmv4_mtp_draft_proposal(
                draft_input
            )
        if (
            is_welmv4_mtp
            and not model_worker_batch.forward_mode.is_idle()
            and not use_welmv4_mtp_draft_proposal
        ):
            raise RuntimeError(
                "WeLM MTP requires a draft proposal from merged_extend_draft."
            )
        if is_welmv4_mtp and model_worker_batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Run draft
        if use_welmv4_mtp_draft_proposal:
            if (
                self._should_use_welmv4_mtp_oe_hash_kernel()
                and getattr(draft_input, "welm_mtp_oe_history_state", None) is None
            ):
                # PD decode bootstrap reaches the proposal branch without an OE
                # history state (no draft prefill ran here), but the fused
                # target-verify hash kernel has no CPU fallback. Rebuild it
                # from the context's CPU prefixes; when the context carries
                # none, derive them from request token history (skip the tail:
                # the transferred bonus token is already in output_ids and is
                # pushed separately via first_token_ids).
                oe_context = getattr(model_worker_batch, "oe_context", None)
                if oe_context is not None:
                    prefix_rows = getattr(oe_context, "hash_prefixes", None)
                    if prefix_rows is None:
                        prefix_rows = self._welmv4_mtp_prefix_rows_from_reqs(
                            model_worker_batch.reqs,
                            self._welmv4_mtp_oe_prefix_width(),
                            skip_latest_output=True,
                        )
                    history_state = self._init_welmv4_mtp_oe_history_from_context(
                        oe_context,
                        device=draft_input.bonus_tokens.device,
                        first_token_ids=draft_input.bonus_tokens,
                        prefix_rows=prefix_rows,
                    )
                    if history_state is not None:
                        draft_input.welm_mtp_oe_history_state = history_state
            parent_list, top_scores_index, draft_tokens = (
                self._build_welmv4_mtp_draft_proposal_results(draft_input)
            )
        else:
            forward_batch, can_cuda_graph = draft_input.prepare_for_v2_draft(
                self.req_to_token_pool,
                model_worker_batch,
                self.cuda_graph_runner,
                self.draft_runner,
                self.topk,
                self.speculative_num_steps,
            )
            cached_mtp_oe_history = getattr(
                draft_input, "welm_mtp_oe_history_state", None
            )
            needs_mtp_oe_history_reinit = (
                cached_mtp_oe_history is None
                or cached_mtp_oe_history.shape[0] != forward_batch.batch_size
            )
            if (
                self._should_use_welmv4_mtp_oe_hash_kernel()
                and not model_worker_batch.forward_mode.is_idle()
                and needs_mtp_oe_history_reinit
            ):
                draft_input.welm_mtp_oe_prefix_rows = (
                    self._welmv4_mtp_prefix_rows_from_reqs(
                        model_worker_batch.reqs,
                        self._welmv4_mtp_oe_prefix_width(),
                        skip_latest_output=not hasattr(draft_input, "future_indices"),
                    )
                )
                if cached_mtp_oe_history is not None:
                    delattr(draft_input, "welm_mtp_oe_history_state")
            self._prepare_welmv4_mtp_draft_decode_entry_history(
                forward_batch,
                draft_input,
            )

        if not use_welmv4_mtp_draft_proposal:
            if can_cuda_graph:
                parent_list, top_scores_index, draft_tokens = (
                    self.cuda_graph_runner.replay(forward_batch)
                )
                if _WELM_MTP_DUMP_ENABLED and self._is_welmv4_mtp_draft_model():
                    _flush_welmv4_mtp_graph_dump(
                        "draft",
                        first_dim_limit=getattr(
                            self.cuda_graph_runner,
                            "raw_bs",
                            forward_batch.batch_size,
                        ),
                    )
            elif _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context("draft"):
                    if (
                        not forward_batch.forward_mode.is_idle()
                        and self.speculative_num_steps > 1
                        and not self._is_welmv4_mtp_draft_model()
                    ):
                        # Skip attention backend init for 1-step draft,
                        # `draft_forward` only does sample in this case.
                        self.draft_attn_backend.init_forward_metadata(forward_batch)
                    parent_list, top_scores_index, draft_tokens = self.draft_forward(
                        forward_batch
                    )
            else:
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                    and not self._is_welmv4_mtp_draft_model()
                ):
                    # Skip attention backend init for 1-step draft,
                    # `draft_forward` only does sample in this case.
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                parent_list, top_scores_index, draft_tokens = self.draft_forward(
                    forward_batch
                )

        if model_worker_batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )

        linear_verify = None
        if self.cuda_graph_runner_for_draft_proposal is not None:
            linear_verify = (
                self.cuda_graph_runner_for_draft_proposal.build_linear_verify_inputs(
                    draft_input, model_worker_batch.seq_lens
                )
            )
        linear_hash_seq_lens = None
        if linear_verify is not None:
            (
                tree_mask,
                position,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                draft_tokens,
                linear_hash_seq_lens,
            ) = linear_verify
        else:
            (
                tree_mask,
                position,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                draft_tokens,
            ) = build_tree_kernel_efficient(
                draft_input.bonus_tokens,
                parent_list,
                top_scores_index,
                draft_tokens,
                model_worker_batch.seq_lens,
                model_worker_batch.seq_lens_sum,
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                self.tree_mask_mode,
                tree_mask_buf,
                position_buf,
            )

        verify_input = EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
            draft_probs=draft_input.draft_probs,
            draft_topk_indices=draft_input.welm_mtp_draft_topk_indices,
            draft_topk_values=draft_input.welm_mtp_draft_topk_values,
        )
        if self._should_use_welmv4_mtp_oe_hash_kernel() and hasattr(
            draft_input, "welm_mtp_oe_history_state"
        ):
            verify_input.welm_mtp_oe_history_state = (
                draft_input.welm_mtp_oe_history_state
            )
        if linear_hash_seq_lens is not None:
            verify_input.welm_mtp_oe_hash_seq_lens = linear_hash_seq_lens
        return verify_input

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        if (
            self._is_welmv4_mtp_draft_model()
            and not forward_batch.forward_mode.is_idle()
        ):
            raise RuntimeError(
                "WeLM MTP draft_forward is not used for active batches; "
                "merged_extend_draft must provide the draft proposal."
            )

        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            step_cache_loc = out_cache_loc[i]
            step_positions = forward_batch.positions[: input_ids.numel()].clone()
            if step_cache_loc.numel() != input_ids.numel():
                step_cache_loc = step_cache_loc[: input_ids.numel()]
            forward_batch.out_cache_loc = step_cache_loc
            forward_batch.positions = step_positions.add(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states

            # Run forward
            logits_output = self.draft_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        # Organize the results
        score_list = torch.cat(score_list, dim=1).flatten(
            1
        )  # b, n, topk; n= 1 + (num_steps-1) * self.topk
        ss_token_list = torch.cat(
            token_list, dim=1
        )  # b, (self.topk + (num_steps-1) * self.topk)
        top_scores = torch.topk(
            score_list, self.speculative_num_draft_tokens - 1, dim=-1
        )
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values
        maybe_detect_oob(
            top_scores_index,
            0,
            ss_token_list.shape[1],
            "draft_forward: top_scores_index OOB for gather on ss_token_list",
        )
        draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            batch_size = parents_list[0].shape[0]
            parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

        return parent_list, top_scores_index, draft_tokens

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ModelWorkerBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
        model_specific_states: Optional[dict] = None,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        global_prefill_counts = self._get_welmv4_mtp_global_prefill_num_tokens(batch)
        global_has_welmv4_mtp_prefill_work = global_prefill_counts is None or any(
            count > 0 for count in global_prefill_counts
        )
        needs_welmv4_mtp_empty_prefill_collective = (
            global_prefill_counts is not None and global_has_welmv4_mtp_prefill_work
        )
        is_welmv4_mtp_draft_model = self._is_welmv4_mtp_draft_model()
        use_welmv4_mtp_idle_prefill = (
            batch.forward_mode.is_idle()
            and batch.is_extend_in_batch
            and needs_welmv4_mtp_empty_prefill_collective
        )
        use_welmv4_mtp_prefill_proposal = (
            is_welmv4_mtp_draft_model
            and self.speculative_num_steps > 1
            and (not batch.forward_mode.is_idle() or use_welmv4_mtp_idle_prefill)
        )
        deferred_return_draft_input = None
        use_welmv4_mtp_empty_prefill_collective = False
        _welm_mtp_trace(
            "draft_extend_for_prefill "
            f"mode={batch.forward_mode} is_extend_in_batch={batch.is_extend_in_batch} "
            f"use_idle_prefill={use_welmv4_mtp_idle_prefill} "
            f"use_prefill_proposal={use_welmv4_mtp_prefill_proposal} "
            f"needs_empty_collective={needs_welmv4_mtp_empty_prefill_collective} "
            f"global_mtp_prefill_tokens={getattr(batch, 'welm_mtp_global_prefill_num_tokens', None)} "
            f"global_tokens={getattr(batch, 'global_num_tokens', None)} "
            f"target_hidden_shape={tuple(target_hidden_states.shape)} "
            f"next_token_shape={tuple(next_token_ids.shape)}"
        )
        if (
            is_welmv4_mtp_draft_model
            and batch.forward_mode.is_idle()
            and batch.is_extend_in_batch
            and not global_has_welmv4_mtp_prefill_work
        ):
            empty_i32 = torch.empty((0,), dtype=torch.int32, device=self.device)
            return EagleDraftInput(
                hidden_states=target_hidden_states[:0],
                bonus_tokens=next_token_ids[:0],
                new_seq_lens=empty_i32,
                num_tokens_per_req=1,
                num_tokens_for_logprob_per_req=1,
                model_specific_states=model_specific_states,
            )

        if use_welmv4_mtp_prefill_proposal:
            batch.capture_hidden_mode = CaptureHiddenMode.LAST
            deferred_prefill_mask = self._get_welmv4_mtp_deferred_prefill_mask(
                batch,
                device=next_token_ids.device,
            )
            if (
                deferred_prefill_mask is not None
                and deferred_prefill_mask.numel() > 0
                and bool(deferred_prefill_mask.all().item())
            ):
                next_draft_input = EagleDraftInput(
                    hidden_states=target_hidden_states,
                    bonus_tokens=next_token_ids,
                    new_seq_lens=batch.seq_lens,
                    num_tokens_per_req=1,
                    num_tokens_for_logprob_per_req=1,
                    model_specific_states=model_specific_states,
                )
                self._prepare_welmv4_mtp_deferred_prefill_draft_input(
                    next_draft_input,
                    batch,
                    target_hidden_states,
                    next_token_ids,
                )
                deferred_return_draft_input = next_draft_input
                if not needs_welmv4_mtp_empty_prefill_collective:
                    batch.spec_info = next_draft_input
                    return next_draft_input
                use_welmv4_mtp_empty_prefill_collective = True

        if use_welmv4_mtp_idle_prefill or use_welmv4_mtp_empty_prefill_collective:
            target_hidden_states = target_hidden_states[:0]
            next_token_ids = next_token_ids[:0]

        # Construct input_ids. The WeLM MTP merged path fills the draft KV cache
        # from the original extend tokens and injects the target next token as
        # the merged query, so it must not use the shifted input layout.
        if not batch.forward_mode.is_idle() and not use_welmv4_mtp_prefill_proposal:
            pt = 0
            for i, extend_len in enumerate(batch.extend_seq_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], next_token_ids[i].reshape(1))
                )
                pt += extend_len

        # Construct spec_info
        next_draft_input = EagleDraftInput(
            hidden_states=target_hidden_states,
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
            model_specific_states=model_specific_states,
        )

        batch.spec_info = next_draft_input
        if use_welmv4_mtp_prefill_proposal:
            deferred_prefill_mask = self._get_welmv4_mtp_deferred_prefill_mask(
                batch,
                device=next_token_ids.device,
            )
            if deferred_prefill_mask is not None:
                next_draft_input.welm_mtp_deferred_prefill_draft_mask = (
                    deferred_prefill_mask
                )
                next_draft_input.welm_mtp_deferred_prefill_draft = bool(
                    deferred_prefill_mask.any().item()
                )

        if use_welmv4_mtp_idle_prefill or use_welmv4_mtp_empty_prefill_collective:
            self._prepare_welmv4_mtp_empty_prefill_collective_batch(batch)

        # Run forward
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        forward_batch.return_logprob = False
        if (
            self._should_use_welmv4_mtp_oe_hash_kernel()
            and batch.oe_context is not None
        ):
            self._prepare_welmv4_mtp_segment_hash_inputs_from_prefixes(forward_batch)
        if use_welmv4_mtp_prefill_proposal:
            first_query_hashed_inputs = None
            first_query_history_state = None
            if batch.forward_mode.is_idle():
                forward_batch.can_run_dp_cuda_graph = False
                forward_batch.custom_last_index = torch.empty(
                    (0,), dtype=torch.long, device=self.device
                )
                forward_batch.custom_last_cache_loc = torch.empty(
                    (0,), dtype=torch.long, device=self.device
                )
            if (
                self._should_use_welmv4_mtp_oe_hash_kernel()
                and not batch.forward_mode.is_idle()
            ):
                entry_history_state = self._init_welmv4_mtp_oe_history_from_extend(
                    forward_batch,
                    first_token_ids=None,
                )
                if entry_history_state is None:
                    raise RuntimeError(
                        "WeLMV4 MTP merged prefill draft is missing entry OE "
                        "history for the first query."
                    )
                (
                    first_query_hashed_inputs,
                    first_query_history_state,
                    first_query_verify_history_state,
                ) = self._compute_welmv4_mtp_first_query_hash_from_entry_history(
                    forward_batch,
                    next_token_ids,
                    entry_history_state,
                )
                next_draft_input.welm_mtp_oe_history_state = (
                    first_query_verify_history_state
                )
            if mm_input_embeds is not None and not batch.forward_mode.is_idle():
                forward_batch.mm_input_embeds = mm_input_embeds
            prefill_branch_cache_locs = None
            if self.topk > 1:
                prefill_branch_cache_locs = (
                    self._reserve_welmv4_mtp_prefill_branch_cache_locs(batch)
                )
                next_draft_input.welm_mtp_prefill_branch_cache_locs = (
                    prefill_branch_cache_locs
                )
            if _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context("merged_extend_draft_prefill"):
                    self._run_welmv4_mtp_merged_extend_draft(
                        forward_batch,
                        next_token_ids,
                        skip_attn_backend_init=False,
                        first_query_hashed_inputs=first_query_hashed_inputs,
                        first_query_history_state=first_query_history_state,
                        draft_path="prefill",
                    )
            else:
                self._run_welmv4_mtp_merged_extend_draft(
                    forward_batch,
                    next_token_ids,
                    skip_attn_backend_init=False,
                    first_query_hashed_inputs=first_query_hashed_inputs,
                    first_query_history_state=first_query_history_state,
                    draft_path="prefill",
                )
            if deferred_return_draft_input is not None:
                return deferred_return_draft_input
            return next_draft_input
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds
        if _WELM_MTP_DUMP_ENABLED:
            with _welmv4_mtp_dump_context("draft_extend_for_prefill"):
                logits_output = self.draft_runner.forward(forward_batch).logits_output
        else:
            logits_output = self.draft_runner.forward(forward_batch).logits_output
        if _WELM_VERIFY_AFTER_DUMP_ENABLED:
            _dump_verify_after_event(
                "draft_extend_for_prefill",
                {
                    "seq_lens": batch.seq_lens,
                    "extend_seq_lens": getattr(batch, "extend_seq_lens", None),
                    "input_ids": batch.input_ids,
                    "next_token_ids": next_token_ids,
                },
            )
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")

        # Update spec_info for the next draft step
        self._capture_for_decode(logits_output, next_draft_input, forward_batch)
        return next_draft_input

    def _prepare_welmv4_mtp_idle_decode_proposal_batch(
        self,
        batch: ModelWorkerBatch,
        draft_input: EagleDraftInput,
        next_draft_input: EagleDraftInput,
    ) -> torch.Tensor:
        device = self.device
        empty_i32 = torch.empty((0,), dtype=torch.int32, device=device)
        empty_i64 = torch.empty((0,), dtype=torch.int64, device=device)
        empty_hidden_states = torch.empty(
            (0, self.draft_runner.model_config.spec_hidden_size),
            dtype=self.draft_runner.dtype,
            device=device,
        )

        draft_input.hidden_states = (
            empty_hidden_states
            if next_draft_input.hidden_states is None
            else next_draft_input.hidden_states
        )
        draft_input.bonus_tokens = (
            empty_i32
            if next_draft_input.bonus_tokens is None
            else next_draft_input.bonus_tokens
        )
        draft_input.new_seq_lens = (
            empty_i32
            if next_draft_input.new_seq_lens is None
            else next_draft_input.new_seq_lens
        )
        draft_input.verify_done = next_draft_input.verify_done
        draft_input.num_correct_drafts = (
            empty_i32
            if next_draft_input.num_correct_drafts is None
            else next_draft_input.num_correct_drafts
        )
        draft_input.num_accept_tokens = (
            empty_i32
            if next_draft_input.num_accept_tokens is None
            else next_draft_input.num_accept_tokens
        )
        draft_input.num_correct_drafts_cpu = list(
            next_draft_input.num_correct_drafts_cpu or []
        )
        draft_input.num_accept_tokens_cpu = list(
            next_draft_input.num_accept_tokens_cpu or []
        )
        draft_input.num_tokens_for_logprob_per_req = 1
        # topk=1 verify never packs mirrored_kv_indices, so idle cycles would
        # hit the proposal-graph mirror-state copy with indices=None and no
        # verify states (idle verify runs no forward) and trip its raise. An
        # empty index tensor routes it into the zero-fill padding branch.
        draft_input.mirrored_kv_indices = (
            empty_i64
            if next_draft_input.mirrored_kv_indices is None
            else next_draft_input.mirrored_kv_indices
        )
        draft_input.model_specific_states = next_draft_input.model_specific_states

        batch.spec_info = draft_input
        batch.input_ids = empty_i64
        batch.req_pool_indices = (
            batch.req_pool_indices[:0]
            if batch.req_pool_indices is not None
            else empty_i64
        )
        batch.seq_lens = batch.seq_lens[:0] if batch.seq_lens is not None else empty_i32
        batch.seq_lens_cpu = (
            batch.seq_lens_cpu[:0]
            if batch.seq_lens_cpu is not None
            else torch.empty((0,), dtype=torch.int32)
        )
        batch.out_cache_loc = (
            batch.out_cache_loc[:0] if batch.out_cache_loc is not None else empty_i64
        )
        batch.seq_lens_sum = 0
        batch.extend_seq_lens = []
        batch.extend_prefix_lens = []
        batch.extend_logprob_start_lens = []
        batch.extend_num_tokens = 0
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.is_extend_in_batch = True
        batch.return_logprob = False
        batch.return_hidden_states = False

        return draft_input.bonus_tokens.to(dtype=torch.int64)

    @staticmethod
    def _packed_accept_path_positions(
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        if accept_index.numel() == 0:
            return accept_index.to(dtype=torch.long).flatten()

        bs, max_accept_len = accept_index.shape
        device = accept_index.device
        row_offsets = (
            torch.arange(bs, device=device, dtype=torch.long) * draft_token_num
        )
        path_order = torch.arange(max_accept_len, device=device, dtype=torch.long)
        valid_accept = path_order.unsqueeze(0) < accept_lens.to(torch.long).unsqueeze(1)
        return (row_offsets.unsqueeze(1) + path_order.unsqueeze(0))[valid_accept]

    def _draft_extend_for_decode(
        self, batch: ModelWorkerBatch, batch_result: GenerationBatchResult
    ):
        is_idle_decode = batch.forward_mode.is_idle()
        draft_continuation_state = batch_result.draft_continuation_state
        assert draft_continuation_state is not None
        next_draft_input = draft_continuation_state.draft_input
        welmv4_mtp_oe_entry_history = getattr(
            getattr(batch, "spec_info", None), "welm_mtp_oe_history_state", None
        )
        welmv4_mtp_oe_next_history = None
        # Batch 2: Draft extend
        draft_hidden_states = next_draft_input.hidden_states
        if draft_hidden_states is None:
            draft_hidden_states = batch_result.logits_output.hidden_states
            if self.topk > 1 and not batch.forward_mode.is_idle():
                raise RuntimeError(
                    "Spec v2 topk>1 draft extend requires packed hidden states "
                    "from the verify step."
                )
        draft_input = EagleDraftInput(
            hidden_states=draft_hidden_states,
            num_tokens_per_req=self.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
            model_specific_states=batch_result.logits_output.model_specific_states,
            mirrored_kv_indices=next_draft_input.mirrored_kv_indices,
        )
        if _WELM_VERIFY_AFTER_DUMP_ENABLED:
            select_index = (
                torch.arange(len(batch.seq_lens), device=self.device)
                * self.speculative_num_draft_tokens
                + batch_result.accept_lens
                - 1
            )
            _dump_verify_after_event(
                "draft_extend_for_decode",
                {
                    "seq_lens": batch.seq_lens,
                    "accept_lens": batch_result.accept_lens,
                    "select_index": select_index,
                    "predict": batch_result.next_token_ids,
                },
            )

        if (
            self._is_welmv4_mtp_draft_model()
            and self.speculative_num_steps > 1
            and (
                not is_idle_decode
                or self.cuda_graph_runner_for_draft_proposal is not None
                or self.topk > 1
            )
        ):
            proposal_runner = self.cuda_graph_runner_for_draft_proposal
            if proposal_runner is not None and proposal_runner.can_replay_from_verify(
                batch, batch_result
            ):
                # The target verify and this call share the compute stream, so
                # its outputs are already ordered.  Write them directly into
                # persistent proposal-graph buffers instead of compacting to a
                # ragged temporary batch and expanding back to width four.
                proposal_runner.replay_from_verify(
                    batch,
                    batch_result,
                    next_draft_input,
                    entry_history_state=welmv4_mtp_oe_entry_history,
                )
                batch.spec_info = next_draft_input
                return
            select_index = (
                torch.arange(len(batch.seq_lens), device=self.device)
                * self.speculative_num_draft_tokens
                + batch_result.accept_lens
                - 1
            )
            if self.plan_stream and next_draft_input.verify_done is not None:
                self.plan_stream.wait_event(next_draft_input.verify_done)
            with self.plan_stream_ctx:
                if is_idle_decode:
                    first_input_ids = (
                        self._prepare_welmv4_mtp_idle_decode_proposal_batch(
                            batch,
                            draft_input,
                            next_draft_input,
                        )
                    )
                else:
                    accept_lens_cpu = batch_result.accept_lens.detach().cpu().tolist()
                    accept_index = getattr(batch_result, "spec_accept_index", None)
                    if accept_index is None:
                        raise RuntimeError(
                            "WeLMV4 MTP merged_extend_draft requires the verify "
                            "accept_index."
                        )
                    accepted_indices = accept_index[accept_index != -1].to(torch.long)
                    if envs.SGLANG_SPEC_OOB_DETECTION.get():
                        torch._assert_async(
                            batch_result.accept_lens.sum() == accepted_indices.numel(),
                            "WeLMV4 MTP accept_index/accept_lens mismatch.",
                        )
                    if self.topk > 1:
                        packed_accepted_indices = self._packed_accept_path_positions(
                            accept_index,
                            batch_result.accept_lens,
                            self.speculative_num_draft_tokens,
                        )
                    else:
                        packed_accepted_indices = accepted_indices
                    max_rows = min(
                        int(batch_result.next_token_ids.numel()),
                        int(batch_result.logits_output.hidden_states.shape[0]),
                        int(batch.out_cache_loc.numel()),
                    )
                    maybe_detect_oob(
                        accepted_indices,
                        0,
                        max_rows,
                        f"WeLMV4 MTP accepted index OOB vs rows={max_rows}",
                    )
                    maybe_detect_oob(
                        packed_accepted_indices,
                        0,
                        max_rows,
                        f"WeLMV4 MTP packed accepted index OOB vs rows={max_rows}",
                    )

                    if self.topk > 1 and next_draft_input.hidden_states is not None:
                        draft_input.hidden_states = next_draft_input.hidden_states[
                            packed_accepted_indices
                        ]
                    else:
                        draft_input.hidden_states = (
                            batch_result.logits_output.hidden_states[accepted_indices]
                        )
                    draft_input.bonus_tokens = next_draft_input.bonus_tokens
                    draft_input.new_seq_lens = next_draft_input.new_seq_lens
                    draft_input.verify_done = next_draft_input.verify_done
                    draft_input.num_tokens_for_logprob_per_req = 1
                    draft_input.num_correct_drafts = batch_result.accept_lens - 1
                    draft_input.num_accept_tokens = batch_result.accept_lens
                    draft_input.num_correct_drafts_cpu = [
                        int(x) - 1 for x in accept_lens_cpu
                    ]
                    draft_input.num_accept_tokens_cpu = [
                        int(x) for x in accept_lens_cpu
                    ]
                    draft_input.mirrored_kv_indices = accepted_indices

                    seq_lens_before = batch.seq_lens
                    seq_lens_cpu_before = batch.seq_lens_cpu
                    if seq_lens_cpu_before is not None:
                        extend_prefix_lens = seq_lens_cpu_before.tolist()
                        accept_lens_cpu_tensor = batch_result.accept_lens.detach().to(
                            device=seq_lens_cpu_before.device,
                            dtype=seq_lens_cpu_before.dtype,
                        )
                        seq_lens_cpu_after = (
                            seq_lens_cpu_before + accept_lens_cpu_tensor
                        )
                    else:
                        extend_prefix_lens = seq_lens_before.detach().cpu().tolist()
                        seq_lens_cpu_after = (
                            next_draft_input.new_seq_lens.detach().cpu()
                        )

                    batch.spec_info = draft_input
                    batch.input_ids = batch_result.next_token_ids[
                        packed_accepted_indices
                    ].to(torch.int64)
                    batch.out_cache_loc = batch.out_cache_loc[packed_accepted_indices]
                    batch.seq_lens = next_draft_input.new_seq_lens
                    batch.seq_lens_cpu = seq_lens_cpu_after
                    batch.seq_lens_sum = int(seq_lens_cpu_after.sum().item())
                    batch.extend_seq_lens = [int(x) for x in accept_lens_cpu]
                    batch.extend_prefix_lens = extend_prefix_lens
                    batch.extend_num_tokens = int(accepted_indices.numel())
                    batch.capture_hidden_mode = CaptureHiddenMode.LAST
                    batch.forward_mode = ForwardMode.DRAFT_EXTEND
                    batch.is_extend_in_batch = True
                    first_input_ids = next_draft_input.bonus_tokens

                forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
                forward_batch.return_logprob = False
                if is_idle_decode:
                    forward_batch.custom_last_index = torch.empty(
                        (0,), dtype=torch.long, device=self.device
                    )
                    forward_batch.custom_last_cache_loc = torch.empty(
                        (0,), dtype=torch.long, device=self.device
                    )
                else:
                    real_last_index = (
                        torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                    )
                    forward_batch.custom_last_index = real_last_index
                    if forward_batch.out_cache_loc is not None:
                        forward_batch.custom_last_cache_loc = (
                            forward_batch.out_cache_loc[real_last_index]
                        )
                first_query_history_state = None
                first_query_hashed_inputs = None
                if self._should_use_welmv4_mtp_oe_hash_kernel() and not is_idle_decode:
                    first_query_history_state = (
                        self._prepare_welmv4_mtp_draft_extend_hash_inputs(
                            forward_batch,
                            batch_result.accept_lens,
                            getattr(
                                batch_result,
                                "welm_mtp_accepted_draft_token_ids",
                                None,
                            ),
                            welmv4_mtp_oe_entry_history,
                        )
                    )
                    if first_query_history_state is not None:
                        next_draft_input.welm_mtp_oe_history_state = (
                            first_query_history_state
                        )
                    first_query_hashed_inputs = (
                        self._select_welmv4_mtp_last_extend_hash_inputs(forward_batch)
                    )
                attn_backend = (
                    self.draft_extend_attn_backend or self.draft_runner.attn_backend
                )
                forward_batch.attn_backend = attn_backend
                can_cuda_graph = (
                    self.cuda_graph_runner_for_draft_proposal is not None
                    and self.cuda_graph_runner_for_draft_proposal.can_run(forward_batch)
                )
                use_variable_decode_extend = (
                    self.topk == 1
                    and self.welmv4_mtp_sample_draft
                    and self._get_welmv4_mtp_draft_sampling_topk() == 0
                    and not can_cuda_graph
                )
                if use_variable_decode_extend and not is_idle_decode:
                    batch.forward_mode = ForwardMode.EXTEND
                    draft_input.num_tokens_per_req = 1
                    draft_input.num_tokens_for_logprob_per_req = 1
                    forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
                    forward_batch.return_logprob = False
                    forward_batch.welm_mtp_variable_decode_extend = True
                    real_last_index = (
                        torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                    )
                    forward_batch.custom_last_index = real_last_index
                    if forward_batch.out_cache_loc is not None:
                        forward_batch.custom_last_cache_loc = (
                            forward_batch.out_cache_loc[real_last_index]
                        )
                    attn_backend = (
                        self.draft_extend_attn_backend or self.draft_runner.attn_backend
                    )
                    forward_batch.attn_backend = attn_backend
                elif use_variable_decode_extend:
                    forward_batch.welm_mtp_variable_decode_extend = True
                if is_idle_decode and not can_cuda_graph:
                    forward_batch.forward_mode = ForwardMode.IDLE
                if not can_cuda_graph and not is_idle_decode:
                    attn_backend.init_forward_metadata(forward_batch)
            if self.plan_stream:
                torch.get_device_module(self.device).current_stream().wait_stream(
                    self.plan_stream
                )

            if can_cuda_graph:
                self.cuda_graph_runner_for_draft_proposal.replay(
                    forward_batch,
                    first_input_ids,
                    first_query_hashed_inputs=first_query_hashed_inputs,
                    first_query_history_state=first_query_history_state,
                )
            elif is_idle_decode:
                if _WELM_MTP_DUMP_ENABLED:
                    with _welmv4_mtp_dump_context("merged_extend_draft_idle_decode"):
                        self._run_welmv4_mtp_merged_extend_draft(
                            forward_batch,
                            first_input_ids,
                            skip_attn_backend_init=True,
                            first_query_hashed_inputs=first_query_hashed_inputs,
                            first_query_history_state=first_query_history_state,
                            draft_path="idle_decode",
                        )
                else:
                    self._run_welmv4_mtp_merged_extend_draft(
                        forward_batch,
                        first_input_ids,
                        skip_attn_backend_init=True,
                        first_query_hashed_inputs=first_query_hashed_inputs,
                        first_query_history_state=first_query_history_state,
                        draft_path="idle_decode",
                    )
            elif _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context("merged_extend_draft_decode"):
                    self._run_welmv4_mtp_merged_extend_draft(
                        forward_batch,
                        first_input_ids,
                        skip_attn_backend_init=True,
                        first_query_hashed_inputs=first_query_hashed_inputs,
                        first_query_history_state=first_query_history_state,
                        draft_path="decode",
                    )
            else:
                self._run_welmv4_mtp_merged_extend_draft(
                    forward_batch,
                    first_input_ids,
                    skip_attn_backend_init=True,
                    first_query_hashed_inputs=first_query_hashed_inputs,
                    first_query_history_state=first_query_history_state,
                    draft_path="decode",
                )
            self._copy_welmv4_mtp_draft_state(
                forward_batch.spec_info,
                next_draft_input,
            )
            return

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch,
                batch_result.next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
            )
            oe_context = getattr(forward_batch, "oe_context", None)
            if (
                self._is_welmv4_mtp_draft_model()
                and self._should_use_welmv4_mtp_oe_hash_kernel()
            ):
                welmv4_mtp_oe_next_history = (
                    self._prepare_welmv4_mtp_draft_extend_hash_inputs(
                        forward_batch,
                        batch_result.accept_lens,
                        getattr(
                            batch_result, "welm_mtp_accepted_draft_token_ids", None
                        ),
                        welmv4_mtp_oe_entry_history,
                    )
                )
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
        if welmv4_mtp_oe_next_history is not None:
            next_draft_input.welm_mtp_oe_history_state = welmv4_mtp_oe_next_history

        if forward_batch.spec_info.num_correct_drafts is None:
            # `batch_result.accept_lens` already includes the bonus token, so use it
            # directly for `num_accept_tokens` and subtract 1 for `num_correct_drafts`.
            forward_batch.spec_info.num_correct_drafts = batch_result.accept_lens - 1
            forward_batch.spec_info.num_accept_tokens = batch_result.accept_lens

        # Run draft extend batch in the main compute stream
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            draft_logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
            if _WELM_MTP_DUMP_ENABLED and self._is_welmv4_mtp_draft_model():
                _flush_welmv4_mtp_graph_dump(
                    "draft_extend",
                    first_dim_limit=int(forward_batch.input_ids.shape[0]),
                )
        else:
            if _WELM_MTP_DUMP_ENABLED:
                with _welmv4_mtp_dump_context("draft_extend"):
                    draft_logits_output = self.draft_runner.forward(
                        forward_batch, skip_attn_backend_init=True
                    ).logits_output
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch, skip_attn_backend_init=True
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        draft_logits_output.hidden_states = draft_logits_output.hidden_states[
            select_index
        ]
        welmv4_mtp_base_positions = None
        if self._is_welmv4_mtp_draft_model() and self.topk > 1:
            welmv4_mtp_base_positions = forward_batch.positions[select_index].clone()

        # Construct the return values
        self._capture_for_decode(
            draft_logits_output,
            next_draft_input,
            forward_batch,
            welmv4_mtp_base_positions,
        )


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self._welmv4_mtp_target_verify_attn_backend = None

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def _get_welmv4_mtp_target_verify_attn_backend(self):
        if self._welmv4_mtp_target_verify_attn_backend is None:
            target_model_runner = self.target_worker.model_runner
            backend = target_model_runner._get_attention_backend_from_str("triton")
            init_forward_metadata = backend.init_forward_metadata

            def init_forward_metadata_with_swa_fallback(forward_batch):
                init_forward_metadata(forward_batch)
                if not forward_batch.forward_mode.is_target_verify():
                    return
                metadata = backend.forward_metadata
                if metadata.window_kv_indptr is None:
                    metadata.window_kv_indptr = metadata.kv_indptr
                    metadata.window_kv_indices = metadata.kv_indices
                    metadata.window_kv_offsets = torch.zeros(
                        metadata.kv_indptr.numel() - 1,
                        dtype=torch.int32,
                        device=metadata.kv_indptr.device,
                    )

            backend.init_forward_metadata = init_forward_metadata_with_swa_fallback
            self._welmv4_mtp_target_verify_attn_backend = backend
        return self._welmv4_mtp_target_verify_attn_backend

    def _should_use_welmv4_mtp_target_verify_attn_backend(self) -> bool:
        return False

    def _should_disable_welmv4_mtp_target_verify_graph(self) -> bool:
        return False

    def clear_cache_pool(self):
        # The allocator and KV cache pool are shared with the target worker.
        # Flush drains WeLM MTP asynchronous proposal work before rebuilding the
        # shared allocator metadata, otherwise late writes can corrupt the next
        # request's freshly reset free list.
        if self.draft_worker._is_welmv4_mtp_draft_model() and torch.cuda.is_available():
            device_module = torch.get_device_module(self.device)
            if self.plan_stream is not None:
                device_module.current_stream().wait_stream(self.plan_stream)
            device_module.synchronize()

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill
            is_welmv4_mtp = self.draft_worker._is_welmv4_mtp_draft_model()
            defer_welmv4_mtp_pd_prefill = (
                self.server_args.disaggregation_mode == "prefill"
                and self.server_args.enable_welm_kv_mirror_opt
                and is_welmv4_mtp
            )
            model_worker_batch.capture_hidden_mode = (
                CaptureHiddenMode.LAST
                if defer_welmv4_mtp_pd_prefill
                else CaptureHiddenMode.FULL
            )
            if (
                is_welmv4_mtp
                and self.topk > 1
                and model_worker_batch.out_cache_loc is not None
            ):
                model_worker_batch.out_cache_loc = (
                    model_worker_batch.out_cache_loc.clone()
                )
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            if defer_welmv4_mtp_pd_prefill:
                self.draft_worker._copy_welmv4_mtp_pd_prefill_mirror_states(
                    model_worker_batch,
                    batch_output.logits_output.model_specific_states,
                )
                next_draft_input = EagleDraftInput(
                    hidden_states=batch_output.logits_output.hidden_states,
                    bonus_tokens=batch_output.next_token_ids,
                    new_seq_lens=model_worker_batch.seq_lens,
                    num_tokens_per_req=1,
                    num_tokens_for_logprob_per_req=1,
                )
                self.draft_worker._prepare_welmv4_mtp_deferred_prefill_draft_input(
                    next_draft_input,
                    model_worker_batch,
                    batch_output.logits_output.hidden_states,
                    batch_output.next_token_ids,
                )
                next_draft_input.capture_hidden_mode = CaptureHiddenMode.LAST
                model_worker_batch.spec_info = next_draft_input
                batch_output.draft_continuation_state = DraftContinuationState(
                    next_draft_input
                )
                return batch_output

            # Draft prefill
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                batch_output.draft_continuation_state = (
                    DraftContinuationState.from_draft_input(
                        self.draft_worker._draft_extend_for_prefill(
                            model_worker_batch,
                            batch_output.logits_output.hidden_states,
                            batch_output.next_token_ids,
                            batch_output.logits_output.mm_input_embeds,
                            batch_output.logits_output.model_specific_states,
                        )
                    )
                )
                return batch_output
        else:
            if model_worker_batch.spec_info is None:
                model_worker_batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self.target_worker.model_config.spec_hidden_size,
                    dtype=self.target_worker.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                verify_input: EagleVerifyInput = self.draft_worker.draft(
                    model_worker_batch
                )
            assert verify_input.is_verify_input()
            # Record a CUDA event after draft() GPU work is dispatched.
            # This event will be waited on by plan_stream in verify()
            # to ensure draft CUDA graph kernels finish before plan_stream
            # begins metadata preparation.
            if self.plan_stream:
                self._draft_done_event = torch.get_device_module(self.device).Event()
                self._draft_done_event.record()
            model_worker_batch.spec_info = verify_input
            batch_output = self.verify(model_worker_batch)
            with (
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.draft_worker._draft_extend_for_decode(
                    model_worker_batch, batch_output
                )
            return batch_output

    def _pack_accept_path_indices(
        self,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        if accept_index.numel() == 0:
            return accept_index.to(dtype=torch.long).flatten()

        bs, max_accept_len = accept_index.shape
        if max_accept_len > draft_token_num:
            raise RuntimeError(
                "Spec v2 accept path is wider than the dense draft row: "
                f"{max_accept_len=} {draft_token_num=}"
            )

        device = accept_index.device
        row_offsets = (
            torch.arange(bs, device=device, dtype=torch.long) * draft_token_num
        )
        draft_offsets = torch.arange(draft_token_num, device=device, dtype=torch.long)
        rank = draft_offsets.unsqueeze(0).expand(bs, draft_token_num).clone()
        rank += max_accept_len

        path_order = torch.arange(max_accept_len, device=device, dtype=torch.long)
        valid_accept = path_order.unsqueeze(0) < accept_lens.to(torch.long).unsqueeze(1)
        accepted_offsets = accept_index.to(torch.long) - row_offsets.unsqueeze(1)
        valid_accept = valid_accept & (
            (accepted_offsets >= 0) & (accepted_offsets < draft_token_num)
        )
        row_ids = torch.arange(bs, device=device, dtype=torch.long)
        safe_offsets = torch.clamp(accepted_offsets, min=0, max=draft_token_num - 1)
        for col in range(max_accept_len):
            col_offsets = safe_offsets[:, col]
            current_rank = rank[row_ids, col_offsets]
            rank[row_ids, col_offsets] = torch.where(
                valid_accept[:, col],
                torch.full_like(current_rank, col),
                current_rank,
            )

        packed_offsets = torch.argsort(rank, dim=1)
        return (row_offsets.unsqueeze(1) + packed_offsets).flatten()

    @staticmethod
    def _sanitize_topk_accept_path(
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        draft_token_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if accept_index.numel() == 0:
            return accept_lens, accept_index

        bs, max_accept_len = accept_index.shape
        device = accept_index.device
        row_offsets = (
            torch.arange(bs, device=device, dtype=torch.long) * draft_token_num
        )
        path_order = torch.arange(max_accept_len, device=device, dtype=torch.long)
        clamped_lens = torch.clamp(
            accept_lens.to(torch.long), min=0, max=max_accept_len
        )
        within_accept = path_order.unsqueeze(0) < clamped_lens.unsqueeze(1)
        accepted_offsets = accept_index.to(torch.long) - row_offsets.unsqueeze(1)
        in_row = (accepted_offsets >= 0) & (accepted_offsets < draft_token_num)
        invalid_pos = torch.where(
            within_accept & ~in_row,
            path_order.unsqueeze(0).expand(bs, max_accept_len),
            torch.full(
                (bs, max_accept_len),
                max_accept_len,
                dtype=torch.long,
                device=device,
            ),
        )
        first_invalid = invalid_pos.min(dim=1).values
        safe_lens = torch.minimum(clamped_lens, first_invalid)
        safe_lens = torch.clamp(safe_lens, min=1)
        safe_accept_index = torch.where(
            path_order.unsqueeze(0) < safe_lens.unsqueeze(1),
            accept_index,
            torch.full_like(accept_index, -1),
        )
        root_indices = row_offsets.to(dtype=accept_index.dtype)
        safe_accept_index[:, 0] = torch.where(
            in_row[:, 0], safe_accept_index[:, 0], root_indices
        )
        return safe_lens.to(dtype=accept_lens.dtype), safe_accept_index

    def _compact_topk_accept_path_in_req_pool(
        self,
        batch: ModelWorkerBatch,
        packed_indices: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        bs = len(batch.seq_lens)
        if bs == 0:
            return batch.out_cache_loc

        packed_cache_locs = batch.out_cache_loc[packed_indices]
        logical_offsets = torch.arange(
            draft_token_num, device=batch.seq_lens.device, dtype=torch.long
        )
        logical_positions = batch.seq_lens.to(torch.long).unsqueeze(1) + (
            logical_offsets.unsqueeze(0)
        )
        req_pool_indices = batch.req_pool_indices.to(torch.long).unsqueeze(1)
        req_pool_indices = req_pool_indices.expand(bs, draft_token_num)
        self.req_to_token_pool.req_to_token[req_pool_indices, logical_positions] = (
            packed_cache_locs.view(bs, draft_token_num).to(
                dtype=self.req_to_token_pool.req_to_token.dtype
            )
        )
        batch.out_cache_loc = packed_cache_locs
        return packed_cache_locs

    def verify(self, batch: ModelWorkerBatch):
        # Since batch.seq_lens is allocated in another stream, we need
        # record_stream() to prevent pytorch gc and reuse the gpu memory
        # while forward_stream is still running.
        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        # Parse args
        verify_input: EagleVerifyInput = batch.spec_info
        verify_input.num_tokens_per_req = verify_input.draft_token_num
        bs = len(batch.seq_lens)
        use_welmv4_mtp_verify_backend = (
            self._should_use_welmv4_mtp_target_verify_attn_backend()
        )
        disable_target_graph_for_welmv4_mtp = (
            _WELM_DISABLE_TARGET_VERIFY_GRAPH_FOR_DUMP
            or self._should_disable_welmv4_mtp_target_verify_graph()
        )
        target_model_runner = self.target_worker.model_runner
        target_graph_runner = None
        target_attn_backend = None
        if disable_target_graph_for_welmv4_mtp:
            target_graph_runner = target_model_runner.graph_runner
            target_model_runner.graph_runner = None
        if use_welmv4_mtp_verify_backend:
            target_attn_backend = target_model_runner.attn_backend
            target_model_runner.attn_backend = (
                self._get_welmv4_mtp_target_verify_attn_backend()
            )

        try:
            # Batch 1: Target verify
            # Prepare for target verify in a separate stream
            with self.plan_stream_ctx:
                # Wait for the draft CUDA graph to finish before plan_stream
                # begins its work. Using an event is more targeted than
                # wait_stream(main_stream) — it only waits for draft GPU
                # work, not all queued main_stream operations.
                if self.plan_stream and hasattr(self, "_draft_done_event"):
                    self.plan_stream.wait_event(self._draft_done_event)
                verify_forward_batch, can_run_cuda_graph = (
                    verify_input.prepare_for_v2_verify(
                        self.req_to_token_pool,
                        batch,
                        self.target_worker,
                        prepare_attn_backend=False,
                    )
                )

            # Correct some buffers due to the overlap plan
            if self.plan_stream:
                torch.get_device_module(self.device).current_stream().wait_stream(
                    self.plan_stream
                )

                # Some values such as custom_mask and position depend on the output of draft,
                # so the previous plan step used the wrong values. Here, we need to run the related
                # computation again to update them to the correct values.
                self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                    verify_input,
                    (
                        self.target_worker.model_runner.graph_runner.bs
                        if can_run_cuda_graph
                        else None
                    ),
                )

            if getattr(verify_input, "welm_mtp_oe_hashed_inputs", None) is not None:
                verify_forward_batch.welm_oe_decode_hashed_inputs = (
                    verify_input.welm_mtp_oe_hashed_inputs
                )
            else:
                self.draft_worker._prepare_welmv4_mtp_target_verify_hash_inputs(
                    verify_forward_batch,
                    getattr(verify_input, "welm_mtp_oe_history_state", None),
                )
                hashed_inputs = getattr(
                    verify_forward_batch, "welm_oe_decode_hashed_inputs", None
                )
                if hashed_inputs is not None:
                    verify_input.welm_mtp_oe_hashed_inputs = hashed_inputs

            if can_run_cuda_graph:
                self.target_worker.model_runner.graph_runner.replay_prepare(
                    verify_forward_batch
                )
            elif not batch.forward_mode.is_idle():
                self.target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )

            if _WELM_VERIFY_AFTER_DUMP_ENABLED and not batch.forward_mode.is_idle():
                graph_runner = self.target_worker.model_runner.graph_runner
                graph_buffers = getattr(graph_runner, "buffers", None)
                _dump_verify_after_event(
                    "verify_before_target_forward",
                    {
                        "can_run_cuda_graph": can_run_cuda_graph,
                        "batch_seq_lens": batch.seq_lens,
                        "batch_seq_lens_cpu": batch.seq_lens_cpu,
                        "batch_out_cache_loc": batch.out_cache_loc,
                        "verify_input_draft_token": verify_input.draft_token,
                        "verify_input_positions": verify_input.positions,
                        "verify_input_retrieve_index": verify_input.retrieve_index,
                        "verify_input_retrieve_next_token": verify_input.retrieve_next_token,
                        "verify_input_retrieve_next_sibling": verify_input.retrieve_next_sibling,
                        "verify_input_custom_mask": verify_input.custom_mask,
                        "graph_input_ids": (
                            None if graph_buffers is None else graph_buffers.input_ids
                        ),
                        "graph_positions": (
                            None if graph_buffers is None else graph_buffers.positions
                        ),
                        "graph_out_cache_loc": (
                            None
                            if graph_buffers is None
                            else graph_buffers.out_cache_loc
                        ),
                        "graph_seq_lens": (
                            None if graph_buffers is None else graph_buffers.seq_lens
                        ),
                    },
                )

            # Prepare grammar data on CPU if needed
            if batch.has_grammar:
                retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
                retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
                draft_tokens_cpu = verify_input.draft_token.view(
                    verify_input.retrieve_next_token.shape
                ).cpu()

            # Run target verify batch in the main compute stream (GPU compute)
            forward_batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch=None,
                forward_batch=verify_forward_batch,
                is_verify=True,
                skip_attn_backend_init=True,
            )
        finally:
            if target_attn_backend is not None:
                target_model_runner.attn_backend = target_attn_backend
            if disable_target_graph_for_welmv4_mtp:
                target_model_runner.graph_runner = target_graph_runner
        logits_output = forward_batch_output.logits_output

        # Generate vocab mask for constrained decoding
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        # Sample
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,
            accept_lens,
            accept_index,
        ) = verify_input.sample(batch, logits_output, vocab_mask)
        if self.topk > 1 and not batch.forward_mode.is_idle():
            accept_lens, accept_index = self._sanitize_topk_accept_path(
                accept_index, accept_lens, self.speculative_num_draft_tokens
            )
        new_seq_lens = batch.seq_lens + accept_lens
        welm_mtp_accepted_draft_token_ids = None
        has_welmv4_mtp_oe_context = (
            self.draft_worker._is_welmv4_mtp_draft_model()
            and not batch.forward_mode.is_idle()
            and (
                (
                    getattr(batch, "oe_context", None) is not None
                    and batch.oe_context.input_ids_buffer is not None
                )
                or (
                    self.draft_worker._should_use_welmv4_mtp_oe_hash_kernel()
                    and getattr(verify_input, "welm_mtp_oe_history_state", None)
                    is not None
                )
            )
        )
        if has_welmv4_mtp_oe_context:
            accepted_mask = accept_index >= 0
            safe_accept_index = torch.where(
                accepted_mask, accept_index, torch.zeros_like(accept_index)
            ).to(torch.long)
            accepted_draft_token_ids = (
                verify_input.draft_token.reshape(-1)
                .index_select(0, safe_accept_index.reshape(-1))
                .reshape_as(accept_index)
                .to(accept_index.dtype)
            )
            welm_mtp_accepted_draft_token_ids = torch.where(
                accepted_mask,
                accepted_draft_token_ids,
                torch.full_like(accept_index, -1),
            )
        if _WELM_VERIFY_AFTER_DUMP_ENABLED:
            verify_dump_extra = {}
            if not batch.forward_mode.is_idle():
                accepted_mask = accept_index >= 0
                accepted_predict = torch.full_like(accept_index, -1)
                accepted_draft = (
                    welm_mtp_accepted_draft_token_ids
                    if welm_mtp_accepted_draft_token_ids is not None
                    else torch.full_like(accept_index, -1)
                )
                if bool(accepted_mask.any().item()):
                    accepted_indices = accept_index[accepted_mask].to(torch.long)
                    accepted_predict[accepted_mask] = predict.index_select(
                        0, accepted_indices
                    ).to(accepted_predict.dtype)
                verify_dump_extra = {
                    "accepted_predict_token_ids": accepted_predict,
                    "accepted_draft_token_ids": accepted_draft,
                }
            _dump_verify_after_event(
                "verify",
                {
                    "seq_lens_before_verify": batch.seq_lens,
                    "new_seq_lens": new_seq_lens,
                    "predict": predict,
                    "accept_lens": accept_lens,
                    "accept_index": accept_index,
                    "draft_token": verify_input.draft_token,
                    "can_run_cuda_graph": can_run_cuda_graph,
                    "speculative_num_draft_tokens": self.speculative_num_draft_tokens,
                    "speculative_num_steps": self.speculative_num_steps,
                    **verify_dump_extra,
                },
            )
        # Update mamba state for hybrid GDN models after verification
        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
        ):
            self._mamba_verify_update(
                batch, verify_input, accept_lens, accept_index, bs
            )

        next_token_ids = predict
        packed_hidden_states = None
        mirrored_kv_indices = None
        if self.topk > 1 and not batch.forward_mode.is_idle():
            packed_indices = self._pack_accept_path_indices(
                accept_index, accept_lens, self.speculative_num_draft_tokens
            )
            next_token_ids = predict[packed_indices]
            mirrored_kv_indices = packed_indices
            if logits_output.hidden_states is not None:
                packed_hidden_states = logits_output.hidden_states[packed_indices]
            self._compact_topk_accept_path_in_req_pool(
                batch, packed_indices, self.speculative_num_draft_tokens
            )

        if not batch.forward_mode.is_idle():
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            if self.topk > 1:
                fill_new_bonus_tokens[(bs,)](
                    next_token_ids,
                    accept_lens,
                    bonus_tokens,
                    self.speculative_num_draft_tokens,
                )
            else:
                all_bonus_tokens = predict[accept_index]
                fill_new_bonus_tokens[(bs,)](
                    all_bonus_tokens,
                    accept_lens,
                    bonus_tokens,
                    self.speculative_num_draft_tokens,
                )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        verify_done = torch.get_device_module(self.device).Event()
        verify_done.record()

        # Construct the next draft input
        next_draft_input = EagleDraftInput(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            verify_done=verify_done,
            hidden_states=packed_hidden_states,
            mirrored_kv_indices=mirrored_kv_indices,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
            draft_continuation_state=DraftContinuationState(next_draft_input),
            accept_lens=accept_lens,
            spec_accept_index=accept_index,
            routed_experts_output=forward_batch_output.routed_experts_output,
            welm_mtp_accepted_draft_token_ids=welm_mtp_accepted_draft_token_ids,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
        )

    def _mamba_verify_update(
        self,
        batch: ModelWorkerBatch,
        verify_input: EagleVerifyInput,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
        bs: int,
    ):
        """Update mamba state for hybrid GDN models after verification."""
        # `accept_lens` already includes the bonus token (drafts + 1 per req).
        accepted_length_with_bonus = accept_lens
        if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
            if verify_input.topk != 1:
                raise ValueError("Spec v2 currently only supports topk = 1.")

            accepted_indices_offset = torch.arange(
                0,
                bs * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=accepted_length_with_bonus.dtype,
                device=accepted_length_with_bonus.device,
            )
            accepted_steps = accepted_length_with_bonus - 1

            if batch.mamba_track_indices is not None:
                # If after verify, the request's seq_lens has crossed a mamba track interval,
                # we need to update the mamba state for the request at the crossing point.
                seq_lens_pre_verify = batch.seq_lens
                seq_lens_post_verify = batch.seq_lens + accepted_length_with_bonus
                mamba_track_interval = self.server_args.mamba_track_interval
                to_track_mask = (
                    seq_lens_pre_verify // mamba_track_interval
                    != seq_lens_post_verify // mamba_track_interval
                )
                tracking_point = (
                    seq_lens_post_verify // mamba_track_interval * mamba_track_interval
                )
                to_track_ith = torch.clamp(
                    tracking_point - seq_lens_pre_verify - 1, min=0
                ).to(torch.int64)
                req_idx = torch.arange(
                    bs,
                    dtype=torch.int64,
                    device=accepted_length_with_bonus.device,
                )
                candidate_track_steps = (
                    accept_index[req_idx, to_track_ith] - accepted_indices_offset
                )
                mamba_steps_to_track = torch.where(
                    to_track_mask,
                    candidate_track_steps,
                    torch.full_like(candidate_track_steps, -1),
                )
            else:
                mamba_steps_to_track = None

            self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
                accepted_steps=accepted_steps,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=self.target_worker.model_runner.model,
            )

    def move_accepted_tokens_to_target_kvcache(
        self,
        batch: ModelWorkerBatch,
        accept_index: torch.Tensor,
        num_correct_drafts: torch.Tensor,
    ):
        """
        Move accepted tokens to the target KV cache.

        Args:
            batch: The batch to run.
            accept_index: The index of the accepted tokens.
            num_correct_drafts: The length of the accepted tokens.
        """
        bs = len(batch.seq_lens)
        size = bs * self.speculative_num_draft_tokens

        tgt_cache_loc = torch.zeros(
            size,
            dtype=torch.int64,
            device=self.device,
        )
        accepted_out_cache_loc = torch.zeros(
            size, dtype=torch.int64, device=self.device
        )
        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + num_correct_drafts,
            tgt_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )
        fill_accepted_out_cache_loc[(size,)](
            accept_index,
            batch.out_cache_loc,
            accepted_out_cache_loc,
            next_power_of_2(size),
        )
        self.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
            tgt_cache_loc, accepted_out_cache_loc
        )

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.draft_worker.draft_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message

    def load_weights_from_distributed(self, received):
        """Load already-received (name, tensor) pairs into the draft model.

        The target receives the weights once via NCCL in the scheduler mixin
        (``tp_worker.receive_weights_from_distributed``); this loads the same
        received tensors into the draft model in-process. Used by WeLM MTP /
        NEXTN draft workers whose ``draft_runner`` owns the MTP head weights.
        """
        self._draft_worker.draft_runner.model.load_weights(received)
        return True, "Succeeded to update draft model weights."
