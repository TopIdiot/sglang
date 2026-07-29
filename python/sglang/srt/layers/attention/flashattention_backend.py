from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.distributed import get_sharded_kv_cp_group
from sglang.srt.layers.attention.attncp_fused_ops import (
    attncp_cp2_fused_q_fa_decode,
    attncp_cp2_fused_q_fa_max_splits,
    attncp_cp2_fused_q_fa_supports_shape,
    attncp_cp2_merge_local_head_slice,
    attncp_cp2_merge_local_remote_head_slice,
    attncp_cp2_pack_local_head_slice,
    attncp_sharded_kv_local_cap,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.cp_sharded_kv import (
    CPShardedKVPageTableResolver,
    CPShardedKVPrefillPlan,
    build_cp_prefill_swa_gather_plan,
    build_cp_sharded_kv_prefill_plan,
    gather_cp_prefill_kv,
    gather_cp_prefill_swa_kv,
    should_use_full_cp_kv_collective,
)
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils.cp_utils import (
    cp_allgather_and_save_kv_cache,
    cp_attn_forward_extend,
    is_cp_kv_sharded,
    is_decode_cp_kv_sharded,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import get_compiler_backend
from sglang.srt.utils.common import get_device_capability

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sgl_kernel import merge_state_v2

_WELM_V4_MODEL_TYPES = {"welmv4_moe"}
_WELM_V4_ARCHITECTURES = {
    "WeLMV4MoeForCausalLM",
    "WeLMV4MoeForCausalLMNextN",
}
_WELM_V4_NUM_SPLITS_ENV = "SGLANG_WELMV4_FLASH_ATTENTION_NUM_SPLITS"
_ATTNCP_DENSE_DECODE_GRAPH_MAX_SEQ_LEN_ENV = (
    "SGLANG_ATTNCP_DENSE_DECODE_GRAPH_MAX_SEQ_LEN"
)
_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_ENV = (
    "SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE"
)
_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_SWA_ENV = (
    "SGLANG_ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_SWA"
)
_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P_ENV = (
    "SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P"
)
_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_ALLOW_LOGPROB_ENV = (
    "SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_ALLOW_LOGPROB"
)
_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P_ENV = (
    "SGLANG_ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P"
)
_ATTNCP_DECODE_CP2_FUSED_MERGE_ENV = "SGLANG_ATTNCP_DECODE_CP2_FUSED_MERGE"
_ATTNCP_DEBUG_METADATA_CHECKS_ENV = "SGLANG_ATTNCP_DEBUG_METADATA_CHECKS"
_ATTNCP_DECODE_CP2_FUSED_Q_FA_MIN_SEQ_CAP = 16384

logger = logging.getLogger(__name__)


def _env_flag_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _is_welm_v4_model(model_config) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    hf_text_config = getattr(model_config, "hf_text_config", None)
    model_type = getattr(hf_config, "model_type", None) or getattr(
        hf_text_config, "model_type", None
    )
    if model_type in _WELM_V4_MODEL_TYPES:
        return True

    architectures = getattr(hf_config, "architectures", None) or getattr(
        hf_text_config, "architectures", None
    )
    return bool(
        architectures and any(arch in _WELM_V4_ARCHITECTURES for arch in architectures)
    )


def _should_use_prefill_cp_kv_ipc(model_runner) -> bool:
    server_args = model_runner.server_args
    return (
        _is_welm_v4_model(model_runner.model_config)
        and model_runner.attn_cp_size == 4
        and server_args.attn_cp_mode == "sharded-kv"
        and getattr(server_args, "disaggregation_mode", "null") != "decode"
    )


def _attncp_decode_fused_q_fa_active(server_args) -> bool:
    return bool(getattr(server_args, "attn_cp_decode_fused_q_fa", False)) and (
        getattr(server_args, "disaggregation_mode", "null") != "prefill"
    )


def _get_local_attention_head_counts(
    model_config, tp_size: Optional[int] = None
) -> tuple[int, int]:
    tp_size = get_attention_tp_size() if tp_size is None else tp_size
    return (
        model_config.hf_text_config.num_attention_heads // tp_size,
        model_config.get_num_kv_heads(tp_size),
    )


def _get_welm_v4_num_splits() -> int:
    env_value = os.getenv(_WELM_V4_NUM_SPLITS_ENV)
    if env_value is None or not env_value.strip():
        return 0
    try:
        num_splits = int(env_value)
    except ValueError as exc:
        raise ValueError(f"{_WELM_V4_NUM_SPLITS_ENV} must be an integer.") from exc
    if num_splits < 0:
        raise ValueError(f"{_WELM_V4_NUM_SPLITS_ENV} must be non-negative.")
    return num_splits


@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    # Sequence lengths for the forward batch
    cache_seqlens_int32: torch.Tensor = None
    # Maximum sequence length for query
    max_seq_len_q: int = 1
    # Maximum sequence length for key
    max_seq_len_k: int = 0
    # Cumulative sequence lengths for query
    cu_seqlens_q: torch.Tensor = None
    # Cumulative sequence lengths for key
    cu_seqlens_k: torch.Tensor = None
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)
    # Page table, the index of KV Cache Tables/Blocks
    page_table: torch.Tensor = None
    # Page table for Sliding Window Attention
    swa_page_table: torch.Tensor = None
    # Cache lengths relative to the first page in swa_page_table. Hybrid SWA
    # decode can keep only an aligned tail of a much longer logical sequence.
    swa_cache_seqlens_int32: torch.Tensor = None
    # Precomputed FA3 scheduler metadata (avoids per-layer prepare_varlen_num_blocks)
    scheduler_metadata: torch.Tensor = None

    # Encoder metadata
    # Cumulative sequence lengths for encoder key
    encoder_cu_seqlens_k: torch.Tensor = None
    # Maximum sequence length for encoder key
    encoder_max_seq_len_k: int = 0
    # Sequence lengths for the forward batch
    encoder_lens_int32: torch.Tensor = None
    # Page table for the encoder
    encoder_page_table: torch.Tensor = None

    # For WeLM KV mirror contracted query rows
    mirror_cu_seqlens_q: torch.Tensor = None
    mirror_max_seq_len_q: int = 1

    # For CP sharded-KV decode. These are compacted once per forward batch so
    # each layer can attend local KV shards without rebuilding full dense KV.
    cp_local_cache_seqlens_int32: torch.Tensor = None
    cp_local_page_table: torch.Tensor = None
    cp_swa_local_cache_seqlens_int32: torch.Tensor = None
    cp_swa_local_page_table: torch.Tensor = None
    cp_swa_local_page_start_offsets: torch.Tensor = None
    cp_swa_local_page_token_counts: torch.Tensor = None
    cp_sharded_kv_prefill_plan: Optional[CPShardedKVPrefillPlan] = None
    cp_sharded_kv_prefill_plan_initialized: bool = False
    requires_exact_logprob: bool = False
    has_empty_cp_shard: bool = False

    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor = None  # cu_seqlens_q for local attention
        local_seqused_k: torch.Tensor = None  # sequence lengths for local attention
        local_block_table: torch.Tensor = None  # block table for local attention
        local_max_query_len: int = 0  # max query length for local attention
        local_max_seq_len: int = 0  # max sequence length for local attention

    local_attn_metadata: Optional[LocalAttentionMetadata] = None

    # For sliding window attention topk>1 spec decoding
    swa_spec_metadata: Optional[FlashAttentionMetadata] = None


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend implementation.

    Note about the init:
    - If no spec decoding
        - FlashAttentionBackend will be init once when the server starts.
    - If spec decoding
        - FlashAttentionBackend will be init once for the target worker
        - FlashAttentionMultiStepBackend will be once for the draft worker
            - It will spawn num_steps FlashAttentionBackend for the draft worker

    Note about CUDA Graph:
    - We only support CUDA Graph for Decode (Normal Decode and Draft Decode) and Target Verify.
    - We don't support CUDA Graph for Extend and Draft Extend.
    - When server init, init_cuda_graph_state will be called first and then init_cuda_graph_capture will be called.
    - For each forward batch, init_replay_cuda_graph will be called first and then replay the graph.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
        fa_impl_ver=3,
    ):
        super().__init__()
        server_args = model_runner.server_args

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.forward_metadata: FlashAttentionMetadata = None
        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.cp_sharded_page_table_resolver = CPShardedKVPageTableResolver(
            self.token_to_kv_pool_allocator
        )
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.is_welm_v4_model = _is_welm_v4_model(model_runner.model_config)
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.skip_prefill = skip_prefill
        self.attn_cp_size = model_runner.attn_cp_size
        self.attn_cp_kv_chunk_size = model_runner.server_args.attn_cp_kv_chunk_size
        self.is_attn_cp_sharded_kv = (
            self.attn_cp_size > 1
            and model_runner.server_args.attn_cp_mode == "sharded-kv"
        )
        self.enable_attn_cp_decode_local_merge = (
            self.is_attn_cp_sharded_kv
            and _env_flag_enabled(
                _ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_ENV, default=True
            )
        )
        self.enable_attn_cp_decode_local_merge_swa = (
            self.enable_attn_cp_decode_local_merge
            and _env_flag_enabled(
                _ATTNCP_EXPERIMENTAL_DECODE_LOCAL_MERGE_SWA_ENV, default=True
            )
        )
        attn_cp_decode_fused_q_fa_requested = bool(
            getattr(
                model_runner.server_args,
                "attn_cp_decode_fused_q_fa",
                False,
            )
        )
        if (
            attn_cp_decode_fused_q_fa_requested
            and not _attncp_decode_fused_q_fa_active(model_runner.server_args)
        ):
            logger.info_once(
                "Ignoring decode-only --attn-cp-decode-fused-q-fa on the "
                "disaggregated Prefill role."
            )
        attn_cp_decode_fused_q_fa_requested = (
            _attncp_decode_fused_q_fa_active(model_runner.server_args)
        )
        self.enable_attn_cp_decode_cp2_q_p2p = (
            self.enable_attn_cp_decode_local_merge
            and _env_flag_enabled(
                _ATTNCP_EXPERIMENTAL_DECODE_CP2_Q_P2P_ENV,
                default=attn_cp_decode_fused_q_fa_requested,
            )
        )
        self.enable_attn_cp_decode_cp2_fused_q_fa = (
            self.enable_attn_cp_decode_local_merge
            and attn_cp_decode_fused_q_fa_requested
        )
        self.attn_cp_decode_cp2_fused_q_fa_min_seq_cap = (
            _ATTNCP_DECODE_CP2_FUSED_Q_FA_MIN_SEQ_CAP
        )
        self.attn_cp_decode_cp2_fused_q_fa_allow_logprob = _env_flag_enabled(
            _ATTNCP_EXPERIMENTAL_DECODE_CP2_FUSED_Q_FA_ALLOW_LOGPROB_ENV,
            default=False,
        )
        self.enable_attn_cp_decode_cp2_olse_p2p = (
            self.enable_attn_cp_decode_local_merge
            and _env_flag_enabled(
                _ATTNCP_EXPERIMENTAL_DECODE_CP2_OLSE_P2P_ENV,
                default=attn_cp_decode_fused_q_fa_requested,
            )
        )
        self.enable_attn_cp_decode_cp2_fused_merge = (
            self.enable_attn_cp_decode_local_merge
            and _env_flag_enabled(_ATTNCP_DECODE_CP2_FUSED_MERGE_ENV, default=True)
        )
        self.debug_attn_cp_metadata_checks = (
            self.is_attn_cp_sharded_kv
            and _env_flag_enabled(_ATTNCP_DEBUG_METADATA_CHECKS_ENV, default=False)
        )
        self.attncp_full_sinks_cache: dict[
            tuple[int, int, torch.dtype, torch.device],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self.attncp_dense_window_static_tensors: dict[
            tuple[int, int, torch.device],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self.attncp_dense_gather_static_tensors: dict[
            tuple[int, int, torch.device],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self.enable_attn_cp_zero_dummy_slot = self.is_attn_cp_sharded_kv
        self.cuda_graph_max_seq_len = int(self.max_context_len)
        self.cuda_graph_max_seq_len_is_explicit = False
        if self.is_attn_cp_sharded_kv:
            graph_seq_cap_arg = int(
                getattr(
                    model_runner.server_args,
                    "attn_cp_decode_cuda_graph_max_seq_len",
                    0,
                )
                or 0
            )
            graph_seq_cap_env = os.getenv(_ATTNCP_DENSE_DECODE_GRAPH_MAX_SEQ_LEN_ENV)
            if graph_seq_cap_arg > 0:
                self.cuda_graph_max_seq_len = graph_seq_cap_arg
                self.cuda_graph_max_seq_len_is_explicit = True
            elif graph_seq_cap_env is not None and graph_seq_cap_env.strip():
                try:
                    self.cuda_graph_max_seq_len = int(graph_seq_cap_env)
                except ValueError as exc:
                    raise ValueError(
                        f"{_ATTNCP_DENSE_DECODE_GRAPH_MAX_SEQ_LEN_ENV} must be an integer"
                    ) from exc
                if self.cuda_graph_max_seq_len <= 0:
                    raise ValueError(
                        f"{_ATTNCP_DENSE_DECODE_GRAPH_MAX_SEQ_LEN_ENV} must be positive"
                    )
                self.cuda_graph_max_seq_len_is_explicit = True
            else:
                self.cuda_graph_max_seq_len = int(self.max_context_len)
            self.cuda_graph_max_seq_len = max(
                1, min(self.cuda_graph_max_seq_len, int(self.max_context_len))
            )

        self.use_sliding_window_kv_pool = (
            isinstance(model_runner.token_to_kv_pool, SWAKVPool)
            and model_runner.token_to_kv_pool.swa_layer_nums > 0
        )
        if self.use_sliding_window_kv_pool:
            self.token_to_kv_pool = model_runner.token_to_kv_pool

        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id

        # Local attention settings
        self.has_local_attention = model_runner.model_config.is_local_attention_model
        if self.has_local_attention:
            assert (
                model_runner.attention_chunk_size is not None
            ), "Attention chunk size is required for local attention"
            self.attention_chunk_size = model_runner.attention_chunk_size

        # The model-wide minimum window sizes shared SWA metadata buffers.  In a
        # hybrid KV pool, the pool mapping remains authoritative per layer.
        self.sliding_window_size = model_runner.sliding_window_size
        self.has_swa = (
            self.sliding_window_size is not None and self.sliding_window_size > -1
        )

        # Select version
        self.fa_impl_ver = fa_impl_ver
        if self.fa_impl_ver == 3:
            from sgl_kernel.flash_attn import (
                flash_attn_varlen_func,
                flash_attn_with_kvcache,
                get_scheduler_metadata,
            )

            self._get_scheduler_metadata = get_scheduler_metadata
        elif self.fa_impl_ver == 4:
            from sglang.jit_kernel.flash_attention_v4 import (
                flash_attn_varlen_func,
                flash_attn_with_kvcache,
            )

            self._get_scheduler_metadata = None
        else:
            raise ValueError(f"Invalid version: {self.fa_impl_ver=}")

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.flash_attn_with_kvcache = flash_attn_with_kvcache

        # Store head info for precomputing FA3 scheduler metadata
        self.head_dim = model_runner.model_config.head_dim
        projection_tp_size = (
            model_runner.tp_size
            if is_decode_cp_kv_sharded(model_runner.server_args)
            else get_attention_tp_size()
        )
        self.num_attention_heads, self.num_kv_heads = (
            _get_local_attention_head_counts(
                model_runner.model_config, tp_size=projection_tp_size
            )
        )
        self.prefill_cp_kv_ipc_transport = None
        use_prefill_cp_kv_ipc = _should_use_prefill_cp_kv_ipc(model_runner)
        if use_prefill_cp_kv_ipc:
            if not self.is_attn_cp_sharded_kv:
                raise RuntimeError(
                    "Automatic Prefill CP K/V IPC was selected outside "
                    "sharded-KV AttnCP"
                )
            if getattr(server_args, "nnodes", 1) != 1:
                raise RuntimeError(
                    "Automatic Prefill CP K/V IPC requires single-node serving"
                )
            if getattr(server_args, "enable_two_batch_overlap", False) or getattr(
                server_args, "enable_pdmux", False
            ):
                raise RuntimeError(
                    "Automatic Prefill CP K/V IPC requires one model forward "
                    "stream and does not support TBO or PD multiplexing"
                )
            if torch.device(model_runner.device).type != "cuda":
                raise RuntimeError("Automatic Prefill CP K/V IPC requires CUDA")
            if self.kv_cache_dtype != torch.bfloat16:
                raise RuntimeError(
                    "Automatic Prefill CP K/V IPC requires the resolved KV dtype BF16"
                )
            v_head_dim = (
                getattr(model_runner.model_config, "v_head_dim", None)
                or self.head_dim
            )
            k_row_width = self.num_kv_heads * self.head_dim
            v_row_width = self.num_kv_heads * v_head_dim
            if k_row_width != 256 or v_row_width != 256:
                raise RuntimeError(
                    "Automatic Prefill CP K/V IPC requires one local "
                    "256-element K/V row"
                )
            transport = getattr(
                model_runner, "_prefill_cp_kv_ipc_transport", None
            )
            if transport is None:
                from sglang.srt.layers.attention.prefill_cp_kv_ipc import (
                    CPKVIPCSourcePushTransport,
                )

                transport = CPKVIPCSourcePushTransport(
                    cp_group=get_sharded_kv_cp_group(),
                    device=model_runner.device,
                    max_rows=self.max_context_len,
                    row_width=k_row_width,
                    max_calls_per_forward=max(
                        1,
                        int(model_runner.model_config.num_hidden_layers)
                        + int(
                            getattr(
                                model_runner.model_config,
                                "num_nextn_predict_layers",
                                0,
                            )
                            or 0
                        ),
                    ),
                )
                model_runner._prefill_cp_kv_ipc_transport = transport
                logger.info(
                    "Using automatic Prefill CP K/V IPC source-push: CP4, BF16, "
                    "row_width=256, arena_bytes=%d",
                    transport.layout.arena_bytes,
                )
            elif (
                transport.layout.max_rows != self.max_context_len
                or transport.layout.row_width != k_row_width
            ):
                raise RuntimeError(
                    "Prefill CP K/V IPC transport shape differs across "
                    "attention backends"
                )
            self.prefill_cp_kv_ipc_transport = transport
        if (
            self.enable_attn_cp_decode_cp2_fused_q_fa
            and model_runner.server_args.disaggregation_mode != "prefill"
        ):
            self._require_attncp_fused_q_fa_shape(
                local_q_heads=self.num_attention_heads,
                local_kv_heads=self.num_kv_heads,
                cp_world_size=self.attn_cp_size,
            )
        _softcapping = getattr(
            model_runner.model_config.hf_text_config, "attn_logit_softcapping", None
        )
        self.has_softcap = _softcapping is not None and _softcapping > 0.0

        # If num_splits == 0, we use a heuristic to automatically determine the number of splits.
        # We set nums splits to 1 if deterministic inference is enabled.
        # See https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ for more details.
        # Furthermore, FA4 does not support num_splits=0 with CUDA Graph, so we set num_splits to 1 if CUDA Graph is enabled.
        device_capability = get_device_capability()
        fa4_no_splitkv = self.fa_impl_ver == 4 and (
            device_capability < (9, 0) or device_capability >= (12, 0)
        )
        if (
            model_runner.server_args.enable_deterministic_inference
            or fa4_no_splitkv
            or (
                self.fa_impl_ver == 4
                and not model_runner.server_args.disable_cuda_graph
            )
        ):
            self.num_splits = 1
        elif self.is_welm_v4_model:
            self.num_splits = _get_welm_v4_num_splits()
        else:
            self.num_splits = 0

        # In embedding mode with no chunked prefill and radix cache disabled,
        # skip KV cache write and use flash_attn_varlen_func with raw K/V
        # instead of flash_attn_with_kvcache, bypassing paged KV cache entirely.
        # Restricted to non-MLA backends: the read-skip elif lives inside the
        # `if not self.use_mla:` branch in forward_extend, while the write-skip
        # guard wraps both set_kv_buffer and set_mla_kv_buffer. Without this
        # gate, MLA + is_embedding would skip the write but still read stale
        # cache via get_key_buffer in the absorbed-MLA path.
        self.fa_skip_kv_cache = (
            server_args.is_embedding
            and server_args.chunked_prefill_size == -1
            and server_args.disable_radix_cache
            and not self.use_mla
        )

        # Skip the FA3 scheduler_metadata precompute (PR #21104) under DP
        # attention. The precomputed buffer can become inconsistent with the
        # num_splits the C++ mha_fwd kernel derives from live cache_seqlens
        # during decode, leading to an OOB read in the split-KV combine kernel
        # (flash_fwd_combine_launch_template.h:52). Leaving scheduler_metadata
        # unset uses the existing per-layer metadata path.
        self._disable_scheduler_metadata_precompute = bool(
            getattr(server_args, "enable_dp_attention", False)
        )

    def _translate_sharded_logical_page_table(
        self, page_table: torch.Tensor
    ) -> torch.Tensor:
        return self.cp_sharded_page_table_resolver.resolve_full(page_table)

    def _resolve_sharded_logical_swa_page_table(
        self, logical_full_pages: torch.Tensor
    ) -> torch.Tensor:
        return self.cp_sharded_page_table_resolver.resolve_swa_pages(
            logical_full_pages
        )

    def _compute_scheduler_metadata(
        self, batch_size, max_seq_len_k, cache_seqlens, cu_seqlens_q
    ):
        """Compute FA3 scheduler metadata for decode.

        Returns the scheduler_metadata tensor, or None if not applicable.
        """
        if self._get_scheduler_metadata is None or self.use_mla:
            return None
        if self._disable_scheduler_metadata_precompute:
            return None
        # Always use window_size=(-1, -1) because scheduler_metadata is only
        # consumed by non-SWA layers (SWA layers skip it in forward_decode).
        return self._get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=1,
            max_seqlen_k=max_seq_len_k,
            num_heads=self.num_attention_heads,
            num_heads_k=self.num_kv_heads,
            headdim=self.head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=self.kv_cache_dtype,
            cu_seqlens_q=cu_seqlens_q,
            page_size=self.page_size,
            causal=True,
            has_softcap=self.has_softcap,
            num_splits=self.num_splits,
        )

    def _gather_sharded_kv_dense(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        page_table_is_physical: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct a temporary dense full-KV table from sharded KV slots."""
        if page_table is None:
            raise RuntimeError("CP sharded-KV gathered attention missing page_table")

        if not page_table_is_physical:
            page_table = self._translate_sharded_logical_page_table(page_table)
        slots = page_table.to(dtype=torch.long)
        batch_size = slots.shape[0]
        if self.page_size == 1:
            max_seq_len = slots.shape[1]
            logical_pos, dense_page_table = (
                self._get_attncp_dense_gather_static_tensors(
                    batch_size, max_seq_len, slots.device
                )
            )
            valid_by_len = logical_pos < cache_seqlens.unsqueeze(1)
            local_valid = valid_by_len & slots.ne(0)
            safe_slots = torch.where(local_valid, slots, torch.zeros_like(slots))

            if self.enable_attn_cp_zero_dummy_slot:
                key_cache[0].zero_()
                value_cache[0].zero_()
            local_k = key_cache[safe_slots][:, :, 0].contiguous()
            local_v = value_cache[safe_slots][:, :, 0].contiguous()
            if not self.enable_attn_cp_zero_dummy_slot:
                local_mask = local_valid.unsqueeze(-1).unsqueeze(-1)
                local_k = local_k * local_mask.to(dtype=local_k.dtype)
                local_v = local_v * local_mask.to(dtype=local_v.dtype)

            cp_group = get_sharded_kv_cp_group()
            full_k, full_v = cp_group.all_reduce_coalesced([local_k, local_v])
            return full_k.contiguous(), full_v.contiguous(), dense_page_table

        max_pages = slots.shape[1]
        page_starts = (
            torch.arange(max_pages, device=slots.device, dtype=torch.long)
            .view(1, max_pages, 1)
            .mul_(self.page_size)
        )
        offsets_in_page = torch.arange(
            self.page_size, device=slots.device, dtype=torch.long
        ).view(1, 1, self.page_size)
        logical_pos = page_starts + offsets_in_page
        page_slots = slots.unsqueeze(-1).expand(batch_size, max_pages, self.page_size)
        valid_by_len = logical_pos < cache_seqlens.to(torch.long).view(batch_size, 1, 1)
        local_valid = valid_by_len & page_slots.ne(0)
        safe_slots = torch.where(local_valid, page_slots, torch.zeros_like(page_slots))

        if self.enable_attn_cp_zero_dummy_slot:
            key_cache[0].zero_()
            value_cache[0].zero_()
        expanded_offsets = offsets_in_page.expand(batch_size, max_pages, -1)
        local_k = key_cache[safe_slots, expanded_offsets].contiguous()
        local_v = value_cache[safe_slots, expanded_offsets].contiguous()
        if not self.enable_attn_cp_zero_dummy_slot:
            local_mask = local_valid.unsqueeze(-1).unsqueeze(-1)
            local_k = local_k * local_mask.to(dtype=local_k.dtype)
            local_v = local_v * local_mask.to(dtype=local_v.dtype)

        cp_group = get_sharded_kv_cp_group()
        full_k, full_v = cp_group.all_reduce_coalesced([local_k, local_v])
        full_k = full_k.contiguous()
        full_v = full_v.contiguous()

        dense_page_table = torch.arange(
            batch_size * max_pages, dtype=torch.int32, device=slots.device
        ).view(batch_size, max_pages)

        # Each batch row maps to a contiguous [bs, max_pages) slice of the
        # dense K/V pages above. Keeping the original page size avoids changing
        # FA3 scheduling and numerics in the correctness fallback.
        return full_k, full_v, dense_page_table

    def _set_sharded_kv_prefill_plan(
        self,
        metadata: FlashAttentionMetadata,
        forward_batch: ForwardBatch,
    ) -> None:
        if metadata.cp_sharded_kv_prefill_plan_initialized:
            return
        metadata.cp_sharded_kv_prefill_plan = build_cp_sharded_kv_prefill_plan(
            logical_page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32,
            prefix_lens=forward_batch.extend_prefix_lens_cpu,
            seq_lens=forward_batch.seq_lens_cpu,
            page_size=self.page_size,
        )
        metadata.cp_sharded_kv_prefill_plan_initialized = True

    def _gather_sharded_kv_dense_prefill_split(
        self,
        plan: Optional[CPShardedKVPrefillPlan],
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Gather prefix and extend KV separately for sharded-KV prefill."""
        if plan is None:
            return None

        prefix_k, prefix_v, _ = self._gather_sharded_kv_dense(
            plan.prefix.logical_page_table,
            plan.prefix.cache_seqlens,
            key_cache,
            value_cache,
        )
        extend_k, extend_v, _ = self._gather_sharded_kv_dense(
            plan.extend.logical_page_table,
            plan.extend.cache_seqlens,
            key_cache,
            value_cache,
        )
        dense_k = torch.cat((prefix_k, extend_k), dim=1).contiguous()
        dense_v = torch.cat((prefix_v, extend_v), dim=1).contiguous()
        dense_page_table = torch.arange(
            plan.total_page_columns,
            dtype=torch.int32,
            device=plan.prefix.logical_page_table.device,
        ).view(1, plan.total_page_columns)
        return dense_k, dense_v, dense_page_table, plan.full_cache_seqlens

    @staticmethod
    def _attncp_forward_batch_requires_exact_logprob(
        forward_batch: Optional[ForwardBatch],
    ) -> bool:
        if forward_batch is None:
            return False
        return bool(getattr(forward_batch, "return_logprob", False))

    def _attncp_current_batch_requires_exact_logprob(self) -> bool:
        return self._attncp_forward_batch_requires_exact_logprob(
            getattr(self, "_replay_forward_batch", None)
        )

    @staticmethod
    def _attncp_seq_lens_has_empty_cp_shard(
        seq_lens_cpu,
        cp_world_size: int,
        cp_kv_chunk_size: int,
    ) -> bool:
        cp_world_size = int(cp_world_size or 0)
        cp_kv_chunk_size = int(cp_kv_chunk_size or 0)
        if cp_world_size <= 1 or cp_kv_chunk_size <= 0 or seq_lens_cpu is None:
            return False

        threshold = (cp_world_size - 1) * cp_kv_chunk_size
        if threshold <= 0:
            return False

        if isinstance(seq_lens_cpu, torch.Tensor):
            if seq_lens_cpu.numel() == 0:
                return False
            return bool((seq_lens_cpu.to(torch.long) <= threshold).any().item())

        try:
            return any(int(seq_len) <= threshold for seq_len in seq_lens_cpu)
        except TypeError:
            return int(seq_lens_cpu) <= threshold

    def attncp_forward_batch_has_empty_cp_shard(
        self, forward_batch: Optional[ForwardBatch]
    ) -> bool:
        if not self.is_attn_cp_sharded_kv or forward_batch is None:
            return False
        return self._attncp_seq_lens_has_empty_cp_shard(
            getattr(forward_batch, "seq_lens_cpu", None),
            self.attn_cp_size,
            self.attn_cp_kv_chunk_size,
        )

    def _get_attncp_dense_gather_static_tensors(
        self,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (batch_size, max_seq_len, device)
        cached = self.attncp_dense_gather_static_tensors.get(key)
        if cached is None:
            if len(self.attncp_dense_gather_static_tensors) >= 32:
                self.attncp_dense_gather_static_tensors.clear()
            cached = (
                torch.arange(max_seq_len, device=device).view(1, max_seq_len),
                torch.arange(
                    batch_size * max_seq_len,
                    dtype=torch.int32,
                    device=device,
                ).view(batch_size, max_seq_len),
            )
            self.attncp_dense_gather_static_tensors[key] = cached
        return cached

    def _gather_sharded_kv_dense_decode_window(
        self,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        window_left: int,
        *,
        page_table_is_physical: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct compact K/V while preserving full logical positions."""
        if page_table is None:
            raise RuntimeError("CP sharded-KV gathered attention missing page_table")

        raw_page_table = page_table
        if not page_table_is_physical:
            page_table = self._translate_sharded_logical_page_table(page_table)
        slots = page_table.to(dtype=torch.long)
        batch_size, max_pages = slots.shape
        max_seq_len = max_pages * self.page_size
        window_tokens = int(window_left) + 1
        if window_tokens <= 0 or window_tokens >= max_seq_len:
            full_k, full_v, dense_page_table = self._gather_sharded_kv_dense(
                raw_page_table,
                cache_seqlens,
                key_cache,
                value_cache,
                page_table_is_physical=page_table_is_physical,
            )
            return full_k, full_v, dense_page_table, cache_seqlens

        static_tensors = self._get_attncp_dense_window_static_tensors(
            batch_size, window_tokens, slots.device
        )
        if static_tensors is None:
            offsets = torch.arange(window_tokens, device=slots.device).unsqueeze(0)
            row_indices = torch.arange(batch_size, device=slots.device).unsqueeze(1)
            compact_slots = torch.arange(
                batch_size * window_tokens, dtype=torch.int32, device=slots.device
            ).view(batch_size, window_tokens)
        else:
            offsets, row_indices, compact_slots = static_tensors

        start = torch.clamp(cache_seqlens.to(torch.long) - window_tokens, min=0)
        cols = start.unsqueeze(1) + offsets
        valid = cols < cache_seqlens.to(torch.long).unsqueeze(1)
        page_cols = torch.div(cols, self.page_size, rounding_mode="floor")
        safe_page_cols = torch.where(valid, page_cols, torch.zeros_like(page_cols))
        offsets_in_page = cols - page_cols * self.page_size
        window_slots = slots[row_indices, safe_page_cols]
        local_valid = valid & window_slots.ne(0)
        safe_slots = torch.where(
            local_valid, window_slots, torch.zeros_like(window_slots)
        )

        if self.enable_attn_cp_zero_dummy_slot:
            key_cache[0].zero_()
            value_cache[0].zero_()
        local_k_window = key_cache[safe_slots, offsets_in_page].contiguous()
        local_v_window = value_cache[safe_slots, offsets_in_page].contiguous()
        if not self.enable_attn_cp_zero_dummy_slot:
            local_mask = local_valid.unsqueeze(-1).unsqueeze(-1)
            local_k_window = local_k_window * local_mask.to(dtype=local_k_window.dtype)
            local_v_window = local_v_window * local_mask.to(dtype=local_v_window.dtype)

        cp_group = get_sharded_kv_cp_group()
        full_k_window, full_v_window = cp_group.all_reduce_coalesced(
            [local_k_window, local_v_window]
        )

        dense_page_table = torch.zeros(
            (batch_size, max_seq_len), dtype=torch.int32, device=slots.device
        )
        dense_page_table.scatter_(dim=1, index=cols, src=compact_slots)
        return (
            full_k_window.contiguous(),
            full_v_window.contiguous(),
            dense_page_table,
            cache_seqlens,
        )

    def _get_attncp_dense_window_static_tensors(
        self,
        batch_size: int,
        window_tokens: int,
        device: torch.device,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        graph_offsets = self.decode_cuda_graph_metadata.get(
            "attncp_dense_window_offsets"
        )
        graph_rows = self.decode_cuda_graph_metadata.get("attncp_dense_window_rows")
        graph_compact_slots = self.decode_cuda_graph_metadata.get(
            "attncp_dense_window_compact_slots"
        )
        if (
            graph_offsets is not None
            and graph_rows is not None
            and graph_compact_slots is not None
            and graph_offsets.numel() >= window_tokens
            and graph_rows.numel() >= batch_size
            and graph_compact_slots.shape[0] >= batch_size
            and graph_compact_slots.shape[1] >= window_tokens
        ):
            return (
                graph_offsets[:window_tokens].view(1, window_tokens),
                graph_rows[:batch_size].view(batch_size, 1),
                graph_compact_slots[:batch_size, :window_tokens],
            )

        key = (batch_size, window_tokens, device)
        cached = self.attncp_dense_window_static_tensors.get(key)
        if cached is None:
            cached = (
                torch.arange(window_tokens, device=device).view(1, window_tokens),
                torch.arange(batch_size, device=device).view(batch_size, 1),
                torch.arange(
                    batch_size * window_tokens,
                    dtype=torch.int32,
                    device=device,
                ).view(batch_size, window_tokens),
            )
            self.attncp_dense_window_static_tensors[key] = cached
        return cached

    def _attncp_local_kv_cap(self, max_seq_len: int) -> int:
        """Upper-bound local CP-owned KV tokens for a global sequence cap."""
        cp_group = get_sharded_kv_cp_group()
        return attncp_sharded_kv_local_cap(
            max_seq_len,
            cp_rank=cp_group.rank_in_group,
            cp_size=cp_group.world_size,
            chunk_size=self.attn_cp_kv_chunk_size,
        )

    def _set_sharded_kv_decode_metadata(
        self,
        metadata: FlashAttentionMetadata,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        *,
        out_page_table: Optional[torch.Tensor] = None,
        out_cache_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Compact the local CP-owned KV slots for sharded-KV decode."""
        if page_table is None:
            raise RuntimeError("CP sharded-KV decode missing page_table")

        page_table = self._translate_sharded_logical_page_table(page_table)
        slots = page_table.to(dtype=torch.int32)
        max_pages = slots.shape[1]
        max_seq_len = max_pages * self.page_size
        local_token_cap = self._attncp_local_kv_cap(max_seq_len)
        local_page_table_cap = max(
            1, (local_token_cap + self.page_size - 1) // self.page_size
        )
        if out_page_table is None:
            local_page_table = torch.empty(
                slots.shape[0],
                local_page_table_cap,
                dtype=slots.dtype,
                device=slots.device,
            )
        else:
            if out_page_table.shape[1] < local_page_table_cap:
                raise RuntimeError(
                    "CP sharded-KV decode local page table buffer is too small: "
                    f"{out_page_table.shape[1]} < {local_page_table_cap}"
                )
            local_page_table = out_page_table[:, :local_page_table_cap]
        local_page_table.zero_()

        page_starts = (
            torch.arange(max_pages, device=slots.device, dtype=torch.long).unsqueeze(0)
            * self.page_size
        )
        cache_lens = cache_seqlens.to(torch.long).unsqueeze(1)
        page_token_counts = torch.clamp(
            cache_lens - page_starts, min=0, max=self.page_size
        ).to(torch.int32)
        local_valid_pages = page_token_counts.gt(0) & slots.ne(0)
        local_cache_seqlens = torch.where(
            local_valid_pages,
            page_token_counts,
            torch.zeros_like(page_token_counts),
        ).sum(dim=1, dtype=torch.int32)

        if max_pages > 0:
            compact_cols = torch.cumsum(local_valid_pages.to(torch.int32), dim=1) - 1
            scatter_cols = torch.where(
                local_valid_pages,
                compact_cols,
                torch.zeros_like(compact_cols),
            ).to(torch.long)
            scatter_slots = torch.where(
                local_valid_pages, slots, torch.zeros_like(slots)
            )
            local_page_table.scatter_reduce_(
                dim=1,
                index=scatter_cols,
                src=scatter_slots,
                reduce="amax",
                include_self=True,
            )

        if out_cache_seqlens is None:
            metadata.cp_local_cache_seqlens_int32 = local_cache_seqlens
        else:
            out_cache_seqlens.copy_(local_cache_seqlens)
            metadata.cp_local_cache_seqlens_int32 = out_cache_seqlens
        metadata.cp_local_page_table = local_page_table

    def _attncp_swa_window_tokens(self) -> Optional[int]:
        if not self.has_swa or self.sliding_window_size is None:
            return None
        window_tokens = int(self.sliding_window_size) + 1
        if window_tokens <= 0 or window_tokens >= int(self.max_context_len):
            return None
        return window_tokens

    def _is_swa_kv_pool_layer(
        self, layer: RadixAttention, token_to_kv_pool
    ) -> bool:
        if self.use_sliding_window_kv_pool:
            is_swa_layer = getattr(token_to_kv_pool, "is_swa_layer", None)
            if not callable(is_swa_layer):
                raise RuntimeError(
                    "Hybrid SWA KV pool must expose is_swa_layer(layer_id)"
                )
            return bool(is_swa_layer(layer.layer_id))
        return (
            layer.sliding_window_size is not None
            and layer.sliding_window_size > -1
        )

    def _set_decode_swa_metadata(
        self,
        metadata: FlashAttentionMetadata,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        *,
        out_page_table: Optional[torch.Tensor] = None,
        out_cache_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Expose the aligned hybrid-SWA tail as a dense FA3 page table."""
        if not self.has_swa or self.sliding_window_size is None:
            raise RuntimeError("SWA decode metadata requested without an SWA window")
        if page_table is None or page_table.dim() != 2:
            raise RuntimeError("SWA decode requires a two-dimensional page table")

        window_tokens = int(self.sliding_window_size) + 1
        if window_tokens <= 0:
            raise RuntimeError(f"Invalid SWA decode window: {window_tokens}")

        slots = page_table.to(dtype=torch.int32)
        batch_size, table_len = slots.shape
        if cache_seqlens.numel() != batch_size:
            raise RuntimeError(
                "SWA decode page table/cache length batch mismatch: "
                f"{batch_size} != {cache_seqlens.numel()}"
            )
        if self.page_size == 1:
            compact_cap = min(table_len, window_tokens)
        else:
            compact_cap = min(
                table_len,
                (window_tokens + 2 * self.page_size - 2) // self.page_size,
            )

        if out_page_table is None:
            compact_page_table = torch.empty(
                batch_size,
                compact_cap,
                dtype=torch.int32,
                device=slots.device,
            )
        else:
            if out_page_table.shape[0] < batch_size:
                raise RuntimeError(
                    "SWA decode page table batch buffer is too small: "
                    f"{out_page_table.shape[0]} < {batch_size}"
                )
            if out_page_table.shape[1] < compact_cap:
                raise RuntimeError(
                    "SWA decode page table buffer is too small: "
                    f"{out_page_table.shape[1]} < {compact_cap}"
                )
            compact_page_table = out_page_table[:batch_size, :compact_cap]
        compact_page_table.zero_()

        cache_lens_long = cache_seqlens.to(torch.long)
        window_start = torch.clamp(cache_lens_long - window_tokens, min=0)
        first_page = torch.div(window_start, self.page_size, rounding_mode="floor")
        relative_cache_seqlens = (
            cache_lens_long - first_page * self.page_size
        ).to(torch.int32)

        if compact_cap > 0:
            page_offsets = torch.arange(
                compact_cap, device=slots.device, dtype=torch.long
            ).unsqueeze(0)
            page_cols = first_page.unsqueeze(1) + page_offsets
            valid = page_cols < table_len
            valid = valid & (
                page_cols * self.page_size < cache_lens_long.unsqueeze(1)
            )
            safe_cols = torch.where(valid, page_cols, torch.zeros_like(page_cols))
            row_indices = torch.arange(
                batch_size, device=slots.device, dtype=torch.long
            ).unsqueeze(1)
            compact_page_table.copy_(
                torch.where(
                    valid,
                    slots[row_indices, safe_cols],
                    torch.zeros_like(safe_cols, dtype=torch.int32),
                )
            )

        if out_cache_seqlens is None:
            metadata.swa_cache_seqlens_int32 = relative_cache_seqlens
        else:
            if out_cache_seqlens.numel() < batch_size:
                raise RuntimeError(
                    "SWA decode cache length buffer is too small: "
                    f"{out_cache_seqlens.numel()} < {batch_size}"
                )
            graph_cache_seqlens = (
                out_cache_seqlens
                if out_cache_seqlens.numel() == batch_size
                else out_cache_seqlens[:batch_size]
            )
            graph_cache_seqlens.copy_(relative_cache_seqlens)
            metadata.swa_cache_seqlens_int32 = graph_cache_seqlens
        metadata.swa_page_table = compact_page_table

    def _set_sharded_kv_decode_swa_metadata(
        self,
        metadata: FlashAttentionMetadata,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        *,
        out_page_table: Optional[torch.Tensor] = None,
        out_cache_seqlens: Optional[torch.Tensor] = None,
        out_page_start_offsets: Optional[torch.Tensor] = None,
        out_page_token_counts: Optional[torch.Tensor] = None,
    ) -> None:
        """Compact physical SWA pages inside the global CP decode window."""
        window_tokens = self._attncp_swa_window_tokens()
        if window_tokens is None:
            return
        if page_table is None:
            raise RuntimeError("CP sharded-KV SWA decode missing page_table")

        slots = page_table.to(dtype=torch.int32)
        batch_size, table_len = slots.shape
        if self.page_size == 1:
            local_page_table_cap = window_tokens
        else:
            local_page_table_cap = min(
                table_len,
                (window_tokens + 2 * self.page_size - 2) // self.page_size,
            )
        if out_page_table is None:
            local_page_table = torch.empty(
                batch_size,
                local_page_table_cap,
                dtype=torch.int32,
                device=slots.device,
            )
        else:
            if out_page_table.shape[1] < local_page_table_cap:
                raise RuntimeError(
                    "CP sharded-KV decode SWA local page table buffer is too small: "
                    f"{out_page_table.shape[1]} < {local_page_table_cap}"
                )
            local_page_table = out_page_table[:, :local_page_table_cap]
        local_page_table.zero_()

        if self.page_size == 1:
            metadata.cp_swa_local_page_start_offsets = None
            metadata.cp_swa_local_page_token_counts = None
            offsets = torch.arange(window_tokens, device=slots.device).unsqueeze(0)
            cache_lens_long = cache_seqlens.to(torch.long)
            start = torch.clamp(cache_lens_long - window_tokens, min=0)
            cols = start.unsqueeze(1) + offsets
            valid = cols < cache_lens_long.unsqueeze(1)
            valid = valid & (cols < table_len)
            safe_cols = torch.where(valid, cols, torch.zeros_like(cols))
            row_indices = torch.arange(batch_size, device=slots.device).unsqueeze(1)
            window_slots = slots[row_indices, safe_cols]
            local_valid = valid & window_slots.ne(0)
            local_cache_seqlens = local_valid.sum(dim=1, dtype=torch.int32)

            if window_tokens > 0:
                compact_cols = torch.cumsum(local_valid.to(torch.int32), dim=1) - 1
                scatter_cols = torch.where(
                    local_valid,
                    compact_cols,
                    torch.zeros_like(compact_cols),
                ).to(torch.long)
                scatter_slots = torch.where(
                    local_valid, window_slots, torch.zeros_like(window_slots)
                )
                local_page_table.scatter_reduce_(
                    dim=1,
                    index=scatter_cols,
                    src=scatter_slots,
                    reduce="amax",
                    include_self=True,
                )
        else:
            if out_page_start_offsets is None:
                local_page_start_offsets = torch.empty_like(local_page_table)
            else:
                if out_page_start_offsets.shape[1] < local_page_table_cap:
                    raise RuntimeError(
                        "CP sharded-KV decode SWA page-offset buffer is too small: "
                        f"{out_page_start_offsets.shape[1]} < {local_page_table_cap}"
                    )
                local_page_start_offsets = out_page_start_offsets[
                    :, :local_page_table_cap
                ]
            if out_page_token_counts is None:
                local_page_token_counts = torch.empty_like(local_page_table)
            else:
                if out_page_token_counts.shape[1] < local_page_table_cap:
                    raise RuntimeError(
                        "CP sharded-KV decode SWA page-count buffer is too small: "
                        f"{out_page_token_counts.shape[1]} < {local_page_table_cap}"
                    )
                local_page_token_counts = out_page_token_counts[
                    :, :local_page_table_cap
                ]
            local_page_start_offsets.zero_()
            local_page_token_counts.zero_()

            page_offsets = torch.arange(
                local_page_table_cap, device=slots.device
            ).unsqueeze(0)
            cache_lens_long = cache_seqlens.to(torch.long)
            window_start = torch.clamp(cache_lens_long - window_tokens, min=0)
            first_page = torch.div(
                window_start, self.page_size, rounding_mode="floor"
            )
            page_cols = first_page.unsqueeze(1) + page_offsets
            valid_page = page_cols < table_len
            page_token_start = page_cols * self.page_size
            valid_start = torch.maximum(page_token_start, window_start.unsqueeze(1))
            valid_end = torch.minimum(
                page_token_start + self.page_size, cache_lens_long.unsqueeze(1)
            )
            page_token_counts = torch.clamp(
                valid_end - valid_start, min=0, max=self.page_size
            ).to(torch.int32)
            page_start_offsets = torch.clamp(
                valid_start - page_token_start, min=0, max=self.page_size - 1
            ).to(torch.int32)
            safe_cols = torch.where(valid_page, page_cols, torch.zeros_like(page_cols))
            row_indices = torch.arange(batch_size, device=slots.device).unsqueeze(1)
            window_slots = slots[row_indices, safe_cols]
            local_valid = valid_page & page_token_counts.gt(0) & window_slots.ne(0)
            valid_counts = torch.where(
                local_valid, page_token_counts, torch.zeros_like(page_token_counts)
            )
            local_cache_seqlens = valid_counts.sum(dim=1, dtype=torch.int32)

            if local_page_table_cap > 0:
                compact_cols = torch.cumsum(local_valid.to(torch.int32), dim=1) - 1
                scatter_cols = torch.where(
                    local_valid,
                    compact_cols,
                    torch.zeros_like(compact_cols),
                ).to(torch.long)
                scatter_slots = torch.where(
                    local_valid, window_slots, torch.zeros_like(window_slots)
                )
                scatter_offsets = torch.where(
                    local_valid,
                    page_start_offsets,
                    torch.zeros_like(page_start_offsets),
                )
                scatter_counts = torch.where(
                    local_valid,
                    page_token_counts,
                    torch.zeros_like(page_token_counts),
                )
                local_page_table.scatter_reduce_(
                    dim=1,
                    index=scatter_cols,
                    src=scatter_slots,
                    reduce="amax",
                    include_self=True,
                )
                local_page_start_offsets.scatter_reduce_(
                    dim=1,
                    index=scatter_cols,
                    src=scatter_offsets,
                    reduce="amax",
                    include_self=True,
                )
                local_page_token_counts.scatter_reduce_(
                    dim=1,
                    index=scatter_cols,
                    src=scatter_counts,
                    reduce="amax",
                    include_self=True,
                )
            metadata.cp_swa_local_page_start_offsets = local_page_start_offsets
            metadata.cp_swa_local_page_token_counts = local_page_token_counts

        if out_cache_seqlens is None:
            metadata.cp_swa_local_cache_seqlens_int32 = local_cache_seqlens
        else:
            out_cache_seqlens.copy_(local_cache_seqlens)
            metadata.cp_swa_local_cache_seqlens_int32 = out_cache_seqlens
        metadata.cp_swa_local_page_table = local_page_table

    def _set_cuda_graph_sharded_kv_decode_metadata(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
        page_table: Optional[torch.Tensor] = None,
    ) -> None:
        local_page_table = page_table if page_table is not None else metadata.page_table
        local_num_pages = local_page_table.shape[1]
        self._set_sharded_kv_decode_metadata(
            metadata,
            local_page_table,
            metadata.cache_seqlens_int32,
            out_page_table=self.decode_cuda_graph_metadata["cp_local_page_table"][
                :bs, :local_num_pages
            ],
            out_cache_seqlens=self.decode_cuda_graph_metadata["cp_local_cache_seqlens"][
                :bs
            ],
        )

    def _set_cuda_graph_sharded_kv_decode_swa_metadata(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
        page_table: Optional[torch.Tensor] = None,
    ) -> None:
        if "cp_swa_local_page_table" not in self.decode_cuda_graph_metadata:
            return
        local_page_table = (
            page_table if page_table is not None else metadata.swa_page_table
        )
        self._set_sharded_kv_decode_swa_metadata(
            metadata,
            local_page_table,
            metadata.cache_seqlens_int32,
            out_page_table=self.decode_cuda_graph_metadata["cp_swa_local_page_table"][:bs],
            out_cache_seqlens=self.decode_cuda_graph_metadata[
                "cp_swa_local_cache_seqlens"
            ][:bs],
            out_page_start_offsets=self.decode_cuda_graph_metadata.get(
                "cp_swa_local_page_start_offsets"
            )[:bs]
            if "cp_swa_local_page_start_offsets" in self.decode_cuda_graph_metadata
            else None,
            out_page_token_counts=self.decode_cuda_graph_metadata.get(
                "cp_swa_local_page_token_counts"
            )[:bs]
            if "cp_swa_local_page_token_counts" in self.decode_cuda_graph_metadata
            else None,
        )

    def _build_attn_cp_local_merge_workspace(
        self, max_bs: int
    ) -> dict[str, torch.Tensor]:
        cp_world_size = self.attn_cp_size
        local_q_heads = self.num_attention_heads
        full_q_heads = local_q_heads * cp_world_size
        head_dim = self.head_dim
        dtype = self.kv_cache_dtype
        device = self.device

        local_merge_workspace = {
            "q_gather": torch.empty(
                cp_world_size * max_bs,
                local_q_heads,
                head_dim,
                dtype=dtype,
                device=device,
            ),
            "q_full": torch.empty(
                max_bs, full_q_heads, head_dim, dtype=dtype, device=device
            ),
            "local_o_full": torch.empty(
                max_bs, full_q_heads, head_dim, dtype=dtype, device=device
            ),
            "local_lse_full": torch.empty(
                max_bs, full_q_heads, dtype=torch.float32, device=device
            ),
            "o_gather": torch.empty(
                cp_world_size * max_bs,
                full_q_heads,
                head_dim,
                dtype=dtype,
                device=device,
            ),
            "lse_gather": torch.empty(
                cp_world_size * max_bs,
                full_q_heads,
                dtype=torch.float32,
                device=device,
            ),
            "merge_current_o": torch.empty(
                max_bs, local_q_heads, head_dim, dtype=dtype, device=device
            ),
            "merge_current_lse": torch.empty(
                max_bs, local_q_heads, dtype=torch.float32, device=device
            ),
            "merge_next_o": torch.empty(
                max_bs, local_q_heads, head_dim, dtype=dtype, device=device
            ),
            "merge_next_lse": torch.empty(
                max_bs, local_q_heads, dtype=torch.float32, device=device
            ),
            "merge_tmp_o": torch.empty(
                max_bs, local_q_heads, head_dim, dtype=dtype, device=device
            ),
            "merge_tmp_lse": torch.empty(
                max_bs, local_q_heads, dtype=torch.float32, device=device
            ),
            "sinks_gather": torch.empty(
                cp_world_size * local_q_heads, dtype=dtype, device=device
            ),
            "sinks_gather_f32": torch.empty(
                cp_world_size * local_q_heads, dtype=torch.float32, device=device
            ),
            "sinks_disabled_full": torch.full(
                (full_q_heads,), -float("inf"), dtype=dtype, device=device
            ),
            "sinks_disabled_local": torch.full(
                (local_q_heads,), -float("inf"), dtype=dtype, device=device
            ),
            "lse_disabled_full": torch.full(
                (full_q_heads,), -float("inf"), dtype=torch.float32, device=device
            ),
            "lse_disabled_local": torch.full(
                (local_q_heads,), -float("inf"), dtype=torch.float32, device=device
            ),
        }

        if (
            self.enable_attn_cp_decode_cp2_q_p2p
            or self.enable_attn_cp_decode_cp2_fused_q_fa
        ):
            local_merge_workspace["q_peer"] = torch.empty(
                max_bs, local_q_heads, head_dim, dtype=dtype, device=device
            )

        if self.enable_attn_cp_decode_cp2_fused_q_fa:
            max_splits = attncp_cp2_fused_q_fa_max_splits(
                self.cuda_graph_max_seq_len
            )
            local_merge_workspace["fused_q_fa_split_o"] = torch.empty(
                max_splits,
                max_bs,
                full_q_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )
            local_merge_workspace["fused_q_fa_split_lse"] = torch.empty(
                max_splits,
                max_bs,
                full_q_heads,
                dtype=torch.float32,
                device=device,
            )

        if self.enable_attn_cp_decode_cp2_olse_p2p:
            local_merge_workspace.update(
                {
                    "o_send": torch.empty(
                        max_bs, local_q_heads, head_dim, dtype=dtype, device=device
                    ),
                    "lse_send": torch.empty(
                        max_bs, local_q_heads, dtype=torch.float32, device=device
                    ),
                    "o_recv": torch.empty(
                        max_bs, local_q_heads, head_dim, dtype=dtype, device=device
                    ),
                    "lse_recv": torch.empty(
                        max_bs, local_q_heads, dtype=torch.float32, device=device
                    ),
                }
            )

        return local_merge_workspace

    def _init_attn_cp_local_merge_cuda_graph_state(self, max_bs: int) -> None:
        self.decode_cuda_graph_metadata["attncp_local_merge"] = (
            self._build_attn_cp_local_merge_workspace(max_bs)
        )

    def _attncp_local_merge_workspace(
        self,
    ) -> Optional[dict[str, torch.Tensor]]:
        return self.decode_cuda_graph_metadata.get("attncp_local_merge")

    @staticmethod
    def _copy_fa_lse(
        out: torch.Tensor,
        lse: torch.Tensor,
        batch_size: int,
        num_heads: int,
    ) -> torch.Tensor:
        out = out[:batch_size, :num_heads]
        if lse.shape == (num_heads, batch_size):
            out.copy_(lse.T)
        elif lse.shape == (batch_size, num_heads):
            out.copy_(lse)
        elif lse.shape == (batch_size, num_heads, 1):
            out.copy_(lse[:, :, 0])
        elif lse.shape == (num_heads, batch_size, 1):
            out.copy_(lse[:, :, 0].T)
        else:
            raise RuntimeError(f"Unexpected FA LSE shape: {tuple(lse.shape)}")
        return out

    @classmethod
    def _normalize_fa_lse(
        cls,
        lse: torch.Tensor,
        batch_size: int,
        num_heads: int,
    ) -> torch.Tensor:
        out = torch.empty(
            (batch_size, num_heads), dtype=torch.float32, device=lse.device
        )
        return cls._copy_fa_lse(out, lse, batch_size, num_heads)

    def _attncp_exchange_cp2_q_peer(
        self,
        q_local: torch.Tensor,
        bufs: dict[str, torch.Tensor],
        batch_size: int,
        local_q_heads: int,
    ) -> Optional[torch.Tensor]:
        cp_group = get_sharded_kv_cp_group()
        if cp_group.world_size != 2:
            return None
        if "q_peer" not in bufs or not q_local.is_contiguous():
            return None

        cp_rank = cp_group.rank_in_group
        peer_rank = 1 - cp_rank
        q_peer = bufs["q_peer"][:batch_size, :local_q_heads, :]
        if self.enable_attn_cp_decode_cp2_q_p2p:
            pynccl_comm = cp_group.pynccl_comm
            if pynccl_comm is None or pynccl_comm.disabled:
                return None
            with pynccl_comm.change_state(enable=True):
                pynccl_comm.group_start()
                pynccl_comm.recv(q_peer, peer_rank)
                pynccl_comm.send(q_local, peer_rank)
                pynccl_comm.group_end()
            return q_peer

        q_gather = bufs.get("q_gather")
        if q_gather is None:
            return None
        head_dim = q_local.shape[2]
        q_gather = q_gather[: cp_group.world_size * batch_size]
        cp_group.all_gather_into_tensor(q_gather, q_local.contiguous())
        q_view = q_gather.view(
            cp_group.world_size, batch_size, local_q_heads, head_dim
        )
        q_peer.copy_(q_view[peer_rank])
        return q_peer

    @staticmethod
    def _require_attncp_fused_q_fa_shape(
        *,
        local_q_heads: int,
        local_kv_heads: int,
        cp_world_size: int,
    ) -> None:
        if attncp_cp2_fused_q_fa_supports_shape(
            local_q_heads,
            local_kv_heads,
            cp_world_size=cp_world_size,
        ):
            return
        raise RuntimeError(
            "Requested --attn-cp-decode-fused-q-fa does not support the "
            "runtime attention head shape: "
            f"local_q_heads={local_q_heads}, "
            f"local_kv_heads={local_kv_heads}, cp_size={cp_world_size}."
        )

    def _attncp_try_fused_q_fa_decode(
        self,
        q_local: torch.Tensor,
        bufs: dict[str, torch.Tensor],
        layer: RadixAttention,
        metadata: FlashAttentionMetadata,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        batch_size: int,
        local_q_heads: int,
        head_dim: int,
        local_page_table: torch.Tensor,
        local_cache_seqlens: torch.Tensor,
        local_page_start_offsets: Optional[torch.Tensor],
        local_page_token_counts: Optional[torch.Tensor],
        attn_window_size: tuple[int, int],
        causal: bool,
        local_kwargs: dict,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Try the experimental Triton provider for CP2 local attention.

        This is the replacement boundary for decode-side Q exchange plus local
        attention over the resident KV shard. The implementation below is a
        KV-stationary Triton path and does not require rebuilding sgl-kernel.
        """
        seq_cap = int(local_page_table.shape[1]) * int(self.page_size)

        if not self.enable_attn_cp_decode_cp2_fused_q_fa:
            return None
        if (
            getattr(metadata, "requires_exact_logprob", False)
            and not self.attn_cp_decode_cp2_fused_q_fa_allow_logprob
        ):
            return None
        cp_group = get_sharded_kv_cp_group()
        self._require_attncp_fused_q_fa_shape(
            local_q_heads=local_q_heads,
            local_kv_heads=int(key_cache.shape[2]),
            cp_world_size=cp_group.world_size,
        )
        has_page_mask = (
            local_page_start_offsets is not None or local_page_token_counts is not None
        )
        if self.page_size not in (1, 16) or layer.v_head_dim != layer.head_dim:
            return None
        if has_page_mask and (
            self.page_size != 16
            or local_page_start_offsets is None
            or local_page_token_counts is None
        ):
            return None
        if metadata.max_seq_len_q != 1:
            return None
        # The compact SWA page table is intentionally short. Keep the long-seq
        # threshold for the normal path, but do not let it force masked SWA
        # page-size=16 back to the dense fallback.
        if (
            not has_page_mask
            and seq_cap < self.attn_cp_decode_cp2_fused_q_fa_min_seq_cap
        ):
            return None
        window_left = attn_window_size[0] if attn_window_size is not None else -1
        window_right = attn_window_size[1] if attn_window_size is not None else -1
        if window_right not in (-1, 0):
            return None
        if not causal:
            return None
        if any(key != "sinks" for key in local_kwargs):
            return None
        if (
            not key_cache.is_contiguous()
            or not value_cache.is_contiguous()
            or not local_cache_seqlens.is_contiguous()
        ):
            return None

        q_peer = self._attncp_exchange_cp2_q_peer(
            q_local, bufs, batch_size, local_q_heads
        )
        if q_peer is None:
            return None

        full_q_heads = local_q_heads * cp_group.world_size
        local_o_full = bufs["local_o_full"][:batch_size, :full_q_heads, :]
        local_lse_full = bufs["local_lse_full"][:batch_size, :full_q_heads]

        attncp_cp2_fused_q_fa_decode(
            q_local,
            q_peer,
            key_cache,
            value_cache,
            local_page_table,
            local_cache_seqlens,
            local_o_full,
            local_lse_full,
            cp_rank=cp_group.rank_in_group,
            softmax_scale=layer.scaling,
            softcap=layer.logit_cap,
            window_left=window_left,
            sinks=local_kwargs.get("sinks"),
            page_size=self.page_size,
            page_start_offsets=local_page_start_offsets,
            page_token_counts=local_page_token_counts,
            split_o=bufs.get("fused_q_fa_split_o"),
            split_lse=bufs.get("fused_q_fa_split_lse"),
            max_splits=bufs["fused_q_fa_split_o"].shape[0],
        )
        return local_o_full, local_lse_full

    def _attncp_gather_full_q(
        self,
        q_local: torch.Tensor,
        bufs: dict[str, torch.Tensor],
        batch_size: int,
        local_q_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        cp_group = get_sharded_kv_cp_group()
        cp_world_size = cp_group.world_size
        if (
            self.enable_attn_cp_decode_cp2_q_p2p
            and cp_world_size == 2
            and q_local.is_contiguous()
        ):
            pynccl_comm = cp_group.pynccl_comm
            if pynccl_comm is not None and not pynccl_comm.disabled:
                cp_rank = cp_group.rank_in_group
                peer_rank = 1 - cp_rank
                head_start = cp_rank * local_q_heads
                peer_head_start = peer_rank * local_q_heads
                q_peer = bufs["q_peer"][:batch_size, :local_q_heads, :]
                q_full = bufs["q_full"][
                    :batch_size, : local_q_heads * cp_world_size, :
                ]
                with pynccl_comm.change_state(enable=True):
                    pynccl_comm.group_start()
                    pynccl_comm.recv(q_peer, peer_rank)
                    pynccl_comm.send(q_local, peer_rank)
                    pynccl_comm.group_end()
                q_full[:, head_start : head_start + local_q_heads, :].copy_(q_local)
                q_full[
                    :, peer_head_start : peer_head_start + local_q_heads, :
                ].copy_(q_peer)
                return q_full

        q_gather = bufs["q_gather"][: cp_world_size * batch_size]
        cp_group.all_gather_into_tensor(q_gather, q_local.contiguous())
        q_view = q_gather.view(cp_world_size, batch_size, local_q_heads, head_dim)
        q_full = bufs["q_full"][:batch_size, : local_q_heads * cp_world_size, :]
        q_full.view(batch_size, cp_world_size, local_q_heads, head_dim).copy_(
            q_view.permute(1, 0, 2, 3)
        )
        return q_full

    def _attncp_get_full_sinks(
        self,
        layer: RadixAttention,
        sinks: torch.Tensor,
        bufs: Optional[dict[str, torch.Tensor]],
        local_q_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_group = get_sharded_kv_cp_group()
        full_q_heads = cp_group.world_size * local_q_heads
        cache_key = (int(layer.layer_id), int(sinks.numel()), sinks.dtype, sinks.device)
        cached = self.attncp_full_sinks_cache.get(cache_key)
        if cached is not None:
            return cached

        if torch.cuda.is_current_stream_capturing():
            if bufs is None:
                raise RuntimeError(
                    "CP sharded-KV decode cannot initialize full attention sinks "
                    "cache during CUDA graph capture without workspace buffers"
                )
            sinks_gather = bufs["sinks_gather"][:full_q_heads]
            cp_group.all_gather_into_tensor(sinks_gather, sinks.contiguous())
            sinks_gather_f32 = bufs["sinks_gather_f32"][:full_q_heads]
            sinks_gather_f32.copy_(sinks_gather)
            return sinks_gather, sinks_gather_f32

        full_sinks = sinks.new_empty(full_q_heads)
        full_sinks_f32 = torch.empty(
            full_q_heads, dtype=torch.float32, device=sinks.device
        )
        cp_group.all_gather_into_tensor(full_sinks, sinks.contiguous())
        full_sinks_f32.copy_(full_sinks)
        cached = (full_sinks, full_sinks_f32)
        self.attncp_full_sinks_cache[cache_key] = cached
        return cached

    def _attncp_apply_empty_local_kv(
        self,
        local_o: torch.Tensor,
        local_lse: torch.Tensor,
        local_cache_seqlens: torch.Tensor,
        empty_lse: torch.Tensor,
    ) -> None:
        empty_local_kv = local_cache_seqlens.to(torch.long).eq(0)
        local_o.masked_fill_(empty_local_kv.view(-1, 1, 1), 0)
        torch.where(
            empty_local_kv.view(-1, 1),
            empty_lse.view(1, -1).expand_as(local_lse),
            local_lse,
            out=local_lse,
        )

    def _attncp_exchange_cp2_local_head_slice(
        self,
        bufs: dict[str, torch.Tensor],
        local_o_full: torch.Tensor,
        local_lse_full: torch.Tensor,
        batch_size: int,
        local_q_heads: int,
        head_dim: int,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        cp_group = get_sharded_kv_cp_group()
        if not self.enable_attn_cp_decode_cp2_olse_p2p:
            return None
        if cp_group.world_size != 2:
            return None

        pynccl_comm = cp_group.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            return None

        cp_rank = cp_group.rank_in_group
        peer_rank = 1 - cp_rank
        peer_head_start = peer_rank * local_q_heads
        peer_head_end = peer_head_start + local_q_heads

        send_o = bufs["o_send"][:batch_size, :local_q_heads, :]
        send_lse = bufs["lse_send"][:batch_size, :local_q_heads]
        recv_o = bufs["o_recv"][:batch_size, :local_q_heads, :]
        recv_lse = bufs["lse_recv"][:batch_size, :local_q_heads]
        if self.enable_attn_cp_decode_cp2_fused_merge:
            attncp_cp2_pack_local_head_slice(
                local_o_full,
                local_lse_full,
                send_o,
                send_lse,
                full_q_heads=local_q_heads * cp_group.world_size,
                local_q_heads=local_q_heads,
                head_dim=head_dim,
                head_start=peer_head_start,
            )
        else:
            send_o.copy_(local_o_full[:, peer_head_start:peer_head_end, :])
            send_lse.copy_(local_lse_full[:, peer_head_start:peer_head_end])

        with pynccl_comm.change_state(enable=True):
            pynccl_comm.group_start()
            pynccl_comm.recv(recv_o, peer_rank)
            pynccl_comm.send(send_o, peer_rank)
            pynccl_comm.recv(recv_lse, peer_rank)
            pynccl_comm.send(send_lse, peer_rank)
            pynccl_comm.group_end()
        return recv_o, recv_lse

    def _attncp_merge_cp2_local_head_slice(
        self,
        bufs: dict[str, torch.Tensor],
        local_o_full: torch.Tensor,
        local_lse_full: torch.Tensor,
        remote_o: torch.Tensor,
        remote_lse: torch.Tensor,
        batch_size: int,
        local_q_heads: int,
        head_dim: int,
        final_o: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cp_group = get_sharded_kv_cp_group()
        head_start = cp_group.rank_in_group * local_q_heads
        head_end = head_start + local_q_heads
        tmp_o = bufs["merge_tmp_o"][:batch_size, :local_q_heads, :]
        out_o = (
            final_o
            if final_o is not None
            and tuple(final_o.shape) == (batch_size, local_q_heads, head_dim)
            and final_o.is_contiguous()
            else tmp_o
        )
        if self.enable_attn_cp_decode_cp2_fused_merge:
            return attncp_cp2_merge_local_remote_head_slice(
                local_o_full,
                local_lse_full,
                remote_o,
                remote_lse,
                out_o,
                full_q_heads=local_q_heads * cp_group.world_size,
                local_q_heads=local_q_heads,
                head_dim=head_dim,
                head_start=head_start,
                local_is_cp0=cp_group.rank_in_group == 0,
            )

        local_o = bufs["merge_current_o"][:batch_size, :local_q_heads, :]
        local_lse = bufs["merge_current_lse"][:batch_size, :local_q_heads]
        tmp_lse = bufs["merge_tmp_lse"][:batch_size, :local_q_heads]
        local_o.copy_(local_o_full[:, head_start:head_end, :])
        local_lse.copy_(local_lse_full[:, head_start:head_end])

        if cp_group.rank_in_group == 0:
            merge_state_v2(local_o, local_lse, remote_o, remote_lse, out_o, tmp_lse)
        else:
            merge_state_v2(remote_o, remote_lse, local_o, local_lse, out_o, tmp_lse)
        return out_o

    def _attncp_merge_local_head_slice(
        self,
        bufs: dict[str, torch.Tensor],
        batch_size: int,
        local_q_heads: int,
        head_dim: int,
        final_o: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cp_group = get_sharded_kv_cp_group()
        cp_world_size = cp_group.world_size
        head_start = cp_group.rank_in_group * local_q_heads
        head_end = head_start + local_q_heads

        gathered_o_flat = bufs["o_gather"][
            : cp_world_size * batch_size, : local_q_heads * cp_world_size, :
        ]
        gathered_lse_flat = bufs["lse_gather"][
            : cp_world_size * batch_size, : local_q_heads * cp_world_size
        ]
        final_o = (
            final_o
            if final_o is not None
            and tuple(final_o.shape) == (batch_size, local_q_heads, head_dim)
            and final_o.is_contiguous()
            else None
        )

        if self.enable_attn_cp_decode_cp2_fused_merge and cp_world_size == 2:
            out_o = (
                final_o
                if final_o is not None
                else bufs["merge_tmp_o"][:batch_size, :local_q_heads, :]
            )
            return attncp_cp2_merge_local_head_slice(
                gathered_o_flat,
                gathered_lse_flat,
                out_o,
                batch_size=batch_size,
                full_q_heads=local_q_heads * cp_world_size,
                local_q_heads=local_q_heads,
                head_dim=head_dim,
                head_start=head_start,
            )

        gathered_o = gathered_o_flat.view(
            cp_world_size, batch_size, local_q_heads * cp_world_size, head_dim
        )
        gathered_lse = gathered_lse_flat.view(
            cp_world_size, batch_size, local_q_heads * cp_world_size
        )

        current_o = bufs["merge_current_o"][:batch_size, :local_q_heads, :]
        current_lse = bufs["merge_current_lse"][:batch_size, :local_q_heads]
        next_o = bufs["merge_next_o"][:batch_size, :local_q_heads, :]
        next_lse = bufs["merge_next_lse"][:batch_size, :local_q_heads]
        tmp_o = bufs["merge_tmp_o"][:batch_size, :local_q_heads, :]
        tmp_lse = bufs["merge_tmp_lse"][:batch_size, :local_q_heads]

        current_o.copy_(gathered_o[0, :, head_start:head_end, :])
        current_lse.copy_(gathered_lse[0, :, head_start:head_end])
        merged_o = current_o
        merged_lse = current_lse
        for cp_idx in range(1, cp_world_size):
            next_o.copy_(gathered_o[cp_idx, :, head_start:head_end, :])
            next_lse.copy_(gathered_lse[cp_idx, :, head_start:head_end])
            out_o = (
                final_o
                if final_o is not None and cp_idx == cp_world_size - 1
                else tmp_o if merged_o is current_o else current_o
            )
            out_lse = tmp_lse if merged_lse is current_lse else current_lse
            merge_state_v2(merged_o, merged_lse, next_o, next_lse, out_o, out_lse)
            merged_o = out_o
            merged_lse = out_lse
        return merged_o

    def _flash_attn_sharded_kv_local_merge_workspace(
        self,
        q_local: torch.Tensor,
        layer: RadixAttention,
        metadata: FlashAttentionMetadata,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        window_size: tuple[int, int],
        causal: bool,
        kwargs: dict,
        out: Optional[torch.Tensor] = None,
        local_page_table: Optional[torch.Tensor] = None,
        local_cache_seqlens: Optional[torch.Tensor] = None,
        local_page_start_offsets: Optional[torch.Tensor] = None,
        local_page_token_counts: Optional[torch.Tensor] = None,
        local_window_size: Optional[tuple[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        bufs = self._attncp_local_merge_workspace()
        if bufs is None:
            return None
        if layer.v_head_dim != layer.head_dim:
            return None

        if local_page_table is None:
            local_page_table = metadata.cp_local_page_table
        if local_cache_seqlens is None:
            local_cache_seqlens = metadata.cp_local_cache_seqlens_int32
        if local_page_table is None or local_cache_seqlens is None:
            raise RuntimeError("CP sharded-KV decode missing local shard metadata")
        attn_window_size = window_size if local_window_size is None else local_window_size

        cp_group = get_sharded_kv_cp_group()
        cp_world_size = cp_group.world_size
        batch_size = q_local.shape[0]
        if batch_size > bufs["q_full"].shape[0]:
            return None
        local_q_heads = q_local.shape[1]
        head_dim = q_local.shape[2]
        full_q_heads = local_q_heads * cp_world_size

        sinks_gather = None
        sinks_gather_f32 = None
        if "sinks" in kwargs:
            sinks_gather, sinks_gather_f32 = self._attncp_get_full_sinks(
                layer, kwargs["sinks"], bufs, local_q_heads
            )

        local_kwargs = dict(kwargs)
        local_kwargs.pop("sinks", None)
        if sinks_gather is not None:
            if cp_group.rank_in_group == 0:
                local_kwargs["sinks"] = sinks_gather[:full_q_heads]
            else:
                local_kwargs["sinks"] = bufs["sinks_disabled_full"][:full_q_heads]

        local_o_full = bufs["local_o_full"][:batch_size, :full_q_heads, :]
        local_lse_full = bufs["local_lse_full"][:batch_size, :full_q_heads]
        fused_result = self._attncp_try_fused_q_fa_decode(
            q_local,
            bufs,
            layer,
            metadata,
            key_cache,
            value_cache,
            batch_size=batch_size,
            local_q_heads=local_q_heads,
            head_dim=head_dim,
            local_page_table=local_page_table,
            local_cache_seqlens=local_cache_seqlens,
            local_page_start_offsets=local_page_start_offsets,
            local_page_token_counts=local_page_token_counts,
            attn_window_size=attn_window_size,
            causal=causal,
            local_kwargs=local_kwargs,
        )
        if fused_result is None:
            if (
                local_page_start_offsets is not None
                or local_page_token_counts is not None
            ):
                return None
            q_full = self._attncp_gather_full_q(
                q_local,
                bufs,
                batch_size,
                local_q_heads,
                head_dim,
            )
            result = self.flash_attn_with_kvcache(
                q=q_full,
                k_cache=key_cache,
                v_cache=value_cache,
                page_table=local_page_table,
                cache_seqlens=local_cache_seqlens,
                cu_seqlens_q=metadata.cu_seqlens_q,
                max_seqlen_q=metadata.max_seq_len_q,
                softmax_scale=layer.scaling,
                causal=causal,
                window_size=attn_window_size,
                softcap=layer.logit_cap,
                num_splits=self.num_splits,
                return_softmax_lse=True,
                out=local_o_full,
                **local_kwargs,
            )
            local_o, local_lse = result[:2]
            local_lse = self._copy_fa_lse(
                local_lse_full,
                local_lse,
                batch_size,
                full_q_heads,
            )
        else:
            local_o, local_lse = fused_result
        empty_lse = (
            sinks_gather_f32[:full_q_heads]
            if sinks_gather is not None and cp_group.rank_in_group == 0
            else bufs["lse_disabled_full"][:full_q_heads]
        )
        self._attncp_apply_empty_local_kv(
            local_o, local_lse, local_cache_seqlens, empty_lse
        )

        o_gather = bufs["o_gather"][: cp_world_size * batch_size, :full_q_heads, :]
        lse_gather = bufs["lse_gather"][
            : cp_world_size * batch_size, :full_q_heads
        ]
        cp2_exchange = self._attncp_exchange_cp2_local_head_slice(
            bufs,
            local_o_full,
            local_lse_full,
            batch_size,
            local_q_heads,
            head_dim,
        )
        if cp2_exchange is None:
            cp_group.all_gather_coalesced(
                [
                    (o_gather, local_o_full.contiguous()),
                    (lse_gather, local_lse_full.contiguous()),
                ]
            )
        if cp2_exchange is None:
            merged_o = self._attncp_merge_local_head_slice(
                bufs, batch_size, local_q_heads, head_dim, final_o=out
            )
        else:
            remote_o, remote_lse = cp2_exchange
            merged_o = self._attncp_merge_cp2_local_head_slice(
                bufs,
                local_o_full,
                local_lse_full,
                remote_o,
                remote_lse,
                batch_size,
                local_q_heads,
                head_dim,
                final_o=out,
            )
        if out is not None and merged_o is not out:
            out.copy_(merged_o)
            merged_o = out
        return merged_o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _flash_attn_sharded_kv_dense(
        self,
        q_local: torch.Tensor,
        layer: RadixAttention,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        *,
        window_size: tuple[int, int],
        causal: bool,
        kwargs: dict,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        scheduler_metadata: Optional[torch.Tensor] = None,
        prefill_plan: Optional[CPShardedKVPrefillPlan] = None,
        page_table_is_physical: bool = False,
    ) -> torch.Tensor:
        window_left = window_size[0] if window_size is not None else -1
        if (
            cu_seqlens_k is None
            and max_seqlen_q == 1
            and 0 <= window_left < self.max_context_len
        ):
            dense_k, dense_v, dense_page_table, dense_cache_seqlens = (
                self._gather_sharded_kv_dense_decode_window(
                    page_table,
                    cache_seqlens,
                    key_cache,
                    value_cache,
                    window_left,
                    page_table_is_physical=page_table_is_physical,
                )
            )
        else:
            split_gather = None
            if (
                cu_seqlens_k is not None
                and (window_size is None or window_left < 0)
                and prefill_plan is not None
            ):
                split_gather = self._gather_sharded_kv_dense_prefill_split(
                    prefill_plan,
                    key_cache,
                    value_cache,
                )
            if split_gather is None:
                dense_k, dense_v, dense_page_table = self._gather_sharded_kv_dense(
                    page_table,
                    cache_seqlens,
                    key_cache,
                    value_cache,
                    page_table_is_physical=page_table_is_physical,
                )
                dense_cache_seqlens = cache_seqlens
            else:
                (
                    dense_k,
                    dense_v,
                    dense_page_table,
                    dense_cache_seqlens,
                ) = split_gather

        dense_page_size = self.page_size if dense_k.dim() == 5 else 1
        if dense_page_size != self.page_size:
            # Some translated fallbacks, such as compact SWA windows, expose
            # temporary K/V as page_size=1. Scheduler metadata precomputed for
            # the original paged cache is not valid for that page table.
            scheduler_metadata = None

        dense_kwargs = dict(kwargs)
        dense_kwargs.pop("ver", None)
        o = self.flash_attn_with_kvcache(
            q=q_local,
            k_cache=dense_k.view(
                -1, dense_page_size, layer.tp_k_head_num, layer.head_dim
            ).contiguous(),
            v_cache=dense_v.view(
                -1, dense_page_size, layer.tp_v_head_num, layer.v_head_dim
            ).contiguous(),
            page_table=dense_page_table,
            cache_seqlens=dense_cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k_new=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=window_size,
            softcap=layer.logit_cap,
            num_splits=self.num_splits,
            out=out,
            scheduler_metadata=scheduler_metadata,
            **dense_kwargs,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _flash_attn_sharded_kv_local_merge(
        self,
        q_local: torch.Tensor,
        layer: RadixAttention,
        metadata: FlashAttentionMetadata,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        window_size: tuple[int, int],
        causal: bool,
        kwargs: dict,
        out: Optional[torch.Tensor] = None,
        local_page_table: Optional[torch.Tensor] = None,
        local_cache_seqlens: Optional[torch.Tensor] = None,
        local_page_start_offsets: Optional[torch.Tensor] = None,
        local_page_token_counts: Optional[torch.Tensor] = None,
        local_window_size: Optional[tuple[int, int]] = None,
    ) -> Optional[torch.Tensor]:
        workspace_o = self._flash_attn_sharded_kv_local_merge_workspace(
            q_local,
            layer,
            metadata,
            key_cache,
            value_cache,
            window_size=window_size,
            causal=causal,
            kwargs=kwargs,
            out=out,
            local_page_table=local_page_table,
            local_cache_seqlens=local_cache_seqlens,
            local_page_start_offsets=local_page_start_offsets,
            local_page_token_counts=local_page_token_counts,
            local_window_size=local_window_size,
        )
        if workspace_o is not None:
            return workspace_o
        if local_page_start_offsets is not None or local_page_token_counts is not None:
            return None

        if local_page_table is None:
            local_page_table = metadata.cp_local_page_table
        if local_cache_seqlens is None:
            local_cache_seqlens = metadata.cp_local_cache_seqlens_int32
        if local_page_table is None or local_cache_seqlens is None:
            raise RuntimeError("CP sharded-KV decode missing local shard metadata")
        attn_window_size = window_size if local_window_size is None else local_window_size

        cp_group = get_sharded_kv_cp_group()
        cp_world_size = cp_group.world_size
        local_q_heads = q_local.shape[1]
        local_head_start = cp_group.rank_in_group * local_q_heads
        q_for_attn = q_local
        local_kwargs = kwargs
        if cp_world_size > 1:
            # CP ranks in a sharded-KV group own different Q-head shards for
            # the same KV-head group. Gather Q heads so every rank computes its
            # local KV shard contribution for the full GQA group.
            q_for_attn = cp_group.all_gather(q_local.contiguous(), dim=1)

        if cp_world_size > 1 and "sinks" in kwargs:
            local_kwargs = dict(kwargs)
            # Sink is a per-Q-head denominator term. Include it in exactly
            # one CP shard before merging softmax states. Put it on CP rank
            # 0 so short sequences whose KV all lives on rank 0 do not rely
            # on sink-only empty-KV rows from the other ranks.
            full_sinks, _ = self._attncp_get_full_sinks(
                layer, kwargs["sinks"], None, local_q_heads
            )
            if cp_group.rank_in_group == 0:
                local_kwargs["sinks"] = full_sinks
            else:
                local_kwargs["sinks"] = kwargs["sinks"].new_full(
                    (q_for_attn.shape[1],), -float("inf")
                )

        result = self.flash_attn_with_kvcache(
            q=q_for_attn,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=local_page_table,
            cache_seqlens=local_cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            max_seqlen_q=metadata.max_seq_len_q,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=attn_window_size,
            softcap=layer.logit_cap,
            num_splits=self.num_splits,
            return_softmax_lse=True,
            **local_kwargs,
        )
        local_o, local_lse = result[:2]

        local_lse = self._normalize_fa_lse(
            local_lse, q_for_attn.shape[0], q_for_attn.shape[1]
        )
        empty_local_kv = local_cache_seqlens.to(torch.long).eq(0)
        if empty_local_kv.any():
            local_o = torch.where(
                empty_local_kv.view(-1, 1, 1),
                torch.zeros_like(local_o),
                local_o,
            )
            if "sinks" in local_kwargs:
                empty_lse = local_kwargs["sinks"].to(local_lse.dtype).view(1, -1)
                empty_lse = empty_lse.expand_as(local_lse)
            else:
                empty_lse = local_lse.new_full(local_lse.shape, -float("inf"))
            local_lse = torch.where(
                empty_local_kv.view(-1, 1),
                empty_lse,
                local_lse,
            )
        if cp_world_size == 1:
            merged_o = local_o
        else:
            gathered_o = cp_group.all_gather(local_o.contiguous(), dim=0).view(
                cp_world_size, *local_o.shape
            )
            gathered_lse = cp_group.all_gather(local_lse.contiguous(), dim=0).view(
                cp_world_size, *local_lse.shape
            )

            merged_o = gathered_o[0].contiguous()
            merged_lse = gathered_lse[0].contiguous()
            for cp_idx in range(1, cp_world_size):
                merged_o, merged_lse = merge_state_v2(
                    merged_o,
                    merged_lse,
                    gathered_o[cp_idx].contiguous(),
                    gathered_lse[cp_idx].contiguous(),
                )
            if merged_o.dtype != local_o.dtype:
                merged_o = merged_o.to(dtype=local_o.dtype)

        if cp_world_size > 1:
            merged_o = merged_o[
                :, local_head_start : local_head_start + local_q_heads, :
            ]

        if out is not None:
            out.copy_(merged_o)
            merged_o = out
        return merged_o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_decode_sharded_kv(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        metadata: FlashAttentionMetadata,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        window_size: tuple[int, int],
        causal: bool,
        kwargs: dict,
        out: Optional[torch.Tensor] = None,
        scheduler_metadata: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        page_table_is_physical: bool = False,
    ) -> torch.Tensor:
        """Run sharded-KV decode without materializing full KV."""
        q_local = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        cache_seqlens = metadata.cache_seqlens_int32
        if q_local.shape[0] != cache_seqlens.numel():
            raise RuntimeError(
                "CP sharded-KV decode only supports one query per sequence, "
                f"got q_rows={q_local.shape[0]} batch={cache_seqlens.numel()}"
            )

        # Local-merge is faster but not numerically equivalent to a single
        # full-KV FA pass for WeLM. Keep it behind an explicit experiment flag.
        window_left = window_size[0] if window_size is not None else -1
        is_swa_window = 0 <= window_left < self.max_context_len
        if is_swa_window and not self.enable_attn_cp_decode_local_merge_swa:
            return self._flash_attn_sharded_kv_dense(
                q_local,
                layer,
                metadata.page_table if page_table is None else page_table,
                cache_seqlens,
                key_cache,
                value_cache,
                metadata.cu_seqlens_q,
                metadata.max_seq_len_q,
                window_size=window_size,
                causal=causal,
                kwargs=kwargs,
                out=out,
                scheduler_metadata=scheduler_metadata,
                page_table_is_physical=page_table_is_physical,
            )

        local_page_table = metadata.cp_local_page_table
        local_cache_seqlens = metadata.cp_local_cache_seqlens_int32
        local_page_start_offsets = None
        local_page_token_counts = None
        local_window_size = window_size
        # Local merge is a numerically different decomposition from a single
        # full-KV FA pass. When a sequence has not yet reached every CP shard,
        # at least one rank contributes an empty softmax state; this is the most
        # fragile case for greedy token-level parity. The condition is computed
        # from CPU sequence lengths while building metadata so layer execution
        # stays CUDA-graph friendly.
        has_empty_cp_shard = bool(getattr(metadata, "has_empty_cp_shard", False))
        if (
            self.enable_attn_cp_decode_local_merge_swa
            and is_swa_window
            and metadata.cp_swa_local_page_table is not None
        ):
            local_page_table = metadata.cp_swa_local_page_table
            local_cache_seqlens = metadata.cp_swa_local_cache_seqlens_int32
            local_page_start_offsets = metadata.cp_swa_local_page_start_offsets
            local_page_token_counts = metadata.cp_swa_local_page_token_counts
            local_window_size = (-1, -1)
        can_use_local_merge = (
            window_left < 0
            or window_left >= self.max_context_len
            or (
                is_swa_window
                and local_page_table is not None
                and local_cache_seqlens is not None
            )
        )
        use_local_merge = (
            self.enable_attn_cp_decode_local_merge
            and not has_empty_cp_shard
            and can_use_local_merge
            and local_page_table is not None
            and (page_table is None or page_table is metadata.page_table)
            and not getattr(metadata, "requires_exact_logprob", False)
        )
        if is_swa_window and local_window_size == (-1, -1):
            use_local_merge = (
                self.enable_attn_cp_decode_local_merge
                and self.enable_attn_cp_decode_local_merge_swa
                and (
                    self.page_size == 1
                    or (
                        self.enable_attn_cp_decode_cp2_fused_q_fa
                        and local_page_start_offsets is not None
                        and local_page_token_counts is not None
                    )
                )
                and not has_empty_cp_shard
                and local_page_table is not None
                and local_cache_seqlens is not None
                and not getattr(metadata, "requires_exact_logprob", False)
            )
        if use_local_merge:
            local_merge_o = self._flash_attn_sharded_kv_local_merge(
                q_local,
                layer,
                metadata,
                key_cache,
                value_cache,
                window_size=window_size,
                causal=causal,
                kwargs=kwargs,
                out=out,
                local_page_table=local_page_table,
                local_cache_seqlens=local_cache_seqlens,
                local_page_start_offsets=local_page_start_offsets,
                local_page_token_counts=local_page_token_counts,
                local_window_size=local_window_size,
            )
            if local_merge_o is not None:
                return local_merge_o

        # Keep the old correctness path for translated page tables such as SWA.
        return self._flash_attn_sharded_kv_dense(
            q_local,
            layer,
            metadata.page_table if page_table is None else page_table,
            cache_seqlens,
            key_cache,
            value_cache,
            metadata.cu_seqlens_q,
            metadata.max_seq_len_q,
            window_size=window_size,
            causal=causal,
            kwargs=kwargs,
            out=out,
            scheduler_metadata=scheduler_metadata,
            page_table_is_physical=page_table_is_physical,
        )

    def _forward_extend_sharded_kv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: FlashAttentionMetadata,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        use_welm_custom_last_q: bool,
        window_size: tuple[int, int],
        causal: bool,
        kwargs: dict,
        page_table: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run sharded-KV prefill with temporary full-KV tensors.

        This is the correctness-first path. It avoids the segment/LSE merge
        reduction order used by ring-style sharded prefill, which can amplify
        small bf16 differences through WeLM MoE routing on long prompts.
        """
        if layer.is_cross_attention:
            raise NotImplementedError(
                "CP sharded-KV prefill does not support cross attention"
            )

        runtime_layout = getattr(
            forward_batch, "attn_cp_prefill_runtime_layout", None
        )
        if runtime_layout is not None:
            return self._forward_extend_sharded_kv_compact(
                q,
                k,
                v,
                layer,
                forward_batch,
                key_cache,
                value_cache,
                use_welm_custom_last_q=use_welm_custom_last_q,
                window_size=window_size,
                causal=causal,
                kwargs=kwargs,
            )

        q_local = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        if q_local.shape[0] == 0:
            return q.new_empty((0, layer.tp_q_head_num * layer.v_head_dim))

        page_table = metadata.page_table if page_table is None else page_table
        if page_table.shape[0] != 1:
            raise NotImplementedError(
                "CP sharded-KV prefill currently requires batch_size=1"
            )
        cache_seqlens = metadata.cache_seqlens_int32
        cu_seqlens_q = metadata.cu_seqlens_q
        cu_seqlens_k = metadata.cu_seqlens_k
        max_seqlen_q = metadata.max_seq_len_q
        if use_welm_custom_last_q:
            cu_seqlens_q = metadata.mirror_cu_seqlens_q
            max_seqlen_q = metadata.mirror_max_seq_len_q
            active_indices = getattr(
                forward_batch, "kv_mirror_active_batch_indices", None
            )
            if (
                active_indices is not None
                and active_indices.numel() != metadata.page_table.shape[0]
            ):
                page_table = page_table[active_indices]
                cache_seqlens = cache_seqlens[active_indices]
                cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
                )

        if self.debug_attn_cp_metadata_checks:
            q_rows = int(cu_seqlens_q[-1].item())
            if q_local.shape[0] != q_rows:
                raise RuntimeError(
                    "CP sharded-KV gathered prefill Q metadata mismatch: "
                    f"q_rows={q_local.shape[0]} cu_seqlens_q[-1]={q_rows}"
                )

        return self._flash_attn_sharded_kv_dense(
            q_local,
            layer,
            page_table,
            cache_seqlens,
            key_cache,
            value_cache,
            cu_seqlens_q,
            max_seqlen_q,
            window_size=window_size,
            causal=causal,
            kwargs=kwargs,
            cu_seqlens_k=cu_seqlens_k,
            prefill_plan=metadata.cp_sharded_kv_prefill_plan,
        )

    @staticmethod
    def _get_cp_prefill_swa_gather_plan(
        forward_batch: ForwardBatch,
        gather_plan,
        runtime_layout,
        *,
        window_left: int,
    ):
        cache = getattr(
            forward_batch, "attn_cp_prefill_swa_kv_gather_plans", None
        )
        if cache is None:
            cache = {}
            forward_batch.attn_cp_prefill_swa_kv_gather_plans = cache
        cache_key = (
            int(window_left),
            bool(runtime_layout.q_is_contracted),
            tuple(runtime_layout.active_per_rank_tokens),
        )
        plan = cache.get(cache_key)
        if plan is None:
            plan = build_cp_prefill_swa_gather_plan(
                plan=gather_plan,
                runtime_layout=runtime_layout,
                window_left=window_left,
            )
            cache[cache_key] = plan
        return plan

    @staticmethod
    def _read_cp_prefill_prefix_kv(
        physical_slots: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        page_size: int,
        *,
        zero_dummy_slots: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = physical_slots.to(device=key_cache.device, dtype=torch.int64)
        if zero_dummy_slots and slots.numel() != 0:
            key_cache[0, 0].zero_()
            value_cache[0, 0].zero_()
        pages = torch.div(slots, page_size, rounding_mode="floor")
        offsets = slots % page_size
        key = key_cache[pages, offsets].contiguous()
        value = value_cache[pages, offsets].contiguous()
        return key, value

    def _resolve_cp_prefill_swa_prefix_slots(
        self,
        prefix_slots: torch.Tensor,
        *,
        missing_slot_message: str,
    ) -> torch.Tensor:
        if prefix_slots.numel() == 0:
            return prefix_slots

        prefix_slots = self.cp_sharded_page_table_resolver.resolve_swa_slots(
            prefix_slots
        )
        no_dummy = torch.all(prefix_slots != 0)
        if prefix_slots.device.type == "cpu":
            if not bool(no_dummy):
                raise RuntimeError(missing_slot_message)
        else:
            torch._assert_async(no_dummy, missing_slot_message)
        return prefix_slots

    def _forward_extend_sharded_kv_swa_compact(
        self,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        runtime_layout,
        gather_plan,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        window_size: tuple[int, int],
        kwargs: dict,
    ) -> torch.Tensor:
        window_left, window_right = window_size
        if window_left <= 0 or window_right != 0:
            raise NotImplementedError(
                "Phase 2 SWA compact gather requires a positive left-only window"
            )
        swa_plan = self._get_cp_prefill_swa_gather_plan(
            forward_batch,
            gather_plan,
            runtime_layout,
            window_left=window_left,
        )

        prefix_send_indices = swa_plan.prefix.local_send_indices
        prefix_slots = gather_plan.prefix.local_physical_slots.index_select(
            0, prefix_send_indices
        )
        prefix_slots = self._resolve_cp_prefill_swa_prefix_slots(
            prefix_slots,
            missing_slot_message=(
                "Phase 2 SWA compact gather selected a missing prefix slot"
            ),
        )
        packed_prefix_k, packed_prefix_v = self._read_cp_prefill_prefix_kv(
            prefix_slots,
            key_cache,
            value_cache,
            self.page_size,
        )

        extend_send_indices = swa_plan.extend.local_send_indices
        packed_extend_k = k_local.index_select(0, extend_send_indices)
        packed_extend_v = v_local.index_select(0, extend_send_indices)
        compact_k, compact_v = gather_cp_prefill_swa_kv(
            plan=swa_plan,
            packed_prefix_k=packed_prefix_k,
            packed_prefix_v=packed_prefix_v,
            packed_extend_k=packed_extend_k,
            packed_extend_v=packed_extend_v,
            cp_group=get_sharded_kv_cp_group(),
        )

        if runtime_layout.active_local_tokens == 0:
            if compact_k is not None or compact_v is not None:
                raise RuntimeError(
                    "empty-Q CP rank unexpectedly received compact SWA K/V"
                )
            return q_local.new_empty(
                (0, layer.tp_q_head_num * layer.v_head_dim)
            )
        if compact_k is None or compact_v is None:
            raise RuntimeError("active-Q CP rank is missing compact SWA K/V")
        if (
            compact_k.shape[0] != swa_plan.compact_token_count
            or compact_v.shape[0] != swa_plan.compact_token_count
        ):
            raise RuntimeError(
                "Phase 2 SWA compact gather returned an unexpected row count"
            )

        output = q_local.new_empty(
            (q_local.shape[0], layer.tp_q_head_num, layer.v_head_dim)
        )
        compact_offset = 0
        for block_index, block in enumerate(runtime_layout.q_blocks):
            k_count = swa_plan.block_k_lengths[block_index]
            q_block = q_local.narrow(0, block.local_start, block.token_count)
            k_block = compact_k.narrow(0, compact_offset, k_count)
            v_block = compact_v.narrow(0, compact_offset, k_count)
            out_block = output.narrow(0, block.local_start, block.token_count)
            block_output = self.flash_attn_varlen_func(
                q=q_block,
                k=k_block,
                v=v_block,
                cu_seqlens_q=block.cu_seqlens_q,
                cu_seqlens_k=swa_plan.block_cu_seqlens_k[block_index],
                max_seqlen_q=block.token_count,
                # Preserve FA3 auto-split scheduling and bf16 reduction order.
                max_seqlen_k=block.visible_kv_end,
                softmax_scale=layer.scaling,
                causal=True,
                window_size=window_size,
                softcap=layer.logit_cap,
                num_splits=self.num_splits,
                out=out_block,
                **kwargs,
            )
            if block_output is not out_block:
                out_block.copy_(block_output)
            compact_offset += k_count
        if compact_offset != swa_plan.compact_token_count:
            raise RuntimeError(
                "Phase 2 SWA attention did not consume all compact K/V rows"
            )
        return output.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_extend_sharded_kv_compact(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        use_welm_custom_last_q: bool,
        window_size: tuple[int, int],
        causal: bool,
        kwargs: dict,
    ) -> torch.Tensor:
        runtime_layout = forward_batch.attn_cp_prefill_runtime_layout
        gather_plan = getattr(
            forward_batch, "attn_cp_prefill_kv_gather_plan", None
        )
        if gather_plan is None:
            raise RuntimeError(
                "Phase 2 sharded-KV prefill is missing its compact gather plan; "
                "dense fallback is disabled"
            )
        if use_welm_custom_last_q and not getattr(
            runtime_layout, "q_is_contracted", False
        ):
            raise RuntimeError(
                "Phase 2 custom-last-Q attention requires a contracted Q runtime"
            )
        if not causal:
            raise NotImplementedError(
                "Phase 2 sharded-KV prefill currently requires causal attention"
            )
        if self.kv_cache_dtype_str != "auto":
            raise NotImplementedError(
                "Phase 2 sharded-KV prefill requires an unquantized KV cache"
            )
        if k is None or v is None:
            raise RuntimeError("Phase 2 sharded-KV prefill requires projected K/V")

        q_local = q.contiguous().view(
            -1, layer.tp_q_head_num, layer.head_dim
        )
        k_local = k.contiguous().view(
            -1, layer.tp_k_head_num, layer.head_dim
        )
        v_local = v.contiguous().view(
            -1, layer.tp_v_head_num, layer.v_head_dim
        )
        expected_q_rows = runtime_layout.active_local_tokens
        expected_kv_rows = runtime_layout.kv_local_tokens
        if q_local.shape[0] != expected_q_rows:
            raise RuntimeError(
                "Phase 2 Q rows do not match the CP-local runtime layout"
            )
        if k_local.shape[0] != expected_kv_rows or v_local.shape[0] != expected_kv_rows:
            raise RuntimeError(
                "Phase 2 K/V rows do not match the owner-local runtime layout"
            )

        use_full_collective = should_use_full_cp_kv_collective(
            runtime_layout,
            page_size=self.page_size,
        )
        use_dense_swa_full_collective = (
            window_size != (-1, -1) and use_full_collective
        )
        if window_size != (-1, -1) and not use_full_collective:
            return self._forward_extend_sharded_kv_swa_compact(
                q_local,
                k_local,
                v_local,
                layer,
                forward_batch,
                runtime_layout,
                gather_plan,
                key_cache,
                value_cache,
                window_size=window_size,
                kwargs=kwargs,
            )

        active_destination_ranks = tuple(
            rank
            for rank, count in enumerate(
                runtime_layout.active_tokens_per_cp_rank()
            )
            if count != 0
        )
        destination_ranks = (
            tuple(range(len(runtime_layout.active_per_rank_tokens)))
            if use_full_collective
            else active_destination_ranks
        )
        ipc_lease = None
        prefill_cp_kv_ipc_transport = getattr(
            self, "prefill_cp_kv_ipc_transport", None
        )
        source_push_plan = getattr(
            forward_batch,
            "attn_cp_prefill_kv_source_push_plan",
            None,
        )
        use_ipc_source_push = (
            source_push_plan is not None
            and not use_dense_swa_full_collective
        )
        if use_ipc_source_push:
            if prefill_cp_kv_ipc_transport is None:
                raise RuntimeError(
                    "Prefill CP K/V IPC source-push plan is missing its transport; "
                    "NCCL gather fallback is disabled"
                )
            expected_kv_rows = (
                runtime_layout.spec.extend_start + runtime_layout.spec.extend_len
            )
            if source_push_plan.logical_token_count != expected_kv_rows:
                raise RuntimeError(
                    "Prefill CP K/V IPC source-push plan does not cover the "
                    "logical attention sequence"
                )
            k_row_width = layer.tp_k_head_num * layer.head_dim
            v_row_width = layer.tp_v_head_num * layer.v_head_dim
            if k_row_width != v_row_width:
                raise RuntimeError(
                    "Prefill CP K/V IPC requires equal K/V row widths"
                )
            if not key_cache.is_contiguous() or not value_cache.is_contiguous():
                raise RuntimeError(
                    "Prefill CP K/V IPC requires contiguous paged K/V cache"
                )
            forward_iter = getattr(forward_batch, "forward_iter", None)
            if forward_iter is None:
                raise RuntimeError(
                    "Prefill CP K/V IPC requires the scheduler forward_iter; "
                    "local epoch fallback is disabled in production"
                )
            ipc_lease = prefill_cp_kv_ipc_transport.push(
                plan=source_push_plan,
                prefix_key_rows=key_cache.view(-1, k_row_width),
                prefix_value_rows=value_cache.view(-1, v_row_width),
                extend_key_rows=k_local.view(-1, k_row_width),
                extend_value_rows=v_local.view(-1, v_row_width),
                destination_ranks=destination_ranks,
                forward_iter=forward_iter,
                operation_index=layer.layer_id,
            )
            if ipc_lease is None:
                full_k = None
                full_v = None
            else:
                full_k = ipc_lease.key.view(
                    -1, layer.tp_k_head_num, layer.head_dim
                )
                full_v = ipc_lease.value.view(
                    -1, layer.tp_v_head_num, layer.v_head_dim
                )
        elif (
            prefill_cp_kv_ipc_transport is not None
            and not use_dense_swa_full_collective
        ):
            raise RuntimeError(
                "Prefill CP K/V IPC is missing its source-push plan; "
                "NCCL gather fallback is disabled"
            )
        else:
            prefix_slots = gather_plan.prefix.local_physical_slots
            if use_dense_swa_full_collective:
                prefix_slots = self._resolve_cp_prefill_swa_prefix_slots(
                    prefix_slots,
                    missing_slot_message=(
                        "Phase 2 short SWA gather selected a missing prefix slot"
                    ),
                )
            local_prefix_k, local_prefix_v = self._read_cp_prefill_prefix_kv(
                prefix_slots,
                key_cache,
                value_cache,
                self.page_size,
            )
            full_k, full_v = gather_cp_prefill_kv(
                plan=gather_plan,
                local_prefix_k=local_prefix_k,
                local_prefix_v=local_prefix_v,
                local_extend_k=k_local,
                local_extend_v=v_local,
                cp_group=get_sharded_kv_cp_group(),
                destination_ranks=destination_ranks,
            )
        try:
            if runtime_layout.active_local_tokens == 0:
                if not use_full_collective and (
                    full_k is not None or full_v is not None
                ):
                    raise RuntimeError(
                        "empty-Q CP rank unexpectedly received gathered K/V"
                    )
                return q_local.new_empty(
                    (0, layer.tp_q_head_num * layer.v_head_dim)
                )
            if full_k is None or full_v is None:
                raise RuntimeError("active-Q CP rank is missing gathered K/V")
            expected_kv_rows = (
                runtime_layout.spec.extend_start + runtime_layout.spec.extend_len
            )
            if (
                full_k.shape[0] != expected_kv_rows
                or full_v.shape[0] != expected_kv_rows
            ):
                raise RuntimeError(
                    "Phase 2 compact KV gather did not reconstruct the logical sequence"
                )

            output = q_local.new_empty(
                (q_local.shape[0], layer.tp_q_head_num, layer.v_head_dim)
            )
            for block in runtime_layout.q_blocks:
                q_block = q_local.narrow(
                    0, block.local_start, block.token_count
                )
                out_block = output.narrow(
                    0, block.local_start, block.token_count
                )
                block_output = self.flash_attn_varlen_func(
                    q=q_block,
                    k=full_k[: block.visible_kv_end],
                    v=full_v[: block.visible_kv_end],
                    cu_seqlens_q=block.cu_seqlens_q,
                    cu_seqlens_k=block.cu_seqlens_k,
                    max_seqlen_q=block.token_count,
                    max_seqlen_k=block.visible_kv_end,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    num_splits=self.num_splits,
                    out=out_block,
                    **kwargs,
                )
                if block_output is not out_block:
                    out_block.copy_(block_output)
            return output.view(
                -1, layer.tp_q_head_num * layer.v_head_dim
            )
        finally:
            if ipc_lease is not None:
                ipc_lease.release()

    @staticmethod
    def _validate_cp_prefill_runtime(forward_batch: ForwardBatch) -> None:
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            return
        split_specs = getattr(
            forward_batch, "attn_cp_prefill_split_specs", None
        )
        phase2_expected = split_specs is not None and any(
            spec is not None for spec in split_specs
        )
        if phase2_expected and getattr(
            forward_batch, "attn_cp_prefill_runtime_layout", None
        ) is None:
            raise RuntimeError(
                "Phase 2 split spec reached FA3 without a CP-local runtime layout; "
                "refusing dense fallback"
            )

    @staticmethod
    def _validate_cp_prefill_kv_write_mode(
        runtime_layout,
        gather_plan,
        *,
        save_kv_cache: bool,
        short_full_participation: bool = False,
    ) -> None:
        if save_kv_cache:
            return
        zero_row_participant = (
            runtime_layout.active_local_tokens == 0
            and getattr(runtime_layout, "kv_local_tokens", 0) == 0
            and (
                gather_plan.prefix.local_physical_slots.numel() != 0
                or short_full_participation
            )
        )
        if not zero_row_participant:
            raise RuntimeError(
                "Phase 2 save_kv_cache=False is valid only for a "
                "prefix-send-only or bounded short-participation CP rank"
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        self._validate_cp_prefill_runtime(forward_batch)
        runtime_layout = getattr(
            forward_batch, "attn_cp_prefill_runtime_layout", None
        )
        if runtime_layout is not None:
            forward_batch.attn_cp_prefill_short_full_kv_collective = (
                should_use_full_cp_kv_collective(
                    runtime_layout,
                    page_size=self.page_size,
                )
            )
        metadata = FlashAttentionMetadata()
        metadata.requires_exact_logprob = (
            self._attncp_forward_batch_requires_exact_logprob(forward_batch)
        )
        metadata.has_empty_cp_shard = self.attncp_forward_batch_has_empty_cp_shard(
            forward_batch
        )
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # Draft Decode
            if forward_batch.spec_info is not None:
                if self.topk <= 1:
                    metadata.cache_seqlens_int32 = (
                        seqlens_in_batch + (self.speculative_step_id + 1)
                    ).to(torch.int32)
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                else:
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,
                        step=decode_length,
                        dtype=torch.int32,
                        device=device,
                    )
                    # shape: [bs, num_steps, topk] -> [bs x topk, num_steps]
                    cache_loc = forward_batch.out_cache_loc.view(
                        -1, self.speculative_num_steps
                    )
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
                # Precompute FA3 scheduler metadata to avoid per-layer
                # prepare_varlen_num_blocks kernel calls
                metadata.scheduler_metadata = self._compute_scheduler_metadata(
                    batch_size,
                    metadata.max_seq_len_k,
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_q,
                )
            # TODO: we need to test this part for llama 4 eagle case
            self._maybe_init_local_attn_metadata(forward_batch, metadata, device)
        elif forward_batch.forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + self.speculative_num_draft_tokens
                ).to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    forward_batch.seq_lens_cpu.max().item()
                    + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                self._maybe_init_local_attn_metadata(forward_batch, metadata, device)
            else:
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens
                    + 1,
                    dtype=torch.int32,
                    device=device,
                )

                # create expand page table
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)
                cols = offsets.expand(
                    forward_batch.seq_lens.numel(), -1
                ) + forward_batch.seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                mask = forward_batch.spec_info.custom_mask[
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)

                # shift table indices to avoid padding
                # non_masked_page_table [[8, 9, 10],   mask (display with int format) [[1, 0, 0],
                #                        [8, 9, 10],                                   [1, 1, 0],
                #                        [8, 9, 10]]                                   [1, 0, 1]]
                # if masked with padding [[8, 0, 0],   our mask without padding       [[8, 9, 10],
                #                        [8, 9, 0],                                    [8, 9, 10],
                #                        [8, 0, 10]]                                   [8, 10, 9]]
                # note here cache_seqlens_int32 is [1, 2, 2] so extra page indices will be ignored in each row
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                # Build keys: if an entry is valid (mask==True), keep its original index;
                # if not, add self.speculative_num_draft_tokens so that it sorts after all valid entries.
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = (
                    forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                self.forward_metadata_spec_decode_expand = metadata_expand

                if self.has_swa:
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand
                    )

        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
            include_draft_extend_v2=True
        ):
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]
            if forward_batch.enable_welm_kv_mirror_opt:
                mirror_num_queries = batch_size
                if forward_batch.welm_kv_mirror_last_q_indices is not None:
                    mirror_num_queries = (
                        forward_batch.welm_kv_mirror_last_q_indices.numel()
                    )
                metadata.mirror_cu_seqlens_q = torch.arange(
                    0, mirror_num_queries + 1, dtype=torch.int32, device=device
                )

            if any(
                forward_batch.extend_prefix_lens_cpu
            ) or forward_batch.forward_mode.is_draft_extend(include_v2=True):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

            # Setup local attention if enabled
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                self._maybe_init_local_attn_metadata(forward_batch, metadata, device)

        # Encoder metadata for cross attention
        if forward_batch.encoder_lens is not None:
            assert (
                forward_batch.encoder_lens.numel() == 1
            ), "Only encoder size 1 is supported for now"

            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()
            metadata.encoder_page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # Currently only support forward_batch.encoder_lens.numel() == 1
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]

        uses_compact_cp_prefill = (
            self.is_attn_cp_sharded_kv
            and getattr(
                forward_batch, "attn_cp_prefill_runtime_layout", None
            )
            is not None
        )
        needs_swa_page_table = (
            self.use_sliding_window_kv_pool and not uses_compact_cp_prefill
        )
        if needs_swa_page_table and not self.is_attn_cp_sharded_kv:
            metadata.swa_page_table = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    metadata.page_table
                )
            )

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )

            if metadata.swa_page_table is not None:
                metadata.swa_page_table = (
                    metadata.swa_page_table[:, self.strided_indices] // self.page_size
                )

            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

            if (
                self.topk > 1
                and forward_batch.forward_mode.is_decode_or_idle()
                and forward_batch.spec_info is not None
            ):
                # Modifies cache_seqlens_int32 and page_table(B, speculative_num_steps).
                last_page_lens = forward_batch.seq_lens % self.page_size
                # First attention handles prefix - last_page_len part.
                metadata.cache_seqlens_int32 -= last_page_lens  # Both (B, )

                # Second attention handles last_page_len + decode part.
                expanded_last_page_lens = last_page_lens.repeat_interleave(self.topk)
                self.forward_metadata_spec_decode_expand.cache_seqlens_int32 += (
                    expanded_last_page_lens
                )
                # NOTE: the max decode length is speculative_num_steps - 1 (one token always generated by draft extend)
                # and we leave one extra for last_page_len, which -> speculative_num_steps for the page table
                expand_page_table = torch.zeros(
                    forward_batch.batch_size * self.topk,
                    self.speculative_num_steps,
                    dtype=torch.int32,
                    device=self.device,
                )
                # shape: [bs, num_steps, topk] -> [bs x topk, num_steps]
                cache_loc = forward_batch.out_cache_loc.view(
                    -1, self.speculative_num_steps
                )
                draft_decode_set_expand_metadata(
                    cache_seqlens_int32=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    page_table=expand_page_table,
                    last_page_lens=last_page_lens,
                    decode_length=decode_length,
                    cache_loc=cache_loc,
                    topk=self.topk,
                    page_size=self.page_size,
                )
                self.forward_metadata_spec_decode_expand.page_table = expand_page_table

        if needs_swa_page_table and self.is_attn_cp_sharded_kv:
            metadata.swa_page_table = self._resolve_sharded_logical_swa_page_table(
                metadata.page_table
            )

        if (
            self.is_attn_cp_sharded_kv
            and forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed(
                include_draft_extend_v2=True
            )
            and getattr(
                forward_batch, "attn_cp_prefill_runtime_layout", None
            ) is None
        ):
            self._set_sharded_kv_prefill_plan(metadata, forward_batch)

        if (
            self.use_sliding_window_kv_pool
            and not self.is_attn_cp_sharded_kv
            and forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is None
        ):
            self._set_decode_swa_metadata(
                metadata,
                metadata.swa_page_table,
                metadata.cache_seqlens_int32,
            )

        if (
            is_cp_kv_sharded()
            and self.enable_attn_cp_decode_local_merge
            and forward_batch.forward_mode.is_decode_or_idle()
            and forward_batch.spec_info is None
        ):
            self._set_sharded_kv_decode_metadata(
                metadata,
                metadata.page_table,
                metadata.cache_seqlens_int32,
            )
            if (
                self.enable_attn_cp_decode_local_merge_swa
                and self.use_sliding_window_kv_pool
                and metadata.swa_page_table is not None
            ):
                self._set_sharded_kv_decode_swa_metadata(
                    metadata,
                    metadata.swa_page_table,
                    metadata.cache_seqlens_int32,
                )

        self.forward_metadata = metadata

    def _per_suffix_attn_compute(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: FlashAttentionMetadata,
        sinks: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        assert not self.use_mla, "per-suffix attention does not support MLA"
        assert not layer.is_cross_attention

        sf = layer.scale_seq_factor
        assert sf > 1, f"per-suffix attention called with scale_seq_factor={sf}"
        assert self.page_size == sf, (
            f"per-suffix attention requires page_size == scale_seq_factor ({sf}), "
            f"got page_size={self.page_size}"
        )

        n_token = q.shape[0]
        n_q_heads = layer.tp_q_head_num
        n_kv_heads = layer.tp_k_head_num
        head_dim = layer.head_dim
        v_head_dim = layer.v_head_dim
        assert n_token % sf == 0
        n_logical = n_token // sf

        q_fold = (
            q.contiguous()
            .view(n_logical, sf, n_q_heads, head_dim)
            .reshape(n_logical, sf * n_q_heads, head_dim)
        )

        key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
            layer.layer_id
        )
        key_cache_fold = key_cache.view(-1, sf, n_kv_heads, head_dim).reshape(
            -1, 1, sf * n_kv_heads, head_dim
        )
        value_cache_fold = value_cache.view(-1, sf, n_kv_heads, v_head_dim).reshape(
            -1, 1, sf * n_kv_heads, v_head_dim
        )

        kwargs = {}
        if self.fa_impl_ver != 3:
            kwargs["ver"] = self.fa_impl_ver
        if sinks is not None:
            kwargs["sinks"] = sinks.repeat(sf)

        sliding = layer.sliding_window_size
        window_size = (
            (sliding, 0) if sliding is not None and sliding > 0 else (-1, -1)
        )
        o = self.flash_attn_with_kvcache(
            q=q_fold,
            k_cache=key_cache_fold,
            v_cache=value_cache_fold,
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens_int32 // sf,
            cu_seqlens_q=(metadata.cu_seqlens_q // sf).to(torch.int32),
            cu_seqlens_k_new=(metadata.cu_seqlens_k // sf).to(torch.int32),
            max_seqlen_q=(metadata.max_seq_len_q + sf - 1) // sf,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=window_size,
            softcap=layer.logit_cap,
            **kwargs,
        )
        return o.view(n_logical, sf, n_q_heads, v_head_dim).reshape(
            n_token, n_q_heads * v_head_dim
        )

    def _per_suffix_scp_attn_compute(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        metadata: FlashAttentionMetadata,
        sinks: Optional[torch.Tensor] = None,
        causal: bool = True,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        from sglang.srt.layers.dp_attention import (
            get_suffix_parallel_rank,
            get_suffix_parallel_size,
        )

        assert not self.use_mla, "suffix-parallel attention does not support MLA"
        assert not layer.is_cross_attention

        sf = get_suffix_parallel_size()
        sp_rank = get_suffix_parallel_rank()
        assert sf > 1, f"suffix-parallel attention called with sf={sf}"
        pool = forward_batch.token_to_kv_pool

        if save_kv_cache and k is not None:
            assert v is not None
            cache_loc = forward_batch.out_cache_loc
            track_loc = cache_loc[sp_rank::sf].contiguous()
            pool.set_kv_buffer(layer, track_loc, k, v, layer.k_scale, layer.v_scale)

        n_q_heads = layer.tp_q_head_num
        n_kv_heads = layer.tp_k_head_num
        head_dim = layer.head_dim
        v_head_dim = layer.v_head_dim

        key_cache, value_cache = pool.get_kv_buffer(layer.layer_id)
        key_cache = key_cache.view(-1, 1, n_kv_heads, head_dim)
        value_cache = value_cache.view(-1, 1, n_kv_heads, v_head_dim)

        page_table_track = metadata.page_table[:, sp_rank::sf].contiguous()
        page_table = pool.translate_loc_from_full_to_suffix(page_table_track)
        kwargs = {}
        if self.fa_impl_ver != 3:
            kwargs["ver"] = self.fa_impl_ver
        if sinks is not None:
            kwargs["sinks"] = sinks

        sliding = layer.sliding_window_size
        window_size = (
            (sliding, 0) if sliding is not None and sliding > 0 else (-1, -1)
        )
        o = self.flash_attn_with_kvcache(
            q=q.contiguous().view(-1, n_q_heads, head_dim),
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=page_table,
            cache_seqlens=metadata.cache_seqlens_int32 // sf,
            cu_seqlens_q=(metadata.cu_seqlens_q // sf).to(torch.int32),
            cu_seqlens_k_new=(metadata.cu_seqlens_k // sf).to(torch.int32),
            max_seqlen_q=(metadata.max_seq_len_q + sf - 1) // sf,
            softmax_scale=layer.scaling,
            causal=causal,
            window_size=window_size,
            softcap=layer.logit_cap,
            **kwargs,
        )
        return o.view(-1, n_q_heads * v_head_dim)

    @staticmethod
    def _is_scp_suffix_layer(layer: RadixAttention, forward_batch: ForwardBatch) -> bool:
        if getattr(layer, "scale_seq_attn_per_suffix", False) and getattr(
            layer, "suffix_parallel", False
        ):
            return True
        is_suffix_layer = getattr(
            getattr(forward_batch, "token_to_kv_pool", None), "is_suffix_layer", None
        )
        return bool(is_suffix_layer is not None and is_suffix_layer(layer.layer_id))

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        runtime_layout = getattr(
            forward_batch, "attn_cp_prefill_runtime_layout", None
        )
        if runtime_layout is not None:
            if layer.is_cross_attention:
                raise NotImplementedError(
                    "Phase 2 sharded-KV prefill does not support cross attention"
                )
            gather_plan = getattr(
                forward_batch, "attn_cp_prefill_kv_gather_plan", None
            )
            if gather_plan is None:
                raise RuntimeError(
                    "Phase 2 sharded-KV prefill is missing its compact gather plan; "
                    "dense fallback is disabled"
                )
            self._validate_cp_prefill_kv_write_mode(
                runtime_layout,
                gather_plan,
                save_kv_cache=save_kv_cache,
                short_full_participation=bool(
                    getattr(
                        forward_batch,
                        "attn_cp_prefill_short_full_kv_collective",
                        False,
                    )
                ),
            )
            if self._is_scp_suffix_layer(layer, forward_batch):
                raise NotImplementedError(
                    "Phase 2 sharded-KV prefill does not support suffix parallel"
                )

        if self._is_scp_suffix_layer(layer, forward_batch):
            return self._per_suffix_scp_attn_compute(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                metadata=self.forward_metadata,
                sinks=sinks,
                causal=not layer.is_cross_attention,
                save_kv_cache=save_kv_cache,
            )

        if k is not None:
            assert v is not None

            is_cp_mode = (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
                and not is_cp_kv_sharded()
            )

            if save_kv_cache and not is_cp_mode and not self.fa_skip_kv_cache:
                cache_loc = (
                    runtime_layout.local_out_cache_loc
                    if runtime_layout is not None
                    else forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                kv_fill_cache_loc = getattr(
                    forward_batch, "welm_mtp_kv_fill_cache_loc", None
                )
                custom_last_cache_loc = getattr(
                    forward_batch, "custom_last_cache_loc", None
                )
                if (
                    not layer.is_cross_attention
                    and getattr(forward_batch, "welm_mtp_merge_kv_fill_draft", False)
                    and kv_fill_cache_loc is not None
                    and k.shape[0] != cache_loc.numel()
                    and k.shape[0] == kv_fill_cache_loc.numel()
                ):
                    cache_loc = kv_fill_cache_loc
                elif (
                    not layer.is_cross_attention
                    and getattr(forward_batch, "welm_kv_mirror_contracted", False)
                    and custom_last_cache_loc is not None
                    and k.shape[0] != cache_loc.numel()
                    and k.shape[0] == custom_last_cache_loc.numel()
                ):
                    cache_loc = custom_last_cache_loc
                if not self.use_mla and k.shape[0] != cache_loc.numel():
                    pool = getattr(forward_batch, "token_to_kv_pool", None)
                    pool_mapping = getattr(pool, "layers_mapping", None)
                    pool_kind = (
                        pool_mapping.get(layer.layer_id)
                        if isinstance(pool_mapping, dict)
                        else None
                    )
                    raise RuntimeError(
                        "FlashAttention KV store shape mismatch before pool write: "
                        f"layer_id={layer.layer_id}, "
                        f"scale_seq_attn_per_suffix={getattr(layer, 'scale_seq_attn_per_suffix', None)}, "
                        f"suffix_parallel={getattr(layer, 'suffix_parallel', None)}, "
                        f"pool_kind={pool_kind}, "
                        f"forward_mode={forward_batch.forward_mode}, "
                        f"q_shape={tuple(q.shape)}, k_shape={tuple(k.shape)}, "
                        f"v_shape={tuple(v.shape)}, cache_loc_numel={cache_loc.numel()}, "
                        f"out_cache_loc_numel={getattr(forward_batch, 'out_cache_loc', cache_loc).numel()}"
                    )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
            if is_cp_mode:
                cp_allgather_and_save_kv_cache(
                    forward_batch, layer, k, v, self.attn_cp_size
                )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        if getattr(layer, "scale_seq_attn_per_suffix", False):
            return self._per_suffix_attn_compute(
                q=q,
                layer=layer,
                forward_batch=forward_batch,
                metadata=metadata,
                sinks=sinks,
                causal=not layer.is_cross_attention,
            )

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        is_swa_layer = self._is_swa_kv_pool_layer(
            layer, forward_batch.token_to_kv_pool
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)
        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case,
        # 4) fa_impl_ver != 4 since fa4 does not currently support fp8 queries and keys.
        if (
            self.kv_cache_dtype_str != "auto"
            and layer.head_dim <= 256
            and self.fa_impl_ver != 4
        ):
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        # Check if we should use local attention
        use_local_attn = (
            self.has_local_attention
            and self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # We do cascade attention for Target Verify with topk > 1. WeLM MTP
        # uses dense tree rows, so real SWA layers need an exact compact KV
        # table instead of a row-index-based local window.
        use_welm_mtp_swa_compact = (
            self.is_welm_v4_model
            and forward_batch.forward_mode.is_target_verify()
            and self.topk > 1
            and is_swa_layer
            and int(layer.sliding_window_size) < int(self.max_context_len)
        )
        use_welm_mtp_swa_cascade = (
            self.is_welm_v4_model
            and forward_batch.forward_mode.is_target_verify()
            and self.topk > 1
            and is_swa_layer
            and not use_welm_mtp_swa_compact
        )
        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify()
            and self.topk > 1
            and (not is_swa_layer or use_welm_mtp_swa_cascade)
        )
        cascade_prefix_window_size = window_size
        cascade_expand_window_size = window_size
        if use_welm_mtp_swa_cascade:
            q_len = int(self.speculative_num_draft_tokens or metadata.max_seq_len_q)
            cascade_prefix_window_size = (
                max(int(layer.sliding_window_size) - q_len, 0),
                max(q_len - 1, 0),
            )
            cascade_expand_window_size = (-1, -1)
        if use_welm_mtp_swa_compact:
            self._init_sliding_window_attn_spec_metadata(
                metadata,
                self.forward_metadata_spec_decode_expand,
                metadata.swa_spec_metadata,
                sliding_window_size=int(layer.sliding_window_size),
                compact_prefix=True,
            )

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks
        cascade_expand_kwargs = kwargs
        if use_cascade_attn and sinks is not None:
            # Attention sink is a single global softmax state. Cascade attention
            # splits KV into prefix and expand branches, so include the sink in
            # exactly one branch before merging states.
            cascade_expand_kwargs = dict(kwargs)
            cascade_expand_kwargs.pop("sinks", None)

        _fa_out = (
            forward_batch._attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            if getattr(forward_batch, "_attn_output", None) is not None
            else None
        )

        use_welm_custom_last_q = (
            getattr(forward_batch, "welm_kv_mirror_contracted", False)
            and getattr(forward_batch, "custom_last_index", None) is not None
            and q.shape[0] == forward_batch.custom_last_index.numel()
        )

        # Get the appropriate page table based on whether we're using local attention
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
        elif (
            is_swa_layer
            and metadata.swa_spec_metadata is not None
            and not use_welm_mtp_swa_cascade
        ):
            swa_spec_metadata = metadata.swa_spec_metadata
            page_table = swa_spec_metadata.page_table
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32
            max_seqlen_q = swa_spec_metadata.max_seq_len_q
            cu_seqlens_k = swa_spec_metadata.cu_seqlens_k
        elif use_welm_custom_last_q:
            page_table = metadata.page_table
            if (
                is_swa_layer
                and self.use_sliding_window_kv_pool
                and runtime_layout is None
            ):
                if metadata.swa_page_table is None:
                    raise RuntimeError(
                        "Hybrid SWA prefill is missing its translated page table"
                    )
                page_table = metadata.swa_page_table
            cu_seqlens_q = metadata.mirror_cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.mirror_max_seq_len_q
            cu_seqlens_k = metadata.cu_seqlens_k
            active_indices = getattr(
                forward_batch, "kv_mirror_active_batch_indices", None
            )
            if (
                active_indices is not None
                and active_indices.numel() != page_table.shape[0]
            ):
                page_table = page_table[active_indices]
                cache_seqlens = cache_seqlens[active_indices]
                cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(cache_seqlens, dim=0, dtype=torch.int32), (1, 0)
                )
        else:
            page_table = metadata.page_table
            if (
                is_swa_layer
                and self.use_sliding_window_kv_pool
                and runtime_layout is None
            ):
                if metadata.swa_page_table is None:
                    raise RuntimeError(
                        "Hybrid SWA prefill is missing its translated page table"
                    )
                page_table = metadata.swa_page_table
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            cu_seqlens_k = metadata.cu_seqlens_k

        # Use Flash Attention for prefill
        if not self.use_mla:
            # Do multi-head attention
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )

            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            if (
                forward_batch.forward_mode.is_context_parallel_extend()
                and forward_batch.attn_cp_metadata is not None
                and self.attn_cp_size > 1
                and not is_cp_kv_sharded()
            ):

                def _fa_cp_attn(
                    q_chunk, cu_seqlens_q_cp, cache_seqlens_cp, max_seqlen_q_cp
                ):
                    return self.flash_attn_with_kvcache(
                        q=q_chunk,
                        k_cache=key_cache,
                        v_cache=value_cache,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens_cp,
                        cu_seqlens_q=cu_seqlens_q_cp,
                        cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                        max_seqlen_q=max_seqlen_q_cp,
                        softmax_scale=layer.scaling,
                        causal=(
                            False
                            if use_cascade_attn or use_welm_mtp_swa_compact
                            else causal
                        ),
                        window_size=(
                            (-1, -1)
                            if use_welm_mtp_swa_compact
                            else (
                                cascade_prefix_window_size
                                if use_cascade_attn
                                else window_size
                            )
                        ),
                        softcap=layer.logit_cap,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        return_softmax_lse=use_cascade_attn,
                        num_splits=self.num_splits,
                        **kwargs,
                    )

                result = cp_attn_forward_extend(
                    forward_batch,
                    q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    self.device,
                    _fa_cp_attn,
                )
            elif is_cp_kv_sharded():
                if use_local_attn:
                    raise NotImplementedError(
                        "CP sharded-KV prefill does not support local attention"
                    )
                if use_cascade_attn:
                    raise NotImplementedError(
                        "CP sharded-KV prefill does not support cascade attention yet"
                    )
                result = self._forward_extend_sharded_kv(
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    metadata,
                    key_cache,
                    value_cache,
                    use_welm_custom_last_q=use_welm_custom_last_q,
                    window_size=window_size,
                    causal=causal and not use_welm_mtp_swa_compact,
                    kwargs=kwargs,
                    page_table=page_table,
                )
            elif self.fa_skip_kv_cache:
                # Embedding mode: skip KV cache read and use raw K/V tensors
                # directly via flash_attn_varlen_func. The KV cache write is
                # also skipped (guarded above). This eliminates store_kvcache
                # and prepare_varlen_num_blocks overhead per layer.
                assert k is not None, "fa_skip_kv_cache requires k to be provided"
                assert k_descale is None and v_descale is None, (
                    "fa_skip_kv_cache uses raw K/V tensors, "
                    "FP8 KV cache descaling is not supported in this mode"
                )
                result = self.flash_attn_varlen_func(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k=k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v=v.view(-1, layer.tp_v_head_num, layer.v_head_dim),
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=causal and not use_welm_mtp_swa_compact,
                    window_size=(
                        (-1, -1)
                        if use_welm_mtp_swa_compact
                        else (
                            cascade_prefix_window_size
                            if use_cascade_attn
                            else window_size
                        )
                    ),
                    softcap=layer.logit_cap,
                    num_splits=self.num_splits,
                    out=_fa_out,
                    **kwargs,
                )
            else:
                result = self.flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=(
                        False
                        if use_cascade_attn or use_welm_mtp_swa_compact
                        else causal
                    ),
                    window_size=(
                        (-1, -1)
                        if use_welm_mtp_swa_compact
                        else (
                            cascade_prefix_window_size
                            if use_cascade_attn
                            else window_size
                        )
                    ),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    num_splits=self.num_splits,
                    out=_fa_out,
                    **kwargs,
                )

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = self.flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    # Here metadata_expand.page_table is not divided with page_size.
                    # This is because we loose the fine control of  what token to attend,
                    # but has to attend to some block completely.
                    k_cache=key_cache.view(-1, 1, layer.tp_k_head_num, layer.head_dim),
                    v_cache=value_cache.view(
                        -1, 1, layer.tp_v_head_num, layer.head_dim
                    ),
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=cascade_expand_window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    num_splits=self.num_splits,
                    **cascade_expand_kwargs,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                # Do multi-head attention with chunked prefix cache
                if forward_batch.attn_attend_prefix_cache:
                    assert not get_global_server_args().disable_chunked_prefix_cache
                    # MHA for chunked prefix kv cache when running model with MLA
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert forward_batch.mha_return_lse
                    output = self.flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                        out=_fa_out,
                        **kwargs,
                    )
                else:
                    # MHA for extend part of sequence without attending prefix kv cache
                    cu_seqlens_k = (
                        metadata.cu_seqlens_q
                        if not forward_batch.mha_one_shot
                        else metadata.cu_seqlens_k
                    )
                    max_seqlen_k = (
                        metadata.max_seq_len_q
                        if not forward_batch.mha_one_shot
                        else metadata.max_seq_len_k
                    )
                    output = self.flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=forward_batch.mha_return_lse,
                        out=_fa_out,
                        **kwargs,
                    )
                if forward_batch.mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:
                assert self.fa_impl_ver == 3, "Only FA3 support here"
                # Do absorbed multi-latent attention
                kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).to(q.dtype)
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                result = self.flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    num_splits=self.num_splits,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        self.flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            num_splits=self.num_splits,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert self.fa_impl_ver in [3], "Only FA3 support decoding"
        if self._is_scp_suffix_layer(layer, forward_batch):
            return self._per_suffix_scp_attn_compute(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                metadata=self.forward_metadata,
                sinks=sinks,
                causal=not layer.is_cross_attention,
                save_kv_cache=save_kv_cache,
            )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla and k.shape[0] != cache_loc.numel():
                    pool = getattr(forward_batch, "token_to_kv_pool", None)
                    pool_mapping = getattr(pool, "layers_mapping", None)
                    pool_kind = (
                        pool_mapping.get(layer.layer_id)
                        if isinstance(pool_mapping, dict)
                        else None
                    )
                    raise RuntimeError(
                        "FlashAttention KV decode store shape mismatch before pool write: "
                        f"layer_id={layer.layer_id}, "
                        f"scale_seq_attn_per_suffix={getattr(layer, 'scale_seq_attn_per_suffix', None)}, "
                        f"suffix_parallel={getattr(layer, 'suffix_parallel', None)}, "
                        f"pool_kind={pool_kind}, "
                        f"forward_mode={forward_batch.forward_mode}, "
                        f"q_shape={tuple(q.shape)}, k_shape={tuple(k.shape)}, "
                        f"v_shape={tuple(v.shape)}, cache_loc_numel={cache_loc.numel()}, "
                        f"out_cache_loc_numel={getattr(forward_batch, 'out_cache_loc', cache_loc).numel()}"
                    )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata
        if getattr(layer, "scale_seq_attn_per_suffix", False):
            return self._per_suffix_attn_compute(
                q=q,
                layer=layer,
                forward_batch=forward_batch,
                metadata=metadata,
                sinks=sinks,
                causal=not layer.is_cross_attention,
            )

        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.has_local_attention
            and self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # When Spec Decode enabled, forward_decode would be called with two mode:
        # 1. DRAFT_DECODE: we enable cascade attention when top_k > 1
        # 2. IDLE: we don’t need cascade attention, spec_info will be none in this case
        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        is_swa_layer = self._is_swa_kv_pool_layer(
            layer, forward_batch.token_to_kv_pool
        )
        window_size = (layer.sliding_window_size, 0) if is_swa_layer else (-1, -1)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks
        cascade_expand_kwargs = kwargs
        if use_cascade_attn and sinks is not None:
            # Attention sink is a single global softmax state. Cascade attention
            # splits KV into prefix and expand branches, so include the sink in
            # exactly one branch before merging states.
            cascade_expand_kwargs = dict(kwargs)
            cascade_expand_kwargs.pop("sinks", None)

        _fa_out = (
            forward_batch._attn_output.view(-1, layer.tp_q_head_num, layer.v_head_dim)
            if getattr(forward_batch, "_attn_output", None) is not None
            else None
        )

        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        if not self.use_mla:
            # Do multi-head attention

            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if is_cp_kv_sharded():
                if layer.is_cross_attention:
                    raise NotImplementedError(
                        "CP sharded-KV decode does not support cross attention"
                    )
                if use_local_attn:
                    raise NotImplementedError(
                        "CP sharded-KV decode does not support local attention"
                    )
                if use_cascade_attn:
                    raise NotImplementedError(
                        "CP sharded-KV decode does not support cascade attention yet"
                    )
                if metadata.max_seq_len_q != 1:
                    raise NotImplementedError(
                        "CP sharded-KV decode currently supports only one query per sequence"
                    )
                if k_descale is not None or v_descale is not None:
                    raise NotImplementedError(
                        "CP sharded-KV decode does not support FP8 KV descaling"
                    )

                page_table = metadata.page_table
                page_table_is_physical = False
                if is_swa_layer and self.use_sliding_window_kv_pool:
                    if metadata.swa_page_table is None:
                        raise RuntimeError(
                            "CP sharded-KV Decode is missing its resolved physical "
                            "SWA page table"
                        )
                    page_table = metadata.swa_page_table
                    page_table_is_physical = True
                sched_meta = None
                if metadata.scheduler_metadata is not None and not is_swa_layer:
                    sched_meta = metadata.scheduler_metadata
                o = self._forward_decode_sharded_kv(
                    q,
                    layer,
                    metadata,
                    key_cache,
                    value_cache,
                    window_size=window_size,
                    causal=causal,
                    kwargs=kwargs,
                    out=_fa_out,
                    scheduler_metadata=sched_meta,
                    page_table=page_table,
                    page_table_is_physical=page_table_is_physical,
                )
            elif layer.is_cross_attention:
                # Always use non-chunked logic for cross-attention
                o = self.flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=self.num_splits,
                    **kwargs,
                )
            elif use_local_attn:
                # Use chunked (local) attention batching for self-attention
                o = self.flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    num_splits=self.num_splits,
                    **kwargs,
                )
            else:
                page_table = metadata.page_table
                if (
                    is_swa_layer
                    and self.use_sliding_window_kv_pool
                    and not use_cascade_attn
                ):
                    if metadata.swa_page_table is None:
                        raise RuntimeError(
                            "Hybrid SWA decode is missing its translated page table"
                        )
                    page_table = metadata.swa_page_table
                cache_seqlens = (
                    metadata.swa_cache_seqlens_int32
                    if is_swa_layer
                    and self.use_sliding_window_kv_pool
                    and not use_cascade_attn
                    and metadata.swa_cache_seqlens_int32 is not None
                    else metadata.cache_seqlens_int32
                )
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_q = metadata.max_seq_len_q
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                # Default: single-token self-attention
                # Use precomputed scheduler_metadata when available and applicable.
                # scheduler_metadata is only valid for non-SWA, non-cascade decode.
                sched_meta = None
                if (
                    metadata.scheduler_metadata is not None
                    and not is_swa_layer
                    and not use_cascade_attn
                ):
                    sched_meta = metadata.scheduler_metadata
                result = self.flash_attn_with_kvcache(
                    q=q_reshaped,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    num_splits=self.num_splits,
                    out=_fa_out,
                    scheduler_metadata=sched_meta,
                    **kwargs,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        self.flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            num_splits=self.num_splits,
                            **cascade_expand_kwargs,
                        )
                    )
                    o, _ = merge_state_v2(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            # Do absorbed multi-latent attention
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                q.dtype
            )
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]
            max_seqlen_q = metadata.max_seq_len_q

            result = self.flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,  # softmax_lse is needed for merge states
                num_splits=self.num_splits,
            )
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = self.flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    num_splits=self.num_splits,
                )
                o, _ = merge_state_v2(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _cuda_graph_decode_metadata_key(self, bs: int):
        if self.is_attn_cp_sharded_kv:
            seq_len = getattr(self, "_cuda_graph_seq_len_fill_value", None)
            if seq_len is not None:
                return (int(bs), int(seq_len))
        return int(bs)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Initialize CUDA graph state for the attention backend.

        Args:
            max_bs (int): Maximum batch size to support in CUDA graphs

        This creates fixed-size tensors that will be reused during CUDA graph replay
        to avoid memory allocations.
        """
        decode_max_context_len = (
            self.cuda_graph_max_seq_len
            if self.is_attn_cp_sharded_kv
            else self.max_context_len
        )
        max_num_pages = (decode_max_context_len + self.page_size - 1) // self.page_size

        # This is being used by normal decode and draft decode when topk == 1
        self.decode_cuda_graph_metadata = {
            "cache_seqlens": torch.zeros(max_bs, dtype=torch.int32, device=self.device),
            "cu_seqlens_q": torch.arange(
                0, max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "cu_seqlens_k": torch.zeros(
                max_bs + 1, dtype=torch.int32, device=self.device
            ),
            "page_table": torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            ),
            "strided_indices": torch.arange(
                0, decode_max_context_len, self.page_size, device=self.device
            ),
        }
        # Pre-allocate scheduler_metadata buffer for CUDA graph
        # Size: 1 (semaphore) + round_up(max_bs, 4) * 4 (causal decode vectors)
        if self._get_scheduler_metadata is not None and not self.use_mla:
            b_rounded = ((max_bs + 3) // 4) * 4
            self._sched_meta_buf = torch.zeros(
                1 + b_rounded * 4, dtype=torch.int32, device=self.device
            )
        else:
            self._sched_meta_buf = None

        # Only allocate local attention buffers if local attention is enabled
        # This prevents OOM errors when local attention is not being used
        if self.has_local_attention:
            # Estimate maximum sizes for local attention metadata
            max_seq_len = self.max_context_len
            page_size = self.page_size or 1
            attn_chunk_size = self.attention_chunk_size
            max_virtual_batches = max_bs * (
                (max_seq_len + attn_chunk_size - 1) // attn_chunk_size
            )
            max_pages_per_block = (attn_chunk_size + page_size - 1) // page_size

            self.decode_cuda_graph_local_attn_metadata = {
                "local_query_start_loc": torch.zeros(
                    max_virtual_batches + 1, dtype=torch.int32, device=self.device
                ),
                "local_seqused_k": torch.zeros(
                    max_virtual_batches, dtype=torch.int32, device=self.device
                ),
                "local_block_table": torch.zeros(
                    max_virtual_batches,
                    max_pages_per_block,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        if self.use_sliding_window_kv_pool:
            self.decode_cuda_graph_metadata["swa_page_table"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )
            if not self.is_attn_cp_sharded_kv:
                window_tokens = int(self.sliding_window_size) + 1
                if self.page_size == 1:
                    compact_cap = min(max_num_pages, window_tokens)
                else:
                    compact_cap = min(
                        max_num_pages,
                        (window_tokens + 2 * self.page_size - 2) // self.page_size,
                    )
                self.decode_cuda_graph_metadata["swa_compact_page_table"] = (
                    torch.zeros(
                        max_bs,
                        compact_cap,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )
                self.decode_cuda_graph_metadata["swa_cache_seqlens"] = torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                )
        if is_cp_kv_sharded() and self.use_sliding_window_kv_pool:
            swa_window_tokens = self._attncp_swa_window_tokens()
            if swa_window_tokens is not None:
                self.decode_cuda_graph_metadata["attncp_dense_window_offsets"] = (
                    torch.arange(
                        swa_window_tokens, dtype=torch.long, device=self.device
                    )
                )
                self.decode_cuda_graph_metadata["attncp_dense_window_rows"] = (
                    torch.arange(max_bs, dtype=torch.long, device=self.device)
                )
                self.decode_cuda_graph_metadata[
                    "attncp_dense_window_compact_slots"
                ] = torch.arange(
                    max_bs * swa_window_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ).view(max_bs, swa_window_tokens)
        if is_cp_kv_sharded() and self.enable_attn_cp_decode_local_merge:
            self.decode_cuda_graph_metadata["cp_local_cache_seqlens"] = torch.zeros(
                max_bs, dtype=torch.int32, device=self.device
            )
            self.decode_cuda_graph_metadata["cp_local_page_table"] = torch.zeros(
                max_bs,
                max_num_pages,
                dtype=torch.int32,
                device=self.device,
            )
            swa_window_tokens = self._attncp_swa_window_tokens()
            if (
                self.enable_attn_cp_decode_local_merge_swa
                and swa_window_tokens is not None
            ):
                if self.page_size == 1:
                    swa_local_table_cap = swa_window_tokens
                else:
                    swa_local_table_cap = (
                        swa_window_tokens + 2 * self.page_size - 2
                    ) // self.page_size
                self.decode_cuda_graph_metadata["cp_swa_local_cache_seqlens"] = (
                    torch.zeros(max_bs, dtype=torch.int32, device=self.device)
                )
                self.decode_cuda_graph_metadata["cp_swa_local_page_table"] = torch.zeros(
                    max_bs,
                    swa_local_table_cap,
                    dtype=torch.int32,
                    device=self.device,
                )
                if self.page_size != 1:
                    self.decode_cuda_graph_metadata[
                        "cp_swa_local_page_start_offsets"
                    ] = torch.zeros(
                        max_bs,
                        swa_local_table_cap,
                        dtype=torch.int32,
                        device=self.device,
                    )
                    self.decode_cuda_graph_metadata[
                        "cp_swa_local_page_token_counts"
                    ] = torch.zeros(
                        max_bs,
                        swa_local_table_cap,
                        dtype=torch.int32,
                        device=self.device,
                    )
            self._init_attn_cp_local_merge_cuda_graph_state(max_bs)

        # This is used by draft decode's first half of metadata when topk > 1
        if self.topk > 1:
            self.draft_decode_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    step=self.topk,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            # This is used by draft decode's second half of metadata when topk > 1
            decode_length = self.speculative_step_id + 1
            self.draft_decode_metadata_topk_expand = {
                "cache_seqlens": torch.full(
                    (max_bs * self.topk,),
                    decode_length,
                    device=self.device,
                    dtype=torch.int32,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.topk + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.arange(
                    0,
                    max_bs * self.topk * decode_length + 1,
                    step=decode_length,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.topk,
                    decode_length + 1,  # Additional page for last partial page
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

        target_verify_num_tokens = (
            self.speculative_num_draft_tokens
            if self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
            else max(1, max_num_tokens // max_bs)
        )
        if target_verify_num_tokens > 0:
            # "page_table_draft_decode" will be set only when spec decoding enabled to save memory
            if (
                self.speculative_num_draft_tokens is not None
                and self.speculative_num_draft_tokens > 0
            ):
                self.decode_cuda_graph_metadata["page_table_draft_decode"] = (
                    torch.zeros(
                        max_bs,
                        max_num_pages,
                        dtype=torch.int32,
                        device=self.device,
                    )
                )

            self.target_verify_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * target_verify_num_tokens + 1,
                    step=target_verify_num_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            if self.use_sliding_window_kv_pool:
                self.target_verify_metadata["swa_page_table"] = torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                )

        if (
            self.speculative_num_draft_tokens is not None
            and self.speculative_num_draft_tokens > 0
        ):
            self.draft_extend_metadata = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.zeros(
                    max_bs + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "strided_indices": torch.arange(
                    0, self.max_context_len, self.page_size, device=self.device
                ),
            }

            if self.use_sliding_window_kv_pool:
                self.draft_extend_metadata["swa_page_table"] = torch.zeros(
                    max_bs,
                    max_num_pages,
                    dtype=torch.int32,
                    device=self.device,
                )

        if self.topk > 1:
            self.target_verify_metadata_topk_normal = {
                "cache_seqlens": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
                "page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            self.target_verify_metadata_topk_expand = {
                "cache_seqlens": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_k": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "cu_seqlens_q": torch.arange(
                    0,
                    max_bs * self.speculative_num_draft_tokens + 1,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "page_table": torch.zeros(
                    max_bs * self.speculative_num_draft_tokens,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=self.device,
                ),
            }

            if self.has_swa:
                self.target_verify_metadata_topk_swa = {
                    "cache_seqlens": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "cu_seqlens_k": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens + 1,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "cu_seqlens_q": torch.arange(
                        0,
                        max_bs * self.speculative_num_draft_tokens + 1,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                    "page_table": torch.zeros(
                        max_bs * self.speculative_num_draft_tokens,
                        self.max_context_len,
                        dtype=torch.int32,
                        device=self.device,
                    ),
                }

        # Only allocate encoder metadata for encoder-decoder models
        if self.is_encoder_decoder:
            self.encoder_metadata = {
                "encoder_page_table": torch.zeros(
                    max_bs,
                    self.max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
                "encoder_lens_int32": torch.zeros(
                    max_bs, dtype=torch.int32, device=self.device
                ),
                "encoder_cu_seqlens_k": torch.zeros(
                    max_bs + 1, dtype=torch.int32, device=self.device
                ),
            }
        else:
            # For decoder-only models, skip encoder_metadata allocation
            self.encoder_metadata = {}

    def _get_target_verify_num_tokens(self, num_tokens: int, bs: int) -> int:
        if self.speculative_num_draft_tokens is not None:
            return self.speculative_num_draft_tokens
        return max(1, num_tokens // bs)

    def _get_target_verify_num_tokens_for_replay(
        self, bs: int, spec_info: Optional[SpecInput]
    ) -> int:
        if self.speculative_num_draft_tokens is not None:
            return self.speculative_num_draft_tokens
        replay_forward_batch = getattr(self, "_replay_forward_batch", None)
        return max(1, getattr(replay_forward_batch, "scale_seq_factor", 1))

    @staticmethod
    def _target_verify_kv_lens(
        seq_lens: torch.Tensor,
        num_verify_tokens: int,
        spec_info: Optional[SpecInput],
    ) -> torch.Tensor:
        if spec_info is None:
            return seq_lens
        return seq_lens + num_verify_tokens

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Initialize forward metadata for capturing CUDA graph."""
        metadata = FlashAttentionMetadata()
        metadata.requires_exact_logprob = (
            self._attncp_current_batch_requires_exact_logprob()
        )

        # metadata_expand is needed for Spec Decoding when top k > 1
        metadata_expand = FlashAttentionMetadata()
        metadata_expand.requires_exact_logprob = metadata.requires_exact_logprob

        device = seq_lens.device
        if forward_mode.is_decode_or_idle():
            if spec_info is not None:
                # Draft Decode
                if self.topk <= 1:
                    # When topk = 1, we use the normal decode metadata
                    metadata.cache_seqlens_int32 = self.decode_cuda_graph_metadata[
                        "cache_seqlens"
                    ][:bs]
                    metadata.max_seq_len_k = seq_lens.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = self.decode_cuda_graph_metadata[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = self.decode_cuda_graph_metadata[
                        "page_table_draft_decode"
                    ][:bs, :]
                    if self.use_sliding_window_kv_pool:
                        metadata.swa_page_table = self.decode_cuda_graph_metadata[
                            "swa_page_table"
                        ][:bs, :]
                    self.decode_cuda_graph_metadata[bs] = metadata
                else:
                    # When top k > 1, we need two specific draft decode metadata, and then merge states
                    # 1. The first half of metadata for prefix tokens
                    metadata.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_normal["cache_seqlens"][:bs]
                    )
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = seq_lens.max().item()
                    metadata.cu_seqlens_q = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_q"
                    ][: bs + 1]
                    metadata.cu_seqlens_k = self.draft_decode_metadata_topk_normal[
                        "cu_seqlens_k"
                    ][: bs + 1]
                    metadata.page_table = self.draft_decode_metadata_topk_normal[
                        "page_table"
                    ][:bs, :]

                    # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                    metadata_expand.cache_seqlens_int32 = (
                        self.draft_decode_metadata_topk_expand["cache_seqlens"][
                            : bs * self.topk
                        ]
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_q"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.cu_seqlens_k = (
                        self.draft_decode_metadata_topk_expand["cu_seqlens_k"][
                            : bs * self.topk + 1
                        ]
                    )
                    metadata_expand.page_table = self.draft_decode_metadata_topk_expand[
                        "page_table"
                    ][: bs * self.topk]
                    self.draft_decode_metadata_topk_normal[bs] = metadata
                    self.draft_decode_metadata_topk_expand[bs] = metadata_expand
            else:
                # Normal Decode
                # Get sequence information
                metadata.cache_seqlens_int32 = seq_lens.to(torch.int32)
                batch_size = len(seq_lens)
                device = seq_lens.device
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
                # Precompute maximum sequence length
                metadata.max_seq_len_k = seq_lens.max().item()
                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                # Precompute page table
                page_table = self.decode_cuda_graph_metadata["page_table"][:bs, :]
                if is_cp_kv_sharded():
                    page_table = page_table[:, :max_seq_pages]
                metadata.page_table = page_table
                if self.use_sliding_window_kv_pool:
                    full_swa_page_table = self.decode_cuda_graph_metadata[
                        "swa_page_table"
                    ][:bs, :]
                    if is_cp_kv_sharded():
                        metadata.swa_page_table = full_swa_page_table[
                            :, :max_seq_pages
                        ]
                    else:
                        self._set_decode_swa_metadata(
                            metadata,
                            full_swa_page_table,
                            metadata.cache_seqlens_int32,
                            out_page_table=self.decode_cuda_graph_metadata[
                                "swa_compact_page_table"
                            ][:bs, :],
                            out_cache_seqlens=self.decode_cuda_graph_metadata[
                                "swa_cache_seqlens"
                            ][:bs],
                        )
                # Precompute cumulative sequence lengths
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata_key = self._cuda_graph_decode_metadata_key(bs)
                self.decode_cuda_graph_metadata[metadata_key] = metadata

                self._maybe_update_local_attn_metadata_for_capture(metadata, batch_size)

                # Compute scheduler_metadata into pre-allocated buffer for CUDA graph capture
                if self._sched_meta_buf is not None:
                    sched = self._compute_scheduler_metadata(
                        batch_size,
                        max(metadata.max_seq_len_k, 1),
                        metadata.cache_seqlens_int32,
                        metadata.cu_seqlens_q,
                    )
                    if sched is not None:
                        n = sched.shape[0]
                        self._sched_meta_buf[:n] = sched
                        self._sched_meta_buf[n:] = 0
                        metadata.scheduler_metadata = self._sched_meta_buf[:n]

                if is_cp_kv_sharded() and self.enable_attn_cp_decode_local_merge:
                    self._set_cuda_graph_sharded_kv_decode_metadata(
                        metadata,
                        bs,
                    )
                    if (
                        self.enable_attn_cp_decode_local_merge_swa
                        and
                        self.use_sliding_window_kv_pool
                        and metadata.swa_page_table is not None
                    ):
                        self._set_cuda_graph_sharded_kv_decode_swa_metadata(
                            metadata,
                            bs,
                        )

        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = self.target_verify_metadata[
                    "cache_seqlens"
                ][:bs]
                num_verify_tokens = self._get_target_verify_num_tokens(num_tokens, bs)
                kv_lens = self._target_verify_kv_lens(
                    seq_lens, num_verify_tokens, spec_info
                )
                metadata.cache_seqlens_int32.copy_(kv_lens)

                metadata.max_seq_len_q = num_verify_tokens
                metadata.max_seq_len_k = kv_lens.max().item()

                metadata.cu_seqlens_q = torch.arange(
                    0,
                    bs * num_verify_tokens + 1,
                    num_verify_tokens,
                    dtype=torch.int32,
                    device=device,
                )

                metadata.cu_seqlens_k = self.target_verify_metadata["cu_seqlens_k"][
                    : (bs + 1)
                ]

                metadata.page_table = self.target_verify_metadata["page_table"][:bs, :]

                if self.use_sliding_window_kv_pool:
                    metadata.swa_page_table = self.target_verify_metadata[
                        "swa_page_table"
                    ][:bs, :]

                self.target_verify_metadata[bs] = metadata
            else:
                # When topk > 1, we need two specific target verify metadata, and then merge states
                # 1. The first half of metadata for prefix tokens
                metadata.cache_seqlens_int32 = self.target_verify_metadata_topk_normal[
                    "cache_seqlens"
                ][:bs]
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                # metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item(), do this in replay
                metadata.cu_seqlens_q = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_q"
                ][: bs + 1]
                metadata.cu_seqlens_k = self.target_verify_metadata_topk_normal[
                    "cu_seqlens_k"
                ][: bs + 1]
                metadata.page_table = self.target_verify_metadata_topk_normal[
                    "page_table"
                ][:bs, :]

                # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                metadata_expand.cache_seqlens_int32 = (
                    self.target_verify_metadata_topk_expand["cache_seqlens"][
                        : bs * self.speculative_num_draft_tokens
                    ]
                )
                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_q"
                ][: bs * self.speculative_num_draft_tokens + 1]
                metadata_expand.cu_seqlens_k = self.target_verify_metadata_topk_expand[
                    "cu_seqlens_k"
                ][: bs * self.speculative_num_draft_tokens + 1]

                metadata_expand.page_table = self.target_verify_metadata_topk_expand[
                    "page_table"
                ][: bs * self.speculative_num_draft_tokens]

                self.target_verify_metadata_topk_normal[bs] = metadata
                self.target_verify_metadata_topk_expand[bs] = metadata_expand

                if self.has_swa:
                    metadata_swa = FlashAttentionMetadata()
                    metadata_swa.cache_seqlens_int32 = (
                        self.target_verify_metadata_topk_swa["cache_seqlens"][
                            : bs * self.speculative_num_draft_tokens
                        ]
                    )
                    metadata_swa.max_seq_len_q = 1
                    metadata_swa.cu_seqlens_q = self.target_verify_metadata_topk_swa[
                        "cu_seqlens_q"
                    ][: bs * self.speculative_num_draft_tokens + 1]
                    metadata_swa.cu_seqlens_k = self.target_verify_metadata_topk_swa[
                        "cu_seqlens_k"
                    ][: bs * self.speculative_num_draft_tokens + 1]

                    metadata_swa.page_table = self.target_verify_metadata_topk_swa[
                        "page_table"
                    ][: bs * self.speculative_num_draft_tokens]
                    self.target_verify_metadata_topk_swa[bs] = metadata_swa
                    metadata.swa_spec_metadata = metadata_swa

        elif forward_mode.is_draft_extend(include_v2=True):
            metadata.cache_seqlens_int32 = self.draft_extend_metadata["cache_seqlens"][
                :bs
            ]
            metadata.cache_seqlens_int32.copy_(seq_lens)

            num_tokens_per_bs = num_tokens // bs
            metadata.max_seq_len_q = num_tokens_per_bs
            metadata.max_seq_len_k = seq_lens.max().item()

            metadata.cu_seqlens_q = torch.arange(
                0,
                bs * num_tokens_per_bs + 1,
                num_tokens_per_bs,
                dtype=torch.int32,
                device=device,
            )

            metadata.cu_seqlens_k = self.draft_extend_metadata["cu_seqlens_k"][
                : (bs + 1)
            ]
            metadata.page_table = self.draft_extend_metadata["page_table"][:bs, :]

            if self.use_sliding_window_kv_pool:
                metadata.swa_page_table = self.draft_extend_metadata["swa_page_table"][
                    :bs, :
                ]

            self.draft_extend_metadata[bs] = metadata

        if encoder_lens is not None:
            encoder_bs = encoder_lens.numel()
            metadata.encoder_lens_int32 = self.encoder_metadata["encoder_lens_int32"][
                :encoder_bs
            ]
            metadata.encoder_cu_seqlens_k = self.encoder_metadata[
                "encoder_cu_seqlens_k"
            ][: (encoder_bs + 1)]

            metadata.encoder_page_table = self.encoder_metadata["encoder_page_table"][
                :bs, :
            ]

        self.forward_metadata = metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: Optional[torch.Tensor] = None,
    ):
        """Initialize forward metadata for replaying CUDA graph."""
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]
        device = seq_lens.device
        metadata = None
        metadata_expand = None

        if forward_mode.is_decode_or_idle():

            if spec_info is not None:
                # Draft Decode
                if self.topk <= 1:
                    # When topk = 1, we use the normal decode metadata
                    metadata = self.decode_cuda_graph_metadata[bs]
                    max_len = seq_lens_cpu.max().item()
                    metadata.max_seq_len_k = max_len + self.speculative_step_id + 1
                    max_seq_pages = (
                        metadata.max_seq_len_k + self.page_size - 1
                    ) // self.page_size

                    normal_decode_set_metadata(
                        metadata.cache_seqlens_int32,
                        metadata.cu_seqlens_k,
                        metadata.page_table,
                        self.req_to_token,
                        req_pool_indices,
                        self.decode_cuda_graph_metadata["strided_indices"],
                        max_seq_pages,
                        seq_lens,
                        self.speculative_step_id + 1,
                        self.page_size,
                        metadata.swa_page_table,
                        (
                            self.token_to_kv_pool
                            if self.use_sliding_window_kv_pool
                            else None
                        ),
                    )

                else:
                    # When top k > 1, we need two specific draft decode metadata, and then merge states
                    # 1. The first half of metadata for prefix tokens
                    metadata = self.draft_decode_metadata_topk_normal[bs]
                    if self.page_size > 1:
                        # First attention handles seq_lens - last_page_lens if page size > 1.
                        last_page_lens = seq_lens % self.page_size
                        seq_lens = seq_lens - last_page_lens
                    metadata.cache_seqlens_int32.copy_(seq_lens)
                    # metadata.max_seq_len_q = self.topk, already set in capture
                    # metadata.cu_seqlens_q already set in capture
                    # metadata.cu_seqlens_k is not needed

                    metadata.max_seq_len_k = seq_lens_cpu.max().item()
                    max_seq_pages = (
                        metadata.max_seq_len_k + self.page_size - 1
                    ) // self.page_size
                    strided_indices = self.decode_cuda_graph_metadata["strided_indices"]
                    strided_indices = strided_indices[:max_seq_pages]
                    page_table = (
                        self.req_to_token[
                            req_pool_indices[:, None],  # shape [bs, 1]
                            strided_indices[None, :],  # shape [1, max_seq_pages]
                        ]
                        // self.page_size
                    )
                    metadata.page_table[:, :max_seq_pages].copy_(page_table)
                    # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                    metadata_expand = self.draft_decode_metadata_topk_expand[bs]
                    decode_length = self.speculative_step_id + 1
                    # shape: [bs, num_steps, topk] -> [bs x topk, num_steps]
                    cache_loc = out_cache_loc.view(-1, self.speculative_num_steps)
                    if self.page_size > 1:
                        draft_decode_set_expand_metadata(
                            cache_seqlens_int32=metadata_expand.cache_seqlens_int32,
                            page_table=metadata_expand.page_table,
                            last_page_lens=last_page_lens,
                            decode_length=decode_length,
                            cache_loc=cache_loc,
                            topk=self.topk,
                            page_size=self.page_size,
                        )
                    else:
                        num_seqs = cache_loc.shape[0]
                        metadata_expand.page_table[:num_seqs, :decode_length].copy_(
                            cache_loc[:, :decode_length]
                        )
                # TODO: Handle local attention metadata for draft decode when llama4 eagle is supported
            else:
                # Normal Decode
                metadata_key = self._cuda_graph_decode_metadata_key(bs)
                metadata = self.decode_cuda_graph_metadata[metadata_key]
                max_len = seq_lens_cpu.max().item()
                max_seq_pages = (max_len + self.page_size - 1) // self.page_size
                metadata.max_seq_len_k = max_len
                full_swa_page_table = (
                    self.decode_cuda_graph_metadata["swa_page_table"][:bs, :]
                    if self.use_sliding_window_kv_pool
                    else None
                )

                normal_decode_set_metadata(
                    metadata.cache_seqlens_int32,
                    metadata.cu_seqlens_k,
                    metadata.page_table,
                    self.req_to_token,
                    req_pool_indices,
                    self.decode_cuda_graph_metadata["strided_indices"],
                    max_seq_pages,
                    seq_lens,
                    0,
                    self.page_size,
                    full_swa_page_table,
                    (
                        self.token_to_kv_pool
                        if self.use_sliding_window_kv_pool
                        and not is_cp_kv_sharded()
                        else None
                    ),
                )

                if self.use_sliding_window_kv_pool and is_cp_kv_sharded():
                    full_swa_page_table[:, :max_seq_pages].copy_(
                        self._resolve_sharded_logical_swa_page_table(
                            metadata.page_table[:, :max_seq_pages]
                        )
                    )

                if self.use_sliding_window_kv_pool and not is_cp_kv_sharded():
                    self._set_decode_swa_metadata(
                        metadata,
                        full_swa_page_table,
                        metadata.cache_seqlens_int32,
                        out_page_table=self.decode_cuda_graph_metadata[
                            "swa_compact_page_table"
                        ][:bs, :],
                        out_cache_seqlens=self.decode_cuda_graph_metadata[
                            "swa_cache_seqlens"
                        ][:bs],
                    )

                self._maybe_update_local_attn_metadata_for_replay(
                    metadata,
                    bs,
                )

                if is_cp_kv_sharded() and self.enable_attn_cp_decode_local_merge:
                    self._set_cuda_graph_sharded_kv_decode_metadata(
                        metadata,
                        bs,
                        metadata.page_table[:, :max_seq_pages],
                    )
                    if (
                        self.enable_attn_cp_decode_local_merge_swa
                        and
                        self.use_sliding_window_kv_pool
                        and metadata.swa_page_table is not None
                    ):
                        self._set_cuda_graph_sharded_kv_decode_swa_metadata(
                            metadata,
                            bs,
                            metadata.swa_page_table[:, :max_seq_pages],
                        )

                # Recompute scheduler_metadata into pre-allocated buffer
                if (
                    self._sched_meta_buf is not None
                    and metadata.scheduler_metadata is not None
                ):
                    sched = self._compute_scheduler_metadata(
                        bs,
                        metadata.max_seq_len_k,
                        metadata.cache_seqlens_int32,
                        metadata.cu_seqlens_q,
                    )
                    if sched is not None:
                        n = sched.shape[0]
                        self._sched_meta_buf[:n] = sched
                        self._sched_meta_buf[n:] = 0

        elif forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata = self.target_verify_metadata[bs]
                num_verify_tokens = self._get_target_verify_num_tokens_for_replay(
                    bs, spec_info
                )
                kv_lens = self._target_verify_kv_lens(
                    seq_lens, num_verify_tokens, spec_info
                )
                metadata.cache_seqlens_int32.copy_(kv_lens)

                metadata.max_seq_len_k = kv_lens.max().item()
                metadata.cu_seqlens_k[1:].copy_(
                    torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
                )
                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                page_indices = self.req_to_token[
                    req_pool_indices[:, None],
                    self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
                ]
                if (
                    self.use_sliding_window_kv_pool
                    and metadata.swa_page_table is not None
                ):
                    swa_page_indices = (
                        self.token_to_kv_pool.translate_loc_from_full_to_swa(
                            page_indices
                        )
                    )
                    metadata.swa_page_table[:, :max_seq_pages].copy_(
                        swa_page_indices // self.page_size
                    )
                page_indices //= self.page_size
                metadata.page_table[:, :max_seq_pages].copy_(page_indices)
            else:
                # When topk > 1, we need two specific target verify metadata, and then merge states
                # 1. The first half of metadata for prefix tokens
                metadata = self.target_verify_metadata_topk_normal[bs]
                metadata.cache_seqlens_int32.copy_(seq_lens)
                # metadata.max_seq_len_q = self.speculative_num_draft_tokens, already set in capture
                metadata.max_seq_len_k = seq_lens_cpu.max().item()
                # metadata.cu_seqlens_q already set in capture
                metadata.cu_seqlens_k[1:].copy_(
                    torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
                )
                max_seq_pages = (
                    metadata.max_seq_len_k + self.page_size - 1
                ) // self.page_size
                page_indices = self.req_to_token[
                    req_pool_indices[:, None],
                    self.decode_cuda_graph_metadata["strided_indices"][:max_seq_pages],
                ]
                page_indices //= self.page_size
                metadata.page_table[:, :max_seq_pages].copy_(page_indices)

                # 2. The second half of metadata for draft tokens (per_batch_num_tokens = topk)
                metadata_expand = self.target_verify_metadata_topk_expand[bs]

                # metadata_expand.max_seq_len_q = 1, already set in capture
                # metadata_expand.cu_seqlens_q already set in capture
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)

                cols = offsets.expand(seq_lens.numel(), -1) + seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                # avoid extracting padded seq indices which will be out of boundary
                mask_extraction_indices[
                    :,
                    spec_info.positions.numel() * self.speculative_num_draft_tokens :,
                ].fill_(0)
                mask = spec_info.custom_mask[mask_extraction_indices].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)

                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                keys = torch.where(
                    mask,
                    col_indices,
                    col_indices + self.speculative_num_draft_tokens,
                )
                _, sort_order = torch.sort(keys, dim=1)

                non_masked_page_table = (
                    self.req_to_token[req_pool_indices, :]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)

                metadata_expand.page_table.copy_(
                    non_masked_page_table.gather(1, sort_order)
                )
                metadata_expand.cache_seqlens_int32.copy_(mask.sum(dim=1))
                metadata_expand.cu_seqlens_k[1:].copy_(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32,
                        dim=0,
                        dtype=torch.int32,
                    )
                )
                if self.has_swa:
                    metadata_swa = self.target_verify_metadata_topk_swa[bs]
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand, metadata_swa
                    )

        elif forward_mode.is_draft_extend():
            metadata = self.draft_extend_metadata[bs]
            metadata.max_seq_len_k = seq_lens_cpu.max().item()
            extend_lens = spec_info.num_accept_tokens[:bs]
            if spec_info.num_accept_tokens_cpu:
                metadata.max_seq_len_q = max(spec_info.num_accept_tokens_cpu)
            else:
                metadata.max_seq_len_q = 1
            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            normal_decode_set_metadata(
                metadata.cache_seqlens_int32,
                metadata.cu_seqlens_k,
                metadata.page_table,
                self.req_to_token,
                req_pool_indices,
                self.draft_extend_metadata["strided_indices"],
                max_seq_pages,
                seq_lens,
                0,
                self.page_size,
                metadata.swa_page_table,
                self.token_to_kv_pool if self.use_sliding_window_kv_pool else None,
            )
            # Fixed-width proposal CUDA graphs capture all query rows at the
            # same length, so cu_seqlens_q is already correct.  Retain the
            # generic update only for variable accepted-token layouts.
            if not (
                spec_info.num_accept_tokens_cpu
                and len(set(spec_info.num_accept_tokens_cpu)) == 1
            ):
                metadata.cu_seqlens_q[1:].copy_(
                    torch.cumsum(extend_lens, dim=0, dtype=torch.int32)
                )

        elif forward_mode.is_draft_extend_v2():
            metadata = self.draft_extend_metadata[bs]
            metadata.cache_seqlens_int32.copy_(seq_lens)

            metadata.max_seq_len_k = seq_lens_cpu.max().item()
            metadata.cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.cache_seqlens_int32, dim=0, dtype=torch.int32)
            )

            extend_seq_lens_tensor = getattr(spec_info, "extend_seq_lens_tensor", None)
            extend_seq_lens_cpu = getattr(spec_info, "extend_seq_lens_cpu", None)
            if extend_seq_lens_tensor is not None:
                extend_seq_lens = extend_seq_lens_tensor.to(torch.int32)
            elif extend_seq_lens_cpu is not None:
                extend_seq_lens = torch.as_tensor(
                    extend_seq_lens_cpu,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                default_extend = getattr(
                    spec_info, "num_tokens_per_req", self.speculative_num_steps + 1
                )
                extend_seq_lens = torch.full(
                    (bs,), default_extend, dtype=torch.int32, device=device
                )
                extend_seq_lens_cpu = [default_extend] * bs

            if extend_seq_lens_cpu:
                metadata.max_seq_len_q = int(max(extend_seq_lens_cpu))
            else:
                metadata.max_seq_len_q = getattr(
                    spec_info, "num_tokens_per_req", self.speculative_num_steps + 1
                )

            metadata.cu_seqlens_q[1:].copy_(
                torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32)
            )

            max_seq_pages = (
                metadata.max_seq_len_k + self.page_size - 1
            ) // self.page_size
            page_indices = self.req_to_token[
                req_pool_indices[:, None],
                self.draft_extend_metadata["strided_indices"][:max_seq_pages],
            ]
            if self.use_sliding_window_kv_pool and metadata.swa_page_table is not None:
                swa_page_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    page_indices
                )
                metadata.swa_page_table[:, :max_seq_pages].copy_(
                    swa_page_indices // self.page_size
                )
            metadata.page_table[:, :max_seq_pages].copy_(page_indices // self.page_size)

        if encoder_lens is not None:
            # Only support encoder size 1 for now
            metadata.encoder_max_seq_len_k = encoder_lens[0]
            metadata.encoder_lens_int32.copy_(encoder_lens[:1])
            metadata.encoder_cu_seqlens_k[1:].copy_(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32)
            )

            metadata.encoder_page_table[:, : metadata.encoder_max_seq_len_k].copy_(
                self.req_to_token[req_pool_indices, : metadata.encoder_max_seq_len_k]
            )

            # Update the regular page table
            page_table = self.req_to_token[
                req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]
            metadata.page_table[:, : metadata.max_seq_len_k].copy_(page_table)

        requires_exact_logprob = self._attncp_current_batch_requires_exact_logprob()
        if metadata is not None:
            metadata.requires_exact_logprob = requires_exact_logprob
        if metadata_expand is not None:
            metadata_expand.requires_exact_logprob = requires_exact_logprob

        self.forward_metadata = metadata
        self.forward_metadata_spec_decode_expand = metadata_expand

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        if self.is_attn_cp_sharded_kv:
            return self.cuda_graph_max_seq_len
        return 1

    def _maybe_init_local_attn_metadata(
        self, forwardbatch: ForwardBatch, metadata: FlashAttentionMetadata, device
    ):
        """Centralized utility to initialize local_attn_metadata if chunked attention is enabled."""
        if not self.has_local_attention:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens_int32 = metadata.cache_seqlens_int32
        if self.use_sliding_window_kv_pool:
            page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                metadata.page_table
            )
        else:
            page_table = metadata.page_table
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seq_lens_np = cache_seqlens_int32.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seq_lens_np,
            page_table,
            self.page_size,
        )

        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )
        metadata.local_attn_metadata = local_metadata

    def _maybe_update_local_attn_metadata_for_capture(
        self, metadata: FlashAttentionMetadata, bs: int
    ):
        """Update local attention metadata during CUDA graph capture phase.

        This method calculates the exact buffer sizes needed for local attention metadata
        during the CUDA graph capture phase, optimizing memory usage by creating views of
        pre-allocated buffers with exactly the sizes needed.
        """
        if not self.has_local_attention:
            return

        seq_lens_capture = metadata.cache_seqlens_int32
        max_seq_len = int(seq_lens_capture.max().item())
        page_table_capture = metadata.page_table

        cu_seqlens_q_np = metadata.cu_seqlens_q.cpu().numpy()
        seqlens_np = seq_lens_capture.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local_np,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seqlens_np,
            page_table_capture,
            self.page_size,
        )

        # Get exact dimensions from the calculation
        q_len = len(cu_seqlens_q_local_np)
        k_len = len(seqlens_k_local_np)
        b0 = block_table_local_np.shape[0] if block_table_local_np.shape[0] > 0 else bs
        b1 = block_table_local_np.shape[1] if block_table_local_np.shape[1] > 0 else 1

        # Create views of the pre-allocated buffers with exactly these sizes
        # This is the key optimization - we only use the memory we actually need
        local_query_start_loc = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ][:q_len]

        local_seqused_k = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"][
            :k_len
        ]

        local_block_table = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ][:b0, :b1]

        metadata.local_attn_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=local_query_start_loc,
            local_seqused_k=local_seqused_k,
            local_block_table=local_block_table,
            local_max_query_len=1,
            local_max_seq_len=max_seq_len,
        )

    def _maybe_update_local_attn_metadata_for_replay(
        self,
        metadata: FlashAttentionMetadata,
        bs: int,
    ):
        """Update preallocated local attention metadata in-place before CUDA graph replay."""
        if not self.has_local_attention:
            return

        # Access preallocated buffers
        local_q_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_query_start_loc"
        ]
        local_k_buf = self.decode_cuda_graph_local_attn_metadata["local_seqused_k"]
        local_block_buf = self.decode_cuda_graph_local_attn_metadata[
            "local_block_table"
        ]
        cu_seqlens_q = self.decode_cuda_graph_metadata["cu_seqlens_q"]

        # Create a modified version for local attention that only processes the last token
        # This mimics the normal decode pattern
        cu_seqlens_q = torch.arange(
            bs + 1, device=cu_seqlens_q.device, dtype=cu_seqlens_q.dtype
        )
        seqlens = metadata.cache_seqlens_int32[:bs]
        # Slice the page_table to match the batch size and actual sequence length
        # This serves three important purposes:
        # 1. Ensures we only process the actual batch size (bs) and not the maximum batch size
        # 2. Limits the sequence length to prevent processing padding tokens or garbage values
        # 3. Prevents zeros in the block table which can cause garbage output during replay
        #
        # Without this slicing, the pre-allocated page_table may contain zeros or invalid indices
        # beyond the actual sequence length, leading to incorrect attention calculations
        max_seq_len = int(seqlens.max().item())
        if self.use_sliding_window_kv_pool:
            sliced_page_table = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                metadata.page_table[:bs, :max_seq_len]
            )
        else:
            sliced_page_table = metadata.page_table[:bs, :max_seq_len]

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seqlens_np = seqlens.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seqlens_np,
            sliced_page_table,
            self.page_size,
        )

        # Convert back to tensors
        device = local_q_buf.device
        cu_seqlens_q_local = torch.from_numpy(cu_seqlens_q_local_np).to(device)
        seqlens_k_local = torch.from_numpy(seqlens_k_local_np).to(device)
        block_table_local = block_table_local.to(device)
        # Get sizes
        q_len = cu_seqlens_q_local.shape[0]
        k_len = seqlens_k_local.shape[0]
        b0, b1 = block_table_local.shape

        # In-place updates into preallocated tensors and zero out the unused space
        local_q_buf[:q_len].copy_(cu_seqlens_q_local)
        local_q_buf[q_len:].fill_(0)
        local_k_buf[:k_len].copy_(seqlens_k_local)
        local_k_buf[k_len:].fill_(0)
        local_block_buf[:b0, :b1].copy_(block_table_local)
        local_block_buf[b0:, :].fill_(0)
        local_block_buf[:b0, b1:].fill_(0)

        if metadata.local_attn_metadata is not None:
            lam = metadata.local_attn_metadata
            lam.local_max_query_len = int(seqlens_q_local_np.max())
            lam.local_max_seq_len = int(seqlens_k_local_np.max())

    def _init_sliding_window_attn_spec_metadata(
        self,
        metadata: FlashAttentionMetadata,
        metadata_expand: FlashAttentionMetadata,
        metadata_swa: Optional[FlashAttentionMetadata] = None,
        sliding_window_size: Optional[int] = None,
        compact_prefix: bool = False,
    ):
        # TODO: support page_size > 1 for swa spec
        assert (
            self.page_size == 1
        ), "FlashAttention backend doesn't support topk > 1 speculative decoding with page size > 1 sliding window attention"

        if compact_prefix:
            assert sliding_window_size is not None
            window_tokens = int(sliding_window_size) + 1
            expand_keep = torch.minimum(
                metadata_expand.cache_seqlens_int32,
                torch.full_like(metadata_expand.cache_seqlens_int32, window_tokens),
            )
            prefix_keep = torch.minimum(
                torch.clamp(window_tokens - expand_keep, min=0),
                metadata.cache_seqlens_int32.repeat_interleave(
                    self.speculative_num_draft_tokens
                ),
            )
            cache_seqlens_int32 = prefix_keep + expand_keep
        else:
            cache_seqlens_int32 = (
                metadata.cache_seqlens_int32.repeat_interleave(
                    self.speculative_num_draft_tokens
                )
                + metadata_expand.cache_seqlens_int32
            )
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        bs = cache_seqlens_int32.shape[0]
        page_table_len = (
            int(sliding_window_size) + 1
            if compact_prefix
            else metadata.max_seq_len_k + metadata_expand.page_table.shape[1]
        )
        page_table = (
            metadata.page_table.new_zeros((bs, page_table_len))
            if metadata_swa is None
            else metadata_swa.page_table
        )

        page_table_a = metadata.page_table
        page_table_b = metadata_expand.page_table
        if self.use_sliding_window_kv_pool:
            page_table_a = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                page_table_a
            )
            page_table_b = self.token_to_kv_pool.translate_loc_from_full_to_swa(
                page_table_b
            )

        if compact_prefix:
            prepare_compact_swa_spec_page_table_triton(
                page_table,
                page_table_a,
                page_table_b,
                metadata.cache_seqlens_int32,
                metadata_expand.cache_seqlens_int32,
                int(sliding_window_size),
                self.speculative_num_draft_tokens,
            )
        else:
            prepare_swa_spec_page_table_triton(
                page_table,
                page_table_a,
                page_table_b,
                metadata.cache_seqlens_int32,
                metadata_expand.cache_seqlens_int32,
                self.speculative_num_draft_tokens,
            )

        if metadata_swa is None:
            metadata_swa = FlashAttentionMetadata()
            metadata_swa.max_seq_len_q = 1
            metadata_swa.cu_seqlens_q = metadata_expand.cu_seqlens_q
            metadata_swa.cache_seqlens_int32 = cache_seqlens_int32
            metadata_swa.cu_seqlens_k = cu_seqlens_k
            metadata_swa.page_table = page_table
        else:
            metadata_swa.cache_seqlens_int32.copy_(cache_seqlens_int32)
            metadata_swa.cu_seqlens_k.copy_(cu_seqlens_k)

        metadata.swa_spec_metadata = metadata_swa


@triton.jit
def _prepare_swa_spec_page_table_kernel(
    dst_ptr,
    src_a_ptr,
    src_b_ptr,
    seq_len_a_ptr,
    seq_len_b_ptr,
    dst_stride_m,
    dst_stride_n,
    a_stride_m,
    a_stride_n,
    b_stride_m,
    b_stride_n,
    LEN_A: tl.constexpr,
    LEN_B: tl.constexpr,
    REPEAT_STEP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    idx_a = pid_m // REPEAT_STEP
    idx_b = pid_m
    seq_len_a = tl.load(seq_len_a_ptr + idx_a)
    seq_len_b = tl.load(seq_len_b_ptr + idx_b)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    total_len = seq_len_a + seq_len_b

    if pid_n * BLOCK_N >= total_len:
        return

    mask = offs_n < total_len
    dst = dst_ptr + pid_m * dst_stride_m + offs_n * dst_stride_n

    if (pid_n + 1) * BLOCK_N < seq_len_a:
        a_ptr = src_a_ptr + idx_a * a_stride_m + offs_n * a_stride_n
        a_mask = mask & (offs_n < LEN_A)
        val = tl.load(a_ptr, mask=a_mask, other=0)
        tl.store(dst, val, mask=mask)
    elif pid_n * BLOCK_N >= seq_len_a:
        offs_b = offs_n - seq_len_a
        b_ptr = src_b_ptr + idx_b * b_stride_m + offs_b * b_stride_n
        b_mask = mask & (offs_b < LEN_B)
        val = tl.load(b_ptr, mask=b_mask, other=0)
        tl.store(dst, val, mask=mask)
    else:
        # mixed part
        a_offs = offs_n
        a_mask = (a_offs < seq_len_a) & (a_offs < LEN_A)
        a_ptr = src_a_ptr + idx_a * a_stride_m + a_offs * a_stride_n
        a_val = tl.load(a_ptr, mask=a_mask, other=0)

        b_offs = offs_n - seq_len_a
        b_mask = (b_offs >= 0) & (b_offs < seq_len_b) & (b_offs < LEN_B)
        b_ptr = src_b_ptr + idx_b * b_stride_m + b_offs * b_stride_n
        b_val = tl.load(b_ptr, mask=b_mask, other=0)

        result = tl.where(offs_n < seq_len_a, a_val, b_val)
        tl.store(dst, result, mask=mask)


def prepare_swa_spec_page_table_triton(
    page_table_dst: torch.Tensor,
    page_table_a: torch.Tensor,
    page_table_b: torch.Tensor,  # expand page table
    seq_len_a: torch.Tensor,
    seq_len_b: torch.Tensor,  # expand seq lens
    speculative_num_draft_tokens: int,
):
    # concat page_table and expand page_table by kv seq length
    bs = seq_len_a.numel()
    bs_expand = seq_len_b.numel()
    assert bs_expand == bs * speculative_num_draft_tokens

    LEN_A = page_table_a.shape[1]
    LEN_B = page_table_b.shape[1]
    LEN_OUT = LEN_A + LEN_B
    REPEAT_STEP = speculative_num_draft_tokens
    BLOCK_N = 256

    grid = (bs_expand, triton.cdiv(LEN_OUT, BLOCK_N))
    _prepare_swa_spec_page_table_kernel[grid](
        page_table_dst,
        page_table_a,
        page_table_b,
        seq_len_a,
        seq_len_b,
        page_table_dst.stride(0),
        page_table_dst.stride(1),
        page_table_a.stride(0),
        page_table_a.stride(1),
        page_table_b.stride(0),
        page_table_b.stride(1),
        LEN_A=LEN_A,
        LEN_B=LEN_B,
        REPEAT_STEP=REPEAT_STEP,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


@triton.jit
def _prepare_compact_swa_spec_page_table_kernel(
    dst_ptr,
    src_a_ptr,
    src_b_ptr,
    seq_len_a_ptr,
    seq_len_b_ptr,
    dst_stride_m,
    dst_stride_n,
    a_stride_m,
    a_stride_n,
    b_stride_m,
    b_stride_n,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    LEN_A: tl.constexpr,
    LEN_B: tl.constexpr,
    REPEAT_STEP: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    idx_a = pid_m // REPEAT_STEP
    seq_len_a = tl.load(seq_len_a_ptr + idx_a)
    seq_len_b = tl.load(seq_len_b_ptr + pid_m)
    window_tokens = SLIDING_WINDOW_SIZE + 1
    keep_b = tl.minimum(seq_len_b, window_tokens)
    prefix_budget = window_tokens - keep_b
    keep_a = tl.minimum(seq_len_a, tl.maximum(prefix_budget, 0))
    start_a = seq_len_a - keep_a
    start_b = seq_len_b - keep_b
    total_len = keep_a + keep_b

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if pid_n * BLOCK_N >= total_len:
        return

    mask = offs_n < total_len
    dst = dst_ptr + pid_m * dst_stride_m + offs_n * dst_stride_n
    use_a = offs_n < keep_a

    a_offs = start_a + offs_n
    a_ptr = src_a_ptr + idx_a * a_stride_m + a_offs * a_stride_n
    a_mask = mask & use_a & (a_offs >= 0) & (a_offs < LEN_A)
    a_val = tl.load(a_ptr, mask=a_mask, other=0)

    b_offs = start_b + offs_n - keep_a
    b_ptr = src_b_ptr + pid_m * b_stride_m + b_offs * b_stride_n
    b_mask = mask & (~use_a) & (b_offs >= 0) & (b_offs < LEN_B)
    b_val = tl.load(b_ptr, mask=b_mask, other=0)

    val = tl.where(use_a, a_val, b_val)
    tl.store(dst, val, mask=mask)


def prepare_compact_swa_spec_page_table_triton(
    page_table_dst: torch.Tensor,
    page_table_a: torch.Tensor,
    page_table_b: torch.Tensor,
    seq_len_a: torch.Tensor,
    seq_len_b: torch.Tensor,
    sliding_window_size: int,
    speculative_num_draft_tokens: int,
):
    bs = seq_len_a.numel()
    bs_expand = seq_len_b.numel()
    assert bs_expand == bs * speculative_num_draft_tokens

    LEN_A = page_table_a.shape[1]
    LEN_B = page_table_b.shape[1]
    REPEAT_STEP = speculative_num_draft_tokens
    BLOCK_N = 256
    max_out = min(page_table_dst.shape[1], int(sliding_window_size) + 1)

    grid = (bs_expand, triton.cdiv(max_out, BLOCK_N))
    _prepare_compact_swa_spec_page_table_kernel[grid](
        page_table_dst,
        page_table_a,
        page_table_b,
        seq_len_a,
        seq_len_b,
        page_table_dst.stride(0),
        page_table_dst.stride(1),
        page_table_a.stride(0),
        page_table_a.stride(1),
        page_table_b.stride(0),
        page_table_b.stride(1),
        SLIDING_WINDOW_SIZE=int(sliding_window_size),
        LEN_A=LEN_A,
        LEN_B=LEN_B,
        REPEAT_STEP=REPEAT_STEP,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )


class FlashAttentionMultiStepBackend:

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        fa_impl_ver: int = 3,
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    fa_impl_ver=fa_impl_ver,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        for i in range(self.speculative_num_steps - 1):
            # TODO: incrementally update the metadata for the later steps,
            # so that they do not need to recompute everything from scratch.
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
            )


@triton.jit
def _fused_metadata_kernel_general(
    # Input tensors
    seq_lens,
    seq_lens_stride_0,
    req_to_token,
    req_to_token_stride_0,
    req_to_token_stride_1,
    req_pool_indices,
    req_pool_indices_stride_0,
    # Output buffers
    cache_seqlens_int32,
    cache_seqlens_int32_stride_0,
    cu_seqlens_k,
    cu_seqlens_k_stride_0,
    page_table,
    page_table_stride_0,
    page_table_stride_1,
    swa_page_table,
    swa_page_table_stride_0,
    swa_page_table_stride_1,
    full_to_swa_mapping,
    full_to_swa_mapping_stride_0,
    # Scalar parameters
    B,
    max_seq_pages,
    page_size: tl.constexpr,
    seq_len_delta: tl.constexpr,
    use_swa: tl.constexpr,
    SHIFT: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # column chunk index

    # 1. Prefix sum (only one block does it)
    if pid_b == 0 and pid_c == 0:
        acc = 0
        for idx in range(B):
            seq = tl.load(seq_lens + idx * seq_lens_stride_0)
            val = (seq + seq_len_delta).to(tl.int32)
            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)
            tl.store(cu_seqlens_k + idx * cu_seqlens_k_stride_0, acc)
            acc += val
        tl.store(cu_seqlens_k + B * cu_seqlens_k_stride_0, acc)

    # 2. Gather for this batch and column chunk
    if max_seq_pages == 0:
        return

    i = pid_b
    # Load row index for this batch (all threads in block have same i)
    row_idx = tl.load(req_pool_indices + i * req_pool_indices_stride_0)
    row_offset = row_idx * req_to_token_stride_0

    col_start = pid_c * BLOCK_COLS
    col_offsets = col_start + tl.arange(0, BLOCK_COLS)
    mask = col_offsets < max_seq_pages

    # Compute column indices in the source tensor (token offset)
    if page_size == 1:
        col_idx = col_offsets
    else:
        col_idx = col_offsets << SHIFT  # faster than multiplication for power-of-two

    # Load page indices from req_to_token
    rt_offsets = row_offset + col_idx * req_to_token_stride_1
    page_index = tl.load(
        req_to_token + rt_offsets, mask=mask, other=0, cache_modifier=".cg"
    )

    # Compute page_table
    if page_size == 1:
        page_table_val = page_index
    else:
        page_table_val = page_index >> SHIFT

    # Store to page_table
    pt_offsets = i * page_table_stride_0 + col_offsets * page_table_stride_1
    tl.store(page_table + pt_offsets, page_table_val, mask=mask, cache_modifier=".cg")

    if use_swa:
        swa_slot = tl.load(
            full_to_swa_mapping + page_index * full_to_swa_mapping_stride_0,
            mask=mask,
            other=0,
            cache_modifier=".cg",
        )
        if page_size == 1:
            swa_val = swa_slot
        else:
            swa_val = swa_slot >> SHIFT
        swa_offsets = (
            i * swa_page_table_stride_0 + col_offsets * swa_page_table_stride_1
        )
        tl.store(swa_page_table + swa_offsets, swa_val, mask=mask, cache_modifier=".cg")


@triton.jit
def _fused_metadata_kernel_ps1_no_swa(
    # Input tensors
    seq_lens,
    seq_lens_stride_0,
    req_to_token,
    req_to_token_stride_0,
    req_to_token_stride_1,
    req_pool_indices,
    req_pool_indices_stride_0,
    # Output buffers
    cache_seqlens_int32,
    cache_seqlens_int32_stride_0,
    cu_seqlens_k,
    cu_seqlens_k_stride_0,
    page_table,
    page_table_stride_0,
    page_table_stride_1,
    # Scalar parameters
    B,
    max_seq_pages,
    seq_len_delta: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_c = tl.program_id(1)  # column chunk index

    # 1. Prefix sum (only one block does it)
    if pid_b == 0 and pid_c == 0:
        acc = 0
        for idx in range(B):
            seq = tl.load(seq_lens + idx * seq_lens_stride_0)
            val = (seq + seq_len_delta).to(tl.int32)
            tl.store(cache_seqlens_int32 + idx * cache_seqlens_int32_stride_0, val)
            tl.store(cu_seqlens_k + idx * cu_seqlens_k_stride_0, acc)
            acc += val
        tl.store(cu_seqlens_k + B * cu_seqlens_k_stride_0, acc)

    # 2. Gather for this batch and column chunk
    if max_seq_pages == 0:
        return

    i = pid_b
    # Load row index for this batch (all threads in block have same i)
    row_idx = tl.load(req_pool_indices + i * req_pool_indices_stride_0)
    row_offset = row_idx * req_to_token_stride_0

    col_start = pid_c * BLOCK_COLS
    col_offsets = col_start + tl.arange(0, BLOCK_COLS)
    mask = col_offsets < max_seq_pages

    # page_size = 1: col_idx = col_offsets
    rt_offsets = row_offset + col_offsets * req_to_token_stride_1
    page_index = tl.load(
        req_to_token + rt_offsets, mask=mask, other=0, cache_modifier=".cg"
    )

    # page_table = page_index // 1 = page_index
    pt_offsets = i * page_table_stride_0 + col_offsets * page_table_stride_1
    tl.store(page_table + pt_offsets, page_index, mask=mask, cache_modifier=".cg")


# Fused Triton kernel implementation
def normal_decode_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    strided_indices: torch.Tensor,
    max_seq_pages: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_len_delta: int,
    page_size: int,
    swa_page_table: Optional[torch.Tensor] = None,
    token_to_kv_pool: Optional[SWAKVPool] = None,
):
    """
    Fused Triton implementation that replaces 4-5 sequential CUDA kernels with 1-2 kernels:
      1. cache_seqlens = seq_lens + seq_len_delta (int64→int32 cast)
      2. cu_seqlens_k = cumsum(cache_seqlens) (prefix-sum)
      3. page_indices = req_to_token[pool_idx, stride_idx] (2-D gather)
      4. page_table = page_indices // page_size (floor-divide)
      5. (optional) swa_page_table for sliding window attention

    Achieves ~5.2x speedup on H200 hardware for typical decode workloads.
    """
    assert (
        page_size > 0 and (page_size & (page_size - 1)) == 0
    ), f"page_size must be a power of two, got {page_size}"

    batch_size = cache_seqlens_int32.shape[0]
    device = seq_lens.device

    # Ensure contiguous memory layout for efficient Triton access
    seq_lens = seq_lens.contiguous()
    req_to_token = req_to_token.contiguous()
    req_pool_indices = req_pool_indices.contiguous()

    # Prepare tensor strides
    seq_lens_stride_0 = seq_lens.stride(0)
    req_to_token_stride_0 = req_to_token.stride(0)
    req_to_token_stride_1 = req_to_token.stride(1)
    req_pool_indices_stride_0 = req_pool_indices.stride(0)
    cache_seqlens_int32_stride_0 = cache_seqlens_int32.stride(0)
    cu_seqlens_k_stride_0 = cu_seqlens_k.stride(0)
    page_table_stride_0 = page_table.stride(0)
    page_table_stride_1 = page_table.stride(1)

    # Check if we should use the specialized fast path for page_size=1, no SWA
    use_swa = swa_page_table is not None and token_to_kv_pool is not None

    if page_size == 1 and not use_swa:
        # Specialized kernel for the common case (page_size=1, no SWA)
        BLOCK_COLS = 256
        if max_seq_pages == 0:
            grid = (1, 1)
        else:
            num_blocks_j = triton.cdiv(max_seq_pages, BLOCK_COLS)
            grid = (batch_size, num_blocks_j)

        _fused_metadata_kernel_ps1_no_swa[grid](
            seq_lens,
            seq_lens_stride_0,
            req_to_token,
            req_to_token_stride_0,
            req_to_token_stride_1,
            req_pool_indices,
            req_pool_indices_stride_0,
            cache_seqlens_int32,
            cache_seqlens_int32_stride_0,
            cu_seqlens_k,
            cu_seqlens_k_stride_0,
            page_table,
            page_table_stride_0,
            page_table_stride_1,
            batch_size,
            max_seq_pages,
            seq_len_delta,
            BLOCK_COLS=BLOCK_COLS,
            num_warps=8,
            num_stages=3,
        )
    else:
        # General kernel for page_size > 1 or SWA cases
        # SWA parameters
        if use_swa:
            assert isinstance(token_to_kv_pool, SWAKVPool)
            swa_page_table = swa_page_table.contiguous()
            swa_page_table_stride_0 = swa_page_table.stride(0)
            swa_page_table_stride_1 = swa_page_table.stride(1)
            # Extract the full_to_swa_index_mapping from token_to_kv_pool
            full_to_swa_mapping = (
                token_to_kv_pool.full_to_swa_index_mapping.contiguous()
            )
            full_to_swa_mapping_stride_0 = full_to_swa_mapping.stride(0)
        else:
            # Dummy tensors (not used)
            swa_page_table = torch.empty(0, dtype=torch.int32, device=device)
            swa_page_table_stride_0 = 0
            swa_page_table_stride_1 = 0
            full_to_swa_mapping = torch.empty(0, dtype=torch.int32, device=device)
            full_to_swa_mapping_stride_0 = 0

        # Kernel configuration
        BLOCK_COLS = 128
        shift = (page_size).bit_length() - 1 if page_size > 1 else 0

        if max_seq_pages == 0:
            grid = (1, 1)
        else:
            num_blocks_j = triton.cdiv(max_seq_pages, BLOCK_COLS)
            grid = (batch_size, num_blocks_j)

        _fused_metadata_kernel_general[grid](
            seq_lens,
            seq_lens_stride_0,
            req_to_token,
            req_to_token_stride_0,
            req_to_token_stride_1,
            req_pool_indices,
            req_pool_indices_stride_0,
            cache_seqlens_int32,
            cache_seqlens_int32_stride_0,
            cu_seqlens_k,
            cu_seqlens_k_stride_0,
            page_table,
            page_table_stride_0,
            page_table_stride_1,
            swa_page_table,
            swa_page_table_stride_0,
            swa_page_table_stride_1,
            full_to_swa_mapping,
            full_to_swa_mapping_stride_0,
            batch_size,
            max_seq_pages,
            page_size,
            seq_len_delta,
            use_swa,
            shift,
            BLOCK_COLS=BLOCK_COLS,
            num_warps=4,
            num_stages=3,
        )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def draft_decode_set_expand_metadata(
    cache_seqlens_int32: torch.Tensor,  # Modifies
    page_table: torch.Tensor,  # Modifies
    last_page_lens: torch.Tensor,
    decode_length: int,
    cache_loc: torch.Tensor,
    topk: int,
    page_size: int,
):
    expanded_last_page_lens = last_page_lens.repeat_interleave(topk)
    cache_seqlens_int32.copy_(decode_length + expanded_last_page_lens)
    cache_loc = (cache_loc // page_size).to(torch.int32)
    if cache_loc.dim() == 1:
        cache_loc = cache_loc.unsqueeze(0)
    # Vectorized torch.unique_consecutive: track value change points then scatter
    mask = torch.ones_like(cache_loc, dtype=torch.bool)
    mask[:, 1:] = cache_loc[:, 1:] != cache_loc[:, :-1]
    positions = mask.cumsum(dim=1) - 1
    num_seqs = cache_loc.shape[0]
    page_table[:num_seqs, :].scatter_(1, positions, cache_loc)


# Copied from:
# https://github.com/houseroad/vllm/blob/4e45bfcaf928bdb9bd952b4ac922a3c205589ae8/vllm/v1/attention/backends/flash_attn.py
#
# Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
# local attention blocks, where each block is passed to the attention kernel
# as an independent local ("virtual") batch item.
#
# For example, if are performing a chunked prefill a batch of 3 sequences:
#   q_seqlens  = [4, 10, 5]
#   kv_seqlens = [6, 17, 9]
# Then normally for regular attention we would compute with an attention mask
#  for batch idx 0 (q_seqlens = 4, kv_seqlens = 6) like:
#   batch idx: 0 (q_seqlens = 4, kv_seqlens = 6)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 | 1 1 1 1 1
#               3 | 1 1 1 1 1 1
#
# for local attention (with attn_chunk_size = 4) we would compute with an
#  attention mask like:
#   batch idx: 0  (q_seqlens = 4, kv_seqlens = 6, attn_chunk_size = 4)
#        k_toks >   0 1 2 3 4 5
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#               2 |         1
#               3 |         1 1
#
# We can simulate this mask using standard flash-attention by breaking the
#  sequences into local ("virtual") batches, where each local batch item is a
#  local attention block, so in this case batch idx 0 would be broken up into:
#
#   local-batch idx: 0 (q_seqlens = 2, kv_seqlens = 4)  (batch 0)
#        k_toks >   0 1 2 3
#        q_toks v  _____________
#               0 | 1 1 1
#               1 | 1 1 1 1
#   local-batch idx: 1 (q_seqlens = 2, kv_seqlens = 2) (batch 0)
#        k_toks >   4 5
#        q_toks v  _____________
#               2 | 1
#               3 | 1 1
#
# e.g. if we have:
#   attn_chunk_size = 4
#   query_start_loc_np = [0, 4, 14, 19] (q_seqlens = [4, 10, 5])
# Then this function would return:
#                           __b0__  ______b1______  __b2__ < orig batch indices
#   q_seqlens_local    = [   2,  2,  1,  4,  4,  1,  4,  1]
#   cu_seqlens_q_local = [0, 4,  6, 10, 14, 18, 19, 23, 24]
#   seqlens_k_local    = [   4,  2,  4,  4,  4,  1,  4,  1]
#   block_table_local  : shape[local_virtual_batches, pages_per_local_batch]
def make_local_attention_virtual_batches(
    attn_chunk_size: int,
    query_start_loc_np: np.ndarray,
    seq_lens_np: np.ndarray,
    block_table: torch.Tensor,
    page_size: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
    """
    Take in `query_start_loc_np` and `seq_lens_np` and break the sequences into
    local attention blocks, where each block is passed to the attention kernel
    as an independent local ("virtual") batch item.

    Args:
        attn_chunk_size: Size of local attention chunks
        query_start_loc_np: Cumulative sum of query lengths (numpy array)
        seq_lens_np: Sequence lengths (numpy array)
        block_table: Block table for KV cache
        page_size: Size of each page in the KV cache

    Returns:
        seqlens_q_local: Query sequence lengths for local attention
        cu_seqlens_q_local: Cumulative sum of query sequence lengths for local attention
        seqlens_k_local: Key sequence lengths for local attention
        block_table_local: Block table for local attention
    """
    # Adjust attention_chunk_size based on the actual sequence length
    # to avoid index out of bounds errors
    max_seq_len = seq_lens_np.max()
    effective_chunk_size = min(attn_chunk_size, max_seq_len)
    # Make sure effective_chunk_size is divisible by page_size
    effective_chunk_size = (effective_chunk_size // page_size) * page_size
    if effective_chunk_size < page_size:
        effective_chunk_size = page_size
    attn_chunk_size = effective_chunk_size

    q_seqlens = query_start_loc_np[1:] - query_start_loc_np[:-1]
    actual_batch_size = seq_lens_np.shape[0]

    # Handle if we are starting in the middle of a local attention block,
    #  we assume q_seqlens > 0 (for all elements), for each batch idx we compute
    #  the number of tokens that are not in the first local attention block and
    #  then we can simply use a cdiv for the rest.
    # For example if we have:
    #   attn_chunk_size = 4
    #   q_seqlens = [4, 10, 5]
    #   k_seqlens = [6, 17, 9]
    # Then we would get:
    #   new_tokens_in_first_block = [2, 1, 4]
    #   local_blocks = [2, 4, 2]
    q_tokens_in_first_block = np.minimum(
        attn_chunk_size - ((seq_lens_np - q_seqlens) % attn_chunk_size), q_seqlens
    ).astype(np.int32)
    tokens_in_last_block = attn_chunk_size + (seq_lens_np % -attn_chunk_size)
    local_blocks = 1 + cdiv(q_seqlens - q_tokens_in_first_block, attn_chunk_size)

    # Once we know the number of local blocks we can compute the request spans
    #  for each batch idx, we can figure out the number of "virtual" requests we
    #  have to make,
    # For the above example we would get:
    #   seqlens_q_local = [2, 2, 1, 4, 4, 1, 4, 1]
    #
    # First Get batched arange. (E.g., [2, 4, 2] -> [0, 1, 0, 1, 2, 3, 0, 1])
    #   (TODO: max a utility to share this code with _prepare_inputs)
    # arange step 1. [2, 4, 2] -> [2, 6, 8]
    cu_num_blocks = np.cumsum(local_blocks)
    virtual_batches = cu_num_blocks[-1]
    # arange step 2. [2, 6, 8] -> [0, 0, 2, 2, 2, 2, 6, 6]
    block_offsets = np.repeat(cu_num_blocks - local_blocks, local_blocks)
    # arange step 3. [0, 1, 0, 1, 2, 3, 0, 1]
    arange = np.arange(virtual_batches, dtype=np.int32) - block_offsets
    # also compute reverse arange (i.e. [1, 0, 3, 2, 1, 0, 1, 0])
    rarange = np.repeat(local_blocks, local_blocks) - arange - 1
    # Then we can compute the seqlens_q_local, handling the fact that the
    #  first and last blocks could be partial
    seqlens_q_local = np.repeat(q_seqlens - q_tokens_in_first_block, local_blocks)
    # set the first block since this may be a partial block
    seqlens_q_local[arange == 0] = q_tokens_in_first_block
    # set the remaining blocks
    seqlens_q_local[arange > 0] = np.minimum(
        seqlens_q_local - attn_chunk_size * (arange - 1), attn_chunk_size
    )[arange > 0]

    # convert from q_seqlens to cu_seqlens_q
    cu_seqlens_q_local = np.pad(np.cumsum(seqlens_q_local), (1, 0)).astype(np.int32)

    # compute the seqlens_k_local,
    #  basically a full local attention block for all but the last block in each
    #  batch
    # For our example this will be:
    #   seqlens_k_local = [4, 2, 4, 4, 4, 1, 4, 1]
    seqlens_k_local = np.full(cu_num_blocks[-1], attn_chunk_size, dtype=np.int32)
    seqlens_k_local[cu_num_blocks - 1] = tokens_in_last_block

    k_seqstarts_absolute = np.repeat(seq_lens_np, local_blocks) - (
        rarange * attn_chunk_size + np.repeat(tokens_in_last_block, local_blocks)
    )
    # For the example the local attention blocks start at:
    #                           _b0_  _____b1_____  _b2_
    #   k_seqstarts_absolute = [0, 4, 4, 8, 12, 16, 4, 8]
    block_starts = k_seqstarts_absolute // page_size

    assert attn_chunk_size % page_size == 0, (
        f"attn_chunk_size {attn_chunk_size} is not "
        f"divisible by page_size {page_size}"
    )
    pages_per_local_batch = attn_chunk_size // page_size

    # Create a block_table for the local attention blocks
    # For out example if we have a block-table like (assuming page_size=2):
    #   block_table = [
    #     [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],  < batch 0
    #     [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  < batch 1
    #     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  < batch 2
    #   ]
    # Then for the local batches we would want a block-table like
    #   block_table_local = [
    #     [  0,  1 ], < local-batch 0, (batch 0, starting from k[0])
    #     [  2,  3 ], < local-batch 1, (batch 0, starting from k[4])
    #     [ 12, 13 ], < local-batch 2, (batch 1, starting from k[4])
    #     [ 14, 15 ], < local-batch 3, (batch 1, starting from k[8])
    #     [ 16, 17 ], < local-batch 4, (batch 1, starting from k[12])
    #     [ 18, 19 ], < local-batch 5, (batch 1, starting from k[16])
    #     [ 22, 23 ], < local-batch 6, (batch 2, starting from k[4])
    #     [ 24, 25 ], < local-batch 7, (batch 2, starting from k[8])
    #   ]
    block_indices = np.broadcast_to(
        np.arange(pages_per_local_batch, dtype=np.int32),
        (virtual_batches, pages_per_local_batch),
    ) + np.expand_dims(block_starts, axis=1)
    # Ensure block_indices doesn't exceed block_table dimensions
    # This is a critical safety check that prevents index out of bounds errors
    # when dealing with large sequences (>8192 tokens) or when the block_table
    # dimensions are smaller than what would be needed for the full attention chunk size.
    block_indices = block_indices.flatten().clip(max=block_table.shape[1] - 1)
    batch_indices = np.repeat(
        np.arange(actual_batch_size, dtype=np.int32),
        local_blocks * pages_per_local_batch,
    )

    # NOTE: https://github.com/pytorch/pytorch/pull/160256 causes performance
    # regression when using numpy arrays (batch and block indices) to index into
    # torch tensor (block_table). As a workaround, convert numpy arrays to torch
    # tensor first, which recovers perf.
    batch_indices_torch = torch.from_numpy(batch_indices)
    block_indices_torch = torch.from_numpy(block_indices)
    block_table_local = block_table[batch_indices_torch, block_indices_torch].view(
        virtual_batches, -1
    )

    return seqlens_q_local, cu_seqlens_q_local, seqlens_k_local, block_table_local


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


# TODO(hebiao064): remove this once we have a better way to handle the merge_state_v2 torch.compile issue
@torch._dynamo.disable()
def merge_state_v2_wrapper(o, s_a, o_exp, s_b):
    return merge_state_v2(o, s_a, o_exp, s_b)
