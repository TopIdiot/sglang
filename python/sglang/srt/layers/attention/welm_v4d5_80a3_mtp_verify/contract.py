from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

ENV_NAME = "SGLANG_WELM_V4D5_80A3_MTP_VERIFY_ATTENTION_BACKEND"

QUERY_TOKENS = 4
Q_HEADS = 6
KV_HEADS = 1
HEAD_DIM = 256
PAGE_SIZE = 16
SUPPORTED_WINDOWS = {(-1, -1), (512, 0)}

AUTO_DATAFLOW = "auto"
WARP_DATAFLOW = "warp_mma_independent_q"
WGMMA_DATAFLOW = "wgmma_kv64_n24"

_MODEL_SIGNATURE = {
    "model_type": "welmv4_moe",
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": HEAD_DIM,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "num_nextn_predict_layers": 4,
}


class VerifyAttentionBackendChoice(str, Enum):
    FA3 = "fa3"
    MK = "mk"


class FallbackReason(str, Enum):
    MODEL_SIGNATURE = "model_signature_mismatch"
    FALLBACK_BACKEND = "fallback_backend_not_fa3"
    TARGET_WORKER = "not_target_worker"
    SPECULATIVE_ALGORITHM = "unsupported_speculative_algorithm"
    SPECULATIVE_CONFIG = "unsupported_speculative_config"
    TP_SIZE = "unsupported_tp_size"
    DP_ATTENTION = "unsupported_dp_attention"
    ATTN_CP = "unsupported_attention_context_parallel"
    PAGE_SIZE = "unsupported_page_size"
    KV_DTYPE = "unsupported_kv_cache_dtype"
    DEVICE = "unsupported_device"
    FORWARD_MODE = "not_target_verify"
    SPEC_METADATA = "invalid_verify_metadata"
    LAYOUT = "unsupported_attention_layout"
    WINDOW = "unsupported_window"
    SOFTCAP = "unsupported_softcap"
    HEAD_SHAPE = "unsupported_local_head_shape"
    QUERY = "invalid_query_tensor"
    KV_CACHE = "invalid_kv_cache"
    PAGE_TABLE = "invalid_page_table"
    CACHE_LENGTH = "invalid_cache_length"
    SINKS = "invalid_attention_sinks"
    METADATA_STATE = "stale_backend_metadata"
    MK_UNAVAILABLE = "mk_unavailable"
    SELF_CHECK = "mk_fa3_self_check_failed"
    PLAN = "mk_plan_failed"
    GRAPH_PLAN = "mk_cuda_graph_plan_unavailable"
    INTERNAL = "mk_internal_support_check_failed"


@dataclass(frozen=True)
class SupportDecision:
    supported: bool
    reason: FallbackReason | None = None
    actual: Any = None
    expected: Any = None
    detail: str | None = None

    @classmethod
    def allow(cls) -> SupportDecision:
        return cls(True)

    @classmethod
    def reject(
        cls,
        reason: FallbackReason,
        *,
        actual: Any = None,
        expected: Any = None,
        detail: str | None = None,
    ) -> SupportDecision:
        return cls(
            False,
            reason=reason,
            actual=actual,
            expected=expected,
            detail=detail,
        )


def get_backend_choice() -> VerifyAttentionBackendChoice:
    raw = os.getenv(ENV_NAME, VerifyAttentionBackendChoice.FA3.value)
    normalized = raw.strip().lower()
    try:
        return VerifyAttentionBackendChoice(normalized)
    except ValueError as exc:
        choices = ", ".join(item.value for item in VerifyAttentionBackendChoice)
        raise ValueError(
            f"Invalid {ENV_NAME}={raw!r}; expected one of: {choices}"
        ) from exc


def is_mk_backend_requested() -> bool:
    return get_backend_choice() is VerifyAttentionBackendChoice.MK


def _hf_config(model_config):
    return getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )


def model_signature(model_config) -> dict[str, Any]:
    config = _hf_config(model_config)
    if config is None:
        return {name: None for name in _MODEL_SIGNATURE}
    return {name: getattr(config, name, None) for name in _MODEL_SIGNATURE}


def is_welm_v4d5_80a3(model_config) -> bool:
    config = _hf_config(model_config)
    if config is None:
        return False
    architectures = tuple(getattr(config, "architectures", None) or ())
    if "WeLMV4MoeForCausalLM" not in architectures:
        return False
    return model_signature(model_config) == _MODEL_SIGNATURE


def validate_static_contract(model_runner, fallback_backend) -> SupportDecision:
    if getattr(model_runner, "is_draft_worker", False):
        return SupportDecision.reject(FallbackReason.TARGET_WORKER)

    signature = model_signature(model_runner.model_config)
    if not is_welm_v4d5_80a3(model_runner.model_config):
        return SupportDecision.reject(
            FallbackReason.MODEL_SIGNATURE,
            actual=signature,
            expected=_MODEL_SIGNATURE,
        )

    target_backend = fallback_backend
    for _ in range(4):
        prefill = getattr(target_backend, "prefill_backend", None)
        decode = getattr(target_backend, "decode_backend", None)
        if prefill is None or decode is None:
            break
        mode = getattr(
            model_runner.server_args, "speculative_attention_mode", "prefill"
        )
        target_backend = decode if mode == "decode" else prefill
    if target_backend.__class__.__name__ != "FlashAttentionBackend":
        return SupportDecision.reject(
            FallbackReason.FALLBACK_BACKEND,
            actual=target_backend.__class__.__name__,
            expected="FlashAttentionBackend",
        )

    server_args = model_runner.server_args
    if getattr(server_args, "speculative_algorithm", None) != "EAGLE":
        return SupportDecision.reject(
            FallbackReason.SPECULATIVE_ALGORITHM,
            actual=getattr(server_args, "speculative_algorithm", None),
            expected="EAGLE",
        )
    speculative = (
        getattr(server_args, "speculative_num_steps", None),
        getattr(server_args, "speculative_eagle_topk", None),
        getattr(server_args, "speculative_num_draft_tokens", None),
    )
    if speculative != (3, 1, QUERY_TOKENS):
        return SupportDecision.reject(
            FallbackReason.SPECULATIVE_CONFIG,
            actual=speculative,
            expected=(3, 1, QUERY_TOKENS),
        )
    if int(getattr(model_runner, "tp_size", -1)) != 4:
        return SupportDecision.reject(
            FallbackReason.TP_SIZE,
            actual=getattr(model_runner, "tp_size", None),
            expected=4,
        )
    # Ordinary data parallelism creates independent TP replicas.  Every
    # replica owns its ModelRunner, KV cache, page tables, and CUDA graphs, so
    # it preserves the local tensor contract consumed by the MK kernel.
    # DP-attention is different: it redistributes attention work/tokens across
    # DP ranks and has not been validated for this model-specific fast path.
    if bool(getattr(server_args, "enable_dp_attention", False)):
        return SupportDecision.reject(
            FallbackReason.DP_ATTENTION,
            actual=True,
            expected=False,
        )
    if int(getattr(model_runner, "attn_cp_size", 1)) != 1:
        return SupportDecision.reject(
            FallbackReason.ATTN_CP,
            actual=getattr(model_runner, "attn_cp_size", None),
            expected=1,
        )
    if int(getattr(model_runner, "page_size", -1)) != PAGE_SIZE:
        return SupportDecision.reject(
            FallbackReason.PAGE_SIZE,
            actual=getattr(model_runner, "page_size", None),
            expected=PAGE_SIZE,
        )
    kv_dtype = getattr(model_runner, "kv_cache_dtype", None)
    if kv_dtype is not torch.bfloat16 and "bfloat16" not in str(kv_dtype):
        return SupportDecision.reject(
            FallbackReason.KV_DTYPE,
            actual=str(kv_dtype),
            expected="torch.bfloat16",
        )
    return SupportDecision.allow()


def validate_runtime_contract(
    *,
    model_runner,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_table: torch.Tensor,
    layer,
    forward_batch,
    window_size: tuple[int, int],
    sinks: torch.Tensor | None,
    causal: bool,
    has_unsupported_layout: bool,
    metadata_ready: bool,
) -> SupportDecision:
    if has_unsupported_layout:
        return SupportDecision.reject(FallbackReason.LAYOUT)
    if not forward_batch.forward_mode.is_target_verify():
        return SupportDecision.reject(FallbackReason.FORWARD_MODE)
    if not metadata_ready:
        return SupportDecision.reject(FallbackReason.METADATA_STATE)

    spec_info = getattr(forward_batch, "spec_info", None)
    spec_values = (
        getattr(spec_info, "topk", None),
        getattr(spec_info, "spec_steps", None),
        getattr(spec_info, "draft_token_num", None),
        getattr(spec_info, "num_tokens_per_req", None),
    )
    if spec_info is None or tuple(map(_safe_int, spec_values)) != (1, 3, 4, 4):
        return SupportDecision.reject(
            FallbackReason.SPEC_METADATA,
            actual=spec_values,
            expected=(1, 3, 4, 4),
        )

    if not causal:
        return SupportDecision.reject(
            FallbackReason.LAYOUT, actual="causal=False", expected="causal=True"
        )
    if window_size not in SUPPORTED_WINDOWS:
        return SupportDecision.reject(
            FallbackReason.WINDOW,
            actual=window_size,
            expected=sorted(SUPPORTED_WINDOWS),
        )
    if float(getattr(layer, "logit_cap", 0.0)) != 0.0:
        return SupportDecision.reject(
            FallbackReason.SOFTCAP,
            actual=getattr(layer, "logit_cap", None),
            expected=0.0,
        )

    local_shape = (
        getattr(layer, "tp_q_head_num", None),
        getattr(layer, "tp_k_head_num", None),
        getattr(layer, "tp_v_head_num", None),
        getattr(layer, "head_dim", None),
        getattr(layer, "v_head_dim", None),
    )
    expected_local_shape = (Q_HEADS, KV_HEADS, KV_HEADS, HEAD_DIM, HEAD_DIM)
    if tuple(map(_safe_int, local_shape)) != expected_local_shape:
        return SupportDecision.reject(
            FallbackReason.HEAD_SHAPE,
            actual=local_shape,
            expected=expected_local_shape,
        )

    batch_size = int(forward_batch.batch_size)
    expected_query_shape = (batch_size * QUERY_TOKENS, Q_HEADS, HEAD_DIM)
    if (
        tuple(query.shape) != expected_query_shape
        or query.dtype is not torch.bfloat16
        or not query.is_contiguous()
    ):
        return SupportDecision.reject(
            FallbackReason.QUERY,
            actual=(tuple(query.shape), str(query.dtype), query.is_contiguous()),
            expected=(expected_query_shape, "torch.bfloat16", True),
        )

    expected_cache_tail = (PAGE_SIZE, KV_HEADS, HEAD_DIM)
    caches_valid = (
        key_cache.ndim == 4
        and value_cache.ndim == 4
        and tuple(key_cache.shape[1:]) == expected_cache_tail
        and tuple(value_cache.shape[1:]) == expected_cache_tail
        and key_cache.dtype is torch.bfloat16
        and value_cache.dtype is torch.bfloat16
        and key_cache.device == query.device
        and value_cache.device == query.device
        and key_cache.is_contiguous()
        and value_cache.is_contiguous()
    )
    if not caches_valid:
        return SupportDecision.reject(
            FallbackReason.KV_CACHE,
            actual=(
                tuple(key_cache.shape),
                tuple(value_cache.shape),
                str(key_cache.dtype),
                str(value_cache.dtype),
            ),
            expected=("[pages,16,1,256]", "torch.bfloat16"),
        )

    page_table_valid = (
        page_table.ndim == 2
        and int(page_table.shape[0]) == batch_size
        and page_table.dtype is torch.int32
        and page_table.device == query.device
        and page_table.is_contiguous()
    )
    if not page_table_valid:
        return SupportDecision.reject(
            FallbackReason.PAGE_TABLE,
            actual=(tuple(page_table.shape), str(page_table.dtype)),
            expected=((batch_size, "max_pages"), "torch.int32"),
        )

    seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if isinstance(seq_lens_cpu, torch.Tensor) and seq_lens_cpu.numel() >= batch_size:
        max_cache_len = int(seq_lens_cpu[:batch_size].max().item()) + QUERY_TOKENS
        required_pages = (max_cache_len + PAGE_SIZE - 1) // PAGE_SIZE
        if required_pages > int(page_table.shape[1]):
            return SupportDecision.reject(
                FallbackReason.CACHE_LENGTH,
                actual={
                    "max_cache_len": max_cache_len,
                    "page_table_pages": int(page_table.shape[1]),
                },
                expected={"required_pages": required_pages},
            )

    if sinks is not None and (
        tuple(sinks.shape) != (Q_HEADS,)
        or sinks.dtype is not torch.bfloat16
        or sinks.device != query.device
        or not sinks.is_contiguous()
    ):
        return SupportDecision.reject(
            FallbackReason.SINKS,
            actual=(tuple(sinks.shape), str(sinks.dtype)),
            expected=((Q_HEADS,), "torch.bfloat16"),
        )

    if query.device.type != "cuda":
        return SupportDecision.reject(
            FallbackReason.DEVICE, actual=str(query.device), expected="cuda:SM90"
        )
    capability = torch.cuda.get_device_capability(query.device)
    if capability != (9, 0):
        return SupportDecision.reject(
            FallbackReason.DEVICE, actual=capability, expected=(9, 0)
        )
    return SupportDecision.allow()


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
