# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""MK fused pre-attention integration for WeLM v4.5 80A3 only."""

import logging
from typing import Any, Optional, Tuple

import torch

from sglang.srt.models.welm_v45_80a3_fused_pre_attn_config import (
    WELM_V45_80A3_FUSED_PRE_ATTN_ENV,
    welm_v45_80a3_fused_pre_attn_enabled,
)

logger = logging.getLogger(__name__)
_WELM_V45_80A3_FUSED_PRE_ATTN_ENV = WELM_V45_80A3_FUSED_PRE_ATTN_ENV
_welm_v45_80a3_fused_pre_attn_enabled = welm_v45_80a3_fused_pre_attn_enabled


_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED = welm_v45_80a3_fused_pre_attn_enabled()
_WELM_MK_FUSED_QKV_FN = None
_WELM_MK_MIRROR_FNS = None
_WELM_MK_FUSED_QKV_CONFIG_ERROR = None
_WELM_MK_FUSED_QKV_IMPORT_FAILED = False
_WELM_MK_MAX_CACHED_PREFILL_HANDLES = 8
_WELM_MK_MAX_FUSED_ROWS = 16384


class _MKCapabilityUnavailable(RuntimeError):
    """The optional MK package does not expose a required fused-QKV symbol."""


def _requires_attention_backend_kv_write(forward_batch: Any) -> bool:
    """Keep paths whose attention backend owns KV placement on the fallback."""
    if getattr(forward_batch, "attn_cp_prefill_runtime_layout", None) is not None:
        return True
    forward_mode = getattr(forward_batch, "forward_mode", None)
    is_context_parallel_extend = getattr(
        forward_mode, "is_context_parallel_extend", None
    )
    if callable(is_context_parallel_extend) and is_context_parallel_extend():
        return True
    return bool(
        getattr(getattr(forward_batch, "attn_backend", None), "fa_skip_kv_cache", False)
    )


def prepare_welm_v45_80a3_fused_pre_attn_cuda_graphs(
    model_runner: Any,
    buffers: Any,
    capture_bs: list[int],
    num_tokens_per_bs: int,
) -> int:
    """Prepare model-owned MK handles before CUDA graph capture."""
    token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    if token_to_kv_pool is None:
        return 0

    unsupported = [
        (int(batch_size), int(batch_size) * int(num_tokens_per_bs))
        for batch_size in capture_bs
        if int(batch_size) * int(num_tokens_per_bs) > _WELM_MK_MAX_FUSED_ROWS
    ]
    if unsupported:
        first_bs, first_rows = unsupported[0]
        logger.warning(
            "WeLM v4.5 80A3 fused pre-attention supports at most %d rows; "
            "%d capture "
            "shape(s), starting at batch size %d (%d rows), will use the "
            "unfused path.",
            _WELM_MK_MAX_FUSED_ROWS,
            len(unsupported),
            first_bs,
            first_rows,
        )

    prepared_count = 0
    for module in model_runner.model.modules():
        prepare_dense = getattr(
            module, "prepare_welm_v45_80a3_fused_pre_attn_cuda_graph", None
        )
        if prepare_dense is None:
            continue

        cache_loc = buffers.out_cache_loc
        if (
            buffers.out_cache_loc_swa is not None
            and hasattr(token_to_kv_pool, "is_swa_layer")
            and token_to_kv_pool.is_swa_layer(module.layer_idx)
        ):
            cache_loc = buffers.out_cache_loc_swa
        for batch_size in capture_bs:
            rows = int(batch_size) * int(num_tokens_per_bs)
            prepared = prepare_dense(
                rows, buffers.positions, cache_loc, token_to_kv_pool
            )
            prepared_count += int(prepared)
    return prepared_count


def _load_welm_mk_fused_qkv():
    global _WELM_MK_FUSED_QKV_CONFIG_ERROR
    global _WELM_MK_FUSED_QKV_FN, _WELM_MK_FUSED_QKV_IMPORT_FAILED
    if _WELM_MK_FUSED_QKV_IMPORT_FAILED:
        return None
    if _WELM_MK_FUSED_QKV_FN is not None:
        return _WELM_MK_FUSED_QKV_FN
    try:
        from mk.errors import MKConfigError
        from mk.kernels import welm_v45_80a3_fused_pre_attn
    except ImportError as exc:
        logger.warning(
            "%s=1 requested but the MK WeLM v4.5 80A3 fused pre-attention "
            "kernel is unavailable: %s",
            _WELM_V45_80A3_FUSED_PRE_ATTN_ENV,
            exc,
        )
        _WELM_MK_FUSED_QKV_IMPORT_FAILED = True
        return None
    _WELM_MK_FUSED_QKV_CONFIG_ERROR = MKConfigError
    _WELM_MK_FUSED_QKV_FN = welm_v45_80a3_fused_pre_attn
    return _WELM_MK_FUSED_QKV_FN


def _prepare_welm_qkv(*args, **kwargs):
    try:
        from mk.kernels import prepare_welm_v45_80a3_fused_pre_attn
    except ImportError as exc:
        raise _MKCapabilityUnavailable from exc

    return prepare_welm_v45_80a3_fused_pre_attn(*args, **kwargs)


def _load_welm_mk_mirror_fns():
    global _WELM_MK_MIRROR_FNS
    if _WELM_MK_MIRROR_FNS is None:
        try:
            from mk.kernels import (
                prepare_welm_v45_80a3_mirror_consumer_fused_pre_attn,
                prepare_welm_v45_80a3_mirror_source_fused_pre_attn,
                welm_v45_80a3_mirror_consumer_fused_pre_attn,
                welm_v45_80a3_mirror_source_fused_pre_attn,
            )
        except ImportError as exc:
            raise _MKCapabilityUnavailable from exc

        _WELM_MK_MIRROR_FNS = {
            "prepare_consumer": (prepare_welm_v45_80a3_mirror_consumer_fused_pre_attn),
            "prepare_source": prepare_welm_v45_80a3_mirror_source_fused_pre_attn,
            "consumer": welm_v45_80a3_mirror_consumer_fused_pre_attn,
            "source": welm_v45_80a3_mirror_source_fused_pre_attn,
        }
    return _WELM_MK_MIRROR_FNS


def _is_mk_config_error(exc: Exception) -> bool:
    try:
        from mk.errors import MKConfigError
    except ImportError:
        return False
    return isinstance(exc, MKConfigError)


def _tensor_owner(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    base = getattr(tensor, "_base", None)
    return tensor if base is None else base


def _tensor_layout_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _mk_rejected_config_key(
    projection_kind: str,
    has_bias: bool,
    tensors: tuple[torch.Tensor, ...],
) -> tuple[Any, ...]:
    return (
        projection_kind,
        int(tensors[0].shape[0]),
        *(_tensor_layout_key(tensor) for tensor in tensors),
        has_bias,
    )


class WeLMV45_80A3FusedPreAttnMixin:
    """Optional MK fusion with a strict WeLM v4.5 80A3 model contract."""

    def _mk_is_standard_projection(self) -> bool:
        raise NotImplementedError

    def _mk_projection_kind(self) -> Optional[str]:
        return "standard" if self._mk_is_standard_projection() else None

    def _mk_mirror_source_keys(self) -> tuple[int, ...]:
        return ()

    def _mk_mirror_consumer_key(self) -> Optional[int]:
        return None

    def _mk_mirror_requires_projection_contract(self, forward_batch: Any) -> bool:
        return False

    def _mk_graph_dump_enabled(self) -> bool:
        raise NotImplementedError

    def prepare_welm_v45_80a3_fused_pre_attn_cuda_graph(
        self,
        rows: int,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        token_to_kv_pool: Any,
    ) -> bool:
        projection_kind = self._mk_projection_kind()
        if (
            not _WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED
            or not getattr(self, "_welm_v45_80a3_fused_pre_attn_model_contract", False)
            or self._mk_graph_dump_enabled()
            or self.is_nextn
            or self.suffix_parallel
            or self.scale_seq_attn_per_suffix
            or projection_kind not in ("standard", "mirror_source", "mirror_consumer")
            or self.qk_norm
            or not self.only_k_norm
            or self.head_dim != 256
            or self.q_size not in (512, 1536)
            or self.kv_size != 256
            # CUDA graph rows are capture_bs * num_tokens_per_bs.  Keep this
            # gate aligned with eager/MK so high-BS speculative graphs can use
            # the large-M kernel instead of silently falling back above 512.
            or not (0 < rows <= _WELM_MK_MAX_FUSED_ROWS)
            or not hasattr(self.qkv_proj, "weight")
            or self.qkv_proj.weight.dtype is not torch.bfloat16
            or cache_loc.dtype not in (torch.int32, torch.int64)
        ):
            return False
        weight = self.qkv_proj.weight
        hidden_size = int(weight.shape[1])
        projection_width = int(weight.shape[0])
        valid_weight_shape = (
            (
                projection_kind == "standard"
                and projection_width == self.q_size + 2 * self.kv_size
            )
            or (
                projection_kind == "mirror_source"
                and projection_width >= self.q_size + 4 * self.kv_size
                and (projection_width - self.q_size - 2 * self.kv_size)
                % (2 * self.kv_size)
                == 0
            )
            or (
                projection_kind == "mirror_consumer" and projection_width == self.q_size
            )
        )
        if hidden_size != 2048 or not valid_weight_shape:
            return False
        key_cache = token_to_kv_pool.get_key_buffer(self.layer_idx)
        value_cache = token_to_kv_pool.get_value_buffer(self.layer_idx)
        if (
            key_cache.dtype != weight.dtype
            or value_cache.dtype != weight.dtype
            or not key_cache.is_contiguous()
            or not value_cache.is_contiguous()
        ):
            return False
        static_input = torch.empty(
            (rows, hidden_size), device=weight.device, dtype=weight.dtype
        )
        mirror_outputs = ()
        try:
            if projection_kind == "standard":
                q_weight = weight[: self.q_size]
                k_weight = weight[self.q_size : self.q_size + self.kv_size]
                v_weight = weight[
                    self.q_size + self.kv_size : self.q_size + 2 * self.kv_size
                ]
                q_bias = k_bias = v_bias = None
                if self.qkv_proj.bias is not None:
                    q_bias = self.qkv_proj.bias[: self.q_size]
                    k_bias = self.qkv_proj.bias[
                        self.q_size : self.q_size + self.kv_size
                    ]
                    v_bias = self.qkv_proj.bias[
                        self.q_size + self.kv_size : self.q_size + 2 * self.kv_size
                    ]
                q, k, v, handle = _prepare_welm_qkv(
                    static_input,
                    q_weight,
                    k_weight,
                    v_weight,
                    self.k_norm.weight,
                    positions[:rows],
                    self.rotary_emb.cos_sin_cache,
                    key_cache,
                    value_cache,
                    cache_loc[:rows],
                    q_bias=q_bias,
                    k_bias=k_bias,
                    v_bias=v_bias,
                    k_norm_eps=self.k_norm.eps,
                    head_dim=self.head_dim,
                )
            elif projection_kind == "mirror_source":
                prepare_source = _load_welm_mk_mirror_fns()["prepare_source"]
                q, k, v, mirror_outputs, handle = prepare_source(
                    static_input,
                    weight,
                    self.k_norm.weight,
                    positions[:rows],
                    self.rotary_emb.cos_sin_cache,
                    key_cache,
                    value_cache,
                    cache_loc[:rows],
                    packed_bias=self.qkv_proj.bias,
                    k_norm_eps=self.k_norm.eps,
                    head_dim=self.head_dim,
                    q_width=self.q_size,
                )
                if len(mirror_outputs) != len(self._mk_mirror_source_keys()):
                    return False
            else:
                prepare_consumer = _load_welm_mk_mirror_fns()["prepare_consumer"]
                placeholder_k = torch.empty(
                    (rows, 256), device=weight.device, dtype=weight.dtype
                )
                placeholder_v = torch.empty_like(placeholder_k)
                q, k, v, handle = prepare_consumer(
                    static_input,
                    weight,
                    placeholder_k,
                    placeholder_v,
                    self.k_norm.weight,
                    positions[:rows],
                    self.rotary_emb.cos_sin_cache,
                    key_cache,
                    value_cache,
                    cache_loc[:rows],
                    q_bias=self.qkv_proj.bias,
                    k_norm_eps=self.k_norm.eps,
                    head_dim=self.head_dim,
                    q_width=self.q_size,
                )
            reusable_workspace = rows <= 32 or bool(
                getattr(handle, "supports_dynamic_input_rebind", False)
            )
            if reusable_workspace:
                handle.prepare_workspace()
        except _MKCapabilityUnavailable:
            # Missing MK package/symbols is an environment capability gap, not a
            # kernel bug: fall back to the unfused path.
            return False
        except Exception as exc:
            # Only MK's explicit "this config is unsupported" signal is a valid
            # reason to fall back. Everything else (including AttributeError,
            # which almost always means a real bug in the fused path) must
            # surface instead of being silently downgraded.
            if _is_mk_config_error(exc):
                return False
            raise
        if not hasattr(self, "_mk_fused_qkv_graph_handles"):
            self._mk_fused_qkv_graph_handles = {}
        self._mk_fused_qkv_graph_handles[rows] = {
            "input": static_input,
            "q": q,
            "k": k,
            "v": v,
            "handle": handle,
            "reusable_workspace": reusable_workspace,
            "projection_kind": projection_kind,
            "mirror_outputs": mirror_outputs,
            "mirror_source_keys": self._mk_mirror_source_keys(),
            "mirror_consumer_key": self._mk_mirror_consumer_key(),
        }
        return True

    def _try_mk_fused_qkv_knorm_rope_kv_write(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
        kv_mirror_states: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if _requires_attention_backend_kv_write(forward_batch):
            return None
        if not _WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED or not getattr(
            self, "_welm_v45_80a3_fused_pre_attn_model_contract", False
        ):
            return None
        try:
            is_capturing = torch.cuda.is_current_stream_capturing()
        except RuntimeError:
            is_capturing = False
        if is_capturing:
            graph_entry = getattr(self, "_mk_fused_qkv_graph_handles", {}).get(
                int(hidden_states.shape[0])
            )
            if graph_entry is None:
                return None
            projection_kind = graph_entry.get("projection_kind", "standard")
            if (
                projection_kind == "mirror_consumer"
                and self._mk_mirror_requires_projection_contract(forward_batch)
            ):
                return None
            if projection_kind == "mirror_source" and kv_mirror_states is None:
                return None
            mirror_k = mirror_v = None
            if projection_kind == "mirror_consumer":
                mirror_key = graph_entry["mirror_consumer_key"]
                mirror_kv = (
                    None
                    if kv_mirror_states is None
                    else kv_mirror_states.get(mirror_key)
                )
                if mirror_kv is None:
                    return None
                mirror_k, mirror_v = mirror_kv
                expected = (int(hidden_states.shape[0]), self.kv_size)
                if (
                    tuple(mirror_k.shape) != expected
                    or tuple(mirror_v.shape) != expected
                ):
                    return None
            if graph_entry.get("reusable_workspace", False):
                rebindings = {0: hidden_states}
                if projection_kind == "mirror_consumer":
                    rebindings.update({18: mirror_k, 19: mirror_v})
                graph_entry["handle"].launch_prepared_rebound(rebindings)
            else:
                graph_entry["input"].copy_(hidden_states)
                if projection_kind == "mirror_consumer":
                    graph_entry["handle"].bind_raw_kv(mirror_k, mirror_v)
                graph_entry["handle"].launch()
            if projection_kind == "mirror_source":
                if kv_mirror_states is None:
                    return None
                kv_mirror_states.update(
                    zip(
                        graph_entry["mirror_source_keys"],
                        graph_entry["mirror_outputs"],
                    )
                )
            elif projection_kind == "mirror_consumer":
                kv_mirror_states.pop(graph_entry["mirror_consumer_key"], None)
                if getattr(self, "need_clear_kv_cache", False):
                    kv_mirror_states.clear()
                return graph_entry["q"], graph_entry["k"], mirror_v
            return graph_entry["q"], graph_entry["k"], graph_entry["v"]
        projection_kind = self._mk_projection_kind()
        if (
            projection_kind == "mirror_consumer"
            and self._mk_mirror_requires_projection_contract(forward_batch)
        ):
            return None
        if (
            self._mk_graph_dump_enabled()
            or self.is_nextn
            or self.suffix_parallel
            or self.scale_seq_attn_per_suffix
            or projection_kind not in ("standard", "mirror_source", "mirror_consumer")
            or self.qk_norm
            or not self.only_k_norm
            or self.head_dim != 256
            or self.q_size not in (512, 1536)
            or self.kv_size != 256
            or hidden_states.ndim != 2
            or hidden_states.shape[1] != 2048
            or not (0 < hidden_states.shape[0] <= _WELM_MK_MAX_FUSED_ROWS)
            or hidden_states.dtype is not torch.bfloat16
            or not hidden_states.is_contiguous()
            or positions.dtype != torch.int64
            or not positions.is_contiguous()
            or positions.shape != (hidden_states.shape[0],)
            or not hasattr(self.qkv_proj, "weight")
            or self.qkv_proj.weight.dtype != hidden_states.dtype
        ):
            return None

        token_to_kv_pool = forward_batch.token_to_kv_pool
        cache_loc = self._mk_cache_loc_for_rows(
            forward_batch, int(hidden_states.shape[0])
        )
        if cache_loc is None:
            return None

        key_cache = token_to_kv_pool.get_key_buffer(self.layer_idx)
        value_cache = token_to_kv_pool.get_value_buffer(self.layer_idx)
        if (
            key_cache.dtype != hidden_states.dtype
            or value_cache.dtype != hidden_states.dtype
            or not key_cache.is_contiguous()
            or not value_cache.is_contiguous()
            or key_cache.numel() % self.kv_size != 0
            or value_cache.numel() % self.kv_size != 0
        ):
            return None
        weight = self.qkv_proj.weight
        rejected_configs = getattr(self, "_mk_fused_qkv_rejected_configs", None)
        rejected_config_tensors = (
            hidden_states,
            weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            key_cache,
            value_cache,
            cache_loc,
        )
        mirror_config_tensors = ()
        try:
            if projection_kind == "mirror_source":
                if kv_mirror_states is None:
                    return None
                source_keys = self._mk_mirror_source_keys()
                if tuple(weight.shape)[1:] != (2048,) or int(
                    weight.shape[0]
                ) != self.q_size + 2 * self.kv_size * (1 + len(source_keys)):
                    return None
                if (
                    rejected_configs is not None
                    and _mk_rejected_config_key(
                        projection_kind,
                        self.qkv_proj.bias is not None,
                        rejected_config_tensors,
                    )
                    in rejected_configs
                ):
                    return None
                q, k, v, mirror_outputs = _load_welm_mk_mirror_fns()["source"](
                    hidden_states,
                    weight,
                    self.k_norm.weight,
                    positions,
                    self.rotary_emb.cos_sin_cache,
                    key_cache,
                    value_cache,
                    cache_loc,
                    packed_bias=self.qkv_proj.bias,
                    k_norm_eps=self.k_norm.eps,
                    head_dim=self.head_dim,
                    q_width=self.q_size,
                )
                kv_mirror_states.update(zip(source_keys, mirror_outputs))
                return q, k, v

            if projection_kind == "mirror_consumer":
                if kv_mirror_states is None or tuple(weight.shape) != (
                    self.q_size,
                    2048,
                ):
                    return None
                mirror_key = self._mk_mirror_consumer_key()
                mirror_kv = kv_mirror_states.get(mirror_key)
                if mirror_kv is None:
                    return None
                raw_k, raw_v = mirror_kv
                expected = (int(hidden_states.shape[0]), self.kv_size)
                if tuple(raw_k.shape) != expected or tuple(raw_v.shape) != expected:
                    return None
                mirror_config_tensors = (raw_k, raw_v)
                if (
                    rejected_configs is not None
                    and _mk_rejected_config_key(
                        projection_kind,
                        self.qkv_proj.bias is not None,
                        rejected_config_tensors + mirror_config_tensors,
                    )
                    in rejected_configs
                ):
                    return None
                q, k, v = _load_welm_mk_mirror_fns()["consumer"](
                    hidden_states,
                    weight,
                    raw_k,
                    raw_v,
                    self.k_norm.weight,
                    positions,
                    self.rotary_emb.cos_sin_cache,
                    key_cache,
                    value_cache,
                    cache_loc,
                    q_bias=self.qkv_proj.bias,
                    k_norm_eps=self.k_norm.eps,
                    head_dim=self.head_dim,
                    q_width=self.q_size,
                )
                kv_mirror_states.pop(mirror_key, None)
                if getattr(self, "need_clear_kv_cache", False):
                    kv_mirror_states.clear()
                return q, k, v

            fused_qkv = _load_welm_mk_fused_qkv()
            if fused_qkv is None or tuple(weight.shape) != (
                self.q_size + 2 * self.kv_size,
                2048,
            ):
                return None
            if (
                rejected_configs is not None
                and _mk_rejected_config_key(
                    projection_kind,
                    self.qkv_proj.bias is not None,
                    rejected_config_tensors,
                )
                in rejected_configs
            ):
                return None
            q_weight = weight[: self.q_size]
            k_weight = weight[self.q_size : self.q_size + self.kv_size]
            v_weight = weight[
                self.q_size + self.kv_size : self.q_size + 2 * self.kv_size
            ]
            q_bias = k_bias = v_bias = None
            if self.qkv_proj.bias is not None:
                q_bias = self.qkv_proj.bias[: self.q_size]
                k_bias = self.qkv_proj.bias[self.q_size : self.q_size + self.kv_size]
                v_bias = self.qkv_proj.bias[
                    self.q_size + self.kv_size : self.q_size + 2 * self.kv_size
                ]
            eager_result = self._try_mk_fused_qkv_reusable_eager(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                self.k_norm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                key_cache,
                value_cache,
                cache_loc,
                q_bias=q_bias,
                k_bias=k_bias,
                v_bias=v_bias,
            )
            if eager_result is not None:
                return eager_result
            return fused_qkv(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                self.k_norm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                key_cache,
                value_cache,
                cache_loc,
                q_bias=q_bias,
                k_bias=k_bias,
                v_bias=v_bias,
                k_norm_eps=self.k_norm.eps,
                head_dim=self.head_dim,
            )
        except _MKCapabilityUnavailable:
            # Missing MK package/symbols is an environment capability gap, not a
            # kernel bug: fall back to the unfused path.
            return None
        except Exception as exc:
            # Only MK's explicit "this config is unsupported" signal is a valid
            # reason to disable the fused path. Anything else (including
            # AttributeError, which almost always means a real bug) must surface.
            is_config_error = (
                _WELM_MK_FUSED_QKV_CONFIG_ERROR is not None
                and isinstance(exc, _WELM_MK_FUSED_QKV_CONFIG_ERROR)
            ) or _is_mk_config_error(exc)
            if not is_config_error:
                raise
            rejected_config = _mk_rejected_config_key(
                projection_kind,
                self.qkv_proj.bias is not None,
                rejected_config_tensors + mirror_config_tensors,
            )
            if rejected_configs is None:
                rejected_configs = set()
                self._mk_fused_qkv_rejected_configs = rejected_configs
            rejected_configs.add(rejected_config)
            logger.warning(
                "Skipping %s for rejected runtime configuration %s: %s",
                _WELM_V45_80A3_FUSED_PRE_ATTN_ENV,
                rejected_config,
                exc,
            )
            return None

    def _try_mk_fused_qkv_reusable_eager(
        self,
        hidden_states: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_loc: torch.Tensor,
        *,
        q_bias: Optional[torch.Tensor],
        k_bias: Optional[torch.Tensor],
        v_bias: Optional[torch.Tensor],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        rows = int(hidden_states.shape[0])
        if not hidden_states.is_cuda or rows > 2048:
            return None
        device_id = hidden_states.device.index
        if device_id is None:
            device_id = torch.cuda.current_device()
        get_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
        stream = (
            int(get_raw_stream(device_id))
            if get_raw_stream is not None
            else int(torch.cuda.current_stream(device=device_id).cuda_stream)
        )

        key = (
            rows,
            cache_loc.dtype,
            stream,
        )
        static_tensor_owners = tuple(
            _tensor_owner(tensor)
            for tensor in (
                q_weight,
                k_weight,
                v_weight,
                q_bias,
                k_bias,
                v_bias,
                k_norm_weight,
                cos_sin_cache,
                key_cache,
                value_cache,
            )
        )
        handles = getattr(self, "_mk_fused_qkv_eager_handles", None)
        entry = handles.get(key) if handles is not None else None
        if entry is not None and any(
            current is not cached
            for current, cached in zip(
                static_tensor_owners, entry["static_tensor_owners"]
            )
        ):
            entry = None
        if entry is None:
            q, k, v, handle = _prepare_welm_qkv(
                hidden_states,
                q_weight,
                k_weight,
                v_weight,
                k_norm_weight,
                positions,
                cos_sin_cache,
                key_cache,
                value_cache,
                cache_loc,
                q_bias=q_bias,
                k_bias=k_bias,
                v_bias=v_bias,
                k_norm_eps=self.k_norm.eps,
                head_dim=self.head_dim,
            )
            if not hasattr(handle, "launch_prepared_rebound"):
                return None
            handle.prepare_workspace(stream=stream)
            handle.launch_prepared(stream=stream)
            if handles is None:
                handles = {}
                self._mk_fused_qkv_eager_handles = handles
            if rows > 32:
                large_keys = [
                    cached_key for cached_key in handles if cached_key[0] > 32
                ]
                if len(large_keys) >= _WELM_MK_MAX_CACHED_PREFILL_HANDLES:
                    handles.pop(large_keys[0])
            entry = {
                "q": q,
                "k": k,
                "v": v,
                "static_tensor_owners": static_tensor_owners,
                "handle": handle,
                "base_dynamic_ptrs": (
                    hidden_states.data_ptr(),
                    positions.data_ptr(),
                    cache_loc.data_ptr(),
                ),
            }
            handles[key] = entry
        else:
            dynamic_ptrs = (
                hidden_states.data_ptr(),
                positions.data_ptr(),
                cache_loc.data_ptr(),
            )
            if dynamic_ptrs == entry["base_dynamic_ptrs"]:
                entry["handle"].launch_prepared(stream=stream)
            else:
                entry["handle"].launch_prepared_rebound(
                    {0: hidden_states, 8: positions, 15: cache_loc},
                    stream=stream,
                )
        return entry["q"], entry["k"], entry["v"]

    def _mk_cache_loc_for_rows(
        self, forward_batch: Any, rows: int
    ) -> Optional[torch.Tensor]:
        token_to_kv_pool = forward_batch.token_to_kv_pool
        cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if hasattr(token_to_kv_pool, "is_swa_layer") and token_to_kv_pool.is_swa_layer(
            self.layer_idx
        ):
            cache_loc = getattr(forward_batch, "out_cache_loc_swa", None)
            if cache_loc is None:
                cache_loc = getattr(token_to_kv_pool, "swa_loc", None)

        candidates = [cache_loc]
        if getattr(forward_batch, "welm_mtp_merge_kv_fill_draft", False):
            candidates.append(
                getattr(forward_batch, "welm_mtp_kv_fill_cache_loc", None)
            )
        if getattr(forward_batch, "welm_kv_mirror_contracted", False):
            candidates.append(getattr(forward_batch, "custom_last_cache_loc", None))

        def valid(candidate: Optional[torch.Tensor]) -> bool:
            return (
                isinstance(candidate, torch.Tensor)
                and candidate.shape == (rows,)
                and candidate.dtype in (torch.int32, torch.int64)
                and candidate.is_contiguous()
            )

        for candidate in candidates:
            if valid(candidate):
                return candidate
        return None
