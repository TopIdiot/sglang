# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""MK Pre-Attn V2 integration for the tuned WeLM v4.5 80A3 TP4 shape."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.models.welm_v45_80a3_h2048_hd256_pre_attn_v2_config import (
    WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_ENV,
    welm_v45_80a3_h2048_hd256_pre_attn_v2_enabled,
)

logger = logging.getLogger(__name__)

_WELM_V45_80A3_FUSED_PRE_ATTN_ENV = WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_ENV
_WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_ENABLED = (
    welm_v45_80a3_h2048_hd256_pre_attn_v2_enabled()
)
_WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_MAX_ROWS = 16383
_MK_V2_API = None
_MK_V2_IMPORT_FAILED = False
_LOGGED_HITS: set[tuple[str, str, int]] = set()
_LOGGED_FALLBACKS: set[tuple[str, int, str]] = set()


class _MKCapabilityUnavailable(RuntimeError):
    """The optional MK package does not provide the production V2 API."""


@dataclass(frozen=True)
class WeLMV45_80A3H2048HD256PreAttnV2Result:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    hidden_states: torch.Tensor
    kv_cache_written: bool = True


def _load_mk_v2_api():
    global _MK_V2_API, _MK_V2_IMPORT_FAILED
    if _MK_V2_API is not None:
        return _MK_V2_API
    if _MK_V2_IMPORT_FAILED:
        raise _MKCapabilityUnavailable
    try:
        from mk.errors import MKConfigError
        from mk.kernels import (
            prepare_welm_v45_80a3_h2048_hd256_idle_mtp_full_qkv_v2,
            prepare_welm_v45_80a3_h2048_hd256_indexed_mtp_optimized_v2,
            prepare_welm_v45_80a3_h2048_hd256_mirror_consumer_optimized_v2,
            prepare_welm_v45_80a3_h2048_hd256_mirror_source_qkv_optimized_v2,
            prepare_welm_v45_80a3_h2048_hd256_standard_qkv_optimized_v2,
        )
    except ImportError as exc:
        logger.warning(
            "%s=1 requested, but MK WeLM H2048/HD256 Pre-Attn V2 is unavailable: %s",
            _WELM_V45_80A3_FUSED_PRE_ATTN_ENV,
            exc,
        )
        _MK_V2_IMPORT_FAILED = True
        raise _MKCapabilityUnavailable from exc
    _MK_V2_API = {
        "config_error": MKConfigError,
        "standard": prepare_welm_v45_80a3_h2048_hd256_standard_qkv_optimized_v2,
        "mirror_source": (
            prepare_welm_v45_80a3_h2048_hd256_mirror_source_qkv_optimized_v2
        ),
        "mirror_consumer": (
            prepare_welm_v45_80a3_h2048_hd256_mirror_consumer_optimized_v2
        ),
        "indexed_mtp": (prepare_welm_v45_80a3_h2048_hd256_indexed_mtp_optimized_v2),
        "idle_mtp": prepare_welm_v45_80a3_h2048_hd256_idle_mtp_full_qkv_v2,
    }
    return _MK_V2_API


def _requires_attention_backend_kv_write(forward_batch: Any) -> bool:
    """Keep paths whose attention backend owns KV placement on the baseline."""
    if getattr(forward_batch, "attn_cp_prefill_runtime_layout", None) is not None:
        return True
    return bool(
        getattr(getattr(forward_batch, "attn_backend", None), "fa_skip_kv_cache", False)
    )


def _q_backed_kv_placeholders(
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Supply shape-only K/V arguments after V2 has already populated the cache."""
    return q[:, :256], q[:, 256:512]


def _plan_name(prepared: Any) -> str:
    plan = getattr(prepared, "plan", None)
    return str(getattr(plan, "name", type(prepared).__name__))


def _log_hit(kind: str, mode: str, rows: int, prepared: Any) -> None:
    plan = getattr(prepared, "plan", None)
    mirror_bank_count = (
        int(getattr(plan, "mirror_bank_count", 0)) if kind == "mirror_source" else 0
    )
    key = (kind, mode, mirror_bank_count)
    if key in _LOGGED_HITS:
        return
    _LOGGED_HITS.add(key)
    logger.info(
        "MK Pre-Attn V2 hit: model=welm_v45_80a3 hidden_size=2048 "
        "head_dim=256 projection=%s mode=%s M=%d route=%s",
        kind,
        mode,
        rows,
        _plan_name(prepared),
    )


def _log_config_fallback(kind: str, rows: int, exc: Exception) -> None:
    key = (kind, rows, str(exc))
    if key in _LOGGED_FALLBACKS:
        return
    _LOGGED_FALLBACKS.add(key)
    logger.warning(
        "MK Pre-Attn V2 rejected projection=%s M=%d; using baseline: %s",
        kind,
        rows,
        exc,
    )


def prepare_welm_v45_80a3_h2048_hd256_pre_attn_v2_cuda_graphs(
    model_runner: Any,
    buffers: Any,
    capture_bs: list[int],
    num_tokens_per_bs: int,
) -> int:
    """Prepare address-stable V2 operations before CUDA graph capture."""
    token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    if token_to_kv_pool is None:
        return 0

    unsupported = [
        (int(bs), int(bs) * int(num_tokens_per_bs))
        for bs in capture_bs
        if int(bs) * int(num_tokens_per_bs)
        > _WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_MAX_ROWS
    ]
    if unsupported:
        first_bs, first_rows = unsupported[0]
        logger.warning(
            "MK Pre-Attn V2 supports M<=%d; %d CUDA graph shape(s), starting "
            "at batch_size=%d (M=%d), will use the baseline",
            _WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_MAX_ROWS,
            len(unsupported),
            first_bs,
            first_rows,
        )

    prepared_count = 0
    for module in model_runner.model.modules():
        prepare = getattr(
            module,
            "prepare_welm_v45_80a3_h2048_hd256_pre_attn_v2_cuda_graph",
            None,
        )
        if prepare is None:
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
            prepared_count += int(
                prepare(rows, buffers.positions, cache_loc, token_to_kv_pool)
            )
    return prepared_count


def prepare_welm_v45_80a3_h2048_hd256_nextn_pre_attn_v2_cuda_graphs(
    model_runner: Any,
    buffers: Any,
    capture_bs: list[int],
    num_tokens_per_bs: int,
    mirror_kv_states: dict[int, tuple[torch.Tensor, torch.Tensor]],
) -> int:
    """Prepare V2 for the unified NextN proposal graph's contracted shapes."""
    token_to_kv_pool = getattr(model_runner, "token_to_kv_pool", None)
    if token_to_kv_pool is None:
        return 0

    prepared_count = 0
    for module in model_runner.model.modules():
        prepare = getattr(
            module,
            "prepare_welm_v45_80a3_h2048_hd256_nextn_pre_attn_v2_cuda_graph",
            None,
        )
        if prepare is None:
            continue
        mirror_key = module._mk_nextn_mirror_key()
        raw_kv = mirror_kv_states.get(mirror_key)
        if raw_kv is None:
            continue
        cache_loc = buffers.out_cache_loc
        if (
            getattr(buffers, "out_cache_loc_swa", None) is not None
            and hasattr(token_to_kv_pool, "is_swa_layer")
            and token_to_kv_pool.is_swa_layer(module.layer_idx)
        ):
            cache_loc = buffers.out_cache_loc_swa
        for batch_size in capture_bs:
            q_rows = int(batch_size)
            kv_rows = q_rows * int(num_tokens_per_bs)
            prepared_count += int(
                prepare(
                    q_rows,
                    kv_rows,
                    raw_kv,
                    buffers.positions,
                    cache_loc,
                    token_to_kv_pool,
                )
            )
    return prepared_count


class WeLMV45_80A3H2048HD256PreAttnV2Mixin:
    """V2-only integration for target layers and both NextN routes."""

    def _mk_projection_kind(self) -> Optional[str]:
        raise NotImplementedError

    def _mk_runtime_projection_kind(self, forward_batch: Any) -> Optional[str]:
        return self._mk_projection_kind()

    def _mk_prepare_indexed_mtp_v2_inputs(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
        kv_mirror_states: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]],
    ) -> Optional[dict[str, Any]]:
        return None

    def _mk_mirror_source_keys(self) -> tuple[int, ...]:
        return ()

    def _mk_mirror_consumer_key(self) -> Optional[int]:
        return None

    def _mk_nextn_mirror_key(self) -> Optional[int]:
        qkv_proj = self.qkv_proj
        mirror_layer_idx = getattr(qkv_proj, "mirror_layer_idx", None)
        return (
            int(mirror_layer_idx)
            if mirror_layer_idx is not None
            else getattr(qkv_proj, "imitated_layer_idx", None)
        )

    def _mk_mirror_requires_projection_contract(self, forward_batch: Any) -> bool:
        return False

    def _mk_graph_dump_enabled(self) -> bool:
        raise NotImplementedError

    def _mk_v2_layer_contract(self, kind: Optional[str], rows: int) -> bool:
        if (
            not _WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_ENABLED
            or not getattr(
                self,
                "_welm_v45_80a3_h2048_hd256_pre_attn_v2_model_contract",
                False,
            )
            or self._mk_graph_dump_enabled()
            or self.suffix_parallel
            or self.scale_seq_attn_per_suffix
            or kind
            not in (
                "standard",
                "mirror_source",
                "mirror_consumer",
                "nextn",
                "indexed_mtp",
                "idle_mtp",
            )
            or (self.is_nextn != (kind in ("nextn", "indexed_mtp", "idle_mtp")))
            or self.qk_norm
            or not self.only_k_norm
            or self.head_dim != 256
            or self.qk_rope_head_dim != 64
            or self.q_size != 1536
            or self.kv_size != 256
            or not 0 < rows <= _WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_MAX_ROWS
            or not hasattr(self.qkv_proj, "weight")
            or self.qkv_proj.bias is not None
        ):
            return False

        weight = self.qkv_proj.weight
        source_banks = len(self._mk_mirror_source_keys())
        expected_shape = {
            "standard": (2048, 2048),
            "mirror_source": (2048 + source_banks * 512, 2048),
            "mirror_consumer": (1536, 2048),
            "nextn": (2048, 2048),
            "indexed_mtp": (2048, 2048),
            "idle_mtp": (2048, 2048),
        }[kind]
        return (
            (kind != "mirror_source" or source_banks in (1, 3))
            and tuple(weight.shape) == expected_shape
            and weight.dtype is torch.bfloat16
            and weight.is_cuda
            and weight.is_contiguous()
            and tuple(self.k_norm.weight.shape) == (256,)
            and self.k_norm.weight.dtype is torch.bfloat16
            and self.k_norm.weight.is_contiguous()
            and self.rotary_emb.cos_sin_cache.dtype is torch.float32
            and self.rotary_emb.cos_sin_cache.is_contiguous()
        )

    @staticmethod
    def _mk_v2_io_contract(
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> bool:
        rows = int(hidden_states.shape[0]) if hidden_states.ndim == 2 else 0
        device = hidden_states.device
        return (
            hidden_states.is_cuda
            and hidden_states.dtype is torch.bfloat16
            and tuple(hidden_states.shape) == (rows, 2048)
            and hidden_states.is_contiguous()
            and positions.device == device
            and positions.dtype is torch.int64
            and tuple(positions.shape) == (rows,)
            and positions.is_contiguous()
            and key_cache.device == device
            and value_cache.device == device
            and key_cache.dtype is torch.bfloat16
            and value_cache.dtype is torch.bfloat16
            and key_cache.is_contiguous()
            and value_cache.is_contiguous()
            and key_cache.numel() == value_cache.numel()
            and key_cache.numel() % 256 == 0
            and cache_loc.device == device
            and cache_loc.dtype in (torch.int32, torch.int64)
            and tuple(cache_loc.shape) == (rows,)
            and cache_loc.is_contiguous()
        )

    def _prepare_v2(
        self,
        kind: str,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_loc: torch.Tensor,
        *,
        raw_k: Optional[torch.Tensor] = None,
        raw_v: Optional[torch.Tensor] = None,
        q_gather_indices: Optional[torch.Tensor] = None,
        kv_gather_indices: Optional[torch.Tensor] = None,
        kv_positions: Optional[torch.Tensor] = None,
    ) -> Any:
        api = _load_mk_v2_api()
        if kind == "idle_mtp":
            return api[kind](hidden_states, self.qkv_proj.weight)
        if kind == "indexed_mtp":
            assert raw_k is not None and raw_v is not None
            assert q_gather_indices is not None and kv_gather_indices is not None
            assert kv_positions is not None
            return api[kind](
                hidden_states,
                self.qkv_proj.weight[:1536],
                raw_k,
                raw_v,
                q_gather_indices,
                kv_gather_indices,
                self.k_norm.weight,
                positions,
                kv_positions,
                self.rotary_emb.cos_sin_cache,
                key_cache,
                value_cache,
                cache_loc,
                k_norm_eps=self.k_norm.eps,
            )
        common = (
            self.k_norm.weight,
            positions,
            self.rotary_emb.cos_sin_cache,
            key_cache,
            value_cache,
            cache_loc,
        )
        if kind == "standard":
            return api[kind](
                hidden_states,
                self.qkv_proj.weight,
                *common,
                k_norm_eps=self.k_norm.eps,
            )
        if kind == "mirror_source":
            return api[kind](
                hidden_states,
                self.qkv_proj.weight,
                *common,
                mirror_bank_count=len(self._mk_mirror_source_keys()),
                k_norm_eps=self.k_norm.eps,
            )
        assert raw_k is not None and raw_v is not None
        return api[kind](
            hidden_states,
            self.qkv_proj.weight,
            raw_k,
            raw_v,
            self.k_norm.weight,
            positions,
            positions,
            self.rotary_emb.cos_sin_cache,
            key_cache,
            value_cache,
            cache_loc,
            k_norm_eps=self.k_norm.eps,
        )

    def prepare_welm_v45_80a3_h2048_hd256_pre_attn_v2_cuda_graph(
        self,
        rows: int,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        token_to_kv_pool: Any,
    ) -> int:
        layer_kind = self._mk_projection_kind()
        rows = int(rows)
        if not self._mk_v2_layer_contract(layer_kind, rows):
            return 0

        weight = self.qkv_proj.weight
        if layer_kind == "nextn":
            return self._prepare_nextn_v2_cuda_graph_ops(
                rows, positions, cache_loc, token_to_kv_pool
            )

        kind = layer_kind
        key_cache = token_to_kv_pool.get_key_buffer(self.layer_idx)
        value_cache = token_to_kv_pool.get_value_buffer(self.layer_idx)
        static_input = torch.empty(
            (rows, 2048), device=weight.device, dtype=torch.bfloat16
        )
        graph_positions = positions[:rows]
        graph_cache_loc = cache_loc[:rows]
        if not self._mk_v2_io_contract(
            static_input,
            graph_positions,
            key_cache,
            value_cache,
            graph_cache_loc,
        ):
            return 0

        raw_k = raw_v = raw_kv = None
        if kind == "mirror_consumer":
            raw_kv = torch.empty(
                (rows, 512), device=weight.device, dtype=torch.bfloat16
            )
            raw_k, raw_v = raw_kv[:, :256], raw_kv[:, 256:]
        try:
            prepared = self._prepare_v2(
                kind,
                static_input,
                graph_positions,
                key_cache,
                value_cache,
                graph_cache_loc,
                raw_k=raw_k,
                raw_v=raw_v,
            )
        except _MKCapabilityUnavailable:
            return 0
        except Exception as exc:
            api = _load_mk_v2_api()
            if not isinstance(exc, api["config_error"]):
                raise
            _log_config_fallback(kind, rows, exc)
            return 0

        if not hasattr(self, "_mk_h2048_hd256_pre_attn_v2_graph_ops"):
            self._mk_h2048_hd256_pre_attn_v2_graph_ops = {}
        self._mk_h2048_hd256_pre_attn_v2_graph_ops[(kind, rows)] = {
            "kind": kind,
            "input": static_input,
            "prepared": prepared,
            "raw_kv": raw_kv,
            "raw_k": raw_k,
            "raw_v": raw_v,
        }
        return 1

    def _prepare_nextn_v2_cuda_graph_ops(
        self,
        rows: int,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        token_to_kv_pool: Any,
        *,
        indexed_kv_rows: Optional[int] = None,
        indexed_raw_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        prepare_idle: bool = True,
    ) -> int:
        """Prepare the common identity-shape NextN graph for idle and active MTP."""
        try:
            api = _load_mk_v2_api()
        except _MKCapabilityUnavailable:
            return 0
        weight = self.qkv_proj.weight
        graph_ops = getattr(self, "_mk_h2048_hd256_pre_attn_v2_graph_ops", None)
        if graph_ops is None:
            graph_ops = {}
            self._mk_h2048_hd256_pre_attn_v2_graph_ops = graph_ops
        prepared_count = 0

        idle_input = torch.empty(
            (rows, 2048), device=weight.device, dtype=torch.bfloat16
        )
        if prepare_idle:
            try:
                idle_prepared = self._prepare_v2(
                    "idle_mtp",
                    idle_input,
                    positions[:rows],
                    idle_input,
                    idle_input,
                    cache_loc[:rows],
                )
            except _MKCapabilityUnavailable:
                return 0
            except Exception as exc:
                if not isinstance(exc, api["config_error"]):
                    raise
                _log_config_fallback("idle_mtp", rows, exc)
            else:
                graph_ops[("idle_mtp", rows)] = {
                    "kind": "idle_mtp",
                    "input": idle_input,
                    "prepared": idle_prepared,
                }
                prepared_count += 1

        key_cache = token_to_kv_pool.get_key_buffer(self.layer_idx)
        value_cache = token_to_kv_pool.get_value_buffer(self.layer_idx)
        indexed_input = torch.empty_like(idle_input)
        if indexed_raw_kv is None:
            raw_kv = torch.empty(
                (rows, 512), device=weight.device, dtype=torch.bfloat16
            )
            raw_k, raw_v = raw_kv[:, :256], raw_kv[:, 256:]
        else:
            raw_kv = None
            raw_k, raw_v = indexed_raw_kv
        kv_rows = rows if indexed_kv_rows is None else int(indexed_kv_rows)
        q_indices = torch.arange(rows, device=weight.device, dtype=torch.int64)
        kv_indices = torch.arange(kv_rows, device=weight.device, dtype=torch.int64)
        q_positions = torch.empty((rows,), device=weight.device, dtype=torch.int64)
        kv_positions = torch.empty((kv_rows,), device=weight.device, dtype=torch.int64)
        graph_cache_loc = torch.empty(
            (kv_rows,), device=weight.device, dtype=cache_loc.dtype
        )
        try:
            indexed_prepared = self._prepare_v2(
                "indexed_mtp",
                indexed_input,
                q_positions,
                key_cache,
                value_cache,
                graph_cache_loc,
                raw_k=raw_k,
                raw_v=raw_v,
                q_gather_indices=q_indices,
                kv_gather_indices=kv_indices,
                kv_positions=kv_positions,
            )
        except _MKCapabilityUnavailable:
            return prepared_count
        except Exception as exc:
            if not isinstance(exc, api["config_error"]):
                raise
            _log_config_fallback("indexed_mtp", rows, exc)
        else:
            project_hidden = getattr(indexed_prepared, "gathered_hidden_states", None)
            if project_hidden is None:
                project_hidden = torch.empty_like(indexed_input)
            graph_ops[("indexed_mtp", rows)] = {
                "kind": "indexed_mtp",
                "input": indexed_input,
                "prepared": indexed_prepared,
                "raw_kv": raw_kv,
                "raw_k": raw_k,
                "raw_v": raw_v,
                "q_indices": q_indices,
                "kv_indices": kv_indices,
                "q_positions": q_positions,
                "kv_positions": kv_positions,
                "cache_loc": graph_cache_loc,
                "project_hidden": project_hidden,
                "project_hidden_from_kernel": hasattr(
                    indexed_prepared, "gathered_hidden_states"
                ),
            }
            prepared_count += 1
        return prepared_count

    def prepare_welm_v45_80a3_h2048_hd256_nextn_pre_attn_v2_cuda_graph(
        self,
        q_rows: int,
        kv_rows: int,
        raw_kv: tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        token_to_kv_pool: Any,
    ) -> int:
        """Prepare one contracted unified-proposal graph bucket for NextN V2."""
        q_rows = int(q_rows)
        kv_rows = int(kv_rows)
        if not self._mk_v2_layer_contract("nextn", q_rows):
            return 0
        if not 0 < kv_rows <= _WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_MAX_ROWS:
            return 0
        return self._prepare_nextn_v2_cuda_graph_ops(
            q_rows,
            positions,
            cache_loc,
            token_to_kv_pool,
            indexed_kv_rows=kv_rows,
            indexed_raw_kv=raw_kv,
            prepare_idle=False,
        )

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
        for candidate in candidates:
            if (
                isinstance(candidate, torch.Tensor)
                and tuple(candidate.shape) == (rows,)
                and candidate.dtype in (torch.int32, torch.int64)
                and candidate.is_contiguous()
            ):
                return candidate
        return None

    def _run_graph_v2(
        self,
        entry: dict[str, Any],
        hidden_states: torch.Tensor,
        kv_mirror_states: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]],
        indexed_inputs: Optional[dict[str, Any]] = None,
    ) -> Optional[WeLMV45_80A3H2048HD256PreAttnV2Result]:
        kind = entry["kind"]
        if kind == "idle_mtp":
            prepared = entry["prepared"]
            q, k, v = prepared.launch_rebound(hidden_states)
            _log_hit(kind, "cuda_graph", int(hidden_states.shape[0]), prepared)
            return WeLMV45_80A3H2048HD256PreAttnV2Result(
                q=q,
                k=k,
                v=v,
                hidden_states=hidden_states,
                kv_cache_written=False,
            )
        if kind == "indexed_mtp":
            if indexed_inputs is None or kv_mirror_states is None:
                return None
            tensor_keys = (
                "raw_k",
                "raw_v",
                "q_indices",
                "kv_indices",
                "q_positions",
                "kv_positions",
                "cache_loc",
            )
            if any(
                tuple(indexed_inputs[key].shape) != tuple(entry[key].shape)
                or indexed_inputs[key].dtype != entry[key].dtype
                or indexed_inputs[key].device != entry[key].device
                or indexed_inputs[key].stride() != entry[key].stride()
                for key in tensor_keys
            ):
                return None
            prepared = entry["prepared"]
            q = prepared.launch_rebound(
                hidden_states,
                indexed_inputs["raw_k"],
                indexed_inputs["raw_v"],
                indexed_inputs["q_indices"],
                indexed_inputs["kv_indices"],
                indexed_inputs["q_positions"],
                indexed_inputs["kv_positions"],
                indexed_inputs["cache_loc"],
            )
            project_hidden = entry["project_hidden"]
            if not entry["project_hidden_from_kernel"]:
                torch.index_select(
                    hidden_states,
                    0,
                    indexed_inputs["q_indices"],
                    out=project_hidden,
                )
            if getattr(self, "need_clear_kv_cache", False):
                kv_mirror_states.clear()
            _log_hit(kind, "cuda_graph", int(q.shape[0]), prepared)
            k, v = _q_backed_kv_placeholders(q)
            return WeLMV45_80A3H2048HD256PreAttnV2Result(
                q=q, k=k, v=v, hidden_states=project_hidden
            )
        if kind == "mirror_source" and kv_mirror_states is None:
            return None
        if kind == "mirror_consumer":
            if kv_mirror_states is None:
                return None
            mirror_key = self._mk_mirror_consumer_key()
            mirror_kv = kv_mirror_states.get(mirror_key)
            if mirror_kv is None:
                return None
            raw_k, raw_v = mirror_kv
            expected = (int(hidden_states.shape[0]), 256)
            if tuple(raw_k.shape) != expected or tuple(raw_v.shape) != expected:
                return None
        prepared = entry["prepared"]
        if kind == "mirror_consumer":
            q = prepared.launch_rebound(hidden_states, raw_k=raw_k, raw_v=raw_v)
        else:
            q = prepared.launch_rebound(hidden_states)
        if kind == "mirror_source":
            outputs = tuple(prepared.mirror_outputs)
            if len(outputs) != len(self._mk_mirror_source_keys()):
                raise RuntimeError("MK Pre-Attn V2 mirror-source bank count changed")
            kv_mirror_states.update(zip(self._mk_mirror_source_keys(), outputs))
        elif kind == "mirror_consumer":
            kv_mirror_states.pop(self._mk_mirror_consumer_key(), None)
            if getattr(self, "need_clear_kv_cache", False):
                kv_mirror_states.clear()
        _log_hit(kind, "cuda_graph", int(hidden_states.shape[0]), prepared)
        k, v = _q_backed_kv_placeholders(q)
        return WeLMV45_80A3H2048HD256PreAttnV2Result(
            q=q, k=k, v=v, hidden_states=hidden_states
        )

    def _try_mk_h2048_hd256_pre_attn_v2(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
        kv_mirror_states: Optional[dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Optional[WeLMV45_80A3H2048HD256PreAttnV2Result]:
        if _requires_attention_backend_kv_write(forward_batch):
            return None
        kind = self._mk_runtime_projection_kind(forward_batch)
        rows = int(hidden_states.shape[0]) if hidden_states.ndim == 2 else 0
        if not self._mk_v2_layer_contract(kind, rows):
            return None
        if kind == "mirror_consumer" and self._mk_mirror_requires_projection_contract(
            forward_batch
        ):
            return None

        indexed_inputs = None
        indexed_cache_loc = None
        if kind == "indexed_mtp":
            indexed_inputs = self._mk_prepare_indexed_mtp_v2_inputs(
                positions, hidden_states, forward_batch, kv_mirror_states
            )
            if indexed_inputs is None:
                return None
            indexed_cache_loc = self._mk_cache_loc_for_rows(
                forward_batch, int(indexed_inputs["kv_indices"].numel())
            )
            if indexed_cache_loc is None:
                return None
            indexed_inputs["cache_loc"] = indexed_cache_loc

        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except RuntimeError:
            capturing = False
        if capturing:
            entry = getattr(self, "_mk_h2048_hd256_pre_attn_v2_graph_ops", {}).get(
                (kind, rows)
            )
            if entry is None:
                return None
            return self._run_graph_v2(
                entry,
                hidden_states,
                kv_mirror_states,
                indexed_inputs=indexed_inputs,
            )

        if kind == "idle_mtp":
            try:
                prepared = self._prepare_v2(
                    kind,
                    hidden_states,
                    positions,
                    hidden_states,
                    hidden_states,
                    positions,
                )
                q, k, v = prepared.launch()
            except _MKCapabilityUnavailable:
                return None
            except Exception as exc:
                api = _load_mk_v2_api()
                if not isinstance(exc, api["config_error"]):
                    raise
                _log_config_fallback(kind, rows, exc)
                return None
            _log_hit(kind, "eager", rows, prepared)
            return WeLMV45_80A3H2048HD256PreAttnV2Result(
                q=q,
                k=k,
                v=v,
                hidden_states=hidden_states,
                kv_cache_written=False,
            )

        token_to_kv_pool = forward_batch.token_to_kv_pool
        kv_rows = (
            int(indexed_inputs["kv_indices"].numel())
            if indexed_inputs is not None
            else rows
        )
        cache_loc = (
            indexed_cache_loc
            if indexed_cache_loc is not None
            else self._mk_cache_loc_for_rows(forward_batch, kv_rows)
        )
        if cache_loc is None:
            return None
        key_cache = token_to_kv_pool.get_key_buffer(self.layer_idx)
        value_cache = token_to_kv_pool.get_value_buffer(self.layer_idx)
        if kind != "indexed_mtp":
            if not self._mk_v2_io_contract(
                hidden_states, positions, key_cache, value_cache, cache_loc
            ):
                return None

        raw_k = raw_v = None
        mirror_key = None
        if kind == "mirror_source":
            if kv_mirror_states is None:
                return None
        elif kind == "mirror_consumer":
            if kv_mirror_states is None:
                return None
            mirror_key = self._mk_mirror_consumer_key()
            mirror_kv = kv_mirror_states.get(mirror_key)
            if mirror_kv is None:
                return None
            raw_k, raw_v = mirror_kv
            expected = (rows, 256)
            if tuple(raw_k.shape) != expected or tuple(raw_v.shape) != expected:
                return None

        if kind == "indexed_mtp":
            raw_k = indexed_inputs["raw_k"]
            raw_v = indexed_inputs["raw_v"]

        try:
            prepared = self._prepare_v2(
                kind,
                hidden_states,
                (
                    indexed_inputs["q_positions"]
                    if indexed_inputs is not None
                    else positions
                ),
                key_cache,
                value_cache,
                cache_loc,
                raw_k=raw_k,
                raw_v=raw_v,
                q_gather_indices=(
                    indexed_inputs["q_indices"] if indexed_inputs is not None else None
                ),
                kv_gather_indices=(
                    indexed_inputs["kv_indices"] if indexed_inputs is not None else None
                ),
                kv_positions=(
                    indexed_inputs["kv_positions"]
                    if indexed_inputs is not None
                    else None
                ),
            )
            q = prepared.launch()
        except _MKCapabilityUnavailable:
            return None
        except Exception as exc:
            api = _load_mk_v2_api()
            if not isinstance(exc, api["config_error"]):
                raise
            _log_config_fallback(kind, rows, exc)
            return None

        if kind == "mirror_source":
            outputs = tuple(prepared.mirror_outputs)
            if len(outputs) != len(self._mk_mirror_source_keys()):
                raise RuntimeError("MK Pre-Attn V2 mirror-source bank count changed")
            kv_mirror_states.update(zip(self._mk_mirror_source_keys(), outputs))
        elif kind == "mirror_consumer":
            kv_mirror_states.pop(mirror_key, None)
            if getattr(self, "need_clear_kv_cache", False):
                kv_mirror_states.clear()
        elif kind == "indexed_mtp" and getattr(self, "need_clear_kv_cache", False):
            kv_mirror_states.clear()
        _log_hit(kind, "eager", rows, prepared)
        k, v = _q_backed_kv_placeholders(q)
        project_hidden = hidden_states
        if kind == "indexed_mtp":
            project_hidden = getattr(prepared, "gathered_hidden_states", None)
            if project_hidden is None:
                project_hidden = hidden_states.index_select(
                    0, indexed_inputs["q_indices"]
                )
        return WeLMV45_80A3H2048HD256PreAttnV2Result(
            q=q, k=k, v=v, hidden_states=project_hidden
        )
