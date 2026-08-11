from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    PAGE_SIZE as _PAGE_SIZE,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    QUERY_TOKENS as _QUERY_TOKENS,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    FallbackReason,
    SupportDecision,
    validate_runtime_contract,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.mk_runner import (
    MkVerifyAttentionRunner as MkVerifyAttention,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


@dataclass
class _GraphBuffers:
    cache_seqlens: torch.Tensor
    full_page_indices: torch.Tensor
    swa_page_indices: torch.Tensor | None
    max_pages: int
    reuses_fallback: bool = False


class WeLMV4D5MTPVerifyAttentionBackend(AttentionBackend):
    """WeLM V4D5 80A3 target-verify MK fast path with an FA3 fallback."""

    def __init__(self, model_runner: ModelRunner, fallback: AttentionBackend):
        super().__init__()
        self.model_runner = model_runner
        self.fallback = fallback
        self.engine = MkVerifyAttention(model_runner)
        self._forward_batch = None
        self._eager_full_page_table = None
        self._eager_swa_page_table = None
        self._graph_buffers: _GraphBuffers | None = None
        self._cuda_graph_capture_bs: tuple[int, ...] = ()
        self._logged_fallbacks: set[tuple[str, int | None]] = set()
        self._fallback_counts: dict[str, int] = {}
        self._logged_mk_use = False
        self._mk_disabled_decision: SupportDecision | None = None

    def set_cuda_graph_capture_bs(self, capture_bs) -> None:
        self._cuda_graph_capture_bs = tuple(int(bs) for bs in capture_bs)

    def _log_fallback(
        self,
        decision: SupportDecision,
        *,
        layer=None,
        forward_batch=None,
    ) -> None:
        reason = (
            decision.reason.value
            if decision.reason is not None
            else FallbackReason.INTERNAL.value
        )
        self._fallback_counts[reason] = self._fallback_counts.get(reason, 0) + 1
        layer_id = getattr(layer, "layer_id", None)
        key = (reason, None)
        if key in self._logged_fallbacks:
            return
        self._logged_fallbacks.add(key)
        logger.warning(
            "WeLM V4D5 80A3 MTP verify attention fallback: "
            "requested_backend=mk selected_backend=fa3 reason=%s actual=%r "
            "expected=%r detail=%r tp_rank=%s layer_id=%s batch_size=%s "
            "fallback_count=%d",
            reason,
            decision.actual,
            decision.expected,
            decision.detail,
            getattr(self.model_runner, "tp_rank", None),
            layer_id,
            getattr(forward_batch, "batch_size", None),
            self._fallback_counts[reason],
        )

    def _log_mk_use(self, layer, forward_batch, *, cuda_graph: bool) -> None:
        if self._logged_mk_use:
            return
        self._logged_mk_use = True
        logger.info(
            "Using MK WeLM V4D5 80A3 MTP verify attention: "
            "tp_rank=%s batch_size=%s q_per_request=%d local_q_heads=%s "
            "local_kv_heads=%s head_dim=%s page_size=%d cuda_graph=%s",
            getattr(self.model_runner, "tp_rank", None),
            getattr(forward_batch, "batch_size", None),
            _QUERY_TOKENS,
            getattr(layer, "tp_q_head_num", None),
            getattr(layer, "tp_k_head_num", None),
            getattr(layer, "head_dim", None),
            _PAGE_SIZE,
            cuda_graph,
        )

    def __getattr__(self, name):
        fallback = self.__dict__.get("fallback")
        if fallback is None:
            raise AttributeError(name)
        return getattr(fallback, name)

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name in {
            "cuda_graph_max_seq_len",
            "_cuda_graph_seq_len_fill_value",
            "_replay_forward_batch",
        }:
            fallback = self.__dict__.get("fallback")
            if fallback is not None:
                setattr(fallback, name, value)

    def _is_swa_layer(self, layer: RadixAttention, pool) -> bool:
        mapping = getattr(pool, "layers_mapping", None)
        entry = mapping.get(layer.layer_id) if isinstance(mapping, dict) else None
        return bool(entry is not None and len(entry) > 1 and entry[1])

    @staticmethod
    def _translate_swa(pool, raw_locations: torch.Tensor) -> torch.Tensor | None:
        translate = getattr(pool, "translate_loc_from_full_to_swa", None)
        if translate is None:
            return None
        return (
            (translate(raw_locations) // _PAGE_SIZE)
            .clamp_min(0)
            .to(torch.int32)
            .contiguous()
        )

    def _build_eager_page_tables(self, forward_batch: ForwardBatch) -> None:
        self._eager_full_page_table = self._eager_swa_page_table = None
        if not forward_batch.forward_mode.is_target_verify():
            return
        fallback_metadata = self._fallback_target_metadata()
        if fallback_metadata is not None:
            full = getattr(fallback_metadata, "page_table", None)
            swa = getattr(fallback_metadata, "swa_page_table", None)
            if isinstance(full, torch.Tensor):
                self._eager_full_page_table = full
                self._eager_swa_page_table = swa
                return
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if not isinstance(seq_lens_cpu, torch.Tensor) or seq_lens_cpu.numel() == 0:
            return
        max_cache_len = int(seq_lens_cpu.max().item()) + _QUERY_TOKENS
        req_pool = forward_batch.req_to_token_pool
        raw = req_pool.req_to_token[
            forward_batch.req_pool_indices, :max_cache_len:_PAGE_SIZE
        ]
        self._eager_full_page_table = (
            (raw // _PAGE_SIZE).clamp_min(0).to(torch.int32).contiguous()
        )
        self._eager_swa_page_table = self._translate_swa(
            forward_batch.token_to_kv_pool, raw
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.fallback.init_forward_metadata(forward_batch)
        self._forward_batch = forward_batch
        self._build_eager_page_tables(forward_batch)
        self.engine.begin_forward(forward_batch, object(), cuda_graph=False)

    def _fallback_target_backend(self):
        """Return the fallback backend that owns target-verify metadata."""

        backend = self.fallback
        # HybridAttnBackend routes target verify to prefill by default and to
        # decode only when explicitly requested.  Avoid constructing a
        # ForwardMode here: graph state is initialized before a batch exists.
        for _ in range(4):
            prefill = getattr(backend, "prefill_backend", None)
            decode = getattr(backend, "decode_backend", None)
            if prefill is None or decode is None:
                break
            mode = getattr(
                self.model_runner.server_args,
                "speculative_attention_mode",
                "prefill",
            )
            backend = decode if mode == "decode" else prefill
        return backend

    def _fallback_target_metadata(self):
        backend = self._fallback_target_backend()
        metadata = getattr(backend, "forward_metadata", None)
        return metadata

    def _ensure_fallback_target_graph_capacity(self, required_max_pages: int) -> bool:
        """Let MK share FA3 target-verify metadata, including the MTP tail page.

        ``ReqToTokenPool`` reserves a few speculative-token slots beyond the
        advertised model context.  For a page-aligned context this requires one
        more page-table column than FA3 normally allocates.  Allocating a second
        MK-owned full/SWA table avoids that capacity mismatch, but it also
        rebuilds both tables on every replay.  At low batch size that duplicate
        metadata work is large enough to hide the verify-attention speedup.

        Grow FA3's target-only buffers before graph capture instead.  Both
        backends then consume the same tensors and the normal FA3 replay setup
        remains the single producer of page-table metadata.
        """

        backend = self._fallback_target_backend()
        state = getattr(backend, "target_verify_metadata", None)
        if not isinstance(state, dict):
            return False

        full = state.get("page_table")
        cache_seqlens = state.get("cache_seqlens")
        if not (
            isinstance(full, torch.Tensor)
            and isinstance(cache_seqlens, torch.Tensor)
            and full.ndim == 2
            and full.dtype == torch.int32
            and cache_seqlens.dtype == torch.int32
        ):
            return False

        current_max_pages = int(full.shape[1])
        if current_max_pages >= required_max_pages:
            return True

        kv_pool = self.model_runner.token_to_kv_pool
        has_swa = hasattr(kv_pool, "translate_loc_from_full_to_swa")
        swa = state.get("swa_page_table")
        if has_swa and not (
            isinstance(swa, torch.Tensor)
            and swa.ndim == 2
            and swa.dtype == torch.int32
            and swa.shape == full.shape
        ):
            return False

        state["page_table"] = full.new_zeros((int(full.shape[0]), required_max_pages))
        if has_swa:
            state["swa_page_table"] = swa.new_zeros(
                (int(swa.shape[0]), required_max_pages)
            )

        # FlashAttention replay indexes ReqToTokenPool through this vector.
        # Extend both copies used by current FA3 versions so the additional
        # speculative tail page is safe in the shared-buffer configuration.
        for metadata in (
            state,
            getattr(backend, "decode_cuda_graph_metadata", None),
        ):
            if not isinstance(metadata, dict):
                continue
            strided_indices = metadata.get("strided_indices")
            if not isinstance(strided_indices, torch.Tensor):
                continue
            if int(strided_indices.numel()) >= required_max_pages:
                continue
            metadata["strided_indices"] = (
                torch.arange(
                    required_max_pages,
                    dtype=strided_indices.dtype,
                    device=strided_indices.device,
                )
                * _PAGE_SIZE
            )

        return True

    def _fallback_graph_buffers(self, required_max_pages: int) -> _GraphBuffers | None:
        backend = self._fallback_target_backend()
        state = getattr(backend, "target_verify_metadata", None)
        if not isinstance(state, dict):
            return None
        cache_seqlens = state.get("cache_seqlens")
        full = state.get("page_table")
        swa = state.get("swa_page_table")
        if not (
            isinstance(cache_seqlens, torch.Tensor)
            and isinstance(full, torch.Tensor)
            and cache_seqlens.dtype == torch.int32
            and full.dtype == torch.int32
            and full.ndim == 2
            and int(full.shape[1]) >= required_max_pages
        ):
            return None
        kv_pool = self.model_runner.token_to_kv_pool
        has_swa = hasattr(kv_pool, "translate_loc_from_full_to_swa")
        if has_swa and not (
            isinstance(swa, torch.Tensor)
            and swa.dtype == torch.int32
            and swa.shape == full.shape
        ):
            return None
        return _GraphBuffers(
            cache_seqlens=cache_seqlens,
            full_page_indices=full,
            swa_page_indices=swa if has_swa else None,
            max_pages=int(full.shape[1]),
            reuses_fallback=True,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.fallback.init_cuda_graph_state(max_bs, max_num_tokens)
        device = torch.device(self.model_runner.device)
        if not self.engine.ensure_self_check(device):
            self._mk_disabled_decision = SupportDecision.reject(
                FallbackReason.SELF_CHECK,
                detail=self.engine.last_fallback_detail,
            )
            self._log_fallback(self._mk_disabled_decision)
            return
        # Speculative verification temporarily addresses draft-token positions
        # past the advertised model context.  ReqToTokenPool deliberately owns
        # that extra tail, so size the MK page table from the pool rather than
        # truncating it to model_config.context_len.  Otherwise a request near
        # an exact page-aligned context boundary (for example 32K) makes MK's
        # runtime replan reject cache_seqlen + _QUERY_TOKENS and kills the
        # scheduler instead of completing through the normal speculative path.
        max_context_len = int(
            getattr(
                self.model_runner.req_to_token_pool,
                "max_context_len",
                self.model_runner.model_config.context_len,
            )
        )
        max_pages = (max_context_len + _PAGE_SIZE - 1) // _PAGE_SIZE
        kv_pool = self.model_runner.token_to_kv_pool
        has_swa = hasattr(kv_pool, "translate_loc_from_full_to_swa")
        self._ensure_fallback_target_graph_capacity(max_pages)
        self._graph_buffers = self._fallback_graph_buffers(max_pages)
        if self._graph_buffers is None:
            self._graph_buffers = _GraphBuffers(
                cache_seqlens=torch.empty(max_bs, dtype=torch.int32, device=device),
                full_page_indices=torch.empty(
                    (max_bs, max_pages), dtype=torch.int32, device=device
                ),
                swa_page_indices=(
                    torch.empty((max_bs, max_pages), dtype=torch.int32, device=device)
                    if has_swa
                    else None
                ),
                max_pages=max_pages,
            )
        else:
            max_pages = self._graph_buffers.max_pages
        # JIT and allocate every fixed-address plan before the outer CUDA Graph
        # capture context starts. Allocating these lazily from a graph warmup
        # can make mutually exclusive variants alias the same capture-pool
        # addresses, leaving only the last-captured variant valid.
        configured_bs = getattr(self, "_cuda_graph_capture_bs", None)
        if not configured_bs:
            configured_bs = getattr(
                self.model_runner.server_args, "cuda_graph_bs", None
            )
        # The runner publishes its exact sparse capture buckets before calling
        # this hook. The max-bs fallback keeps direct/unit callers bounded
        # without allocating plans for every integer batch size.
        capture_bs = sorted(
            {int(bs) for bs in (configured_bs or (max_bs,)) if 0 < int(bs) <= max_bs}
        )
        windows = [(-1, -1)]
        if has_swa:
            windows.append((512, 0))
        for bs in capture_bs:
            for window_size in windows:
                pages = (
                    self._graph_buffers.swa_page_indices
                    if window_size[0] == 512
                    else self._graph_buffers.full_page_indices
                )
                if pages is None:
                    continue
                execution = self.engine.precompile_cuda_graph_execution(
                    (bs, max_pages),
                    pages[:bs].reshape(-1),
                    window_size,
                )
                if execution is None:
                    self._log_fallback(
                        SupportDecision.reject(
                            FallbackReason.GRAPH_PLAN,
                            actual={"batch_size": bs, "window": window_size},
                            expected="precompiled MK CUDA Graph execution",
                            detail=self.engine.last_fallback_detail,
                        )
                    )

    def _fill_graph_buffers(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        valid_pages: int | None = None,
    ) -> None:
        buffers = self._graph_buffers
        if buffers is None:
            return
        buffers.cache_seqlens[:bs].copy_(
            (seq_lens[:bs] + _QUERY_TOKENS).to(torch.int32)
        )
        req_pool = self.model_runner.req_to_token_pool
        # The graph allocations keep the maximum-context stride, but runtime
        # attention only reads pages covered by the exact cache lengths.  Do
        # not rebuild the unused 256K-context tail on every replay: for common
        # short-context batches that work is a material fixed overhead.
        pages_to_copy = buffers.max_pages
        if valid_pages is not None:
            pages_to_copy = max(1, min(int(valid_pages), buffers.max_pages))
        max_tokens = pages_to_copy * _PAGE_SIZE
        raw = req_pool.req_to_token[req_pool_indices[:bs], :max_tokens:_PAGE_SIZE]
        buffers.full_page_indices[:bs, :pages_to_copy].copy_(
            (raw // _PAGE_SIZE).clamp_min(0).to(torch.int32)
        )
        if buffers.swa_page_indices is not None:
            swa = self._translate_swa(self.model_runner.token_to_kv_pool, raw)
            if swa is not None:
                buffers.swa_page_indices[:bs, :pages_to_copy].copy_(swa)

    def _copy_graph_attrs_to_fallback(self) -> list[str]:
        copied = []
        for name in ("_cuda_graph_seq_len_fill_value", "_replay_forward_batch"):
            if hasattr(self, name):
                setattr(self.fallback, name, getattr(self, name))
                copied.append(name)
        return copied

    def _clear_fallback_graph_attrs(self, names: list[str]) -> None:
        for name in names:
            setattr(self.fallback, name, None)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: torch.Tensor | None,
        forward_mode: ForwardMode,
        spec_info: SpecInput | None,
    ):
        self._forward_batch = None
        copied = self._copy_graph_attrs_to_fallback()
        try:
            self.fallback.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )
        finally:
            self._clear_fallback_graph_attrs(copied)
        self._fill_graph_buffers(bs, req_pool_indices, seq_lens)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: torch.Tensor | None,
        forward_mode: ForwardMode,
        spec_info: SpecInput | None,
        seq_lens_cpu: torch.Tensor | None,
    ):
        self._forward_batch = None
        copied = self._copy_graph_attrs_to_fallback()
        try:
            self.fallback.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )
        finally:
            self._clear_fallback_graph_attrs(copied)
        if self._graph_buffers is None or not self._graph_buffers.reuses_fallback:
            valid_pages = None
            if isinstance(seq_lens_cpu, torch.Tensor) and seq_lens_cpu.numel() >= bs:
                max_cache_len = int(seq_lens_cpu[:bs].max().item()) + _QUERY_TOKENS
                valid_pages = (max_cache_len + _PAGE_SIZE - 1) // _PAGE_SIZE
            self._fill_graph_buffers(
                bs,
                req_pool_indices,
                seq_lens,
                valid_pages=valid_pages,
            )
        if isinstance(seq_lens_cpu, torch.Tensor):
            self.engine.replan_cuda_graph(bs, seq_lens_cpu[:bs])

    def get_cuda_graph_seq_len_fill_value(self):
        return self.fallback.get_cuda_graph_seq_len_fill_value()

    def on_after_cuda_graph_warmup(self):
        return self.fallback.on_after_cuda_graph_warmup()

    def get_verify_buffers_to_fill_after_draft(self):
        return self.fallback.get_verify_buffers_to_fill_after_draft()

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        return self.fallback.update_verify_buffers_to_fill_after_draft(
            spec_info, cuda_graph_bs
        )

    def _unsupported_layout(self, layer, forward_batch) -> bool:
        return bool(
            getattr(self.model_runner, "attn_cp_size", 1) > 1
            or getattr(forward_batch, "attn_cp_prefill_runtime_layout", None)
            is not None
            or getattr(forward_batch, "attn_cp_metadata", None) is not None
            or getattr(forward_batch, "welm_kv_mirror_contracted", False)
            or getattr(forward_batch, "welm_mtp_merge_kv_fill_draft", False)
            or getattr(self.fallback, "has_local_attention", False)
            or getattr(layer, "is_cross_attention", False)
        )

    def _store_kv(self, k, v, layer, forward_batch) -> None:
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
            layer.k_scale,
            layer.v_scale,
        )

    def _try_mk(self, q, k, v, layer, forward_batch, save_kv_cache, sinks):
        if not forward_batch.forward_mode.is_target_verify():
            return None, False
        if self._mk_disabled_decision is not None:
            return None, False
        if self._graph_buffers is None and not self.engine.ensure_self_check(q.device):
            self._mk_disabled_decision = SupportDecision.reject(
                FallbackReason.SELF_CHECK,
                detail=self.engine.last_fallback_detail,
            )
            self._log_fallback(
                self._mk_disabled_decision,
                layer=layer,
                forward_batch=forward_batch,
            )
            return None, False
        if k is None or v is None or not save_kv_cache:
            self._log_fallback(
                SupportDecision.reject(
                    FallbackReason.LAYOUT,
                    actual={
                        "key_is_none": k is None,
                        "value_is_none": v is None,
                        "save_kv_cache": save_kv_cache,
                    },
                    expected="K/V tensors with save_kv_cache=True",
                ),
                layer=layer,
                forward_batch=forward_batch,
            )
            return None, False

        pool = forward_batch.token_to_kv_pool
        is_swa = self._is_swa_layer(layer, pool)
        window_size = (int(layer.sliding_window_size), 0) if is_swa else (-1, -1)
        key_cache, value_cache = pool.get_kv_buffer(layer.layer_id)
        key_cache = key_cache.view(-1, _PAGE_SIZE, layer.tp_k_head_num, layer.head_dim)
        value_cache = value_cache.view(
            -1, _PAGE_SIZE, layer.tp_v_head_num, layer.v_head_dim
        )
        query = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        output = getattr(forward_batch, "_attn_output", None)
        if output is not None:
            output = output.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        cuda_graph = self._graph_buffers is not None and self._forward_batch is None
        # CUDA graph forward batches do not pass through init_forward_metadata.
        if cuda_graph:
            buffers = self._graph_buffers
            bs = int(forward_batch.batch_size)
            pages = buffers.swa_page_indices if is_swa else buffers.full_page_indices
            if pages is None:
                self._log_fallback(
                    SupportDecision.reject(
                        FallbackReason.PAGE_TABLE,
                        actual=None,
                        expected="CUDA Graph page table",
                    ),
                    layer=layer,
                    forward_batch=forward_batch,
                )
                return None, False
            self.engine.begin_forward(forward_batch, object(), cuda_graph=True)
            decision = validate_runtime_contract(
                model_runner=self.model_runner,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                page_table=pages[:bs],
                layer=layer,
                forward_batch=forward_batch,
                window_size=window_size,
                sinks=sinks,
                causal=True,
                has_unsupported_layout=self._unsupported_layout(layer, forward_batch),
                metadata_ready=True,
            )
            if not decision.supported:
                self._log_fallback(decision, layer=layer, forward_batch=forward_batch)
                return None, False
            if not self.engine._is_supported(
                query,
                key_cache,
                value_cache,
                pages[:bs],
                layer,
                forward_batch,
                window_size,
                sinks,
                causal=True,
                has_unsupported_layout=False,
            ):
                self._log_fallback(
                    SupportDecision.reject(FallbackReason.INTERNAL),
                    layer=layer,
                    forward_batch=forward_batch,
                )
                return None, False
            self._store_kv(k, v, layer, forward_batch)
            result = self.engine.try_run_cuda_graph(
                query,
                key_cache,
                value_cache,
                pages[:bs].reshape(-1),
                buffers.cache_seqlens[:bs],
                (bs, buffers.max_pages),
                layer,
                forward_batch,
                window_size,
                sinks,
                output,
                causal=True,
                has_unsupported_layout=False,
            )
        else:
            page_table = (
                self._eager_swa_page_table if is_swa else self._eager_full_page_table
            )
            if page_table is None:
                self._log_fallback(
                    SupportDecision.reject(
                        FallbackReason.PAGE_TABLE,
                        actual=None,
                        expected="eager page table",
                    ),
                    layer=layer,
                    forward_batch=forward_batch,
                )
                return None, False
            decision = validate_runtime_contract(
                model_runner=self.model_runner,
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                page_table=page_table,
                layer=layer,
                forward_batch=forward_batch,
                window_size=window_size,
                sinks=sinks,
                causal=True,
                has_unsupported_layout=self._unsupported_layout(layer, forward_batch),
                metadata_ready=self._forward_batch is forward_batch,
            )
            if not decision.supported:
                self._log_fallback(decision, layer=layer, forward_batch=forward_batch)
                return None, False
            if not self.engine._is_supported(
                query,
                key_cache,
                value_cache,
                page_table,
                layer,
                forward_batch,
                window_size,
                sinks,
                causal=True,
                has_unsupported_layout=False,
            ):
                self._log_fallback(
                    SupportDecision.reject(FallbackReason.INTERNAL),
                    layer=layer,
                    forward_batch=forward_batch,
                )
                return None, False
            self._store_kv(k, v, layer, forward_batch)
            result = self.engine.try_run(
                query,
                key_cache,
                value_cache,
                page_table,
                layer,
                forward_batch,
                window_size,
                sinks,
                output,
                causal=True,
                has_unsupported_layout=False,
            )
        if result is None:
            self._log_fallback(
                SupportDecision.reject(
                    FallbackReason.GRAPH_PLAN if cuda_graph else FallbackReason.PLAN,
                    detail=getattr(self.engine, "last_fallback_detail", None),
                ),
                layer=layer,
                forward_batch=forward_batch,
            )
        else:
            self._log_mk_use(layer, forward_batch, cuda_graph=cuda_graph)
        return result, True

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        result, kv_stored = self._try_mk(
            q, k, v, layer, forward_batch, save_kv_cache, kwargs.get("sinks")
        )
        if result is not None:
            return result.view(-1, layer.tp_q_head_num * layer.v_head_dim)
        return self.fallback.forward_extend(
            q,
            k,
            v,
            layer,
            forward_batch,
            save_kv_cache=save_kv_cache and not kv_stored,
            **kwargs,
        )

    def forward_decode(self, *args, **kwargs):
        return self.fallback.forward_decode(*args, **kwargs)

    def get_indexer_metadata(self, layer_id: int, forward_batch: ForwardBatch):
        return self.fallback.get_indexer_metadata(layer_id, forward_batch)
