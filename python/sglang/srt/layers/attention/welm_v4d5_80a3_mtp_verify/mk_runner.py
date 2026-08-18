from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    AUTO_DATAFLOW as _AUTO_DATAFLOW,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    HEAD_DIM as _HEAD_DIM,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    KV_HEADS as _KV_HEADS,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    PAGE_SIZE as _PAGE_SIZE,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    Q_HEADS as _Q_HEADS,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    QUERY_TOKENS as _QUERY_TOKENS,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    SUPPORTED_WINDOWS as _SUPPORTED_WINDOWS,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    is_mk_backend_requested as is_mk_verify_attention_enabled,
)

logger = logging.getLogger(__name__)


def _aligned_empty(size: int, alignment: int, device: torch.device) -> torch.Tensor:
    owner = torch.empty(size + alignment - 1, dtype=torch.uint8, device=device)
    offset = (-int(owner.data_ptr())) % alignment
    return owner[offset : offset + size]


@dataclass
class _Execution:
    plan: object
    page_indices: torch.Tensor
    workspace: torch.Tensor
    partial_scratch: torch.Tensor | None


@dataclass(frozen=True)
class SelfCheckMetrics:
    window_size: tuple[int, int]
    mismatches: int
    max_abs_diff: float
    mean_abs_diff: float


class MkVerifyAttention:
    """Optional fixed-shape MK fast path for linear WeLM MTP target verify."""

    def __init__(self, model_runner):
        self._enabled = is_mk_verify_attention_enabled()
        self._model_runner = model_runner
        model_config = model_runner.model_config
        hf_config = getattr(model_config, "hf_text_config", None) or getattr(
            model_config, "hf_config", None
        )
        architectures = getattr(hf_config, "architectures", None) or ()
        self._is_welm = getattr(hf_config, "model_type", None) == "welmv4_moe" or any(
            str(name).startswith("WeLMV4MoeForCausalLM") for name in architectures
        )
        self._forward_batch = None
        self._metadata = None
        self._cuda_graph = False
        self._executions: dict[tuple[int, int], _Execution] = {}
        self._graph_executions: dict[tuple[int, int, int], _Execution] = {}
        self._graph_executions_by_bs: dict[
            int, list[tuple[tuple[int, int, int], _Execution]]
        ] = {}
        self._graph_failed_keys: set[tuple[int, int, int]] = set()
        self._failed_keys: set[tuple[int, int]] = set()
        self._api = None
        self._fallback_errors = ()
        self._import_attempted = False
        self._self_check_complete = False
        self._self_check_passed = False
        self.self_check_metrics: tuple[SelfCheckMetrics, ...] = ()
        self.last_fallback_detail: str | None = None

    def begin_forward(self, forward_batch, metadata, *, cuda_graph: bool) -> None:
        self._forward_batch = forward_batch
        self._metadata = metadata
        self._cuda_graph = cuda_graph
        self._executions.clear()
        self._failed_keys.clear()

    def precompile_cuda_graph_execution(
        self,
        page_table_shape: tuple[int, int],
        page_indices: torch.Tensor,
        window_size: tuple[int, int],
    ) -> _Execution | None:
        """Create fixed-address graph resources before graph capture starts."""

        key = (
            int(page_table_shape[0]),
            int(page_table_shape[1]),
            int(window_size[0]),
        )
        execution = self._graph_executions.get(key)
        if execution is not None:
            return execution
        if key in self._graph_failed_keys or not self._load_api():
            return None
        graph_fallback_errors = self._fallback_errors + (TypeError,)
        try:
            batch_size, max_pages = page_table_shape
            host_indptr = [row * max_pages for row in range(batch_size + 1)]
            host_cache_lens = [max_pages * _PAGE_SIZE] * batch_size
            host_cu_q = [row * _QUERY_TOKENS for row in range(batch_size + 1)]
            plan_fn, _, _, _ = self._api
            sm_count = torch.cuda.get_device_properties(
                page_indices.device
            ).multi_processor_count
            partial_capacity = 4 * int(sm_count) + 2 * int(batch_size)
            record_capacity = partial_capacity + int(batch_size)
            plan = plan_fn(
                page_table_shape,
                host_indptr,
                host_cache_lens,
                host_cu_q,
                page_indices.device,
                max_seqlen_q=_QUERY_TOKENS,
                causal=True,
                softcap=0.0,
                window_size=window_size,
                attention_dataflow=_AUTO_DATAFLOW,
                partial_merge_mode="two_kernel",
                dynamic_runtime_metadata=True,
                runtime_plan_capacity=(
                    record_capacity,
                    int(batch_size),
                    partial_capacity,
                ),
            )
            workspace = _aligned_empty(
                int(plan.workspace_size),
                int(plan.workspace_alignment),
                plan.device,
            )
            partial_scratch = None
            if int(plan.partial_scratch_size) > 0:
                partial_scratch = _aligned_empty(
                    int(plan.partial_scratch_size),
                    int(plan.partial_scratch_alignment),
                    plan.device,
                )
            execution = _Execution(plan, page_indices, workspace, partial_scratch)
            self._graph_executions[key] = execution
            self._graph_executions_by_bs.setdefault(int(batch_size), []).append(
                (key, execution)
            )
            return execution
        except graph_fallback_errors as exc:
            self._graph_failed_keys.add(key)
            self.last_fallback_detail = f"{type(exc).__name__}: {exc}"
            logger.warning_once(
                "MK could not precompile a CUDA Graph verify-attention "
                "variant; it will use the configured fallback backend: %s",
                exc,
            )
            return None

    def _load_api(self) -> bool:
        if self._import_attempted:
            return self._api is not None
        self._import_attempted = True
        try:
            from mk.errors import MKCompileError, MKConfigError
            from mk.kernels.verify_attention_welmv45 import (
                verify_attention_welmv45_plan,
                verify_attention_welmv45_prepare,
                verify_attention_welmv45_replan,
                verify_attention_welmv45_run,
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            self.last_fallback_detail = f"{type(exc).__name__}: {exc}"
            logger.warning_once(
                "MK WeLM verify attention is unavailable; using the configured "
                "attention backend instead: %s",
                exc,
            )
            return False
        self._api = (
            verify_attention_welmv45_plan,
            verify_attention_welmv45_replan,
            verify_attention_welmv45_prepare,
            verify_attention_welmv45_run,
        )
        self._fallback_errors = (MKConfigError, MKCompileError)
        return True

    def ensure_self_check(self, device: torch.device) -> bool:
        """Run one startup-only MK/FA3 numerical check for Full and SWA512."""

        if self._self_check_complete:
            return self._self_check_passed
        self._self_check_complete = True
        if not self._load_api():
            return False

        try:
            metrics = (
                self._run_self_check_case(device, (-1, -1), (32, 48)),
                self._run_self_check_case(device, (512, 0), (528, 544)),
            )
        except self._fallback_errors as exc:
            self.last_fallback_detail = f"{type(exc).__name__}: {exc}"
            return False
        except (ImportError, TypeError) as exc:
            self.last_fallback_detail = f"{type(exc).__name__}: {exc}"
            return False

        self.self_check_metrics = metrics
        failed = [item for item in metrics if item.mismatches]
        if failed:
            self.last_fallback_detail = "; ".join(
                f"window={item.window_size} mismatches={item.mismatches} "
                f"max_abs={item.max_abs_diff:.6f}"
                for item in failed
            )
            logger.error(
                "MK WeLM V4D5 80A3 verify attention self-check failed: %s",
                self.last_fallback_detail,
            )
            return False

        self._self_check_passed = True
        logger.info(
            "MK WeLM V4D5 80A3 verify attention self-check passed: %s",
            ", ".join(
                f"window={item.window_size} max_abs={item.max_abs_diff:.6f} "
                f"mean_abs={item.mean_abs_diff:.6f}"
                for item in metrics
            ),
        )
        return True

    def _run_self_check_case(
        self,
        device: torch.device,
        window_size: tuple[int, int],
        cache_lens: tuple[int, ...],
    ) -> SelfCheckMetrics:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache

        batch_size = len(cache_lens)
        page_counts = tuple(
            (length + _PAGE_SIZE - 1) // _PAGE_SIZE for length in cache_lens
        )
        max_pages = max(page_counts)
        page_indptr = [0]
        for count in page_counts:
            page_indptr.append(page_indptr[-1] + count)
        num_pages = page_indptr[-1]
        page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
        dense_page_table = torch.zeros(
            (batch_size, max_pages), dtype=torch.int32, device=device
        )
        offset = 0
        for row, count in enumerate(page_counts):
            dense_page_table[row, :count].copy_(page_indices[offset : offset + count])
            offset += count

        generator = torch.Generator(device=device)
        generator.manual_seed(20260810 + int(window_size[0]))
        query = torch.randn(
            batch_size * _QUERY_TOKENS,
            _Q_HEADS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        key_cache = torch.randn(
            num_pages,
            _PAGE_SIZE,
            _KV_HEADS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        value_cache = torch.randn(
            key_cache.shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        sinks = torch.randn(
            _Q_HEADS,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        mk_output = torch.empty_like(query)
        fa3_output = torch.empty_like(query)
        cu_seqlens_q_host = [row * _QUERY_TOKENS for row in range(batch_size + 1)]

        plan_fn, _, prepare_fn, run_fn = self._api
        plan = plan_fn(
            (batch_size, max_pages),
            page_indptr,
            cache_lens,
            cu_seqlens_q_host,
            device,
            max_seqlen_q=_QUERY_TOKENS,
            causal=True,
            softcap=0.0,
            window_size=window_size,
            attention_dataflow=_AUTO_DATAFLOW,
            partial_merge_mode="two_kernel",
        )
        workspace = _aligned_empty(
            int(plan.workspace_size), int(plan.workspace_alignment), plan.device
        )
        partial_scratch = None
        if int(plan.partial_scratch_size) > 0:
            partial_scratch = _aligned_empty(
                int(plan.partial_scratch_size),
                int(plan.partial_scratch_alignment),
                plan.device,
            )

        prepare_fn(plan, workspace)
        run_fn(
            plan,
            query,
            key_cache,
            value_cache,
            page_indices,
            sinks,
            mk_output,
            workspace,
            partial_scratch,
            1.0 / (_HEAD_DIM**0.5),
        )
        flash_attn_with_kvcache(
            q=query,
            k_cache=key_cache,
            v_cache=value_cache,
            page_table=dense_page_table,
            cache_seqlens=torch.tensor(cache_lens, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(
                0,
                (batch_size + 1) * _QUERY_TOKENS,
                _QUERY_TOKENS,
                dtype=torch.int32,
                device=device,
            ),
            max_seqlen_q=_QUERY_TOKENS,
            softmax_scale=1.0 / (_HEAD_DIM**0.5),
            causal=True,
            window_size=window_size,
            softcap=0.0,
            num_splits=0,
            out=fa3_output,
            ver=3,
            sinks=sinks,
        )
        torch.cuda.synchronize(device)
        diff = (mk_output.float() - fa3_output.float()).abs()
        close = torch.isclose(
            mk_output.float(), fa3_output.float(), atol=3.0e-2, rtol=3.0e-2
        )
        return SelfCheckMetrics(
            window_size=window_size,
            mismatches=int((~close).sum().item()),
            max_abs_diff=float(diff.max().item()),
            mean_abs_diff=float(diff.mean().item()),
        )

    def _is_supported(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        page_table: torch.Tensor,
        layer,
        forward_batch,
        window_size: tuple[int, int],
        sinks: torch.Tensor | None,
        *,
        causal: bool,
        has_unsupported_layout: bool,
    ) -> bool:
        if not self._enabled or not self._is_welm or has_unsupported_layout:
            return False
        if self._forward_batch is not forward_batch or self._metadata is None:
            return False
        if not forward_batch.forward_mode.is_target_verify():
            return False

        spec_info = getattr(forward_batch, "spec_info", None)
        if (
            spec_info is None
            or int(getattr(spec_info, "topk", -1)) != 1
            or int(getattr(spec_info, "spec_steps", -1)) != _QUERY_TOKENS - 1
            or int(getattr(spec_info, "draft_token_num", -1)) != _QUERY_TOKENS
            or int(getattr(spec_info, "num_tokens_per_req", -1)) != _QUERY_TOKENS
        ):
            return False
        if (
            getattr(self._model_runner.server_args, "speculative_algorithm", None)
            != "EAGLE"
        ):
            return False

        batch_size = int(forward_batch.batch_size)
        if (
            int(getattr(self._model_runner, "page_size", -1)) != _PAGE_SIZE
            or not causal
            or window_size not in _SUPPORTED_WINDOWS
            or float(getattr(layer, "logit_cap", 0.0)) != 0.0
            or int(getattr(layer, "tp_q_head_num", -1)) != _Q_HEADS
            or int(getattr(layer, "tp_k_head_num", -1)) != _KV_HEADS
            or int(getattr(layer, "tp_v_head_num", -1)) != _KV_HEADS
            or int(getattr(layer, "head_dim", -1)) != _HEAD_DIM
            or int(getattr(layer, "v_head_dim", -1)) != _HEAD_DIM
            or query.shape != (batch_size * _QUERY_TOKENS, _Q_HEADS, _HEAD_DIM)
            or query.dtype is not torch.bfloat16
            or not query.is_contiguous()
        ):
            return False
        expected_cache_tail = (_PAGE_SIZE, _KV_HEADS, _HEAD_DIM)
        if (
            key_cache.ndim != 4
            or value_cache.ndim != 4
            or tuple(key_cache.shape[1:]) != expected_cache_tail
            or tuple(value_cache.shape[1:]) != expected_cache_tail
            or key_cache.dtype is not torch.bfloat16
            or value_cache.dtype is not torch.bfloat16
            or key_cache.device != query.device
            or value_cache.device != query.device
            or page_table.device != query.device
            or not key_cache.is_contiguous()
            or not value_cache.is_contiguous()
            or page_table.ndim != 2
            or page_table.shape[0] != batch_size
            or page_table.dtype is not torch.int32
            or not page_table.is_contiguous()
        ):
            return False
        if sinks is not None and (
            sinks.shape != (_Q_HEADS,)
            or sinks.dtype is not torch.bfloat16
            or sinks.device != query.device
            or not sinks.is_contiguous()
        ):
            return False
        if query.device.type != "cuda" or query.device != key_cache.device:
            return False
        return torch.cuda.get_device_capability(query.device) == (9, 0)

    def _host_cache_lens(self) -> tuple[int, ...] | None:
        seq_lens_cpu = getattr(self._forward_batch, "seq_lens_cpu", None)
        if (
            not isinstance(seq_lens_cpu, torch.Tensor)
            or seq_lens_cpu.device.type != "cpu"
        ):
            return None
        values = tuple(int(value) + _QUERY_TOKENS for value in seq_lens_cpu.tolist())
        if len(values) != int(self._forward_batch.batch_size):
            return None
        return values

    def _make_execution(
        self,
        page_table: torch.Tensor,
        window_size: tuple[int, int],
    ) -> _Execution | None:
        cache_lens = self._host_cache_lens()
        if cache_lens is None:
            return None
        page_counts = tuple(
            (length + _PAGE_SIZE - 1) // _PAGE_SIZE for length in cache_lens
        )
        if not page_counts or max(page_counts) > int(page_table.shape[1]):
            return None

        page_indptr = [0]
        for count in page_counts:
            page_indptr.append(page_indptr[-1] + count)
        page_indices = torch.cat(
            [page_table[row, :count] for row, count in enumerate(page_counts)]
        ).contiguous()
        cu_seqlens_q = [row * _QUERY_TOKENS for row in range(len(cache_lens) + 1)]

        plan_fn, _, _, _ = self._api
        plan = plan_fn(
            (len(cache_lens), int(page_table.shape[1])),
            page_indptr,
            cache_lens,
            cu_seqlens_q,
            page_table.device,
            max_seqlen_q=_QUERY_TOKENS,
            causal=True,
            softcap=0.0,
            window_size=window_size,
        )
        workspace = _aligned_empty(
            int(plan.workspace_size), int(plan.workspace_alignment), plan.device
        )
        partial_scratch = None
        if int(plan.partial_scratch_size) > 0:
            partial_scratch = _aligned_empty(
                int(plan.partial_scratch_size),
                int(plan.partial_scratch_alignment),
                plan.device,
            )
        return _Execution(plan, page_indices, workspace, partial_scratch)

    def try_run(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        page_table: torch.Tensor,
        layer,
        forward_batch,
        window_size: tuple[int, int],
        sinks: torch.Tensor | None,
        output: torch.Tensor | None,
        *,
        causal: bool,
        has_unsupported_layout: bool,
    ) -> torch.Tensor | None:
        if self._cuda_graph:
            return None
        if not self._is_supported(
            query,
            key_cache,
            value_cache,
            page_table,
            layer,
            forward_batch,
            window_size,
            sinks,
            causal=causal,
            has_unsupported_layout=has_unsupported_layout,
        ):
            return None
        if not self._load_api():
            return None

        key = (int(window_size[0]), int(page_table.data_ptr()))
        if key in self._failed_keys:
            return None
        try:
            execution = self._executions.get(key)
            if execution is None:
                execution = self._make_execution(page_table, window_size)
                if execution is None:
                    return None
                self._executions[key] = execution
            if output is None:
                output = torch.empty_like(query)
            _, _, prepare_fn, run_fn = self._api
            prepare_fn(execution.plan, execution.workspace)
            return run_fn(
                execution.plan,
                query,
                key_cache,
                value_cache,
                execution.page_indices,
                sinks,
                output,
                execution.workspace,
                execution.partial_scratch,
                float(layer.scaling),
            )
        except self._fallback_errors as exc:
            self._failed_keys.add(key)
            self.last_fallback_detail = f"{type(exc).__name__}: {exc}"
            logger.warning_once(
                "MK rejected a WeLM verify-attention configuration; falling back "
                "to the configured attention backend: %s",
                exc,
            )
            return None

    def try_run_cuda_graph(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        page_indices: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table_shape: tuple[int, int],
        layer,
        forward_batch,
        window_size: tuple[int, int],
        sinks: torch.Tensor | None,
        output: torch.Tensor | None,
        *,
        causal: bool,
        has_unsupported_layout: bool,
    ) -> torch.Tensor | None:
        if not self._cuda_graph or not self._is_supported(
            query,
            key_cache,
            value_cache,
            page_indices.view(page_table_shape),
            layer,
            forward_batch,
            window_size,
            sinks,
            causal=causal,
            has_unsupported_layout=has_unsupported_layout,
        ):
            return None
        if not self._load_api():
            return None
        if output is None:
            output = torch.empty_like(query)
        key = (
            page_table_shape[0],
            page_table_shape[1],
            int(window_size[0]),
        )
        if key in self._graph_failed_keys:
            return None
        graph_fallback_errors = self._fallback_errors + (TypeError,)
        try:
            execution = self._graph_executions.get(key)
            if execution is None:
                execution = self.precompile_cuda_graph_execution(
                    page_table_shape,
                    page_indices,
                    window_size,
                )
                if execution is None:
                    return None
            _, _, prepare_fn, run_fn = self._api
            prepare_fn(execution.plan, execution.workspace)
            return run_fn(
                execution.plan,
                query,
                key_cache,
                value_cache,
                page_indices,
                sinks,
                output,
                execution.workspace,
                execution.partial_scratch,
                float(layer.scaling),
                runtime_cache_seqlens=cache_seqlens,
            )
        except graph_fallback_errors as exc:
            self._graph_failed_keys.add(key)
            self.last_fallback_detail = f"{type(exc).__name__}: {exc}"
            logger.warning_once(
                "MK rejected CUDA Graph WeLM verify attention; falling back to "
                "the configured attention backend: %s",
                exc,
            )
            return None

    def replan_cuda_graph(self, bs: int, seq_lens_cpu: torch.Tensor) -> None:
        if not self._graph_executions or not self._load_api():
            return
        _, replan_fn, _, _ = self._api
        for _, execution in self._graph_executions_by_bs.get(int(bs), ()):
            replan_fn(
                execution.plan,
                seq_lens_cpu,
                cache_seqlen_offset=_QUERY_TOKENS,
            )


MkVerifyAttentionRunner = MkVerifyAttention
