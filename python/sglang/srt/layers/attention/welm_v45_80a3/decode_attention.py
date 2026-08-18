# Vendored from the k-dash kernel source package welm/v45_80a3_attention.
# The CUDA kernel.so is resolved at runtime by k_dash.get(); that repo is a
# k-dash Source Package, not a Python distribution, so the host planner is
# mirrored here. Keep edits in sync with the upstream repo.
from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass
import heapq
import math
from typing import Sequence as TypingSequence

import numpy as np
import torch

from .kernel_runtime import (
    ConfigError as MKConfigError,
    LaunchError as MKLaunchError,
    prepare_decode_kernel,
)
from .decode_executors import (
    DECODE_DEFAULT_MAX_SPLITS,
    DECODE_DEFAULT_NUM_STAGES,
    DECODE_DEFAULT_NUM_SUB_TASKS,
    DECODE_DEFAULT_OUTPUT_MERGE_WARPS,
    resolve_decode_executor,
)


_ARG_DEFS = (
    ("query", "void*"),
    ("key_cache", "void*"),
    ("value_cache", "void*"),
    ("page_ids", "int32_t*"),
    ("output", "void*"),
    ("lse", "float*"),
    ("sinks", "void*"),
    ("split_plan", "int32_t*"),
    ("max_pages", "int32_t"),
    ("num_cache_pages", "int32_t"),
    ("num_split_records", "int32_t"),
    ("sm_scale", "float"),
    ("window_left", "int32_t"),
    ("fma_max_tokens", "int32_t"),
    ("scratch_workspace", "void*"),
    ("gate_values", "void*"),
)
_LOCAL_Q_HEADS = 6
_LOCAL_KV_HEADS = 1
_HEAD_DIM = 256
_PAGE_SIZE = 16
_DEFAULT_NUM_STAGES = 4
_DEFAULT_OUTPUT_MERGE_WARPS = 8
_SUPPORTED_NUM_STAGES = (1, 2, 3, 4, 5)
_SUPPORTED_OUTPUT_MERGE_WARPS = (1, 2, 4, 8)
_PLAN_TOKEN_FIELD = 4
_PLAN_BEGIN_TOKEN_FIELD = 5
_PLAN_WORKER_FIELD = 6
_PLAN_ACTIVE_SPLITS_FIELD = 7
_PLAN_MERGE_WORKER_FIELD = 8
_PLAN_FIELDS = 9
_DEFAULT_AUTO_NUM_SPLITS = 2
_MAX_MERGE_WORKERS = 32
_MIN_SPLIT_TAIL_TOKENS = 1024
_DEFAULT_FMA_MAX_TOKENS = 1024
_AUTO_SPLIT_TARGET_RECORDS_NUM = 5
_AUTO_SPLIT_TARGET_RECORDS_DEN = 2
_AUTO_SPLIT_SHORT_MAX_SPLITS = 20
_AUTO_SPLIT_LONG_MAX_SPLITS = 64
_AUTO_SPLIT_LONG_CONTEXT_TOKENS = 65536
_AUTO_SPLIT_MIN_TOTAL_TOKENS = 1024
_AUTO_SPLIT_MIN_CHUNK_PAGES = 4


@dataclass
class _NativePlanHandle:
    ptr: object
    library: object
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.library.plan_destroy(int(self.ptr))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True)
class _SplitPlan:
    split_plan: torch.Tensor | None
    split_token_counts: torch.Tensor
    max_split_tokens: int
    max_splits: int
    num_split_records: int
    chunk_pages: int
    worker_count: int
    split_plan_buffer: torch.Tensor | None = None
    native_handle: _NativePlanHandle | None = None


@dataclass
class DecodeAttentionWelmv45Plan:
    _prepared: PreparedDecodeKernel
    _plan: _SplitPlan
    num_sub_tasks: int
    max_pages: int
    num_cache_pages: int
    sm_scale: float
    has_sinks: bool
    window_left: int
    fma_max_tokens: int
    worker_count: int
    num_stages: int
    output_merge_warps: int
    direct_fma_only: bool
    has_headwise_gate: bool
    device: torch.device

    @property
    def max_splits(self) -> int:
        return int(self._plan.max_splits)

    @property
    def num_split_records(self) -> int:
        return int(self._plan.num_split_records)

    @property
    def workspace_size(self) -> int:
        if self._plan.native_handle is None:
            return 0
        return _native_plan_workspace_size(self._plan.native_handle)

    @property
    def scratch_workspace_size(self) -> int:
        if self._plan.native_handle is None:
            return 0
        return _native_plan_scratch_workspace_size(self._plan.native_handle)

    def close(self) -> None:
        _close_native_plan(self._plan)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass
class DecodeAttentionWelmv45Workspace:
    _prepared: PreparedDecodeKernel
    _plan: _SplitPlan
    _decode_plan: DecodeAttentionWelmv45Plan | None
    workspace_size: int
    workspace: torch.Tensor
    cpu_workspace: torch.Tensor
    scratch_workspace_size: int
    scratch_workspace: torch.Tensor | None
    num_layers: int
    num_sub_tasks: int
    max_pages: int
    num_cache_pages: int
    sm_scale: float
    has_sinks: bool
    window_left: int
    fma_max_tokens: int
    worker_count: int
    num_stages: int
    output_merge_warps: int
    direct_fma_only: bool
    has_headwise_gate: bool
    device: torch.device


def decode_attention_welmv45_plan(
    token_counts: torch.Tensor | Sequence[int],
    *,
    device: torch.device | str | int | None = None,
    max_pages: int,
    num_cache_pages: int,
    has_sinks: bool = True,
    sm_scale: float | None = None,
    worker_count: int | None = None,
    num_stages: int = _DEFAULT_NUM_STAGES,
    output_merge_warps: int = _DEFAULT_OUTPUT_MERGE_WARPS,
    num_splits: int | None = None,
    split_kv_chunk_size: int | None = None,
    fma_max_tokens: int = _DEFAULT_FMA_MAX_TOKENS,
    window_size: tuple[int, int] | None = None,
    has_headwise_gate: bool = False,
) -> DecodeAttentionWelmv45Plan:
    batch = _planner_token_counts_len(token_counts)
    token_counts_cpu = _normalize_planner_token_counts(token_counts, batch=batch)
    if batch <= 0:
        raise MKConfigError("token_counts must be non-empty")
    plan_device = _normalize_plan_device(device)
    if not isinstance(has_sinks, bool):
        raise MKConfigError("has_sinks must be a bool")
    if not isinstance(has_headwise_gate, bool):
        raise MKConfigError("has_headwise_gate must be a bool")
    max_pages = int(max_pages)
    if max_pages <= 0:
        raise MKConfigError("max_pages must be positive")
    num_cache_pages = int(num_cache_pages)
    if num_cache_pages <= 0:
        raise MKConfigError("num_cache_pages must be positive")
    resolved_worker_count = _resolve_worker_count_from_device(plan_device, worker_count)
    _validate_tuning(
        worker_count=resolved_worker_count,
        num_stages=num_stages,
        output_merge_warps=output_merge_warps,
        num_splits=num_splits,
        split_kv_chunk_size=split_kv_chunk_size,
    )
    fma_max_tokens = _validate_fma_max_tokens(fma_max_tokens)
    window_left = _normalize_window_left(window_size)
    scale = 1.0 / math.sqrt(float(_HEAD_DIM)) if sm_scale is None else float(sm_scale)
    shape_plan = _make_split_plan(
        token_counts=token_counts_cpu,
        max_pages=max_pages,
        worker_count=resolved_worker_count,
        num_splits=num_splits,
        split_kv_chunk_size=split_kv_chunk_size,
        window_left=window_left,
        fma_max_tokens=fma_max_tokens,
        include_split_token_counts=False,
    )
    _validate_window_direct_plan(
        plan=shape_plan,
        window_left=window_left,
        fma_max_tokens=fma_max_tokens,
    )
    direct_fma_only = (
        window_left >= 0
        and shape_plan.max_splits == 1
        and shape_plan.max_split_tokens <= fma_max_tokens
    )
    _validate_decode_aot_profile(
        num_sub_tasks=batch,
        max_splits=shape_plan.max_splits,
        worker_count=resolved_worker_count,
        num_stages=num_stages,
        output_merge_warps=output_merge_warps,
        has_sinks=has_sinks,
        has_window=window_left >= 0,
        direct_fma_only=direct_fma_only,
        has_headwise_gate=has_headwise_gate,
    )
    prepared = prepare_decode_kernel(
        worker_count=resolved_worker_count,
        device_id=_device_index(plan_device),
        has_window=window_left >= 0,
    )
    native_plan = _create_native_plan(
        prepared,
        plan=shape_plan,
        max_pages=max_pages,
        num_cache_pages=num_cache_pages,
        scale=scale,
        window_left=window_left,
        fma_max_tokens=fma_max_tokens,
        worker_count=resolved_worker_count,
        batch_size=batch,
    )
    return DecodeAttentionWelmv45Plan(
        _prepared=prepared,
        _plan=native_plan,
        num_sub_tasks=batch,
        max_pages=max_pages,
        num_cache_pages=num_cache_pages,
        sm_scale=scale,
        has_sinks=has_sinks,
        window_left=window_left,
        fma_max_tokens=fma_max_tokens,
        worker_count=resolved_worker_count,
        num_stages=int(num_stages),
        output_merge_warps=int(output_merge_warps),
        direct_fma_only=direct_fma_only,
        has_headwise_gate=has_headwise_gate,
        device=plan_device,
    )


def decode_attention_welmv45_init_workspace(
    page_ids: torch.Tensor | DecodeAttentionWelmv45Plan,
    token_counts: torch.Tensor | Sequence[int] | None = None,
    *,
    num_cache_pages: int | None = None,
    num_layers: int = 1,
    has_sinks: bool = True,
    sm_scale: float | None = None,
    worker_count: int | None = None,
    num_stages: int = _DEFAULT_NUM_STAGES,
    output_merge_warps: int = _DEFAULT_OUTPUT_MERGE_WARPS,
    num_splits: int | None = None,
    split_kv_chunk_size: int | None = None,
    fma_max_tokens: int = _DEFAULT_FMA_MAX_TOKENS,
    window_size: tuple[int, int] | None = None,
    has_headwise_gate: bool = False,
    workspace: DecodeAttentionWelmv45Workspace | None = None,
    allow_resize: bool = True,
) -> DecodeAttentionWelmv45Workspace:
    if isinstance(page_ids, DecodeAttentionWelmv45Plan):
        if not isinstance(token_counts, torch.Tensor):
            raise MKConfigError(
                "page_ids tensor must be passed as the second argument when "
                "init_workspace is called with a DecodeAttentionWelmv45Plan"
            )
        return _init_workspace_from_decode_plan(
            page_ids,
            token_counts,
            num_layers=num_layers,
            workspace=workspace,
            allow_resize=allow_resize,
        )
    if token_counts is None:
        raise MKConfigError("token_counts must be provided")
    _validate_page_ids(page_ids)
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
        raise MKConfigError("num_layers must be a positive integer")
    if not isinstance(has_sinks, bool):
        raise MKConfigError("has_sinks must be a bool")
    if not isinstance(has_headwise_gate, bool):
        raise MKConfigError("has_headwise_gate must be a bool")
    if num_cache_pages is None:
        raise MKConfigError("num_cache_pages must be provided")
    num_cache_pages = int(num_cache_pages)
    if num_cache_pages <= 0:
        raise MKConfigError("num_cache_pages must be positive")

    resolved_worker_count = _resolve_worker_count_from_device(page_ids.device, worker_count)
    _validate_tuning(
        worker_count=resolved_worker_count,
        num_stages=num_stages,
        output_merge_warps=output_merge_warps,
        num_splits=num_splits,
        split_kv_chunk_size=split_kv_chunk_size,
    )
    fma_max_tokens = _validate_fma_max_tokens(fma_max_tokens)
    window_left = _normalize_window_left(window_size)
    max_pages = int(page_ids.shape[1])
    scale = 1.0 / math.sqrt(float(_HEAD_DIM)) if sm_scale is None else float(sm_scale)
    _validate_planner_token_counts_metadata(
        token_counts,
        batch=int(page_ids.shape[0]),
    )

    existing_workspace: DecodeAttentionWelmv45Workspace | None = None
    if isinstance(workspace, DecodeAttentionWelmv45Workspace):
        existing_workspace = workspace
    elif workspace is not None:
        raise MKConfigError("workspace must be a DecodeAttentionWelmv45Workspace")

    if _current_stream_is_capturing():
        if existing_workspace is None:
            raise MKConfigError(
                "CUDA graph capture workspace init requires an existing workspace handle"
            )
        direct_fma_only = (
            window_left >= 0
            and existing_workspace._plan.max_splits == 1
            and existing_workspace._plan.max_split_tokens <= fma_max_tokens
        )
        _validate_reusable_workspace(
            workspace=existing_workspace,
            plan=existing_workspace._plan,
            num_layers=num_layers,
            num_sub_tasks=int(page_ids.shape[0]),
            max_pages=max_pages,
            num_cache_pages=num_cache_pages,
            scale=scale,
            has_sinks=has_sinks,
            window_left=window_left,
            fma_max_tokens=fma_max_tokens,
            worker_count=resolved_worker_count,
            num_stages=num_stages,
            output_merge_warps=output_merge_warps,
            direct_fma_only=direct_fma_only,
            has_headwise_gate=has_headwise_gate,
            device=page_ids.device,
        )
        _init_workspace_layers(
            existing_workspace,
            _init_workspace_args(existing_workspace._plan, existing_workspace),
        )
        return existing_workspace

    token_counts_cpu = _normalize_planner_token_counts(
        token_counts,
        batch=int(page_ids.shape[0]),
    )
    plan = _make_split_plan(
        token_counts=token_counts_cpu,
        max_pages=max_pages,
        worker_count=resolved_worker_count,
        num_splits=num_splits,
        split_kv_chunk_size=split_kv_chunk_size,
        window_left=window_left,
        fma_max_tokens=fma_max_tokens,
        include_split_token_counts=False,
        split_plan_buffer=(
            existing_workspace._plan.split_plan_buffer
            if existing_workspace is not None
            else None
        ),
    )
    _validate_window_direct_plan(
        plan=plan,
        window_left=window_left,
        fma_max_tokens=fma_max_tokens,
    )

    direct_fma_only = (
        window_left >= 0
        and plan.max_splits == 1
        and plan.max_split_tokens <= fma_max_tokens
    )
    _validate_decode_aot_profile(
        num_sub_tasks=int(page_ids.shape[0]),
        max_splits=plan.max_splits,
        worker_count=resolved_worker_count,
        num_stages=num_stages,
        output_merge_warps=output_merge_warps,
        has_sinks=has_sinks,
        has_window=window_left >= 0,
        direct_fma_only=direct_fma_only,
        has_headwise_gate=has_headwise_gate,
    )
    base_args = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        plan.split_plan,
        int(page_ids.shape[0]),
        max_pages,
        num_cache_pages,
        plan.num_split_records,
        scale,
        window_left,
        fma_max_tokens,
        None,
        None,
    )
    if existing_workspace is not None:
        _validate_reusable_workspace(
            workspace=existing_workspace,
            plan=plan,
            num_layers=num_layers,
            num_sub_tasks=int(page_ids.shape[0]),
            max_pages=max_pages,
            num_cache_pages=num_cache_pages,
            scale=scale,
            has_sinks=has_sinks,
            window_left=window_left,
            fma_max_tokens=fma_max_tokens,
            worker_count=resolved_worker_count,
            num_stages=num_stages,
            output_merge_warps=output_merge_warps,
            direct_fma_only=direct_fma_only,
            has_headwise_gate=has_headwise_gate,
            device=page_ids.device,
        )
        native_plan = _create_native_plan(
            existing_workspace._prepared,
            plan=plan,
            max_pages=max_pages,
            num_cache_pages=num_cache_pages,
            scale=scale,
            window_left=window_left,
            fma_max_tokens=fma_max_tokens,
            worker_count=resolved_worker_count,
            batch_size=int(page_ids.shape[0]),
        )
        _resize_workspace_buffers_for_plan(
            existing_workspace,
            native_plan,
            allow_resize=allow_resize,
        )
        _close_workspace_owned_plan(existing_workspace, replacement=native_plan)
        existing_workspace._plan = native_plan
        existing_workspace._decode_plan = None
        _init_workspace_layers(
            existing_workspace,
            _init_workspace_args(native_plan, existing_workspace),
        )
        return existing_workspace

    prepared = prepare_decode_kernel(
        worker_count=resolved_worker_count,
        device_id=_device_index(page_ids.device),
        has_window=window_left >= 0,
    )
    plan = _create_native_plan(
        prepared,
        plan=plan,
        max_pages=max_pages,
        num_cache_pages=num_cache_pages,
        scale=scale,
        window_left=window_left,
        fma_max_tokens=fma_max_tokens,
        worker_count=resolved_worker_count,
        batch_size=int(page_ids.shape[0]),
    )
    workspace_size = _native_plan_workspace_size(plan.native_handle)
    scratch_workspace_size = _scratch_workspace_size(prepared, plan)
    scratch_workspace = (
        torch.empty(
            (scratch_workspace_size,),
            dtype=torch.uint8,
            device=page_ids.device,
        )
        if scratch_workspace_size > 0
        else None
    )
    total_workspace_size = workspace_size * int(num_layers)
    workspace_buffer = torch.empty(
        (total_workspace_size,),
        dtype=torch.uint8,
        device=page_ids.device,
    )
    cpu_workspace = torch.empty(
        (workspace_size,),
        dtype=torch.uint8,
        device="cpu",
        pin_memory=True,
    )
    result = DecodeAttentionWelmv45Workspace(
        _prepared=prepared,
        _plan=plan,
        _decode_plan=None,
        workspace_size=workspace_size,
        workspace=workspace_buffer,
        cpu_workspace=cpu_workspace,
        scratch_workspace_size=scratch_workspace_size,
        scratch_workspace=scratch_workspace,
        num_layers=int(num_layers),
        num_sub_tasks=int(page_ids.shape[0]),
        max_pages=max_pages,
        num_cache_pages=num_cache_pages,
        sm_scale=scale,
        has_sinks=has_sinks,
        window_left=window_left,
        fma_max_tokens=fma_max_tokens,
        worker_count=resolved_worker_count,
        num_stages=num_stages,
        output_merge_warps=output_merge_warps,
        direct_fma_only=direct_fma_only,
        has_headwise_gate=has_headwise_gate,
        device=page_ids.device,
    )
    _init_workspace_layers(result, _init_workspace_args(plan, result))
    return result


def _init_workspace_from_decode_plan(
    plan: DecodeAttentionWelmv45Plan,
    page_ids: torch.Tensor,
    *,
    num_layers: int,
    workspace: DecodeAttentionWelmv45Workspace | None,
    allow_resize: bool = True,
) -> DecodeAttentionWelmv45Workspace:
    _validate_page_ids(page_ids)
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
        raise MKConfigError("num_layers must be a positive integer")
    if int(page_ids.shape[0]) != plan.num_sub_tasks:
        raise MKConfigError("page_ids batch does not match plan")
    if int(page_ids.shape[1]) != plan.max_pages:
        raise MKConfigError("page_ids max_pages does not match plan")
    if _device_index(page_ids.device) != _device_index(plan.device):
        raise MKConfigError("page_ids device does not match plan")

    existing_workspace: DecodeAttentionWelmv45Workspace | None = None
    if isinstance(workspace, DecodeAttentionWelmv45Workspace):
        existing_workspace = workspace
    elif workspace is not None:
        raise MKConfigError("workspace must be a DecodeAttentionWelmv45Workspace")

    if existing_workspace is not None:
        _validate_reusable_workspace(
            workspace=existing_workspace,
            plan=plan._plan,
            num_layers=num_layers,
            num_sub_tasks=plan.num_sub_tasks,
            max_pages=plan.max_pages,
            num_cache_pages=plan.num_cache_pages,
            scale=plan.sm_scale,
            has_sinks=plan.has_sinks,
            window_left=plan.window_left,
            fma_max_tokens=plan.fma_max_tokens,
            worker_count=plan.worker_count,
            num_stages=plan.num_stages,
            output_merge_warps=plan.output_merge_warps,
            direct_fma_only=plan.direct_fma_only,
            has_headwise_gate=plan.has_headwise_gate,
            device=plan.device,
        )
        _resize_workspace_buffers_for_plan(
            existing_workspace,
            plan._plan,
            allow_resize=allow_resize,
        )
        _close_workspace_owned_plan(existing_workspace, replacement=plan._plan)
        existing_workspace._prepared = plan._prepared
        existing_workspace._plan = plan._plan
        existing_workspace._decode_plan = plan
        _init_workspace_layers(
            existing_workspace,
            _init_workspace_args(plan._plan, existing_workspace),
        )
        return existing_workspace

    workspace_size = plan.workspace_size
    if workspace_size <= 0:
        raise MKConfigError("plan workspace size must be positive")
    scratch_workspace_size = plan.scratch_workspace_size
    scratch_workspace = (
        torch.empty(
            (scratch_workspace_size,),
            dtype=torch.uint8,
            device=plan.device,
        )
        if scratch_workspace_size > 0
        else None
    )
    result = DecodeAttentionWelmv45Workspace(
        _prepared=plan._prepared,
        _plan=plan._plan,
        _decode_plan=plan,
        workspace_size=workspace_size,
        workspace=torch.empty(
            (workspace_size * int(num_layers),),
            dtype=torch.uint8,
            device=plan.device,
        ),
        cpu_workspace=torch.empty(
            (workspace_size,),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        ),
        scratch_workspace_size=scratch_workspace_size,
        scratch_workspace=scratch_workspace,
        num_layers=int(num_layers),
        num_sub_tasks=plan.num_sub_tasks,
        max_pages=plan.max_pages,
        num_cache_pages=plan.num_cache_pages,
        sm_scale=plan.sm_scale,
        has_sinks=plan.has_sinks,
        window_left=plan.window_left,
        fma_max_tokens=plan.fma_max_tokens,
        worker_count=plan.worker_count,
        num_stages=plan.num_stages,
        output_merge_warps=plan.output_merge_warps,
        direct_fma_only=plan.direct_fma_only,
        has_headwise_gate=plan.has_headwise_gate,
        device=plan.device,
    )
    _init_workspace_layers(result, _init_workspace_args(plan._plan, result))
    return result


def decode_attention_welmv45_run(
    workspace: DecodeAttentionWelmv45Workspace,
    *,
    layer_id: int,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_ids: torch.Tensor,
    sinks: torch.Tensor | None = None,
    gate_values: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(workspace, DecodeAttentionWelmv45Workspace):
        raise MKConfigError("workspace must be a DecodeAttentionWelmv45Workspace")
    if not isinstance(layer_id, int) or isinstance(layer_id, bool):
        raise MKConfigError("layer_id must be an integer")
    if layer_id < 0 or layer_id >= workspace.num_layers:
        raise MKConfigError("layer_id out of range")
    if (sinks is not None) != workspace.has_sinks:
        raise MKConfigError("sinks presence must match workspace has_sinks")
    output, lse = _validate_runtime_tensors(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        page_ids=page_ids,
        sinks=sinks,
        output=output,
        lse=lse,
    )
    gate_values = _validate_gate_values(
        query=query,
        gate_values=gate_values,
        has_headwise_gate=workspace.has_headwise_gate,
    )
    if int(query.shape[0]) != workspace.num_sub_tasks:
        raise MKConfigError("query batch does not match workspace")
    if int(page_ids.shape[1]) != workspace.max_pages:
        raise MKConfigError("page_ids max_pages does not match workspace")
    if int(key_cache.shape[0]) != workspace.num_cache_pages:
        raise MKConfigError("key_cache num pages does not match workspace")
    runtime_args = (
        query,
        key_cache,
        value_cache,
        page_ids,
        output,
        lse,
        sinks,
        workspace._plan.split_plan,
        workspace.num_sub_tasks,
        workspace.max_pages,
        workspace.num_cache_pages,
        workspace._plan.num_split_records,
        workspace.sm_scale,
        workspace.window_left,
        workspace.fma_max_tokens,
        workspace.scratch_workspace,
        gate_values,
    )
    workspace._prepared._library.launch_prepared(
        _workspace_args(workspace, layer_id, runtime_args=runtime_args),
        device_id=workspace._prepared._device_id,
        stream=_current_stream_ptr(workspace.device),
        context=workspace._prepared._context,
    )
    return output, lse





def _resolve_worker_count(query: torch.Tensor, worker_count: int | None) -> int:
    if worker_count is not None:
        return worker_count
    device_index = query.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _resolve_worker_count_from_device(
    device: torch.device,
    worker_count: int | None,
) -> int:
    if worker_count is not None:
        return worker_count
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _current_stream_is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return False


def _current_stream_ptr(device: torch.device) -> int:
    with torch.cuda.device(device):
        return int(torch.cuda.current_stream(device=device).cuda_stream)




def _validate_decode_aot_profile(
    *,
    num_sub_tasks: int,
    max_splits: int,
    worker_count: int,
    num_stages: int,
    output_merge_warps: int,
    has_sinks: bool,
    has_window: bool,
    direct_fma_only: bool,
    has_headwise_gate: bool,
) -> None:
    if int(num_sub_tasks) <= 0 or int(num_sub_tasks) > DECODE_DEFAULT_NUM_SUB_TASKS:
        raise MKConfigError(
            f"batch/num_sub_tasks={num_sub_tasks} exceeds decode AOT capacity "
            f"{DECODE_DEFAULT_NUM_SUB_TASKS}"
        )
    if int(max_splits) <= 0 or int(max_splits) > DECODE_DEFAULT_MAX_SPLITS:
        raise MKConfigError(
            f"max_splits={max_splits} exceeds decode AOT capacity "
            f"{DECODE_DEFAULT_MAX_SPLITS}"
        )
    if int(num_stages) != DECODE_DEFAULT_NUM_STAGES:
        raise MKConfigError(
            f"decode AOT profiles use num_stages="
            f"{DECODE_DEFAULT_NUM_STAGES}, got {num_stages}"
        )
    if int(output_merge_warps) != DECODE_DEFAULT_OUTPUT_MERGE_WARPS:
        raise MKConfigError(
            f"decode AOT profiles use output_merge_warps="
            f"{DECODE_DEFAULT_OUTPUT_MERGE_WARPS}, got {output_merge_warps}"
        )
    if not has_sinks:
        raise MKConfigError("decode AOT profiles are compiled with has_sinks=True")
    if direct_fma_only or has_headwise_gate:
        raise MKConfigError(
            "decode AOT profiles do not include direct-fma-only / "
            "headwise-gate specializations yet; build a dedicated executor"
        )
    del worker_count, has_window


def _scratch_workspace_size(prepared: PreparedDecodeKernel, plan: _SplitPlan | None = None) -> int:
    if plan is not None and plan.native_handle is not None:
        return _native_plan_scratch_workspace_size(plan.native_handle)
    return int(prepared._library.scratch_workspace_size())


def _create_native_plan(
    prepared: PreparedDecodeKernel,
    *,
    plan: _SplitPlan,
    max_pages: int,
    num_cache_pages: int,
    scale: float,
    window_left: int,
    fma_max_tokens: int,
    worker_count: int,
    batch_size: int | None = None,
) -> _SplitPlan:
    if plan.split_plan is None:
        raise MKConfigError("split_plan must be provided by the Python planner")
    split_plan = plan.split_plan
    if split_plan.device.type != "cpu" or split_plan.dtype is not torch.int32:
        raise MKConfigError("split_plan must be a CPU int32 planner output")
    if not split_plan.is_contiguous():
        split_plan = split_plan.contiguous()
    library = prepared._library
    if batch_size is None:
        raise MKConfigError("batch_size is required when creating a native decode plan")
    handle_ptr = library.plan_create(
        split_plan,
        int(plan.num_split_records),
        int(plan.max_split_tokens),
        int(plan.max_splits),
        int(plan.chunk_pages),
        int(batch_size),
        int(max_pages),
        int(num_cache_pages),
        float(scale),
        int(window_left),
        int(fma_max_tokens),
    )
    handle = _NativePlanHandle(ptr=int(handle_ptr), library=library)
    return _SplitPlan(
        split_plan=split_plan,
        split_token_counts=plan.split_token_counts,
        max_split_tokens=int(plan.max_split_tokens),
        max_splits=int(plan.max_splits),
        num_split_records=int(plan.num_split_records),
        chunk_pages=int(plan.chunk_pages),
        worker_count=worker_count,
        split_plan_buffer=plan.split_plan_buffer if split_plan is plan.split_plan else split_plan,
        native_handle=handle,
    )


def _close_native_plan(plan: _SplitPlan) -> None:
    if plan.native_handle is not None:
        plan.native_handle.close()


def _close_workspace_owned_plan(
    workspace: DecodeAttentionWelmv45Workspace,
    *,
    replacement: _SplitPlan | None = None,
) -> None:
    if workspace._decode_plan is None and workspace._plan is not replacement:
        _close_native_plan(workspace._plan)


def _native_plan_scratch_workspace_size(handle: _NativePlanHandle) -> int:
    return int(handle.library.plan_scratch_workspace_size(int(handle.ptr)))


def _native_plan_workspace_size(handle: _NativePlanHandle) -> int:
    return int(handle.library.plan_workspace_size(int(handle.ptr)))


def _init_workspace_layers_native(
    workspace: DecodeAttentionWelmv45Workspace,
    *,
    stream: int,
    copy_to_device: bool,
) -> None:
    del stream  # stream is taken from the torch current stream inside the .so
    handle = workspace._plan.native_handle
    if handle is None:
        raise MKConfigError("workspace does not have a native plan")
    library = workspace._prepared._library
    library.plan_init_workspace(
        int(handle.ptr),
        workspace.scratch_workspace,
        workspace.workspace,
        workspace.cpu_workspace,
        int(workspace.workspace_size),
        int(workspace.num_layers),
        1 if copy_to_device else 0,
    )


def _init_workspace_layers_many(
    workspace: DecodeAttentionWelmv45Workspace,
    *,
    runtime_args: tuple[object, ...],
    stream: int,
    copy_to_device: bool,
) -> None:
    del stream
    library = workspace._prepared._library
    # runtime_args layout matches decode launch ABI without workspaces:
    # 0..6 tensors, 7 split_plan, 8 batch, 9 max_pages, 10 num_cache_pages,
    # 11 num_split_records, 12 sm_scale, 13 window_left, 14 fma_max_tokens,
    # 15 scratch, 16 gate
    library.init_workspace_same_plan(
        runtime_args[7],
        int(runtime_args[11]),
        int(runtime_args[8]),
        int(runtime_args[9]),
        int(runtime_args[10]),
        float(runtime_args[12]),
        int(runtime_args[13]),
        int(runtime_args[14]),
        workspace.scratch_workspace,
        workspace.workspace,
        workspace.cpu_workspace,
        int(workspace.workspace_size),
        int(workspace.num_layers),
        1 if copy_to_device else 0,
    )


def _init_workspace_layer_fast(
    workspace: DecodeAttentionWelmv45Workspace,
    layer_id: int,
    *,
    runtime_args: tuple[object, ...],
    stream: int,
) -> None:
    del stream
    begin = int(layer_id) * int(workspace.workspace_size)
    end = begin + int(workspace.workspace_size)
    library = workspace._prepared._library
    library.init_workspace_same_plan(
        runtime_args[7],
        int(runtime_args[11]),
        int(runtime_args[8]),
        int(runtime_args[9]),
        int(runtime_args[10]),
        float(runtime_args[12]),
        int(runtime_args[13]),
        int(runtime_args[14]),
        workspace.scratch_workspace,
        workspace.workspace[begin:end],
        workspace.cpu_workspace[: int(workspace.workspace_size)],
        int(workspace.workspace_size),
        1,
        1,
    )


def _init_workspace_layers(
    workspace: DecodeAttentionWelmv45Workspace,
    runtime_args: tuple[object, ...],
    *,
    copy_to_device: bool = True,
) -> None:
    if workspace._plan.native_handle is not None:
        _init_workspace_layers_native(
            workspace,
            stream=_current_stream_ptr(workspace.device),
            copy_to_device=copy_to_device,
        )
        return
    stream = _current_stream_ptr(workspace.device)
    if workspace.num_layers > 1 and not _current_stream_is_capturing():
        _init_workspace_layers_many(
            workspace,
            runtime_args=runtime_args,
            stream=stream,
            copy_to_device=copy_to_device,
        )
        return
    if not copy_to_device:
        raise MKConfigError("CPU-only workspace reset requires a multi-layer workspace")
    for layer_id in range(workspace.num_layers):
        _init_workspace_layer_fast(
            workspace,
            layer_id,
            runtime_args=runtime_args,
            stream=stream,
        )


def _init_workspace_args(
    plan: _SplitPlan,
    workspace: DecodeAttentionWelmv45Workspace,
) -> tuple[object, ...]:
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        plan.split_plan,
        workspace.num_sub_tasks,
        workspace.max_pages,
        workspace.num_cache_pages,
        plan.num_split_records,
        workspace.sm_scale,
        workspace.window_left,
        workspace.fma_max_tokens,
        workspace.scratch_workspace,
        None,
    )


def _workspace_args(
    workspace: DecodeAttentionWelmv45Workspace,
    layer_id: int,
    *,
    runtime_args: tuple[object, ...],
) -> tuple[object, ...]:
    begin = int(layer_id) * int(workspace.workspace_size)
    end = begin + int(workspace.workspace_size)
    return (
        *runtime_args,
        workspace.workspace[begin:end],
        workspace.cpu_workspace[: int(workspace.workspace_size)],
    )


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise MKConfigError("device must be CUDA")
    return int(torch.cuda.current_device() if device.index is None else device.index)


def _normalize_plan_device(device: torch.device | str | int | None) -> torch.device:
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    if isinstance(device, int):
        return torch.device("cuda", int(device))
    plan_device = torch.device(device)
    if plan_device.type != "cuda":
        raise MKConfigError("plan device must be CUDA")
    if plan_device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return plan_device


def _validate_reusable_workspace(
    *,
    workspace: DecodeAttentionWelmv45Workspace,
    plan: _SplitPlan,
    num_layers: int,
    num_sub_tasks: int,
    max_pages: int,
    num_cache_pages: int,
    scale: float,
    has_sinks: bool,
    window_left: int,
    fma_max_tokens: int,
    worker_count: int,
    num_stages: int,
    output_merge_warps: int,
    direct_fma_only: bool,
    has_headwise_gate: bool,
    device: torch.device,
) -> None:
    if not isinstance(workspace, DecodeAttentionWelmv45Workspace):
        raise MKConfigError("workspace must be a DecodeAttentionWelmv45Workspace")
    if _device_index(workspace.device) != _device_index(device):
        raise MKConfigError("workspace device does not match page_ids")
    if int(workspace.workspace.numel()) < int(workspace.workspace_size) * int(
        workspace.num_layers
    ):
        raise MKConfigError("workspace buffer is too small")
    if int(workspace.cpu_workspace.numel()) < int(workspace.workspace_size):
        raise MKConfigError("workspace CPU template buffer is too small")
    if workspace.num_layers != int(num_layers):
        raise MKConfigError("workspace num_layers does not match")
    if workspace.num_sub_tasks != int(num_sub_tasks):
        raise MKConfigError("workspace batch does not match")
    if workspace.max_pages != int(max_pages):
        raise MKConfigError("workspace max_pages does not match")
    if workspace.num_cache_pages != int(num_cache_pages):
        raise MKConfigError("workspace num_cache_pages does not match")
    if workspace.has_sinks != bool(has_sinks):
        raise MKConfigError("workspace has_sinks does not match")
    if workspace.window_left != int(window_left):
        raise MKConfigError("workspace window_left does not match")
    if workspace.fma_max_tokens != int(fma_max_tokens):
        raise MKConfigError("workspace fma_max_tokens does not match")
    if workspace.worker_count != int(worker_count):
        raise MKConfigError("workspace worker_count does not match")
    if workspace.num_stages != int(num_stages):
        raise MKConfigError("workspace num_stages does not match")
    if workspace.output_merge_warps != int(output_merge_warps):
        raise MKConfigError("workspace output_merge_warps does not match")
    if workspace.direct_fma_only != bool(direct_fma_only):
        raise MKConfigError("workspace direct_fma_only does not match")
    if workspace.has_headwise_gate != bool(has_headwise_gate):
        raise MKConfigError("workspace has_headwise_gate does not match")
    if workspace._plan.max_splits != plan.max_splits:
        raise MKConfigError("workspace max_splits does not match")
    if workspace._plan.worker_count != plan.worker_count:
        raise MKConfigError("workspace worker_count does not match")
    if not math.isclose(workspace.sm_scale, float(scale), rel_tol=0.0, abs_tol=0.0):
        raise MKConfigError("workspace sm_scale does not match")


def _resize_workspace_buffers_for_plan(
    workspace: DecodeAttentionWelmv45Workspace,
    plan: _SplitPlan,
    *,
    allow_resize: bool = True,
) -> None:
    if plan.native_handle is None:
        return
    required_workspace_size = _native_plan_workspace_size(plan.native_handle)
    if required_workspace_size <= 0:
        raise MKConfigError("native plan workspace size must be positive")
    required_gpu = int(required_workspace_size) * int(workspace.num_layers)
    if not allow_resize and (
        int(required_workspace_size) > int(workspace.workspace_size)
        or int(workspace.workspace.numel()) < required_gpu
        or int(workspace.cpu_workspace.numel()) < int(required_workspace_size)
    ):
        raise MKConfigError(
            "decode_attention_welmv45 workspace capacity exceeded and "
            "allow_resize is false"
        )
    if int(required_workspace_size) > int(workspace.workspace_size):
        workspace.workspace_size = int(required_workspace_size)
    required_gpu = int(workspace.workspace_size) * int(workspace.num_layers)
    if int(workspace.workspace.numel()) < required_gpu:
        workspace.workspace = torch.empty(
            (required_gpu,),
            dtype=torch.uint8,
            device=workspace.device,
        )
    if int(workspace.cpu_workspace.numel()) < int(workspace.workspace_size):
        workspace.cpu_workspace = torch.empty(
            (int(workspace.workspace_size),),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )


def _validate_page_ids(page_ids: torch.Tensor) -> None:
    if not isinstance(page_ids, torch.Tensor):
        raise MKConfigError("page_ids must be a torch.Tensor")
    if page_ids.device.type != "cuda":
        raise MKConfigError("page_ids must be a CUDA tensor")
    if page_ids.dtype is not torch.int32:
        raise MKConfigError("page_ids must be torch.int32")
    if page_ids.ndim != 2:
        raise MKConfigError("page_ids shape must be (batch, max_pages)")
    if not page_ids.is_contiguous():
        raise MKConfigError("page_ids must be contiguous")
    if int(page_ids.shape[0]) <= 0 or int(page_ids.shape[1]) <= 0:
        raise MKConfigError("page_ids must be non-empty")


def _validate_planner_token_counts_metadata(
    token_counts: torch.Tensor | Sequence[int],
    *,
    batch: int,
) -> None:
    if isinstance(token_counts, torch.Tensor):
        if token_counts.device.type != "cpu":
            raise MKConfigError("token_counts must be a CPU int32 planner input")
        if token_counts.dtype is not torch.int32:
            raise MKConfigError("token_counts must be torch.int32")
        if tuple(token_counts.shape) != (int(batch),):
            raise MKConfigError("token_counts shape must be (batch,)")
        return
    if isinstance(token_counts, Sequence):
        if len(token_counts) != int(batch):
            raise MKConfigError("token_counts shape must be (batch,)")
        return
    raise MKConfigError("token_counts must be a CPU int32 planner input")


def _planner_token_counts_len(token_counts: torch.Tensor | Sequence[int]) -> int:
    if isinstance(token_counts, torch.Tensor):
        if token_counts.ndim != 1:
            raise MKConfigError("token_counts shape must be (batch,)")
        return int(token_counts.numel())
    if isinstance(token_counts, Sequence):
        return int(len(token_counts))
    raise MKConfigError("token_counts must be a CPU int32 planner input")


def _normalize_planner_token_counts(
    token_counts: torch.Tensor | Sequence[int],
    *,
    batch: int,
) -> torch.Tensor:
    if not isinstance(token_counts, torch.Tensor):
        if isinstance(token_counts, Sequence):
            token_counts = torch.tensor(tuple(token_counts), dtype=torch.int32)
        else:
            raise MKConfigError(
                "token_counts must be a CPU int32 planner input"
            )
    if token_counts.device.type != "cpu":
        raise MKConfigError("token_counts must be a CPU int32 planner input")
    if token_counts.dtype is not torch.int32:
        raise MKConfigError("token_counts must be torch.int32")
    if tuple(token_counts.shape) != (int(batch),):
        raise MKConfigError("token_counts shape must be (batch,)")
    if not token_counts.is_contiguous():
        token_counts = token_counts.contiguous()
    return token_counts.detach()



def _should_auto_split(token_counts: torch.Tensor) -> bool:
    if int(token_counts.numel()) < 64:
        return False
    if token_counts.device.type != "cpu" or token_counts.dtype is not torch.int32:
        raise MKConfigError("token_counts must be a CPU int32 planner input")
    token_counts_cpu = token_counts.detach()
    max_tokens = int(torch.clamp(token_counts_cpu, min=0).max().item())
    return max_tokens >= 8192




def _validate_runtime_tensors(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_ids: torch.Tensor,
    sinks: torch.Tensor | None,
    output: torch.Tensor | None,
    lse: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_cuda_bf16_contiguous(query, name="query")
    _validate_cuda_bf16_contiguous(key_cache, name="key_cache")
    _validate_cuda_bf16_contiguous(value_cache, name="value_cache")
    if query.ndim != 3 or tuple(query.shape[1:]) != (_LOCAL_Q_HEADS, _HEAD_DIM):
        raise MKConfigError(
            f"query shape must be (batch, {_LOCAL_Q_HEADS}, {_HEAD_DIM})"
        )
    if key_cache.ndim != 4 or tuple(key_cache.shape[1:]) != (
        _PAGE_SIZE,
        _LOCAL_KV_HEADS,
        _HEAD_DIM,
    ):
        raise MKConfigError(
            "key_cache shape must be "
            f"(num_pages, {_PAGE_SIZE}, {_LOCAL_KV_HEADS}, {_HEAD_DIM})"
        )
    if tuple(value_cache.shape) != tuple(key_cache.shape):
        raise MKConfigError("value_cache shape must match key_cache shape")
    if query.device != key_cache.device or query.device != value_cache.device:
        raise MKConfigError("query, key_cache, and value_cache must share a device")

    if not isinstance(page_ids, torch.Tensor):
        raise MKConfigError("page_ids must be a torch.Tensor")
    if page_ids.device.type != "cuda" or page_ids.device != query.device:
        raise MKConfigError("page_ids must be a CUDA tensor on the query device")
    if page_ids.dtype is not torch.int32:
        raise MKConfigError("page_ids must be torch.int32")
    if page_ids.ndim != 2 or int(page_ids.shape[0]) != int(query.shape[0]):
        raise MKConfigError("page_ids shape must be (batch, max_pages)")
    if not page_ids.is_contiguous():
        raise MKConfigError("page_ids must be contiguous")
    if int(page_ids.shape[1]) <= 0:
        raise MKConfigError("max_pages must be positive")

    if sinks is not None:
        _validate_cuda_bf16_contiguous(sinks, name="sinks")
        if sinks.device != query.device:
            raise MKConfigError("sinks must be on the query device")
        if tuple(sinks.shape) != (_LOCAL_Q_HEADS,):
            raise MKConfigError(f"sinks shape must be ({_LOCAL_Q_HEADS},)")

    expected_output_shape = tuple(query.shape)
    if output is None:
        output = torch.empty_like(query)
    else:
        _validate_cuda_bf16_contiguous(output, name="output")
        if output.device != query.device:
            raise MKConfigError("output must be on the query device")
        if tuple(output.shape) != expected_output_shape:
            raise MKConfigError(f"output shape must be {expected_output_shape}")

    expected_lse_shape = (int(query.shape[0]), _LOCAL_Q_HEADS)
    if lse is None:
        lse = torch.empty(expected_lse_shape, device=query.device, dtype=torch.float32)
    else:
        if not isinstance(lse, torch.Tensor):
            raise MKConfigError("lse must be a torch.Tensor")
        if lse.device.type != "cuda" or lse.device != query.device:
            raise MKConfigError("lse must be a CUDA tensor on the query device")
        if lse.dtype is not torch.float32:
            raise MKConfigError("lse must be torch.float32")
        if tuple(lse.shape) != expected_lse_shape:
            raise MKConfigError(f"lse shape must be {expected_lse_shape}")
        if not lse.is_contiguous():
            raise MKConfigError("lse must be contiguous")
    return output, lse


def _validate_gate_values(
    *,
    query: torch.Tensor,
    gate_values: torch.Tensor | None,
    has_headwise_gate: bool,
) -> torch.Tensor | None:
    if not has_headwise_gate:
        if gate_values is not None:
            raise MKConfigError(
                "gate_values requires workspace has_headwise_gate=True"
            )
        return None
    if gate_values is None:
        raise MKConfigError("gate_values must be provided when has_headwise_gate=True")
    _validate_cuda_bf16_contiguous(gate_values, name="gate_values")
    if gate_values.device != query.device:
        raise MKConfigError("gate_values must be on the query device")
    expected_shape = (int(query.shape[0]), _LOCAL_Q_HEADS)
    if tuple(gate_values.shape) != expected_shape:
        raise MKConfigError(f"gate_values shape must be {expected_shape}")
    return gate_values


def _validate_cuda_bf16_contiguous(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise MKConfigError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise MKConfigError(f"{name} must be a CUDA tensor")
    if tensor.dtype is not torch.bfloat16:
        raise MKConfigError(f"{name} must be torch.bfloat16")
    if not tensor.is_contiguous():
        raise MKConfigError(f"{name} must be contiguous")


def _validate_tuning(
    *,
    worker_count: int,
    num_stages: int,
    output_merge_warps: int,
    num_splits: int | None,
    split_kv_chunk_size: int | None,
) -> None:
    if not isinstance(worker_count, int) or isinstance(worker_count, bool):
        raise MKConfigError("worker_count must be an integer")
    if worker_count <= 0:
        raise MKConfigError("worker_count must be positive")
    if num_stages not in _SUPPORTED_NUM_STAGES:
        choices = ", ".join(str(value) for value in _SUPPORTED_NUM_STAGES)
        raise MKConfigError(f"num_stages must be one of: {choices}")
    if output_merge_warps not in _SUPPORTED_OUTPUT_MERGE_WARPS:
        choices = ", ".join(str(value) for value in _SUPPORTED_OUTPUT_MERGE_WARPS)
        raise MKConfigError(f"output_merge_warps must be one of: {choices}")
    if num_splits is not None:
        if not isinstance(num_splits, int) or isinstance(num_splits, bool):
            raise MKConfigError("num_splits must be an integer")
        if num_splits <= 0:
            raise MKConfigError("num_splits must be positive")
    if split_kv_chunk_size is not None:
        if not isinstance(split_kv_chunk_size, int) or isinstance(
            split_kv_chunk_size, bool
        ):
            raise MKConfigError("split_kv_chunk_size must be an integer")
        if split_kv_chunk_size <= 0:
            raise MKConfigError("split_kv_chunk_size must be positive")


def _validate_fma_max_tokens(fma_max_tokens: int) -> int:
    if not isinstance(fma_max_tokens, int) or isinstance(fma_max_tokens, bool):
        raise MKConfigError("fma_max_tokens must be an integer")
    if fma_max_tokens < 0:
        raise MKConfigError("fma_max_tokens must be non-negative")
    return int(fma_max_tokens)


def _normalize_window_left(window_size: tuple[int, int] | None) -> int:
    if window_size is None:
        return -1
    if not isinstance(window_size, tuple) or len(window_size) != 2:
        raise MKConfigError("window_size must be a tuple(left, right)")
    left, right = window_size
    if not isinstance(left, int) or isinstance(left, bool):
        raise MKConfigError("window_size left must be an integer")
    if not isinstance(right, int) or isinstance(right, bool):
        raise MKConfigError("window_size right must be an integer")
    if left < 0 and right < 0:
        return -1
    if left < 0:
        raise MKConfigError("SWA window_size left must be non-negative")
    if right != 0:
        raise MKConfigError("decode_attention_welmv45 only supports right window 0")
    return int(left)


def _request_window_range(
    token_count: int,
    *,
    max_tokens: int,
    window_left: int,
) -> tuple[int, int]:
    end = min(max(int(token_count), 0), max_tokens)
    if window_left < 0:
        return 0, end
    begin = max(end - int(window_left) - 1, 0)
    return begin, end


def _validate_window_direct_plan(
    *,
    plan: _SplitPlan,
    window_left: int,
    fma_max_tokens: int,
) -> None:
    if (
        window_left >= 0
        and plan.max_splits == 1
        and plan.max_split_tokens > fma_max_tokens
    ):
        raise MKConfigError(
            "SWA direct WMMA is not supported; increase fma_max_tokens or use "
            "num_splits > 1"
        )


def _make_split_plan(
    *,
    token_counts: torch.Tensor,
    max_pages: int,
    worker_count: int,
    num_splits: int | None,
    split_kv_chunk_size: int | None,
    window_left: int = -1,
    fma_max_tokens: int | None = None,
    include_split_token_counts: bool = True,
    split_plan_buffer: torch.Tensor | None = None,
) -> _SplitPlan:
    if worker_count <= 0:
        raise MKConfigError("worker_count must be positive")
    if token_counts.device.type != "cpu" or token_counts.dtype is not torch.int32:
        raise MKConfigError("token_counts must be a CPU int32 planner input")
    token_counts_cpu = token_counts.detach()
    batch = int(token_counts_cpu.numel())
    max_tokens = max_pages * _PAGE_SIZE
    if window_left < 0:
        full_tokens = np.clip(token_counts_cpu.numpy(), 0, max_tokens).astype(
            np.int32,
            copy=False,
        )
        effective_token_counts = full_tokens
        max_active_pages = max_pages
    else:
        token_values = [int(raw) for raw in token_counts_cpu.tolist()]
        ranges = [
            _request_window_range(raw, max_tokens=max_tokens, window_left=window_left)
            for raw in token_values
        ]
        effective_token_counts = [end - begin for begin, end in ranges]
        max_active_pages = (
            max(
                math.ceil(end / _PAGE_SIZE) - begin // _PAGE_SIZE
                if end > begin
                else 0
                for begin, end in ranges
            )
            if ranges
            else 0
        )
    actual_max_tokens = (
        int(effective_token_counts.max())
        if isinstance(effective_token_counts, np.ndarray) and effective_token_counts.size
        else max(effective_token_counts) if effective_token_counts else 0
    )
    total_tokens = (
        int(effective_token_counts.sum())
        if isinstance(effective_token_counts, np.ndarray)
        else sum(effective_token_counts)
    )
    split_page_cap = max_pages if window_left < 0 else max_active_pages
    force_window_split = (
        window_left >= 0
        and fma_max_tokens is not None
        and actual_max_tokens > int(fma_max_tokens)
    )

    if split_page_cap <= 0:
        split_page_cap = 1

    if num_splits is not None:
        max_splits = int(num_splits)
        chunk_pages = max(1, math.ceil(split_page_cap / max_splits))
    else:
        max_splits = _auto_max_splits(
            batch=batch,
            actual_max_tokens=actual_max_tokens,
            total_tokens=total_tokens,
            max_pages=split_page_cap,
            worker_count=worker_count,
        )
        if split_kv_chunk_size is not None and max_splits > 1:
            chunk_pages_from_arg = max(
                1, math.ceil(int(split_kv_chunk_size) / _PAGE_SIZE)
            )
            max_splits = max(
                max_splits,
                math.ceil(split_page_cap / chunk_pages_from_arg),
            )
        max_splits = min(max_splits, split_page_cap)
        if force_window_split and actual_max_tokens > 0:
            max_splits = max(max_splits, 2)
        chunk_pages = max(1, math.ceil(split_page_cap / max_splits))

    if window_left < 0:
        return _make_full_split_plan_from_array(
            tokens=full_tokens,
            max_pages=max_pages,
            worker_count=worker_count,
            max_splits=max_splits,
            chunk_pages=chunk_pages,
            include_split_token_counts=include_split_token_counts,
            split_plan_buffer=split_plan_buffer,
        )

    return _make_window_split_plan_from_ranges(
        ranges=ranges,
        max_splits=max_splits,
        chunk_pages=chunk_pages,
        worker_count=worker_count,
        include_split_token_counts=include_split_token_counts,
    )


def _make_full_split_plan_from_array(
    *,
    tokens: np.ndarray,
    max_pages: int,
    worker_count: int,
    max_splits: int,
    chunk_pages: int,
    include_split_token_counts: bool,
    split_plan_buffer: torch.Tensor | None,
) -> _SplitPlan:
    batch = int(tokens.size)
    active_pages = (tokens + _PAGE_SIZE - 1) // _PAGE_SIZE
    active_splits_array = np.minimum(
        (active_pages + chunk_pages - 1) // chunk_pages,
        max_splits,
    ).astype(np.int32, copy=False)
    active_splits = active_splits_array.tolist()

    split_grid = np.arange(max_splits, dtype=np.int32)
    active_mask = split_grid[None, :] < active_splits_array[:, None]
    request_ids, split_ids = np.nonzero(active_mask)
    request_ids = request_ids.astype(np.int32, copy=False)
    split_ids = split_ids.astype(np.int32, copy=False)
    begin_pages = split_ids * int(chunk_pages)
    num_pages = np.minimum(int(chunk_pages), int(max_pages) - begin_pages).astype(
        np.int32,
        copy=False,
    )
    split_tokens = np.minimum(
        tokens[request_ids] - begin_pages * _PAGE_SIZE,
        num_pages * _PAGE_SIZE,
    ).astype(np.int32, copy=False)
    max_split_tokens = int(split_tokens.max()) if split_tokens.size else 0
    if include_split_token_counts:
        split_token_counts_array = np.zeros((batch, max_splits), dtype=np.int32)
        split_token_counts_array[request_ids, split_ids] = split_tokens
    else:
        split_token_counts_array = None

    if split_tokens.size:
        order = np.argsort(-split_tokens, kind="stable")
        request_ids = request_ids[order]
        split_ids = split_ids[order]
        begin_pages = begin_pages[order]
        num_pages = num_pages[order]
        split_tokens = split_tokens[order]

    return _finish_split_plan_from_arrays(
        request_ids=request_ids,
        split_ids=split_ids,
        begin_pages=begin_pages,
        num_pages=num_pages,
        split_tokens=split_tokens,
        active_splits=active_splits,
        split_token_counts_array=split_token_counts_array,
        max_split_tokens=max_split_tokens,
        max_splits=max_splits,
        chunk_pages=chunk_pages,
        worker_count=worker_count,
    )


def _make_window_split_plan_from_ranges(
    *,
    ranges: list[tuple[int, int]],
    max_splits: int,
    chunk_pages: int,
    worker_count: int,
    include_split_token_counts: bool,
) -> _SplitPlan:
    batch = len(ranges)
    active_splits = [0] * batch
    split_token_counts_flat = (
        array("i", [0]) * (batch * max_splits)
        if include_split_token_counts
        else None
    )
    max_split_tokens = 0
    records: list[tuple[int, int, int, int, int, int]] = []
    for request, (window_begin, window_end) in enumerate(ranges):
        effective_tokens = window_end - window_begin
        if effective_tokens <= 0:
            continue
        first_page = window_begin // _PAGE_SIZE
        last_page = math.ceil(window_end / _PAGE_SIZE)
        active_pages = last_page - first_page
        request_splits = min(max_splits, math.ceil(active_pages / chunk_pages))
        active_splits[request] = request_splits
        for split in range(request_splits):
            begin_page = first_page + split * chunk_pages
            num_pages = min(chunk_pages, last_page - begin_page)
            split_begin = max(window_begin, begin_page * _PAGE_SIZE)
            split_end = min(window_end, (begin_page + num_pages) * _PAGE_SIZE)
            split_tokens = max(0, split_end - split_begin)
            token_begin = split_begin - begin_page * _PAGE_SIZE
            if split_token_counts_flat is not None:
                split_token_counts_flat[request * max_splits + split] = split_tokens
            if split_tokens > max_split_tokens:
                max_split_tokens = split_tokens
            records.append(
                (request, split, begin_page, num_pages, split_tokens, token_begin)
            )

    return _finish_split_plan(
        records=records,
        active_splits=active_splits,
        split_token_counts_flat=split_token_counts_flat,
        max_split_tokens=max_split_tokens,
        max_splits=max_splits,
        chunk_pages=chunk_pages,
        worker_count=worker_count,
    )


def _finish_split_plan(
    *,
    records: list[tuple[int, int, int, int, int, int]],
    active_splits: list[int],
    split_token_counts_flat: array | None,
    max_split_tokens: int,
    max_splits: int,
    chunk_pages: int,
    worker_count: int,
) -> _SplitPlan:
    records.sort(key=lambda row: row[_PLAN_TOKEN_FIELD], reverse=True)
    worker_loads = [0] * worker_count
    worker_task_counts = [0] * worker_count
    assigned_records: list[tuple[int, int, int, int, int, int, int]] = []
    if len(records) <= worker_count:
        for worker, record in enumerate(records):
            assigned_records.append((*record, worker))
            worker_loads[worker] = record[_PLAN_TOKEN_FIELD]
            worker_task_counts[worker] = 1
    else:
        worker_heap = [(0, worker) for worker in range(worker_count)]
        heapq.heapify(worker_heap)
        for record in records:
            load, worker = heapq.heappop(worker_heap)
            assigned_records.append((*record, worker))
            load += record[_PLAN_TOKEN_FIELD]
            worker_loads[worker] = load
            worker_task_counts[worker] += 1
            heapq.heappush(worker_heap, (load, worker))

    merge_workers = _make_merge_worker_ids(
        worker_loads=worker_loads,
        worker_task_counts=worker_task_counts,
        batch=len(active_splits),
        max_splits=max_splits,
    )
    split_plan, num_split_records = _make_split_plan_tensor(
        assigned_records=assigned_records,
        active_splits=active_splits,
        merge_workers=merge_workers,
    )
    split_token_counts_cpu = _make_split_token_counts_tensor(
        split_token_counts_flat,
        batch=len(active_splits),
        max_splits=max_splits,
    )
    return _SplitPlan(
        split_plan=split_plan,
        split_token_counts=split_token_counts_cpu,
        max_split_tokens=max_split_tokens,
        max_splits=max_splits,
        num_split_records=num_split_records,
        chunk_pages=chunk_pages,
        worker_count=worker_count,
        split_plan_buffer=split_plan,
    )


def _finish_split_plan_from_arrays(
    *,
    request_ids: np.ndarray,
    split_ids: np.ndarray,
    begin_pages: np.ndarray,
    num_pages: np.ndarray,
    split_tokens: np.ndarray,
    active_splits: list[int],
    split_token_counts_array: np.ndarray | None,
    max_split_tokens: int,
    max_splits: int,
    chunk_pages: int,
    worker_count: int,
) -> _SplitPlan:
    worker_loads = [0] * worker_count
    worker_task_counts = [0] * worker_count
    workers = np.empty((int(split_tokens.size),), dtype=np.int32)
    if int(split_tokens.size) <= worker_count:
        record_count = int(split_tokens.size)
        workers[:] = np.arange(record_count, dtype=np.int32)
        for worker, token_count in enumerate(split_tokens.tolist()):
            worker_loads[worker] = int(token_count)
            worker_task_counts[worker] = 1
    else:
        worker_heap = [(0, worker) for worker in range(worker_count)]
        heapq.heapify(worker_heap)
        for idx, token_count in enumerate(split_tokens.tolist()):
            load, worker = heapq.heappop(worker_heap)
            workers[idx] = worker
            load += int(token_count)
            worker_loads[worker] = load
            worker_task_counts[worker] += 1
            heapq.heappush(worker_heap, (load, worker))

    merge_workers = np.asarray(
        _make_merge_worker_ids(
            worker_loads=worker_loads,
            worker_task_counts=worker_task_counts,
            batch=len(active_splits),
            max_splits=max_splits,
        ),
        dtype=np.int32,
    )
    zero_requests = np.nonzero(np.asarray(active_splits, dtype=np.int32) == 0)[0].astype(
        np.int32,
        copy=False,
    )
    num_records = int(split_tokens.size) + int(zero_requests.size)
    split_plan_capacity = (
        len(active_splits) * int(max_splits)
        if split_token_counts_array is None
        else num_records
    )
    split_plan_array = np.empty((split_plan_capacity, _PLAN_FIELDS), dtype=np.int32)
    split_plan_rows = split_plan_array[:num_records]
    active_splits_array = np.asarray(active_splits, dtype=np.int32)
    record_count = int(split_tokens.size)
    if record_count > 0:
        split_plan_rows[:record_count, 0] = request_ids
        split_plan_rows[:record_count, 1] = split_ids
        split_plan_rows[:record_count, 2] = begin_pages
        split_plan_rows[:record_count, 3] = num_pages
        split_plan_rows[:record_count, 4] = split_tokens
        split_plan_rows[:record_count, 5] = 0
        split_plan_rows[:record_count, 6] = workers
        split_plan_rows[:record_count, 7] = active_splits_array[request_ids]
        split_plan_rows[:record_count, 8] = merge_workers[request_ids]
    if zero_requests.size > 0:
        zero_rows = split_plan_rows[record_count:]
        zero_rows[:, 0] = zero_requests
        zero_rows[:, 1] = -1
        zero_rows[:, 2:6] = 0
        zero_rows[:, 6] = merge_workers[zero_requests]
        zero_rows[:, 7] = 0
        zero_rows[:, 8] = zero_rows[:, 6]

    if split_token_counts_array is None:
        split_token_counts = torch.empty(
            (len(active_splits), 0),
            dtype=torch.int32,
            device="cpu",
        )
    else:
        split_token_counts = torch.from_numpy(split_token_counts_array)

    split_plan_buffer = torch.from_numpy(split_plan_array)
    return _SplitPlan(
        split_plan=split_plan_buffer[:num_records],
        split_token_counts=split_token_counts,
        max_split_tokens=max_split_tokens,
        max_splits=max_splits,
        num_split_records=num_records,
        chunk_pages=chunk_pages,
        worker_count=worker_count,
        split_plan_buffer=split_plan_buffer,
    )


def _usable_split_plan_buffer(
    buffer: torch.Tensor | None,
    *,
    capacity: int,
) -> torch.Tensor | None:
    if buffer is None:
        return None
    if (
        buffer.device.type != "cpu"
        or buffer.dtype is not torch.int32
        or not buffer.is_contiguous()
        or buffer.numel() < int(capacity) * _PLAN_FIELDS
    ):
        return None
    return buffer.reshape(-1, _PLAN_FIELDS)[: int(capacity)]


def _make_split_plan_tensor(
    *,
    assigned_records: list[tuple[int, int, int, int, int, int, int]],
    active_splits: TypingSequence[int],
    merge_workers: TypingSequence[int],
) -> tuple[torch.Tensor, int]:
    flat = array("i")
    seen = [False] * len(active_splits)
    for record in assigned_records:
        request = int(record[0])
        seen[request] = True
        flat.extend(
            (
                *record,
                int(active_splits[request]),
                int(merge_workers[request]),
            )
        )
    for request, was_seen in enumerate(seen):
        if was_seen:
            continue
        merge_worker = int(merge_workers[request])
        flat.extend((request, -1, 0, 0, 0, 0, merge_worker, 0, merge_worker))
    num_records = len(flat) // _PLAN_FIELDS
    return _int_array_to_tensor(flat, fields=_PLAN_FIELDS), num_records


def _make_split_token_counts_tensor(
    flat: array | None,
    *,
    batch: int,
    max_splits: int,
) -> torch.Tensor:
    if flat is None:
        return torch.empty((int(batch), 0), dtype=torch.int32, device="cpu")
    return _int_array_to_tensor(flat, fields=max_splits)


def _int_array_to_tensor(flat: array, *, fields: int) -> torch.Tensor:
    if len(flat) == 0:
        return torch.empty((0, int(fields)), dtype=torch.int32, device="cpu")
    return torch.frombuffer(flat, dtype=torch.int32).view(-1, int(fields))


def _auto_max_splits(
    *,
    batch: int,
    actual_max_tokens: int,
    total_tokens: int,
    max_pages: int,
    worker_count: int,
) -> int:
    if (
        batch <= 0
        or actual_max_tokens <= 0
        or total_tokens < _AUTO_SPLIT_MIN_TOTAL_TOKENS
        or max_pages <= 1
    ):
        return 1

    per_request_cap = (
        _AUTO_SPLIT_LONG_MAX_SPLITS
        if actual_max_tokens >= _AUTO_SPLIT_LONG_CONTEXT_TOKENS
        else _AUTO_SPLIT_SHORT_MAX_SPLITS
    )
    target_records = max(
        1,
        math.ceil(
            worker_count
            * _AUTO_SPLIT_TARGET_RECORDS_NUM
            / _AUTO_SPLIT_TARGET_RECORDS_DEN
        ),
    )
    target_records = min(target_records, batch * per_request_cap)
    max_splits = math.ceil(actual_max_tokens * target_records / total_tokens)
    max_splits = max(1, min(per_request_cap, max_splits))
    chunk_page_cap = max(1, math.ceil(max_pages / _AUTO_SPLIT_MIN_CHUNK_PAGES))
    return max(1, min(max_splits, chunk_page_cap))


def _make_merge_workers(*, worker_loads: list[int], batch: int) -> torch.Tensor:
    return torch.tensor(
        _make_merge_worker_ids(worker_loads=worker_loads, batch=batch),
        dtype=torch.int32,
        device="cpu",
    )


def _max_local_tasks_per_worker(
    *, batch: int, max_splits: int, worker_count: int
) -> int:
    merge_worker_count = max(1, min(batch, worker_count, _MAX_MERGE_WORKERS))
    return (
        (batch * max_splits + worker_count - 1) // worker_count
        + 2
        + (batch + merge_worker_count - 1) // merge_worker_count
        + 1
    )


def _make_merge_worker_ids(
    *,
    worker_loads: list[int],
    batch: int,
    worker_task_counts: list[int] | None = None,
    max_splits: int = 1,
) -> list[int]:
    worker_count = len(worker_loads)
    if worker_count <= 0:
        raise MKConfigError("worker_count must be positive")
    if batch <= 0:
        return []

    ranked_workers = sorted(
        range(worker_count), key=lambda idx: (worker_loads[idx], idx)
    )
    if worker_task_counts is None or max_splits <= 1:
        worker_task_counts = [0] * worker_count
        max_local_tasks = batch + 1
    elif len(worker_task_counts) != worker_count:
        raise MKConfigError("worker_task_counts must match worker_loads")
    else:
        max_local_tasks = _max_local_tasks_per_worker(
            batch=batch,
            max_splits=max_splits,
            worker_count=worker_count,
        )

    remaining_slots = [
        max(0, int(max_local_tasks) - int(worker_task_counts[idx]))
        for idx in range(worker_count)
    ]
    merge_worker_pool = [idx for idx in ranked_workers if remaining_slots[idx] > 0]
    if not merge_worker_pool:
        raise MKConfigError("merge worker queue capacity exceeded")
    merge_worker_count = max(
        1, min(batch, len(merge_worker_pool), _MAX_MERGE_WORKERS)
    )
    selected_workers = merge_worker_pool[:merge_worker_count]
    if sum(remaining_slots[idx] for idx in selected_workers) < batch:
        selected_workers = sorted(
            merge_worker_pool,
            key=lambda idx: (-remaining_slots[idx], worker_loads[idx], idx),
        )[:merge_worker_count]
        selected_workers.sort(key=lambda idx: (worker_loads[idx], idx))

    selected_remaining = {idx: remaining_slots[idx] for idx in selected_workers}
    merge_workers: list[int] = []
    cursor = 0
    for _request in range(batch):
        for _attempt in range(len(selected_workers)):
            worker = selected_workers[cursor % len(selected_workers)]
            cursor += 1
            if selected_remaining[worker] > 0:
                selected_remaining[worker] -= 1
                merge_workers.append(worker)
                break
        else:
            raise MKConfigError("merge worker queue capacity exceeded")
    return merge_workers


