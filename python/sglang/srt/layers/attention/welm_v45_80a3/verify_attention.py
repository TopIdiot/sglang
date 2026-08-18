# Vendored from the k-dash kernel source package welm/v45_80a3_attention.
# The CUDA kernel.so is resolved at runtime by k_dash.get(); that repo is a
# k-dash Source Package, not a Python distribution, so the host planner is
# mirrored here. Keep edits in sync with the upstream repo.
from __future__ import annotations

import math
from dataclasses import dataclass, field
import heapq
from collections.abc import Sequence

import torch

from .kernel_runtime import (
    ConfigError as MKConfigError,
    prepare_kernel,
)


_ARG_DEFS = (
    ("query", "void*"),
    ("key_cache", "void*"),
    ("value_cache", "void*"),
    ("page_indptr", "int32_t*"),
    ("page_indices", "int32_t*"),
    ("cache_seqlens", "int32_t*"),
    ("cu_seqlens_q", "int32_t*"),
    ("sinks", "void*"),
    ("output", "void*"),
    ("total_q", "int32_t"),
    ("batch_size", "int32_t"),
    ("max_seqlen_q", "int32_t"),
    ("softmax_scale", "float"),
    ("window_left", "int32_t"),
    ("window_right", "int32_t"),
)

_WGMMA_DIRECT_ARG_DEFS = (
    ("query", "void*"),
    ("key_cache", "void*"),
    ("value_cache", "void*"),
    ("num_cache_pages", "int32_t"),
    ("page_indptr", "int32_t*"),
    ("page_indices", "int32_t*"),
    ("cache_seqlens", "int32_t*"),
    ("sinks", "void*"),
    ("output", "void*"),
    ("batch_size", "int32_t"),
    ("softmax_scale", "float"),
    ("window_left", "int32_t"),
)

_WGMMA_PERSISTENT_ARG_DEFS = (
    ("query", "void*"),
    ("key_cache", "void*"),
    ("value_cache", "void*"),
    ("num_cache_pages", "int32_t"),
    ("batch_size", "int32_t"),
    ("page_indices", "int32_t*"),
    ("sinks", "void*"),
    ("output", "void*"),
    ("plan_records", "int32_t*"),
    ("num_plan_records", "int32_t"),
    ("num_events", "int32_t"),
    ("num_partial_slots", "int32_t"),
    ("softmax_scale", "float"),
    ("partial_scratch", "void*"),
    ("workspace", "void*"),
    ("cpu_workspace", "void*"),
    ("runtime_cache_seqlens", "int32_t*"),
    ("runtime_page_stride", "int32_t"),
    ("runtime_window_left", "int32_t"),
    ("plan_record_capacity", "int32_t"),
    ("event_capacity", "int32_t"),
    ("partial_slot_capacity", "int32_t"),
)

_PLAN_FIELDS = 16
_PLAN_DIRECT = 0
_PLAN_PARTIAL = 1
_PLAN_MERGE = 2
_WORKSPACE_ALIGNMENT = 256
_PARTIAL_ALIGNMENT = 256
_PARTIAL_O_BYTES = 4 * 48 * 32 * 4
_PARTIAL_ML_BYTES = 4 * 6 * 4
_WARP_MMA_DATAFLOW = "warp_mma_independent_q"
_WGMMA_DATAFLOW = "wgmma_kv64_n24"
_COREV2_DATAFLOW = "wgmma_kv64_n24_corev2"
_AUTO_DATAFLOW = "auto"
_DEFAULT_DATAFLOW = _AUTO_DATAFLOW
_TWO_KERNEL_PARTIAL_MERGE = "two_kernel"
_LAST_ARRIVER_PARTIAL_MERGE = "last_arriver"
_AUTO_PARTIAL_MERGE = "auto"
_DEFAULT_PARTIAL_MERGE_MODE = _AUTO_PARTIAL_MERGE
_RUNTIME_PLANNER_MAX_BATCH = 256
VERIFY_ATTENTION_PARTIAL_MERGE_MODES = (
    _AUTO_PARTIAL_MERGE,
    _TWO_KERNEL_PARTIAL_MERGE,
    _LAST_ARRIVER_PARTIAL_MERGE,
)

# Same-run H20 A/B shows that the last-arriver atomic is profitable when one
# launch removes many independent, shallow merge events.  Keep that topology
# rule as the short-context policy; the cost model below separately evaluates
# deep fan-in.  Neither policy uses batch/window labels or fixture identity.
_LAST_ARRIVER_MAX_PARTIALS_PER_EVENT = 4
_LAST_ARRIVER_MERGE_COST_CREDIT = 0.40


def _use_wgmma_static_persistent_executor(
    num_direct_tasks: int,
    num_partial_tasks: int,
    total_attention_blocks: int,
    worker_count: int,
) -> bool:
    """Use persistent workers when unified scheduling amortizes best."""
    mixed_wave = num_direct_tasks > 0 and num_partial_tasks > 0
    shallow_direct_waves = (
        num_partial_tasks == 0
        and num_direct_tasks > worker_count
        and total_attention_blocks <= 8 * num_direct_tasks
    )
    return mixed_wave or shallow_direct_waves


def _resolve_partial_merge_mode(
    partial_merge_mode: str,
    split_counts: Sequence[int],
    attention_dataflow: str,
    worker_count: int = 78,
    block_counts: Sequence[int] | None = None,
) -> str:
    """Choose a merge executor from event topology and scheduled KV work."""

    if partial_merge_mode != _AUTO_PARTIAL_MERGE:
        return partial_merge_mode
    if attention_dataflow != _WARP_MMA_DATAFLOW:
        return _TWO_KERNEL_PARTIAL_MERGE
    event_partial_counts = [count for count in split_counts if count > 1]
    if (
        # The two-kernel path uses six head-parallel merge CTAs per event
        # while that layout fits in one physical worker wave.  Beyond that
        # boundary it falls back to one heavier all-head CTA per event; this
        # is the regime where eliminating the extra launch repays atomics.
        len(event_partial_counts) * 6 > worker_count
        and max(event_partial_counts, default=0)
        <= _LAST_ARRIVER_MAX_PARTIALS_PER_EVENT
    ):
        return _LAST_ARRIVER_PARTIAL_MERGE
    if len(event_partial_counts) * 6 <= worker_count:
        return _TWO_KERNEL_PARTIAL_MERGE
    if block_counts is not None and event_partial_counts:
        two_kernel_cost = _attention_schedule_cost(
            block_counts,
            split_counts,
            worker_count=worker_count,
            attention_dataflow=attention_dataflow,
            partial_merge_mode=_TWO_KERNEL_PARTIAL_MERGE,
        )
        last_arriver_cost = _attention_schedule_cost(
            block_counts,
            split_counts,
            worker_count=worker_count,
            attention_dataflow=attention_dataflow,
            partial_merge_mode=_LAST_ARRIVER_PARTIAL_MERGE,
        )
        if last_arriver_cost < two_kernel_cost:
            return _LAST_ARRIVER_PARTIAL_MERGE
    return _TWO_KERNEL_PARTIAL_MERGE


def _work_balanced_split_counts(
    block_counts: Sequence[int],
    *,
    tokens_per_block: int,
    worker_count: int,
) -> list[int]:
    """Partition effective KV work into coarse, approximately one-wave chunks.

    The chunk floor and 256-token quantization mirror the persistent-attention
    scheduling granularity used by FlashInfer.  Per-request ceil division is
    intentional: the resulting logical partial count may exceed the physical
    worker count for heterogeneous batches, while the actual persistent launch
    remains fixed at exactly one CTA per SM.
    """

    total_tokens = sum(tokens_per_block * blocks for blocks in block_counts)
    raw_chunk_tokens = math.ceil(1.10 * total_tokens / worker_count)
    chunk_tokens = (
        128
        if raw_chunk_tokens <= 128
        else math.ceil(raw_chunk_tokens / 256) * 256
    )
    return [
        min(
            blocks,
            max(1, math.ceil(tokens_per_block * blocks / chunk_tokens)),
        )
        for blocks in block_counts
    ]


# Non-negative wave-makespan fit from the H20 packed-PV representative sweep.
# These coefficients depend only on physical launch waves, per-wave KV blocks,
# and merge fan-in; batch/window labels are deliberately absent from the model.
_WARP_TASK_FIXED_COST = 11.52069028
_WARP_BLOCK_COST = 1.58275269
_WARP_MERGE_FIXED_COST = 1.57760745
_WARP_MERGE_SHARD_COST = 0.64547019
_WARP_MERGE_ALL_HEADS_COST = 2.66450638
# H20 fit for the single-launch last-arriver path on the FA3-loss split sweep.
# Unlike the two-kernel path, launch-wave boundaries do not add host launches;
# the residual cost is the grid-wide KV makespan plus a small per-partial
# publication cost.  The shallow-event heuristic below remains in place for
# short contexts where this deep-fan-in fit's intercept is not representative.
_WARP_LAST_ARRIVER_DIRECT_BLOCK_COST = 1.71623638
_WARP_LAST_ARRIVER_PARTIAL_FIXED_COST = 10.23675619
_WARP_LAST_ARRIVER_PARTIAL_BLOCK_COST = 1.63374226
_WARP_LAST_ARRIVER_PARTIAL_TASK_COST = 0.14545944
_WGMMA_KERNEL_FIXED_COST = 0.3198019456145704
_WGMMA_TASK_FIXED_COST = 14.953081498573994
_WGMMA_BLOCK_COST = 3.8766255028017356
_WGMMA_MERGE_FIXED_COST = 0.0
_WGMMA_MERGE_FAN_IN_COST = 1.230793131647856


@dataclass(frozen=True)
class _AttentionScheduleFeatures:
    attention_fixed_units: int
    attention_block_units: int
    merge_waves: int
    merge_fan_in_wave_sum: int
    merge_head_parallel: bool
    persistent_worker_profiles: tuple[tuple[int, int], ...] = ()
    direct_attention_fixed_units: int = 0
    direct_attention_block_units: int = 0
    partial_attention_fixed_units: int = 0
    partial_attention_block_units: int = 0
    num_partial_tasks: int = 0


def _allocate_split_counts(
    block_counts: Sequence[int],
    target_tasks: int,
    *,
    max_splits_per_request: int | None = None,
) -> list[int]:
    split_counts = [1] * len(block_counts)
    split_heap = [
        (-block_counts[index], index)
        for index in range(len(block_counts))
        if block_counts[index] > 1
    ]
    heapq.heapify(split_heap)
    total_tasks = len(split_counts)
    while total_tasks < target_tasks and split_heap:
        _, request = heapq.heappop(split_heap)
        split_counts[request] += 1
        total_tasks += 1
        request_limit = block_counts[request]
        if max_splits_per_request is not None:
            request_limit = min(request_limit, max_splits_per_request)
        if split_counts[request] < request_limit:
            next_cost = block_counts[request] / split_counts[request]
            heapq.heappush(split_heap, (-next_cost, request))
    return split_counts


def _attention_schedule_features(
    block_counts: Sequence[int],
    split_counts: Sequence[int],
    *,
    worker_count: int,
    attention_dataflow: str,
) -> _AttentionScheduleFeatures:
    """Describe the two-kernel attention and fan-in launch waves.

    A physical launch contains at most ``worker_count`` CTAs.  Its duration is
    therefore modeled by the largest segment in each Warp launch wave.  WGMMA
    instead uses one fixed 78-CTA persistent launch, so its feature is the
    largest LPT-assigned worker load.  Final fan-in is a separate phase for
    both.  This is intentionally a shape model: batch size and window mode
    affect the result only through the effective KV spans.
    """

    if attention_dataflow not in (
        _WARP_MMA_DATAFLOW,
        _WGMMA_DATAFLOW,
        _COREV2_DATAFLOW,
    ):
        raise ValueError(f"unsupported attention dataflow: {attention_dataflow}")

    direct_blocks: list[int] = []
    partial_blocks: list[int] = []
    all_blocks: list[int] = []
    ordered = sorted(
        zip(block_counts, split_counts), reverse=True
    )
    for blocks, splits in ordered:
        destination = direct_blocks if splits == 1 else partial_blocks
        for split in range(splits):
            segment_blocks = (
                (split + 1) * blocks // splits - split * blocks // splits
            )
            destination.append(segment_blocks)
            all_blocks.append(segment_blocks)

    persistent_worker_profiles: tuple[tuple[int, int], ...] = ()
    direct_attention_fixed_units = 0
    direct_attention_block_units = 0
    partial_attention_fixed_units = 0
    partial_attention_block_units = 0
    for tasks, is_partial in (
        (direct_blocks, False),
        (partial_blocks, True),
    ):
        for offset in range(0, len(tasks), worker_count):
            if is_partial:
                partial_attention_fixed_units += 1
                partial_attention_block_units += max(
                    tasks[offset : offset + worker_count]
                )
            else:
                direct_attention_fixed_units += 1
                direct_attention_block_units += max(
                    tasks[offset : offset + worker_count]
                )
    if attention_dataflow == _WARP_MMA_DATAFLOW:
        attention_fixed_units = (
            direct_attention_fixed_units + partial_attention_fixed_units
        )
        attention_block_units = (
            direct_attention_block_units + partial_attention_block_units
        )
    else:
        worker_heap = [(0, worker) for worker in range(worker_count)]
        heapq.heapify(worker_heap)
        worker_tasks = [0] * worker_count
        worker_blocks = [0] * worker_count
        for segment_blocks in all_blocks:
            load, worker = heapq.heappop(worker_heap)
            worker_tasks[worker] += 1
            worker_blocks[worker] += segment_blocks
            heapq.heappush(
                worker_heap,
                (load + 48 + 64 * segment_blocks, worker),
            )
        persistent_worker_profiles = tuple(
            zip(worker_tasks, worker_blocks)
        )
        attention_fixed_units = max(worker_tasks)
        attention_block_units = max(worker_blocks)
    merge_splits = [splits for splits in split_counts if splits > 1]
    if not merge_splits:
        return _AttentionScheduleFeatures(
            attention_fixed_units,
            attention_block_units,
            0,
            0,
            False,
            persistent_worker_profiles,
            direct_attention_fixed_units,
            direct_attention_block_units,
            partial_attention_fixed_units,
            partial_attention_block_units,
            len(partial_blocks),
        )

    # Keep this threshold aligned with C++ runtime_schedule_cost() and the
    # dynamic launch dispatch in verify_attention_welmv45.cuh. Dynamic graph
    # capture initializes every request at max_pages (so merge count=batch)
    # and reserves event_capacity=batch; all three paths then use <= 3.
    merge_row_parallel = (
        attention_dataflow == _WARP_MMA_DATAFLOW
        and len(merge_splits) <= 3
    )
    merge_head_parallel = (
        attention_dataflow == _WARP_MMA_DATAFLOW
        and (
            merge_row_parallel
            or len(merge_splits) * 6 <= worker_count
        )
    )
    if merge_row_parallel:
        merge_fan_in = [
            math.ceil(splits / 12)
            for splits in merge_splits
            for _ in range(24)
        ]
    elif merge_head_parallel:
        merge_fan_in = [
            math.ceil(splits / 3)
            for splits in merge_splits
            for _ in range(6)
        ]
    else:
        merge_fan_in = merge_splits
    merge_waves = 0
    merge_fan_in_wave_sum = 0
    for offset in range(0, len(merge_fan_in), worker_count):
        merge_waves += 1
        merge_fan_in_wave_sum += max(
            merge_fan_in[offset : offset + worker_count]
        )
    return _AttentionScheduleFeatures(
        attention_fixed_units,
        attention_block_units,
        merge_waves,
        merge_fan_in_wave_sum,
        merge_head_parallel,
        persistent_worker_profiles,
        direct_attention_fixed_units,
        direct_attention_block_units,
        partial_attention_fixed_units,
        partial_attention_block_units,
        len(partial_blocks),
    )


def _attention_schedule_cost(
    block_counts: Sequence[int],
    split_counts: Sequence[int],
    *,
    worker_count: int,
    attention_dataflow: str,
    partial_merge_mode: str = _TWO_KERNEL_PARTIAL_MERGE,
) -> float:
    features = _attention_schedule_features(
        block_counts,
        split_counts,
        worker_count=worker_count,
        attention_dataflow=attention_dataflow,
    )
    if attention_dataflow == _WARP_MMA_DATAFLOW:
        merge_fan_in_cost = (
            _WARP_MERGE_SHARD_COST
            if features.merge_head_parallel
            else _WARP_MERGE_ALL_HEADS_COST
        )
        two_kernel_cost = (
            _WARP_TASK_FIXED_COST * features.attention_fixed_units
            + _WARP_BLOCK_COST * features.attention_block_units
            + _WARP_MERGE_FIXED_COST * features.merge_waves
            + merge_fan_in_cost * features.merge_fan_in_wave_sum
        )
        if not any(splits > 1 for splits in split_counts):
            return two_kernel_cost
        event_partial_counts = [splits for splits in split_counts if splits > 1]
        shallow_last_arriver = (
            len(event_partial_counts) * 6 > worker_count
            and max(event_partial_counts, default=0)
            <= _LAST_ARRIVER_MAX_PARTIALS_PER_EVENT
        )
        last_arriver_cost = (
            _WARP_LAST_ARRIVER_DIRECT_BLOCK_COST
            * features.direct_attention_block_units
            + _WARP_LAST_ARRIVER_PARTIAL_FIXED_COST
            + _WARP_LAST_ARRIVER_PARTIAL_BLOCK_COST
            * features.partial_attention_block_units
            + _WARP_LAST_ARRIVER_PARTIAL_TASK_COST
            * features.num_partial_tasks
        )
        if partial_merge_mode == _LAST_ARRIVER_PARTIAL_MERGE:
            return last_arriver_cost
        if partial_merge_mode == _AUTO_PARTIAL_MERGE:
            if shallow_last_arriver:
                return two_kernel_cost - _LAST_ARRIVER_MERGE_COST_CREDIT * (
                    _WARP_MERGE_FIXED_COST * features.merge_waves
                    + merge_fan_in_cost * features.merge_fan_in_wave_sum
                )
            if features.merge_head_parallel:
                return two_kernel_cost
            return min(two_kernel_cost, last_arriver_cost)
        return two_kernel_cost
    attention_cost = max(
        _WGMMA_TASK_FIXED_COST * task_count
        + _WGMMA_BLOCK_COST * block_count
        for task_count, block_count in features.persistent_worker_profiles
    )
    return (
        _WGMMA_KERNEL_FIXED_COST
        + attention_cost
        + _WGMMA_MERGE_FIXED_COST * features.merge_waves
        + _WGMMA_MERGE_FAN_IN_COST * features.merge_fan_in_wave_sum
    )


def _cost_model_split_counts(
    block_counts: Sequence[int],
    *,
    worker_count: int,
    attention_dataflow: str = _WARP_MMA_DATAFLOW,
    partial_merge_mode: str = _TWO_KERNEL_PARTIAL_MERGE,
) -> list[int]:
    """Select KV splits by minimizing estimated attention + merge makespan.

    Search is intentionally allowed to cross the physical-SM boundary.  The
    objective itself decides whether an additional logical wave is worth its
    shorter, better-balanced segments; no ``partial_count <= SM count`` rule is
    encoded here.
    """

    batch = len(block_counts)
    maximum_target = sum(block_counts)
    best_splits = [1] * batch
    best_cost = _attention_schedule_cost(
        block_counts,
        best_splits,
        worker_count=worker_count,
        attention_dataflow=attention_dataflow,
        partial_merge_mode=partial_merge_mode,
    )
    candidate = [1] * batch
    split_heap = [
        (-block_counts[index], index)
        for index in range(batch)
        if block_counts[index] > 1
    ]
    heapq.heapify(split_heap)
    targets_without_improvement = 0
    # The mixed direct/partial cost curve is non-convex at physical-wave
    # boundaries.  Two stale waves can stop immediately before converting the
    # last direct requests into a substantially cheaper all-partial topology.
    # Scan one additional wave while retaining a bounded planner cost.
    stale_target_limit = max(3 * worker_count, batch)
    for _ in range(batch + 1, maximum_target + 1):
        if not split_heap:
            break
        _, request = heapq.heappop(split_heap)
        candidate[request] += 1
        if candidate[request] < block_counts[request]:
            heapq.heappush(
                split_heap,
                (-block_counts[request] / candidate[request], request),
            )
        candidate_cost = _attention_schedule_cost(
            block_counts,
            candidate,
            worker_count=worker_count,
            attention_dataflow=attention_dataflow,
            partial_merge_mode=partial_merge_mode,
        )
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_splits = candidate.copy()
            targets_without_improvement = 0
        else:
            targets_without_improvement += 1
            if targets_without_improvement >= stale_target_limit:
                break
    return best_splits


def _resolve_attention_dataflow(
    attention_dataflow: str,
    span_page_counts: Sequence[int],
    worker_count: int,
) -> str:
    """Choose the lowest-cost dataflow from effective per-request KV spans."""

    return _resolve_attention_schedule(
        attention_dataflow,
        span_page_counts,
        worker_count,
        partial_merge_mode=_AUTO_PARTIAL_MERGE,
    )[0]


def _resolve_attention_schedule(
    attention_dataflow: str,
    span_page_counts: Sequence[int],
    worker_count: int,
    *,
    partial_merge_mode: str = _AUTO_PARTIAL_MERGE,
) -> tuple[str, list[int], float]:
    """Return dataflow, KV split counts, and estimated makespan."""

    if attention_dataflow != _AUTO_DATAFLOW:
        pages_per_block = 2 if attention_dataflow == _WARP_MMA_DATAFLOW else 4
        block_counts = [
            (pages + pages_per_block - 1) // pages_per_block
            for pages in span_page_counts
        ]
        split_counts = _cost_model_split_counts(
            block_counts,
            worker_count=worker_count,
            attention_dataflow=attention_dataflow,
            partial_merge_mode=partial_merge_mode,
        )
        cost = _attention_schedule_cost(
            block_counts,
            split_counts,
            worker_count=worker_count,
            attention_dataflow=attention_dataflow,
            partial_merge_mode=partial_merge_mode,
        )
        return (
            attention_dataflow,
            split_counts,
            cost,
        )
    candidates: list[tuple[float, str, list[int]]] = []
    for candidate_dataflow, pages_per_block in (
        (_WARP_MMA_DATAFLOW, 2),
        (_WGMMA_DATAFLOW, 4),
    ):
        block_counts = [
            (pages + pages_per_block - 1) // pages_per_block
            for pages in span_page_counts
        ]
        split_counts = _cost_model_split_counts(
            block_counts,
            worker_count=worker_count,
            attention_dataflow=candidate_dataflow,
            partial_merge_mode=partial_merge_mode,
        )
        cost = _attention_schedule_cost(
            block_counts,
            split_counts,
            worker_count=worker_count,
            attention_dataflow=candidate_dataflow,
            partial_merge_mode=partial_merge_mode,
        )
        candidates.append((cost, candidate_dataflow, split_counts))
    cost, candidate_dataflow, split_counts = min(candidates)
    return candidate_dataflow, split_counts, cost


@dataclass
class VerifyAttentionWelmv45Plan:
    """Pointer-free, shape-dynamic schedule for the fixed WeLM verify shape."""

    _prepared: object
    _plan_records: torch.Tensor
    _cpu_workspace_template: torch.Tensor
    _cpu_replan_staging: tuple[torch.Tensor, ...] = field(
        default_factory=tuple, repr=False
    )
    _cpp_replan_stager: object | None = field(default=None, repr=False)
    _cpp_planner_state: torch.Tensor | None = field(default=None, repr=False)
    _replan_meta: torch.Tensor | None = field(default=None, repr=False)
    _gpu_workspace_template: torch.Tensor | None = None
    _ready_workspace_ptrs: set[int] = field(default_factory=set, repr=False)
    batch_size: int = 0
    page_table_shape: tuple[int, int] = (0, 0)
    page_indices_count: int = 0
    num_plan_records: int = 0
    num_direct_tasks: int = 0
    num_partial_tasks: int = 0
    num_merge_tasks: int = 0
    num_events: int = 0
    num_partial_slots: int = 0
    plan_record_capacity: int = 0
    event_capacity: int = 0
    partial_slot_capacity: int = 0
    worker_count: int = 0
    window_size: tuple[int, int] = (-1, -1)
    attention_dataflow: str = _DEFAULT_DATAFLOW
    partial_merge_mode: str = _DEFAULT_PARTIAL_MERGE_MODE
    estimated_schedule_cost: float = 0.0
    estimated_attention_fixed_units: int = 0
    estimated_attention_block_units: int = 0
    estimated_merge_waves: int = 0
    estimated_merge_fan_in_wave_sum: int = 0
    estimated_merge_head_parallel: bool = False
    workspace_size: int = 0
    workspace_alignment: int = _WORKSPACE_ALIGNMENT
    partial_scratch_size: int = 0
    partial_scratch_alignment: int = _PARTIAL_ALIGNMENT
    dynamic_runtime_metadata: bool = False
    runtime_auto_dataflow: bool = False
    selected_attention_dataflow: str = _WARP_MMA_DATAFLOW
    device: torch.device = torch.device("cpu")


def verify_attention_welmv45_plan(
    page_table_shape: tuple[int, int],
    page_indptr: torch.Tensor | Sequence[int],
    cache_seqlens: torch.Tensor | Sequence[int],
    cu_seqlens_q: torch.Tensor | Sequence[int],
    device: torch.device | str | int,
    *,
    max_seqlen_q: int = 4,
    causal: bool = True,
    softcap: float = 0.0,
    window_size: tuple[int, int] = (-1, -1),
    schedule_policy: str | None = None,
    attention_dataflow: str | None = None,
    partial_merge_mode: str = _DEFAULT_PARTIAL_MERGE_MODE,
    kv_producer_warps: int = 1,
    kv_phase_stages: int = 3,
    worker_count: int | None = None,
    target_task_records: int | None = None,
    dynamic_runtime_metadata: bool = False,
    runtime_plan_capacity: tuple[int, int, int] | None = None,
) -> VerifyAttentionWelmv45Plan:
    """Create the runtime-sized persistent schedule from host metadata only.

    The four verify tokens always form one linear causal draft.  Arbitrary
    speculative-decode trees and caller-provided attention masks are outside
    this operator's contract; the planner derives every token's causal/window
    interval directly from ``cache_seqlens``.  With
    ``dynamic_runtime_metadata=True``, the plan fixes only the maximum padded
    page stride and task topology; each launch derives exact spans from a CUDA
    ``runtime_cache_seqlens`` tensor, making the plan safe for graph replay.

    Prefer ``schedule_policy`` (``auto`` / ``warp_mma_independent_q`` /
    ``wgmma_kv64_n24`` / ``wgmma_kv64_n24_corev2``). ``attention_dataflow`` is
    kept as a backward-compatible alias for the same string.
    """

    if schedule_policy is None and attention_dataflow is None:
        attention_dataflow = _DEFAULT_DATAFLOW
    elif schedule_policy is None:
        pass
    elif attention_dataflow is None:
        attention_dataflow = schedule_policy
    elif schedule_policy != attention_dataflow:
        raise MKConfigError(
            "schedule_policy and attention_dataflow disagree; pass only one"
        )
    assert attention_dataflow is not None

    if (
        not isinstance(page_table_shape, tuple)
        or len(page_table_shape) != 2
        or int(page_table_shape[0]) <= 0
        or int(page_table_shape[1]) <= 0
    ):
        raise MKConfigError(
            "page_table_shape must be a positive (batch, max_pages) pair"
        )
    batch = int(page_table_shape[0])
    max_pages = int(page_table_shape[1])
    indptr = _host_int32_values(page_indptr, "page_indptr")
    lengths = _host_int32_values(cache_seqlens, "cache_seqlens")
    cu_q = _host_int32_values(cu_seqlens_q, "cu_seqlens_q")
    if len(indptr) != batch + 1 or len(lengths) != batch or len(cu_q) != batch + 1:
        raise MKConfigError("host schedule metadata batch dimensions do not match")
    if indptr[0] != 0 or any(b < a for a, b in zip(indptr, indptr[1:])):
        raise MKConfigError("page_indptr must be monotonic and start at zero")
    if any(value != 4 * index for index, value in enumerate(cu_q)):
        raise MKConfigError("cu_seqlens_q must equal [0, 4, ..., 4 * batch]")
    if max_seqlen_q != 4:
        raise MKConfigError("verify attention requires max_seqlen_q == 4")
    if causal is not True or float(softcap) != 0.0:
        raise MKConfigError("verify attention requires causal=True and softcap=0.0")
    if window_size not in ((-1, -1), (512, 0)):
        raise MKConfigError("window_size must be (-1, -1) or (512, 0)")
    if attention_dataflow not in (
        _AUTO_DATAFLOW,
        _WARP_MMA_DATAFLOW,
        _WGMMA_DATAFLOW,
        _COREV2_DATAFLOW,
    ):
        raise MKConfigError(
            "schedule_policy must be auto, warp_mma_independent_q, "
            "wgmma_kv64_n24, or wgmma_kv64_n24_corev2"
        )
    if partial_merge_mode not in (
        _AUTO_PARTIAL_MERGE,
        _TWO_KERNEL_PARTIAL_MERGE,
        _LAST_ARRIVER_PARTIAL_MERGE,
    ):
        raise MKConfigError(
            "partial_merge_mode must be auto, two_kernel, or last_arriver"
        )
    if kv_producer_warps != 1 or kv_phase_stages != 3:
        raise MKConfigError(
            "verify attention requires one KV producer warp and 3 KV phase stages"
        )
    runtime_auto_dataflow = bool(
        dynamic_runtime_metadata and attention_dataflow == _AUTO_DATAFLOW
    )
    if dynamic_runtime_metadata:
        if batch > _RUNTIME_PLANNER_MAX_BATCH:
            raise MKConfigError(
                "dynamic runtime metadata supports batch size at most "
                f"{_RUNTIME_PLANNER_MAX_BATCH}"
            )
        if any(
            indptr[index + 1] - indptr[index] != max_pages
            for index in range(batch)
        ):
            raise MKConfigError(
                "dynamic runtime metadata requires a padded max_pages stride"
            )
        if attention_dataflow not in (
            _AUTO_DATAFLOW,
            _WARP_MMA_DATAFLOW,
            _WGMMA_DATAFLOW,
        ):
            raise MKConfigError(
                "dynamic runtime metadata requires auto, warp_mma_independent_q, "
                "or wgmma_kv64_n24"
            )
        partial_merge_mode = _TWO_KERNEL_PARTIAL_MERGE
        lengths = [max_pages * 16] * batch

    plan_device = _normalize_cuda_device(device)
    properties = torch.cuda.get_device_properties(plan_device)
    sm_count = int(properties.multi_processor_count)
    resolved_workers = sm_count if worker_count is None else int(worker_count)
    if resolved_workers != sm_count:
        raise MKConfigError("worker_count must equal the target GPU SM count")
    if resolved_workers <= 0:
        raise MKConfigError("worker_count must be positive")

    span_page_counts: list[int] = []
    first_pages: list[int] = []
    request_pages: list[int] = []
    for request, cache_len in enumerate(lengths):
        num_pages = indptr[request + 1] - indptr[request]
        if num_pages <= 0 or num_pages > max_pages:
            raise MKConfigError("each request must have 1..max_pages logical pages")
        if cache_len < 4 or cache_len > num_pages * 16:
            raise MKConfigError(
                "cache_seqlens must fit the paged cache and include 4 verify tokens"
            )
        first_valid = 0 if window_size[0] < 0 else max(0, cache_len - 4 - 512)
        first_page = (first_valid // 64) * 4
        # The last allocation page may be padding.  Scheduling only the pages
        # that can contain valid cache tokens both matches FA3's logical span
        # and avoids making allocator slack part of the kernel cost model.
        last_page = (cache_len + 15) // 16
        span_pages = last_page - first_page
        if span_pages <= 0:
            raise MKConfigError("planned verify span cannot be empty")
        first_pages.append(first_page)
        request_pages.append(last_page)
        span_page_counts.append(span_pages)

    schedule_dataflow = attention_dataflow
    if (
        partial_merge_mode == _LAST_ARRIVER_PARTIAL_MERGE
        and attention_dataflow == _AUTO_DATAFLOW
    ):
        schedule_dataflow = _WARP_MMA_DATAFLOW
    attention_dataflow, cost_model_split_counts, estimated_schedule_cost = (
        _resolve_attention_schedule(
            schedule_dataflow,
            span_page_counts,
            resolved_workers,
            partial_merge_mode=partial_merge_mode,
        )
    )
    if (
        partial_merge_mode == _LAST_ARRIVER_PARTIAL_MERGE
        and attention_dataflow != _WARP_MMA_DATAFLOW
    ):
        raise MKConfigError(
            "last_arriver partial merge is currently implemented only for "
            "warp_mma_independent_q"
        )
    pages_per_block = 2 if attention_dataflow == _WARP_MMA_DATAFLOW else 4
    block_counts = [
        (pages + pages_per_block - 1) // pages_per_block
        for pages in span_page_counts
    ]

    if target_task_records is None:
        # Choose the number of logical partials from effective KV work rather
        # than targeting a fixed number of task records.  This avoids turning
        # Window-512 requests into one tiny CTA per KV32 block and removes the
        # former low-batch 2.5-wave reduction cliff.
        split_counts = cost_model_split_counts

        # A CUDA Graph plan owns fixed-address record and partial buffers.  A
        # heterogeneous runtime batch can have a larger cost-model optimum
        # than the homogeneous max-context topology used during capture.  In
        # that case keep the runtime metadata exact, but cap only the number of
        # logical splits so the refreshed schedule fits the captured buffers.
        # Reserve one merge record per request; requests that remain direct
        # simply leave part of that reservation unused.
        if runtime_plan_capacity is not None:
            record_capacity, event_capacity, partial_capacity = map(
                int, runtime_plan_capacity
            )
            desired_partials = sum(
                splits for splits in split_counts if splits > 1
            )
            desired_merges = sum(splits > 1 for splits in split_counts)
            if (
                desired_partials > partial_capacity
                or desired_merges > event_capacity
                or sum(split_counts) + desired_merges > record_capacity
            ):
                capacity_target = min(
                    sum(block_counts),
                    partial_capacity,
                    record_capacity - batch,
                )
                if event_capacity < batch or capacity_target < batch:
                    raise MKConfigError(
                        "runtime plan capacity cannot hold one task per request"
                    )
                split_counts = _allocate_split_counts(
                    block_counts,
                    capacity_target,
                )
    elif dynamic_runtime_metadata:
        # The dynamic persistent executor always launches one CTA per physical
        # SM. Replanning only rewrites the fixed-capacity work-indptr, work-item,
        # and merge regions consumed by those captured launch nodes.
        name = (
            "verify_attention_welmv45_wgmma_optimized_dynamic_persistent_w"
            f"{resolved_workers}"
        )
        kernel_defs = (
            "mk::kernelv2::VerifyAttentionWelmv45WgmmaOptimizedPersistentKernel<"
            f"{resolved_workers}>"
        )
    else:
        target = int(target_task_records)
        if target < batch:
            raise MKConfigError("target_task_records cannot be smaller than batch size")
        target = min(target, sum(block_counts))
        split_counts = _allocate_split_counts(
            block_counts,
            target,
        )
        estimated_schedule_cost = 0.0
    schedule_features = _attention_schedule_features(
        block_counts,
        split_counts,
        worker_count=resolved_workers,
        attention_dataflow=attention_dataflow,
    )
    partial_merge_mode = _resolve_partial_merge_mode(
        partial_merge_mode,
        split_counts,
        attention_dataflow,
        resolved_workers,
        block_counts,
    )
    estimated_schedule_cost = _attention_schedule_cost(
        block_counts,
        split_counts,
        worker_count=resolved_workers,
        attention_dataflow=attention_dataflow,
        partial_merge_mode=partial_merge_mode,
    )

    attention_rows: list[list[int]] = []
    merge_specs: list[tuple[int, int, int, int]] = []
    worker_heap = [(0, worker) for worker in range(resolved_workers)]
    heapq.heapify(worker_heap)
    partial_cursor = 0
    event_cursor = 0
    for request in sorted(range(batch), key=lambda r: block_counts[r], reverse=True):
        splits = split_counts[request]
        request_partial_begin = partial_cursor
        request_event = event_cursor if splits > 1 else -1
        for split in range(splits):
            block_begin_in_span = split * block_counts[request] // splits
            block_end_in_span = (split + 1) * block_counts[request] // splits
            begin_page = first_pages[request] + block_begin_in_span * pages_per_block
            end_page = min(
                request_pages[request],
                first_pages[request] + block_end_in_span * pages_per_block,
            )
            num_pages = end_page - begin_page
            segment_tokens = num_pages * 16
            valid_begin: list[int] = []
            valid_end: list[int] = []
            origin = begin_page * 16
            cache_len = lengths[request]
            for token in range(4):
                causal_position = cache_len - 4 + token
                global_begin = (
                    0
                    if window_size[0] < 0
                    else max(0, causal_position - window_size[0])
                )
                valid_begin.append(max(0, min(segment_tokens, global_begin - origin)))
                valid_end.append(
                    max(0, min(segment_tokens, causal_position + 1 - origin))
                )
            load, worker = heapq.heappop(worker_heap)
            kind = _PLAN_DIRECT if splits == 1 else _PLAN_PARTIAL
            partial_slot = -1 if splits == 1 else partial_cursor
            row = [
                kind,
                worker,
                request,
                indptr[request] + begin_page,
                num_pages,
                *valid_begin,
                *valid_end,
                partial_slot,
                splits if splits > 1 else 0,
                request_event,
            ]
            if dynamic_runtime_metadata:
                # Runtime kernels reinterpret these otherwise-static interval
                # slots as the split ordinal and split count, then derive the
                # exact page span and causal/window intervals from device
                # cache_seqlens on every CUDA Graph replay.
                row[5] = split
                row[6] = splits
            if len(row) != _PLAN_FIELDS:
                raise AssertionError("internal verify plan row width mismatch")
            attention_rows.append(row)
            segment_blocks = block_end_in_span - block_begin_in_span
            predicted_cost = (
                48 + 64 * segment_blocks
                if attention_dataflow in (_WGMMA_DATAFLOW, _COREV2_DATAFLOW)
                else segment_blocks
            )
            heapq.heappush(worker_heap, (load + predicted_cost, worker))
            if splits > 1:
                partial_cursor += 1
        if splits > 1:
            merge_specs.append((request, request_partial_begin, splits, request_event))
            event_cursor += 1

    merge_rows: list[list[int]] = []
    for request, partial_begin, partial_count, event_id in merge_specs:
        load, worker = heapq.heappop(worker_heap)
        row = [
            _PLAN_MERGE,
            worker,
            request,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            partial_begin,
            partial_count,
            event_id,
        ]
        merge_rows.append(row)
        heapq.heappush(worker_heap, (load + 1, worker))

    rows = attention_rows + merge_rows
    num_direct_tasks = sum(row[0] == _PLAN_DIRECT for row in rows)
    num_partial_tasks = sum(row[0] == _PLAN_PARTIAL for row in rows)
    plan_records = torch.tensor(rows, dtype=torch.int32, device="cpu").contiguous()
    if runtime_plan_capacity is None:
        plan_record_capacity = len(rows)
        event_capacity = event_cursor
        partial_slot_capacity = partial_cursor
    else:
        plan_record_capacity, event_capacity, partial_slot_capacity = map(
            int, runtime_plan_capacity
        )
        if (
            plan_record_capacity < len(rows)
            or event_capacity < event_cursor
            or partial_slot_capacity < partial_cursor
        ):
            raise MKConfigError("runtime plan exceeds its fixed capacity")
    if runtime_auto_dataflow:
        name = f"verify_attention_welmv45_auto_typed_w{resolved_workers}"
        kernel_defs = (
            "mk::kernelv2::VerifyAttentionWelmv45AutoTypedKernel<"
            f"{resolved_workers}>"
        )
    elif attention_dataflow == _COREV2_DATAFLOW:
        name = f"verify_attention_welmv45_wgmma_corev2_w{resolved_workers}"
        kernel_defs = (
            f"mk::kernelv2::VerifyAttentionWelmv45WgmmaKernel<{resolved_workers}>"
        )
    elif attention_dataflow == _WARP_MMA_DATAFLOW:
        if partial_merge_mode == _LAST_ARRIVER_PARTIAL_MERGE:
            name = (
                "verify_attention_welmv45_warp_mma_last_arriver_w"
                f"{resolved_workers}"
            )
            kernel_defs = (
                "mk::kernelv2::VerifyAttentionWelmv45WarpMmaLastArriverKernel<"
                f"{resolved_workers}>"
            )
        else:
            name = f"verify_attention_welmv45_warp_mma_typed_w{resolved_workers}"
            kernel_defs = (
                "mk::kernelv2::VerifyAttentionWelmv45WarpMmaTypedKernel<"
                f"{resolved_workers}>"
            )
    else:
        use_static_persistent = _use_wgmma_static_persistent_executor(
            num_direct_tasks,
            num_partial_tasks,
            sum(block_counts),
            resolved_workers,
        )
        if use_static_persistent:
            name = (
                "verify_attention_welmv45_wgmma_optimized_persistent_w"
                f"{resolved_workers}"
            )
            kernel_defs = (
                "mk::kernelv2::VerifyAttentionWelmv45WgmmaOptimizedPersistentKernel<"
                f"{resolved_workers}>"
            )
        else:
            name = (
                "verify_attention_welmv45_wgmma_optimized_typed_w"
                f"{resolved_workers}"
            )
            kernel_defs = (
                "mk::kernelv2::VerifyAttentionWelmv45WgmmaOptimizedTypedKernel<"
                f"{resolved_workers}>"
            )
    base_args = (
        None,
        None,
        None,
        0,
        batch,
        None,
        None,
        None,
        plan_records,
        len(rows),
        event_cursor,
        partial_cursor,
        1.0 / math.sqrt(256.0),
        None,
        None,
        None,
        None,
        0,
        -1,
        plan_record_capacity,
        event_capacity,
        partial_slot_capacity,
    )
    prepared = prepare_kernel(
        schedule_policy=attention_dataflow,
        partial_merge_mode=partial_merge_mode,
        use_wgmma_static_persistent=_use_wgmma_static_persistent_executor(
            num_direct_tasks,
            num_partial_tasks,
            sum(block_counts),
            resolved_workers,
        ),
        worker_count=resolved_workers,
        name=name,
    )
    stream = int(torch.cuda.current_stream(plan_device).cuda_stream)
    workspace_size = prepared._library.workspace_size(
        base_args,
        device_id=plan_device.index,
        stream=stream,
        context=prepared._context,
    )
    if workspace_size <= 0:
        raise MKConfigError("verify attention workspace size must be positive")
    cpu_template = torch.empty(
        (workspace_size,), dtype=torch.uint8, device="cpu", pin_memory=True
    )
    template_args_list = list(base_args)
    template_args_list[15] = cpu_template
    template_args = tuple(template_args_list)
    prepared._library.init_cpu_workspace(
        template_args,
        device_id=plan_device.index,
        stream=stream,
        context=prepared._context,
    )
    if attention_dataflow == _WARP_MMA_DATAFLOW:
        cpu_template[:16].view(torch.int32)[3] = int(not dynamic_runtime_metadata)
    partial_m_padded = _align_up(
        partial_slot_capacity * _PARTIAL_ML_BYTES, 256
    )
    partial_scratch_size = (
        partial_slot_capacity * _PARTIAL_O_BYTES
        + partial_m_padded
        + partial_slot_capacity * _PARTIAL_ML_BYTES
    )
    # Graph capture uses a fixed-address D2D copy. Runtime exact replanning
    # alternates between two pinned CPU buffers and asynchronously refreshes
    # this stable GPU staging buffer before replay.
    gpu_workspace_template = (
        cpu_template.to(device=plan_device)
        if dynamic_runtime_metadata
        else None
    )
    cpu_replan_staging = ()
    cpp_replan_stager = None
    cpp_planner_state = None
    replan_meta = None
    if dynamic_runtime_metadata:
        second_cpu_template = torch.empty(
            (workspace_size,), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        second_cpu_template.copy_(cpu_template)
        cpu_replan_staging = (cpu_template, second_cpu_template)
        stager_args = (
            base_args[:14]
            + (gpu_workspace_template, cpu_template)
            + base_args[16:17]
            + (max_pages, window_size[0])
            + base_args[19:]
        )
        # Header + topology age + last block counts + capacity-clamped split
        # counts.  The C++ planner keeps a correct slowly varying split
        # topology across KV32 boundaries, while still rewriting exact causal
        # intervals every step and periodically refreshing the cost optimum.
        planner_state_size = (3 + 2 * batch) * (
            2 if runtime_auto_dataflow else 1
        )
        cpp_planner_state = torch.zeros(
            (planner_state_size,), dtype=torch.int32, device="cpu", pin_memory=True
        )
        replan_meta = torch.empty(
            (5,), dtype=torch.int32, device="cpu", pin_memory=True
        )
        cpp_replan_stager = prepared._library.create_full_plan_stager(
            cpu_replan_staging,
            gpu_workspace_template,
            workspace_size,
            stager_args,
            device_id=plan_device.index,
            stream=stream,
            context=prepared._context,
        )
    return VerifyAttentionWelmv45Plan(
        _prepared=prepared,
        _plan_records=plan_records,
        _cpu_workspace_template=cpu_template,
        _cpu_replan_staging=cpu_replan_staging,
        _cpp_replan_stager=cpp_replan_stager,
        _cpp_planner_state=cpp_planner_state,
        _replan_meta=replan_meta,
        _gpu_workspace_template=gpu_workspace_template,
        batch_size=batch,
        page_table_shape=(batch, max_pages),
        page_indices_count=indptr[-1],
        num_plan_records=len(rows),
        num_direct_tasks=num_direct_tasks,
        num_partial_tasks=num_partial_tasks,
        num_merge_tasks=sum(row[0] == _PLAN_MERGE for row in rows),
        num_events=event_cursor,
        num_partial_slots=partial_cursor,
        plan_record_capacity=plan_record_capacity,
        event_capacity=event_capacity,
        partial_slot_capacity=partial_slot_capacity,
        worker_count=resolved_workers,
        window_size=window_size,
        attention_dataflow=attention_dataflow,
        partial_merge_mode=partial_merge_mode,
        estimated_schedule_cost=estimated_schedule_cost,
        estimated_attention_fixed_units=schedule_features.attention_fixed_units,
        estimated_attention_block_units=(
            schedule_features.attention_block_units
        ),
        estimated_merge_waves=schedule_features.merge_waves,
        estimated_merge_fan_in_wave_sum=(
            schedule_features.merge_fan_in_wave_sum
        ),
        estimated_merge_head_parallel=schedule_features.merge_head_parallel,
        workspace_size=workspace_size,
        partial_scratch_size=partial_scratch_size,
        dynamic_runtime_metadata=dynamic_runtime_metadata,
        runtime_auto_dataflow=runtime_auto_dataflow,
        selected_attention_dataflow=(
            _WARP_MMA_DATAFLOW
            if runtime_auto_dataflow
            else attention_dataflow
        ),
        device=plan_device,
    )


def verify_attention_welmv45_replan(
    plan: VerifyAttentionWelmv45Plan,
    page_indptr_or_cache_seqlens: torch.Tensor | Sequence[int],
    cache_seqlens: torch.Tensor | Sequence[int] | None = None,
    cu_seqlens_q: torch.Tensor | Sequence[int] | None = None,
    *,
    cache_seqlen_offset: int = 0,
) -> float:
    """Refresh a graph plan with the C++ exact planner and async staging.

    The two-argument form ``replan(plan, cache_seqlens)`` is the allocation-free
    serving path.  The legacy four-argument form remains accepted; its
    page-indptr and query-indptr are fixed by the CUDA Graph page stride.
    """

    if not plan.dynamic_runtime_metadata:
        raise MKConfigError("replan requires a dynamic runtime metadata plan")
    runtime_lengths = (
        page_indptr_or_cache_seqlens
        if cache_seqlens is None
        else cache_seqlens
    )
    if not isinstance(runtime_lengths, torch.Tensor):
        # Compatibility for direct API users. SGLang always supplies its
        # existing CPU int32 metadata tensor and does not enter this branch.
        runtime_lengths = torch.tensor(runtime_lengths, dtype=torch.int32)
    if (
        runtime_lengths.device.type != "cpu"
        or runtime_lengths.dtype is not torch.int32
        or not runtime_lengths.is_contiguous()
        or runtime_lengths.shape != (plan.batch_size,)
    ):
        raise MKConfigError(
            "replan cache_seqlens must be contiguous CPU int32 [batch]"
        )
    if plan._gpu_workspace_template is None:
        raise MKConfigError("dynamic graph plan is missing GPU staging storage")
    if (
        len(plan._cpu_replan_staging) != 2
        or plan._cpp_replan_stager is None
        or plan._replan_meta is None
    ):
        raise MKConfigError("dynamic graph plan is missing double-buffered staging")
    stream = int(torch.cuda.current_stream(plan.device).cuda_stream)
    estimated_cost_q10 = plan._prepared._library.replan_and_stage(
        plan._cpp_replan_stager,
        runtime_lengths,
        plan.batch_size,
        int(cache_seqlen_offset),
        0,
        plan._cpp_planner_state,
        0 if plan._cpp_planner_state is None else plan._cpp_planner_state.numel(),
        plan._replan_meta,
        stream=stream,
    )
    if plan.runtime_auto_dataflow:
        plan.selected_attention_dataflow = (
            _WGMMA_DATAFLOW
            if int(plan._replan_meta[3]) < 0
            else _WARP_MMA_DATAFLOW
        )
    return float(estimated_cost_q10) / 1024.0


def verify_attention_welmv45_prepare(
    plan: VerifyAttentionWelmv45Plan,
    workspace: torch.Tensor,
) -> None:
    """Prepare a caller-owned workspace for one launch.

    Dynamic two-kernel plans never mutate their schedule workspace.  Their C++
    replanner already stages the exact schedule into a stable GPU buffer on
    the replay stream, so CUDA Graph kernels can consume that buffer directly.
    Avoiding a redundant D2D restore here removes one copy node per model
    layer.  Static plans retain the original restore semantics.
    """

    _validate_byte_buffer(
        workspace,
        "workspace",
        plan.workspace_size,
        plan.workspace_alignment,
        plan.device,
    )
    if not plan.dynamic_runtime_metadata:
        workspace[: plan.workspace_size].copy_(
            plan._cpu_workspace_template, non_blocking=False
        )
    plan._ready_workspace_ptrs.add(int(workspace.data_ptr()))


def verify_attention_welmv45_run(
    plan: VerifyAttentionWelmv45Plan,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_indices: torch.Tensor,
    sinks: torch.Tensor | None,
    output: torch.Tensor,
    workspace: torch.Tensor,
    partial_scratch: torch.Tensor | None,
    softmax_scale: float | None = None,
    runtime_cache_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Consume a prepared workspace and launch the selected executor.

    There is intentionally no runtime ``mask`` argument.  Causality is fixed
    by the linear four-token verify contract captured in the plan records.
    Dynamic plans require ``prepare`` on every launch to reset split/merge
    state; their GPU template makes that reset CUDA Graph-capturable.
    """

    _validate_byte_buffer(
        workspace,
        "workspace",
        plan.workspace_size,
        plan.workspace_alignment,
        plan.device,
    )
    workspace_ptr = int(workspace.data_ptr())
    if workspace_ptr not in plan._ready_workspace_ptrs:
        raise MKConfigError(
            "workspace is single-use; call prepare() before every run()"
        )
    if plan.dynamic_runtime_metadata:
        if (
            not isinstance(runtime_cache_seqlens, torch.Tensor)
            or runtime_cache_seqlens.device != plan.device
            or runtime_cache_seqlens.dtype is not torch.int32
            or not runtime_cache_seqlens.is_contiguous()
            or runtime_cache_seqlens.shape != (plan.batch_size,)
        ):
            raise MKConfigError(
                "dynamic runtime cache_seqlens must be contiguous CUDA int32 [batch]"
            )
    elif runtime_cache_seqlens is not None:
        raise MKConfigError(
            "runtime_cache_seqlens requires a dynamic runtime metadata plan"
        )
    _validate_persistent_runtime(
        plan, query, key_cache, value_cache, page_indices, sinks, output
    )
    if plan.partial_scratch_size > 0:
        if partial_scratch is None:
            raise MKConfigError("partial_scratch is required by this plan")
        _validate_byte_buffer(
            partial_scratch,
            "partial_scratch",
            plan.partial_scratch_size,
            plan.partial_scratch_alignment,
            plan.device,
        )
    elif partial_scratch is not None:
        _validate_byte_buffer(
            partial_scratch,
            "partial_scratch",
            0,
            plan.partial_scratch_alignment,
            plan.device,
        )
    scale = (
        1.0 / math.sqrt(float(query.shape[-1]))
        if softmax_scale is None
        else float(softmax_scale)
    )
    # Dynamic runtime-plan executors use separate attention and merge kernels;
    # their headers and task arrays are immutable during execution.  Reuse the
    # stable GPU staging allocation updated by replan_and_stage() instead of
    # copying it into the caller workspace once per layer inside CUDA Graph.
    runtime_workspace = (
        plan._gpu_workspace_template
        if plan.dynamic_runtime_metadata
        else workspace
    )
    if runtime_workspace is None:
        raise MKConfigError("dynamic graph plan is missing GPU staging storage")
    runtime_args = (
        query,
        key_cache,
        value_cache,
        int(key_cache.shape[0]),
        plan.batch_size,
        page_indices,
        sinks,
        output,
        plan._plan_records,
        plan.num_plan_records,
        plan.num_events,
        plan.num_partial_slots,
        scale,
        partial_scratch,
        runtime_workspace,
        plan._cpu_workspace_template,
        runtime_cache_seqlens,
        plan.page_table_shape[1] if plan.dynamic_runtime_metadata else 0,
        plan.window_size[0] if plan.dynamic_runtime_metadata else -1,
        plan.plan_record_capacity,
        plan.event_capacity,
        plan.partial_slot_capacity,
    )
    stream = int(torch.cuda.current_stream(plan.device).cuda_stream)
    plan._prepared._library.launch_prepared(
        runtime_args,
        device_id=plan.device.index,
        stream=stream,
        context=plan._prepared._context,
    )
    if not plan.dynamic_runtime_metadata:
        plan._ready_workspace_ptrs.remove(workspace_ptr)
    return output


def verify_attention_welmv45_mock(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    page_table: tuple[tuple[int, int], torch.Tensor, torch.Tensor],
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    sinks: torch.Tensor | None = None,
    max_seqlen_q: int,
    softmax_scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the verify-attention API scaffold.

    The native implementation is intentionally a no-op. When ``output`` is not
    supplied, Python zero-initializes it once so that the mock result is stable;
    repeated calls with a supplied output do not modify that tensor.  Like the
    production API, this scaffold accepts no external attention mask and does
    not model speculative-decode trees.
    """

    shape, page_indptr, page_indices = _validate_inputs(
        query,
        key_cache,
        value_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        sinks,
        max_seqlen_q,
        window_size,
    )
    if output is None:
        output = torch.zeros_like(query)
    else:
        _validate_output(query, output)

    scale = (
        1.0 / math.sqrt(float(query.shape[-1]))
        if softmax_scale is None
        else float(softmax_scale)
    )
    del shape, page_indptr, page_indices, scale
    raise MKConfigError("verify_attention mock path is removed in the k-dash package")


def verify_attention_welmv45_wgmma_direct(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    page_table: tuple[tuple[int, int], torch.Tensor, torch.Tensor],
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    sinks: torch.Tensor | None = None,
    max_seqlen_q: int = 4,
    softmax_scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the first fixed-shape ``WGMMAKV64N24`` DirectOutput path.

    This entry point intentionally exposes only the no-outer-split milestone.
    It already uses the locked KV64 grouping and exact SM90
    ``wgmma.m64n24k16`` QK/PV shapes; planner-owned outer splits and the
    persistent CoreV2 queue are layered on by the plan/prepare/run API.  Its
    four query tokens are an implicit linear causal draft; arbitrary masks and
    speculative-decode trees are unsupported.
    """

    shape, page_indptr, page_indices = _validate_inputs(
        query,
        key_cache,
        value_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        sinks,
        max_seqlen_q,
        window_size,
    )
    if max_seqlen_q != 4 or query.shape[0] != int(shape[0]) * 4:
        raise MKConfigError(
            "WGMMA direct requires exactly four query tokens per request"
        )
    if window_size not in ((-1, -1), (512, 0)):
        raise MKConfigError(
            "WGMMA direct supports only full context or window (512, 0)"
        )
    if output is None:
        output = torch.empty_like(query)
    else:
        _validate_output(query, output)

    scale = (
        1.0 / math.sqrt(float(query.shape[-1]))
        if softmax_scale is None
        else float(softmax_scale)
    )
    del shape, page_indptr, page_indices, scale, output
    raise MKConfigError(
        "verify_attention WGMMA direct entry is removed; use plan/prepare/run"
    )


def _validate_inputs(
    query,
    key_cache,
    value_cache,
    page_table,
    cache_seqlens,
    cu_seqlens_q,
    sinks,
    max_seqlen_q,
    window_size,
) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor]:
    tensors = {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise MKConfigError(f"{name} must be a torch.Tensor")
        if tensor.device.type != "cuda":
            raise MKConfigError(f"{name} must be on CUDA")
        if not tensor.is_contiguous():
            raise MKConfigError(f"{name} must be contiguous")
    if query.ndim != 3 or tuple(query.shape[1:]) != (6, 256):
        raise MKConfigError("query must have shape (total_q, 6, 256)")
    if query.dtype is not torch.bfloat16:
        raise MKConfigError("query must have dtype torch.bfloat16")
    expected_cache_tail = (16, 1, 256)
    for name, tensor in (("key_cache", key_cache), ("value_cache", value_cache)):
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != expected_cache_tail:
            raise MKConfigError(f"{name} must have shape (num_pages, 16, 1, 256)")
        if tensor.dtype is not query.dtype or tensor.device != query.device:
            raise MKConfigError(f"{name} must match query dtype and device")
    if key_cache.shape != value_cache.shape:
        raise MKConfigError("key_cache and value_cache must have identical shapes")
    if not isinstance(page_table, tuple) or len(page_table) != 3:
        raise MKConfigError("page_table must be (shape, indptr, indices)")
    shape, page_indptr, page_indices = page_table
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise MKConfigError("page_table shape must be a two-item tuple")
    for name, tensor in (("page_indptr", page_indptr), ("page_indices", page_indices)):
        if not isinstance(tensor, torch.Tensor) or tensor.device != query.device:
            raise MKConfigError(f"{name} must be a CUDA tensor on the query device")
        if tensor.dtype is not torch.int32 or not tensor.is_contiguous():
            raise MKConfigError(f"{name} must be contiguous torch.int32")
    batch = int(shape[0])
    if page_indptr.numel() != batch + 1 or cache_seqlens.numel() != batch:
        raise MKConfigError("page table and cache_seqlens batch sizes must match")
    if cu_seqlens_q.dtype is not torch.int32 or cache_seqlens.dtype is not torch.int32:
        raise MKConfigError("sequence metadata must have dtype torch.int32")
    if cu_seqlens_q.numel() != batch + 1:
        raise MKConfigError("cu_seqlens_q must have batch_size + 1 elements")
    if sinks is not None:
        if sinks.device != query.device or sinks.dtype is not query.dtype:
            raise MKConfigError("sinks must match query dtype and device")
        if tuple(sinks.shape) != (6,) or not sinks.is_contiguous():
            raise MKConfigError("sinks must be contiguous with shape (6,)")
    if not isinstance(max_seqlen_q, int) or max_seqlen_q <= 0:
        raise MKConfigError("max_seqlen_q must be a positive integer")
    if (
        not isinstance(window_size, tuple)
        or len(window_size) != 2
        or not all(isinstance(value, int) for value in window_size)
    ):
        raise MKConfigError("window_size must be a pair of integers")
    return (int(shape[0]), int(shape[1])), page_indptr, page_indices


def _validate_output(query: torch.Tensor, output: torch.Tensor) -> None:
    if not isinstance(output, torch.Tensor):
        raise MKConfigError("output must be a torch.Tensor")
    if output.shape != query.shape or output.dtype != query.dtype:
        raise MKConfigError("output must match query shape and dtype")
    if output.device != query.device or not output.is_contiguous():
        raise MKConfigError("output must be contiguous on the query device")


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _host_int32_values(value: torch.Tensor | Sequence[int], name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise MKConfigError(
                f"{name} must be host metadata; CUDA tensors are rejected"
            )
        if (
            value.dtype is not torch.int32
            or not value.is_contiguous()
            or value.ndim != 1
        ):
            raise MKConfigError(
                f"{name} must be a contiguous one-dimensional CPU int32 tensor"
            )
        return [int(item) for item in value.tolist()]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise MKConfigError(f"{name} must be a CPU int32 tensor or integer sequence")
    result: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise MKConfigError(f"{name} must contain only integers")
        if item < -(1 << 31) or item >= (1 << 31):
            raise MKConfigError(f"{name} contains a value outside int32 range")
        result.append(int(item))
    return result


def _normalize_cuda_device(device: torch.device | str | int) -> torch.device:
    if isinstance(device, int):
        result = torch.device("cuda", int(device))
    else:
        result = torch.device(device)
        if result.type == "cuda" and result.index is None:
            result = torch.device("cuda", torch.cuda.current_device())
    if result.type != "cuda" or result.index is None:
        raise MKConfigError("device must identify a CUDA device")
    return result


def _validate_byte_buffer(
    tensor: torch.Tensor,
    name: str,
    required_bytes: int,
    alignment: int,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise MKConfigError(f"{name} must be a torch.Tensor")
    if (
        tensor.device != device
        or tensor.dtype is not torch.uint8
        or not tensor.is_contiguous()
    ):
        raise MKConfigError(f"{name} must be contiguous CUDA uint8 on the plan device")
    if tensor.numel() < required_bytes:
        raise MKConfigError(f"{name} capacity is smaller than the plan requirement")
    if int(tensor.data_ptr()) % alignment != 0:
        raise MKConfigError(
            f"{name} pointer does not satisfy {alignment}-byte alignment"
        )


def _validate_persistent_runtime(
    plan: VerifyAttentionWelmv45Plan,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    page_indices: torch.Tensor,
    sinks: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    if query.device != plan.device:
        raise MKConfigError("query device does not match plan")
    if (
        query.dtype is not torch.bfloat16
        or not query.is_contiguous()
        or tuple(query.shape) != (plan.batch_size * 4, 6, 256)
    ):
        raise MKConfigError("query must be contiguous BF16 [batch * 4, 6, 256]")
    expected_tail = (16, 1, 256)
    for name, cache in (("key_cache", key_cache), ("value_cache", value_cache)):
        if (
            cache.device != plan.device
            or cache.dtype is not torch.bfloat16
            or not cache.is_contiguous()
            or cache.ndim != 4
            or tuple(cache.shape[1:]) != expected_tail
        ):
            raise MKConfigError(
                f"{name} must be contiguous BF16 [num_pages, 16, 1, 256]"
            )
    if key_cache.shape != value_cache.shape:
        raise MKConfigError("key_cache and value_cache shapes must match")
    if (
        page_indices.device != plan.device
        or page_indices.dtype is not torch.int32
        or not page_indices.is_contiguous()
        or page_indices.ndim != 1
        or page_indices.numel() < plan.page_indices_count
    ):
        raise MKConfigError(
            "page_indices must be a sufficiently large contiguous CUDA int32 tensor"
        )
    if sinks is not None and (
        sinks.device != plan.device
        or sinks.dtype is not torch.bfloat16
        or not sinks.is_contiguous()
        or tuple(sinks.shape) != (6,)
    ):
        raise MKConfigError("sinks must be contiguous BF16 [6] on the plan device")
    _validate_output(query, output)
