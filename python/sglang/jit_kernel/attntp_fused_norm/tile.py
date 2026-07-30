"""Replicated AttnTP FP32 tile pipeline runners."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import torch
from tvm_ffi import Module

from sglang.jit_kernel.attntp_fused_norm.ipc import (
    AttnTPNormSpec,
    NormInternalPrecision,
    OutputMode,
    decode_output_capacity,
)
from sglang.jit_kernel.attntp_fused_norm.resources import (
    AttnTPSymmetricMemoryResource,
)
from sglang.jit_kernel.attntp_fused_norm.symm import (
    _validate_device_scalar,
    _validate_kernel_tuning,
    _validate_run_tensors,
    _validate_true_symm_spec,
)
from sglang.jit_kernel.attntp_fused_norm.tile_tuning import (
    ConsumerCohorts,
    TilePipelineShape,
    WeightPlacement,
    validate_shape_for_hidden_size,
)
from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

_ALIGNMENT = 128
_SUPPORTED_PHASES = ("prefill", "decode")
_SUPPORTED_ROWS_PER_TILE = (1, 2, 4, 8, 16, 32)
_SUPPORTED_RING_STAGES = (2, 4, 8, 16, 32, 64, 128)
_SUPPORTED_LAUNCH_BOUNDS_MIN_BLOCKS = (0, 1, 2, 3, 4)
_CONTROL_HEADER_BYTES = 128
_TICKET_BYTES = torch.int64.itemsize


def _align_bytes(value: int) -> int:
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


@dataclass(frozen=True)
class TilePipelineArenaLayout:
    input_bytes: int
    reduced_offset_bytes: int
    reduced_bytes: int
    control_offset_bytes: int
    input_ready_offset_bytes: int
    group_ready_offset_bytes: int
    kernel_done_offset_bytes: int
    completion_counter_offset_bytes: int
    ready_offset_bytes: int
    consumed_offset_bytes: int
    consumed_owner_stride_bytes: int
    control_bytes: int
    total_bytes: int


@dataclass(frozen=True, order=True)
class TilePipelineControlKey:
    capacity: int
    rows_per_tile: int
    ring_stages: int

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or self.capacity <= 0:
            raise ValueError("tile control capacity must be positive")
        if self.rows_per_tile not in _SUPPORTED_ROWS_PER_TILE:
            raise ValueError(f"rows_per_tile must be one of {_SUPPORTED_ROWS_PER_TILE}")
        if self.ring_stages not in _SUPPORTED_RING_STAGES:
            raise ValueError(f"ring_stages must be one of {_SUPPORTED_RING_STAGES}")


@dataclass(frozen=True)
class SharedTilePipelineArenaLayout:
    input_bytes: int
    layouts: Mapping[TilePipelineControlKey, TilePipelineArenaLayout]
    total_bytes: int

    def layout_for(self, key: TilePipelineControlKey) -> TilePipelineArenaLayout:
        try:
            return self.layouts[key]
        except KeyError as error:
            raise RuntimeError(
                "tile pipeline control layout is missing for "
                f"capacity={key.capacity}, rows_per_tile={key.rows_per_tile}, "
                f"ring_stages={key.ring_stages}"
            ) from error


def validate_tile_pipeline_spec(spec: AttnTPNormSpec) -> None:
    _validate_true_symm_spec(spec)
    if spec.output_mode is not OutputMode.REPLICATED:
        raise ValueError("tile pipeline requires replicated output mode")
    if spec.internal_precision is not NormInternalPrecision.FULL_FP32:
        raise ValueError("tile pipeline requires full_fp32 internal precision")


def _validate_protocol_tuning(
    *,
    hidden_size: int,
    rows_per_tile: int,
    ring_stages: int,
    block_size: int,
    signal_backoff: int,
    launch_bounds_min_blocks: int,
    consumer_cohorts: ConsumerCohorts,
    weight_placement: WeightPlacement,
    blocks_per_sm: int | None,
    producer_blocks_per_sm: int | None,
) -> None:
    _validate_kernel_tuning(
        hidden_size=hidden_size,
        block_size=block_size,
        signal_backoff=signal_backoff,
        blocks_per_sm=blocks_per_sm,
    )
    if type(rows_per_tile) is not int or rows_per_tile not in _SUPPORTED_ROWS_PER_TILE:
        raise ValueError(f"rows_per_tile must be one of {_SUPPORTED_ROWS_PER_TILE}")
    if type(ring_stages) is not int or ring_stages not in _SUPPORTED_RING_STAGES:
        raise ValueError(f"ring_stages must be one of {_SUPPORTED_RING_STAGES}")
    if (
        type(launch_bounds_min_blocks) is not int
        or launch_bounds_min_blocks not in _SUPPORTED_LAUNCH_BOUNDS_MIN_BLOCKS
    ):
        raise ValueError(
            "launch_bounds_min_blocks must be one of "
            f"{_SUPPORTED_LAUNCH_BOUNDS_MIN_BLOCKS}"
        )
    if producer_blocks_per_sm is not None and (
        type(producer_blocks_per_sm) is not int or producer_blocks_per_sm <= 0
    ):
        raise ValueError("producer_blocks_per_sm must be positive or None")
    if (
        blocks_per_sm is not None
        and producer_blocks_per_sm is not None
        and producer_blocks_per_sm >= blocks_per_sm
    ):
        raise ValueError(
            "producer_blocks_per_sm must leave at least one consumer block per SM"
        )
    validate_shape_for_hidden_size(
        TilePipelineShape(
            rows_per_tile=rows_per_tile,
            ring_stages=ring_stages,
            block_size=block_size,
            signal_backoff=signal_backoff,
            launch_bounds_min_blocks=launch_bounds_min_blocks,
            consumer_cohorts=consumer_cohorts,
            weight_placement=weight_placement,
        ),
        hidden_size,
    )


def tile_pipeline_arena_layout(
    *,
    capacity: int,
    spec: AttnTPNormSpec,
    rows_per_tile: int,
    ring_stages: int,
) -> TilePipelineArenaLayout:
    validate_tile_pipeline_spec(spec)
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("tile pipeline capacity must be a positive integer")
    if type(rows_per_tile) is not int or rows_per_tile not in _SUPPORTED_ROWS_PER_TILE:
        raise ValueError(f"rows_per_tile must be one of {_SUPPORTED_ROWS_PER_TILE}")
    if type(ring_stages) is not int or ring_stages not in _SUPPORTED_RING_STAGES:
        raise ValueError(f"ring_stages must be one of {_SUPPORTED_RING_STAGES}")

    input_bytes = capacity * spec.hidden_size * torch.bfloat16.itemsize
    reduced_offset_bytes = _align_bytes(input_bytes)
    reduced_bytes = (
        rows_per_tile * ring_stages * spec.hidden_size * torch.float32.itemsize
    )
    control_offset_bytes = _align_bytes(reduced_offset_bytes + reduced_bytes)

    input_ready_offset_bytes = control_offset_bytes
    group_ready_offset_bytes = input_ready_offset_bytes + _TICKET_BYTES
    kernel_done_offset_bytes = group_ready_offset_bytes + _TICKET_BYTES
    completion_counter_offset_bytes = kernel_done_offset_bytes + _TICKET_BYTES
    ready_offset_bytes = control_offset_bytes + _CONTROL_HEADER_BYTES
    ready_bytes = _align_bytes(ring_stages * _TICKET_BYTES)
    consumed_offset_bytes = ready_offset_bytes + ready_bytes
    consumed_owner_stride_bytes = _align_bytes(ring_stages * _TICKET_BYTES)
    consumed_bytes = spec.attn_tp_size * consumed_owner_stride_bytes
    control_bytes = consumed_offset_bytes - control_offset_bytes + consumed_bytes
    total_bytes = _align_bytes(control_offset_bytes + control_bytes)

    return TilePipelineArenaLayout(
        input_bytes=input_bytes,
        reduced_offset_bytes=reduced_offset_bytes,
        reduced_bytes=reduced_bytes,
        control_offset_bytes=control_offset_bytes,
        input_ready_offset_bytes=input_ready_offset_bytes,
        group_ready_offset_bytes=group_ready_offset_bytes,
        kernel_done_offset_bytes=kernel_done_offset_bytes,
        completion_counter_offset_bytes=completion_counter_offset_bytes,
        ready_offset_bytes=ready_offset_bytes,
        consumed_offset_bytes=consumed_offset_bytes,
        consumed_owner_stride_bytes=consumed_owner_stride_bytes,
        control_bytes=control_bytes,
        total_bytes=total_bytes,
    )


def shared_tile_pipeline_arena_layout(
    *,
    capacity: int,
    spec: AttnTPNormSpec,
    control_keys: Iterable[TilePipelineControlKey],
    reserved_bytes_after_input: int = 0,
) -> SharedTilePipelineArenaLayout:
    validate_tile_pipeline_spec(spec)
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("shared tile pipeline capacity must be positive")
    keys = tuple(control_keys)
    if not keys:
        raise ValueError("at least one tile pipeline control layout is required")
    if len(keys) != len(set(keys)):
        raise ValueError("tile pipeline control keys must be unique")
    if any(key.capacity > capacity for key in keys):
        raise ValueError("tile control capacity exceeds shared payload capacity")
    if type(reserved_bytes_after_input) is not int or reserved_bytes_after_input < 0:
        raise ValueError("reserved bytes after input must be non-negative")

    input_bytes = capacity * spec.hidden_size * torch.bfloat16.itemsize
    cursor = _align_bytes(input_bytes + reserved_bytes_after_input)
    layouts = {}
    for key in sorted(keys):
        local = tile_pipeline_arena_layout(
            capacity=key.capacity,
            spec=spec,
            rows_per_tile=key.rows_per_tile,
            ring_stages=key.ring_stages,
        )
        translation = cursor - local.reduced_offset_bytes
        if translation < 0:
            raise RuntimeError("shared tile control layout overlaps input payload")
        translated = TilePipelineArenaLayout(
            input_bytes=local.input_bytes,
            reduced_offset_bytes=local.reduced_offset_bytes + translation,
            reduced_bytes=local.reduced_bytes,
            control_offset_bytes=local.control_offset_bytes + translation,
            input_ready_offset_bytes=(local.input_ready_offset_bytes + translation),
            group_ready_offset_bytes=(local.group_ready_offset_bytes + translation),
            kernel_done_offset_bytes=(local.kernel_done_offset_bytes + translation),
            completion_counter_offset_bytes=(
                local.completion_counter_offset_bytes + translation
            ),
            ready_offset_bytes=local.ready_offset_bytes + translation,
            consumed_offset_bytes=local.consumed_offset_bytes + translation,
            consumed_owner_stride_bytes=local.consumed_owner_stride_bytes,
            control_bytes=local.control_bytes,
            total_bytes=local.total_bytes + translation,
        )
        layouts[key] = translated
        cursor = _align_bytes(translated.total_bytes)

    return SharedTilePipelineArenaLayout(
        input_bytes=input_bytes,
        layouts=MappingProxyType(layouts),
        total_bytes=cursor,
    )


@cache_once
def _jit_attntp_tile_pipeline_module(
    attn_tp_size: int,
    hidden_size: int,
    phase: str,
    rows_per_tile: int,
    ring_stages: int,
    block_size: int,
    signal_backoff: int,
    launch_bounds_min_blocks: int,
    consumer_cohorts: ConsumerCohorts,
    weight_placement: WeightPlacement,
) -> Module:
    if phase not in _SUPPORTED_PHASES:
        raise ValueError(f"phase must be one of {_SUPPORTED_PHASES}")
    if launch_bounds_min_blocks not in _SUPPORTED_LAUNCH_BOUNDS_MIN_BLOCKS:
        raise ValueError(
            "launch_bounds_min_blocks must be one of "
            f"{_SUPPORTED_LAUNCH_BOUNDS_MIN_BLOCKS}"
        )
    validate_shape_for_hidden_size(
        TilePipelineShape(
            rows_per_tile=rows_per_tile,
            ring_stages=ring_stages,
            block_size=block_size,
            signal_backoff=signal_backoff,
            launch_bounds_min_blocks=launch_bounds_min_blocks,
            consumer_cohorts=consumer_cohorts,
            weight_placement=weight_placement,
        ),
        hidden_size,
    )
    args = make_cpp_args(
        torch.bfloat16,
        attn_tp_size,
        hidden_size,
        phase == "decode",
        rows_per_tile,
        ring_stages,
        block_size,
        signal_backoff,
        launch_bounds_min_blocks,
        consumer_cohorts.value,
        weight_placement is WeightPlacement.SHARED,
    )
    class_name = f"FusedAttnTPReplicatedTilePipeline<{args}>"
    return load_jit(
        f"{phase}_attntp_replicated_tile_pipeline",
        *args,
        cuda_files=[
            "distributed/attntp_fused_norm/attntp_replicated_tile_pipeline.cuh"
        ],
        cuda_wrappers=[
            ("fused_tile_pipeline", f"{class_name}::run"),
            ("get_max_occupancy", f"{class_name}::get_max_occupancy"),
        ],
        extra_cuda_cflags=[
            f"-DSGL_ATTNTP_TILE_BLOCK_SIZE={block_size}",
            (
                "-DSGL_ATTNTP_TILE_LAUNCH_BOUNDS_MIN_BLOCKS="
                f"{launch_bounds_min_blocks}"
            ),
        ],
    )


class _AttnTPReplicatedTilePipelineRunnerBase:
    def _initialize_tile_pipeline(
        self,
        *,
        group,
        device: torch.device,
        spec: AttnTPNormSpec,
        rows: int,
        phase: str,
        rows_per_tile: int,
        ring_stages: int,
        block_size: int,
        signal_backoff: int,
        launch_bounds_min_blocks: int,
        consumer_cohorts: ConsumerCohorts,
        weight_placement: WeightPlacement,
        blocks_per_sm: int | None,
        producer_blocks_per_sm: int | None,
    ) -> None:
        arena_layout = tile_pipeline_arena_layout(
            capacity=rows,
            spec=spec,
            rows_per_tile=rows_per_tile,
            ring_stages=ring_stages,
        )
        resource = AttnTPSymmetricMemoryResource(
            group=group,
            device=device,
            attn_tp_size=spec.attn_tp_size,
            hidden_size=spec.hidden_size,
            capacity=rows,
            total_bytes=arena_layout.total_bytes,
        )
        try:
            self._initialize_tile_pipeline_from_resource(
                resource=resource,
                spec=spec,
                rows=rows,
                phase=phase,
                arena_layout=arena_layout,
                rows_per_tile=rows_per_tile,
                ring_stages=ring_stages,
                block_size=block_size,
                signal_backoff=signal_backoff,
                launch_bounds_min_blocks=launch_bounds_min_blocks,
                consumer_cohorts=consumer_cohorts,
                weight_placement=weight_placement,
                blocks_per_sm=blocks_per_sm,
                producer_blocks_per_sm=producer_blocks_per_sm,
                owns_resource=True,
            )
        except Exception:
            resource.close()
            raise

    def _initialize_tile_pipeline_from_resource(
        self,
        *,
        resource,
        spec: AttnTPNormSpec,
        rows: int,
        phase: str,
        arena_layout: TilePipelineArenaLayout,
        rows_per_tile: int,
        ring_stages: int,
        block_size: int,
        signal_backoff: int,
        launch_bounds_min_blocks: int,
        consumer_cohorts: ConsumerCohorts,
        weight_placement: WeightPlacement,
        blocks_per_sm: int | None,
        producer_blocks_per_sm: int | None,
        owns_resource: bool,
    ) -> None:
        validate_tile_pipeline_spec(spec)
        if type(rows) is not int or rows <= 0:
            raise ValueError("tile pipeline rows must be a positive integer")
        if phase not in _SUPPORTED_PHASES:
            raise ValueError(f"phase must be one of {_SUPPORTED_PHASES}")
        _validate_protocol_tuning(
            hidden_size=spec.hidden_size,
            rows_per_tile=rows_per_tile,
            ring_stages=ring_stages,
            block_size=block_size,
            signal_backoff=signal_backoff,
            launch_bounds_min_blocks=launch_bounds_min_blocks,
            consumer_cohorts=consumer_cohorts,
            weight_placement=weight_placement,
            blocks_per_sm=blocks_per_sm,
            producer_blocks_per_sm=producer_blocks_per_sm,
        )

        self.phase = phase
        self.rows = rows
        self.capacity = rows
        self.spec = spec
        self.rows_per_tile = rows_per_tile
        self.ring_stages = ring_stages
        self.block_size = block_size
        self.signal_backoff = signal_backoff
        self.launch_bounds_min_blocks = launch_bounds_min_blocks
        self.consumer_cohorts = consumer_cohorts
        self.weight_placement = weight_placement
        self.blocks_per_sm = blocks_per_sm
        self.producer_blocks_per_sm = producer_blocks_per_sm
        self.device = torch.device(resource.device)
        if self.device.type != "cuda":
            raise ValueError("tile pipeline requires a CUDA device")
        if not isinstance(arena_layout, TilePipelineArenaLayout):
            raise ValueError("arena_layout must be a TilePipelineArenaLayout")
        expected_input_bytes = rows * spec.hidden_size * torch.bfloat16.itemsize
        if arena_layout.input_bytes != expected_input_bytes:
            raise ValueError("tile pipeline arena input size does not match rows")
        expected_reduced_bytes = (
            rows_per_tile * ring_stages * spec.hidden_size * torch.float32.itemsize
        )
        if arena_layout.reduced_bytes != expected_reduced_bytes:
            raise ValueError("tile pipeline arena scratch size does not match tuning")
        shared_input_bytes = (
            resource.capacity * spec.hidden_size * torch.bfloat16.itemsize
        )
        if arena_layout.reduced_offset_bytes < shared_input_bytes:
            raise ValueError("tile pipeline control state overlaps shared payload")
        if arena_layout.total_bytes % torch.bfloat16.itemsize != 0:
            raise RuntimeError("tile pipeline arena is not BF16-storage aligned")
        resource.require(
            attn_tp_size=spec.attn_tp_size,
            hidden_size=spec.hidden_size,
            capacity=rows,
            required_bytes=arena_layout.total_bytes,
        )
        self.arena_layout = arena_layout
        self.rank = resource.rank
        self._resource = resource
        self._owns_resource = owns_resource
        self._input = resource.input_view_for(rows)
        self._storage = getattr(resource, "storage", None)
        self._handle = getattr(resource, "_handle", None)
        self._pointer_table = resource.pointer_table
        self._closed = False

    @property
    def input_view(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("tile pipeline runner is closed")
        return self._input

    def capture(self):
        if self._closed:
            raise RuntimeError("tile pipeline runner is closed")
        return self._resource.capture()

    def _run_tile_pipeline_out(
        self,
        *,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        output: torch.Tensor,
        residual_out: torch.Tensor,
        actual_rows: torch.Tensor | None,
        owner_start: torch.Tensor | None,
        o_norm_eps: float,
        post_norm_eps: float,
    ) -> None:
        if self._closed:
            raise RuntimeError("tile pipeline runner is closed")
        runtime_rows = residual.shape[0] if residual.ndim == 2 else 0
        if runtime_rows <= 0 or runtime_rows > self.rows:
            raise ValueError(
                "tile pipeline runtime rows must be in "
                f"[1, {self.rows}], got {runtime_rows}"
            )
        input_view = self._input[:runtime_rows]
        _validate_run_tensors(
            input_view=input_view,
            residual=residual,
            o_norm_weight=o_norm_weight,
            post_norm_weight=post_norm_weight,
            output=output,
            residual_out=residual_out,
            expected_output_rows=runtime_rows,
            device=self.device,
        )
        if self.phase == "decode":
            if actual_rows is not None or owner_start is not None:
                raise ValueError(
                    "Decode tile pipeline does not accept Prefill metadata"
                )
        else:
            if actual_rows is None or owner_start is None:
                raise ValueError(
                    "Prefill tile pipeline requires actual_rows and owner_start"
                )
            _validate_device_scalar(actual_rows, "actual_rows", self.device)
            _validate_device_scalar(owner_start, "owner_start", self.device)
        module = _jit_attntp_tile_pipeline_module(
            self.spec.attn_tp_size,
            self.spec.hidden_size,
            self.phase,
            self.rows_per_tile,
            self.ring_stages,
            self.block_size,
            self.signal_backoff,
            self.launch_bounds_min_blocks,
            self.consumer_cohorts,
            self.weight_placement,
        )
        layout = self.arena_layout
        module.fused_tile_pipeline(
            input_view,
            self._pointer_table,
            layout.reduced_offset_bytes,
            layout.input_ready_offset_bytes,
            layout.group_ready_offset_bytes,
            layout.kernel_done_offset_bytes,
            layout.completion_counter_offset_bytes,
            layout.ready_offset_bytes,
            layout.consumed_offset_bytes,
            layout.consumed_owner_stride_bytes,
            residual,
            o_norm_weight,
            post_norm_weight,
            output,
            residual_out,
            actual_rows,
            owner_start,
            self.rank,
            0 if self.blocks_per_sm is None else self.blocks_per_sm,
            (0 if self.producer_blocks_per_sm is None else self.producer_blocks_per_sm),
            float(o_norm_eps),
            float(post_norm_eps),
        )

    def get_max_occupancy(self) -> int:
        module = _jit_attntp_tile_pipeline_module(
            self.spec.attn_tp_size,
            self.spec.hidden_size,
            self.phase,
            self.rows_per_tile,
            self.ring_stages,
            self.block_size,
            self.signal_backoff,
            self.launch_bounds_min_blocks,
            self.consumer_cohorts,
            self.weight_placement,
        )
        return int(module.get_max_occupancy())

    def close(self) -> None:
        if self._closed:
            return
        resource = getattr(self, "_resource", None)
        owns_resource = getattr(self, "_owns_resource", False)
        if resource is not None and owns_resource:
            resource.close()
        elif resource is None:
            torch.cuda.synchronize(self.device)
        self._input = None
        self._storage = None
        self._handle = None
        self._pointer_table = 0
        self._resource = None
        self._closed = True


class PrefillAttnTPReplicatedTilePipelineRunner(
    _AttnTPReplicatedTilePipelineRunnerBase
):
    def __init__(
        self,
        *,
        group,
        device: torch.device,
        spec: AttnTPNormSpec,
        capacity: int,
        rows_per_tile: int = 4,
        ring_stages: int = 64,
        block_size: int = 256,
        signal_backoff: int = 64,
        launch_bounds_min_blocks: int = 0,
        consumer_cohorts: ConsumerCohorts = ConsumerCohorts.ONE,
        weight_placement: WeightPlacement = WeightPlacement.REGISTER,
        blocks_per_sm: int | None = None,
        producer_blocks_per_sm: int | None = None,
    ) -> None:
        self._initialize_tile_pipeline(
            group=group,
            device=device,
            spec=spec,
            rows=capacity,
            phase="prefill",
            rows_per_tile=rows_per_tile,
            ring_stages=ring_stages,
            block_size=block_size,
            signal_backoff=signal_backoff,
            launch_bounds_min_blocks=launch_bounds_min_blocks,
            consumer_cohorts=consumer_cohorts,
            weight_placement=weight_placement,
            blocks_per_sm=blocks_per_sm,
            producer_blocks_per_sm=producer_blocks_per_sm,
        )

    @classmethod
    def from_resource(
        cls,
        *,
        resource,
        spec: AttnTPNormSpec,
        capacity: int,
        arena_layout: TilePipelineArenaLayout,
        rows_per_tile: int = 4,
        ring_stages: int = 64,
        block_size: int = 256,
        signal_backoff: int = 64,
        launch_bounds_min_blocks: int = 0,
        consumer_cohorts: ConsumerCohorts = ConsumerCohorts.ONE,
        weight_placement: WeightPlacement = WeightPlacement.REGISTER,
        blocks_per_sm: int | None = None,
        producer_blocks_per_sm: int | None = None,
    ) -> PrefillAttnTPReplicatedTilePipelineRunner:
        runner = cls.__new__(cls)
        runner._initialize_tile_pipeline_from_resource(
            resource=resource,
            spec=spec,
            rows=capacity,
            phase="prefill",
            arena_layout=arena_layout,
            rows_per_tile=rows_per_tile,
            ring_stages=ring_stages,
            block_size=block_size,
            signal_backoff=signal_backoff,
            launch_bounds_min_blocks=launch_bounds_min_blocks,
            consumer_cohorts=consumer_cohorts,
            weight_placement=weight_placement,
            blocks_per_sm=blocks_per_sm,
            producer_blocks_per_sm=producer_blocks_per_sm,
            owns_resource=False,
        )
        return runner

    def run_out(
        self,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        output: torch.Tensor,
        residual_out: torch.Tensor,
        actual_rows: torch.Tensor,
        owner_start: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
    ) -> None:
        self._run_tile_pipeline_out(
            residual=residual,
            o_norm_weight=o_norm_weight,
            post_norm_weight=post_norm_weight,
            output=output,
            residual_out=residual_out,
            actual_rows=actual_rows,
            owner_start=owner_start,
            o_norm_eps=o_norm_eps,
            post_norm_eps=post_norm_eps,
        )


class DecodeAttnTPReplicatedTilePipelineRunner(_AttnTPReplicatedTilePipelineRunnerBase):
    def __init__(
        self,
        *,
        group,
        device: torch.device,
        spec: AttnTPNormSpec,
        rows: int,
        rows_per_tile: int = 1,
        ring_stages: int = 4,
        block_size: int = 256,
        signal_backoff: int = 64,
        launch_bounds_min_blocks: int = 0,
        consumer_cohorts: ConsumerCohorts = ConsumerCohorts.ONE,
        weight_placement: WeightPlacement = WeightPlacement.REGISTER,
        blocks_per_sm: int | None = None,
        producer_blocks_per_sm: int | None = None,
    ) -> None:
        decode_output_capacity(rows, spec)
        self._initialize_tile_pipeline(
            group=group,
            device=device,
            spec=spec,
            rows=rows,
            phase="decode",
            rows_per_tile=rows_per_tile,
            ring_stages=ring_stages,
            block_size=block_size,
            signal_backoff=signal_backoff,
            launch_bounds_min_blocks=launch_bounds_min_blocks,
            consumer_cohorts=consumer_cohorts,
            weight_placement=weight_placement,
            blocks_per_sm=blocks_per_sm,
            producer_blocks_per_sm=producer_blocks_per_sm,
        )

    @classmethod
    def from_resource(
        cls,
        *,
        resource,
        spec: AttnTPNormSpec,
        rows: int,
        arena_layout: TilePipelineArenaLayout,
        rows_per_tile: int = 1,
        ring_stages: int = 4,
        block_size: int = 256,
        signal_backoff: int = 64,
        launch_bounds_min_blocks: int = 0,
        consumer_cohorts: ConsumerCohorts = ConsumerCohorts.ONE,
        weight_placement: WeightPlacement = WeightPlacement.REGISTER,
        blocks_per_sm: int | None = None,
        producer_blocks_per_sm: int | None = None,
    ) -> DecodeAttnTPReplicatedTilePipelineRunner:
        decode_output_capacity(rows, spec)
        runner = cls.__new__(cls)
        runner._initialize_tile_pipeline_from_resource(
            resource=resource,
            spec=spec,
            rows=rows,
            phase="decode",
            arena_layout=arena_layout,
            rows_per_tile=rows_per_tile,
            ring_stages=ring_stages,
            block_size=block_size,
            signal_backoff=signal_backoff,
            launch_bounds_min_blocks=launch_bounds_min_blocks,
            consumer_cohorts=consumer_cohorts,
            weight_placement=weight_placement,
            blocks_per_sm=blocks_per_sm,
            producer_blocks_per_sm=producer_blocks_per_sm,
            owns_resource=False,
        )
        return runner

    def run_out(
        self,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        output: torch.Tensor,
        residual_out: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
    ) -> None:
        self._run_tile_pipeline_out(
            residual=residual,
            o_norm_weight=o_norm_weight,
            post_norm_weight=post_norm_weight,
            output=output,
            residual_out=residual_out,
            actual_rows=None,
            owner_start=None,
            o_norm_eps=o_norm_eps,
            post_norm_eps=post_norm_eps,
        )
