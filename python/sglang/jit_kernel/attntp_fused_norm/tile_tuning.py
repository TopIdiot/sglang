"""Reusable configuration model for AttnTP tile-pipeline autotuning."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable

TUNING_SCHEMA_VERSION = 4
AUTOTUNE_POLICY_VERSION = 1
SUPPORTED_PHASES = ("prefill", "decode")
SUPPORTED_HIDDEN_SIZES = (2048, 4096)
SUPPORTED_ROWS_PER_TILE = (1, 2, 4, 8, 16, 32)
SUPPORTED_RING_STAGES = (2, 4, 8, 16, 32, 64, 128)
SUPPORTED_BLOCK_SIZES = (128, 256, 512)
SUPPORTED_LAUNCH_BOUNDS_MIN_BLOCKS = (0, 1, 2, 3, 4)


class ConsumerCohorts(int, Enum):
    ONE = 1
    TWO = 2
    FOUR = 4


class WeightPlacement(str, Enum):
    REGISTER = "register_weight"
    SHARED = "shared_weight"


def _require_plain_int(name: str, value: int, *, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, order=True)
class TilePipelineShape:
    rows_per_tile: int
    ring_stages: int
    block_size: int
    signal_backoff: int
    launch_bounds_min_blocks: int = 0
    consumer_cohorts: ConsumerCohorts = ConsumerCohorts.ONE
    weight_placement: WeightPlacement = WeightPlacement.REGISTER

    def __post_init__(self) -> None:
        if self.rows_per_tile not in SUPPORTED_ROWS_PER_TILE:
            raise ValueError(f"rows_per_tile must be one of {SUPPORTED_ROWS_PER_TILE}")
        if self.ring_stages not in SUPPORTED_RING_STAGES:
            raise ValueError(f"ring_stages must be one of {SUPPORTED_RING_STAGES}")
        if self.block_size not in SUPPORTED_BLOCK_SIZES:
            raise ValueError(f"block_size must be one of {SUPPORTED_BLOCK_SIZES}")
        _require_plain_int("signal_backoff", self.signal_backoff)
        if self.launch_bounds_min_blocks not in SUPPORTED_LAUNCH_BOUNDS_MIN_BLOCKS:
            raise ValueError(
                "launch_bounds_min_blocks must be one of "
                f"{SUPPORTED_LAUNCH_BOUNDS_MIN_BLOCKS}"
            )
        if not isinstance(self.consumer_cohorts, ConsumerCohorts):
            raise ValueError("consumer_cohorts must be a ConsumerCohorts")
        if not isinstance(self.weight_placement, WeightPlacement):
            raise ValueError("weight_placement must be a WeightPlacement")

    def to_dict(self) -> dict[str, int | str]:
        payload = asdict(self)
        payload["consumer_cohorts"] = self.consumer_cohorts.value
        payload["consumer_threads"] = self.block_size // self.consumer_cohorts.value
        payload["weight_placement"] = self.weight_placement.value
        return payload


@dataclass(frozen=True, order=True)
class TilePipelineSchedule:
    blocks_per_sm: int
    producer_blocks_per_sm: int

    def __post_init__(self) -> None:
        _require_plain_int("blocks_per_sm", self.blocks_per_sm, minimum=2)
        _require_plain_int(
            "producer_blocks_per_sm",
            self.producer_blocks_per_sm,
        )
        if self.producer_blocks_per_sm >= self.blocks_per_sm:
            raise ValueError(
                "producer_blocks_per_sm must leave at least one consumer block"
            )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, order=True)
class TilePipelineTuning:
    shape: TilePipelineShape
    schedule: TilePipelineSchedule

    def to_dict(self) -> dict[str, dict[str, int | str]]:
        return {
            "shape": self.shape.to_dict(),
            "schedule": self.schedule.to_dict(),
        }

    def stable_key(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class TilePipelineMeasurement:
    tuning: TilePipelineTuning
    median_ms: float
    p90_ms: float

    def __post_init__(self) -> None:
        for name, value in (
            ("median_ms", self.median_ms),
            ("p90_ms", self.p90_ms),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class TilePipelineTuningKey:
    gpu_name: str
    compute_capability: tuple[int, int]
    sm_count: int
    attn_tp_size: int
    hidden_size: int
    phase: str
    row_bucket: int
    topology_identity: str
    search_space_identity: str
    schema_version: int = TUNING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.gpu_name:
            raise ValueError("gpu_name must be non-empty")
        if (
            type(self.compute_capability) is not tuple
            or len(self.compute_capability) != 2
        ):
            raise ValueError("compute_capability must be a (major, minor) tuple")
        for name, value in zip(
            ("compute_capability.major", "compute_capability.minor"),
            self.compute_capability,
            strict=True,
        ):
            _require_plain_int(name, value, minimum=0)
        _require_plain_int("sm_count", self.sm_count)
        if self.attn_tp_size not in (2, 4, 8):
            raise ValueError("attn_tp_size must be one of (2, 4, 8)")
        if self.hidden_size not in (2048, 4096):
            raise ValueError("hidden_size must be one of (2048, 4096)")
        if self.phase not in SUPPORTED_PHASES:
            raise ValueError(f"phase must be one of {SUPPORTED_PHASES}")
        _require_plain_int("row_bucket", self.row_bucket)
        if not self.topology_identity:
            raise ValueError("topology_identity must be non-empty")
        if len(self.search_space_identity) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.search_space_identity
        ):
            raise ValueError("search_space_identity must be a lowercase SHA-256 digest")
        if self.schema_version != TUNING_SCHEMA_VERSION:
            raise ValueError(f"schema_version must equal {TUNING_SCHEMA_VERSION}")

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["compute_capability"] = list(self.compute_capability)
        return payload

    def stable_key(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


def shape_space_identity(
    shapes: Iterable[TilePipelineShape],
) -> str:
    ordered_shapes = tuple(sorted(set(shapes)))
    if not ordered_shapes:
        raise ValueError("tile-pipeline shape space must be non-empty")
    if any(not isinstance(shape, TilePipelineShape) for shape in ordered_shapes):
        raise ValueError("shape space must contain TilePipelineShape values")
    payload = {
        "autotune_policy_version": AUTOTUNE_POLICY_VERSION,
        "shapes": [shape.to_dict() for shape in ordered_shapes],
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()


def validate_shape_for_hidden_size(
    shape: TilePipelineShape,
    hidden_size: int,
) -> None:
    if not isinstance(shape, TilePipelineShape):
        raise ValueError("shape must be a TilePipelineShape")
    if hidden_size not in SUPPORTED_HIDDEN_SIZES:
        raise ValueError(f"hidden_size must be one of {SUPPORTED_HIDDEN_SIZES}")
    if (
        shape.consumer_cohorts is not ConsumerCohorts.ONE
        or shape.weight_placement is not WeightPlacement.REGISTER
    ) and (hidden_size != 4096 or shape.block_size != 512):
        raise ValueError(
            "multiple consumer cohorts and shared weights require an "
            "H4096 physical 512-thread CTA"
        )
    if hidden_size % (shape.block_size * 8) != 0 or hidden_size // (
        shape.block_size * 8
    ) not in (1, 2, 4):
        raise ValueError(
            f"block_size {shape.block_size} is unsupported for H{hidden_size}"
        )


def build_shape_space(
    *,
    phase: str,
    hidden_size: int,
    rows_per_tiles: Iterable[int],
    ring_stages: Iterable[int],
    block_sizes: Iterable[int],
    signal_backoffs: Iterable[int],
    launch_bounds_min_blocks: Iterable[int] = (0,),
    consumer_cohorts: Iterable[ConsumerCohorts] = (ConsumerCohorts.ONE,),
    weight_placements: Iterable[WeightPlacement] = (WeightPlacement.REGISTER,),
) -> tuple[TilePipelineShape, ...]:
    if phase not in SUPPORTED_PHASES:
        raise ValueError(f"phase must be one of {SUPPORTED_PHASES}")
    if hidden_size not in SUPPORTED_HIDDEN_SIZES:
        raise ValueError(f"hidden_size must be one of {SUPPORTED_HIDDEN_SIZES}")
    row_values = tuple(rows_per_tiles)
    ring_values = tuple(ring_stages)
    block_values = tuple(block_sizes)
    backoff_values = tuple(signal_backoffs)
    launch_bounds_values = tuple(launch_bounds_min_blocks)
    cohort_values = tuple(consumer_cohorts)
    weight_values = tuple(weight_placements)
    if any(block not in SUPPORTED_BLOCK_SIZES for block in block_values):
        raise ValueError(f"block_sizes must use {SUPPORTED_BLOCK_SIZES}")
    if any(not isinstance(cohort, ConsumerCohorts) for cohort in cohort_values):
        raise ValueError("consumer_cohorts must contain ConsumerCohorts values")
    if any(not isinstance(placement, WeightPlacement) for placement in weight_values):
        raise ValueError("weight_placements must contain WeightPlacement values")
    shapes = {
        TilePipelineShape(
            rows,
            ring,
            block,
            backoff,
            min_blocks,
            cohort,
            placement,
        )
        for rows in row_values
        for ring in ring_values
        for block in block_values
        if (hidden_size % (block * 8) == 0 and hidden_size // (block * 8) in (1, 2, 4))
        for backoff in backoff_values
        for min_blocks in launch_bounds_values
        for cohort in cohort_values
        for placement in weight_values
    }
    if not shapes:
        raise ValueError("tile-pipeline shape space must be non-empty")
    for shape in shapes:
        validate_shape_for_hidden_size(shape, hidden_size)
    return tuple(sorted(shapes))


def build_schedule_space(
    *,
    shape: TilePipelineShape,
    occupancy: int,
) -> tuple[TilePipelineTuning, ...]:
    if not isinstance(shape, TilePipelineShape):
        raise ValueError("shape must be a TilePipelineShape")
    _require_plain_int("occupancy", occupancy, minimum=2)
    return tuple(
        TilePipelineTuning(
            shape=shape,
            schedule=TilePipelineSchedule(
                blocks_per_sm=blocks_per_sm,
                producer_blocks_per_sm=producer_blocks_per_sm,
            ),
        )
        for blocks_per_sm in range(2, occupancy + 1)
        for producer_blocks_per_sm in range(1, blocks_per_sm)
    )


def default_tuning(
    *,
    shape: TilePipelineShape,
    occupancy: int,
) -> TilePipelineTuning:
    _require_plain_int("occupancy", occupancy, minimum=2)
    return TilePipelineTuning(
        shape=shape,
        schedule=TilePipelineSchedule(
            blocks_per_sm=occupancy,
            producer_blocks_per_sm=max(1, occupancy // 2),
        ),
    )


def select_top_tunings(
    measurements: Iterable[TilePipelineMeasurement],
    *,
    top_k: int,
) -> tuple[TilePipelineMeasurement, ...]:
    _require_plain_int("top_k", top_k)
    ordered = sorted(
        measurements,
        key=lambda measurement: (
            measurement.p90_ms,
            measurement.median_ms,
            measurement.tuning,
        ),
    )
    if not ordered:
        raise ValueError("at least one tuning measurement is required")
    return tuple(ordered[:top_k])


def build_stage_two_tunings(
    *,
    stage_one_measurements: Iterable[TilePipelineMeasurement],
    occupancies: dict[TilePipelineShape, int],
    top_k: int,
) -> tuple[TilePipelineTuning, ...]:
    measurements = tuple(stage_one_measurements)
    selected = select_top_tunings(measurements, top_k=top_k)
    stage_one_tunings = {measurement.tuning for measurement in measurements}
    expanded = set()
    for measurement in selected:
        shape = measurement.tuning.shape
        try:
            occupancy = occupancies[shape]
        except KeyError as error:
            raise ValueError(f"missing occupancy for shape {shape}") from error
        expanded.update(
            tuning
            for tuning in build_schedule_space(
                shape=shape,
                occupancy=occupancy,
            )
            if tuning not in stage_one_tunings
        )
    return tuple(sorted(expanded))
