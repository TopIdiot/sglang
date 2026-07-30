"""Pure-Python contracts for the AttnTP fused norm microbenchmark tuner."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import statistics
from typing import Callable, Iterable, Protocol, Sequence, TypeVar


class Backend(str, Enum):
    IPC = "ipc"
    SYMM = "symm"


class Algorithm(str, Enum):
    SOURCE_PUSH = "source_push"
    OWNER_PULL = "owner_pull"
    SYMM = "symm"


class _StableCandidate(Protocol):
    @property
    def stable_id(self) -> str: ...


_Candidate = TypeVar("_Candidate", bound=_StableCandidate)


_SUPPORTED_PHASES = ("prefill", "decode")
_SUPPORTED_ATTN_TP_SIZES = (2, 4, 8)
_SUPPORTED_HIDDEN_SIZES = (2048, 4096)
_SUPPORTED_OUTPUT_MODES = (
    "replicated",
    "single_contributor",
    "token_scattered",
)
_SUPPORTED_PRECISIONS = ("reference_bf16", "full_fp32")
_SUPPORTED_EXECUTIONS = ("eager", "graph")
_SUPPORTED_BLOCK_SIZES = (128, 256, 512)
_SUPPORTED_SIGNAL_BACKOFFS = (32, 64, 128, 256)
_SUPPORTED_ROWS_PER_TILE = (1, 2, 4, 8)


def _require_member(name: str, value, supported: Sequence) -> None:
    if value not in supported:
        raise ValueError(f"{name} must be one of {tuple(supported)}, got {value!r}")


def _normalize_unique_positive(name: str, values: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(set(values)))
    if not normalized or any(
        type(value) is not int or value <= 0 for value in normalized
    ):
        raise ValueError(f"{name} must contain positive integers")
    return normalized


@dataclass(frozen=True)
class TuningShape:
    phase: str
    attn_tp_size: int
    hidden_size: int
    output_mode: str
    internal_precision: str
    execution: str

    def __post_init__(self) -> None:
        _require_member("phase", self.phase, _SUPPORTED_PHASES)
        _require_member("attn_tp_size", self.attn_tp_size, _SUPPORTED_ATTN_TP_SIZES)
        _require_member("hidden_size", self.hidden_size, _SUPPORTED_HIDDEN_SIZES)
        _require_member("output_mode", self.output_mode, _SUPPORTED_OUTPUT_MODES)
        _require_member(
            "internal_precision", self.internal_precision, _SUPPORTED_PRECISIONS
        )
        _require_member("execution", self.execution, _SUPPORTED_EXECUTIONS)

    @property
    def stable_id(self) -> str:
        return (
            f"{self.phase}-n{self.attn_tp_size}-h{self.hidden_size}-"
            f"{self.output_mode}-{self.internal_precision}-{self.execution}"
        )

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "attn_tp_size": self.attn_tp_size,
            "hidden_size": self.hidden_size,
            "output_mode": self.output_mode,
            "internal_precision": self.internal_precision,
            "execution": self.execution,
        }

    @classmethod
    def from_dict(cls, value: dict) -> TuningShape:
        return cls(**value)


@dataclass(frozen=True)
class CompileSpec:
    shape: TuningShape
    backend: Backend
    algorithm: Algorithm
    block_size: int
    signal_backoff: int
    rows_per_tile: int

    def __post_init__(self) -> None:
        if not isinstance(self.shape, TuningShape):
            raise ValueError("shape must be a TuningShape")
        if not isinstance(self.backend, Backend):
            raise ValueError("backend must be a Backend")
        if not isinstance(self.algorithm, Algorithm):
            raise ValueError("algorithm must be an Algorithm")
        _require_member("block_size", self.block_size, _SUPPORTED_BLOCK_SIZES)
        vector_denominator = self.block_size * 8
        if (
            self.shape.hidden_size % vector_denominator != 0
            or self.shape.hidden_size // vector_denominator not in (1, 2, 4)
        ):
            raise ValueError(
                f"block_size {self.block_size} is unsupported for hidden size "
                f"{self.shape.hidden_size}"
            )

        if self.algorithm is Algorithm.OWNER_PULL:
            if self.backend is not Backend.IPC:
                raise ValueError("owner_pull requires the IPC backend")
            if self.signal_backoff != 0 or self.rows_per_tile != 0:
                raise ValueError("owner_pull does not consume signal or tile knobs")
        elif self.algorithm is Algorithm.SOURCE_PUSH:
            if self.backend is not Backend.IPC:
                raise ValueError("source_push requires the IPC backend")
            _require_member(
                "signal_backoff", self.signal_backoff, _SUPPORTED_SIGNAL_BACKOFFS
            )
            if self.shape.phase == "prefill":
                _require_member(
                    "rows_per_tile", self.rows_per_tile, _SUPPORTED_ROWS_PER_TILE
                )
            elif self.rows_per_tile != 0:
                raise ValueError("decode source_push does not consume rows_per_tile")
        else:
            if self.backend is not Backend.SYMM:
                raise ValueError("symm algorithm requires the Symm backend")
            _require_member(
                "signal_backoff", self.signal_backoff, _SUPPORTED_SIGNAL_BACKOFFS
            )
            if self.rows_per_tile != 0:
                raise ValueError("symm does not consume rows_per_tile")

    @property
    def stable_id(self) -> str:
        return (
            f"{self.shape.stable_id}-{self.backend.value}-{self.algorithm.value}-"
            f"b{self.block_size}-wait{self.signal_backoff}-tile{self.rows_per_tile}"
        )

    def to_dict(self) -> dict:
        return {
            "shape": self.shape.to_dict(),
            "backend": self.backend.value,
            "algorithm": self.algorithm.value,
            "block_size": self.block_size,
            "signal_backoff": self.signal_backoff,
            "rows_per_tile": self.rows_per_tile,
        }

    @classmethod
    def from_dict(cls, value: dict) -> CompileSpec:
        return cls(
            shape=TuningShape.from_dict(value["shape"]),
            backend=Backend(value["backend"]),
            algorithm=Algorithm(value["algorithm"]),
            block_size=value["block_size"],
            signal_backoff=value["signal_backoff"],
            rows_per_tile=value["rows_per_tile"],
        )


@dataclass(frozen=True)
class RunCandidate:
    compile_spec: CompileSpec
    blocks_per_sm: int

    def __post_init__(self) -> None:
        if not isinstance(self.compile_spec, CompileSpec):
            raise ValueError("compile_spec must be a CompileSpec")
        if type(self.blocks_per_sm) is not int or self.blocks_per_sm <= 0:
            raise ValueError("blocks_per_sm must be a positive integer")

    @property
    def stable_id(self) -> str:
        return f"{self.compile_spec.stable_id}-grid{self.blocks_per_sm}"

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.stable_id,
            "compile_spec": self.compile_spec.to_dict(),
            "blocks_per_sm": self.blocks_per_sm,
        }


@dataclass(frozen=True)
class CompileResult:
    spec: CompileSpec
    max_occupancy: int | None
    compile_ms: float
    error: str | None

    def __post_init__(self) -> None:
        if self.compile_ms < 0:
            raise ValueError("compile_ms must be non-negative")
        if self.error is None:
            if self.max_occupancy is None or self.max_occupancy <= 0:
                raise ValueError("successful compilation requires positive occupancy")
        elif self.max_occupancy is not None:
            raise ValueError("failed compilation cannot report occupancy")

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def succeeded(
        cls,
        spec: CompileSpec,
        *,
        max_occupancy: int,
        compile_ms: float,
    ) -> CompileResult:
        return cls(spec, max_occupancy, compile_ms, None)

    @classmethod
    def failed(
        cls,
        spec: CompileSpec,
        *,
        error: str,
        compile_ms: float,
    ) -> CompileResult:
        if not error:
            raise ValueError("compile failure requires a non-empty error")
        return cls(spec, None, compile_ms, error)

    def to_dict(self) -> dict:
        return {
            "compile_spec": self.spec.to_dict(),
            "compile_spec_id": self.spec.stable_id,
            "ok": self.ok,
            "max_occupancy": self.max_occupancy,
            "compile_ms": self.compile_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict) -> CompileResult:
        return cls(
            spec=CompileSpec.from_dict(value["compile_spec"]),
            max_occupancy=value["max_occupancy"],
            compile_ms=value["compile_ms"],
            error=value["error"],
        )


@dataclass(frozen=True)
class Measurement:
    candidate: RunCandidate
    median_ms: float
    samples_ms: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.median_ms <= 0:
            raise ValueError("median_ms must be positive")
        if not self.samples_ms or any(sample <= 0 for sample in self.samples_ms):
            raise ValueError("samples_ms must contain positive values")

    def to_dict(self) -> dict:
        return {
            **self.candidate.to_dict(),
            "median_ms": self.median_ms,
            "samples_ms": list(self.samples_ms),
        }

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.9)


@dataclass(frozen=True)
class Winner:
    candidate: RunCandidate
    reason: str
    speedup_over_preferred: float | None

    def to_dict(self) -> dict:
        return {
            **self.candidate.to_dict(),
            "reason": self.reason,
            "speedup_over_preferred": self.speedup_over_preferred,
        }


@dataclass(frozen=True)
class CollectiveCandidateRejection:
    candidate_id: str
    stage: str
    rank_errors: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.stage not in ("construction", "measurement"):
            raise ValueError("rejection stage must be construction or measurement")
        if not self.rank_errors:
            raise ValueError("rank_errors must not be empty")
        if any(rank < 0 or not error for rank, error in self.rank_errors):
            raise ValueError("rank_errors must contain nonnegative ranks and errors")


@dataclass(frozen=True)
class CollectiveSweepResult:
    measurements: tuple[Measurement, ...]
    rejections: tuple[CollectiveCandidateRejection, ...]


def order_collective_candidates(
    candidates: Iterable[_Candidate],
) -> tuple[_Candidate, ...]:
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.stable_id))
    if not ordered:
        raise ValueError("at least one collective candidate is required")
    candidate_ids = tuple(candidate.stable_id for candidate in ordered)
    if any(not candidate_id for candidate_id in candidate_ids):
        raise ValueError("collective candidate IDs must not be empty")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("collective candidate IDs must be unique")
    return ordered


def validate_collective_candidate_orders(
    rank_candidate_ids: Iterable[Iterable[str]],
) -> tuple[str, ...]:
    orders = tuple(tuple(candidate_ids) for candidate_ids in rank_candidate_ids)
    if not orders:
        raise ValueError("at least one rank candidate order is required")
    expected = orders[0]
    if not expected or any(not candidate_id for candidate_id in expected):
        raise ValueError("rank candidate orders must contain nonempty IDs")
    if len(expected) != len(set(expected)):
        raise ValueError("rank candidate orders must not contain duplicate IDs")
    if any(order != expected for order in orders[1:]):
        raise RuntimeError(
            "collective ranks must use the same ordered candidates; "
            f"rank_orders={orders}"
        )
    return expected


def synchronize_collective_candidate_construction(
    candidate,
    *,
    local_error: str | None,
    gather_errors_fn: Callable[[str | None], Iterable[str | None]],
    candidate_id: str | None = None,
    stage: str = "construction",
):
    if stage not in ("construction", "measurement"):
        raise ValueError("stage must be construction or measurement")
    resolved_id = candidate_id or getattr(candidate, "stable_id", None)
    if not resolved_id:
        raise ValueError("candidate_id is required when construction returns no runner")
    gathered_errors = tuple(gather_errors_fn(local_error))
    if not gathered_errors:
        raise ValueError("candidate construction must gather every rank status")
    rank_errors = tuple(
        (rank, error) for rank, error in enumerate(gathered_errors) if error is not None
    )
    if rank_errors:
        if candidate is not None:
            candidate.close()
        return (
            None,
            CollectiveCandidateRejection(
                candidate_id=resolved_id,
                stage=stage,
                rank_errors=rank_errors,
            ),
        )
    if candidate is None:
        raise RuntimeError(
            f"collective candidate {resolved_id} returned no runner without an error"
        )
    return candidate, None


def aggregate_max_rank_samples(
    rank_samples: Iterable[Iterable[float]],
) -> tuple[float, ...]:
    samples = tuple(tuple(values) for values in rank_samples)
    if not samples or not samples[0]:
        raise ValueError("rank samples must not be empty")
    sample_count = len(samples[0])
    if any(len(values) != sample_count for values in samples):
        raise ValueError("every rank must report the same number of latency samples")
    if any(
        not math.isfinite(value) or value <= 0 for values in samples for value in values
    ):
        raise ValueError("rank latency samples must be finite and positive")
    return tuple(max(values) for values in zip(*samples, strict=True))


def measure_collective_candidates(
    candidates: Iterable[_Candidate],
    *,
    gather_candidate_orders_fn: Callable[
        [tuple[str, ...]],
        Iterable[Iterable[str]],
    ],
    prepare_fn: Callable[[_Candidate], object],
    gather_errors_fn: Callable[
        [str, str, str | None],
        Iterable[str | None],
    ],
    measure_fn: Callable[[object, _Candidate], Iterable[float]],
    gather_samples_fn: Callable[
        [tuple[float, ...]],
        Iterable[Iterable[float]],
    ],
) -> CollectiveSweepResult:
    """Measure collective candidates in lockstep without owning tune lifecycle."""

    ordered = order_collective_candidates(candidates)
    candidate_ids = tuple(candidate.stable_id for candidate in ordered)
    validate_collective_candidate_orders(gather_candidate_orders_fn(candidate_ids))

    measurements = []
    rejections = []
    for candidate in ordered:
        candidate_id = candidate.stable_id
        runner = None
        local_error = None
        try:
            runner = prepare_fn(candidate)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"

        runner, rejection = synchronize_collective_candidate_construction(
            runner,
            candidate_id=candidate_id,
            local_error=local_error,
            gather_errors_fn=lambda error: gather_errors_fn(
                "construction",
                candidate_id,
                error,
            ),
            stage="construction",
        )
        if rejection is not None:
            rejections.append(rejection)
            continue

        try:
            local_samples = ()
            local_error = None
            try:
                local_samples = tuple(measure_fn(runner, candidate))
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"

            runner, rejection = synchronize_collective_candidate_construction(
                runner,
                candidate_id=candidate_id,
                local_error=local_error,
                gather_errors_fn=lambda error: gather_errors_fn(
                    "measurement",
                    candidate_id,
                    error,
                ),
                stage="measurement",
            )
            if rejection is not None:
                rejections.append(rejection)
                continue

            samples = aggregate_max_rank_samples(gather_samples_fn(local_samples))
            measurements.append(
                Measurement(
                    candidate=candidate,
                    median_ms=float(statistics.median(samples)),
                    samples_ms=samples,
                )
            )
        finally:
            if runner is not None:
                runner.close()

    return CollectiveSweepResult(
        measurements=tuple(measurements),
        rejections=tuple(rejections),
    )


def generate_compile_specs(
    shape: TuningShape,
    *,
    block_sizes: Iterable[int],
    signal_backoffs: Iterable[int],
    source_push_rows_per_tile: Iterable[int],
) -> tuple[CompileSpec, ...]:
    blocks = _normalize_unique_positive("block_sizes", block_sizes)
    backoffs = _normalize_unique_positive("signal_backoffs", signal_backoffs)
    tiles = _normalize_unique_positive(
        "source_push_rows_per_tile", source_push_rows_per_tile
    )
    specs: list[CompileSpec] = []
    for block_size in blocks:
        try:
            specs.append(
                CompileSpec(
                    shape=shape,
                    backend=Backend.IPC,
                    algorithm=Algorithm.OWNER_PULL,
                    block_size=block_size,
                    signal_backoff=0,
                    rows_per_tile=0,
                )
            )
        except ValueError as error:
            if "block_size" not in str(error):
                raise
            continue
        for signal_backoff in backoffs:
            source_tiles = tiles if shape.phase == "prefill" else (0,)
            for rows_per_tile in source_tiles:
                specs.append(
                    CompileSpec(
                        shape=shape,
                        backend=Backend.IPC,
                        algorithm=Algorithm.SOURCE_PUSH,
                        block_size=block_size,
                        signal_backoff=signal_backoff,
                        rows_per_tile=rows_per_tile,
                    )
                )
            specs.append(
                CompileSpec(
                    shape=shape,
                    backend=Backend.SYMM,
                    algorithm=Algorithm.SYMM,
                    block_size=block_size,
                    signal_backoff=signal_backoff,
                    rows_per_tile=0,
                )
            )
    return tuple(sorted(specs, key=lambda spec: spec.stable_id))


def expand_run_candidates(
    spec: CompileSpec,
    *,
    max_occupancy: int,
    blocks_per_sm: Iterable[int],
) -> tuple[RunCandidate, ...]:
    if type(max_occupancy) is not int or max_occupancy <= 0:
        raise ValueError("max_occupancy must be a positive integer")
    blocks = _normalize_unique_positive("blocks_per_sm", blocks_per_sm)
    candidates = tuple(
        RunCandidate(spec, value) for value in blocks if value <= max_occupancy
    )
    if not candidates:
        raise ValueError("no blocks_per_sm candidate fits the kernel occupancy")
    return candidates


def _measurement_sort_key(measurement: Measurement) -> tuple[float, str]:
    return measurement.median_ms, measurement.candidate.stable_id


def select_finalists(
    measurements: Iterable[Measurement],
    *,
    count: int,
) -> tuple[RunCandidate, ...]:
    if type(count) is not int or count <= 0:
        raise ValueError("finalist count must be a positive integer")
    ordered = sorted(measurements, key=_measurement_sort_key)
    if not ordered:
        raise ValueError("cannot select finalists without measurements")
    ids = [measurement.candidate.stable_id for measurement in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("measurements contain duplicate candidates")
    return tuple(measurement.candidate for measurement in ordered[:count])


def select_winner(
    measurements: Iterable[Measurement],
    *,
    minimum_speedup: float,
) -> Winner:
    if minimum_speedup <= 1.0:
        raise ValueError("minimum_speedup must be greater than one")
    ordered = sorted(measurements, key=_measurement_sort_key)
    if not ordered:
        raise ValueError("cannot select a winner without measurements")
    ids = [measurement.candidate.stable_id for measurement in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("measurements contain duplicate candidates")
    fastest = ordered[0]
    if len(ordered) == 1:
        return Winner(fastest.candidate, "only_valid_candidate", None)

    preferred = sorted(
        (
            measurement
            for measurement in ordered
            if measurement.candidate.compile_spec.algorithm is Algorithm.OWNER_PULL
        ),
        key=_measurement_sort_key,
    )
    if not preferred:
        return Winner(fastest.candidate, "no_preferred_candidate", None)
    preferred_best = preferred[0]
    if fastest.candidate == preferred_best.candidate:
        return Winner(fastest.candidate, "preferred_fastest", 1.0)

    speedup = preferred_best.median_ms / fastest.median_ms
    if speedup < minimum_speedup:
        return Winner(
            preferred_best.candidate,
            "preferred_within_noise_band",
            speedup,
        )
    return Winner(fastest.candidate, "measured_speedup", speedup)


def percentile(samples: tuple[float, ...], fraction: float) -> float:
    if not samples:
        raise ValueError("percentile samples must not be empty")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be in [0, 1]")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def select_lowest_p90_measurement(measurements: Iterable):
    values = tuple(measurements)
    if not values:
        raise ValueError("cannot select a winner without measurements")
    candidate_ids = [measurement.candidate.stable_id for measurement in values]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("measurements contain duplicate candidates")
    return min(
        values,
        key=lambda measurement: (
            measurement.p90_ms,
            measurement.median_ms,
            measurement.candidate.stable_id,
        ),
    )


def select_topology_winner(
    measurements: Iterable[Measurement],
) -> Winner:
    winner = select_lowest_p90_measurement(measurements)
    return Winner(
        winner.candidate,
        "lowest_p90_then_median",
        None,
    )


def build_report(
    *,
    shape: TuningShape,
    compile_results: Iterable[CompileResult],
    coarse_measurements: Iterable[Measurement],
    final_measurements: Iterable[Measurement],
    winner: Winner,
) -> dict:
    return {
        "schema_version": 1,
        "shape": shape.to_dict(),
        "compile_results": [result.to_dict() for result in compile_results],
        "coarse_measurements": [
            measurement.to_dict() for measurement in coarse_measurements
        ],
        "final_measurements": [
            measurement.to_dict() for measurement in final_measurements
        ],
        "winner": winner.to_dict(),
    }
