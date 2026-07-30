"""Production tuning contracts for fused AttnTP reduction and WeLM norms."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping

from sglang.jit_kernel.autotune import (
    ManifestIdentity,
    iter_bounded_compilation,
    load_or_tune_coordinated_manifest,
    production_compile_worker_count,
    production_manifest_path,
)
from sglang.jit_kernel.attntp_fused_norm.ipc import OutputMode

logger = logging.getLogger(__name__)

_PHASES = ("prefill", "decode")
_TOPOLOGIES = ("tp", "dp", "cp")
_EXECUTION_MODES = ("eager", "graph")
_SUPPORTED_ATTN_TP_SIZES = (1, 2, 4, 8)
_SUPPORTED_HIDDEN_SIZES = (2048, 4096)
_PRODUCTION_ROW_BUCKETS = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536)
_PRODUCTION_NORM_EPS = 1e-6
_PRODUCTION_OUTPUT_MAX_ABS = 0.125
_PRODUCTION_OUTPUT_MEAN_ABS = 2e-3
_PRODUCTION_RESIDUAL_MAX_ABS = 0.125
_PRODUCTION_RESIDUAL_MEAN_ABS = 5e-4
_PRODUCTION_COARSE_WARMUP = 2
_PRODUCTION_COARSE_ITERATIONS = 5
_PRODUCTION_FINAL_WARMUP = 5
_PRODUCTION_FINAL_ITERATIONS = 20
_PRODUCTION_FINALISTS_PER_FAMILY = 4

SearchProfile = Literal["quick", "balanced", "broad"]
PRODUCTION_SEARCH_PROFILE: SearchProfile = "balanced"
_SEARCH_PROFILES = ("quick", "balanced", "broad")

_ParameterValue = bool | int | float | str | None
_ShapeKey = tuple[str, str, int, int, str, OutputMode]


def _require_search_profile(profile: str) -> None:
    if profile not in _SEARCH_PROFILES:
        raise ValueError(f"profile must be one of {_SEARCH_PROFILES}, got {profile!r}")


def build_production_row_buckets(max_rows: int) -> tuple[int, ...]:
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    buckets = [value for value in _PRODUCTION_ROW_BUCKETS if value < max_rows]
    buckets.append(max_rows)
    return tuple(buckets)


def build_production_manifest_identity(
    required_keys: Iterable[AttnTPFusedNormWorkloadKey],
    *,
    device,
    topology_extra: Mapping[str, object] | None = None,
    profile: SearchProfile = PRODUCTION_SEARCH_PROFILE,
) -> ManifestIdentity[AttnTPFusedNormWorkloadKey]:
    import torch
    import tvm_ffi

    _require_search_profile(profile)
    keys = tuple(required_keys)
    if not keys:
        raise ValueError("production identity requires at least one workload")
    shape_key = keys[0].shape_key
    if any(key.shape_key != shape_key for key in keys[1:]):
        raise ValueError(
            "production identity workloads must share one phase and topology"
        )

    source_root = Path(__file__).resolve().parent
    jit_source_root = (
        source_root.parent / "csrc" / "distributed" / "attntp_fused_norm"
    )
    source_paths = (
        source_root / "tuning.py",
        source_root / "candidates.py",
        source_root / "resources.py",
        source_root / "ipc.py",
        source_root / "symm.py",
        source_root / "tile.py",
        source_root / "tile_tuning.py",
        *sorted(
            path
            for path in jit_source_root.rglob("*")
            if path.is_file()
            and path.suffix in {".cc", ".cpp", ".cu", ".cuh", ".h"}
        ),
    )
    source_hash = hashlib.sha256()
    for source_path in source_paths:
        source_hash.update(source_path.read_bytes())

    properties = torch.cuda.get_device_properties(device)
    first = keys[0]
    topology_payload = {
        "phase": first.phase,
        "topology": first.topology,
        "attn_tp_size": first.attn_tp_size,
        "hidden_size": first.hidden_size,
        "execution": first.execution,
        "output_mode": first.output_mode.value,
        "extra": dict(topology_extra or {}),
    }
    return ManifestIdentity(
        kernel_abi=source_hash.hexdigest(),
        compiler_version=str(getattr(tvm_ffi, "__version__", "unknown")),
        cuda_version=str(torch.version.cuda or "unknown"),
        gpu_fingerprint="|".join(
            (
                str(properties.name),
                f"cc{properties.major}.{properties.minor}",
                f"sm{properties.multi_processor_count}",
                f"mem{properties.total_memory}",
            )
        ),
        topology_fingerprint=json.dumps(
            topology_payload,
            sort_keys=True,
            separators=(",", ":"),
        ),
        dtype="partial=bf16,accum=fp32,output=bf16,residual=fp32",
        profile=(f"{profile}:v1-coarse2x5-final4-per-family-final5x20"),
        required_keys=keys,
    )


def attntp_production_manifest_path(
    cache_root: str | Path,
    identity: ManifestIdentity[AttnTPFusedNormWorkloadKey],
) -> Path:
    return production_manifest_path(
        cache_root,
        "attntp_fused_norm",
        identity,
    )


def resolve_current_output_mode(topology: str) -> OutputMode:
    """Return the output layout required by the current production framework."""

    if topology not in _TOPOLOGIES:
        raise ValueError(f"topology must be one of {_TOPOLOGIES}, got {topology!r}")
    return OutputMode.REPLICATED


@dataclass(frozen=True)
class AttnTPFusedNormWorkloadKey:
    phase: str
    topology: str
    attn_tp_size: int
    hidden_size: int
    row_bucket: int
    execution: str
    output_mode: OutputMode

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError(f"phase must be one of {_PHASES}, got {self.phase!r}")
        if self.topology not in _TOPOLOGIES:
            raise ValueError(
                f"topology must be one of {_TOPOLOGIES}, got {self.topology!r}"
            )
        if (
            type(self.attn_tp_size) is not int
            or self.attn_tp_size not in _SUPPORTED_ATTN_TP_SIZES
        ):
            raise ValueError(
                "attn_tp_size must be one of "
                f"{_SUPPORTED_ATTN_TP_SIZES}, got {self.attn_tp_size!r}"
            )
        if (
            type(self.hidden_size) is not int
            or self.hidden_size not in _SUPPORTED_HIDDEN_SIZES
        ):
            raise ValueError(
                "hidden_size must be one of "
                f"{_SUPPORTED_HIDDEN_SIZES}, got {self.hidden_size!r}"
            )
        if type(self.row_bucket) is not int or self.row_bucket <= 0:
            raise ValueError("row_bucket must be a positive integer")
        if self.execution not in _EXECUTION_MODES:
            raise ValueError(
                f"execution must be one of {_EXECUTION_MODES}, got {self.execution!r}"
            )
        if not isinstance(self.output_mode, OutputMode):
            raise ValueError("output_mode must be an OutputMode")

    @classmethod
    def current(
        cls,
        *,
        phase: str,
        topology: str,
        attn_tp_size: int,
        hidden_size: int,
        row_bucket: int,
        execution: str,
    ) -> AttnTPFusedNormWorkloadKey:
        return cls(
            phase=phase,
            topology=topology,
            attn_tp_size=attn_tp_size,
            hidden_size=hidden_size,
            row_bucket=row_bucket,
            execution=execution,
            output_mode=resolve_current_output_mode(topology),
        )

    @property
    def shape_key(self) -> _ShapeKey:
        return (
            self.phase,
            self.topology,
            self.attn_tp_size,
            self.hidden_size,
            self.execution,
            self.output_mode,
        )

    def encode(self) -> str:
        return ":".join(
            (
                self.phase,
                self.topology,
                f"attntp{self.attn_tp_size}",
                f"h{self.hidden_size}",
                f"rows{self.row_bucket}",
                self.execution,
                self.output_mode.value,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "topology": self.topology,
            "attn_tp_size": self.attn_tp_size,
            "hidden_size": self.hidden_size,
            "row_bucket": self.row_bucket,
            "execution": self.execution,
            "output_mode": self.output_mode.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AttnTPFusedNormWorkloadKey:
        return cls(
            phase=str(payload["phase"]),
            topology=str(payload["topology"]),
            attn_tp_size=int(payload["attn_tp_size"]),
            hidden_size=int(payload["hidden_size"]),
            row_bucket=int(payload["row_bucket"]),
            execution=str(payload["execution"]),
            output_mode=OutputMode(str(payload["output_mode"])),
        )


@dataclass(frozen=True)
class AttnTPFusedNormCandidate:
    family: str
    output_mode: OutputMode
    parameters: tuple[tuple[str, _ParameterValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("candidate family must be nonempty")
        if not isinstance(self.output_mode, OutputMode):
            raise ValueError("candidate output_mode must be an OutputMode")
        if type(self.parameters) is not tuple:
            raise ValueError("candidate parameters must be a tuple")

        names = []
        for parameter in self.parameters:
            if (
                type(parameter) is not tuple
                or len(parameter) != 2
                or not isinstance(parameter[0], str)
                or not parameter[0]
            ):
                raise ValueError(
                    "each candidate parameter must be a nonempty name/value tuple"
                )
            if not isinstance(parameter[1], (bool, int, float, str, type(None))):
                raise ValueError(
                    "candidate parameter values must be JSON scalar values"
                )
            names.append(parameter[0])
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError(
                "candidate parameters must have unique names in sorted order"
            )

    @property
    def stable_id(self) -> str:
        suffix = ",".join(f"{name}={value}" for name, value in self.parameters)
        base = f"{self.family}:{self.output_mode.value}"
        return f"{base}:{suffix}" if suffix else base

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "output_mode": self.output_mode.value,
            "parameters": [list(parameter) for parameter in self.parameters],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AttnTPFusedNormCandidate:
        raw_parameters = payload.get("parameters", ())
        if not isinstance(raw_parameters, (list, tuple)):
            raise ValueError("serialized candidate parameters must be a sequence")
        parameters = tuple(
            (str(parameter[0]), parameter[1]) for parameter in raw_parameters
        )
        return cls(
            family=str(payload["family"]),
            output_mode=OutputMode(str(payload["output_mode"])),
            parameters=parameters,
        )


@dataclass(frozen=True)
class AttnTPProductionCompilation:
    occupancies: Mapping[str, int]
    errors: Mapping[str, str]


@dataclass(frozen=True)
class AttnTPProductionCandidateSpace:
    candidates: tuple[AttnTPFusedNormCandidate, ...]
    compilation: AttnTPProductionCompilation


@dataclass
class AttnTPProductionBenchmarkState:
    partial: object
    residual: object
    o_norm_weight: object
    post_norm_weight: object
    output: object
    residual_out: object
    expected_output: object
    expected_residual_out: object
    actual_rows: object | None
    lane_rotation: object | None
    lane_rotation_value: int
    device: object


def build_production_benchmark_state(
    workload: AttnTPFusedNormWorkloadKey,
    *,
    rank: int,
    device,
    device_group,
    norm_eps: float = _PRODUCTION_NORM_EPS,
) -> AttnTPProductionBenchmarkState:
    """Build deterministic inputs and an ordered FP32 collective reference."""

    import torch
    import torch.distributed as dist

    if not isinstance(workload, AttnTPFusedNormWorkloadKey):
        raise ValueError("workload must be an AttnTPFusedNormWorkloadKey")
    if type(rank) is not int or rank < 0 or rank >= workload.attn_tp_size:
        raise ValueError("rank must be inside the AttnTP group")
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("production AttnTP tuning requires a CUDA device")
    if dist.get_world_size(group=device_group) != workload.attn_tp_size:
        raise ValueError("device-group size must match AttnTP size")
    if dist.get_rank(group=device_group) != rank:
        raise ValueError("device-group rank must match the tuning rank")

    torch.cuda.set_device(device)
    shape = (workload.row_bucket, workload.hidden_size)
    partial_generator = torch.Generator(device=device)
    partial_generator.manual_seed(0xA77F0000 + rank)
    shared_generator = torch.Generator(device=device)
    shared_generator.manual_seed(0xA77F2026)
    partial = torch.randn(
        shape,
        dtype=torch.bfloat16,
        device=device,
        generator=partial_generator,
    )
    residual = torch.randn(
        shape,
        dtype=torch.float32,
        device=device,
        generator=shared_generator,
    )
    o_norm_weight = torch.randn(
        (workload.hidden_size,),
        dtype=torch.bfloat16,
        device=device,
        generator=shared_generator,
    )
    post_norm_weight = torch.randn(
        (workload.hidden_size,),
        dtype=torch.bfloat16,
        device=device,
        generator=shared_generator,
    )

    peer_partials = [torch.empty_like(partial) for _ in range(workload.attn_tp_size)]
    dist.all_gather(peer_partials, partial, group=device_group)
    reduced = torch.zeros_like(residual)
    for peer_partial in peer_partials:
        reduced.add_(peer_partial.float())
    del peer_partials
    o_norm_scale = torch.rsqrt(reduced.square().mean(dim=-1, keepdim=True) + norm_eps)
    o_norm = reduced * o_norm_scale * o_norm_weight.float()
    expected_residual_out = residual + o_norm
    post_norm_scale = torch.rsqrt(
        expected_residual_out.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    expected_output = (
        expected_residual_out * post_norm_scale * post_norm_weight.float()
    ).to(torch.bfloat16)

    return AttnTPProductionBenchmarkState(
        partial=partial,
        residual=residual,
        o_norm_weight=o_norm_weight,
        post_norm_weight=post_norm_weight,
        output=torch.empty(shape, dtype=torch.bfloat16, device=device),
        residual_out=torch.empty(shape, dtype=torch.float32, device=device),
        expected_output=expected_output,
        expected_residual_out=expected_residual_out,
        actual_rows=(
            torch.tensor(
                [workload.row_bucket],
                dtype=torch.int32,
                device=device,
            )
            if workload.phase == "prefill"
            else None
        ),
        lane_rotation=(
            torch.zeros((1,), dtype=torch.int32, device=device)
            if workload.phase == "prefill"
            else None
        ),
        lane_rotation_value=(
            1
            if workload.phase == "prefill"
            and workload.topology == "cp"
            and workload.attn_tp_size > 1
            else 0
        ),
        device=device,
    )


def _run_production_benchmark_candidate(
    workload: AttnTPFusedNormWorkloadKey,
    candidate: AttnTPFusedNormCandidate,
    runner,
    state: AttnTPProductionBenchmarkState,
    *,
    norm_eps: float,
) -> None:
    from sglang.jit_kernel.attntp_fused_norm.candidates import (
        run_prepared_production_candidate_out,
    )

    actual_rows = None
    lane_rotation = None
    if workload.phase == "prefill":
        actual_rows = state.actual_rows
        lane_rotation = state.lane_rotation
        if actual_rows is None or lane_rotation is None:
            raise RuntimeError("Prefill benchmark metadata tensors are unavailable")
        actual_rows.fill_(workload.row_bucket)
        if workload.topology == "cp":
            lane_rotation.fill_(state.lane_rotation_value)

    run_prepared_production_candidate_out(
        phase=workload.phase,
        topology=workload.topology,
        candidate=candidate,
        runner=runner,
        partial=state.partial,
        residual=state.residual,
        o_norm_weight=state.o_norm_weight,
        post_norm_weight=state.post_norm_weight,
        output=state.output,
        residual_out=state.residual_out,
        actual_rows=actual_rows,
        lane_rotation=lane_rotation,
        o_norm_eps=norm_eps,
        post_norm_eps=norm_eps,
    )


def _validate_production_benchmark_outputs(
    state: AttnTPProductionBenchmarkState,
    *,
    candidate_id: str,
) -> None:
    output_diff = (state.output.float() - state.expected_output.float()).abs()
    residual_diff = (state.residual_out - state.expected_residual_out).abs()
    errors = {
        "output_max_abs": float(output_diff.max().item()),
        "output_mean_abs": float(output_diff.mean().item()),
        "residual_max_abs": float(residual_diff.max().item()),
        "residual_mean_abs": float(residual_diff.mean().item()),
    }
    if not all(math.isfinite(value) for value in errors.values()):
        raise RuntimeError(
            "AttnTP fused norm candidate produced non-finite output for "
            f"{candidate_id}: {errors}"
        )
    if (
        errors["output_max_abs"] > _PRODUCTION_OUTPUT_MAX_ABS
        or errors["output_mean_abs"] > _PRODUCTION_OUTPUT_MEAN_ABS
        or errors["residual_max_abs"] > _PRODUCTION_RESIDUAL_MAX_ABS
        or errors["residual_mean_abs"] > _PRODUCTION_RESIDUAL_MEAN_ABS
    ):
        raise RuntimeError(
            "AttnTP fused norm candidate correctness failed for "
            f"{candidate_id}: {errors}"
        )


def measure_prepared_production_candidate(
    workload: AttnTPFusedNormWorkloadKey,
    candidate: AttnTPFusedNormCandidate,
    runner,
    state: AttnTPProductionBenchmarkState,
    *,
    cpu_group,
    warmup: int,
    iterations: int,
    norm_eps: float = _PRODUCTION_NORM_EPS,
) -> tuple[float, ...]:
    """Validate and measure one prepared collective candidate."""

    import torch
    import torch.distributed as dist

    for name, value in (("warmup", warmup), ("iterations", iterations)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")

    def launch() -> None:
        _run_production_benchmark_candidate(
            workload,
            candidate,
            runner,
            state,
            norm_eps=norm_eps,
        )

    dist.barrier(group=cpu_group)
    launch()
    torch.cuda.synchronize(state.device)
    _validate_production_benchmark_outputs(
        state,
        candidate_id=candidate.stable_id,
    )

    graph = None
    operation = launch
    if workload.execution == "graph":
        dist.barrier(group=cpu_group)
        graph = torch.cuda.CUDAGraph()
        with runner.capture():
            with torch.cuda.graph(graph):
                launch()
        dist.barrier(group=cpu_group)
        state.output.fill_(float("nan"))
        state.residual_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(state.device)
        _validate_production_benchmark_outputs(
            state,
            candidate_id=candidate.stable_id,
        )
        operation = graph.replay

    try:
        for _ in range(warmup):
            dist.barrier(group=cpu_group)
            operation()
        torch.cuda.synchronize(state.device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(iterations):
            dist.barrier(group=cpu_group)
            start.record()
            operation()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
        return tuple(samples)
    finally:
        if graph is not None:
            del graph
            torch.cuda.synchronize(state.device)


def compile_production_candidate_occupancies(
    workload: AttnTPFusedNormWorkloadKey,
    candidates: Iterable[AttnTPFusedNormCandidate],
    *,
    participant_index: int,
    participant_count: int,
    gather_fn: Callable[
        [Mapping[str, tuple[int | None, str | None]]],
        Iterable[Mapping[str, tuple[int | None, str | None]]],
    ],
    cpu_count: int | None = None,
    compile_candidate_fn=None,
) -> AttnTPProductionCompilation:
    """Compile partitioned AttnTP modules through the shared bounded queue."""

    from sglang.jit_kernel.attntp_fused_norm.candidates import (
        compile_production_candidate,
        production_candidate_compile_id,
    )
    from sglang.jit_kernel.utils import _jit_compile_context

    if not isinstance(workload, AttnTPFusedNormWorkloadKey):
        raise ValueError("workload must be an AttnTPFusedNormWorkloadKey")
    if (
        type(participant_count) is not int
        or participant_count <= 0
        or type(participant_index) is not int
        or participant_index < 0
        or participant_index >= participant_count
    ):
        raise ValueError("participant index must be inside participant count")
    values = tuple(candidates)
    if not values:
        raise ValueError("at least one production candidate is required")
    representatives = {}
    for candidate in values:
        compile_id = production_candidate_compile_id(workload, candidate)
        representatives.setdefault(compile_id, candidate)
    ordered = tuple(sorted(representatives.items()))
    local_jobs = tuple(
        (compile_id, candidate)
        for index, (compile_id, candidate) in enumerate(ordered)
        if index % participant_count == participant_index
    )
    compile_fn = compile_candidate_fn or compile_production_candidate
    workers = production_compile_worker_count(
        cpu_count if cpu_count is not None else (os.cpu_count() or 1),
        participant_count,
    )
    local_status = {}
    with _jit_compile_context():
        completions = iter_bounded_compilation(
            local_jobs,
            compile_fn=lambda job: compile_fn(workload, job[1]),
            max_workers=workers,
            max_pending=max(workers, workers * 2),
        )
        for completion in completions:
            compile_id, _ = completion.job
            if completion.error is None:
                local_status[compile_id] = (int(completion.compiled), None)
            else:
                local_status[compile_id] = (
                    None,
                    f"{type(completion.error).__name__}: {completion.error}",
                )

    gathered = tuple(gather_fn(local_status))
    if not gathered:
        raise RuntimeError("production compilation gathered no participant status")
    required_ids = set(representatives)
    merged = {}
    for statuses in gathered:
        unexpected = set(statuses) - required_ids
        if unexpected:
            raise RuntimeError(
                "production compilation returned unexpected IDs: "
                f"{tuple(sorted(unexpected))}"
            )
        for compile_id, status in statuses.items():
            if (
                type(status) is not tuple
                or len(status) != 2
                or ((status[0] is None) == (status[1] is None))
            ):
                raise RuntimeError(
                    f"invalid production compile status for {compile_id}"
                )
            previous = merged.setdefault(compile_id, status)
            if previous != status:
                raise RuntimeError(
                    "production compile status differs across participants for "
                    f"{compile_id}: {previous} != {status}"
                )
    missing = required_ids - set(merged)
    if missing:
        raise RuntimeError(
            "production compilation is missing IDs: " + ", ".join(sorted(missing))
        )
    occupancies = {
        compile_id: int(occupancy)
        for compile_id, (occupancy, error) in merged.items()
        if error is None
    }
    errors = {
        compile_id: str(error)
        for compile_id, (occupancy, error) in merged.items()
        if occupancy is None
    }
    return AttnTPProductionCompilation(
        occupancies=MappingProxyType(occupancies),
        errors=MappingProxyType(errors),
    )


def build_compiled_production_candidate_space(
    workload: AttnTPFusedNormWorkloadKey,
    *,
    participant_index: int,
    participant_count: int,
    gather_fn: Callable[
        [Mapping[str, tuple[int | None, str | None]]],
        Iterable[Mapping[str, tuple[int | None, str | None]]],
    ],
    cpu_count: int | None = None,
    compile_candidate_fn=None,
    profile: SearchProfile = PRODUCTION_SEARCH_PROFILE,
) -> AttnTPProductionCandidateSpace:
    from sglang.jit_kernel.attntp_fused_norm.candidates import (
        build_production_non_tile_candidates,
        build_production_tile_candidates,
        build_production_tile_shapes,
        filter_production_candidates_by_occupancy,
        production_candidate_compile_id,
        tile_tuning_from_production_candidate,
    )

    _require_search_profile(profile)
    non_tile = build_production_non_tile_candidates(workload, profile=profile)
    tile_shapes = build_production_tile_shapes(workload, profile=profile)
    tile_compile_candidates = build_production_tile_candidates(
        workload,
        shapes=tile_shapes,
        occupancies={shape: 2 for shape in tile_shapes},
    )
    compilation = compile_production_candidate_occupancies(
        workload,
        (*non_tile, *tile_compile_candidates),
        participant_index=participant_index,
        participant_count=participant_count,
        gather_fn=gather_fn,
        cpu_count=cpu_count,
        compile_candidate_fn=compile_candidate_fn,
    )

    successful_non_tile = tuple(
        candidate
        for candidate in non_tile
        if production_candidate_compile_id(workload, candidate)
        in compilation.occupancies
    )
    accepted_non_tile = (
        filter_production_candidates_by_occupancy(
            workload,
            successful_non_tile,
            occupancies=dict(compilation.occupancies),
        )
        if successful_non_tile
        else ()
    )
    shape_occupancies = {shape: 0 for shape in tile_shapes}
    for candidate in tile_compile_candidates:
        shape = tile_tuning_from_production_candidate(candidate).shape
        compile_id = production_candidate_compile_id(workload, candidate)
        shape_occupancies[shape] = compilation.occupancies.get(compile_id, 0)
    accepted_tile = (
        build_production_tile_candidates(
            workload,
            shapes=tile_shapes,
            occupancies=shape_occupancies,
        )
        if any(occupancy >= 2 for occupancy in shape_occupancies.values())
        else ()
    )
    candidates = tuple(
        sorted(
            {*accepted_non_tile, *accepted_tile},
            key=lambda candidate: candidate.stable_id,
        )
    )
    if not candidates:
        details = ", ".join(
            f"{compile_id}: {error}"
            for compile_id, error in sorted(compilation.errors.items())
        )
        raise RuntimeError(
            "every production AttnTP fused norm candidate was rejected"
            + (f": {details}" if details else "")
        )
    return AttnTPProductionCandidateSpace(
        candidates=candidates,
        compilation=compilation,
    )


@dataclass(frozen=True)
class AttnTPFusedNormWinner:
    candidate: AttnTPFusedNormCandidate
    row_bucket: int
    median_ms: float
    p90_ms: float

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, AttnTPFusedNormCandidate):
            raise ValueError("winner candidate must be an AttnTPFusedNormCandidate")
        if type(self.row_bucket) is not int or self.row_bucket <= 0:
            raise ValueError("winner row_bucket must be a positive integer")
        for name, value in (
            ("median_ms", self.median_ms),
            ("p90_ms", self.p90_ms),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"winner {name} must be finite and positive")
        if self.p90_ms < self.median_ms:
            raise ValueError("winner p90_ms must be at least median_ms")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_dict(),
            "row_bucket": self.row_bucket,
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AttnTPFusedNormWinner:
        candidate = payload["candidate"]
        if not isinstance(candidate, Mapping):
            raise ValueError("serialized winner candidate must be a mapping")
        return cls(
            candidate=AttnTPFusedNormCandidate.from_dict(candidate),
            row_bucket=int(payload["row_bucket"]),
            median_ms=float(payload["median_ms"]),
            p90_ms=float(payload["p90_ms"]),
        )


def _winner_score(winner: AttnTPFusedNormWinner) -> tuple[float, float, str]:
    return (
        winner.p90_ms,
        winner.median_ms,
        winner.candidate.stable_id,
    )


def tune_production_workload(
    workload: AttnTPFusedNormWorkloadKey,
    candidates: Iterable[AttnTPFusedNormCandidate],
    *,
    coarse_family_finalists: int,
    gather_candidate_orders_fn,
    prepare_fn,
    gather_errors_fn,
    coarse_measure_fn,
    final_measure_fn,
    gather_samples_fn,
) -> AttnTPFusedNormWinner:
    """Run a collective coarse/final sweep for one row bucket."""

    from sglang.jit_kernel.attntp_fused_norm.autotune import (
        measure_collective_candidates,
        select_finalists,
        select_lowest_p90_measurement,
    )

    if not isinstance(workload, AttnTPFusedNormWorkloadKey):
        raise ValueError("workload must be an AttnTPFusedNormWorkloadKey")
    if type(coarse_family_finalists) is not int or coarse_family_finalists <= 0:
        raise ValueError("coarse_family_finalists must be a positive integer")
    values = tuple(candidates)
    if not values:
        raise ValueError("production workload requires at least one candidate")
    if any(not isinstance(candidate, AttnTPFusedNormCandidate) for candidate in values):
        raise ValueError(
            "production workload candidates must be AttnTPFusedNormCandidate"
        )

    coarse = measure_collective_candidates(
        values,
        gather_candidate_orders_fn=gather_candidate_orders_fn,
        prepare_fn=prepare_fn,
        gather_errors_fn=gather_errors_fn,
        measure_fn=coarse_measure_fn,
        gather_samples_fn=gather_samples_fn,
    )
    if not coarse.measurements:
        rejected = ", ".join(
            f"{rejection.candidate_id}@{rejection.stage}"
            for rejection in coarse.rejections
        )
        raise RuntimeError(
            "every coarse AttnTP fused norm candidate failed"
            + (f": {rejected}" if rejected else "")
        )

    finalists = []
    families = sorted(
        {measurement.candidate.family for measurement in coarse.measurements}
    )
    for family in families:
        family_measurements = tuple(
            measurement
            for measurement in coarse.measurements
            if measurement.candidate.family == family
        )
        finalists.extend(
            select_finalists(
                family_measurements,
                count=min(
                    coarse_family_finalists,
                    len(family_measurements),
                ),
            )
        )

    final = measure_collective_candidates(
        finalists,
        gather_candidate_orders_fn=gather_candidate_orders_fn,
        prepare_fn=prepare_fn,
        gather_errors_fn=gather_errors_fn,
        measure_fn=final_measure_fn,
        gather_samples_fn=gather_samples_fn,
    )
    if not final.measurements:
        rejected = ", ".join(
            f"{rejection.candidate_id}@{rejection.stage}"
            for rejection in final.rejections
        )
        raise RuntimeError(
            "every finalist AttnTP fused norm candidate failed"
            + (f": {rejected}" if rejected else "")
        )
    measurement = select_lowest_p90_measurement(final.measurements)
    return AttnTPFusedNormWinner(
        candidate=measurement.candidate,
        row_bucket=workload.row_bucket,
        median_ms=measurement.median_ms,
        p90_ms=measurement.p90_ms,
    )


def run_local_production_autotune(
    workloads: Iterable[AttnTPFusedNormWorkloadKey],
    *,
    participant_index: int,
    participant_count: int,
    compilation_gather_fn,
    attn_tp_group,
    device,
    cpu_count: int | None = None,
    profile: SearchProfile = PRODUCTION_SEARCH_PROFILE,
) -> dict[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner]:
    """Compile once, then tune every row bucket inside each AttnTP group."""

    import torch

    from sglang.jit_kernel.attntp_fused_norm.candidates import (
        prepare_production_candidate,
    )
    from sglang.jit_kernel.attntp_fused_norm.resources import (
        build_candidate_resource_set,
    )

    _require_search_profile(profile)
    keys = tuple(
        sorted(
            workloads,
            key=lambda workload: (
                workload.row_bucket,
                workload.encode(),
            ),
        )
    )
    if not keys:
        raise ValueError("production autotune requires at least one workload")
    if len(keys) != len(set(keys)):
        raise ValueError("production autotune workloads must be unique")
    shape_key = keys[0].shape_key
    if any(key.shape_key != shape_key for key in keys[1:]):
        raise ValueError("one production autotune run must use one phase and topology")
    if attn_tp_group.world_size != keys[0].attn_tp_size:
        raise ValueError("AttnTP group size does not match production workloads")
    if attn_tp_group.rank_in_group not in range(attn_tp_group.world_size):
        raise ValueError("AttnTP rank is outside its process group")

    compile_workload = max(keys, key=lambda key: key.row_bucket)
    candidate_space = build_compiled_production_candidate_space(
        compile_workload,
        participant_index=participant_index,
        participant_count=participant_count,
        gather_fn=compilation_gather_fn,
        cpu_count=cpu_count,
        profile=profile,
    )
    candidates = candidate_space.candidates
    if candidate_space.compilation.errors and attn_tp_group.rank_in_group == 0:
        logger.warning(
            "AttnTP fused norm rejected production candidate compilation: %s",
            ", ".join(
                f"{compile_id}={error}"
                for compile_id, error in sorted(
                    candidate_space.compilation.errors.items()
                )
            ),
        )
    resource_sets_by_family = {}
    resource_errors = {}
    resource_sets = []
    for families in (
        ("ipc_owner_pull", "ipc_source_push"),
        ("direct_symm", "tile_pipeline"),
    ):
        family_entries = tuple(
            (workload, candidate)
            for workload in keys
            for candidate in candidates
            if candidate.family in families
        )
        if not family_entries:
            continue
        local_resource_set = None
        local_error = None
        try:
            local_resource_set = build_candidate_resource_set(
                family_entries,
                group=attn_tp_group.cpu_group,
                device=device,
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
        gathered = tuple(attn_tp_group.all_gather_object((families, local_error)))
        if any(payload[0] != families for payload in gathered):
            if local_resource_set is not None:
                local_resource_set.close()
            raise RuntimeError(
                "AttnTP autotune ranks constructed different resource "
                f"families: {gathered}"
            )
        rank_errors = tuple(
            (rank, error)
            for rank, (_families, error) in enumerate(gathered)
            if error is not None
        )
        if rank_errors:
            if local_resource_set is not None:
                local_resource_set.close()
            message = ", ".join(f"rank{rank}={error}" for rank, error in rank_errors)
            if attn_tp_group.rank_in_group == 0:
                logger.warning(
                    "AttnTP fused norm resource families %s are unavailable: %s",
                    families,
                    message,
                )
            for family in families:
                resource_errors[family] = message
            continue
        if local_resource_set is None:
            raise RuntimeError(
                f"AttnTP resource families {families} returned no resource"
            )
        resource_sets.append(local_resource_set)
        for family in families:
            if any(candidate.family == family for candidate in candidates):
                resource_sets_by_family[family] = local_resource_set
    candidates = tuple(
        candidate
        for candidate in candidates
        if candidate.family in resource_sets_by_family
    )
    if not candidates:
        raise RuntimeError("every AttnTP fused norm resource family is unavailable")

    def gather_candidate_orders(candidate_ids):
        return attn_tp_group.all_gather_object(candidate_ids)

    def gather_errors(stage, candidate_id, local_error):
        gathered = tuple(
            attn_tp_group.all_gather_object((stage, candidate_id, local_error))
        )
        expected = (stage, candidate_id)
        if any(payload[:2] != expected for payload in gathered):
            raise RuntimeError(
                f"AttnTP autotune ranks entered different candidate stages: {gathered}"
            )
        errors = tuple(payload[2] for payload in gathered)
        if attn_tp_group.rank_in_group == 0 and any(
            error is not None for error in errors
        ):
            logger.warning(
                "AttnTP fused norm candidate %s failed during %s: %s",
                candidate_id,
                stage,
                ", ".join(
                    f"rank{rank}={error}"
                    for rank, error in enumerate(errors)
                    if error is not None
                ),
            )
        return errors

    def gather_samples(local_samples):
        return attn_tp_group.all_gather_object(local_samples)

    winners = {}
    try:
        for workload in keys:
            state = build_production_benchmark_state(
                workload,
                rank=attn_tp_group.rank_in_group,
                device=device,
                device_group=attn_tp_group.device_group,
            )

            def prepare(candidate):
                try:
                    resource_set = resource_sets_by_family[candidate.family]
                except KeyError as error:
                    details = resource_errors.get(
                        candidate.family,
                        "resource family was not constructed",
                    )
                    raise RuntimeError(
                        "AttnTP fused norm resource is unavailable for "
                        f"{candidate.family}: {details}"
                    ) from error
                tile_layout = (
                    resource_set.tile_layout_for(workload, candidate)
                    if candidate.family == "tile_pipeline"
                    else None
                )
                return prepare_production_candidate(
                    workload,
                    candidate,
                    group=attn_tp_group.cpu_group,
                    device=device,
                    resource=resource_set.resource_for(candidate),
                    tile_arena_layout=tile_layout,
                )

            def measure(candidate_runner, candidate, *, warmup, iterations):
                return measure_prepared_production_candidate(
                    workload,
                    candidate,
                    candidate_runner,
                    state,
                    cpu_group=attn_tp_group.cpu_group,
                    warmup=warmup,
                    iterations=iterations,
                )

            winners[workload] = tune_production_workload(
                workload,
                candidates,
                coarse_family_finalists=_PRODUCTION_FINALISTS_PER_FAMILY,
                gather_candidate_orders_fn=gather_candidate_orders,
                prepare_fn=prepare,
                gather_errors_fn=gather_errors,
                coarse_measure_fn=lambda runner, candidate: measure(
                    runner,
                    candidate,
                    warmup=_PRODUCTION_COARSE_WARMUP,
                    iterations=_PRODUCTION_COARSE_ITERATIONS,
                ),
                final_measure_fn=lambda runner, candidate: measure(
                    runner,
                    candidate,
                    warmup=_PRODUCTION_FINAL_WARMUP,
                    iterations=_PRODUCTION_FINAL_ITERATIONS,
                ),
                gather_samples_fn=gather_samples,
            )
    finally:
        for resource_set in reversed(resource_sets):
            resource_set.close()
        torch.cuda.synchronize(device)
    return winners


def load_or_tune_attntp_manifest(
    path: str | Path,
    identity: ManifestIdentity[AttnTPFusedNormWorkloadKey],
    *,
    local_tune_fn: Callable[
        [],
        Mapping[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner],
    ],
    synchronize_cache_fn: Callable[
        [Mapping[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner] | None],
        Mapping[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner] | None,
    ],
    gather_fn: Callable[
        [Mapping[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner]],
        Iterable[Mapping[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner]],
    ],
    publish: bool,
) -> dict[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner]:
    """Bind AttnTP winner codecs and scoring to the shared v3 autotune engine."""

    return load_or_tune_coordinated_manifest(
        path,
        identity,
        local_tune_fn=local_tune_fn,
        synchronize_cache_fn=synchronize_cache_fn,
        gather_fn=gather_fn,
        score_fn=_winner_score,
        encode_winner=lambda winner: winner.to_dict(),
        decode_winner=AttnTPFusedNormWinner.from_dict,
        publish=publish,
    )


@dataclass(frozen=True)
class AttnTPFusedNormRegistry:
    winners: Mapping[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner]
    row_buckets_by_shape: Mapping[_ShapeKey, tuple[int, ...]]

    @classmethod
    def build(
        cls,
        *,
        required_keys: Iterable[AttnTPFusedNormWorkloadKey],
        winners: Mapping[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner],
    ) -> AttnTPFusedNormRegistry:
        required = tuple(required_keys)
        if not required:
            raise ValueError("at least one workload key is required")
        if len(required) != len(set(required)):
            raise ValueError("required workload keys must be unique")

        required_set = set(required)
        winner_keys = set(winners)
        missing = required_set - winner_keys
        unexpected = winner_keys - required_set
        if missing or unexpected:
            details = []
            if missing:
                details.append(
                    "no winner for "
                    + ", ".join(sorted(key.encode() for key in missing))
                )
            if unexpected:
                details.append(
                    "unexpected winner for "
                    + ", ".join(sorted(key.encode() for key in unexpected))
                )
            raise RuntimeError("; ".join(details))

        copied_winners = {}
        buckets: dict[_ShapeKey, list[int]] = {}
        for key in required:
            winner = winners[key]
            if winner.row_bucket != key.row_bucket:
                raise ValueError(
                    "winner row bucket does not match workload key: "
                    f"{winner.row_bucket} != {key.row_bucket}"
                )
            if winner.candidate.output_mode is not key.output_mode:
                raise ValueError(
                    "winner output mode does not match workload key: "
                    f"{winner.candidate.output_mode.value} != {key.output_mode.value}"
                )
            copied_winners[key] = winner
            buckets.setdefault(key.shape_key, []).append(key.row_bucket)

        return cls(
            winners=MappingProxyType(copied_winners),
            row_buckets_by_shape=MappingProxyType(
                {
                    shape: tuple(sorted(shape_buckets))
                    for shape, shape_buckets in buckets.items()
                }
            ),
        )

    def resolve(self, key: AttnTPFusedNormWorkloadKey) -> AttnTPFusedNormWinner:
        try:
            return self.winners[key]
        except KeyError as exc:
            raise RuntimeError(
                f"fused AttnTP norm winner is missing for {key.encode()}"
            ) from exc

    def resolve_runtime(
        self,
        *,
        phase: str,
        topology: str,
        attn_tp_size: int,
        hidden_size: int,
        rows: int,
        execution: str,
    ) -> tuple[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner]:
        return self.resolve_runtime_for_mode(
            phase=phase,
            topology=topology,
            attn_tp_size=attn_tp_size,
            hidden_size=hidden_size,
            rows=rows,
            execution=execution,
            output_mode=resolve_current_output_mode(topology),
        )

    def resolve_runtime_for_mode(
        self,
        *,
        phase: str,
        topology: str,
        attn_tp_size: int,
        hidden_size: int,
        rows: int,
        execution: str,
        output_mode: OutputMode,
    ) -> tuple[AttnTPFusedNormWorkloadKey, AttnTPFusedNormWinner]:
        if type(rows) is not int or rows <= 0:
            raise ValueError("runtime rows must be a positive integer")
        requested = AttnTPFusedNormWorkloadKey(
            phase=phase,
            topology=topology,
            attn_tp_size=attn_tp_size,
            hidden_size=hidden_size,
            row_bucket=rows,
            execution=execution,
            output_mode=output_mode,
        )
        if requested in self.winners:
            return requested, self.winners[requested]

        buckets = self.row_buckets_by_shape.get(requested.shape_key, ())
        index = bisect_left(buckets, rows)
        if index == len(buckets):
            if buckets:
                raise RuntimeError(
                    "fused AttnTP norm runtime rows exceed the largest prepared "
                    f"row bucket: rows={rows}, largest={buckets[-1]}, "
                    f"shape={requested.encode()}"
                )
            raise RuntimeError(
                "fused AttnTP norm has no prepared row buckets for "
                f"{requested.encode()}"
            )

        key = AttnTPFusedNormWorkloadKey(
            phase=phase,
            topology=topology,
            attn_tp_size=attn_tp_size,
            hidden_size=hidden_size,
            row_bucket=buckets[index],
            execution=execution,
            output_mode=output_mode,
        )
        return key, self.resolve(key)
