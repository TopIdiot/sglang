"""Candidate-space contracts shared by AttnTP production and benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

from sglang.jit_kernel.attntp_fused_norm.tuning import (
    AttnTPFusedNormCandidate,
    AttnTPFusedNormWorkloadKey,
)
from sglang.jit_kernel.attntp_fused_norm.tile_tuning import (
    ConsumerCohorts,
    TilePipelineSchedule,
    TilePipelineShape,
    TilePipelineTuning,
    WeightPlacement,
    build_shape_space,
)

_SUPPORTED_PHASES = ("prefill", "decode")
_SUPPORTED_BACKENDS = ("direct_symm", "ipc_owner_pull", "ipc_source_push")
_SUPPORTED_ATTN_TP_SIZES = (2, 4, 8)
_SUPPORTED_HIDDEN_SIZES = (2048, 4096)
_SUPPORTED_BLOCK_SIZES = (128, 256, 512)
_SUPPORTED_SIGNAL_BACKOFFS = (32, 64, 128, 256)
_SUPPORTED_PREFILL_ROWS_PER_TILE = (1, 2, 4, 8)
_SUPPORTED_RESIDENT_BLOCKS = tuple(range(2, 9))
_SUPPORTED_SEARCH_PROFILES = ("quick", "balanced", "broad")
_PRODUCTION_BLOCK_SIZES = {
    "quick": (256,),
    "balanced": (128, 256, 512),
    "broad": (128, 256, 512),
}
_PRODUCTION_SIGNAL_BACKOFFS = {
    "quick": (32,),
    "balanced": (32, 64),
    "broad": (32, 64, 128, 256),
}
_PRODUCTION_PREFILL_ROWS_PER_TILE = {
    "quick": (4,),
    "balanced": (2, 4, 8),
    "broad": (1, 2, 4, 8, 16),
}
_PRODUCTION_RESIDENT_BLOCKS = {
    "quick": (2, 4, 8),
    "balanced": tuple(range(2, 9)),
    "broad": tuple(range(2, 9)),
}
_PRODUCTION_RING_STAGES = {
    "prefill": {
        "quick": (32,),
        "balanced": (32, 64),
        "broad": (16, 32, 64, 128),
    },
    "decode": {
        "quick": (4,),
        "balanced": (2, 4, 8),
        "broad": (2, 4, 8, 16),
    },
}
_PRODUCTION_DECODE_ROWS_PER_TILE = {
    "quick": (1,),
    "balanced": (1, 2),
    "broad": (1, 2, 4),
}
_PRODUCTION_LAUNCH_BOUNDS = {
    "quick": (0,),
    "balanced": (0, 3),
    "broad": (0, 1, 2, 3, 4),
}


def _require_member(name: str, value, supported: tuple) -> None:
    if value not in supported:
        raise ValueError(f"{name} must be one of {supported}, got {value!r}")


def _require_search_profile(profile: str) -> None:
    _require_member("profile", profile, _SUPPORTED_SEARCH_PROFILES)


@dataclass(frozen=True)
class BackendCandidateSpec:
    phase: str
    backend: str
    attn_tp_size: int
    hidden_size: int
    block_size: int
    signal_backoff: int
    rows_per_tile: int
    blocks_per_sm: int

    def __post_init__(self) -> None:
        _require_member("phase", self.phase, _SUPPORTED_PHASES)
        _require_member("backend", self.backend, _SUPPORTED_BACKENDS)
        _require_member(
            "attn_tp_size",
            self.attn_tp_size,
            _SUPPORTED_ATTN_TP_SIZES,
        )
        _require_member(
            "hidden_size",
            self.hidden_size,
            _SUPPORTED_HIDDEN_SIZES,
        )
        _require_member("block_size", self.block_size, _SUPPORTED_BLOCK_SIZES)
        if self.backend == "ipc_owner_pull":
            if self.signal_backoff != 0:
                raise ValueError("ipc_owner_pull does not consume signal_backoff")
        else:
            _require_member(
                "signal_backoff",
                self.signal_backoff,
                _SUPPORTED_SIGNAL_BACKOFFS,
            )
        _require_member(
            "blocks_per_sm",
            self.blocks_per_sm,
            _SUPPORTED_RESIDENT_BLOCKS,
        )

        vector_denominator = self.block_size * 8
        if (
            self.hidden_size % vector_denominator != 0
            or self.hidden_size // vector_denominator not in (1, 2, 4)
        ):
            raise ValueError(
                f"block_size {self.block_size} is unsupported for hidden "
                f"size {self.hidden_size}"
            )

        if self.backend == "direct_symm":
            if self.rows_per_tile != 0:
                raise ValueError("direct_symm does not consume rows_per_tile")
        elif self.backend == "ipc_owner_pull":
            if self.rows_per_tile != 0:
                raise ValueError("ipc_owner_pull does not consume rows_per_tile")
        elif self.phase == "prefill":
            _require_member(
                "rows_per_tile",
                self.rows_per_tile,
                _SUPPORTED_PREFILL_ROWS_PER_TILE,
            )
        elif self.rows_per_tile != 0:
            raise ValueError("decode ipc_source_push does not consume rows_per_tile")

    @property
    def compile_id(self) -> str:
        return (
            f"{self.phase}-n{self.attn_tp_size}-h{self.hidden_size}-"
            f"{self.backend}-b{self.block_size}-wait{self.signal_backoff}-"
            f"tile{self.rows_per_tile}"
        )

    @property
    def stable_id(self) -> str:
        return f"{self.compile_id}-grid{self.blocks_per_sm}"

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.stable_id,
            "compile_id": self.compile_id,
            "phase": self.phase,
            "backend": self.backend,
            "attn_tp_size": self.attn_tp_size,
            "hidden_size": self.hidden_size,
            "block_size": self.block_size,
            "signal_backoff": self.signal_backoff,
            "rows_per_tile": self.rows_per_tile,
            "blocks_per_sm": self.blocks_per_sm,
        }


@dataclass(frozen=True)
class BackendCandidateRejection:
    candidate_id: str
    phase: str
    backend: str
    reason: str

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        _require_member("phase", self.phase, _SUPPORTED_PHASES)
        _require_member("backend", self.backend, _SUPPORTED_BACKENDS)
        if not self.reason:
            raise ValueError("reason must not be empty")

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "phase": self.phase,
            "backend": self.backend,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class BackendCandidateSpace:
    candidates: tuple[BackendCandidateSpec, ...]
    rejections: tuple[BackendCandidateRejection, ...]


def backend_spec_to_production_candidate(
    spec: BackendCandidateSpec,
    *,
    output_mode,
) -> AttnTPFusedNormCandidate:
    if not isinstance(spec, BackendCandidateSpec):
        raise ValueError("spec must be a BackendCandidateSpec")
    return AttnTPFusedNormCandidate(
        family=spec.backend,
        output_mode=output_mode,
        parameters=(
            ("block_size", spec.block_size),
            ("blocks_per_sm", spec.blocks_per_sm),
            ("rows_per_tile", spec.rows_per_tile),
            ("signal_backoff", spec.signal_backoff),
        ),
    )


def backend_spec_from_production_candidate(
    candidate: AttnTPFusedNormCandidate,
    *,
    phase: str,
    attn_tp_size: int,
    hidden_size: int,
) -> BackendCandidateSpec:
    if not isinstance(candidate, AttnTPFusedNormCandidate):
        raise ValueError("candidate must be an AttnTPFusedNormCandidate")
    if candidate.family not in _SUPPORTED_BACKENDS:
        raise ValueError(
            "backend candidate family must be direct_symm, ipc_owner_pull, "
            "or ipc_source_push"
        )
    parameters = dict(candidate.parameters)
    expected = {
        "block_size",
        "blocks_per_sm",
        "rows_per_tile",
        "signal_backoff",
    }
    if set(parameters) != expected or any(
        type(parameters[name]) is not int for name in expected
    ):
        raise ValueError(
            "backend candidate parameters must contain integer "
            f"{tuple(sorted(expected))}"
        )
    return BackendCandidateSpec(
        phase=phase,
        backend=candidate.family,
        attn_tp_size=attn_tp_size,
        hidden_size=hidden_size,
        block_size=parameters["block_size"],
        signal_backoff=parameters["signal_backoff"],
        rows_per_tile=parameters["rows_per_tile"],
        blocks_per_sm=parameters["blocks_per_sm"],
    )


def tile_tuning_to_production_candidate(
    tuning: TilePipelineTuning,
    *,
    output_mode,
) -> AttnTPFusedNormCandidate:
    if not isinstance(tuning, TilePipelineTuning):
        raise ValueError("tuning must be a TilePipelineTuning")
    shape = tuning.shape
    schedule = tuning.schedule
    return AttnTPFusedNormCandidate(
        family="tile_pipeline",
        output_mode=output_mode,
        parameters=(
            ("block_size", shape.block_size),
            ("blocks_per_sm", schedule.blocks_per_sm),
            ("consumer_cohorts", shape.consumer_cohorts.value),
            ("launch_bounds_min_blocks", shape.launch_bounds_min_blocks),
            ("producer_blocks_per_sm", schedule.producer_blocks_per_sm),
            ("ring_stages", shape.ring_stages),
            ("rows_per_tile", shape.rows_per_tile),
            ("signal_backoff", shape.signal_backoff),
            ("weight_placement", shape.weight_placement.value),
        ),
    )


def tile_tuning_from_production_candidate(
    candidate: AttnTPFusedNormCandidate,
) -> TilePipelineTuning:
    if not isinstance(candidate, AttnTPFusedNormCandidate):
        raise ValueError("candidate must be an AttnTPFusedNormCandidate")
    if candidate.family != "tile_pipeline":
        raise ValueError("tile tuning requires a tile_pipeline candidate")
    parameters = dict(candidate.parameters)
    expected = {
        "block_size",
        "blocks_per_sm",
        "consumer_cohorts",
        "launch_bounds_min_blocks",
        "producer_blocks_per_sm",
        "ring_stages",
        "rows_per_tile",
        "signal_backoff",
        "weight_placement",
    }
    if set(parameters) != expected:
        raise ValueError(
            f"tile candidate parameters must contain {tuple(sorted(expected))}"
        )
    integer_names = expected - {"weight_placement"}
    if any(type(parameters[name]) is not int for name in integer_names):
        raise ValueError("tile candidate numeric parameters must be integers")
    if not isinstance(parameters["weight_placement"], str):
        raise ValueError("tile candidate weight_placement must be a string")
    return TilePipelineTuning(
        shape=TilePipelineShape(
            rows_per_tile=parameters["rows_per_tile"],
            ring_stages=parameters["ring_stages"],
            block_size=parameters["block_size"],
            signal_backoff=parameters["signal_backoff"],
            launch_bounds_min_blocks=parameters["launch_bounds_min_blocks"],
            consumer_cohorts=ConsumerCohorts(parameters["consumer_cohorts"]),
            weight_placement=WeightPlacement(parameters["weight_placement"]),
        ),
        schedule=TilePipelineSchedule(
            blocks_per_sm=parameters["blocks_per_sm"],
            producer_blocks_per_sm=parameters["producer_blocks_per_sm"],
        ),
    )


def build_production_non_tile_candidates(
    workload: AttnTPFusedNormWorkloadKey,
    *,
    profile: str = "balanced",
) -> tuple[AttnTPFusedNormCandidate, ...]:
    """Build the bounded non-Tile production search space."""

    if not isinstance(workload, AttnTPFusedNormWorkloadKey):
        raise ValueError("workload must be an AttnTPFusedNormWorkloadKey")
    _require_search_profile(profile)
    space = build_non_tile_candidate_space(
        phase=workload.phase,
        attn_tp_size=workload.attn_tp_size,
        hidden_size=workload.hidden_size,
        candidates=_SUPPORTED_BACKENDS,
        block_sizes=_PRODUCTION_BLOCK_SIZES[profile],
        signal_backoffs=_PRODUCTION_SIGNAL_BACKOFFS[profile],
        prefill_rows_per_tile=_PRODUCTION_PREFILL_ROWS_PER_TILE[profile],
        resident_blocks=_PRODUCTION_RESIDENT_BLOCKS[profile],
    )
    candidates = {
        backend_spec_to_production_candidate(
            spec,
            output_mode=workload.output_mode,
        )
        for spec in space.candidates
    }
    return tuple(sorted(candidates, key=lambda candidate: candidate.stable_id))


def build_production_tile_shapes(
    workload: AttnTPFusedNormWorkloadKey,
    *,
    profile: str = "balanced",
) -> tuple[TilePipelineShape, ...]:
    """Build the compile-time Tile shapes used by production autotuning."""

    if not isinstance(workload, AttnTPFusedNormWorkloadKey):
        raise ValueError("workload must be an AttnTPFusedNormWorkloadKey")
    if workload.output_mode.value != "replicated":
        raise ValueError("production Tile candidates require replicated output")
    _require_search_profile(profile)
    rows_per_tiles = (
        _PRODUCTION_PREFILL_ROWS_PER_TILE[profile]
        if workload.phase == "prefill"
        else _PRODUCTION_DECODE_ROWS_PER_TILE[profile]
    )
    return build_shape_space(
        phase=workload.phase,
        hidden_size=workload.hidden_size,
        rows_per_tiles=rows_per_tiles,
        ring_stages=_PRODUCTION_RING_STAGES[workload.phase][profile],
        block_sizes=_PRODUCTION_BLOCK_SIZES[profile],
        signal_backoffs=_PRODUCTION_SIGNAL_BACKOFFS[profile],
        launch_bounds_min_blocks=_PRODUCTION_LAUNCH_BOUNDS[profile],
        consumer_cohorts=(ConsumerCohorts.ONE,),
        weight_placements=(WeightPlacement.REGISTER,),
    )


def build_production_tile_candidates(
    workload: AttnTPFusedNormWorkloadKey,
    *,
    shapes: Iterable[TilePipelineShape],
    occupancies: dict[TilePipelineShape, int],
) -> tuple[AttnTPFusedNormCandidate, ...]:
    """Expand compiled Tile shapes into a bounded runtime schedule space."""

    if not isinstance(workload, AttnTPFusedNormWorkloadKey):
        raise ValueError("workload must be an AttnTPFusedNormWorkloadKey")
    ordered_shapes = tuple(sorted(set(shapes)))
    if not ordered_shapes:
        raise ValueError("at least one Tile shape is required")
    missing = set(ordered_shapes) - set(occupancies)
    if missing:
        raise ValueError(f"missing Tile occupancies for {tuple(sorted(missing))}")

    candidates = set()
    for shape in ordered_shapes:
        occupancy = occupancies[shape]
        if type(occupancy) is not int or occupancy < 0:
            raise ValueError("Tile occupancy must be a non-negative integer")
        if occupancy < 2:
            continue
        blocks_per_sm_values = {2, min(4, occupancy), occupancy}
        for blocks_per_sm in sorted(blocks_per_sm_values):
            producer_values = {1, max(1, blocks_per_sm // 2)}
            for producer_blocks_per_sm in sorted(producer_values):
                if producer_blocks_per_sm >= blocks_per_sm:
                    continue
                candidates.add(
                    tile_tuning_to_production_candidate(
                        TilePipelineTuning(
                            shape=shape,
                            schedule=TilePipelineSchedule(
                                blocks_per_sm=blocks_per_sm,
                                producer_blocks_per_sm=(producer_blocks_per_sm),
                            ),
                        ),
                        output_mode=workload.output_mode,
                    )
                )
    if not candidates:
        raise RuntimeError("no production Tile candidate fits kernel occupancy")
    return tuple(sorted(candidates, key=lambda candidate: candidate.stable_id))


def production_candidate_compile_id(
    workload: AttnTPFusedNormWorkloadKey,
    candidate: AttnTPFusedNormCandidate,
) -> str:
    """Return the CUDA-module identity, excluding runtime launch schedules."""

    if not isinstance(workload, AttnTPFusedNormWorkloadKey):
        raise ValueError("workload must be an AttnTPFusedNormWorkloadKey")
    if not isinstance(candidate, AttnTPFusedNormCandidate):
        raise ValueError("candidate must be an AttnTPFusedNormCandidate")
    if candidate.output_mode is not workload.output_mode:
        raise ValueError("candidate output mode does not match workload")
    if candidate.family in _SUPPORTED_BACKENDS:
        return backend_spec_from_production_candidate(
            candidate,
            phase=workload.phase,
            attn_tp_size=workload.attn_tp_size,
            hidden_size=workload.hidden_size,
        ).compile_id
    if candidate.family == "tile_pipeline":
        shape = tile_tuning_from_production_candidate(candidate).shape
        shape_payload = json.dumps(
            shape.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            f"{workload.phase}-n{workload.attn_tp_size}-"
            f"h{workload.hidden_size}-tile_pipeline-{shape_payload}"
        )
    raise ValueError(f"unsupported production candidate family: {candidate.family}")


def compile_production_candidate(
    workload: AttnTPFusedNormWorkloadKey,
    candidate: AttnTPFusedNormCandidate,
) -> int:
    """Compile one existing JIT module and return its occupancy."""

    production_candidate_compile_id(workload, candidate)
    if candidate.family in _SUPPORTED_BACKENDS:
        spec = backend_spec_from_production_candidate(
            candidate,
            phase=workload.phase,
            attn_tp_size=workload.attn_tp_size,
            hidden_size=workload.hidden_size,
        )
        from sglang.jit_kernel.attntp_fused_norm import (
            ipc as ipc_norm,
            symm as symm_norm,
        )

        common = (
            workload.attn_tp_size,
            workload.hidden_size,
            workload.output_mode,
            ipc_norm.NormInternalPrecision.FULL_FP32,
        )
        if candidate.family == "direct_symm":
            factory = (
                symm_norm._jit_prefill_attntp_fused_symm_norm_module
                if workload.phase == "prefill"
                else symm_norm._jit_decode_attntp_fused_symm_norm_module
            )
            module = factory(
                *common,
                spec.block_size,
                spec.signal_backoff,
            )
        elif candidate.family == "ipc_owner_pull":
            factory = (
                ipc_norm._jit_prefill_attntp_owner_pull_norm_module
                if workload.phase == "prefill"
                else ipc_norm._jit_decode_attntp_owner_pull_norm_module
            )
            module = factory(
                *common,
                spec.block_size,
            )
        elif workload.phase == "prefill":
            module = ipc_norm._jit_prefill_attntp_source_push_norm_module(
                *common,
                spec.rows_per_tile,
                spec.block_size,
                spec.signal_backoff,
            )
        else:
            module = ipc_norm._jit_decode_attntp_source_push_norm_module(
                *common,
                spec.block_size,
                spec.signal_backoff,
            )
    elif candidate.family == "tile_pipeline":
        from sglang.jit_kernel.attntp_fused_norm import tile

        shape = tile_tuning_from_production_candidate(candidate).shape
        module = tile._jit_attntp_tile_pipeline_module(
            workload.attn_tp_size,
            workload.hidden_size,
            workload.phase,
            shape.rows_per_tile,
            shape.ring_stages,
            shape.block_size,
            shape.signal_backoff,
            shape.launch_bounds_min_blocks,
            shape.consumer_cohorts,
            shape.weight_placement,
        )
    else:
        raise ValueError(f"unsupported production candidate family: {candidate.family}")
    occupancy = int(module.get_max_occupancy())
    if occupancy < 0:
        raise RuntimeError("compiled production candidate reported negative occupancy")
    return occupancy


def prepare_production_candidate(
    workload: AttnTPFusedNormWorkloadKey,
    candidate: AttnTPFusedNormCandidate,
    *,
    group,
    device,
    resource=None,
    tile_arena_layout=None,
):
    """Decode a production candidate into an existing prepared runner."""

    production_candidate_compile_id(workload, candidate)
    from sglang.jit_kernel.attntp_fused_norm import ipc as ipc_norm

    norm_spec = ipc_norm.AttnTPNormSpec(
        attn_tp_size=workload.attn_tp_size,
        hidden_size=workload.hidden_size,
        output_mode=workload.output_mode,
        internal_precision=ipc_norm.NormInternalPrecision.FULL_FP32,
    )
    if candidate.family in _SUPPORTED_BACKENDS:
        backend_spec = backend_spec_from_production_candidate(
            candidate,
            phase=workload.phase,
            attn_tp_size=workload.attn_tp_size,
            hidden_size=workload.hidden_size,
        )
        common = {
            "spec": norm_spec,
            "block_size": backend_spec.block_size,
            "signal_backoff": backend_spec.signal_backoff,
            "blocks_per_sm": backend_spec.blocks_per_sm,
        }
        if candidate.family == "direct_symm":
            from sglang.jit_kernel.attntp_fused_norm import symm as symm_norm

            runner_cls = (
                symm_norm.PrefillAttnTPFusedSymmNormRunner
                if workload.phase == "prefill"
                else symm_norm.DecodeAttnTPFusedSymmNormRunner
            )
            rows_name = "capacity" if workload.phase == "prefill" else "max_rows"
        else:
            runner_cls = (
                ipc_norm.PrefillAttnTPFusedIPCNormRunner
                if workload.phase == "prefill"
                else ipc_norm.DecodeAttnTPFusedIPCNormRunner
            )
            rows_name = "capacity" if workload.phase == "prefill" else "max_rows"
            if workload.phase == "prefill":
                common["algorithm"] = (
                    ipc_norm.PrefillCommunicationAlgorithm.OWNER_PULL
                    if candidate.family == "ipc_owner_pull"
                    else ipc_norm.PrefillCommunicationAlgorithm.SOURCE_PUSH
                )
            else:
                common["algorithm"] = (
                    ipc_norm.DecodeCommunicationAlgorithm.OWNER_PULL
                    if candidate.family == "ipc_owner_pull"
                    else ipc_norm.DecodeCommunicationAlgorithm.SOURCE_PUSH
                )
            if workload.phase == "prefill" and candidate.family == "ipc_source_push":
                common["rows_per_tile"] = backend_spec.rows_per_tile
        common[rows_name] = workload.row_bucket
        if resource is None:
            return runner_cls(group=group, device=device, **common)
        return runner_cls.from_resource(resource=resource, **common)

    if candidate.family == "tile_pipeline":
        from sglang.jit_kernel.attntp_fused_norm import tile

        tuning = tile_tuning_from_production_candidate(candidate)
        shape = tuning.shape
        schedule = tuning.schedule
        runner_cls = (
            tile.PrefillAttnTPReplicatedTilePipelineRunner
            if workload.phase == "prefill"
            else tile.DecodeAttnTPReplicatedTilePipelineRunner
        )
        common = {
            "spec": norm_spec,
            "rows_per_tile": shape.rows_per_tile,
            "ring_stages": shape.ring_stages,
            "block_size": shape.block_size,
            "signal_backoff": shape.signal_backoff,
            "launch_bounds_min_blocks": shape.launch_bounds_min_blocks,
            "consumer_cohorts": shape.consumer_cohorts,
            "weight_placement": shape.weight_placement,
            "blocks_per_sm": schedule.blocks_per_sm,
            "producer_blocks_per_sm": schedule.producer_blocks_per_sm,
        }
        rows_name = "capacity" if workload.phase == "prefill" else "rows"
        common[rows_name] = workload.row_bucket
        if resource is None:
            return runner_cls(group=group, device=device, **common)
        if tile_arena_layout is None:
            raise ValueError("shared Tile production runner requires an arena layout")
        return runner_cls.from_resource(
            resource=resource,
            arena_layout=tile_arena_layout,
            **common,
        )
    raise ValueError(f"unsupported production candidate family: {candidate.family}")


def run_prepared_production_candidate_out(
    *,
    phase: str,
    topology: str,
    candidate: AttnTPFusedNormCandidate,
    runner,
    partial,
    residual,
    o_norm_weight,
    post_norm_weight,
    output,
    residual_out,
    actual_rows,
    lane_rotation,
    o_norm_eps: float,
    post_norm_eps: float,
) -> None:
    """Launch one existing prepared runner with a uniform production ABI."""

    if phase not in _SUPPORTED_PHASES:
        raise ValueError(f"phase must be one of {_SUPPORTED_PHASES}")
    if topology not in ("tp", "dp", "cp"):
        raise ValueError("topology must be tp, dp, or cp")
    if phase == "decode":
        if actual_rows is not None or lane_rotation is not None:
            raise ValueError("Decode candidates do not accept Prefill metadata")
    else:
        if actual_rows is None or lane_rotation is None:
            raise ValueError("Prefill candidates require actual_rows and lane_rotation")
    if not isinstance(candidate, AttnTPFusedNormCandidate):
        raise ValueError("candidate must be an AttnTPFusedNormCandidate")
    if candidate.family in ("direct_symm", "tile_pipeline"):
        runner.input_view[: partial.shape[0]].copy_(partial)
        if phase == "prefill":
            runner.run_out(
                residual,
                o_norm_weight,
                post_norm_weight,
                output,
                residual_out,
                actual_rows,
                lane_rotation,
                o_norm_eps,
                post_norm_eps,
            )
        else:
            runner.run_out(
                residual,
                o_norm_weight,
                post_norm_weight,
                output,
                residual_out,
                o_norm_eps,
                post_norm_eps,
            )
        return
    if candidate.family in ("ipc_owner_pull", "ipc_source_push"):
        if phase == "prefill":
            runner.run_out(
                partial,
                residual,
                o_norm_weight,
                post_norm_weight,
                output,
                residual_out,
                actual_rows,
                lane_rotation,
                o_norm_eps,
                post_norm_eps,
            )
        else:
            runner.run_out(
                partial,
                residual,
                o_norm_weight,
                post_norm_weight,
                output,
                residual_out,
                o_norm_eps,
                post_norm_eps,
            )
        return
    raise ValueError(f"unsupported production candidate family: {candidate.family}")


def filter_production_candidates_by_occupancy(
    workload: AttnTPFusedNormWorkloadKey,
    candidates: Iterable[AttnTPFusedNormCandidate],
    *,
    occupancies: dict[str, int],
) -> tuple[AttnTPFusedNormCandidate, ...]:
    """Filter runtime schedules using shared compile-time occupancy results."""

    values = tuple(candidates)
    if not values:
        raise ValueError("at least one production candidate is required")
    missing = {
        production_candidate_compile_id(workload, candidate) for candidate in values
    } - set(occupancies)
    if missing:
        raise RuntimeError(
            "missing occupancy for production compile IDs: "
            + ", ".join(sorted(missing))
        )

    accepted = []
    for candidate in values:
        compile_id = production_candidate_compile_id(workload, candidate)
        occupancy = occupancies[compile_id]
        if type(occupancy) is not int or occupancy < 0:
            raise ValueError("production occupancy must be a non-negative integer")
        if occupancy < 2:
            continue
        if candidate.family == "tile_pipeline":
            blocks_per_sm = tile_tuning_from_production_candidate(
                candidate
            ).schedule.blocks_per_sm
        elif candidate.family in _SUPPORTED_BACKENDS:
            blocks_per_sm = backend_spec_from_production_candidate(
                candidate,
                phase=workload.phase,
                attn_tp_size=workload.attn_tp_size,
                hidden_size=workload.hidden_size,
            ).blocks_per_sm
        else:
            raise ValueError(
                f"unsupported production candidate family: {candidate.family}"
            )
        if blocks_per_sm <= occupancy:
            accepted.append(candidate)
    if not accepted:
        raise RuntimeError("every production candidate was rejected by occupancy")
    return tuple(sorted(accepted, key=lambda candidate: candidate.stable_id))


def _raw_candidate_id(
    *,
    phase: str,
    backend: str,
    attn_tp_size: int,
    hidden_size: int,
    block_size,
    signal_backoff,
    rows_per_tile,
    blocks_per_sm,
) -> str:
    return (
        f"{phase}-n{attn_tp_size}-h{hidden_size}-{backend}-"
        f"b{block_size}-wait{signal_backoff}-tile{rows_per_tile}-"
        f"grid{blocks_per_sm}"
    )


def build_non_tile_candidate_space(
    *,
    phase: str,
    attn_tp_size: int,
    hidden_size: int,
    candidates: Iterable[str],
    block_sizes: Iterable[int],
    signal_backoffs: Iterable[int],
    prefill_rows_per_tile: Iterable[int],
    resident_blocks: Iterable[int],
) -> BackendCandidateSpace:
    _require_member("phase", phase, _SUPPORTED_PHASES)
    _require_member(
        "attn_tp_size",
        attn_tp_size,
        _SUPPORTED_ATTN_TP_SIZES,
    )
    _require_member("hidden_size", hidden_size, _SUPPORTED_HIDDEN_SIZES)

    backend_values = tuple(sorted(set(candidates)))
    if not backend_values:
        raise ValueError("at least one non-Tile candidate is required")
    unknown_backends = set(backend_values) - set(_SUPPORTED_BACKENDS)
    if unknown_backends:
        raise ValueError(
            "non-Tile candidates must be direct_symm, ipc_owner_pull, "
            "or ipc_source_push, "
            f"got {tuple(sorted(unknown_backends))}"
        )

    blocks = tuple(sorted(set(block_sizes)))
    backoffs = tuple(sorted(set(signal_backoffs)))
    prefill_tiles = tuple(sorted(set(prefill_rows_per_tile)))
    resident_values = tuple(sorted(set(resident_blocks)))

    specs: set[BackendCandidateSpec] = set()
    rejections: dict[str, BackendCandidateRejection] = {}
    for backend in backend_values:
        rows_per_tiles = (
            prefill_tiles
            if backend == "ipc_source_push" and phase == "prefill"
            else (0,)
        )
        backend_backoffs = (0,) if backend == "ipc_owner_pull" else backoffs
        for block_size in blocks:
            for signal_backoff in backend_backoffs:
                for rows_per_tile in rows_per_tiles:
                    for blocks_per_sm in resident_values:
                        candidate_id = _raw_candidate_id(
                            phase=phase,
                            backend=backend,
                            attn_tp_size=attn_tp_size,
                            hidden_size=hidden_size,
                            block_size=block_size,
                            signal_backoff=signal_backoff,
                            rows_per_tile=rows_per_tile,
                            blocks_per_sm=blocks_per_sm,
                        )
                        try:
                            specs.add(
                                BackendCandidateSpec(
                                    phase=phase,
                                    backend=backend,
                                    attn_tp_size=attn_tp_size,
                                    hidden_size=hidden_size,
                                    block_size=block_size,
                                    signal_backoff=signal_backoff,
                                    rows_per_tile=rows_per_tile,
                                    blocks_per_sm=blocks_per_sm,
                                )
                            )
                        except ValueError as error:
                            rejections[candidate_id] = BackendCandidateRejection(
                                candidate_id=candidate_id,
                                phase=phase,
                                backend=backend,
                                reason=str(error),
                            )
    return BackendCandidateSpace(
        candidates=tuple(sorted(specs, key=lambda spec: spec.stable_id)),
        rejections=tuple(
            rejections[candidate_id] for candidate_id in sorted(rejections)
        ),
    )
