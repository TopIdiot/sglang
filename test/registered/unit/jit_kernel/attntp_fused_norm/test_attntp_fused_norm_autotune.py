import ast
import json
import inspect
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.jit_kernel.attntp_fused_norm.ipc import OutputMode
from sglang.jit_kernel.attntp_fused_norm.autotune import (
    Algorithm,
    Backend,
    CollectiveCandidateRejection,
    CompileResult,
    CompileSpec,
    Measurement,
    TuningShape,
    aggregate_max_rank_samples,
    build_report,
    expand_run_candidates,
    generate_compile_specs,
    measure_collective_candidates,
    order_collective_candidates,
    select_finalists,
    select_winner,
    synchronize_collective_candidate_construction,
    validate_collective_candidate_orders,
)
from sglang.jit_kernel.attntp_fused_norm.candidates import (
    build_production_non_tile_candidates,
    build_production_tile_candidates,
    build_production_tile_shapes,
    compile_production_candidate,
    filter_production_candidates_by_occupancy,
    prepare_production_candidate,
    production_candidate_compile_id,
)
from sglang.jit_kernel.attntp_fused_norm.tuning import (
    PRODUCTION_SEARCH_PROFILE,
    AttnTPProductionCandidateSpace,
    AttnTPProductionCompilation,
    AttnTPFusedNormCandidate,
    AttnTPFusedNormWorkloadKey,
    build_compiled_production_candidate_space,
    compile_production_candidate_occupancies,
    run_local_production_autotune,
    tune_production_workload,
)


def test_production_search_profiles_are_nested_and_default_balanced() -> None:
    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=4096,
        execution="eager",
    )
    profiles = ("quick", "balanced", "broad")
    non_tile = {
        profile: set(build_production_non_tile_candidates(key, profile=profile))
        for profile in profiles
    }
    tile_shapes = {
        profile: set(build_production_tile_shapes(key, profile=profile))
        for profile in profiles
    }

    assert PRODUCTION_SEARCH_PROFILE == "balanced"
    assert set(build_production_non_tile_candidates(key)) == non_tile["balanced"]
    assert set(build_production_tile_shapes(key)) == tile_shapes["balanced"]
    assert non_tile["quick"] < non_tile["balanced"] < non_tile["broad"]
    assert tile_shapes["quick"] < tile_shapes["balanced"] < tile_shapes["broad"]


@pytest.mark.parametrize(
    ("phase", "topology", "expected_updates"),
    (
        ("decode", "tp", ()),
        ("decode", "dp", ()),
        ("prefill", "tp", (("actual_rows", 17),)),
        ("prefill", "dp", (("actual_rows", 17),)),
        (
            "prefill",
            "cp",
            (("actual_rows", 17), ("lane_rotation", 1)),
        ),
    ),
)
def test_production_benchmark_times_only_runtime_metadata_for_its_contract(
    monkeypatch,
    phase: str,
    topology: str,
    expected_updates: tuple[tuple[str, int], ...],
) -> None:
    from sglang.jit_kernel.attntp_fused_norm import candidates as candidate_module
    from sglang.jit_kernel.attntp_fused_norm import tuning as tuning_module

    events = []

    class RecordedScalar:
        def __init__(self, name: str) -> None:
            self.name = name

        def fill_(self, value: int) -> None:
            events.append((self.name, value))

    workload = AttnTPFusedNormWorkloadKey.current(
        phase=phase,
        topology=topology,
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=17,
        execution="graph" if phase == "decode" else "eager",
    )
    candidate = AttnTPFusedNormCandidate(
        family="ipc_source_push",
        output_mode=OutputMode.REPLICATED,
    )
    state = SimpleNamespace(
        partial=object(),
        residual=object(),
        o_norm_weight=object(),
        post_norm_weight=object(),
        output=object(),
        residual_out=object(),
        actual_rows=RecordedScalar("actual_rows"),
        lane_rotation=RecordedScalar("lane_rotation"),
        lane_rotation_value=1,
    )

    monkeypatch.setattr(
        candidate_module,
        "run_prepared_production_candidate_out",
        lambda **_kwargs: events.append(("kernel", None)),
    )

    tuning_module._run_production_benchmark_candidate(
        workload,
        candidate,
        object(),
        state,
        norm_eps=1e-6,
    )

    assert tuple(events[:-1]) == expected_updates
    assert events[-1] == ("kernel", None)


@pytest.mark.parametrize("bad_value", (float("nan"), float("inf")))
def test_production_correctness_gate_rejects_non_finite_outputs(bad_value) -> None:
    from sglang.jit_kernel.attntp_fused_norm import tuning as tuning_module

    state = SimpleNamespace(
        output=torch.tensor([[bad_value]], dtype=torch.float32),
        expected_output=torch.zeros((1, 1), dtype=torch.float32),
        residual_out=torch.zeros((1, 1), dtype=torch.float32),
        expected_residual_out=torch.zeros((1, 1), dtype=torch.float32),
    )

    with pytest.raises(RuntimeError, match="non-finite"):
        tuning_module._validate_production_benchmark_outputs(
            state,
            candidate_id="bad-candidate",
        )


def test_graph_candidate_is_cleared_replayed_and_validated_before_timing(
    monkeypatch,
) -> None:
    from sglang.jit_kernel.attntp_fused_norm import tuning as tuning_module

    events = []

    class Buffer:
        def __init__(self, name):
            self.name = name

        def fill_(self, value):
            events.append((self.name, "fill", value))

    class Graph:
        def replay(self):
            events.append(("graph", "replay"))

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            events.append(("event", "record"))

        def synchronize(self):
            events.append(("event", "synchronize"))

        def elapsed_time(self, _other):
            return 1.0

    workload = AttnTPFusedNormWorkloadKey.current(
        phase="decode",
        topology="tp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=1,
        execution="graph",
    )
    candidate = AttnTPFusedNormCandidate(
        family="ipc_source_push",
        output_mode=OutputMode.REPLICATED,
    )
    state = SimpleNamespace(
        output=Buffer("output"),
        residual_out=Buffer("residual_out"),
        device=torch.device("cuda:0"),
    )
    runner = SimpleNamespace(capture=lambda: nullcontext())

    monkeypatch.setattr(
        tuning_module,
        "_run_production_benchmark_candidate",
        lambda *_args, **_kwargs: events.append(("kernel", "launch")),
    )
    monkeypatch.setattr(
        tuning_module,
        "_validate_production_benchmark_outputs",
        lambda *_args, **_kwargs: events.append(("output", "validate")),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda _graph: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)

    tuning_module.measure_prepared_production_candidate(
        workload,
        candidate,
        runner,
        state,
        cpu_group=object(),
        warmup=1,
        iterations=1,
    )

    assert events.count(("output", "validate")) == 2
    assert len(
        [event for event in events if event[:2] == ("output", "fill")]
    ) == 1
    assert len(
        [event for event in events if event[:2] == ("residual_out", "fill")]
    ) == 1
    assert events.count(("graph", "replay")) == 3


def test_attntp_package_does_not_own_a_second_autotune_engine() -> None:
    package_dir = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/jit_kernel/attntp_fused_norm"
    )
    forbidden_definitions = {
        "_read_manifest",
        "_write_manifest",
        "iter_bounded_compilation",
        "load_or_tune_coordinated_manifest",
        "load_or_tune_manifest",
        "production_compile_worker_count",
        "production_manifest_path",
    }
    forbidden_imports = {
        "concurrent.futures",
        "tempfile",
    }

    for source_path in package_dir.glob("*.py"):
        module = ast.parse(source_path.read_text())
        definitions = {
            node.name
            for node in ast.walk(module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        imports = {
            alias.name
            for node in ast.walk(module)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module
            for node in ast.walk(module)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )

        assert not definitions & forbidden_definitions, source_path
        assert not imports & forbidden_imports, source_path


def make_shape(**overrides) -> TuningShape:
    values = {
        "phase": "prefill",
        "attn_tp_size": 8,
        "hidden_size": 2048,
        "output_mode": "token_scattered",
        "internal_precision": "reference_bf16",
        "execution": "graph",
    }
    values.update(overrides)
    return TuningShape(**values)


def make_spec(
    algorithm: Algorithm,
    *,
    block_size: int = 256,
    signal_backoff: int = 0,
    rows_per_tile: int = 0,
) -> CompileSpec:
    backend = Backend.SYMM if algorithm is Algorithm.SYMM else Backend.IPC
    return CompileSpec(
        shape=make_shape(),
        backend=backend,
        algorithm=algorithm,
        block_size=block_size,
        signal_backoff=signal_backoff,
        rows_per_tile=rows_per_tile,
    )


def test_collective_candidates_have_one_deterministic_order() -> None:
    candidates = (
        expand_run_candidates(
            make_spec(Algorithm.SYMM, signal_backoff=64),
            max_occupancy=2,
            blocks_per_sm=(1, 2),
        )[1],
        expand_run_candidates(
            make_spec(Algorithm.OWNER_PULL),
            max_occupancy=1,
            blocks_per_sm=(1,),
        )[0],
    )

    ordered = order_collective_candidates(reversed(candidates))
    candidate_ids = tuple(candidate.stable_id for candidate in ordered)

    assert candidate_ids == tuple(
        sorted(candidate.stable_id for candidate in candidates)
    )
    assert (
        validate_collective_candidate_orders(
            (candidate_ids, candidate_ids, candidate_ids)
        )
        == candidate_ids
    )
    with pytest.raises(RuntimeError, match="same ordered candidates"):
        validate_collective_candidate_orders(
            (candidate_ids, candidate_ids[:1], candidate_ids)
        )


def test_production_candidate_space_is_grouped_and_deterministic() -> None:
    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=4096,
        execution="eager",
    )

    non_tile = build_production_non_tile_candidates(key)
    shapes = build_production_tile_shapes(key)
    tile = build_production_tile_candidates(
        key,
        shapes=shapes,
        occupancies={shape: 4 for shape in shapes},
    )

    assert {candidate.family for candidate in non_tile} == {
        "direct_symm",
        "ipc_owner_pull",
        "ipc_source_push",
    }
    owner_pull = tuple(
        candidate
        for candidate in non_tile
        if candidate.family == "ipc_owner_pull"
    )
    assert owner_pull
    assert all(
        dict(candidate.parameters)["rows_per_tile"] == 0
        and dict(candidate.parameters)["signal_backoff"] == 0
        for candidate in owner_pull
    )
    assert {candidate.family for candidate in tile} == {"tile_pipeline"}
    assert all(
        candidate.output_mode is OutputMode.REPLICATED
        for candidate in (*non_tile, *tile)
    )
    candidate_ids = tuple(candidate.stable_id for candidate in (*non_tile, *tile))
    assert len(candidate_ids) == len(set(candidate_ids))
    assert non_tile == tuple(
        sorted(non_tile, key=lambda candidate: candidate.stable_id)
    )
    assert tile == tuple(sorted(tile, key=lambda candidate: candidate.stable_id))


def test_production_compile_ids_exclude_runtime_schedule() -> None:
    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=4096,
        execution="eager",
    )
    non_tile = build_production_non_tile_candidates(key)
    shapes = build_production_tile_shapes(key)
    tile = build_production_tile_candidates(
        key,
        shapes=shapes,
        occupancies={shape: 4 for shape in shapes},
    )

    non_tile_by_compile_id = {}
    for candidate in non_tile:
        non_tile_by_compile_id.setdefault(
            production_candidate_compile_id(key, candidate), []
        ).append(candidate)
    tile_by_compile_id = {}
    for candidate in tile:
        tile_by_compile_id.setdefault(
            production_candidate_compile_id(key, candidate), []
        ).append(candidate)

    assert any(len(candidates) > 1 for candidates in non_tile_by_compile_id.values())
    assert any(len(candidates) > 1 for candidates in tile_by_compile_id.values())
    assert all(
        len({candidate.stable_id for candidate in candidates}) == len(candidates)
        for candidates in (
            *non_tile_by_compile_id.values(),
            *tile_by_compile_id.values(),
        )
    )


def test_production_occupancy_filter_uses_shared_compile_results() -> None:
    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=4096,
        execution="eager",
    )
    candidates = build_production_non_tile_candidates(key)
    compile_ids = {
        production_candidate_compile_id(key, candidate) for candidate in candidates
    }

    filtered = filter_production_candidates_by_occupancy(
        key,
        candidates,
        occupancies={compile_id: 4 for compile_id in compile_ids},
    )

    assert filtered
    assert all(
        dict(candidate.parameters)["blocks_per_sm"] <= 4 for candidate in filtered
    )
    with pytest.raises(RuntimeError, match="missing occupancy"):
        filter_production_candidates_by_occupancy(
            key,
            candidates,
            occupancies={},
        )


def test_compile_production_candidate_uses_existing_jit_module(
    monkeypatch,
) -> None:
    from sglang.jit_kernel.attntp_fused_norm import ipc as ipc_norm

    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=4096,
        execution="eager",
    )
    candidate = next(
        candidate
        for candidate in build_production_non_tile_candidates(key)
        if candidate.family == "ipc_source_push"
    )
    calls = []

    class Module:
        @staticmethod
        def get_max_occupancy():
            return 6

    monkeypatch.setattr(
        ipc_norm,
        "_jit_prefill_attntp_source_push_norm_module",
        lambda *args: calls.append(args) or Module(),
    )

    assert compile_production_candidate(key, candidate) == 6
    assert calls == [
        (
            2,
            2048,
            OutputMode.REPLICATED,
            ipc_norm.NormInternalPrecision.FULL_FP32,
            dict(candidate.parameters)["rows_per_tile"],
            dict(candidate.parameters)["block_size"],
            dict(candidate.parameters)["signal_backoff"],
        )
    ]


def test_compile_production_owner_pull_uses_full_fp32_jit_module(
    monkeypatch,
) -> None:
    from sglang.jit_kernel.attntp_fused_norm import ipc as ipc_norm

    key = AttnTPFusedNormWorkloadKey.current(
        phase="decode",
        topology="tp",
        attn_tp_size=4,
        hidden_size=2048,
        row_bucket=64,
        execution="graph",
    )
    candidate = next(
        candidate
        for candidate in build_production_non_tile_candidates(key)
        if candidate.family == "ipc_owner_pull"
    )
    calls = []

    class Module:
        @staticmethod
        def get_max_occupancy():
            return 7

    monkeypatch.setattr(
        ipc_norm,
        "_jit_decode_attntp_owner_pull_norm_module",
        lambda *args: calls.append(args) or Module(),
    )

    assert compile_production_candidate(key, candidate) == 7
    assert calls == [
        (
            4,
            2048,
            OutputMode.REPLICATED,
            ipc_norm.NormInternalPrecision.FULL_FP32,
            dict(candidate.parameters)["block_size"],
        )
    ]


def test_production_compilation_partitions_and_merges_compile_ids() -> None:
    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=4096,
        execution="eager",
    )
    representatives = {}
    for candidate in build_production_non_tile_candidates(key):
        representatives.setdefault(
            production_candidate_compile_id(key, candidate),
            candidate,
        )
    compile_ids = tuple(sorted(representatives))[:4]
    candidates = tuple(representatives[compile_id] for compile_id in compile_ids)
    local_ids = set(compile_ids[::2])
    peer_ids = set(compile_ids[1::2])
    compiled = []

    result = compile_production_candidate_occupancies(
        key,
        candidates,
        participant_index=0,
        participant_count=2,
        cpu_count=8,
        compile_candidate_fn=lambda workload, candidate: (
            compiled.append(production_candidate_compile_id(workload, candidate)) or 6
        ),
        gather_fn=lambda local: (
            local,
            {compile_id: (5, None) for compile_id in peer_ids},
        ),
    )

    assert isinstance(result, AttnTPProductionCompilation)
    assert set(compiled) == local_ids
    assert result.errors == {}
    assert result.occupancies == {
        **{compile_id: 6 for compile_id in local_ids},
        **{compile_id: 5 for compile_id in peer_ids},
    }


def test_compiled_production_space_keeps_all_valid_backend_families() -> None:
    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=4096,
        execution="eager",
    )

    space = build_compiled_production_candidate_space(
        key,
        participant_index=0,
        participant_count=1,
        cpu_count=4,
        compile_candidate_fn=lambda _workload, _candidate: 4,
        gather_fn=lambda local: (local,),
    )

    assert isinstance(space, AttnTPProductionCandidateSpace)
    assert {candidate.family for candidate in space.candidates} == {
        "direct_symm",
        "ipc_owner_pull",
        "ipc_source_push",
        "tile_pipeline",
    }
    assert not space.compilation.errors


def test_production_workload_tuning_refines_each_family_and_uses_p90() -> None:
    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=4096,
        execution="eager",
    )
    candidates = (
        AttnTPFusedNormCandidate(
            family=family,
            output_mode=OutputMode.REPLICATED,
            parameters=(("variant", variant),),
        )
        for family in ("direct_symm", "ipc_source_push")
        for variant in (0, 1)
    )
    samples = {
        "direct_symm:replicated:variant=0": (0.8, 0.8, 0.8),
        "direct_symm:replicated:variant=1": (0.7, 0.7, 0.7),
        "ipc_source_push:replicated:variant=0": (0.6, 0.6, 1.2),
        "ipc_source_push:replicated:variant=1": (0.75, 0.75, 0.75),
    }
    prepared = []
    closed = []

    class Runner:
        def __init__(self, candidate_id: str) -> None:
            self.candidate_id = candidate_id

        def close(self) -> None:
            closed.append(self.candidate_id)

    winner = tune_production_workload(
        key,
        candidates,
        coarse_family_finalists=1,
        gather_candidate_orders_fn=lambda local: (local,),
        prepare_fn=lambda candidate: (
            prepared.append(candidate.stable_id) or Runner(candidate.stable_id)
        ),
        gather_errors_fn=lambda _stage, _candidate_id, error: (error,),
        coarse_measure_fn=lambda runner, _candidate: samples[runner.candidate_id],
        final_measure_fn=lambda runner, _candidate: samples[runner.candidate_id],
        gather_samples_fn=lambda local: (local,),
    )

    assert winner.candidate.stable_id == ("direct_symm:replicated:variant=1")
    assert winner.median_ms == pytest.approx(0.7)
    assert winner.p90_ms == pytest.approx(0.7)
    assert len(prepared) == 6
    assert closed == prepared


def test_local_production_autotune_compiles_once_and_reuses_resources(
    monkeypatch,
) -> None:
    from sglang.jit_kernel.attntp_fused_norm import candidates as candidate_module
    from sglang.jit_kernel.attntp_fused_norm import resources as resource_module
    from sglang.jit_kernel.attntp_fused_norm import tuning as tuning_module

    keys = tuple(
        AttnTPFusedNormWorkloadKey.current(
            phase="prefill",
            topology="cp",
            attn_tp_size=2,
            hidden_size=2048,
            row_bucket=rows,
            execution="eager",
        )
        for rows in (16, 64)
    )
    candidate = AttnTPFusedNormCandidate(
        family="ipc_source_push",
        output_mode=OutputMode.REPLICATED,
    )
    compile_calls = []
    resource_entries = []
    resource_closed = []
    prepared = []

    class ResourceSet:
        def resource_for(self, _candidate):
            return "resource"

        def close(self):
            resource_closed.append(True)

    class Runner:
        def close(self):
            pass

    class Group:
        world_size = 2
        rank_in_group = 0
        cpu_group = "cpu"
        device_group = "device"

        @staticmethod
        def all_gather_object(value):
            return (value,)

    monkeypatch.setattr(
        tuning_module,
        "build_compiled_production_candidate_space",
        lambda workload, **_kwargs: (
            compile_calls.append(workload)
            or AttnTPProductionCandidateSpace(
                candidates=(candidate,),
                compilation=AttnTPProductionCompilation(
                    occupancies={},
                    errors={},
                ),
            )
        ),
    )
    monkeypatch.setattr(
        resource_module,
        "build_candidate_resource_set",
        lambda entries, **_kwargs: (resource_entries.extend(entries) or ResourceSet()),
    )
    monkeypatch.setattr(
        tuning_module,
        "build_production_benchmark_state",
        lambda workload, **_kwargs: SimpleNamespace(rows=workload.row_bucket),
    )
    monkeypatch.setattr(
        candidate_module,
        "prepare_production_candidate",
        lambda workload, _candidate, **_kwargs: (
            prepared.append(workload.row_bucket) or Runner()
        ),
    )
    monkeypatch.setattr(
        tuning_module,
        "measure_prepared_production_candidate",
        lambda workload, _candidate, _runner, _state, **_kwargs: (
            workload.row_bucket / 1000,
        ),
    )
    monkeypatch.setattr("torch.cuda.synchronize", lambda _device: None)

    winners = run_local_production_autotune(
        keys,
        participant_index=0,
        participant_count=1,
        compilation_gather_fn=lambda local: (local,),
        attn_tp_group=Group(),
        device="cuda:0",
    )

    assert compile_calls == [keys[1]]
    assert resource_entries == [
        (keys[0], candidate),
        (keys[1], candidate),
    ]
    assert tuple(winners) == keys
    assert prepared == [16, 16, 64, 64]
    assert resource_closed == [True]


def test_local_production_autotune_rejects_unavailable_resource_family(
    monkeypatch,
    caplog,
) -> None:
    from sglang.jit_kernel.attntp_fused_norm import candidates as candidate_module
    from sglang.jit_kernel.attntp_fused_norm import resources as resource_module
    from sglang.jit_kernel.attntp_fused_norm import tuning as tuning_module

    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=16,
        execution="eager",
    )
    direct = AttnTPFusedNormCandidate(
        family="direct_symm",
        output_mode=OutputMode.REPLICATED,
    )
    ipc = AttnTPFusedNormCandidate(
        family="ipc_source_push",
        output_mode=OutputMode.REPLICATED,
    )
    allocations = []
    closed = []

    class ResourceSet:
        def resource_for(self, _candidate):
            return "ipc-resource"

        def close(self):
            closed.append(True)

    class Runner:
        def close(self):
            pass

    class Group:
        world_size = 2
        rank_in_group = 0
        cpu_group = "cpu"
        device_group = "device"

        @staticmethod
        def all_gather_object(value):
            return (value,)

    monkeypatch.setattr(
        tuning_module,
        "build_compiled_production_candidate_space",
        lambda *_args, **_kwargs: AttnTPProductionCandidateSpace(
            candidates=(direct, ipc),
            compilation=AttnTPProductionCompilation(
                occupancies={},
                errors={"bad-compile-id": "compile failed"},
            ),
        ),
    )

    def build_resources(entries, **_kwargs):
        entries = tuple(entries)
        families = {candidate.family for _workload, candidate in entries}
        allocations.append(families)
        if "direct_symm" in families:
            raise RuntimeError("Symm is unavailable")
        return ResourceSet()

    monkeypatch.setattr(
        resource_module,
        "build_candidate_resource_set",
        build_resources,
    )
    monkeypatch.setattr(
        tuning_module,
        "build_production_benchmark_state",
        lambda workload, **_kwargs: SimpleNamespace(rows=workload.row_bucket),
    )
    monkeypatch.setattr(
        candidate_module,
        "prepare_production_candidate",
        lambda _workload, candidate, **_kwargs: (
            Runner()
            if candidate.family == "ipc_source_push"
            else (_ for _ in ()).throw(AssertionError("Symm must be rejected"))
        ),
    )
    monkeypatch.setattr(
        tuning_module,
        "measure_prepared_production_candidate",
        lambda *_args, **_kwargs: (0.1,),
    )
    monkeypatch.setattr("torch.cuda.synchronize", lambda _device: None)

    with caplog.at_level(
        "WARNING",
        logger="sglang.jit_kernel.attntp_fused_norm.tuning",
    ):
        winners = run_local_production_autotune(
            (key,),
            participant_index=0,
            participant_count=1,
            compilation_gather_fn=lambda local: (local,),
            attn_tp_group=Group(),
            device="cuda:0",
        )

    assert allocations == [
        {"ipc_source_push"},
        {"direct_symm"},
    ]
    assert winners[key].candidate is ipc
    assert closed == [True]
    assert "bad-compile-id" in caplog.text
    assert "Symm is unavailable" in caplog.text


def test_prepare_production_candidate_reuses_existing_runner(
    monkeypatch,
) -> None:
    from sglang.jit_kernel.attntp_fused_norm import ipc as ipc_norm

    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=4096,
        execution="eager",
    )
    candidate = next(
        candidate
        for candidate in build_production_non_tile_candidates(key)
        if candidate.family == "ipc_source_push"
    )
    calls = []
    expected_runner = object()

    monkeypatch.setattr(
        ipc_norm,
        "PrefillAttnTPFusedIPCNormRunner",
        lambda **kwargs: calls.append(kwargs) or expected_runner,
    )

    actual = prepare_production_candidate(
        key,
        candidate,
        group="attntp-cpu-group",
        device="cuda:3",
    )

    assert actual is expected_runner
    assert calls[0]["group"] == "attntp-cpu-group"
    assert calls[0]["device"] == "cuda:3"
    assert calls[0]["capacity"] == 4096
    assert calls[0]["algorithm"] is ipc_norm.PrefillCommunicationAlgorithm.SOURCE_PUSH
    assert calls[0]["spec"].output_mode is OutputMode.REPLICATED
    assert (
        calls[0]["spec"].internal_precision is ipc_norm.NormInternalPrecision.FULL_FP32
    )


def test_prepare_production_owner_pull_reuses_full_fp32_runner(
    monkeypatch,
) -> None:
    from sglang.jit_kernel.attntp_fused_norm import ipc as ipc_norm

    key = AttnTPFusedNormWorkloadKey.current(
        phase="decode",
        topology="tp",
        attn_tp_size=4,
        hidden_size=2048,
        row_bucket=64,
        execution="graph",
    )
    candidate = next(
        candidate
        for candidate in build_production_non_tile_candidates(key)
        if candidate.family == "ipc_owner_pull"
    )
    calls = []
    expected_runner = object()

    monkeypatch.setattr(
        ipc_norm,
        "DecodeAttnTPFusedIPCNormRunner",
        lambda **kwargs: calls.append(kwargs) or expected_runner,
    )

    actual = prepare_production_candidate(
        key,
        candidate,
        group="attntp-cpu-group",
        device="cuda:3",
    )

    assert actual is expected_runner
    assert calls[0]["algorithm"] is ipc_norm.DecodeCommunicationAlgorithm.OWNER_PULL
    assert calls[0]["spec"].internal_precision is ipc_norm.NormInternalPrecision.FULL_FP32
    assert calls[0]["max_rows"] == 64
    assert calls[0]["block_size"] == dict(candidate.parameters)["block_size"]


def test_prepare_shared_ipc_candidate_uses_workload_capacity(
    monkeypatch,
) -> None:
    from sglang.jit_kernel.attntp_fused_norm import ipc as ipc_norm

    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=16,
        execution="eager",
    )
    candidate = next(
        candidate
        for candidate in build_production_non_tile_candidates(key)
        if candidate.family == "ipc_source_push"
    )
    resource = SimpleNamespace(max_rows=4096)
    calls = []
    expected_runner = object()

    class Runner:
        @classmethod
        def from_resource(cls, **kwargs):
            calls.append(kwargs)
            return expected_runner

    monkeypatch.setattr(ipc_norm, "PrefillAttnTPFusedIPCNormRunner", Runner)

    actual = prepare_production_candidate(
        key,
        candidate,
        group="attntp-cpu-group",
        device="cuda:3",
        resource=resource,
    )

    assert actual is expected_runner
    assert calls[0]["resource"] is resource
    assert calls[0]["capacity"] == 16


def test_collective_construction_failure_closes_local_candidate() -> None:
    class Candidate:
        stable_id = "candidate-a"

        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    candidate = Candidate()

    prepared, rejection = synchronize_collective_candidate_construction(
        candidate,
        local_error=None,
        gather_errors_fn=lambda error: (error, "rank 1 compile failed", None),
    )

    assert prepared is None
    assert rejection == CollectiveCandidateRejection(
        candidate_id="candidate-a",
        stage="construction",
        rank_errors=((1, "rank 1 compile failed"),),
    )
    assert candidate.close_calls == 1


def test_collective_construction_validates_stage_on_success() -> None:
    class Candidate:
        stable_id = "candidate-a"

        @staticmethod
        def close():
            pass

    with pytest.raises(ValueError, match="stage"):
        synchronize_collective_candidate_construction(
            Candidate(),
            local_error=None,
            gather_errors_fn=lambda error: (error, None),
            stage="typo",
        )


def test_collective_latency_uses_max_rank_for_each_sample() -> None:
    assert aggregate_max_rank_samples(
        (
            (1.0, 4.0, 2.0),
            (3.0, 2.0, 5.0),
            (2.0, 3.0, 1.0),
        )
    ) == (3.0, 4.0, 5.0)

    with pytest.raises(ValueError, match="same number"):
        aggregate_max_rank_samples(((1.0, 2.0), (3.0,)))


def test_collective_measurement_adapter_rejects_before_launch_and_uses_rank_max() -> (
    None
):
    candidates = order_collective_candidates(
        (
            expand_run_candidates(
                make_spec(Algorithm.SYMM, signal_backoff=64),
                max_occupancy=1,
                blocks_per_sm=(1,),
            )[0],
            expand_run_candidates(
                make_spec(Algorithm.OWNER_PULL),
                max_occupancy=1,
                blocks_per_sm=(1,),
            )[0],
        )
    )
    rejected_id = candidates[0].stable_id
    runners = {}
    measured = []

    class Runner:
        def __init__(self, candidate_id):
            self.stable_id = candidate_id
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    def prepare(candidate):
        runner = Runner(candidate.stable_id)
        runners[candidate.stable_id] = runner
        return runner

    def gather_errors(stage, candidate_id, local_error):
        if stage == "construction" and candidate_id == rejected_id:
            return (local_error, "rank 1 rejected candidate")
        return (local_error, None)

    result = measure_collective_candidates(
        candidates,
        gather_candidate_orders_fn=lambda candidate_ids: (
            candidate_ids,
            candidate_ids,
        ),
        prepare_fn=prepare,
        gather_errors_fn=gather_errors,
        measure_fn=lambda runner, candidate: measured.append(candidate.stable_id)
        or (1.0, 4.0, 2.0),
        gather_samples_fn=lambda local_samples: (
            local_samples,
            (3.0, 2.0, 5.0),
        ),
    )

    assert tuple(
        measurement.candidate.stable_id for measurement in result.measurements
    ) == (candidates[1].stable_id,)
    assert result.measurements[0].samples_ms == (3.0, 4.0, 5.0)
    assert result.rejections == (
        CollectiveCandidateRejection(
            candidate_id=rejected_id,
            stage="construction",
            rank_errors=((1, "rank 1 rejected candidate"),),
        ),
    )
    assert measured == [candidates[1].stable_id]
    assert all(runner.close_calls == 1 for runner in runners.values())


def test_generate_compile_specs_covers_all_legal_knobs() -> None:
    specs = generate_compile_specs(
        make_shape(),
        block_sizes=(128, 256),
        signal_backoffs=(64, 128),
        source_push_rows_per_tile=(2, 4),
    )

    owner_pull = [spec for spec in specs if spec.algorithm is Algorithm.OWNER_PULL]
    source_push = [spec for spec in specs if spec.algorithm is Algorithm.SOURCE_PUSH]
    symm = [spec for spec in specs if spec.algorithm is Algorithm.SYMM]

    assert len(owner_pull) == 2
    assert len(source_push) == 8
    assert len(symm) == 4
    assert len({spec.stable_id for spec in specs}) == len(specs)
    assert specs == tuple(sorted(specs, key=lambda spec: spec.stable_id))


def test_compile_spec_round_trips_through_json() -> None:
    spec = make_spec(
        Algorithm.SOURCE_PUSH,
        block_size=128,
        signal_backoff=256,
        rows_per_tile=4,
    )

    restored = CompileSpec.from_dict(json.loads(json.dumps(spec.to_dict())))

    assert restored == spec
    assert restored.stable_id == spec.stable_id


def test_decode_generation_does_not_invent_prefill_tile_knobs() -> None:
    specs = generate_compile_specs(
        make_shape(phase="decode"),
        block_sizes=(256,),
        signal_backoffs=(64, 128),
        source_push_rows_per_tile=(2, 4),
    )

    source_push = [spec for spec in specs if spec.algorithm is Algorithm.SOURCE_PUSH]
    assert len(source_push) == 2
    assert all(spec.rows_per_tile == 0 for spec in source_push)


@pytest.mark.parametrize(
    ("algorithm", "signal_backoff", "rows_per_tile"),
    [
        (Algorithm.OWNER_PULL, 64, 0),
        (Algorithm.OWNER_PULL, 0, 4),
        (Algorithm.SOURCE_PUSH, 0, 4),
        (Algorithm.SOURCE_PUSH, 64, 0),
        (Algorithm.SYMM, 0, 0),
        (Algorithm.SYMM, 64, 4),
    ],
)
def test_compile_spec_rejects_knobs_irrelevant_to_algorithm(
    algorithm: Algorithm,
    signal_backoff: int,
    rows_per_tile: int,
) -> None:
    with pytest.raises(ValueError):
        make_spec(
            algorithm,
            signal_backoff=signal_backoff,
            rows_per_tile=rows_per_tile,
        )


def test_compile_spec_rejects_unsupported_block_shape() -> None:
    with pytest.raises(ValueError, match="block_size"):
        make_spec(Algorithm.OWNER_PULL, block_size=192)


def test_expand_run_candidates_respects_kernel_occupancy() -> None:
    spec = make_spec(Algorithm.SYMM, signal_backoff=64)

    candidates = expand_run_candidates(
        spec,
        max_occupancy=3,
        blocks_per_sm=(1, 2, 3, 4, 8),
    )

    assert [candidate.blocks_per_sm for candidate in candidates] == [1, 2, 3]
    assert len({candidate.stable_id for candidate in candidates}) == 3


def test_select_finalists_is_deterministic() -> None:
    candidates = [
        expand_run_candidates(
            make_spec(Algorithm.OWNER_PULL),
            max_occupancy=3,
            blocks_per_sm=(1, 2, 3),
        )[index]
        for index in range(3)
    ]
    measurements = [
        Measurement(candidate=candidates[0], median_ms=1.0, samples_ms=(1.0,)),
        Measurement(candidate=candidates[1], median_ms=0.8, samples_ms=(0.8,)),
        Measurement(candidate=candidates[2], median_ms=0.8, samples_ms=(0.8,)),
    ]

    finalists = select_finalists(measurements, count=2)

    assert finalists == tuple(
        sorted(candidates[1:], key=lambda candidate: candidate.stable_id)
    )


def test_select_winner_keeps_owner_pull_inside_three_percent_noise_band() -> None:
    owner = expand_run_candidates(
        make_spec(Algorithm.OWNER_PULL),
        max_occupancy=1,
        blocks_per_sm=(1,),
    )[0]
    symm = expand_run_candidates(
        make_spec(Algorithm.SYMM, signal_backoff=64),
        max_occupancy=1,
        blocks_per_sm=(1,),
    )[0]

    winner = select_winner(
        (
            Measurement(candidate=owner, median_ms=1.0, samples_ms=(1.0,)),
            Measurement(candidate=symm, median_ms=0.975, samples_ms=(0.975,)),
        ),
        minimum_speedup=1.03,
    )

    assert winner.candidate == owner
    assert winner.reason == "preferred_within_noise_band"


def test_select_winner_switches_after_three_percent_improvement() -> None:
    owner = expand_run_candidates(
        make_spec(Algorithm.OWNER_PULL),
        max_occupancy=1,
        blocks_per_sm=(1,),
    )[0]
    symm = expand_run_candidates(
        make_spec(Algorithm.SYMM, signal_backoff=64),
        max_occupancy=1,
        blocks_per_sm=(1,),
    )[0]

    winner = select_winner(
        (
            Measurement(candidate=owner, median_ms=1.0, samples_ms=(1.0,)),
            Measurement(candidate=symm, median_ms=0.96, samples_ms=(0.96,)),
        ),
        minimum_speedup=1.03,
    )

    assert winner.candidate == symm
    assert winner.reason == "measured_speedup"


def test_report_serializes_compile_failures_and_winner() -> None:
    owner_spec = make_spec(Algorithm.OWNER_PULL)
    symm_spec = make_spec(Algorithm.SYMM, signal_backoff=64)
    owner = expand_run_candidates(
        owner_spec,
        max_occupancy=1,
        blocks_per_sm=(1,),
    )[0]
    measurement = Measurement(
        candidate=owner,
        median_ms=1.0,
        samples_ms=(0.9, 1.0, 1.1),
    )
    winner = select_winner((measurement,), minimum_speedup=1.03)

    report = build_report(
        shape=make_shape(),
        compile_results=(
            CompileResult.succeeded(owner_spec, max_occupancy=3, compile_ms=12.5),
            CompileResult.failed(symm_spec, error="nvcc failed", compile_ms=4.0),
        ),
        coarse_measurements=(measurement,),
        final_measurements=(measurement,),
        winner=winner,
    )
    encoded = json.dumps(report, sort_keys=True)

    assert '"error": "nvcc failed"' in encoded
    assert report["winner"]["candidate_id"] == owner.stable_id
    assert report["winner"]["reason"] == "only_valid_candidate"

    for compile_result in report["compile_results"]:
        assert CompileResult.from_dict(compile_result).to_dict() == compile_result


def test_ipc_jit_specializes_real_tuning_knobs() -> None:
    from sglang.jit_kernel.attntp_fused_norm import ipc

    assert list(
        inspect.signature(ipc._jit_prefill_attntp_source_push_norm_module).parameters
    ) == [
        "attn_tp_size",
        "hidden_size",
        "output_mode",
        "internal_precision",
        "rows_per_tile",
        "block_size",
        "signal_backoff",
    ]
    assert list(
        inspect.signature(ipc._jit_prefill_attntp_owner_pull_norm_module).parameters
    ) == [
        "attn_tp_size",
        "hidden_size",
        "output_mode",
        "internal_precision",
        "block_size",
    ]
    assert list(
        inspect.signature(ipc._jit_decode_attntp_source_push_norm_module).parameters
    ) == [
        "attn_tp_size",
        "hidden_size",
        "output_mode",
        "internal_precision",
        "block_size",
        "signal_backoff",
    ]
    assert list(
        inspect.signature(ipc._jit_decode_attntp_owner_pull_norm_module).parameters
    ) == [
        "attn_tp_size",
        "hidden_size",
        "output_mode",
        "internal_precision",
        "block_size",
    ]

    assert list(
        inspect.signature(ipc.PrefillAttnTPFusedIPCNormRunner.__init__).parameters
    ) == [
        "self",
        "group",
        "device",
        "spec",
        "capacity",
        "algorithm",
        "block_size",
        "signal_backoff",
        "rows_per_tile",
        "blocks_per_sm",
    ]
    assert list(
        inspect.signature(ipc.DecodeAttnTPFusedIPCNormRunner.__init__).parameters
    ) == [
        "self",
        "group",
        "device",
        "spec",
        "max_rows",
        "algorithm",
        "block_size",
        "signal_backoff",
        "blocks_per_sm",
    ]

    repo_root = Path(__file__).resolve().parents[5]
    header_dir = (
        repo_root / "python/sglang/jit_kernel/csrc/distributed/attntp_fused_norm"
    )
    for header_name in (
        "prefill_attntp_fused_ipc_norm.cuh",
        "decode_attntp_fused_ipc_norm.cuh",
    ):
        source = (header_dir / header_name).read_text()
        assert "uint32_t kSignalBackoff_" in source
        assert "__nanosleep(Trait::kSignalBackoff)" in source
        assert "max_blocks_per_sm_i64" in source
