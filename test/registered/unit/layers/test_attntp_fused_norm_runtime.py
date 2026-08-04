from __future__ import annotations

import inspect
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.jit_kernel.attntp_fused_norm.ipc import OutputMode
from sglang.jit_kernel.attntp_fused_norm.tuning import (
    AttnTPFusedNormCandidate,
    AttnTPFusedNormRegistry,
    AttnTPFusedNormWinner,
    AttnTPFusedNormWorkloadKey,
)
from sglang.srt.context_parallel import build_cp_prefill_split_spec
from sglang.srt.environ import envs
from sglang.srt.layers.attntp_fused_norm import (
    AttnTPFusedNormManager,
    PreparedAttnTPFusedNormRunner,
    attntp_fused_norm_communicator_name,
    get_or_create_attntp_fused_norm_manager,
    get_prefill_cp_attntp_fused_norm_manager,
    max_prefill_cp_local_tokens,
    prepare_attntp_fused_norm_before_kv_pool,
)
from sglang.srt.layers.communicator import LayerCommunicator, ScatterMode
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)


def test_attntp_fused_norm_environment_gate_defaults_off() -> None:
    with envs.SGLANG_ENABLE_ATTNTP_FUSED_NORM.override(False):
        assert not envs.SGLANG_ENABLE_ATTNTP_FUSED_NORM.get()
    with envs.SGLANG_ENABLE_ATTNTP_FUSED_NORM.override(True):
        assert envs.SGLANG_ENABLE_ATTNTP_FUSED_NORM.get()


@pytest.mark.parametrize(
    ("phase", "topology", "expected"),
    (
        ("prefill", "cp", "prefill_cp_attntp2_fused_norm"),
        ("prefill", "tp", "attntp_fused_norm_prefill_tp"),
        ("decode", "tp", "attntp_fused_norm_decode_tp"),
        ("prefill", "dp", "attntp_fused_norm_prefill_dp"),
        ("decode", "dp", "attntp_fused_norm_decode_dp"),
    ),
)
def test_attntp_manager_names_isolate_phase_and_topology(
    phase: str,
    topology: str,
    expected: str,
) -> None:
    assert attntp_fused_norm_communicator_name(phase, topology) == expected


def test_manager_separates_prefill_cp_rotation_from_decode_forward() -> None:
    forward_parameters = inspect.signature(AttnTPFusedNormManager.forward).parameters
    prefill_cp_parameters = inspect.signature(
        AttnTPFusedNormManager.forward_prefill_cp
    ).parameters

    assert "owner_start" not in forward_parameters
    assert "lane_rotation" not in forward_parameters
    assert "lane_rotation" in prefill_cp_parameters


def test_attntp_manager_factory_reuses_only_exact_topology() -> None:
    communicators = {}

    class Group:
        world_size = 4
        device = torch.device("cuda:0")

        @staticmethod
        def get_graph_capture_communicator(name):
            return communicators.get(name)

        @staticmethod
        def register_graph_capture_communicator(name, communicator):
            communicators[name] = communicator

    group = Group()
    first = get_or_create_attntp_fused_norm_manager(
        group=group,
        phase="prefill",
        topology="dp",
        hidden_size=4096,
        max_rows=16384,
    )
    second = get_or_create_attntp_fused_norm_manager(
        group=group,
        phase="prefill",
        topology="dp",
        hidden_size=4096,
        max_rows=16384,
    )

    assert first is second
    assert first.attn_tp_size == 4
    assert communicators == {"attntp_fused_norm_prefill_dp": first}
    with pytest.raises(RuntimeError, match="incompatible topology or capacity"):
        get_or_create_attntp_fused_norm_manager(
            group=group,
            phase="prefill",
            topology="dp",
            hidden_size=2048,
            max_rows=16384,
        )


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("page_size", [1, 4, 16])
def test_prefill_cp_capacity_covers_every_rotated_zigzag_split(
    cp_size: int,
    page_size: int,
) -> None:
    max_global_tokens = 65
    bound = max_prefill_cp_local_tokens(
        max_global_tokens=max_global_tokens,
        cp_size=cp_size,
        page_size=page_size,
    )

    for extend_len in range(max_global_tokens + 1):
        for leading_offset in range(page_size):
            extend_start = 2 * page_size + leading_offset
            for rotation in range(cp_size):
                for leading_owner in range(cp_size):
                    spec = build_cp_prefill_split_spec(
                        extend_start=extend_start,
                        extend_len=extend_len,
                        cp_size=cp_size,
                        page_size=page_size,
                        owner_rotation=rotation,
                        leading_page_owner=(
                            leading_owner if leading_offset and extend_len else None
                        ),
                    )
                    assert max(spec.per_rank_tokens, default=0) <= bound


def test_prefill_cp_factory_registers_one_deferred_manager_per_pair() -> None:
    communicators = {}

    class Group:
        world_size = 2
        device = torch.device("cuda:0")

        @staticmethod
        def get_graph_capture_communicator(name):
            return communicators.get(name)

        @staticmethod
        def register_graph_capture_communicator(name, communicator):
            communicators[name] = communicator

    group = Group()
    first = get_prefill_cp_attntp_fused_norm_manager(
        group=group,
        max_global_tokens=16384,
        cp_size=4,
        page_size=16,
        hidden_size=2048,
    )
    second = get_prefill_cp_attntp_fused_norm_manager(
        group=group,
        max_global_tokens=16384,
        cp_size=4,
        page_size=16,
        hidden_size=2048,
    )

    assert second is first
    assert first.phase == "prefill"
    assert first.topology == "cp"
    assert first.attn_tp_size == 2
    assert first.hidden_size == 2048
    assert first.max_rows == max_prefill_cp_local_tokens(
        max_global_tokens=16384,
        cp_size=4,
        page_size=16,
    )
    assert first.registry is None


def test_attntp_resources_are_installed_before_kv_pool_capacity_is_sized() -> None:
    source = inspect.getsource(ModelRunnerKVCacheMixin.init_memory_pool)

    assert source.index("prepare_attntp_fused_norm_before_kv_pool") < source.index(
        "_resolve_memory_pool_config"
    )


def test_dp_fused_norm_full_output_uses_replicate_gather(monkeypatch) -> None:
    import sglang.srt.layers.communicator as communicator_module

    normalized = torch.arange(8, dtype=torch.bfloat16).view(4, 2)
    residual = normalized.float() + 100
    global_hidden = torch.empty((8, 2), dtype=torch.bfloat16)
    calls = []
    communicator = object.__new__(LayerCommunicator)
    communicator.layer_scatter_modes = SimpleNamespace(
        mlp_mode=ScatterMode.FULL,
    )
    communicator._context = SimpleNamespace(
        attn_tp_size=2,
        attn_tp_rank=1,
        attn_dp_size=4,
    )
    monkeypatch.setattr(
        communicator_module,
        "get_global_dp_buffer",
        lambda _group: global_hidden,
    )
    monkeypatch.setattr(communicator_module, "get_tp_group", lambda: "global-tp")

    def gather_replicate(output, value, batch):
        calls.append((output, value, batch))
        output.copy_(torch.cat((value, value + 20), dim=0))

    monkeypatch.setattr(
        communicator_module,
        "dp_gather_replicate",
        gather_replicate,
        raising=False,
    )
    batch = SimpleNamespace()

    hidden, returned_residual = communicator.prepare_mlp_from_fused_attntp(
        normalized,
        residual,
        batch,
    )

    assert hidden is global_hidden
    assert returned_residual is residual
    assert calls == [(global_hidden, normalized, batch)]


def test_dp_fused_norm_scattered_output_slices_without_collective() -> None:
    normalized = torch.arange(16, dtype=torch.bfloat16).view(8, 2)
    residual = normalized.float() + 100
    communicator = object.__new__(LayerCommunicator)
    communicator.layer_scatter_modes = SimpleNamespace(
        mlp_mode=ScatterMode.SCATTERED,
    )
    communicator._context = SimpleNamespace(
        attn_tp_size=2,
        attn_tp_rank=1,
        attn_dp_size=4,
    )

    hidden, returned_residual = communicator.prepare_mlp_from_fused_attntp(
        normalized,
        residual,
        SimpleNamespace(),
    )

    assert torch.equal(hidden, normalized[4:])
    assert torch.equal(returned_residual, residual[4:])


@pytest.mark.parametrize("family", ("direct_symm", "tile_pipeline"))
def test_prefill_prepared_runner_copies_only_actual_rows_for_arena_backends(
    family: str,
) -> None:
    rows = 5
    raw_runner = _FakeArenaPrefillRunner(capacity=16)
    prepared = PreparedAttnTPFusedNormRunner(
        phase="prefill",
        candidate=_candidate(family),
        runner=raw_runner,
        capacity=16,
        topology="cp",
    )
    partial, residual, o_weight, post_weight = _inputs(rows)

    output, residual_out = prepared.forward_prefill_cp(
        partial,
        residual,
        o_weight,
        post_weight,
        1e-6,
        1e-6,
        lane_rotation=0,
    )

    assert torch.equal(raw_runner.input_view[:rows], partial)
    assert raw_runner.calls == [(rows, rows, 0)]
    assert output.shape == partial.shape
    assert output.dtype is torch.bfloat16
    assert residual_out.shape == residual.shape
    assert residual_out.dtype is torch.float32


@pytest.mark.parametrize("family", ("ipc_source_push", "ipc_owner_pull"))
def test_prefill_prepared_ipc_runner_consumes_partial_without_arena_copy(
    family: str,
) -> None:
    rows = 5
    raw_runner = _FakeIPCPrefillRunner(capacity=16)
    prepared = PreparedAttnTPFusedNormRunner(
        phase="prefill",
        candidate=_candidate(family),
        runner=raw_runner,
        capacity=16,
        topology="cp",
    )
    partial, residual, o_weight, post_weight = _inputs(rows)

    prepared.forward_prefill_cp(
        partial,
        residual,
        o_weight,
        post_weight,
        1e-6,
        1e-6,
        lane_rotation=1,
    )

    assert raw_runner.partial is partial
    assert raw_runner.calls == [(rows, rows, 1)]


def test_manager_resolves_covering_bucket_without_changing_replicated_rows() -> None:
    key_16 = _key(16)
    winner_16 = _winner("ipc_source_push", 16)
    registry = AttnTPFusedNormRegistry.build(
        required_keys=(key_16,),
        winners={key_16: winner_16},
    )
    prepared = PreparedAttnTPFusedNormRunner(
        phase="prefill",
        candidate=winner_16.candidate,
        runner=_FakeIPCPrefillRunner(capacity=16),
        capacity=16,
        topology="cp",
    )
    manager = AttnTPFusedNormManager(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
    )
    manager.install(registry=registry, prepared={key_16: prepared})
    partial, residual, o_weight, post_weight = _inputs(5)

    output, residual_out, fp32_output = manager.forward_prefill_cp(
        partial,
        residual,
        o_weight,
        post_weight,
        1e-6,
        1e-6,
        execution="eager",
        lane_rotation=0,
    )

    assert output.shape == partial.shape
    assert residual_out.shape == residual.shape
    assert fp32_output is None


def test_manager_logs_runtime_active_only_after_first_kernel_dispatch(caplog) -> None:
    key_16 = _key(16)
    winner_16 = _winner("ipc_source_push", 16)
    registry = AttnTPFusedNormRegistry.build(
        required_keys=(key_16,),
        winners={key_16: winner_16},
    )
    raw_runner = _FakeIPCPrefillRunner(capacity=16)
    prepared = PreparedAttnTPFusedNormRunner(
        phase="prefill",
        candidate=winner_16.candidate,
        runner=raw_runner,
        capacity=16,
        topology="cp",
    )
    manager = AttnTPFusedNormManager(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        max_rows=16,
    )
    manager.install(registry=registry, prepared={key_16: prepared})
    caplog.set_level(
        logging.INFO,
        logger="sglang.srt.layers.attntp_fused_norm",
    )

    manager.forward_prefill_cp(
        *_inputs(0),
        1e-6,
        1e-6,
        execution="eager",
        lane_rotation=0,
    )
    assert not any(
        "ATTNTP_FUSED_NORM=1 ACTIVE" in message for message in caplog.messages
    )

    for _ in range(2):
        manager.forward_prefill_cp(
            *_inputs(5),
            1e-6,
            1e-6,
            execution="eager",
            lane_rotation=1,
        )

    active_messages = [
        message
        for message in caplog.messages
        if "SGLANG_ENABLE_ATTNTP_FUSED_NORM=1 ACTIVE" in message
    ]
    assert len(active_messages) == 1
    assert "unfused norm path bypassed" in active_messages[0]
    assert "phase=prefill" in active_messages[0]
    assert "topology=cp" in active_messages[0]
    assert "family=ipc_source_push" in active_messages[0]
    assert "rows=5" in active_messages[0]
    assert raw_runner.calls == [(5, 5, 1), (5, 5, 1)]


def test_decode_manager_chunks_rows_above_kernel_capacity() -> None:
    key_256 = _decode_key(256)
    winner_256 = _winner("ipc_source_push", 256)
    registry = AttnTPFusedNormRegistry.build(
        required_keys=(key_256,),
        winners={key_256: winner_256},
    )
    raw_runner = _FakeIPCDecodeRunner()
    prepared = PreparedAttnTPFusedNormRunner(
        phase="decode",
        candidate=winner_256.candidate,
        runner=raw_runner,
        capacity=256,
    )
    manager = AttnTPFusedNormManager(
        phase="decode",
        topology="tp",
        attn_tp_size=2,
        hidden_size=2048,
        max_rows=4096,
    )
    manager.install(registry=registry, prepared={key_256: prepared})
    partial, residual, o_weight, post_weight = _inputs(600)

    output, residual_out, fp32_output = manager.forward(
        partial,
        residual,
        o_weight,
        post_weight,
        1e-6,
        1e-6,
        execution="eager",
    )

    assert raw_runner.calls == [256, 256, 88]
    assert torch.equal(output, partial)
    assert torch.equal(residual_out, residual + 1)
    assert fp32_output is None


def test_decode_cp_prepared_tile_runner_uses_fixed_rank_contract() -> None:
    rows = 88
    raw_runner = _FakeArenaDecodeRunner(capacity=256)
    prepared = PreparedAttnTPFusedNormRunner(
        phase="decode",
        candidate=_candidate("tile_pipeline"),
        runner=raw_runner,
        capacity=256,
        topology="cp",
    )
    partial, residual, o_weight, post_weight = _inputs(rows)

    prepared.forward(
        partial,
        residual,
        o_weight,
        post_weight,
        1e-6,
        1e-6,
    )

    assert torch.equal(raw_runner.input_view[:rows], partial)
    assert raw_runner.calls == [(rows, rows)]


def test_manager_handles_collectively_empty_rows_without_launching_kernel() -> None:
    key_16 = _key(16)
    winner_16 = _winner("ipc_source_push", 16)
    registry = AttnTPFusedNormRegistry.build(
        required_keys=(key_16,),
        winners={key_16: winner_16},
    )
    raw_runner = _FakeIPCPrefillRunner(capacity=16)
    prepared = PreparedAttnTPFusedNormRunner(
        phase="prefill",
        candidate=winner_16.candidate,
        runner=raw_runner,
        capacity=16,
        topology="cp",
    )
    manager = AttnTPFusedNormManager(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
    )
    manager.install(registry=registry, prepared={key_16: prepared})

    output, residual_out, fp32_output = manager.forward_prefill_cp(
        torch.empty((0, 2048), dtype=torch.bfloat16),
        torch.empty((0, 2048), dtype=torch.float32),
        torch.ones((2048,), dtype=torch.bfloat16),
        torch.ones((2048,), dtype=torch.bfloat16),
        1e-6,
        1e-6,
        execution="eager",
        lane_rotation=0,
    )

    assert output.shape == (0, 2048)
    assert output.dtype is torch.bfloat16
    assert residual_out.shape == (0, 2048)
    assert residual_out.dtype is torch.float32
    assert fp32_output is None
    assert raw_runner.calls == []


def test_manager_fails_fast_when_enabled_path_has_no_prepared_winner() -> None:
    key_16 = _key(16)
    registry = AttnTPFusedNormRegistry.build(
        required_keys=(key_16,),
        winners={key_16: _winner("ipc_source_push", 16)},
    )
    manager = AttnTPFusedNormManager(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
    )

    with pytest.raises(RuntimeError, match="not installed"):
        manager.forward_prefill_cp(
            *_inputs(5),
            1e-6,
            1e-6,
            execution="eager",
            lane_rotation=0,
        )
    with pytest.raises(RuntimeError, match="prepared runner"):
        manager.install(registry=registry, prepared={})


def test_manager_closes_shared_resources_once_after_prepared_runners() -> None:
    events = []
    key = _key(16)
    winner = _winner("ipc_source_push", 16)
    registry = AttnTPFusedNormRegistry.build(
        required_keys=(key,),
        winners={key: winner},
    )
    raw_runner = _FakeIPCPrefillRunner(capacity=16, events=events)
    prepared = PreparedAttnTPFusedNormRunner(
        phase="prefill",
        candidate=winner.candidate,
        runner=raw_runner,
        capacity=16,
        topology="cp",
    )
    resource = _FakeResource(events)
    manager = AttnTPFusedNormManager(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
    )
    manager.install(
        registry=registry,
        prepared={key: prepared},
        resources=(resource, resource),
    )

    manager.close()
    manager.close()

    assert events == ["runner-close", "resource-close"]


def test_manager_installs_winners_with_shared_production_resources(
    monkeypatch,
) -> None:
    from sglang.jit_kernel.attntp_fused_norm import candidates as candidate_module
    from sglang.jit_kernel.attntp_fused_norm import resources as resource_module

    events = []
    keys = (_key(16), _key(64))
    winners = {key: _winner("ipc_source_push", key.row_bucket) for key in keys}
    resource = _FakeResource(events)
    resource_set = _FakeProductionResourceSet(resource)
    raw_runners = []

    def prepare(workload, candidate, **kwargs):
        assert kwargs["group"] == "attntp-cpu"
        assert kwargs["device"] == torch.device("cuda:0")
        assert kwargs["resource"] is resource
        assert kwargs["tile_arena_layout"] is None
        runner = _FakeIPCPrefillRunner(
            capacity=workload.row_bucket,
            events=events,
        )
        raw_runners.append((candidate, runner))
        return runner

    monkeypatch.setattr(
        resource_module,
        "build_production_resource_set",
        lambda selected, **_kwargs: selected == winners and resource_set,
    )
    monkeypatch.setattr(
        candidate_module,
        "prepare_production_candidate",
        prepare,
    )
    manager = AttnTPFusedNormManager(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        max_rows=64,
    )

    manager.install_winners(
        winners,
        group="attntp-cpu",
        device=torch.device("cuda:0"),
    )

    assert tuple(manager.registry.winners) == keys
    assert tuple(manager.prepared) == keys
    assert len(raw_runners) == 2
    manager.close()
    assert events == [
        "runner-close",
        "runner-close",
        "resource-close",
    ]


def test_startup_preparation_reuses_coordinated_manifest_and_installs(
    monkeypatch,
) -> None:
    import sglang.srt.distributed as distributed
    from sglang.jit_kernel.attntp_fused_norm import tuning as tuning_module

    manager = AttnTPFusedNormManager(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        max_rows=16,
    )
    required_keys = []
    install_calls = []
    communicator_name = attntp_fused_norm_communicator_name("prefill", "cp")

    class AttnTPGroup:
        world_size = 2
        rank_in_group = 0
        cpu_group = "attntp-cpu"
        device = torch.device("cuda:0")

        @staticmethod
        def get_graph_capture_communicator(name):
            return manager if name == communicator_name else None

    class WorldGroup:
        local_rank = 0
        rank_in_group = 0

        @staticmethod
        def all_gather_object(value):
            return (value,)

    def build_identity(keys, **_kwargs):
        required_keys.extend(keys)
        return SimpleNamespace(required_keys=tuple(keys))

    def load_manifest(
        _path,
        _identity,
        *,
        local_tune_fn,
        synchronize_cache_fn,
        gather_fn,
        publish,
    ):
        del local_tune_fn, gather_fn
        cached = {
            key: _winner("ipc_source_push", key.row_bucket) for key in required_keys
        }
        assert synchronize_cache_fn(cached) == cached
        assert publish
        return cached

    monkeypatch.setattr(distributed, "get_attn_tp_group", lambda: AttnTPGroup())
    monkeypatch.setattr(distributed, "get_world_group", lambda: WorldGroup())
    monkeypatch.setattr(
        tuning_module,
        "build_production_manifest_identity",
        build_identity,
    )
    monkeypatch.setattr(
        tuning_module,
        "attntp_production_manifest_path",
        lambda _cache, _identity: Path("/tmp/attntp-test.json"),
    )
    monkeypatch.setattr(
        tuning_module,
        "load_or_tune_attntp_manifest",
        load_manifest,
    )
    monkeypatch.setattr(
        manager,
        "install_winners",
        lambda winners, **kwargs: install_calls.append((winners, kwargs)),
    )
    model_runner = SimpleNamespace(
        enable_attntp_fused_norm=True,
        device="cuda",
        server_args=SimpleNamespace(
            attn_cp_size=4,
            page_size=16,
            node_rank=0,
        ),
    )

    result = prepare_attntp_fused_norm_before_kv_pool(model_runner)

    assert result == (manager,)
    assert [key.row_bucket for key in required_keys] == [1, 4, 16]
    assert len(install_calls) == 1
    assert install_calls[0][1] == {
        "group": "attntp-cpu",
        "device": torch.device("cuda:0"),
    }
    assert manager.manifest_path == "/tmp/attntp-test.json"


def test_decode_startup_caps_tuned_buckets_at_kernel_chunk_capacity(
    monkeypatch,
) -> None:
    import sglang.srt.distributed as distributed
    from sglang.jit_kernel.attntp_fused_norm import tuning as tuning_module

    manager = AttnTPFusedNormManager(
        phase="decode",
        topology="tp",
        attn_tp_size=2,
        hidden_size=2048,
        max_rows=4096,
    )
    required_keys = []
    install_calls = []

    class AttnTPGroup:
        world_size = 2
        rank_in_group = 0
        cpu_group = "attntp-cpu"
        device = torch.device("cuda:0")

        @staticmethod
        def get_graph_capture_communicator(name):
            return manager if name == "attntp_fused_norm_decode_tp" else None

    class WorldGroup:
        local_rank = 0
        rank_in_group = 0

        @staticmethod
        def all_gather_object(value):
            return (value,)

    def build_identity(keys, **_kwargs):
        required_keys.extend(keys)
        return SimpleNamespace(required_keys=tuple(keys))

    def load_manifest(
        _path,
        _identity,
        *,
        local_tune_fn,
        synchronize_cache_fn,
        gather_fn,
        publish,
    ):
        del local_tune_fn, gather_fn
        cached = {
            key: _winner("ipc_source_push", key.row_bucket) for key in required_keys
        }
        assert synchronize_cache_fn(cached) == cached
        assert publish
        return cached

    monkeypatch.setattr(distributed, "get_attn_tp_group", lambda: AttnTPGroup())
    monkeypatch.setattr(distributed, "get_world_group", lambda: WorldGroup())
    monkeypatch.setattr(
        tuning_module,
        "build_production_manifest_identity",
        build_identity,
    )
    monkeypatch.setattr(
        tuning_module,
        "attntp_production_manifest_path",
        lambda _cache, _identity: Path("/tmp/attntp-decode-test.json"),
    )
    monkeypatch.setattr(
        tuning_module,
        "load_or_tune_attntp_manifest",
        load_manifest,
    )
    monkeypatch.setattr(
        manager,
        "install_winners",
        lambda winners, **kwargs: install_calls.append((winners, kwargs)),
    )
    model_runner = SimpleNamespace(
        enable_attntp_fused_norm=True,
        device="cuda",
        server_args=SimpleNamespace(
            attn_cp_size=1,
            dp_size=1,
            page_size=16,
            node_rank=0,
        ),
    )

    result = prepare_attntp_fused_norm_before_kv_pool(model_runner)

    assert result == (manager,)
    assert [key.row_bucket for key in required_keys] == [1, 4, 16, 64, 256]
    assert len(install_calls) == 1
    assert manager.manifest_path == "/tmp/attntp-decode-test.json"


def test_startup_preparation_fails_when_enabled_without_deferred_manager(
    monkeypatch,
) -> None:
    import sglang.srt.distributed as distributed

    class AttnTPGroup:
        device = torch.device("cuda:0")

        @staticmethod
        def get_graph_capture_communicator(_name):
            return None

    class WorldGroup:
        local_rank = 0
        rank_in_group = 0

        @staticmethod
        def all_gather_object(value):
            return (value,)

    monkeypatch.setattr(distributed, "get_attn_tp_group", lambda: AttnTPGroup())
    monkeypatch.setattr(distributed, "get_world_group", lambda: WorldGroup())
    model_runner = SimpleNamespace(
        enable_attntp_fused_norm=True,
        device=torch.device("cuda:0"),
        server_args=SimpleNamespace(
            attn_cp_size=4,
            page_size=16,
            node_rank=0,
        ),
    )

    with pytest.raises(RuntimeError, match="enabled.*no deferred manager"):
        prepare_attntp_fused_norm_before_kv_pool(model_runner)


class _FakeArenaPrefillRunner:
    def __init__(self, *, capacity: int) -> None:
        self.device = torch.device("cpu")
        self.input_view = torch.zeros(
            (capacity, 2048),
            dtype=torch.bfloat16,
        )
        self.calls = []

    def run_out(
        self,
        residual,
        _o_weight,
        _post_weight,
        output,
        residual_out,
        actual_rows,
        lane_rotation,
        _o_eps,
        _post_eps,
    ) -> None:
        self.calls.append(
            (
                residual.shape[0],
                output.shape[0],
                int(lane_rotation.item()),
            )
        )

    def close(self) -> None:
        pass


class _FakeIPCPrefillRunner:
    def __init__(self, *, capacity: int, events=None) -> None:
        self.device = torch.device("cpu")
        self.capacity = capacity
        self.calls = []
        self.partial = None
        self.events = events

    def run_out(
        self,
        partial,
        residual,
        _o_weight,
        _post_weight,
        output,
        residual_out,
        actual_rows,
        lane_rotation,
        _o_eps,
        _post_eps,
    ) -> None:
        self.partial = partial
        self.calls.append(
            (
                residual.shape[0],
                output.shape[0],
                int(lane_rotation.item()),
            )
        )

    def close(self) -> None:
        if self.events is not None:
            self.events.append("runner-close")


class _FakeIPCDecodeRunner:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.calls = []

    def run_out(
        self,
        partial,
        residual,
        _o_weight,
        _post_weight,
        output,
        residual_out,
        _o_eps,
        _post_eps,
    ) -> None:
        self.calls.append(partial.shape[0])
        output.copy_(partial)
        residual_out.copy_(residual + 1)

    def close(self) -> None:
        pass


class _FakeArenaDecodeRunner:
    def __init__(self, *, capacity: int) -> None:
        self.device = torch.device("cpu")
        self.input_view = torch.zeros(
            (capacity, 2048),
            dtype=torch.bfloat16,
        )
        self.calls = []

    def run_out(
        self,
        residual,
        _o_weight,
        _post_weight,
        output,
        _residual_out,
        _o_eps,
        _post_eps,
    ) -> None:
        self.calls.append((residual.shape[0], output.shape[0]))
        output.copy_(self.input_view[: residual.shape[0]])

    def close(self) -> None:
        pass


class _FakeResource:
    def __init__(self, events) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("resource-close")


class _FakeProductionResourceSet:
    def __init__(self, resource) -> None:
        self.resources = (resource,)
        self.resource = resource

    def resource_for(self, _candidate):
        return self.resource

    def close(self) -> None:
        self.resource.close()


def _inputs(rows: int):
    return (
        torch.arange(rows * 2048, dtype=torch.float32)
        .reshape(rows, 2048)
        .to(torch.bfloat16),
        torch.zeros((rows, 2048), dtype=torch.float32),
        torch.ones((2048,), dtype=torch.bfloat16),
        torch.ones((2048,), dtype=torch.bfloat16),
    )


def _candidate(family: str) -> AttnTPFusedNormCandidate:
    return AttnTPFusedNormCandidate(
        family=family,
        output_mode=OutputMode.REPLICATED,
    )


def _key(row_bucket: int) -> AttnTPFusedNormWorkloadKey:
    return AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=row_bucket,
        execution="eager",
    )


def _decode_key(row_bucket: int) -> AttnTPFusedNormWorkloadKey:
    return AttnTPFusedNormWorkloadKey.current(
        phase="decode",
        topology="tp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=row_bucket,
        execution="eager",
    )


def _winner(family: str, row_bucket: int) -> AttnTPFusedNormWinner:
    return AttnTPFusedNormWinner(
        candidate=_candidate(family),
        row_bucket=row_bucket,
        median_ms=0.1,
        p90_ms=0.11,
    )
