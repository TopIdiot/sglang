import json
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.jit_kernel.autotune import ManifestIdentity
from sglang.jit_kernel.attntp_fused_norm.ipc import OutputMode
from sglang.jit_kernel.attntp_fused_norm.tuning import (
    AttnTPFusedNormCandidate,
    AttnTPFusedNormRegistry,
    AttnTPFusedNormWinner,
    AttnTPFusedNormWorkloadKey,
    attntp_production_manifest_path,
    build_production_manifest_identity,
    build_production_row_buckets,
    load_or_tune_attntp_manifest,
    resolve_current_output_mode,
)


def test_attntp_kernel_release_includes_production_runtime_wiring() -> None:
    runtime_spec = importlib.util.find_spec("sglang.srt.layers.attntp_fused_norm")
    assert (
        runtime_spec is not None
    ), "AttnTP fused kernels were merged without their production runtime"

    from sglang.srt.environ import envs
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
        ModelRunnerKVCacheMixin,
    )
    from sglang.srt.models.welmv4 import (
        Qwen2MoeAttention,
        Qwen2MoeDecoderLayer,
    )

    assert hasattr(
        envs, "SGLANG_ENABLE_ATTNTP_FUSED_NORM"
    ), "AttnTP fused kernels were merged without their environment gate"
    init_memory_pool_source = inspect.getsource(
        ModelRunnerKVCacheMixin.init_memory_pool
    )
    assert (
        "prepare_attntp_fused_norm_before_kv_pool" in init_memory_pool_source
    ), "AttnTP fused kernels were merged without their startup preparation hook"
    assert hasattr(
        GroupCoordinator, "register_graph_capture_communicator"
    ), "AttnTP fused runtime was merged without graph-capture registration"
    assert hasattr(
        GroupCoordinator, "get_graph_capture_communicator"
    ), "AttnTP fused runtime was merged without graph-capture lookup"

    layer_init_source = inspect.getsource(Qwen2MoeDecoderLayer.__init__)
    layer_forward_source = inspect.getsource(Qwen2MoeDecoderLayer.forward)
    attention_forward_source = inspect.getsource(Qwen2MoeAttention.forward)
    assert "_welm_create_tp_dp_attntp_fused_norm_managers" in layer_init_source
    assert "_welm_create_prefill_cp_attntp2_fused_norm_runner" in layer_init_source
    assert "prepare_mlp_fused_attntp2" in layer_forward_source
    assert "tp_dp_attntp_fused_norm_manager.forward" in layer_forward_source
    assert "skip_o_proj_reduce" in attention_forward_source


@pytest.mark.parametrize("topology", ("tp", "dp", "cp"))
def test_current_production_topologies_use_replicated_output(topology: str) -> None:
    assert resolve_current_output_mode(topology) is OutputMode.REPLICATED

    key = AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology=topology,
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=1024,
        execution="eager",
    )
    assert key.output_mode is OutputMode.REPLICATED


def test_scattered_candidate_round_trips_but_is_not_selected_for_production() -> None:
    candidate = AttnTPFusedNormCandidate(
        family="ipc_source_push",
        output_mode=OutputMode.TOKEN_SCATTERED,
        parameters=(("block_size", 256), ("rows_per_tile", 4)),
    )

    restored = AttnTPFusedNormCandidate.from_dict(
        json.loads(json.dumps(candidate.to_dict()))
    )

    assert restored == candidate
    assert restored.output_mode is OutputMode.TOKEN_SCATTERED
    assert resolve_current_output_mode("cp") is OutputMode.REPLICATED


def test_registry_resolves_exact_then_smallest_covering_row_bucket() -> None:
    key_1k = _key(row_bucket=1024)
    key_4k = _key(row_bucket=4096)
    winner_1k = _winner("small", row_bucket=1024)
    winner_4k = _winner("large", row_bucket=4096)
    registry = AttnTPFusedNormRegistry.build(
        required_keys=(key_1k, key_4k),
        winners={key_1k: winner_1k, key_4k: winner_4k},
    )

    assert registry.resolve_runtime(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        rows=1024,
        execution="eager",
    ) == (key_1k, winner_1k)
    assert registry.resolve_runtime(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        rows=1025,
        execution="eager",
    ) == (key_4k, winner_4k)


def test_registry_rejects_missing_winners() -> None:
    key_1k = _key(row_bucket=1024)
    key_4k = _key(row_bucket=4096)

    with pytest.raises(RuntimeError, match="no winner"):
        AttnTPFusedNormRegistry.build(
            required_keys=(key_1k, key_4k),
            winners={key_1k: _winner("small", row_bucket=1024)},
        )


def test_registry_fails_above_largest_prepared_bucket() -> None:
    key = _key(row_bucket=4096)
    registry = AttnTPFusedNormRegistry.build(
        required_keys=(key,),
        winners={key: _winner("large", row_bucket=4096)},
    )

    with pytest.raises(RuntimeError, match="largest prepared row bucket"):
        registry.resolve_runtime(
            phase="prefill",
            topology="cp",
            attn_tp_size=2,
            hidden_size=2048,
            rows=4097,
            execution="eager",
        )


def test_registry_rejects_winner_with_different_output_semantics() -> None:
    key = _key(row_bucket=1024)
    scattered = AttnTPFusedNormWinner(
        candidate=AttnTPFusedNormCandidate(
            family="ipc_source_push",
            output_mode=OutputMode.TOKEN_SCATTERED,
        ),
        row_bucket=1024,
        median_ms=0.1,
        p90_ms=0.11,
    )

    with pytest.raises(ValueError, match="output mode"):
        AttnTPFusedNormRegistry.build(
            required_keys=(key,),
            winners={key: scattered},
        )


@pytest.mark.parametrize(
    ("max_rows", "expected"),
    (
        (1, (1,)),
        (5, (1, 4, 5)),
        (4096, (1, 4, 16, 64, 256, 1024, 4096)),
        (8191, (1, 4, 16, 64, 256, 1024, 4096, 8191)),
    ),
)
def test_production_row_buckets_end_at_exact_capacity(
    max_rows: int,
    expected: tuple[int, ...],
) -> None:
    assert build_production_row_buckets(max_rows) == expected


def test_attntp_manifest_uses_collective_score_through_shared_engine(tmp_path) -> None:
    key = _key(row_bucket=1024)
    local = _winner(
        "z-local",
        row_bucket=1024,
        median_ms=0.09,
        p90_ms=0.20,
    )
    peer = _winner(
        "a-peer",
        row_bucket=1024,
        median_ms=0.10,
        p90_ms=0.19,
    )
    identity = ManifestIdentity(
        kernel_abi="attntp-v1",
        compiler_version="tvm-ffi-0.1.9",
        cuda_version="12.9",
        gpu_fingerprint="H20-sm90",
        topology_fingerprint="attntp2-nvlink",
        dtype="bfloat16",
        profile="balanced",
        required_keys=(key,),
    )

    winners = load_or_tune_attntp_manifest(
        tmp_path / "attntp.json",
        identity,
        local_tune_fn=lambda: {key: local},
        synchronize_cache_fn=lambda cached: None,
        gather_fn=lambda local_winners: (local_winners, {key: peer}),
        publish=False,
    )

    assert winners == {key: peer}


def test_production_manifest_identity_covers_kernel_topology_and_buckets(
    monkeypatch,
    tmp_path,
) -> None:
    keys = (_key(row_bucket=1024), _key(row_bucket=4096))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            name="H20",
            major=9,
            minor=0,
            multi_processor_count=78,
            total_memory=96 * 1024**3,
        ),
    )

    identity = build_production_manifest_identity(
        reversed(keys),
        device="cuda:0",
        topology_extra={"cp_size": 4},
    )
    broad_identity = build_production_manifest_identity(
        reversed(keys),
        device="cuda:0",
        topology_extra={"cp_size": 4},
        profile="broad",
    )
    path = attntp_production_manifest_path(tmp_path, identity)
    broad_path = attntp_production_manifest_path(tmp_path, broad_identity)

    assert identity.required_keys == keys
    assert identity.profile.startswith("balanced:")
    assert broad_identity.profile.startswith("broad:")
    assert identity.compiler_version
    assert identity.cuda_version
    assert identity.gpu_fingerprint == (f"H20|cc9.0|sm78|mem{96 * 1024**3}")
    assert json.loads(identity.topology_fingerprint)["extra"] == {"cp_size": 4}
    assert path.parent == tmp_path / "attntp_fused_norm"
    assert path.suffix == ".json"
    assert broad_path != path


def test_production_manifest_identity_hashes_jit_cuda_sources(
    monkeypatch,
) -> None:
    read_paths = []
    original_read_bytes = Path.read_bytes

    def record_read_bytes(path):
        read_paths.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", record_read_bytes)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            name="H20",
            major=9,
            minor=0,
            multi_processor_count=78,
            total_memory=96 * 1024**3,
        ),
    )

    build_production_manifest_identity(
        (_key(row_bucket=1024),),
        device="cuda:0",
    )

    source_names = {path.name for path in read_paths}
    assert "attntp_fused_ipc_norm_common.cuh" in source_names
    assert "prefill_attntp_fused_ipc_norm.cuh" in source_names
    assert "decode_attntp_fused_ipc_norm.cuh" in source_names
    assert "attntp_replicated_tile_pipeline.cuh" in source_names


def _key(*, row_bucket: int) -> AttnTPFusedNormWorkloadKey:
    return AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=row_bucket,
        execution="eager",
    )


def _winner(
    name: str,
    *,
    row_bucket: int,
    median_ms: float = 0.1,
    p90_ms: float = 0.11,
) -> AttnTPFusedNormWinner:
    return AttnTPFusedNormWinner(
        candidate=AttnTPFusedNormCandidate(
            family=name,
            output_mode=OutputMode.REPLICATED,
        ),
        row_bucket=row_bucket,
        median_ms=median_ms,
        p90_ms=p90_ms,
    )
