from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def _load_candidate_module():
    try:
        return importlib.import_module("sglang.jit_kernel.attntp_fused_norm.tile")
    except ModuleNotFoundError as error:
        pytest.fail(f"production AttnTP tile pipeline module is missing: {error}")


@pytest.mark.parametrize("attn_tp_size", [2, 4, 8])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
def test_tile_pipeline_accepts_full_fp32_replicated_contract(
    attn_tp_size: int,
    hidden_size: int,
) -> None:
    module = _load_candidate_module()

    module.validate_tile_pipeline_spec(
        module.AttnTPNormSpec(
            attn_tp_size=attn_tp_size,
            hidden_size=hidden_size,
            output_mode=module.OutputMode.REPLICATED,
            internal_precision=module.NormInternalPrecision.FULL_FP32,
        )
    )


def test_tile_pipeline_rejects_lossy_or_non_replicated_contracts() -> None:
    module = _load_candidate_module()

    with pytest.raises(ValueError, match="full_fp32"):
        module.validate_tile_pipeline_spec(
            module.AttnTPNormSpec(
                attn_tp_size=4,
                hidden_size=2048,
                output_mode=module.OutputMode.REPLICATED,
                internal_precision=module.NormInternalPrecision.REFERENCE_BF16,
            )
        )
    with pytest.raises(ValueError, match="replicated output mode"):
        module.validate_tile_pipeline_spec(
            module.AttnTPNormSpec(
                attn_tp_size=4,
                hidden_size=2048,
                output_mode=module.OutputMode.TOKEN_SCATTERED,
                internal_precision=module.NormInternalPrecision.FULL_FP32,
            )
        )


def test_tile_pipeline_arena_keeps_fp32_ring_bounded() -> None:
    module = _load_candidate_module()
    spec = module.AttnTPNormSpec(
        attn_tp_size=8,
        hidden_size=4096,
        output_mode=module.OutputMode.REPLICATED,
        internal_precision=module.NormInternalPrecision.FULL_FP32,
    )

    small = module.tile_pipeline_arena_layout(
        capacity=1024,
        spec=spec,
        rows_per_tile=4,
        ring_stages=8,
    )
    large = module.tile_pipeline_arena_layout(
        capacity=65536,
        spec=spec,
        rows_per_tile=4,
        ring_stages=8,
    )

    expected_ring_bytes = 4 * 8 * 4096 * 4
    assert small.reduced_bytes == expected_ring_bytes
    assert large.reduced_bytes == expected_ring_bytes
    assert small.reduced_offset_bytes == small.input_bytes
    assert large.reduced_offset_bytes == large.input_bytes
    assert (
        small.total_bytes - small.input_bytes == large.total_bytes - large.input_bytes
    )
    assert small.control_offset_bytes % 128 == 0
    assert small.total_bytes % 128 == 0


def test_tile_pipeline_supports_large_decode_tiles() -> None:
    module = _load_candidate_module()
    spec = module.AttnTPNormSpec(
        attn_tp_size=4,
        hidden_size=4096,
        output_mode=module.OutputMode.REPLICATED,
        internal_precision=module.NormInternalPrecision.FULL_FP32,
    )

    layout = module.tile_pipeline_arena_layout(
        capacity=256,
        spec=spec,
        rows_per_tile=32,
        ring_stages=4,
    )

    assert layout.reduced_bytes == 32 * 4 * 4096 * 4
    with pytest.raises(ValueError, match="rows_per_tile"):
        module.tile_pipeline_arena_layout(
            capacity=256,
            spec=spec,
            rows_per_tile=64,
            ring_stages=4,
        )


def test_tile_pipeline_prefill_api_is_varlen() -> None:
    module = _load_candidate_module()
    runner = module.PrefillAttnTPReplicatedTilePipelineRunner

    assert isinstance(runner.input_view, property)
    assert list(inspect.signature(runner.__init__).parameters) == [
        "self",
        "group",
        "device",
        "spec",
        "capacity",
        "rows_per_tile",
        "ring_stages",
        "block_size",
        "signal_backoff",
        "launch_bounds_min_blocks",
        "consumer_cohorts",
        "weight_placement",
        "blocks_per_sm",
        "producer_blocks_per_sm",
    ]
    assert list(inspect.signature(runner.run_out).parameters) == [
        "self",
        "residual",
        "o_norm_weight",
        "post_norm_weight",
        "output",
        "residual_out",
        "actual_rows",
        "owner_start",
        "o_norm_eps",
        "post_norm_eps",
    ]


def test_tile_pipeline_prefill_uses_actual_tensor_rows_with_capacity_arena(
    monkeypatch,
) -> None:
    module = _load_candidate_module()
    spec = module.AttnTPNormSpec(
        attn_tp_size=2,
        hidden_size=2048,
        output_mode=module.OutputMode.REPLICATED,
        internal_precision=module.NormInternalPrecision.FULL_FP32,
    )
    runner = object.__new__(module.PrefillAttnTPReplicatedTilePipelineRunner)
    runner.spec = spec
    runner.phase = "prefill"
    runner.rows = 16
    runner.capacity = 16
    runner.rows_per_tile = 4
    runner.ring_stages = 8
    runner.block_size = 256
    runner.signal_backoff = 64
    runner.launch_bounds_min_blocks = 0
    runner.consumer_cohorts = module.ConsumerCohorts.ONE
    runner.weight_placement = module.WeightPlacement.REGISTER
    runner.blocks_per_sm = 2
    runner.producer_blocks_per_sm = 1
    runner.device = torch.device("cpu")
    runner.rank = 0
    runner._closed = False
    runner._input = torch.empty((16, 2048), dtype=torch.bfloat16)
    runner._pointer_table = 1
    runner.arena_layout = module.tile_pipeline_arena_layout(
        capacity=16,
        spec=spec,
        rows_per_tile=4,
        ring_stages=8,
    )
    launched_input_shapes = []

    class FakeModule:
        @staticmethod
        def fused_tile_pipeline(input_view, *_args):
            launched_input_shapes.append(tuple(input_view.shape))

    monkeypatch.setattr(
        module,
        "_jit_attntp_tile_pipeline_module",
        lambda *_args: FakeModule(),
    )
    rows = 5
    runner.run_out(
        torch.empty((rows, 2048), dtype=torch.float32),
        torch.empty((2048,), dtype=torch.bfloat16),
        torch.empty((2048,), dtype=torch.bfloat16),
        torch.empty((rows, 2048), dtype=torch.bfloat16),
        torch.empty((rows, 2048), dtype=torch.float32),
        torch.tensor([rows], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        1e-6,
        1e-6,
    )

    assert launched_input_shapes == [(rows, 2048)]


def test_tile_pipeline_decode_has_a_separate_schedule() -> None:
    module = _load_candidate_module()
    runner = module.DecodeAttnTPReplicatedTilePipelineRunner

    assert isinstance(runner.input_view, property)
    assert list(inspect.signature(runner.__init__).parameters) == [
        "self",
        "group",
        "device",
        "spec",
        "rows",
        "rows_per_tile",
        "ring_stages",
        "block_size",
        "signal_backoff",
        "launch_bounds_min_blocks",
        "consumer_cohorts",
        "weight_placement",
        "blocks_per_sm",
        "producer_blocks_per_sm",
    ]
    assert list(inspect.signature(runner.run_out).parameters) == [
        "self",
        "residual",
        "o_norm_weight",
        "post_norm_weight",
        "output",
        "residual_out",
        "o_norm_eps",
        "post_norm_eps",
    ]


def test_tile_pipeline_decode_does_not_forward_prefill_metadata() -> None:
    module = _load_candidate_module()
    runner = object.__new__(module.DecodeAttnTPReplicatedTilePipelineRunner)
    captured = {}
    runner._run_tile_pipeline_out = lambda **kwargs: captured.update(kwargs)

    runner.run_out(
        object(),
        object(),
        object(),
        object(),
        object(),
        1e-6,
        1e-6,
    )

    assert captured["actual_rows"] is None
    assert captured["owner_start"] is None


def test_tile_pipeline_resets_completion_counter_for_each_launch() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    source = (
        repo_root / "python/sglang/jit_kernel/csrc/distributed/attntp_fused_norm/"
        "attntp_replicated_tile_pipeline.cuh"
    ).read_text()

    assert "tile_store_release_gpu(local_completion_counter, 0);" in source
    assert "if (completion == gridDim.x - 1)" in source
    assert "completion % gridDim.x" not in source
    assert "constexpr uint64_t kTileTicketEpochStride = uint64_t{1} << 32;" in source
    assert "epoch * kTileTicketEpochStride" in source
    assert "params.max_owner_tiles" not in source


def test_tile_pipeline_close_synchronizes_before_releasing_peer_storage(
    monkeypatch,
) -> None:
    module = _load_candidate_module()
    runner = object.__new__(module._AttnTPReplicatedTilePipelineRunnerBase)
    storage = object()
    runner._closed = False
    runner.device = torch.device("cuda:0")
    runner._input = object()
    runner._storage = storage
    runner._handle = object()
    runner._pointer_table = 1234
    synchronized = False

    def fake_synchronize(device) -> None:
        nonlocal synchronized
        assert device == runner.device
        assert runner._storage is storage
        synchronized = True

    monkeypatch.setattr(torch.cuda, "synchronize", fake_synchronize)

    runner.close()

    assert synchronized
    assert runner._storage is None
    assert runner._pointer_table == 0
    assert runner._closed


def test_tile_pipeline_jit_specializes_protocol_shape() -> None:
    module = _load_candidate_module()

    assert list(
        inspect.signature(module._jit_attntp_tile_pipeline_module).parameters
    ) == [
        "attn_tp_size",
        "hidden_size",
        "phase",
        "rows_per_tile",
        "ring_stages",
        "block_size",
        "signal_backoff",
        "launch_bounds_min_blocks",
        "consumer_cohorts",
        "weight_placement",
    ]
    source = inspect.getsource(module._jit_attntp_tile_pipeline_module)
    assert source.count('("fused_tile_pipeline",') == 1
    assert "consumer_cohorts.value" in source
    assert "weight_placement is WeightPlacement.SHARED" in source
    assert "reduce_kernel" not in source
    assert "gather_kernel" not in source


def test_tile_pipeline_rejects_hidden_incompatible_consumer_shapes() -> None:
    module = _load_candidate_module()
    common = dict(
        hidden_size=2048,
        rows_per_tile=4,
        ring_stages=64,
        block_size=256,
        signal_backoff=64,
        launch_bounds_min_blocks=0,
        blocks_per_sm=None,
        producer_blocks_per_sm=None,
    )

    with pytest.raises(ValueError, match="H4096 physical 512-thread CTA"):
        module._validate_protocol_tuning(
            **common,
            consumer_cohorts=module.ConsumerCohorts.TWO,
            weight_placement=module.WeightPlacement.REGISTER,
        )
    with pytest.raises(ValueError, match="H4096 physical 512-thread CTA"):
        module._validate_protocol_tuning(
            **common,
            consumer_cohorts=module.ConsumerCohorts.ONE,
            weight_placement=module.WeightPlacement.SHARED,
        )


def test_tile_pipeline_candidate_is_not_imported_by_production_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    forbidden_import = "sglang.jit_kernel.attntp_fused_norm.tile"
    offenders = []
    for path in (repo_root / "python" / "sglang" / "srt").rglob("*.py"):
        if forbidden_import in path.read_text():
            offenders.append(path.relative_to(repo_root).as_posix())

    assert offenders == []


def test_tile_pipeline_header_uses_one_cooperative_producer_consumer_grid() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    source = (
        repo_root / "python/sglang/jit_kernel/csrc/distributed/attntp_fused_norm/"
        "attntp_replicated_tile_pipeline.cuh"
    ).read_text()

    assert source.count("attntp_replicated_tile_pipeline_kernel(") == 1
    assert source.count("__global__") == 1
    assert "kFullFP32" in source
    assert "float accumulators" in source
    assert "producer_blocks" in source
    assert "consumer_blocks" in source
    assert "get_max_occupancy" in source
    assert "grid_blocks <= max_resident_blocks" in source
    assert "cudaDevAttrCooperativeLaunch" in source
    assert "cudaLaunchAttributeCooperative" in source
    assert "attr.val.cooperative = 1" in source
    assert "cudaLaunchKernelEx" in source
    assert "LaunchKernel(" not in source
    assert "__launch_bounds__" in source
    assert "kLaunchBoundsMinBlocks" in source
    assert "st.release.sys.global.u64" in source
    assert "ld.acquire.sys.global.u64" in source
    assert "st.release.gpu.global.u64" in source
    assert "ld.acquire.gpu.global.u64" in source
    assert "atom.gpu.global.add.u64" in source
    assert "atom.sys.global.add.u64" not in source
    assert "static_assert(!kDecode || kRowsPerTile == 1)" not in source
    assert "kRowsPerTile == 32" in source


def test_tile_pipeline_header_specializes_h4096_consumer_cohorts() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    source = (
        repo_root / "python/sglang/jit_kernel/csrc/distributed/attntp_fused_norm/"
        "attntp_replicated_tile_pipeline.cuh"
    ).read_text()

    assert '#include "attntp_tile_pipeline_norm.cuh"' in source
    assert "kConsumerCohorts" in source
    assert "kSharedWeights" in source
    assert "tile_consume_cohorts" in source
    assert "consumer_id * Trait::kConsumerCohorts + cohort" in source
    assert "params.consumer_blocks * Trait::kConsumerCohorts" in source
    assert "apply_welm_norm_pipeline_full_fp32_cohort" in source
    assert "kWeightSharedBytes" in source
    assert "kernel, Trait::kBlockSize, Trait::kWeightSharedBytes" in source
    assert "apply_welm_norm_pipeline_full_fp32<Trait>" in source


def test_cohort_norm_header_uses_only_subgroup_barriers() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    csrc_dir = repo_root / "python/sglang/jit_kernel/csrc/distributed/attntp_fused_norm"
    path = csrc_dir / "attntp_tile_pipeline_norm.cuh"

    assert path.exists(), f"cohort norm header is missing: {path}"
    source = path.read_text()
    assert "CohortNormTrait" in source
    assert "cohort_reduce_sum" in source
    assert "apply_welm_norm_pipeline_full_fp32_cohort" in source
    assert "bar.sync" in source
    assert "__syncthreads()" not in source

    common_source = (csrc_dir / "attntp_fused_ipc_norm_common.cuh").read_text()
    assert "CohortNormTrait" not in common_source
    assert "apply_welm_norm_pipeline_full_fp32_cohort" not in common_source
