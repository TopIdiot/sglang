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
        return importlib.import_module("sglang.jit_kernel.attntp_fused_norm.symm")
    except ModuleNotFoundError as error:
        pytest.fail(f"AttnTP Symm kernel module is missing: {error}")


def test_symm_runner_reuses_v2_contract_types() -> None:
    symm = _load_candidate_module()
    ipc = importlib.import_module("sglang.jit_kernel.attntp_fused_norm.ipc")

    assert symm.AttnTPNormSpec is ipc.AttnTPNormSpec
    assert symm.NormInternalPrecision is ipc.NormInternalPrecision
    assert symm.OutputMode is ipc.OutputMode


def test_prefill_runner_owns_symmetric_input() -> None:
    module = _load_candidate_module()

    init_parameters = list(
        inspect.signature(module.PrefillAttnTPFusedSymmNormRunner.__init__).parameters
    )
    run_parameters = list(
        inspect.signature(module.PrefillAttnTPFusedSymmNormRunner.run_out).parameters
    )

    assert init_parameters == [
        "self",
        "group",
        "device",
        "spec",
        "capacity",
        "block_size",
        "signal_backoff",
        "blocks_per_sm",
    ]
    assert run_parameters == [
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
    assert "partial" not in run_parameters
    assert isinstance(module.PrefillAttnTPFusedSymmNormRunner.input_view, property)


def test_prefill_runner_uses_actual_tensor_rows_with_capacity_sized_arena(
    monkeypatch,
) -> None:
    module = _load_candidate_module()
    spec = module.AttnTPNormSpec(
        attn_tp_size=2,
        hidden_size=2048,
        output_mode=module.OutputMode.REPLICATED,
        internal_precision=module.NormInternalPrecision.FULL_FP32,
    )
    runner = object.__new__(module.PrefillAttnTPFusedSymmNormRunner)
    runner.spec = spec
    runner.capacity = 16
    runner.output_capacity = 16
    runner.block_size = 256
    runner.signal_backoff = 64
    runner.blocks_per_sm = 2
    runner.device = torch.device("cpu")
    runner.rank = 0
    runner._closed = False
    runner._input = torch.empty((16, 2048), dtype=torch.bfloat16)
    runner._peer_pointer = 1
    runner._multicast_pointer = 0
    runner._multicast_signal_offset_bytes = 0
    runner._multicast_signal_slots = 0
    runner._partial_pointer_table = 2
    runner._signal_pad = torch.empty((2,), dtype=torch.uint32)
    runner._peer_signal_pointer = 3
    launched_input_shapes = []

    class FakeModule:
        @staticmethod
        def fused_symm_norm(input_view, *_args):
            launched_input_shapes.append(tuple(input_view.shape))

    monkeypatch.setattr(
        module,
        "_jit_prefill_attntp_fused_symm_norm_module",
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


def test_decode_runner_owns_symmetric_input() -> None:
    module = _load_candidate_module()

    init_parameters = list(
        inspect.signature(module.DecodeAttnTPFusedSymmNormRunner.__init__).parameters
    )
    run_parameters = list(
        inspect.signature(module.DecodeAttnTPFusedSymmNormRunner.run_out).parameters
    )

    assert init_parameters == [
        "self",
        "group",
        "device",
        "spec",
        "max_rows",
        "block_size",
        "signal_backoff",
        "blocks_per_sm",
    ]
    assert run_parameters == [
        "self",
        "residual",
        "o_norm_weight",
        "post_norm_weight",
        "output",
        "residual_out",
        "o_norm_eps",
        "post_norm_eps",
    ]
    assert "partial" not in run_parameters
    assert isinstance(module.DecodeAttnTPFusedSymmNormRunner.input_view, property)


def test_symm_candidate_is_not_imported_by_production_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    forbidden_import = "sglang.jit_kernel.attntp_fused_norm.symm"
    offenders = []
    for path in (repo_root / "python" / "sglang" / "srt").rglob("*.py"):
        if forbidden_import in path.read_text():
            offenders.append(path.relative_to(repo_root).as_posix())

    assert offenders == []


def test_distributed_symm_headers_use_hopper_multimem_reduction() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    header_dir = (
        repo_root / "python/sglang/jit_kernel/csrc/distributed/attntp_fused_norm"
    )

    for header_name in (
        "prefill_attntp_fused_symm_norm.cuh",
        "decode_attntp_fused_symm_norm.cuh",
    ):
        source = (header_dir / header_name).read_text()
        assert "multimem.ld_reduce" in source
        assert "multimem.red.release.sys.global.add.u32" in source
        assert "multicast_partial" in source
        assert "local_multicast_signals" in source
        assert "multimem_all_reduce_" not in source


def test_symm_jit_specializes_real_launch_knobs() -> None:
    module = _load_candidate_module()
    prefill_parameters = list(
        inspect.signature(module._jit_prefill_attntp_fused_symm_norm_module).parameters
    )
    decode_parameters = list(
        inspect.signature(module._jit_decode_attntp_fused_symm_norm_module).parameters
    )

    expected = [
        "attn_tp_size",
        "hidden_size",
        "output_mode",
        "internal_precision",
        "block_size",
        "signal_backoff",
    ]
    assert prefill_parameters == expected
    assert decode_parameters == expected

    repo_root = Path(__file__).resolve().parents[5]
    header_dir = (
        repo_root / "python/sglang/jit_kernel/csrc/distributed/attntp_fused_norm"
    )
    common = (header_dir / "attntp_fused_ipc_norm_common.cuh").read_text()
    assert "uint32_t kBlockSize_" in common
    assert "kBlockSize = kBlockSize_" in common
    for header_name in (
        "prefill_attntp_fused_symm_norm.cuh",
        "decode_attntp_fused_symm_norm.cuh",
    ):
        source = (header_dir / header_name).read_text()
        assert "uint32_t kSignalBackoff_" in source
        assert "__nanosleep(Trait::kSignalBackoff)" in source
        assert "max_blocks_per_sm_i64" in source


def test_symm_backend_supports_distributed_attntp_sizes() -> None:
    module = _load_candidate_module()

    for attn_tp_size in (2, 4, 8):
        module._validate_true_symm_spec(
            module.AttnTPNormSpec(
                attn_tp_size=attn_tp_size,
                hidden_size=2048,
                output_mode=module.OutputMode.REPLICATED,
            )
        )


def test_symm_backend_rejects_attntp1() -> None:
    module = _load_candidate_module()

    with pytest.raises(NotImplementedError, match="distributed AttnTP"):
        module._validate_true_symm_spec(
            module.AttnTPNormSpec(
                attn_tp_size=1,
                hidden_size=2048,
                output_mode=module.OutputMode.REPLICATED,
            )
        )


def test_symm_backend_supports_current_hidden_sizes() -> None:
    module = _load_candidate_module()

    for hidden_size in (2048, 4096):
        module._validate_true_symm_spec(
            module.AttnTPNormSpec(
                attn_tp_size=2,
                hidden_size=hidden_size,
                output_mode=module.OutputMode.REPLICATED,
            )
        )


def test_symm_runner_synchronizes_counter_initialization_across_ranks() -> None:
    module = _load_candidate_module()
    source = inspect.getsource(module.AttnTPSymmetricMemoryResource.__init__)

    local_sync = "torch.cuda.synchronize(self.device)"
    group_sync = "dist.barrier(group=group)"
    assert local_sync in source
    assert group_sync in source
    assert source.index(local_sync) < source.index(group_sync)
