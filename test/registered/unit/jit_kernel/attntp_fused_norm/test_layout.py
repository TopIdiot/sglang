from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
JIT_ROOT = REPO_ROOT / "python" / "sglang" / "jit_kernel"
CSRC_ROOT = JIT_ROOT / "csrc" / "distributed"
BENCHMARK_ROOT = REPO_ROOT / "benchmark" / "kernels"


def test_attntp_fused_norm_uses_dedicated_directories() -> None:
    assert (JIT_ROOT / "attntp_fused_norm" / "ipc.py").is_file()
    assert (JIT_ROOT / "attntp_fused_norm" / "symm.py").is_file()
    assert (JIT_ROOT / "attntp_fused_norm" / "tile.py").is_file()
    assert (CSRC_ROOT / "attntp_fused_norm").is_dir()
    assert (JIT_ROOT / "tests" / "attntp_fused_norm").is_dir()

    scattered_modules = {
        path.name
        for path in JIT_ROOT.glob("attntp*.py")
        if path.name != "attntp_fused_norm.py"
    }
    assert scattered_modules == set()


def test_attntp_fused_norm_removes_obsolete_implementations() -> None:
    obsolete_paths = (
        JIT_ROOT / "attntp2_fused_ipc_norm.py",
        JIT_ROOT / "attntp_decode_pipeline.py",
        JIT_ROOT / "attntp_pipeline_layout.py",
        JIT_ROOT / "attntp_prefill_owner_pipeline.py",
        JIT_ROOT / "attntp_replicated_norm.py",
        JIT_ROOT / "attntp_replicated_symm_norm.py",
        CSRC_ROOT / "attntp2_fused_ipc_norm.cuh",
        CSRC_ROOT / "attntp_decode_pipeline.cuh",
        CSRC_ROOT / "attntp_prefill_owner_pipeline.cuh",
        CSRC_ROOT / "attntp_replicated_symm_norm.cuh",
    )
    assert not [path for path in obsolete_paths if path.exists()]


def test_attntp_fused_norm_keeps_one_minimal_benchmark() -> None:
    benchmark_dir = BENCHMARK_ROOT / "attntp_fused_norm"
    assert {path.name for path in benchmark_dir.iterdir()} == {
        "README.md",
        "benchmark.py",
    }
    assert not (BENCHMARK_ROOT / "attntp2_fused_ipc_norm").exists()
    assert not (BENCHMARK_ROOT / "attntp_fused_ipc_norm_v2").exists()
