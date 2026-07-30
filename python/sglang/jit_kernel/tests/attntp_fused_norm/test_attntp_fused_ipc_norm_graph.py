"""CUDA Graph coverage for production AttnTP fused IPC norm kernels."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from sglang.jit_kernel.attntp_fused_norm.ipc import (
    NormInternalPrecision,
    OutputMode,
    PrefillCommunicationAlgorithm,
)
from sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm import (
    _init_distributed,
    _run_prefill_cuda_graph_varlen,
)


@pytest.mark.parametrize("nproc", [2, 4, 8])
@pytest.mark.parametrize(
    "algorithm", [algorithm.value for algorithm in PrefillCommunicationAlgorithm]
)
@pytest.mark.parametrize("output_mode", [mode.value for mode in OutputMode])
@pytest.mark.parametrize(
    "internal_precision",
    [precision.value for precision in NormInternalPrecision],
)
def test_prefill_attntp_cuda_graph(
    nproc: int,
    algorithm: str,
    output_mode: str,
    internal_precision: str,
) -> None:
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"Requires {nproc} GPUs")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc}",
        "--module",
        "sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm_graph",
        "--algorithm",
        algorithm,
        "--output-mode",
        output_mode,
        "--internal-precision",
        internal_precision,
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"CUDA Graph torchrun timed out for N={nproc}, "
            f"algorithm={algorithm}, output_mode={output_mode}\n"
            f"internal_precision={internal_precision}\n"
            f"{error.stdout}"
        ) from error

    assert result.returncode == 0, (
        f"CUDA Graph torchrun failed for N={nproc}, "
        f"algorithm={algorithm}, output_mode={output_mode}\n"
        f"internal_precision={internal_precision}\n"
        f"{result.stdout}"
    )


@torch.inference_mode()
def _worker_main(
    algorithm: str,
    output_mode: str,
    internal_precision: str,
) -> None:
    rank, device, cpu_group, nccl_group = _init_distributed()
    torch.cuda.set_stream(torch.cuda.Stream())
    _run_prefill_cuda_graph_varlen(
        rank,
        device,
        cpu_group,
        nccl_group,
        algorithm=PrefillCommunicationAlgorithm(algorithm),
        output_mode=OutputMode(output_mode),
        internal_precision=NormInternalPrecision(internal_precision),
    )
    dist.destroy_process_group()


def _main() -> None:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Graph worker must be launched with torchrun")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=[algorithm.value for algorithm in PrefillCommunicationAlgorithm],
    )
    parser.add_argument(
        "--output-mode",
        required=True,
        choices=[mode.value for mode in OutputMode],
    )
    parser.add_argument(
        "--internal-precision",
        choices=[precision.value for precision in NormInternalPrecision],
        default=NormInternalPrecision.REFERENCE_BF16.value,
    )
    arguments = parser.parse_args()
    _worker_main(
        arguments.algorithm,
        arguments.output_mode,
        arguments.internal_precision,
    )


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _main()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
