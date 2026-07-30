"""Eager and CUDA Graph coverage for Decode AttnTP fused IPC norm kernels."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from sglang.jit_kernel.attntp_fused_norm.ipc import (
    AttnTPNormSpec,
    DecodeAttnTPFusedIPCNormRunner,
    DecodeCommunicationAlgorithm,
    NormInternalPrecision,
    OutputMode,
    balanced_row_range,
)
from sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm import (
    NORM_EPS,
    _assert_source_push_outputs,
    _init_distributed,
    _make_inputs,
    _reference_for_precision,
)

DECODE_ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
GRAPH_REPLAYS = 1000


@pytest.mark.parametrize("nproc", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "algorithm", [algorithm.value for algorithm in DecodeCommunicationAlgorithm]
)
@pytest.mark.parametrize("output_mode", [mode.value for mode in OutputMode])
@pytest.mark.parametrize(
    "internal_precision",
    [precision.value for precision in NormInternalPrecision],
)
def test_decode_attntp_eager(
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
        "sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm_decode",
        "--algorithm",
        algorithm,
        "--output-mode",
        output_mode,
        "--internal-precision",
        internal_precision,
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"Decode eager torchrun failed for N={nproc}, "
        f"algorithm={algorithm}, output_mode={output_mode}, "
        f"internal_precision={internal_precision}\n"
        f"{result.stdout}"
    )


@pytest.mark.parametrize("nproc", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "algorithm", [algorithm.value for algorithm in DecodeCommunicationAlgorithm]
)
@pytest.mark.parametrize("output_mode", [mode.value for mode in OutputMode])
@pytest.mark.parametrize(
    "internal_precision",
    [precision.value for precision in NormInternalPrecision],
)
def test_decode_attntp_cuda_graph(
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
        "sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm_decode",
        "--algorithm",
        algorithm,
        "--output-mode",
        output_mode,
        "--internal-precision",
        internal_precision,
        "--graph",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, (
        f"Decode CUDA Graph torchrun failed for N={nproc}, "
        f"algorithm={algorithm}, output_mode={output_mode}, "
        f"internal_precision={internal_precision}\n"
        f"{result.stdout}"
    )


def _expected_owned_rows(
    *,
    reference,
    output_mode: OutputMode,
    rank: int,
    world_size: int,
    owner: int,
    rows: int,
):
    if output_mode is OutputMode.REPLICATED:
        offset = 0
        valid_rows = rows
    elif output_mode is OutputMode.SINGLE_CONTRIBUTOR:
        offset = 0
        valid_rows = rows if rank == owner else 0
    else:
        offset, valid_rows = balanced_row_range(
            total_rows=rows,
            rank=rank,
            attn_tp_size=world_size,
            owner_start=owner,
        )
    return (
        reference[0][offset : offset + valid_rows],
        reference[1][offset : offset + valid_rows],
        valid_rows,
    )


@torch.inference_mode()
def _worker_main(
    algorithm: str,
    output_mode: str,
    internal_precision: str,
) -> None:
    rank, device, cpu_group, nccl_group = _init_distributed()
    torch.cuda.set_stream(torch.cuda.Stream())
    world_size = dist.get_world_size(group=cpu_group)
    selected_algorithm = DecodeCommunicationAlgorithm(algorithm)
    selected_mode = OutputMode(output_mode)
    selected_precision = NormInternalPrecision(internal_precision)

    for hidden_size in (2048, 4096):
        spec = AttnTPNormSpec(
            attn_tp_size=world_size,
            hidden_size=hidden_size,
            output_mode=selected_mode,
            internal_precision=selected_precision,
        )
        runner = DecodeAttnTPFusedIPCNormRunner(
            group=cpu_group,
            device=device,
            spec=spec,
            max_rows=max(DECODE_ROWS),
            algorithm=selected_algorithm,
        )
        try:
            for row_index, rows in enumerate(DECODE_ROWS):
                owner = 0
                inputs = _make_inputs(
                    rows,
                    rank=rank,
                    device=device,
                    seed=500000 + hidden_size + row_index * 1000,
                    hidden_size=hidden_size,
                )
                peer_partials = [torch.empty_like(inputs[0]) for _ in range(world_size)]
                dist.all_gather(peer_partials, inputs[0], group=nccl_group)
                reference = _reference_for_precision(
                    selected_precision,
                    peer_partials,
                    inputs[1],
                    inputs[2],
                    inputs[3],
                )
                output = torch.full(
                    (runner.output_capacity_for(rows), hidden_size),
                    17.0,
                    dtype=torch.bfloat16,
                    device=device,
                )
                residual_out = torch.full(
                    (runner.output_capacity_for(rows), hidden_size),
                    -23.0,
                    dtype=torch.float32,
                    device=device,
                )
                runner.run_out(
                    *inputs,
                    output,
                    residual_out,
                    NORM_EPS,
                    NORM_EPS,
                )
                torch.cuda.synchronize()

                if selected_mode is OutputMode.REPLICATED:
                    offset = 0
                    valid_rows = rows
                elif selected_mode is OutputMode.SINGLE_CONTRIBUTOR:
                    offset = 0
                    valid_rows = rows if rank == owner else 0
                else:
                    offset, valid_rows = balanced_row_range(
                        total_rows=rows,
                        rank=rank,
                        attn_tp_size=world_size,
                        owner_start=owner,
                    )
                _assert_source_push_outputs(
                    output,
                    residual_out,
                    (
                        reference[0][offset : offset + valid_rows],
                        reference[1][offset : offset + valid_rows],
                    ),
                    valid_rows,
                    cpu_group=cpu_group,
                    label=(
                        f"decode {algorithm} N={world_size} H={hidden_size} "
                        f"mode={output_mode} precision={internal_precision} "
                        f"owner={owner} rows={rows}"
                    ),
                )
        finally:
            runner.close()

    dist.destroy_process_group()


@torch.inference_mode()
def _graph_worker_main(
    algorithm: str,
    output_mode: str,
    internal_precision: str,
    graph_replays: int = GRAPH_REPLAYS,
) -> None:
    rank, device, cpu_group, nccl_group = _init_distributed()
    torch.cuda.set_stream(torch.cuda.Stream())
    world_size = dist.get_world_size(group=cpu_group)
    selected_algorithm = DecodeCommunicationAlgorithm(algorithm)
    selected_mode = OutputMode(output_mode)
    selected_precision = NormInternalPrecision(internal_precision)

    for hidden_size in (2048, 4096):
        spec = AttnTPNormSpec(
            attn_tp_size=world_size,
            hidden_size=hidden_size,
            output_mode=selected_mode,
            internal_precision=selected_precision,
        )
        runner = DecodeAttnTPFusedIPCNormRunner(
            group=cpu_group,
            device=device,
            spec=spec,
            max_rows=max(DECODE_ROWS),
            algorithm=selected_algorithm,
        )
        graph_states = []
        graph = None
        state = None
        try:
            for row_index, rows in enumerate(DECODE_ROWS):
                inputs = _make_inputs(
                    rows,
                    rank=rank,
                    device=device,
                    seed=600000 + hidden_size + row_index * 1000,
                    hidden_size=hidden_size,
                )
                output = torch.full(
                    (runner.output_capacity_for(rows), hidden_size),
                    17.0,
                    dtype=torch.bfloat16,
                    device=device,
                )
                residual_out = torch.full(
                    (runner.output_capacity_for(rows), hidden_size),
                    -23.0,
                    dtype=torch.float32,
                    device=device,
                )
                runner.run_out(
                    *inputs,
                    output,
                    residual_out,
                    NORM_EPS,
                    NORM_EPS,
                )
                torch.cuda.synchronize()
                dist.barrier(group=cpu_group)

                graph = torch.cuda.CUDAGraph()
                with runner.capture():
                    with torch.cuda.graph(graph):
                        runner.run_out(
                            *inputs,
                            output,
                            residual_out,
                            NORM_EPS,
                            NORM_EPS,
                        )
                graph_states.append(
                    (
                        rows,
                        inputs,
                        output,
                        residual_out,
                        graph,
                    )
                )

            for row_index, state in enumerate(graph_states):
                (
                    rows,
                    inputs,
                    output,
                    residual_out,
                    graph,
                ) = state
                replay_inputs = _make_inputs(
                    rows,
                    rank=rank,
                    device=device,
                    seed=700000 + hidden_size + row_index * 1000,
                    hidden_size=hidden_size,
                )
                for fixed, current in zip(inputs, replay_inputs):
                    fixed.copy_(current)
                peer_partials = [torch.empty_like(inputs[0]) for _ in range(world_size)]
                dist.all_gather(peer_partials, inputs[0], group=nccl_group)
                reference = _reference_for_precision(
                    selected_precision,
                    peer_partials,
                    inputs[1],
                    inputs[2],
                    inputs[3],
                )
                owner = 0
                output.fill_(17.0)
                residual_out.fill_(-23.0)
                graph.replay()
                torch.cuda.synchronize()
                expected_output, expected_residual, valid_rows = _expected_owned_rows(
                    reference=reference,
                    output_mode=selected_mode,
                    rank=rank,
                    world_size=world_size,
                    owner=owner,
                    rows=rows,
                )
                _assert_source_push_outputs(
                    output,
                    residual_out,
                    (expected_output, expected_residual),
                    valid_rows,
                    cpu_group=cpu_group,
                    label=(
                        f"decode-graph {algorithm} N={world_size} "
                        f"H={hidden_size} mode={output_mode} "
                        f"precision={internal_precision} "
                        f"owner={owner} rows={rows}"
                    ),
                )

            final_owners = [0] * len(graph_states)
            for replay in range(graph_replays):
                state_index = replay % len(graph_states)
                (
                    _,
                    _,
                    output,
                    residual_out,
                    graph,
                ) = graph_states[state_index]
                owner = 0
                output.fill_(17.0)
                residual_out.fill_(-23.0)
                graph.replay()
                final_owners[state_index] = owner
            torch.cuda.synchronize()

            for state, owner in zip(graph_states, final_owners):
                rows, inputs, output, residual_out, _ = state
                peer_partials = [torch.empty_like(inputs[0]) for _ in range(world_size)]
                dist.all_gather(peer_partials, inputs[0], group=nccl_group)
                reference = _reference_for_precision(
                    selected_precision,
                    peer_partials,
                    inputs[1],
                    inputs[2],
                    inputs[3],
                )
                expected_output, expected_residual, valid_rows = _expected_owned_rows(
                    reference=reference,
                    output_mode=selected_mode,
                    rank=rank,
                    world_size=world_size,
                    owner=owner,
                    rows=rows,
                )
                _assert_source_push_outputs(
                    output,
                    residual_out,
                    (expected_output, expected_residual),
                    valid_rows,
                    cpu_group=cpu_group,
                    label=(
                        f"decode-graph-stress {algorithm} N={world_size} "
                        f"H={hidden_size} mode={output_mode} "
                        f"precision={internal_precision} "
                        f"owner={owner} rows={rows}"
                    ),
                )
        finally:
            _ = None
            graph = None
            state = None
            graph_states.clear()
            runner.close()

    dist.destroy_process_group()


def _main() -> None:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Decode worker must be launched with torchrun")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=[algorithm.value for algorithm in DecodeCommunicationAlgorithm],
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
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--graph-replays", type=int, default=GRAPH_REPLAYS)
    arguments = parser.parse_args()
    if arguments.graph:
        _graph_worker_main(
            arguments.algorithm,
            arguments.output_mode,
            arguments.internal_precision,
            arguments.graph_replays,
        )
    else:
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
