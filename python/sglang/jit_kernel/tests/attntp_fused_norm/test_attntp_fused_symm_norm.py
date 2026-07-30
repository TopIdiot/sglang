"""Correctness coverage for production AttnTP true Symm norm kernels."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from sglang.jit_kernel.attntp_fused_norm.ipc import balanced_row_range
from sglang.jit_kernel.attntp_fused_norm.symm import (
    AttnTPNormSpec,
    NormInternalPrecision,
    OutputMode,
    PrefillAttnTPFusedSymmNormRunner,
)
from sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm import (
    NORM_EPS,
    _assert_source_push_outputs,
    _init_distributed,
    _make_inputs,
    _reference_for_precision,
)
from sglang.jit_kernel.tests.utils import multiprocess_main, multiprocess_test

CAPACITY = 257
ROW_COUNTS = (0, 1, 17, 256, 257)
LARGE_GRID_ROWS = 4096


def test_prefill_attntp2_true_symm() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires 2 GPUs")
    multiprocess_test(__file__, 2)


@torch.inference_mode()
def _worker_main() -> None:
    rank, device, cpu_group, nccl_group = _init_distributed()
    torch.cuda.set_stream(torch.cuda.Stream())

    for hidden_size in (2048, 4096):
        for precision in NormInternalPrecision:
            for output_mode in OutputMode:
                spec = AttnTPNormSpec(
                    attn_tp_size=2,
                    hidden_size=hidden_size,
                    output_mode=output_mode,
                    internal_precision=precision,
                )
                runner = PrefillAttnTPFusedSymmNormRunner(
                    group=cpu_group,
                    device=device,
                    spec=spec,
                    capacity=CAPACITY,
                )
                try:
                    for index, rows in enumerate(ROW_COUNTS):
                        owner = index % 2
                        inputs = _make_inputs(
                            CAPACITY,
                            rank=rank,
                            device=device,
                            seed=(
                                700000
                                + hidden_size
                                + list(NormInternalPrecision).index(precision) * 10000
                                + list(OutputMode).index(output_mode) * 1000
                                + index * 100
                            ),
                            hidden_size=hidden_size,
                        )
                        runner.input_view.copy_(inputs[0])
                        peer_partials = [
                            torch.empty_like(runner.input_view) for _ in range(2)
                        ]
                        dist.all_gather(
                            peer_partials,
                            runner.input_view,
                            group=nccl_group,
                        )
                        reference = _reference_for_precision(
                            precision,
                            [partial[:rows] for partial in peer_partials],
                            inputs[1][:rows],
                            inputs[2],
                            inputs[3],
                        )
                        output = torch.full(
                            (runner.output_capacity, hidden_size),
                            17.0,
                            dtype=torch.bfloat16,
                            device=device,
                        )
                        residual_out = torch.full(
                            (runner.output_capacity, hidden_size),
                            -23.0,
                            dtype=torch.float32,
                            device=device,
                        )
                        actual_rows = torch.tensor(
                            [rows], dtype=torch.int32, device=device
                        )
                        owner_start = torch.tensor(
                            [owner], dtype=torch.int32, device=device
                        )

                        runner.run_out(
                            inputs[1],
                            inputs[2],
                            inputs[3],
                            output,
                            residual_out,
                            actual_rows,
                            owner_start,
                            NORM_EPS,
                            NORM_EPS,
                        )
                        torch.cuda.synchronize()

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
                                attn_tp_size=2,
                                owner_start=owner,
                            )
                        expected = (
                            reference[0][offset : offset + valid_rows],
                            reference[1][offset : offset + valid_rows],
                        )
                        _assert_source_push_outputs(
                            output,
                            residual_out,
                            expected,
                            valid_rows,
                            cpu_group=cpu_group,
                            label=(
                                f"true-symm H={hidden_size} "
                                f"precision={precision.value} "
                                f"mode={output_mode.value} owner={owner} rows={rows}"
                            ),
                        )

                    graph_inputs = _make_inputs(
                        CAPACITY,
                        rank=rank,
                        device=device,
                        seed=(
                            800000
                            + hidden_size
                            + list(NormInternalPrecision).index(precision) * 10000
                            + list(OutputMode).index(output_mode) * 1000
                        ),
                        hidden_size=hidden_size,
                    )
                    runner.input_view.copy_(graph_inputs[0])
                    graph_output = torch.full(
                        (runner.output_capacity, hidden_size),
                        17.0,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    graph_residual_out = torch.full(
                        (runner.output_capacity, hidden_size),
                        -23.0,
                        dtype=torch.float32,
                        device=device,
                    )
                    graph_actual_rows = torch.tensor(
                        [CAPACITY], dtype=torch.int32, device=device
                    )
                    graph_owner_start = torch.zeros(
                        (1,), dtype=torch.int32, device=device
                    )
                    runner.run_out(
                        graph_inputs[1],
                        graph_inputs[2],
                        graph_inputs[3],
                        graph_output,
                        graph_residual_out,
                        graph_actual_rows,
                        graph_owner_start,
                        NORM_EPS,
                        NORM_EPS,
                    )
                    torch.cuda.synchronize()

                    graph = torch.cuda.CUDAGraph()
                    with runner.capture():
                        with torch.cuda.graph(graph):
                            runner.run_out(
                                graph_inputs[1],
                                graph_inputs[2],
                                graph_inputs[3],
                                graph_output,
                                graph_residual_out,
                                graph_actual_rows,
                                graph_owner_start,
                                NORM_EPS,
                                NORM_EPS,
                            )

                    replay_rows = 17
                    replay_owner = 1
                    replay_inputs = _make_inputs(
                        CAPACITY,
                        rank=rank,
                        device=device,
                        seed=(
                            900000
                            + hidden_size
                            + list(NormInternalPrecision).index(precision) * 10000
                            + list(OutputMode).index(output_mode) * 1000
                        ),
                        hidden_size=hidden_size,
                    )
                    runner.input_view.copy_(replay_inputs[0])
                    graph_inputs[1].copy_(replay_inputs[1])
                    graph_inputs[2].copy_(replay_inputs[2])
                    graph_inputs[3].copy_(replay_inputs[3])
                    peer_partials = [
                        torch.empty_like(runner.input_view) for _ in range(2)
                    ]
                    dist.all_gather(
                        peer_partials,
                        runner.input_view,
                        group=nccl_group,
                    )
                    reference = _reference_for_precision(
                        precision,
                        [partial[:replay_rows] for partial in peer_partials],
                        graph_inputs[1][:replay_rows],
                        graph_inputs[2],
                        graph_inputs[3],
                    )
                    graph_output.fill_(17.0)
                    graph_residual_out.fill_(-23.0)
                    graph_actual_rows.fill_(replay_rows)
                    graph_owner_start.fill_(replay_owner)
                    graph.replay()
                    torch.cuda.synchronize()

                    if output_mode is OutputMode.REPLICATED:
                        offset = 0
                        valid_rows = replay_rows
                    elif output_mode is OutputMode.SINGLE_CONTRIBUTOR:
                        offset = 0
                        valid_rows = replay_rows if rank == replay_owner else 0
                    else:
                        offset, valid_rows = balanced_row_range(
                            total_rows=replay_rows,
                            rank=rank,
                            attn_tp_size=2,
                            owner_start=replay_owner,
                        )
                    _assert_source_push_outputs(
                        graph_output,
                        graph_residual_out,
                        (
                            reference[0][offset : offset + valid_rows],
                            reference[1][offset : offset + valid_rows],
                        ),
                        valid_rows,
                        cpu_group=cpu_group,
                        label=(
                            f"true-symm-graph H={hidden_size} "
                            f"precision={precision.value} "
                            f"mode={output_mode.value}"
                        ),
                    )
                    graph = None
                finally:
                    runner.close()

    spec = AttnTPNormSpec(
        attn_tp_size=2,
        hidden_size=2048,
        output_mode=OutputMode.TOKEN_SCATTERED,
    )
    runner = PrefillAttnTPFusedSymmNormRunner(
        group=cpu_group,
        device=device,
        spec=spec,
        capacity=LARGE_GRID_ROWS,
    )
    try:
        inputs = _make_inputs(
            LARGE_GRID_ROWS,
            rank=rank,
            device=device,
            seed=1300000,
            hidden_size=spec.hidden_size,
        )
        runner.input_view.copy_(inputs[0])
        peer_partials = [torch.empty_like(runner.input_view) for _ in range(2)]
        dist.all_gather(peer_partials, runner.input_view, group=nccl_group)
        reference = _reference_for_precision(
            spec.internal_precision,
            peer_partials,
            inputs[1],
            inputs[2],
            inputs[3],
        )
        output = torch.full(
            (runner.output_capacity, spec.hidden_size),
            17.0,
            dtype=torch.bfloat16,
            device=device,
        )
        residual_out = torch.full(
            (runner.output_capacity, spec.hidden_size),
            -23.0,
            dtype=torch.float32,
            device=device,
        )
        actual_rows = torch.tensor([LARGE_GRID_ROWS], dtype=torch.int32, device=device)
        owner_start = torch.zeros((1,), dtype=torch.int32, device=device)
        runner.run_out(
            inputs[1],
            inputs[2],
            inputs[3],
            output,
            residual_out,
            actual_rows,
            owner_start,
            NORM_EPS,
            NORM_EPS,
        )
        torch.cuda.synchronize()
        offset, valid_rows = balanced_row_range(
            total_rows=LARGE_GRID_ROWS,
            rank=rank,
            attn_tp_size=2,
            owner_start=0,
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
            label="true-symm-large-grid",
        )

        graph = torch.cuda.CUDAGraph()
        with runner.capture():
            with torch.cuda.graph(graph):
                runner.run_out(
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    output,
                    residual_out,
                    actual_rows,
                    owner_start,
                    NORM_EPS,
                    NORM_EPS,
                )
        for _ in range(1000):
            graph.replay()
        torch.cuda.synchronize()
        _assert_source_push_outputs(
            output,
            residual_out,
            (
                reference[0][offset : offset + valid_rows],
                reference[1][offset : offset + valid_rows],
            ),
            valid_rows,
            cpu_group=cpu_group,
            label="true-symm-large-grid-graph-reuse",
        )
        graph = None
    finally:
        runner.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _worker_main()
    else:
        multiprocess_main(2, _worker_main)
