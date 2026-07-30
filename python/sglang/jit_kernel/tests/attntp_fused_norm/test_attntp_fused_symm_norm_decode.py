"""Decode coverage for production AttnTP true Symm norm kernels."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from sglang.jit_kernel.attntp_fused_norm.ipc import balanced_row_range
from sglang.jit_kernel.attntp_fused_norm.symm import (
    AttnTPNormSpec,
    DecodeAttnTPFusedSymmNormRunner,
    NormInternalPrecision,
    OutputMode,
)
from sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm import (
    NORM_EPS,
    _assert_source_push_outputs,
    _init_distributed,
    _make_inputs,
    _reference_for_precision,
)
from sglang.jit_kernel.tests.utils import multiprocess_main, multiprocess_test

MAX_ROWS = 256
ROW_COUNTS = (1, 2, 17, 128, 256)


def test_decode_attntp2_true_symm() -> None:
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
                runner = DecodeAttnTPFusedSymmNormRunner(
                    group=cpu_group,
                    device=device,
                    spec=spec,
                    max_rows=MAX_ROWS,
                )
                try:
                    for index, rows in enumerate(ROW_COUNTS):
                        owner = 0
                        inputs = _make_inputs(
                            rows,
                            rank=rank,
                            device=device,
                            seed=(
                                1000000
                                + hidden_size
                                + list(NormInternalPrecision).index(precision) * 10000
                                + list(OutputMode).index(output_mode) * 1000
                                + index * 100
                            ),
                            hidden_size=hidden_size,
                        )
                        runner.input_view[:rows].copy_(inputs[0])
                        peer_partials = [torch.empty_like(inputs[0]) for _ in range(2)]
                        dist.all_gather(
                            peer_partials,
                            runner.input_view[:rows],
                            group=nccl_group,
                        )
                        reference = _reference_for_precision(
                            precision,
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
                            inputs[1],
                            inputs[2],
                            inputs[3],
                            output,
                            residual_out,
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
                                f"true-symm-decode H={hidden_size} "
                                f"precision={precision.value} "
                                f"mode={output_mode.value} owner={owner} rows={rows}"
                            ),
                        )

                    graph_rows = 17
                    graph_inputs = _make_inputs(
                        graph_rows,
                        rank=rank,
                        device=device,
                        seed=(
                            1100000
                            + hidden_size
                            + list(NormInternalPrecision).index(precision) * 10000
                            + list(OutputMode).index(output_mode) * 1000
                        ),
                        hidden_size=hidden_size,
                    )
                    runner.input_view[:graph_rows].copy_(graph_inputs[0])
                    graph_output = torch.full(
                        (runner.output_capacity_for(graph_rows), hidden_size),
                        17.0,
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    graph_residual_out = torch.full(
                        (runner.output_capacity_for(graph_rows), hidden_size),
                        -23.0,
                        dtype=torch.float32,
                        device=device,
                    )
                    runner.run_out(
                        graph_inputs[1],
                        graph_inputs[2],
                        graph_inputs[3],
                        graph_output,
                        graph_residual_out,
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
                                NORM_EPS,
                                NORM_EPS,
                            )

                    replay_owner = 0
                    replay_inputs = _make_inputs(
                        graph_rows,
                        rank=rank,
                        device=device,
                        seed=(
                            1200000
                            + hidden_size
                            + list(NormInternalPrecision).index(precision) * 10000
                            + list(OutputMode).index(output_mode) * 1000
                        ),
                        hidden_size=hidden_size,
                    )
                    runner.input_view[:graph_rows].copy_(replay_inputs[0])
                    graph_inputs[1].copy_(replay_inputs[1])
                    graph_inputs[2].copy_(replay_inputs[2])
                    graph_inputs[3].copy_(replay_inputs[3])
                    peer_partials = [
                        torch.empty_like(replay_inputs[0]) for _ in range(2)
                    ]
                    dist.all_gather(
                        peer_partials,
                        runner.input_view[:graph_rows],
                        group=nccl_group,
                    )
                    reference = _reference_for_precision(
                        precision,
                        peer_partials,
                        graph_inputs[1],
                        graph_inputs[2],
                        graph_inputs[3],
                    )
                    graph_output.fill_(17.0)
                    graph_residual_out.fill_(-23.0)
                    graph.replay()
                    torch.cuda.synchronize()

                    if output_mode is OutputMode.REPLICATED:
                        offset = 0
                        valid_rows = graph_rows
                    elif output_mode is OutputMode.SINGLE_CONTRIBUTOR:
                        offset = 0
                        valid_rows = graph_rows if rank == replay_owner else 0
                    else:
                        offset, valid_rows = balanced_row_range(
                            total_rows=graph_rows,
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
                            f"true-symm-decode-graph H={hidden_size} "
                            f"precision={precision.value} "
                            f"mode={output_mode.value}"
                        ),
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
