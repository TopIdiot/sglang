"""Distributed correctness for the replicated AttnTP FP32 tile pipeline."""

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
    NormInternalPrecision,
    OutputMode,
)
from sglang.jit_kernel.attntp_fused_norm.tile import (
    DecodeAttnTPReplicatedTilePipelineRunner,
    PrefillAttnTPReplicatedTilePipelineRunner,
)
from sglang.jit_kernel.attntp_fused_norm.tile_tuning import (
    ConsumerCohorts,
    WeightPlacement,
)
from sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm import (
    NORM_EPS,
    _full_fp32_reference,
    _init_distributed,
    _make_inputs,
)


@pytest.mark.parametrize("nproc", [2, 4, 8])
def test_replicated_tile_pipeline_cuda_graph(nproc: int) -> None:
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"Requires {nproc} GPUs")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc}",
        "--module",
        "sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_replicated_tile_pipeline",
        "--phase",
        "both",
        "--hidden-sizes",
        "2048",
        "4096",
    ]
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1200,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"tile pipeline torchrun timed out for N={nproc}\n{error.stdout}"
        ) from error
    assert result.returncode == 0, (
        f"tile pipeline torchrun failed for N={nproc}\n{result.stdout}"
    )


def _reference(inputs, actual_rows: int, world_size: int, nccl_group):
    partial, residual, o_weight, post_weight = inputs
    peer_partials = [torch.empty_like(partial) for _ in range(world_size)]
    dist.all_gather(peer_partials, partial, group=nccl_group)
    return _full_fp32_reference(
        [peer[:actual_rows] for peer in peer_partials],
        residual[:actual_rows],
        o_weight,
        post_weight,
    )


def _assert_replicated(
    *,
    output: torch.Tensor,
    residual_out: torch.Tensor,
    expected,
    actual_rows: int,
    world_size: int,
    cpu_group,
    nccl_group,
    label: str,
) -> None:
    expected_output, expected_residual = expected
    actual_output = output[:actual_rows]
    actual_residual = residual_out[:actual_rows]
    output_diff = (actual_output.float() - expected_output.float()).abs()
    residual_diff = (actual_residual - expected_residual).abs()
    output_max = output_diff.max().item() if output_diff.numel() else 0.0
    output_mean = output_diff.mean().item() if output_diff.numel() else 0.0
    residual_max = residual_diff.max().item() if residual_diff.numel() else 0.0
    residual_mean = residual_diff.mean().item() if residual_diff.numel() else 0.0

    peer_outputs = [torch.empty_like(actual_output) for _ in range(world_size)]
    peer_residuals = [torch.empty_like(actual_residual) for _ in range(world_size)]
    dist.all_gather(peer_outputs, actual_output, group=nccl_group)
    dist.all_gather(peer_residuals, actual_residual, group=nccl_group)
    output_guard_ok = torch.equal(
        output[actual_rows:],
        torch.full_like(output[actual_rows:], 17.0),
    )
    residual_guard_ok = torch.equal(
        residual_out[actual_rows:],
        torch.full_like(residual_out[actual_rows:], -23.0),
    )
    failed_local = (
        output_max > 0.125
        or output_mean > 2e-3
        or residual_max > 0.125
        or residual_mean > 5e-4
        or any(not torch.equal(actual_output, peer) for peer in peer_outputs)
        or any(not torch.equal(actual_residual, peer) for peer in peer_residuals)
        or not output_guard_ok
        or not residual_guard_ok
    )
    failed = torch.tensor([int(failed_local)], dtype=torch.int32)
    dist.all_reduce(failed, group=cpu_group)
    if failed.item():
        raise RuntimeError(
            f"{label} mismatch: output_max={output_max:.6f}, "
            f"output_mean={output_mean:.6f}, "
            f"residual_max={residual_max:.6f}, "
            f"residual_mean={residual_mean:.6f}, "
            f"output_guard_ok={output_guard_ok}, "
            f"residual_guard_ok={residual_guard_ok}"
        )


def _spec(world_size: int, hidden_size: int) -> AttnTPNormSpec:
    return AttnTPNormSpec(
        attn_tp_size=world_size,
        hidden_size=hidden_size,
        output_mode=OutputMode.REPLICATED,
        internal_precision=NormInternalPrecision.FULL_FP32,
    )


@torch.inference_mode()
def _run_prefill(
    *,
    hidden_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    cpu_group,
    nccl_group,
    consumer_cohorts: ConsumerCohorts,
    weight_placement: WeightPlacement,
) -> None:
    rows_per_tile = 2
    ring_stages = 2
    capacity = world_size * rows_per_tile * ring_stages + 3
    runner = PrefillAttnTPReplicatedTilePipelineRunner(
        group=cpu_group,
        device=device,
        spec=_spec(world_size, hidden_size),
        capacity=capacity,
        rows_per_tile=rows_per_tile,
        ring_stages=ring_stages,
        block_size=(
            512
            if (
                consumer_cohorts is not ConsumerCohorts.ONE
                or weight_placement is not WeightPlacement.REGISTER
            )
            else 256
        ),
        consumer_cohorts=consumer_cohorts,
        weight_placement=weight_placement,
    )
    graph = None
    try:
        graph_inputs = _make_inputs(
            capacity,
            rank,
            device,
            seed=100000 + hidden_size + capacity,
            hidden_size=hidden_size,
        )
        runner.input_view.copy_(graph_inputs[0])
        residual = graph_inputs[1].clone()
        o_weight = graph_inputs[2].clone()
        post_weight = graph_inputs[3].clone()
        output = torch.full(
            (capacity, hidden_size),
            17.0,
            dtype=torch.bfloat16,
            device=device,
        )
        residual_out = torch.full(
            (capacity, hidden_size),
            -23.0,
            dtype=torch.float32,
            device=device,
        )
        actual_rows = torch.tensor(
            [capacity],
            dtype=torch.int32,
            device=device,
        )
        owner_start = torch.zeros((1,), dtype=torch.int32, device=device)
        runner.run_out(
            residual,
            o_weight,
            post_weight,
            output,
            residual_out,
            actual_rows,
            owner_start,
            NORM_EPS,
            NORM_EPS,
        )
        torch.cuda.synchronize(device)
        dist.barrier(group=cpu_group)

        graph = torch.cuda.CUDAGraph()
        with runner.capture():
            with torch.cuda.graph(graph):
                runner.run_out(
                    residual,
                    o_weight,
                    post_weight,
                    output,
                    residual_out,
                    actual_rows,
                    owner_start,
                    NORM_EPS,
                    NORM_EPS,
                )

        row_sequence = (
            capacity,
            capacity - 1,
            world_size + 1,
            1,
            capacity,
        )
        for epoch, rows in enumerate(row_sequence):
            inputs = _make_inputs(
                capacity,
                rank,
                device,
                seed=200000 + hidden_size + epoch * 1000,
                hidden_size=hidden_size,
            )
            expected = _reference(inputs, rows, world_size, nccl_group)
            runner.input_view.copy_(inputs[0])
            residual.copy_(inputs[1])
            o_weight.copy_(inputs[2])
            post_weight.copy_(inputs[3])
            output.fill_(17.0)
            residual_out.fill_(-23.0)
            actual_rows.fill_(rows)
            owner_start.fill_(epoch % world_size)
            graph.replay()
            torch.cuda.synchronize(device)
            _assert_replicated(
                output=output,
                residual_out=residual_out,
                expected=expected,
                actual_rows=rows,
                world_size=world_size,
                cpu_group=cpu_group,
                nccl_group=nccl_group,
                label=(
                    f"prefill N={world_size} H={hidden_size} rows={rows} "
                    f"cohorts={consumer_cohorts.value} "
                    f"weights={weight_placement.value} "
                    f"owner={epoch % world_size} epoch={epoch}"
                ),
            )
    finally:
        if graph is not None:
            del graph
            torch.cuda.synchronize(device)
        runner.close()
        dist.barrier(group=cpu_group)


@torch.inference_mode()
def _run_decode(
    *,
    hidden_size: int,
    rank: int,
    world_size: int,
    device: torch.device,
    cpu_group,
    nccl_group,
) -> None:
    for rows, rows_per_tile in (
        (1, 1),
        (2 * world_size + 1, 1),
        (2 * world_size + 1, 4),
    ):
        runner = DecodeAttnTPReplicatedTilePipelineRunner(
            group=cpu_group,
            device=device,
            spec=_spec(world_size, hidden_size),
            rows=rows,
            rows_per_tile=rows_per_tile,
            ring_stages=2,
        )
        graph = None
        try:
            graph_inputs = _make_inputs(
                rows,
                rank,
                device,
                seed=300000 + hidden_size + rows,
                hidden_size=hidden_size,
            )
            runner.input_view.copy_(graph_inputs[0])
            residual = graph_inputs[1].clone()
            o_weight = graph_inputs[2].clone()
            post_weight = graph_inputs[3].clone()
            output = torch.empty_like(graph_inputs[0])
            residual_out = torch.empty_like(graph_inputs[1])
            runner.run_out(
                residual,
                o_weight,
                post_weight,
                output,
                residual_out,
                NORM_EPS,
                NORM_EPS,
            )
            torch.cuda.synchronize(device)
            dist.barrier(group=cpu_group)

            graph = torch.cuda.CUDAGraph()
            with runner.capture():
                with torch.cuda.graph(graph):
                    runner.run_out(
                        residual,
                        o_weight,
                        post_weight,
                        output,
                        residual_out,
                        NORM_EPS,
                        NORM_EPS,
                    )
            for epoch in range(4):
                inputs = _make_inputs(
                    rows,
                    rank,
                    device,
                    seed=400000 + hidden_size + rows + epoch * 1000,
                    hidden_size=hidden_size,
                )
                expected = _reference(inputs, rows, world_size, nccl_group)
                runner.input_view.copy_(inputs[0])
                residual.copy_(inputs[1])
                o_weight.copy_(inputs[2])
                post_weight.copy_(inputs[3])
                graph.replay()
                torch.cuda.synchronize(device)
                _assert_replicated(
                    output=output,
                    residual_out=residual_out,
                    expected=expected,
                    actual_rows=rows,
                    world_size=world_size,
                    cpu_group=cpu_group,
                    nccl_group=nccl_group,
                    label=(
                        f"decode N={world_size} H={hidden_size} rows={rows} "
                        f"rows_per_tile={rows_per_tile} "
                        f"owner={epoch % world_size} epoch={epoch}"
                    ),
                )
        finally:
            if graph is not None:
                del graph
                torch.cuda.synchronize(device)
            runner.close()
            dist.barrier(group=cpu_group)


@torch.inference_mode()
def _worker_main(
    phase: str,
    hidden_sizes: tuple[int, ...],
    consumer_cohorts: ConsumerCohorts,
    weight_placement: WeightPlacement,
) -> None:
    if any(hidden_size != 4096 for hidden_size in hidden_sizes) and (
        consumer_cohorts is not ConsumerCohorts.ONE
        or weight_placement is not WeightPlacement.REGISTER
    ):
        raise ValueError("multiple cohorts and shared weights require H4096")
    rank, device, cpu_group, nccl_group = _init_distributed()
    world_size = dist.get_world_size(group=cpu_group)
    torch.cuda.set_stream(torch.cuda.Stream())
    for hidden_size in hidden_sizes:
        if phase in ("prefill", "both"):
            _run_prefill(
                hidden_size=hidden_size,
                rank=rank,
                world_size=world_size,
                device=device,
                cpu_group=cpu_group,
                nccl_group=nccl_group,
                consumer_cohorts=consumer_cohorts,
                weight_placement=weight_placement,
            )
        if phase in ("decode", "both"):
            _run_decode(
                hidden_size=hidden_size,
                rank=rank,
                world_size=world_size,
                device=device,
                cpu_group=cpu_group,
                nccl_group=nccl_group,
            )
    dist.destroy_process_group()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prefill", "decode", "both"),
        default="both",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        choices=(2048, 4096),
        default=(2048, 4096),
    )
    parser.add_argument(
        "--consumer-cohorts",
        type=int,
        choices=(1, 2, 4),
        default=1,
    )
    parser.add_argument(
        "--weight-placement",
        choices=("register", "shared"),
        default="register",
    )
    args = parser.parse_args()
    _worker_main(
        args.phase,
        tuple(args.hidden_sizes),
        ConsumerCohorts(args.consumer_cohorts),
        (
            WeightPlacement.REGISTER
            if args.weight_placement == "register"
            else WeightPlacement.SHARED
        ),
    )


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _main()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
