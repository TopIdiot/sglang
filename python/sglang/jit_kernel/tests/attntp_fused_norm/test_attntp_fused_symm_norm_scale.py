"""AttnTP4/8 coverage for production fused SymmetricMemory kernels."""

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
    PrefillAttnTPFusedSymmNormRunner,
)
from sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm import (
    NORM_EPS,
    _assert_source_push_outputs,
    _init_distributed,
    _make_inputs,
    _reference_for_precision,
)
from sglang.jit_kernel.tests.utils import multiprocess_test

HIDDEN_SIZE = 2048
PREFILL_CAPACITY = 257
PREFILL_ROWS = 17
DECODE_ROWS = 17


@pytest.mark.parametrize("nproc", [4, 8])
def test_attntp_multimem_scale(nproc: int) -> None:
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"Requires {nproc} GPUs")
    multiprocess_test(__file__, nproc)


def _valid_slice(
    *,
    rows: int,
    rank: int,
    world_size: int,
    owner_start: int,
    output_mode: OutputMode,
) -> tuple[int, int]:
    if output_mode is OutputMode.REPLICATED:
        return 0, rows
    if output_mode is OutputMode.SINGLE_CONTRIBUTOR:
        return 0, rows if rank == owner_start else 0
    return balanced_row_range(
        total_rows=rows,
        rank=rank,
        attn_tp_size=world_size,
        owner_start=owner_start,
    )


def _gather_reference(
    *,
    partial: torch.Tensor,
    residual: torch.Tensor,
    o_norm_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    precision: NormInternalPrecision,
    group,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    partials = [torch.empty_like(partial) for _ in range(world_size)]
    dist.all_gather(partials, partial, group=group)
    return _reference_for_precision(
        precision,
        partials,
        residual,
        o_norm_weight,
        post_norm_weight,
    )


@torch.inference_mode()
def _worker_main() -> None:
    rank, device, cpu_group, nccl_group = _init_distributed()
    world_size = dist.get_world_size(group=cpu_group)
    if world_size not in (4, 8):
        raise RuntimeError(f"scale test requires AttnTP4 or AttnTP8, got {world_size}")
    torch.cuda.set_stream(torch.cuda.Stream())
    owner_start = world_size - 1

    for precision in NormInternalPrecision:
        for output_mode in OutputMode:
            spec = AttnTPNormSpec(
                attn_tp_size=world_size,
                hidden_size=HIDDEN_SIZE,
                output_mode=output_mode,
                internal_precision=precision,
            )
            runner = PrefillAttnTPFusedSymmNormRunner(
                group=cpu_group,
                device=device,
                spec=spec,
                capacity=PREFILL_CAPACITY,
            )
            try:
                inputs = _make_inputs(
                    PREFILL_CAPACITY,
                    rank=rank,
                    device=device,
                    seed=(
                        1400000
                        + world_size * 10000
                        + list(NormInternalPrecision).index(precision) * 1000
                        + list(OutputMode).index(output_mode) * 100
                    ),
                    hidden_size=HIDDEN_SIZE,
                )
                runner.input_view.copy_(inputs[0])
                reference = _gather_reference(
                    partial=runner.input_view[:PREFILL_ROWS],
                    residual=inputs[1][:PREFILL_ROWS],
                    o_norm_weight=inputs[2],
                    post_norm_weight=inputs[3],
                    precision=precision,
                    group=nccl_group,
                    world_size=world_size,
                )
                output = torch.full(
                    (runner.output_capacity, HIDDEN_SIZE),
                    17.0,
                    dtype=torch.bfloat16,
                    device=device,
                )
                residual_out = torch.full(
                    (runner.output_capacity, HIDDEN_SIZE),
                    -23.0,
                    dtype=torch.float32,
                    device=device,
                )
                actual_rows = torch.tensor(
                    [PREFILL_ROWS], dtype=torch.int32, device=device
                )
                owner = torch.tensor([owner_start], dtype=torch.int32, device=device)
                runner.run_out(
                    inputs[1],
                    inputs[2],
                    inputs[3],
                    output,
                    residual_out,
                    actual_rows,
                    owner,
                    NORM_EPS,
                    NORM_EPS,
                )
                torch.cuda.synchronize()
                offset, valid_rows = _valid_slice(
                    rows=PREFILL_ROWS,
                    rank=rank,
                    world_size=world_size,
                    owner_start=owner_start,
                    output_mode=output_mode,
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
                        f"true-symm-prefill-N{world_size} "
                        f"precision={precision.value} mode={output_mode.value}"
                    ),
                )

                if (
                    precision is NormInternalPrecision.FULL_FP32
                    and output_mode is OutputMode.TOKEN_SCATTERED
                ):
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
                                owner,
                                NORM_EPS,
                                NORM_EPS,
                            )
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
                        label=f"true-symm-prefill-N{world_size}-graph",
                    )
                    graph = None
            finally:
                runner.close()

            runner = DecodeAttnTPFusedSymmNormRunner(
                group=cpu_group,
                device=device,
                spec=spec,
                max_rows=DECODE_ROWS,
            )
            try:
                inputs = _make_inputs(
                    DECODE_ROWS,
                    rank=rank,
                    device=device,
                    seed=(
                        1500000
                        + world_size * 10000
                        + list(NormInternalPrecision).index(precision) * 1000
                        + list(OutputMode).index(output_mode) * 100
                    ),
                    hidden_size=HIDDEN_SIZE,
                )
                runner.input_view.copy_(inputs[0])
                reference = _gather_reference(
                    partial=runner.input_view,
                    residual=inputs[1],
                    o_norm_weight=inputs[2],
                    post_norm_weight=inputs[3],
                    precision=precision,
                    group=nccl_group,
                    world_size=world_size,
                )
                output_rows = runner.output_capacity_for(DECODE_ROWS)
                output = torch.full(
                    (output_rows, HIDDEN_SIZE),
                    17.0,
                    dtype=torch.bfloat16,
                    device=device,
                )
                residual_out = torch.full(
                    (output_rows, HIDDEN_SIZE),
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
                offset, valid_rows = _valid_slice(
                    rows=DECODE_ROWS,
                    rank=rank,
                    world_size=world_size,
                    owner_start=0,
                    output_mode=output_mode,
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
                        f"true-symm-decode-N{world_size} "
                        f"precision={precision.value} mode={output_mode.value}"
                    ),
                )

                if (
                    precision is NormInternalPrecision.FULL_FP32
                    and output_mode is OutputMode.TOKEN_SCATTERED
                ):
                    graph = torch.cuda.CUDAGraph()
                    with runner.capture():
                        with torch.cuda.graph(graph):
                            runner.run_out(
                                inputs[1],
                                inputs[2],
                                inputs[3],
                                output,
                                residual_out,
                                NORM_EPS,
                                NORM_EPS,
                            )
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
                        label=f"true-symm-decode-N{world_size}-graph",
                    )
                    graph = None
            finally:
                runner.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("launch scale coverage with torchrun")
    _worker_main()
