"""Prefill/Decode communicator isolation for AttnTP fused IPC norm."""

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
    OutputMode,
    PrefillAttnTPFusedIPCNormRunner,
    PrefillCommunicationAlgorithm,
    balanced_row_range,
)
from sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm import (
    NORM_EPS,
    _assert_source_push_outputs,
    _init_distributed,
    _make_inputs,
    _ordered_reference,
)

ALGORITHM_PAIRS = tuple(
    (prefill.value, decode.value)
    for prefill in PrefillCommunicationAlgorithm
    for decode in DecodeCommunicationAlgorithm
)


@pytest.mark.parametrize("nproc", [2, 4, 8])
@pytest.mark.parametrize("prefill_algorithm,decode_algorithm", ALGORITHM_PAIRS)
def test_prefill_decode_communicator_coexistence(
    nproc: int,
    prefill_algorithm: str,
    decode_algorithm: str,
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
        "sglang.jit_kernel.tests.attntp_fused_norm.test_attntp_fused_ipc_norm_coexistence",
        "--prefill-algorithm",
        prefill_algorithm,
        "--decode-algorithm",
        decode_algorithm,
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "Prefill/Decode coexistence torchrun failed for "
        f"N={nproc}, prefill={prefill_algorithm}, "
        f"decode={decode_algorithm}\n{result.stdout}"
    )


def _owned_reference(
    reference,
    *,
    rank: int,
    world_size: int,
    owner: int,
    rows: int,
):
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


def _reference_for_inputs(inputs, *, world_size: int, nccl_group):
    peer_partials = [torch.empty_like(inputs[0]) for _ in range(world_size)]
    dist.all_gather(peer_partials, inputs[0], group=nccl_group)
    return _ordered_reference(
        peer_partials,
        inputs[1],
        inputs[2],
        inputs[3],
    )


def _new_outputs(runner, rows: int, hidden_size: int, device):
    output_rows = runner.output_capacity_for(rows)
    return (
        torch.full(
            (output_rows, hidden_size),
            17.0,
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.full(
            (output_rows, hidden_size),
            -23.0,
            dtype=torch.float32,
            device=device,
        ),
    )


def _assert_owned(
    output,
    residual_out,
    reference,
    *,
    rank: int,
    world_size: int,
    owner: int,
    rows: int,
    cpu_group,
    label: str,
) -> None:
    expected_output, expected_residual, valid_rows = _owned_reference(
        reference,
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
        label=label,
    )


@torch.inference_mode()
def _worker_main(prefill_algorithm: str, decode_algorithm: str) -> None:
    rank, device, cpu_group, nccl_group = _init_distributed()
    torch.cuda.set_stream(torch.cuda.Stream())
    world_size = dist.get_world_size(group=cpu_group)
    hidden_size = 2048
    prefill_capacity = 17
    prefill_rows = 13
    decode_rows = 7
    spec = AttnTPNormSpec(
        attn_tp_size=world_size,
        hidden_size=hidden_size,
        output_mode=OutputMode.TOKEN_SCATTERED,
    )
    prefill_runner = PrefillAttnTPFusedIPCNormRunner(
        group=cpu_group,
        device=device,
        spec=spec,
        capacity=prefill_capacity,
        algorithm=PrefillCommunicationAlgorithm(prefill_algorithm),
    )
    decode_runner = DecodeAttnTPFusedIPCNormRunner(
        group=cpu_group,
        device=device,
        spec=spec,
        max_rows=decode_rows,
        algorithm=DecodeCommunicationAlgorithm(decode_algorithm),
    )
    prefill_closed = False
    prefill_graph = None
    decode_graph = None
    try:
        prefill_inputs = _make_inputs(
            prefill_capacity,
            rank=rank,
            device=device,
            seed=810000,
            hidden_size=hidden_size,
        )
        decode_inputs = _make_inputs(
            decode_rows,
            rank=rank,
            device=device,
            seed=820000,
            hidden_size=hidden_size,
        )
        prefill_reference = _reference_for_inputs(
            tuple(
                value[:prefill_rows] if value.ndim == 2 else value
                for value in prefill_inputs
            ),
            world_size=world_size,
            nccl_group=nccl_group,
        )
        decode_reference = _reference_for_inputs(
            decode_inputs,
            world_size=world_size,
            nccl_group=nccl_group,
        )
        prefill_output, prefill_residual_out = _new_outputs(
            prefill_runner,
            prefill_capacity,
            hidden_size,
            device,
        )
        decode_output, decode_residual_out = _new_outputs(
            decode_runner,
            decode_rows,
            hidden_size,
            device,
        )
        actual_rows = torch.tensor([prefill_rows], dtype=torch.int32, device=device)
        prefill_owner = torch.zeros((1,), dtype=torch.int32, device=device)

        prefill_runner.run_out(
            *prefill_inputs,
            prefill_output,
            prefill_residual_out,
            actual_rows,
            prefill_owner,
            NORM_EPS,
            NORM_EPS,
        )
        decode_runner.run_out(
            *decode_inputs,
            decode_output,
            decode_residual_out,
            NORM_EPS,
            NORM_EPS,
        )
        torch.cuda.synchronize()
        _assert_owned(
            prefill_output,
            prefill_residual_out,
            prefill_reference,
            rank=rank,
            world_size=world_size,
            owner=0,
            rows=prefill_rows,
            cpu_group=cpu_group,
            label="coexistence-prefill-eager",
        )
        _assert_owned(
            decode_output,
            decode_residual_out,
            decode_reference,
            rank=rank,
            world_size=world_size,
            owner=1,
            rows=decode_rows,
            cpu_group=cpu_group,
            label="coexistence-decode-eager",
        )

        prefill_graph = torch.cuda.CUDAGraph()
        with prefill_runner.capture():
            with torch.cuda.graph(prefill_graph):
                prefill_runner.run_out(
                    *prefill_inputs,
                    prefill_output,
                    prefill_residual_out,
                    actual_rows,
                    prefill_owner,
                    NORM_EPS,
                    NORM_EPS,
                )
        decode_graph = torch.cuda.CUDAGraph()
        with decode_runner.capture():
            with torch.cuda.graph(decode_graph):
                decode_runner.run_out(
                    *decode_inputs,
                    decode_output,
                    decode_residual_out,
                    NORM_EPS,
                    NORM_EPS,
                )

        final_prefill_owner = 0
        final_decode_owner = 0
        for replay in range(100):
            final_prefill_owner = replay % world_size
            prefill_output.fill_(17.0)
            prefill_residual_out.fill_(-23.0)
            decode_output.fill_(17.0)
            decode_residual_out.fill_(-23.0)
            prefill_owner.fill_(final_prefill_owner)
            prefill_graph.replay()
            decode_graph.replay()
        torch.cuda.synchronize()
        _assert_owned(
            prefill_output,
            prefill_residual_out,
            prefill_reference,
            rank=rank,
            world_size=world_size,
            owner=final_prefill_owner,
            rows=prefill_rows,
            cpu_group=cpu_group,
            label="coexistence-prefill-graph",
        )
        _assert_owned(
            decode_output,
            decode_residual_out,
            decode_reference,
            rank=rank,
            world_size=world_size,
            owner=final_decode_owner,
            rows=decode_rows,
            cpu_group=cpu_group,
            label="coexistence-decode-graph",
        )

        prefill_graph = None
        prefill_runner.close()
        prefill_closed = True
        surviving_owner = 0
        decode_output.fill_(17.0)
        decode_residual_out.fill_(-23.0)
        decode_runner.run_out(
            *decode_inputs,
            decode_output,
            decode_residual_out,
            NORM_EPS,
            NORM_EPS,
        )
        torch.cuda.synchronize()
        _assert_owned(
            decode_output,
            decode_residual_out,
            decode_reference,
            rank=rank,
            world_size=world_size,
            owner=surviving_owner,
            rows=decode_rows,
            cpu_group=cpu_group,
            label="coexistence-decode-after-prefill-close",
        )
    finally:
        prefill_graph = None
        decode_graph = None
        if not prefill_closed:
            prefill_runner.close()
        decode_runner.close()
        dist.destroy_process_group()


def _main() -> None:
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Coexistence worker must be launched with torchrun")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefill-algorithm",
        required=True,
        choices=[algorithm.value for algorithm in PrefillCommunicationAlgorithm],
    )
    parser.add_argument(
        "--decode-algorithm",
        required=True,
        choices=[algorithm.value for algorithm in DecodeCommunicationAlgorithm],
    )
    arguments = parser.parse_args()
    _worker_main(arguments.prefill_algorithm, arguments.decode_algorithm)


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        _main()
    else:
        sys.exit(pytest.main([__file__, "-v", "-s"]))
