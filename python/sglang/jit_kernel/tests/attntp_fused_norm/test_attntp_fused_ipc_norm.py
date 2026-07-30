"""Correctness tests for production AttnTP fused IPC norm kernels."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.attntp_fused_norm.ipc import (
    AttnTPNormSpec,
    NormInternalPrecision,
    OutputMode,
    PrefillCommunicationAlgorithm,
    PrefillAttnTPFusedIPCNormRunner,
    balanced_row_range,
    fused_prefill_attntp_norm,
    fused_prefill_attntp_norm_out,
    required_push_buffer_bytes,
)
from sglang.jit_kernel.tests.utils import multiprocess_main, multiprocess_test
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.layers.welmv4_op import mmq_style_norm_after_attn

HIDDEN_SIZE = 2048
NORM_EPS = 1e-5
TOKEN_COUNTS = (1, 17, 1024)
LOCAL_TOKEN_COUNTS = (0, 1, 3, 17, 257)
LOCAL_HIDDEN_SIZES = (2048, 4096)
SOURCE_PUSH_CAPACITY = 257


@pytest.mark.parametrize("nproc", [1, 2, 4, 8])
def test_prefill_attntp_candidate(nproc: int) -> None:
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"Requires {nproc} GPUs")
    multiprocess_test(__file__, nproc)


def _init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coordinator = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    return (
        local_rank,
        device,
        coordinator.cpu_group,
        coordinator.device_group,
    )


def _make_inputs(
    tokens: int,
    rank: int,
    device: torch.device,
    seed: int,
    hidden_size: int = HIDDEN_SIZE,
):
    local_generator = torch.Generator(device=device)
    local_generator.manual_seed(seed + rank)
    shared_generator = torch.Generator(device=device)
    shared_generator.manual_seed(seed + 10000)
    partial = torch.randn(
        (tokens, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=local_generator,
    )
    residual = torch.randn(
        (tokens, hidden_size),
        dtype=torch.float32,
        device=device,
        generator=shared_generator,
    )
    o_weight = torch.randn(
        (hidden_size,),
        dtype=torch.bfloat16,
        device=device,
        generator=shared_generator,
    )
    post_weight = torch.randn(
        (hidden_size,),
        dtype=torch.bfloat16,
        device=device,
        generator=shared_generator,
    )
    return partial, residual, o_weight, post_weight


def _ordered_reference(partials, residual, o_weight, post_weight):
    reduced_fp32 = torch.zeros_like(residual)
    for partial in partials:
        reduced_fp32 = reduced_fp32 + partial.float()
    reduced = reduced_fp32.to(torch.bfloat16)

    o_norm_scale = torch.rsqrt(
        reduced.float().square().mean(dim=-1, keepdim=True) + NORM_EPS
    )
    o_norm = (reduced.float() * o_norm_scale * o_weight.float()).to(torch.bfloat16)

    residual_out = residual + o_norm.float()
    post_norm_input = residual_out.to(torch.bfloat16)
    post_norm_scale = torch.rsqrt(
        post_norm_input.float().square().mean(dim=-1, keepdim=True) + NORM_EPS
    )
    output = (post_norm_input.float() * post_norm_scale * post_weight.float()).to(
        torch.bfloat16
    )
    return output, residual_out


def _full_fp32_reference(partials, residual, o_weight, post_weight):
    reduced = torch.zeros_like(residual)
    for partial in partials:
        reduced = reduced + partial.float()

    o_norm_scale = torch.rsqrt(reduced.square().mean(dim=-1, keepdim=True) + NORM_EPS)
    o_norm = reduced * o_norm_scale * o_weight.float()

    residual_out = residual + o_norm
    post_norm_scale = torch.rsqrt(
        residual_out.square().mean(dim=-1, keepdim=True) + NORM_EPS
    )
    output = (residual_out * post_norm_scale * post_weight.float()).to(torch.bfloat16)
    return output, residual_out


def _reference_for_precision(
    internal_precision,
    partials,
    residual,
    o_weight,
    post_weight,
):
    reference_fn = (
        _ordered_reference
        if internal_precision is NormInternalPrecision.REFERENCE_BF16
        else _full_fp32_reference
    )
    return reference_fn(partials, residual, o_weight, post_weight)


def _reference(inputs, nccl_group):
    partial, residual, o_weight, post_weight = inputs
    reduced = partial.clone()
    dist.all_reduce(reduced, group=nccl_group)
    output, residual_out, _ = mmq_style_norm_after_attn(
        reduced,
        residual,
        o_weight,
        post_weight,
        NORM_EPS,
    )
    return output, residual_out


def _assert_local_outputs(actual, expected, *, label: str) -> None:
    output, residual_out = actual
    expected_output, expected_residual = expected
    output_diff = (output.float() - expected_output.float()).abs()
    residual_diff = (residual_out - expected_residual).abs()
    output_max = output_diff.max().item() if output_diff.numel() else 0.0
    output_mean = output_diff.mean().item() if output_diff.numel() else 0.0
    residual_max = residual_diff.max().item() if residual_diff.numel() else 0.0
    residual_mean = residual_diff.mean().item() if residual_diff.numel() else 0.0
    if (
        output_max > 0.125
        or output_mean > 2e-3
        or residual_max > 0.125
        or residual_mean > 5e-4
    ):
        raise RuntimeError(
            f"{label} mismatch: output_max={output_max:.6f}, "
            f"output_mean={output_mean:.6f}, "
            f"residual_max={residual_max:.6f}, "
            f"residual_mean={residual_mean:.6f}"
        )


@torch.inference_mode()
def _local_worker_main() -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_stream(torch.cuda.Stream())

    for hidden_size in LOCAL_HIDDEN_SIZES:
        for tokens in LOCAL_TOKEN_COUNTS:
            inputs = _make_inputs(
                tokens,
                rank=0,
                device=device,
                seed=5000 + hidden_size + tokens,
                hidden_size=hidden_size,
            )
            expected = _ordered_reference([inputs[0]], inputs[1], inputs[2], inputs[3])

            guarded_output = torch.full(
                (tokens + 2, hidden_size),
                17.0,
                dtype=torch.bfloat16,
                device=device,
            )
            guarded_residual = torch.full(
                (tokens + 2, hidden_size),
                -23.0,
                dtype=torch.float32,
                device=device,
            )
            fused_prefill_attntp_norm_out(
                None,
                *inputs,
                guarded_output[:tokens],
                guarded_residual[:tokens],
                NORM_EPS,
                NORM_EPS,
                attn_tp_size=1,
            )
            torch.cuda.synchronize()

            _assert_local_outputs(
                (guarded_output[:tokens], guarded_residual[:tokens]),
                expected,
                label=f"AttnTP1 H={hidden_size} tokens={tokens}",
            )
            if not torch.equal(
                guarded_output[tokens:],
                torch.full_like(guarded_output[tokens:], 17.0),
            ):
                raise RuntimeError("BF16 output guard was modified")
            if not torch.equal(
                guarded_residual[tokens:],
                torch.full_like(guarded_residual[tokens:], -23.0),
            ):
                raise RuntimeError("FP32 residual guard was modified")


def _assert_outputs(
    actual,
    expected,
    *,
    cpu_group,
    nccl_group,
    label: str,
):
    output, residual_out = actual
    expected_output, expected_residual = expected
    peer_outputs = [torch.empty_like(output) for _ in range(2)]
    peer_residuals = [torch.empty_like(residual_out) for _ in range(2)]
    dist.all_gather(peer_outputs, output, group=nccl_group)
    dist.all_gather(peer_residuals, residual_out, group=nccl_group)

    output_diff = (output.float() - expected_output.float()).abs()
    residual_diff = (residual_out - expected_residual).abs()
    failed_local = (
        output_diff.max().item() > 0.125
        or output_diff.mean().item() > 2e-3
        or residual_diff.max().item() > 0.125
        or residual_diff.mean().item() > 5e-4
        or not torch.equal(peer_outputs[0], peer_outputs[1])
        or not torch.equal(peer_residuals[0], peer_residuals[1])
    )
    failed = torch.tensor([int(failed_local)], dtype=torch.int32)
    dist.all_reduce(failed, group=cpu_group)
    if failed.item():
        raise RuntimeError(
            f"{label} mismatch: output_max={output_diff.max().item():.6f}, "
            f"output_mean={output_diff.mean().item():.6f}, "
            f"residual_max={residual_diff.max().item():.6f}, "
            f"residual_mean={residual_diff.mean().item():.6f}"
        )


@torch.inference_mode()
def _run_attntp2_regression(
    rank,
    device,
    cpu_group,
    nccl_group,
) -> None:
    communicator = CustomAllReduceV2(
        cpu_group,
        device,
        max_pull_size=0,
        max_push_size=required_push_buffer_bytes(max(TOKEN_COUNTS)),
    )
    if communicator.disabled:
        raise RuntimeError("production CustomAllReduceV2 is disabled")

    for tokens in TOKEN_COUNTS:
        inputs = _make_inputs(tokens, rank, device, seed=1000 + tokens)
        expected = _reference(inputs, nccl_group)
        actual = fused_prefill_attntp_norm(
            communicator.obj,
            *inputs,
            NORM_EPS,
            NORM_EPS,
        )
        torch.cuda.synchronize()
        _assert_outputs(
            actual,
            expected,
            cpu_group=cpu_group,
            nccl_group=nccl_group,
            label=f"eager tokens={tokens}",
        )

    empty_inputs = _make_inputs(0, rank, device, seed=2000)
    empty_output, empty_residual = fused_prefill_attntp_norm(
        communicator.obj,
        *empty_inputs,
        NORM_EPS,
        NORM_EPS,
    )
    if empty_output.shape != (0, HIDDEN_SIZE):
        raise RuntimeError("empty BF16 output shape changed")
    if empty_residual.shape != (0, HIDDEN_SIZE):
        raise RuntimeError("empty FP32 residual shape changed")

    tokens = 17
    static_inputs = _make_inputs(tokens, rank, device, seed=3000)
    output = torch.empty_like(static_inputs[0])
    residual_out = torch.empty_like(static_inputs[1])
    fused_prefill_attntp_norm_out(
        communicator.obj,
        *static_inputs,
        output,
        residual_out,
        NORM_EPS,
        NORM_EPS,
    )
    torch.cuda.synchronize()
    dist.barrier(group=cpu_group)

    graph = torch.cuda.CUDAGraph()
    with communicator.capture():
        with torch.cuda.graph(graph):
            fused_prefill_attntp_norm_out(
                communicator.obj,
                *static_inputs,
                output,
                residual_out,
                NORM_EPS,
                NORM_EPS,
            )

    replay_inputs = _make_inputs(tokens, rank, device, seed=4000)
    replay_expected = _reference(replay_inputs, nccl_group)
    for static, replay in zip(static_inputs, replay_inputs):
        static.copy_(replay)
    graph.replay()
    torch.cuda.synchronize()
    _assert_outputs(
        (output, residual_out),
        replay_expected,
        cpu_group=cpu_group,
        nccl_group=nccl_group,
        label="cuda-graph tokens=17",
    )

    communicator.close()


def _assert_source_push_outputs(
    output,
    residual_out,
    expected,
    valid_rows: int,
    *,
    cpu_group,
    label: str,
) -> None:
    expected_output, expected_residual = expected
    actual_output = output[:valid_rows]
    actual_residual = residual_out[:valid_rows]
    output_diff = (actual_output.float() - expected_output.float()).abs()
    residual_diff = (actual_residual - expected_residual).abs()
    output_max = output_diff.max().item() if output_diff.numel() else 0.0
    output_mean = output_diff.mean().item() if output_diff.numel() else 0.0
    residual_max = residual_diff.max().item() if residual_diff.numel() else 0.0
    residual_mean = residual_diff.mean().item() if residual_diff.numel() else 0.0
    output_guard_ok = torch.equal(
        output[valid_rows:],
        torch.full_like(output[valid_rows:], 17.0),
    )
    residual_guard_ok = torch.equal(
        residual_out[valid_rows:],
        torch.full_like(residual_out[valid_rows:], -23.0),
    )
    failed_local = (
        output_max > 0.125
        or output_mean > 2e-3
        or residual_max > 0.125
        or residual_mean > 5e-4
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


@torch.inference_mode()
def _run_source_push_matrix(
    rank,
    device,
    cpu_group,
    nccl_group,
) -> None:
    world_size = dist.get_world_size(group=cpu_group)
    actual_row_counts = tuple(
        dict.fromkeys(
            (
                0,
                1,
                world_size - 1,
                world_size,
                world_size + 1,
                17,
                SOURCE_PUSH_CAPACITY,
            )
        )
    )
    actual_rows = torch.zeros((1,), dtype=torch.int32, device=device)
    owner_start = torch.zeros((1,), dtype=torch.int32, device=device)

    for hidden_size in LOCAL_HIDDEN_SIZES:
        for algorithm in PrefillCommunicationAlgorithm:
            for output_mode in OutputMode:
                _run_prefill_specialization(
                    rank=rank,
                    device=device,
                    cpu_group=cpu_group,
                    nccl_group=nccl_group,
                    world_size=world_size,
                    actual_row_counts=actual_row_counts,
                    actual_rows=actual_rows,
                    owner_start=owner_start,
                    hidden_size=hidden_size,
                    algorithm=algorithm,
                    output_mode=output_mode,
                )


@torch.inference_mode()
def _run_prefill_signal_reuse_stress(
    rank,
    device,
    cpu_group,
    nccl_group,
) -> None:
    world_size = dist.get_world_size(group=cpu_group)
    hidden_size = 2048
    capacities = (17, SOURCE_PUSH_CAPACITY)
    replay_sequence = (
        (0, capacities[0]),
        (1, 1),
        (0, capacities[0] // 2 + 1),
        (1, 0),
        (0, capacities[0] - 1),
        (1, capacities[1]),
        (0, 0),
        (1, capacities[1] // 2 + 1),
        (0, 1),
        (1, capacities[1] - 1),
    )

    for algorithm in PrefillCommunicationAlgorithm:
        spec = AttnTPNormSpec(
            attn_tp_size=world_size,
            hidden_size=hidden_size,
            output_mode=OutputMode.REPLICATED,
        )
        runner = PrefillAttnTPFusedIPCNormRunner(
            group=cpu_group,
            device=device,
            spec=spec,
            capacity=max(capacities),
            algorithm=algorithm,
        )
        actual_rows = torch.zeros((1,), dtype=torch.int32, device=device)
        owner_start = torch.zeros((1,), dtype=torch.int32, device=device)
        try:
            for replay, (runner_index, rows) in enumerate(replay_sequence):
                capacity = capacities[runner_index]
                owner = replay % world_size
                inputs = _make_inputs(
                    capacity,
                    rank=rank,
                    device=device,
                    seed=300000 + replay * 1000,
                    hidden_size=hidden_size,
                )
                peer_partials = [torch.empty_like(inputs[0]) for _ in range(world_size)]
                dist.all_gather(peer_partials, inputs[0], group=nccl_group)
                reference = _ordered_reference(
                    [partial[:rows] for partial in peer_partials],
                    inputs[1][:rows],
                    inputs[2],
                    inputs[3],
                )
                output = torch.full(
                    (runner.output_capacity_for(capacity), hidden_size),
                    17.0,
                    dtype=torch.bfloat16,
                    device=device,
                )
                residual_out = torch.full(
                    (runner.output_capacity_for(capacity), hidden_size),
                    -23.0,
                    dtype=torch.float32,
                    device=device,
                )
                actual_rows.fill_(rows)
                owner_start.fill_(owner)
                runner.run_out(
                    *inputs,
                    output,
                    residual_out,
                    actual_rows,
                    owner_start,
                    NORM_EPS,
                    NORM_EPS,
                )
                torch.cuda.synchronize()
                _assert_source_push_outputs(
                    output,
                    residual_out,
                    reference,
                    rows,
                    cpu_group=cpu_group,
                    label=(
                        f"signal-reuse {algorithm.value} N={world_size} "
                        f"capacity={capacity} owner={owner} rows={rows}"
                    ),
                )
        finally:
            runner.close()


@torch.inference_mode()
def _run_prefill_cuda_graph_varlen(
    rank,
    device,
    cpu_group,
    nccl_group,
    *,
    algorithm,
    output_mode,
    internal_precision=NormInternalPrecision.REFERENCE_BF16,
) -> None:
    world_size = dist.get_world_size(group=cpu_group)
    hidden_size = 2048
    capacities = (17, SOURCE_PUSH_CAPACITY)
    spec = AttnTPNormSpec(
        attn_tp_size=world_size,
        hidden_size=hidden_size,
        output_mode=output_mode,
        internal_precision=internal_precision,
    )
    runner = PrefillAttnTPFusedIPCNormRunner(
        group=cpu_group,
        device=device,
        spec=spec,
        capacity=max(capacities),
        algorithm=algorithm,
    )
    graph_states = []
    graph = None
    state = None

    try:
        for capacity_index, capacity in enumerate(capacities):
            inputs = _make_inputs(
                capacity,
                rank=rank,
                device=device,
                seed=400000 + capacity_index * 1000,
                hidden_size=hidden_size,
            )
            output_capacity = runner.output_capacity_for(capacity)
            output = torch.full(
                (output_capacity, hidden_size),
                17.0,
                dtype=torch.bfloat16,
                device=device,
            )
            residual_out = torch.full(
                (output_capacity, hidden_size),
                -23.0,
                dtype=torch.float32,
                device=device,
            )
            actual_rows = torch.tensor([capacity], dtype=torch.int32, device=device)
            owner_start = torch.zeros((1,), dtype=torch.int32, device=device)

            runner.run_out(
                *inputs,
                output,
                residual_out,
                actual_rows,
                owner_start,
                NORM_EPS,
                NORM_EPS,
            )
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with runner.capture():
                with torch.cuda.graph(graph):
                    runner.run_out(
                        *inputs,
                        output,
                        residual_out,
                        actual_rows,
                        owner_start,
                        NORM_EPS,
                        NORM_EPS,
                    )
            graph_states.append(
                (
                    capacity,
                    inputs,
                    output,
                    residual_out,
                    actual_rows,
                    owner_start,
                    graph,
                )
            )

        for replay in range(5):
            for capacity_index, state in enumerate(graph_states):
                (
                    capacity,
                    inputs,
                    output,
                    residual_out,
                    actual_rows,
                    owner_start,
                    graph,
                ) = state
                rows = (
                    capacity,
                    1,
                    capacity // 2 + 1,
                    0,
                    capacity - 1,
                )[replay]
                replay_inputs = _make_inputs(
                    capacity,
                    rank=rank,
                    device=device,
                    seed=410000 + replay * 1000 + capacity_index * 100,
                    hidden_size=hidden_size,
                )
                for fixed, current in zip(inputs, replay_inputs):
                    fixed.copy_(current)
                peer_partials = [torch.empty_like(inputs[0]) for _ in range(world_size)]
                dist.all_gather(peer_partials, inputs[0], group=nccl_group)
                reference = _reference_for_precision(
                    internal_precision,
                    [partial[:rows] for partial in peer_partials],
                    inputs[1][:rows],
                    inputs[2],
                    inputs[3],
                )
                owner = (replay * len(capacities) + capacity_index) % world_size
                output.fill_(17.0)
                residual_out.fill_(-23.0)
                actual_rows.fill_(rows)
                owner_start.fill_(owner)
                graph.replay()
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
                        attn_tp_size=world_size,
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
                        f"cuda-graph {algorithm.value} N={world_size} "
                        f"capacity={capacity} mode={output_mode.value} "
                        f"precision={internal_precision.value} "
                        f"owner={owner} rows={rows}"
                    ),
                )

        final_replays = [None] * len(graph_states)
        for replay in range(1000):
            capacity_index = replay % len(graph_states)
            (
                capacity,
                _,
                output,
                residual_out,
                actual_rows,
                owner_start,
                graph,
            ) = graph_states[capacity_index]
            rows = (
                capacity,
                1,
                capacity // 2 + 1,
                0,
                capacity - 1,
            )[(replay // len(graph_states)) % 5]
            owner = replay % world_size
            output.fill_(17.0)
            residual_out.fill_(-23.0)
            actual_rows.fill_(rows)
            owner_start.fill_(owner)
            graph.replay()
            final_replays[capacity_index] = (rows, owner)
        torch.cuda.synchronize()

        for state, final_replay in zip(graph_states, final_replays):
            (
                capacity,
                inputs,
                output,
                residual_out,
                _,
                _,
                _,
            ) = state
            rows, owner = final_replay
            peer_partials = [torch.empty_like(inputs[0]) for _ in range(world_size)]
            dist.all_gather(peer_partials, inputs[0], group=nccl_group)
            reference = _reference_for_precision(
                internal_precision,
                [partial[:rows] for partial in peer_partials],
                inputs[1][:rows],
                inputs[2],
                inputs[3],
            )
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
                    f"cuda-graph-stress {algorithm.value} N={world_size} "
                    f"capacity={capacity} mode={output_mode.value} "
                    f"precision={internal_precision.value} "
                    f"owner={owner} rows={rows}"
                ),
            )
    finally:
        _ = None
        graph = None
        state = None
        graph_states.clear()
        runner.close()


def _run_prefill_specialization(
    *,
    rank,
    device,
    cpu_group,
    nccl_group,
    world_size,
    actual_row_counts,
    actual_rows,
    owner_start,
    hidden_size,
    algorithm,
    output_mode,
) -> None:
    spec = AttnTPNormSpec(
        attn_tp_size=world_size,
        hidden_size=hidden_size,
        output_mode=output_mode,
    )
    runner = PrefillAttnTPFusedIPCNormRunner(
        group=cpu_group,
        device=device,
        spec=spec,
        capacity=SOURCE_PUSH_CAPACITY,
        algorithm=algorithm,
    )
    try:
        for owner in range(world_size):
            for rows in actual_row_counts:
                seed = (
                    100000
                    + hidden_size
                    + owner * 1000
                    + rows
                    + list(OutputMode).index(output_mode) * 10000
                    + list(PrefillCommunicationAlgorithm).index(algorithm) * 1000000
                )
                inputs = _make_inputs(
                    SOURCE_PUSH_CAPACITY,
                    rank=rank,
                    device=device,
                    seed=seed,
                    hidden_size=hidden_size,
                )
                peer_partials = [torch.empty_like(inputs[0]) for _ in range(world_size)]
                dist.all_gather(
                    peer_partials,
                    inputs[0],
                    group=nccl_group,
                )
                reference = _ordered_reference(
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
                actual_rows.fill_(rows)
                owner_start.fill_(owner)
                runner.run_out(
                    *inputs,
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
                        attn_tp_size=world_size,
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
                        f"{algorithm.value} N={world_size} "
                        f"H={hidden_size} "
                        f"mode={output_mode.value} owner={owner} "
                        f"rows={rows}"
                    ),
                )
    finally:
        runner.close()


@torch.inference_mode()
def _run_cross_group_isolation(global_rank, device) -> None:
    for subgroup_size in (2, 4):
        groups = []
        selected = None
        for start in range(0, 8, subgroup_size):
            ranks = list(range(start, start + subgroup_size))
            cpu_group = dist.new_group(ranks=ranks, backend="gloo")
            nccl_group = dist.new_group(ranks=ranks, backend="nccl")
            groups.append((ranks, cpu_group, nccl_group))
            if global_rank in ranks:
                selected = (start, cpu_group, nccl_group)

        if selected is None:
            raise RuntimeError("global rank was not assigned to an AttnTP subgroup")
        start, cpu_group, nccl_group = selected
        rank = dist.get_rank(group=cpu_group)
        rows = subgroup_size + 1
        owner = 1
        hidden_size = 2048
        spec = AttnTPNormSpec(
            attn_tp_size=subgroup_size,
            hidden_size=hidden_size,
            output_mode=OutputMode.TOKEN_SCATTERED,
        )
        runner = PrefillAttnTPFusedIPCNormRunner(
            group=cpu_group,
            device=device,
            spec=spec,
            capacity=17,
            algorithm=PrefillCommunicationAlgorithm.SOURCE_PUSH,
        )
        try:
            inputs = _make_inputs(
                17,
                rank=rank,
                device=device,
                seed=200000 + subgroup_size * 1000 + start,
                hidden_size=hidden_size,
            )
            peer_partials = [torch.empty_like(inputs[0]) for _ in range(subgroup_size)]
            dist.all_gather(peer_partials, inputs[0], group=nccl_group)
            reference = _ordered_reference(
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
            actual_rows = torch.tensor([rows], dtype=torch.int32, device=device)
            owner_start = torch.tensor([owner], dtype=torch.int32, device=device)
            runner.run_out(
                *inputs,
                output,
                residual_out,
                actual_rows,
                owner_start,
                NORM_EPS,
                NORM_EPS,
            )
            torch.cuda.synchronize()
            offset, valid_rows = balanced_row_range(
                total_rows=rows,
                rank=rank,
                attn_tp_size=subgroup_size,
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
                label=(f"parallel-groups N={subgroup_size} group_start={start}"),
            )
        finally:
            runner.close()

        dist.barrier()
        for ranks, cpu_group, nccl_group in groups:
            if global_rank in ranks:
                dist.destroy_process_group(nccl_group)
                dist.destroy_process_group(cpu_group)


@torch.inference_mode()
def _run_full_fp32_prefill_smoke(
    rank,
    device,
    cpu_group,
    nccl_group,
) -> None:
    world_size = dist.get_world_size(group=cpu_group)
    capacity = 17
    rows = world_size + 1
    owner = 1 % world_size

    for algorithm in PrefillCommunicationAlgorithm:
        for output_mode in OutputMode:
            for hidden_size in LOCAL_HIDDEN_SIZES:
                spec = AttnTPNormSpec(
                    attn_tp_size=world_size,
                    hidden_size=hidden_size,
                    output_mode=output_mode,
                    internal_precision=NormInternalPrecision.FULL_FP32,
                )
                runner = PrefillAttnTPFusedIPCNormRunner(
                    group=cpu_group,
                    device=device,
                    spec=spec,
                    capacity=capacity,
                    algorithm=algorithm,
                )
                try:
                    inputs = _make_inputs(
                        capacity,
                        rank=rank,
                        device=device,
                        seed=(
                            900000
                            + hidden_size
                            + list(OutputMode).index(output_mode) * 1000
                            + list(PrefillCommunicationAlgorithm).index(algorithm)
                            * 10000
                        ),
                        hidden_size=hidden_size,
                    )
                    peer_partials = [
                        torch.empty_like(inputs[0]) for _ in range(world_size)
                    ]
                    dist.all_gather(
                        peer_partials,
                        inputs[0],
                        group=nccl_group,
                    )
                    reference = _full_fp32_reference(
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
                    actual_rows = torch.tensor([rows], dtype=torch.int32, device=device)
                    owner_start = torch.tensor(
                        [owner], dtype=torch.int32, device=device
                    )
                    runner.run_out(
                        *inputs,
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
                            f"full-fp32 {algorithm.value} N={world_size} "
                            f"H={hidden_size} mode={output_mode.value}"
                        ),
                    )
                finally:
                    runner.close()


@torch.inference_mode()
def _distributed_worker_main() -> None:
    rank, device, cpu_group, nccl_group = _init_distributed()
    torch.cuda.set_stream(torch.cuda.Stream())
    if dist.get_world_size(group=cpu_group) == 2:
        _run_attntp2_regression(
            rank,
            device,
            cpu_group,
            nccl_group,
        )
    _run_source_push_matrix(
        rank,
        device,
        cpu_group,
        nccl_group,
    )
    _run_full_fp32_prefill_smoke(
        rank,
        device,
        cpu_group,
        nccl_group,
    )
    _run_prefill_signal_reuse_stress(
        rank,
        device,
        cpu_group,
        nccl_group,
    )
    if dist.get_world_size() == 8:
        _run_cross_group_isolation(rank, device)
    dist.destroy_process_group()


def worker_main() -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size == 1:
        _local_worker_main()
    else:
        _distributed_worker_main()


if __name__ == "__main__":
    multiprocess_main(__file__, worker_main)
