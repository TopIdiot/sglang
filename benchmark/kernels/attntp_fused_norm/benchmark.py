#!/usr/bin/env python3
"""Benchmark production AttnTP fused IPC norm against NCCL AllReduce + MMQ norm."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, ContextManager

import torch
import torch.distributed as dist

import sglang.srt.distributed.parallel_state as ps
from sglang.jit_kernel.attntp_fused_norm.ipc import (
    AttnTPNormSpec,
    DecodeAttnTPFusedIPCNormRunner,
    DecodeCommunicationAlgorithm,
    NormInternalPrecision,
    OutputMode,
    PrefillAttnTPFusedIPCNormRunner,
    PrefillCommunicationAlgorithm,
    balanced_row_range,
    fused_prefill_attntp_norm_out,
    get_fused_local_attntp_max_occupancy,
)
from sglang.jit_kernel.attntp_fused_norm.symm import (
    DecodeAttnTPFusedSymmNormRunner,
    PrefillAttnTPFusedSymmNormRunner,
)
from sglang.srt.layers.welmv4_op import (
    mmq_style_norm_after_attn,
    welm_use_previous_precision,
)

NORM_EPS = 1e-5
L2_FLUSH_BYTES = 256 * 1024 * 1024
SUPPORTED_ATTN_TP = (1, 2, 4, 8)
DEFAULT_PREFILL_ROWS = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    8192,
    16384,
    32768,
    65536,
)
DEFAULT_DECODE_ROWS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class LatencySummary:
    median_ms: float
    p90_ms: float
    minimum_ms: float
    maximum_ms: float


@dataclass(frozen=True)
class ErrorSummary:
    output_max_abs: float
    output_mean_abs: float
    residual_max_abs: float
    residual_mean_abs: float
    output_guard_ok: bool
    residual_guard_ok: bool


@dataclass
class FusedInvocation:
    operation: Callable[[], None]
    capture_context: Callable[[], ContextManager]
    output: torch.Tensor
    residual_out: torch.Tensor
    close: Callable[[], None]
    algorithm: str
    max_blocks_per_sm: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prefill", "decode", "both"),
        default="both",
    )
    parser.add_argument(
        "--topology",
        choices=("tp", "dp", "cp"),
        default="tp",
    )
    parser.add_argument(
        "--prefill-rows",
        type=int,
        nargs="+",
        default=list(DEFAULT_PREFILL_ROWS),
    )
    parser.add_argument(
        "--decode-rows",
        type=int,
        nargs="+",
        default=list(DEFAULT_DECODE_ROWS),
    )
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        choices=(2048, 4096),
        default=[2048, 4096],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("ipc", "symm"),
        default=["ipc"],
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("source_push", "owner_pull"),
        default=["source_push", "owner_pull"],
    )
    parser.add_argument(
        "--output-modes",
        nargs="+",
        choices=[mode.value for mode in OutputMode],
        default=[OutputMode.TOKEN_SCATTERED.value],
    )
    parser.add_argument(
        "--internal-precision",
        choices=[precision.value for precision in NormInternalPrecision],
        default=NormInternalPrecision.REFERENCE_BF16.value,
    )
    parser.add_argument(
        "--execution",
        choices=("eager", "graph"),
        default="graph",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON result path written by rank 0.",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations positive")
    if any(rows <= 0 for rows in args.prefill_rows):
        parser.error("--prefill-rows must contain positive values")
    if any(rows <= 0 or rows > 256 for rows in args.decode_rows):
        parser.error("--decode-rows must be in [1, 256]")
    args.internal_precision = NormInternalPrecision(args.internal_precision)
    return args


def init_distributed():
    missing = [
        name for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK") if name not in os.environ
    ]
    if missing:
        raise RuntimeError(f"Run with torchrun; missing variables: {missing}")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size not in SUPPORTED_ATTN_TP:
        raise RuntimeError(
            f"AttnTP world size must be one of {SUPPORTED_ATTN_TP}, got {world_size}"
        )
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="gloo")
    ps._WORLD = coordinator = ps.init_world_group(
        ranks=list(range(world_size)),
        local_rank=local_rank,
        backend="nccl",
    )
    device_backend = str(dist.get_backend(coordinator.device_group)).lower()
    if device_backend != "nccl":
        raise RuntimeError(
            "AttnTP fused norm baseline requires an NCCL device process group, "
            f"got {device_backend}"
        )
    return (
        rank,
        world_size,
        device,
        coordinator.cpu_group,
        coordinator.device_group,
    )


def make_inputs(
    rows: int,
    hidden_size: int,
    rank: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rank_generator = torch.Generator(device=device)
    rank_generator.manual_seed(seed + rank)
    shared_generator = torch.Generator(device=device)
    shared_generator.manual_seed(seed + 10000)
    partial = torch.randn(
        (rows, hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=rank_generator,
    )
    residual = torch.randn(
        (rows, hidden_size),
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


def output_capacity(rows: int, world_size: int, mode: OutputMode) -> int:
    if mode is OutputMode.TOKEN_SCATTERED:
        return (rows + world_size - 1) // world_size
    return rows


def estimated_nvlink_payload_bytes(
    *,
    phase: str,
    rows: int,
    hidden_size: int,
    world_size: int,
    algorithm: str,
    mode: OutputMode,
    internal_precision: NormInternalPrecision,
) -> int:
    if world_size == 1:
        return 0
    partial_bytes = rows * hidden_size * torch.bfloat16.itemsize
    if mode is not OutputMode.REPLICATED:
        return partial_bytes * (world_size - 1)
    if phase == "prefill" and algorithm == "source_push":
        gather_multiplier = (
            2 if internal_precision is NormInternalPrecision.FULL_FP32 else 1
        )
        return (1 + gather_multiplier) * partial_bytes * (world_size - 1)
    return partial_bytes * world_size * (world_size - 1)


def update_runtime_metadata(
    *,
    phase: str,
    topology: str,
    rows: int,
    actual_rows,
    lane_rotation,
    lane_rotation_value: int,
) -> None:
    if phase == "decode":
        if actual_rows is not None or lane_rotation is not None:
            raise ValueError("Decode benchmark must not receive Prefill metadata")
        return
    if phase != "prefill":
        raise ValueError("phase must be prefill or decode")
    if topology not in ("tp", "dp", "cp"):
        raise ValueError("topology must be tp, dp, or cp")
    if actual_rows is None or lane_rotation is None:
        raise ValueError("Prefill benchmark requires runtime metadata tensors")
    actual_rows.fill_(rows)
    if topology == "cp":
        lane_rotation.fill_(lane_rotation_value)


def create_fused_invocation(
    *,
    backend: str,
    phase: str,
    algorithm: str,
    spec: AttnTPNormSpec,
    inputs,
    rows: int,
    topology: str,
    lane_rotation: torch.Tensor | None,
    lane_rotation_value: int,
    cpu_group,
    device: torch.device,
    block_size: int = 256,
    signal_backoff: int = 64,
    rows_per_tile: int = 1,
    blocks_per_sm: int | None = None,
) -> FusedInvocation:
    partial, residual, o_weight, post_weight = inputs
    output_rows = output_capacity(rows, spec.attn_tp_size, spec.output_mode)
    output = torch.full(
        (output_rows, spec.hidden_size),
        17.0,
        dtype=torch.bfloat16,
        device=device,
    )
    residual_out = torch.full(
        (output_rows, spec.hidden_size),
        -23.0,
        dtype=torch.float32,
        device=device,
    )

    if backend == "symm":
        if phase == "prefill":
            runner = PrefillAttnTPFusedSymmNormRunner(
                group=cpu_group,
                device=device,
                spec=spec,
                capacity=rows,
                block_size=block_size,
                signal_backoff=signal_backoff,
                blocks_per_sm=blocks_per_sm,
            )
            actual_rows = torch.tensor([rows], dtype=torch.int32, device=device)

            def operation() -> None:
                update_runtime_metadata(
                    phase=phase,
                    topology=topology,
                    rows=rows,
                    actual_rows=actual_rows,
                    lane_rotation=lane_rotation,
                    lane_rotation_value=lane_rotation_value,
                )
                runner.input_view.copy_(partial)
                runner.run_out(
                    residual,
                    o_weight,
                    post_weight,
                    output,
                    residual_out,
                    actual_rows,
                    lane_rotation,
                    NORM_EPS,
                    NORM_EPS,
                )

        else:
            runner = DecodeAttnTPFusedSymmNormRunner(
                group=cpu_group,
                device=device,
                spec=spec,
                max_rows=rows,
                block_size=block_size,
                signal_backoff=signal_backoff,
                blocks_per_sm=blocks_per_sm,
            )

            def operation() -> None:
                runner.input_view[:rows].copy_(partial)
                runner.run_out(
                    residual,
                    o_weight,
                    post_weight,
                    output,
                    residual_out,
                    NORM_EPS,
                    NORM_EPS,
                )

        if spec.attn_tp_size in (4, 8):
            symm_algorithm = (
                "symm_multimem"
                if spec.internal_precision is NormInternalPrecision.REFERENCE_BF16
                else "symm_pointer_table_fp32"
            )
        else:
            symm_algorithm = "symm_peer_fp32"

        return FusedInvocation(
            operation=operation,
            capture_context=runner.capture,
            output=output,
            residual_out=residual_out,
            close=runner.close,
            algorithm=symm_algorithm,
            max_blocks_per_sm=runner.get_max_occupancy(),
        )

    if backend != "ipc":
        raise ValueError(f"unknown fused backend: {backend}")

    if phase == "prefill" and spec.attn_tp_size == 1:

        def operation() -> None:
            fused_prefill_attntp_norm_out(
                None,
                partial,
                residual,
                o_weight,
                post_weight,
                output,
                residual_out,
                NORM_EPS,
                NORM_EPS,
                attn_tp_size=1,
                internal_precision=spec.internal_precision,
            )

        return FusedInvocation(
            operation=operation,
            capture_context=nullcontext,
            output=output,
            residual_out=residual_out,
            close=lambda: None,
            algorithm="local",
            max_blocks_per_sm=get_fused_local_attntp_max_occupancy(
                spec.hidden_size,
                internal_precision=spec.internal_precision,
            ),
        )

    if phase == "prefill":
        runner = PrefillAttnTPFusedIPCNormRunner(
            group=cpu_group,
            device=device,
            spec=spec,
            capacity=rows,
            algorithm=PrefillCommunicationAlgorithm(algorithm),
            block_size=block_size,
            signal_backoff=signal_backoff,
            rows_per_tile=rows_per_tile,
            blocks_per_sm=blocks_per_sm,
        )
        actual_rows = torch.tensor([rows], dtype=torch.int32, device=device)

        def operation() -> None:
            update_runtime_metadata(
                phase=phase,
                topology=topology,
                rows=rows,
                actual_rows=actual_rows,
                lane_rotation=lane_rotation,
                lane_rotation_value=lane_rotation_value,
            )
            runner.run_out(
                partial,
                residual,
                o_weight,
                post_weight,
                output,
                residual_out,
                actual_rows,
                lane_rotation,
                NORM_EPS,
                NORM_EPS,
            )

    else:
        runner = DecodeAttnTPFusedIPCNormRunner(
            group=cpu_group,
            device=device,
            spec=spec,
            max_rows=rows,
            algorithm=DecodeCommunicationAlgorithm(algorithm),
            block_size=block_size,
            signal_backoff=signal_backoff,
            blocks_per_sm=blocks_per_sm,
        )

        def operation() -> None:
            runner.run_out(
                partial,
                residual,
                o_weight,
                post_weight,
                output,
                residual_out,
                NORM_EPS,
                NORM_EPS,
            )

    return FusedInvocation(
        operation=operation,
        capture_context=runner.capture,
        output=output,
        residual_out=residual_out,
        close=runner.close,
        algorithm=algorithm,
        max_blocks_per_sm=runner.get_max_occupancy(),
    )


def baseline_algorithm_for_mode(mode: OutputMode) -> str:
    if mode is OutputMode.TOKEN_SCATTERED:
        return "nccl_reduce_scatter_local_mmq"
    return "nccl_all_reduce_full_mmq"


def create_baseline_operation(
    inputs,
    nccl_group,
    *,
    mode: OutputMode,
    rank: int,
    world_size: int,
    owner: int,
):
    partial, residual, o_weight, post_weight = inputs

    if mode is OutputMode.TOKEN_SCATTERED:
        if owner != 0:
            raise ValueError("token-scattered NCCL baseline requires owner_start=0")
        ranges = [
            balanced_row_range(
                total_rows=partial.shape[0],
                rank=destination,
                attn_tp_size=world_size,
                owner_start=0,
            )
            for destination in range(world_size)
        ]
        sizes = [count for _, count in ranges]
        offset, count = ranges[rank]
        reduced = torch.empty(
            (count, partial.shape[1]),
            dtype=partial.dtype,
            device=partial.device,
        )
        residual_local = residual[offset : offset + count]
        result = {}
        empty_outputs = (
            torch.empty_like(reduced),
            torch.empty_like(reduced, dtype=torch.float32),
            torch.empty_like(reduced, dtype=torch.float32),
        )
        equal_counts = all(size == sizes[0] for size in sizes)

        def prepare() -> None:
            pass

        def operation() -> None:
            if equal_counts:
                dist.reduce_scatter_tensor(
                    reduced,
                    partial,
                    group=nccl_group,
                )
            else:
                for destination, (chunk_offset, chunk_count) in enumerate(ranges):
                    if chunk_count == 0:
                        continue
                    chunk = partial[chunk_offset : chunk_offset + chunk_count]
                    if rank == destination:
                        reduced.copy_(chunk)
                        chunk = reduced
                    dist.reduce(chunk, dst=destination, group=nccl_group)
            if count == 0:
                result["outputs"] = empty_outputs
                return
            result["outputs"] = mmq_style_norm_after_attn(
                reduced,
                residual_local,
                o_weight,
                post_weight,
                NORM_EPS,
            )

        return prepare, operation, result

    reduced = torch.empty_like(partial)
    result = {}

    def prepare() -> None:
        reduced.copy_(partial)

    def operation() -> None:
        dist.all_reduce(reduced, group=nccl_group)
        result["outputs"] = mmq_style_norm_after_attn(
            reduced,
            residual,
            o_weight,
            post_weight,
            NORM_EPS,
        )

    return prepare, operation, result


def create_fp32_correctness_reference(inputs, nccl_group):
    partial, residual, o_weight, post_weight = inputs
    reduced_fp32 = partial.float()
    dist.all_reduce(reduced_fp32, group=nccl_group)
    return mmq_style_norm_after_attn(
        reduced_fp32.to(torch.bfloat16),
        residual,
        o_weight,
        post_weight,
        NORM_EPS,
    )


def create_full_fp32_correctness_reference(inputs, nccl_group):
    partial, residual, o_weight, post_weight = inputs
    reduced = partial.float()
    dist.all_reduce(reduced, group=nccl_group)
    o_norm_scale = torch.rsqrt(reduced.square().mean(dim=-1, keepdim=True) + NORM_EPS)
    o_norm = reduced * o_norm_scale * o_weight.float()
    residual_out = residual + o_norm
    post_norm_scale = torch.rsqrt(
        residual_out.square().mean(dim=-1, keepdim=True) + NORM_EPS
    )
    output = (residual_out * post_norm_scale * post_weight.float()).to(torch.bfloat16)
    return output, residual_out, None


def capture_operation(
    operation: Callable[[], None],
    capture_context: Callable[[], ContextManager],
    cpu_group,
) -> torch.cuda.CUDAGraph:
    dist.barrier(group=cpu_group)
    graph = torch.cuda.CUDAGraph()
    with capture_context():
        with torch.cuda.graph(graph):
            operation()
    dist.barrier(group=cpu_group)
    return graph


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples: list[float]) -> LatencySummary:
    return LatencySummary(
        median_ms=float(statistics.median(samples)),
        p90_ms=percentile(samples, 0.9),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
    )


def group_max(value: float, cpu_group) -> float:
    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=cpu_group)
    return float(tensor.item())


def group_max_samples(samples: list[float], cpu_group) -> list[float]:
    tensor = torch.tensor(samples, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=cpu_group)
    return tensor.tolist()


def group_all(value: bool, cpu_group) -> bool:
    tensor = torch.tensor([int(value)], dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(tensor.item())


def measure(
    operation: Callable[[], None],
    prepare: Callable[[], None],
    flush_l2: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    cpu_group,
) -> tuple[LatencySummary, list[float]]:
    dist.barrier(group=cpu_group)
    for _ in range(warmup):
        prepare()
        flush_l2()
        operation()
    torch.cuda.synchronize()

    dist.barrier(group=cpu_group)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(start_events, end_events):
        prepare()
        flush_l2()
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()

    samples = group_max_samples(
        [start.elapsed_time(end) for start, end in zip(start_events, end_events)],
        cpu_group,
    )
    return summarize(samples), samples


def expected_range(
    *,
    rows: int,
    rank: int,
    world_size: int,
    owner: int,
    mode: OutputMode,
) -> tuple[int, int]:
    if mode is OutputMode.REPLICATED:
        return 0, rows
    if mode is OutputMode.SINGLE_CONTRIBUTOR:
        return 0, rows if rank == owner else 0
    return balanced_row_range(
        total_rows=rows,
        rank=rank,
        attn_tp_size=world_size,
        owner_start=owner,
    )


def compare_outputs(
    fused: FusedInvocation,
    reference_outputs,
    *,
    rows: int,
    rank: int,
    world_size: int,
    owner: int,
    mode: OutputMode,
    cpu_group,
) -> ErrorSummary:
    reference_output, reference_residual, _ = reference_outputs
    offset, valid_rows = expected_range(
        rows=rows,
        rank=rank,
        world_size=world_size,
        owner=owner,
        mode=mode,
    )
    expected_output = reference_output[offset : offset + valid_rows]
    expected_residual = reference_residual[offset : offset + valid_rows]
    output_diff = (fused.output[:valid_rows].float() - expected_output.float()).abs()
    residual_diff = (fused.residual_out[:valid_rows] - expected_residual).abs()
    local = ErrorSummary(
        output_max_abs=(output_diff.max().item() if output_diff.numel() else 0.0),
        output_mean_abs=(output_diff.mean().item() if output_diff.numel() else 0.0),
        residual_max_abs=(residual_diff.max().item() if residual_diff.numel() else 0.0),
        residual_mean_abs=(
            residual_diff.mean().item() if residual_diff.numel() else 0.0
        ),
        output_guard_ok=torch.equal(
            fused.output[valid_rows:],
            torch.full_like(fused.output[valid_rows:], 17.0),
        ),
        residual_guard_ok=torch.equal(
            fused.residual_out[valid_rows:],
            torch.full_like(fused.residual_out[valid_rows:], -23.0),
        ),
    )
    return ErrorSummary(
        output_max_abs=group_max(local.output_max_abs, cpu_group),
        output_mean_abs=group_max(local.output_mean_abs, cpu_group),
        residual_max_abs=group_max(local.residual_max_abs, cpu_group),
        residual_mean_abs=group_max(local.residual_mean_abs, cpu_group),
        output_guard_ok=group_all(local.output_guard_ok, cpu_group),
        residual_guard_ok=group_all(local.residual_guard_ok, cpu_group),
    )


def compare_baseline_outputs(
    baseline_outputs,
    reference_outputs,
    *,
    rows: int,
    rank: int,
    world_size: int,
    owner: int,
    mode: OutputMode,
    cpu_group,
) -> ErrorSummary:
    output, residual_out, _ = baseline_outputs
    reference_output, reference_residual, _ = reference_outputs
    offset, valid_rows = expected_range(
        rows=rows,
        rank=rank,
        world_size=world_size,
        owner=owner,
        mode=mode,
    )
    if mode is OutputMode.TOKEN_SCATTERED:
        expected_shape = (valid_rows, reference_output.shape[1])
        if output.shape != expected_shape or residual_out.shape != expected_shape:
            raise RuntimeError(
                "token-scattered baseline output shape mismatch: "
                f"got {tuple(output.shape)} and {tuple(residual_out.shape)}, "
                f"expected {expected_shape}"
            )
        actual_output = output
        actual_residual = residual_out
    else:
        actual_output = output[offset : offset + valid_rows]
        actual_residual = residual_out[offset : offset + valid_rows]

    expected_output = reference_output[offset : offset + valid_rows]
    expected_residual = reference_residual[offset : offset + valid_rows]
    output_diff = (actual_output.float() - expected_output.float()).abs()
    residual_diff = (actual_residual - expected_residual).abs()
    local = ErrorSummary(
        output_max_abs=(output_diff.max().item() if output_diff.numel() else 0.0),
        output_mean_abs=(output_diff.mean().item() if output_diff.numel() else 0.0),
        residual_max_abs=(residual_diff.max().item() if residual_diff.numel() else 0.0),
        residual_mean_abs=(
            residual_diff.mean().item() if residual_diff.numel() else 0.0
        ),
        output_guard_ok=True,
        residual_guard_ok=True,
    )
    return ErrorSummary(
        output_max_abs=group_max(local.output_max_abs, cpu_group),
        output_mean_abs=group_max(local.output_mean_abs, cpu_group),
        residual_max_abs=group_max(local.residual_max_abs, cpu_group),
        residual_mean_abs=group_max(local.residual_mean_abs, cpu_group),
        output_guard_ok=True,
        residual_guard_ok=True,
    )


def assert_correct(errors: ErrorSummary, label: str) -> None:
    if (
        errors.output_max_abs > 0.125
        or errors.output_mean_abs > 2e-3
        or errors.residual_max_abs > 0.125
        or errors.residual_mean_abs > 5e-4
        or not errors.output_guard_ok
        or not errors.residual_guard_ok
    ):
        raise RuntimeError(f"{label} correctness failed: {errors}")


def assert_baseline_correct(errors: ErrorSummary, label: str) -> None:
    if (
        errors.output_max_abs > 0.5
        or errors.output_mean_abs > 1e-2
        or errors.residual_max_abs > 0.5
        or errors.residual_mean_abs > 1e-2
        or not errors.output_guard_ok
        or not errors.residual_guard_ok
    ):
        raise RuntimeError(f"{label} correctness failed: {errors}")


def run_case(
    *,
    backend: str,
    phase: str,
    topology: str,
    rows: int,
    hidden_size: int,
    algorithm: str,
    mode: OutputMode,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
    cpu_group,
    nccl_group,
    flush_l2: Callable[[], None],
    block_size: int = 256,
    signal_backoff: int = 64,
    rows_per_tile: int = 1,
    blocks_per_sm: int | None = None,
    measure_baseline: bool = True,
    inputs_override=None,
    check_correctness: bool = True,
    lane_rotation_override: int | None = None,
) -> dict:
    seed = 100000 + rows + hidden_size + list(OutputMode).index(mode) * 1000
    inputs = (
        make_inputs(rows, hidden_size, rank, device, seed)
        if inputs_override is None
        else inputs_override
    )
    rotation_seed = (
        rows.bit_length()
        + hidden_size // 2048
        + int(algorithm == "owner_pull")
        + list(OutputMode).index(mode)
    )
    if lane_rotation_override is not None:
        if phase != "prefill" or topology != "cp":
            raise ValueError("lane_rotation_override is valid only for Prefill CP")
        if (
            type(lane_rotation_override) is not int
            or lane_rotation_override < 0
            or lane_rotation_override >= world_size
        ):
            raise ValueError(f"lane_rotation_override must be in [0, {world_size})")
        owner = lane_rotation_override
    elif phase != "prefill" or topology != "cp":
        owner = 0
    elif measure_baseline and mode is OutputMode.TOKEN_SCATTERED:
        owner = 0
    else:
        owner = rotation_seed % world_size
    lane_rotation = (
        torch.zeros((1,), dtype=torch.int32, device=device)
        if phase == "prefill"
        else None
    )
    spec = AttnTPNormSpec(
        attn_tp_size=world_size,
        hidden_size=hidden_size,
        output_mode=mode,
        internal_precision=args.internal_precision,
    )
    fused = create_fused_invocation(
        backend=backend,
        phase=phase,
        algorithm=algorithm,
        spec=spec,
        inputs=inputs,
        rows=rows,
        topology=topology,
        lane_rotation=lane_rotation,
        lane_rotation_value=owner,
        cpu_group=cpu_group,
        device=device,
        block_size=block_size,
        signal_backoff=signal_backoff,
        rows_per_tile=rows_per_tile,
        blocks_per_sm=blocks_per_sm,
    )
    baseline_prepare = None
    baseline_operation = None
    baseline_result = None
    if measure_baseline:
        baseline_prepare, baseline_operation, baseline_result = (
            create_baseline_operation(
                inputs,
                nccl_group,
                mode=mode,
                rank=rank,
                world_size=world_size,
                owner=owner,
            )
        )
    baseline_graph = None
    fused_graph = None
    baseline_timed = None
    fused_timed = None
    try:
        if measure_baseline:
            baseline_prepare()
            baseline_operation()
        errors = None
        baseline_errors = None
        if check_correctness:
            correctness_reference_fn = (
                create_fp32_correctness_reference
                if args.internal_precision is NormInternalPrecision.REFERENCE_BF16
                else create_full_fp32_correctness_reference
            )
            correctness_reference = correctness_reference_fn(inputs, nccl_group)
            if measure_baseline:
                baseline_errors = compare_baseline_outputs(
                    baseline_result["outputs"],
                    correctness_reference,
                    rows=rows,
                    rank=rank,
                    world_size=world_size,
                    owner=owner,
                    mode=mode,
                    cpu_group=cpu_group,
                )
                assert_baseline_correct(
                    baseline_errors,
                    f"baseline {phase} N={world_size} H={hidden_size} "
                    f"rows={rows} mode={mode.value}",
                )
            fused.operation()
            torch.cuda.synchronize()
            errors = compare_outputs(
                fused,
                correctness_reference,
                rows=rows,
                rank=rank,
                world_size=world_size,
                owner=owner,
                mode=mode,
                cpu_group=cpu_group,
            )
            assert_correct(
                errors,
                f"{phase} N={world_size} H={hidden_size} rows={rows} "
                f"algorithm={fused.algorithm} mode={mode.value} "
                f"precision={args.internal_precision.value}",
            )

        baseline_latency = None
        baseline_samples = None
        fused_latency = None
        fused_samples = None
        if not args.correctness_only:
            if args.execution == "graph":
                if measure_baseline:
                    baseline_prepare()
                    baseline_graph = capture_operation(
                        baseline_operation,
                        nullcontext,
                        cpu_group,
                    )
                fused_graph = capture_operation(
                    fused.operation,
                    fused.capture_context,
                    cpu_group,
                )
                if measure_baseline:
                    baseline_timed = baseline_graph.replay
                fused_timed = fused_graph.replay
            else:
                if measure_baseline:
                    baseline_timed = baseline_operation
                fused_timed = fused.operation
            if measure_baseline:
                baseline_latency, baseline_samples = measure(
                    baseline_timed,
                    baseline_prepare,
                    flush_l2,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    cpu_group=cpu_group,
                )
            fused_latency, fused_samples = measure(
                fused_timed,
                lambda: None,
                flush_l2,
                warmup=args.warmup,
                iterations=args.iterations,
                cpu_group=cpu_group,
            )

        speedup = (
            None
            if baseline_latency is None or fused_latency is None
            else baseline_latency.median_ms / fused_latency.median_ms
        )
        nvlink_payload_bytes = estimated_nvlink_payload_bytes(
            phase=phase,
            rows=rows,
            hidden_size=hidden_size,
            world_size=world_size,
            algorithm=fused.algorithm,
            mode=mode,
            internal_precision=args.internal_precision,
        )
        fused_nvlink_payload_gbps = (
            None
            if fused_latency is None
            else nvlink_payload_bytes / fused_latency.median_ms / 1e6
        )
        _, local_valid_rows = expected_range(
            rows=rows,
            rank=rank,
            world_size=world_size,
            owner=owner,
            mode=mode,
        )
        return {
            "phase": phase,
            "topology": topology,
            "backend": backend,
            "rows": rows,
            "hidden_size": hidden_size,
            "algorithm": fused.algorithm,
            "output_mode": mode.value,
            "internal_precision": args.internal_precision.value,
            "block_size": block_size,
            "signal_backoff": signal_backoff,
            "rows_per_tile": rows_per_tile,
            "selected_blocks_per_sm": blocks_per_sm,
            "lane_rotation": (
                owner if phase == "prefill" and topology == "cp" else None
            ),
            "baseline_algorithm": (
                baseline_algorithm_for_mode(mode) if measure_baseline else None
            ),
            "input_partial_bytes_per_rank": rows * hidden_size * 2,
            "estimated_nvlink_payload_bytes_group": nvlink_payload_bytes,
            "estimated_nvlink_payload_gbps_group": fused_nvlink_payload_gbps,
            "fused_output_rows_per_rank_capacity": fused.output.shape[0],
            "fused_output_rows_local_valid": local_valid_rows,
            "fused_max_blocks_per_sm": fused.max_blocks_per_sm,
            "baseline": (
                None
                if baseline_latency is None
                else {
                    **asdict(baseline_latency),
                    "samples_ms": baseline_samples,
                }
            ),
            "fused": (
                None
                if fused_latency is None
                else {
                    **asdict(fused_latency),
                    "samples_ms": fused_samples,
                }
            ),
            "speedup_median": speedup,
            "baseline_errors": (
                None if baseline_errors is None else asdict(baseline_errors)
            ),
            "errors": None if errors is None else asdict(errors),
        }
    finally:
        if baseline_timed is not None:
            del baseline_timed
        if fused_timed is not None:
            del fused_timed
        if baseline_graph is not None:
            del baseline_graph
        if fused_graph is not None:
            del fused_graph
        fused.close()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if welm_use_previous_precision():
        raise RuntimeError(
            "V2 fused IPC norm does not implement the previous-precision WeLM path"
        )
    rank, world_size, device, cpu_group, nccl_group = init_distributed()
    torch.cuda.set_stream(torch.cuda.Stream())
    l2_flush_buffer = torch.empty(
        L2_FLUSH_BYTES // torch.empty((), dtype=torch.int32).element_size(),
        dtype=torch.int32,
        device=device,
    )

    def flush_l2() -> None:
        l2_flush_buffer.zero_()

    phases = ("prefill", "decode") if args.phase == "both" else (args.phase,)
    results = []
    try:
        for phase in phases:
            row_values = args.prefill_rows if phase == "prefill" else args.decode_rows
            for hidden_size in args.hidden_sizes:
                for rows in row_values:
                    for mode_name in args.output_modes:
                        mode = OutputMode(mode_name)
                        for backend in args.backends:
                            if backend == "symm":
                                algorithms = ["symm_direct"]
                            else:
                                algorithms = (
                                    ["local"] if world_size == 1 else args.algorithms
                                )
                            for algorithm in algorithms:
                                row = run_case(
                                    backend=backend,
                                    phase=phase,
                                    topology=args.topology,
                                    rows=rows,
                                    hidden_size=hidden_size,
                                    algorithm=(
                                        "source_push"
                                        if algorithm == "local"
                                        else algorithm
                                    ),
                                    mode=mode,
                                    args=args,
                                    rank=rank,
                                    world_size=world_size,
                                    device=device,
                                    cpu_group=cpu_group,
                                    nccl_group=nccl_group,
                                    flush_l2=flush_l2,
                                )
                                results.append(row)
                                if rank == 0:
                                    print(json.dumps(row, sort_keys=True), flush=True)

        payload = {
            "gpu": torch.cuda.get_device_name(device),
            "attn_tp_size": world_size,
            "topology": args.topology,
            "execution": args.execution,
            "internal_precision": args.internal_precision.value,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "l2_flush_bytes_per_replay": L2_FLUSH_BYTES,
            "baseline_contract": (
                "token_scattered uses BF16 NCCL reduce_scatter(v) + local "
                "mmq_style_norm_after_attn; other modes use BF16 NCCL "
                "all_reduce + full mmq_style_norm_after_attn; both include "
                "MMQ's unused FP32 normalized output"
            ),
            "correctness_reference_contract": (
                "FP32 NCCL all_reduce, one BF16 cast, then "
                "mmq_style_norm_after_attn; not timed"
                if args.internal_precision is NormInternalPrecision.REFERENCE_BF16
                else "FP32 NCCL all_reduce, FP32 O-Norm, FP32 residual "
                "add, FP32 Post-Norm, then one BF16 hidden cast; not timed"
            ),
            "fused_contract": (
                "rank-order FP32 partial accumulation, one BF16 cast, "
                "O-Norm, FP32 residual add, one BF16 cast, Post-Norm; "
                "returns BF16 hidden and FP32 residual only"
                if args.internal_precision is NormInternalPrecision.REFERENCE_BF16
                else "rank-order FP32 partial accumulation, FP32 O-Norm, "
                "FP32 residual add, FP32 Post-Norm, then one BF16 hidden "
                "cast; returns BF16 hidden and FP32 residual only"
            ),
            "results": results,
        }
        if rank == 0 and args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
