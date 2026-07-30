"""AttnTP fused IPC and WeLM norm kernel family."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist
from tvm_ffi import Module

from sglang.jit_kernel.attntp_fused_norm.resources import AttnTPIPCResource
from sglang.jit_kernel.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
)

_WORLD_SIZE = 2
_HIDDEN_SIZE = 2048
_ALIGNMENT = 128
_SUPPORTED_ROWS_PER_SIGNAL = (1, 2, 4, 8)
_SUPPORTED_CACHE_MODES = ("peer", "local")
_SUPPORTED_MIN_BLOCKS_PER_SM = (4, 5, 6)
_SUPPORTED_ATTN_TP_SIZES = (1, 2, 4, 8)
_SUPPORTED_HIDDEN_SIZES = (2048, 4096)
_SUPPORTED_BLOCK_SIZES = (128, 256, 512)
_SUPPORTED_SIGNAL_BACKOFFS = (32, 64, 128, 256)
_SUPPORTED_SOURCE_PUSH_ROWS_PER_TILE = (1, 2, 4, 8)
_SOURCE_PUSH_ROWS_PER_TILE = 1
MAX_DECODE_ROWS = 256


class OutputMode(str, Enum):
    REPLICATED = "replicated"
    SINGLE_CONTRIBUTOR = "single_contributor"
    TOKEN_SCATTERED = "token_scattered"


class NormInternalPrecision(str, Enum):
    REFERENCE_BF16 = "reference_bf16"
    FULL_FP32 = "full_fp32"


class PrefillCommunicationAlgorithm(str, Enum):
    SOURCE_PUSH = "source_push"
    OWNER_PULL = "owner_pull"


class DecodeCommunicationAlgorithm(str, Enum):
    SOURCE_PUSH = "source_push"
    OWNER_PULL = "owner_pull"


_OUTPUT_MODE_CPP_VALUE = {
    OutputMode.REPLICATED: 0,
    OutputMode.SINGLE_CONTRIBUTOR: 1,
    OutputMode.TOKEN_SCATTERED: 2,
}

_NORM_INTERNAL_PRECISION_CPP_VALUE = {
    NormInternalPrecision.REFERENCE_BF16: 0,
    NormInternalPrecision.FULL_FP32: 1,
}


def _validate_attn_tp_size(attn_tp_size: int) -> None:
    if type(attn_tp_size) is not int or attn_tp_size not in _SUPPORTED_ATTN_TP_SIZES:
        raise ValueError(
            f"AttnTP size must be one of {_SUPPORTED_ATTN_TP_SIZES}, "
            f"got {attn_tp_size!r}"
        )


def _validate_generalized_kernel_tuning(
    *,
    hidden_size: int,
    block_size: int,
    signal_backoff: int | None,
    rows_per_tile: int | None,
    blocks_per_sm: int | None,
) -> None:
    if type(block_size) is not int or block_size not in _SUPPORTED_BLOCK_SIZES:
        raise ValueError(f"block_size must be one of {_SUPPORTED_BLOCK_SIZES}")
    vector_denominator = block_size * 8
    if (
        hidden_size % vector_denominator != 0
        or hidden_size // vector_denominator not in (1, 2, 4)
    ):
        raise ValueError(
            f"block_size {block_size} is unsupported for hidden size {hidden_size}"
        )
    if signal_backoff is not None and (
        type(signal_backoff) is not int
        or signal_backoff not in _SUPPORTED_SIGNAL_BACKOFFS
    ):
        raise ValueError(f"signal_backoff must be one of {_SUPPORTED_SIGNAL_BACKOFFS}")
    if rows_per_tile is not None and (
        type(rows_per_tile) is not int
        or rows_per_tile not in _SUPPORTED_SOURCE_PUSH_ROWS_PER_TILE
    ):
        raise ValueError(
            f"rows_per_tile must be one of {_SUPPORTED_SOURCE_PUSH_ROWS_PER_TILE}"
        )
    if blocks_per_sm is not None and (
        type(blocks_per_sm) is not int or blocks_per_sm <= 0
    ):
        raise ValueError("blocks_per_sm must be a positive integer or None")


@dataclass(frozen=True)
class AttnTPNormSpec:
    attn_tp_size: int
    hidden_size: int
    output_mode: OutputMode
    dtype: torch.dtype = torch.bfloat16
    internal_precision: NormInternalPrecision = NormInternalPrecision.REFERENCE_BF16

    def __post_init__(self) -> None:
        _validate_attn_tp_size(self.attn_tp_size)
        if (
            type(self.hidden_size) is not int
            or self.hidden_size not in _SUPPORTED_HIDDEN_SIZES
        ):
            raise ValueError(
                f"hidden size must be one of {_SUPPORTED_HIDDEN_SIZES}, "
                f"got {self.hidden_size!r}"
            )
        if self.dtype is not torch.bfloat16:
            raise ValueError("AttnTP fused IPC norm requires BF16")
        if not isinstance(self.output_mode, OutputMode):
            raise ValueError("output_mode must be an OutputMode")
        if not isinstance(self.internal_precision, NormInternalPrecision):
            raise ValueError("internal_precision must be a NormInternalPrecision")


def balanced_row_range(
    *,
    total_rows: int,
    rank: int,
    attn_tp_size: int,
    owner_start: int = 0,
) -> tuple[int, int]:
    _validate_attn_tp_size(attn_tp_size)
    if type(total_rows) is not int or total_rows < 0:
        raise ValueError("total_rows must be a non-negative integer")
    if type(rank) is not int or rank < 0 or rank >= attn_tp_size:
        raise ValueError(f"rank must be in [0, {attn_tp_size}), got {rank!r}")
    if type(owner_start) is not int or owner_start < 0 or owner_start >= attn_tp_size:
        raise ValueError(
            f"owner_start must be in [0, {attn_tp_size}), got {owner_start!r}"
        )

    logical_rank = (rank - owner_start) % attn_tp_size
    base, remainder = divmod(total_rows, attn_tp_size)
    count = base + int(logical_rank < remainder)
    offset = logical_rank * base + min(logical_rank, remainder)
    return offset, count


def output_rows_for_rank(
    *,
    total_rows: int,
    rank: int,
    attn_tp_size: int,
    owner_start: int,
    output_mode: OutputMode,
) -> int:
    offset, scattered_count = balanced_row_range(
        total_rows=total_rows,
        rank=rank,
        attn_tp_size=attn_tp_size,
        owner_start=owner_start,
    )
    del offset
    if not isinstance(output_mode, OutputMode):
        raise ValueError("output_mode must be an OutputMode")
    if output_mode is OutputMode.REPLICATED:
        return total_rows
    if output_mode is OutputMode.SINGLE_CONTRIBUTOR:
        return total_rows if rank == owner_start else 0
    return scattered_count


def _select_fused_prefill_attntp_specialization(
    tokens: int,
) -> tuple[str, int, int]:
    # Tuned on H20 with four concurrent PrefillAttnTP pairs.
    tokens = int(tokens)
    if tokens <= 0:
        raise ValueError("tokens must be positive for specialization")
    if tokens == 1:
        return "local", 4, 4
    if tokens <= 256:
        return "peer", 1, 4
    if tokens <= 768:
        return "local", 4, 4
    return "local", 8, 4


def _align_bytes(size: int) -> int:
    return (int(size) + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def required_push_buffer_bytes(
    tokens: int,
    hidden_size: int = _HIDDEN_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    tokens = int(tokens)
    hidden_size = int(hidden_size)
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    if hidden_size != _HIDDEN_SIZE:
        raise ValueError(
            f"PrefillAttnTP fused IPC norm requires hidden size {_HIDDEN_SIZE}"
        )
    if dtype != torch.bfloat16:
        raise ValueError("PrefillAttnTP fused IPC norm requires BF16 partials")
    data_bytes = tokens * hidden_size * torch.bfloat16.itemsize
    signal_bytes = tokens * 4
    return _align_bytes(data_bytes) + _align_bytes(signal_bytes)


def source_push_output_capacity(
    capacity: int,
    spec: AttnTPNormSpec,
) -> int:
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if spec.attn_tp_size == 1:
        raise NotImplementedError(
            "source-push communication requires AttnTP size 2, 4, or 8"
        )
    if spec.output_mode in (
        OutputMode.REPLICATED,
        OutputMode.SINGLE_CONTRIBUTOR,
    ):
        return capacity
    return (capacity + spec.attn_tp_size - 1) // spec.attn_tp_size


def required_source_push_buffer_bytes(
    capacity: int,
    spec: AttnTPNormSpec,
) -> int:
    source_push_output_capacity(capacity, spec)
    owner_capacity = (
        capacity
        if spec.output_mode is OutputMode.SINGLE_CONTRIBUTOR
        else (capacity + spec.attn_tp_size - 1) // spec.attn_tp_size
    )
    source_payload_bytes = _align_bytes(
        owner_capacity * spec.hidden_size * torch.bfloat16.itemsize
    )
    max_tiles = (
        owner_capacity + _SOURCE_PUSH_ROWS_PER_TILE - 1
    ) // _SOURCE_PUSH_ROWS_PER_TILE
    signal_bytes = _align_bytes(max_tiles * torch.int32.itemsize)
    buffer_bytes = source_payload_bytes + signal_bytes
    if spec.output_mode is OutputMode.REPLICATED:
        gather_dtype = (
            torch.float32
            if spec.internal_precision is NormInternalPrecision.FULL_FP32
            else torch.bfloat16
        )
        gather_payload_bytes = _align_bytes(
            owner_capacity * spec.hidden_size * gather_dtype.itemsize
        )
        buffer_bytes += gather_payload_bytes + signal_bytes
    return buffer_bytes


def required_owner_pull_buffer_bytes(
    capacity: int,
    spec: AttnTPNormSpec,
) -> int:
    source_push_output_capacity(capacity, spec)
    return _align_bytes(capacity * spec.hidden_size * torch.bfloat16.itemsize)


def decode_output_capacity(rows: int, spec: AttnTPNormSpec) -> int:
    if type(rows) is not int or rows <= 0 or rows > MAX_DECODE_ROWS:
        raise ValueError(f"Decode rows must be in [1, {MAX_DECODE_ROWS}], got {rows!r}")
    if spec.output_mode is OutputMode.TOKEN_SCATTERED:
        return (rows + spec.attn_tp_size - 1) // spec.attn_tp_size
    return rows


def required_decode_source_push_buffer_bytes(
    max_rows: int,
    spec: AttnTPNormSpec,
) -> int:
    decode_output_capacity(max_rows, spec)
    if spec.attn_tp_size == 1:
        return 0
    slot_rows = (
        (max_rows + spec.attn_tp_size - 1) // spec.attn_tp_size
        if spec.output_mode is OutputMode.TOKEN_SCATTERED
        else max_rows
    )
    payload_bytes = _align_bytes(slot_rows * spec.hidden_size * torch.bfloat16.itemsize)
    signal_bytes = _align_bytes(slot_rows * torch.int32.itemsize)
    return payload_bytes + signal_bytes


def required_decode_owner_pull_buffer_bytes(
    max_rows: int,
    spec: AttnTPNormSpec,
) -> int:
    decode_output_capacity(max_rows, spec)
    if spec.attn_tp_size == 1:
        return 0
    return _align_bytes(max_rows * spec.hidden_size * torch.bfloat16.itemsize)


@cache_once
def _jit_prefill_attntp_fused_ipc_norm_module(
    rows_per_signal: int,
    cache_mode: str,
    min_blocks_per_sm: int,
    internal_precision: NormInternalPrecision,
) -> Module:
    if internal_precision is not NormInternalPrecision.REFERENCE_BF16:
        raise NotImplementedError(
            "legacy Prefill AttnTP2 IPC norm supports only reference_bf16; "
            "use PrefillAttnTPFusedIPCNormRunner for full_fp32"
        )
    internal_precision_value = _NORM_INTERNAL_PRECISION_CPP_VALUE[internal_precision]
    args = make_cpp_args(
        torch.bfloat16,
        _WORLD_SIZE,
        _HIDDEN_SIZE,
        rows_per_signal,
        cache_mode == "peer",
        min_blocks_per_sm,
        False,
        internal_precision_value,
    )
    class_name = f"FusedPrefillAttnTPIPCNorm<{args}>"
    return load_jit(
        "prefill_attntp_fused_ipc_norm",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/attntp_fused_norm/prefill_attntp_fused_ipc_norm.cuh"],
        cuda_wrappers=[
            ("fused_prefill_attntp_norm", f"{class_name}::run"),
            ("get_max_occupancy", f"{class_name}::get_max_occupancy"),
        ],
    )


@cache_once
def _jit_local_attntp_fused_norm_module(
    hidden_size: int,
    internal_precision: NormInternalPrecision,
) -> Module:
    internal_precision_value = _NORM_INTERNAL_PRECISION_CPP_VALUE[internal_precision]
    args = make_cpp_args(
        torch.bfloat16,
        hidden_size,
        internal_precision_value,
    )
    class_name = f"FusedLocalAttnTPNorm<{args}>"
    return load_jit(
        "local_attntp_fused_norm",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/attntp_fused_norm/prefill_attntp_fused_ipc_norm.cuh"],
        cuda_wrappers=[
            ("fused_local_attntp_norm", f"{class_name}::run"),
            ("get_max_occupancy", f"{class_name}::get_max_occupancy"),
        ],
    )


@cache_once
def _jit_prefill_attntp_source_push_norm_module(
    attn_tp_size: int,
    hidden_size: int,
    output_mode: OutputMode,
    internal_precision: NormInternalPrecision,
    rows_per_tile: int,
    block_size: int,
    signal_backoff: int,
) -> Module:
    output_mode_value = _OUTPUT_MODE_CPP_VALUE[output_mode]
    internal_precision_value = _NORM_INTERNAL_PRECISION_CPP_VALUE[internal_precision]
    args = make_cpp_args(
        torch.bfloat16,
        attn_tp_size,
        hidden_size,
        output_mode_value,
        rows_per_tile,
        internal_precision_value,
        block_size,
        signal_backoff,
    )
    class_name = f"FusedPrefillAttnTPSourcePushNorm<{args}>"
    return load_jit(
        "prefill_attntp_source_push_norm",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/attntp_fused_norm/prefill_attntp_fused_ipc_norm.cuh"],
        cuda_wrappers=[
            ("fused_source_push_norm", f"{class_name}::run"),
            ("get_max_occupancy", f"{class_name}::get_max_occupancy"),
        ],
    )


@cache_once
def _jit_prefill_attntp_owner_pull_norm_module(
    attn_tp_size: int,
    hidden_size: int,
    output_mode: OutputMode,
    internal_precision: NormInternalPrecision,
    block_size: int,
) -> Module:
    output_mode_value = _OUTPUT_MODE_CPP_VALUE[output_mode]
    internal_precision_value = _NORM_INTERNAL_PRECISION_CPP_VALUE[internal_precision]
    args = make_cpp_args(
        torch.bfloat16,
        attn_tp_size,
        hidden_size,
        output_mode_value,
        internal_precision_value,
        block_size,
    )
    class_name = f"FusedPrefillAttnTPOwnerPullNorm<{args}>"
    return load_jit(
        "prefill_attntp_owner_pull_norm",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/attntp_fused_norm/prefill_attntp_fused_ipc_norm.cuh"],
        cuda_wrappers=[
            ("fused_owner_pull_norm", f"{class_name}::run"),
            ("get_max_occupancy", f"{class_name}::get_max_occupancy"),
        ],
    )


@cache_once
def _jit_decode_attntp_source_push_norm_module(
    attn_tp_size: int,
    hidden_size: int,
    output_mode: OutputMode,
    internal_precision: NormInternalPrecision,
    block_size: int,
    signal_backoff: int,
) -> Module:
    output_mode_value = _OUTPUT_MODE_CPP_VALUE[output_mode]
    internal_precision_value = _NORM_INTERNAL_PRECISION_CPP_VALUE[internal_precision]
    args = make_cpp_args(
        torch.bfloat16,
        attn_tp_size,
        hidden_size,
        output_mode_value,
        internal_precision_value,
        block_size,
        signal_backoff,
    )
    class_name = f"FusedDecodeAttnTPSourcePushNorm<{args}>"
    return load_jit(
        "decode_attntp_source_push_norm",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/attntp_fused_norm/decode_attntp_fused_ipc_norm.cuh"],
        cuda_wrappers=[
            ("fused_source_push_norm", f"{class_name}::run"),
            ("get_max_occupancy", f"{class_name}::get_max_occupancy"),
        ],
    )


@cache_once
def _jit_decode_attntp_owner_pull_norm_module(
    attn_tp_size: int,
    hidden_size: int,
    output_mode: OutputMode,
    internal_precision: NormInternalPrecision,
    block_size: int,
) -> Module:
    output_mode_value = _OUTPUT_MODE_CPP_VALUE[output_mode]
    internal_precision_value = _NORM_INTERNAL_PRECISION_CPP_VALUE[internal_precision]
    args = make_cpp_args(
        torch.bfloat16,
        attn_tp_size,
        hidden_size,
        output_mode_value,
        internal_precision_value,
        block_size,
    )
    class_name = f"FusedDecodeAttnTPOwnerPullNorm<{args}>"
    return load_jit(
        "decode_attntp_owner_pull_norm",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/attntp_fused_norm/decode_attntp_fused_ipc_norm.cuh"],
        cuda_wrappers=[
            ("fused_owner_pull_norm", f"{class_name}::run"),
            ("get_max_occupancy", f"{class_name}::get_max_occupancy"),
        ],
    )


@cache_once
def _jit_decode_attntp_local_norm_module(
    hidden_size: int,
    output_mode: OutputMode,
    internal_precision: NormInternalPrecision,
) -> Module:
    output_mode_value = _OUTPUT_MODE_CPP_VALUE[output_mode]
    internal_precision_value = _NORM_INTERNAL_PRECISION_CPP_VALUE[internal_precision]
    args = make_cpp_args(
        torch.bfloat16,
        hidden_size,
        output_mode_value,
        internal_precision_value,
    )
    class_name = f"FusedDecodeAttnTPLocalNorm<{args}>"
    return load_jit(
        "decode_attntp_local_norm",
        *args,
        extra_ldflags=["-lcuda"],
        cuda_files=["distributed/attntp_fused_norm/decode_attntp_fused_ipc_norm.cuh"],
        cuda_wrappers=[
            ("fused_local_norm", f"{class_name}::run"),
            ("get_max_occupancy", f"{class_name}::get_max_occupancy"),
        ],
    )


def _validate_rows_per_signal(rows_per_signal: int) -> None:
    if rows_per_signal not in _SUPPORTED_ROWS_PER_SIGNAL:
        raise ValueError(f"rows_per_signal must be one of {_SUPPORTED_ROWS_PER_SIGNAL}")


def _validate_cache_mode(cache_mode: str) -> None:
    if cache_mode not in _SUPPORTED_CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {_SUPPORTED_CACHE_MODES}")


def _validate_min_blocks_per_sm(
    min_blocks_per_sm: int,
    rows_per_signal: int,
    cache_mode: str,
) -> None:
    if min_blocks_per_sm not in _SUPPORTED_MIN_BLOCKS_PER_SM:
        raise ValueError(
            f"min_blocks_per_sm must be one of {_SUPPORTED_MIN_BLOCKS_PER_SM}"
        )
    if min_blocks_per_sm != 4 and (rows_per_signal != 8 or cache_mode != "local"):
        raise ValueError(
            "high-occupancy specialization requires rows_per_signal=8 "
            "and cache_mode='local'"
        )


def get_fused_local_attntp_max_occupancy(
    hidden_size: int,
    *,
    internal_precision: NormInternalPrecision = (NormInternalPrecision.REFERENCE_BF16),
) -> int:
    hidden_size = int(hidden_size)
    if hidden_size not in _SUPPORTED_HIDDEN_SIZES:
        raise ValueError(
            f"hidden size must be one of {_SUPPORTED_HIDDEN_SIZES}, got {hidden_size!r}"
        )
    module = _jit_local_attntp_fused_norm_module(
        hidden_size,
        internal_precision,
    )
    return int(module.get_max_occupancy())


def _validate_fused_inputs(
    partial: torch.Tensor,
    residual: torch.Tensor,
    o_norm_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
) -> None:
    if partial.device.type != "cuda":
        raise ValueError("partial must be a CUDA tensor")
    if partial.dtype != torch.bfloat16:
        raise ValueError("partial must use BF16")
    if partial.ndim != 2 or partial.shape[1] not in _SUPPORTED_HIDDEN_SIZES:
        raise ValueError(
            "partial must have shape [tokens, hidden_size] with hidden_size "
            f"in {_SUPPORTED_HIDDEN_SIZES}"
        )
    if not partial.is_contiguous():
        raise ValueError("partial must be contiguous")
    if residual.shape != partial.shape or residual.dtype != torch.float32:
        raise ValueError("residual must be FP32 and have the same shape as partial")
    if residual.device != partial.device or not residual.is_contiguous():
        raise ValueError("residual must be contiguous and on the partial device")
    for name, weight in (
        ("o_norm_weight", o_norm_weight),
        ("post_norm_weight", post_norm_weight),
    ):
        if (
            weight.shape != (partial.shape[1],)
            or weight.dtype != torch.bfloat16
            or weight.device != partial.device
            or not weight.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous BF16 [{partial.shape[1]}] on "
                "the partial device"
            )


def _allocate_outputs(
    partial: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.empty_like(partial)
    residual_out = torch.empty_like(residual)
    return output, residual_out


def _validate_fused_outputs(
    partial: torch.Tensor,
    output: torch.Tensor,
    residual_out: torch.Tensor,
) -> None:
    if output.shape != partial.shape:
        raise ValueError(
            "output must have the same shape as partial, got "
            f"{tuple(output.shape)} and {tuple(partial.shape)}"
        )
    if output.dtype != torch.bfloat16:
        raise ValueError("output must be BF16")
    if output.device != partial.device or not output.is_contiguous():
        raise ValueError("output must be contiguous and on the partial device")
    if residual_out.shape != partial.shape:
        raise ValueError(
            "residual_out must have shape "
            f"{tuple(partial.shape)}, got {tuple(residual_out.shape)}"
        )
    if residual_out.dtype != torch.float32:
        raise ValueError("residual_out must be FP32")
    if residual_out.device != partial.device or not residual_out.is_contiguous():
        raise ValueError("residual_out must be contiguous and on the partial device")


def _validate_specialization_options(
    rows_per_signal: int | None = None,
    cache_mode: str | None = None,
    min_blocks_per_sm: int | None = None,
) -> tuple[int | None, str | None, int | None]:
    if rows_per_signal is not None:
        rows_per_signal = int(rows_per_signal)
        _validate_rows_per_signal(rows_per_signal)
    if cache_mode is not None:
        _validate_cache_mode(cache_mode)
    if (rows_per_signal is None) != (cache_mode is None):
        raise ValueError(
            "rows_per_signal and cache_mode must both be explicit or both "
            "use automatic specialization"
        )
    if min_blocks_per_sm is not None and rows_per_signal is None:
        raise ValueError("min_blocks_per_sm cannot override automatic specialization")
    return rows_per_signal, cache_mode, min_blocks_per_sm


def _launch_fused_prefill_attntp_norm(
    custom_ar,
    partial: torch.Tensor,
    residual: torch.Tensor,
    o_norm_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    output: torch.Tensor,
    residual_out: torch.Tensor,
    o_norm_eps: float,
    post_norm_eps: float,
    rows_per_signal: int | None,
    cache_mode: str | None,
    min_blocks_per_sm: int | None,
    attn_tp_size: int,
    internal_precision: NormInternalPrecision,
) -> None:
    _validate_attn_tp_size(attn_tp_size)
    if not isinstance(internal_precision, NormInternalPrecision):
        raise ValueError("internal_precision must be a NormInternalPrecision")
    hidden_size = partial.shape[1]

    if attn_tp_size == 1:
        if custom_ar is not None:
            raise ValueError("AttnTP1 local norm requires custom_ar=None")
        if (
            rows_per_signal is not None
            or cache_mode is not None
            or min_blocks_per_sm is not None
        ):
            raise ValueError(
                "AttnTP1 local norm does not accept IPC specialization options"
            )
        if partial.shape[0] == 0:
            return
        module = _jit_local_attntp_fused_norm_module(
            hidden_size,
            internal_precision,
        )
        module.fused_local_attntp_norm(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            output,
            residual_out,
            float(o_norm_eps),
            float(post_norm_eps),
        )
        return

    if attn_tp_size != 2:
        raise NotImplementedError(
            f"Prefill AttnTP{attn_tp_size} communication is not implemented"
        )
    if hidden_size != _HIDDEN_SIZE:
        raise NotImplementedError(
            "Prefill AttnTP2 communication currently supports hidden size "
            f"{_HIDDEN_SIZE}, got {hidden_size}"
        )
    if custom_ar is None:
        raise ValueError("Prefill AttnTP2 norm requires a CustomAllReduce object")
    if partial.shape[0] == 0:
        return

    if rows_per_signal is None:
        cache_mode, rows_per_signal, min_blocks_per_sm = (
            _select_fused_prefill_attntp_specialization(partial.shape[0])
        )
    else:
        min_blocks_per_sm = 4 if min_blocks_per_sm is None else int(min_blocks_per_sm)
        _validate_min_blocks_per_sm(
            min_blocks_per_sm,
            rows_per_signal,
            cache_mode,
        )
    module = _jit_prefill_attntp_fused_ipc_norm_module(
        rows_per_signal,
        cache_mode,
        min_blocks_per_sm,
        internal_precision,
    )
    module.fused_prefill_attntp_norm(
        custom_ar,
        partial,
        residual,
        o_norm_weight,
        post_norm_weight,
        output,
        residual_out,
        float(o_norm_eps),
        float(post_norm_eps),
    )


def fused_prefill_attntp_norm_out(
    custom_ar,
    partial: torch.Tensor,
    residual: torch.Tensor,
    o_norm_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    output: torch.Tensor,
    residual_out: torch.Tensor,
    o_norm_eps: float,
    post_norm_eps: float,
    *,
    rows_per_signal: int | None = None,
    cache_mode: str | None = None,
    min_blocks_per_sm: int | None = None,
    attn_tp_size: int = _WORLD_SIZE,
    internal_precision: NormInternalPrecision = (NormInternalPrecision.REFERENCE_BF16),
) -> None:
    """Launch into fixed caller-owned outputs after eager JIT warmup."""
    rows_per_signal, cache_mode, min_blocks_per_sm = _validate_specialization_options(
        rows_per_signal,
        cache_mode,
        min_blocks_per_sm,
    )
    _validate_fused_inputs(
        partial,
        residual,
        o_norm_weight,
        post_norm_weight,
    )
    _validate_fused_outputs(
        partial,
        output,
        residual_out,
    )
    _launch_fused_prefill_attntp_norm(
        custom_ar,
        partial,
        residual,
        o_norm_weight,
        post_norm_weight,
        output,
        residual_out,
        o_norm_eps,
        post_norm_eps,
        rows_per_signal,
        cache_mode,
        min_blocks_per_sm,
        attn_tp_size,
        internal_precision,
    )


def fused_prefill_attntp_norm(
    custom_ar,
    partial: torch.Tensor,
    residual: torch.Tensor,
    o_norm_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    o_norm_eps: float,
    post_norm_eps: float,
    *,
    rows_per_signal: int | None = None,
    cache_mode: str | None = None,
    min_blocks_per_sm: int | None = None,
    attn_tp_size: int = _WORLD_SIZE,
    internal_precision: NormInternalPrecision = (NormInternalPrecision.REFERENCE_BF16),
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_per_signal, cache_mode, min_blocks_per_sm = _validate_specialization_options(
        rows_per_signal,
        cache_mode,
        min_blocks_per_sm,
    )
    _validate_fused_inputs(
        partial,
        residual,
        o_norm_weight,
        post_norm_weight,
    )
    output, residual_out = _allocate_outputs(partial, residual)
    _launch_fused_prefill_attntp_norm(
        custom_ar,
        partial,
        residual,
        o_norm_weight,
        post_norm_weight,
        output,
        residual_out,
        o_norm_eps,
        post_norm_eps,
        rows_per_signal,
        cache_mode,
        min_blocks_per_sm,
        attn_tp_size,
        internal_precision,
    )
    return output, residual_out


def _synchronize_ipc_communicator_initialization(*, group, device) -> None:
    # A peer must not publish into storage that another rank can still clear.
    torch.cuda.synchronize(device)
    dist.barrier(group=group)


class PrefillAttnTPFusedIPCNormRunner:
    """Benchmark-only fixed-capacity varlen Prefill IPC runner."""

    def __init__(
        self,
        *,
        group,
        device: torch.device,
        spec: AttnTPNormSpec,
        capacity: int,
        algorithm: PrefillCommunicationAlgorithm,
        block_size: int = 256,
        signal_backoff: int = 64,
        rows_per_tile: int = _SOURCE_PUSH_ROWS_PER_TILE,
        blocks_per_sm: int | None = None,
    ) -> None:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        self.spec = spec
        if not isinstance(algorithm, PrefillCommunicationAlgorithm):
            raise ValueError("algorithm must be a PrefillCommunicationAlgorithm")
        _validate_generalized_kernel_tuning(
            hidden_size=spec.hidden_size,
            block_size=block_size,
            signal_backoff=(
                signal_backoff
                if algorithm is PrefillCommunicationAlgorithm.SOURCE_PUSH
                else None
            ),
            rows_per_tile=(
                rows_per_tile
                if algorithm is PrefillCommunicationAlgorithm.SOURCE_PUSH
                else None
            ),
            blocks_per_sm=blocks_per_sm,
        )
        self.algorithm = algorithm
        self.block_size = block_size
        self.signal_backoff = signal_backoff
        self.rows_per_tile = rows_per_tile
        self.blocks_per_sm = blocks_per_sm
        self.capacity = capacity
        self.arena_capacity = capacity
        self.output_capacity = source_push_output_capacity(capacity, spec)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Prefill IPC runner requires a CUDA device")
        world_size = dist.get_world_size(group=group)
        if world_size != spec.attn_tp_size:
            raise ValueError(
                "process-group size must match AttnTP size, got "
                f"{world_size} and {spec.attn_tp_size}"
            )
        self.rank = dist.get_rank(group=group)
        if algorithm is PrefillCommunicationAlgorithm.SOURCE_PUSH:
            max_pull_size = 0
            max_push_size = required_source_push_buffer_bytes(capacity, spec)
        else:
            max_pull_size = required_owner_pull_buffer_bytes(capacity, spec)
            max_push_size = 0
        self.communicator = CustomAllReduceV2(
            group,
            self.device,
            max_pull_size=max_pull_size,
            max_push_size=max_push_size,
        )
        if self.communicator.disabled:
            raise RuntimeError(
                "Prefill IPC runner requires enabled CUDA IPC CustomAllReduceV2"
            )
        _synchronize_ipc_communicator_initialization(
            group=group,
            device=self.device,
        )
        self._resource = AttnTPIPCResource.from_communicator(
            communicator=self.communicator,
            device=self.device,
            attn_tp_size=spec.attn_tp_size,
            max_rows=capacity,
            max_pull_size=max_pull_size,
            max_push_size=max_push_size,
            source_push_capacity=(
                capacity
                if algorithm is PrefillCommunicationAlgorithm.SOURCE_PUSH
                else 0
            ),
        )
        self._owns_resource = True
        self._closed = False

    @classmethod
    def from_resource(
        cls,
        *,
        resource,
        spec: AttnTPNormSpec,
        capacity: int,
        algorithm: PrefillCommunicationAlgorithm,
        block_size: int = 256,
        signal_backoff: int = 64,
        rows_per_tile: int = _SOURCE_PUSH_ROWS_PER_TILE,
        blocks_per_sm: int | None = None,
    ) -> PrefillAttnTPFusedIPCNormRunner:
        if not isinstance(algorithm, PrefillCommunicationAlgorithm):
            raise ValueError("algorithm must be a PrefillCommunicationAlgorithm")
        _validate_generalized_kernel_tuning(
            hidden_size=spec.hidden_size,
            block_size=block_size,
            signal_backoff=(
                signal_backoff
                if algorithm is PrefillCommunicationAlgorithm.SOURCE_PUSH
                else None
            ),
            rows_per_tile=(
                rows_per_tile
                if algorithm is PrefillCommunicationAlgorithm.SOURCE_PUSH
                else None
            ),
            blocks_per_sm=blocks_per_sm,
        )
        output_capacity = source_push_output_capacity(capacity, spec)
        if algorithm is PrefillCommunicationAlgorithm.SOURCE_PUSH:
            arena_capacity = resource.source_push_capacity
            if arena_capacity < capacity:
                raise ValueError(
                    "source-push arena capacity must cover runner capacity"
                )
            min_pull_size = 0
            min_push_size = required_source_push_buffer_bytes(capacity, spec)
        else:
            arena_capacity = capacity
            min_pull_size = required_owner_pull_buffer_bytes(capacity, spec)
            min_push_size = 0
        resource.require(
            attn_tp_size=spec.attn_tp_size,
            min_rows=capacity,
            min_pull_size=min_pull_size,
            min_push_size=min_push_size,
        )
        device = torch.device(resource.device)
        if device.type != "cuda":
            raise ValueError("Prefill IPC runner requires a CUDA device")

        runner = cls.__new__(cls)
        runner.spec = spec
        runner.algorithm = algorithm
        runner.block_size = block_size
        runner.signal_backoff = signal_backoff
        runner.rows_per_tile = rows_per_tile
        runner.blocks_per_sm = blocks_per_sm
        runner.capacity = capacity
        runner.arena_capacity = arena_capacity
        runner.output_capacity = output_capacity
        runner.device = device
        runner.rank = resource.rank
        runner.communicator = resource.communicator
        runner._resource = resource
        runner._owns_resource = False
        runner._closed = False
        return runner

    def output_capacity_for(self, capacity: int) -> int:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if capacity > self.capacity:
            raise ValueError(
                f"capacity {capacity} exceeds runner maximum {self.capacity}"
            )
        return source_push_output_capacity(capacity, self.spec)

    def run_out(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        output: torch.Tensor,
        residual_out: torch.Tensor,
        actual_rows: torch.Tensor,
        owner_start: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
    ) -> None:
        if self._closed:
            raise RuntimeError("Prefill IPC runner is closed")
        _validate_fused_inputs(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
        )
        input_capacity = partial.shape[0]
        output_capacity = self.output_capacity_for(input_capacity)
        expected_input_shape = (input_capacity, self.spec.hidden_size)
        if partial.shape != expected_input_shape:
            raise ValueError(f"partial must have shape {expected_input_shape}")
        expected_output_shape = (
            output_capacity,
            self.spec.hidden_size,
        )
        if (
            output.shape != expected_output_shape
            or output.dtype != torch.bfloat16
            or output.device != self.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "output must be contiguous BF16 with shape "
                f"{expected_output_shape} on {self.device}"
            )
        if (
            residual_out.shape != expected_output_shape
            or residual_out.dtype != torch.float32
            or residual_out.device != self.device
            or not residual_out.is_contiguous()
        ):
            raise ValueError(
                "residual_out must be contiguous FP32 with shape "
                f"{expected_output_shape} on {self.device}"
            )
        for name, scalar in (
            ("actual_rows", actual_rows),
            ("owner_start", owner_start),
        ):
            if (
                scalar.shape != (1,)
                or scalar.dtype != torch.int32
                or scalar.device != self.device
                or not scalar.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be a contiguous CUDA int32 tensor with shape [1]"
                )

        if self.algorithm is PrefillCommunicationAlgorithm.SOURCE_PUSH:
            module = _jit_prefill_attntp_source_push_norm_module(
                self.spec.attn_tp_size,
                self.spec.hidden_size,
                self.spec.output_mode,
                self.spec.internal_precision,
                self.rows_per_tile,
                self.block_size,
                self.signal_backoff,
            )
            module.fused_source_push_norm(
                self.communicator.obj,
                partial,
                residual,
                o_norm_weight,
                post_norm_weight,
                output,
                residual_out,
                actual_rows,
                owner_start,
                int(self.arena_capacity),
                0 if self.blocks_per_sm is None else self.blocks_per_sm,
                float(o_norm_eps),
                float(post_norm_eps),
            )
        else:
            module = _jit_prefill_attntp_owner_pull_norm_module(
                self.spec.attn_tp_size,
                self.spec.hidden_size,
                self.spec.output_mode,
                self.spec.internal_precision,
                self.block_size,
            )
            module.fused_owner_pull_norm(
                self.communicator.obj,
                partial,
                residual,
                o_norm_weight,
                post_norm_weight,
                output,
                residual_out,
                actual_rows,
                owner_start,
                0 if self.blocks_per_sm is None else self.blocks_per_sm,
                float(o_norm_eps),
                float(post_norm_eps),
            )

    def get_max_occupancy(self) -> int:
        if self.algorithm is PrefillCommunicationAlgorithm.SOURCE_PUSH:
            module = _jit_prefill_attntp_source_push_norm_module(
                self.spec.attn_tp_size,
                self.spec.hidden_size,
                self.spec.output_mode,
                self.spec.internal_precision,
                self.rows_per_tile,
                self.block_size,
                self.signal_backoff,
            )
        else:
            module = _jit_prefill_attntp_owner_pull_norm_module(
                self.spec.attn_tp_size,
                self.spec.hidden_size,
                self.spec.output_mode,
                self.spec.internal_precision,
                self.block_size,
            )
        return int(module.get_max_occupancy())

    def capture(self):
        if self._closed:
            raise RuntimeError("Prefill IPC runner is closed")
        return self._resource.capture()

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_resource:
            self._resource.close()
        self._closed = True


class DecodeAttnTPFusedIPCNormRunner:
    """Benchmark-only fixed-row Decode IPC runner."""

    def __init__(
        self,
        *,
        group,
        device: torch.device,
        spec: AttnTPNormSpec,
        max_rows: int,
        algorithm: DecodeCommunicationAlgorithm,
        block_size: int = 256,
        signal_backoff: int = 64,
        blocks_per_sm: int | None = None,
    ) -> None:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        if not isinstance(spec, AttnTPNormSpec):
            raise ValueError("spec must be an AttnTPNormSpec")
        if not isinstance(algorithm, DecodeCommunicationAlgorithm):
            raise ValueError("algorithm must be a DecodeCommunicationAlgorithm")
        _validate_generalized_kernel_tuning(
            hidden_size=spec.hidden_size,
            block_size=block_size,
            signal_backoff=(
                signal_backoff
                if algorithm is DecodeCommunicationAlgorithm.SOURCE_PUSH
                else None
            ),
            rows_per_tile=None,
            blocks_per_sm=blocks_per_sm,
        )
        decode_output_capacity(max_rows, spec)
        self.spec = spec
        self.algorithm = algorithm
        self.block_size = block_size
        self.signal_backoff = signal_backoff
        self.blocks_per_sm = blocks_per_sm
        self.max_rows = max_rows
        self.arena_rows = max_rows
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Decode IPC runner requires a CUDA device")
        world_size = dist.get_world_size(group=group)
        if world_size != spec.attn_tp_size:
            raise ValueError(
                "process-group size must match AttnTP size, got "
                f"{world_size} and {spec.attn_tp_size}"
            )
        self.rank = dist.get_rank(group=group)
        self.communicator = None
        self._resource = None
        self._owns_resource = False
        if spec.attn_tp_size > 1:
            if algorithm is DecodeCommunicationAlgorithm.SOURCE_PUSH:
                max_pull_size = 0
                max_push_size = required_decode_source_push_buffer_bytes(max_rows, spec)
            else:
                max_pull_size = required_decode_owner_pull_buffer_bytes(max_rows, spec)
                max_push_size = 0
            self.communicator = CustomAllReduceV2(
                group,
                self.device,
                max_pull_size=max_pull_size,
                max_push_size=max_push_size,
            )
            if self.communicator.disabled:
                raise RuntimeError(
                    "Decode IPC runner requires enabled CUDA IPC CustomAllReduceV2"
                )
            _synchronize_ipc_communicator_initialization(
                group=group,
                device=self.device,
            )
            self._resource = AttnTPIPCResource.from_communicator(
                communicator=self.communicator,
                device=self.device,
                attn_tp_size=spec.attn_tp_size,
                max_rows=max_rows,
                max_pull_size=max_pull_size,
                max_push_size=max_push_size,
                source_push_capacity=(
                    max_rows
                    if algorithm is DecodeCommunicationAlgorithm.SOURCE_PUSH
                    else 0
                ),
            )
            self._owns_resource = True
        self._closed = False

    @classmethod
    def from_resource(
        cls,
        *,
        resource,
        spec: AttnTPNormSpec,
        max_rows: int,
        algorithm: DecodeCommunicationAlgorithm,
        block_size: int = 256,
        signal_backoff: int = 64,
        blocks_per_sm: int | None = None,
    ) -> DecodeAttnTPFusedIPCNormRunner:
        if not isinstance(spec, AttnTPNormSpec):
            raise ValueError("spec must be an AttnTPNormSpec")
        if not isinstance(algorithm, DecodeCommunicationAlgorithm):
            raise ValueError("algorithm must be a DecodeCommunicationAlgorithm")
        _validate_generalized_kernel_tuning(
            hidden_size=spec.hidden_size,
            block_size=block_size,
            signal_backoff=(
                signal_backoff
                if algorithm is DecodeCommunicationAlgorithm.SOURCE_PUSH
                else None
            ),
            rows_per_tile=None,
            blocks_per_sm=blocks_per_sm,
        )
        decode_output_capacity(max_rows, spec)
        arena_rows = max_rows
        if spec.attn_tp_size == 1:
            if resource is not None:
                raise ValueError("AttnTP1 Decode must not receive an IPC resource")
            device = torch.device("cuda")
            rank = 0
            communicator = None
        else:
            if algorithm is DecodeCommunicationAlgorithm.SOURCE_PUSH:
                arena_rows = resource.source_push_capacity
                if arena_rows < max_rows:
                    raise ValueError(
                        "source-push arena rows must cover runner rows"
                    )
                min_pull_size = 0
                min_push_size = required_decode_source_push_buffer_bytes(max_rows, spec)
            else:
                arena_rows = max_rows
                min_pull_size = required_decode_owner_pull_buffer_bytes(max_rows, spec)
                min_push_size = 0
            resource.require(
                attn_tp_size=spec.attn_tp_size,
                min_rows=max_rows,
                min_pull_size=min_pull_size,
                min_push_size=min_push_size,
            )
            device = torch.device(resource.device)
            rank = resource.rank
            communicator = resource.communicator
        if device.type != "cuda":
            raise ValueError("Decode IPC runner requires a CUDA device")

        runner = cls.__new__(cls)
        runner.spec = spec
        runner.algorithm = algorithm
        runner.block_size = block_size
        runner.signal_backoff = signal_backoff
        runner.blocks_per_sm = blocks_per_sm
        runner.max_rows = max_rows
        runner.arena_rows = arena_rows
        runner.device = device
        runner.rank = rank
        runner.communicator = communicator
        runner._resource = resource
        runner._owns_resource = False
        runner._closed = False
        return runner

    def output_capacity_for(self, rows: int) -> int:
        if type(rows) is not int or rows <= 0 or rows > self.max_rows:
            raise ValueError(
                f"Decode rows must be in [1, {self.max_rows}], got {rows!r}"
            )
        return decode_output_capacity(rows, self.spec)

    def run_out(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        output: torch.Tensor,
        residual_out: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
    ) -> None:
        if self._closed:
            raise RuntimeError("Decode IPC runner is closed")
        _validate_fused_inputs(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
        )
        rows = partial.shape[0]
        output_rows = self.output_capacity_for(rows)
        if partial.shape != (rows, self.spec.hidden_size):
            raise ValueError(f"partial must have shape {(rows, self.spec.hidden_size)}")
        expected_output_shape = (output_rows, self.spec.hidden_size)
        if (
            output.shape != expected_output_shape
            or output.dtype != torch.bfloat16
            or output.device != self.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "output must be contiguous BF16 with shape "
                f"{expected_output_shape} on {self.device}"
            )
        if (
            residual_out.shape != expected_output_shape
            or residual_out.dtype != torch.float32
            or residual_out.device != self.device
            or not residual_out.is_contiguous()
        ):
            raise ValueError(
                "residual_out must be contiguous FP32 with shape "
                f"{expected_output_shape} on {self.device}"
            )
        if self.spec.attn_tp_size == 1:
            module = _jit_decode_attntp_local_norm_module(
                self.spec.hidden_size,
                self.spec.output_mode,
                self.spec.internal_precision,
            )
            module.fused_local_norm(
                partial,
                residual,
                o_norm_weight,
                post_norm_weight,
                output,
                residual_out,
                float(o_norm_eps),
                float(post_norm_eps),
            )
        elif self.algorithm is DecodeCommunicationAlgorithm.SOURCE_PUSH:
            module = _jit_decode_attntp_source_push_norm_module(
                self.spec.attn_tp_size,
                self.spec.hidden_size,
                self.spec.output_mode,
                self.spec.internal_precision,
                self.block_size,
                self.signal_backoff,
            )
            module.fused_source_push_norm(
                self.communicator.obj,
                partial,
                residual,
                o_norm_weight,
                post_norm_weight,
                output,
                residual_out,
                int(self.arena_rows),
                0 if self.blocks_per_sm is None else self.blocks_per_sm,
                float(o_norm_eps),
                float(post_norm_eps),
            )
        else:
            module = _jit_decode_attntp_owner_pull_norm_module(
                self.spec.attn_tp_size,
                self.spec.hidden_size,
                self.spec.output_mode,
                self.spec.internal_precision,
                self.block_size,
            )
            module.fused_owner_pull_norm(
                self.communicator.obj,
                partial,
                residual,
                o_norm_weight,
                post_norm_weight,
                output,
                residual_out,
                0 if self.blocks_per_sm is None else self.blocks_per_sm,
                float(o_norm_eps),
                float(post_norm_eps),
            )

    def capture(self):
        if self._closed:
            raise RuntimeError("Decode IPC runner is closed")
        if self._resource is None:
            return nullcontext()
        return self._resource.capture()

    def get_max_occupancy(self) -> int:
        if self.spec.attn_tp_size == 1:
            module = _jit_decode_attntp_local_norm_module(
                self.spec.hidden_size,
                self.spec.output_mode,
                self.spec.internal_precision,
            )
        else:
            if self.algorithm is DecodeCommunicationAlgorithm.SOURCE_PUSH:
                module = _jit_decode_attntp_source_push_norm_module(
                    self.spec.attn_tp_size,
                    self.spec.hidden_size,
                    self.spec.output_mode,
                    self.spec.internal_precision,
                    self.block_size,
                    self.signal_backoff,
                )
            else:
                module = _jit_decode_attntp_owner_pull_norm_module(
                    self.spec.attn_tp_size,
                    self.spec.hidden_size,
                    self.spec.output_mode,
                    self.spec.internal_precision,
                    self.block_size,
                )
        return int(module.get_max_occupancy())

    def close(self) -> None:
        if self._closed:
            return
        if self._resource is not None and self._owns_resource:
            self._resource.close()
        self._closed = True
