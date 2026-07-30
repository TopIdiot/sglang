"""AttnTP fused SymmetricMemory and WeLM norm kernels."""

from __future__ import annotations

import torch
from tvm_ffi import Module

from sglang.jit_kernel.attntp_fused_norm.ipc import (
    AttnTPNormSpec,
    NormInternalPrecision,
    OutputMode,
    _NORM_INTERNAL_PRECISION_CPP_VALUE,
    _OUTPUT_MODE_CPP_VALUE,
    decode_output_capacity,
    source_push_output_capacity,
)
from sglang.jit_kernel.attntp_fused_norm.resources import (
    AttnTPSymmetricMemoryResource,
)
from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

_SUPPORTED_BLOCK_SIZES = (128, 256, 512)
_SUPPORTED_SIGNAL_BACKOFFS = (32, 64, 128, 256)
_MAX_TUNABLE_BLOCKS_PER_SM = 8


def _validate_true_symm_spec(spec: AttnTPNormSpec) -> None:
    if not isinstance(spec, AttnTPNormSpec):
        raise ValueError("spec must be an AttnTPNormSpec")
    if spec.attn_tp_size == 1:
        raise NotImplementedError(
            "true fused Symm requires distributed AttnTP; use the local fused "
            "kernel for AttnTP1"
        )


def _validate_kernel_tuning(
    *,
    hidden_size: int,
    block_size: int,
    signal_backoff: int,
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
    if (
        type(signal_backoff) is not int
        or signal_backoff not in _SUPPORTED_SIGNAL_BACKOFFS
    ):
        raise ValueError(f"signal_backoff must be one of {_SUPPORTED_SIGNAL_BACKOFFS}")
    if blocks_per_sm is not None and (
        type(blocks_per_sm) is not int or blocks_per_sm <= 0
    ):
        raise ValueError("blocks_per_sm must be a positive integer or None")


@cache_once
def _jit_prefill_attntp_fused_symm_norm_module(
    attn_tp_size: int,
    hidden_size: int,
    output_mode: OutputMode,
    internal_precision: NormInternalPrecision,
    block_size: int,
    signal_backoff: int,
) -> Module:
    args = make_cpp_args(
        torch.bfloat16,
        hidden_size,
        _OUTPUT_MODE_CPP_VALUE[output_mode],
        _NORM_INTERNAL_PRECISION_CPP_VALUE[internal_precision],
        attn_tp_size,
        block_size,
        signal_backoff,
    )
    class_name = f"FusedPrefillAttnTPSymmNorm<{args}>"
    cuda_wrappers = [
        ("fused_symm_norm", f"{class_name}::run"),
        ("get_max_occupancy", f"{class_name}::get_max_occupancy"),
    ]
    return load_jit(
        "prefill_attntp_fused_symm_norm",
        *args,
        cuda_files=["distributed/attntp_fused_norm/prefill_attntp_fused_symm_norm.cuh"],
        cuda_wrappers=cuda_wrappers,
    )


@cache_once
def _jit_decode_attntp_fused_symm_norm_module(
    attn_tp_size: int,
    hidden_size: int,
    output_mode: OutputMode,
    internal_precision: NormInternalPrecision,
    block_size: int,
    signal_backoff: int,
) -> Module:
    args = make_cpp_args(
        torch.bfloat16,
        hidden_size,
        _OUTPUT_MODE_CPP_VALUE[output_mode],
        _NORM_INTERNAL_PRECISION_CPP_VALUE[internal_precision],
        attn_tp_size,
        block_size,
        signal_backoff,
    )
    class_name = f"FusedDecodeAttnTPSymmNorm<{args}>"
    return load_jit(
        "decode_attntp_fused_symm_norm",
        *args,
        cuda_files=["distributed/attntp_fused_norm/decode_attntp_fused_symm_norm.cuh"],
        cuda_wrappers=[
            ("fused_symm_norm", f"{class_name}::run"),
            ("get_max_occupancy", f"{class_name}::get_max_occupancy"),
        ],
    )


def _direct_symm_resource_layout(
    *,
    device: torch.device,
    spec: AttnTPNormSpec,
    rows: int,
) -> tuple[int, int, int]:
    input_bytes = rows * spec.hidden_size * torch.bfloat16.itemsize
    if spec.attn_tp_size < 4:
        return input_bytes, 0, 0
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    sync_bytes = 2 * _MAX_TUNABLE_BLOCKS_PER_SM * sm_count * torch.uint32.itemsize
    row_bytes = spec.hidden_size * torch.bfloat16.itemsize
    sync_rows = (sync_bytes + row_bytes - 1) // row_bytes
    allocated_sync_bytes = sync_rows * row_bytes
    return (
        input_bytes + allocated_sync_bytes,
        input_bytes,
        allocated_sync_bytes // torch.uint32.itemsize,
    )


class _AttnTPFusedSymmNormRunnerBase:
    def _initialize_symm(
        self,
        *,
        group,
        device: torch.device,
        spec: AttnTPNormSpec,
        rows: int,
        block_size: int,
        signal_backoff: int,
        blocks_per_sm: int | None,
    ) -> None:
        device = torch.device(device)
        total_bytes, signal_offset, signal_slots = _direct_symm_resource_layout(
            device=device,
            spec=spec,
            rows=rows,
        )
        resource = AttnTPSymmetricMemoryResource(
            group=group,
            device=device,
            attn_tp_size=spec.attn_tp_size,
            hidden_size=spec.hidden_size,
            capacity=rows,
            total_bytes=total_bytes,
            direct_signal_offset_bytes=signal_offset,
            direct_signal_slots=signal_slots,
        )
        try:
            self._initialize_symm_from_resource(
                resource=resource,
                spec=spec,
                rows=rows,
                block_size=block_size,
                signal_backoff=signal_backoff,
                blocks_per_sm=blocks_per_sm,
                owns_resource=True,
            )
        except Exception:
            resource.close()
            raise

    def _initialize_symm_from_resource(
        self,
        *,
        resource,
        spec: AttnTPNormSpec,
        rows: int,
        block_size: int,
        signal_backoff: int,
        blocks_per_sm: int | None,
        owns_resource: bool,
    ) -> None:
        _validate_true_symm_spec(spec)
        _validate_kernel_tuning(
            hidden_size=spec.hidden_size,
            block_size=block_size,
            signal_backoff=signal_backoff,
            blocks_per_sm=blocks_per_sm,
        )
        if type(rows) is not int or rows <= 0:
            raise ValueError("Symm row capacity must be a positive integer")
        self.spec = spec
        self.block_size = block_size
        self.signal_backoff = signal_backoff
        self.blocks_per_sm = blocks_per_sm
        self.device = torch.device(resource.device)
        if self.device.type != "cuda":
            raise ValueError("true fused Symm runner requires a CUDA device")
        resource.require(
            attn_tp_size=spec.attn_tp_size,
            hidden_size=spec.hidden_size,
            capacity=rows,
            required_bytes=rows * spec.hidden_size * torch.bfloat16.itemsize,
        )
        resource.require_direct_symm(capacity=rows)
        self.rank = resource.rank
        self._resource = resource
        self._owns_resource = owns_resource
        self._storage = getattr(resource, "storage", None)
        self._input = resource.input_view_for(rows)
        self._handle = getattr(resource, "_handle", None)
        self._partial_pointer_table = resource.pointer_table
        self._multicast_pointer = resource.multicast_pointer
        self._multicast_signal_offset_bytes = resource.direct_signal_offset_bytes
        self._multicast_signal_slots = resource.direct_signal_slots
        self._peer_pointer = resource.peer_pointer
        self._signal_pad = resource.signal_pad
        self._peer_signal_pointer = resource.peer_signal_pointer
        self._closed = False

    @property
    def input_view(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("true fused Symm runner is closed")
        return self._input

    def capture(self):
        if self._closed:
            raise RuntimeError("true fused Symm runner is closed")
        return self._resource.capture()

    def close(self) -> None:
        if self._closed:
            return
        resource = getattr(self, "_resource", None)
        owns_resource = getattr(self, "_owns_resource", False)
        if resource is not None and owns_resource:
            resource.close()
        elif resource is None:
            torch.cuda.synchronize(self.device)
        self._closed = True
        self._resource = None
        self._handle = None
        self._input = None
        self._storage = None
        self._signal_pad = None
        self._peer_pointer = 0
        self._peer_signal_pointer = 0
        self._partial_pointer_table = 0
        self._multicast_pointer = 0
        self._multicast_signal_offset_bytes = 0
        self._multicast_signal_slots = 0


class PrefillAttnTPFusedSymmNormRunner(_AttnTPFusedSymmNormRunnerBase):
    """Fixed-capacity prefill runner whose producer writes a Symm input view."""

    def __init__(
        self,
        *,
        group,
        device: torch.device,
        spec: AttnTPNormSpec,
        capacity: int,
        block_size: int = 256,
        signal_backoff: int = 64,
        blocks_per_sm: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.output_capacity = source_push_output_capacity(capacity, spec)
        self._initialize_symm(
            group=group,
            device=device,
            spec=spec,
            rows=capacity,
            block_size=block_size,
            signal_backoff=signal_backoff,
            blocks_per_sm=blocks_per_sm,
        )

    @classmethod
    def from_resource(
        cls,
        *,
        resource,
        spec: AttnTPNormSpec,
        capacity: int,
        block_size: int = 256,
        signal_backoff: int = 64,
        blocks_per_sm: int | None = None,
    ) -> PrefillAttnTPFusedSymmNormRunner:
        runner = cls.__new__(cls)
        runner.capacity = capacity
        runner.output_capacity = source_push_output_capacity(capacity, spec)
        runner._initialize_symm_from_resource(
            resource=resource,
            spec=spec,
            rows=capacity,
            block_size=block_size,
            signal_backoff=signal_backoff,
            blocks_per_sm=blocks_per_sm,
            owns_resource=False,
        )
        return runner

    def run_out(
        self,
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
            raise RuntimeError("true fused Symm runner is closed")
        runtime_rows = residual.shape[0] if residual.ndim == 2 else 0
        if runtime_rows <= 0 or runtime_rows > self.capacity:
            raise ValueError(
                "Prefill Symm runtime rows must be in "
                f"[1, {self.capacity}], got {runtime_rows}"
            )
        input_view = self._input[:runtime_rows]
        _validate_run_tensors(
            input_view=input_view,
            residual=residual,
            o_norm_weight=o_norm_weight,
            post_norm_weight=post_norm_weight,
            output=output,
            residual_out=residual_out,
            expected_output_rows=source_push_output_capacity(
                runtime_rows,
                self.spec,
            ),
            device=self.device,
        )
        _validate_device_scalar(actual_rows, "actual_rows", self.device)
        _validate_device_scalar(owner_start, "owner_start", self.device)

        module = _jit_prefill_attntp_fused_symm_norm_module(
            self.spec.attn_tp_size,
            self.spec.hidden_size,
            self.spec.output_mode,
            self.spec.internal_precision,
            self.block_size,
            self.signal_backoff,
        )
        module.fused_symm_norm(
            input_view,
            self._peer_pointer,
            self._multicast_pointer,
            self._multicast_signal_offset_bytes,
            self._multicast_signal_slots,
            self._partial_pointer_table,
            self._signal_pad,
            self._peer_signal_pointer,
            residual,
            o_norm_weight,
            post_norm_weight,
            output,
            residual_out,
            actual_rows,
            owner_start,
            self.rank,
            0 if self.blocks_per_sm is None else self.blocks_per_sm,
            float(o_norm_eps),
            float(post_norm_eps),
        )

    def get_max_occupancy(self) -> int:
        module = _jit_prefill_attntp_fused_symm_norm_module(
            self.spec.attn_tp_size,
            self.spec.hidden_size,
            self.spec.output_mode,
            self.spec.internal_precision,
            self.block_size,
            self.signal_backoff,
        )
        return int(module.get_max_occupancy())


class DecodeAttnTPFusedSymmNormRunner(_AttnTPFusedSymmNormRunnerBase):
    """Fixed-capacity decode runner with a decode-specialized Symm kernel."""

    def __init__(
        self,
        *,
        group,
        device: torch.device,
        spec: AttnTPNormSpec,
        max_rows: int,
        block_size: int = 256,
        signal_backoff: int = 64,
        blocks_per_sm: int | None = None,
    ) -> None:
        decode_output_capacity(max_rows, spec)
        self.max_rows = max_rows
        self._initialize_symm(
            group=group,
            device=device,
            spec=spec,
            rows=max_rows,
            block_size=block_size,
            signal_backoff=signal_backoff,
            blocks_per_sm=blocks_per_sm,
        )

    @classmethod
    def from_resource(
        cls,
        *,
        resource,
        spec: AttnTPNormSpec,
        max_rows: int,
        block_size: int = 256,
        signal_backoff: int = 64,
        blocks_per_sm: int | None = None,
    ) -> DecodeAttnTPFusedSymmNormRunner:
        decode_output_capacity(max_rows, spec)
        runner = cls.__new__(cls)
        runner.max_rows = max_rows
        runner._initialize_symm_from_resource(
            resource=resource,
            spec=spec,
            rows=max_rows,
            block_size=block_size,
            signal_backoff=signal_backoff,
            blocks_per_sm=blocks_per_sm,
            owns_resource=False,
        )
        return runner

    def output_capacity_for(self, rows: int) -> int:
        if type(rows) is not int or rows <= 0 or rows > self.max_rows:
            raise ValueError(
                f"Decode rows must be in [1, {self.max_rows}], got {rows!r}"
            )
        return decode_output_capacity(rows, self.spec)

    def run_out(
        self,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        output: torch.Tensor,
        residual_out: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
    ) -> None:
        if self._closed:
            raise RuntimeError("true fused Symm runner is closed")
        if residual.ndim != 2:
            raise ValueError("decode residual must be two-dimensional")
        rows = residual.shape[0]
        output_rows = self.output_capacity_for(rows)
        input_view = self._input[:rows]
        _validate_run_tensors(
            input_view=input_view,
            residual=residual,
            o_norm_weight=o_norm_weight,
            post_norm_weight=post_norm_weight,
            output=output,
            residual_out=residual_out,
            expected_output_rows=output_rows,
            device=self.device,
        )
        module = _jit_decode_attntp_fused_symm_norm_module(
            self.spec.attn_tp_size,
            self.spec.hidden_size,
            self.spec.output_mode,
            self.spec.internal_precision,
            self.block_size,
            self.signal_backoff,
        )
        module.fused_symm_norm(
            input_view,
            self._peer_pointer,
            self._multicast_pointer,
            self._multicast_signal_offset_bytes,
            self._multicast_signal_slots,
            self._partial_pointer_table,
            self._signal_pad,
            self._peer_signal_pointer,
            residual,
            o_norm_weight,
            post_norm_weight,
            output,
            residual_out,
            self.rank,
            0 if self.blocks_per_sm is None else self.blocks_per_sm,
            float(o_norm_eps),
            float(post_norm_eps),
        )

    def get_max_occupancy(self) -> int:
        module = _jit_decode_attntp_fused_symm_norm_module(
            self.spec.attn_tp_size,
            self.spec.hidden_size,
            self.spec.output_mode,
            self.spec.internal_precision,
            self.block_size,
            self.signal_backoff,
        )
        return int(module.get_max_occupancy())


def _validate_device_scalar(
    tensor: torch.Tensor,
    name: str,
    device: torch.device,
) -> None:
    if (
        tensor.shape != (1,)
        or tensor.dtype != torch.int32
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be a contiguous CUDA int32 tensor with shape [1]"
        )


def _validate_run_tensors(
    *,
    input_view: torch.Tensor,
    residual: torch.Tensor,
    o_norm_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    output: torch.Tensor,
    residual_out: torch.Tensor,
    expected_output_rows: int,
    device: torch.device,
) -> None:
    if (
        residual.shape != input_view.shape
        or residual.dtype != torch.float32
        or residual.device != device
        or not residual.is_contiguous()
    ):
        raise ValueError("residual must be contiguous FP32 with the input shape")
    hidden_size = input_view.shape[1]
    for name, weight in (
        ("o_norm_weight", o_norm_weight),
        ("post_norm_weight", post_norm_weight),
    ):
        if (
            weight.shape != (hidden_size,)
            or weight.dtype != torch.bfloat16
            or weight.device != device
            or not weight.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous BF16 [{hidden_size}]")
    expected_output_shape = (expected_output_rows, hidden_size)
    if (
        output.shape != expected_output_shape
        or output.dtype != torch.bfloat16
        or output.device != device
        or not output.is_contiguous()
    ):
        raise ValueError(
            f"output must be contiguous BF16 with shape {expected_output_shape}"
        )
    if (
        residual_out.shape != expected_output_shape
        or residual_out.dtype != torch.float32
        or residual_out.device != device
        or not residual_out.is_contiguous()
    ):
        raise ValueError(
            f"residual_out must be contiguous FP32 with shape {expected_output_shape}"
        )
