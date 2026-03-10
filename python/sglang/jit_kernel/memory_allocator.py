"""Custom memory allocators using CUDA pinned memory and unified memory.

This module provides JIT-compiled memory allocation utilities that were
previously in the external ``prc_custom_ops`` package:

- ``custom_empty``: allocate tensor with ``cudaMallocHost`` (pinned host memory,
  presented as a CUDA device tensor)
- ``unified_empty``: allocate tensor with ``cudaMallocManaged`` (unified memory,
  CPU-preferred)
- ``unified_empty_with_device``: allocate unified memory tensor with GPU device
  access hint
- ``unified_prefetch_to_gpu``: prefetch unified memory tensor to GPU
- ``unified_prefetch_to_cpu``: prefetch unified memory tensor to CPU
- ``pin_memory`` / ``unpin_memory``: pin/unpin existing host tensor memory
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


# ---------------------------------------------------------------------------
# JIT module loading
# ---------------------------------------------------------------------------


@cache_once
def _jit_memory_allocator_module() -> Module:
    return load_jit(
        "memory_allocator",
        cuda_files=["memory_allocator.cuh"],
        cuda_wrappers=[
            ("cuda_malloc_host", "cuda_malloc_host"),
            ("cuda_free_host", "cuda_free_host"),
            ("cuda_malloc_managed", "cuda_malloc_managed"),
            ("cuda_free_managed", "cuda_free_managed"),
            ("unified_prefetch_to_gpu", "unified_prefetch_to_gpu"),
            ("unified_prefetch_to_cpu", "unified_prefetch_to_cpu"),
            ("pin_memory", "pin_memory"),
            ("unpin_memory", "unpin_memory"),
            ("mem_advise_preferred_location_cpu", "mem_advise_preferred_location_cpu"),
            ("mem_advise_accessed_by", "mem_advise_accessed_by"),
            ("mem_prefetch_async", "mem_prefetch_async"),
            ("set_device", "set_device"),
        ],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DTYPE_SIZE = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.bool: 1,
}


def _element_size(dtype: torch.dtype) -> int:
    if dtype in _DTYPE_SIZE:
        return _DTYPE_SIZE[dtype]
    return torch.tensor([], dtype=dtype).element_size()


def _numel(sizes) -> int:
    n = 1
    for s in sizes:
        n *= s
    return n


@cache_once
def _get_cudart():
    """Return a ctypes handle to the CUDA runtime library."""
    import ctypes

    try:
        return ctypes.CDLL("libcudart.so")
    except OSError:
        return ctypes.CDLL("libcudart.so.12")


def _make_tensor_from_ptr(
    ptr: int,
    sizes: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | str,
    free_fn,
) -> torch.Tensor:
    """Create a torch.Tensor from a raw memory pointer with a custom destructor.

    Mirrors the C++ ``torch::from_blob(ptr, sizes, deleter, TensorOptions().dtype(dtype).device(device))``.

    For pinned host memory (``cudaMallocHost``) the tensor is presented as a
    **CUDA tensor** so that ``copy_`` dispatches to the CUDA copy path.
    Under Unified Virtual Addressing (UVA) the pinned host pointer is directly
    usable as a CUDA device pointer, so we pass it to
    ``_construct_storage_from_data_pointer`` without translation.

    IMPORTANT: The destructor guard is attached to the *storage* object (not
    the tensor) because ``nn.Parameter(tensor)`` does **not** inherit custom
    Python attributes from the wrapped tensor.  The storage, however, is
    shared between the original tensor and any Parameter wrapping it, so
    the guard survives as long as the underlying data is alive.
    """

    numel = _numel(sizes)
    element_size = _element_size(dtype)
    total_bytes = numel * element_size

    device = torch.device(device) if isinstance(device, str) else device

    # Create an UntypedStorage that wraps the raw pointer.
    # For cudaMallocHost pointers the host address is valid on both host and
    # device under UVA – no cudaHostGetDevicePointer translation needed.
    storage = torch._C._construct_storage_from_data_pointer(ptr, device, total_bytes)

    # Build a tensor from the storage with the desired shape and contiguous strides.
    ndim = len(sizes)
    strides: tuple[int, ...] = ()
    if ndim > 0:
        strides_list = [0] * ndim
        strides_list[-1] = 1
        for i in range(ndim - 2, -1, -1):
            strides_list[i] = strides_list[i + 1] * sizes[i + 1]
        strides = tuple(strides_list)

    tensor = torch.tensor([], dtype=dtype, device=device).set_(
        storage, 0, sizes, strides
    )

    # Prevent premature release: attach a destructor guard on the *storage*
    # object.  We deliberately do NOT set it on the tensor, because
    # ``nn.Parameter(tensor)`` does not inherit custom attributes from the
    # wrapped tensor – the original tensor may be GC'd while the Parameter
    # (sharing the same storage) is still alive.  Storing the guard on the
    # storage guarantees the destructor outlives the data.
    #
    # We avoid ``weakref.finalize`` for the same reason – the weak-ref
    # target (original tensor) can die before the storage.
    class _prevent_free:
        """prevent premature deallocation of the backing memory.

        Stored on the ``UntypedStorage`` object as ``storage._prevent_free``.
        When the storage is garbage-collected, ``__del__`` fires and calls
        the provided ``free_fn`` to release the CUDA memory.
        """

        __slots__ = ("_ptr", "_free", "_freed")

        def __init__(self, p, f):
            self._ptr = p
            self._free = f
            self._freed = False

        def __del__(self):
            if self._freed:
                return
            self._freed = True
            try:
                self._free(self._ptr)
            except Exception:
                pass

    guard = _prevent_free(ptr, free_fn)
    storage._prevent_free = guard

    return tensor


# ---------------------------------------------------------------------------
# Public API: Memory allocation functions
# ---------------------------------------------------------------------------


def custom_empty(
    sizes: tuple[int, ...] | list[int],
    dtype: torch.dtype = torch.float32,
    device_id: int = 0,
) -> torch.Tensor:
    """Create an empty tensor backed by ``cudaMallocHost`` (pinned host memory).

    The tensor is allocated in page-locked (pinned) host memory, enabling
    zero-copy GPU access and fast asynchronous H2D / D2H transfers.

    Args:
        sizes: Shape of the tensor.
        dtype: Data type (default: ``torch.float32``).
        device_id: CUDA device ID (default: 0).

    Returns:
        A ``torch.Tensor`` backed by pinned host memory.
    """
    import ctypes

    sizes = tuple(sizes)
    total_bytes = _numel(sizes) * _element_size(dtype)

    # Guard against zero-size allocations: cudaMallocHost(0) returns a null
    # pointer which would cause undefined behaviour in storage construction.
    if total_bytes == 0:
        return torch.empty(sizes, dtype=dtype, device=torch.device("cuda", device_id))

    # Use ctypes to call cudaMallocHost directly – this matches the approach
    # validated by test_pinned2.py on the execution node.
    cudart = _get_cudart()
    torch.cuda.set_device(device_id)

    host_ptr = ctypes.c_void_p(0)
    err = cudart.cudaMallocHost(ctypes.byref(host_ptr), ctypes.c_size_t(total_bytes))
    if err != 0:
        raise RuntimeError(
            f"cudaMallocHost failed with error code {err} "
            f"(requested {total_bytes} bytes)"
        )
    ptr = host_ptr.value

    def _free_host(p, _cudart=cudart, _c_void_p=ctypes.c_void_p):
        _cudart.cudaFreeHost(_c_void_p(p))

    tensor = _make_tensor_from_ptr(
        ptr, sizes, dtype, torch.device("cuda", device_id), _free_host
    )

    # Synchronize and check for latent CUDA errors right after tensor creation.
    torch.cuda.synchronize(device_id)

    # Quick sanity test: try a small fill to verify the tensor is actually usable.
    try:
        if tensor.numel() > 0:
            tensor.view(-1)[0:1].data.fill_(0)
            torch.cuda.synchronize(device_id)
    except Exception:
        raise

    return tensor


def unified_empty(
    sizes: tuple[int, ...] | list[int],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create an empty tensor backed by ``cudaMallocManaged`` (unified memory).

    The data is placed in unified memory with CPU as the preferred location,
    and is prefetched to CPU.

    Args:
        sizes: Shape of the tensor.
        dtype: Data type (default: ``torch.float32``).

    Returns:
        A ``torch.Tensor`` on CPU backed by unified memory.
    """
    sizes = tuple(sizes)
    total_bytes = _numel(sizes) * _element_size(dtype)

    if total_bytes == 0:
        return torch.empty(sizes, dtype=dtype, device="cpu")

    module = _jit_memory_allocator_module()
    ptr = module.cuda_malloc_managed(total_bytes)

    # Set preferred location to CPU and prefetch
    module.mem_advise_preferred_location_cpu(ptr, total_bytes)
    module.mem_prefetch_async(ptr, total_bytes, -1)  # cudaCpuDeviceId = -1

    return _make_tensor_from_ptr(ptr, sizes, dtype, "cpu", module.cuda_free_managed)


def unified_empty_with_device(
    sizes: tuple[int, ...] | list[int],
    dtype: torch.dtype = torch.float32,
    device_id: int = 0,
) -> torch.Tensor:
    """Create an empty tensor backed by ``cudaMallocManaged`` with GPU access hint.

    The data is in unified memory with the specified GPU set as an accessor.

    Args:
        sizes: Shape of the tensor.
        dtype: Data type (default: ``torch.float32``).
        device_id: CUDA device ID (default: 0).

    Returns:
        A ``torch.Tensor`` backed by unified memory.
    """
    sizes = tuple(sizes)
    total_bytes = _numel(sizes) * _element_size(dtype)

    if total_bytes == 0:
        return torch.empty(sizes, dtype=dtype, device=torch.device("cuda", device_id))

    module = _jit_memory_allocator_module()
    module.set_device(device_id)
    ptr = module.cuda_malloc_managed(total_bytes)

    # Set memory advise: accessed by the specified device
    module.mem_advise_accessed_by(ptr, total_bytes, device_id)

    return _make_tensor_from_ptr(
        ptr, sizes, dtype, torch.device("cuda", device_id), module.cuda_free_managed
    )


# ---------------------------------------------------------------------------
# Public API: Memory management functions (JIT-compiled, operate on tensors)
# ---------------------------------------------------------------------------


def unified_prefetch_to_gpu(tensor: torch.Tensor, device_id: int = 0) -> None:
    """Prefetch a unified memory tensor to the specified GPU device.

    Args:
        tensor: A contiguous tensor backed by unified memory.
        device_id: Target CUDA device ID (default: 0).
    """
    module = _jit_memory_allocator_module()
    module.unified_prefetch_to_gpu(tensor, device_id)


def unified_prefetch_to_cpu(tensor: torch.Tensor) -> None:
    """Prefetch a unified memory tensor back to CPU.

    Args:
        tensor: A contiguous tensor backed by unified memory.
    """
    module = _jit_memory_allocator_module()
    module.unified_prefetch_to_cpu(tensor)


def pin_memory(tensor: torch.Tensor) -> None:
    """Pin the memory of an existing CPU tensor (make it page-locked).

    This enables faster H2D transfers for the tensor.

    Args:
        tensor: A contiguous CPU tensor.
    """
    module = _jit_memory_allocator_module()
    module.pin_memory(tensor)


def unpin_memory(tensor: torch.Tensor) -> None:
    """Unpin previously pinned host memory of a CPU tensor.

    Args:
        tensor: A contiguous CPU tensor that was previously pinned.
    """
    module = _jit_memory_allocator_module()
    module.unpin_memory(tensor)
