"""Tests for the memory_allocator JIT kernel module."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# custom_empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int32])
@pytest.mark.parametrize("sizes", [(128,), (4, 32), (2, 3, 16), (1024,)])
def test_custom_empty_shape_dtype(sizes, dtype):
    from sglang.jit_kernel.memory_allocator import custom_empty

    t = custom_empty(sizes, dtype=dtype)
    assert t.shape == sizes
    assert t.dtype == dtype


def test_custom_empty_default():
    from sglang.jit_kernel.memory_allocator import custom_empty

    t = custom_empty((64,))
    assert t.shape == (64,)
    assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# unified_empty
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_unified_empty(dtype):
    from sglang.jit_kernel.memory_allocator import unified_empty

    t = unified_empty((256,), dtype=dtype)
    assert t.shape == (256,)
    assert t.dtype == dtype


# ---------------------------------------------------------------------------
# unified_empty_with_device
# ---------------------------------------------------------------------------


def test_unified_empty_with_device():
    from sglang.jit_kernel.memory_allocator import unified_empty_with_device

    t = unified_empty_with_device((128,), dtype=torch.float32, device_id=0)
    assert t.shape == (128,)
    assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# pin / unpin memory
# ---------------------------------------------------------------------------


def test_pin_unpin_memory():
    from sglang.jit_kernel.memory_allocator import pin_memory, unpin_memory

    t = torch.randn(1024, dtype=torch.float32)
    pin_memory(t)
    unpin_memory(t)


# ---------------------------------------------------------------------------
# unified_prefetch
# ---------------------------------------------------------------------------


def test_unified_prefetch_round_trip():
    from sglang.jit_kernel.memory_allocator import (
        unified_empty,
        unified_prefetch_to_cpu,
        unified_prefetch_to_gpu,
    )

    t = unified_empty((512,), dtype=torch.float32)
    unified_prefetch_to_gpu(t, device_id=0)
    torch.cuda.synchronize()
    unified_prefetch_to_cpu(t)
    torch.cuda.synchronize()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
