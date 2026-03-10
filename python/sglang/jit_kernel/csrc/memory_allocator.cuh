// Custom memory allocators for tensors.
// Provides pinned memory (cudaMallocHost) and unified memory (cudaMallocManaged) operations.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace {

#define MEMORY_ALLOC_CUDA_CHECK(call)                                                \
  do {                                                                               \
    cudaError_t err = (call);                                                        \
    host::RuntimeCheck(err == cudaSuccess, "CUDA Error: ", cudaGetErrorString(err)); \
  } while (0)

// ---------------------------------------------------------------------------
// Memory allocation: returns raw pointer as int64_t
// Python side wraps it into a tensor via torch.from_dlpack / ctypes.
// ---------------------------------------------------------------------------

int64_t cuda_malloc_host(int64_t size_bytes) {
  void* ptr = nullptr;
  MEMORY_ALLOC_CUDA_CHECK(cudaMallocHost(&ptr, static_cast<size_t>(size_bytes)));
  return reinterpret_cast<int64_t>(ptr);
}

void cuda_free_host(int64_t ptr) {
  MEMORY_ALLOC_CUDA_CHECK(cudaFreeHost(reinterpret_cast<void*>(ptr)));
}

int64_t cuda_malloc_managed(int64_t size_bytes) {
  void* ptr = nullptr;
  MEMORY_ALLOC_CUDA_CHECK(cudaMallocManaged(&ptr, static_cast<size_t>(size_bytes)));
  return reinterpret_cast<int64_t>(ptr);
}

void cuda_free_managed(int64_t ptr) {
  MEMORY_ALLOC_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
}

// ---------------------------------------------------------------------------
// Memory prefetch operations
// ---------------------------------------------------------------------------

void unified_prefetch_to_gpu(tvm::ffi::TensorView tensor, int64_t device_id) {
  using namespace host;
  RuntimeCheck(tensor.is_contiguous(), "Tensor must be contiguous for prefetching");
  void* data_ptr = tensor.data_ptr();
  int64_t numel = 1;
  for (int i = 0; i < tensor.dim(); ++i) {
    numel *= tensor.size(i);
  }
  size_t element_size = (tensor.dtype().bits * tensor.dtype().lanes + 7) / 8;
  size_t size_bytes = static_cast<size_t>(numel) * element_size;

  MEMORY_ALLOC_CUDA_CHECK(cudaSetDevice(static_cast<int>(device_id)));
  // Use cudaStreamPerThread so the prefetch is ordered with respect to
  // work submitted on the calling thread's per-thread default stream,
  // rather than the legacy default stream (NULL / 0).
  MEMORY_ALLOC_CUDA_CHECK(cudaMemPrefetchAsync(data_ptr, size_bytes, static_cast<int>(device_id), cudaStreamPerThread));
}

void unified_prefetch_to_cpu(tvm::ffi::TensorView tensor) {
  using namespace host;
  RuntimeCheck(tensor.is_contiguous(), "Tensor must be contiguous for prefetching");
  void* data_ptr = tensor.data_ptr();
  int64_t numel = 1;
  for (int i = 0; i < tensor.dim(); ++i) {
    numel *= tensor.size(i);
  }
  size_t element_size = (tensor.dtype().bits * tensor.dtype().lanes + 7) / 8;
  size_t size_bytes = static_cast<size_t>(numel) * element_size;

  MEMORY_ALLOC_CUDA_CHECK(cudaMemPrefetchAsync(data_ptr, size_bytes, cudaCpuDeviceId, cudaStreamPerThread));
}

// ---------------------------------------------------------------------------
// Memory pinning operations
// ---------------------------------------------------------------------------

void pin_memory(tvm::ffi::TensorView tensor) {
  using namespace host;
  RuntimeCheck(tensor.is_contiguous(), "Tensor must be contiguous for pinning");
  void* data_ptr = tensor.data_ptr();
  int64_t numel = 1;
  for (int i = 0; i < tensor.dim(); ++i) {
    numel *= tensor.size(i);
  }
  size_t element_size = (tensor.dtype().bits * tensor.dtype().lanes + 7) / 8;
  size_t size_bytes = static_cast<size_t>(numel) * element_size;

  MEMORY_ALLOC_CUDA_CHECK(cudaHostRegister(data_ptr, size_bytes, cudaHostRegisterDefault));
}

void unpin_memory(tvm::ffi::TensorView tensor) {
  void* data_ptr = tensor.data_ptr();
  MEMORY_ALLOC_CUDA_CHECK(cudaHostUnregister(data_ptr));
}

// ---------------------------------------------------------------------------
// Unified memory advise
// ---------------------------------------------------------------------------

void mem_advise_preferred_location_cpu(int64_t ptr, int64_t size_bytes) {
  MEMORY_ALLOC_CUDA_CHECK(cudaMemAdvise(
      reinterpret_cast<void*>(ptr),
      static_cast<size_t>(size_bytes),
      cudaMemAdviseSetPreferredLocation,
      cudaCpuDeviceId));
}

void mem_advise_accessed_by(int64_t ptr, int64_t size_bytes, int64_t device_id) {
  MEMORY_ALLOC_CUDA_CHECK(cudaMemAdvise(
      reinterpret_cast<void*>(ptr),
      static_cast<size_t>(size_bytes),
      cudaMemAdviseSetAccessedBy,
      static_cast<int>(device_id)));
}

void mem_prefetch_async(int64_t ptr, int64_t size_bytes, int64_t device_id) {
  if (device_id >= 0) {
    MEMORY_ALLOC_CUDA_CHECK(cudaSetDevice(static_cast<int>(device_id)));
  }
  MEMORY_ALLOC_CUDA_CHECK(cudaMemPrefetchAsync(
      reinterpret_cast<void*>(ptr), static_cast<size_t>(size_bytes), static_cast<int>(device_id), cudaStreamPerThread));
}

void set_device(int64_t device_id) {
  MEMORY_ALLOC_CUDA_CHECK(cudaSetDevice(static_cast<int>(device_id)));
}

#undef MEMORY_ALLOC_CUDA_CHECK

}  // namespace
