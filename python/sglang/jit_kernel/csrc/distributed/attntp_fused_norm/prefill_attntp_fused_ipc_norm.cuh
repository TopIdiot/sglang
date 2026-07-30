#include <sgl_kernel/ffi.h>
#include <sgl_kernel/math.cuh>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/distributed/common.cuh>
#include <sgl_kernel/distributed/custom_all_reduce.cuh>

#include "attntp_fused_ipc_norm_common.cuh"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

using device::distributed::PullController, device::distributed::PushController;
using host::distributed::AllReduceData, host::distributed::CustomAllReduceBase,
    host::distributed::CustomAllReduceRef;

template <typename T>
SGL_DEVICE T* byte_offset(void* base, uint64_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(base) + offset);
}

template <typename T>
SGL_DEVICE const T* byte_offset(const void* base, uint64_t offset) {
  return reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(base) + offset);
}

template <typename T>
SGL_DEVICE void store_volatile_16b(void* address, const T& value) {
  static_assert(sizeof(T) == 16 && alignof(T) == 16);
  const uint4 raw = *reinterpret_cast<const uint4*>(&value);
  asm volatile(
      "st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};"
      :
      : "r"(raw.x), "r"(raw.y), "r"(raw.z), "r"(raw.w), "l"(address)
      : "memory");
}

template <typename T>
SGL_DEVICE T load_volatile_16b(const void* address) {
  static_assert(sizeof(T) == 16 && alignof(T) == 16);
  uint4 raw;
  asm volatile(
      "ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
      : "l"(address)
      : "memory");
  return *reinterpret_cast<const T*>(&raw);
}

SGL_DEVICE void store_release_system(uint32_t* address, uint32_t value) {
  asm volatile(
      "st.release.sys.global.u32 [%0], %1;"
      :
      : "l"(address), "r"(value)
      : "memory");
}

SGL_DEVICE uint32_t load_acquire_system(const uint32_t* address) {
  uint32_t value;
  asm volatile(
      "ld.acquire.sys.global.u32 %0, [%1];"
      : "=r"(value)
      : "l"(address)
      : "memory");
  return value;
}

template <uint32_t kNumWarps>
SGL_DEVICE float block_reduce_sum(float value, float* warp_sums) {
  const uint32_t lane = threadIdx.x % device::kWarpThreads;
  const uint32_t warp = threadIdx.x / device::kWarpThreads;
  value = device::warp::reduce_sum(value);
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();

  if (warp == 0) {
    const float warp_value = lane < kNumWarps ? warp_sums[lane] : 0.0f;
    const float total = device::warp::reduce_sum<kNumWarps>(warp_value);
    if (lane == 0) warp_sums[0] = total;
  }
  __syncthreads();
  return warp_sums[0];
}

struct FusedPrefillAttnTPIPCNormParams {
  void* buffer[device::distributed::kMaxNumGPU];
  const void* partial;
  const void* residual;
  const void* o_norm_weight;
  const void* post_norm_weight;
  void* output;
  void* residual_out;
  uint64_t buffer_bytes;
  uint64_t epoch_bytes;
  uint64_t signal_offset_bytes;
  float o_norm_eps;
  float post_norm_eps;
  uint32_t rank;
  uint32_t num_tokens;
  uint32_t num_tiles;
  uint32_t num_controller_slots;
};

struct FusedLocalAttnTPNormParams {
  const void* partial;
  const void* residual;
  const void* o_norm_weight;
  const void* post_norm_weight;
  void* output;
  void* residual_out;
  float o_norm_eps;
  float post_norm_eps;
  uint32_t num_tokens;
};

template <typename Trait>
__global__ void fused_local_attntp_norm_kernel(
    const FusedLocalAttnTPNormParams __grid_constant__ params) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kVectorsPerThread = Trait::kVectorsPerThread;

  __shared__ float warp_sums[Trait::kNumWarps];

  BF16Storage o_weight_values[kVectorsPerThread];
  BF16Storage post_weight_values[kVectorsPerThread];
#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
    const uint32_t storage_index =
        threadIdx.x * kVectorsPerThread + vector;
    o_weight_values[vector].load(params.o_norm_weight, storage_index);
    post_weight_values[vector].load(
        params.post_norm_weight, storage_index);
  }

  for (uint32_t row = blockIdx.x; row < params.num_tokens;
       row += gridDim.x) {
    const auto* partial_row = static_cast<const DType*>(params.partial) +
                              static_cast<uint64_t>(row) * kHiddenSize;
    const auto* residual_row = static_cast<const float*>(params.residual) +
                               static_cast<uint64_t>(row) * kHiddenSize;
    auto* output_row = static_cast<DType*>(params.output) +
                       static_cast<uint64_t>(row) * kHiddenSize;
    auto* residual_out_row = static_cast<float*>(params.residual_out) +
                             static_cast<uint64_t>(row) * kHiddenSize;

    if constexpr (
        Trait::kInternalPrecision ==
        sglang::jit_kernel::attntp::NormInternalPrecision::kFullFP32) {
      float activation_values[kVectorsPerThread]
                             [Trait::kElementsPerVector];
#pragma unroll
      for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
        BF16Storage local_values;
        const uint32_t storage_index =
            threadIdx.x * kVectorsPerThread + vector;
        local_values.load(partial_row, storage_index);
#pragma unroll
        for (uint32_t index = 0; index < Trait::kElementsPerVector;
             ++index) {
          activation_values[vector][index] =
              device::cast<float>(local_values[index]);
        }
      }
      sglang::jit_kernel::attntp::apply_welm_norm_pipeline_full_fp32<
          Trait>(
          activation_values,
          residual_row,
          o_weight_values,
          post_weight_values,
          output_row,
          residual_out_row,
          params.o_norm_eps,
          params.post_norm_eps,
          warp_sums);
    } else {
      BF16Storage reduced_values[kVectorsPerThread];
#pragma unroll
      for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
        BF16Storage local_values;
        const uint32_t storage_index =
            threadIdx.x * kVectorsPerThread + vector;
        local_values.load(partial_row, storage_index);
#pragma unroll
        for (uint32_t index = 0; index < Trait::kElementsPerVector;
             ++index) {
          float reduced = 0.0f;
          reduced += device::cast<float>(local_values[index]);
          reduced_values[vector][index] = device::cast<DType>(reduced);
        }
      }
      sglang::jit_kernel::attntp::apply_welm_norm_pipeline<Trait>(
          reduced_values,
          residual_row,
          o_weight_values,
          post_weight_values,
          output_row,
          residual_out_row,
          params.o_norm_eps,
          params.post_norm_eps,
          warp_sums);
    }
  }
}

template <
    typename DType,
    uint32_t kHiddenSize,
    uint32_t kInternalPrecision>
struct FusedLocalAttnTPNorm {
  using Trait = sglang::jit_kernel::attntp::NormPipelineTrait<
      DType,
      kHiddenSize,
      kInternalPrecision>;
  static constexpr auto kernel = fused_local_attntp_norm_kernel<Trait>;

  static void run(
      const tvm::ffi::Tensor partial,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const float o_norm_eps,
      const float post_norm_eps) {
    using namespace host;

    auto num_tokens_symbol = SymbolicSize{"num_tokens"};
    auto device_symbol = SymbolicDevice{};
    device_symbol.set_options<kDLCUDA>();
    TensorMatcher({num_tokens_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(partial);
    TensorMatcher({num_tokens_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual);
    TensorMatcher({kHiddenSize})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(o_norm_weight);
    TensorMatcher({kHiddenSize})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(post_norm_weight);
    TensorMatcher({num_tokens_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(output);
    TensorMatcher({num_tokens_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual_out);

    const auto device = device_symbol.unwrap();
    const auto num_tokens =
        static_cast<uint32_t>(num_tokens_symbol.unwrap());
    RuntimeCheck(
        num_tokens > 0, "Local AttnTP fused norm requires non-empty input");
    for (const auto* pointer :
         {partial.data_ptr(),
          residual.data_ptr(),
          o_norm_weight.data_ptr(),
          post_norm_weight.data_ptr(),
          output.data_ptr(),
          residual_out.data_ptr()}) {
      RuntimeCheck(
          std::bit_cast<intptr_t>(pointer) % 16 == 0,
          "Local AttnTP fused norm tensors must be 16-byte aligned");
    }

    int device_id = 0;
    host::RuntimeDeviceCheck(cudaGetDevice(&device_id));
    const uint32_t max_kernel_blocks =
        get_max_occupancy() * host::runtime::get_sm_count(device_id);
    const uint32_t num_blocks =
        std::min(num_tokens, max_kernel_blocks);

    FusedLocalAttnTPNormParams params{};
    params.partial = partial.data_ptr();
    params.residual = residual.data_ptr();
    params.o_norm_weight = o_norm_weight.data_ptr();
    params.post_norm_weight = post_norm_weight.data_ptr();
    params.output = output.data_ptr();
    params.residual_out = residual_out.data_ptr();
    params.o_norm_eps = o_norm_eps;
    params.post_norm_eps = post_norm_eps;
    params.num_tokens = num_tokens;

    LaunchKernel(num_blocks, Trait::kBlockSize, device)(kernel, params);
  }

  static uint32_t get_max_occupancy() {
    return host::runtime::get_blocks_per_sm(kernel, Trait::kBlockSize);
  }
};

template <
    typename DType_,
    uint32_t kNumGPU_,
    uint32_t kHiddenSize_,
    uint32_t kRowsPerSignal_,
    bool kCachePeerTile_,
    uint32_t kMinBlocksPerSM_,
    bool kUsePDL_,
    uint32_t kInternalPrecision_>
struct FusedPrefillAttnTPIPCNormTrait {
  using DType = DType_;
  static constexpr uint32_t kNumGPU = kNumGPU_;
  static constexpr uint32_t kHiddenSize = kHiddenSize_;
  static constexpr uint32_t kRowsPerSignal = kRowsPerSignal_;
  static constexpr bool kCachePeerTile = kCachePeerTile_;
  static constexpr bool kUsePDL = kUsePDL_;
  static constexpr auto kInternalPrecision =
      static_cast<sglang::jit_kernel::attntp::NormInternalPrecision>(
          kInternalPrecision_);
  static constexpr uint32_t kBlockSize = 256;
  static constexpr uint32_t kNumWarps = kBlockSize / device::kWarpThreads;
  static constexpr uint32_t kElementsPerThread = kHiddenSize / kBlockSize;
  static constexpr uint32_t kMinBlocksPerSM = kMinBlocksPerSM_;
  using BF16Storage = device::AlignedVector<DType, kElementsPerThread>;
  using FP32Storage = device::AlignedVector<float, kElementsPerThread / 2>;

  static_assert(kNumGPU == 2, "AttnTP fused norm requires exactly two GPUs");
  static_assert(std::is_same_v<DType, bf16_t>, "AttnTP fused norm requires BF16");
  static_assert(kHiddenSize == 2048, "AttnTP fused norm requires hidden size 2048");
  static_assert(
      kRowsPerSignal == 1 || kRowsPerSignal == 2 ||
      kRowsPerSignal == 4 || kRowsPerSignal == 8,
      "AttnTP fused norm supports 1, 2, 4 or 8 rows per signal");
  static_assert(
      kMinBlocksPerSM == 4 || kMinBlocksPerSM == 5 ||
      kMinBlocksPerSM == 6,
      "AttnTP fused norm supports min occupancy 4, 5 or 6");
  static_assert(
      kMinBlocksPerSM == 4 ||
          (!kCachePeerTile && kRowsPerSignal == 8),
      "High-occupancy AttnTP fused norm requires rows8/local");
  static_assert(kHiddenSize % kBlockSize == 0);
  static_assert(kElementsPerThread == 8);
  static_assert(sizeof(BF16Storage) == 16);
  static_assert(sizeof(FP32Storage) == 16);
  static_assert(
      sglang::jit_kernel::attntp::is_supported_internal_precision(
          kInternalPrecision));
};

template <typename Trait>
SGL_DEVICE void prefill_attntp_publish_tile(
    const FusedPrefillAttnTPIPCNormParams& params,
    const uint32_t tile,
    void* send_slot,
    void* recv_slot,
    typename Trait::BF16Storage* cached_tile) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;

  constexpr uint32_t kRowBytes =
      Trait::kHiddenSize * sizeof(DType);
  constexpr uint32_t kRowsPerSignal = Trait::kRowsPerSignal;

  auto* send_signal = byte_offset<uint32_t>(
      send_slot,
      params.signal_offset_bytes + static_cast<uint64_t>(tile) * 4);
  auto* recv_signal = byte_offset<uint32_t>(
      recv_slot,
      params.signal_offset_bytes + static_cast<uint64_t>(tile) * 4);

  if (threadIdx.x == 0) {
    while (load_acquire_system(send_signal) != 0) {
      __nanosleep(64);
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t row_in_tile = 0; row_in_tile < kRowsPerSignal;
       ++row_in_tile) {
    const uint32_t row = tile * kRowsPerSignal + row_in_tile;
    if (row >= params.num_tokens) break;
    const auto* partial_row = byte_offset<DType>(
        params.partial, static_cast<uint64_t>(row) * kRowBytes);
    auto* send_row =
        byte_offset<void>(send_slot, static_cast<uint64_t>(row) * kRowBytes);
    BF16Storage local_values;
    local_values.load(partial_row, threadIdx.x);
    if constexpr (!Trait::kCachePeerTile) {
      cached_tile[row_in_tile * Trait::kBlockSize + threadIdx.x] =
          local_values;
    }
    store_volatile_16b(
        byte_offset<void>(
            send_row,
            static_cast<uint64_t>(threadIdx.x) * sizeof(BF16Storage)),
        local_values);
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) store_release_system(send_signal, 1);

  if (threadIdx.x == 0) {
    while (load_acquire_system(recv_signal) != 1) {
      __nanosleep(64);
    }
  }
  __syncthreads();
  (void)load_acquire_system(recv_signal);

  if constexpr (Trait::kCachePeerTile) {
#pragma unroll
    for (uint32_t row_in_tile = 0; row_in_tile < kRowsPerSignal;
         ++row_in_tile) {
      const uint32_t row = tile * kRowsPerSignal + row_in_tile;
      if (row >= params.num_tokens) break;
      const auto* recv_row = byte_offset<void>(
          recv_slot, static_cast<uint64_t>(row) * kRowBytes);
      cached_tile[row_in_tile * Trait::kBlockSize + threadIdx.x] =
          load_volatile_16b<BF16Storage>(byte_offset<void>(
              recv_row,
              static_cast<uint64_t>(threadIdx.x) *
                  sizeof(BF16Storage)));
    }
    __syncthreads();
    if (threadIdx.x == 0) store_release_system(recv_signal, 0);
  }
}

template <typename Trait>
SGL_DEVICE void prefill_attntp_compute_tile(
    const FusedPrefillAttnTPIPCNormParams& params,
    const uint32_t tile,
    void* recv_slot,
    const typename Trait::BF16Storage& persistent_o_weight,
    const typename Trait::BF16Storage& persistent_post_weight,
    typename Trait::BF16Storage* cached_tile,
    float* warp_sums) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;
  using FP32Storage = typename Trait::FP32Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kRowBytes = kHiddenSize * sizeof(DType);
  constexpr uint32_t kRowsPerSignal = Trait::kRowsPerSignal;
  constexpr uint32_t kNumWarps = Trait::kNumWarps;
  constexpr uint32_t kElementsPerThread = Trait::kElementsPerThread;

  auto* recv_signal = byte_offset<uint32_t>(
      recv_slot,
      params.signal_offset_bytes + static_cast<uint64_t>(tile) * 4);

#pragma unroll
  for (uint32_t row_in_tile = 0; row_in_tile < kRowsPerSignal;
       ++row_in_tile) {
    const uint32_t row = tile * kRowsPerSignal + row_in_tile;
    if (row >= params.num_tokens) break;
    BF16Storage peer_values;
    BF16Storage local_values;
    if constexpr (Trait::kCachePeerTile) {
      const auto* partial_row = byte_offset<DType>(
          params.partial, static_cast<uint64_t>(row) * kRowBytes);
      peer_values =
          cached_tile[row_in_tile * Trait::kBlockSize + threadIdx.x];
      local_values.load(partial_row, threadIdx.x);
    } else {
      const auto* recv_row = byte_offset<void>(
          recv_slot, static_cast<uint64_t>(row) * kRowBytes);
      peer_values = load_volatile_16b<BF16Storage>(byte_offset<void>(
          recv_row,
          static_cast<uint64_t>(threadIdx.x) * sizeof(BF16Storage)));
      local_values =
          cached_tile[row_in_tile * Trait::kBlockSize + threadIdx.x];
    }

    BF16Storage reduced_values;
    float o_norm_sum = 0.0f;
#pragma unroll
    for (uint32_t index = 0; index < kElementsPerThread; ++index) {
      const float reduced = device::cast<float>(local_values[index]) +
                            device::cast<float>(peer_values[index]);
      reduced_values[index] = device::cast<DType>(reduced);
      const float rounded = device::cast<float>(reduced_values[index]);
      o_norm_sum += rounded * rounded;
    }

    const float o_norm_total =
        block_reduce_sum<kNumWarps>(o_norm_sum, warp_sums);
    const float o_norm_scale =
        rsqrtf(o_norm_total / static_cast<float>(kHiddenSize) +
               params.o_norm_eps);

    BF16Storage o_norm_values;
    if constexpr (Trait::kMinBlocksPerSM == 6) {
      BF16Storage o_weight;
      o_weight.load(params.o_norm_weight, threadIdx.x);
#pragma unroll
      for (uint32_t index = 0; index < kElementsPerThread; ++index) {
        const float normalized =
            device::cast<float>(reduced_values[index]) * o_norm_scale *
            device::cast<float>(o_weight[index]);
        o_norm_values[index] = device::cast<DType>(normalized);
      }
    } else {
#pragma unroll
      for (uint32_t index = 0; index < kElementsPerThread; ++index) {
        const float normalized =
            device::cast<float>(reduced_values[index]) * o_norm_scale *
            device::cast<float>(persistent_o_weight[index]);
        o_norm_values[index] = device::cast<DType>(normalized);
      }
    }

    const auto* residual_row = byte_offset<float>(
        params.residual,
        static_cast<uint64_t>(row) * kHiddenSize * sizeof(float));
    auto* residual_out_row = byte_offset<float>(
        params.residual_out,
        static_cast<uint64_t>(row) * kHiddenSize * sizeof(float));
    FP32Storage residual_values_0;
    FP32Storage residual_values_1;
    residual_values_0.load(residual_row, threadIdx.x * 2);
    residual_values_1.load(residual_row, threadIdx.x * 2 + 1);

    FP32Storage residual_out_values_0;
    FP32Storage residual_out_values_1;
    BF16Storage post_norm_inputs;
    float post_norm_sum = 0.0f;
#pragma unroll
    for (uint32_t index = 0; index < kElementsPerThread; ++index) {
      const float residual_value =
          index < kElementsPerThread / 2
              ? residual_values_0[index]
              : residual_values_1[index - kElementsPerThread / 2];
      const float residual_sum =
          device::cast<float>(o_norm_values[index]) + residual_value;
      if (index < kElementsPerThread / 2) {
        residual_out_values_0[index] = residual_sum;
      } else {
        residual_out_values_1[index - kElementsPerThread / 2] =
            residual_sum;
      }
      post_norm_inputs[index] = device::cast<DType>(residual_sum);
      const float rounded = device::cast<float>(post_norm_inputs[index]);
      post_norm_sum += rounded * rounded;
    }
    residual_out_values_0.store(residual_out_row, threadIdx.x * 2);
    residual_out_values_1.store(residual_out_row, threadIdx.x * 2 + 1);

    const float post_norm_total =
        block_reduce_sum<kNumWarps>(post_norm_sum, warp_sums);
    const float post_norm_scale =
        rsqrtf(post_norm_total / static_cast<float>(kHiddenSize) +
               params.post_norm_eps);

    BF16Storage output_values;
    if constexpr (Trait::kMinBlocksPerSM == 6) {
      BF16Storage post_weight;
      post_weight.load(params.post_norm_weight, threadIdx.x);
#pragma unroll
      for (uint32_t index = 0; index < kElementsPerThread; ++index) {
        const float normalized =
            device::cast<float>(post_norm_inputs[index]) *
            post_norm_scale * device::cast<float>(post_weight[index]);
        output_values[index] = device::cast<DType>(normalized);
      }
    } else {
#pragma unroll
      for (uint32_t index = 0; index < kElementsPerThread; ++index) {
        const float normalized =
            device::cast<float>(post_norm_inputs[index]) *
            post_norm_scale *
            device::cast<float>(persistent_post_weight[index]);
        output_values[index] = device::cast<DType>(normalized);
      }
    }
    auto* output_row = byte_offset<DType>(
        params.output, static_cast<uint64_t>(row) * kRowBytes);
    output_values.store(output_row, threadIdx.x);
  }

  __syncthreads();
  if constexpr (!Trait::kCachePeerTile) {
    if (threadIdx.x == 0) store_release_system(recv_signal, 0);
  }
}

template <typename Trait>
__global__ __launch_bounds__(
    Trait::kBlockSize,
    Trait::kMinBlocksPerSM) void fused_prefill_attntp_ipc_norm_kernel(
    const FusedPrefillAttnTPIPCNormParams __grid_constant__ params,
    const PushController __grid_constant__ ctrl) {
  using BF16Storage = typename Trait::BF16Storage;

  constexpr uint32_t kRowsPerSignal = Trait::kRowsPerSignal;
  constexpr uint32_t kNumWarps = Trait::kNumWarps;

  __shared__ float warp_sums[kNumWarps];
  __shared__ BF16Storage cached_tile[
      kRowsPerSignal * Trait::kBlockSize];

  const uint32_t block = blockIdx.x;
  const uint32_t peer = 1 - params.rank;
  const uint64_t epoch_offset = ctrl.epoch() * params.epoch_bytes;
  void* send_slot = byte_offset<void>(
      params.buffer[peer],
      epoch_offset + static_cast<uint64_t>(params.rank) * params.buffer_bytes);
  void* recv_slot = byte_offset<void>(
      params.buffer[params.rank],
      epoch_offset + static_cast<uint64_t>(peer) * params.buffer_bytes);

  device::PDLWaitPrimary<Trait::kUsePDL>();
  device::PDLTriggerSecondary<Trait::kUsePDL>();

  BF16Storage persistent_o_weight;
  BF16Storage persistent_post_weight;
  if constexpr (Trait::kMinBlocksPerSM != 6) {
    persistent_o_weight.load(params.o_norm_weight, threadIdx.x);
    persistent_post_weight.load(params.post_norm_weight, threadIdx.x);
  }

  for (uint32_t tile = block; tile < params.num_tiles; tile += gridDim.x) {
    prefill_attntp_publish_tile<Trait>(
        params, tile, send_slot, recv_slot, cached_tile);
    prefill_attntp_compute_tile<Trait>(
        params,
        tile,
        recv_slot,
        persistent_o_weight,
        persistent_post_weight,
        cached_tile,
        warp_sums);
  }

  ctrl.exit();
  if constexpr (Trait::kMinBlocksPerSM == 4) {
    if (threadIdx.x == 0) {
      for (uint32_t signal = gridDim.x + blockIdx.x;
           signal < params.num_controller_slots;
           signal += gridDim.x) {
        ctrl.exit_unsafe(signal);
      }
    }
  }
}

__global__ void prefill_attntp_push_controller_cleanup_kernel(
    const PushController ctrl,
    const uint32_t first_signal,
    const uint32_t num_signals) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < num_signals) {
    ctrl.exit_unsafe(first_signal + index);
  }
}

template <
    typename DType,
    uint32_t kNumGPU,
    uint32_t kHiddenSize,
    uint32_t kRowsPerSignal,
    bool kCachePeerTile,
    uint32_t kMinBlocksPerSM,
    bool kUsePDL,
    uint32_t kInternalPrecision>
struct FusedPrefillAttnTPIPCNorm : public CustomAllReduceBase {
  using Trait = FusedPrefillAttnTPIPCNormTrait<
      DType,
      kNumGPU,
      kHiddenSize,
      kRowsPerSignal,
      kCachePeerTile,
      kMinBlocksPerSM,
      kUsePDL,
      kInternalPrecision>;
  static constexpr auto kernel = fused_prefill_attntp_ipc_norm_kernel<Trait>;

  static uint64_t get_fixed_signal_offset(const uint64_t buffer_bytes) {
    // A shape-dependent signal offset can alias payload left by another graph.
    constexpr uint64_t kRowBytes = kHiddenSize * sizeof(DType);
    uint64_t max_tokens = buffer_bytes / kRowBytes;
    while (max_tokens != 0) {
      const uint64_t data_bytes = max_tokens * kRowBytes;
      const uint64_t signal_bytes =
          (max_tokens * sizeof(uint32_t) + 127) / 128 * 128;
      if (data_bytes + signal_bytes <= buffer_bytes) return data_bytes;
      --max_tokens;
    }
    return 0;
  }

  void _run(
      const tvm::ffi::Tensor partial,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const float o_norm_eps,
      const float post_norm_eps) {
    using namespace host;

    auto num_tokens_symbol = SymbolicSize{"num_tokens"};
    auto device_symbol = SymbolicDevice{};
    device_symbol.set_options<kDLCUDA>();
    TensorMatcher({num_tokens_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(partial);
    TensorMatcher({num_tokens_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual);
    TensorMatcher({kHiddenSize})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(o_norm_weight);
    TensorMatcher({kHiddenSize})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(post_norm_weight);
    TensorMatcher({num_tokens_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(output);
    TensorMatcher({num_tokens_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual_out);
    const auto device = device_symbol.unwrap();
    const auto num_tokens = static_cast<uint32_t>(num_tokens_symbol.unwrap());
    const uint64_t data_bytes =
        static_cast<uint64_t>(num_tokens) * kHiddenSize * sizeof(DType);
    const uint32_t num_tiles =
        div_ceil(num_tokens, Trait::kRowsPerSignal);
    const uint64_t signal_bytes =
        div_ceil(static_cast<uint64_t>(num_tiles) * sizeof(uint32_t), 128) *
        128;
    const uint64_t signal_offset_bytes = get_fixed_signal_offset(
        static_cast<uint64_t>(m_push_buffer_bytes));
    const uint64_t signal_capacity_bytes =
        static_cast<uint64_t>(m_push_buffer_bytes) - signal_offset_bytes;

    RuntimeCheck(m_num_gpu == kNumGPU, "PrefillAttnTP fused norm requires world size 2");
    RuntimeCheck(m_push_ctrl.has_value(), "Push controller is not initialized");
    RuntimeCheck(
        data_bytes <= signal_offset_bytes,
        "Push payload region is too small, required bytes: ",
        data_bytes,
        ", available bytes: ",
        signal_offset_bytes);
    RuntimeCheck(
        signal_bytes <= signal_capacity_bytes,
        "Push signal region is too small, required bytes: ",
        signal_bytes,
        ", available bytes: ",
        signal_capacity_bytes);
    for (const auto* pointer :
         {partial.data_ptr(),
          residual.data_ptr(),
          o_norm_weight.data_ptr(),
          post_norm_weight.data_ptr(),
          output.data_ptr(),
          residual_out.data_ptr()}) {
      RuntimeCheck(
          std::bit_cast<intptr_t>(pointer) % 16 == 0,
          "PrefillAttnTP fused norm tensors must be 16-byte aligned");
    }

    [[maybe_unused]] static const uint32_t max_kernel_blocks = [] {
      int device_id = 0;
      host::RuntimeDeviceCheck(cudaGetDevice(&device_id));
      return get_max_occupancy() *
             host::runtime::get_sm_count(device_id);
    }();
    const uint32_t num_blocks = std::min(
        {num_tiles, m_max_num_cta_push, max_kernel_blocks});
    const uint32_t num_clean =
        m_max_num_cta_push - num_blocks;
    RuntimeCheck(num_blocks > 0, "PrefillAttnTP fused norm requires non-empty input");
    RuntimeCheck(num_blocks <= m_max_num_cta_push, "Invalid push CTA count");

    FusedPrefillAttnTPIPCNormParams params{};
    for (uint32_t index = 0; index < kNumGPU; ++index) {
      params.buffer[index] = get_push_buffer(m_peer_storage[index]);
    }
    params.partial = partial.data_ptr();
    params.residual = residual.data_ptr();
    params.o_norm_weight = o_norm_weight.data_ptr();
    params.post_norm_weight = post_norm_weight.data_ptr();
    params.output = output.data_ptr();
    params.residual_out = residual_out.data_ptr();
    params.buffer_bytes = m_push_buffer_bytes;
    params.epoch_bytes =
        static_cast<uint64_t>(kNumGPU) * m_push_buffer_bytes;
    params.signal_offset_bytes = signal_offset_bytes;
    params.o_norm_eps = o_norm_eps;
    params.post_norm_eps = post_norm_eps;
    params.rank = m_rank;
    params.num_tokens = num_tokens;
    params.num_tiles = num_tiles;
    params.num_controller_slots = m_max_num_cta_push;

    LaunchKernel(num_blocks, Trait::kBlockSize, device)
        .enable_pdl(kUsePDL)(kernel, params, *m_push_ctrl);
    if constexpr (kMinBlocksPerSM != 4) {
      if (num_clean == 0) return;
      constexpr uint32_t kCleanupBlockSize = 256;
      LaunchKernel(
          div_ceil(num_clean, kCleanupBlockSize),
          kCleanupBlockSize,
          device)(
          prefill_attntp_push_controller_cleanup_kernel,
          *m_push_ctrl,
          num_blocks,
          num_clean);
    }
  }

  static void run(
      CustomAllReduceRef object,
      const tvm::ffi::Tensor partial,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const float o_norm_eps,
      const float post_norm_eps) {
    using Self = FusedPrefillAttnTPIPCNorm;
    static_cast<Self*>(object.get())
        ->_run(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            output,
            residual_out,
            o_norm_eps,
            post_norm_eps);
  }

  static uint32_t get_max_occupancy() {
    return host::runtime::get_blocks_per_sm(kernel, Trait::kBlockSize);
  }
};

struct FusedPrefillAttnTPSourcePushNormParams {
  void* buffer[device::distributed::kMaxNumGPU];
  const void* partial;
  const void* residual;
  const void* o_norm_weight;
  const void* post_norm_weight;
  void* output;
  void* residual_out;
  const int32_t* actual_rows;
  const int32_t* owner_start;
  uint64_t buffer_bytes;
  uint64_t epoch_bytes;
  uint64_t signal_offset_bytes;
  uint64_t gather_payload_offset_bytes;
  uint64_t gather_signal_offset_bytes;
  float o_norm_eps;
  float post_norm_eps;
  uint32_t rank;
  uint32_t capacity;
  uint32_t owner_capacity;
  uint32_t max_tiles_per_owner;
  uint32_t num_global_tiles;
};

template <
    typename DType_,
    uint32_t kNumGPU_,
    uint32_t kHiddenSize_,
    uint32_t kOutputMode_,
    uint32_t kRowsPerTile_,
    uint32_t kInternalPrecision_,
    uint32_t kBlockSize_,
    uint32_t kSignalBackoff_>
struct FusedPrefillAttnTPSourcePushNormTrait
    : public sglang::jit_kernel::attntp::NormPipelineTrait<
          DType_,
          kHiddenSize_,
          kInternalPrecision_,
          kBlockSize_> {
  using Base = sglang::jit_kernel::attntp::NormPipelineTrait<
      DType_,
      kHiddenSize_,
      kInternalPrecision_,
      kBlockSize_>;
  using DType = typename Base::DType;
  static constexpr uint32_t kNumGPU = kNumGPU_;
  static constexpr uint32_t kRowsPerTile = kRowsPerTile_;
  static constexpr uint32_t kSignalBackoff = kSignalBackoff_;
  static constexpr auto kOutputMode =
      static_cast<sglang::jit_kernel::attntp::OutputMode>(kOutputMode_);

  static_assert(
      sglang::jit_kernel::attntp::is_supported_attn_tp_size(kNumGPU));
  static_assert(kNumGPU >= 2);
  static_assert(
      kOutputMode ==
              sglang::jit_kernel::attntp::OutputMode::kReplicated ||
          kOutputMode ==
              sglang::jit_kernel::attntp::OutputMode::kSingleContributor ||
          kOutputMode ==
              sglang::jit_kernel::attntp::OutputMode::kTokenScattered);
  static_assert(
      kRowsPerTile == 1 || kRowsPerTile == 2 ||
      kRowsPerTile == 4 || kRowsPerTile == 8);
  static_assert(
      kSignalBackoff == 32 || kSignalBackoff == 64 ||
      kSignalBackoff == 128 || kSignalBackoff == 256);
};

template <typename Trait>
__global__ void fused_prefill_attntp_source_push_norm_kernel(
    const FusedPrefillAttnTPSourcePushNormParams __grid_constant__ params,
    const PushController __grid_constant__ ctrl) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kRowBytes = kHiddenSize * sizeof(DType);
  constexpr uint32_t kVectorsPerThread = Trait::kVectorsPerThread;
  constexpr uint32_t kElementsPerVector = Trait::kElementsPerVector;

  const int32_t actual_rows_signed = *params.actual_rows;
  const int32_t owner_start_signed = *params.owner_start;
  if (actual_rows_signed < 0 ||
      static_cast<uint32_t>(actual_rows_signed) > params.capacity ||
      owner_start_signed < 0 ||
      static_cast<uint32_t>(owner_start_signed) >= Trait::kNumGPU) {
    if (threadIdx.x == 0) asm volatile("trap;");
    return;
  }
  const uint32_t actual_rows = static_cast<uint32_t>(actual_rows_signed);
  const uint32_t owner_start = static_cast<uint32_t>(owner_start_signed);
  const uint32_t signal_epoch = ctrl.epoch() + 1;
  const uint64_t epoch_offset =
      static_cast<uint64_t>(signal_epoch - 1) * params.epoch_bytes;

  __shared__ float warp_sums[Trait::kNumWarps];
  BF16Storage o_weight_values[kVectorsPerThread];
  BF16Storage post_weight_values[kVectorsPerThread];
#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
    const uint32_t storage_index =
        threadIdx.x * kVectorsPerThread + vector;
    o_weight_values[vector].load(params.o_norm_weight, storage_index);
    post_weight_values[vector].load(
        params.post_norm_weight, storage_index);
  }

  for (uint32_t global_tile = blockIdx.x;
       global_tile < params.num_global_tiles;
       global_tile += gridDim.x) {
    uint32_t owner = owner_start;
    uint32_t local_tile = global_tile;
    sglang::jit_kernel::attntp::BalancedRowRange owner_range{
        0, actual_rows};
    if constexpr (
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kTokenScattered) {
      const uint32_t owner_slot =
          global_tile / params.max_tiles_per_owner;
      local_tile = global_tile % params.max_tiles_per_owner;
      owner = (owner_start + owner_slot) % Trait::kNumGPU;
      owner_range = sglang::jit_kernel::attntp::balanced_row_range(
          actual_rows, owner, Trait::kNumGPU, owner_start);
    }

    const uint32_t local_row_start =
        local_tile * Trait::kRowsPerTile;
    if (local_row_start >= owner_range.count) continue;
    const uint32_t rows_in_tile =
        owner_range.count - local_row_start < Trait::kRowsPerTile
            ? owner_range.count - local_row_start
            : Trait::kRowsPerTile;

    void* send_slot = byte_offset<void>(
        params.buffer[owner],
        epoch_offset +
            static_cast<uint64_t>(params.rank) * params.buffer_bytes);
    auto* send_signal = byte_offset<uint32_t>(
        send_slot,
        params.signal_offset_bytes +
            static_cast<uint64_t>(local_tile) * sizeof(uint32_t));
    if (threadIdx.x == 0) {
      while (load_acquire_system(send_signal) != 0) {
        __nanosleep(Trait::kSignalBackoff);
      }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t row_in_tile = 0;
         row_in_tile < Trait::kRowsPerTile;
         ++row_in_tile) {
      if (row_in_tile >= rows_in_tile) break;
      const uint32_t local_row = local_row_start + row_in_tile;
      const uint32_t global_row = owner_range.offset + local_row;
      const auto* partial_row = static_cast<const DType*>(params.partial) +
                                static_cast<uint64_t>(global_row) *
                                    kHiddenSize;
      auto* send_row = byte_offset<void>(
          send_slot, static_cast<uint64_t>(local_row) * kRowBytes);
#pragma unroll
      for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
        BF16Storage values;
        const uint32_t storage_index =
            threadIdx.x * kVectorsPerThread + vector;
        values.load(partial_row, storage_index);
        store_volatile_16b(
            byte_offset<void>(
                send_row,
                static_cast<uint64_t>(storage_index) *
                    sizeof(BF16Storage)),
            values);
      }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      store_release_system(send_signal, signal_epoch);
    }
    __syncthreads();

    if (params.rank != owner) continue;

    void* owner_epoch_base = byte_offset<void>(
        params.buffer[params.rank], epoch_offset);
    if (threadIdx.x == 0) {
#pragma unroll
      for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
        const auto* source_signal = byte_offset<uint32_t>(
            owner_epoch_base,
            static_cast<uint64_t>(source) * params.buffer_bytes +
                params.signal_offset_bytes +
                static_cast<uint64_t>(local_tile) * sizeof(uint32_t));
        while (load_acquire_system(source_signal) != signal_epoch) {
          __nanosleep(Trait::kSignalBackoff);
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t row_in_tile = 0;
         row_in_tile < Trait::kRowsPerTile;
         ++row_in_tile) {
      if (row_in_tile >= rows_in_tile) break;
      const uint32_t local_row = local_row_start + row_in_tile;
      const uint32_t global_row = owner_range.offset + local_row;
      const auto* residual_row =
          static_cast<const float*>(params.residual) +
          static_cast<uint64_t>(global_row) * kHiddenSize;
      auto* output_row = static_cast<DType*>(params.output) +
                         static_cast<uint64_t>(local_row) * kHiddenSize;
      auto* residual_out_row = static_cast<float*>(params.residual_out) +
                               static_cast<uint64_t>(local_row) *
                                   kHiddenSize;
      if constexpr (
          Trait::kInternalPrecision ==
          sglang::jit_kernel::attntp::NormInternalPrecision::kFullFP32) {
        float activation_values[kVectorsPerThread][kElementsPerVector];
#pragma unroll
        for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
#pragma unroll
          for (uint32_t index = 0; index < kElementsPerVector; ++index) {
            activation_values[vector][index] = 0.0f;
          }
#pragma unroll
          for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
            const auto* source_row = byte_offset<void>(
                owner_epoch_base,
                static_cast<uint64_t>(source) * params.buffer_bytes +
                    static_cast<uint64_t>(local_row) * kRowBytes);
            const uint32_t storage_index =
                threadIdx.x * kVectorsPerThread + vector;
            const BF16Storage source_values =
                load_volatile_16b<BF16Storage>(byte_offset<void>(
                    source_row,
                    static_cast<uint64_t>(storage_index) *
                        sizeof(BF16Storage)));
#pragma unroll
            for (uint32_t index = 0; index < kElementsPerVector; ++index) {
              activation_values[vector][index] +=
                  device::cast<float>(source_values[index]);
            }
          }
        }
        sglang::jit_kernel::attntp::apply_welm_norm_pipeline_full_fp32<
            Trait>(
            activation_values,
            residual_row,
            o_weight_values,
            post_weight_values,
            output_row,
            residual_out_row,
            params.o_norm_eps,
            params.post_norm_eps,
            warp_sums);
      } else {
        BF16Storage reduced_values[kVectorsPerThread];
#pragma unroll
        for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
          float accumulators[kElementsPerVector] = {};
#pragma unroll
          for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
            const auto* source_row = byte_offset<void>(
                owner_epoch_base,
                static_cast<uint64_t>(source) * params.buffer_bytes +
                    static_cast<uint64_t>(local_row) * kRowBytes);
            const uint32_t storage_index =
                threadIdx.x * kVectorsPerThread + vector;
            const BF16Storage source_values =
                load_volatile_16b<BF16Storage>(byte_offset<void>(
                    source_row,
                    static_cast<uint64_t>(storage_index) *
                        sizeof(BF16Storage)));
#pragma unroll
            for (uint32_t index = 0; index < kElementsPerVector; ++index) {
              accumulators[index] +=
                  device::cast<float>(source_values[index]);
            }
          }
#pragma unroll
          for (uint32_t index = 0; index < kElementsPerVector; ++index) {
            reduced_values[vector][index] =
                device::cast<DType>(accumulators[index]);
          }
        }
        sglang::jit_kernel::attntp::apply_welm_norm_pipeline<Trait>(
            reduced_values,
            residual_row,
            o_weight_values,
            post_weight_values,
            output_row,
            residual_out_row,
            params.o_norm_eps,
            params.post_norm_eps,
            warp_sums);
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
      for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
        auto* source_signal = byte_offset<uint32_t>(
            owner_epoch_base,
            static_cast<uint64_t>(source) * params.buffer_bytes +
                params.signal_offset_bytes +
                static_cast<uint64_t>(local_tile) * sizeof(uint32_t));
        store_release_system(source_signal, 0);
      }
    }
    __syncthreads();
  }

  ctrl.exit();
}

template <typename Trait>
__global__ void fused_prefill_attntp_replicated_source_push_norm_kernel(
    const FusedPrefillAttnTPSourcePushNormParams __grid_constant__ params,
    const PushController __grid_constant__ ctrl) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;
  using FP32Storage = typename Trait::FP32Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kRowBytes = kHiddenSize * sizeof(DType);
  constexpr uint32_t kGatherRowBytes =
      Trait::kInternalPrecision ==
              sglang::jit_kernel::attntp::NormInternalPrecision::kFullFP32
          ? kHiddenSize * sizeof(float)
          : kRowBytes;
  constexpr uint32_t kVectorsPerThread = Trait::kVectorsPerThread;
  constexpr uint32_t kElementsPerVector = Trait::kElementsPerVector;

  const int32_t actual_rows_signed = *params.actual_rows;
  const int32_t owner_start_signed = *params.owner_start;
  if (actual_rows_signed < 0 ||
      static_cast<uint32_t>(actual_rows_signed) > params.capacity ||
      owner_start_signed < 0 ||
      static_cast<uint32_t>(owner_start_signed) >= Trait::kNumGPU) {
    if (threadIdx.x == 0) asm volatile("trap;");
    return;
  }
  const uint32_t actual_rows = static_cast<uint32_t>(actual_rows_signed);
  const uint32_t owner_start = static_cast<uint32_t>(owner_start_signed);
  const uint32_t signal_epoch = ctrl.epoch() + 1;
  const uint64_t epoch_offset =
      static_cast<uint64_t>(signal_epoch - 1) * params.epoch_bytes;

  __shared__ float warp_sums[Trait::kNumWarps];
  BF16Storage o_weight_values[kVectorsPerThread];
  BF16Storage post_weight_values[kVectorsPerThread];
#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
    const uint32_t storage_index =
        threadIdx.x * kVectorsPerThread + vector;
    o_weight_values[vector].load(params.o_norm_weight, storage_index);
    post_weight_values[vector].load(
        params.post_norm_weight, storage_index);
  }

  for (uint32_t global_tile = blockIdx.x;
       global_tile < params.num_global_tiles;
       global_tile += gridDim.x) {
    const uint32_t owner_slot =
        global_tile / params.max_tiles_per_owner;
    const uint32_t local_tile =
        global_tile % params.max_tiles_per_owner;
    const uint32_t owner =
        (owner_start + owner_slot) % Trait::kNumGPU;
    const auto owner_range =
        sglang::jit_kernel::attntp::balanced_row_range(
            actual_rows, owner, Trait::kNumGPU, owner_start);
    const uint32_t local_row_start =
        local_tile * Trait::kRowsPerTile;
    if (local_row_start >= owner_range.count) continue;
    const uint32_t rows_in_tile =
        owner_range.count - local_row_start < Trait::kRowsPerTile
            ? owner_range.count - local_row_start
            : Trait::kRowsPerTile;

    void* send_slot = byte_offset<void>(
        params.buffer[owner],
        epoch_offset +
            static_cast<uint64_t>(params.rank) * params.buffer_bytes);
    auto* send_signal = byte_offset<uint32_t>(
        send_slot,
        params.signal_offset_bytes +
            static_cast<uint64_t>(local_tile) * sizeof(uint32_t));
    if (threadIdx.x == 0) {
      while (load_acquire_system(send_signal) != 0) {
        __nanosleep(Trait::kSignalBackoff);
      }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t row_in_tile = 0;
         row_in_tile < Trait::kRowsPerTile;
         ++row_in_tile) {
      if (row_in_tile >= rows_in_tile) break;
      const uint32_t local_row = local_row_start + row_in_tile;
      const uint32_t global_row = owner_range.offset + local_row;
      const auto* partial_row = static_cast<const DType*>(params.partial) +
                                static_cast<uint64_t>(global_row) *
                                    kHiddenSize;
      auto* send_row = byte_offset<void>(
          send_slot, static_cast<uint64_t>(local_row) * kRowBytes);
#pragma unroll
      for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
        BF16Storage values;
        const uint32_t storage_index =
            threadIdx.x * kVectorsPerThread + vector;
        values.load(partial_row, storage_index);
        store_volatile_16b(
            byte_offset<void>(
                send_row,
                static_cast<uint64_t>(storage_index) *
                    sizeof(BF16Storage)),
            values);
      }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      store_release_system(send_signal, signal_epoch);
    }
    __syncthreads();

    if (params.rank == owner) {
      void* owner_epoch_base = byte_offset<void>(
          params.buffer[params.rank], epoch_offset);
      if (threadIdx.x == 0) {
#pragma unroll
        for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
          const auto* source_signal = byte_offset<uint32_t>(
              owner_epoch_base,
              static_cast<uint64_t>(source) * params.buffer_bytes +
                  params.signal_offset_bytes +
                  static_cast<uint64_t>(local_tile) *
                      sizeof(uint32_t));
          while (load_acquire_system(source_signal) != signal_epoch) {
            __nanosleep(Trait::kSignalBackoff);
          }
        }
#pragma unroll
        for (uint32_t destination = 0;
             destination < Trait::kNumGPU;
             ++destination) {
          const auto* gather_signal = byte_offset<uint32_t>(
              params.buffer[destination],
              epoch_offset +
                  static_cast<uint64_t>(owner) * params.buffer_bytes +
                  params.gather_signal_offset_bytes +
                  static_cast<uint64_t>(local_tile) *
                      sizeof(uint32_t));
          while (load_acquire_system(gather_signal) != 0) {
            __nanosleep(Trait::kSignalBackoff);
          }
        }
      }
      __syncthreads();

#pragma unroll
      for (uint32_t row_in_tile = 0;
           row_in_tile < Trait::kRowsPerTile;
           ++row_in_tile) {
        if (row_in_tile >= rows_in_tile) break;
        const uint32_t local_row = local_row_start + row_in_tile;
#pragma unroll
        for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
          float accumulators[kElementsPerVector] = {};
          const uint32_t storage_index =
              threadIdx.x * kVectorsPerThread + vector;
#pragma unroll
          for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
            const auto* source_row = byte_offset<void>(
                owner_epoch_base,
                static_cast<uint64_t>(source) * params.buffer_bytes +
                    static_cast<uint64_t>(local_row) * kRowBytes);
            const BF16Storage source_values =
                load_volatile_16b<BF16Storage>(byte_offset<void>(
                    source_row,
                    static_cast<uint64_t>(storage_index) *
                        sizeof(BF16Storage)));
#pragma unroll
            for (uint32_t index = 0;
                 index < kElementsPerVector;
                 ++index) {
              accumulators[index] +=
                  device::cast<float>(source_values[index]);
            }
          }
          if constexpr (
              Trait::kInternalPrecision ==
              sglang::jit_kernel::attntp::NormInternalPrecision::
                  kFullFP32) {
            FP32Storage reduced_values[2];
#pragma unroll
            for (uint32_t index = 0; index < kElementsPerVector; ++index) {
              reduced_values[index / (kElementsPerVector / 2)]
                            [index % (kElementsPerVector / 2)] =
                  accumulators[index];
            }
#pragma unroll
            for (uint32_t destination = 0;
                 destination < Trait::kNumGPU;
                 ++destination) {
              auto* gather_row = byte_offset<void>(
                  params.buffer[destination],
                  epoch_offset +
                      static_cast<uint64_t>(owner) * params.buffer_bytes +
                      params.gather_payload_offset_bytes +
                      static_cast<uint64_t>(local_row) * kGatherRowBytes);
#pragma unroll
              for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t fp32_storage_index =
                    threadIdx.x * (2 * kVectorsPerThread) +
                    vector * 2 + half;
                store_volatile_16b(
                    byte_offset<void>(
                        gather_row,
                        static_cast<uint64_t>(fp32_storage_index) *
                            sizeof(FP32Storage)),
                    reduced_values[half]);
              }
            }
          } else {
            BF16Storage reduced_values;
#pragma unroll
            for (uint32_t index = 0; index < kElementsPerVector; ++index) {
              reduced_values[index] =
                  device::cast<DType>(accumulators[index]);
            }
#pragma unroll
            for (uint32_t destination = 0;
                 destination < Trait::kNumGPU;
                 ++destination) {
              auto* gather_row = byte_offset<void>(
                  params.buffer[destination],
                  epoch_offset +
                      static_cast<uint64_t>(owner) * params.buffer_bytes +
                      params.gather_payload_offset_bytes +
                      static_cast<uint64_t>(local_row) * kGatherRowBytes);
              store_volatile_16b(
                  byte_offset<void>(
                      gather_row,
                      static_cast<uint64_t>(storage_index) *
                          sizeof(BF16Storage)),
                  reduced_values);
            }
          }
        }
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
#pragma unroll
        for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
          auto* source_signal = byte_offset<uint32_t>(
              owner_epoch_base,
              static_cast<uint64_t>(source) * params.buffer_bytes +
                  params.signal_offset_bytes +
                  static_cast<uint64_t>(local_tile) *
                      sizeof(uint32_t));
          store_release_system(source_signal, 0);
        }
#pragma unroll
        for (uint32_t destination = 0;
             destination < Trait::kNumGPU;
             ++destination) {
          auto* gather_signal = byte_offset<uint32_t>(
              params.buffer[destination],
              epoch_offset +
                  static_cast<uint64_t>(owner) * params.buffer_bytes +
                  params.gather_signal_offset_bytes +
                  static_cast<uint64_t>(local_tile) *
                      sizeof(uint32_t));
          store_release_system(gather_signal, signal_epoch);
        }
      }
      __syncthreads();
    }

    void* local_gather_slot = byte_offset<void>(
        params.buffer[params.rank],
        epoch_offset +
            static_cast<uint64_t>(owner) * params.buffer_bytes);
    auto* local_gather_signal = byte_offset<uint32_t>(
        local_gather_slot,
        params.gather_signal_offset_bytes +
            static_cast<uint64_t>(local_tile) * sizeof(uint32_t));
    if (threadIdx.x == 0) {
      while (load_acquire_system(local_gather_signal) != signal_epoch) {
        __nanosleep(Trait::kSignalBackoff);
      }
    }
    __syncthreads();

#pragma unroll
    for (uint32_t row_in_tile = 0;
         row_in_tile < Trait::kRowsPerTile;
         ++row_in_tile) {
      if (row_in_tile >= rows_in_tile) break;
      const uint32_t local_row = local_row_start + row_in_tile;
      const uint32_t global_row = owner_range.offset + local_row;
      const auto* gather_row = byte_offset<void>(
          local_gather_slot,
          params.gather_payload_offset_bytes +
              static_cast<uint64_t>(local_row) * kGatherRowBytes);
      const auto* residual_row =
          static_cast<const float*>(params.residual) +
          static_cast<uint64_t>(global_row) * kHiddenSize;
      auto* output_row = static_cast<DType*>(params.output) +
                         static_cast<uint64_t>(global_row) * kHiddenSize;
      auto* residual_out_row = static_cast<float*>(params.residual_out) +
                               static_cast<uint64_t>(global_row) *
                                   kHiddenSize;
      if constexpr (
          Trait::kInternalPrecision ==
          sglang::jit_kernel::attntp::NormInternalPrecision::kFullFP32) {
        float activation_values[kVectorsPerThread][kElementsPerVector];
#pragma unroll
        for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
#pragma unroll
          for (uint32_t half = 0; half < 2; ++half) {
            const uint32_t storage_index =
                threadIdx.x * (2 * kVectorsPerThread) + vector * 2 + half;
            const FP32Storage values =
                load_volatile_16b<FP32Storage>(byte_offset<void>(
                    gather_row,
                    static_cast<uint64_t>(storage_index) *
                        sizeof(FP32Storage)));
#pragma unroll
            for (uint32_t index = 0;
                 index < kElementsPerVector / 2;
                 ++index) {
              activation_values[vector]
                               [half * (kElementsPerVector / 2) + index] =
                  values[index];
            }
          }
        }
        sglang::jit_kernel::attntp::apply_welm_norm_pipeline_full_fp32<
            Trait>(
            activation_values,
            residual_row,
            o_weight_values,
            post_weight_values,
            output_row,
            residual_out_row,
            params.o_norm_eps,
            params.post_norm_eps,
            warp_sums);
      } else {
        BF16Storage reduced_values[kVectorsPerThread];
#pragma unroll
        for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
          const uint32_t storage_index =
              threadIdx.x * kVectorsPerThread + vector;
          reduced_values[vector] =
              load_volatile_16b<BF16Storage>(byte_offset<void>(
                  gather_row,
                  static_cast<uint64_t>(storage_index) *
                      sizeof(BF16Storage)));
        }
        sglang::jit_kernel::attntp::apply_welm_norm_pipeline<Trait>(
            reduced_values,
            residual_row,
            o_weight_values,
            post_weight_values,
            output_row,
            residual_out_row,
            params.o_norm_eps,
            params.post_norm_eps,
            warp_sums);
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      store_release_system(local_gather_signal, 0);
    }
    __syncthreads();
  }

  ctrl.exit();
}

template <
    typename DType,
    uint32_t kNumGPU,
    uint32_t kHiddenSize,
    uint32_t kOutputMode,
    uint32_t kRowsPerTile,
    uint32_t kInternalPrecision,
    uint32_t kBlockSize,
    uint32_t kSignalBackoff>
struct FusedPrefillAttnTPSourcePushNorm : public CustomAllReduceBase {
  using Trait = FusedPrefillAttnTPSourcePushNormTrait<
      DType,
      kNumGPU,
      kHiddenSize,
      kOutputMode,
      kRowsPerTile,
      kInternalPrecision,
      kBlockSize,
      kSignalBackoff>;
  static constexpr auto kernel =
      Trait::kOutputMode ==
              sglang::jit_kernel::attntp::OutputMode::kReplicated
          ? fused_prefill_attntp_replicated_source_push_norm_kernel<Trait>
          : fused_prefill_attntp_source_push_norm_kernel<Trait>;

  void _run(
      const tvm::ffi::Tensor partial,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const tvm::ffi::Tensor actual_rows,
      const tvm::ffi::Tensor owner_start,
      const int64_t arena_capacity_i64,
      const int64_t max_blocks_per_sm_i64,
      const float o_norm_eps,
      const float post_norm_eps) {
    using namespace host;

    auto capacity_symbol = SymbolicSize{"capacity"};
    auto output_capacity_symbol = SymbolicSize{"output_capacity"};
    auto device_symbol = SymbolicDevice{};
    device_symbol.set_options<kDLCUDA>();
    TensorMatcher({capacity_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(partial);
    TensorMatcher({capacity_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual);
    TensorMatcher({kHiddenSize})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(o_norm_weight)
        .verify(post_norm_weight);
    TensorMatcher({output_capacity_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(output);
    TensorMatcher({output_capacity_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual_out);
    TensorMatcher({1})
        .with_dtype<int32_t>()
        .with_device(device_symbol)
        .verify(actual_rows)
        .verify(owner_start);

    const auto capacity_i64 = capacity_symbol.unwrap();
    const auto output_capacity_i64 = output_capacity_symbol.unwrap();
    RuntimeCheck(
        capacity_i64 > 0 && capacity_i64 <= UINT32_MAX,
        "Source-push capacity must fit uint32");
    RuntimeCheck(
        arena_capacity_i64 >= capacity_i64 &&
            arena_capacity_i64 <= UINT32_MAX,
        "Source-push arena capacity must cover input capacity");
    RuntimeCheck(
        max_blocks_per_sm_i64 >= 0,
        "Source-push blocks per SM must be non-negative");
    const uint32_t capacity = static_cast<uint32_t>(capacity_i64);
    const uint32_t arena_capacity =
        static_cast<uint32_t>(arena_capacity_i64);
    const bool is_single_contributor =
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kSingleContributor;
    const bool is_replicated =
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kReplicated;
    const uint32_t owner_capacity = is_single_contributor
        ? arena_capacity
        : div_ceil(arena_capacity, kNumGPU);
    const uint32_t current_owner_capacity = is_single_contributor
        ? capacity
        : div_ceil(capacity, kNumGPU);
    const uint32_t expected_output_capacity =
        is_replicated || is_single_contributor
        ? capacity
        : current_owner_capacity;
    RuntimeCheck(
        output_capacity_i64 == expected_output_capacity,
        "Source-push output capacity mismatch, expected ",
        expected_output_capacity,
        ", got ",
        output_capacity_i64);
    const uint32_t max_tiles_per_owner =
        div_ceil(current_owner_capacity, Trait::kRowsPerTile);
    const uint32_t num_global_tiles =
        is_single_contributor ? max_tiles_per_owner
                              : kNumGPU * max_tiles_per_owner;
    const uint64_t signal_offset_bytes =
        static_cast<uint64_t>(owner_capacity) * kHiddenSize *
        sizeof(DType);
    const uint32_t arena_signal_slots = owner_capacity;
    const uint64_t signal_bytes =
        div_ceil(
            static_cast<uint64_t>(arena_signal_slots) * sizeof(uint32_t),
            static_cast<uint64_t>(128)) *
        128;
    const uint64_t gather_payload_offset_bytes =
        signal_offset_bytes + signal_bytes;
    constexpr uint64_t kGatherElementBytes =
        Trait::kInternalPrecision ==
                sglang::jit_kernel::attntp::NormInternalPrecision::kFullFP32
            ? sizeof(float)
            : sizeof(DType);
    const uint64_t gather_signal_offset_bytes =
        gather_payload_offset_bytes +
        static_cast<uint64_t>(owner_capacity) * kHiddenSize *
            kGatherElementBytes;
    const uint64_t required_buffer_bytes =
        is_replicated ? gather_signal_offset_bytes + signal_bytes
                      : signal_offset_bytes + signal_bytes;

    RuntimeCheck(m_num_gpu == kNumGPU, "Source-push world size mismatch");
    RuntimeCheck(
        m_push_ctrl.has_value(), "Source-push controller is not initialized");
    RuntimeCheck(
        required_buffer_bytes <=
            static_cast<uint64_t>(m_push_buffer_bytes),
        "Source-push buffer is too small, required ",
        required_buffer_bytes,
        ", available ",
        m_push_buffer_bytes);
    for (const auto* pointer :
         {partial.data_ptr(),
          residual.data_ptr(),
          o_norm_weight.data_ptr(),
          post_norm_weight.data_ptr(),
          output.data_ptr(),
          residual_out.data_ptr()}) {
      RuntimeCheck(
          std::bit_cast<intptr_t>(pointer) % 16 == 0,
          "Source-push tensors must be 16-byte aligned");
    }

    const auto device = device_symbol.unwrap();
    int device_id = 0;
    host::RuntimeDeviceCheck(cudaGetDevice(&device_id));
    const uint32_t occupancy = get_max_occupancy();
    const uint32_t selected_blocks_per_sm = max_blocks_per_sm_i64 == 0
                                                ? occupancy
                                                : static_cast<uint32_t>(max_blocks_per_sm_i64);
    RuntimeCheck(
        selected_blocks_per_sm > 0 && selected_blocks_per_sm <= occupancy,
        "Source-push blocks per SM exceeds kernel occupancy");
    const uint32_t max_kernel_blocks =
        selected_blocks_per_sm * host::runtime::get_sm_count(device_id);
    const uint32_t num_blocks = std::min(
        {num_global_tiles, m_max_num_cta_push, max_kernel_blocks});
    RuntimeCheck(num_blocks > 0, "Source-push requires at least one CTA");

    FusedPrefillAttnTPSourcePushNormParams params{};
    for (uint32_t index = 0; index < kNumGPU; ++index) {
      params.buffer[index] = get_push_buffer(m_peer_storage[index]);
    }
    params.partial = partial.data_ptr();
    params.residual = residual.data_ptr();
    params.o_norm_weight = o_norm_weight.data_ptr();
    params.post_norm_weight = post_norm_weight.data_ptr();
    params.output = output.data_ptr();
    params.residual_out = residual_out.data_ptr();
    params.actual_rows = static_cast<const int32_t*>(actual_rows.data_ptr());
    params.owner_start = static_cast<const int32_t*>(owner_start.data_ptr());
    params.buffer_bytes = m_push_buffer_bytes;
    params.epoch_bytes =
        static_cast<uint64_t>(kNumGPU) * m_push_buffer_bytes;
    params.signal_offset_bytes = signal_offset_bytes;
    params.gather_payload_offset_bytes = gather_payload_offset_bytes;
    params.gather_signal_offset_bytes = gather_signal_offset_bytes;
    params.o_norm_eps = o_norm_eps;
    params.post_norm_eps = post_norm_eps;
    params.rank = m_rank;
    params.capacity = capacity;
    params.owner_capacity = owner_capacity;
    params.max_tiles_per_owner = max_tiles_per_owner;
    params.num_global_tiles = num_global_tiles;

    LaunchKernel(num_blocks, Trait::kBlockSize, device)(
        kernel, params, *m_push_ctrl);
  }

  static void run(
      CustomAllReduceRef object,
      const tvm::ffi::Tensor partial,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const tvm::ffi::Tensor actual_rows,
      const tvm::ffi::Tensor owner_start,
      const int64_t arena_capacity,
      const int64_t max_blocks_per_sm,
      const float o_norm_eps,
      const float post_norm_eps) {
    using Self = FusedPrefillAttnTPSourcePushNorm;
    static_cast<Self*>(object.get())
        ->_run(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            output,
            residual_out,
            actual_rows,
            owner_start,
            arena_capacity,
            max_blocks_per_sm,
            o_norm_eps,
            post_norm_eps);
  }

  static uint32_t get_max_occupancy() {
    return host::runtime::get_blocks_per_sm(kernel, Trait::kBlockSize);
  }
};

struct FusedPrefillAttnTPOwnerPullNormParams {
  const void* residual;
  const void* o_norm_weight;
  const void* post_norm_weight;
  void* output;
  void* residual_out;
  const int32_t* actual_rows;
  const int32_t* owner_start;
  float o_norm_eps;
  float post_norm_eps;
  uint32_t rank;
  uint32_t capacity;
  uint32_t owner_capacity;
  uint32_t num_global_tiles;
};

template <
    typename DType_,
    uint32_t kNumGPU_,
    uint32_t kHiddenSize_,
    uint32_t kOutputMode_,
    uint32_t kInternalPrecision_,
    uint32_t kBlockSize_>
struct FusedPrefillAttnTPOwnerPullNormTrait
    : public sglang::jit_kernel::attntp::NormPipelineTrait<
          DType_,
          kHiddenSize_,
          kInternalPrecision_,
          kBlockSize_> {
  using Base = sglang::jit_kernel::attntp::NormPipelineTrait<
      DType_,
      kHiddenSize_,
      kInternalPrecision_,
      kBlockSize_>;
  using DType = typename Base::DType;
  static constexpr uint32_t kNumGPU = kNumGPU_;
  static constexpr auto kOutputMode =
      static_cast<sglang::jit_kernel::attntp::OutputMode>(kOutputMode_);

  static_assert(
      sglang::jit_kernel::attntp::is_supported_attn_tp_size(kNumGPU));
  static_assert(kNumGPU >= 2);
};

template <typename Trait>
__global__ void fused_prefill_attntp_owner_pull_norm_kernel(
    const AllReduceData* __restrict__ data,
    const FusedPrefillAttnTPOwnerPullNormParams __grid_constant__ params,
    const PullController __grid_constant__ ctrl) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kVectorsPerThread = Trait::kVectorsPerThread;
  constexpr uint32_t kElementsPerVector = Trait::kElementsPerVector;

  const int32_t actual_rows_signed = *params.actual_rows;
  const int32_t owner_start_signed = *params.owner_start;
  if (actual_rows_signed < 0 ||
      static_cast<uint32_t>(actual_rows_signed) > params.capacity ||
      owner_start_signed < 0 ||
      static_cast<uint32_t>(owner_start_signed) >= Trait::kNumGPU) {
    if (threadIdx.x == 0) asm volatile("trap;");
    return;
  }
  const uint32_t actual_rows = static_cast<uint32_t>(actual_rows_signed);
  const uint32_t owner_start = static_cast<uint32_t>(owner_start_signed);

  const DType* partials[Trait::kNumGPU];
#pragma unroll
  for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
    partials[source] = static_cast<const DType*>(data->input[source]);
  }
  __shared__ float warp_sums[Trait::kNumWarps];
  BF16Storage o_weight_values[kVectorsPerThread];
  BF16Storage post_weight_values[kVectorsPerThread];
#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
    const uint32_t storage_index =
        threadIdx.x * kVectorsPerThread + vector;
    o_weight_values[vector].load(params.o_norm_weight, storage_index);
    post_weight_values[vector].load(
        params.post_norm_weight, storage_index);
  }

  ctrl.sync</*kFence=*/false, /*kStart=*/true>(
      params.rank, Trait::kNumGPU);

  for (uint32_t global_tile = blockIdx.x;
       global_tile < params.num_global_tiles;
       global_tile += gridDim.x) {
    uint32_t global_row = global_tile;
    uint32_t output_row_index = global_tile;
    if constexpr (
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kSingleContributor) {
      if (params.rank != owner_start || global_tile >= actual_rows) continue;
    } else if constexpr (
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kTokenScattered) {
      const uint32_t owner_slot = global_tile / params.owner_capacity;
      const uint32_t local_row = global_tile % params.owner_capacity;
      const uint32_t owner =
          (owner_start + owner_slot) % Trait::kNumGPU;
      const auto owner_range =
          sglang::jit_kernel::attntp::balanced_row_range(
              actual_rows, owner, Trait::kNumGPU, owner_start);
      if (params.rank != owner || local_row >= owner_range.count) continue;
      global_row = owner_range.offset + local_row;
      output_row_index = local_row;
    } else {
      if (global_tile >= actual_rows) continue;
    }

    const auto* residual_row =
        static_cast<const float*>(params.residual) +
        static_cast<uint64_t>(global_row) * kHiddenSize;
    auto* output_row = static_cast<DType*>(params.output) +
                       static_cast<uint64_t>(output_row_index) *
                           kHiddenSize;
    auto* residual_out_row = static_cast<float*>(params.residual_out) +
                             static_cast<uint64_t>(output_row_index) *
                                 kHiddenSize;
    if constexpr (
        Trait::kInternalPrecision ==
        sglang::jit_kernel::attntp::NormInternalPrecision::kFullFP32) {
      float activation_values[kVectorsPerThread][kElementsPerVector];
#pragma unroll
      for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
#pragma unroll
        for (uint32_t index = 0; index < kElementsPerVector; ++index) {
          activation_values[vector][index] = 0.0f;
        }
        const uint32_t storage_index =
            threadIdx.x * kVectorsPerThread + vector;
#pragma unroll
        for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
          BF16Storage source_values;
          source_values.load(
              partials[source] +
                  static_cast<uint64_t>(global_row) * kHiddenSize,
              storage_index);
#pragma unroll
          for (uint32_t index = 0; index < kElementsPerVector; ++index) {
            activation_values[vector][index] +=
                device::cast<float>(source_values[index]);
          }
        }
      }
      sglang::jit_kernel::attntp::apply_welm_norm_pipeline_full_fp32<
          Trait>(
          activation_values,
          residual_row,
          o_weight_values,
          post_weight_values,
          output_row,
          residual_out_row,
          params.o_norm_eps,
          params.post_norm_eps,
          warp_sums);
    } else {
      BF16Storage reduced_values[kVectorsPerThread];
#pragma unroll
      for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
        float accumulators[kElementsPerVector] = {};
        const uint32_t storage_index =
            threadIdx.x * kVectorsPerThread + vector;
#pragma unroll
        for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
          BF16Storage source_values;
          source_values.load(
              partials[source] +
                  static_cast<uint64_t>(global_row) * kHiddenSize,
              storage_index);
#pragma unroll
          for (uint32_t index = 0; index < kElementsPerVector; ++index) {
            accumulators[index] +=
                device::cast<float>(source_values[index]);
          }
        }
#pragma unroll
        for (uint32_t index = 0; index < kElementsPerVector; ++index) {
          reduced_values[vector][index] =
              device::cast<DType>(accumulators[index]);
        }
      }
      sglang::jit_kernel::attntp::apply_welm_norm_pipeline<Trait>(
          reduced_values,
          residual_row,
          o_weight_values,
          post_weight_values,
          output_row,
          residual_out_row,
          params.o_norm_eps,
          params.post_norm_eps,
          warp_sums);
    }
  }

  ctrl.sync</*kFence=*/true, /*kStart=*/false>(
      params.rank, Trait::kNumGPU);
}

template <
    typename DType,
    uint32_t kNumGPU,
    uint32_t kHiddenSize,
    uint32_t kOutputMode,
    uint32_t kInternalPrecision,
    uint32_t kBlockSize>
struct FusedPrefillAttnTPOwnerPullNorm : public CustomAllReduceBase {
  using Trait = FusedPrefillAttnTPOwnerPullNormTrait<
      DType,
      kNumGPU,
      kHiddenSize,
      kOutputMode,
      kInternalPrecision,
      kBlockSize>;
  static constexpr auto kernel =
      fused_prefill_attntp_owner_pull_norm_kernel<Trait>;

  void _run(
      const tvm::ffi::Tensor partial,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const tvm::ffi::Tensor actual_rows,
      const tvm::ffi::Tensor owner_start,
      const int64_t max_blocks_per_sm_i64,
      const float o_norm_eps,
      const float post_norm_eps) {
    using namespace host;

    auto capacity_symbol = SymbolicSize{"capacity"};
    auto output_capacity_symbol = SymbolicSize{"output_capacity"};
    auto device_symbol = SymbolicDevice{};
    device_symbol.set_options<kDLCUDA>();
    TensorMatcher({capacity_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(partial);
    TensorMatcher({capacity_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual);
    TensorMatcher({kHiddenSize})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(o_norm_weight)
        .verify(post_norm_weight);
    TensorMatcher({output_capacity_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(output);
    TensorMatcher({output_capacity_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual_out);
    TensorMatcher({1})
        .with_dtype<int32_t>()
        .with_device(device_symbol)
        .verify(actual_rows)
        .verify(owner_start);

    const auto capacity_i64 = capacity_symbol.unwrap();
    const auto output_capacity_i64 = output_capacity_symbol.unwrap();
    RuntimeCheck(
        capacity_i64 > 0 && capacity_i64 <= UINT32_MAX,
        "Owner-pull capacity must fit uint32");
    RuntimeCheck(
        max_blocks_per_sm_i64 >= 0,
        "Owner-pull blocks per SM must be non-negative");
    const uint32_t capacity = static_cast<uint32_t>(capacity_i64);
    const uint32_t owner_capacity = div_ceil(capacity, kNumGPU);
    const uint32_t expected_output_capacity =
        Trait::kOutputMode ==
                sglang::jit_kernel::attntp::OutputMode::kTokenScattered
            ? owner_capacity
            : capacity;
    RuntimeCheck(
        output_capacity_i64 == expected_output_capacity,
        "Owner-pull output capacity mismatch, expected ",
        expected_output_capacity,
        ", got ",
        output_capacity_i64);
    const uint32_t num_global_tiles =
        Trait::kOutputMode ==
                sglang::jit_kernel::attntp::OutputMode::kTokenScattered
            ? kNumGPU * owner_capacity
            : capacity;

    RuntimeCheck(m_num_gpu == kNumGPU, "Owner-pull world size mismatch");
    RuntimeCheck(
        m_pull_ctrl.has_value(), "Owner-pull controller is not initialized");
    const uint64_t input_bytes =
        static_cast<uint64_t>(capacity) * kHiddenSize * sizeof(DType);
    RuntimeCheck(
        input_bytes <= static_cast<uint64_t>(m_pull_buffer_bytes),
        "Owner-pull buffer is too small, required ",
        input_bytes,
        ", available ",
        m_pull_buffer_bytes);
    for (const auto* pointer :
         {partial.data_ptr(),
          residual.data_ptr(),
          o_norm_weight.data_ptr(),
          post_norm_weight.data_ptr(),
          output.data_ptr(),
          residual_out.data_ptr()}) {
      RuntimeCheck(
          std::bit_cast<intptr_t>(pointer) % 16 == 0,
          "Owner-pull tensors must be 16-byte aligned");
    }

    const auto device = device_symbol.unwrap();
    int device_id = 0;
    host::RuntimeDeviceCheck(cudaGetDevice(&device_id));
    const uint32_t occupancy = get_max_occupancy();
    const uint32_t selected_blocks_per_sm = max_blocks_per_sm_i64 == 0
                                                ? occupancy
                                                : static_cast<uint32_t>(max_blocks_per_sm_i64);
    RuntimeCheck(
        selected_blocks_per_sm > 0 && selected_blocks_per_sm <= occupancy,
        "Owner-pull blocks per SM exceeds kernel occupancy");
    const uint32_t max_kernel_blocks =
        selected_blocks_per_sm * host::runtime::get_sm_count(device_id);
    const uint32_t num_blocks = std::min(
        {num_global_tiles, m_max_num_cta_pull, max_kernel_blocks});
    RuntimeCheck(num_blocks > 0, "Owner-pull requires at least one CTA");

    FusedPrefillAttnTPOwnerPullNormParams params{};
    params.residual = residual.data_ptr();
    params.o_norm_weight = o_norm_weight.data_ptr();
    params.post_norm_weight = post_norm_weight.data_ptr();
    params.output = output.data_ptr();
    params.residual_out = residual_out.data_ptr();
    params.actual_rows = static_cast<const int32_t*>(actual_rows.data_ptr());
    params.owner_start = static_cast<const int32_t*>(owner_start.data_ptr());
    params.o_norm_eps = o_norm_eps;
    params.post_norm_eps = post_norm_eps;
    params.rank = m_rank;
    params.capacity = capacity;
    params.owner_capacity = owner_capacity;
    params.num_global_tiles = num_global_tiles;

    const auto stream = LaunchKernel::resolve_device(device);
    const AllReduceData* data_ptr;
    bool is_capturing = false;
    if (m_is_graph_capturing) {
      cudaStreamCaptureStatus status;
      RuntimeDeviceCheck(cudaStreamIsCapturing(stream, &status));
      is_capturing = status == cudaStreamCaptureStatusActive;
    }
    if (is_capturing) {
      data_ptr = allocate_graph_capture_input(partial.data_ptr());
    } else {
      RuntimeDeviceCheck(cudaMemcpyAsync(
          get_pull_buffer(m_storage),
          partial.data_ptr(),
          input_bytes,
          cudaMemcpyDeviceToDevice,
          stream));
      data_ptr = get_data_ptr();
    }
    LaunchKernel(num_blocks, Trait::kBlockSize, stream)(
        kernel, data_ptr, params, *m_pull_ctrl);
  }

  static void run(
      CustomAllReduceRef object,
      const tvm::ffi::Tensor partial,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const tvm::ffi::Tensor actual_rows,
      const tvm::ffi::Tensor owner_start,
      const int64_t max_blocks_per_sm,
      const float o_norm_eps,
      const float post_norm_eps) {
    using Self = FusedPrefillAttnTPOwnerPullNorm;
    static_cast<Self*>(object.get())
        ->_run(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            output,
            residual_out,
            actual_rows,
            owner_start,
            max_blocks_per_sm,
            o_norm_eps,
            post_norm_eps);
  }

  static uint32_t get_max_occupancy() {
    return host::runtime::get_blocks_per_sm(kernel, Trait::kBlockSize);
  }
};

}  // namespace
