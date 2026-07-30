#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <sgl_kernel/distributed/common.cuh>
#include <sgl_kernel/distributed/custom_all_reduce.cuh>

#include "attntp_fused_ipc_norm_common.cuh"

#include <algorithm>
#include <bit>
#include <cstdint>

namespace {

using device::distributed::PullController, device::distributed::PushController;
using host::distributed::AllReduceData, host::distributed::CustomAllReduceBase,
    host::distributed::CustomAllReduceRef;

template <typename T>
SGL_DEVICE T* decode_byte_offset(void* base, uint64_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(base) + offset);
}

template <typename T>
SGL_DEVICE const T* decode_byte_offset(const void* base, uint64_t offset) {
  return reinterpret_cast<const T*>(
      reinterpret_cast<const uint8_t*>(base) + offset);
}

template <typename T>
SGL_DEVICE void decode_store_volatile_16b(void* address, const T& value) {
  static_assert(sizeof(T) == 16 && alignof(T) == 16);
  const uint4 raw = *reinterpret_cast<const uint4*>(&value);
  asm volatile(
      "st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};"
      :
      : "r"(raw.x), "r"(raw.y), "r"(raw.z), "r"(raw.w), "l"(address)
      : "memory");
}

template <typename T>
SGL_DEVICE T decode_load_volatile_16b(const void* address) {
  static_assert(sizeof(T) == 16 && alignof(T) == 16);
  uint4 raw;
  asm volatile(
      "ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
      : "=r"(raw.x), "=r"(raw.y), "=r"(raw.z), "=r"(raw.w)
      : "l"(address)
      : "memory");
  return *reinterpret_cast<const T*>(&raw);
}

SGL_DEVICE void decode_store_release_system(
    uint32_t* address,
    uint32_t value) {
  asm volatile(
      "st.release.sys.global.u32 [%0], %1;"
      :
      : "l"(address), "r"(value)
      : "memory");
}

SGL_DEVICE uint32_t decode_load_acquire_system(const uint32_t* address) {
  uint32_t value;
  asm volatile(
      "ld.acquire.sys.global.u32 %0, [%1];"
      : "=r"(value)
      : "l"(address)
      : "memory");
  return value;
}

struct FusedDecodeAttnTPSourcePushNormParams {
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
  uint32_t rows;
  uint32_t owner_capacity;
  uint32_t num_global_rows;
};

template <
    typename DType_,
    uint32_t kNumGPU_,
    uint32_t kHiddenSize_,
    uint32_t kOutputMode_,
    uint32_t kInternalPrecision_,
    uint32_t kBlockSize_,
    uint32_t kSignalBackoff_>
struct FusedDecodeAttnTPSourcePushNormTrait
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
  static constexpr uint32_t kSignalBackoff = kSignalBackoff_;
  static constexpr auto kOutputMode =
      static_cast<sglang::jit_kernel::attntp::OutputMode>(kOutputMode_);

  static_assert(
      sglang::jit_kernel::attntp::is_supported_attn_tp_size(kNumGPU));
  static_assert(kNumGPU >= 2);
  static_assert(
      kSignalBackoff == 32 || kSignalBackoff == 64 ||
      kSignalBackoff == 128 || kSignalBackoff == 256);
};

template <typename Trait>
__global__ void fused_decode_attntp_source_push_norm_kernel(
    const FusedDecodeAttnTPSourcePushNormParams __grid_constant__ params,
    const PushController __grid_constant__ ctrl) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kRowBytes = kHiddenSize * sizeof(DType);
  constexpr uint32_t kVectorsPerThread = Trait::kVectorsPerThread;
  constexpr uint32_t kElementsPerVector = Trait::kElementsPerVector;

  const uint32_t epoch = ctrl.epoch();
  const uint32_t signal_value = epoch + 1;
  const uint64_t epoch_offset =
      static_cast<uint64_t>(epoch) * params.epoch_bytes;

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

  for (uint32_t global_slot = blockIdx.x;
       global_slot < params.num_global_rows;
       global_slot += gridDim.x) {
    uint32_t owner = 0;
    uint32_t local_row = global_slot;
    uint32_t global_row = global_slot;
    if constexpr (
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kTokenScattered) {
      const uint32_t owner_slot = global_slot / params.owner_capacity;
      local_row = global_slot % params.owner_capacity;
      owner = owner_slot % Trait::kNumGPU;
      const auto owner_range =
          sglang::jit_kernel::attntp::balanced_row_range(
              params.rows, owner, Trait::kNumGPU, 0);
      if (local_row >= owner_range.count) continue;
      global_row = owner_range.offset + local_row;
    } else if (global_row >= params.rows) {
      continue;
    }

    BF16Storage local_values[kVectorsPerThread];
#pragma unroll
    for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
      const uint32_t storage_index =
          threadIdx.x * kVectorsPerThread + vector;
      local_values[vector].load(
          static_cast<const DType*>(params.partial) +
              static_cast<uint64_t>(global_row) * kHiddenSize,
          storage_index);
    }

    if constexpr (
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kReplicated) {
      if (threadIdx.x == 0) {
#pragma unroll
        for (uint32_t destination = 0;
             destination < Trait::kNumGPU;
             ++destination) {
          const auto* signal = decode_byte_offset<uint32_t>(
              params.buffer[destination],
              epoch_offset +
                  static_cast<uint64_t>(params.rank) * params.buffer_bytes +
                  params.signal_offset_bytes +
                  static_cast<uint64_t>(local_row) * sizeof(uint32_t));
          while (decode_load_acquire_system(signal) != 0) {
            __nanosleep(Trait::kSignalBackoff);
          }
        }
      }
      __syncthreads();

#pragma unroll
      for (uint32_t destination = 0;
           destination < Trait::kNumGPU;
           ++destination) {
        void* destination_row = decode_byte_offset<void>(
            params.buffer[destination],
            epoch_offset +
                static_cast<uint64_t>(params.rank) * params.buffer_bytes +
                static_cast<uint64_t>(local_row) * kRowBytes);
#pragma unroll
        for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
          const uint32_t storage_index =
              threadIdx.x * kVectorsPerThread + vector;
          decode_store_volatile_16b(
              decode_byte_offset<void>(
                  destination_row,
                  static_cast<uint64_t>(storage_index) *
                      sizeof(BF16Storage)),
              local_values[vector]);
        }
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
#pragma unroll
        for (uint32_t destination = 0;
             destination < Trait::kNumGPU;
             ++destination) {
          auto* signal = decode_byte_offset<uint32_t>(
              params.buffer[destination],
              epoch_offset +
                  static_cast<uint64_t>(params.rank) * params.buffer_bytes +
                  params.signal_offset_bytes +
                  static_cast<uint64_t>(local_row) * sizeof(uint32_t));
          decode_store_release_system(signal, signal_value);
        }
      }
      owner = params.rank;
    } else {
      void* send_slot = decode_byte_offset<void>(
          params.buffer[owner],
          epoch_offset +
              static_cast<uint64_t>(params.rank) * params.buffer_bytes);
      auto* send_signal = decode_byte_offset<uint32_t>(
          send_slot,
          params.signal_offset_bytes +
              static_cast<uint64_t>(local_row) * sizeof(uint32_t));
      if (threadIdx.x == 0) {
        while (decode_load_acquire_system(send_signal) != 0) {
          __nanosleep(Trait::kSignalBackoff);
        }
      }
      __syncthreads();

      void* send_row = decode_byte_offset<void>(
          send_slot, static_cast<uint64_t>(local_row) * kRowBytes);
#pragma unroll
      for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
        const uint32_t storage_index =
            threadIdx.x * kVectorsPerThread + vector;
        decode_store_volatile_16b(
            decode_byte_offset<void>(
                send_row,
                static_cast<uint64_t>(storage_index) * sizeof(BF16Storage)),
            local_values[vector]);
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        decode_store_release_system(send_signal, signal_value);
      }
      if (params.rank != owner) continue;
    }

    void* receive_epoch = decode_byte_offset<void>(
        params.buffer[params.rank], epoch_offset);
    if (threadIdx.x == 0) {
#pragma unroll
      for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
        const auto* source_signal = decode_byte_offset<uint32_t>(
            receive_epoch,
            static_cast<uint64_t>(source) * params.buffer_bytes +
                params.signal_offset_bytes +
                static_cast<uint64_t>(local_row) * sizeof(uint32_t));
        while (decode_load_acquire_system(source_signal) != signal_value) {
          __nanosleep(Trait::kSignalBackoff);
        }
      }
    }
    __syncthreads();

    const auto* residual_row =
        static_cast<const float*>(params.residual) +
        static_cast<uint64_t>(global_row) * kHiddenSize;
    const uint32_t output_row_index =
        Trait::kOutputMode ==
                sglang::jit_kernel::attntp::OutputMode::kReplicated
            ? global_row
            : local_row;
    auto* output_row = static_cast<DType*>(params.output) +
                       static_cast<uint64_t>(output_row_index) * kHiddenSize;
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
          const auto* source_row = decode_byte_offset<void>(
              receive_epoch,
              static_cast<uint64_t>(source) * params.buffer_bytes +
                  static_cast<uint64_t>(local_row) * kRowBytes);
          const BF16Storage source_values =
              decode_load_volatile_16b<BF16Storage>(
                  decode_byte_offset<void>(
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
        const uint32_t storage_index =
            threadIdx.x * kVectorsPerThread + vector;
#pragma unroll
        for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
          const auto* source_row = decode_byte_offset<void>(
              receive_epoch,
              static_cast<uint64_t>(source) * params.buffer_bytes +
                  static_cast<uint64_t>(local_row) * kRowBytes);
          const BF16Storage source_values =
              decode_load_volatile_16b<BF16Storage>(
                  decode_byte_offset<void>(
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

    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
      for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
        auto* source_signal = decode_byte_offset<uint32_t>(
            receive_epoch,
            static_cast<uint64_t>(source) * params.buffer_bytes +
                params.signal_offset_bytes +
                static_cast<uint64_t>(local_row) * sizeof(uint32_t));
        decode_store_release_system(source_signal, 0);
      }
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
    uint32_t kInternalPrecision,
    uint32_t kBlockSize,
    uint32_t kSignalBackoff>
struct FusedDecodeAttnTPSourcePushNorm : public CustomAllReduceBase {
  using Trait = FusedDecodeAttnTPSourcePushNormTrait<
      DType,
      kNumGPU,
      kHiddenSize,
      kOutputMode,
      kInternalPrecision,
      kBlockSize,
      kSignalBackoff>;
  static constexpr auto kernel =
      fused_decode_attntp_source_push_norm_kernel<Trait>;

  void _run(
      const tvm::ffi::Tensor partial,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const int64_t arena_rows_i64,
      const int64_t max_blocks_per_sm_i64,
      const float o_norm_eps,
      const float post_norm_eps) {
    using namespace host;

    auto rows_symbol = SymbolicSize{"rows"};
    auto output_rows_symbol = SymbolicSize{"output_rows"};
    auto device_symbol = SymbolicDevice{};
    device_symbol.set_options<kDLCUDA>();
    TensorMatcher({rows_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(partial);
    TensorMatcher({rows_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual);
    TensorMatcher({kHiddenSize})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(o_norm_weight)
        .verify(post_norm_weight);
    TensorMatcher({output_rows_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(output);
    TensorMatcher({output_rows_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual_out);
    const auto rows_i64 = rows_symbol.unwrap();
    const auto output_rows_i64 = output_rows_symbol.unwrap();
    RuntimeCheck(
        rows_i64 > 0 && rows_i64 <= 256,
        "Decode source-push rows must be in [1, 256]");
    RuntimeCheck(
        arena_rows_i64 >= rows_i64 && arena_rows_i64 <= 256,
        "Decode source-push arena rows must cover input rows");
    RuntimeCheck(
        max_blocks_per_sm_i64 >= 0,
        "Decode source-push blocks per SM must be non-negative");
    const uint32_t rows = static_cast<uint32_t>(rows_i64);
    const uint32_t arena_rows = static_cast<uint32_t>(arena_rows_i64);
    const bool is_scattered =
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kTokenScattered;
    const uint32_t slot_rows =
        is_scattered ? div_ceil(arena_rows, kNumGPU) : arena_rows;
    const uint32_t owner_capacity = div_ceil(rows, kNumGPU);
    const uint32_t expected_output_rows =
        is_scattered ? owner_capacity : rows;
    RuntimeCheck(
        output_rows_i64 == expected_output_rows,
        "Decode source-push output rows mismatch");
    const uint32_t num_global_rows =
        is_scattered ? kNumGPU * owner_capacity : rows;
    const uint64_t signal_offset_bytes =
        static_cast<uint64_t>(slot_rows) * kHiddenSize * sizeof(DType);
    const uint64_t signal_bytes =
        div_ceil(
            static_cast<uint64_t>(slot_rows) * sizeof(uint32_t),
            static_cast<uint64_t>(128)) *
        128;
    const uint64_t required_buffer_bytes =
        signal_offset_bytes + signal_bytes;

    RuntimeCheck(
        m_num_gpu == kNumGPU, "Decode source-push world size mismatch");
    RuntimeCheck(
        m_push_ctrl.has_value(),
        "Decode source-push controller is not initialized");
    RuntimeCheck(
        required_buffer_bytes <=
            static_cast<uint64_t>(m_push_buffer_bytes),
        "Decode source-push buffer is too small");
    for (const auto* pointer :
         {partial.data_ptr(),
          residual.data_ptr(),
          o_norm_weight.data_ptr(),
          post_norm_weight.data_ptr(),
          output.data_ptr(),
          residual_out.data_ptr()}) {
      RuntimeCheck(
          std::bit_cast<intptr_t>(pointer) % 16 == 0,
          "Decode source-push tensors must be 16-byte aligned");
    }

    const auto device = device_symbol.unwrap();
    int device_id = 0;
    RuntimeDeviceCheck(cudaGetDevice(&device_id));
    const uint32_t occupancy = get_max_occupancy();
    const uint32_t selected_blocks_per_sm = max_blocks_per_sm_i64 == 0
                                                ? occupancy
                                                : static_cast<uint32_t>(max_blocks_per_sm_i64);
    RuntimeCheck(
        selected_blocks_per_sm > 0 && selected_blocks_per_sm <= occupancy,
        "Decode source-push blocks per SM exceeds kernel occupancy");
    const uint32_t max_kernel_blocks =
        selected_blocks_per_sm * host::runtime::get_sm_count(device_id);
    const uint32_t num_blocks = std::min(
        {num_global_rows, m_max_num_cta_push, max_kernel_blocks});
    RuntimeCheck(num_blocks > 0, "Decode source-push requires one CTA");

    FusedDecodeAttnTPSourcePushNormParams params{};
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
    params.rows = rows;
    params.owner_capacity = owner_capacity;
    params.num_global_rows = num_global_rows;

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
      const int64_t arena_rows,
      const int64_t max_blocks_per_sm,
      const float o_norm_eps,
      const float post_norm_eps) {
    using Self = FusedDecodeAttnTPSourcePushNorm;
    static_cast<Self*>(object.get())
        ->_run(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            output,
            residual_out,
            arena_rows,
            max_blocks_per_sm,
            o_norm_eps,
            post_norm_eps);
  }

  static uint32_t get_max_occupancy() {
    return host::runtime::get_blocks_per_sm(kernel, Trait::kBlockSize);
  }
};

struct FusedDecodeAttnTPOwnerPullNormParams {
  const void* residual;
  const void* o_norm_weight;
  const void* post_norm_weight;
  void* output;
  void* residual_out;
  float o_norm_eps;
  float post_norm_eps;
  uint32_t rank;
  uint32_t rows;
  uint32_t owner_capacity;
  uint32_t num_global_rows;
};

template <
    typename DType_,
    uint32_t kNumGPU_,
    uint32_t kHiddenSize_,
    uint32_t kOutputMode_,
    uint32_t kInternalPrecision_,
    uint32_t kBlockSize_>
struct FusedDecodeAttnTPOwnerPullNormTrait
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
__global__ void fused_decode_attntp_owner_pull_norm_kernel(
    const AllReduceData* __restrict__ data,
    const FusedDecodeAttnTPOwnerPullNormParams __grid_constant__ params,
    const PullController __grid_constant__ ctrl) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kVectorsPerThread = Trait::kVectorsPerThread;
  constexpr uint32_t kElementsPerVector = Trait::kElementsPerVector;

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

  for (uint32_t global_slot = blockIdx.x;
       global_slot < params.num_global_rows;
       global_slot += gridDim.x) {
    uint32_t global_row = global_slot;
    uint32_t output_row_index = global_slot;
    if constexpr (
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kSingleContributor) {
      if (params.rank != 0 || global_slot >= params.rows) continue;
    } else if constexpr (
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kTokenScattered) {
      const uint32_t owner_slot = global_slot / params.owner_capacity;
      const uint32_t local_row = global_slot % params.owner_capacity;
      const uint32_t owner = owner_slot % Trait::kNumGPU;
      const auto owner_range =
          sglang::jit_kernel::attntp::balanced_row_range(
              params.rows, owner, Trait::kNumGPU, 0);
      if (params.rank != owner || local_row >= owner_range.count) continue;
      global_row = owner_range.offset + local_row;
      output_row_index = local_row;
    } else if (global_slot >= params.rows) {
      continue;
    }

    const auto* residual_row =
        static_cast<const float*>(params.residual) +
        static_cast<uint64_t>(global_row) * kHiddenSize;
    auto* output_row = static_cast<DType*>(params.output) +
                       static_cast<uint64_t>(output_row_index) * kHiddenSize;
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
struct FusedDecodeAttnTPOwnerPullNorm : public CustomAllReduceBase {
  using Trait = FusedDecodeAttnTPOwnerPullNormTrait<
      DType,
      kNumGPU,
      kHiddenSize,
      kOutputMode,
      kInternalPrecision,
      kBlockSize>;
  static constexpr auto kernel =
      fused_decode_attntp_owner_pull_norm_kernel<Trait>;

  void _run(
      const tvm::ffi::Tensor partial,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const int64_t max_blocks_per_sm_i64,
      const float o_norm_eps,
      const float post_norm_eps) {
    using namespace host;

    auto rows_symbol = SymbolicSize{"rows"};
    auto output_rows_symbol = SymbolicSize{"output_rows"};
    auto device_symbol = SymbolicDevice{};
    device_symbol.set_options<kDLCUDA>();
    TensorMatcher({rows_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(partial);
    TensorMatcher({rows_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual);
    TensorMatcher({kHiddenSize})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(o_norm_weight)
        .verify(post_norm_weight);
    TensorMatcher({output_rows_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(output);
    TensorMatcher({output_rows_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual_out);
    const auto rows_i64 = rows_symbol.unwrap();
    const auto output_rows_i64 = output_rows_symbol.unwrap();
    RuntimeCheck(
        rows_i64 > 0 && rows_i64 <= 256,
        "Decode owner-pull rows must be in [1, 256]");
    RuntimeCheck(
        max_blocks_per_sm_i64 >= 0,
        "Decode owner-pull blocks per SM must be non-negative");
    const uint32_t rows = static_cast<uint32_t>(rows_i64);
    const uint32_t owner_capacity = div_ceil(rows, kNumGPU);
    const bool is_scattered =
        Trait::kOutputMode ==
        sglang::jit_kernel::attntp::OutputMode::kTokenScattered;
    const uint32_t expected_output_rows =
        is_scattered ? owner_capacity : rows;
    RuntimeCheck(
        output_rows_i64 == expected_output_rows,
        "Decode owner-pull output rows mismatch");
    const uint32_t num_global_rows =
        is_scattered ? kNumGPU * owner_capacity : rows;

    RuntimeCheck(
        m_num_gpu == kNumGPU, "Decode owner-pull world size mismatch");
    RuntimeCheck(
        m_pull_ctrl.has_value(),
        "Decode owner-pull controller is not initialized");
    const uint64_t input_bytes =
        static_cast<uint64_t>(rows) * kHiddenSize * sizeof(DType);
    RuntimeCheck(
        input_bytes <= static_cast<uint64_t>(m_pull_buffer_bytes),
        "Decode owner-pull buffer is too small");
    for (const auto* pointer :
         {partial.data_ptr(),
          residual.data_ptr(),
          o_norm_weight.data_ptr(),
          post_norm_weight.data_ptr(),
          output.data_ptr(),
          residual_out.data_ptr()}) {
      RuntimeCheck(
          std::bit_cast<intptr_t>(pointer) % 16 == 0,
          "Decode owner-pull tensors must be 16-byte aligned");
    }

    const auto device = device_symbol.unwrap();
    int device_id = 0;
    RuntimeDeviceCheck(cudaGetDevice(&device_id));
    const uint32_t occupancy = get_max_occupancy();
    const uint32_t selected_blocks_per_sm = max_blocks_per_sm_i64 == 0
                                                ? occupancy
                                                : static_cast<uint32_t>(max_blocks_per_sm_i64);
    RuntimeCheck(
        selected_blocks_per_sm > 0 && selected_blocks_per_sm <= occupancy,
        "Decode owner-pull blocks per SM exceeds kernel occupancy");
    const uint32_t max_kernel_blocks =
        selected_blocks_per_sm * host::runtime::get_sm_count(device_id);
    const uint32_t num_blocks = std::min(
        {num_global_rows, m_max_num_cta_pull, max_kernel_blocks});
    RuntimeCheck(num_blocks > 0, "Decode owner-pull requires one CTA");

    FusedDecodeAttnTPOwnerPullNormParams params{};
    params.residual = residual.data_ptr();
    params.o_norm_weight = o_norm_weight.data_ptr();
    params.post_norm_weight = post_norm_weight.data_ptr();
    params.output = output.data_ptr();
    params.residual_out = residual_out.data_ptr();
    params.o_norm_eps = o_norm_eps;
    params.post_norm_eps = post_norm_eps;
    params.rank = m_rank;
    params.rows = rows;
    params.owner_capacity = owner_capacity;
    params.num_global_rows = num_global_rows;

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
      const int64_t max_blocks_per_sm,
      const float o_norm_eps,
      const float post_norm_eps) {
    using Self = FusedDecodeAttnTPOwnerPullNorm;
    static_cast<Self*>(object.get())
        ->_run(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            output,
            residual_out,
            max_blocks_per_sm,
            o_norm_eps,
            post_norm_eps);
  }

  static uint32_t get_max_occupancy() {
    return host::runtime::get_blocks_per_sm(kernel, Trait::kBlockSize);
  }
};

struct FusedDecodeAttnTPLocalNormParams {
  const void* partial;
  const void* residual;
  const void* o_norm_weight;
  const void* post_norm_weight;
  void* output;
  void* residual_out;
  float o_norm_eps;
  float post_norm_eps;
  uint32_t rows;
};

template <typename Trait>
__global__ void fused_decode_attntp_local_norm_kernel(
    const FusedDecodeAttnTPLocalNormParams __grid_constant__ params) {
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

  for (uint32_t row = blockIdx.x; row < params.rows; row += gridDim.x) {
    const auto* partial_row =
        static_cast<const DType*>(params.partial) +
        static_cast<uint64_t>(row) * kHiddenSize;
    const auto* residual_row =
        static_cast<const float*>(params.residual) +
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
        BF16Storage partial_values;
        const uint32_t storage_index =
            threadIdx.x * kVectorsPerThread + vector;
        partial_values.load(partial_row, storage_index);
#pragma unroll
        for (uint32_t index = 0; index < Trait::kElementsPerVector;
             ++index) {
          activation_values[vector][index] =
              device::cast<float>(partial_values[index]);
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
        reduced_values[vector].load(partial_row, storage_index);
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
    uint32_t kOutputMode,
    uint32_t kInternalPrecision>
struct FusedDecodeAttnTPLocalNorm {
  using Trait = sglang::jit_kernel::attntp::NormPipelineTrait<
      DType,
      kHiddenSize,
      kInternalPrecision>;
  static constexpr auto kernel =
      fused_decode_attntp_local_norm_kernel<Trait>;

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
    static_assert(
        kOutputMode <= static_cast<uint32_t>(
                           sglang::jit_kernel::attntp::OutputMode::
                               kTokenScattered));

    auto rows_symbol = SymbolicSize{"rows"};
    auto device_symbol = SymbolicDevice{};
    device_symbol.set_options<kDLCUDA>();
    TensorMatcher({rows_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(partial)
        .verify(output);
    TensorMatcher({rows_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<float>()
        .with_device(device_symbol)
        .verify(residual)
        .verify(residual_out);
    TensorMatcher({kHiddenSize})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(o_norm_weight)
        .verify(post_norm_weight);
    const auto rows_i64 = rows_symbol.unwrap();
    RuntimeCheck(
        rows_i64 > 0 && rows_i64 <= 256,
        "Decode local rows must be in [1, 256]");
    for (const auto* pointer :
         {partial.data_ptr(),
          residual.data_ptr(),
          o_norm_weight.data_ptr(),
          post_norm_weight.data_ptr(),
          output.data_ptr(),
          residual_out.data_ptr()}) {
      RuntimeCheck(
          std::bit_cast<intptr_t>(pointer) % 16 == 0,
          "Decode local tensors must be 16-byte aligned");
    }

    const uint32_t rows = static_cast<uint32_t>(rows_i64);
    const auto device = device_symbol.unwrap();
    int device_id = 0;
    RuntimeDeviceCheck(cudaGetDevice(&device_id));
    const uint32_t max_kernel_blocks =
        get_max_occupancy() * host::runtime::get_sm_count(device_id);
    const uint32_t num_blocks = std::min(rows, max_kernel_blocks);

    FusedDecodeAttnTPLocalNormParams params{};
    params.partial = partial.data_ptr();
    params.residual = residual.data_ptr();
    params.o_norm_weight = o_norm_weight.data_ptr();
    params.post_norm_weight = post_norm_weight.data_ptr();
    params.output = output.data_ptr();
    params.residual_out = residual_out.data_ptr();
    params.o_norm_eps = o_norm_eps;
    params.post_norm_eps = post_norm_eps;
    params.rows = rows;

    LaunchKernel(num_blocks, Trait::kBlockSize, device)(kernel, params);
  }

  static uint32_t get_max_occupancy() {
    return host::runtime::get_blocks_per_sm(kernel, Trait::kBlockSize);
  }
};

}  // namespace
