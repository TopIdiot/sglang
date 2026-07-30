#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/vec.cuh>

#include "attntp_fused_ipc_norm_common.cuh"
#include <algorithm>
#include <cstdint>

namespace {

SGL_DEVICE uint32_t symm_atomic_add_system(uint32_t* address, uint32_t value) {
  uint32_t previous;
  asm volatile("atom.sys.global.add.u32 %0, [%1], %2;" : "=r"(previous) : "l"(address), "r"(value) : "memory");
  return previous;
}

SGL_DEVICE uint32_t symm_load_acquire_system(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

SGL_DEVICE void symm_store_release_system(uint32_t* address, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
}

SGL_DEVICE void symm_multimem_red_release_add_u32(uint32_t* address, uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile(
      "multimem.red.release.sys.global.add.u32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
#else
  asm volatile("trap;");
#endif
}

template <typename BF16Storage>
SGL_DEVICE void symm_multimem_load_reduce_bf16x8(const void* address, BF16Storage& values) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t word0;
  uint32_t word1;
  uint32_t word2;
  uint32_t word3;
  asm volatile("multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
               : "=r"(word0), "=r"(word1), "=r"(word2), "=r"(word3)
               : "l"(address)
               : "memory");
  auto* words = reinterpret_cast<uint32_t*>(values.data());
  words[0] = word0;
  words[1] = word1;
  words[2] = word2;
  words[3] = word3;
#else
  asm volatile("trap;");
#endif
}

struct FusedDecodeAttnTPSymmNormParams {
  const void* partials[2];
  const void* multicast_partial;
  uint32_t* local_multicast_signals;
  uint32_t* multicast_signals;
  const void* const* partial_pointer_table;
  uint32_t* local_signals;
  uint32_t* peer_signals;
  const void* residual;
  const void* o_norm_weight;
  const void* post_norm_weight;
  void* output;
  void* residual_out;
  float o_norm_eps;
  float post_norm_eps;
  uint32_t rank;
  uint32_t rows;
  uint32_t signal_stride;
  uint32_t multicast_signal_stride;
};

template <
    typename DType_,
    uint32_t kHiddenSize_,
    uint32_t kOutputMode_,
    uint32_t kInternalPrecision_,
    uint32_t kNumGPU_,
    uint32_t kBlockSize_,
    uint32_t kSignalBackoff_>
struct FusedDecodeAttnTPSymmNormTrait
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
  static constexpr auto kOutputMode = static_cast<sglang::jit_kernel::attntp::OutputMode>(kOutputMode_);
  static_assert(kNumGPU == 2 || kNumGPU == 4 || kNumGPU == 8);
  static_assert(kSignalBackoff == 32 || kSignalBackoff == 64 || kSignalBackoff == 128 || kSignalBackoff == 256);
};

template <typename Trait>
SGL_DEVICE void symm_wait_for_group_epoch(
    const FusedDecodeAttnTPSymmNormParams& params, const uint32_t signal_offset, const uint32_t signal_epoch) {
  if constexpr (Trait::kNumGPU == 2) {
    while (symm_load_acquire_system(params.peer_signals + signal_offset) != signal_epoch) {
      __nanosleep(Trait::kSignalBackoff);
    }
  } else {
    const uint32_t expected = signal_epoch * Trait::kNumGPU;
    while (symm_load_acquire_system(params.local_multicast_signals + signal_offset) != expected) {
      __nanosleep(Trait::kSignalBackoff);
    }
  }
}

template <typename Trait>
SGL_DEVICE void symm_publish_group_epoch(
    const FusedDecodeAttnTPSymmNormParams& params, const uint32_t signal_offset, const uint32_t signal_epoch) {
  if constexpr (Trait::kNumGPU == 2) {
    symm_store_release_system(params.local_signals + signal_offset, signal_epoch);
  } else {
    symm_multimem_red_release_add_u32(params.multicast_signals + signal_offset, 1);
  }
}

template <typename Trait>
SGL_DEVICE const typename Trait::DType*
symm_source_partial(const FusedDecodeAttnTPSymmNormParams& params, const uint32_t source) {
  if constexpr (Trait::kNumGPU == 2) {
    return static_cast<const typename Trait::DType*>(params.partials[source]);
  } else {
    return static_cast<const typename Trait::DType*>(params.partial_pointer_table[source]);
  }
}

template <typename Trait>
__global__ void fused_decode_attntp_symm_norm_kernel(const FusedDecodeAttnTPSymmNormParams __grid_constant__ params) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kVectorsPerThread = Trait::kVectorsPerThread;
  constexpr uint32_t kElementsPerVector = Trait::kElementsPerVector;

  __shared__ uint32_t signal_epoch;
  if (threadIdx.x == 0) {
    if constexpr (Trait::kNumGPU == 2) __threadfence_system();
    signal_epoch = symm_atomic_add_system(params.local_signals + blockIdx.x, 1) + 1;
    if constexpr (Trait::kNumGPU > 2) {
      symm_publish_group_epoch<Trait>(params, blockIdx.x, signal_epoch);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    symm_wait_for_group_epoch<Trait>(params, blockIdx.x, signal_epoch);
  }
  __syncthreads();

  __shared__ float warp_sums[Trait::kNumWarps];
  BF16Storage o_weight_values[kVectorsPerThread];
  BF16Storage post_weight_values[kVectorsPerThread];
#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
    const uint32_t storage_index = threadIdx.x * kVectorsPerThread + vector;
    o_weight_values[vector].load(params.o_norm_weight, storage_index);
    post_weight_values[vector].load(params.post_norm_weight, storage_index);
  }

  const uint32_t work_rows = Trait::kOutputMode == sglang::jit_kernel::attntp::OutputMode::kTokenScattered
                                 ? (params.rows + Trait::kNumGPU - 1) / Trait::kNumGPU
                                 : params.rows;
  for (uint32_t work_row = blockIdx.x; work_row < work_rows; work_row += gridDim.x) {
    uint32_t global_row = work_row;
    const uint32_t output_row_index = work_row;
    if constexpr (Trait::kOutputMode == sglang::jit_kernel::attntp::OutputMode::kSingleContributor) {
      if (params.rank != 0) continue;
    } else if constexpr (Trait::kOutputMode == sglang::jit_kernel::attntp::OutputMode::kTokenScattered) {
      const auto owner_range =
          sglang::jit_kernel::attntp::balanced_row_range(params.rows, params.rank, Trait::kNumGPU, 0);
      if (output_row_index >= owner_range.count) continue;
      global_row = owner_range.offset + output_row_index;
    }

    const auto* residual_row =
        static_cast<const float*>(params.residual) + static_cast<uint64_t>(global_row) * kHiddenSize;
    auto* output_row = static_cast<DType*>(params.output) + static_cast<uint64_t>(output_row_index) * kHiddenSize;
    auto* residual_out_row =
        static_cast<float*>(params.residual_out) + static_cast<uint64_t>(output_row_index) * kHiddenSize;

    if constexpr (Trait::kInternalPrecision == sglang::jit_kernel::attntp::NormInternalPrecision::kFullFP32) {
      float activation_values[kVectorsPerThread][kElementsPerVector];
#pragma unroll
      for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
#pragma unroll
        for (uint32_t index = 0; index < kElementsPerVector; ++index) {
          activation_values[vector][index] = 0.0f;
        }
        const uint32_t storage_index = threadIdx.x * kVectorsPerThread + vector;
#pragma unroll
        for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
          BF16Storage source_values;
          source_values.load(
              symm_source_partial<Trait>(params, source) + static_cast<uint64_t>(global_row) * kHiddenSize,
              storage_index);
#pragma unroll
          for (uint32_t index = 0; index < kElementsPerVector; ++index) {
            activation_values[vector][index] += device::cast<float>(source_values[index]);
          }
        }
      }
      sglang::jit_kernel::attntp::apply_welm_norm_pipeline_full_fp32<Trait>(
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
        const uint32_t storage_index = threadIdx.x * kVectorsPerThread + vector;
        if constexpr (Trait::kNumGPU == 4 || Trait::kNumGPU == 8) {
          const auto* multicast_row =
              static_cast<const DType*>(params.multicast_partial) + static_cast<uint64_t>(global_row) * kHiddenSize;
          symm_multimem_load_reduce_bf16x8(
              multicast_row + static_cast<uint64_t>(storage_index) * kElementsPerVector, reduced_values[vector]);
        } else {
          float accumulators[kElementsPerVector] = {};
#pragma unroll
          for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
            BF16Storage source_values;
            source_values.load(
                symm_source_partial<Trait>(params, source) + static_cast<uint64_t>(global_row) * kHiddenSize,
                storage_index);
#pragma unroll
            for (uint32_t index = 0; index < kElementsPerVector; ++index) {
              accumulators[index] += device::cast<float>(source_values[index]);
            }
          }
#pragma unroll
          for (uint32_t index = 0; index < kElementsPerVector; ++index) {
            reduced_values[vector][index] = device::cast<DType>(accumulators[index]);
          }
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
    const uint32_t signal_offset = Trait::kNumGPU == 2 ? params.signal_stride + blockIdx.x
                                                       : params.multicast_signal_stride + blockIdx.x;
    symm_publish_group_epoch<Trait>(params, signal_offset, signal_epoch);
    symm_wait_for_group_epoch<Trait>(params, signal_offset, signal_epoch);
  }
  __syncthreads();
}

template <
    typename DType,
    uint32_t kHiddenSize,
    uint32_t kOutputMode,
    uint32_t kInternalPrecision,
    uint32_t kNumGPU,
    uint32_t kBlockSize,
    uint32_t kSignalBackoff>
struct FusedDecodeAttnTPSymmNorm {
  using Trait = FusedDecodeAttnTPSymmNormTrait<
      DType,
      kHiddenSize,
      kOutputMode,
      kInternalPrecision,
      kNumGPU,
      kBlockSize,
      kSignalBackoff>;
  static constexpr auto kernel = fused_decode_attntp_symm_norm_kernel<Trait>;

  static void
  run(const tvm::ffi::Tensor partial,
      const int64_t peer_pointer,
      const int64_t multicast_pointer,
      const int64_t multicast_signal_offset_bytes_i64,
      const int64_t multicast_signal_slots_i64,
      const int64_t partial_pointer_table,
      const tvm::ffi::Tensor signal_pad,
      const int64_t peer_signal_pointer,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const int64_t rank_i64,
      const int64_t max_blocks_per_sm_i64,
      const float o_norm_eps,
      const float post_norm_eps) {
    using namespace host;

    auto rows_symbol = SymbolicSize{"rows"};
    auto output_rows_symbol = SymbolicSize{"output_rows"};
    auto signal_slots_symbol = SymbolicSize{"signal_slots"};
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
    TensorMatcher({signal_slots_symbol})
        .with_strides({1})
        .with_dtype<uint32_t>()
        .with_device(device_symbol)
        .verify(signal_pad);

    const int64_t rows_i64 = rows_symbol.unwrap();
    const int64_t output_rows_i64 = output_rows_symbol.unwrap();
    RuntimeCheck(rows_i64 > 0 && rows_i64 <= UINT32_MAX, "Symm decode rows must fit uint32");
    RuntimeCheck(rank_i64 >= 0 && rank_i64 < Trait::kNumGPU, "Symm rank is outside the AttnTP group");
    RuntimeCheck(max_blocks_per_sm_i64 >= 0, "Symm blocks per SM must be non-negative");
    RuntimeCheck(multicast_signal_offset_bytes_i64 >= 0, "Symm multicast signal offset must be non-negative");
    RuntimeCheck(multicast_signal_slots_i64 >= 0, "Symm multicast signal slots must be non-negative");
    if constexpr (Trait::kNumGPU == 2) {
      RuntimeCheck(peer_pointer > 0, "Symm peer pointer must be non-zero");
      RuntimeCheck(peer_signal_pointer > 0, "Symm peer signal pointer must be non-zero");
    } else {
      RuntimeCheck(partial_pointer_table > 0, "Symm device buffer-pointer table must be non-zero");
      RuntimeCheck(multicast_pointer > 0, "Symm multicast pointer must be non-zero");
      RuntimeCheck(
          multicast_signal_offset_bytes_i64 >= rows_i64 * kHiddenSize * static_cast<int64_t>(sizeof(DType)),
          "Symm multicast signals overlap the partial input");
      RuntimeCheck(
          multicast_signal_offset_bytes_i64 % alignof(uint32_t) == 0,
          "Symm multicast signal offset must be uint32 aligned");
    }
    const uint32_t rows = static_cast<uint32_t>(rows_i64);
    const uint32_t expected_output_rows = Trait::kOutputMode == sglang::jit_kernel::attntp::OutputMode::kTokenScattered
                                              ? div_ceil(rows, Trait::kNumGPU)
                                              : rows;
    RuntimeCheck(
        output_rows_i64 == expected_output_rows,
        "Symm decode output rows mismatch, expected ",
        expected_output_rows,
        ", got ",
        output_rows_i64);

    for (const auto pointer :
         {reinterpret_cast<uintptr_t>(partial.data_ptr()),
          reinterpret_cast<uintptr_t>(residual.data_ptr()),
          reinterpret_cast<uintptr_t>(o_norm_weight.data_ptr()),
          reinterpret_cast<uintptr_t>(post_norm_weight.data_ptr()),
          reinterpret_cast<uintptr_t>(output.data_ptr()),
          reinterpret_cast<uintptr_t>(residual_out.data_ptr())}) {
      RuntimeCheck(pointer % 16 == 0, "Symm tensors must be 16-byte aligned");
    }

    const uint32_t rank = static_cast<uint32_t>(rank_i64);
    FusedDecodeAttnTPSymmNormParams params{};
    if constexpr (Trait::kNumGPU == 2) {
      params.partials[rank] = partial.data_ptr();
      params.partials[1 - rank] = reinterpret_cast<const void*>(static_cast<uintptr_t>(peer_pointer));
    }
    params.multicast_partial = reinterpret_cast<const void*>(static_cast<uintptr_t>(multicast_pointer));
    if constexpr (Trait::kNumGPU > 2) {
      const auto signal_offset_bytes = static_cast<uint64_t>(multicast_signal_offset_bytes_i64);
      params.local_multicast_signals = reinterpret_cast<uint32_t*>(
          reinterpret_cast<uintptr_t>(partial.data_ptr()) + signal_offset_bytes);
      params.multicast_signals =
          reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(multicast_pointer) + signal_offset_bytes);
    }
    params.partial_pointer_table = reinterpret_cast<const void* const*>(static_cast<uintptr_t>(partial_pointer_table));
    params.local_signals = static_cast<uint32_t*>(signal_pad.data_ptr());
    params.peer_signals = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(peer_signal_pointer));
    params.residual = residual.data_ptr();
    params.o_norm_weight = o_norm_weight.data_ptr();
    params.post_norm_weight = post_norm_weight.data_ptr();
    params.output = output.data_ptr();
    params.residual_out = residual_out.data_ptr();
    params.o_norm_eps = o_norm_eps;
    params.post_norm_eps = post_norm_eps;
    params.rank = rank;
    params.rows = rows;

    const auto device = device_symbol.unwrap();
    int device_id = 0;
    RuntimeDeviceCheck(cudaGetDevice(&device_id));
    if constexpr (Trait::kNumGPU == 4 || Trait::kNumGPU == 8) {
      int compute_capability_major = 0;
      RuntimeDeviceCheck(
          cudaDeviceGetAttribute(&compute_capability_major, cudaDevAttrComputeCapabilityMajor, device_id));
      RuntimeCheck(compute_capability_major >= 9, "Symm multimem requires compute capability 9.0 or newer");
    }
    const uint32_t occupancy = get_max_occupancy();
    const uint32_t selected_blocks_per_sm = max_blocks_per_sm_i64 == 0
                                                ? occupancy
                                                : static_cast<uint32_t>(max_blocks_per_sm_i64);
    RuntimeCheck(
        selected_blocks_per_sm > 0 && selected_blocks_per_sm <= occupancy,
        "Symm blocks per SM exceeds kernel occupancy");
    const uint32_t max_blocks = selected_blocks_per_sm * runtime::get_sm_count(device_id);
    const uint32_t work_rows =
        Trait::kOutputMode == sglang::jit_kernel::attntp::OutputMode::kTokenScattered ? expected_output_rows : rows;
    const uint32_t num_blocks = std::min(work_rows, max_blocks);
    RuntimeCheck(num_blocks > 0, "Symm decode kernel requires at least one CTA");
    const int64_t signal_slots_i64 = signal_slots_symbol.unwrap();
    RuntimeCheck(signal_slots_i64 >= 2 && signal_slots_i64 <= UINT32_MAX, "Symm signal pad size must fit uint32");
    const uint32_t signal_stride = static_cast<uint32_t>(signal_slots_i64 / 2);
    RuntimeCheck(num_blocks <= signal_stride, "Symm signal pad is too small for the resident grid");
    params.signal_stride = signal_stride;
    if constexpr (Trait::kNumGPU > 2) {
      RuntimeCheck(
          multicast_signal_slots_i64 >= 2 && multicast_signal_slots_i64 <= UINT32_MAX,
          "Symm multicast signal storage must fit uint32");
      const uint32_t multicast_signal_stride = static_cast<uint32_t>(multicast_signal_slots_i64 / 2);
      RuntimeCheck(num_blocks <= multicast_signal_stride, "Symm multicast signal storage is too small");
      params.multicast_signal_stride = multicast_signal_stride;
    }

    const auto stream = LaunchKernel::resolve_device(device);
    LaunchKernel(num_blocks, Trait::kBlockSize, stream)(kernel, params);
  }

  static uint32_t get_max_occupancy() {
    return host::runtime::get_blocks_per_sm(kernel, Trait::kBlockSize);
  }
};

}  // namespace
