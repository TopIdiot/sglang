#pragma once

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <cstdint>
#include <type_traits>

namespace sglang::jit_kernel::attntp {

enum class OutputMode : uint32_t {
  kReplicated = 0,
  kSingleContributor = 1,
  kTokenScattered = 2,
};

enum class NormInternalPrecision : uint32_t {
  kReferenceBF16 = 0,
  kFullFP32 = 1,
};

struct BalancedRowRange {
  uint32_t offset;
  uint32_t count;
};

__host__ __device__ constexpr bool is_supported_attn_tp_size(
    const uint32_t attn_tp_size) {
  return attn_tp_size == 1 || attn_tp_size == 2 ||
         attn_tp_size == 4 || attn_tp_size == 8;
}

__host__ __device__ constexpr bool is_supported_hidden_size(
    const uint32_t hidden_size) {
  return hidden_size == 2048 || hidden_size == 4096;
}

__host__ __device__ constexpr bool is_supported_internal_precision(
    const NormInternalPrecision internal_precision) {
  return internal_precision == NormInternalPrecision::kReferenceBF16 ||
         internal_precision == NormInternalPrecision::kFullFP32;
}

__host__ __device__ constexpr BalancedRowRange balanced_row_range(
    const uint32_t total_rows,
    const uint32_t rank,
    const uint32_t attn_tp_size,
    const uint32_t owner_start) {
  const uint32_t logical_rank =
      (rank + attn_tp_size - owner_start) % attn_tp_size;
  const uint32_t base = total_rows / attn_tp_size;
  const uint32_t remainder = total_rows % attn_tp_size;
  return {
      logical_rank * base +
          (logical_rank < remainder ? logical_rank : remainder),
      base + static_cast<uint32_t>(logical_rank < remainder),
  };
}

__host__ __device__ constexpr uint32_t output_rows_for_rank(
    const uint32_t total_rows,
    const uint32_t rank,
    const uint32_t attn_tp_size,
    const uint32_t owner_start,
    const OutputMode output_mode) {
  if (output_mode == OutputMode::kReplicated) return total_rows;
  if (output_mode == OutputMode::kSingleContributor) {
    return rank == owner_start ? total_rows : 0;
  }
  return balanced_row_range(
             total_rows, rank, attn_tp_size, owner_start)
      .count;
}

template <
    typename DType_,
    uint32_t kHiddenSize_,
    uint32_t kInternalPrecision_ = 0,
    uint32_t kBlockSize_ = 256>
struct NormPipelineTrait {
  using DType = DType_;
  static constexpr uint32_t kHiddenSize = kHiddenSize_;
  static constexpr auto kInternalPrecision =
      static_cast<NormInternalPrecision>(kInternalPrecision_);
  static constexpr uint32_t kBlockSize = kBlockSize_;
  static constexpr uint32_t kNumWarps =
      kBlockSize / device::kWarpThreads;
  static constexpr uint32_t kElementsPerVector = 8;
  static constexpr uint32_t kVectorsPerThread =
      kHiddenSize / (kBlockSize * kElementsPerVector);
  using BF16Storage =
      device::AlignedVector<DType, kElementsPerVector>;
  using FP32Storage =
      device::AlignedVector<float, kElementsPerVector / 2>;

  static_assert(std::is_same_v<DType, bf16_t>);
  static_assert(is_supported_hidden_size(kHiddenSize));
  static_assert(is_supported_internal_precision(kInternalPrecision));
  static_assert(kBlockSize == 128 || kBlockSize == 256 || kBlockSize == 512);
  static_assert(
      kHiddenSize % (kBlockSize * kElementsPerVector) == 0);
  static_assert(kVectorsPerThread == 1 || kVectorsPerThread == 2 || kVectorsPerThread == 4);
  static_assert(sizeof(BF16Storage) == 16);
  static_assert(sizeof(FP32Storage) == 16);
};

template <uint32_t kNumWarps>
SGL_DEVICE float reduce_block_sum(float value, float* warp_sums) {
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

template <typename Trait>
SGL_DEVICE void apply_welm_norm_pipeline(
    const typename Trait::BF16Storage (
        &reduced_values)[Trait::kVectorsPerThread],
    const float* residual_row,
    const typename Trait::BF16Storage (
        &o_weight_values)[Trait::kVectorsPerThread],
    const typename Trait::BF16Storage (
        &post_weight_values)[Trait::kVectorsPerThread],
    typename Trait::DType* output_row,
    float* residual_out_row,
    const float o_norm_eps,
    const float post_norm_eps,
    float* warp_sums) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;
  using FP32Storage = typename Trait::FP32Storage;

  constexpr uint32_t kElementsPerVector = Trait::kElementsPerVector;
  constexpr uint32_t kVectorsPerThread = Trait::kVectorsPerThread;

  float o_norm_sum = 0.0f;
#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
#pragma unroll
    for (uint32_t index = 0; index < kElementsPerVector; ++index) {
      const float value = device::cast<float>(reduced_values[vector][index]);
      o_norm_sum += value * value;
    }
  }
  const float o_norm_total =
      reduce_block_sum<Trait::kNumWarps>(o_norm_sum, warp_sums);
  const float o_norm_scale =
      rsqrtf(o_norm_total / static_cast<float>(Trait::kHiddenSize) +
             o_norm_eps);

  BF16Storage o_norm_values[kVectorsPerThread];
#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
#pragma unroll
    for (uint32_t index = 0; index < kElementsPerVector; ++index) {
      const float normalized =
          device::cast<float>(reduced_values[vector][index]) *
          o_norm_scale *
          device::cast<float>(o_weight_values[vector][index]);
      o_norm_values[vector][index] = device::cast<DType>(normalized);
    }
  }

  BF16Storage post_norm_inputs[kVectorsPerThread];
  float post_norm_sum = 0.0f;
#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
    FP32Storage residual_values[2];
    FP32Storage residual_out_values[2];
#pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
      const uint32_t storage_index =
          threadIdx.x * (2 * kVectorsPerThread) + vector * 2 + half;
      residual_values[half].load(residual_row, storage_index);
    }
#pragma unroll
    for (uint32_t index = 0; index < kElementsPerVector; ++index) {
      const uint32_t half = index / (kElementsPerVector / 2);
      const uint32_t half_index = index % (kElementsPerVector / 2);
      const float residual_sum =
          device::cast<float>(o_norm_values[vector][index]) +
          residual_values[half][half_index];
      residual_out_values[half][half_index] = residual_sum;
      post_norm_inputs[vector][index] = device::cast<DType>(residual_sum);
      const float rounded =
          device::cast<float>(post_norm_inputs[vector][index]);
      post_norm_sum += rounded * rounded;
    }
#pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
      const uint32_t storage_index =
          threadIdx.x * (2 * kVectorsPerThread) + vector * 2 + half;
      residual_out_values[half].store(residual_out_row, storage_index);
    }
  }

  const float post_norm_total =
      reduce_block_sum<Trait::kNumWarps>(post_norm_sum, warp_sums);
  const float post_norm_scale =
      rsqrtf(post_norm_total / static_cast<float>(Trait::kHiddenSize) +
             post_norm_eps);

#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
    BF16Storage output_values;
#pragma unroll
    for (uint32_t index = 0; index < kElementsPerVector; ++index) {
      const float normalized =
          device::cast<float>(post_norm_inputs[vector][index]) *
          post_norm_scale *
          device::cast<float>(post_weight_values[vector][index]);
      output_values[index] = device::cast<DType>(normalized);
    }
    const uint32_t storage_index =
        threadIdx.x * kVectorsPerThread + vector;
    output_values.store(output_row, storage_index);
  }
}

template <typename Trait>
SGL_DEVICE void apply_welm_norm_pipeline_full_fp32(
    float (&activation_values)[Trait::kVectorsPerThread]
                              [Trait::kElementsPerVector],
    const float* residual_row,
    const typename Trait::BF16Storage (
        &o_weight_values)[Trait::kVectorsPerThread],
    const typename Trait::BF16Storage (
        &post_weight_values)[Trait::kVectorsPerThread],
    typename Trait::DType* output_row,
    float* residual_out_row,
    const float o_norm_eps,
    const float post_norm_eps,
    float* warp_sums) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;
  using FP32Storage = typename Trait::FP32Storage;

  constexpr uint32_t kElementsPerVector = Trait::kElementsPerVector;
  constexpr uint32_t kVectorsPerThread = Trait::kVectorsPerThread;

  float o_norm_sum = 0.0f;
#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
#pragma unroll
    for (uint32_t index = 0; index < kElementsPerVector; ++index) {
      const float value = activation_values[vector][index];
      o_norm_sum += value * value;
    }
  }
  const float o_norm_total =
      reduce_block_sum<Trait::kNumWarps>(o_norm_sum, warp_sums);
  const float o_norm_scale =
      rsqrtf(o_norm_total / static_cast<float>(Trait::kHiddenSize) +
             o_norm_eps);

#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
#pragma unroll
    for (uint32_t index = 0; index < kElementsPerVector; ++index) {
      activation_values[vector][index] *=
          o_norm_scale *
          device::cast<float>(o_weight_values[vector][index]);
    }
  }

  float post_norm_sum = 0.0f;
#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
    FP32Storage residual_values[2];
    FP32Storage residual_out_values[2];
#pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
      const uint32_t storage_index =
          threadIdx.x * (2 * kVectorsPerThread) + vector * 2 + half;
      residual_values[half].load(residual_row, storage_index);
    }
#pragma unroll
    for (uint32_t index = 0; index < kElementsPerVector; ++index) {
      const uint32_t half = index / (kElementsPerVector / 2);
      const uint32_t half_index = index % (kElementsPerVector / 2);
      const float residual_sum =
          activation_values[vector][index] +
          residual_values[half][half_index];
      activation_values[vector][index] = residual_sum;
      residual_out_values[half][half_index] = residual_sum;
      post_norm_sum += residual_sum * residual_sum;
    }
#pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
      const uint32_t storage_index =
          threadIdx.x * (2 * kVectorsPerThread) + vector * 2 + half;
      residual_out_values[half].store(residual_out_row, storage_index);
    }
  }

  const float post_norm_total =
      reduce_block_sum<Trait::kNumWarps>(post_norm_sum, warp_sums);
  const float post_norm_scale =
      rsqrtf(post_norm_total / static_cast<float>(Trait::kHiddenSize) +
             post_norm_eps);

#pragma unroll
  for (uint32_t vector = 0; vector < kVectorsPerThread; ++vector) {
    BF16Storage output_values;
#pragma unroll
    for (uint32_t index = 0; index < kElementsPerVector; ++index) {
      const float normalized =
          activation_values[vector][index] * post_norm_scale *
          device::cast<float>(post_weight_values[vector][index]);
      output_values[index] = device::cast<DType>(normalized);
    }
    const uint32_t storage_index =
        threadIdx.x * kVectorsPerThread + vector;
    output_values.store(output_row, storage_index);
  }
}

}  // namespace sglang::jit_kernel::attntp
