#pragma once

#include "attntp_fused_ipc_norm_common.cuh"

#include <cstdint>

namespace sglang::jit_kernel::attntp {

template <typename BaseTrait, uint32_t kCohortCount_>
struct CohortNormTrait : BaseTrait {
  using Base = BaseTrait;
  using DType = typename Base::DType;
  using BF16Storage = typename Base::BF16Storage;
  using FP32Storage = typename Base::FP32Storage;

  static constexpr uint32_t kCohortCount = kCohortCount_;
  static constexpr uint32_t kCohortThreads =
      Base::kBlockSize / kCohortCount;
  static constexpr uint32_t kWarpsPerCohort =
      kCohortThreads / device::kWarpThreads;
  static constexpr uint32_t kElementsPerVector =
      Base::kElementsPerVector;
  static constexpr uint32_t kVectorsPerThread =
      Base::kHiddenSize /
      (kCohortThreads * kElementsPerVector);

  static_assert(Base::kHiddenSize == 4096);
  static_assert(Base::kBlockSize == 512);
  static_assert(
      kCohortCount == 1 || kCohortCount == 2 ||
      kCohortCount == 4);
  static_assert(
      kCohortThreads == 512 || kCohortThreads == 256 ||
      kCohortThreads == 128);
  static_assert(
      kVectorsPerThread == 1 || kVectorsPerThread == 2 ||
      kVectorsPerThread == 4);
  static_assert(
      kCohortThreads * kVectorsPerThread *
          kElementsPerVector ==
      Base::kHiddenSize);
};

template <typename Trait>
SGL_DEVICE uint32_t cohort_id() {
  return threadIdx.x / Trait::kCohortThreads;
}

template <typename Trait>
SGL_DEVICE uint32_t cohort_thread_id() {
  return threadIdx.x % Trait::kCohortThreads;
}

template <typename Trait>
SGL_DEVICE void cohort_barrier() {
  const uint32_t barrier_id = cohort_id<Trait>() + 1;
  asm volatile(
      "bar.sync %0, %1;"
      :
      : "r"(barrier_id), "r"(Trait::kCohortThreads)
      : "memory");
}

template <typename Trait>
SGL_DEVICE float cohort_reduce_sum(
    float value,
    float* shared_warp_sums,
    float* shared_totals) {
  const uint32_t local_thread = cohort_thread_id<Trait>();
  const uint32_t lane = local_thread % device::kWarpThreads;
  const uint32_t cohort_warp =
      local_thread / device::kWarpThreads;
  const uint32_t cohort = cohort_id<Trait>();
  const uint32_t warp_offset =
      cohort * Trait::kWarpsPerCohort;

  value = device::warp::reduce_sum(value);
  if (lane == 0) {
    shared_warp_sums[warp_offset + cohort_warp] = value;
  }
  cohort_barrier<Trait>();

  if (cohort_warp == 0) {
    const float warp_value =
        lane < Trait::kWarpsPerCohort
        ? shared_warp_sums[warp_offset + lane]
        : 0.0f;
    const float total =
        device::warp::reduce_sum<Trait::kWarpsPerCohort>(
            warp_value);
    if (lane == 0) shared_totals[cohort] = total;
  }
  cohort_barrier<Trait>();
  return shared_totals[cohort];
}

template <typename Trait>
struct CohortRegisterWeightProvider {
  using BF16Storage = typename Trait::BF16Storage;

  BF16Storage o_norm[Trait::kVectorsPerThread];
  BF16Storage post_norm[Trait::kVectorsPerThread];

  SGL_DEVICE void load(
      const void* o_norm_weight,
      const void* post_norm_weight) {
    const uint32_t local_thread = cohort_thread_id<Trait>();
#pragma unroll
    for (uint32_t vector = 0;
         vector < Trait::kVectorsPerThread;
         ++vector) {
      const uint32_t storage_index =
          local_thread * Trait::kVectorsPerThread + vector;
      o_norm[vector].load(o_norm_weight, storage_index);
      post_norm[vector].load(post_norm_weight, storage_index);
    }
  }

  SGL_DEVICE BF16Storage load_o_norm(
      const uint32_t vector) const {
    return o_norm[vector];
  }

  SGL_DEVICE BF16Storage load_post_norm(
      const uint32_t vector) const {
    return post_norm[vector];
  }
};

template <typename Trait>
struct CohortSharedWeightProvider {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;

  const DType* o_norm;
  const DType* post_norm;

  SGL_DEVICE BF16Storage load_o_norm(
      const uint32_t vector) const {
    BF16Storage values;
    const uint32_t storage_index =
        cohort_thread_id<Trait>() *
            Trait::kVectorsPerThread +
        vector;
    values.load(o_norm, storage_index);
    return values;
  }

  SGL_DEVICE BF16Storage load_post_norm(
      const uint32_t vector) const {
    BF16Storage values;
    const uint32_t storage_index =
        cohort_thread_id<Trait>() *
            Trait::kVectorsPerThread +
        vector;
    values.load(post_norm, storage_index);
    return values;
  }
};

template <typename Trait, typename WeightProvider>
SGL_DEVICE void apply_welm_norm_pipeline_full_fp32_cohort(
    float (&activation_values)[Trait::kVectorsPerThread]
                              [Trait::kElementsPerVector],
    const float* residual_row,
    const WeightProvider& weights,
    typename Trait::DType* output_row,
    float* residual_out_row,
    const float o_norm_eps,
    const float post_norm_eps,
    float* shared_warp_sums,
    float* shared_totals) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;
  using FP32Storage = typename Trait::FP32Storage;

  constexpr uint32_t kElementsPerVector =
      Trait::kElementsPerVector;
  constexpr uint32_t kVectorsPerThread =
      Trait::kVectorsPerThread;
  const uint32_t local_thread = cohort_thread_id<Trait>();

  float o_norm_sum = 0.0f;
#pragma unroll
  for (uint32_t vector = 0;
       vector < kVectorsPerThread;
       ++vector) {
#pragma unroll
    for (uint32_t index = 0;
         index < kElementsPerVector;
         ++index) {
      const float value = activation_values[vector][index];
      o_norm_sum += value * value;
    }
  }
  const float o_norm_total = cohort_reduce_sum<Trait>(
      o_norm_sum, shared_warp_sums, shared_totals);
  const float o_norm_scale = rsqrtf(
      o_norm_total / static_cast<float>(Trait::kHiddenSize) +
      o_norm_eps);

#pragma unroll
  for (uint32_t vector = 0;
       vector < kVectorsPerThread;
       ++vector) {
    const BF16Storage weight_values =
        weights.load_o_norm(vector);
#pragma unroll
    for (uint32_t index = 0;
         index < kElementsPerVector;
         ++index) {
      activation_values[vector][index] *=
          o_norm_scale *
          device::cast<float>(weight_values[index]);
    }
  }

  float post_norm_sum = 0.0f;
#pragma unroll
  for (uint32_t vector = 0;
       vector < kVectorsPerThread;
       ++vector) {
    FP32Storage residual_values[2];
    FP32Storage residual_out_values[2];
#pragma unroll
    for (uint32_t half = 0; half < 2; ++half) {
      const uint32_t storage_index =
          local_thread * (2 * kVectorsPerThread) +
          vector * 2 + half;
      residual_values[half].load(
          residual_row, storage_index);
    }
#pragma unroll
    for (uint32_t index = 0;
         index < kElementsPerVector;
         ++index) {
      const uint32_t half =
          index / (kElementsPerVector / 2);
      const uint32_t half_index =
          index % (kElementsPerVector / 2);
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
          local_thread * (2 * kVectorsPerThread) +
          vector * 2 + half;
      residual_out_values[half].store(
          residual_out_row, storage_index);
    }
  }

  const float post_norm_total = cohort_reduce_sum<Trait>(
      post_norm_sum, shared_warp_sums, shared_totals);
  const float post_norm_scale = rsqrtf(
      post_norm_total /
          static_cast<float>(Trait::kHiddenSize) +
      post_norm_eps);

#pragma unroll
  for (uint32_t vector = 0;
       vector < kVectorsPerThread;
       ++vector) {
    const BF16Storage weight_values =
        weights.load_post_norm(vector);
    BF16Storage output_values;
#pragma unroll
    for (uint32_t index = 0;
         index < kElementsPerVector;
         ++index) {
      const float normalized =
          activation_values[vector][index] *
          post_norm_scale *
          device::cast<float>(weight_values[index]);
      output_values[index] =
          device::cast<DType>(normalized);
    }
    const uint32_t storage_index =
        local_thread * kVectorsPerThread + vector;
    output_values.store(output_row, storage_index);
  }
}

}  // namespace sglang::jit_kernel::attntp
