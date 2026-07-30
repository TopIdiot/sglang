#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/vec.cuh>

#include "attntp_fused_ipc_norm_common.cuh"
#include "attntp_tile_pipeline_norm.cuh"

#include <algorithm>
#include <cstdint>

#ifndef SGL_ATTNTP_TILE_BLOCK_SIZE
#error "SGL_ATTNTP_TILE_BLOCK_SIZE must be defined by the JIT wrapper"
#endif

#ifndef SGL_ATTNTP_TILE_LAUNCH_BOUNDS_MIN_BLOCKS
#error "SGL_ATTNTP_TILE_LAUNCH_BOUNDS_MIN_BLOCKS must be defined by the JIT wrapper"
#endif

#if SGL_ATTNTP_TILE_LAUNCH_BOUNDS_MIN_BLOCKS == 0
#define SGL_ATTNTP_TILE_LAUNCH_BOUNDS
#else
#define SGL_ATTNTP_TILE_LAUNCH_BOUNDS                    \
  __launch_bounds__(                                     \
      SGL_ATTNTP_TILE_BLOCK_SIZE,                        \
      SGL_ATTNTP_TILE_LAUNCH_BOUNDS_MIN_BLOCKS)
#endif

namespace {

constexpr uint64_t kTileTicketEpochStride = uint64_t{1} << 32;

SGL_DEVICE uint64_t tile_load_acquire_system(const uint64_t* address) {
  uint64_t value;
  asm volatile(
      "ld.acquire.sys.global.u64 %0, [%1];"
      : "=l"(value)
      : "l"(address)
      : "memory");
  return value;
}

SGL_DEVICE void tile_store_release_system(
    uint64_t* address,
    uint64_t value) {
  asm volatile(
      "st.release.sys.global.u64 [%0], %1;"
      :
      : "l"(address), "l"(value)
      : "memory");
}

SGL_DEVICE uint64_t tile_load_acquire_gpu(const uint64_t* address) {
  uint64_t value;
  asm volatile(
      "ld.acquire.gpu.global.u64 %0, [%1];"
      : "=l"(value)
      : "l"(address)
      : "memory");
  return value;
}

SGL_DEVICE void tile_store_release_gpu(
    uint64_t* address,
    uint64_t value) {
  asm volatile(
      "st.release.gpu.global.u64 [%0], %1;"
      :
      : "l"(address), "l"(value)
      : "memory");
}

SGL_DEVICE uint64_t tile_atomic_add_gpu(
    uint64_t* address,
    uint64_t value) {
  uint64_t previous;
  asm volatile(
      "atom.gpu.global.add.u64 %0, [%1], %2;"
      : "=l"(previous)
      : "l"(address), "l"(value)
      : "memory");
  return previous;
}

struct AttnTPReplicatedTilePipelineParams {
  const void* const* pointer_table;
  void* local_base;
  const void* residual;
  const void* o_norm_weight;
  const void* post_norm_weight;
  void* output;
  void* residual_out;
  const int32_t* actual_rows;
  const int32_t* owner_start;
  uint64_t reduced_offset_bytes;
  uint64_t input_ready_offset_bytes;
  uint64_t group_ready_offset_bytes;
  uint64_t kernel_done_offset_bytes;
  uint64_t completion_counter_offset_bytes;
  uint64_t ready_offset_bytes;
  uint64_t consumed_offset_bytes;
  uint64_t consumed_owner_stride_bytes;
  float o_norm_eps;
  float post_norm_eps;
  uint32_t rank;
  uint32_t capacity;
  uint32_t producer_blocks;
  uint32_t consumer_blocks;
};

template <typename Trait>
SGL_DEVICE uint32_t tile_actual_rows(
    const AttnTPReplicatedTilePipelineParams& params) {
  if constexpr (Trait::kDecode) {
    return params.capacity;
  } else {
    return static_cast<uint32_t>(*params.actual_rows);
  }
}

template <typename Trait>
SGL_DEVICE uint32_t tile_owner_start(
    const AttnTPReplicatedTilePipelineParams& params) {
  if constexpr (Trait::kDecode) {
    return 0;
  } else {
    return static_cast<uint32_t>(*params.owner_start);
  }
}

template <
    typename DType_,
    uint32_t kNumGPU_,
    uint32_t kHiddenSize_,
    bool kDecode_,
    uint32_t kRowsPerTile_,
    uint32_t kRingStages_,
    uint32_t kBlockSize_,
    uint32_t kSignalBackoff_,
    uint32_t kLaunchBoundsMinBlocks_,
    uint32_t kConsumerCohorts_,
    bool kSharedWeights_>
struct AttnTPReplicatedTilePipelineTrait
    : sglang::jit_kernel::attntp::NormPipelineTrait<
          DType_,
          kHiddenSize_,
          static_cast<uint32_t>(
              sglang::jit_kernel::attntp::NormInternalPrecision::kFullFP32),
          kBlockSize_> {
  using Base = sglang::jit_kernel::attntp::NormPipelineTrait<
      DType_,
      kHiddenSize_,
      static_cast<uint32_t>(
          sglang::jit_kernel::attntp::NormInternalPrecision::kFullFP32),
      kBlockSize_>;
  using DType = typename Base::DType;
  using BF16Storage = typename Base::BF16Storage;
  using FP32Storage = typename Base::FP32Storage;

  static constexpr uint32_t kNumGPU = kNumGPU_;
  static constexpr uint32_t kHiddenSize = kHiddenSize_;
  static constexpr bool kDecode = kDecode_;
  static constexpr uint32_t kRowsPerTile = kRowsPerTile_;
  static constexpr uint32_t kRingStages = kRingStages_;
  static constexpr uint32_t kBlockSize = kBlockSize_;
  static constexpr uint32_t kSignalBackoff = kSignalBackoff_;
  static constexpr uint32_t kLaunchBoundsMinBlocks =
      kLaunchBoundsMinBlocks_;
  static constexpr uint32_t kConsumerCohorts =
      kConsumerCohorts_;
  static constexpr bool kSharedWeights = kSharedWeights_;
  static constexpr uint32_t kWeightSharedBytes =
      kSharedWeights ? 2 * kHiddenSize * sizeof(DType) : 0;
  static constexpr uint32_t kNumWarps = Base::kNumWarps;
  static constexpr uint32_t kElementsPerVector =
      Base::kElementsPerVector;
  static constexpr uint32_t kVectorsPerThread =
      Base::kVectorsPerThread;
  static constexpr auto kInternalPrecision =
      Base::kInternalPrecision;

  static_assert(kNumGPU == 2 || kNumGPU == 4 || kNumGPU == 8);
  static_assert(
      kRowsPerTile == 1 || kRowsPerTile == 2 ||
      kRowsPerTile == 4 || kRowsPerTile == 8 ||
      kRowsPerTile == 16 || kRowsPerTile == 32);
  static_assert(
      kRingStages == 2 || kRingStages == 4 ||
      kRingStages == 8 || kRingStages == 16 ||
      kRingStages == 32 || kRingStages == 64 ||
      kRingStages == 128);
  static_assert((kRingStages & (kRingStages - 1)) == 0);
  static_assert(
      kLaunchBoundsMinBlocks >= 0 &&
      kLaunchBoundsMinBlocks <= 4);
  static_assert(kBlockSize == SGL_ATTNTP_TILE_BLOCK_SIZE);
  static_assert(
      kLaunchBoundsMinBlocks ==
      SGL_ATTNTP_TILE_LAUNCH_BOUNDS_MIN_BLOCKS);
  static_assert(
      kConsumerCohorts == 1 || kConsumerCohorts == 2 ||
      kConsumerCohorts == 4);
  static_assert(
      (kConsumerCohorts == 1 && !kSharedWeights) ||
      (kHiddenSize == 4096 && kBlockSize == 512));
  static_assert(
      kInternalPrecision ==
      sglang::jit_kernel::attntp::NormInternalPrecision::kFullFP32);
};

template <typename T = void>
SGL_DEVICE T* tile_byte_offset(void* base, uint64_t offset) {
  return reinterpret_cast<T*>(
      reinterpret_cast<uintptr_t>(base) + offset);
}

template <typename T = const void>
SGL_DEVICE const T* tile_byte_offset(
    const void* base,
    uint64_t offset) {
  return reinterpret_cast<const T*>(
      reinterpret_cast<uintptr_t>(base) + offset);
}

template <typename Trait>
SGL_DEVICE void* tile_rank_base(
    const AttnTPReplicatedTilePipelineParams& params,
    uint32_t rank) {
  return const_cast<void*>(params.pointer_table[rank]);
}

template <typename Trait>
SGL_DEVICE uint64_t* tile_rank_signal(
    const AttnTPReplicatedTilePipelineParams& params,
    uint32_t rank,
    uint64_t offset) {
  return tile_byte_offset<uint64_t>(
      tile_rank_base<Trait>(params, rank), offset);
}

template <typename Trait>
SGL_DEVICE void tile_wait_for_ticket(
    const uint64_t* address,
    uint64_t expected) {
  while (tile_load_acquire_system(address) < expected) {
    __nanosleep(Trait::kSignalBackoff);
  }
}

template <typename Trait>
SGL_DEVICE void tile_wait_for_local_ticket(
    const uint64_t* address,
    uint64_t expected) {
  while (tile_load_acquire_gpu(address) < expected) {
    __nanosleep(Trait::kSignalBackoff);
  }
}

template <typename Trait>
SGL_DEVICE uint64_t tile_ticket(
    const AttnTPReplicatedTilePipelineParams& params,
    uint64_t epoch,
    uint32_t local_tile) {
  return epoch * kTileTicketEpochStride +
         static_cast<uint64_t>(local_tile) + 1;
}

template <typename Trait>
SGL_DEVICE void tile_wait_for_inputs(
    const AttnTPReplicatedTilePipelineParams& params,
    uint64_t epoch) {
  auto* local_input_ready = tile_rank_signal<Trait>(
      params, params.rank, params.input_ready_offset_bytes);
  auto* local_group_ready = tile_rank_signal<Trait>(
      params, params.rank, params.group_ready_offset_bytes);
  auto* local_completion_counter = tile_rank_signal<Trait>(
      params, params.rank, params.completion_counter_offset_bytes);

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    tile_store_release_system(local_input_ready, epoch);
#pragma unroll
    for (uint32_t source = 0; source < Trait::kNumGPU; ++source) {
      tile_wait_for_ticket<Trait>(
          tile_rank_signal<Trait>(
              params, source, params.input_ready_offset_bytes),
          epoch);
    }
    tile_store_release_gpu(local_completion_counter, 0);
    tile_store_release_gpu(local_group_ready, epoch);
  }
  if (threadIdx.x == 0) {
    tile_wait_for_local_ticket<Trait>(
        local_group_ready, epoch);
  }
  __syncthreads();
}

template <typename Trait>
SGL_DEVICE void tile_produce(
    const AttnTPReplicatedTilePipelineParams& params,
    uint64_t epoch) {
  using BF16Storage = typename Trait::BF16Storage;
  using FP32Storage = typename Trait::FP32Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kElementsPerVector =
      Trait::kElementsPerVector;
  constexpr uint32_t kVectorsPerThread =
      Trait::kVectorsPerThread;
  constexpr uint64_t kScratchTileBytes =
      static_cast<uint64_t>(Trait::kRowsPerTile) *
      kHiddenSize * sizeof(float);

  const uint32_t producer_id = blockIdx.x;
  const uint32_t actual_rows = tile_actual_rows<Trait>(params);
  const uint32_t owner_start = tile_owner_start<Trait>(params);
  const auto owner_range =
      sglang::jit_kernel::attntp::balanced_row_range(
          actual_rows,
          params.rank,
          Trait::kNumGPU,
          owner_start);
  const uint32_t owner_tiles =
      (owner_range.count + Trait::kRowsPerTile - 1) /
      Trait::kRowsPerTile;

  for (uint32_t local_tile = producer_id;
       local_tile < owner_tiles;
       local_tile += params.producer_blocks) {
    const uint32_t slot =
        local_tile & (Trait::kRingStages - 1);
    if (local_tile >= Trait::kRingStages) {
      const uint64_t previous_ticket = tile_ticket<Trait>(
          params,
          epoch,
          local_tile - Trait::kRingStages);
      if (threadIdx.x == 0) {
#pragma unroll
        for (uint32_t consumer = 0;
             consumer < Trait::kNumGPU;
             ++consumer) {
          const uint64_t consumed_offset =
              params.consumed_offset_bytes +
              static_cast<uint64_t>(params.rank) *
                  params.consumed_owner_stride_bytes +
              static_cast<uint64_t>(slot) * sizeof(uint64_t);
          tile_wait_for_ticket<Trait>(
              tile_rank_signal<Trait>(
                  params, consumer, consumed_offset),
              previous_ticket);
        }
      }
      __syncthreads();
    }

    const uint32_t local_row_start =
        local_tile * Trait::kRowsPerTile;
    const uint32_t remaining_rows =
        owner_range.count - local_row_start;
    const uint32_t rows_in_tile =
        remaining_rows < Trait::kRowsPerTile
            ? remaining_rows
            : Trait::kRowsPerTile;
    auto* scratch_tile = tile_byte_offset<float>(
        params.local_base,
        params.reduced_offset_bytes +
            static_cast<uint64_t>(slot) * kScratchTileBytes);

    for (uint32_t row_in_tile = 0;
         row_in_tile < rows_in_tile;
         ++row_in_tile) {
      const uint32_t global_row =
          owner_range.offset + local_row_start + row_in_tile;
      auto* scratch_row =
          scratch_tile +
          static_cast<uint64_t>(row_in_tile) * kHiddenSize;
#pragma unroll
      for (uint32_t vector = 0;
           vector < kVectorsPerThread;
           ++vector) {
        float accumulators[kElementsPerVector] = {};
        const uint32_t storage_index =
            threadIdx.x * kVectorsPerThread + vector;
#pragma unroll
        for (uint32_t source = 0;
             source < Trait::kNumGPU;
             ++source) {
          const auto* source_base =
              static_cast<const typename Trait::DType*>(
                  params.pointer_table[source]);
          BF16Storage source_values;
          source_values.load(
              source_base +
                  static_cast<uint64_t>(global_row) *
                      kHiddenSize,
              storage_index);
#pragma unroll
          for (uint32_t element = 0;
               element < kElementsPerVector;
               ++element) {
            accumulators[element] +=
                device::cast<float>(source_values[element]);
          }
        }

        FP32Storage reduced_values[2];
#pragma unroll
        for (uint32_t element = 0;
             element < kElementsPerVector;
             ++element) {
          reduced_values[
              element / (kElementsPerVector / 2)]
                        [element % (kElementsPerVector / 2)] =
              accumulators[element];
        }
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          const uint32_t fp32_storage_index =
              threadIdx.x * (2 * kVectorsPerThread) +
              vector * 2 + half;
          reduced_values[half].store(
              scratch_row, fp32_storage_index);
        }
      }
    }

    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
      auto* ready = tile_rank_signal<Trait>(
          params,
          params.rank,
          params.ready_offset_bytes +
              static_cast<uint64_t>(slot) * sizeof(uint64_t));
      tile_store_release_system(
          ready,
          tile_ticket<Trait>(
              params, epoch, local_tile));
    }
    __syncthreads();
  }
}

template <typename Trait>
SGL_DEVICE void tile_consume(
    const AttnTPReplicatedTilePipelineParams& params,
    uint64_t epoch) {
  using BF16Storage = typename Trait::BF16Storage;
  using FP32Storage = typename Trait::FP32Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kElementsPerVector =
      Trait::kElementsPerVector;
  constexpr uint32_t kVectorsPerThread =
      Trait::kVectorsPerThread;
  constexpr uint64_t kScratchTileBytes =
      static_cast<uint64_t>(Trait::kRowsPerTile) *
      kHiddenSize * sizeof(float);

  __shared__ float warp_sums[Trait::kNumWarps];
  BF16Storage o_weight_values[kVectorsPerThread];
  BF16Storage post_weight_values[kVectorsPerThread];
#pragma unroll
  for (uint32_t vector = 0;
       vector < kVectorsPerThread;
       ++vector) {
    const uint32_t storage_index =
        threadIdx.x * kVectorsPerThread + vector;
    o_weight_values[vector].load(
        params.o_norm_weight, storage_index);
    post_weight_values[vector].load(
        params.post_norm_weight, storage_index);
  }

  const uint32_t consumer_id =
      blockIdx.x - params.producer_blocks;
  const uint32_t actual_rows = tile_actual_rows<Trait>(params);
  const uint32_t owner_start = tile_owner_start<Trait>(params);
  const uint32_t max_owner_rows =
      (actual_rows + Trait::kNumGPU - 1) /
      Trait::kNumGPU;
  const uint32_t owner_tile_rounds =
      (max_owner_rows + Trait::kRowsPerTile - 1) /
      Trait::kRowsPerTile;
  const uint32_t schedule_tiles =
      owner_tile_rounds * Trait::kNumGPU;

  for (uint32_t scheduled_tile = consumer_id;
       scheduled_tile < schedule_tiles;
       scheduled_tile += params.consumer_blocks) {
    const uint32_t logical_owner =
        scheduled_tile % Trait::kNumGPU;
    const uint32_t local_tile =
        scheduled_tile / Trait::kNumGPU;
    const uint32_t owner =
        (owner_start + logical_owner) % Trait::kNumGPU;
    const auto owner_range =
        sglang::jit_kernel::attntp::balanced_row_range(
            actual_rows,
            owner,
            Trait::kNumGPU,
            owner_start);
    const uint32_t local_row_start =
        local_tile * Trait::kRowsPerTile;
    if (local_row_start >= owner_range.count) continue;

    const uint32_t slot =
        local_tile & (Trait::kRingStages - 1);
    const uint64_t expected_ticket =
        tile_ticket<Trait>(params, epoch, local_tile);
    if (threadIdx.x == 0) {
      tile_wait_for_ticket<Trait>(
          tile_rank_signal<Trait>(
              params,
              owner,
              params.ready_offset_bytes +
                  static_cast<uint64_t>(slot) *
                      sizeof(uint64_t)),
          expected_ticket);
    }
    __syncthreads();

    const uint32_t remaining_rows =
        owner_range.count - local_row_start;
    const uint32_t rows_in_tile =
        remaining_rows < Trait::kRowsPerTile
            ? remaining_rows
            : Trait::kRowsPerTile;
    const auto* owner_scratch = tile_byte_offset<float>(
        tile_rank_base<Trait>(params, owner),
        params.reduced_offset_bytes +
            static_cast<uint64_t>(slot) * kScratchTileBytes);
    for (uint32_t row_in_tile = 0;
         row_in_tile < rows_in_tile;
         ++row_in_tile) {
      const uint32_t global_row =
          owner_range.offset + local_row_start + row_in_tile;
      const auto* scratch_row =
          owner_scratch +
          static_cast<uint64_t>(row_in_tile) * kHiddenSize;
      float activation_values[kVectorsPerThread]
                             [kElementsPerVector];
#pragma unroll
      for (uint32_t vector = 0;
           vector < kVectorsPerThread;
           ++vector) {
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          const uint32_t storage_index =
              threadIdx.x * (2 * kVectorsPerThread) +
              vector * 2 + half;
          FP32Storage values;
          values.load(scratch_row, storage_index);
#pragma unroll
          for (uint32_t element = 0;
               element < kElementsPerVector / 2;
               ++element) {
            activation_values[vector]
                             [half * (kElementsPerVector / 2) +
                              element] = values[element];
          }
        }
      }

      const auto* residual_row =
          static_cast<const float*>(params.residual) +
          static_cast<uint64_t>(global_row) * kHiddenSize;
      auto* output_row =
          static_cast<typename Trait::DType*>(params.output) +
          static_cast<uint64_t>(global_row) * kHiddenSize;
      auto* residual_out_row =
          static_cast<float*>(params.residual_out) +
          static_cast<uint64_t>(global_row) * kHiddenSize;
      sglang::jit_kernel::attntp::
          apply_welm_norm_pipeline_full_fp32<Trait>(
              activation_values,
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
      auto* consumed = tile_rank_signal<Trait>(
          params,
          params.rank,
          params.consumed_offset_bytes +
              static_cast<uint64_t>(owner) *
                  params.consumed_owner_stride_bytes +
              static_cast<uint64_t>(slot) * sizeof(uint64_t));
      tile_store_release_system(consumed, expected_ticket);
    }
    __syncthreads();
  }
}

template <
    typename Trait,
    typename CohortTrait,
    typename WeightProvider>
SGL_DEVICE void tile_consume_cohort_schedule(
    const AttnTPReplicatedTilePipelineParams& params,
    uint64_t epoch,
    const WeightProvider& weights,
    float* warp_sums,
    float* shared_totals) {
  using FP32Storage = typename CohortTrait::FP32Storage;

  constexpr uint32_t kHiddenSize = Trait::kHiddenSize;
  constexpr uint32_t kElementsPerVector =
      CohortTrait::kElementsPerVector;
  constexpr uint32_t kVectorsPerThread =
      CohortTrait::kVectorsPerThread;
  constexpr uint64_t kScratchTileBytes =
      static_cast<uint64_t>(Trait::kRowsPerTile) *
      kHiddenSize * sizeof(float);

  const uint32_t consumer_id =
      blockIdx.x - params.producer_blocks;
  const uint32_t cohort =
      sglang::jit_kernel::attntp::cohort_id<CohortTrait>();
  const uint32_t local_thread =
      sglang::jit_kernel::attntp::
          cohort_thread_id<CohortTrait>();
  const uint32_t actual_rows = tile_actual_rows<Trait>(params);
  const uint32_t owner_start = tile_owner_start<Trait>(params);
  const uint32_t max_owner_rows =
      (actual_rows + Trait::kNumGPU - 1) /
      Trait::kNumGPU;
  const uint32_t owner_tile_rounds =
      (max_owner_rows + Trait::kRowsPerTile - 1) /
      Trait::kRowsPerTile;
  const uint32_t schedule_tiles =
      owner_tile_rounds * Trait::kNumGPU;
  const uint32_t schedule_stride =
      params.consumer_blocks * Trait::kConsumerCohorts;

  for (uint32_t scheduled_tile =
           consumer_id * Trait::kConsumerCohorts + cohort;
       scheduled_tile < schedule_tiles;
       scheduled_tile += schedule_stride) {
    const uint32_t logical_owner =
        scheduled_tile % Trait::kNumGPU;
    const uint32_t local_tile =
        scheduled_tile / Trait::kNumGPU;
    const uint32_t owner =
        (owner_start + logical_owner) % Trait::kNumGPU;
    const auto owner_range =
        sglang::jit_kernel::attntp::balanced_row_range(
            actual_rows,
            owner,
            Trait::kNumGPU,
            owner_start);
    const uint32_t local_row_start =
        local_tile * Trait::kRowsPerTile;
    if (local_row_start >= owner_range.count) continue;

    const uint32_t slot =
        local_tile & (Trait::kRingStages - 1);
    const uint64_t expected_ticket =
        tile_ticket<Trait>(params, epoch, local_tile);
    if (local_thread == 0) {
      tile_wait_for_ticket<Trait>(
          tile_rank_signal<Trait>(
              params,
              owner,
              params.ready_offset_bytes +
                  static_cast<uint64_t>(slot) *
                      sizeof(uint64_t)),
          expected_ticket);
    }
    sglang::jit_kernel::attntp::
        cohort_barrier<CohortTrait>();

    const uint32_t remaining_rows =
        owner_range.count - local_row_start;
    const uint32_t rows_in_tile =
        remaining_rows < Trait::kRowsPerTile
            ? remaining_rows
            : Trait::kRowsPerTile;
    const auto* owner_scratch = tile_byte_offset<float>(
        tile_rank_base<Trait>(params, owner),
        params.reduced_offset_bytes +
            static_cast<uint64_t>(slot) * kScratchTileBytes);
    for (uint32_t row_in_tile = 0;
         row_in_tile < rows_in_tile;
         ++row_in_tile) {
      const uint32_t global_row =
          owner_range.offset + local_row_start + row_in_tile;
      const auto* scratch_row =
          owner_scratch +
          static_cast<uint64_t>(row_in_tile) * kHiddenSize;
      float activation_values[kVectorsPerThread]
                             [kElementsPerVector];
#pragma unroll
      for (uint32_t vector = 0;
           vector < kVectorsPerThread;
           ++vector) {
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          const uint32_t storage_index =
              local_thread * (2 * kVectorsPerThread) +
              vector * 2 + half;
          FP32Storage values;
          values.load(scratch_row, storage_index);
#pragma unroll
          for (uint32_t element = 0;
               element < kElementsPerVector / 2;
               ++element) {
            activation_values[vector]
                             [half * (kElementsPerVector / 2) +
                              element] = values[element];
          }
        }
      }

      const auto* residual_row =
          static_cast<const float*>(params.residual) +
          static_cast<uint64_t>(global_row) * kHiddenSize;
      auto* output_row =
          static_cast<typename Trait::DType*>(params.output) +
          static_cast<uint64_t>(global_row) * kHiddenSize;
      auto* residual_out_row =
          static_cast<float*>(params.residual_out) +
          static_cast<uint64_t>(global_row) * kHiddenSize;
      sglang::jit_kernel::attntp::
          apply_welm_norm_pipeline_full_fp32_cohort<
              CohortTrait>(
              activation_values,
              residual_row,
              weights,
              output_row,
              residual_out_row,
              params.o_norm_eps,
              params.post_norm_eps,
              warp_sums,
              shared_totals);
    }

    sglang::jit_kernel::attntp::
        cohort_barrier<CohortTrait>();
    if (local_thread == 0) {
      auto* consumed = tile_rank_signal<Trait>(
          params,
          params.rank,
          params.consumed_offset_bytes +
              static_cast<uint64_t>(owner) *
                  params.consumed_owner_stride_bytes +
              static_cast<uint64_t>(slot) * sizeof(uint64_t));
      tile_store_release_system(consumed, expected_ticket);
    }
    sglang::jit_kernel::attntp::
        cohort_barrier<CohortTrait>();
  }
}

template <typename Trait>
SGL_DEVICE void tile_consume_cohorts(
    const AttnTPReplicatedTilePipelineParams& params,
    uint64_t epoch,
    unsigned char* weight_shared) {
  using DType = typename Trait::DType;
  using BF16Storage = typename Trait::BF16Storage;
  using CohortTrait =
      sglang::jit_kernel::attntp::CohortNormTrait<
          Trait,
          Trait::kConsumerCohorts>;

  __shared__ float warp_sums[Trait::kNumWarps];
  __shared__ float shared_totals[Trait::kConsumerCohorts];

  if constexpr (Trait::kSharedWeights) {
    auto* shared_o_norm =
        reinterpret_cast<DType*>(weight_shared);
    auto* shared_post_norm =
        shared_o_norm + Trait::kHiddenSize;
    constexpr uint32_t kWeightVectors =
        Trait::kHiddenSize / Trait::kElementsPerVector;
    for (uint32_t storage_index = threadIdx.x;
         storage_index < kWeightVectors;
         storage_index += Trait::kBlockSize) {
      BF16Storage o_values;
      BF16Storage post_values;
      o_values.load(params.o_norm_weight, storage_index);
      post_values.load(
          params.post_norm_weight, storage_index);
      o_values.store(shared_o_norm, storage_index);
      post_values.store(shared_post_norm, storage_index);
    }
    __syncthreads();
    const sglang::jit_kernel::attntp::
        CohortSharedWeightProvider<CohortTrait>
            weights{shared_o_norm, shared_post_norm};
    tile_consume_cohort_schedule<
        Trait,
        CohortTrait>(
        params,
        epoch,
        weights,
        warp_sums,
        shared_totals);
  } else {
    sglang::jit_kernel::attntp::
        CohortRegisterWeightProvider<CohortTrait> weights;
    weights.load(
        params.o_norm_weight, params.post_norm_weight);
    tile_consume_cohort_schedule<
        Trait,
        CohortTrait>(
        params,
        epoch,
        weights,
        warp_sums,
        shared_totals);
  }
}

template <typename Trait>
__global__ SGL_ATTNTP_TILE_LAUNCH_BOUNDS void
attntp_replicated_tile_pipeline_kernel(
    const AttnTPReplicatedTilePipelineParams
        __grid_constant__ params) {
  if constexpr (!Trait::kDecode) {
    const int32_t actual_rows_signed = *params.actual_rows;
    const int32_t owner_start_signed = *params.owner_start;
    if (actual_rows_signed < 0 ||
        static_cast<uint32_t>(actual_rows_signed) >
            params.capacity ||
        owner_start_signed < 0 ||
        owner_start_signed >=
            static_cast<int32_t>(Trait::kNumGPU)) {
      if (threadIdx.x == 0) asm volatile("trap;");
      return;
    }
  }

  const uint64_t previous_epoch =
      tile_load_acquire_system(tile_rank_signal<Trait>(
          params,
          params.rank,
          params.kernel_done_offset_bytes));
  const uint64_t current_epoch = previous_epoch + 1;
  tile_wait_for_inputs<Trait>(params, current_epoch);

  extern __shared__ __align__(16)
      unsigned char weight_shared[];
  if (blockIdx.x < params.producer_blocks) {
    tile_produce<Trait>(params, current_epoch);
  } else {
    if constexpr (
        Trait::kConsumerCohorts == 1 &&
        !Trait::kSharedWeights) {
      tile_consume<Trait>(params, current_epoch);
    } else {
      tile_consume_cohorts<Trait>(
          params, current_epoch, weight_shared);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    const uint64_t completion = tile_atomic_add_gpu(
        tile_rank_signal<Trait>(
            params,
            params.rank,
            params.completion_counter_offset_bytes),
        1);
    if (completion == gridDim.x - 1) {
      __threadfence_system();
      tile_store_release_system(
          tile_rank_signal<Trait>(
              params,
              params.rank,
              params.kernel_done_offset_bytes),
          current_epoch);
#pragma unroll
      for (uint32_t rank = 0;
           rank < Trait::kNumGPU;
           ++rank) {
        tile_wait_for_ticket<Trait>(
            tile_rank_signal<Trait>(
                params,
                rank,
                params.kernel_done_offset_bytes),
            current_epoch);
      }
    }
  }
}

template <
    typename DType,
    uint32_t kNumGPU,
    uint32_t kHiddenSize,
    bool kDecode,
    uint32_t kRowsPerTile,
    uint32_t kRingStages,
    uint32_t kBlockSize,
    uint32_t kSignalBackoff,
    uint32_t kLaunchBoundsMinBlocks,
    uint32_t kConsumerCohorts,
    bool kSharedWeights>
struct FusedAttnTPReplicatedTilePipeline {
  using Trait = AttnTPReplicatedTilePipelineTrait<
      DType,
      kNumGPU,
      kHiddenSize,
      kDecode,
      kRowsPerTile,
      kRingStages,
      kBlockSize,
      kSignalBackoff,
      kLaunchBoundsMinBlocks,
      kConsumerCohorts,
      kSharedWeights>;
  static constexpr auto kernel =
      attntp_replicated_tile_pipeline_kernel<Trait>;

  static void run(
      const tvm::ffi::Tensor partial,
      const int64_t pointer_table_i64,
      const int64_t reduced_offset_bytes_i64,
      const int64_t input_ready_offset_bytes_i64,
      const int64_t group_ready_offset_bytes_i64,
      const int64_t kernel_done_offset_bytes_i64,
      const int64_t completion_counter_offset_bytes_i64,
      const int64_t ready_offset_bytes_i64,
      const int64_t consumed_offset_bytes_i64,
      const int64_t consumed_owner_stride_bytes_i64,
      const tvm::ffi::Tensor residual,
      const tvm::ffi::Tensor o_norm_weight,
      const tvm::ffi::Tensor post_norm_weight,
      const tvm::ffi::Tensor output,
      const tvm::ffi::Tensor residual_out,
      const tvm::ffi::Optional<tvm::ffi::TensorView> actual_rows,
      const tvm::ffi::Optional<tvm::ffi::TensorView> owner_start,
      const int64_t rank_i64,
      const int64_t blocks_per_sm_i64,
      const int64_t producer_blocks_per_sm_i64,
      const float o_norm_eps,
      const float post_norm_eps) {
    using namespace host;

    auto capacity_symbol = SymbolicSize{"capacity"};
    auto device_symbol = SymbolicDevice{};
    device_symbol.set_options<kDLCUDA>();
    TensorMatcher({capacity_symbol, kHiddenSize})
        .with_strides({kHiddenSize, 1})
        .with_dtype<DType>()
        .with_device(device_symbol)
        .verify(partial)
        .verify(output);
    TensorMatcher({capacity_symbol, kHiddenSize})
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
    if constexpr (Trait::kDecode) {
      RuntimeCheck(
          !actual_rows.has_value() && !owner_start.has_value(),
          "Decode tile pipeline must not receive Prefill metadata");
    } else {
      RuntimeCheck(
          actual_rows.has_value() && owner_start.has_value(),
          "Prefill tile pipeline requires actual_rows and owner_start");
      TensorMatcher({1})
          .with_dtype<int32_t>()
          .with_device(device_symbol)
          .verify(actual_rows.value())
          .verify(owner_start.value());
    }

    const int64_t capacity_i64 = capacity_symbol.unwrap();
    RuntimeCheck(
        capacity_i64 > 0 && capacity_i64 <= UINT32_MAX,
        "tile pipeline capacity must fit uint32");
    if constexpr (Trait::kDecode) {
      RuntimeCheck(
          capacity_i64 <= 256,
          "tile pipeline Decode rows must be in [1, 256]");
    }
    RuntimeCheck(
        pointer_table_i64 > 0,
        "tile pipeline pointer table must be non-zero");
    RuntimeCheck(
        rank_i64 >= 0 && rank_i64 < Trait::kNumGPU,
        "tile pipeline rank is outside the AttnTP group");
    RuntimeCheck(
        blocks_per_sm_i64 >= 0 &&
            producer_blocks_per_sm_i64 >= 0,
        "tile pipeline block counts must be non-negative");

    const int64_t input_bytes =
        capacity_i64 * kHiddenSize *
        static_cast<int64_t>(sizeof(DType));
    constexpr int64_t kScratchBytes =
        static_cast<int64_t>(Trait::kRingStages) *
        Trait::kRowsPerTile * kHiddenSize *
        static_cast<int64_t>(sizeof(float));
    RuntimeCheck(
        reduced_offset_bytes_i64 >= input_bytes &&
            reduced_offset_bytes_i64 % 128 == 0,
        "tile pipeline reduced storage offset is invalid");
    RuntimeCheck(
        input_ready_offset_bytes_i64 >=
            reduced_offset_bytes_i64 + kScratchBytes &&
            input_ready_offset_bytes_i64 % alignof(uint64_t) == 0,
        "tile pipeline control storage overlaps reduced tiles");
    RuntimeCheck(
        group_ready_offset_bytes_i64 ==
            input_ready_offset_bytes_i64 +
                static_cast<int64_t>(sizeof(uint64_t)) &&
            kernel_done_offset_bytes_i64 ==
                group_ready_offset_bytes_i64 +
                    static_cast<int64_t>(sizeof(uint64_t)) &&
            completion_counter_offset_bytes_i64 ==
                kernel_done_offset_bytes_i64 +
                    static_cast<int64_t>(sizeof(uint64_t)),
        "tile pipeline control header layout mismatch");
    RuntimeCheck(
        ready_offset_bytes_i64 >
            completion_counter_offset_bytes_i64 &&
            ready_offset_bytes_i64 % 128 == 0,
        "tile pipeline ready-ticket offset is invalid");
    RuntimeCheck(
        consumed_offset_bytes_i64 >=
            ready_offset_bytes_i64 +
                static_cast<int64_t>(Trait::kRingStages) *
                    static_cast<int64_t>(sizeof(uint64_t)) &&
            consumed_owner_stride_bytes_i64 >=
                static_cast<int64_t>(Trait::kRingStages) *
                    static_cast<int64_t>(sizeof(uint64_t)) &&
            consumed_owner_stride_bytes_i64 % 128 == 0,
        "tile pipeline consumed-ticket layout is invalid");

    for (const auto pointer :
         {reinterpret_cast<uintptr_t>(partial.data_ptr()),
          reinterpret_cast<uintptr_t>(residual.data_ptr()),
          reinterpret_cast<uintptr_t>(
              o_norm_weight.data_ptr()),
          reinterpret_cast<uintptr_t>(
              post_norm_weight.data_ptr()),
          reinterpret_cast<uintptr_t>(output.data_ptr()),
          reinterpret_cast<uintptr_t>(
              residual_out.data_ptr())}) {
      RuntimeCheck(
          pointer % 16 == 0,
          "tile pipeline tensors must be 16-byte aligned");
    }

    const auto device = device_symbol.unwrap();
    int device_id = 0;
    RuntimeDeviceCheck(cudaGetDevice(&device_id));
    int cooperative_launch = 0;
    RuntimeDeviceCheck(cudaDeviceGetAttribute(
        &cooperative_launch,
        cudaDevAttrCooperativeLaunch,
        device_id));
    RuntimeCheck(
        cooperative_launch != 0,
        "tile pipeline requires CUDA cooperative launch support");
    const uint32_t occupancy = get_max_occupancy();
    const uint32_t selected_blocks_per_sm =
        blocks_per_sm_i64 == 0
            ? occupancy
            : static_cast<uint32_t>(blocks_per_sm_i64);
    RuntimeCheck(
        selected_blocks_per_sm >= 2 &&
            selected_blocks_per_sm <= occupancy,
        "tile pipeline requires at least two resident blocks per SM");
    const uint32_t producer_blocks_per_sm =
        producer_blocks_per_sm_i64 == 0
            ? std::max<uint32_t>(
                  1, selected_blocks_per_sm / 2)
            : static_cast<uint32_t>(
                  producer_blocks_per_sm_i64);
    RuntimeCheck(
        producer_blocks_per_sm > 0 &&
            producer_blocks_per_sm <
                selected_blocks_per_sm,
        "tile pipeline must reserve resident consumer blocks");

    const uint32_t sm_count =
        runtime::get_sm_count(device_id);
    const uint32_t max_resident_blocks =
        selected_blocks_per_sm * sm_count;
    const uint32_t capacity =
        static_cast<uint32_t>(capacity_i64);
    const uint32_t owner_capacity =
        div_ceil(capacity, Trait::kNumGPU);
    const uint32_t max_owner_tiles =
        div_ceil(owner_capacity, Trait::kRowsPerTile);
    const uint32_t producer_blocks = std::min(
        {max_owner_tiles,
         Trait::kRingStages,
         producer_blocks_per_sm * sm_count});
    const uint32_t max_valid_tiles =
        div_ceil(capacity, Trait::kRowsPerTile);
    const uint32_t max_consumer_blocks =
        div_ceil(
            max_valid_tiles,
            Trait::kConsumerCohorts);
    const uint32_t consumer_blocks = std::min(
        max_consumer_blocks,
        (selected_blocks_per_sm -
         producer_blocks_per_sm) *
            sm_count);
    const uint32_t grid_blocks =
        producer_blocks + consumer_blocks;
    RuntimeCheck(
        producer_blocks > 0 && consumer_blocks > 0,
        "tile pipeline requires producer and consumer blocks");
    RuntimeCheck(
        grid_blocks <= max_resident_blocks,
        "tile pipeline grid exceeds resident capacity");

    AttnTPReplicatedTilePipelineParams params{};
    params.pointer_table =
        reinterpret_cast<const void* const*>(
            static_cast<uintptr_t>(pointer_table_i64));
    params.local_base = partial.data_ptr();
    params.residual = residual.data_ptr();
    params.o_norm_weight = o_norm_weight.data_ptr();
    params.post_norm_weight = post_norm_weight.data_ptr();
    params.output = output.data_ptr();
    params.residual_out = residual_out.data_ptr();
    if constexpr (!Trait::kDecode) {
      params.actual_rows =
          static_cast<const int32_t*>(
              actual_rows.value().data_ptr());
      params.owner_start =
          static_cast<const int32_t*>(
              owner_start.value().data_ptr());
    }
    params.reduced_offset_bytes =
        static_cast<uint64_t>(
            reduced_offset_bytes_i64);
    params.input_ready_offset_bytes =
        static_cast<uint64_t>(
            input_ready_offset_bytes_i64);
    params.group_ready_offset_bytes =
        static_cast<uint64_t>(
            group_ready_offset_bytes_i64);
    params.kernel_done_offset_bytes =
        static_cast<uint64_t>(
            kernel_done_offset_bytes_i64);
    params.completion_counter_offset_bytes =
        static_cast<uint64_t>(
            completion_counter_offset_bytes_i64);
    params.ready_offset_bytes =
        static_cast<uint64_t>(
            ready_offset_bytes_i64);
    params.consumed_offset_bytes =
        static_cast<uint64_t>(
            consumed_offset_bytes_i64);
    params.consumed_owner_stride_bytes =
        static_cast<uint64_t>(
            consumed_owner_stride_bytes_i64);
    params.o_norm_eps = o_norm_eps;
    params.post_norm_eps = post_norm_eps;
    params.rank = static_cast<uint32_t>(rank_i64);
    params.capacity = capacity;
    params.producer_blocks = producer_blocks;
    params.consumer_blocks = consumer_blocks;

    const auto stream = static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(
            device.device_type,
            device.device_id));
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeCooperative;
    attr.val.cooperative = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(grid_blocks);
    config.blockDim = dim3(Trait::kBlockSize);
    config.dynamicSmemBytes = Trait::kWeightSharedBytes;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    RuntimeDeviceCheck(
        cudaLaunchKernelEx(&config, kernel, params));
  }

  static uint32_t get_max_occupancy() {
    return host::runtime::get_blocks_per_sm(
        kernel, Trait::kBlockSize, Trait::kWeightSharedBytes);
  }
};

}  // namespace

#undef SGL_ATTNTP_TILE_LAUNCH_BOUNDS
