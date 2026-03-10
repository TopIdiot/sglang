// N-gram operations for speculative decoding.
// Provides build_ngram_with_tree, build_ngram_with_target_verify,
// and assign_ngram_input_ids_draft_extend_after_decode kernels.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// CUDA Kernels
// ---------------------------------------------------------------------------

__global__ void build_ngram_with_tree_kernel(
    int64_t* ngram_input_ids,
    int64_t* parent_list,
    int64_t* token_list,
    int64_t* current_parrent_list,
    int64_t* buffer,
    int topk,
    int gram_n,
    int buffer_size,
    int i,
    int parent_list_stride,
    int token_list_stride) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (tid >= topk) {
    return;
  }
  int64_t current_pos = current_parrent_list[bid * topk + tid];
  int gram = gram_n - 1 - i;
  if (gram > 0) {
    ngram_input_ids[bid * topk + tid] = buffer[(bid + 1) * buffer_size - gram];
    return;
  }
  int64_t parent_token;
  int layer = i;
  for (int gram_ids = 0; gram_ids < gram_n - 1; gram_ids++) {
    int pre_layer_num_node = topk + topk * topk * (layer - 1);
    int cur_layer_pos = static_cast<int>(current_pos) - pre_layer_num_node;
    int parent_layer_pos = cur_layer_pos / topk;
    int parent_offset = 1 + topk * (layer - 1);
    int parent_pos = parent_layer_pos + parent_offset;
    parent_pos = static_cast<int>(parent_list[bid * parent_list_stride + parent_pos]);
    parent_token = token_list[bid * token_list_stride + parent_pos];
    current_pos = parent_pos;
    layer--;
  }
  ngram_input_ids[bid * topk + tid] = parent_token;
}

__global__ void build_target_verify_ngram_kernel(
    int64_t* ngram_input_ids,
    int64_t* buffer,
    int64_t* draft_token_ids,
    bool* tree_mask,
    int64_t* positions,
    int64_t* seq_lens,
    int gram_n,
    int draft_token_num,
    int buffer_size) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (tid != 0) {
    return;
  }
  int seq_id = bid / draft_token_num;
  int64_t mask_offset = 0;
  for (int j = 0; j < seq_id; j++) {
    int64_t mask_len = seq_lens[j] + draft_token_num;
    mask_offset += draft_token_num * mask_len;
  }
  int64_t seq_len = seq_lens[seq_id];
  int64_t mask_len = seq_len + draft_token_num;
  mask_offset += (bid % draft_token_num) * mask_len;

  int target_gram = gram_n;
  int64_t res = 0;
  for (int64_t idx = seq_len + draft_token_num - 1; idx >= seq_len; idx--) {
    if (tree_mask[mask_offset + idx]) {
      target_gram--;
      if (target_gram == 0) {
        res = draft_token_ids[seq_id * draft_token_num + idx - seq_len];
        break;
      }
    }
  }
  if (target_gram != 0) {
    res = buffer[(seq_id + 1) * buffer_size - target_gram - 1];
  }
  ngram_input_ids[bid] = res;
}

__global__ void assign_ngram_input_ids_draft_extend_after_decode_kernel(
    int64_t* input_ids,
    int64_t* buffer,
    int64_t* input_ids_gram,
    int32_t* accept_length,
    int gram_n,
    int buffer_size,
    bool update_buffer) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  int gram = gram_n - 1;
  int accum_accept_len = 0;
  for (int j = 0; j < bid; j++) {
    accum_accept_len += accept_length[j];
  }
  int curr_accept_len = accept_length[bid];
  if (tid < curr_accept_len) {
    if (tid >= gram) {
      input_ids_gram[accum_accept_len + tid] = input_ids[accum_accept_len + tid - gram];
    } else {
      input_ids_gram[accum_accept_len + tid] = buffer[bid * buffer_size + buffer_size - (gram - tid)];
    }
  }
  // Early return: buffer update path is disabled in original code
  if (true) {
    return;
  }

  if (tid >= buffer_size) {
    return;
  }
  int64_t new_buffer[10];
  int remained_size = buffer_size - curr_accept_len;
  if (tid < remained_size) {
    new_buffer[tid] = buffer[bid * buffer_size + buffer_size - remained_size + tid];
  } else {
    new_buffer[tid] = input_ids[accum_accept_len + tid - remained_size];
  }
  buffer[bid * buffer_size + tid] = new_buffer[tid];
}

// ---------------------------------------------------------------------------
// Host-side wrapper functions (tvm_ffi interface)
// ---------------------------------------------------------------------------

void build_ngram_with_tree(
    tvm::ffi::TensorView ngram_input_ids,
    tvm::ffi::TensorView parent_list,
    tvm::ffi::TensorView token_list,
    tvm::ffi::TensorView current_parrent_list,
    tvm::ffi::TensorView buffer,
    int64_t buffer_size,
    int64_t gram_n,
    int64_t topk,
    int64_t i) {
  using namespace host;

  SymbolicDevice device_;
  TensorMatcher({details::kAnySize}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(ngram_input_ids);

  const DLDevice device = device_.unwrap();
  int bs = static_cast<int>(parent_list.size(0));
  int parent_list_stride = static_cast<int>(parent_list.stride(0));
  int token_list_stride = static_cast<int>(token_list.stride(0));

  LaunchKernel(bs, 32, device)(
      build_ngram_with_tree_kernel,
      static_cast<int64_t*>(ngram_input_ids.data_ptr()),
      static_cast<int64_t*>(parent_list.data_ptr()),
      static_cast<int64_t*>(token_list.data_ptr()),
      static_cast<int64_t*>(current_parrent_list.data_ptr()),
      static_cast<int64_t*>(buffer.data_ptr()),
      static_cast<int>(topk),
      static_cast<int>(gram_n),
      static_cast<int>(buffer_size),
      static_cast<int>(i),
      parent_list_stride,
      token_list_stride);
}

void build_ngram_with_target_verify(
    tvm::ffi::TensorView ngram_input_ids,
    tvm::ffi::TensorView buffer,
    tvm::ffi::TensorView draft_token_ids,
    tvm::ffi::TensorView tree_mask,
    tvm::ffi::TensorView positions,
    tvm::ffi::TensorView seq_lens,
    int64_t gram_n,
    int64_t draft_token_num,
    int64_t buffer_size) {
  using namespace host;

  SymbolicDevice device_;
  TensorMatcher({details::kAnySize}).with_dtype<int64_t>().with_device<kDLCUDA>(device_).verify(seq_lens);

  const DLDevice device = device_.unwrap();
  int bs = static_cast<int>(seq_lens.size(0));

  LaunchKernel(bs * static_cast<int>(draft_token_num), 32, device)(
      build_target_verify_ngram_kernel,
      static_cast<int64_t*>(ngram_input_ids.data_ptr()),
      static_cast<int64_t*>(buffer.data_ptr()),
      static_cast<int64_t*>(draft_token_ids.data_ptr()),
      static_cast<bool*>(tree_mask.data_ptr()),
      static_cast<int64_t*>(positions.data_ptr()),
      static_cast<int64_t*>(seq_lens.data_ptr()),
      static_cast<int>(gram_n),
      static_cast<int>(draft_token_num),
      static_cast<int>(buffer_size));
}

void assign_ngram_input_ids_draft_extend_after_decode(
    tvm::ffi::TensorView input_ids,
    tvm::ffi::TensorView buffer,
    tvm::ffi::TensorView input_ids_gram,
    tvm::ffi::TensorView accept_length,
    int64_t gram_n,
    int64_t buffer_size,
    int64_t update_buffer) {
  using namespace host;

  RuntimeCheck(buffer_size < 10, "buffer_size should be less than 10, got ", buffer_size);

  SymbolicDevice device_;
  TensorMatcher({details::kAnySize}).with_dtype<int32_t>().with_device<kDLCUDA>(device_).verify(accept_length);

  const DLDevice device = device_.unwrap();
  int bs = static_cast<int>(accept_length.size(0));

  LaunchKernel(bs, 32, device)(
      assign_ngram_input_ids_draft_extend_after_decode_kernel,
      static_cast<int64_t*>(input_ids.data_ptr()),
      static_cast<int64_t*>(buffer.data_ptr()),
      static_cast<int64_t*>(input_ids_gram.data_ptr()),
      static_cast<int32_t*>(accept_length.data_ptr()),
      static_cast<int>(gram_n),
      static_cast<int>(buffer_size),
      static_cast<bool>(update_buffer != 0));
}

}  // namespace
