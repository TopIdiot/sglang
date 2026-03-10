from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def assign_ngram_input_ids_kernel(
    input_ids: torch.Tensor,
    input_ids_gram: torch.Tensor,
    gram_n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRAM_BLOCK_SIZE: tl.constexpr,
    extend_len: torch.Tensor,
):
    data_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    gram_offset = gram_n - 1
    mask = data_offset < extend_len
    data = tl.load(input_ids + data_offset, mask)
    tl.store(
        input_ids_gram + data_offset + gram_offset,
        data,
        data_offset + gram_offset < extend_len,
    )
    prefix_offset = tl.arange(0, GRAM_BLOCK_SIZE)
    tl.store(
        input_ids_gram + prefix_offset,
        0,
        prefix_offset < tl.minimum(gram_offset, extend_len),
    )


def assign_ngram_input_ids_draft_extend(
    input_ids: torch.Tensor,
    input_ids_gram: torch.Tensor,
    extend_lens: List[int],
    gram_n: int,
):
    pt = 0
    GRAM_BLOCK_SIZE = triton.next_power_of_2(gram_n - 1)
    for i, extend_len in enumerate(extend_lens):
        grid = triton.cdiv(extend_len, 128)
        assign_ngram_input_ids_kernel[(grid,)](
            input_ids[pt : pt + extend_len],
            input_ids_gram[pt : pt + extend_len],
            gram_n,
            128,
            GRAM_BLOCK_SIZE,
            extend_len,
        )
        pt += extend_len


@triton.jit
def assign_ngram_input_ids_draft_decode_first_token_kernel(
    input_ids_buffer: torch.Tensor,
    input_ids_gram_decode: torch.Tensor,
    buffer_size: tl.constexpr,
    gram_n: tl.constexpr,
    topk: tl.constexpr,
    topk_block: tl.constexpr,
):
    bid = tl.program_id(0)

    gram_offset = bid * buffer_size + buffer_size - gram_n + 1
    data = tl.load(input_ids_buffer + gram_offset)

    repeat_data = tl.full((topk_block,), data, dtype=input_ids_gram_decode.dtype)
    save_offset = tl.arange(0, topk_block)
    tl.store(
        input_ids_gram_decode + bid * topk + save_offset,
        repeat_data,
        save_offset < topk,
    )


def assign_ngram_input_ids_draft_decode_first_token(
    input_ids_buffer: torch.Tensor,
    input_ids_gram_decode: torch.Tensor,
    seq_lens: torch.Tensor,
    gram_n: int,
    topk: int,
    buffer_size: int,
):
    bs = seq_lens.numel()
    topk_block = triton.next_power_of_2(topk)
    assign_ngram_input_ids_draft_decode_first_token_kernel[(bs,)](
        input_ids_buffer,
        input_ids_gram_decode,
        buffer_size,
        gram_n,
        topk,
        topk_block,
    )


@triton.jit
def assign_ngram_buffer_kernel(
    input_ids: torch.Tensor,
    buffer: torch.Tensor,
    seq_lens: torch.Tensor,
    buffer_size: tl.constexpr,
    bs_block: tl.constexpr,
    buffer_size_block: tl.constexpr,
):
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_block)
    cu_seq_len = tl.sum(tl.load(seq_lens + bs_offset, bs_offset <= bid))
    seq_len = tl.load(seq_lens + bid)
    pre_bound = cu_seq_len - seq_len
    data_offset = tl.arange(0, buffer_size_block)
    load_offset = cu_seq_len - buffer_size + data_offset
    mask = (data_offset < buffer_size) & (load_offset >= pre_bound)
    data = tl.load(input_ids + load_offset, mask, other=0)
    tl.store(buffer + bid * buffer_size + data_offset, data, data_offset < buffer_size)


def assign_ngram_buffer(
    input_ids: torch.Tensor,
    buffer: torch.Tensor,
    seq_lens: torch.Tensor,
    buffer_size: int,
):
    bs = seq_lens.numel()
    bs_block = triton.next_power_of_2(bs)
    buffer_size_block = triton.next_power_of_2(buffer_size)
    assign_ngram_buffer_kernel[(bs,)](
        input_ids, buffer, seq_lens, buffer_size, bs_block, buffer_size_block
    )


@triton.jit
def assign_buffer_kernel(
    buffer: torch.Tensor,
    new_buffer: torch.Tensor,
    keep_indices: torch.Tensor,
    buffer_size: tl.constexpr,
    buffer_block: tl.constexpr,
):
    bid = tl.program_id(0)
    offset = tl.load(keep_indices + bid)
    buffer_offset = tl.arange(0, buffer_block)
    data = tl.load(
        buffer + offset * buffer_size + buffer_offset, buffer_offset < buffer_size
    )
    tl.store(
        new_buffer + bid * buffer_size + buffer_offset,
        data,
        buffer_offset < buffer_size,
    )


def filter_buffer(buffer: torch.Tensor, keep_indices: torch.Tensor, buffer_size: int):
    size = keep_indices.numel()
    new_buffer = torch.empty(
        (size * buffer_size), device=buffer.device, dtype=buffer.dtype
    )
    buffer_block = triton.next_power_of_2(buffer_size)
    assign_buffer_kernel[(size,)](
        buffer, new_buffer, keep_indices, buffer_size, buffer_block
    )
    return new_buffer
