from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _translate_cache_loc_kernel(
    full_cache_loc,
    full_to_swa,
    swa_cache_loc,
    num_tokens,
    mapping_size,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_tokens
    full_loc = tl.load(full_cache_loc + offsets, mask=mask, other=0).to(tl.int64)
    full_loc = tl.where(full_loc < 0, full_loc + mapping_size, full_loc)
    swa_loc = tl.load(full_to_swa + full_loc, mask=mask, other=0)
    tl.store(swa_cache_loc + offsets, swa_loc, mask=mask)


def translate_welm_mtp_cache_loc(
    *,
    full_cache_loc: torch.Tensor,
    full_to_swa: torch.Tensor,
    swa_cache_loc: torch.Tensor,
) -> None:
    """Translate fixed proposal-graph cache locations with one graphable launch."""
    num_tokens = int(full_cache_loc.numel())
    if num_tokens == 0:
        return
    if int(swa_cache_loc.numel()) != num_tokens:
        raise ValueError("Full and SWA cache-location buffers must have equal size.")
    if swa_cache_loc.dtype != torch.int32:
        raise ValueError("The SWA cache-location buffer must use torch.int32.")
    block_size = 256
    _translate_cache_loc_kernel[(triton.cdiv(num_tokens, block_size),)](
        full_cache_loc,
        full_to_swa,
        swa_cache_loc,
        num_tokens,
        int(full_to_swa.numel()),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )


@triton.jit
def _pack_graph_inputs_kernel(
    input_ids,
    out_cache_loc,
    positions,
    hidden_states,
    mirrored_kv_indices,
    cached_hash,
    accept_lens,
    extend_start_loc,
    output_input_ids,
    output_cache_loc,
    output_positions,
    output_hidden_states,
    output_mirrored_kv_indices,
    output_hash,
    custom_last_index,
    custom_last_cache_loc,
    raw_bs,
    input_hash_stride,
    output_hash_stride,
    mirror_padding_index,
    TOKENS_PER_BS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    NUM_BRANCHES: tl.constexpr,
    HAS_MIRROR_INPUT: tl.constexpr,
    HAS_HASH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    dst_idx = tl.program_id(0)
    row = dst_idx // TOKENS_PER_BS
    local_pos = dst_idx % TOKENS_PER_BS
    real_row = row < raw_bs
    real_len = tl.load(accept_lens + row, mask=real_row, other=0).to(tl.int64)
    src_start = tl.load(extend_start_loc + row, mask=real_row, other=0).to(tl.int64)
    valid = real_row & (local_pos < real_len)
    has_real_token = real_row & (real_len > 0)
    pad = has_real_token & (local_pos >= real_len)
    last_src_idx = src_start + tl.maximum(real_len - 1, 0)
    src_idx = tl.where(valid, src_start + local_pos, last_src_idx)

    token = tl.load(input_ids + src_idx, mask=valid | pad, other=0)
    position = tl.load(positions + src_idx, mask=valid | pad, other=0)
    position += tl.where(pad, local_pos - real_len + 1, 0)
    cache_loc = tl.load(out_cache_loc + src_idx, mask=valid, other=0)
    tl.store(output_input_ids + dst_idx, token)
    tl.store(output_positions + dst_idx, position)
    tl.store(output_cache_loc + dst_idx, cache_loc)

    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    hidden = tl.load(
        hidden_states + src_idx * HIDDEN_SIZE + hidden_offsets,
        mask=valid & hidden_mask,
        other=0.0,
    )
    tl.store(
        output_hidden_states + dst_idx * HIDDEN_SIZE + hidden_offsets,
        hidden,
        mask=hidden_mask,
    )

    if HAS_MIRROR_INPUT:
        mirror_idx = tl.load(mirrored_kv_indices + src_idx, mask=valid, other=0)
    else:
        mirror_idx = src_idx
    mirror_idx = tl.where(valid, mirror_idx, mirror_padding_index)
    tl.store(output_mirrored_kv_indices + dst_idx, mirror_idx)

    if HAS_HASH:
        branch_offsets = tl.arange(0, BLOCK_B)
        branch_mask = branch_offsets < NUM_BRANCHES
        hash_values = tl.load(
            cached_hash + branch_offsets * input_hash_stride + src_idx,
            mask=valid & branch_mask,
            other=0,
        )
        tl.store(
            output_hash + branch_offsets * output_hash_stride + dst_idx,
            hash_values,
            mask=branch_mask,
        )

    write_row = local_pos == 0
    last_dst_idx = row * TOKENS_PER_BS + tl.where(
        real_len > 0, real_len - 1, TOKENS_PER_BS - 1
    )
    last_cache_loc = tl.load(
        out_cache_loc + last_src_idx,
        mask=has_real_token,
        other=0,
    )
    tl.store(custom_last_index + row, last_dst_idx, mask=write_row)
    tl.store(custom_last_cache_loc + row, last_cache_loc, mask=write_row)


def pack_welm_mtp_graph_inputs(
    *,
    input_ids: torch.Tensor,
    out_cache_loc: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    mirrored_kv_indices: Optional[torch.Tensor],
    cached_hash: Optional[torch.Tensor],
    accept_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    output_input_ids: torch.Tensor,
    output_cache_loc: torch.Tensor,
    output_positions: torch.Tensor,
    output_hidden_states: torch.Tensor,
    output_mirrored_kv_indices: torch.Tensor,
    output_hash: Optional[torch.Tensor],
    custom_last_index: torch.Tensor,
    custom_last_cache_loc: torch.Tensor,
    raw_bs: int,
    graph_num_tokens: int,
    tokens_per_bs: int,
    mirror_padding_index: int,
) -> None:
    """Pack ragged accepted tokens directly into fixed CUDA-graph rows."""
    if graph_num_tokens <= 0:
        return
    if graph_num_tokens % tokens_per_bs != 0:
        raise ValueError(
            f"graph_num_tokens={graph_num_tokens} is not divisible by "
            f"tokens_per_bs={tokens_per_bs}."
        )
    hidden_size = int(hidden_states.shape[-1])
    if hidden_size != int(output_hidden_states.shape[-1]):
        raise ValueError("Input/output hidden sizes do not match.")
    num_branches = int(cached_hash.shape[0]) if cached_hash is not None else 0
    if (cached_hash is None) != (output_hash is None):
        raise ValueError("cached_hash and output_hash must be both present or absent.")
    if output_hash is not None and int(output_hash.shape[0]) != num_branches:
        raise ValueError("Input/output OE hash branch counts do not match.")

    mirror_input = mirrored_kv_indices if mirrored_kv_indices is not None else input_ids
    hash_input = cached_hash if cached_hash is not None else input_ids
    hash_output = output_hash if output_hash is not None else output_input_ids
    block_h = triton.next_power_of_2(hidden_size)
    block_b = triton.next_power_of_2(max(1, num_branches))
    _pack_graph_inputs_kernel[(graph_num_tokens,)](
        input_ids,
        out_cache_loc,
        positions,
        hidden_states,
        mirror_input,
        hash_input,
        accept_lens,
        extend_start_loc,
        output_input_ids,
        output_cache_loc,
        output_positions,
        output_hidden_states,
        output_mirrored_kv_indices,
        hash_output,
        custom_last_index,
        custom_last_cache_loc,
        raw_bs,
        int(cached_hash.stride(0)) if cached_hash is not None else 0,
        int(output_hash.stride(0)) if output_hash is not None else 0,
        mirror_padding_index,
        TOKENS_PER_BS=tokens_per_bs,
        HIDDEN_SIZE=hidden_size,
        NUM_BRANCHES=num_branches,
        HAS_MIRROR_INPUT=mirrored_kv_indices is not None,
        HAS_HASH=cached_hash is not None,
        BLOCK_H=block_h,
        BLOCK_B=block_b,
        num_warps=8 if hidden_size >= 1024 else 4,
    )


@triton.jit
def _pack_verify_handoff_kernel(
    predict,
    accept_index,
    accept_lens,
    target_cache_loc,
    target_hidden_states,
    old_seq_lens,
    req_pool_indices,
    bonus_tokens,
    output_input_ids,
    output_cache_loc,
    output_positions,
    output_hidden_states,
    output_mirrored_kv_indices,
    output_seq_lens,
    output_req_pool_indices,
    output_first_input_ids,
    output_base_positions,
    custom_last_index,
    custom_last_cache_loc,
    raw_bs,
    mirror_padding_index,
    seq_len_fill_value,
    TOKENS_PER_BS: tl.constexpr,
    ACCEPT_WIDTH: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Write target-verify outputs directly into fixed draft-graph rows.

    The generic path first materializes a ragged accepted-token batch and then
    expanded it back to fixed-width CUDA-graph rows.  For top-k=1 MTP the
    accepted path already lives in ``accept_index``; doing the gather here
    removes both intermediate tensors and all of their Python bookkeeping.
    """
    dst_idx = tl.program_id(0)
    row = dst_idx // TOKENS_PER_BS
    local_pos = dst_idx % TOKENS_PER_BS
    real_row = row < raw_bs
    real_len = tl.load(accept_lens + row, mask=real_row, other=0).to(tl.int64)
    real_len = tl.minimum(tl.maximum(real_len, 1), TOKENS_PER_BS)
    valid = real_row & (local_pos < real_len)
    pad = real_row & (local_pos >= real_len)
    selected_pos = tl.minimum(local_pos, real_len - 1)
    accept_slot = row * ACCEPT_WIDTH + selected_pos
    src_idx = tl.load(accept_index + accept_slot, mask=real_row, other=0).to(tl.int64)
    # The first ``real_len`` entries are required to be valid.  Keep a safe
    # source for release builds as malformed accept data is diagnosed by the
    # existing optional OOB assertions in the caller.
    src_idx = tl.maximum(src_idx, 0)

    token = tl.load(predict + src_idx, mask=real_row, other=0)
    old_seq_len = tl.load(old_seq_lens + row, mask=real_row, other=0).to(tl.int64)
    cache_loc = tl.load(target_cache_loc + src_idx, mask=valid, other=0)
    tl.store(output_input_ids + dst_idx, token)
    tl.store(
        output_positions + dst_idx,
        tl.where(real_row, old_seq_len + local_pos, 0),
    )
    tl.store(output_cache_loc + dst_idx, cache_loc)
    tl.store(
        output_mirrored_kv_indices + dst_idx,
        tl.where(valid, src_idx, mirror_padding_index),
    )

    hidden_offsets = tl.arange(0, BLOCK_H)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    hidden = tl.load(
        target_hidden_states + src_idx * HIDDEN_SIZE + hidden_offsets,
        mask=valid & hidden_mask,
        other=0.0,
    )
    tl.store(
        output_hidden_states + dst_idx * HIDDEN_SIZE + hidden_offsets,
        hidden,
        mask=hidden_mask,
    )

    if local_pos == 0:
        last_src_slot = row * ACCEPT_WIDTH + real_len - 1
        last_src_idx = tl.load(accept_index + last_src_slot, mask=real_row, other=0).to(
            tl.int64
        )
        last_src_idx = tl.maximum(last_src_idx, 0)
        last_cache = tl.load(target_cache_loc + last_src_idx, mask=real_row, other=0)
        req_idx = tl.load(req_pool_indices + row, mask=real_row, other=0)
        bonus = tl.load(bonus_tokens + row, mask=real_row, other=0)
        tl.store(
            output_seq_lens + row,
            tl.where(real_row, old_seq_len + real_len, seq_len_fill_value),
        )
        tl.store(output_req_pool_indices + row, req_idx)
        tl.store(output_first_input_ids + row, bonus)
        tl.store(
            output_base_positions + row,
            tl.where(real_row, old_seq_len + real_len - 1, 0),
        )
        tl.store(
            custom_last_index + row,
            row * TOKENS_PER_BS + tl.where(real_row, real_len - 1, TOKENS_PER_BS - 1),
        )
        tl.store(custom_last_cache_loc + row, last_cache)


def pack_welm_mtp_verify_handoff(
    *,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    target_cache_loc: torch.Tensor,
    target_hidden_states: torch.Tensor,
    old_seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    bonus_tokens: torch.Tensor,
    output_input_ids: torch.Tensor,
    output_cache_loc: torch.Tensor,
    output_positions: torch.Tensor,
    output_hidden_states: torch.Tensor,
    output_mirrored_kv_indices: torch.Tensor,
    output_seq_lens: torch.Tensor,
    output_req_pool_indices: torch.Tensor,
    output_first_input_ids: torch.Tensor,
    output_base_positions: torch.Tensor,
    custom_last_index: torch.Tensor,
    custom_last_cache_loc: torch.Tensor,
    raw_bs: int,
    graph_bs: int,
    tokens_per_bs: int,
    mirror_padding_index: int,
    seq_len_fill_value: int,
) -> None:
    """Fused device-resident target-verify to draft-graph handoff."""
    if graph_bs <= 0:
        return
    if accept_index.ndim != 2:
        raise ValueError("accept_index must be two-dimensional.")
    if int(accept_index.shape[1]) < tokens_per_bs:
        raise ValueError(
            "accept_index width is smaller than the fixed draft graph row."
        )
    hidden_size = int(target_hidden_states.shape[-1])
    if hidden_size != int(output_hidden_states.shape[-1]):
        raise ValueError("Target/output hidden sizes do not match.")
    graph_num_tokens = graph_bs * tokens_per_bs
    _pack_verify_handoff_kernel[(graph_num_tokens,)](
        predict,
        accept_index,
        accept_lens,
        target_cache_loc,
        target_hidden_states,
        old_seq_lens,
        req_pool_indices,
        bonus_tokens,
        output_input_ids,
        output_cache_loc,
        output_positions,
        output_hidden_states,
        output_mirrored_kv_indices,
        output_seq_lens,
        output_req_pool_indices,
        output_first_input_ids,
        output_base_positions,
        custom_last_index,
        custom_last_cache_loc,
        raw_bs,
        mirror_padding_index,
        seq_len_fill_value,
        TOKENS_PER_BS=tokens_per_bs,
        ACCEPT_WIDTH=int(accept_index.shape[1]),
        HIDDEN_SIZE=hidden_size,
        BLOCK_H=triton.next_power_of_2(hidden_size),
        num_warps=8 if hidden_size >= 1024 else 4,
    )


@triton.jit
def _gather_last_hash_kernel(
    dense_hash,
    accept_lens,
    query_hash,
    raw_bs,
    dense_hash_stride,
    query_hash_stride,
    TOKENS_PER_BS: tl.constexpr,
):
    branch = tl.program_id(0)
    row = tl.program_id(1)
    valid = row < raw_bs
    accepted = tl.load(accept_lens + row, mask=valid, other=1).to(tl.int64)
    accepted = tl.minimum(tl.maximum(accepted, 1), TOKENS_PER_BS)
    src = branch * dense_hash_stride + row * TOKENS_PER_BS + accepted - 1
    value = tl.load(dense_hash + src, mask=valid, other=0)
    tl.store(query_hash + branch * query_hash_stride + row, value, mask=valid)


def gather_welm_mtp_last_hash(
    dense_hash: torch.Tensor,
    accept_lens: torch.Tensor,
    query_hash: torch.Tensor,
    *,
    raw_bs: int,
    tokens_per_bs: int,
) -> None:
    """Gather the last real accepted-token OE hash for every graph row."""
    if raw_bs <= 0:
        return
    if dense_hash.ndim != 2 or query_hash.ndim != 2:
        raise ValueError("OE hash inputs must be two-dimensional.")
    if int(dense_hash.shape[0]) != int(query_hash.shape[0]):
        raise ValueError("OE hash branch counts do not match.")
    _gather_last_hash_kernel[(int(dense_hash.shape[0]), raw_bs)](
        dense_hash,
        accept_lens,
        query_hash,
        raw_bs,
        int(dense_hash.stride(0)),
        int(query_hash.stride(0)),
        TOKENS_PER_BS=tokens_per_bs,
        num_warps=1,
    )


@triton.jit
def _build_linear_verify_inputs_kernel(
    bonus_tokens,
    proposal_tokens,
    seq_lens,
    output_tokens,
    output_positions,
    num_rows,
    TOKENS_PER_BS: tl.constexpr,
):
    idx = tl.program_id(0)
    row = idx // TOKENS_PER_BS
    local_pos = idx % TOKENS_PER_BS
    valid = row < num_rows
    bonus = tl.load(bonus_tokens + row, mask=valid, other=0)
    proposal_idx = row * (TOKENS_PER_BS - 1) + tl.maximum(local_pos - 1, 0)
    proposal = tl.load(proposal_tokens + proposal_idx, mask=valid, other=0)
    seq_len = tl.load(seq_lens + row, mask=valid, other=0)
    tl.store(output_tokens + idx, tl.where(local_pos == 0, bonus, proposal), mask=valid)
    tl.store(output_positions + idx, seq_len + local_pos, mask=valid)


def build_welm_mtp_linear_verify_inputs(
    *,
    bonus_tokens: torch.Tensor,
    proposal_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    output_tokens: torch.Tensor,
    output_positions: torch.Tensor,
    batch_size: int,
    tokens_per_bs: int,
) -> None:
    """Build the fixed four-token top-k=1 verify chain in one launch."""
    if batch_size <= 0:
        return
    if int(proposal_tokens.numel()) < batch_size * (tokens_per_bs - 1):
        raise ValueError("Linear MTP proposal token buffer is too small.")
    num_tokens = batch_size * tokens_per_bs
    _build_linear_verify_inputs_kernel[(num_tokens,)](
        bonus_tokens,
        proposal_tokens,
        seq_lens,
        output_tokens,
        output_positions,
        batch_size,
        TOKENS_PER_BS=tokens_per_bs,
        num_warps=1,
    )


@triton.jit
def _pack_linear_graph_outputs_kernel(
    sampled_p0,
    sampled_p1,
    sampled_p2,
    sampled_i0,
    sampled_i1,
    sampled_i2,
    draft_i0,
    draft_i1,
    draft_i2,
    draft_v0,
    draft_v1,
    draft_v2,
    bonus_tokens,
    base_positions,
    hot_token_map,
    output_topk_p,
    output_topk_index,
    output_proposal_tokens,
    output_draft_indices,
    output_draft_values,
    output_verify_tokens,
    output_verify_positions,
    num_rows,
    P0_ROW_STRIDE: tl.constexpr,
    P1_ROW_STRIDE: tl.constexpr,
    P2_ROW_STRIDE: tl.constexpr,
    I0_ROW_STRIDE: tl.constexpr,
    I1_ROW_STRIDE: tl.constexpr,
    I2_ROW_STRIDE: tl.constexpr,
    DI0_ROW_STRIDE: tl.constexpr,
    DI1_ROW_STRIDE: tl.constexpr,
    DI2_ROW_STRIDE: tl.constexpr,
    DV0_ROW_STRIDE: tl.constexpr,
    DV1_ROW_STRIDE: tl.constexpr,
    DV2_ROW_STRIDE: tl.constexpr,
    DRAFT_TOPK: tl.constexpr,
    VERIFY_WIDTH: tl.constexpr,
    HAS_HOT_MAP: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Pack the fixed MTP3 top-k=1 graph outputs with one small kernel."""
    row = tl.program_id(0)
    valid = row < num_rows

    p0 = tl.load(sampled_p0 + row * P0_ROW_STRIDE, mask=valid, other=0.0)
    p1 = tl.load(sampled_p1 + row * P1_ROW_STRIDE, mask=valid, other=0.0)
    p2 = tl.load(sampled_p2 + row * P2_ROW_STRIDE, mask=valid, other=0.0)
    i0 = tl.load(sampled_i0 + row * I0_ROW_STRIDE, mask=valid, other=0).to(tl.int64)
    i1 = tl.load(sampled_i1 + row * I1_ROW_STRIDE, mask=valid, other=0).to(tl.int64)
    i2 = tl.load(sampled_i2 + row * I2_ROW_STRIDE, mask=valid, other=0).to(tl.int64)

    output_base = row * 3
    tl.store(output_topk_p + output_base + 0, p0, mask=valid)
    tl.store(output_topk_p + output_base + 1, p1, mask=valid)
    tl.store(output_topk_p + output_base + 2, p2, mask=valid)
    tl.store(output_topk_index + output_base + 0, i0, mask=valid)
    tl.store(output_topk_index + output_base + 1, i1, mask=valid)
    tl.store(output_topk_index + output_base + 2, i2, mask=valid)

    if HAS_HOT_MAP:
        proposal0 = tl.load(hot_token_map + i0, mask=valid, other=0)
        proposal1 = tl.load(hot_token_map + i1, mask=valid, other=0)
        proposal2 = tl.load(hot_token_map + i2, mask=valid, other=0)
    else:
        proposal0 = i0
        proposal1 = i1
        proposal2 = i2
    tl.store(output_proposal_tokens + output_base + 0, proposal0, mask=valid)
    tl.store(output_proposal_tokens + output_base + 1, proposal1, mask=valid)
    tl.store(output_proposal_tokens + output_base + 2, proposal2, mask=valid)

    k = tl.arange(0, BLOCK_K)
    k_mask = valid & (k < DRAFT_TOPK)
    sparse_base = row * VERIFY_WIDTH * DRAFT_TOPK
    for step in tl.static_range(0, 4):
        target = sparse_base + step * DRAFT_TOPK + k
        if step == 0:
            draft_indices = tl.load(
                draft_i0 + row * DI0_ROW_STRIDE + k, mask=k_mask, other=0
            )
            draft_values = tl.load(
                draft_v0 + row * DV0_ROW_STRIDE + k, mask=k_mask, other=0.0
            )
        elif step == 1:
            draft_indices = tl.load(
                draft_i1 + row * DI1_ROW_STRIDE + k, mask=k_mask, other=0
            )
            draft_values = tl.load(
                draft_v1 + row * DV1_ROW_STRIDE + k, mask=k_mask, other=0.0
            )
        elif step == 2:
            draft_indices = tl.load(
                draft_i2 + row * DI2_ROW_STRIDE + k, mask=k_mask, other=0
            )
            draft_values = tl.load(
                draft_v2 + row * DV2_ROW_STRIDE + k, mask=k_mask, other=0.0
            )
        else:
            draft_indices = tl.zeros((BLOCK_K,), tl.int64)
            draft_values = tl.zeros((BLOCK_K,), tl.float32)
        tl.store(output_draft_indices + target, draft_indices, mask=k_mask)
        tl.store(output_draft_values + target, draft_values, mask=k_mask)

    # ``base_positions`` is written by the verify->draft handoff as
    # ``new_seq_len - 1`` and is not mutated by the captured draft forward.
    # In contrast, ForwardBatch.seq_lens is model-internal graph state and is
    # not an authoritative source for the next target verify positions.
    bonus = tl.load(bonus_tokens + row, mask=valid, other=0)
    base_position = tl.load(base_positions + row, mask=valid, other=0)
    verify_base = row * VERIFY_WIDTH
    tl.store(output_verify_tokens + verify_base + 0, bonus, mask=valid)
    tl.store(output_verify_tokens + verify_base + 1, proposal0, mask=valid)
    tl.store(output_verify_tokens + verify_base + 2, proposal1, mask=valid)
    tl.store(output_verify_tokens + verify_base + 3, proposal2, mask=valid)
    for pos in tl.static_range(0, 4):
        tl.store(
            output_verify_positions + verify_base + pos,
            base_position + 1 + pos,
            mask=valid,
        )


def pack_welm_mtp_linear_graph_outputs(
    *,
    sampled_probs: list[torch.Tensor],
    sampled_indices: list[torch.Tensor],
    draft_topk_indices: list[torch.Tensor],
    draft_topk_values: list[torch.Tensor],
    bonus_tokens: torch.Tensor,
    base_positions: torch.Tensor,
    hot_token_map: Optional[torch.Tensor],
    output_topk_p: torch.Tensor,
    output_topk_index: torch.Tensor,
    output_proposal_tokens: torch.Tensor,
    output_draft_indices: torch.Tensor,
    output_draft_values: torch.Tensor,
    output_verify_tokens: torch.Tensor,
    output_verify_positions: torch.Tensor,
    batch_size: int,
) -> None:
    """Fuse graph-tail assembly for the fixed three-step top-k=1 chain."""
    if batch_size <= 0:
        return
    if not (
        len(sampled_probs)
        == len(sampled_indices)
        == len(draft_topk_indices)
        == len(draft_topk_values)
        == 3
    ):
        raise ValueError("Fused linear graph output packing requires MTP3 inputs.")
    if (
        output_draft_indices.ndim != 3
        or output_draft_values.shape != output_draft_indices.shape
    ):
        raise ValueError("Draft sparse output buffers must be matching 3D tensors.")
    if int(output_draft_indices.shape[1]) != 4:
        raise ValueError(
            "Fused linear graph output packing requires verify width four."
        )
    draft_topk = int(output_draft_indices.shape[2])
    hot_map = sampled_indices[0] if hot_token_map is None else hot_token_map
    _pack_linear_graph_outputs_kernel[(batch_size,)](
        sampled_probs[0],
        sampled_probs[1],
        sampled_probs[2],
        sampled_indices[0],
        sampled_indices[1],
        sampled_indices[2],
        draft_topk_indices[0],
        draft_topk_indices[1],
        draft_topk_indices[2],
        draft_topk_values[0],
        draft_topk_values[1],
        draft_topk_values[2],
        bonus_tokens,
        base_positions,
        hot_map,
        output_topk_p,
        output_topk_index,
        output_proposal_tokens,
        output_draft_indices,
        output_draft_values,
        output_verify_tokens,
        output_verify_positions,
        batch_size,
        P0_ROW_STRIDE=sampled_probs[0].stride(0),
        P1_ROW_STRIDE=sampled_probs[1].stride(0),
        P2_ROW_STRIDE=sampled_probs[2].stride(0),
        I0_ROW_STRIDE=sampled_indices[0].stride(0),
        I1_ROW_STRIDE=sampled_indices[1].stride(0),
        I2_ROW_STRIDE=sampled_indices[2].stride(0),
        DI0_ROW_STRIDE=draft_topk_indices[0].stride(0),
        DI1_ROW_STRIDE=draft_topk_indices[1].stride(0),
        DI2_ROW_STRIDE=draft_topk_indices[2].stride(0),
        DV0_ROW_STRIDE=draft_topk_values[0].stride(0),
        DV1_ROW_STRIDE=draft_topk_values[1].stride(0),
        DV2_ROW_STRIDE=draft_topk_values[2].stride(0),
        DRAFT_TOPK=draft_topk,
        VERIFY_WIDTH=4,
        HAS_HOT_MAP=hot_token_map is not None,
        BLOCK_K=triton.next_power_of_2(draft_topk),
        num_warps=1,
    )
