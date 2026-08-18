from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.layers.utils.hash import murmur_hash32

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is optional on non-CUDA builds.
    triton = None
    tl = None


_SUPPORTED_TOPK = {1, 2, 4, 8, 16}
_UINT32_SCALE = 1.0 / float(1 << 32)


def welm_mtp_deterministic_uniforms(
    *,
    sampling_info,
    positions: Optional[torch.Tensor],
    batch_size: int,
    width: int,
    salt: int,
) -> Optional[torch.Tensor]:
    if sampling_info is None or positions is None or batch_size <= 0 or width <= 0:
        return None

    sampling_seed = getattr(sampling_info, "sampling_seed", None)
    if sampling_seed is None or sampling_seed.numel() < batch_size:
        return None
    if not sampling_seed.is_cuda or not positions.is_cuda:
        return None

    seeds = sampling_seed[:batch_size].reshape(-1).to(torch.uint64)
    base_positions = positions[:batch_size].reshape(-1).to(torch.int64)
    cols = torch.arange(
        int(salt),
        int(salt) + int(width),
        dtype=torch.int64,
        device=positions.device,
    )
    hashed = murmur_hash32(seeds, base_positions, cols)
    return (hashed.to(torch.float32) + 0.5) * _UINT32_SCALE


def welm_mtp_batch_base_positions(
    positions: Optional[torch.Tensor],
    *,
    batch_size: int,
    draft_token_num: int,
) -> Optional[torch.Tensor]:
    if positions is None or batch_size <= 0:
        return None
    if positions.numel() >= batch_size * draft_token_num:
        return positions.reshape(batch_size, draft_token_num)[:, 0]
    if positions.numel() >= batch_size:
        return positions[:batch_size]
    return None


def welm_mtp_sample_from_weights_with_uniform(
    weights: torch.Tensor, uniform: torch.Tensor
) -> torch.Tensor:
    weights = weights.to(dtype=torch.float32)
    total = weights.sum()
    threshold = uniform.to(dtype=torch.float32) * total
    cdf = torch.cumsum(weights, dim=0)
    return torch.sum(cdf < threshold).clamp(max=weights.numel() - 1)


def welmv4_mtp_apply_root_only_sampling(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    retrieve_index: torch.Tensor,
    root_only_verify_mask: torch.Tensor,
    target_predict: Optional[torch.Tensor] = None,
    target_probs: Optional[torch.Tensor] = None,
    target_topk_indices: Optional[torch.Tensor] = None,
    target_topk_values: Optional[torch.Tensor] = None,
    root_uniforms: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Overwrite seed rows with one root Target sample and no accepted drafts."""
    batch_size = int(accept_token_num.numel())
    if (
        root_only_verify_mask.dtype != torch.bool
        or root_only_verify_mask.ndim != 1
        or root_only_verify_mask.numel() != batch_size
    ):
        raise RuntimeError("WeLM MTP root-only sampling mask is not request-aligned.")
    root_rows = torch.nonzero(root_only_verify_mask, as_tuple=False).flatten()
    if root_rows.numel() == 0:
        return None

    has_sparse_target = (
        target_topk_indices is not None or target_topk_values is not None
    )
    source_count = int(target_predict is not None) + int(target_probs is not None)
    source_count += int(has_sparse_target)
    if source_count != 1 or has_sparse_target != (
        target_topk_indices is not None and target_topk_values is not None
    ):
        raise RuntimeError(
            "WeLM MTP root-only sampling requires exactly one Target distribution."
        )

    def root_source_rows(value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] == batch_size:
            return value[root_rows, 0]
        if value.shape[0] == root_rows.numel():
            return value[:, 0]
        raise RuntimeError(
            "WeLM MTP root-only Target rows are neither batch nor root aligned."
        )

    if target_predict is not None:
        sampled_tokens = root_source_rows(target_predict).to(dtype=predicts.dtype)
    else:
        if target_probs is not None:
            weights = root_source_rows(target_probs).to(dtype=torch.float32)
            token_ids = None
        else:
            weights = root_source_rows(target_topk_values).to(dtype=torch.float32)
            token_ids = root_source_rows(target_topk_indices)

        if root_uniforms is None:
            uniforms = torch.rand(
                (root_rows.numel(),), dtype=torch.float32, device=weights.device
            )
        elif root_uniforms.numel() == batch_size:
            uniforms = root_uniforms.reshape(-1).index_select(0, root_rows)
        elif root_uniforms.numel() == root_rows.numel():
            uniforms = root_uniforms.reshape(-1)
        else:
            raise RuntimeError(
                "WeLM MTP root-only sampling uniforms are not request-aligned."
            )
        totals = weights.sum(dim=1)
        torch._assert_async(
            torch.all(totals > 0),
            "WeLM MTP root-only Target distribution has zero probability mass.",
        )
        thresholds = uniforms * totals
        sample_pos = torch.sum(
            torch.cumsum(weights, dim=1) < thresholds[:, None], dim=1
        ).clamp(max=weights.shape[1] - 1)
        sampled_tokens = (
            sample_pos
            if token_ids is None
            else token_ids.gather(1, sample_pos[:, None]).squeeze(1)
        ).to(dtype=predicts.dtype)

    root_predict_indices = retrieve_index[root_rows, 0].to(dtype=torch.long)
    accept_index[root_rows] = -1
    accept_index[root_rows, 0] = root_predict_indices.to(dtype=accept_index.dtype)
    accept_token_num[root_rows] = 0
    predicts[root_predict_indices] = sampled_tokens
    return root_rows, sampled_tokens


if triton is not None:

    @triton.jit
    def _local_topk_kernel(
        logits_ptr,
        local_values_ptr,
        local_indices_ptr,
        vocab_size: tl.constexpr,
        num_blocks: tl.constexpr,
        block_size: tl.constexpr,
        topk: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < vocab_size

        vals = tl.load(
            logits_ptr + row * vocab_size + offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        out_base = (row * num_blocks + block) * topk

        for k in tl.static_range(0, topk):
            max_val = tl.max(vals, axis=0)
            max_pos = tl.min(
                tl.where(vals == max_val, offsets, vocab_size + block_size),
                axis=0,
            )
            tl.store(local_values_ptr + out_base + k, max_val)
            tl.store(local_indices_ptr + out_base + k, max_pos)
            vals = tl.where(offsets == max_pos, -float("inf"), vals)

    @triton.jit
    def _reduce_sample_kernel(
        local_values_ptr,
        local_indices_ptr,
        temperature_ptr,
        top_p_ptr,
        uniform_ptr,
        sampled_probs_ptr,
        sampled_indices_ptr,
        topk_indices_ptr,
        topk_probs_ptr,
        num_blocks: tl.constexpr,
        topk: tl.constexpr,
        reduce_block: tl.constexpr,
        use_top_p: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, reduce_block)
        num_candidates = num_blocks * topk
        mask = offsets < num_candidates
        local_base = row * num_candidates

        vals = tl.load(
            local_values_ptr + local_base + offsets,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        out_base = row * topk

        for k in tl.static_range(0, topk):
            max_val = tl.max(vals, axis=0)
            max_pos = tl.min(
                tl.where(vals == max_val, offsets, num_candidates + reduce_block),
                axis=0,
            )
            token_id = tl.load(local_indices_ptr + local_base + max_pos)
            tl.store(topk_indices_ptr + out_base + k, token_id)
            tl.store(topk_probs_ptr + out_base + k, max_val)
            vals = tl.where(offsets == max_pos, -float("inf"), vals)

        k_offsets = tl.arange(0, topk)
        selected_logits = tl.load(topk_probs_ptr + out_base + k_offsets).to(tl.float32)
        temperature = tl.load(temperature_ptr + row)
        scaled = selected_logits / temperature
        stable_max = tl.max(scaled, axis=0)
        exp_vals = tl.exp(scaled - stable_max)
        denom = tl.sum(exp_vals, axis=0)
        probs = exp_vals / denom
        if use_top_p:
            # Candidates are in descending-logit order. Match
            # sgl_kernel.top_p_renorm_prob by retaining the candidate that
            # crosses the nucleus threshold, then renormalizing the prefix.
            top_p = tl.load(top_p_ptr + row)
            pre_filter_cdf = tl.cumsum(probs, axis=0)
            prefix_keep = (pre_filter_cdf - probs) < top_p
            # The SGL kernel retains every probability tied with the cutoff
            # value, even if only one of those equal-valued candidates is the
            # mathematical crossing point.
            cutoff = tl.min(tl.where(prefix_keep, probs, float("inf")), axis=0)
            probs = tl.where(probs >= cutoff, probs, 0.0)
            probs = probs / tl.sum(probs, axis=0)
        tl.store(topk_probs_ptr + out_base + k_offsets, probs)

        cdf = tl.cumsum(probs, axis=0)
        uniform = tl.load(uniform_ptr + row)
        sample_pos = tl.min(tl.where(cdf >= uniform, k_offsets, topk - 1), axis=0)
        sampled_token_id = tl.load(topk_indices_ptr + out_base + sample_pos)
        sampled_prob = tl.load(topk_probs_ptr + out_base + sample_pos)
        tl.store(sampled_indices_ptr + row, sampled_token_id)
        tl.store(sampled_probs_ptr + row, sampled_prob)

    @triton.jit
    def _reduce_local_topk_pack_kernel(
        local_values_ptr,
        local_indices_ptr,
        packed_ptr,
        num_blocks: tl.constexpr,
        topk: tl.constexpr,
        reduce_block: tl.constexpr,
        index_offset: tl.constexpr,
    ):
        """Reduce block candidates and pack FP32 values/global ids together."""
        row = tl.program_id(0)
        candidate_count = num_blocks * topk
        offsets = tl.arange(0, reduce_block)
        mask = offsets < candidate_count
        source_base = row * candidate_count
        values = tl.load(
            local_values_ptr + source_base + offsets,
            mask=mask,
            other=-float("inf"),
        )
        indices = tl.load(
            local_indices_ptr + source_base + offsets,
            mask=mask,
            other=0x7FFFFFFF,
        ).to(tl.int64)
        output_base = row * (2 * topk)
        for k in tl.static_range(0, topk):
            max_value = tl.max(values, axis=0)
            # Deterministic tie-break by the actual local vocabulary id.
            max_index = tl.min(
                tl.where(values == max_value, indices, 0x7FFFFFFF), axis=0
            )
            tl.store(packed_ptr + output_base + k, max_value)
            # The global vocabulary is far below 2**24, so FP32 represents
            # every token id exactly and permits one collective for value+id.
            tl.store(
                packed_ptr + output_base + topk + k,
                (max_index + index_offset).to(tl.float32),
            )
            values = tl.where(indices == max_index, -float("inf"), values)

    @triton.jit
    def _unpack_gathered_topk_kernel(
        gathered_ptr,
        values_ptr,
        indices_ptr,
        rows,
        topk: tl.constexpr,
        world_size: tl.constexpr,
        block_k: tl.constexpr,
    ):
        row = tl.program_id(0)
        valid_row = row < rows
        k = tl.arange(0, block_k)
        valid = valid_row & (k < topk)
        gathered_row_base = row * world_size * 2 * topk
        output_row_base = row * world_size * topk
        for rank in tl.static_range(0, world_size):
            source_base = gathered_row_base + rank * 2 * topk
            target_base = output_row_base + rank * topk
            values = tl.load(gathered_ptr + source_base + k, mask=valid, other=0.0)
            indices = tl.load(
                gathered_ptr + source_base + topk + k, mask=valid, other=0.0
            ).to(tl.int64)
            tl.store(values_ptr + target_base + k, values, mask=valid)
            tl.store(indices_ptr + target_base + k, indices, mask=valid)

    @triton.jit
    def _verify_top1_sparse_kernel(
        predicts_ptr,
        accept_index_ptr,
        accept_token_num_ptr,
        candidates_ptr,
        retrieve_index_ptr,
        retrieve_next_token_ptr,
        target_topk_indices_ptr,
        target_topk_values_ptr,
        draft_topk_indices_ptr,
        draft_topk_values_ptr,
        uniform_ptr,
        bs: tl.constexpr,
        draft_token_num: tl.constexpr,
        accept_width: tl.constexpr,
        num_draft_steps: tl.constexpr,
        topk: tl.constexpr,
        block_topk: tl.constexpr,
        uniform_width: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_topk)
        topk_mask = offsets < topk

        parent = tl.full((), 0, dtype=tl.int64)
        accepted = tl.full((), 0, dtype=tl.int32)
        active = row >= 0

        retrieve_base = row * draft_token_num
        topk_base = row * draft_token_num * topk
        accept_base = row * accept_width
        uniform_base = row * uniform_width

        parent_predict_idx = tl.load(retrieve_index_ptr + retrieve_base)
        tl.store(accept_index_ptr + accept_base, parent_predict_idx)

        for step in tl.static_range(0, num_draft_steps):
            next_local = tl.load(retrieve_next_token_ptr + retrieve_base + parent)
            valid_child = active & (next_local >= 0)
            safe_child = tl.maximum(next_local, 0)
            draft_token_id = tl.load(candidates_ptr + retrieve_base + safe_child)

            parent_topk_base = topk_base + parent * topk
            target_indices = tl.load(
                target_topk_indices_ptr + parent_topk_base + offsets,
                mask=topk_mask,
                other=-1,
            )
            target_values = tl.load(
                target_topk_values_ptr + parent_topk_base + offsets,
                mask=topk_mask,
                other=0.0,
            ).to(tl.float32)
            draft_indices = tl.load(
                draft_topk_indices_ptr + parent_topk_base + offsets,
                mask=topk_mask,
                other=-2,
            )
            draft_values = tl.load(
                draft_topk_values_ptr + parent_topk_base + offsets,
                mask=topk_mask,
                other=0.0,
            ).to(tl.float32)

            p = tl.sum(
                tl.where(target_indices == draft_token_id, target_values, 0.0), axis=0
            )
            q = tl.sum(
                tl.where(draft_indices == draft_token_id, draft_values, 0.0), axis=0
            )
            accept_prob = tl.minimum(p / tl.maximum(q, 1.0e-30), 1.0)
            coin = tl.load(uniform_ptr + uniform_base + step)
            accepted_this = valid_child & (coin <= accept_prob)

            tl.store(
                predicts_ptr + parent_predict_idx,
                draft_token_id,
                mask=accepted_this,
            )
            child_predict_idx = tl.load(retrieve_index_ptr + retrieve_base + safe_child)
            next_accepted = accepted + accepted_this.to(tl.int32)
            tl.store(
                accept_index_ptr + accept_base + next_accepted,
                child_predict_idx,
                mask=accepted_this,
            )

            parent = tl.where(accepted_this, safe_child, parent)
            parent_predict_idx = tl.where(
                accepted_this, child_predict_idx, parent_predict_idx
            )
            accepted = next_accepted
            active = accepted_this

        sample_topk_base = topk_base + parent * topk
        target_indices = tl.load(
            target_topk_indices_ptr + sample_topk_base + offsets,
            mask=topk_mask,
            other=-1,
        )
        target_values = tl.load(
            target_topk_values_ptr + sample_topk_base + offsets,
            mask=topk_mask,
            other=0.0,
        ).to(tl.float32)
        q_on_target = tl.zeros((block_topk,), dtype=tl.float32)
        for k in tl.static_range(0, topk):
            draft_idx = tl.load(draft_topk_indices_ptr + sample_topk_base + k)
            draft_val = tl.load(draft_topk_values_ptr + sample_topk_base + k).to(
                tl.float32
            )
            q_on_target += tl.where(target_indices == draft_idx, draft_val, 0.0)

        residual_values = tl.maximum(target_values - q_on_target, 0.0)
        residual_sum = tl.sum(tl.where(topk_mask, residual_values, 0.0), axis=0)
        target_sum = tl.sum(tl.where(topk_mask, target_values, 0.0), axis=0)
        use_residual = (accepted < num_draft_steps) & (residual_sum > 0.0)
        sample_values = tl.where(use_residual, residual_values, target_values)
        sample_total = tl.maximum(
            tl.where(use_residual, residual_sum, target_sum), 1.0e-30
        )
        sample_threshold = (
            tl.load(uniform_ptr + uniform_base + num_draft_steps) * sample_total
        )
        cdf = tl.cumsum(tl.where(topk_mask, sample_values, 0.0), axis=0)
        sample_pos = tl.min(
            tl.where((cdf >= sample_threshold) & topk_mask, offsets, topk - 1),
            axis=0,
        )
        sampled_token = tl.sum(
            tl.where(offsets == sample_pos, target_indices, 0), axis=0
        )

        tl.store(predicts_ptr + parent_predict_idx, sampled_token.to(tl.int32))
        tl.store(accept_token_num_ptr + row, accepted)

    @triton.jit
    def _verify_top1_dense_target_accept_kernel(
        predicts_ptr,
        accept_index_ptr,
        accept_token_num_ptr,
        sample_parent_ptr,
        sample_want_residual_ptr,
        candidates_ptr,
        retrieve_index_ptr,
        retrieve_next_token_ptr,
        target_probs_ptr,
        draft_topk_indices_ptr,
        draft_topk_values_ptr,
        uniform_ptr,
        vocab_size: tl.constexpr,
        draft_token_num: tl.constexpr,
        accept_width: tl.constexpr,
        num_draft_steps: tl.constexpr,
        topk: tl.constexpr,
        block_topk: tl.constexpr,
        uniform_width: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_topk)
        topk_mask = offsets < topk

        parent = tl.full((), 0, dtype=tl.int64)
        accepted = tl.full((), 0, dtype=tl.int32)
        active = row >= 0

        retrieve_base = row * draft_token_num
        topk_base = row * draft_token_num * topk
        target_base = row * draft_token_num * vocab_size
        accept_base = row * accept_width
        uniform_base = row * uniform_width

        parent_predict_idx = tl.load(retrieve_index_ptr + retrieve_base)
        tl.store(accept_index_ptr + accept_base, parent_predict_idx)

        for step in tl.static_range(0, num_draft_steps):
            next_local = tl.load(retrieve_next_token_ptr + retrieve_base + parent)
            valid_child = active & (next_local >= 0)
            safe_child = tl.maximum(next_local, 0)
            draft_token_id = tl.load(candidates_ptr + retrieve_base + safe_child)

            p = tl.load(
                target_probs_ptr + target_base + parent * vocab_size + draft_token_id
            ).to(tl.float32)
            parent_topk_base = topk_base + parent * topk
            draft_indices = tl.load(
                draft_topk_indices_ptr + parent_topk_base + offsets,
                mask=topk_mask,
                other=-2,
            )
            draft_values = tl.load(
                draft_topk_values_ptr + parent_topk_base + offsets,
                mask=topk_mask,
                other=0.0,
            ).to(tl.float32)
            q = tl.sum(
                tl.where(draft_indices == draft_token_id, draft_values, 0.0), axis=0
            )
            accept_prob = tl.minimum(p / tl.maximum(q, 1.0e-30), 1.0)
            coin = tl.load(uniform_ptr + uniform_base + step)
            accepted_this = valid_child & (coin <= accept_prob)

            tl.store(
                predicts_ptr + parent_predict_idx,
                draft_token_id,
                mask=accepted_this,
            )
            child_predict_idx = tl.load(retrieve_index_ptr + retrieve_base + safe_child)
            next_accepted = accepted + accepted_this.to(tl.int32)
            tl.store(
                accept_index_ptr + accept_base + next_accepted,
                child_predict_idx,
                mask=accepted_this,
            )

            parent = tl.where(accepted_this, safe_child, parent)
            parent_predict_idx = tl.where(
                accepted_this, child_predict_idx, parent_predict_idx
            )
            accepted = next_accepted
            active = accepted_this

        tl.store(sample_parent_ptr + row, parent)
        tl.store(
            sample_want_residual_ptr + row,
            (accepted < num_draft_steps).to(tl.int32),
        )
        tl.store(accept_token_num_ptr + row, accepted)

    @triton.jit
    def _verify_top1_dense_target_block_sum_kernel(
        target_probs_ptr,
        draft_topk_indices_ptr,
        draft_topk_values_ptr,
        sample_parent_ptr,
        target_block_sums_ptr,
        residual_block_sums_ptr,
        vocab_size: tl.constexpr,
        num_blocks: tl.constexpr,
        block_size: tl.constexpr,
        draft_token_num: tl.constexpr,
        topk: tl.constexpr,
        block_topk: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < vocab_size
        topk_offsets = tl.arange(0, block_topk)
        topk_mask = topk_offsets < topk

        parent = tl.load(sample_parent_ptr + row)
        target_base = row * draft_token_num * vocab_size + parent * vocab_size
        vals = tl.load(
            target_probs_ptr + target_base + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        target_sum = tl.sum(vals, axis=0)

        topk_base = row * draft_token_num * topk + parent * topk
        draft_indices = tl.load(
            draft_topk_indices_ptr + topk_base + topk_offsets,
            mask=topk_mask,
            other=-1,
        )
        draft_values = tl.load(
            draft_topk_values_ptr + topk_base + topk_offsets,
            mask=topk_mask,
            other=0.0,
        ).to(tl.float32)
        q = tl.zeros((block_size,), dtype=tl.float32)
        for k in tl.static_range(0, topk):
            draft_idx = tl.load(draft_topk_indices_ptr + topk_base + k)
            draft_val = tl.load(draft_topk_values_ptr + topk_base + k).to(tl.float32)
            q += tl.where(offsets == draft_idx, draft_val, 0.0)
        residual_sum = tl.sum(tl.maximum(vals - q, 0.0), axis=0)

        out = row * num_blocks + block
        tl.store(target_block_sums_ptr + out, target_sum)
        tl.store(residual_block_sums_ptr + out, residual_sum)

    @triton.jit
    def _verify_top1_dense_target_sample_kernel(
        predicts_ptr,
        retrieve_index_ptr,
        target_probs_ptr,
        draft_topk_indices_ptr,
        draft_topk_values_ptr,
        sample_parent_ptr,
        sample_want_residual_ptr,
        target_block_sums_ptr,
        residual_block_sums_ptr,
        uniform_ptr,
        vocab_size: tl.constexpr,
        num_blocks: tl.constexpr,
        block_size: tl.constexpr,
        reduce_block: tl.constexpr,
        draft_token_num: tl.constexpr,
        num_draft_steps: tl.constexpr,
        topk: tl.constexpr,
        block_topk: tl.constexpr,
        uniform_width: tl.constexpr,
    ):
        row = tl.program_id(0)
        block_offsets = tl.arange(0, reduce_block)
        block_mask = block_offsets < num_blocks
        parent = tl.load(sample_parent_ptr + row)
        want_residual = tl.load(sample_want_residual_ptr + row) != 0

        sums_base = row * num_blocks
        target_sums = tl.load(
            target_block_sums_ptr + sums_base + block_offsets,
            mask=block_mask,
            other=0.0,
        ).to(tl.float32)
        residual_sums = tl.load(
            residual_block_sums_ptr + sums_base + block_offsets,
            mask=block_mask,
            other=0.0,
        ).to(tl.float32)
        target_total = tl.sum(target_sums, axis=0)
        residual_total = tl.sum(residual_sums, axis=0)
        use_residual = want_residual & (residual_total > 0.0)
        block_sums = tl.where(use_residual, residual_sums, target_sums)
        total = tl.maximum(
            tl.where(use_residual, residual_total, target_total), 1.0e-30
        )
        threshold = tl.load(uniform_ptr + row * uniform_width + num_draft_steps) * total
        block_cdf = tl.cumsum(block_sums, axis=0)
        selected_block = tl.min(
            tl.where(
                (block_cdf >= threshold) & block_mask,
                block_offsets,
                num_blocks - 1,
            ),
            axis=0,
        )
        prev_cdf = tl.sum(
            tl.where(block_offsets < selected_block, block_sums, 0.0), axis=0
        )
        threshold_in_block = threshold - prev_cdf

        offsets = selected_block * block_size + tl.arange(0, block_size)
        mask = offsets < vocab_size
        target_base = row * draft_token_num * vocab_size + parent * vocab_size
        vals = tl.load(
            target_probs_ptr + target_base + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        topk_base = row * draft_token_num * topk + parent * topk
        q = tl.zeros((block_size,), dtype=tl.float32)
        for k in tl.static_range(0, topk):
            draft_idx = tl.load(draft_topk_indices_ptr + topk_base + k)
            draft_val = tl.load(draft_topk_values_ptr + topk_base + k).to(tl.float32)
            q += tl.where(offsets == draft_idx, draft_val, 0.0)
        sample_vals = tl.where(use_residual, tl.maximum(vals - q, 0.0), vals)
        sample_cdf = tl.cumsum(tl.where(mask, sample_vals, 0.0), axis=0)
        local_offsets = tl.arange(0, block_size)
        last_valid = tl.minimum(
            block_size - 1, vocab_size - selected_block * block_size - 1
        )
        sample_pos = tl.min(
            tl.where(
                (sample_cdf >= threshold_in_block) & mask,
                local_offsets,
                last_valid,
            ),
            axis=0,
        )
        sampled_token = selected_block * block_size + sample_pos
        parent_predict_idx = tl.load(
            retrieve_index_ptr + row * draft_token_num + parent
        )
        tl.store(predicts_ptr + parent_predict_idx, sampled_token.to(tl.int32))


def can_use_welmv4_mtp_fused_topk_sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    uniform: torch.Tensor,
    topk: int,
    top_p: Optional[torch.Tensor] = None,
) -> bool:
    return (
        triton is not None
        and logits.is_cuda
        and temperature.is_cuda
        and uniform.is_cuda
        and logits.dim() == 2
        and temperature.numel() >= logits.shape[0]
        and uniform.numel() >= logits.shape[0]
        and (top_p is None or (top_p.is_cuda and top_p.numel() >= logits.shape[0]))
        and int(topk) in _SUPPORTED_TOPK
        and 0 < int(topk) < logits.shape[1]
        and logits.shape[0] > 0
    )


def welmv4_mtp_local_topk_pack(
    logits: torch.Tensor,
    topk: int,
    *,
    index_offset: int = 0,
    block_size: int = 1024,
) -> Optional[torch.Tensor]:
    """Return each shard's exact top-k as one packed FP32 value/id tensor."""
    if (
        triton is None
        or not logits.is_cuda
        or logits.ndim != 2
        or logits.shape[0] <= 0
        or logits.shape[1] <= 0
        or int(topk) not in _SUPPORTED_TOPK
        or int(topk) > int(logits.shape[1])
        or int(index_offset) < 0
        or int(index_offset) + int(logits.shape[1]) > (1 << 24)
    ):
        return None
    rows, vocab_size = logits.shape
    topk = int(topk)
    num_blocks = triton.cdiv(vocab_size, block_size)
    reduce_block = triton.next_power_of_2(num_blocks * topk)
    local_values = torch.empty(
        (rows, num_blocks, topk), dtype=torch.float32, device=logits.device
    )
    local_indices = torch.empty(
        (rows, num_blocks, topk), dtype=torch.int64, device=logits.device
    )
    packed = torch.empty((rows, 2 * topk), dtype=torch.float32, device=logits.device)
    _local_topk_kernel[(rows, num_blocks)](
        logits,
        local_values,
        local_indices,
        vocab_size,
        num_blocks,
        block_size,
        topk,
        num_warps=8,
    )
    _reduce_local_topk_pack_kernel[(rows,)](
        local_values,
        local_indices,
        packed,
        num_blocks,
        topk,
        reduce_block,
        int(index_offset),
        num_warps=8,
    )
    return packed


def welmv4_mtp_unpack_gathered_topk(
    gathered: torch.Tensor,
    output_indices: torch.Tensor,
    *,
    topk: int,
    world_size: int,
) -> Optional[torch.Tensor]:
    """Unpack rank-major packed candidates after one TP all-gather."""
    topk = int(topk)
    world_size = int(world_size)
    if (
        triton is None
        or not gathered.is_cuda
        or not output_indices.is_cuda
        or gathered.ndim != 2
        or output_indices.ndim != 2
        or gathered.dtype != torch.float32
        or output_indices.dtype != torch.int64
        or topk not in _SUPPORTED_TOPK
        or world_size <= 0
        or int(gathered.shape[1]) != world_size * 2 * topk
        or tuple(output_indices.shape) != (int(gathered.shape[0]), world_size * topk)
    ):
        return None
    values = torch.empty(
        output_indices.shape, dtype=torch.float32, device=gathered.device
    )
    _unpack_gathered_topk_kernel[(int(gathered.shape[0]),)](
        gathered,
        values,
        output_indices,
        int(gathered.shape[0]),
        topk=topk,
        world_size=world_size,
        block_k=triton.next_power_of_2(topk),
        num_warps=1,
    )
    return values


def welmv4_mtp_fused_topk_softmax_sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    uniform: torch.Tensor,
    topk: int,
    *,
    top_p: Optional[torch.Tensor] = None,
    block_size: int = 1024,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return sampled prob/id plus sparse topK q distribution for WeLMV4 MTP.

    This is an exact topK over the full vocab for small K. When ``top_p`` is
    provided, nucleus filtering is applied to those candidates before sample.
    """
    if not can_use_welmv4_mtp_fused_topk_sample(
        logits, temperature, uniform, topk, top_p
    ):
        return None

    rows, vocab_size = logits.shape
    topk = int(topk)
    num_blocks = triton.cdiv(vocab_size, block_size)
    reduce_block = triton.next_power_of_2(num_blocks * topk)

    local_values = torch.empty(
        (rows, num_blocks, topk), dtype=torch.float32, device=logits.device
    )
    local_indices = torch.empty(
        (rows, num_blocks, topk), dtype=torch.int64, device=logits.device
    )
    sampled_probs = torch.empty((rows, 1), dtype=torch.float32, device=logits.device)
    sampled_indices = torch.empty((rows, 1), dtype=torch.int64, device=logits.device)
    topk_indices = torch.empty((rows, topk), dtype=torch.int64, device=logits.device)
    topk_probs = torch.empty((rows, topk), dtype=torch.float32, device=logits.device)

    temperature = temperature.reshape(-1)
    uniform = uniform.reshape(-1)
    use_top_p = top_p is not None
    top_p = temperature if top_p is None else top_p.reshape(-1)

    _local_topk_kernel[(rows, num_blocks)](
        logits,
        local_values,
        local_indices,
        vocab_size,
        num_blocks,
        block_size,
        topk,
        num_warps=8,
    )
    _reduce_sample_kernel[(rows,)](
        local_values,
        local_indices,
        temperature,
        top_p,
        uniform,
        sampled_probs,
        sampled_indices,
        topk_indices,
        topk_probs,
        num_blocks,
        topk,
        reduce_block,
        use_top_p=use_top_p,
        num_warps=8,
    )
    return sampled_probs, sampled_indices, topk_indices, topk_probs


def can_use_welmv4_mtp_fused_verify_top1_sparse(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    target_topk_indices: torch.Tensor,
    target_topk_values: torch.Tensor,
    draft_topk_indices: torch.Tensor,
    draft_topk_values: torch.Tensor,
) -> bool:
    if triton is None:
        return False
    tensors = (
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        target_topk_indices,
        target_topk_values,
        draft_topk_indices,
        draft_topk_values,
    )
    if not all(t.is_cuda for t in tensors):
        return False
    if candidates.dim() != 2 or accept_index.dim() != 2:
        return False
    if target_topk_indices.shape != target_topk_values.shape:
        return False
    if draft_topk_indices.shape != draft_topk_values.shape:
        return False
    if target_topk_indices.shape != draft_topk_indices.shape:
        return False
    if target_topk_indices.dim() != 3:
        return False
    topk = int(target_topk_indices.shape[-1])
    return (
        int(candidates.shape[0]) > 0
        and int(candidates.shape[1]) > 1
        and int(accept_index.shape[1]) > 1
        and int(accept_index.shape[1]) <= int(candidates.shape[1])
        and topk in _SUPPORTED_TOPK
    )


def welmv4_mtp_fused_verify_top1_sparse(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    target_topk_indices: torch.Tensor,
    target_topk_values: torch.Tensor,
    draft_topk_indices: torch.Tensor,
    draft_topk_values: torch.Tensor,
    uniforms: Optional[torch.Tensor] = None,
) -> bool:
    """Fused topK-restricted verify sampling for WeLMV4 MTP topk=1 chains."""
    if not can_use_welmv4_mtp_fused_verify_top1_sparse(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        target_topk_indices,
        target_topk_values,
        draft_topk_indices,
        draft_topk_values,
    ):
        return False

    candidates = candidates.contiguous()
    retrieve_index = retrieve_index.contiguous()
    retrieve_next_token = retrieve_next_token.contiguous()
    target_topk_indices = target_topk_indices.contiguous()
    target_topk_values = target_topk_values.contiguous()
    draft_topk_indices = draft_topk_indices.contiguous()
    draft_topk_values = draft_topk_values.contiguous()

    bs = int(candidates.shape[0])
    draft_token_num = int(candidates.shape[1])
    accept_width = int(accept_index.shape[1])
    num_draft_steps = min(accept_width - 1, draft_token_num - 1)
    topk = int(target_topk_indices.shape[-1])
    block_topk = triton.next_power_of_2(topk)
    uniform_width = num_draft_steps + 1
    if uniforms is None:
        uniforms = torch.empty(
            (bs, uniform_width), dtype=torch.float32, device=candidates.device
        )
        uniforms.uniform_()
    else:
        if uniforms.shape[0] < bs or uniforms.shape[1] < uniform_width:
            return False
        uniforms = uniforms[:bs, :uniform_width].contiguous()

    _verify_top1_sparse_kernel[(bs,)](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        target_topk_indices,
        target_topk_values,
        draft_topk_indices,
        draft_topk_values,
        uniforms,
        bs,
        draft_token_num,
        accept_width,
        num_draft_steps,
        topk,
        block_topk,
        uniform_width,
        num_warps=1,
    )
    return True


def can_use_welmv4_mtp_fused_verify_top1_dense_target(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    target_probs: torch.Tensor,
    draft_topk_indices: torch.Tensor,
    draft_topk_values: torch.Tensor,
) -> bool:
    if triton is None:
        return False
    tensors = (
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        target_probs,
        draft_topk_indices,
        draft_topk_values,
    )
    if not all(t.is_cuda for t in tensors):
        return False
    if candidates.dim() != 2 or accept_index.dim() != 2:
        return False
    if target_probs.dim() != 3:
        return False
    if draft_topk_indices.shape != draft_topk_values.shape:
        return False
    if draft_topk_indices.dim() != 3:
        return False
    if target_probs.shape[:2] != candidates.shape:
        return False
    if draft_topk_indices.shape[:2] != candidates.shape:
        return False

    topk = int(draft_topk_indices.shape[-1])
    return (
        int(candidates.shape[0]) > 0
        and int(candidates.shape[1]) > 1
        and int(accept_index.shape[1]) > 1
        and int(accept_index.shape[1]) <= int(candidates.shape[1])
        and int(target_probs.shape[-1]) > 0
        and topk in _SUPPORTED_TOPK
    )


def welmv4_mtp_fused_verify_top1_dense_target(
    *,
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    target_probs: torch.Tensor,
    draft_topk_indices: torch.Tensor,
    draft_topk_values: torch.Tensor,
    uniforms: Optional[torch.Tensor] = None,
    block_size: int = 1024,
) -> bool:
    """Fused verify sampling for full-vocab target p and sparse topK draft q."""
    if not can_use_welmv4_mtp_fused_verify_top1_dense_target(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        target_probs,
        draft_topk_indices,
        draft_topk_values,
    ):
        return False

    candidates = candidates.contiguous()
    retrieve_index = retrieve_index.contiguous()
    retrieve_next_token = retrieve_next_token.contiguous()
    target_probs = target_probs.contiguous()
    draft_topk_indices = draft_topk_indices.contiguous()
    draft_topk_values = draft_topk_values.contiguous()

    bs = int(candidates.shape[0])
    draft_token_num = int(candidates.shape[1])
    accept_width = int(accept_index.shape[1])
    vocab_size = int(target_probs.shape[-1])
    num_draft_steps = min(accept_width - 1, draft_token_num - 1)
    topk = int(draft_topk_indices.shape[-1])
    block_topk = triton.next_power_of_2(topk)
    num_blocks = triton.cdiv(vocab_size, block_size)
    reduce_block = triton.next_power_of_2(num_blocks)
    uniform_width = num_draft_steps + 1

    if uniforms is None:
        uniforms = torch.empty(
            (bs, uniform_width), dtype=torch.float32, device=candidates.device
        )
        uniforms.uniform_()
    else:
        if uniforms.shape[0] < bs or uniforms.shape[1] < uniform_width:
            return False
        uniforms = uniforms[:bs, :uniform_width].contiguous()
    sample_parent = torch.empty((bs,), dtype=torch.int64, device=candidates.device)
    sample_want_residual = torch.empty(
        (bs,), dtype=torch.int32, device=candidates.device
    )
    target_block_sums = torch.empty(
        (bs, num_blocks), dtype=torch.float32, device=candidates.device
    )
    residual_block_sums = torch.empty_like(target_block_sums)

    _verify_top1_dense_target_accept_kernel[(bs,)](
        predicts,
        accept_index,
        accept_token_num,
        sample_parent,
        sample_want_residual,
        candidates,
        retrieve_index,
        retrieve_next_token,
        target_probs,
        draft_topk_indices,
        draft_topk_values,
        uniforms,
        vocab_size,
        draft_token_num,
        accept_width,
        num_draft_steps,
        topk,
        block_topk,
        uniform_width,
        num_warps=1,
    )
    _verify_top1_dense_target_block_sum_kernel[(bs, num_blocks)](
        target_probs,
        draft_topk_indices,
        draft_topk_values,
        sample_parent,
        target_block_sums,
        residual_block_sums,
        vocab_size,
        num_blocks,
        block_size,
        draft_token_num,
        topk,
        block_topk,
        num_warps=8,
    )
    _verify_top1_dense_target_sample_kernel[(bs,)](
        predicts,
        retrieve_index,
        target_probs,
        draft_topk_indices,
        draft_topk_values,
        sample_parent,
        sample_want_residual,
        target_block_sums,
        residual_block_sums,
        uniforms,
        vocab_size,
        num_blocks,
        block_size,
        reduce_block,
        draft_token_num,
        num_draft_steps,
        topk,
        block_topk,
        uniform_width,
        num_warps=8,
    )
    return True
