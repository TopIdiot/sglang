import pytest
import torch

from sglang.srt.speculative.welmv4_mtp_staging import (
    build_welm_mtp_linear_verify_inputs,
    gather_welm_mtp_last_hash,
    pack_welm_mtp_graph_inputs,
    pack_welm_mtp_linear_graph_outputs,
    pack_welm_mtp_verify_handoff,
    translate_welm_mtp_cache_loc,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_translate_welm_mtp_cache_loc_is_cuda_graph_replayable():
    full_cache_loc = torch.tensor([7, 0, -1, 3], dtype=torch.int64, device="cuda")
    full_to_swa = torch.arange(128, dtype=torch.int64, device="cuda") + 100
    full_to_swa[-1] = -1
    swa_cache_loc = torch.empty(4, dtype=torch.int32, device="cuda")

    graph = torch.cuda.CUDAGraph()
    translate_welm_mtp_cache_loc(
        full_cache_loc=full_cache_loc,
        full_to_swa=full_to_swa,
        swa_cache_loc=swa_cache_loc,
    )
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        translate_welm_mtp_cache_loc(
            full_cache_loc=full_cache_loc,
            full_to_swa=full_to_swa,
            swa_cache_loc=swa_cache_loc,
        )

    full_cache_loc.copy_(torch.tensor([2, 5, -1, 0], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    assert swa_cache_loc.cpu().tolist() == [102, 105, -1, 100]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("with_mirror", [False, True])
@pytest.mark.parametrize("with_hash", [False, True])
def test_pack_welm_mtp_graph_inputs(with_mirror, with_hash):
    device = "cuda"
    tokens_per_bs = 4
    raw_bs = 3
    graph_bs = 4
    graph_num_tokens = graph_bs * tokens_per_bs
    hidden_size = 16
    accept_lens_cpu = [2, 1, 4]
    src_tokens = sum(accept_lens_cpu)

    input_ids = torch.arange(100, 100 + src_tokens, dtype=torch.int64, device=device)
    out_cache_loc = torch.arange(
        1000, 1000 + src_tokens, dtype=torch.int64, device=device
    )
    positions = torch.tensor(
        [10, 11, 20, 30, 31, 32, 33], dtype=torch.int64, device=device
    )
    hidden_states = torch.arange(
        src_tokens * hidden_size, dtype=torch.float32, device=device
    ).reshape(src_tokens, hidden_size)
    mirrored_kv_indices = (
        torch.arange(2000, 2000 + src_tokens, dtype=torch.int64, device=device)
        if with_mirror
        else None
    )
    cached_hash = (
        torch.arange(3 * src_tokens, dtype=torch.int64, device=device).reshape(
            3, src_tokens
        )
        if with_hash
        else None
    )
    accept_lens = torch.tensor(accept_lens_cpu, dtype=torch.int64, device=device)
    extend_start_loc = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)

    output_input_ids = torch.full(
        (graph_num_tokens,), -1, dtype=torch.int64, device=device
    )
    output_cache_loc = torch.full_like(output_input_ids, -1)
    output_positions = torch.full_like(output_input_ids, -1)
    output_hidden_states = torch.full(
        (graph_num_tokens, hidden_size), -1, dtype=torch.float32, device=device
    )
    mirror_padding_index = 9999
    output_mirror = torch.full_like(output_input_ids, -1)
    output_hash = (
        torch.full((3, graph_num_tokens), -1, dtype=torch.int64, device=device)
        if with_hash
        else None
    )
    custom_last_index = torch.full((graph_bs,), -1, dtype=torch.int64, device=device)
    custom_last_cache_loc = torch.full_like(custom_last_index, -1)

    pack_welm_mtp_graph_inputs(
        input_ids=input_ids,
        out_cache_loc=out_cache_loc,
        positions=positions,
        hidden_states=hidden_states,
        mirrored_kv_indices=mirrored_kv_indices,
        cached_hash=cached_hash,
        accept_lens=accept_lens,
        extend_start_loc=extend_start_loc,
        output_input_ids=output_input_ids,
        output_cache_loc=output_cache_loc,
        output_positions=output_positions,
        output_hidden_states=output_hidden_states,
        output_mirrored_kv_indices=output_mirror,
        output_hash=output_hash,
        custom_last_index=custom_last_index,
        custom_last_cache_loc=custom_last_cache_loc,
        raw_bs=raw_bs,
        graph_num_tokens=graph_num_tokens,
        tokens_per_bs=tokens_per_bs,
        mirror_padding_index=mirror_padding_index,
    )
    torch.cuda.synchronize()

    expected_ids = torch.zeros(graph_num_tokens, dtype=torch.int64)
    expected_cache = torch.zeros_like(expected_ids)
    expected_positions = torch.zeros_like(expected_ids)
    expected_hidden = torch.zeros(graph_num_tokens, hidden_size)
    expected_mirror = torch.full_like(expected_ids, mirror_padding_index)
    expected_hash = (
        torch.zeros(3, graph_num_tokens, dtype=torch.int64) if with_hash else None
    )
    src_start = 0
    for row, real_len in enumerate(accept_lens_cpu):
        dst_start = row * tokens_per_bs
        expected_ids[dst_start : dst_start + real_len] = input_ids[
            src_start : src_start + real_len
        ].cpu()
        expected_cache[dst_start : dst_start + real_len] = out_cache_loc[
            src_start : src_start + real_len
        ].cpu()
        expected_positions[dst_start : dst_start + real_len] = positions[
            src_start : src_start + real_len
        ].cpu()
        expected_hidden[dst_start : dst_start + real_len] = hidden_states[
            src_start : src_start + real_len
        ].cpu()
        if with_mirror:
            expected_mirror[dst_start : dst_start + real_len] = mirrored_kv_indices[
                src_start : src_start + real_len
            ].cpu()
        else:
            expected_mirror[dst_start : dst_start + real_len] = torch.arange(
                src_start, src_start + real_len
            )
        if with_hash:
            expected_hash[:, dst_start : dst_start + real_len] = cached_hash[
                :, src_start : src_start + real_len
            ].cpu()
        pad_len = tokens_per_bs - real_len
        if pad_len:
            expected_ids[dst_start + real_len : dst_start + tokens_per_bs] = input_ids[
                src_start + real_len - 1
            ].cpu()
            expected_positions[dst_start + real_len : dst_start + tokens_per_bs] = (
                positions[src_start + real_len - 1].cpu() + torch.arange(1, pad_len + 1)
            )
        src_start += real_len

    assert torch.equal(output_input_ids.cpu(), expected_ids)
    assert torch.equal(output_cache_loc.cpu(), expected_cache)
    assert torch.equal(output_positions.cpu(), expected_positions)
    assert torch.equal(output_hidden_states.cpu(), expected_hidden)
    assert torch.equal(output_mirror.cpu(), expected_mirror)
    if with_hash:
        assert torch.equal(output_hash.cpu(), expected_hash)
    assert torch.equal(custom_last_index.cpu(), torch.tensor([1, 4, 11, 15]))
    assert torch.equal(custom_last_cache_loc.cpu(), torch.tensor([1001, 1002, 1006, 0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pack_welm_mtp_verify_handoff():
    device = "cuda"
    raw_bs, graph_bs, width, hidden_size = 3, 4, 4, 16
    num_candidates = raw_bs * width
    predict = torch.arange(100, 100 + num_candidates, device=device, dtype=torch.int32)
    accept_index = torch.tensor(
        [[0, 1, -1, -1], [4, -1, -1, -1], [8, 9, 10, 11]],
        device=device,
        dtype=torch.int32,
    )
    accept_lens = torch.tensor([2, 1, 4], device=device, dtype=torch.int32)
    cache = torch.arange(1000, 1000 + num_candidates, device=device)
    hidden = torch.arange(
        num_candidates * hidden_size, device=device, dtype=torch.float32
    ).reshape(num_candidates, hidden_size)
    old_seq_lens = torch.tensor([10, 20, 30], device=device, dtype=torch.int32)
    req_pool = torch.tensor([5, 6, 7], device=device)
    bonus = torch.tensor([201, 202, 203], device=device, dtype=torch.int32)

    out_ids = torch.full((graph_bs * width,), -1, device=device, dtype=torch.int64)
    out_cache = torch.full_like(out_ids, -1)
    out_pos = torch.full_like(out_ids, -1)
    out_hidden = torch.full(
        (graph_bs * width, hidden_size), -1, device=device, dtype=torch.float32
    )
    out_mirror = torch.full_like(out_ids, -1)
    out_seq = torch.full((graph_bs,), -1, device=device, dtype=torch.int32)
    out_req = torch.full((graph_bs,), -1, device=device, dtype=torch.int64)
    out_first = torch.full((graph_bs,), -1, device=device, dtype=torch.int64)
    out_base = torch.full((graph_bs,), -1, device=device, dtype=torch.int64)
    last_idx = torch.full((graph_bs,), -1, device=device, dtype=torch.int64)
    last_cache = torch.full((graph_bs,), -1, device=device, dtype=torch.int64)

    pack_welm_mtp_verify_handoff(
        predict=predict,
        accept_index=accept_index,
        accept_lens=accept_lens,
        target_cache_loc=cache,
        target_hidden_states=hidden,
        old_seq_lens=old_seq_lens,
        req_pool_indices=req_pool,
        bonus_tokens=bonus,
        output_input_ids=out_ids,
        output_cache_loc=out_cache,
        output_positions=out_pos,
        output_hidden_states=out_hidden,
        output_mirrored_kv_indices=out_mirror,
        output_seq_lens=out_seq,
        output_req_pool_indices=out_req,
        output_first_input_ids=out_first,
        output_base_positions=out_base,
        custom_last_index=last_idx,
        custom_last_cache_loc=last_cache,
        raw_bs=raw_bs,
        graph_bs=graph_bs,
        tokens_per_bs=width,
        mirror_padding_index=99,
        seq_len_fill_value=1,
    )
    torch.cuda.synchronize()

    assert out_ids.cpu().tolist() == [
        100,
        101,
        101,
        101,
        104,
        104,
        104,
        104,
        108,
        109,
        110,
        111,
        0,
        0,
        0,
        0,
    ]
    assert out_cache.cpu().tolist() == [
        1000,
        1001,
        0,
        0,
        1004,
        0,
        0,
        0,
        1008,
        1009,
        1010,
        1011,
        0,
        0,
        0,
        0,
    ]
    assert out_pos.cpu().tolist() == [
        10,
        11,
        12,
        13,
        20,
        21,
        22,
        23,
        30,
        31,
        32,
        33,
        0,
        0,
        0,
        0,
    ]
    assert out_mirror.cpu().tolist() == [
        0,
        1,
        99,
        99,
        4,
        99,
        99,
        99,
        8,
        9,
        10,
        11,
        99,
        99,
        99,
        99,
    ]
    assert out_seq.cpu().tolist() == [12, 21, 34, 1]
    assert out_req.cpu().tolist() == [5, 6, 7, 0]
    assert out_first.cpu().tolist() == [201, 202, 203, 0]
    assert out_base.cpu().tolist() == [11, 20, 33, 0]
    assert last_idx.cpu().tolist() == [1, 4, 11, 15]
    assert last_cache.cpu().tolist() == [1001, 1004, 1011, 0]
    assert torch.equal(out_hidden[:2].cpu(), hidden[:2].cpu())
    assert torch.count_nonzero(out_hidden[2:4]).item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_gather_welm_mtp_last_hash():
    dense = torch.arange(3 * 16, device="cuda", dtype=torch.int64).reshape(3, 16)
    accept_lens = torch.tensor([2, 1, 4], device="cuda", dtype=torch.int32)
    query = torch.full((3, 4), -1, device="cuda", dtype=torch.int64)
    gather_welm_mtp_last_hash(dense, accept_lens, query, raw_bs=3, tokens_per_bs=4)
    torch.cuda.synchronize()
    expected = dense[:, torch.tensor([1, 4, 11], device="cuda")]
    assert torch.equal(query[:, :3], expected)
    assert torch.equal(query[:, 3], torch.full((3,), -1, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_build_welm_mtp_linear_verify_inputs():
    bonus = torch.tensor([10, 20], device="cuda", dtype=torch.int32)
    proposal = torch.tensor(
        [[11, 12, 13], [21, 22, 23]], device="cuda", dtype=torch.int64
    )
    seq_lens = torch.tensor([100, 200], device="cuda", dtype=torch.int32)
    tokens = torch.empty(8, device="cuda", dtype=torch.int64)
    positions = torch.empty(8, device="cuda", dtype=torch.int64)
    build_welm_mtp_linear_verify_inputs(
        bonus_tokens=bonus,
        proposal_tokens=proposal,
        seq_lens=seq_lens,
        output_tokens=tokens,
        output_positions=positions,
        batch_size=2,
        tokens_per_bs=4,
    )
    torch.cuda.synchronize()
    assert tokens.cpu().tolist() == [10, 11, 12, 13, 20, 21, 22, 23]
    assert positions.cpu().tolist() == [100, 101, 102, 103, 200, 201, 202, 203]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pack_welm_mtp_linear_graph_outputs():
    batch_size, draft_topk = 2, 8
    # Use deliberately non-contiguous row views.  The live CUDA graph sampler
    # returns views whose row stride can be larger than one/top-k, and treating
    # them as packed memory silently reads unrelated token IDs.
    sampled_probs = []
    sampled_indices = []
    draft_indices = []
    draft_values = []
    for step in range(3):
        prob_storage = torch.full((batch_size, 3), -1.0, device="cuda")
        prob_storage[:, 1] = torch.tensor([0.1 + step, 0.2 + step], device="cuda")
        sampled_probs.append(prob_storage[:, 1:2])

        index_storage = torch.full(
            (batch_size, 3), -1, device="cuda", dtype=torch.int64
        )
        index_storage[:, 1] = torch.tensor(
            [11 + step, 21 + step], device="cuda", dtype=torch.int64
        )
        sampled_indices.append(index_storage[:, 1:2])

        index_values = torch.arange(
            step * 100,
            step * 100 + batch_size * draft_topk,
            device="cuda",
            dtype=torch.int64,
        ).reshape(batch_size, draft_topk)
        index_storage_3d = torch.full(
            (batch_size, 2, draft_topk), -1, device="cuda", dtype=torch.int64
        )
        index_storage_3d[:, 1, :] = index_values
        draft_indices.append(index_storage_3d[:, 1, :])

        value_storage_3d = torch.full((batch_size, 2, draft_topk), -1.0, device="cuda")
        value_storage_3d[:, 1, :] = index_values.float() / 1000
        draft_values.append(value_storage_3d[:, 1, :])
    topk_p = torch.empty((batch_size, 3), device="cuda")
    topk_index = torch.empty((batch_size, 3), device="cuda", dtype=torch.int64)
    proposal = torch.empty_like(topk_index)
    sparse_i = torch.empty(
        (batch_size, 4, draft_topk), device="cuda", dtype=torch.int64
    )
    sparse_v = torch.empty(
        (batch_size, 4, draft_topk), device="cuda", dtype=torch.float32
    )
    bonus = torch.tensor([10, 20], device="cuda", dtype=torch.int64)
    base_positions = torch.tensor([99, 199], device="cuda", dtype=torch.int64)
    verify_tokens = torch.empty(batch_size * 4, device="cuda", dtype=torch.int64)
    verify_positions = torch.empty_like(verify_tokens)
    pack_welm_mtp_linear_graph_outputs(
        sampled_probs=sampled_probs,
        sampled_indices=sampled_indices,
        draft_topk_indices=draft_indices,
        draft_topk_values=draft_values,
        bonus_tokens=bonus,
        base_positions=base_positions,
        hot_token_map=None,
        output_topk_p=topk_p,
        output_topk_index=topk_index,
        output_proposal_tokens=proposal,
        output_draft_indices=sparse_i,
        output_draft_values=sparse_v,
        output_verify_tokens=verify_tokens,
        output_verify_positions=verify_positions,
        batch_size=batch_size,
    )
    torch.cuda.synchronize()

    assert topk_index.cpu().tolist() == [[11, 12, 13], [21, 22, 23]]
    assert torch.equal(proposal, topk_index)
    assert torch.equal(sparse_i[:, 0], draft_indices[0])
    assert torch.equal(sparse_i[:, 1], draft_indices[1])
    assert torch.equal(sparse_i[:, 2], draft_indices[2])
    assert torch.count_nonzero(sparse_i[:, 3]).item() == 0
    assert torch.count_nonzero(sparse_v[:, 3]).item() == 0
    assert verify_tokens.cpu().tolist() == [10, 11, 12, 13, 20, 21, 22, 23]
    assert verify_positions.cpu().tolist() == [100, 101, 102, 103, 200, 201, 202, 203]
