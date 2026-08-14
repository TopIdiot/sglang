from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="WeLM MTP fused sampling requires CUDA"
)


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_fused_topk_top_p_sample_matches_reference(batch_size: int):
    from sgl_kernel import top_p_renorm_prob

    from sglang.srt.speculative.welmv4_mtp_sampling import (
        welmv4_mtp_fused_topk_softmax_sample,
    )

    torch.manual_seed(42 + batch_size)
    logits = torch.randn(batch_size, 8192, dtype=torch.bfloat16, device="cuda")
    temperature = torch.linspace(0.7, 1.3, batch_size, device="cuda")
    top_p = torch.linspace(0.75, 0.99, batch_size, device="cuda")
    uniform = torch.linspace(0.01, 0.99, batch_size, device="cuda")

    sampled_p, sampled_i, top_i, top_probs = welmv4_mtp_fused_topk_softmax_sample(
        logits,
        temperature,
        uniform,
        8,
        top_p=top_p,
    )
    ref_logits, ref_i = torch.topk(logits.float(), 8, dim=-1, sorted=True)
    ref_probs = top_p_renorm_prob(
        torch.softmax(ref_logits / temperature[:, None], dim=-1), top_p
    )
    ref_pos = torch.sum(
        torch.cumsum(ref_probs, dim=-1) < uniform[:, None],
        dim=-1,
        keepdim=True,
    ).long()
    ref_pos.clamp_(max=7)
    ref_sampled_i = torch.gather(ref_i, 1, ref_pos)
    ref_sampled_p = torch.gather(ref_probs, 1, ref_pos)

    # torch.topk does not promise which index wins an equal-value BF16 tie.
    # Compare the selected values, probabilities, and sampled value so either
    # valid tie order is accepted without weakening the numerical check.
    torch.testing.assert_close(
        torch.gather(logits.float(), 1, top_i), ref_logits, rtol=0, atol=0
    )
    torch.testing.assert_close(top_probs, ref_probs, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(sampled_p, ref_sampled_p, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        torch.gather(logits.float(), 1, sampled_i),
        torch.gather(logits.float(), 1, ref_sampled_i),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize(
    ("world_size", "local_vocab"),
    [
        (2, 4096),
        # Production WeLM shape: 155,648 vocabulary entries sharded over TP8.
        # This also proves that packing token IDs through FP32 remains exact at
        # the largest ID used by the deployed model.
        (8, 155648 // 8),
    ],
)
def test_packed_distributed_topk_union_contains_exact_global_topk(
    batch_size: int, world_size: int, local_vocab: int
):
    from sglang.srt.speculative.welmv4_mtp_sampling import (
        welmv4_mtp_local_topk_pack,
        welmv4_mtp_unpack_gathered_topk,
    )

    torch.manual_seed(1729 + batch_size)
    topk = 8
    shards = [
        torch.randn(batch_size, local_vocab, dtype=torch.bfloat16, device="cuda")
        for _ in range(world_size)
    ]
    packed = [
        welmv4_mtp_local_topk_pack(shard, topk, index_offset=rank * local_vocab)
        for rank, shard in enumerate(shards)
    ]
    assert all(item is not None for item in packed)
    gathered = torch.cat(packed, dim=-1)
    candidate_ids = torch.empty(
        (batch_size, world_size * topk), dtype=torch.int64, device="cuda"
    )
    candidate_values = welmv4_mtp_unpack_gathered_topk(
        gathered,
        candidate_ids,
        topk=topk,
        world_size=world_size,
    )
    assert candidate_values is not None

    global_logits = torch.cat(shards, dim=-1).float()
    reference_values, _ = torch.topk(global_logits, topk, dim=-1, sorted=True)
    candidate_global_values = torch.gather(global_logits, 1, candidate_ids)
    torch.testing.assert_close(
        candidate_values, candidate_global_values, rtol=0, atol=0
    )
    union_values, _ = torch.topk(candidate_values, topk, dim=-1, sorted=True)
    torch.testing.assert_close(union_values, reference_values, rtol=0, atol=0)


def test_packed_distributed_topk_rejects_inexact_fp32_token_ids():
    from sglang.srt.speculative.welmv4_mtp_sampling import (
        welmv4_mtp_local_topk_pack,
    )

    logits = torch.randn(1, 16, dtype=torch.bfloat16, device="cuda")
    assert (
        welmv4_mtp_local_topk_pack(
            logits,
            8,
            index_offset=(1 << 24) - 15,
        )
        is None
    )


def test_persistent_handoff_requires_captured_sampling_mode():
    from sglang.srt.speculative.welmv4_mtp_draft_proposal_cuda_graph_runner import (
        WelmMTPDraftProposalCudaGraphRunner,
    )

    runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
        WelmMTPDraftProposalCudaGraphRunner
    )
    runner.persistent_handoff = True
    runner.topk = 1
    runner.dp_size = 1
    runner.capture_bs = [1]
    runner.num_tokens_per_bs = 5
    runner.use_dp_sampling_consensus = False
    runner.sample_draft = False
    runner.use_top_p = False
    runner.eagle_worker = SimpleNamespace(
        _should_sample_welmv4_mtp_draft=lambda batch: True,
        _should_use_welmv4_mtp_draft_top_p=lambda batch: runner.requires_top_p,
    )
    runner.requires_top_p = False
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens_cpu=[1],
        seq_lens=torch.tensor([1]),
        out_cache_loc=torch.tensor([0]),
        req_pool_indices=torch.tensor([0]),
    )
    batch_result = SimpleNamespace(
        spec_accept_index=torch.zeros((1, 5), dtype=torch.long),
        accept_lens=torch.ones(1, dtype=torch.long),
        logits_output=SimpleNamespace(hidden_states=torch.empty((1, 1))),
    )

    assert not runner.can_replay_from_verify(batch, batch_result)

    runner.sample_draft = True
    assert runner.can_replay_from_verify(batch, batch_result)

    runner.requires_top_p = True
    assert not runner.can_replay_from_verify(batch, batch_result)

    runner.use_top_p = True
    assert runner.can_replay_from_verify(batch, batch_result)
