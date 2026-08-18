from types import SimpleNamespace
from unittest.mock import patch

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
    runner.graphs_by_mode = {(False, False): {}}
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

    runner.graphs_by_mode[(True, False)] = {}
    assert runner.can_replay_from_verify(batch, batch_result)

    runner.requires_top_p = True
    assert not runner.can_replay_from_verify(batch, batch_result)

    runner.graphs_by_mode[(True, True)] = {}
    assert runner.can_replay_from_verify(batch, batch_result)


def test_draft_sampling_topk_defaults_when_unset_or_invalid(monkeypatch):
    from sglang.srt.speculative.eagle_worker_v2 import (
        _WELM_MTP_DRAFT_SAMPLING_TOPK_ENV,
        EagleDraftWorker,
    )

    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.draft_runner = SimpleNamespace(model_config=SimpleNamespace(vocab_size=1000))

    monkeypatch.delenv(_WELM_MTP_DRAFT_SAMPLING_TOPK_ENV, raising=False)
    assert worker._get_welmv4_mtp_draft_sampling_topk() == 8

    monkeypatch.setenv(_WELM_MTP_DRAFT_SAMPLING_TOPK_ENV, "20")
    assert worker._get_welmv4_mtp_draft_sampling_topk() == 20

    monkeypatch.setenv(_WELM_MTP_DRAFT_SAMPLING_TOPK_ENV, "0")
    with pytest.raises(ValueError, match="must be in"):
        worker._get_welmv4_mtp_draft_sampling_topk()

    monkeypatch.setenv(_WELM_MTP_DRAFT_SAMPLING_TOPK_ENV, "1000")
    with pytest.raises(ValueError, match="must be in"):
        worker._get_welmv4_mtp_draft_sampling_topk()


def test_draft_sampling_topk_bounded_by_draft_logits_width(monkeypatch):
    from sglang.srt.speculative.eagle_worker_v2 import (
        _WELM_MTP_DRAFT_SAMPLING_TOPK_ENV,
        EagleDraftWorker,
    )

    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.draft_runner = SimpleNamespace(
        model_config=SimpleNamespace(
            vocab_size=1000,
            hf_config=SimpleNamespace(hot_vocab_size=100),
        )
    )

    monkeypatch.setenv(_WELM_MTP_DRAFT_SAMPLING_TOPK_ENV, "20")
    assert worker._get_welmv4_mtp_draft_sampling_topk() == 20

    # K within the main vocab but beyond the hot-vocab logits width must be
    # rejected: torch.topk over the draft logits cannot satisfy it.
    monkeypatch.setenv(_WELM_MTP_DRAFT_SAMPLING_TOPK_ENV, "200")
    with pytest.raises(ValueError, match="must be in"):
        worker._get_welmv4_mtp_draft_sampling_topk()


def _root_only_sampling_state():
    predicts = torch.arange(8, device="cuda", dtype=torch.int32) + 100
    accept_index = torch.tensor(
        [[0, 1, 2, -1], [4, 5, 6, -1]], device="cuda", dtype=torch.int32
    )
    correct_drafts = torch.tensor([2, 2], device="cuda", dtype=torch.int32)
    retrieve_index = torch.arange(8, device="cuda", dtype=torch.int64).reshape(2, 4)
    root_only = torch.tensor([False, True], device="cuda", dtype=torch.bool)
    return predicts, accept_index, correct_drafts, retrieve_index, root_only


def _assert_root_only_sampling_result(
    predicts, accept_index, correct_drafts, expected_token
):
    assert predicts[:4].cpu().tolist() == [100, 101, 102, 103]
    assert predicts[4].item() == expected_token
    assert accept_index[0].cpu().tolist() == [0, 1, 2, -1]
    assert accept_index[1].cpu().tolist() == [4, -1, -1, -1]
    assert correct_drafts.cpu().tolist() == [2, 0]


def test_root_only_greedy_sampling_preserves_ordinary_rows():
    from sglang.srt.speculative.welmv4_mtp_sampling import (
        welmv4_mtp_apply_root_only_sampling,
    )

    state = _root_only_sampling_state()
    target_predict = torch.tensor(
        [[1, 2, 3, 4], [9, 8, 7, 6]], device="cuda", dtype=torch.int64
    )

    welmv4_mtp_apply_root_only_sampling(
        predicts=state[0],
        accept_index=state[1],
        accept_token_num=state[2],
        retrieve_index=state[3],
        root_only_verify_mask=state[4],
        target_predict=target_predict,
    )

    _assert_root_only_sampling_result(*state[:3], expected_token=9)


def test_root_only_dense_sampling_uses_one_root_uniform():
    from sglang.srt.speculative.welmv4_mtp_sampling import (
        welmv4_mtp_apply_root_only_sampling,
    )

    state = _root_only_sampling_state()
    target_probs = torch.zeros((2, 4, 5), device="cuda", dtype=torch.float32)
    target_probs[1, 0] = torch.tensor(
        [0.1, 0.2, 0.7, 0.0, 0.0], device="cuda"
    )

    welmv4_mtp_apply_root_only_sampling(
        predicts=state[0],
        accept_index=state[1],
        accept_token_num=state[2],
        retrieve_index=state[3],
        root_only_verify_mask=state[4],
        target_probs=target_probs,
        root_uniforms=torch.tensor([0.9, 0.25], device="cuda"),
    )

    _assert_root_only_sampling_result(*state[:3], expected_token=1)


def test_root_only_sparse_sampling_uses_root_target_distribution():
    from sglang.srt.speculative.welmv4_mtp_sampling import (
        welmv4_mtp_apply_root_only_sampling,
    )

    state = _root_only_sampling_state()
    target_topk_indices = torch.zeros((2, 4, 3), device="cuda", dtype=torch.int64)
    target_topk_values = torch.zeros((2, 4, 3), device="cuda", dtype=torch.float32)
    target_topk_indices[1, 0] = torch.tensor([4, 2, 7], device="cuda")
    target_topk_values[1, 0] = torch.tensor([0.1, 0.6, 0.3], device="cuda")

    welmv4_mtp_apply_root_only_sampling(
        predicts=state[0],
        accept_index=state[1],
        accept_token_num=state[2],
        retrieve_index=state[3],
        root_only_verify_mask=state[4],
        target_topk_indices=target_topk_indices,
        target_topk_values=target_topk_values,
        root_uniforms=torch.tensor([0.9, 0.5], device="cuda"),
    )

    _assert_root_only_sampling_result(*state[:3], expected_token=2)


def _root_only_greedy_verify_input():
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    class NoDevicePredicateTensor(torch.Tensor):
        def any(self, *args, **kwargs):
            raise AssertionError("root-only sampling read a device predicate")

    draft_tokens = torch.tensor(
        [20, 0, 0, 0], device="cuda", dtype=torch.int64
    )
    verify_input = EagleVerifyInput(
        draft_token=draft_tokens,
        custom_mask=torch.ones((16,), device="cuda", dtype=torch.bool),
        positions=torch.arange(4, device="cuda", dtype=torch.int64),
        retrieve_index=torch.arange(
            4, device="cuda", dtype=torch.int64
        ).reshape(1, 4),
        retrieve_next_token=torch.tensor(
            [[1, 2, 3, -1]], device="cuda", dtype=torch.int64
        ),
        retrieve_next_sibling=torch.full(
            (1, 4), -1, device="cuda", dtype=torch.int64
        ),
        retrieve_cum_len=None,
        spec_steps=3,
        topk=1,
        draft_token_num=4,
        capture_hidden_mode=None,
        seq_lens_sum=None,
        seq_lens_cpu=None,
        welm_mtp_root_only_verify_mask=torch.ones(
            (1,), device="cuda", dtype=torch.bool
        ).as_subclass(NoDevicePredicateTensor),
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([20], device="cuda", dtype=torch.int32),
        sampling_info=SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            is_all_greedy=True,
        ),
        input_ids=draft_tokens,
    )
    return verify_input, batch


def test_root_only_verify_then_ordinary_verify_accepts_drafts():
    verify_input, batch = _root_only_greedy_verify_input()
    root_logits = torch.full((4, 32), -100.0, device="cuda")
    root_logits[0, 7] = 10

    root_predict, root_accept_lens, root_accept_index = verify_input.sample(
        batch, SimpleNamespace(next_token_logits=root_logits)
    )

    assert root_accept_lens.cpu().tolist() == [1]
    assert root_accept_index[0].cpu().tolist() == [0, -1, -1, -1]
    assert root_predict[0].item() == 7

    verify_input.welm_mtp_root_only_verify_mask = None
    verify_input.draft_token = torch.tensor(
        [7, 8, 9, 10], device="cuda", dtype=torch.int64
    )
    batch.input_ids = verify_input.draft_token
    ordinary_logits = torch.full((4, 32), -100.0, device="cuda")
    for row, token in enumerate((8, 9, 10, 11)):
        ordinary_logits[row, token] = 10

    _, ordinary_accept_lens, ordinary_accept_index = verify_input.sample(
        batch, SimpleNamespace(next_token_logits=ordinary_logits)
    )

    assert ordinary_accept_lens.cpu().tolist() == [4]
    assert ordinary_accept_index[0].cpu().tolist() == [0, 1, 2, 3]


def test_deterministic_target_only_sampling_uses_contiguous_uniforms():
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    draft_tokens = torch.tensor(
        [20, 21, 22, 23, 30, 31, 32, 33], device="cuda", dtype=torch.int64
    )
    verify_input = EagleVerifyInput(
        draft_token=draft_tokens,
        custom_mask=torch.ones((32,), device="cuda", dtype=torch.bool),
        positions=torch.arange(8, device="cuda", dtype=torch.int64),
        retrieve_index=torch.arange(
            8, device="cuda", dtype=torch.int64
        ).reshape(2, 4),
        retrieve_next_token=torch.tensor(
            [[1, 2, 3, -1], [1, 2, 3, -1]], device="cuda", dtype=torch.int64
        ),
        retrieve_next_sibling=torch.full(
            (2, 4), -1, device="cuda", dtype=torch.int64
        ),
        retrieve_cum_len=None,
        spec_steps=3,
        topk=1,
        draft_token_num=4,
        capture_hidden_mode=None,
        seq_lens_sum=None,
        seq_lens_cpu=None,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([20, 30], device="cuda", dtype=torch.int32),
        sampling_info=SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            is_all_greedy=False,
            sampling_seed=torch.tensor([123, 456], device="cuda", dtype=torch.int64),
            temperatures=torch.ones((2, 1), device="cuda"),
            need_top_k_sampling=False,
            need_top_p_sampling=False,
        ),
        input_ids=draft_tokens,
    )
    logits = torch.full((8, 64), -100.0, device="cuda")
    for row, token in enumerate((21, 22, 23, 24, 31, 32, 33, 34)):
        logits[row, token] = 100

    with (
        patch(
            "sglang.srt.speculative.eagle_info_v2.get_tp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        patch(
            "sglang.srt.speculative.eagle_info_v2.get_global_server_args",
            return_value=SimpleNamespace(
                speculative_accept_threshold_single=1.0,
                speculative_accept_threshold_acc=1.0,
            ),
        ),
    ):
        _, accept_lens, _ = verify_input.sample(
            batch, SimpleNamespace(next_token_logits=logits)
        )

    assert accept_lens.shape == (2,)


def test_root_only_verify_samples_from_grammar_masked_target_root():
    verify_input, batch = _root_only_greedy_verify_input()

    class BoolMaskGrammar:
        @staticmethod
        def apply_vocab_mask(logits, vocab_mask):
            logits.masked_fill_(~vocab_mask, -float("inf"))

    verify_input.grammar = BoolMaskGrammar()
    logits = torch.full((4, 32), -100.0, device="cuda")
    logits[0, 7] = 10
    logits[0, 5] = 5
    vocab_mask = torch.zeros_like(logits, dtype=torch.bool)
    vocab_mask[:, 0] = True
    vocab_mask[0, 5] = True

    predict, accept_lens, accept_index = verify_input.sample(
        batch,
        SimpleNamespace(next_token_logits=logits),
        vocab_mask=vocab_mask,
    )

    assert accept_lens.cpu().tolist() == [1]
    assert accept_index[0].cpu().tolist() == [0, -1, -1, -1]
    assert predict[0].item() == 5


@pytest.mark.parametrize(
    ("oe_grams", "oe_vocab_sizes", "prompt"),
    [
        ((2, 3), (97, 101), [11, 12, 13, 14]),
        ((2, 4), (97, 103), [21, 22, 23, 24, 25]),
    ],
)
def test_root_only_oe_bootstrap_matches_decode_reference(
    oe_grams, oe_vocab_sizes, prompt
):
    from sglang.jit_kernel.welm_oe import (
        welm_oe_hash_decode_from_prefixes_cuda,
    )
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    history_width = max(oe_grams)
    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.device = torch.device("cuda")
    worker.topk = 1
    worker.draft_runner = SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=1024)
    )
    worker._should_use_welmv4_mtp_oe_hash_kernel = lambda: True
    worker._welmv4_mtp_oe_hash_config = lambda: (
        oe_grams,
        oe_vocab_sizes,
        history_width,
    )
    worker._welmv4_mtp_oe_prefix_width = lambda: history_width - 1

    ordinary_history = torch.arange(
        100, 100 + history_width, dtype=torch.int64, device="cuda"
    )
    draft_input = SimpleNamespace(
        bonus_tokens=torch.tensor([90, prompt[-1]], device="cuda"),
        welm_mtp_root_only_verify_mask=torch.tensor(
            [False, True], dtype=torch.bool, device="cuda"
        ),
        welm_mtp_oe_history_state=torch.stack(
            (ordinary_history, torch.zeros_like(ordinary_history))
        ),
    )
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(origin_input_ids=[80, 90], output_ids=[]),
            SimpleNamespace(origin_input_ids=prompt, output_ids=[]),
        ]
    )
    worker._prepare_welmv4_mtp_root_only_oe_history(draft_input, batch)

    torch.testing.assert_close(
        draft_input.welm_mtp_oe_history_state[0], ordinary_history
    )
    root_history = draft_input.welm_mtp_oe_history_state[1:2]
    expected_root_history = torch.tensor(
        [prompt[-history_width:]], dtype=torch.int64, device="cuda"
    )
    torch.testing.assert_close(root_history, expected_root_history)

    generated = torch.tensor([31], dtype=torch.int64, device="cuda")
    query_hash, _, next_verify_history = (
        worker._compute_welmv4_mtp_first_query_hash_from_entry_history(
            SimpleNamespace(), generated, root_history
        )
    )
    expected_next_history = torch.cat(
        (root_history[:, 1:], generated[:, None]), dim=1
    )
    torch.testing.assert_close(next_verify_history, expected_next_history)

    reference_hash = torch.empty(
        (len(oe_grams), 1), dtype=torch.int64, device="cuda"
    )
    decode_prefixes = [
        int(root_history[0, -1 - lag].item()) for lag in range(history_width - 1)
    ]
    welm_oe_hash_decode_from_prefixes_cuda(
        generated,
        decode_prefixes,
        oe_grams,
        oe_vocab_sizes,
        reference_hash,
        vocab_size=1024,
    )
    torch.testing.assert_close(query_hash, reference_hash)


def test_eagle_verify_sparse_fused_sampling_applies_root_only_mask():
    from sglang.srt.speculative.eagle_info import EagleVerifyInput

    draft_tokens = torch.tensor(
        [10, 11, 12, 13, 20, 0, 0, 0], device="cuda", dtype=torch.int64
    )
    retrieve_index = torch.arange(8, device="cuda", dtype=torch.int64).reshape(2, 4)
    retrieve_next_token = torch.tensor(
        [[1, 2, 3, -1], [1, 2, 3, -1]], device="cuda", dtype=torch.int64
    )
    draft_topk_indices = torch.zeros((2, 4, 4), device="cuda", dtype=torch.int64)
    draft_topk_values = torch.zeros((2, 4, 4), device="cuda", dtype=torch.float32)
    for parent, token in enumerate((11, 12, 13)):
        draft_topk_indices[0, parent, 0] = token
        draft_topk_values[0, parent, 0] = 1
    verify_input = EagleVerifyInput(
        draft_token=draft_tokens,
        custom_mask=torch.ones((32,), device="cuda", dtype=torch.bool),
        positions=torch.arange(8, device="cuda", dtype=torch.int64),
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=torch.full(
            (2, 4), -1, device="cuda", dtype=torch.int64
        ),
        retrieve_cum_len=None,
        spec_steps=3,
        topk=1,
        draft_token_num=4,
        capture_hidden_mode=None,
        seq_lens_sum=None,
        seq_lens_cpu=None,
        draft_topk_indices=draft_topk_indices,
        draft_topk_values=draft_topk_values,
        welm_mtp_root_only_verify_mask=torch.tensor(
            [False, True], device="cuda", dtype=torch.bool
        ),
    )
    sampling_info = SimpleNamespace(
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
        is_all_greedy=False,
        sampling_seed=torch.tensor([123, 456], device="cuda", dtype=torch.int64),
        temperatures=torch.ones((2, 1), device="cuda"),
        need_top_p_sampling=False,
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([10, 20], device="cuda", dtype=torch.int32),
        sampling_info=sampling_info,
        input_ids=draft_tokens,
    )
    logits = torch.full((8, 32), -100.0, device="cuda")
    for row, token in enumerate((11, 12, 13, 14, 7, 0, 0, 9)):
        logits[row, token] = 100

    with (
        patch(
            "sglang.srt.speculative.eagle_info_v2.get_tp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        patch(
            "sglang.srt.speculative.eagle_info_v2._welm_mtp_verify_sampling_topk",
            return_value=4,
        ),
    ):
        predict, accept_lens, accept_index = verify_input.sample(
            batch, SimpleNamespace(next_token_logits=logits)
        )

    assert accept_lens.cpu().tolist() == [4, 1]
    assert accept_index[1].cpu().tolist() == [4, -1, -1, -1]
    assert predict[4].item() == 7

    with (
        patch(
            "sglang.srt.speculative.eagle_info_v2.get_tp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        patch(
            "sglang.srt.speculative.eagle_info_v2._welm_mtp_verify_sampling_topk",
            return_value=4,
        ),
        patch(
            "sglang.srt.speculative.eagle_info_v2."
            "welmv4_mtp_fused_verify_top1_sparse",
            side_effect=RuntimeError("fused failure"),
        ),
        pytest.raises(RuntimeError, match="root-only sparse verify fast path failed"),
    ):
        verify_input.sample(batch, SimpleNamespace(next_token_logits=logits))
