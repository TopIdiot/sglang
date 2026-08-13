from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.models import welmv4
from sglang.srt.models import welmv4_token_owner as token_owner
from sglang.srt.speculative.welmv4_mtp_draft_proposal_cuda_graph_runner import (
    WelmMTPDraftProposalCudaGraphRunner,
)


def valid_capability_kwargs():
    return {
        "enabled": True,
        "dp_attention_enabled": True,
        "attn_cp_size": 1,
        "attn_tp_size": 2,
        "pp_size": 1,
        "global_tp_size": 8,
        "global_tp_rank": 5,
        "attn_tp_rank": 1,
        "moe_a2a_backend": "none",
        "use_previous_precision": False,
        "suffix_parallel_enabled": False,
        "tbo_enabled": False,
        "router_replay_enabled": False,
        "mk_router_enabled": False,
        "speculative_enabled": False,
        "speculative_moe_a2a_backend": None,
        "deepep_mode": "auto",
        "disaggregation_mode": "null",
        "attn_tp_input_scattered": False,
    }


@pytest.mark.parametrize("moe_a2a_backend", ["none", "deepep"])
def test_welm_dp_attention_supports_token_owner(moe_a2a_backend):
    kwargs = valid_capability_kwargs()
    kwargs["moe_a2a_backend"] = moe_a2a_backend

    assert token_owner.validate_welm_token_owner_capability(**kwargs)


@pytest.mark.parametrize(
    ("deepep_mode", "disaggregation_mode"),
    [("auto", "null"), ("auto", "decode"), ("low_latency", "decode")],
)
@pytest.mark.parametrize("speculative_moe_a2a_backend", [None, "deepep"])
def test_welm_token_owner_supports_speculative_deepep_graph(
    deepep_mode, disaggregation_mode, speculative_moe_a2a_backend
):
    kwargs = valid_capability_kwargs()
    kwargs.update(
        speculative_enabled=True,
        moe_a2a_backend="deepep",
        speculative_moe_a2a_backend=speculative_moe_a2a_backend,
        deepep_mode=deepep_mode,
        disaggregation_mode=disaggregation_mode,
    )

    assert token_owner.validate_welm_token_owner_capability(**kwargs)


@pytest.mark.parametrize("speculative_enabled", [False, True])
def test_welm_token_owner_rejects_low_latency_for_non_disaggregated_prefill(
    speculative_enabled,
):
    kwargs = valid_capability_kwargs()
    kwargs.update(
        speculative_enabled=speculative_enabled,
        moe_a2a_backend="deepep",
        deepep_mode="low_latency",
    )

    with pytest.raises(NotImplementedError, match="decode-only"):
        token_owner.validate_welm_token_owner_capability(**kwargs)


@pytest.mark.parametrize(
    ("target_backend", "speculative_backend"),
    [
        ("deepep", "none"),
        ("none", "deepep"),
        ("none", "mooncake"),
    ],
)
def test_welm_token_owner_rejects_mismatched_speculative_moe_backend(
    target_backend, speculative_backend
):
    kwargs = valid_capability_kwargs()
    kwargs.update(
        speculative_enabled=True,
        moe_a2a_backend=target_backend,
        speculative_moe_a2a_backend=speculative_backend,
    )

    with pytest.raises(NotImplementedError, match="Global TP MoE or matching DeepEP"):
        token_owner.validate_welm_token_owner_capability(**kwargs)


def test_welm_runtime_token_owner_capability_uses_parallel_groups(monkeypatch):
    monkeypatch.setattr(
        token_owner,
        "get_global_server_args",
        lambda: SimpleNamespace(
            enable_token_owner=True,
            enable_two_batch_overlap=False,
            enable_moe_router_replay=False,
            speculative_algorithm=None,
            speculative_moe_a2a_backend=None,
            deepep_mode="auto",
            disaggregation_mode="null",
            enable_attn_tp_input_scattered=False,
            enable_return_routed_experts=False,
            expert_distribution_recorder_mode=None,
        ),
    )
    monkeypatch.setattr(token_owner, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(token_owner, "get_attention_cp_size", lambda: 1)
    monkeypatch.setattr(token_owner, "get_attention_tp_size", lambda: 4)
    monkeypatch.setattr(
        token_owner,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=8, rank_in_group=6),
    )
    monkeypatch.setattr(
        token_owner,
        "get_attn_tp_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=2),
    )
    monkeypatch.setattr(
        token_owner,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(value="none"),
    )
    monkeypatch.setattr(token_owner, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(token_owner, "is_suffix_parallel_enabled", lambda: False)
    monkeypatch.setattr(
        token_owner,
        "get_mk_moe_router_mode",
        lambda: token_owner.MkMoeRouterMode.OFF,
    )

    assert welmv4._welm_token_owner_enabled(pp_size=1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dp_attention_enabled": False}, "DP Attention"),
        ({"attn_cp_size": 2}, "AttnCP1"),
        ({"attn_tp_size": 1}, "AttnTP>1"),
        ({"pp_size": 2}, "PP1"),
        ({"global_tp_size": 7}, "divisible"),
        ({"attn_tp_rank": 0}, "rank ordering"),
        ({"moe_a2a_backend": "mooncake"}, "none or DeepEP"),
        ({"use_previous_precision": True}, "previous-precision"),
        ({"suffix_parallel_enabled": True}, "suffix parallel"),
        ({"tbo_enabled": True}, "TBO"),
        ({"router_replay_enabled": True}, "Router Replay"),
        ({"mk_router_enabled": True}, "MK MoE Router"),
        ({"attn_tp_input_scattered": True}, "input scatter"),
    ],
)
def test_welm_token_owner_rejects_unsupported_capability(overrides, message):
    kwargs = valid_capability_kwargs()
    kwargs.update(overrides)

    with pytest.raises((RuntimeError, NotImplementedError), match=message):
        token_owner.validate_welm_token_owner_capability(**kwargs)


def test_welm_token_owner_runtime_builds_local_and_global_layouts():
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=2),
    )
    forward_batch = SimpleNamespace(global_num_tokens_cpu=[5, 3])

    local_layout = runtime.local_layout(forward_batch)
    global_layout = runtime.global_layout(forward_batch)

    assert local_layout.owner_sizes == (2, 1)
    assert local_layout.local_owner_rank == 0
    assert local_layout.local_valid_rows == 2
    assert global_layout.owner_sizes == (3, 2, 2, 1)
    assert global_layout.local_owner_rank == 2
    assert global_layout.local_valid_rows == 2


def test_welm_token_owner_runtime_recomputes_layout_only_after_invalidation():
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=1),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=1),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=[8, 4],
        original_global_num_tokens_cpu=[8, 4],
        welm_kv_mirror_contract_flags=[True, False],
        welm_kv_mirror_last_q_indices_cpu=[3, 7],
        welm_kv_mirror_active_batch_indices_cpu=[0, 2],
    )

    runtime.begin_forward(forward_batch)
    initial = runtime.local_layout(forward_batch)
    assert runtime.local_layout(forward_batch) is initial

    forward_batch.global_num_tokens_cpu = [3, 4]
    forward_batch._welm_kv_mirror_contracted_dp_metadata_rows = 3
    runtime.invalidate()
    contracted = runtime.local_layout(forward_batch)

    assert contracted is not initial
    assert contracted.owner_sizes == (2, 1)


@pytest.mark.parametrize(
    ("original_tokens", "owner_count", "last_q", "active", "output_size", "expected"),
    [
        (12, 2, [2, 7, 11], [0, 2, 3], 4, (2, 2)),
        (12, 4, [2, 5, 8, 11], [0, 1, 2, 3], 4, (1, 1, 1, 1)),
        (8, 4, [1, 2], [1, 2], 4, (2, 2, 0, 0)),
        (0, 4, [], [], 3, (3, 0, 0, 0)),
    ],
)
def test_welm_kv_mirror_owner_sizes_preserve_survivor_lanes(
    original_tokens,
    owner_count,
    last_q,
    active,
    output_size,
    expected,
):
    assert (
        token_owner.kv_mirror_owner_sizes(
            original_token_count=original_tokens,
            owner_count=owner_count,
            last_q_indices=last_q,
            active_batch_indices=active,
            output_size=output_size,
        )
        == expected
    )


def test_welm_kv_mirror_init_publishes_cpu_rows_from_extend_lens(monkeypatch):
    monkeypatch.setattr(welmv4, "_welm_kv_mirror_row_alignment", lambda: 1)
    forward_batch = SimpleNamespace(
        kv_mirror_output_size=None,
        attn_cp_prefill_runtime_layout=None,
        welm_kv_mirror_last_q_indices=None,
        custom_last_index=None,
        extend_seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        extend_seq_lens_cpu=[3, 2],
        out_cache_loc=None,
        req_to_token_pool=None,
        req_pool_indices=None,
        seq_lens=None,
    )

    assert welmv4._welm_init_kv_mirror_last_q_indices(forward_batch)
    assert forward_batch.custom_last_index.tolist() == [2, 4]
    assert forward_batch.welm_kv_mirror_last_q_indices_cpu == [2, 4]
    assert forward_batch.welm_kv_mirror_active_batch_indices_cpu == [0, 1]


def test_welm_kv_mirror_init_publishes_empty_cpu_rows_for_idle_owner(
    monkeypatch,
):
    monkeypatch.setattr(welmv4, "_welm_kv_mirror_row_alignment", lambda: 4)
    monkeypatch.setattr(
        welmv4,
        "get_global_server_args",
        lambda: SimpleNamespace(enable_token_owner=True),
    )
    forward_batch = SimpleNamespace(
        kv_mirror_output_size=None,
        attn_cp_prefill_runtime_layout=None,
        welm_kv_mirror_last_q_indices=None,
        custom_last_index=torch.empty(0, dtype=torch.long, device="meta"),
        extend_seq_lens_cpu=None,
        out_cache_loc=None,
        req_to_token_pool=None,
        req_pool_indices=None,
        seq_lens=None,
    )

    assert welmv4._welm_init_kv_mirror_last_q_indices(forward_batch)
    assert forward_batch.welm_kv_mirror_last_q_indices_cpu == []
    assert forward_batch.welm_kv_mirror_active_batch_indices_cpu == []


def test_welm_token_owner_runtime_switches_to_mirror_proxy_layout():
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=1),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=1),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=[8, 4],
        original_global_num_tokens_cpu=[8, 4],
        welm_kv_mirror_contract_flags=[True, False],
        extend_seq_lens_cpu=[4, 4],
        welm_kv_mirror_last_q_indices_cpu=[3, 7],
        welm_kv_mirror_active_batch_indices_cpu=[0, 2],
    )

    initial = runtime.local_layout(forward_batch)
    assert initial.owner_sizes == (4, 4)

    forward_batch.global_num_tokens_cpu = [3, 4]
    forward_batch._welm_kv_mirror_contracted_dp_metadata_rows = 3
    runtime.invalidate()

    local_layout = runtime.local_layout(forward_batch)
    global_layout = runtime.global_layout(forward_batch)

    assert local_layout is not initial
    assert local_layout.owner_sizes == (2, 1)
    assert global_layout.owner_sizes == (3, 0, 2, 2)


def test_welm_nextn_contraction_rebalances_fresh_owner_rows():
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=0),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=[3, 4],
        original_global_num_tokens_cpu=[8, 4],
        welm_mtp_merge_kv_fill_draft=True,
        _welm_kv_mirror_contracted_dp_metadata_rows=3,
    )

    assert runtime.local_layout(forward_batch).owner_sizes == (2, 1)
    assert runtime.global_layout(forward_batch).owner_sizes == (2, 1, 2, 2)
    assert not runtime.local_contraction_active(forward_batch)


def test_welm_mtp_owner_graph_tracks_only_contracted_valid_rows(monkeypatch):
    import sglang.srt.speculative.welmv4_mtp_draft_proposal_cuda_graph_runner as mtp_graph

    runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
        WelmMTPDraftProposalCudaGraphRunner
    )
    runner.use_token_owner = True
    runner.buffers = SimpleNamespace(
        num_token_non_padded=torch.tensor([-1], dtype=torch.int32)
    )
    runner.num_tokens_per_bs = 4
    runner.contracted_rows_by_bs = {8: 8}
    monkeypatch.setattr(mtp_graph, "get_attention_tp_size", lambda: 4)
    monkeypatch.setattr(mtp_graph, "get_attention_tp_rank", lambda: 2, raising=False)

    runner._update_num_token_non_padded(graph_bs=8, raw_bs=5)

    assert runner.buffers.num_token_non_padded.item() == 1

    runner.contracted_rows_by_bs.clear()
    runner._update_num_token_non_padded(graph_bs=8, raw_bs=5)

    # Before contraction, request padding is interleaved and cannot be
    # represented as a trailing valid-row count.
    assert runner.buffers.num_token_non_padded.item() == 8


def test_welm_mtp_non_owner_graph_does_not_publish_num_token_non_padded():
    runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
        WelmMTPDraftProposalCudaGraphRunner
    )
    runner.use_token_owner = False
    runner.buffers = SimpleNamespace(num_token_non_padded=None)
    runner.num_tokens_per_bs = 4
    runner.contracted_rows_by_bs = {}

    runner._update_num_token_non_padded(graph_bs=8, raw_bs=5)

    assert runner.buffers.num_token_non_padded is None


@pytest.mark.parametrize(
    (
        "sample_draft",
        "has_fixed_sampling_params",
        "fixed_top_p",
        "global_has_non_greedy_sampling",
        "global_needs_top_p_sampling",
        "expected_mode",
    ),
    [
        (True, False, None, False, False, (False, False)),
        (True, False, None, True, False, (True, False)),
        (True, False, None, True, True, (True, True)),
        (True, True, None, False, False, (True, False)),
        (True, True, None, False, True, (True, True)),
        (True, True, 0.95, False, False, (True, True)),
        (True, True, 1.0, True, True, (True, False)),
        (False, False, None, True, True, (False, False)),
    ],
)
def test_welm_mtp_proposal_graph_uses_global_sampling_mode(
    sample_draft,
    has_fixed_sampling_params,
    fixed_top_p,
    global_has_non_greedy_sampling,
    global_needs_top_p_sampling,
    expected_mode,
):
    runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
        WelmMTPDraftProposalCudaGraphRunner
    )
    runner.eagle_worker = SimpleNamespace(
        _is_welmv4_mtp_draft_sampling_enabled=lambda: sample_draft,
        welmv4_mtp_draft_fixed_top_p=fixed_top_p,
        _has_welmv4_mtp_fixed_draft_sampling_params=(lambda: has_fixed_sampling_params),
    )
    runner.topk = 1
    runner.use_dp_sampling_consensus = True
    runner.supports_draft_top_p = True
    forward_batch = SimpleNamespace(
        global_has_non_greedy_sampling=global_has_non_greedy_sampling,
        global_needs_top_p_sampling=global_needs_top_p_sampling,
    )

    assert runner._required_draft_sampling_mode(forward_batch) == expected_mode


def test_welm_mtp_proposal_graph_keeps_local_mode_without_dp_consensus():
    runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
        WelmMTPDraftProposalCudaGraphRunner
    )
    runner.use_dp_sampling_consensus = False
    runner.eagle_worker = SimpleNamespace(
        _should_sample_welmv4_mtp_draft=lambda _batch: True,
        _should_use_welmv4_mtp_draft_top_p=lambda _batch: True,
    )

    assert runner._required_draft_sampling_mode(SimpleNamespace()) == (True, True)


def test_welm_mtp_proposal_graph_rejects_skip_all_gather(monkeypatch):
    import sglang.srt.speculative.welmv4_mtp_draft_proposal_cuda_graph_runner as mtp_graph

    monkeypatch.setenv("SGLANG_SCHEDULER_SKIP_ALL_GATHER", "1")

    with pytest.raises(ValueError, match="scheduler all-gather"):
        mtp_graph._validate_dp_proposal_graph_consensus_config(
            enable_dp_attention=True,
            dp_size=4,
        )

    mtp_graph._validate_dp_proposal_graph_consensus_config(
        enable_dp_attention=False,
        dp_size=1,
    )


def test_welm_mtp_owner_graph_fails_fast_on_sampling_mode_mismatch():
    runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
        WelmMTPDraftProposalCudaGraphRunner
    )
    runner.use_token_owner = True
    runner.use_dp_sampling_consensus = True
    runner.supports_draft_top_p = True
    runner.topk = 1
    runner.sample_draft = True
    runner.use_top_p = False
    runner.eagle_worker = SimpleNamespace(
        _is_welmv4_mtp_draft_sampling_enabled=lambda: True,
        welmv4_mtp_draft_fixed_top_p=None,
        _has_welmv4_mtp_fixed_draft_sampling_params=lambda: True,
    )
    forward_batch = SimpleNamespace(
        forward_mode=welmv4.ForwardMode.DRAFT_EXTEND,
        global_has_non_greedy_sampling=True,
        global_needs_top_p_sampling=True,
    )

    with pytest.raises(RuntimeError, match="sampling mode mismatch"):
        runner.can_run(forward_batch)


@pytest.mark.parametrize(
    ("graphs", "max_bs", "can_run_dp_cuda_graph", "reason"),
    [
        ({1: object()}, 1, True, "capture bucket"),
        ({2: object()}, 2, False, "scheduler disabled"),
    ],
)
def test_welm_mtp_owner_graph_rejects_unexpected_fast_path_miss(
    graphs,
    max_bs,
    can_run_dp_cuda_graph,
    reason,
):
    runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
        WelmMTPDraftProposalCudaGraphRunner
    )
    runner.use_token_owner = True
    runner.require_mlp_tp_gather = False
    runner.require_mlp_sync = True
    runner.disable_padding = False
    runner.graphs = graphs
    runner.max_bs = max_bs
    runner.num_tokens_per_bs = 4
    runner.sample_draft = False
    runner.use_top_p = False
    runner.use_dp_sampling_consensus = False
    runner.eagle_worker = SimpleNamespace(
        _should_sample_welmv4_mtp_draft=lambda _batch: False,
    )
    forward_batch = SimpleNamespace(
        forward_mode=welmv4.ForwardMode.DRAFT_EXTEND,
        input_ids=torch.arange(8),
        out_cache_loc=torch.arange(8),
        batch_size=2,
        global_num_reqs_cpu=[2, 1],
        can_run_dp_cuda_graph=can_run_dp_cuda_graph,
    )

    with pytest.raises(RuntimeError, match=reason):
        runner.can_run(forward_batch)


def test_welm_token_owner_aligns_mirror_residual_to_original_lane():
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=1),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=1),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=[3, 4],
        original_global_num_tokens_cpu=[8, 4],
        welm_kv_mirror_contract_flags=[True, False],
        extend_seq_lens_cpu=[4, 4],
        welm_kv_mirror_last_q_indices_cpu=[3, 7],
        welm_kv_mirror_active_batch_indices_cpu=[0, 2],
        _welm_kv_mirror_contracted_dp_metadata_rows=3,
    )
    old_owner_residual = torch.tensor([[4.0], [5.0], [6.0], [7.0]])

    contracted = runtime.align_kv_mirror_residual(
        old_owner_residual,
        forward_batch,
    )

    assert contracted.tolist() == [[7.0]]


def test_welm_token_owner_does_not_contract_non_contracting_dp_rank():
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=1),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=3),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=[3, 4],
        original_global_num_tokens_cpu=[8, 4],
        welm_kv_mirror_contract_flags=[True, False],
        _welm_kv_mirror_contracted_dp_metadata_rows=4,
    )

    assert not runtime.local_contraction_active(forward_batch)
    assert runtime.local_layout(forward_batch).owner_sizes == (2, 2)


def test_welm_token_owner_runtime_recomputes_valid_mask_after_invalidation():
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=0),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=[3, 4],
        original_global_num_tokens_cpu=[8, 4],
        welm_kv_mirror_contract_flags=[True, False],
        welm_kv_mirror_last_q_indices_cpu=[1, 7],
        welm_kv_mirror_active_batch_indices_cpu=[0, 2],
        _welm_kv_mirror_contracted_dp_metadata_rows=3,
    )

    first = runtime.valid_local_mask(forward_batch, device=torch.device("cpu"))
    assert first.tolist() == [True, False]
    forward_batch.welm_kv_mirror_active_batch_indices_cpu = [1, 2]
    runtime.invalidate()
    second = runtime.valid_local_mask(forward_batch, device=torch.device("cpu"))
    assert second.tolist() == [False, True]
    assert second is not first


def test_welm_deepep_owner_context_masks_non_survivor_rows():
    layout = token_owner.TokenOwnerLayout.from_owner_sizes(
        (2, 1),
        local_owner_rank=0,
    )
    context = token_owner.DeepEPRouterContext(
        layout=layout,
        valid_local_mask=torch.tensor([False, True]),
    )
    hidden = torch.tensor([[0.0], [2.0]])
    topk_output = StandardTopKOutput(
        topk_weights=torch.tensor([[0.5, 0.5], [0.75, 0.25]]),
        topk_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
        router_logits=torch.empty((2, 0)),
    )

    output_hidden, output_topk = context.prepare_moe_inputs(hidden, topk_output)

    assert output_hidden is hidden
    assert output_topk.topk_ids.tolist() == [[-1, -1], [3, 4]]
    assert output_topk.topk_weights.tolist() == [[0.0, 0.0], [0.75, 0.25]]


def test_welm_token_owner_router_gathers_hidden_and_topk_metadata():
    layout = token_owner.TokenOwnerLayout.from_owner_sizes(
        (2, 1, 1, 0),
        local_owner_rank=1,
    )
    local_hidden = torch.tensor([[1.0, 2.0]])
    local_weights = torch.tensor([[0.75, 0.25]])
    local_ids = torch.tensor([[2, 5]], dtype=torch.int64)
    topk_output = StandardTopKOutput(
        topk_weights=local_weights,
        topk_ids=local_ids,
        router_logits=torch.empty((1, 0)),
    )

    class FakeGroup:
        world_size = 4
        rank_in_group = 1

        def all_gatherv(self, inputs, sizes):
            assert all(
                actual is expected
                for actual, expected in zip(
                    inputs,
                    (local_hidden, local_weights, local_ids),
                    strict=True,
                )
            )
            assert sizes == [2, 1, 1, 0]
            return [
                torch.arange(8, dtype=torch.float32).view(4, 2),
                torch.full((4, 2), 0.5),
                torch.arange(8, dtype=torch.int64).view(4, 2),
            ]

    context = token_owner.GlobalTPRouterContext(
        layout=layout,
        global_tp_group=FakeGroup(),
    )

    full_hidden, full_topk = context.prepare_moe_inputs(
        local_hidden,
        topk_output,
    )

    assert full_hidden.shape == (4, 2)
    assert full_topk.topk_weights.shape == (4, 2)
    assert full_topk.topk_ids.shape == (4, 2)
    assert full_topk.router_logits.shape == (4, 0)


def test_welm_token_owner_router_uses_local_proxy_before_global_gather():
    local_layout = token_owner.TokenOwnerLayout.from_owner_sizes(
        (2, 1),
        local_owner_rank=1,
    )
    global_layout = token_owner.TokenOwnerLayout.from_owner_sizes(
        (3, 0, 2, 2),
        local_owner_rank=1,
    )
    local_hidden = torch.tensor([[3.0]])
    local_weights = torch.tensor([[0.75, 0.25]])
    local_ids = torch.tensor([[2, 5]], dtype=torch.int64)
    topk_output = StandardTopKOutput(
        topk_weights=local_weights,
        topk_ids=local_ids,
        router_logits=torch.empty((1, 0)),
    )

    class LocalGroup:
        world_size = 2
        rank_in_group = 1

        def all_gatherv(self, inputs, sizes):
            assert sizes == [2, 1]
            assert inputs == [local_hidden, local_weights, local_ids]
            return [
                torch.arange(3, dtype=torch.float32).view(3, 1),
                torch.full((3, 2), 0.5),
                torch.arange(6, dtype=torch.int64).view(3, 2),
            ]

    class GlobalGroup:
        world_size = 4
        rank_in_group = 1

        def all_gatherv(self, inputs, sizes):
            assert sizes == [3, 0, 2, 2]
            assert all(tensor.shape[0] == 0 for tensor in inputs)
            return [
                torch.arange(7, dtype=torch.float32).view(7, 1),
                torch.full((7, 2), 0.5),
                torch.arange(14, dtype=torch.int64).view(7, 2),
            ]

    context = token_owner.GlobalTPRouterContext(
        layout=global_layout,
        local_layout=local_layout,
        global_tp_group=GlobalGroup(),
        attn_tp_group=LocalGroup(),
    )

    full_hidden, full_topk = context.prepare_moe_inputs(local_hidden, topk_output)

    assert full_hidden.shape == (7, 1)
    assert full_topk.topk_ids.shape == (7, 2)


def test_global_tp_owner_routes_cuda_graph_padding(monkeypatch):
    observed_non_padded = []

    class FakeTopK:
        topk_config = SimpleNamespace(top_k=1)

        def __call__(
            self,
            hidden_states,
            router_logits,
            *,
            num_token_non_padded,
        ):
            observed_non_padded.append(num_token_non_padded)
            rows = hidden_states.shape[0]
            return StandardTopKOutput(
                topk_weights=torch.ones((rows, 1)),
                topk_ids=torch.zeros((rows, 1), dtype=torch.int64),
                router_logits=router_logits,
            )

    class FakeGroup:
        world_size = 1
        rank_in_group = 0

        @staticmethod
        def all_gatherv(inputs, sizes):
            assert sizes == [2]
            return inputs

    block = torch.nn.Module()
    block.layer_id = 0
    block.tp_size = 1
    block._shared_expert_replicated = False
    block.use_mxfp8 = False
    block.use_previous_precision_router = False
    block.use_mmq_router_linear_v2 = False
    block._mk_moe_router = None
    block.shared_expert = None
    block.shared_expert_gate = None
    block.gate = SimpleNamespace(weight=torch.zeros((4, 1)))
    block.topk = FakeTopK()
    block.experts = lambda hidden_states, _topk_output: hidden_states
    monkeypatch.setattr(
        welmv4,
        "mmq_style_router_linear",
        lambda hidden_states, _weight, **_kwargs: torch.zeros(
            (hidden_states.shape[0], 4)
        ),
    )

    hidden_states = torch.tensor([[1.0], [0.0]])
    forward_batch = SimpleNamespace(num_token_non_padded=torch.tensor(1))
    context = token_owner.GlobalTPRouterContext(
        layout=token_owner.TokenOwnerLayout.from_owner_sizes(
            (2,),
            local_owner_rank=0,
        ),
        global_tp_group=FakeGroup(),
    )

    output = welmv4.Qwen2MoeSparseMoeBlock.forward(
        block,
        hidden_states,
        None,
        forward_batch=forward_batch,
        router_context=context,
    )

    assert output.shape == hidden_states.shape
    assert observed_non_padded == [None]
