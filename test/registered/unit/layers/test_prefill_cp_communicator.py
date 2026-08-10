"""CPU contract tests for persistent-token prefill CP communication."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models import welmv4 as welmv4_model
from sglang.srt.context_parallel import (
    build_cp_prefill_split_spec,
    materialize_cp_prefill_runtime_layout,
)
from sglang.srt.layers.communicator_prefill_cp import (
    PrefillCPLayerCommunicator,
    global_tp_destination_sizes,
    pair_lane_sizes,
)
from sglang.srt.models.welmv4 import (
    _welm_localize_cp_prefill_rows,
    _welm_select_layer_communicator,
)
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


@dataclass
class _Layout:
    cp_rank: int
    active_local_tokens: int
    counts: tuple[int, ...]
    owner_rotation: int = 0

    def active_tokens_per_cp_rank(self):
        return self.counts

    @property
    def spec(self):
        return SimpleNamespace(owner_rotation=self.owner_rotation)


class _RecordingGroup:
    def __init__(self, world_size, rank, *, gather_output=None):
        self.world_size = world_size
        self.rank_in_group = rank
        self.gather_output = gather_output
        self.calls = []

    def all_gatherv(self, tensor, sizes):
        self.calls.append(("all_gatherv", tensor, tuple(sizes)))
        output = self.gather_output
        if output is None:
            output = tensor
        if isinstance(tensor, list):
            return output
        return [output]

    def reduce_scatterv(self, tensor, sizes):
        sizes = tuple(sizes)
        self.calls.append(("reduce_scatterv", tensor, sizes))
        start = sum(sizes[: self.rank_in_group])
        return tensor.narrow(0, start, sizes[self.rank_in_group]).clone()


class _SummingReduceScatterGroup(_RecordingGroup):
    def __init__(self, world_size, rank, all_rank_partials):
        super().__init__(world_size, rank)
        self.all_rank_partials = all_rank_partials

    def reduce_scatterv(self, tensor, sizes):
        sizes = tuple(sizes)
        self.calls.append(("reduce_scatterv", tensor, sizes))
        assert tensor is self.all_rank_partials[self.rank_in_group]
        reduced = torch.stack(self.all_rank_partials).sum(dim=0)
        start = sum(sizes[: self.rank_in_group])
        return reduced.narrow(0, start, sizes[self.rank_in_group]).clone()


class _PostAttentionNorm:
    def __init__(self, normalized):
        self.normalized = normalized
        self.calls = []

    def __call__(self, hidden_states, residual, clone_fp32_out=False):
        self.calls.append((hidden_states, residual, clone_fp32_out))
        if clone_fp32_out:
            return self.normalized, residual, self.normalized.float()
        return self.normalized, residual


class _PrecisionSeparatingInputNorm:
    def __init__(self, normalized, fp32_normalized):
        self.normalized = normalized
        self.fp32_normalized = fp32_normalized
        self.calls = []

    def __call__(
        self,
        hidden_states,
        residual=None,
        *,
        residual_after_layernorm=False,
        clone_fp32_out=False,
    ):
        self.calls.append(
            (
                hidden_states,
                residual,
                residual_after_layernorm,
                clone_fp32_out,
            )
        )
        if clone_fp32_out:
            return self.normalized, self.normalized, self.fp32_normalized
        if residual is None:
            return self.normalized
        return self.normalized, residual


def _communicator(
    layout,
    *,
    lane=0,
    cp_output=None,
    pair_output=None,
    use_ep_dispatch=False,
    fused_attntp2_norm_runner=None,
):
    cp_group = _RecordingGroup(4, layout.cp_rank, gather_output=cp_output)
    pair_group = _RecordingGroup(2, lane, gather_output=pair_output)
    tp_group = _RecordingGroup(8, layout.cp_rank * 2 + lane)
    communicator = PrefillCPLayerCommunicator(
        input_layernorm=None,
        post_attention_layernorm=None,
        cp_group=cp_group,
        attn_tp_group=pair_group,
        global_tp_group=tp_group,
        use_ep_dispatch=use_ep_dispatch,
        fused_attntp2_norm_runner=fused_attntp2_norm_runner,
    )
    return communicator, cp_group, pair_group, tp_group


def test_pair_and_global_destination_sizes_match_tensor_split():
    assert pair_lane_sizes(5) == (3, 2)
    assert pair_lane_sizes(0) == (0, 0)
    assert global_tp_destination_sizes((5, 0, 3, 2)) == (
        3,
        2,
        0,
        0,
        2,
        1,
        1,
        1,
    )


def test_router_context_reuses_global_tp_owner_mapping():
    counts = (5, 0, 3, 2)
    layout = _Layout(
        cp_rank=2,
        active_local_tokens=3,
        counts=counts,
    )
    communicator, _, _, _ = _communicator(layout, lane=1)
    forward_batch = SimpleNamespace(attn_cp_prefill_runtime_layout=layout)

    context = communicator.build_router_context(forward_batch)
    global_hidden = torch.arange(20, dtype=torch.float32).view(10, 2)

    assert context.destination_sizes == (3, 2, 0, 0, 2, 1, 1, 1)
    assert context.owner_start == 7
    assert context.owner_count == 1
    assert context.global_token_count == 10
    assert torch.equal(context.local_rows(global_hidden), global_hidden[7:8])


def test_router_context_gathers_only_metadata_in_rank_packed_order():
    counts = (5, 0, 3, 2)
    layout = _Layout(
        cp_rank=2,
        active_local_tokens=3,
        counts=counts,
    )
    full_weights = torch.arange(20, dtype=torch.float32).view(10, 2)
    full_ids = torch.arange(20, dtype=torch.int64).view(10, 2)
    communicator, _, _, tp_group = _communicator(layout, lane=1)
    tp_group.gather_output = [full_weights, full_ids]
    context = communicator.build_router_context(
        SimpleNamespace(attn_cp_prefill_runtime_layout=layout)
    )
    local_weights = full_weights[7:8]
    local_ids = full_ids[7:8]

    weights, ids = context.gather_routing_metadata(local_weights, local_ids)

    assert weights is full_weights
    assert ids is full_ids
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int64
    assert len(tp_group.calls) == 1
    assert tp_group.calls[0][0] == "all_gatherv"
    assert tp_group.calls[0][1][0] is local_weights
    assert tp_group.calls[0][1][1] is local_ids
    assert tp_group.calls[0][2] == (3, 2, 0, 0, 2, 1, 1, 1)


def test_router_context_empty_owner_still_joins_metadata_collective():
    counts = (3, 0, 0, 0)
    layout = _Layout(
        cp_rank=1,
        active_local_tokens=0,
        counts=counts,
    )
    full_weights = torch.arange(6, dtype=torch.float32).view(3, 2)
    full_ids = torch.arange(6, dtype=torch.int64).view(3, 2)
    communicator, _, _, tp_group = _communicator(layout, lane=0)
    tp_group.gather_output = [full_weights, full_ids]
    context = communicator.build_router_context(
        SimpleNamespace(attn_cp_prefill_runtime_layout=layout)
    )

    weights, ids = context.gather_routing_metadata(
        torch.empty((0, 2), dtype=torch.float32),
        torch.empty((0, 2), dtype=torch.int64),
    )

    assert torch.equal(weights, full_weights)
    assert torch.equal(ids, full_ids)
    assert context.owner_count == 0
    assert len(tp_group.calls) == 1


def test_router_context_all_zero_runtime_skips_metadata_collective():
    counts = (0, 0, 0, 0)
    layout = _Layout(cp_rank=1, active_local_tokens=0, counts=counts)
    communicator, _, _, tp_group = _communicator(layout, lane=1)
    context = communicator.build_router_context(
        SimpleNamespace(attn_cp_prefill_runtime_layout=layout)
    )
    local_weights = torch.empty((0, 2), dtype=torch.float32)
    local_ids = torch.empty((0, 2), dtype=torch.int64)

    weights, ids = context.gather_routing_metadata(local_weights, local_ids)

    assert weights is local_weights
    assert ids is local_ids
    assert tp_group.calls == []


def test_router_context_rejects_router_replay():
    layout = _Layout(cp_rank=0, active_local_tokens=3, counts=(3, 0, 0, 0))
    communicator, _, _, _ = _communicator(layout)

    with pytest.raises(NotImplementedError, match="Router Replay"):
        communicator.build_router_context(
            SimpleNamespace(
                attn_cp_prefill_runtime_layout=layout,
                router_replay_topk_ids=torch.zeros((3, 1, 1), dtype=torch.int32),
            )
        )


def test_non_ep_mlp_validation_requires_local_router_contract():
    layout = _Layout(cp_rank=0, active_local_tokens=1, counts=(1, 0, 0, 0))
    communicator, _, _, _ = _communicator(layout, use_ep_dispatch=False)

    with pytest.raises(RuntimeError, match="router validator"):
        communicator.validate_mlp(SimpleNamespace())


def test_non_ep_mlp_validation_accepts_local_router_contract():
    layout = _Layout(cp_rank=0, active_local_tokens=1, counts=(1, 0, 0, 0))
    communicator, _, _, _ = _communicator(layout, use_ep_dispatch=False)
    validated = []
    mlp = SimpleNamespace(validate_local_router=lambda: validated.append(True))

    communicator.validate_mlp(mlp)

    assert validated == [True]


def test_same_lane_cp_gather_reconstructs_rank_packed_global_rows():
    counts = (5, 0, 3, 2)
    layout = _Layout(cp_rank=2, active_local_tokens=3, counts=counts)
    global_rows = torch.arange(10, dtype=torch.float32).view(10, 1)
    communicator, cp_group, _, _ = _communicator(
        layout, cp_output=global_rows
    )
    local_rows = global_rows[5:8]

    gathered = communicator.gather_global_tp_input(local_rows, layout)

    assert torch.equal(gathered, global_rows)
    assert len(cp_group.calls) == 1
    assert cp_group.calls[0][0] == "all_gatherv"
    assert cp_group.calls[0][1] is local_rows
    assert cp_group.calls[0][2] == counts


def test_global_reduce_scatter_routes_cp_lane_destination():
    counts = (5, 0, 3, 2)
    layout = _Layout(cp_rank=2, active_local_tokens=3, counts=counts)
    communicator, _, _, tp_group = _communicator(layout)
    global_sum = torch.arange(10, dtype=torch.float32).view(10, 1)

    local_part = communicator.reduce_scatter_global_tp_output(global_sum, layout)

    assert torch.equal(local_part, global_sum[5:7])
    assert len(tp_group.calls) == 1
    assert tp_group.calls[0][0] == "reduce_scatterv"
    assert tp_group.calls[0][1] is global_sum
    assert tp_group.calls[0][2] == (3, 2, 0, 0, 2, 1, 1, 1)


def test_pair_gather_restores_cp_local_rows():
    counts = (5, 0, 3, 2)
    layout = _Layout(cp_rank=2, active_local_tokens=3, counts=counts)
    full_local = torch.arange(3, dtype=torch.float32).view(3, 1)
    communicator, _, pair_group, _ = _communicator(
        layout, pair_output=full_local
    )
    lane_part = full_local[:2]

    restored = communicator.restore_attn_pair(lane_part, layout)

    assert torch.equal(restored, full_local)
    assert len(pair_group.calls) == 1
    assert pair_group.calls[0][0] == "all_gatherv"
    assert pair_group.calls[0][1] is lane_part
    assert pair_group.calls[0][2] == (2, 1)


def test_reduce_scatter_then_pair_gather_matches_global_tp_all_reduce():
    counts = (5, 0, 3, 2)
    total_tokens = sum(counts)
    all_rank_partials = [
        torch.arange(total_tokens * 3, dtype=torch.float32).view(total_tokens, 3)
        * (rank + 1)
        + rank
        for rank in range(8)
    ]
    global_sum = torch.stack(all_rank_partials).sum(dim=0)

    cp_start = 0
    for cp_rank, local_count in enumerate(counts):
        expected_local = global_sum.narrow(0, cp_start, local_count)
        cp_start += local_count
        for lane in range(2):
            layout = _Layout(
                cp_rank=cp_rank,
                active_local_tokens=local_count,
                counts=counts,
            )
            global_tp_rank = 2 * cp_rank + lane
            communicator = PrefillCPLayerCommunicator(
                input_layernorm=None,
                post_attention_layernorm=None,
                cp_group=_RecordingGroup(4, cp_rank),
                attn_tp_group=_RecordingGroup(
                    2, lane, gather_output=expected_local
                ),
                global_tp_group=_SummingReduceScatterGroup(
                    8, global_tp_rank, all_rank_partials
                ),
                use_ep_dispatch=False,
            )

            local_part = communicator.reduce_scatter_global_tp_output(
                all_rank_partials[global_tp_rank], layout
            )
            restored = communicator.restore_attn_pair(local_part, layout)

            assert torch.equal(restored, expected_local)


@pytest.mark.parametrize(
    ("lane", "expected"),
    [(0, [0.0, 1.0, 2.0]), (1, [3.0, 4.0])],
)
def test_ep_scatter_uses_one_copy_pair_partition(lane, expected):
    layout = _Layout(cp_rank=0, active_local_tokens=5, counts=(5, 0, 0, 0))
    cp_group = _RecordingGroup(4, 0)
    pair_group = _RecordingGroup(2, lane)
    tp_group = _RecordingGroup(8, lane)
    communicator = PrefillCPLayerCommunicator(
        input_layernorm=None,
        post_attention_layernorm=None,
        cp_group=cp_group,
        attn_tp_group=pair_group,
        global_tp_group=tp_group,
    )
    hidden = torch.arange(5, dtype=torch.float32).view(5, 1)

    scattered = communicator.scatter_ep_input(hidden, layout)

    assert scattered.flatten().tolist() == expected
    assert cp_group.calls == []
    assert tp_group.calls == []


def test_zero_token_rank_skips_empty_pair_restore_collective():
    counts = (3, 0, 0, 0)
    layout = _Layout(cp_rank=1, active_local_tokens=0, counts=counts)
    global_rows = torch.arange(3, dtype=torch.float32).view(3, 1)
    communicator, cp_group, pair_group, tp_group = _communicator(
        layout,
        cp_output=global_rows,
        pair_output=torch.empty((0, 1)),
    )

    gathered = communicator.gather_global_tp_input(torch.empty((0, 1)), layout)
    local_part = communicator.reduce_scatter_global_tp_output(gathered, layout)
    restored = communicator.restore_attn_pair(local_part, layout)

    assert restored.shape == (0, 1)
    assert [call[0] for call in cp_group.calls] == ["all_gatherv"]
    assert [call[0] for call in tp_group.calls] == ["reduce_scatterv"]
    assert pair_group.calls == []


def test_all_zero_runtime_skips_every_layer_transition_collective():
    counts = (0, 0, 0, 0)
    layout = _Layout(cp_rank=2, active_local_tokens=0, counts=counts)
    communicator, cp_group, pair_group, tp_group = _communicator(layout)
    empty = torch.empty((0, 1))
    forward_batch = type("Batch", (), {"attn_cp_prefill_runtime_layout": layout})()

    gathered = communicator.gather_global_tp_input(empty, layout)
    reduced = communicator.reduce_scatter_global_tp_output(empty, layout)
    hidden, residual = communicator.postprocess_layer(empty, empty, forward_batch)

    assert gathered is empty
    assert reduced is empty
    assert hidden is empty
    assert residual is empty
    assert not communicator.has_active_mlp_tokens(forward_batch)
    assert cp_group.calls == []
    assert pair_group.calls == []
    assert tp_group.calls == []


def test_wrong_group_topology_fails_before_collective():
    layout = _Layout(cp_rank=2, active_local_tokens=3, counts=(5, 0, 3, 2))
    communicator = PrefillCPLayerCommunicator(
        input_layernorm=None,
        post_attention_layernorm=None,
        cp_group=_RecordingGroup(2, 0),
        attn_tp_group=_RecordingGroup(4, 2),
        global_tp_group=_RecordingGroup(8, 4),
    )

    with pytest.raises(RuntimeError, match="topology"):
        communicator.gather_global_tp_input(torch.empty((3, 1)), layout)


@pytest.mark.parametrize("has_residual", [False, True])
def test_prepare_attn_preserves_true_fp32_post_norm_residual(has_residual):
    layout = _Layout(cp_rank=0, active_local_tokens=2, counts=(2, 0, 0, 0))
    hidden = torch.tensor([[1.0], [-2.0]], dtype=torch.bfloat16)
    incoming_residual = (
        torch.tensor([[0.5], [0.25]], dtype=torch.float32)
        if has_residual
        else None
    )
    fp32_normalized = torch.tensor(
        [[1.0039061], [-2.0078123]], dtype=torch.float32
    )
    normalized = fp32_normalized.to(torch.bfloat16)
    assert not torch.equal(normalized.float(), fp32_normalized)
    norm = _PrecisionSeparatingInputNorm(normalized, fp32_normalized)
    communicator, _, _, _ = _communicator(layout)
    communicator.input_layernorm = norm
    forward_batch = type("Batch", (), {"attn_cp_prefill_runtime_layout": layout})()

    output, residual = communicator.prepare_attn(
        hidden,
        incoming_residual,
        forward_batch,
        residual_after_layernorm=True,
    )

    assert output is normalized
    assert residual is fp32_normalized
    assert norm.calls == [(hidden, incoming_residual, True, True)]


def test_prepare_attn_empty_ppln_rank_keeps_fp32_residual_contract():
    layout = _Layout(cp_rank=1, active_local_tokens=0, counts=(2, 0, 0, 0))
    communicator, _, _, _ = _communicator(layout)
    hidden = torch.empty((0, 4), dtype=torch.bfloat16)
    forward_batch = type("Batch", (), {"attn_cp_prefill_runtime_layout": layout})()

    output, residual = communicator.prepare_attn(
        hidden,
        None,
        forward_batch,
        residual_after_layernorm=True,
    )

    assert output is hidden
    assert residual.shape == hidden.shape
    assert residual.dtype == torch.float32


def test_prepare_mlp_gathers_only_normalized_hidden_and_keeps_residual_local():
    counts = (5, 0, 3, 2)
    layout = _Layout(cp_rank=2, active_local_tokens=3, counts=counts)
    normalized_local = torch.arange(3, dtype=torch.float32).view(3, 1) + 10
    global_normalized = torch.arange(10, dtype=torch.float32).view(10, 1) + 10
    residual = torch.arange(3, dtype=torch.float32).view(3, 1) + 100
    norm = _PostAttentionNorm(normalized_local)
    communicator, cp_group, _, _ = _communicator(
        layout, cp_output=global_normalized
    )
    communicator.post_attention_layernorm = norm
    forward_batch = type("Batch", (), {"attn_cp_prefill_runtime_layout": layout})()

    hidden, returned_residual = communicator.prepare_mlp(
        torch.zeros_like(normalized_local), residual, forward_batch
    )

    assert torch.equal(hidden, global_normalized)
    assert returned_residual is residual
    assert norm.calls[0][1] is residual
    assert norm.calls[0][2] is False
    assert cp_group.calls[0][1] is normalized_local


@pytest.mark.parametrize(
    ("lane", "expected"),
    [(0, [10.0, 11.0, 12.0]), (1, [13.0, 14.0])],
)
def test_ep_prepare_mlp_keeps_one_token_copy_and_skips_global_gather(
    lane, expected
):
    layout = _Layout(cp_rank=0, active_local_tokens=5, counts=(5, 0, 0, 0))
    normalized_local = torch.arange(5, dtype=torch.float32).view(5, 1) + 10
    residual = torch.arange(5, dtype=torch.float32).view(5, 1) + 100
    norm = _PostAttentionNorm(normalized_local)
    communicator, cp_group, pair_group, tp_group = _communicator(
        layout, lane=lane, use_ep_dispatch=True
    )
    communicator.post_attention_layernorm = norm
    forward_batch = type("Batch", (), {"attn_cp_prefill_runtime_layout": layout})()

    hidden, returned_residual = communicator.prepare_mlp(
        torch.zeros_like(normalized_local), residual, forward_batch
    )

    assert hidden.flatten().tolist() == expected
    assert returned_residual is residual
    assert cp_group.calls == []
    assert pair_group.calls == []
    assert tp_group.calls == []


class _RecordingFusedNormRunner:
    def __init__(self, normalized, residual_out, fp32_output):
        self.normalized = normalized
        self.residual_out = residual_out
        self.fp32_output = fp32_output
        self.calls = []

    def forward_prefill_cp(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.normalized, self.residual_out, self.fp32_output


def test_fused_attntp2_prepare_mlp_gathers_normalized_and_keeps_fp32_residual():
    counts = (5, 0, 3, 2)
    layout = _Layout(
        cp_rank=2,
        active_local_tokens=3,
        counts=counts,
        owner_rotation=3,
    )
    partial = torch.arange(3, dtype=torch.bfloat16).view(3, 1)
    residual = torch.arange(3, dtype=torch.float32).view(3, 1) + 100
    normalized_local = partial + 10
    normalized_global = torch.arange(10, dtype=torch.bfloat16).view(10, 1) + 10
    residual_out = residual + 20
    fp32_output = normalized_local.float()
    runner = _RecordingFusedNormRunner(
        normalized_local, residual_out, fp32_output
    )
    communicator, cp_group, _, _ = _communicator(
        layout,
        cp_output=normalized_global,
        fused_attntp2_norm_runner=runner,
    )
    communicator.post_attention_layernorm = SimpleNamespace(
        weight=torch.ones(1, dtype=torch.bfloat16), eps=2e-5
    )
    o_norm = SimpleNamespace(
        weight=torch.full((1,), 2, dtype=torch.bfloat16), eps=1e-5
    )
    forward_batch = SimpleNamespace(attn_cp_prefill_runtime_layout=layout)

    hidden, returned_residual = communicator.prepare_mlp_fused_attntp2(
        partial, residual, forward_batch, o_norm=o_norm
    )

    assert torch.equal(hidden, normalized_global)
    assert returned_residual is residual_out
    assert runner.calls == [
        (
            (
                partial,
                residual,
                o_norm.weight,
                communicator.post_attention_layernorm.weight,
                o_norm.eps,
                communicator.post_attention_layernorm.eps,
            ),
            {"lane_rotation": 1},
        )
    ]
    assert cp_group.calls[0][1] is normalized_local


def test_fused_attntp2_empty_owner_skips_pair_collective_and_joins_cp_gather():
    counts = (3, 0, 0, 0)
    layout = _Layout(cp_rank=1, active_local_tokens=0, counts=counts)
    empty_hidden = torch.empty((0, 1), dtype=torch.bfloat16)
    empty_residual = torch.empty((0, 1), dtype=torch.float32)
    normalized_global = torch.arange(3, dtype=torch.bfloat16).view(3, 1)
    runner = _RecordingFusedNormRunner(None, None, None)
    communicator, cp_group, pair_group, _ = _communicator(
        layout,
        cp_output=normalized_global,
        fused_attntp2_norm_runner=runner,
    )
    communicator.post_attention_layernorm = SimpleNamespace(
        weight=torch.ones(1, dtype=torch.bfloat16), eps=2e-5
    )
    forward_batch = SimpleNamespace(attn_cp_prefill_runtime_layout=layout)

    hidden, returned_residual = communicator.prepare_mlp_fused_attntp2(
        empty_hidden,
        empty_residual,
        forward_batch,
        o_norm=SimpleNamespace(
            weight=torch.ones(1, dtype=torch.bfloat16),
            eps=1e-5,
        ),
    )

    assert torch.equal(hidden, normalized_global)
    assert returned_residual is empty_residual
    assert runner.calls == []
    assert [call[0] for call in cp_group.calls] == ["all_gatherv"]
    assert pair_group.calls == []


def test_fused_attntp2_prepare_mlp_fails_without_registered_runner():
    layout = _Layout(cp_rank=0, active_local_tokens=1, counts=(1, 0, 0, 0))
    communicator, _, _, _ = _communicator(layout)
    forward_batch = SimpleNamespace(attn_cp_prefill_runtime_layout=layout)

    with pytest.raises(RuntimeError, match="fused IPC norm runner is unavailable"):
        communicator.prepare_mlp_fused_attntp2(
            torch.ones((1, 1), dtype=torch.bfloat16),
            torch.ones((1, 1), dtype=torch.float32),
            forward_batch,
            o_norm=SimpleNamespace(
                weight=torch.ones(1, dtype=torch.bfloat16), eps=1e-5
            ),
        )


@pytest.mark.parametrize(
    ("lane", "lane_rows"),
    [(0, [20.0, 21.0, 22.0]), (1, [23.0, 24.0])],
)
def test_ep_postprocess_restores_pair_without_global_reduce_scatter(
    lane, lane_rows
):
    layout = _Layout(cp_rank=0, active_local_tokens=5, counts=(5, 0, 0, 0))
    restored = torch.arange(5, dtype=torch.float32).view(5, 1) + 20
    communicator, cp_group, pair_group, tp_group = _communicator(
        layout,
        lane=lane,
        pair_output=restored,
        use_ep_dispatch=True,
    )
    forward_batch = type("Batch", (), {"attn_cp_prefill_runtime_layout": layout})()

    hidden, residual = communicator.postprocess_layer(
        torch.tensor(lane_rows).view(-1, 1), None, forward_batch
    )

    assert torch.equal(hidden, restored)
    assert residual is None
    assert cp_group.calls == []
    assert tp_group.calls == []
    assert [call[0] for call in pair_group.calls] == ["all_gatherv"]


def test_ep_mlp_validation_accepts_full_ep_and_replicated_shared_expert():
    layout = _Layout(cp_rank=0, active_local_tokens=5, counts=(5, 0, 0, 0))
    communicator, _, _, _ = _communicator(layout, use_ep_dispatch=True)
    mlp = SimpleNamespace(
        experts=SimpleNamespace(moe_tp_size=1, moe_ep_size=8),
        shared_expert=SimpleNamespace(
            gate_up_proj=SimpleNamespace(tp_size=1),
            down_proj=SimpleNamespace(tp_size=1),
        ),
    )

    communicator.validate_mlp(mlp)


@pytest.mark.parametrize(
    ("moe_tp_size", "moe_ep_size", "match"),
    [(2, 4, "MoE-TP1"), (1, 4, "EP8")],
)
def test_ep_mlp_validation_rejects_mixed_expert_topology(
    moe_tp_size, moe_ep_size, match
):
    layout = _Layout(cp_rank=0, active_local_tokens=5, counts=(5, 0, 0, 0))
    communicator, _, _, _ = _communicator(layout, use_ep_dispatch=True)
    mlp = SimpleNamespace(
        experts=SimpleNamespace(
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        ),
        shared_expert=None,
    )

    with pytest.raises(NotImplementedError, match=match):
        communicator.validate_mlp(mlp)


def test_ep_mlp_validation_requires_replicated_shared_expert():
    layout = _Layout(cp_rank=0, active_local_tokens=5, counts=(5, 0, 0, 0))
    communicator, _, _, _ = _communicator(layout, use_ep_dispatch=True)
    mlp = SimpleNamespace(
        experts=SimpleNamespace(moe_tp_size=1, moe_ep_size=8),
        shared_expert=SimpleNamespace(
            gate_up_proj=SimpleNamespace(tp_size=8),
            down_proj=SimpleNamespace(tp_size=8),
        ),
    )

    with pytest.raises(RuntimeError, match="shared expert"):
        communicator.validate_mlp(mlp)


def test_cp_prefill_fused_norm_fallback_is_explicit_and_rate_limited(
    caplog, monkeypatch
):
    monkeypatch.setattr(
        welmv4_model, "_WELM_CP_FUSED_NORM_FALLBACK_WARNED", False
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(_enable_attntp_fused_norm=False),
    )
    kwargs = dict(
        use_previous_precision=False,
        residual_after_layernorm=True,
        use_o_norm=True,
        o_norm_needs_attn_tp_reduce=False,
    )

    with caplog.at_level(logging.WARNING, logger=welmv4_model.__name__):
        assert not welmv4_model._welm_should_use_mmq_norm_after_attn(
            **kwargs, use_prefill_cp_communicator=True
        )
        assert not welmv4_model._welm_should_use_mmq_norm_after_attn(
            **kwargs, use_prefill_cp_communicator=True
        )

    messages = [
        record.getMessage()
        for record in caplog.records
        if "mmq_style_norm_after_attn" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "falling back to the unfused norm path" in messages[0]
    assert welmv4_model._welm_should_use_mmq_norm_after_attn(
        **kwargs, use_prefill_cp_communicator=False
    )


def test_cp_prefill_selects_dedicated_attntp2_fused_norm(monkeypatch):
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(_enable_attntp_fused_norm=True),
    )
    kwargs = dict(
        use_previous_precision=False,
        residual_after_layernorm=True,
        use_o_norm=True,
        o_norm_needs_attn_tp_reduce=False,
    )
    attention = SimpleNamespace(
        attn_tp_size=2,
        o_proj=SimpleNamespace(
            reduce_results=True,
            use_attention_tp_reduce=True,
        ),
        o_proj_suffix_parallel_reduce=False,
    )

    assert welmv4_model._welm_should_use_prefill_cp_attntp2_fused_norm(
        **kwargs,
        use_prefill_cp_communicator=True,
        attention=attention,
        hidden_size=2048,
    )
    assert not welmv4_model._welm_should_use_prefill_cp_attntp2_fused_norm(
        **kwargs,
        use_prefill_cp_communicator=False,
        attention=attention,
        hidden_size=2048,
    )
    assert not welmv4_model._welm_should_use_mmq_norm_after_attn(
        **kwargs, use_prefill_cp_communicator=True
    )
    assert welmv4_model._welm_should_use_mmq_norm_after_attn(
        **kwargs, use_prefill_cp_communicator=False
    )


def test_postprocess_reduces_partial_then_restores_pair_without_residual():
    counts = (5, 0, 3, 2)
    layout = _Layout(cp_rank=2, active_local_tokens=3, counts=counts)
    partial = torch.arange(10, dtype=torch.float32).view(10, 1)
    restored = torch.tensor([[5.0], [6.0], [7.0]])
    residual = torch.tensor([[50.0], [60.0], [70.0]])
    communicator, _, pair_group, tp_group = _communicator(
        layout, pair_output=restored
    )
    forward_batch = type("Batch", (), {"attn_cp_prefill_runtime_layout": layout})()

    hidden, returned_residual = communicator.postprocess_layer(
        partial, residual, forward_batch
    )

    assert torch.equal(hidden, restored)
    assert returned_residual is residual
    assert tp_group.calls[0][1] is partial
    assert pair_group.calls[0][1].shape == (2, 1)
    assert communicator.should_use_reduce_scatter(forward_batch)


def test_welm_selects_prefill_communicator_per_forward_only():
    legacy = object()
    prefill = object()
    layer = type(
        "Layer",
        (),
        {
            "layer_communicator": legacy,
            "prefill_cp_communicator": prefill,
        },
    )()

    selected, is_prefill_cp = _welm_select_layer_communicator(
        layer,
        type("Batch", (), {"attn_cp_prefill_runtime_layout": object()})(),
    )
    assert selected is prefill
    assert is_prefill_cp

    selected, is_prefill_cp = _welm_select_layer_communicator(
        layer,
        type("Batch", (), {"attn_cp_prefill_runtime_layout": None})(),
    )
    assert selected is legacy
    assert not is_prefill_cp


def test_welm_localizes_hidden_and_positions_without_rewriting_full_batch_rows():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=16,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    out_cache_loc = torch.zeros(16, dtype=torch.int64)
    out_cache_loc[:4] = torch.arange(4) + 100
    out_cache_loc[12:] = torch.arange(4) + 104
    full_input_ids = torch.arange(16)
    full_positions = torch.arange(16, dtype=torch.int32)
    runtime = materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=0,
        input_ids=full_input_ids,
        positions=full_positions,
        out_cache_loc=out_cache_loc,
    )
    hidden = torch.arange(32, dtype=torch.float32).view(16, 2)
    batch = type(
        "Batch",
        (),
        {
            "attn_cp_prefill_runtime_layout": runtime,
            "input_ids": full_input_ids,
            "positions": full_positions,
        },
    )()

    local_hidden, local_positions = _welm_localize_cp_prefill_rows(
        hidden, full_positions, batch
    )

    assert torch.equal(
        local_hidden, hidden.index_select(0, runtime.local_extend_indices)
    )
    assert torch.equal(local_positions, runtime.local_positions)
    assert torch.equal(batch.input_ids, full_input_ids)
    assert torch.equal(batch.positions, full_positions)
