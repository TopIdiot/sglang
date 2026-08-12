from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import welmv4 as welmv4_model
from sglang.srt.models.welm_deferred_mirror import (
    WelmDeferredExecutionRole,
    WelmDeferredMirrorPair,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


class _Attention(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.layer_idx = layer_id
        self.num_kv_heads = 1
        self.head_dim = 4
        self.qk_norm = False
        self.only_k_norm = True
        self.k_norm = nn.LayerNorm(4)
        self.rotary_emb = object()
        self.scale_seq_factor = 1
        self.scale_seq_attn_per_suffix = False
        self.attn = SimpleNamespace(layer_id=layer_id)
        self.deferred_target_kv_finalizers = nn.ModuleDict()


class _Layer(nn.Module):
    def __init__(self, layer_id):
        super().__init__()
        self.self_attn = _Attention(layer_id)


def test_monolithic_finalizers_borrow_target_modules_without_registration():
    model = welmv4_model.Qwen2MoeModel.__new__(welmv4_model.Qwen2MoeModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_Layer(i) for i in range(4)])
    model.start_layer = 0
    model.end_layer = 4
    model.deferred_execution = SimpleNamespace(
        role=WelmDeferredExecutionRole.MONOLITHIC,
        plan=SimpleNamespace(
            pairs=(
                WelmDeferredMirrorPair(0, 2),
                WelmDeferredMirrorPair(1, 3),
            )
        ),
    )

    model._bind_monolithic_deferred_target_kv_finalizers()

    for source_id, target_id in ((0, 2), (1, 3)):
        source = model.layers[source_id].self_attn
        target = model.layers[target_id].self_attn
        finalizer = source.deferred_target_kv_finalizers[str(target_id)]
        assert finalizer.k_norm is target.k_norm
        assert finalizer.rotary_emb is target.rotary_emb
        assert finalizer.cache_layer is target.attn
        assert list(finalizer.named_parameters()) == []

    parameter_names = [name for name, _param in model.named_parameters()]
    assert not any("deferred_target_kv_finalizers" in name for name in parameter_names)
    state_keys = list(model.state_dict())
    assert not any("deferred_target_kv_finalizers" in name for name in state_keys)
    for target_id in (2, 3):
        target_name = f"layers.{target_id}.self_attn.k_norm.weight"
        assert parameter_names.count(target_name) == 1
        assert state_keys.count(target_name) == 1


class _CountingLayer(nn.Module):
    def __init__(self, layer_id, calls):
        super().__init__()
        self.layer_id = layer_id
        self.calls = calls
        self.kv_mirror_layers = ()

    def forward(
        self,
        _positions,
        hidden_states,
        _forward_batch,
        residual,
        kv_mirror_states,
    ):
        self.calls.append((self.layer_id, hidden_states.shape[0]))
        return hidden_states + 1, residual, kv_mirror_states


@pytest.mark.parametrize(
    ("deferred_prefill", "expected_layers", "expected_value", "norm_calls"),
    [
        (True, [0, 1], 2, 0),
        (False, [0, 1, 2, 3], 4, 1),
    ],
)
def test_monolithic_runtime_cutoff_preserves_ordinary_full_forward(
    monkeypatch, deferred_prefill, expected_layers, expected_value, norm_calls
):
    calls = []
    model = welmv4_model.Qwen2MoeModel.__new__(welmv4_model.Qwen2MoeModel)
    nn.Module.__init__(model)
    model.token_owner_runtime = None
    model.config = SimpleNamespace(num_hidden_layers=4)
    model.deferred_execution = SimpleNamespace(
        role=WelmDeferredExecutionRole.MONOLITHIC,
        prefill_execution_end_layer=2,
        omit_final_output=False,
    )
    model.execution_end_layer = 4
    model.start_layer = 0
    model.end_layer = 4
    model.layers = nn.ModuleList([_CountingLayer(i, calls) for i in range(4)])
    model.layers_to_capture = []
    model.mk_moe_router = None
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.embed_tokens = object()
    model.oe_grams = []
    model.oe_vocab_sizes = []
    model.vocab_size = 128
    model.scale_seq_times = 0
    model.norm = MagicMock(side_effect=lambda hidden: hidden)
    model.norm.weight = torch.ones(4, dtype=torch.bfloat16)
    monkeypatch.setattr(
        welmv4_model,
        "welm_embeddings",
        lambda **_kwargs: torch.zeros((2, 4), dtype=torch.bfloat16),
    )
    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda _batch: False
    )
    monkeypatch.setattr(welmv4_model, "_set_welm_kv_mirror_states", lambda *_args: None)
    monkeypatch.setattr(
        welmv4_model,
        "model_forward_maybe_tbo",
        MagicMock(side_effect=AssertionError("TBO must not run")),
    )
    monkeypatch.setattr(
        welmv4_model.ScatterMode,
        "model_input_output",
        lambda: object(),
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(with_current_layer=lambda _layer: nullcontext()),
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        welm_deferred_prefill=deferred_prefill,
        can_run_tbo=deferred_prefill,
        spec_info=None,
        spec_algorithm=None,
        capture_hidden_mode=SimpleNamespace(need_capture=lambda: False),
        model_specific_states=None,
        attn_cp_prefill_runtime_layout=None,
    )

    output = model(
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        forward_batch,
    )

    assert [layer for layer, _rows in calls] == expected_layers
    assert torch.equal(
        output,
        torch.full((2, 4), expected_value, dtype=torch.bfloat16),
    )
    assert model.norm.call_count == norm_calls


@pytest.mark.parametrize(
    ("dp_rank", "flags", "local_deferred", "local_rows", "expected_rows"),
    [
        (0, [True, False], True, 2, 0),
        (1, [True, False], False, 3, 3),
    ],
)
def test_deferred_dp_cutoff_builds_one_synchronized_suffix_layout(
    monkeypatch, dp_rank, flags, local_deferred, local_rows, expected_rows
):
    hidden = torch.ones((local_rows, 4), dtype=torch.bfloat16)
    residual = torch.full_like(hidden, 2)
    positions = torch.arange(local_rows, dtype=torch.int64)
    forward_batch = SimpleNamespace(
        welm_deferred_prefill=local_deferred,
        welm_deferred_prefill_flags=flags,
        global_num_tokens_cpu=[2, 3],
        global_num_tokens_gpu=torch.tensor([2, 3], dtype=torch.int64),
        global_num_tokens_for_logprob_cpu=[1, 3],
        global_num_tokens_for_logprob_gpu=torch.tensor([1, 3], dtype=torch.int64),
    )
    update_metadata = MagicMock()
    monkeypatch.setattr(
        "sglang.srt.layers.dp_attention.get_attention_dp_rank",
        lambda: dp_rank,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_update_contracted_dp_metadata", update_metadata
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)

    output, output_residual, output_positions, suffix_is_empty = (
        welmv4_model._welm_apply_deferred_prefill_dp_cutoff(
            hidden,
            residual,
            positions,
            forward_batch,
        )
    )

    assert output.shape == (expected_rows, 4)
    assert output_residual.shape == (expected_rows, 4)
    assert output_positions.shape == (expected_rows,)
    assert suffix_is_empty is False
    assert forward_batch.welm_deferred_prefill_suffix_active is local_deferred
    update_metadata.assert_called_once_with(
        forward_batch,
        expected_rows,
        synchronized_global_num_tokens=[0, 3],
        synchronized_global_num_tokens_for_logprob=[0, 3],
        force_sum_len=True,
    )


@pytest.mark.parametrize(
    (
        "flags",
        "peer_rows",
        "expected_calls",
        "expected_rows",
        "expected_invalidations",
    ),
    [
        ([True, False], 3, [(0, 2), (1, 2), (2, 0), (3, 0)], 0, 1),
        ([True, False], 0, [(0, 2), (1, 2)], 0, 0),
        ([True, True], 2, [(0, 2), (1, 2)], 0, 0),
    ],
)
def test_deferred_dp_runtime_continues_only_when_a_peer_needs_suffix(
    monkeypatch,
    flags,
    peer_rows,
    expected_calls,
    expected_rows,
    expected_invalidations,
):
    calls = []
    model = welmv4_model.Qwen2MoeModel.__new__(welmv4_model.Qwen2MoeModel)
    nn.Module.__init__(model)
    model.token_owner_runtime = MagicMock()
    model.config = SimpleNamespace(num_hidden_layers=4)
    model.deferred_execution = SimpleNamespace(
        role=WelmDeferredExecutionRole.MONOLITHIC,
        prefill_execution_end_layer=2,
        omit_final_output=False,
    )
    model.execution_end_layer = 4
    model.start_layer = 0
    model.end_layer = 4
    model.layers = nn.ModuleList([_CountingLayer(i, calls) for i in range(4)])
    model.layers_to_capture = []
    model.mk_moe_router = None
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.embed_tokens = object()
    model.oe_grams = []
    model.oe_vocab_sizes = []
    model.vocab_size = 128
    model.scale_seq_times = 0
    model.norm = MagicMock(side_effect=lambda hidden: hidden)
    model.norm.weight = torch.ones(4, dtype=torch.bfloat16)
    monkeypatch.setattr(
        welmv4_model,
        "welm_embeddings",
        lambda **_kwargs: torch.zeros((2, 4), dtype=torch.bfloat16),
    )
    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda _batch: False
    )
    monkeypatch.setattr(welmv4_model, "_set_welm_kv_mirror_states", lambda *_: None)
    monkeypatch.setattr(
        welmv4_model,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(with_current_layer=lambda _layer: nullcontext()),
    )
    monkeypatch.setattr(
        "sglang.srt.layers.dp_attention.get_attention_dp_rank", lambda: 0
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_update_contracted_dp_metadata", MagicMock()
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        welm_deferred_prefill=True,
        welm_deferred_prefill_flags=flags,
        global_num_tokens_cpu=[2, peer_rows],
        global_num_tokens_gpu=torch.tensor(
            [2, peer_rows], dtype=torch.int64
        ),
        global_num_tokens_for_logprob_cpu=[1, peer_rows],
        global_num_tokens_for_logprob_gpu=torch.tensor(
            [1, peer_rows], dtype=torch.int64
        ),
        can_run_tbo=False,
        spec_info=None,
        spec_algorithm=None,
        capture_hidden_mode=SimpleNamespace(need_capture=lambda: False),
        model_specific_states=None,
        attn_cp_prefill_runtime_layout=None,
    )

    output = model(
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        forward_batch,
    )

    assert calls == expected_calls
    assert output.shape == (expected_rows, 4)
    assert model.token_owner_runtime.invalidate.call_count == expected_invalidations


def test_mirror_q_allows_missing_kv_only_for_deferred_zero_row_suffix(monkeypatch):
    projection = welmv4_model.MirrorQProjection.__new__(
        welmv4_model.MirrorQProjection
    )
    nn.Module.__init__(projection)
    projection.imitated_layer_idx = 1
    projection.mirror_layer_idx = 3
    projection._apply_qkv = MagicMock(
        side_effect=lambda hidden: hidden.new_empty((hidden.shape[0], 8))
    )
    hidden = torch.empty((0, 4), dtype=torch.bfloat16)
    forward_batch = SimpleNamespace(
        welm_deferred_prefill=True,
        welm_deferred_prefill_suffix_active=True,
    )
    attn = SimpleNamespace(q_size=8, kv_size=4, need_clear_kv_cache=False)
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda _batch: False
    )

    q, k, v, projected_hidden = projection(
        attn, hidden, forward_batch, kv_mirror_states={}
    )

    assert q.shape == (0, 8)
    assert k.shape == (0, 4)
    assert v.shape == (0, 4)
    assert projected_hidden is hidden
    projection._apply_qkv.assert_not_called()

    forward_batch.welm_deferred_prefill_suffix_active = False
    with pytest.raises(RuntimeError, match="Missing mirrored KV activation"):
        projection(attn, hidden, forward_batch, kv_mirror_states={})


def test_deferred_zero_row_suffix_still_participates_in_dp_collectives(monkeypatch):
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: True)
    forward_batch = SimpleNamespace(
        global_num_tokens_gpu=torch.tensor([0, 3], dtype=torch.int64),
        welm_deferred_prefill_suffix_active=True,
        forward_mode=ForwardMode.EXTEND,
        is_extend_in_batch=True,
    )

    assert welmv4_model._welm_needs_empty_dp_collectives(
        forward_batch, is_nextn=False
    )

    forward_batch.welm_deferred_prefill_suffix_active = False
    assert not welmv4_model._welm_needs_empty_dp_collectives(
        forward_batch, is_nextn=False
    )
