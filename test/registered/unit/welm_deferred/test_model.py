from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import welmv4 as welmv4_model
from sglang.srt.models import welmv4_token_owner as token_owner
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
        self.mtp_direct_kv_finalizers = nn.ModuleDict()


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


@pytest.mark.parametrize("page_size", [1, 16])
def test_direct_finalizer_writes_canonical_slot_through_swa_pool(page_size):
    canonical_slot = 2 * page_size
    physical_slot = 3 * page_size
    destination_pool = SWAKVPool.__new__(SWAKVPool)
    destination_pool.page_size = page_size
    destination_pool.swa_loc = None
    destination_pool.layers_mapping = {4: (0, True)}
    destination_pool.swa_kv_pool = MagicMock()
    destination_pool.full_kv_pool = MagicMock()
    mapping = torch.zeros((canonical_slot + 1,), dtype=torch.int64)
    mapping[canonical_slot] = physical_slot
    destination_pool.register_mapping(mapping)

    finalizer = welmv4_model.WelmMirrorTargetKVFinalizer.__new__(
        welmv4_model.WelmMirrorTargetKVFinalizer
    )
    nn.Module.__init__(finalizer)
    finalizer.target_layer_id = 4
    finalizer.num_kv_heads = 1
    finalizer.head_dim = 4
    finalizer.scale_seq_factor = 1
    finalizer.scale_rope_positions = False
    finalizer.apply_k_norm = False
    finalizer.k_norm = nn.Identity()
    finalizer.rotary_emb = SimpleNamespace(forward_k_only_cuda=lambda *_args: None)
    finalizer.cache_layer = welmv4_model.WelmDeferredKVCacheLayer(
        layer_id=4,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=4,
        v_head_dim=4,
    )
    finalizer.destination_pool = destination_pool
    forward_batch = SimpleNamespace(
        out_cache_loc=torch.tensor([canonical_slot], dtype=torch.int64),
        token_to_kv_pool=MagicMock(),
        attn_cp_prefill_runtime_layout=None,
    )
    key = torch.arange(4, dtype=torch.bfloat16).view(1, 4)
    value = torch.arange(4, 8, dtype=torch.bfloat16).view(1, 4)

    finalizer(torch.tensor([3], dtype=torch.int64), key, value, forward_batch)

    destination_pool.swa_kv_pool.set_kv_buffer.assert_called_once()
    args = destination_pool.swa_kv_pool.set_kv_buffer.call_args.args
    assert args[0] is None
    torch.testing.assert_close(
        args[1], torch.tensor([physical_slot], dtype=torch.int32)
    )
    torch.testing.assert_close(args[2].view_as(key), key)
    torch.testing.assert_close(args[3].view_as(value), value)
    assert destination_pool.swa_kv_pool.set_kv_buffer.call_args.kwargs == {
        "layer_id_override": 0
    }
    destination_pool.full_kv_pool.set_kv_buffer.assert_not_called()
    forward_batch.token_to_kv_pool.set_kv_buffer.assert_not_called()


def test_bind_mtp_direct_kv_uses_preserved_target_layer_count():
    class DirectAttention(_Attention):
        def __init__(self, layer_id, projection):
            super().__init__(layer_id)
            self.qkv_proj = projection
            self.mtp_direct_kv_finalizers = nn.ModuleDict()

    class DirectLayer(nn.Module):
        def __init__(self, layer_id, projection):
            super().__init__()
            self.self_attn = DirectAttention(layer_id, projection)

    def mirror_projection(projection_cls, mirror_layer_idx):
        projection = projection_cls.__new__(projection_cls)
        nn.Module.__init__(projection)
        projection.imitated_layer_idx = 0
        projection.mirror_layer_idx = mirror_layer_idx
        return projection

    target = welmv4_model.WeLMV4MoeForCausalLM.__new__(
        welmv4_model.WeLMV4MoeForCausalLM
    )
    nn.Module.__init__(target)
    target.config = SimpleNamespace(
        num_hidden_layers=1,
        num_target_hidden_layers=4,
        num_nextn_predict_layers=1,
        kv_mirror_layers=[2, 4],
        kv_mirror_imitated_layers=[0, 1],
    )
    target.model = welmv4_model.Qwen2MoeModel.__new__(welmv4_model.Qwen2MoeModel)
    nn.Module.__init__(target.model)
    target.model.config = target.config
    target.model.start_layer = 0
    target.model.end_layer = 4
    target.model.layers = nn.ModuleList(
        [
            DirectLayer(0, nn.Identity()),
            DirectLayer(1, nn.Identity()),
            DirectLayer(2, mirror_projection(welmv4_model.MirrorQProjection, 2)),
            DirectLayer(3, nn.Identity()),
        ]
    )

    draft = SimpleNamespace(
        model=SimpleNamespace(
            decoder_layers=nn.ModuleList(
                [
                    DirectLayer(
                        0,
                        mirror_projection(welmv4_model.NextnMirrorQProjection, 4),
                    )
                ]
            )
        )
    )
    target_pool = object()
    draft_pool = object()
    target.model.layers[0].self_attn.deferred_target_kv_finalizers["2"] = (
        nn.Identity()
    )
    target.model.layers[1].self_attn.deferred_target_kv_finalizers["4"] = (
        nn.Identity()
    )

    target.model.bind_mtp_direct_kv(
        draft.model,
        target_kv_pool=target_pool,
        draft_kv_pool=draft_pool,
    )

    base_finalizer = target.model.layers[0].self_attn.mtp_direct_kv_finalizers["2"]
    nextn_finalizer = target.model.layers[1].self_attn.mtp_direct_kv_finalizers["4"]
    assert base_finalizer.target_layer_id == 2
    assert base_finalizer.cache_layer.layer_id == 2
    assert base_finalizer.destination_pool is target_pool
    assert nextn_finalizer.target_layer_id == 4
    assert nextn_finalizer.cache_layer.layer_id == 0
    assert nextn_finalizer.destination_pool is draft_pool
    assert target.model.layers[2].self_attn.qkv_proj.mirror_kv_cache_ready
    assert draft.model.decoder_layers[0].self_attn.qkv_proj.mirror_kv_cache_ready
    assert not target.model.layers[0].self_attn.deferred_target_kv_finalizers
    assert not target.model.layers[1].self_attn.deferred_target_kv_finalizers
    assert target.model.mtp_direct_kv_enabled


def test_bind_storage_draft_kv_maps_logical_nextn_to_local_layer():
    model = welmv4_model.Qwen2MoeModel.__new__(welmv4_model.Qwen2MoeModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        kv_mirror_layers=[4],
        kv_mirror_imitated_layers=[1],
    )
    model.layers = nn.ModuleList([_Layer(i) for i in range(4)])
    model.start_layer = 0
    model.end_layer = 4
    model.mtp_direct_kv_enabled = False
    finalizer = welmv4_model.WelmMirrorTargetKVFinalizer(
        target_layer_id=4,
        num_kv_heads=1,
        head_dim=4,
        qk_norm=False,
        k_norm=False,
        qk_norm_eps=1e-5,
        rotary_emb=object(),
        scale_seq_factor=1,
        scale_rope_positions=False,
    )
    model.layers[1].self_attn.deferred_target_kv_finalizers["4"] = finalizer
    draft_config = SimpleNamespace(
        full_attention_layer_ids=[],
        swa_attention_layer_ids=[0],
        head_dim=4,
        v_head_dim=4,
        get_num_kv_heads=lambda tp_size: 1,
    )
    draft_pool = SimpleNamespace(
        page_size=16,
        head_num=1,
        head_dim=4,
        swa_kv_pool=SimpleNamespace(head_num=1, head_dim=4, v_head_dim=4),
        full_kv_pool=SimpleNamespace(head_num=1, head_dim=4, v_head_dim=4),
        layers_mapping={0: (0, True)},
    )

    model.bind_mtp_storage_draft_kv(
        draft_config,
        draft_kv_pool=draft_pool,
    )

    assert finalizer.target_layer_id == 4
    assert finalizer.cache_layer.layer_id == 0
    assert finalizer.destination_pool is draft_pool
    assert model.layers[1].self_attn.mtp_direct_kv_finalizers["4"] is finalizer
    assert "4" not in model.layers[1].self_attn.deferred_target_kv_finalizers
    assert model.mtp_direct_kv_enabled


def test_storage_prefill_builds_nextn_finalizers_without_deferred_execution():
    model = welmv4_model.Qwen2MoeModel.__new__(welmv4_model.Qwen2MoeModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        kv_mirror_layers=[4],
        kv_mirror_imitated_layers=[1],
        scale_seq_times=0,
        scale_seq_attn_per_suffix_layerwise=[],
        _sglang_welm_mtp_storage_draft_kv=True,
    )
    model.layers = nn.ModuleList([_Layer(i) for i in range(4)])
    model.start_layer = 0
    model.end_layer = 4

    model._bind_storage_draft_target_kv_finalizers()

    finalizers = model.layers[1].self_attn.deferred_target_kv_finalizers
    assert tuple(finalizers) == ("4",)
    assert finalizers["4"].target_layer_id == 4


@pytest.mark.parametrize(("target_pool", "draft_pool"), [(None, object()), (object(), None)])
def test_bind_mtp_direct_kv_rejects_missing_destination_pool(
    target_pool, draft_pool
):
    model = welmv4_model.Qwen2MoeModel.__new__(welmv4_model.Qwen2MoeModel)
    nn.Module.__init__(model)
    model.mtp_direct_kv_enabled = False

    with pytest.raises(RuntimeError, match="destination pools"):
        model.bind_mtp_direct_kv(
            SimpleNamespace(),
            target_kv_pool=target_pool,
            draft_kv_pool=draft_pool,
        )


def test_direct_source_finalization_consumes_all_mirror_kv_without_deferred_flag():
    calls = []

    class Finalizer(nn.Module):
        def __init__(self, target_layer_id):
            super().__init__()
            self.target_layer_id = target_layer_id

        def forward(self, positions, key, value, forward_batch):
            calls.append((self.target_layer_id, positions, key, value, forward_batch))

    attention = SimpleNamespace(
        mtp_direct_kv_finalizers=nn.ModuleDict(
            {"2": Finalizer(2), "4": Finalizer(4)}
        ),
        deferred_target_kv_finalizers=nn.ModuleDict({"3": Finalizer(3)}),
    )
    positions = torch.tensor([0, 1], dtype=torch.int64)
    forward_batch = SimpleNamespace(welm_deferred_prefill=False)
    states = {
        layer: (
            torch.full((2, 4), layer, dtype=torch.bfloat16),
            torch.full((2, 4), -layer, dtype=torch.bfloat16),
        )
        for layer in (2, 4)
    }

    welmv4_model._welm_finalize_source_kv(
        attention,
        positions,
        forward_batch,
        states,
    )

    assert [target for target, *_rest in calls] == [2, 4]
    assert states == {}


def test_storage_source_finalizes_direct_nextn_and_deferred_base_kv():
    calls = []

    class Finalizer(nn.Module):
        def __init__(self, target_layer_id):
            super().__init__()
            self.target_layer_id = target_layer_id

        def forward(self, positions, key, value, forward_batch):
            calls.append(self.target_layer_id)

    attention = SimpleNamespace(
        mtp_direct_kv_finalizers=nn.ModuleDict({"4": Finalizer(4)}),
        deferred_target_kv_finalizers=nn.ModuleDict({"3": Finalizer(3)}),
    )
    states = {
        layer: (torch.ones((1, 4)), torch.ones((1, 4))) for layer in (3, 4)
    }

    welmv4_model._welm_finalize_source_kv(
        attention,
        torch.tensor([0]),
        SimpleNamespace(welm_deferred_prefill=True),
        states,
    )

    assert calls == [4, 3]
    assert states == {}


@pytest.mark.parametrize(
    ("projection_cls", "apply_method"),
    [
        (welmv4_model.MirrorQProjection, "_apply_qkv"),
        (welmv4_model.NextnMirrorQProjection, "_apply_q_only"),
    ],
)
def test_direct_mirror_consumer_projects_q_without_mirror_tensor(
    monkeypatch, projection_cls, apply_method
):
    projection = projection_cls.__new__(projection_cls)
    nn.Module.__init__(projection)
    projection.imitated_layer_idx = 1
    projection.mirror_layer_idx = 4
    projection.mirror_kv_cache_ready = True
    project_q = MagicMock(
        side_effect=lambda hidden, *_args: hidden.new_empty((hidden.shape[0], 8))
    )
    setattr(projection, apply_method, project_q)
    hidden = torch.empty((2, 4), dtype=torch.bfloat16)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_prefill_runtime_layout=None,
        spec_info=SimpleNamespace(mirrored_kv_indices=None),
    )
    attn = SimpleNamespace(q_size=8, kv_size=4, need_clear_kv_cache=False)
    states = {}
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda _batch: False
    )

    q, k, v, projected_hidden = projection(
        attn, hidden, forward_batch, kv_mirror_states=states
    )

    assert q.shape == (2, 8)
    assert k is None
    assert v is None
    assert projected_hidden is hidden
    assert states == {}
    project_q.assert_called_once()


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
    (
        "deferred_prefill",
        "direct_kv",
        "expected_layers",
        "expected_value",
        "norm_calls",
    ),
    [
        (True, False, [0, 1], 2, 0),
        (False, False, [0, 1, 2, 3], 4, 1),
        (True, True, [0, 1], 2, 0),
    ],
)
def test_monolithic_runtime_cutoff_preserves_ordinary_full_forward(
    monkeypatch,
    deferred_prefill,
    direct_kv,
    expected_layers,
    expected_value,
    norm_calls,
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
    model.mtp_direct_kv_enabled = direct_kv
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
    take_nextn = MagicMock(
        side_effect=AssertionError("direct K/V must not materialize mirror payload")
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_take_nextn_kv_mirror_states", take_nextn
    )
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
        spec_algorithm=SimpleNamespace(is_eagle=lambda: True) if direct_kv else None,
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
    take_nextn.assert_not_called()


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


def test_deferred_dp_cutoff_rebuilds_token_owner_plan(monkeypatch):
    flags = [True, False]
    peer_rows = 3
    calls = []
    model = welmv4_model.Qwen2MoeModel.__new__(welmv4_model.Qwen2MoeModel)
    nn.Module.__init__(model)
    model.token_owner_runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=0),
    )
    model.token_owner_boundary_layer = 2
    model.disable_prefill_mirror_token_owner = False
    model.decode_token_owner_enabled = True
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
        token_owner,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_deepep=lambda: False),
    )
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
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        global_forward_modes=[ForwardMode.EXTEND, ForwardMode.EXTEND],
        enable_welm_kv_mirror_opt=True,
        welm_deferred_prefill=True,
        welm_deferred_prefill_flags=flags,
        welm_kv_mirror_contract_flags=[False, not flags[1] and peer_rows > 0],
        original_global_num_tokens_cpu=[2, peer_rows],
        global_num_reqs_cpu=[1, min(peer_rows, 1)],
        welm_kv_mirror_last_q_indices_cpu=[],
        welm_kv_mirror_active_batch_indices_cpu=[],
        welm_kv_mirror_output_size=1,
        global_num_tokens_cpu=[2, peer_rows],
        global_num_tokens_gpu=torch.tensor(
            [2, peer_rows], dtype=torch.int64
        ),
        global_num_tokens_for_logprob_cpu=[1, peer_rows],
        global_num_tokens_for_logprob_gpu=torch.tensor(
            [1, peer_rows], dtype=torch.int64
        ),
        global_dp_buffer_len=2 + peer_rows,
        dp_padding_mode=welmv4_model.DpPaddingMode.SUM_LEN,
        dp_local_start_pos=torch.tensor(0),
        dp_local_num_tokens=torch.tensor(2),
        num_token_non_padded=None,
        scale_seq_factor=1,
        is_extend_in_batch=True,
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

    assert calls == [(0, 2), (1, 2), (2, 0), (3, 0)]
    assert output.shape == (0, 4)
    plan_state = model.token_owner_runtime.state_for(
        forward_batch, token_owner.TokenOwnerLayerIdentity.PRE_MIRROR
    )
    assert plan_state.input_local_layout.valid_token_count == 0
    boundary_state = model.token_owner_runtime.state_for(
        forward_batch, token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY
    )
    assert boundary_state.transitions_rows
    assert boundary_state.owner_output


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
