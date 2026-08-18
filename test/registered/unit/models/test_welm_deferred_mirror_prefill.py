from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from sglang.srt.layers.rotary_embedding import base as rotary_base
from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.models import welmv4 as welmv4_model
from sglang.srt.models.welm_deferred_mirror import (
    WelmDeferredExecutionRole,
    WelmDeferredModelExecution,
    bind_welm_deferred_model_execution,
    build_welm_deferred_mirror_plan,
    get_welm_deferred_model_execution,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _mirror_config():
    return SimpleNamespace(
        num_hidden_layers=48,
        num_nextn_predict_layers=1,
        kv_mirror_layers=list(range(48, 32, -1)),
        kv_mirror_imitated_layers=list(range(16)),
    )


def _model_config(role: str | None):
    config = _mirror_config()
    config.pad_token_id = None
    config.vocab_size = 128
    config.hidden_size = 16
    config.rms_norm_eps = 1e-5
    config.oe_dim = 0
    config.oe_grams = []
    config.oe_vocab_sizes = []
    config.scale_seq_times = 0
    if role is not None:
        bind_welm_deferred_model_execution(
            config,
            build_welm_deferred_mirror_plan(config),
            role=role,
            capture_nextn=True,
        )
    return config


def test_role_specific_execution_keeps_logical_layers_but_prunes_only_prefill():
    plan = build_welm_deferred_mirror_plan(_mirror_config())

    prefill = WelmDeferredModelExecution(role="prefill", plan=plan)
    decode = WelmDeferredModelExecution(role="decode", plan=plan)
    monolithic = WelmDeferredModelExecution(role="monolithic", plan=plan)

    assert prefill.logical_num_layers == 48
    assert prefill.execution_end_layer == 33
    assert prefill.omitted_layer_ids == tuple(range(33, 48))
    assert prefill.omit_final_output
    assert decode.logical_num_layers == 48
    assert decode.execution_end_layer == 48
    assert decode.omitted_layer_ids == ()
    assert not decode.omit_final_output
    assert monolithic.logical_num_layers == 48
    assert monolithic.execution_end_layer == 48
    assert monolithic.prefill_execution_end_layer == 33
    assert monolithic.omitted_layer_ids == ()
    assert not monolithic.omit_final_output


def test_execution_binding_is_explicit_and_round_trips_through_config():
    config = _mirror_config()
    plan = build_welm_deferred_mirror_plan(config)

    execution = bind_welm_deferred_model_execution(config, plan, role="prefill")

    assert get_welm_deferred_model_execution(config) is execution
    with pytest.raises(ValueError, match="prefill or decode"):
        bind_welm_deferred_model_execution(config, plan, role="invalid")


def test_prefill_and_monolithic_project_base_and_nextn_pairs():
    prefill_config = _model_config("prefill")
    decode_config = _model_config("decode")
    monolithic_config = _model_config("monolithic")

    assert welmv4_model._welm_effective_kv_mirror_pairs(prefill_config) == (
        [*range(33, 48), 48],
        [*range(15, 0, -1), 0],
    )
    assert welmv4_model._welm_effective_kv_mirror_pairs(decode_config) == (
        list(range(48, 32, -1)),
        list(range(16)),
    )
    assert welmv4_model._welm_effective_kv_mirror_pairs(monolithic_config) == (
        [*range(33, 48), 48],
        [*range(15, 0, -1), 0],
    )


def test_deferred_execution_does_not_capture_nextn_unless_requested():
    config = _mirror_config()
    bind_welm_deferred_model_execution(
        config,
        build_welm_deferred_mirror_plan(config),
        role="prefill",
    )

    assert welmv4_model._welm_effective_kv_mirror_pairs(config) == (
        list(range(33, 48)),
        list(range(15, 0, -1)),
    )


def test_nextn_capture_source_must_be_in_executed_prefill_prefix():
    config = _model_config("prefill")
    config.kv_mirror_imitated_layers[0] = 40

    with pytest.raises(ValueError, match="executed Target prefix"):
        welmv4_model._welm_effective_kv_mirror_pairs(config)


def test_take_nextn_capture_states_requires_complete_layer_set():
    config = _model_config("prefill")
    nextn_state = (
        torch.ones((2, 4), dtype=torch.bfloat16),
        torch.full((2, 4), 2, dtype=torch.bfloat16),
    )
    states = {48: nextn_state}

    captured = welmv4_model._welm_take_nextn_kv_mirror_states(config, states)

    assert captured == {48: nextn_state}
    assert states == {}

    with pytest.raises(RuntimeError, match="missing NextN mirror K/V.*48"):
        welmv4_model._welm_take_nextn_kv_mirror_states(config, {})


class _FakeDecoderLayer(nn.Module):
    def __init__(self, layer_id, **_kwargs):
        super().__init__()
        self.layer_id = layer_id


def _patch_model_construction(monkeypatch):
    pp_group = SimpleNamespace(
        is_first_rank=True,
        is_last_rank=True,
        rank_in_group=0,
        world_size=1,
    )
    monkeypatch.setattr(welmv4_model, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        welmv4_model, "_welm_token_owner_enabled", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            enable_welm_v45_80a3_mk_moe_router=False,
            welm_shared_embedding_policy="disabled",
            welm_vocab_padding_size=0,
        ),
    )
    monkeypatch.setattr(
        welmv4_model,
        "VocabParallelEmbedding",
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        welmv4_model,
        "WelmV4FusedRMSNorm",
        lambda *_args, **_kwargs: nn.Identity(),
    )

    def make_layers(num_layers, layer_fn, **_kwargs):
        layers = nn.ModuleList(
            [
                layer_fn(idx=layer_id, prefix=f"layers.{layer_id}")
                for layer_id in range(num_layers)
            ]
        )
        return layers, 0, num_layers

    monkeypatch.setattr(welmv4_model, "make_layers", make_layers)
    monkeypatch.setattr(
        welmv4_model.Qwen2MoeModel,
        "_bind_monolithic_deferred_target_kv_finalizers",
        lambda _self: None,
    )


@pytest.mark.parametrize(
    ("role", "expected_real_layers", "expect_missing_norm"),
    [
        ("prefill", 33, True),
        ("decode", 48, False),
        ("monolithic", 48, False),
        (None, 48, False),
    ],
)
def test_model_construction_preserves_48_logical_slots(
    monkeypatch, role, expected_real_layers, expect_missing_norm
):
    _patch_model_construction(monkeypatch)

    model = welmv4_model.Qwen2MoeModel(
        _model_config(role),
        decoder_layer_type=_FakeDecoderLayer,
    )

    assert len(model.layers) == 48
    assert model.execution_end_layer == expected_real_layers
    assert all(
        isinstance(layer, _FakeDecoderLayer)
        for layer in model.layers[:expected_real_layers]
    )
    assert all(
        isinstance(layer, PPMissingLayer)
        for layer in model.layers[expected_real_layers:]
    )
    assert isinstance(model.norm, PPMissingLayer) is expect_missing_norm


def test_prefill_causal_lm_does_not_construct_lm_head(monkeypatch):
    config = _model_config("prefill")
    monkeypatch.setattr(
        welmv4_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: object())
    monkeypatch.setattr(
        welmv4_model,
        "Qwen2MoeModel",
        lambda *_args, **_kwargs: SimpleNamespace(scale_seq_times=0),
    )

    def unexpected_lm_head(*_args, **_kwargs):
        raise AssertionError("deferred Prefill must not construct LM head")

    monkeypatch.setattr(welmv4_model, "ParallelLMHead", unexpected_lm_head)
    monkeypatch.setattr(welmv4_model, "LogitsProcessor", MagicMock())

    model = welmv4_model.WeLMV4MoeForCausalLM(config)

    assert isinstance(model.lm_head, PPMissingLayer)


def test_deferred_prefill_skips_decode_cuda_graph_capture(monkeypatch):
    from sglang.srt.model_executor import model_runner as model_runner_module

    runner = model_runner_module.ModelRunner.__new__(model_runner_module.ModelRunner)
    runner.is_generation = True
    runner.device = "cuda"
    runner.gpu_id = 0
    runner.welm_deferred_mirror_plan = object()
    runner.server_args = SimpleNamespace(
        disaggregation_mode="prefill",
        disable_cuda_graph=False,
        model_impl="auto",
    )
    monkeypatch.setattr(
        model_runner_module,
        "CudaGraphRunner",
        MagicMock(
            side_effect=AssertionError(
                "deferred Prefill must not capture a decode CUDA graph"
            )
        ),
    )
    monkeypatch.setattr(
        model_runner_module,
        "get_available_gpu_memory",
        MagicMock(return_value=1.0),
    )

    runner.init_device_graphs()

    assert runner.graph_runner is None
    assert runner.graph_mem_usage == 0


def test_monolithic_does_not_take_prefill_cuda_graph_skip(caplog):
    from sglang.srt.model_executor import model_runner as model_runner_module

    runner = model_runner_module.ModelRunner.__new__(model_runner_module.ModelRunner)
    runner.is_generation = True
    runner.device = "cpu"
    runner.gpu_id = 0
    runner.welm_deferred_mirror_plan = object()
    runner.server_args = SimpleNamespace(
        disaggregation_mode="null",
        disable_cuda_graph=False,
        enable_torch_compile=False,
        model_impl="auto",
    )

    runner.init_device_graphs()

    assert "Skip decode CUDA graph capture" not in caplog.text


@pytest.mark.parametrize("target_layer_id", [33, 34])
def test_target_finalizer_applies_target_norm_rope_and_layer_identity(
    monkeypatch, target_layer_id
):
    captured = {}
    finalizer = welmv4_model.WelmMirrorTargetKVFinalizer.__new__(
        welmv4_model.WelmMirrorTargetKVFinalizer
    )
    nn.Module.__init__(finalizer)
    finalizer.target_layer_id = target_layer_id
    finalizer.num_kv_heads = 1
    finalizer.head_dim = 4
    finalizer.scale_seq_factor = 1
    finalizer.scale_rope_positions = False
    finalizer.apply_k_norm = True
    finalizer.k_norm = SimpleNamespace(
        weight=torch.ones(4, dtype=torch.bfloat16),
        eps=1e-5,
    )
    finalizer.rotary_emb = SimpleNamespace(
        forward_k_only_cuda=lambda positions, key: captured.update(
            rope_positions=positions.clone(), rope_key=key.clone()
        )
    )
    finalizer.cache_layer = welmv4_model.WelmDeferredKVCacheLayer(
        layer_id=target_layer_id,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=4,
        v_head_dim=4,
    )
    raw_k = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    raw_v = torch.arange(8, 16, dtype=torch.bfloat16).view(2, 4)
    normalized_k = raw_k + 10
    monkeypatch.setattr(
        welmv4_model,
        "mmq_style_k_rms_norm",
        lambda key, _weight, _eps: normalized_k.view(2, 1, 4),
    )

    def write_kv(
        cache_layer, key, value, forward_batch, *, token_to_kv_pool=None
    ):
        captured["cache_layer"] = cache_layer
        captured["write_key"] = key.clone()
        captured["write_value"] = value.clone()
        captured["forward_batch"] = forward_batch
        captured["token_to_kv_pool"] = token_to_kv_pool

    monkeypatch.setattr(welmv4_model, "_welm_write_kv_cache_only", write_kv)
    forward_batch = SimpleNamespace(
        out_cache_loc=torch.tensor([0, 1], dtype=torch.int64)
    )
    positions = torch.tensor([5, 6], dtype=torch.int64)

    finalizer(positions, raw_k, raw_v, forward_batch)

    assert torch.equal(captured["rope_positions"], positions)
    assert torch.equal(captured["rope_key"], normalized_k)
    assert captured["cache_layer"].layer_id == target_layer_id
    assert captured["token_to_kv_pool"] is None
    assert torch.equal(captured["write_key"], normalized_k)
    assert torch.equal(captured["write_value"], raw_v)
    assert captured["forward_batch"] is forward_batch


@pytest.mark.parametrize(
    ("target_layer_id", "expected_pool_kind"),
    [(33, "swa"), (34, "full")],
)
def test_deferred_target_layer_identity_routes_existing_swa_and_full_pools(
    target_layer_id, expected_pool_kind
):
    captured = {}

    class RoutingPool:
        layer_kinds = {33: "swa", 34: "full"}

        def set_kv_buffer(
            self, layer, loc, key, value, k_scale, v_scale
        ):
            captured["pool_kind"] = self.layer_kinds[layer.layer_id]
            captured["loc"] = loc.clone()
            captured["key"] = key.clone()
            captured["value"] = value.clone()

    cache_layer = welmv4_model.WelmDeferredKVCacheLayer(
        layer_id=target_layer_id,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=4,
        v_head_dim=4,
    )
    key = torch.arange(4, dtype=torch.bfloat16).view(1, 4)
    value = torch.arange(4, 8, dtype=torch.bfloat16).view(1, 4)
    forward_batch = SimpleNamespace(
        out_cache_loc=torch.tensor([7], dtype=torch.int64),
        token_to_kv_pool=RoutingPool(),
    )

    welmv4_model._welm_write_kv_cache_only(
        cache_layer,
        key,
        value,
        forward_batch,
    )

    assert captured["pool_kind"] == expected_pool_kind
    assert torch.equal(captured["loc"], forward_batch.out_cache_loc)
    assert torch.equal(captured["key"].view(1, 4), key)
    assert torch.equal(captured["value"].view(1, 4), value)


def test_deferred_target_kv_write_uses_existing_cp_local_cache_locations():
    captured = {}

    class RecordingPool:
        def set_kv_buffer(
            self, layer, loc, key, value, k_scale, v_scale
        ):
            captured["layer"] = layer
            captured["loc"] = loc.clone()
            captured["key"] = key.clone()
            captured["value"] = value.clone()

    cache_layer = welmv4_model.WelmDeferredKVCacheLayer(
        layer_id=33,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=4,
        v_head_dim=4,
    )
    key = torch.arange(8, dtype=torch.bfloat16).view(2, 4)
    value = torch.arange(8, 16, dtype=torch.bfloat16).view(2, 4)
    local_cache_loc = torch.tensor([17, 23], dtype=torch.int64)
    forward_batch = SimpleNamespace(
        out_cache_loc=torch.tensor([0, 17, 23, 0], dtype=torch.int64),
        attn_cp_prefill_runtime_layout=SimpleNamespace(
            local_out_cache_loc=local_cache_loc
        ),
        token_to_kv_pool=RecordingPool(),
    )

    welmv4_model._welm_write_kv_cache_only(
        cache_layer,
        key,
        value,
        forward_batch,
    )

    assert captured["layer"] is cache_layer
    assert torch.equal(captured["loc"], local_cache_loc)
    assert torch.equal(captured["key"].view(2, 4), key)
    assert torch.equal(captured["value"].view(2, 4), value)


def test_source_finalization_consumes_raw_mirror_state_immediately():
    captured = {}

    class Finalizer(nn.Module):
        def forward(self, positions, key, value, forward_batch):
            captured["args"] = (positions, key, value, forward_batch)

    states = {
        33: (
            torch.ones((2, 4), dtype=torch.bfloat16),
            torch.full((2, 4), 2, dtype=torch.bfloat16),
        )
    }
    finalizers = nn.ModuleDict({"33": Finalizer()})
    positions = torch.tensor([0, 1], dtype=torch.int64)
    forward_batch = object()

    welmv4_model._welm_finalize_target_kv(
        finalizers,
        positions,
        forward_batch,
        states,
    )

    assert states == {}
    assert captured["args"][0] is positions
    assert captured["args"][3] is forward_batch


def test_source_finalization_fails_when_projection_did_not_produce_target_kv():
    finalizers = nn.ModuleDict({"33": nn.Identity()})

    with pytest.raises(RuntimeError, match="target layer 33"):
        welmv4_model._welm_finalize_target_kv(
            finalizers,
            torch.tensor([0]),
            object(),
            {},
        )


def test_target_finalizer_fails_instead_of_dropping_nonempty_kv_without_cache():
    finalizer = welmv4_model.WelmMirrorTargetKVFinalizer.__new__(
        welmv4_model.WelmMirrorTargetKVFinalizer
    )
    nn.Module.__init__(finalizer)
    finalizer.target_layer_id = 33
    finalizer.num_kv_heads = 1
    finalizer.head_dim = 4
    finalizer.scale_seq_factor = 1
    finalizer.scale_rope_positions = False
    finalizer.apply_k_norm = False
    finalizer.k_norm = nn.Identity()
    finalizer.rotary_emb = SimpleNamespace(forward_k_only_cuda=lambda *_args: None)
    finalizer.cache_layer = welmv4_model.WelmDeferredKVCacheLayer(
        layer_id=33,
        tp_k_head_num=1,
        tp_v_head_num=1,
        qk_head_dim=4,
        v_head_dim=4,
    )

    with pytest.raises(RuntimeError, match="no cache destination.*layer 33"):
        finalizer(
            torch.tensor([0], dtype=torch.int64),
            torch.ones((1, 4), dtype=torch.bfloat16),
            torch.ones((1, 4), dtype=torch.bfloat16),
            SimpleNamespace(out_cache_loc=None),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_k_only_rope_matches_existing_qk_rope_exactly(monkeypatch):
    monkeypatch.setattr(
        rotary_base,
        "get_global_server_args",
        lambda: SimpleNamespace(rl_on_policy_target=None),
    )
    rope = welmv4_model.WelmV4InplaceRotaryEmbedding(
        head_size=8,
        rotary_dim=4,
        max_position_embeddings=32,
        base=10000,
        is_neox_style=True,
        dtype=torch.bfloat16,
    ).cuda()
    positions = torch.arange(7, device="cuda", dtype=torch.int64)
    generator = torch.Generator(device="cuda").manual_seed(7)
    query = torch.randn(
        (7, 16),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    key = torch.randn(
        (7, 8),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    expected_key = key.clone()
    actual_key = key.clone()

    rope.forward_cuda(positions, query, expected_key)
    rope.forward_k_only_cuda(positions, actual_key)

    torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("scale_rope_positions", [False, True])
def test_target_finalizer_matches_target_layer_kv_preparation_exactly(
    monkeypatch, scale_rope_positions
):
    monkeypatch.setattr(
        rotary_base,
        "get_global_server_args",
        lambda: SimpleNamespace(rl_on_policy_target=None),
    )
    rope = welmv4_model.WelmV4InplaceRotaryEmbedding(
        head_size=8,
        rotary_dim=4,
        max_position_embeddings=64,
        base=10000,
        is_neox_style=True,
        dtype=torch.bfloat16,
    ).cuda()
    finalizer = welmv4_model.WelmMirrorTargetKVFinalizer(
        target_layer_id=33,
        num_kv_heads=2,
        head_dim=8,
        qk_norm=False,
        k_norm=True,
        qk_norm_eps=1e-5,
        rotary_emb=rope,
        scale_seq_factor=2,
        scale_rope_positions=scale_rope_positions,
    ).cuda()
    generator = torch.Generator(device="cuda").manual_seed(11)
    finalizer.k_norm.weight.data.copy_(
        torch.randn(
            (8,),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
    )
    positions = torch.arange(9, device="cuda", dtype=torch.int64) * 3
    raw_key = torch.randn(
        (9, 16),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    raw_value = torch.randn(
        (9, 16),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    expected_key = welmv4_model.mmq_style_k_rms_norm(
        raw_key.view(9, 2, 8).contiguous(),
        finalizer.k_norm.weight,
        finalizer.k_norm.eps,
    ).view_as(raw_key)
    dummy_query = torch.randn(
        (9, 16),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    rope_positions = positions // 2 if scale_rope_positions else positions
    rope.forward_cuda(rope_positions, dummy_query, expected_key)

    captured = {}

    def write_kv(
        cache_layer, key, value, forward_batch, *, token_to_kv_pool=None
    ):
        captured["layer_id"] = cache_layer.layer_id
        captured["key"] = key.clone()
        captured["value"] = value.clone()
        captured["forward_batch"] = forward_batch
        captured["token_to_kv_pool"] = token_to_kv_pool

    monkeypatch.setattr(welmv4_model, "_welm_write_kv_cache_only", write_kv)
    forward_batch = SimpleNamespace(
        out_cache_loc=torch.arange(9, device="cuda", dtype=torch.int64)
    )

    finalizer(positions, raw_key.clone(), raw_value, forward_batch)

    assert captured["layer_id"] == 33
    assert captured["token_to_kv_pool"] is None
    assert captured["forward_batch"] is forward_batch
    torch.testing.assert_close(captured["key"], expected_key, rtol=0, atol=0)
    torch.testing.assert_close(captured["value"], raw_value, rtol=0, atol=0)


def _make_weight_loader_model(
    monkeypatch, *, include_v_route=True, storage_only=False
):
    loaded = {}

    class FakeProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

            def weight_loader(param, tensor, slot):
                loaded.setdefault("projection", {})[slot] = tensor.clone()
                param.data.copy_(tensor.reshape_as(param))

            self.weight.weight_loader = weight_loader

        def param_mapping(self, _layer_idx):
            mapping = {
                "k2_0": "model.layers.33.self_attn.k_proj.weight",
            }
            if include_v_route:
                mapping["v2_0"] = "model.layers.33.self_attn.v_proj.weight"
            return mapping

    class FakeFinalizer(nn.Module):
        def __init__(self, target_layer_id=33):
            super().__init__()
            self.target_layer_id = target_layer_id
            self.k_norm = nn.Module()
            self.k_norm.weight = nn.Parameter(
                torch.zeros(4, dtype=torch.bfloat16)
            )

    monkeypatch.setattr(
        welmv4_model, "ImitateQkvMultiBankKvProjection", FakeProjection
    )
    monkeypatch.setattr(
        welmv4_model, "WelmMirrorTargetKVFinalizer", FakeFinalizer
    )
    monkeypatch.setattr(
        welmv4_model.FusedMoE,
        "make_expert_params_mapping",
        lambda **_kwargs: [],
    )

    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = FakeProjection()
            self.deferred_target_kv_finalizers = nn.ModuleDict(
                {"33": FakeFinalizer(33), "48": FakeFinalizer(48)}
            )

    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = FakeAttention()

    class FakeBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(
                [PPMissingLayer() for _ in range(15)] + [FakeLayer()]
            )
            self.start_layer = 0
            self.end_layer = 48

    model = welmv4_model.WeLMV4MoeForCausalLM.__new__(
        welmv4_model.WeLMV4MoeForCausalLM
    )
    nn.Module.__init__(model)
    config = _model_config(None if storage_only else "prefill")
    config._sglang_welm_mtp_storage_draft_kv = storage_only
    config.num_experts = 0
    model.config = config
    model.model = FakeBaseModel()
    model.deferred_execution = get_welm_deferred_model_execution(config)
    return model, loaded


def test_deferred_weight_loader_relocates_target_kv_and_knorm(monkeypatch):
    model, loaded = _make_weight_loader_model(monkeypatch)
    target_k = torch.tensor([3], dtype=torch.bfloat16)
    target_v = torch.tensor([5], dtype=torch.bfloat16)
    target_k_norm = torch.arange(4, dtype=torch.bfloat16)
    omitted_q = torch.ones(8, dtype=torch.bfloat16)
    omitted_norm = torch.ones(4, dtype=torch.bfloat16)
    omitted_head = torch.ones(16, dtype=torch.bfloat16)

    model.load_weights(
        [
            ("model.layers.33.self_attn.k_proj.weight", target_k),
            ("model.layers.33.self_attn.v_proj.weight", target_v),
            ("model.layers.33.self_attn.k_norm.weight", target_k_norm),
            ("model.layers.33.self_attn.q_proj.weight", omitted_q),
            ("model.norm.weight", omitted_norm),
            ("lm_head.weight", omitted_head),
        ]
    )

    assert torch.equal(loaded["projection"]["k2_0"], target_k)
    assert torch.equal(loaded["projection"]["v2_0"], target_v)
    finalizer = model.model.layers[15].self_attn.deferred_target_kv_finalizers["33"]
    assert torch.equal(finalizer.k_norm.weight, target_k_norm)


def test_deferred_weight_loader_relocates_nextn_knorm(monkeypatch):
    model, _ = _make_weight_loader_model(monkeypatch)
    target_k_norm = torch.arange(4, dtype=torch.bfloat16)

    model.load_weights(
        [("model.layers.48.self_attn.k_norm.weight", target_k_norm)]
    )

    finalizer = model.model.layers[15].self_attn.deferred_target_kv_finalizers["48"]
    assert torch.equal(finalizer.k_norm.weight, target_k_norm)


def test_storage_only_weight_loader_relocates_nextn_knorm_without_deferred(
    monkeypatch,
):
    model, _ = _make_weight_loader_model(monkeypatch, storage_only=True)
    target_k_norm = torch.arange(4, dtype=torch.bfloat16)

    model.load_weights(
        [("model.layers.48.self_attn.k_norm.weight", target_k_norm)]
    )

    finalizer = model.model.layers[15].self_attn.deferred_target_kv_finalizers["48"]
    assert torch.equal(finalizer.k_norm.weight, target_k_norm)


def test_deferred_weight_loader_rejects_missing_required_target_kv_route(monkeypatch):
    model, _ = _make_weight_loader_model(monkeypatch, include_v_route=False)

    with pytest.raises(RuntimeError, match="target V projection.*layer 33"):
        model.load_weights(
            [
                (
                    "model.layers.33.self_attn.v_proj.weight",
                    torch.ones(1, dtype=torch.bfloat16),
                )
            ]
        )


def test_deferred_weight_loader_rejects_duplicate_relocated_weight(monkeypatch):
    model, _ = _make_weight_loader_model(monkeypatch)
    weight = torch.arange(4, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="duplicate relocated weight"):
        model.load_weights(
            [
                ("model.layers.33.self_attn.k_norm.weight", weight),
                ("model.layers.33.self_attn.k_norm.weight", weight),
            ]
        )


def test_deferred_weight_loader_rejects_duplicate_relocated_k_before_second_load(
    monkeypatch,
):
    model, loaded = _make_weight_loader_model(monkeypatch)
    weight = torch.ones(1, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="duplicate relocated weight"):
        model.load_weights(
            [
                ("model.layers.33.self_attn.k_proj.weight", weight),
                ("model.layers.33.self_attn.k_proj.weight", weight),
            ]
        )

    assert torch.equal(loaded["projection"]["k2_0"], weight)
