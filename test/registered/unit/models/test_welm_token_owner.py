import dataclasses
from types import SimpleNamespace

import pytest
import torch

from sglang.srt import server_args as server_args_module
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor import model_runner
from sglang.srt.models import welmv4
from sglang.srt.models import welmv4_nextn
from sglang.srt.models import welmv4_token_owner as token_owner
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.welmv4_mtp_draft_proposal_cuda_graph_runner import (
    WelmMTPDraftProposalCudaGraphRunner,
    _use_pretranslated_swa_cache_loc,
)


def valid_capability_kwargs():
    return {
        "enabled": True,
        "dp_attention_enabled": True,
        "dp_size": 4,
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


def pure_tp_capability_kwargs():
    kwargs = valid_capability_kwargs()
    kwargs.update(
        dp_attention_enabled=False,
        dp_size=1,
        attn_tp_size=4,
        global_tp_size=4,
        global_tp_rank=3,
        attn_tp_rank=3,
    )
    return kwargs


def test_token_owner_disables_mtp_swa_graph_staging():
    assert not _use_pretranslated_swa_cache_loc(
        is_hybrid_swa=True, use_token_owner=True
    )
    assert _use_pretranslated_swa_cache_loc(
        is_hybrid_swa=True, use_token_owner=False
    )
    assert not _use_pretranslated_swa_cache_loc(
        is_hybrid_swa=False, use_token_owner=False
    )


@pytest.mark.parametrize(
    (
        "override",
        "dp_attention_enabled",
        "dp_size",
        "disaggregation_mode",
        "expected",
    ),
    [
        (None, False, 1, "null", False),
        (None, False, 1, "prefill", True),
        (None, False, 1, "decode", True),
        (None, True, 2, "null", True),
        (True, True, 2, "decode", False),
        (False, False, 1, "null", True),
    ],
)
def test_welm_decode_token_owner_policy(
    override,
    dp_attention_enabled,
    dp_size,
    disaggregation_mode,
    expected,
):
    assert (
        server_args_module.resolve_welm_decode_token_owner_enabled(
            token_owner_enabled=True,
            disable_override=override,
            dp_attention_enabled=dp_attention_enabled,
            dp_size=dp_size,
            disaggregation_mode=disaggregation_mode,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("forward_mode", "variable_decode_extend", "global_modes", "expected"),
    [
        (welmv4.ForwardMode.EXTEND, False, None, True),
        (welmv4.ForwardMode.MIXED, False, None, True),
        (welmv4.ForwardMode.SPLIT_PREFILL, False, None, True),
        (welmv4.ForwardMode.DECODE, False, None, False),
        (welmv4.ForwardMode.IDLE, False, None, False),
        (welmv4.ForwardMode.TARGET_VERIFY, False, None, False),
        (welmv4.ForwardMode.DRAFT_EXTEND, False, None, False),
        (welmv4.ForwardMode.DRAFT_EXTEND_V2, False, None, False),
        (
            welmv4.ForwardMode.DECODE,
            False,
            [welmv4.ForwardMode.DECODE, welmv4.ForwardMode.EXTEND],
            True,
        ),
        (
            welmv4.ForwardMode.IDLE,
            False,
            [welmv4.ForwardMode.EXTEND, welmv4.ForwardMode.IDLE],
            True,
        ),
        (
            welmv4.ForwardMode.DRAFT_EXTEND,
            False,
            [welmv4.ForwardMode.DRAFT_EXTEND, welmv4.ForwardMode.IDLE],
            False,
        ),
        (welmv4.ForwardMode.EXTEND, True, None, False),
        (
            welmv4.ForwardMode.EXTEND,
            True,
            [welmv4.ForwardMode.EXTEND, welmv4.ForwardMode.IDLE],
            True,
        ),
    ],
)
def test_welm_token_owner_forward_phase_policy(
    forward_mode,
    variable_decode_extend,
    global_modes,
    expected,
):
    forward_batch = SimpleNamespace(
        forward_mode=forward_mode,
        welm_mtp_variable_decode_extend=variable_decode_extend,
        global_forward_modes=global_modes,
    )

    assert (
        welmv4._welm_token_owner_enabled_for_forward(
            forward_batch,
            decode_token_owner_enabled=False,
        )
        is expected
    )


def test_welm_non_owner_mirror_contracts_residual_before_attntp_split(
    monkeypatch,
):
    captured = {}

    class FakeAttention:
        kv_mirror_layer_idx = 33
        attn_tp_size = 2
        attn_tp_rank = 1
        use_o_norm = False
        o_norm_needs_attn_tp_reduce = False
        o_proj_suffix_parallel_reduce = False
        o_proj = SimpleNamespace(reduce_results=True)

        def __call__(self, **kwargs):
            return kwargs["hidden_states"][-1:]

    def prepare_mlp(hidden_states, residual, *_args, **_kwargs):
        captured["residual"] = residual
        return hidden_states, residual

    communicator = SimpleNamespace(
        layer_scatter_modes=SimpleNamespace(
            layer_input_mode=welmv4.ScatterMode.SCATTERED,
            mlp_mode=welmv4.ScatterMode.SCATTERED,
        ),
        prepare_attn=lambda hidden, residual, *_a, **_k: (hidden, residual),
        prepare_mlp=prepare_mlp,
        should_use_reduce_scatter=lambda *_: False,
        postprocess_layer=lambda hidden, residual, *_a, **_k: (hidden, residual),
    )
    mlp = lambda hidden, *_args, **_kwargs: hidden
    mlp.tp_size = 2
    layer = SimpleNamespace(
        enable_token_owner=False,
        layer_communicator=communicator,
        prefill_cp_communicator=object(),
        tp_dp_attntp_fused_norm_managers={},
        _prefill_cp_mlp_validated=False,
        hidden_size=1,
        ppln=True,
        config_layer_id=1,
        prenorm_layer_idx=[],
        self_attn=FakeAttention(),
        kv_mirror_layers=[33],
        is_nextn=False,
        is_final_layer=False,
        layer_id=33,
        mlp=mlp,
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=None,
        custom_last_index=torch.tensor([3]),
        kv_mirror_active_batch_indices=torch.tensor([0]),
        kv_mirror_output_size=2,
        dp_padding_mode=None,
        forward_mode=welmv4.ForwardMode.EXTEND,
    )

    monkeypatch.setattr(welmv4, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(welmv4, "is_suffix_parallel_enabled", lambda: False)
    monkeypatch.setattr(
        welmv4, "_welm_select_layer_communicator", lambda *_: (communicator, False)
    )
    monkeypatch.setattr(
        welmv4, "_welm_select_tp_dp_attntp_fused_norm_manager", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        welmv4, "_welm_should_use_mmq_norm_after_attn", lambda **_kwargs: False
    )
    monkeypatch.setattr(
        welmv4,
        "_welm_should_use_prefill_cp_attntp2_fused_norm",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        welmv4, "_welm_needs_empty_dp_collectives", lambda *_a, **_k: False
    )
    monkeypatch.setattr(welmv4, "_welm_should_dispatch_attention", lambda *_: True)
    monkeypatch.setattr(
        welmv4, "_welm_should_sync_kv_mirror_dp_metadata", lambda *_: False
    )
    monkeypatch.setattr(
        welmv4, "_welm_should_contract_kv_mirror", lambda *_: True
    )
    monkeypatch.setattr(
        welmv4, "_welm_kv_mirror_row_alignment", lambda *_: 2
    )

    welmv4.Qwen2MoeDecoderLayer.forward(
        layer,
        positions=torch.arange(4),
        hidden_states=torch.arange(1, 5, dtype=torch.bfloat16).reshape(4, 1),
        forward_batch=forward_batch,
        residual=None,
        kv_mirror_states={},
    )

    assert captured["residual"].tolist() == [[0.0]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_welm_token_owner_reuses_completed_count_staging():
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=0),
    )
    destination = torch.empty(2, dtype=torch.int64, device="cuda")

    runtime._copy_counts(destination, [1, 2], staging_slot="tokens")
    torch.cuda.synchronize()
    first_source = runtime._count_staging["tokens"][0].tensor
    runtime._copy_counts(destination, [3, 4], staging_slot="tokens")
    torch.cuda.synchronize()

    assert runtime._count_staging["tokens"][0].tensor is first_source
    assert len(runtime._count_staging["tokens"]) == 1
    assert destination.cpu().tolist() == [3, 4]


@pytest.mark.parametrize(
    (
        "mlp_mode",
        "ppln",
        "use_o_norm",
        "o_proj_reduces",
        "suffix_reduce",
        "error",
    ),
    [
        (welmv4.ScatterMode.SCATTERED, True, True, True, False, None),
        (welmv4.ScatterMode.FULL, False, False, True, False, None),
        (
            welmv4.ScatterMode.FULL,
            False,
            True,
            False,
            False,
            "requires the native pure-TP o_proj reduction topology",
        ),
    ],
)
def test_welm_pure_tp_prefill_mirror_exit_reduces_only_in_output_communicator(
    monkeypatch,
    mlp_mode,
    ppln,
    use_o_norm,
    o_proj_reduces,
    suffix_reduce,
    error,
):
    captured = {}
    input_mode = (
        welmv4.ScatterMode.SCATTERED
        if mlp_mode == welmv4.ScatterMode.SCATTERED
        else welmv4.ScatterMode.TP_ATTN_FULL
    )

    class OwnerCommunicator:
        def prepare_attn(self, hidden_states, residual, _forward_batch, **kwargs):
            return hidden_states, residual

    class ReplicatedCommunicator:
        layer_scatter_modes = SimpleNamespace(
            layer_input_mode=input_mode,
            mlp_mode=mlp_mode,
        )

        def prepare_attn(self, hidden_states, residual, _forward_batch, **_kwargs):
            return hidden_states, residual

        def prepare_mlp(self, hidden_states, residual, _forward_batch, **_kwargs):
            captured["prepare_mlp_shapes"] = (
                tuple(hidden_states.shape),
                tuple(residual.shape),
            )
            if mlp_mode == welmv4.ScatterMode.SCATTERED:
                hidden_states = hidden_states.tensor_split(2)[1]
            return hidden_states, residual

        def should_use_reduce_scatter(self, _forward_batch):
            return False

        def postprocess_layer(
            self, hidden_states, residual, _forward_batch, **_kwargs
        ):
            captured["postprocess"] = True
            return hidden_states, residual

    class FakeAttention:
        kv_mirror_layer_idx = 33
        attn_tp_size = 2
        attn_tp_rank = 1
        o_proj_suffix_parallel_reduce = suffix_reduce
        o_norm = SimpleNamespace(weight=torch.ones(1), eps=1e-5)

        def __init__(self):
            self.use_o_norm = use_o_norm
            self.o_proj = SimpleNamespace(reduce_results=o_proj_reduces)
            self.o_norm_needs_attn_tp_reduce = (
                use_o_norm and not o_proj_reduces
            )

        def __call__(self, **kwargs):
            captured["attention_kwargs"] = kwargs
            return kwargs["hidden_states"]

    class FakeMLP:
        tp_size = 2

        def __call__(
            self,
            hidden_states,
            hidden_states_fp32,
            _forward_batch,
            _use_reduce_scatter,
            **_kwargs,
        ):
            captured["mlp_hidden_states_fp32"] = hidden_states_fp32
            return hidden_states

    owner_communicator = OwnerCommunicator()
    replicated_communicator = ReplicatedCommunicator()
    owner_layout = token_owner.TokenOwnerLayout.balanced(
        valid_token_count=4,
        owner_count=2,
        local_owner_rank=1,
    )
    owner_state = token_owner.TokenOwnerLayerState(
        transitions_rows=True,
        owner_input=True,
        owner_output=False,
        input_local_layout=owner_layout,
        output_local_layout=None,
        output_global_layout=None,
        transport_local_layout=None,
        local_contracts_rows=False,
        survivor_source_rows=(),
        survivor_output_rows=(),
        router_context=None,
    )
    tail_state = dataclasses.replace(
        owner_state,
        transitions_rows=False,
        owner_input=False,
        input_local_layout=None,
    )

    class FakeRuntime:
        def state_for(self, _forward_batch, identity):
            return {
                token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY: owner_state,
                token_owner.TokenOwnerLayerIdentity.MIRROR_TAIL: tail_state,
            }[identity]

        def publish_boundary_transport(self, _forward_batch, state):
            assert state is owner_state

        def contract_boundary_residual(
            self, residual, state, *, output_scattered
        ):
            assert state is owner_state
            return residual.tensor_split(2)[1] if output_scattered else residual

    layer = SimpleNamespace(
        enable_token_owner=True,
        disable_prefill_mirror_token_owner=True,
        token_owner_uses_global_tp_moe=False,
        token_owner_runtime=FakeRuntime(),
        token_owner_layer_identity=(
            token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY
        ),
        layer_communicator=owner_communicator,
        prefill_mirror_communicator=replicated_communicator,
        prefill_cp_communicator=object(),
        tp_dp_attntp_fused_norm_managers={},
        hidden_size=1,
        ppln=ppln,
        config_layer_id=1,
        prenorm_layer_idx=[],
        self_attn=FakeAttention(),
        kv_mirror_layers=[33],
        is_nextn=False,
        is_final_layer=False,
        layer_id=33,
        mlp=FakeMLP(),
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=None,
        dp_padding_mode=None,
        forward_mode=welmv4.ForwardMode.EXTEND,
    )
    hidden_states = torch.ones((4, 1), dtype=torch.bfloat16)
    residual = None if ppln else torch.ones_like(hidden_states)

    monkeypatch.setattr(welmv4, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4, "is_suffix_parallel_enabled", lambda: False)
    monkeypatch.setattr(
        welmv4, "_welm_needs_empty_dp_collectives", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        welmv4, "_welm_should_dispatch_attention", lambda *_args: True
    )
    monkeypatch.setattr(
        welmv4, "_welm_should_sync_kv_mirror_dp_metadata", lambda *_: True
    )
    monkeypatch.setattr(
        welmv4,
        "_welm_update_contracted_dp_metadata",
        lambda *_args, **_kwargs: pytest.fail(
            "forward-plan layers must not enter legacy DP metadata sync"
        ),
    )
    monkeypatch.setattr(
        welmv4, "_welm_should_contract_kv_mirror", lambda *_: False
    )
    monkeypatch.setattr(welmv4, "_welm_kv_mirror_row_alignment", lambda *_: 1)

    if error is not None:
        with pytest.raises(RuntimeError, match=error):
            welmv4.Qwen2MoeDecoderLayer.forward(
                layer,
                positions=torch.arange(4),
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                residual=residual,
                kv_mirror_states={},
            )
        return

    welmv4.Qwen2MoeDecoderLayer.forward(
        layer,
        positions=torch.arange(4),
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        residual=residual,
        kv_mirror_states={},
    )

    assert forward_batch.welm_kv_mirror_full_q_attention is False
    assert captured["attention_kwargs"]["skip_o_proj_reduce"] is True
    assert captured["attention_kwargs"]["skip_o_norm"] is True
    assert captured["prepare_mlp_shapes"] == (
        (4, 1),
        (2, 1) if mlp_mode == welmv4.ScatterMode.SCATTERED else (4, 1),
    )
    assert captured["mlp_hidden_states_fp32"] is None
    assert captured["postprocess"] is True

    layer.token_owner_layer_identity = token_owner.TokenOwnerLayerIdentity.MIRROR_TAIL
    welmv4.Qwen2MoeDecoderLayer.forward(
        layer,
        positions=torch.arange(4),
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        residual=residual,
        kv_mirror_states={},
    )


@pytest.mark.parametrize("moe_a2a_backend", ["none", "deepep"])
def test_welm_dp_attention_supports_token_owner(moe_a2a_backend):
    kwargs = valid_capability_kwargs()
    kwargs["moe_a2a_backend"] = moe_a2a_backend

    assert token_owner.validate_welm_token_owner_capability(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"global_tp_size": 8}, "single owner group"),
        ({"dp_size": 2}, "DP=1"),
    ],
)
def test_welm_pure_tp_rejects_unsupported_group_topology(overrides, message):
    kwargs = pure_tp_capability_kwargs()
    kwargs.update(overrides)

    with pytest.raises(NotImplementedError, match=message):
        token_owner.validate_welm_token_owner_capability(**kwargs)


@pytest.mark.parametrize(
    ("moe_a2a_backend", "speculative_moe_a2a_backend"),
    [
        ("none", None),
        ("none", "none"),
        ("deepep", None),
        ("deepep", "deepep"),
    ],
)
def test_welm_pure_tp_supports_mtp_backends(
    moe_a2a_backend,
    speculative_moe_a2a_backend,
):
    kwargs = pure_tp_capability_kwargs()
    kwargs.update(
        speculative_enabled=True,
        moe_a2a_backend=moe_a2a_backend,
        speculative_moe_a2a_backend=speculative_moe_a2a_backend,
        deepep_mode="auto",
    )

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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
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


def test_welm_token_owner_rejects_pdmux():
    kwargs = valid_capability_kwargs()
    kwargs["pdmux_enabled"] = True

    with pytest.raises(NotImplementedError, match="PDMux"):
        token_owner.validate_welm_token_owner_capability(**kwargs)


def test_pure_tp_owner_metadata_does_not_trigger_unpaired_mlp_sync_post(monkeypatch):
    from sglang.srt.model_executor.model_runner import ModelRunner

    calls = []
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = "cuda"
    runner.graph_runner = None
    runner.hisparse_coordinator = None
    runner.server_args = SimpleNamespace(
        enable_token_owner=True,
        enable_dp_attention=False,
        attn_cp_size=1,
        _welm_token_owner_args_validated=True,
    )
    runner.pp_group = SimpleNamespace(is_last_rank=True)
    monkeypatch.setattr(
        model_runner,
        "set_is_extend_in_batch",
        lambda value: calls.append(f"deepep:{value}"),
    )

    def forward_extend(forward_batch, **_kwargs):
        forward_batch.global_num_tokens_cpu = [forward_batch.batch_size]
        return object(), False

    runner.forward_extend = forward_extend
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_cuda_graph=lambda: False,
            is_decode=lambda: False,
            is_split_prefill=lambda: False,
            is_extend=lambda **_kwargs: True,
            is_idle=lambda: False,
        ),
        global_num_tokens_cpu=None,
        num_token_non_padded=None,
        global_num_tokens_gpu=None,
        out_cache_loc_swa=None,
        is_extend_in_batch=True,
        batch_size=3,
        prepare_mlp_sync_batch=lambda _runner: calls.append("prepare"),
        prepare_attn_tp_scatter_input=lambda _runner: calls.append("scatter"),
        post_forward_mlp_sync_batch=lambda _ret: calls.append("post"),
    )

    runner._forward_raw(
        forward_batch,
        skip_attn_backend_init=False,
        pp_proxy_tensors=None,
    )

    assert calls == ["deepep:True", "scatter"]


def test_non_welm_token_owner_flag_does_not_skip_mlp_sync():
    server_args = SimpleNamespace(
        enable_token_owner=True,
        enable_dp_attention=False,
        attn_cp_size=1,
        enable_mixed_chunk=False,
        dp_size=1,
        disaggregation_mode="null",
        _welm_token_owner_args_validated=False,
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=[4],
        forward_mode=welmv4.ForwardMode.EXTEND,
        global_forward_modes=None,
    )

    assert model_runner._uses_mlp_sync_batch(server_args, forward_batch)


@pytest.mark.parametrize(
    ("forward_mode", "effective_decode_owner", "expected"),
    [
        (welmv4.ForwardMode.EXTEND, False, False),
        (welmv4.ForwardMode.DECODE, False, True),
        (welmv4.ForwardMode.DECODE, True, False),
    ],
)
def test_pure_tp_mlp_sync_follows_effective_token_owner_phase(
    monkeypatch, forward_mode, effective_decode_owner, expected
):
    monkeypatch.setattr(
        model_runner,
        "get_welm_decode_token_owner_enabled",
        lambda _server_args: effective_decode_owner,
        raising=False,
    )
    server_args = SimpleNamespace(
        enable_token_owner=True,
        enable_dp_attention=False,
        attn_cp_size=1,
        _welm_token_owner_args_validated=True,
    )
    forward_batch = SimpleNamespace(
        forward_mode=forward_mode,
        welm_mtp_variable_decode_extend=False,
        global_forward_modes=None,
        global_num_tokens_cpu=[1],
    )

    assert model_runner._uses_mlp_sync_batch(server_args, forward_batch) is expected


def test_welm_nextn_initializes_owner_runtime_from_selected_rows(monkeypatch):
    begin_forward_calls = []

    def begin_forward(_forward_batch, **kwargs):
        begin_forward_calls.append(kwargs)

    model = welmv4_nextn.WeLMV4ModelNextN.__new__(
        welmv4_nextn.WeLMV4ModelNextN
    )
    torch.nn.Module.__init__(model)
    model.token_owner_runtime = SimpleNamespace(begin_forward=begin_forward)
    model.decode_token_owner_enabled = True
    model._build_merged_query_embedding_only = lambda *_args: torch.zeros((8, 4))

    def assert_owner_state_is_installed(forward_batch):
        assert forward_batch.welm_token_owner_enabled

    monkeypatch.setattr(
        welmv4_nextn, "_welm_should_contract_kv_mirror", lambda _batch: True
    )
    monkeypatch.setattr(
        welmv4_nextn,
        "_welm_init_kv_mirror_last_q_indices",
        assert_owner_state_is_installed,
    )
    monkeypatch.setattr(
        welmv4_nextn,
        "_welm_select_kv_mirror_rows",
        lambda hidden, _batch, **_kwargs: hidden[:0],
    )
    monkeypatch.setattr(
        welmv4_nextn,
        "_welm_scatter_kv_mirror_rows",
        lambda hidden, _batch: hidden,
    )
    monkeypatch.setattr(
        welmv4_nextn,
        "_welm_update_contracted_dp_metadata",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(welmv4_nextn, "is_dp_attention_enabled", lambda: False)

    forward_batch = SimpleNamespace(
        mtp_step_idx=0,
        kv_fill_only=True,
        kv_mirror_output_size=0,
        spec_info=SimpleNamespace(hidden_states=torch.zeros((8, 4))),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        is_extend_in_batch=False,
        global_num_tokens_gpu=None,
    )

    hidden_states, next_hidden_states = model(
        torch.arange(8),
        torch.arange(8),
        forward_batch,
    )

    assert hidden_states.shape == next_hidden_states.shape == (0, 4)
    assert begin_forward_calls == [
        {
            "single_group_physical_rows": 0,
            "transition": token_owner.TokenOwnerTransition.NONE,
            "device": torch.device("cpu"),
        }
    ]


def test_welm_mtp_pruned_logits_metadata_keeps_layer_dp_counts(monkeypatch):
    layer_counts = torch.tensor([16, 16], dtype=torch.int32)
    logprob_counts = torch.tensor([16, 16], dtype=torch.int32)
    logits_metadata = SimpleNamespace(
        global_num_tokens_gpu=layer_counts,
        global_num_tokens_for_logprob_gpu=logprob_counts,
        global_num_tokens_for_logprob_cpu=[4, 4],
        global_dp_buffer_len=32,
        dp_local_start_pos=torch.tensor(16),
        dp_local_num_tokens=torch.tensor(16),
        welm_kv_mirror_contracted=False,
    )
    monkeypatch.setattr(
        welmv4_nextn.LogitsMetadata,
        "from_forward_batch",
        lambda _batch: logits_metadata,
    )
    monkeypatch.setattr(welmv4_nextn, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(welmv4_nextn, "get_attention_dp_rank", lambda: 1)

    prepared = welmv4_nextn._welm_prepare_mtp_pruned_logits_metadata(
        forward_batch := SimpleNamespace(
            _welm_mtp_contract_global_num_tokens_cpu=[4, 4]
        ),
        local_num_tokens=4,
    )

    assert layer_counts.tolist() == [16, 16]
    assert prepared.global_num_tokens_gpu.tolist() == [4, 4]
    assert prepared.global_num_tokens_gpu is not layer_counts
    assert prepared.global_num_tokens_gpu is logprob_counts
    assert prepared.global_num_tokens_for_logprob_gpu is logprob_counts
    assert prepared.global_dp_buffer_len == 8
    assert prepared.dp_local_start_pos is None
    assert prepared.dp_local_num_tokens is None
    assert prepared.welm_kv_mirror_contracted
    assert forward_batch._welm_mtp_contracted_dp_metadata_rows == 4
    assert not hasattr(forward_batch, "welm_kv_mirror_contracted")


def make_token_owner_plan_for_test(
    *,
    transition,
    pre_rows,
    post_rows,
    contract_flags,
    last_q_indices,
    active_batch_indices,
):
    owner_count = 2
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=owner_count, rank_in_group=0),
        global_tp_group=SimpleNamespace(
            world_size=owner_count * len(pre_rows), rank_in_group=0
        ),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=list(pre_rows),
        original_global_num_tokens_cpu=list(pre_rows),
        global_num_tokens_for_logprob_cpu=list(pre_rows),
        global_num_reqs_cpu=list(post_rows),
        welm_kv_mirror_contract_flags=list(contract_flags),
        welm_kv_mirror_last_q_indices_cpu=list(last_q_indices),
        welm_kv_mirror_active_batch_indices_cpu=list(active_batch_indices),
        welm_kv_mirror_output_size=post_rows[0],
        dp_padding_mode=None,
    )
    plan = runtime.begin_forward(
        forward_batch,
        single_group_physical_rows=(pre_rows[0] if len(pre_rows) == 1 else None),
        transition=transition,
        device=torch.device("cpu"),
    )
    return runtime, plan, forward_batch


def test_welm_token_owner_forward_plan_is_immutable_and_lookup_is_stable():
    runtime, plan, forward_batch = make_token_owner_plan_for_test(
        transition=token_owner.TokenOwnerTransition.CONTRACT_KEEP_OWNER,
        pre_rows=(8,),
        post_rows=(3,),
        contract_flags=(True,),
        last_q_indices=(1, 3, 7),
        active_batch_indices=(0, 1, 2),
    )

    before = runtime.state_for(
        forward_batch, token_owner.TokenOwnerLayerIdentity.PRE_MIRROR
    )
    boundary = runtime.state_for(
        forward_batch, token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY
    )
    tail = runtime.state_for(
        forward_batch, token_owner.TokenOwnerLayerIdentity.MIRROR_TAIL
    )

    assert before.owner_input and before.owner_output
    assert boundary.owner_input and boundary.owner_output
    assert boundary.transitions_rows
    assert tail.owner_input and tail.owner_output
    assert before.output_local_layout.owner_sizes == (4, 4)
    assert boundary.output_local_layout.owner_sizes == (2, 1)
    assert boundary.local_contracts_rows
    assert boundary.survivor_source_rows == (1, 3)
    assert boundary.survivor_output_rows == (0, 1)
    assert tail is plan.state_for(token_owner.TokenOwnerLayerIdentity.MIRROR_TAIL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.transition = token_owner.TokenOwnerTransition.NONE


def test_welm_token_owner_forward_plan_exits_owner_at_boundary():
    runtime, plan, forward_batch = make_token_owner_plan_for_test(
        transition=token_owner.TokenOwnerTransition.CONTRACT_EXIT_OWNER,
        pre_rows=(8,),
        post_rows=(3,),
        contract_flags=(True,),
        last_q_indices=(1, 3, 7),
        active_batch_indices=(0, 1, 2),
    )

    boundary = runtime.state_for(
        forward_batch, token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY
    )
    tail = runtime.state_for(
        forward_batch, token_owner.TokenOwnerLayerIdentity.MIRROR_TAIL
    )

    assert boundary.owner_input and not boundary.owner_output
    assert boundary.transitions_rows
    assert boundary.output_local_layout is None
    assert boundary.output_global_layout is None
    assert boundary.local_contracts_rows
    assert boundary.survivor_source_rows == (1, 3, 7)
    assert boundary.survivor_output_rows == (0, 1, 2)
    assert not tail.owner_input and not tail.owner_output
    assert tail.input_local_layout is None
    assert tail.transport_local_layout is None
    assert plan.post_global_num_tokens_cpu == (3,)


def test_welm_token_owner_forward_plan_contracts_only_flagged_dp_group():
    _, plan, _ = make_token_owner_plan_for_test(
        transition=token_owner.TokenOwnerTransition.CONTRACT_KEEP_OWNER,
        pre_rows=(8, 4),
        post_rows=(3, 4),
        contract_flags=(True, False),
        last_q_indices=(1, 3, 7),
        active_batch_indices=(0, 1, 2),
    )

    boundary = plan.state_for(token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY)
    tail = plan.state_for(token_owner.TokenOwnerLayerIdentity.MIRROR_TAIL)
    assert plan.post_global_num_tokens_cpu == (3, 4)
    assert boundary.output_local_layout.owner_sizes == (2, 1)
    assert boundary.output_global_layout.owner_sizes == (3, 0, 2, 2)


def test_welm_token_owner_plan_contracts_boundary_residual_in_owner_domain():
    runtime, plan, _ = make_token_owner_plan_for_test(
        transition=token_owner.TokenOwnerTransition.CONTRACT_KEEP_OWNER,
        pre_rows=(8,),
        post_rows=(3,),
        contract_flags=(True,),
        last_q_indices=(1, 3, 7),
        active_batch_indices=(0, 1, 2),
    )
    boundary = plan.state_for(token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY)

    contracted = runtime.contract_boundary_residual(
        torch.arange(4, dtype=torch.float32).view(4, 1),
        boundary,
        output_scattered=False,
    )

    assert contracted.tolist() == [[1.0], [3.0]]
    with pytest.raises(RuntimeError, match="has 3 rows, expected 4"):
        runtime.contract_boundary_residual(
            torch.zeros((3, 1)), boundary, output_scattered=False
        )


def test_welm_token_owner_plan_contracts_boundary_residual_in_full_domain():
    runtime, plan, _ = make_token_owner_plan_for_test(
        transition=token_owner.TokenOwnerTransition.CONTRACT_EXIT_OWNER,
        pre_rows=(8,),
        post_rows=(4,),
        contract_flags=(True,),
        last_q_indices=(1, 3, 7),
        active_batch_indices=(0, 1, 2),
    )
    boundary = plan.state_for(token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY)
    residual = torch.arange(8, dtype=torch.float32).view(8, 1)

    contracted = runtime.contract_boundary_residual(
        residual, boundary, output_scattered=False
    )

    assert contracted.tolist() == [[1.0], [3.0], [7.0], [0.0]]


def test_welm_token_owner_plan_publishes_boundary_transport_once_from_plan(
    monkeypatch,
):
    buffer_updates = []
    extend_updates = []
    monkeypatch.setattr(
        token_owner,
        "set_dp_buffer_len",
        lambda *args: buffer_updates.append(args),
    )
    monkeypatch.setattr(
        token_owner,
        "set_is_extend_in_batch",
        lambda enabled: extend_updates.append(enabled),
    )
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=0),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=[8, 4],
        original_global_num_tokens_cpu=[8, 4],
        global_num_tokens_for_logprob_cpu=[8, 4],
        global_num_tokens_gpu=torch.tensor([8, 4]),
        global_num_tokens_for_logprob_gpu=torch.tensor([8, 4]),
        global_num_reqs_cpu=[3, 4],
        welm_kv_mirror_contract_flags=[True, False],
        welm_kv_mirror_last_q_indices_cpu=[1, 3, 7],
        welm_kv_mirror_active_batch_indices_cpu=[0, 1, 2],
        welm_kv_mirror_output_size=3,
        kv_mirror_output_size=3,
        _welm_kv_mirror_row_pad=0,
        dp_padding_mode=welmv4.DpPaddingMode.MAX_LEN,
        global_dp_buffer_len=8,
        dp_local_start_pos=torch.tensor(0),
        dp_local_num_tokens=torch.tensor(8),
        num_token_non_padded=torch.tensor(8, dtype=torch.int32),
        num_token_non_padded_cpu=4,
        is_extend_in_batch=True,
        _welm_force_low_latency_deepep=False,
    )
    plan = runtime.begin_forward(
        forward_batch,
        single_group_physical_rows=None,
        transition=token_owner.TokenOwnerTransition.CONTRACT_EXIT_OWNER,
        device=torch.device("cpu"),
    )
    boundary = plan.state_for(token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY)
    original_non_padded = forward_batch.num_token_non_padded

    runtime.publish_boundary_transport(forward_batch, boundary)

    assert forward_batch.global_num_tokens_cpu == [3, 4]
    assert forward_batch.global_num_tokens_gpu.tolist() == [3, 4]
    assert forward_batch.global_num_tokens_for_logprob_cpu == [3, 4]
    assert forward_batch.global_num_tokens_for_logprob_gpu.tolist() == [3, 4]
    assert forward_batch.global_dp_buffer_len == 7
    assert forward_batch.dp_local_start_pos is None
    assert forward_batch.dp_local_num_tokens is None
    assert forward_batch.num_token_non_padded.item() == 2
    assert forward_batch.num_token_non_padded is not original_non_padded
    assert original_non_padded.item() == 8
    assert forward_batch.num_token_non_padded_cpu == 2
    assert buffer_updates == [(7, 3, False, [3, 4])]
    assert extend_updates == [True]


@pytest.mark.parametrize(
    ("forward_batch", "transition", "match"),
    [
        (SimpleNamespace(), "NONE", "CPU global token counts"),
        (
            SimpleNamespace(global_num_tokens_cpu=[-1]),
            "NONE",
            "invalid global token counts",
        ),
        (
            SimpleNamespace(
                global_num_tokens_cpu=[8],
                original_global_num_tokens_cpu=[8],
                global_num_reqs_cpu=[3],
                welm_kv_mirror_contract_flags=[True, False],
                welm_kv_mirror_last_q_indices_cpu=[1, 3, 7],
                welm_kv_mirror_active_batch_indices_cpu=[0, 1, 2],
            ),
            "CONTRACT_KEEP_OWNER",
            "one flag per owner group",
        ),
        (
            SimpleNamespace(
                global_num_tokens_cpu=[8],
                original_global_num_tokens_cpu=[8],
                global_num_reqs_cpu=[3],
                welm_kv_mirror_contract_flags=[True],
                welm_kv_mirror_last_q_indices_cpu=[1, 8],
                welm_kv_mirror_active_batch_indices_cpu=[0, 1],
            ),
            "CONTRACT_KEEP_OWNER",
            "outside the original rows",
        ),
    ],
)
def test_welm_token_owner_forward_plan_rejects_invalid_cpu_metadata(
    forward_batch, transition, match
):
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
        global_tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
    )

    with pytest.raises((RuntimeError, ValueError), match=match):
        runtime.begin_forward(
            forward_batch,
            single_group_physical_rows=None,
            transition=getattr(token_owner.TokenOwnerTransition, transition),
            device=torch.device("cpu"),
        )


def test_welm_token_owner_forward_plan_rejects_different_forward_batch():
    runtime, _, _ = make_token_owner_plan_for_test(
        transition=token_owner.TokenOwnerTransition.NONE,
        pre_rows=(4,),
        post_rows=(4,),
        contract_flags=(False,),
        last_q_indices=(),
        active_batch_indices=(),
    )

    with pytest.raises(RuntimeError, match="different forward batch"):
        runtime.state_for(
            SimpleNamespace(), token_owner.TokenOwnerLayerIdentity.PRE_MIRROR
        )


@pytest.mark.parametrize(
    (
        "local_start",
        "local_end",
        "execution_end",
        "mirror_targets",
        "expected_boundary",
        "expected",
    ),
    [
        (
            0,
            48,
            48,
            (33, 37),
            33,
            (token_owner.TokenOwnerLayerIdentity.PRE_MIRROR,) * 33
            + (token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY,)
            + (token_owner.TokenOwnerLayerIdentity.MIRROR_TAIL,) * 14,
        ),
        (
            0,
            33,
            33,
            (33, 37),
            None,
            (token_owner.TokenOwnerLayerIdentity.PRE_MIRROR,) * 33,
        ),
        (
            34,
            48,
            48,
            (33, 37),
            33,
            (token_owner.TokenOwnerLayerIdentity.MIRROR_TAIL,) * 14,
        ),
        (
            4,
            12,
            12,
            (),
            None,
            (token_owner.TokenOwnerLayerIdentity.PRE_MIRROR,) * 8,
        ),
    ],
)
def test_welm_token_owner_layer_identity_is_static_for_execution_range(
    local_start,
    local_end,
    execution_end,
    mirror_targets,
    expected_boundary,
    expected,
):
    boundary, identities = token_owner.classify_token_owner_layers(
        local_start_layer=local_start,
        local_end_layer=local_end,
        execution_start_layer=0,
        execution_end_layer=execution_end,
        mirror_targets=mirror_targets,
    )

    assert boundary == expected_boundary
    assert identities == expected


@pytest.mark.parametrize(
    ("forward_batch", "kwargs", "expected"),
    [
        (
            SimpleNamespace(
                forward_mode=welmv4.ForwardMode.EXTEND,
                extend_num_tokens=17,
                batch_size=3,
                global_num_tokens_cpu=None,
            ),
            {},
            17,
        ),
        (
            SimpleNamespace(
                forward_mode=welmv4.ForwardMode.DECODE,
                batch_size=5,
                global_num_tokens_cpu=None,
            ),
            {},
            5,
        ),
        (
            SimpleNamespace(
                forward_mode=welmv4.ForwardMode.TARGET_VERIFY,
                batch_size=3,
                spec_info=SimpleNamespace(num_tokens_per_req=4),
                global_num_tokens_cpu=None,
            ),
            {},
            12,
        ),
        (
            SimpleNamespace(
                forward_mode=welmv4.ForwardMode.DECODE,
                batch_size=5,
                global_num_tokens_cpu=[16],
            ),
            {},
            16,
        ),
        (
            SimpleNamespace(
                forward_mode=welmv4.ForwardMode.DRAFT_EXTEND,
                welm_kv_mirror_output_size=5,
            ),
            {"post_pruning": True, "row_alignment": 4},
            8,
        ),
    ],
)
def test_welm_token_owner_physical_rows_uses_explicit_cpu_metadata(
    forward_batch, kwargs, expected
):
    assert welmv4._welm_token_owner_physical_rows(forward_batch, **kwargs) == expected


def test_welm_token_owner_physical_rows_does_not_depend_on_hidden_shape():
    forward_batch = SimpleNamespace(
        forward_mode=welmv4.ForwardMode.EXTEND,
        extend_num_tokens=9,
        batch_size=1,
        global_num_tokens_cpu=None,
    )

    small_hidden = torch.empty((1, 4))
    large_hidden = torch.empty((99, 4))

    assert small_hidden.shape != large_hidden.shape
    assert welmv4._welm_token_owner_physical_rows(forward_batch) == 9


@pytest.mark.parametrize(
    ("exit_owner", "expected"),
    [
        (False, token_owner.TokenOwnerTransition.CONTRACT_KEEP_OWNER),
        (True, token_owner.TokenOwnerTransition.CONTRACT_EXIT_OWNER),
    ],
)
def test_welm_token_owner_transition_uses_ordinary_prefill_policy(
    exit_owner, expected
):
    forward_batch = SimpleNamespace(
        forward_mode=welmv4.ForwardMode.EXTEND,
        enable_welm_kv_mirror_opt=True,
        return_logprob=False,
    )

    assert (
        token_owner.resolve_token_owner_transition(
            forward_batch,
            has_mirror_boundary=True,
            exit_owner=exit_owner,
        )
        is expected
    )


def test_welm_token_owner_transition_is_shared_by_mixed_dp_batch():
    forward_batch = SimpleNamespace(
        forward_mode=welmv4.ForwardMode.IDLE,
        enable_welm_kv_mirror_opt=True,
        global_forward_modes=[
            welmv4.ForwardMode.EXTEND.value,
            welmv4.ForwardMode.IDLE.value,
        ],
        welm_kv_mirror_contract_flags=[True, False],
    )

    assert (
        token_owner.resolve_token_owner_transition(
            forward_batch,
            has_mirror_boundary=True,
            exit_owner=True,
        )
        is token_owner.TokenOwnerTransition.CONTRACT_EXIT_OWNER
    )


def test_welm_pure_tp_forward_plan_uses_scheduler_output_without_dp_counts():
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
        global_tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_reqs_cpu=None,
        welm_kv_mirror_contract_flags=None,
        welm_kv_mirror_last_q_indices_cpu=[1, 7],
        welm_kv_mirror_active_batch_indices_cpu=[0, 1],
        welm_kv_mirror_output_size=2,
        dp_padding_mode=None,
    )

    plan = runtime.begin_forward(
        forward_batch,
        single_group_physical_rows=8,
        transition=token_owner.TokenOwnerTransition.CONTRACT_KEEP_OWNER,
        device=torch.device("cpu"),
    )

    assert not plan.uses_dp_transport
    assert plan.post_global_num_tokens_cpu == (2,)
    assert not hasattr(forward_batch, "original_global_num_tokens_cpu")
    assert forward_batch.global_num_tokens_cpu is None
    assert forward_batch.global_num_reqs_cpu is None


def test_welm_forward_plan_supports_alignment_padding_beyond_input_rows(monkeypatch):
    monkeypatch.setattr(
        token_owner,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_deepep=lambda: True),
    )
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=4, rank_in_group=0),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=0),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_reqs_cpu=None,
        welm_kv_mirror_contract_flags=None,
        welm_kv_mirror_last_q_indices_cpu=[0],
        welm_kv_mirror_active_batch_indices_cpu=[0],
        welm_kv_mirror_output_size=1,
        dp_padding_mode=None,
    )

    plan = runtime.begin_forward(
        forward_batch,
        single_group_physical_rows=1,
        transition=token_owner.TokenOwnerTransition.CONTRACT_EXIT_OWNER,
        device=torch.device("cpu"),
    )
    boundary = plan.state_for(token_owner.TokenOwnerLayerIdentity.MIRROR_BOUNDARY)
    tail = plan.state_for(token_owner.TokenOwnerLayerIdentity.MIRROR_TAIL)

    assert plan.post_global_num_tokens_cpu == (4,)
    assert (
        plan.post_global_num_tokens_cpu[0]
        - forward_batch.welm_kv_mirror_output_size
        == 3
    )
    assert boundary.transport_local_layout.owner_sizes == (1, 1, 1, 1)
    assert tail.transport_local_layout is boundary.transport_local_layout
    assert runtime.contract_boundary_residual(
        torch.tensor([[7.0]]), boundary, output_scattered=False
    ).tolist() == [[7.0], [0.0], [0.0], [0.0]]


def test_welm_forward_plan_rejects_more_survivors_than_input_rows(monkeypatch):
    monkeypatch.setattr(
        token_owner,
        "get_moe_a2a_backend",
        lambda: SimpleNamespace(is_deepep=lambda: True),
    )
    runtime = token_owner.WeLMTokenOwnerRuntime(
        attn_tp_group=SimpleNamespace(world_size=4, rank_in_group=0),
        global_tp_group=SimpleNamespace(world_size=4, rank_in_group=0),
    )
    forward_batch = SimpleNamespace(
        global_num_tokens_cpu=None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_reqs_cpu=None,
        welm_kv_mirror_contract_flags=None,
        welm_kv_mirror_last_q_indices_cpu=[0, 0],
        welm_kv_mirror_active_batch_indices_cpu=[0, 1],
        welm_kv_mirror_output_size=2,
        dp_padding_mode=None,
    )

    with pytest.raises(RuntimeError, match="survivor count exceeds"):
        runtime.begin_forward(
            forward_batch,
            single_group_physical_rows=1,
            transition=token_owner.TokenOwnerTransition.CONTRACT_EXIT_OWNER,
            device=torch.device("cpu"),
        )


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
    monkeypatch.setattr(
        welmv4, "_welm_kv_mirror_row_alignment", lambda _forward_batch=None: 1
    )
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


def test_welm_mtp_pure_tp_graph_uses_local_request_count():
    runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
        WelmMTPDraftProposalCudaGraphRunner
    )
    runner.use_token_owner = True
    runner.require_mlp_tp_gather = False
    runner.dp_size = 1
    forward_batch = SimpleNamespace(
        batch_size=7,
        global_num_reqs_cpu=None,
        global_num_tokens_cpu=None,
    )

    assert runner._get_dp_cuda_graph_request_bs(forward_batch) == 7


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
    runner.graphs_by_mode = {(True, False): {}}
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
        ({2: object()}, 2, True, "valid EagleDraftInput"),
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
    runner.dp_size = 2
    runner.disable_padding = False
    runner.graphs_by_mode = {(False, False): graphs}
    runner.max_bs = max_bs
    runner.num_tokens_per_bs = 4
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
        spec_info=None,
    )

    with pytest.raises(RuntimeError, match=reason):
        runner.can_run(forward_batch)


@pytest.mark.parametrize(
    (
        "enabled",
        "topk",
        "has_fixed_params",
        "fixed_top_p",
        "supports_top_p",
        "expected_modes",
    ),
    [
        (False, 1, False, None, True, [(False, False)]),
        (True, 2, False, None, True, [(False, False)]),
        (True, 1, True, 0.8, True, [(True, True)]),
        (True, 1, True, 1.0, True, [(True, False)]),
        (True, 1, True, 0.8, False, [(True, False)]),
        (True, 1, True, None, True, [(True, False), (True, True)]),
        (True, 1, True, None, False, [(True, False)]),
        (True, 1, False, None, True, [(False, False), (True, False), (True, True)]),
        (True, 1, False, None, False, [(False, False), (True, False)]),
    ],
)
def test_welm_mtp_capture_sampling_modes_follow_policy(
    enabled,
    topk,
    has_fixed_params,
    fixed_top_p,
    supports_top_p,
    expected_modes,
):
    import sglang.srt.speculative.welmv4_mtp_draft_proposal_cuda_graph_runner as mtp_graph

    worker = SimpleNamespace(
        _is_welmv4_mtp_draft_sampling_enabled=lambda: enabled,
        _has_welmv4_mtp_fixed_draft_sampling_params=lambda: has_fixed_params,
        welmv4_mtp_draft_fixed_top_p=fixed_top_p,
    )

    modes = mtp_graph._compute_capture_sampling_modes(
        worker, topk=topk, supports_draft_top_p=supports_top_p
    )

    assert modes == expected_modes


def test_welm_mtp_graph_activation_switches_family():
    runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
        WelmMTPDraftProposalCudaGraphRunner
    )
    greedy_graphs, random_graphs = {1: object()}, {1: object()}
    greedy_outputs, random_outputs = {1: object()}, {1: object()}
    runner.graphs_by_mode = {
        (False, False): greedy_graphs,
        (True, False): random_graphs,
    }
    runner.output_buffers_by_mode = {
        (False, False): greedy_outputs,
        (True, False): random_outputs,
    }

    runner._activate_sampling_mode((True, False))
    assert runner.sample_draft and not runner.use_top_p
    assert runner.graphs is random_graphs
    assert runner.output_buffers is random_outputs

    runner._activate_sampling_mode((False, False))
    assert not runner.sample_draft and not runner.use_top_p
    assert runner.graphs is greedy_graphs
    assert runner.output_buffers is greedy_outputs


def test_welm_mtp_pure_tp_graph_ignores_dp_scheduler_permission():
    runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
        WelmMTPDraftProposalCudaGraphRunner
    )
    runner.use_token_owner = True
    runner.require_mlp_tp_gather = False
    runner.require_mlp_sync = True
    runner.dp_size = 1
    runner.use_dp_sampling_consensus = False
    runner.disable_padding = False
    runner.graphs_by_mode = {(False, False): {2: object()}}
    runner.max_bs = 2
    runner.capture_bs = [2]
    runner.num_tokens_per_bs = 4
    runner.sample_draft = False
    runner.use_top_p = False
    runner.eagle_worker = SimpleNamespace(
        _should_sample_welmv4_mtp_draft=lambda _batch: False
    )
    spec_info = EagleDraftInput(
        hidden_states=torch.zeros((8, 1)),
        num_accept_tokens_cpu=[4, 4],
        num_accept_tokens=torch.tensor([4, 4], dtype=torch.int32),
        num_correct_drafts=torch.tensor([3, 3], dtype=torch.int32),
    )
    forward_batch = SimpleNamespace(
        forward_mode=welmv4.ForwardMode.DRAFT_EXTEND,
        input_ids=torch.arange(8),
        out_cache_loc=torch.arange(8),
        batch_size=2,
        global_num_reqs_cpu=None,
        can_run_dp_cuda_graph=False,
        spec_info=spec_info,
    )

    assert runner.can_run(forward_batch)


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
