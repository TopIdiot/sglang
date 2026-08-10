"""CPU contracts for WeLM persistent-token prefill CP MoE layouts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import welmv4 as welmv4_model
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _Backend:
    def __init__(self, is_none: bool):
        self._is_none = is_none

    def is_none(self):
        return self._is_none


def test_shared_expert_stays_tp_sharded_without_ep_dispatch(monkeypatch):
    monkeypatch.setattr(
        welmv4_model, "get_moe_a2a_backend", lambda: _Backend(is_none=True)
    )

    assert welmv4_model._welm_shared_expert_parallel_kwargs() == {}


def test_shared_expert_is_replicated_for_ep_dispatch(monkeypatch):
    monkeypatch.setattr(
        welmv4_model, "get_moe_a2a_backend", lambda: _Backend(is_none=False)
    )

    assert welmv4_model._welm_shared_expert_parallel_kwargs() == {
        "tp_rank": 0,
        "tp_size": 1,
    }


def _fused_norm_eligible_attention():
    return SimpleNamespace(
        use_o_norm=True,
        o_norm_needs_attn_tp_reduce=False,
        attn_tp_size=2,
        o_proj_suffix_parallel_reduce=False,
        o_proj=SimpleNamespace(
            reduce_results=True,
            use_attention_tp_reduce=True,
        ),
    )


def test_decode_role_does_not_create_prefill_cp_fused_norm_runner(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            enable_prefill_context_parallel=True,
            attn_cp_mode="sharded-kv",
            disaggregation_mode="decode",
        ),
    )
    monkeypatch.setattr(
        welmv4_model, "get_prefill_cp_attntp_fused_norm_manager", factory
    )

    runner = welmv4_model._welm_create_prefill_cp_attntp2_fused_norm_runner(
        attention=_fused_norm_eligible_attention(),
        hidden_size=2048,
        residual_after_layernorm=True,
    )

    assert runner is None
    factory.assert_not_called()


def test_prefill_role_does_not_create_fused_norm_runner_when_disabled(
    monkeypatch,
):
    group = object()
    factory = MagicMock()
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            enable_prefill_context_parallel=True,
            attn_cp_mode="sharded-kv",
            disaggregation_mode="prefill",
            max_prefill_tokens=32768,
            attn_cp_size=4,
            page_size=16,
            _enable_attntp_fused_norm=False,
        ),
    )
    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: group)
    monkeypatch.setattr(
        welmv4_model, "get_prefill_cp_attntp_fused_norm_manager", factory
    )

    actual = welmv4_model._welm_create_prefill_cp_attntp2_fused_norm_runner(
        attention=_fused_norm_eligible_attention(),
        hidden_size=2048,
        residual_after_layernorm=True,
    )

    assert actual is None
    factory.assert_not_called()


def test_prefill_cp_fused_norm_workspace_covers_chunked_prefill(monkeypatch):
    group = object()
    runner = object()
    factory = MagicMock(return_value=runner)
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            enable_prefill_context_parallel=True,
            attn_cp_mode="sharded-kv",
            disaggregation_mode="prefill",
            max_prefill_tokens=16384,
            chunked_prefill_size=32768,
            attn_cp_size=2,
            page_size=16,
            _enable_attntp_fused_norm=True,
        ),
    )
    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: group)
    monkeypatch.setattr(
        welmv4_model, "get_prefill_cp_attntp_fused_norm_manager", factory
    )

    actual = welmv4_model._welm_create_prefill_cp_attntp2_fused_norm_runner(
        attention=_fused_norm_eligible_attention(),
        hidden_size=2048,
        residual_after_layernorm=True,
    )

    assert actual is runner
    factory.assert_called_once_with(
        group=group,
        max_global_tokens=32768,
        cp_size=2,
        page_size=16,
        hidden_size=2048,
    )


def test_prefill_role_uses_startup_attntp_fused_norm_snapshot(monkeypatch):
    group = object()
    runner = object()
    factory = MagicMock(return_value=runner)
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            enable_prefill_context_parallel=True,
            attn_cp_mode="sharded-kv",
            disaggregation_mode="prefill",
            max_prefill_tokens=32768,
            attn_cp_size=4,
            page_size=16,
            _enable_attntp_fused_norm=True,
        ),
    )
    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: group)
    monkeypatch.setattr(
        welmv4_model, "get_prefill_cp_attntp_fused_norm_manager", factory
    )

    actual = welmv4_model._welm_create_prefill_cp_attntp2_fused_norm_runner(
        attention=_fused_norm_eligible_attention(),
        hidden_size=2048,
        residual_after_layernorm=True,
    )

    assert actual is runner
    factory.assert_called_once_with(
        group=group,
        max_global_tokens=32768,
        cp_size=4,
        page_size=16,
        hidden_size=2048,
    )


@pytest.mark.parametrize(
    ("dp_enabled", "topology"),
    ((False, "tp"), (True, "dp")),
)
def test_tp_dp_fused_norm_managers_follow_pd_role(
    monkeypatch,
    dp_enabled,
    topology,
):
    group = SimpleNamespace(world_size=2)
    prefill_manager = object()
    factory = MagicMock(return_value=prefill_manager)
    attention = SimpleNamespace(
        use_o_norm=True,
        attn_tp_size=2,
        o_proj_suffix_parallel_reduce=False,
        o_norm_needs_attn_tp_reduce=dp_enabled,
        o_proj=SimpleNamespace(
            reduce_results=not dp_enabled,
            use_attention_tp_reduce=False,
        ),
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            _enable_attntp_fused_norm=True,
            _attntp_fused_norm_prefill_max_rows=4096,
            enable_prefill_context_parallel=False,
            attn_cp_mode="null",
            disaggregation_mode="prefill",
            max_prefill_tokens=32768,
            max_running_requests=128,
            cuda_graph_max_bs=256,
            speculative_num_draft_tokens=4,
        ),
    )
    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(
        welmv4_model, "is_dp_attention_enabled", lambda: dp_enabled
    )
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: group)
    monkeypatch.setattr(
        welmv4_model,
        "get_or_create_attntp_fused_norm_manager",
        factory,
    )

    managers = welmv4_model._welm_create_tp_dp_attntp_fused_norm_managers(
        attention=attention,
        hidden_size=2048,
        residual_after_layernorm=True,
    )

    assert managers == {"prefill": prefill_manager}
    factory.assert_called_once_with(
        group=group,
        phase="prefill",
        topology=topology,
        hidden_size=2048,
        max_rows=4096 if topology == "tp" else 32768,
    )


def test_tp_dp_fused_norm_non_disaggregated_role_registers_decode_capacity(
    monkeypatch,
):
    group = SimpleNamespace(world_size=4)
    managers = {"prefill": object(), "decode": object()}
    factory = MagicMock(side_effect=(managers["prefill"], managers["decode"]))
    attention = SimpleNamespace(
        use_o_norm=True,
        attn_tp_size=4,
        o_proj_suffix_parallel_reduce=False,
        o_norm_needs_attn_tp_reduce=False,
        o_proj=SimpleNamespace(
            reduce_results=True,
            use_attention_tp_reduce=False,
        ),
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            _enable_attntp_fused_norm=True,
            _attntp_fused_norm_prefill_max_rows=4096,
            enable_prefill_context_parallel=False,
            attn_cp_mode="null",
            disaggregation_mode="null",
            max_prefill_tokens=16384,
            max_running_requests=128,
            cuda_graph_max_bs=256,
            speculative_num_draft_tokens=4,
        ),
    )
    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(
        welmv4_model, "is_dp_attention_enabled", lambda: False
    )
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: group)
    monkeypatch.setattr(
        welmv4_model,
        "get_or_create_attntp_fused_norm_manager",
        factory,
    )

    actual = welmv4_model._welm_create_tp_dp_attntp_fused_norm_managers(
        attention=attention,
        hidden_size=4096,
        residual_after_layernorm=True,
    )

    assert actual == managers
    assert factory.call_args_list == [
        call(
            group=group,
            phase="prefill",
            topology="tp",
            hidden_size=4096,
            max_rows=4096,
        ),
        call(
            group=group,
            phase="decode",
            topology="tp",
            hidden_size=4096,
            max_rows=1024,
        ),
    ]


def test_decode_fused_norm_reserves_auto_request_upper_bound(monkeypatch):
    group = SimpleNamespace(world_size=2)
    decode_manager = object()
    factory = MagicMock(return_value=decode_manager)
    attention = SimpleNamespace(
        use_o_norm=True,
        attn_tp_size=2,
        o_proj_suffix_parallel_reduce=False,
        o_norm_needs_attn_tp_reduce=False,
        o_proj=SimpleNamespace(
            reduce_results=True,
            use_attention_tp_reduce=False,
        ),
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            _enable_attntp_fused_norm=True,
            enable_prefill_context_parallel=False,
            attn_cp_mode="null",
            disaggregation_mode="decode",
            max_prefill_tokens=16384,
            max_running_requests=None,
            cuda_graph_max_bs=128,
            speculative_num_draft_tokens=None,
            dp_size=1,
        ),
    )
    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(
        welmv4_model, "is_dp_attention_enabled", lambda: False
    )
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: group)
    monkeypatch.setattr(
        welmv4_model,
        "get_or_create_attntp_fused_norm_manager",
        factory,
    )

    actual = welmv4_model._welm_create_tp_dp_attntp_fused_norm_managers(
        attention=attention,
        hidden_size=2048,
        residual_after_layernorm=True,
    )

    assert actual == {"decode": decode_manager}
    factory.assert_called_once_with(
        group=group,
        phase="decode",
        topology="tp",
        hidden_size=2048,
        max_rows=4096,
    )


def test_dp_decode_fused_norm_uses_per_worker_request_capacity(monkeypatch):
    group = SimpleNamespace(world_size=2)
    decode_manager = object()
    factory = MagicMock(return_value=decode_manager)
    attention = SimpleNamespace(
        use_o_norm=True,
        attn_tp_size=2,
        o_proj_suffix_parallel_reduce=False,
        o_norm_needs_attn_tp_reduce=True,
        o_proj=SimpleNamespace(
            reduce_results=False,
            use_attention_tp_reduce=False,
        ),
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            _enable_attntp_fused_norm=True,
            enable_prefill_context_parallel=False,
            attn_cp_mode="null",
            disaggregation_mode="decode",
            max_prefill_tokens=16384,
            max_running_requests=128,
            cuda_graph_max_bs=32,
            speculative_num_draft_tokens=None,
            dp_size=2,
        ),
    )
    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(
        welmv4_model, "is_dp_attention_enabled", lambda: True
    )
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: group)
    monkeypatch.setattr(
        welmv4_model,
        "get_or_create_attntp_fused_norm_manager",
        factory,
    )

    actual = welmv4_model._welm_create_tp_dp_attntp_fused_norm_managers(
        attention=attention,
        hidden_size=2048,
        residual_after_layernorm=True,
    )

    assert actual == {"decode": decode_manager}
    factory.assert_called_once_with(
        group=group,
        phase="decode",
        topology="dp",
        hidden_size=2048,
        max_rows=64,
    )


@pytest.mark.parametrize(
    ("num_rows", "expected_manager"),
    ((4096, True), (4097, False)),
)
def test_tp_prefill_fused_norm_honors_max_rows(
    monkeypatch, num_rows, expected_manager
):
    manager = SimpleNamespace(
        phase="prefill",
        topology="tp",
        attn_tp_size=2,
        hidden_size=2048,
        max_rows=4096,
    )
    layer = SimpleNamespace(
        tp_dp_attntp_fused_norm_managers={"prefill": manager},
        self_attn=SimpleNamespace(attn_tp_size=2),
        hidden_size=2048,
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_num_tokens=num_rows,
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)

    actual = welmv4_model._welm_select_tp_dp_attntp_fused_norm_manager(
        layer,
        forward_batch,
        use_prefill_cp_communicator=False,
    )

    assert (actual is manager) is expected_manager


def test_tp_fused_norm_rejects_non_bf16_model_dtype(monkeypatch):
    group = SimpleNamespace(world_size=2)
    attention = _fused_norm_eligible_attention()
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            _enable_attntp_fused_norm=True,
            enable_prefill_context_parallel=False,
            attn_cp_mode="null",
            disaggregation_mode="decode",
            max_prefill_tokens=16384,
            max_running_requests=128,
            cuda_graph_max_bs=128,
            speculative_num_draft_tokens=1,
            speculative_eagle_topk=1,
            dp_size=1,
        ),
    )
    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: group)

    with pytest.raises(NotImplementedError, match="BF16"):
        welmv4_model._welm_create_tp_dp_attntp_fused_norm_managers(
            attention=attention,
            hidden_size=2048,
            residual_after_layernorm=True,
            model_dtype=torch.float16,
        )


def test_tp_prefill_fused_norm_capacity_fallback_warns_once(monkeypatch):
    manager = SimpleNamespace(
        phase="prefill",
        topology="tp",
        attn_tp_size=2,
        hidden_size=2048,
        max_rows=2048,
    )
    layer = SimpleNamespace(
        tp_dp_attntp_fused_norm_managers={"prefill": manager},
        self_attn=SimpleNamespace(attn_tp_size=2),
        hidden_size=2048,
    )
    warnings = []
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(
        welmv4_model.logger,
        "warning",
        lambda *args: warnings.append(args),
    )
    monkeypatch.setattr(
        welmv4_model,
        "_WELM_TP_FUSED_NORM_LONG_PREFILL_FALLBACK_WARNED",
        False,
        raising=False,
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_num_tokens=2049,
    )

    for _ in range(2):
        assert (
            welmv4_model._welm_select_tp_dp_attntp_fused_norm_manager(
                layer,
                forward_batch,
                use_prefill_cp_communicator=False,
            )
            is None
        )

    assert len(warnings) == 1
    assert warnings[0][1:] == (2049, 2048)


def test_dp_prefill_fused_norm_ignores_tp_max_rows(monkeypatch):
    manager = SimpleNamespace(
        phase="prefill",
        topology="dp",
        attn_tp_size=2,
        hidden_size=2048,
    )
    layer = SimpleNamespace(
        tp_dp_attntp_fused_norm_managers={"prefill": manager},
        self_attn=SimpleNamespace(attn_tp_size=2),
        hidden_size=2048,
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: True)

    actual = welmv4_model._welm_select_tp_dp_attntp_fused_norm_manager(
        layer,
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_num_tokens=32768,
        ),
        use_prefill_cp_communicator=False,
    )

    assert actual is manager


@pytest.mark.parametrize(
    "forward_mode",
    (
        ForwardMode.DECODE,
        ForwardMode.IDLE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.DRAFT_EXTEND,
        ForwardMode.DRAFT_EXTEND_V2,
    ),
)
def test_non_disaggregated_worker_selects_decode_manager_for_decode_workloads(
    monkeypatch,
    forward_mode,
):
    managers = {
        phase: SimpleNamespace(
            phase=phase,
            topology="tp",
            attn_tp_size=2,
            hidden_size=2048,
        )
        for phase in ("prefill", "decode")
    }
    layer = SimpleNamespace(
        tp_dp_attntp_fused_norm_managers=managers,
        self_attn=SimpleNamespace(attn_tp_size=2),
        hidden_size=2048,
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)

    actual = welmv4_model._welm_select_tp_dp_attntp_fused_norm_manager(
        layer,
        SimpleNamespace(forward_mode=forward_mode),
        use_prefill_cp_communicator=False,
    )

    assert actual is managers["decode"]


@pytest.mark.parametrize(
    "forward_mode",
    (
        ForwardMode.EXTEND,
        ForwardMode.MIXED,
        ForwardMode.SPLIT_PREFILL,
        ForwardMode.DLLM_EXTEND,
    ),
)
def test_non_disaggregated_worker_selects_prefill_manager_for_prefill_workloads(
    monkeypatch,
    forward_mode,
):
    managers = {
        phase: SimpleNamespace(
            phase=phase,
            topology="tp",
            attn_tp_size=2,
            hidden_size=2048,
            max_rows=4096,
        )
        for phase in ("prefill", "decode")
    }
    layer = SimpleNamespace(
        tp_dp_attntp_fused_norm_managers=managers,
        self_attn=SimpleNamespace(attn_tp_size=2),
        hidden_size=2048,
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)

    actual = welmv4_model._welm_select_tp_dp_attntp_fused_norm_manager(
        layer,
        SimpleNamespace(
            forward_mode=forward_mode,
            extend_num_tokens=1024,
        ),
        use_prefill_cp_communicator=False,
    )

    assert actual is managers["prefill"]


def test_single_prefill_manager_handles_idle_dp_rank(monkeypatch):
    manager = SimpleNamespace(
        phase="prefill",
        topology="dp",
        attn_tp_size=2,
        hidden_size=2048,
    )
    layer = SimpleNamespace(
        tp_dp_attntp_fused_norm_managers={"prefill": manager},
        self_attn=SimpleNamespace(attn_tp_size=2),
        hidden_size=2048,
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: True)

    actual = welmv4_model._welm_select_tp_dp_attntp_fused_norm_manager(
        layer,
        SimpleNamespace(forward_mode=ForwardMode.IDLE),
        use_prefill_cp_communicator=False,
    )

    assert actual is manager


def test_welm_mtp_variable_decode_extend_selects_decode_manager(monkeypatch):
    managers = {
        phase: SimpleNamespace(
            phase=phase,
            topology="tp",
            attn_tp_size=2,
            hidden_size=2048,
        )
        for phase in ("prefill", "decode")
    }
    layer = SimpleNamespace(
        tp_dp_attntp_fused_norm_managers=managers,
        self_attn=SimpleNamespace(attn_tp_size=2),
        hidden_size=2048,
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)

    actual = welmv4_model._welm_select_tp_dp_attntp_fused_norm_manager(
        layer,
        SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            welm_mtp_variable_decode_extend=True,
        ),
        use_prefill_cp_communicator=False,
    )

    assert actual is managers["decode"]


def test_tp_dp_fused_norm_rejects_speculative_topk_greater_than_one(monkeypatch):
    group = SimpleNamespace(world_size=2)
    attention = _fused_norm_eligible_attention()
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            _enable_attntp_fused_norm=True,
            enable_prefill_context_parallel=False,
            attn_cp_mode="null",
            disaggregation_mode="decode",
            max_prefill_tokens=16384,
            max_running_requests=128,
            cuda_graph_max_bs=128,
            speculative_num_draft_tokens=8,
            speculative_eagle_topk=2,
            dp_size=1,
        ),
    )
    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: group)
    monkeypatch.setattr(
        welmv4_model,
        "get_or_create_attntp_fused_norm_manager",
        MagicMock(),
    )

    with pytest.raises(NotImplementedError, match="topk=1"):
        welmv4_model._welm_create_tp_dp_attntp_fused_norm_managers(
            attention=attention,
            hidden_size=2048,
            residual_after_layernorm=True,
        )


def test_prefill_cp_fused_norm_rejects_speculative_topk_greater_than_one(
    monkeypatch,
):
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            _enable_attntp_fused_norm=True,
            enable_prefill_context_parallel=True,
            attn_cp_mode="sharded-kv",
            disaggregation_mode="prefill",
            max_prefill_tokens=32768,
            attn_cp_size=4,
            page_size=16,
            speculative_eagle_topk=2,
        ),
    )
    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: object())
    monkeypatch.setattr(
        welmv4_model,
        "get_prefill_cp_attntp_fused_norm_manager",
        MagicMock(),
    )

    with pytest.raises(NotImplementedError, match="topk=1"):
        welmv4_model._welm_create_prefill_cp_attntp2_fused_norm_runner(
            attention=_fused_norm_eligible_attention(),
            hidden_size=2048,
            residual_after_layernorm=True,
        )


def test_decode_fused_norm_capacity_covers_scale_seq_rows(monkeypatch):
    group = SimpleNamespace(world_size=2)
    manager = object()
    factory = MagicMock(return_value=manager)
    attention = _fused_norm_eligible_attention()
    attention.scale_seq_times = 3
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(
            _enable_attntp_fused_norm=True,
            enable_prefill_context_parallel=False,
            attn_cp_mode="null",
            disaggregation_mode="decode",
            max_prefill_tokens=16384,
            max_running_requests=8,
            cuda_graph_max_bs=8,
            speculative_num_draft_tokens=1,
            speculative_eagle_topk=1,
            dp_size=1,
        ),
    )
    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4_model, "get_attn_tp_group", lambda: group)
    monkeypatch.setattr(
        welmv4_model,
        "get_or_create_attntp_fused_norm_manager",
        factory,
    )

    actual = welmv4_model._welm_create_tp_dp_attntp_fused_norm_managers(
        attention=attention,
        hidden_size=2048,
        residual_after_layernorm=True,
    )

    assert actual == {"decode": manager}
    factory.assert_called_once_with(
        group=group,
        phase="decode",
        topology="tp",
        hidden_size=2048,
        max_rows=32,
    )


@pytest.mark.parametrize("ep_dispatch", [False, True])
def test_prefill_cp_ppln_keeps_communicator_fp32_residual(
    monkeypatch, ep_dispatch
):
    captured = {}
    fp32_residual = torch.tensor([[1.0039061]], dtype=torch.float32)

    class PrefillCommunicator:
        use_ep_dispatch = ep_dispatch

        def validate_mlp(self, _mlp):
            captured["validated"] = True

        def prepare_attn(
            self, hidden_states, residual, forward_batch, **kwargs
        ):
            captured["prepare_attn_kwargs"] = kwargs
            return hidden_states, fp32_residual

        def prepare_mlp(self, hidden_states, residual, forward_batch):
            captured["prepare_mlp_residual"] = residual
            return hidden_states, residual

        def has_active_mlp_tokens(self, forward_batch):
            return True

        def build_router_context(self, forward_batch):
            assert not ep_dispatch
            captured["router_context_batch"] = forward_batch
            return "router-context"

        def postprocess_layer(self, hidden_states, residual, forward_batch):
            return hidden_states, residual

    class FakeAttention:
        use_o_norm = False
        o_norm_needs_attn_tp_reduce = False
        kv_mirror_layer_idx = -1

        def __call__(self, **kwargs):
            return kwargs["hidden_states"]

    class FakeMLP:
        tp_size = 2

        def __call__(
            self,
            hidden_states,
            hidden_states_fp32,
            forward_batch,
            use_reduce_scatter,
            **kwargs,
        ):
            captured["mlp_hidden_states_fp32"] = hidden_states_fp32
            captured["mlp_kwargs"] = kwargs
            return hidden_states

    layer = SimpleNamespace(
        layer_communicator=object(),
        prefill_cp_communicator=PrefillCommunicator(),
        _prefill_cp_mlp_validated=False,
        hidden_size=1,
        ppln=True,
        config_layer_id=1,
        prenorm_layer_idx=[],
        self_attn=FakeAttention(),
        kv_mirror_layers=[],
        is_nextn=False,
        is_final_layer=False,
        layer_id=0,
        mlp=FakeMLP(),
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=object(),
        dp_padding_mode=None,
    )

    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(_enable_attntp_fused_norm=False),
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_suffix_parallel_enabled", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "_welm_needs_empty_dp_collectives",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_dispatch_attention", lambda *_args: True
    )
    monkeypatch.setattr(
        welmv4_model,
        "_welm_should_sync_kv_mirror_dp_metadata",
        lambda *_: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda *_: False
    )

    welmv4_model.Qwen2MoeDecoderLayer.forward(
        layer,
        positions=torch.arange(1),
        hidden_states=torch.ones((1, 1), dtype=torch.bfloat16),
        forward_batch=forward_batch,
        residual=None,
        kv_mirror_states={},
    )

    assert captured["validated"] is True
    assert captured["prepare_attn_kwargs"] == {
        "residual_after_layernorm": True
    }
    assert captured["prepare_mlp_residual"] is fp32_residual
    assert captured["mlp_hidden_states_fp32"] is None
    if ep_dispatch:
        assert "router_context_batch" not in captured
        assert captured["mlp_kwargs"]["router_context"] is None
    else:
        assert captured["router_context_batch"] is forward_batch
        assert captured["mlp_kwargs"]["router_context"] == "router-context"


def test_prefill_cp_ppln_routes_attntp2_partial_through_fused_norm(monkeypatch):
    captured = {}
    fp32_residual = torch.full((1, 2048), 1.0039061, dtype=torch.float32)
    fused_residual = torch.full((1, 2048), 2.0039062, dtype=torch.float32)

    class PrefillCommunicator:
        use_ep_dispatch = False

        def validate_mlp(self, _mlp):
            captured["validated"] = True

        def prepare_attn(
            self, hidden_states, residual, forward_batch, **kwargs
        ):
            return hidden_states, fp32_residual

        def prepare_mlp(self, *_args, **_kwargs):
            raise AssertionError("eligible AttnTP2 path used unfused prepare_mlp")

        def prepare_mlp_fused_attntp2(
            self, hidden_states, residual, forward_batch, *, o_norm
        ):
            captured["fused_args"] = (
                hidden_states,
                residual,
                forward_batch,
                o_norm,
            )
            return hidden_states, fused_residual

        def has_active_mlp_tokens(self, forward_batch):
            return True

        def build_router_context(self, forward_batch):
            return "router-context"

        def postprocess_layer(self, hidden_states, residual, forward_batch):
            return hidden_states, residual

    o_norm = SimpleNamespace(
        weight=torch.ones(2048, dtype=torch.bfloat16), eps=1e-5
    )

    class FakeAttention:
        use_o_norm = True
        o_norm_needs_attn_tp_reduce = False
        kv_mirror_layer_idx = -1
        attn_tp_size = 2
        o_proj_suffix_parallel_reduce = False
        o_proj = SimpleNamespace(
            reduce_results=True,
            use_attention_tp_reduce=True,
        )

        def __init__(self):
            self.o_norm = o_norm

        def __call__(self, **kwargs):
            captured["attention_kwargs"] = kwargs
            return kwargs["hidden_states"]

    class FakeMLP:
        tp_size = 2

        def __call__(
            self,
            hidden_states,
            hidden_states_fp32,
            forward_batch,
            use_reduce_scatter,
            **kwargs,
        ):
            captured["mlp_hidden_states_fp32"] = hidden_states_fp32
            return hidden_states

    communicator = PrefillCommunicator()
    layer = SimpleNamespace(
        layer_communicator=object(),
        prefill_cp_communicator=communicator,
        _prefill_cp_mlp_validated=False,
        hidden_size=2048,
        ppln=True,
        config_layer_id=1,
        prenorm_layer_idx=[],
        self_attn=FakeAttention(),
        kv_mirror_layers=[],
        is_nextn=False,
        is_final_layer=False,
        layer_id=0,
        mlp=FakeMLP(),
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=object(),
        dp_padding_mode=None,
    )

    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_server_args",
        lambda: SimpleNamespace(_enable_attntp_fused_norm=True),
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_suffix_parallel_enabled", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "_welm_needs_empty_dp_collectives",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_dispatch_attention", lambda *_args: True
    )
    monkeypatch.setattr(
        welmv4_model,
        "_welm_should_sync_kv_mirror_dp_metadata",
        lambda *_: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda *_: False
    )

    welmv4_model.Qwen2MoeDecoderLayer.forward(
        layer,
        positions=torch.arange(1),
        hidden_states=torch.ones((1, 2048), dtype=torch.bfloat16),
        forward_batch=forward_batch,
        residual=None,
        kv_mirror_states={},
    )

    assert captured["attention_kwargs"]["skip_o_norm"] is True
    assert captured["attention_kwargs"]["skip_o_proj_reduce"] is True
    assert captured["fused_args"][1] is fp32_residual
    assert captured["fused_args"][2] is forward_batch
    assert captured["fused_args"][3] is o_norm
    assert captured["mlp_hidden_states_fp32"] is None


def test_tp_ppln_routes_partial_through_autotuned_fused_norm(monkeypatch):
    captured = {}
    fused_residual = torch.full((1, 2048), 2.0, dtype=torch.float32)

    class InputNorm:
        weight = torch.ones(2048, dtype=torch.bfloat16)

        def __call__(self, hidden_states, residual, **kwargs):
            captured["input_norm_kwargs"] = kwargs
            return hidden_states, None, hidden_states.float()

    class Manager:
        phase = "prefill"
        topology = "tp"
        attn_tp_size = 2
        hidden_size = 2048
        max_rows = 4096

        def forward(
            self,
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            o_norm_eps,
            post_norm_eps,
            **kwargs,
        ):
            captured["fused_args"] = (
                partial,
                residual,
                o_norm_weight,
                post_norm_weight,
                o_norm_eps,
                post_norm_eps,
                kwargs,
            )
            return partial + 1, fused_residual, None

    class FakeAttention:
        use_o_norm = True
        o_norm_needs_attn_tp_reduce = False
        kv_mirror_layer_idx = -1
        attn_tp_size = 2
        o_proj_suffix_parallel_reduce = False
        o_proj = SimpleNamespace(
            reduce_results=True,
            use_attention_tp_reduce=False,
        )
        o_norm = SimpleNamespace(
            weight=torch.ones(2048, dtype=torch.bfloat16),
            eps=1e-5,
        )

        def __call__(self, **kwargs):
            captured["attention_kwargs"] = kwargs
            return kwargs["hidden_states"] + 2

    class FakeMLP:
        tp_size = 2

        def __call__(
            self,
            hidden_states,
            hidden_states_fp32,
            forward_batch,
            use_reduce_scatter,
            **kwargs,
        ):
            captured["mlp_args"] = (
                hidden_states,
                hidden_states_fp32,
                use_reduce_scatter,
            )
            return hidden_states

    layer = SimpleNamespace(
        layer_communicator=object(),
        prefill_cp_communicator=object(),
        tp_dp_attntp_fused_norm_managers={"prefill": Manager()},
        hidden_size=2048,
        ppln=True,
        config_layer_id=1,
        prenorm_layer_idx=[],
        input_layernorm=InputNorm(),
        post_attention_layernorm=SimpleNamespace(
            weight=torch.ones(2048, dtype=torch.bfloat16),
            eps=2e-5,
        ),
        self_attn=FakeAttention(),
        kv_mirror_layers=[],
        is_nextn=False,
        is_final_layer=False,
        layer_id=0,
        mlp=FakeMLP(),
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=None,
        dp_padding_mode=None,
        forward_mode=ForwardMode.EXTEND,
        extend_num_tokens=1,
    )

    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_suffix_parallel_enabled", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "mmq_style_norm_after_attn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("autotuned path called legacy fused norm")
        ),
    )
    monkeypatch.setattr(
        welmv4_model,
        "_welm_needs_empty_dp_collectives",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_dispatch_attention", lambda *_args: True
    )
    monkeypatch.setattr(
        welmv4_model,
        "_welm_should_sync_kv_mirror_dp_metadata",
        lambda *_: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda *_: False
    )

    welmv4_model.Qwen2MoeDecoderLayer.forward(
        layer,
        positions=torch.arange(1),
        hidden_states=torch.ones((1, 2048), dtype=torch.bfloat16),
        forward_batch=forward_batch,
        residual=None,
        kv_mirror_states={},
    )

    assert captured["attention_kwargs"]["skip_o_norm"] is True
    assert captured["attention_kwargs"]["skip_o_proj_reduce"] is True
    assert captured["fused_args"][6] == {"execution": "eager"}
    assert captured["fused_args"][1].dtype is torch.float32
    assert captured["mlp_args"][0][0, 0].item() == 4
    assert captured["mlp_args"][1] is None
    assert captured["mlp_args"][2] is False


def test_dp_ppln_routes_partial_through_autotuned_fused_norm(monkeypatch):
    captured = {}
    fused_hidden = torch.full((2, 2048), 4.0, dtype=torch.bfloat16)
    fused_residual = torch.full((2, 2048), 5.0, dtype=torch.float32)
    mlp_hidden = torch.full((4, 2048), 6.0, dtype=torch.bfloat16)

    class LayerCommunicator:
        def prepare_attn(self, hidden_states, residual, forward_batch):
            return torch.cat((hidden_states, hidden_states)), residual

        def prepare_mlp(self, *_args, **_kwargs):
            raise AssertionError("autotuned path called legacy prepare_mlp")

        def prepare_mlp_from_fused_attntp(
            self, hidden_states, residual, forward_batch
        ):
            captured["fused_mlp_args"] = (hidden_states, residual, forward_batch)
            return mlp_hidden, residual

        def should_use_reduce_scatter(self, forward_batch):
            return False

        def postprocess_layer(self, hidden_states, residual, forward_batch):
            captured["postprocess_args"] = (hidden_states, residual, forward_batch)
            return hidden_states, residual

    class Manager:
        phase = "decode"
        topology = "dp"
        attn_tp_size = 2
        hidden_size = 2048

        def forward(self, partial, residual, *_args, **kwargs):
            captured["fused_args"] = (partial, residual, kwargs)
            return fused_hidden, fused_residual, None

    class FakeAttention:
        use_o_norm = True
        o_norm_needs_attn_tp_reduce = True
        kv_mirror_layer_idx = -1
        attn_tp_size = 2
        o_proj_suffix_parallel_reduce = False
        o_proj = SimpleNamespace(
            reduce_results=False,
            use_attention_tp_reduce=False,
        )
        o_norm = SimpleNamespace(
            weight=torch.ones(2048, dtype=torch.bfloat16),
            eps=1e-5,
        )

        def __call__(self, **kwargs):
            captured["attention_kwargs"] = kwargs
            return kwargs["hidden_states"] + 2

    class FakeMLP:
        tp_size = 2

        def __call__(
            self,
            hidden_states,
            hidden_states_fp32,
            forward_batch,
            use_reduce_scatter,
            **kwargs,
        ):
            captured["mlp_args"] = (
                hidden_states,
                hidden_states_fp32,
                use_reduce_scatter,
            )
            return hidden_states

    communicator = LayerCommunicator()
    layer = SimpleNamespace(
        layer_communicator=communicator,
        prefill_cp_communicator=object(),
        tp_dp_attntp_fused_norm_managers={"decode": Manager()},
        layer_scatter_modes=SimpleNamespace(
            layer_input_mode=welmv4_model.ScatterMode.TP_ATTN_FULL,
            mlp_mode=welmv4_model.ScatterMode.FULL,
        ),
        hidden_size=2048,
        ppln=True,
        config_layer_id=1,
        prenorm_layer_idx=[],
        post_attention_layernorm=SimpleNamespace(
            weight=torch.ones(2048, dtype=torch.bfloat16),
            eps=2e-5,
        ),
        self_attn=FakeAttention(),
        kv_mirror_layers=[],
        is_nextn=False,
        is_final_layer=False,
        layer_id=0,
        mlp=FakeMLP(),
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=None,
        dp_padding_mode=None,
        forward_mode=ForwardMode.DECODE,
    )

    monkeypatch.setattr(
        welmv4_model, "welm_use_previous_precision", lambda: False
    )
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(welmv4_model, "is_suffix_parallel_enabled", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "_welm_needs_empty_dp_collectives",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_dispatch_attention", lambda *_args: True
    )
    monkeypatch.setattr(
        welmv4_model,
        "_welm_should_sync_kv_mirror_dp_metadata",
        lambda *_: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda *_: False
    )

    welmv4_model.Qwen2MoeDecoderLayer.forward(
        layer,
        positions=torch.arange(1),
        hidden_states=torch.ones((1, 2048), dtype=torch.bfloat16),
        forward_batch=forward_batch,
        residual=None,
        kv_mirror_states={},
    )

    assert captured["attention_kwargs"]["skip_o_norm"] is True
    assert captured["attention_kwargs"]["skip_o_proj_reduce"] is True
    assert captured["fused_args"][1].shape == (2, 2048)
    assert captured["fused_args"][1].dtype is torch.float32
    assert captured["fused_args"][2] == {"execution": "graph"}
    assert captured["fused_mlp_args"][0] is fused_hidden
    assert captured["fused_mlp_args"][1] is fused_residual
    assert captured["mlp_args"][0] is mlp_hidden
    assert captured["mlp_args"][1:] == (None, False)
    assert captured["postprocess_args"][1] is fused_residual


def test_prefill_cp_moe_routes_only_owner_rows_but_keeps_full_expert_input(
    monkeypatch,
):
    hidden = torch.arange(24, dtype=torch.bfloat16).view(6, 4)
    hidden_fp32 = hidden.float()
    local_hidden = hidden[2:4]
    full_weights = torch.arange(12, dtype=torch.float32).view(6, 2)
    full_ids = torch.arange(12, dtype=torch.int64).view(6, 2)
    captured = {}

    class RouterContext:
        def local_rows(self, tensor):
            captured.setdefault("local_rows", []).append(tensor)
            return tensor[2:4]

        def prepare_moe_inputs(self, expert_hidden, topk_output):
            captured["local_weights"] = topk_output.topk_weights
            captured["local_ids"] = topk_output.topk_ids
            return expert_hidden, StandardTopKOutput(
                topk_weights=full_weights,
                topk_ids=full_ids,
                router_logits=topk_output.router_logits.new_empty((6, 0)),
            )

    class SharedExpert:
        def __call__(self, tensor):
            captured["shared_hidden"] = tensor
            return torch.zeros_like(tensor)

    class TopK:
        def __call__(
            self, routing_hidden, router_logits, num_token_non_padded=None
        ):
            assert num_token_non_padded is None
            captured["topk_hidden"] = routing_hidden
            captured["router_logits"] = router_logits
            return StandardTopKOutput(
                topk_weights=torch.ones((2, 2), dtype=torch.float32),
                topk_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
                router_logits=router_logits,
            )

    class Experts:
        def __call__(self, tensor, topk_output):
            captured["expert_hidden"] = tensor
            captured["expert_topk"] = topk_output
            return tensor.clone()

    def router_linear(tensor, weight, *, use_mxfp8):
        assert not use_mxfp8
        captured["router_hidden"] = tensor
        return torch.arange(8, dtype=torch.float32).view(2, 4)

    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4_model, "mmq_style_router_linear", router_linear)
    monkeypatch.setattr(
        welmv4_model,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(recording=False),
    )
    monkeypatch.setattr(
        welmv4_model, "get_global_experts_capturer", lambda: None
    )
    block = SimpleNamespace(
        _mk_moe_router=None,
        layer_id=3,
        shared_expert=SharedExpert(),
        shared_expert_gate=None,
        gate=SimpleNamespace(weight=torch.empty((4, 4))),
        topk=TopK(),
        experts=Experts(),
        tp_size=1,
        router_score_func="sigmoid",
        use_mxfp8=False,
        use_mmq_router_linear_v2=False,
        use_previous_precision_router=False,
    )

    output = welmv4_model.Qwen2MoeSparseMoeBlock.forward(
        block,
        hidden,
        hidden_fp32,
        forward_batch=SimpleNamespace(),
        router_context=RouterContext(),
    )

    assert torch.equal(output, hidden)
    assert torch.equal(captured["shared_hidden"], hidden)
    assert torch.equal(captured["router_hidden"], local_hidden)
    assert torch.equal(captured["topk_hidden"], local_hidden)
    assert torch.equal(captured["expert_hidden"], hidden)
    assert captured["expert_topk"].topk_weights is full_weights
    assert captured["expert_topk"].topk_ids is full_ids
    assert captured["expert_topk"].router_logits.shape == (6, 0)


def test_prefill_cp_default_router_does_not_require_fp32_hidden(monkeypatch):
    hidden = torch.arange(24, dtype=torch.bfloat16).view(6, 4)
    full_weights = torch.ones((6, 2), dtype=torch.float32)
    full_ids = torch.zeros((6, 2), dtype=torch.int64)
    captured = {}

    class RouterContext:
        def local_rows(self, tensor):
            assert torch.equal(tensor, hidden)
            captured["local_rows_calls"] = captured.get("local_rows_calls", 0) + 1
            return tensor[2:4]

        def prepare_moe_inputs(self, expert_hidden, topk_output):
            return expert_hidden, StandardTopKOutput(
                topk_weights=full_weights,
                topk_ids=full_ids,
                router_logits=topk_output.router_logits.new_empty((6, 0)),
            )

    class TopK:
        def __call__(
            self, routing_hidden, router_logits, num_token_non_padded=None
        ):
            assert num_token_non_padded is None
            return StandardTopKOutput(
                topk_weights=torch.ones((2, 2), dtype=torch.float32),
                topk_ids=torch.zeros((2, 2), dtype=torch.int64),
                router_logits=router_logits,
            )

    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "mmq_style_router_linear",
        lambda tensor, _weight, *, use_mxfp8: torch.ones((tensor.shape[0], 4)),
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(recording=False),
    )
    monkeypatch.setattr(welmv4_model, "get_global_experts_capturer", lambda: None)
    block = SimpleNamespace(
        _mk_moe_router=None,
        layer_id=3,
        shared_expert=None,
        shared_expert_gate=None,
        gate=SimpleNamespace(weight=torch.empty((4, 4))),
        topk=TopK(),
        experts=lambda tensor, _topk: tensor,
        tp_size=1,
        router_score_func="sigmoid",
        use_mxfp8=False,
        use_mmq_router_linear_v2=False,
        use_previous_precision_router=False,
    )

    output = welmv4_model.Qwen2MoeSparseMoeBlock.forward(
        block,
        hidden,
        None,
        forward_batch=SimpleNamespace(),
        router_context=RouterContext(),
    )

    assert torch.equal(output, hidden)
    assert captured["local_rows_calls"] == 1


def test_prefill_cp_moe_rejects_nonstandard_local_topk(monkeypatch):
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)

    class RouterContext:
        @staticmethod
        def local_rows(tensor):
            return tensor[:1]

        @staticmethod
        def prepare_moe_inputs(_hidden_states, topk_output):
            if not isinstance(topk_output, StandardTopKOutput):
                raise RuntimeError(
                    "owner-local prefill CP routing requires Standard TopK output"
                )
            raise AssertionError("unexpected Standard TopK output")

    context = RouterContext()
    block = SimpleNamespace(
        _mk_moe_router=None,
        layer_id=0,
        shared_expert=None,
        shared_expert_gate=None,
        gate=SimpleNamespace(weight=torch.empty((4, 4))),
        topk=lambda *_args, **_kwargs: object(),
        experts=MagicMock(),
        tp_size=1,
        router_score_func="sigmoid",
        use_mxfp8=False,
        use_mmq_router_linear_v2=False,
        use_previous_precision_router=False,
    )
    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "mmq_style_router_linear",
        lambda tensor, _weight, *, use_mxfp8: torch.ones((tensor.shape[0], 4)),
    )
    monkeypatch.setattr(
        welmv4_model,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(recording=False),
    )
    monkeypatch.setattr(
        welmv4_model, "get_global_experts_capturer", lambda: None
    )

    with pytest.raises(RuntimeError, match="Standard TopK"):
        welmv4_model.Qwen2MoeSparseMoeBlock.forward(
            block,
            hidden,
            hidden.float(),
            forward_batch=SimpleNamespace(),
            router_context=context,
        )


@pytest.mark.parametrize(
    "backend",
    [
        welmv4_model.MoeRunnerBackend.TRITON,
        welmv4_model.MoeRunnerBackend.DEEP_GEMM,
    ],
)
def test_prefill_cp_local_router_accepts_metadata_only_moe_runners(backend):
    block = SimpleNamespace(
        experts=SimpleNamespace(
            quant_method=SimpleNamespace(
                runner=SimpleNamespace(runner_backend=backend)
            )
        )
    )

    welmv4_model.Qwen2MoeSparseMoeBlock.validate_local_router(block)


def test_prefill_cp_local_router_rejects_runner_that_consumes_logits():
    block = SimpleNamespace(
        experts=SimpleNamespace(
            quant_method=SimpleNamespace(
                runner=SimpleNamespace(
                    runner_backend=welmv4_model.MoeRunnerBackend.MARLIN
                )
            )
        )
    )

    with pytest.raises(NotImplementedError, match="got marlin"):
        welmv4_model.Qwen2MoeSparseMoeBlock.validate_local_router(
            block
        )


def test_non_dp_decode_does_not_select_layer_communicator_reduce_scatter(monkeypatch):
    communicator = SimpleNamespace(
        should_use_reduce_scatter=MagicMock(return_value=True)
    )
    captured = {}

    class FakeNorm:
        weight = torch.ones(1)

        def __call__(self, hidden_states, residual, **kwargs):
            if kwargs.get("clone_fp32_out"):
                return hidden_states, residual, hidden_states.float()
            return hidden_states, residual

    class FakeAttention:
        use_o_norm = False
        o_norm_needs_attn_tp_reduce = False
        kv_mirror_layer_idx = -1

        def __call__(self, **kwargs):
            return kwargs["hidden_states"]

    class FakeMLP:
        tp_size = 2

        def __call__(
            self,
            hidden_states,
            hidden_states_fp32,
            forward_batch,
            use_reduce_scatter,
            **kwargs,
        ):
            captured["use_reduce_scatter"] = use_reduce_scatter
            return hidden_states

    layer = SimpleNamespace(
        layer_communicator=communicator,
        prefill_cp_communicator=object(),
        _prefill_cp_mlp_validated=False,
        hidden_size=1,
        ppln=False,
        config_layer_id=0,
        prenorm_layer_idx=[],
        input_layernorm=FakeNorm(),
        post_attention_layernorm=FakeNorm(),
        self_attn=FakeAttention(),
        kv_mirror_layers=[],
        is_nextn=False,
        is_final_layer=False,
        layer_id=0,
        mlp=FakeMLP(),
    )
    forward_batch = SimpleNamespace(
        attn_cp_prefill_runtime_layout=None,
        dp_padding_mode=object(),
    )

    monkeypatch.setattr(welmv4_model, "welm_use_previous_precision", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4_model, "is_suffix_parallel_enabled", lambda: False)
    monkeypatch.setattr(
        welmv4_model,
        "_welm_needs_empty_dp_collectives",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_dispatch_attention", lambda *_args: True
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_sync_kv_mirror_dp_metadata", lambda *_: False
    )
    monkeypatch.setattr(
        welmv4_model, "_welm_should_contract_kv_mirror", lambda *_: False
    )

    welmv4_model.Qwen2MoeDecoderLayer.forward(
        layer,
        positions=torch.arange(1),
        hidden_states=torch.ones((1, 1)),
        forward_batch=forward_batch,
        residual=None,
        kv_mirror_states={},
    )

    assert captured["use_reduce_scatter"] is False
    communicator.should_use_reduce_scatter.assert_not_called()
