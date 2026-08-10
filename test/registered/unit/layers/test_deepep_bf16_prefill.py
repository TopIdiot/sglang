from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.token_dispatcher import deepep as deepep_dispatcher
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPNormalDispatchOutput,
    _DeepEPDispatcherImplNormal,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def test_deepep_normal_dispatch_honors_explicit_bf16_config(monkeypatch):
    dispatcher = object.__new__(_DeepEPDispatcherImplNormal)
    dispatcher.async_finish = False
    dispatcher.quant_config = {"bf16_dispatch": True}
    hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
    topk_output = SimpleNamespace(
        topk_ids=torch.zeros((2, 1), dtype=torch.int64),
        topk_weights=torch.ones((2, 1), dtype=torch.float32),
    )

    monkeypatch.setattr(
        deepep_dispatcher,
        "get_moe_runner_backend",
        lambda: MoeRunnerBackend.DEEP_GEMM,
    )
    monkeypatch.setattr(deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True)
    monkeypatch.setattr(
        deepep_dispatcher,
        "sglang_per_token_group_quant_fp8",
        lambda *_args, **_kwargs: pytest.fail("BF16 dispatch was quantized to FP8"),
        raising=False,
    )

    dispatched, topk_ids, topk_weights, previous_event = dispatcher.dispatch_a(
        hidden_states,
        topk_output,
    )

    assert dispatched is hidden_states
    assert topk_ids is topk_output.topk_ids
    assert topk_weights is topk_output.topk_weights
    assert previous_event is None


def test_unquantized_deepep_normal_uses_deep_gemm_runner():
    expected = object()
    captured = {}

    class QuantMethod:
        runner = SimpleNamespace(runner_backend=MoeRunnerBackend.DEEP_GEMM)

        def apply(self, *, layer, dispatch_output):
            captured["layer"] = layer
            captured["dispatch_output"] = dispatch_output
            return expected

    moe = object.__new__(DeepEPMoE)
    torch.nn.Module.__init__(moe)
    moe.deprecate_flag = False
    moe.quant_config = None
    moe.quant_method = QuantMethod()
    dispatch_output = DeepEPNormalDispatchOutput(
        hidden_states=torch.ones((1, 4), dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_ids=torch.zeros((1, 1), dtype=torch.int64),
        topk_weights=torch.ones((1, 1), dtype=torch.float32),
        num_recv_tokens_per_expert=[1],
    )

    result = moe.run_moe_core(dispatch_output)

    assert result is expected
    assert captured == {"layer": moe, "dispatch_output": dispatch_output}
