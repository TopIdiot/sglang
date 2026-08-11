from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    ENV_NAME,
    FallbackReason,
    VerifyAttentionBackendChoice,
    get_backend_choice,
    is_welm_v4d5_80a3,
    validate_static_contract,
)


class FlashAttentionBackend:
    pass


def _model_config(**overrides):
    values = {
        "model_type": "welmv4_moe",
        "architectures": ["WeLMV4MoeForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "num_nextn_predict_layers": 4,
    }
    values.update(overrides)
    return SimpleNamespace(hf_text_config=SimpleNamespace(**values))


def _runner(*, server_arg_overrides=None, **overrides):
    server_args = {
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
        "speculative_attention_mode": "prefill",
        "dp_size": 1,
        "enable_dp_attention": False,
    }
    server_args.update(server_arg_overrides or {})
    values = {
        "model_config": _model_config(),
        "is_draft_worker": False,
        "tp_size": 4,
        "attn_cp_size": 1,
        "page_size": 16,
        "kv_cache_dtype": torch.bfloat16,
        "server_args": SimpleNamespace(**server_args),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_backend_choice_defaults_to_fa3(monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    assert get_backend_choice() is VerifyAttentionBackendChoice.FA3

    monkeypatch.setenv(ENV_NAME, "mk")
    assert get_backend_choice() is VerifyAttentionBackendChoice.MK


def test_backend_choice_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "enabled")
    with pytest.raises(ValueError, match=ENV_NAME):
        get_backend_choice()


def test_exact_welm_v4d5_80a3_signature():
    assert is_welm_v4d5_80a3(_model_config())
    assert not is_welm_v4d5_80a3(_model_config(num_experts=256))


def test_static_contract_accepts_pinned_configuration():
    assert validate_static_contract(_runner(), FlashAttentionBackend()).supported


def test_static_contract_accepts_independent_dp_replicas():
    runner = _runner(server_arg_overrides={"dp_size": 2})
    assert validate_static_contract(runner, FlashAttentionBackend()).supported


@pytest.mark.parametrize(
    "runner,reason",
    [
        (
            _runner(model_config=_model_config(num_hidden_layers=47)),
            FallbackReason.MODEL_SIGNATURE,
        ),
        (_runner(tp_size=8), FallbackReason.TP_SIZE),
        (_runner(page_size=64), FallbackReason.PAGE_SIZE),
        (_runner(kv_cache_dtype=torch.float8_e4m3fn), FallbackReason.KV_DTYPE),
        (_runner(attn_cp_size=2), FallbackReason.ATTN_CP),
        (
            _runner(server_arg_overrides={"enable_dp_attention": True}),
            FallbackReason.DP_ATTENTION,
        ),
    ],
)
def test_static_contract_reports_precise_reason(runner, reason):
    decision = validate_static_contract(runner, FlashAttentionBackend())
    assert not decision.supported
    assert decision.reason is reason
