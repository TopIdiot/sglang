from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from sglang.srt.models.welm_deferred_mirror import (
    DeferredDecodeInputKind,
    WelmDeferredMirrorPair,
    WelmPDExecutionMode,
    build_welm_deferred_mirror_plan,
    resolve_welm_deferred_mirror_plan,
)


def _production_text_config(**overrides):
    values = {
        "num_hidden_layers": 48,
        "kv_mirror_layers": list(range(48, 32, -1)),
        "kv_mirror_imitated_layers": list(range(16)),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _model_config(**overrides):
    values = {
        "hf_config": SimpleNamespace(
            architectures=["WeLMV4MoeForCausalLM"],
        ),
        "hf_text_config": _production_text_config(),
        "is_multimodal": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _server_args(**overrides):
    values = {
        "welm_kv_mirror_pd_mode": "deferred-last-prompt",
        "enable_welm_kv_mirror_opt": True,
        "disaggregation_mode": "prefill",
        "disaggregation_transfer_backend": "mooncake",
        "attention_backend": "fa3",
        "prefill_attention_backend": None,
        "decode_attention_backend": None,
        "pp_size": 1,
        "speculative_algorithm": None,
        "speculative_eagle_topk": 1,
        "disable_overlap_schedule": False,
        "enable_hierarchical_cache": False,
        "disaggregation_decode_enable_offload_kvcache": False,
        "enable_dp_attention": False,
        "attn_cp_mode": "none",
        "attn_cp_size": 1,
        "enable_lora": False,
        "enable_suffix_parallel": False,
        "enable_mixed_chunk": False,
        "kv_cache_dtype": "auto",
        "tp_size": 4,
        "dp_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _mtp_model_config():
    return _model_config(
        hf_config=SimpleNamespace(
            architectures=["WeLMV4MoeForCausalLM"],
            num_nextn_predict_layers=1,
        )
    )


def test_execution_mode_and_input_kind_have_stable_wire_values():
    assert WelmPDExecutionMode.LEGACY.value == "legacy"
    assert WelmPDExecutionMode.DEFERRED_LAST_PROMPT.value == "deferred-last-prompt"
    assert DeferredDecodeInputKind.TOKEN_ID == 1
    assert DeferredDecodeInputKind.EMBEDDING == 2


def test_production_plan_excludes_nextn_and_locks_fingerprint():
    plan = build_welm_deferred_mirror_plan(_production_text_config())

    assert plan.num_hidden_layers == 48
    assert plan.execution_end_layer == 33
    assert plan.pairs == tuple(
        WelmDeferredMirrorPair(source_layer=source, target_layer=target)
        for source, target in zip(range(15, 0, -1), range(33, 48))
    )
    assert plan.fingerprint == (
        "12cae9b49a46e2bcb571da4d88773891b8b81e15789e16824287a6479a6cb1af"
    )


def test_plan_is_immutable_and_independent_of_config_pair_order():
    config = _production_text_config()
    reversed_config = _production_text_config(
        kv_mirror_layers=list(reversed(config.kv_mirror_layers)),
        kv_mirror_imitated_layers=list(
            reversed(config.kv_mirror_imitated_layers)
        ),
    )

    plan = build_welm_deferred_mirror_plan(config)
    assert build_welm_deferred_mirror_plan(reversed_config) == plan
    with pytest.raises(FrozenInstanceError):
        plan.execution_end_layer = 34


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "kv_mirror_layers": [33, 34],
                "kv_mirror_imitated_layers": [1],
            },
            "same length",
        ),
        (
            {
                "kv_mirror_layers": [33, 35],
                "kv_mirror_imitated_layers": [1, 2],
            },
            "contiguous suffix",
        ),
        (
            {
                "kv_mirror_layers": [33, 33, 34],
                "kv_mirror_imitated_layers": [1, 2, 3],
            },
            "duplicate target",
        ),
        (
            {
                "kv_mirror_layers": list(range(33, 48)),
                "kv_mirror_imitated_layers": [
                    *range(15, 1, -1),
                    33,
                ],
            },
            "before execution_end_layer",
        ),
        (
            {
                "kv_mirror_layers": [48],
                "kv_mirror_imitated_layers": [0],
            },
            "base mirror target",
        ),
    ],
)
def test_plan_rejects_malformed_base_suffix(overrides, message):
    with pytest.raises(ValueError, match=message):
        build_welm_deferred_mirror_plan(
            _production_text_config(**overrides)
        )


def test_legacy_mode_skips_deferred_validation_and_plan_construction():
    args = _server_args(
        welm_kv_mirror_pd_mode="legacy",
        enable_welm_kv_mirror_opt=False,
        disaggregation_mode="null",
        attention_backend="triton",
    )
    invalid_model = _model_config(
        hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"])
    )

    assert (
        resolve_welm_deferred_mirror_plan(
            args,
            invalid_model,
            use_previous_precision=True,
        )
        is None
    )


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_deferred_mode_resolves_same_rank_free_plan_for_both_roles(role):
    plan = resolve_welm_deferred_mirror_plan(
        _server_args(disaggregation_mode=role),
        _model_config(),
        use_previous_precision=False,
    )

    assert plan is not None
    assert plan.execution_end_layer == 33
    assert not hasattr(plan, "tp_rank")
    assert not hasattr(plan, "cp_rank")
    assert not hasattr(plan, "dp_rank")


def test_deferred_mode_resolves_monolithic_without_transfer_backend():
    plan = resolve_welm_deferred_mirror_plan(
        _server_args(
            disaggregation_mode="null",
            disaggregation_transfer_backend=None,
        ),
        _model_config(),
        use_previous_precision=False,
    )

    assert plan is not None
    assert plan.execution_end_layer == 33


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
def test_deferred_mode_allows_welm_mtp_spec_v2(role):
    plan = resolve_welm_deferred_mirror_plan(
        _server_args(
            disaggregation_mode=role,
            disaggregation_transfer_backend=(
                None if role == "null" else "mooncake"
            ),
            speculative_algorithm="EAGLE",
        ),
        _mtp_model_config(),
        use_previous_precision=False,
    )

    assert plan is not None
    assert plan.execution_end_layer == 33


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_deferred_pd_roles_still_require_mooncake(role):
    with pytest.raises(ValueError, match="mooncake"):
        resolve_welm_deferred_mirror_plan(
            _server_args(
                disaggregation_mode=role,
                disaggregation_transfer_backend="nixl",
            ),
            _model_config(),
            use_previous_precision=False,
        )


@pytest.mark.parametrize("role", ["null", "prefill", "decode"])
def test_deferred_mode_rejects_mixed_chunk_for_every_role(role):
    with pytest.raises(ValueError, match="mixed chunk"):
        resolve_welm_deferred_mirror_plan(
            _server_args(
                disaggregation_mode=role,
                enable_mixed_chunk=True,
            ),
            _model_config(),
            use_previous_precision=False,
        )


@pytest.mark.parametrize(
    ("args_overrides", "model_overrides", "previous_precision", "message"),
    [
        ({"welm_kv_mirror_pd_mode": "broken"}, {}, False, "unknown"),
        ({"enable_welm_kv_mirror_opt": False}, {}, False, "mirror-opt"),
        (
            {"disaggregation_transfer_backend": "nixl"},
            {},
            False,
            "mooncake",
        ),
        ({"attention_backend": "triton"}, {}, False, "FA3"),
        ({"pp_size": 2}, {}, False, "pipeline parallel"),
        ({"speculative_algorithm": "EAGLE"}, {}, False, "WeLM MTP"),
        (
            {
                "speculative_algorithm": "EAGLE",
                "speculative_eagle_topk": 2,
            },
            {"hf_config": _mtp_model_config().hf_config},
            False,
            "topk=1",
        ),
        (
            {
                "speculative_algorithm": "EAGLE",
                "disable_overlap_schedule": True,
            },
            {"hf_config": _mtp_model_config().hf_config},
            False,
            "overlap",
        ),
        (
            {
                "speculative_algorithm": "EAGLE",
                "attn_cp_size": 2,
            },
            {"hf_config": _mtp_model_config().hf_config},
            False,
            "AttnCP",
        ),
        ({"enable_hierarchical_cache": True}, {}, False, "HiCache"),
        (
            {"disaggregation_decode_enable_offload_kvcache": True},
            {},
            False,
            "KV offload",
        ),
        ({"enable_lora": True}, {}, False, "LoRA"),
        ({"enable_suffix_parallel": True}, {}, False, "suffix parallel"),
        (
            {},
            {
                "hf_config": SimpleNamespace(
                    architectures=["WeLMV4MoeForCausalLM"],
                    scale_seq_times=1,
                )
            },
            False,
            "Scale-Seq",
        ),
        (
            {},
            {
                "hf_text_config": _production_text_config(scale_seq_times=1),
            },
            False,
            "Scale-Seq",
        ),
        ({"kv_cache_dtype": "fp8_e4m3"}, {}, False, "KV cache dtype"),
        (
            {
                "disaggregation_mode": "decode",
                "enable_dp_attention": True,
                "attn_cp_mode": "sharded-kv",
                "attn_cp_size": 2,
            },
            {},
            False,
            "DP attention.*sharded-KV CP",
        ),
        (
            {},
            {
                "hf_config": SimpleNamespace(
                    architectures=["WeLMV4VLMForConditionalGeneration"]
                ),
                "is_multimodal": True,
            },
            False,
            "language-only WeLMV4",
        ),
        ({}, {}, True, "previous-precision"),
    ],
)
def test_deferred_mode_fails_fast_for_unsupported_runtime(
    args_overrides,
    model_overrides,
    previous_precision,
    message,
):
    with pytest.raises(ValueError, match=message):
        resolve_welm_deferred_mirror_plan(
            _server_args(**args_overrides),
            _model_config(**model_overrides),
            use_previous_precision=previous_precision,
        )
