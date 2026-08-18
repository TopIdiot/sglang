import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify import (
    maybe_wrap_welm_v4d5_80a3_mtp_verify_backend,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.backend import (
    WeLMV4D5MTPVerifyAttentionBackend,
)
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    ENV_NAME,
    FallbackReason,
    SupportDecision,
)


def test_default_fa3_returns_original_backend(monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    fallback = object()
    assert (
        maybe_wrap_welm_v4d5_80a3_mtp_verify_backend(SimpleNamespace(), fallback)
        is fallback
    )


def test_mk_request_on_wrong_model_logs_explicit_fallback(monkeypatch, caplog):
    monkeypatch.setenv(ENV_NAME, "mk")
    runner = SimpleNamespace(
        is_draft_worker=False,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="other")),
        tp_rank=2,
    )
    fallback = object()
    with caplog.at_level(logging.WARNING):
        result = maybe_wrap_welm_v4d5_80a3_mtp_verify_backend(runner, fallback)
    assert result is fallback
    assert "requested_backend=mk selected_backend=fa3" in caplog.text
    assert "model_signature_mismatch" in caplog.text
    assert "tp_rank=2" in caplog.text


def test_fallback_log_is_deduplicated_but_counted(caplog):
    backend = object.__new__(WeLMV4D5MTPVerifyAttentionBackend)
    backend.model_runner = SimpleNamespace(tp_rank=1)
    backend._logged_fallbacks = set()
    backend._fallback_counts = {}
    decision = SupportDecision.reject(FallbackReason.PAGE_SIZE, actual=64, expected=16)

    with caplog.at_level(logging.WARNING):
        backend._log_fallback(decision)
        backend._log_fallback(decision)

    assert caplog.text.count("unsupported_page_size") == 1
    assert backend._fallback_counts[FallbackReason.PAGE_SIZE.value] == 2


def test_mk_grows_and_reuses_fa3_target_verify_graph_buffers():
    owner = SimpleNamespace(
        target_verify_metadata={
            "cache_seqlens": torch.zeros(2, dtype=torch.int32),
            "page_table": torch.zeros((2, 4), dtype=torch.int32),
            "swa_page_table": torch.zeros((2, 4), dtype=torch.int32),
            "strided_indices": torch.arange(0, 64, 16),
        },
        decode_cuda_graph_metadata={
            "strided_indices": torch.arange(0, 64, 16),
        },
    )
    backend = object.__new__(WeLMV4D5MTPVerifyAttentionBackend)
    backend.fallback = SimpleNamespace()
    backend.model_runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(
            translate_loc_from_full_to_swa=lambda locations: locations
        ),
        tp_rank=0,
    )
    backend._fallback_target_backend = lambda: owner

    assert backend._ensure_fallback_target_graph_capacity(5)
    buffers = backend._fallback_graph_buffers(5)

    assert buffers is not None
    assert buffers.reuses_fallback
    assert buffers.max_pages == 5
    assert buffers.full_page_indices is owner.target_verify_metadata["page_table"]
    assert buffers.swa_page_indices is owner.target_verify_metadata["swa_page_table"]
    assert owner.target_verify_metadata["strided_indices"].tolist() == [
        0,
        16,
        32,
        48,
        64,
    ]
    assert owner.decode_cuda_graph_metadata["strided_indices"].tolist() == [
        0,
        16,
        32,
        48,
        64,
    ]


def test_mk_q_only_reads_prepopulated_cache_without_storing(monkeypatch):
    backend = object.__new__(WeLMV4D5MTPVerifyAttentionBackend)
    backend.model_runner = SimpleNamespace(attn_cp_size=1, tp_rank=0)
    backend.fallback = SimpleNamespace(has_local_attention=False)
    backend._mk_disabled_decision = None
    backend._forward_batch = None
    backend._logged_fallbacks = set()
    backend._fallback_counts = {}
    backend._logged_mk_use = False
    backend._graph_buffers = SimpleNamespace(
        full_page_indices=torch.zeros((1, 1), dtype=torch.int32),
        swa_page_indices=None,
        cache_seqlens=torch.ones(1, dtype=torch.int32),
        max_pages=1,
    )
    expected = torch.ones((1, 1, 256), dtype=torch.bfloat16)
    backend.engine = SimpleNamespace(
        begin_forward=lambda *_args, **_kwargs: None,
        _is_supported=lambda *_args, **_kwargs: True,
        try_run_cuda_graph=lambda *_args, **_kwargs: expected,
    )
    backend._store_kv = MagicMock()
    monkeypatch.setattr(
        "sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.backend.validate_runtime_contract",
        lambda **_kwargs: SupportDecision.allow(),
    )

    key_cache = torch.zeros((16, 1, 256), dtype=torch.bfloat16)
    pool = SimpleNamespace(
        layers_mapping={},
        get_kv_buffer=lambda _layer_id: (key_cache, key_cache.clone()),
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=256,
        v_head_dim=256,
        sliding_window_size=-1,
        is_cross_attention=False,
        welm_mirror_kv_cache_ready=True,
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True),
        token_to_kv_pool=pool,
        batch_size=1,
    )

    result, kv_stored = backend._try_mk(
        torch.zeros((1, 256), dtype=torch.bfloat16),
        None,
        None,
        layer,
        forward_batch,
        False,
        None,
    )

    assert result is expected
    assert not kv_stored
    backend._store_kv.assert_not_called()


def test_mk_q_only_miss_is_not_allowed_to_fallback():
    backend = object.__new__(WeLMV4D5MTPVerifyAttentionBackend)
    backend._try_mk = MagicMock(return_value=(None, False))
    backend.fallback = MagicMock()
    layer = SimpleNamespace(welm_mirror_kv_cache_ready=True)

    with pytest.raises(RuntimeError, match="cache-ready Q-only"):
        backend.forward_extend(
            torch.empty((1, 256), dtype=torch.bfloat16),
            None,
            None,
            layer,
            SimpleNamespace(),
            save_kv_cache=False,
        )

    backend.fallback.forward_extend.assert_not_called()
