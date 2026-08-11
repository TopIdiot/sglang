import logging
from types import SimpleNamespace

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
