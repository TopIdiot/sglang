from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import (
    ENV_NAME,
    VerifyAttentionBackendChoice,
    get_backend_choice,
    validate_static_contract,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def maybe_wrap_welm_v4d5_80a3_mtp_verify_backend(
    model_runner: ModelRunner,
    fallback_backend: AttentionBackend,
) -> AttentionBackend:
    """Install the model-specific MK verify wrapper when explicitly selected."""

    choice = get_backend_choice()
    if choice is VerifyAttentionBackendChoice.FA3:
        return fallback_backend

    # NEXTN uses the same checkpoint for its draft worker.  Draft attention is
    # outside this backend's scope and is ordinary delegation, not a fallback.
    if getattr(model_runner, "is_draft_worker", False):
        return fallback_backend

    decision = validate_static_contract(model_runner, fallback_backend)
    if not decision.supported:
        logger.warning(
            "WeLM V4D5 80A3 MTP verify attention fallback: "
            "requested_backend=mk selected_backend=fa3 reason=%s actual=%r "
            "expected=%r detail=%r tp_rank=%s env=%s",
            decision.reason.value,
            decision.actual,
            decision.expected,
            decision.detail,
            getattr(model_runner, "tp_rank", None),
            ENV_NAME,
        )
        return fallback_backend

    from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.backend import (
        WeLMV4D5MTPVerifyAttentionBackend,
    )

    logger.info(
        "Selected WeLM V4D5 80A3 MTP verify attention backend: "
        "requested_backend=mk fallback_backend=fa3 tp_rank=%s",
        getattr(model_runner, "tp_rank", None),
    )
    return WeLMV4D5MTPVerifyAttentionBackend(model_runner, fallback_backend)


__all__ = [
    "ENV_NAME",
    "maybe_wrap_welm_v4d5_80a3_mtp_verify_backend",
]
