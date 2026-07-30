# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Dependency-free feature gate for WeLM v4.5 80A3 fused pre-attention."""

import os

WELM_V45_80A3_FUSED_PRE_ATTN_ENV = "SGLANG_WELM_V45_80A3_FUSED_PRE_ATTN"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def welm_v45_80a3_fused_pre_attn_enabled() -> bool:
    return (
        os.getenv(WELM_V45_80A3_FUSED_PRE_ATTN_ENV, "0").strip().lower()
        in _TRUE_VALUES
    )
