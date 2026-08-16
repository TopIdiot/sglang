# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Dependency-free gate for the tuned WeLM H2048/HD256 Pre-Attn V2."""

import os

WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_ENV = "SGLANG_WELM_V45_80A3_FUSED_PRE_ATTN"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def welm_v45_80a3_h2048_hd256_pre_attn_v2_enabled() -> bool:
    return (
        os.getenv(WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_ENV, "0").strip().lower()
        in _TRUE_VALUES
    )
