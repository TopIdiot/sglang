# Vendored from the k-dash kernel source package welm/v45_80a3_attention.
# The CUDA kernel.so is resolved at runtime by k_dash.get(); that repo is a
# k-dash Source Package, not a Python distribution, so the host planner is
# mirrored here. Keep edits in sync with the upstream repo.
"""Naming for compiled decode-attention executors.

``DecodeExecutor`` is the concrete CUDA specialization baked into one
``kernel.so``. The host planner still chooses split counts / stages at
runtime, but those must fit the compiled capacity of the selected executor.

Artifact filenames: ``decode_attn_w78_<executor>.so``.
"""

from __future__ import annotations

from enum import Enum


class DecodeExecutor(str, Enum):
    """Compiled decode specialization (k-dash ``executor`` / ``K_DASH_EXECUTOR``)."""

    # Capacity AOT profile: batch<=256, splits<=64, stages=4, merge_warps=8,
    # sinks on, no sliding window.
    DECODE_DEFAULT = "decode_default"
    # Same capacity with HasWindow=true for SWA layers (e.g. window_left=512).
    DECODE_WINDOW = "decode_window"


EXECUTOR_VALUES = tuple(member.value for member in DecodeExecutor)

# Compile-time capacities baked into decode AOT profiles (must match FFI).
DECODE_DEFAULT_NUM_SUB_TASKS = 256
DECODE_DEFAULT_MAX_SPLITS = 64
DECODE_DEFAULT_NUM_STAGES = 4
DECODE_DEFAULT_OUTPUT_MERGE_WARPS = 8


def resolve_decode_executor(*, has_window: bool) -> DecodeExecutor:
    if has_window:
        return DecodeExecutor.DECODE_WINDOW
    return DecodeExecutor.DECODE_DEFAULT
