# Vendored from the k-dash kernel source package welm/v45_80a3_attention.
# The CUDA kernel.so is resolved at runtime by k_dash.get(); that repo is a
# k-dash Source Package, not a Python distribution, so the host planner is
# mirrored here. Keep edits in sync with the upstream repo.
"""Shared naming for compiled verify-attention executors.

Two layers (do not conflate them):

1. ``VerifySchedulePolicy`` — what the host planner / bench asks for
   (``auto`` may still choose among concrete executors at plan time).
2. ``VerifyExecutor`` — the concrete CUDA specialization baked into one
   ``kernel.so`` / AOT BuildSpec.

Artifact filenames use the executor value, e.g.
``verify_attn_w78_warp_mma_last_arriver.so``.
"""

from __future__ import annotations

from enum import Enum


class VerifySchedulePolicy(str, Enum):
    """Host-facing schedule policy passed to ``plan(..., schedule_policy=...)``."""

    AUTO = "auto"
    WARP_MMA = "warp_mma_independent_q"
    WGMMA = "wgmma_kv64_n24"
    WGMMA_COREV2 = "wgmma_kv64_n24_corev2"


class VerifyExecutor(str, Enum):
    """Compiled kernel specialization (k-dash ``executor`` arg / ``K_DASH_EXECUTOR``)."""

    AUTO_DUAL_TYPED = "auto_dual_typed"
    WARP_MMA_TYPED = "warp_mma_typed"
    WARP_MMA_LAST_ARRIVER = "warp_mma_last_arriver"
    WGMMA_TYPED = "wgmma_typed"
    WGMMA_PERSISTENT_TYPED = "wgmma_persistent_typed"
    WGMMA_COREV2 = "wgmma_corev2"


class VerifyPartialMerge(str, Enum):
    AUTO = "auto"
    TWO_KERNEL = "two_kernel"
    LAST_ARRIVER = "last_arriver"


# Backward-compatible aliases used inside the ported planner.
POLICY_AUTO = VerifySchedulePolicy.AUTO.value
POLICY_WARP_MMA = VerifySchedulePolicy.WARP_MMA.value
POLICY_WGMMA = VerifySchedulePolicy.WGMMA.value
POLICY_WGMMA_COREV2 = VerifySchedulePolicy.WGMMA_COREV2.value

EXECUTOR_VALUES = tuple(member.value for member in VerifyExecutor)


def resolve_verify_executor(
    schedule_policy: str,
    *,
    partial_merge_mode: str,
    use_wgmma_static_persistent: bool,
) -> VerifyExecutor:
    """Map a resolved schedule policy (+ merge/persistent knobs) to one executor.

    Call this *after* ``auto`` has been lowered to ``warp_mma_*`` / ``wgmma_*``.
    Passing ``auto`` here selects the dual-typed dynamic executor
    (``AUTO_DUAL_TYPED``), which is only for dynamic-runtime plans.
    """

    if schedule_policy == VerifySchedulePolicy.AUTO.value:
        return VerifyExecutor.AUTO_DUAL_TYPED
    if schedule_policy == VerifySchedulePolicy.WGMMA_COREV2.value:
        return VerifyExecutor.WGMMA_COREV2
    if schedule_policy == VerifySchedulePolicy.WARP_MMA.value:
        if partial_merge_mode == VerifyPartialMerge.LAST_ARRIVER.value:
            return VerifyExecutor.WARP_MMA_LAST_ARRIVER
        return VerifyExecutor.WARP_MMA_TYPED
    if schedule_policy == VerifySchedulePolicy.WGMMA.value:
        if use_wgmma_static_persistent:
            return VerifyExecutor.WGMMA_PERSISTENT_TYPED
        return VerifyExecutor.WGMMA_TYPED
    raise ValueError(f"unsupported verify schedule_policy={schedule_policy!r}")
