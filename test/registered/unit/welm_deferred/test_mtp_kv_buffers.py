from dataclasses import FrozenInstanceError

import pytest
import torch

from sglang.srt.speculative.welmv4_mtp_kv import (
    build_welmv4_mtp_kv_mirror_buffer_spec,
    copy_welmv4_mtp_kv_mirror_states_to_buffers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from mtp_test_utils import model_config

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def test_mirror_buffer_spec_is_derived_from_target_config():
    spec = build_welmv4_mtp_kv_mirror_buffer_spec(
        model_config=model_config(),
        attention_tp_size=4,
        dtype=torch.bfloat16,
        device="cpu",
    )

    assert spec.mirror_layers == (48, 49)
    assert spec.tensor_size == 2 * 128
    assert spec.dtype is torch.bfloat16
    assert spec.device == torch.device("cpu")
    with pytest.raises(FrozenInstanceError):
        spec.tensor_size = 1


def test_mirror_copy_preserves_unwritten_rows_and_trims_source_padding():
    buffers = {
        "48.k": torch.full((5, 4), -1, dtype=torch.bfloat16),
        "48.v": torch.full((5, 4), -2, dtype=torch.bfloat16),
    }
    source_k = torch.arange(12, dtype=torch.bfloat16).view(3, 4)
    source_v = source_k + 20

    copy_welmv4_mtp_kv_mirror_states_to_buffers(
        buffers,
        {"welm_kv_mirror_states": {48: (source_k, source_v)}},
        rows=2,
    )

    assert torch.equal(buffers["48.k"][:2], source_k[:2])
    assert torch.equal(buffers["48.v"][:2], source_v[:2])
    assert torch.equal(
        buffers["48.k"][2:], torch.full((3, 4), -1, dtype=torch.bfloat16)
    )
    assert torch.equal(
        buffers["48.v"][2:], torch.full((3, 4), -2, dtype=torch.bfloat16)
    )
