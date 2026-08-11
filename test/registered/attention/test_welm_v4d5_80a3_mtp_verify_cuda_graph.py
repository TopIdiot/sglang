from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.contract import ENV_NAME
from sglang.srt.layers.attention.welm_v4d5_80a3_mtp_verify.mk_runner import (
    MkVerifyAttentionRunner,
)
from sglang.test.ci.ci_register import register_cuda_ci

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

register_cuda_ci(est_time=120, stage="stage-b", runner_config="1-gpu-small")


def test_mk_startup_self_check_matches_fa3(monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("WeLM V4D5 verify kernel requires SM90")
    monkeypatch.setenv(ENV_NAME, "mk")
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                model_type="welmv4_moe",
                architectures=["WeLMV4MoeForCausalLM"],
            )
        )
    )
    mk_runner = MkVerifyAttentionRunner(runner)
    assert mk_runner.ensure_self_check(torch.device("cuda"))
    assert len(mk_runner.self_check_metrics) == 2
    assert all(item.mismatches == 0 for item in mk_runner.self_check_metrics)
