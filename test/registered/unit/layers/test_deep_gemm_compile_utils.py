from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.deep_gemm_wrapper import compile_utils


def test_bf16_grouped_warmup_uses_positional_grouped_layout():
    calls = []

    def grouped_gemm(a, b, out, grouped_layout):
        calls.append((a, b, out, grouped_layout))

    executor = compile_utils._BF16GroupedContWarmupExecutor.__new__(
        compile_utils._BF16GroupedContWarmupExecutor
    )
    executor.a = torch.empty((4, 2))
    executor.b = torch.empty((2, 3, 2))
    executor.out = torch.empty((4, 3))
    executor.m_indices = torch.arange(4, dtype=torch.int32)

    with patch.object(
        compile_utils,
        "deep_gemm",
        SimpleNamespace(m_grouped_bf16_gemm_nt_contiguous=grouped_gemm),
    ):
        executor.execute(2)

    assert len(calls) == 1
    assert torch.equal(calls[0][3], executor.m_indices[:2])
