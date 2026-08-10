"""Focused SM120 FlashAttention-4 regression tests."""

import math

import pytest
import torch

from sglang.jit_kernel.flash_attention_v4 import (
    flash_attn_varlen_func,
)
from sglang.jit_kernel.flash_attn.cute.flash_fwd import (
    FlashAttentionForwardSm80,
)
from sglang.jit_kernel.flash_attn.cute.flash_fwd_sm120 import (
    FlashAttentionForwardSm120,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=50,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-small",
)

if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)):
    pytest.skip(
        "SM120 FlashAttention-4 test requires CUDA SM 12.0.",
        allow_module_level=True,
    )


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    window_left: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP32 causal attention-with-sink output and LSE."""
    seq = q.shape[0]
    scale = q.shape[-1] ** -0.5
    scores = (
        torch.einsum(
            "qhd,kd->hqk",
            q.float(),
            k[:, 0].float(),
        )
        * scale
    )
    q_idx = torch.arange(seq, device=q.device)[:, None]
    kv_idx = torch.arange(seq, device=q.device)[None, :]
    mask = kv_idx <= q_idx
    if window_left is not None:
        mask &= kv_idx >= q_idx - window_left
    scores.masked_fill_(~mask[None], -torch.inf)

    row_max = torch.maximum(scores.amax(dim=-1), sinks.float()[:, None])
    weights = torch.exp(scores - row_max[:, :, None])
    denominator = weights.sum(dim=-1) + torch.exp(sinks.float()[:, None] - row_max)
    output = (
        torch.einsum(
            "hqk,kd->qhd",
            weights,
            v[:, 0].float(),
        )
        / denominator.T[:, :, None]
    )
    lse = torch.log(denominator) + row_max
    return output, lse


def test_sm120_owns_learnable_sink_capability():
    assert not FlashAttentionForwardSm80.supports_learnable_sink
    assert FlashAttentionForwardSm120.supports_learnable_sink


@pytest.mark.parametrize(
    ("head_dim", "expected"),
    [
        (64, (64, 64)),
        (128, (64, 64)),
        (192, (128, 64)),
        (224, (64, 64)),
        (256, (64, 64)),
    ],
)
def test_sm120_tile_selection(head_dim, expected):
    assert FlashAttentionForwardSm120.get_fwd_tile_size(head_dim, head_dim) == expected


@pytest.mark.parametrize("window_left", [None, 250])
def test_sm120_varlen_mqa_hd256_learnable_sink(window_left):
    """Cover the WeLMv4 Q6/KV1/hd256 global and local prefill shapes."""
    torch.manual_seed(1234)
    seq, num_q_heads, head_dim = 512, 6, 256
    q = torch.randn(seq, num_q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(seq, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
    window_size = (None, None) if window_left is None else (window_left, 0)
    out_ref, lse_ref = _reference(q, k, v, sinks, window_left)

    outputs = []
    for pack_gqa in (False, True, None):
        out, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=seq,
            max_seqlen_k=seq,
            softmax_scale=head_dim**-0.5,
            causal=True,
            window_size=window_size,
            sinks=sinks,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )
        output_error = (out.float() - out_ref).abs()
        lse_error = (lse - lse_ref).abs()
        assert output_error.max().item() < 1e-2
        assert output_error.mean().item() < 5e-4
        assert lse_error.max().item() < 5e-5
        outputs.append(out)

    torch.testing.assert_close(outputs[0], outputs[1], atol=0.0, rtol=0.0)
    torch.testing.assert_close(outputs[1], outputs[2], atol=0.0, rtol=0.0)


def test_sm120_varlen_padding_ctas_are_inert_across_tile_specializations():
    """Padding CTAs must not consume stale SMEM or write another batch's output."""
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_q_heads, head_dim = 6, 256

    def make_inputs(lengths, seed):
        torch.manual_seed(seed)
        total = sum(lengths)
        q = torch.randn(
            total,
            num_q_heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn(total, 1, head_dim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        sinks = torch.randn(num_q_heads, device="cuda", dtype=torch.bfloat16)
        cuts = [0]
        for length in lengths:
            cuts.append(cuts[-1] + length)
        cu_seqlens = torch.tensor(cuts, device="cuda", dtype=torch.int32)
        return q, k, v, sinks, cu_seqlens

    def run(inputs, lengths, pack_gqa):
        q, k, v, sinks, cu_seqlens = inputs
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(lengths),
            softmax_scale=head_dim**-0.5,
            causal=True,
            sinks=sinks,
            pack_gqa=pack_gqa,
            return_softmax_lse=True,
        )

    # Exercise the three SM-normalized selector regions before the padding
    # case. This is the order that exposed stale shared-memory state.
    rows_per_sm = (58, 84, 100)
    for seed, ratio in enumerate(rows_per_sm):
        seq = math.ceil(ratio * num_sms / num_q_heads)
        inputs = make_inputs([seq], seed)
        for pack_gqa in (False, True, False):
            run(inputs, [seq], pack_gqa)
        torch.cuda.synchronize()

    # At 64 query rows/SM, two batches select M64 while the conservative
    # varlen grid contains padding CTAs.
    seq = math.ceil(32 * num_sms / num_q_heads)
    lengths = [seq, seq]
    inputs = make_inputs(lengths, 10)
    q, k, v, sinks, _ = inputs
    references = [
        _reference(
            q[start : start + seq],
            k[start : start + seq],
            v[start : start + seq],
            sinks,
            None,
        )
        for start in (0, seq)
    ]
    out_ref = torch.cat([reference[0] for reference in references], dim=0)
    lse_ref = torch.cat([reference[1] for reference in references], dim=1)

    for pack_gqa in (False, True, None):
        out, lse = run(inputs, lengths, pack_gqa)
        output_error = (out.float() - out_ref).abs()
        lse_error = (lse - lse_ref).abs()
        assert output_error.max().item() < 1e-2
        assert output_error.mean().item() < 5e-4
        assert lse_error.max().item() < 5e-5
