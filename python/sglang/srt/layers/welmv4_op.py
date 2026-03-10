from typing import Optional

import torch
import triton
import triton.language as tl
from torch import nn

from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.layers.utils import MultiPlatformOp


@triton.jit
def _do_rms_norm(hidden, gamma, cols: int, eps: tl.constexpr):
    inv_rms = tl.math.rsqrt(tl.sum(hidden * hidden, axis=-1) / cols + eps)
    out = hidden * inv_rms
    out *= gamma.to(hidden.dtype)
    return out


@triton.jit
def rms_norm_kernel(  # pylint: disable=too-many-arguments,too-many-locals
    hidden_states_ptr: tl.tensor,
    reisdual_ptr: tl.tensor,
    gamma_ptr: tl.tensor,
    out_ptr: tl.tensor,
    out_residual_ptr: tl.tensor,
    out_copy_ptr: tl.tensor,
    rows: int,
    cols: int,
    eps: float,
    hidden_states_row_stride: int,
    residual_row_stride: int,
    residual_after_layernorm: tl.constexpr,
    NUM_SMS: tl.constexpr,  # pylint: disable=invalid-name
    BLOCK_SIZE: tl.constexpr,  # pylint: disable=invalid-name
):
    row_start = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK_SIZE)
    mask = cols_off < cols
    gamma_shm = tl.load(gamma_ptr + cols_off, mask=mask, other=0.0)

    original_dtype = hidden_states_ptr.dtype.element_ty
    for row_id in tl.range(row_start, rows, NUM_SMS, num_stages=4):
        h_offs = (row_id * hidden_states_row_stride + cols_off).to(tl.int64)
        r_offs = (row_id * residual_row_stride + cols_off).to(tl.int64)
        h = tl.load(hidden_states_ptr + h_offs, mask=mask, other=0.0).to(tl.float32)
        if reisdual_ptr is not None:
            r = tl.load(reisdual_ptr + r_offs, mask=mask, other=0.0).to(tl.float32)
            h = h + r

        output_offs = (row_id * cols + cols_off).to(tl.int64)
        if not residual_after_layernorm and out_residual_ptr is not None:
            tl.store(
                out_residual_ptr + output_offs,
                h.to(reisdual_ptr.dtype.element_ty),
                mask=mask,
            )

        out = _do_rms_norm(h, gamma_shm, cols, eps)
        if out_copy_ptr is not None:
            tl.store(out_copy_ptr + output_offs, out, mask=mask)

        out = out.to(original_dtype)
        if residual_after_layernorm:
            tl.store(out_residual_ptr + output_offs, out, mask=mask)

        tl.store(out_ptr + output_offs, out, mask=mask)


class WelmV4FusedRMSNorm(MultiPlatformOp):
    def __init__(
        self, hidden_size: int, eps: float = 1e-6, weight_dtype: Optional = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        self.num_sms = 78 * 8

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        residual_after_layernorm: bool = False,
        clone_fp32_out: bool = False,
    ):
        output = torch.empty_like(x)
        fp32_out = None
        out_residual = None
        if residual is not None or residual_after_layernorm:
            out_residual = torch.empty_like(x)
        if clone_fp32_out:
            fp32_out = torch.empty_like(x, dtype=torch.float32)
        cols = x.shape[-1]
        rows = x.numel() // cols

        if residual is not None:
            residual_row_stride = residual.stride(0)
        else:
            residual_row_stride = 0

        num_sms = min(rows, self.num_sms)
        block_size = triton.next_power_of_2(cols)
        rms_norm_kernel[(num_sms,)](
            x,
            residual,
            self.weight,
            output,
            out_residual,
            fp32_out,
            rows,
            cols,
            self.eps,
            x.stride(0),
            residual_row_stride,
            residual_after_layernorm,
            num_sms,
            block_size,
        )
        if out_residual is None:
            out_residual = x

        if clone_fp32_out:
            return output, out_residual, fp32_out
        else:
            return output, out_residual


@triton.jit
def sigmoid_mul_kernel(
    x: tl.tensor,
    y: tl.tensor,
    rows: int,
    cols: tl.constexpr,
    y_row_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    row_start = tl.program_id(0)
    col_off = tl.arange(0, BLOCK_SIZE)
    mask = col_off < cols
    for row_id in tl.range(row_start, rows, NUM_SMS, num_stages=4):
        y_off = row_id * y_row_stride + col_off
        y_data = tl.load(y + y_off, mask=mask, other=0.0)
        x_data = tl.load(x + row_id).to(tl.float32)
        out_data = tl.sigmoid(x_data) * y_data
        tl.store(y + y_off, out_data.to(y.dtype.element_ty), mask=mask)


# return sigmoid(x) * y
def inplace_sigmoid_mul(x: torch.Tensor, y: torch.Tensor):
    num_sms = 78 * 8
    cols = y.shape[-1]
    rows = y.numel() // cols
    block_size = triton.next_power_of_2(cols)
    sigmoid_mul_kernel[(num_sms,)](x, y, rows, cols, y.stride(-2), block_size, num_sms)


@triton.jit
def _rope(  # pylint: disable=too-many-arguments, too-many-locals
    data_ptr: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    num_heads: tl.constexpr,
    num_heads_blocked: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    num_head_offset = tl.arange(0, num_heads_blocked)
    rope_dim_offset = tl.arange(0, rope_dim)
    mask = num_head_offset[:, None] < num_heads
    x = tl.load(
        data_ptr + num_head_offset[:, None] * head_dim + rope_dim_offset[None, :],
        mask=mask,
    )
    x = x.reshape(num_heads_blocked, 2, half_rope_dim).trans(0, 2, 1)
    x1, x2 = x.split()
    x_out1 = x1 * cos - x2 * sin
    x_out2 = x1 * sin + x2 * cos
    x_out = tl.join(x_out1, x_out2).trans(0, 2, 1).reshape(num_heads_blocked, rope_dim)
    tl.store(
        data_ptr + num_head_offset[:, None] * head_dim + rope_dim_offset[None, :],
        x_out,
        mask=mask,
    )


@triton.jit
def _welmv4_inplace_rope_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    position_ptr: tl.tensor,
    cos_sin_cache_ptr: tl.tensor,
    last_index_ptr: tl.tensor,
    N: int,
    BS: int,
    q_token_stride: tl.constexpr,
    k_token_stride: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    num_sms: tl.constexpr,
    num_stages: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_k_heads: tl.constexpr,
    num_q_heads_blocked: tl.constexpr,
    num_k_heads_blocked: tl.constexpr,
):
    half_rope_dim: tl.constexpr = rope_dim // 2
    cos_off = tl.arange(0, half_rope_dim)
    sin_off = tl.arange(half_rope_dim, rope_dim)
    for token_id in tl.range(tl.program_id(0), N, num_sms, num_stages=num_stages):
        position_id = tl.load(position_ptr + token_id)
        cos_sin_cache = tl.load(cos_sin_cache_ptr + position_id * rope_dim + cos_off)
        sin_sin_cache = tl.load(cos_sin_cache_ptr + position_id * rope_dim + sin_off)
        q_data_ptr = q_ptr + token_id * q_token_stride + head_dim - rope_dim
        k_data_ptr = k_ptr + token_id * k_token_stride + head_dim - rope_dim
        _rope(
            k_data_ptr,
            cos_sin_cache,
            sin_sin_cache,
            num_k_heads,
            num_k_heads_blocked,
            head_dim,
            rope_dim,
        )
        if last_index_ptr is not None:
            if token_id < BS:
                position_id = tl.load(last_index_ptr + token_id)
                cos_sin_cache = tl.load(
                    cos_sin_cache_ptr + position_id * rope_dim + cos_off
                )
                sin_sin_cache = tl.load(
                    cos_sin_cache_ptr + position_id * rope_dim + sin_off
                )
                _rope(
                    q_data_ptr,
                    cos_sin_cache,
                    sin_sin_cache,
                    num_q_heads,
                    num_q_heads_blocked,
                    head_dim,
                    rope_dim,
                )
        else:
            _rope(
                q_data_ptr,
                cos_sin_cache,
                sin_sin_cache,
                num_q_heads,
                num_q_heads_blocked,
                head_dim,
                rope_dim,
            )


class WelmV4InplaceRotaryEmbedding(RotaryEmbedding):
    """WelmV4 rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        self.num_sms = 78 * 8

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[object] = None,
        last_index: Optional[torch.Tensor] = None,
    ):
        query = query.view(query.shape[0], -1, self.head_size)
        key = key.view(key.shape[0], -1, self.head_size)
        N = positions.shape[0]
        num_sms = min(N, self.num_sms)
        num_stages = 4
        BS = last_index.numel() if last_index is not None else 0
        _welmv4_inplace_rope_kernel[(num_sms,)](
            query,
            key,
            positions,
            self.cos_sin_cache,
            last_index,
            N,
            BS,
            query.stride(0),
            key.stride(0),
            self.head_size,
            self.rotary_dim,
            num_sms,
            num_stages,
            query.shape[-2],
            key.shape[-2],
            triton.next_power_of_2(query.shape[-2]),
            triton.next_power_of_2(key.shape[-2]),
        )
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s
