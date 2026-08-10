# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.

import math
from functools import lru_cache, partial
from types import SimpleNamespace
from typing import Callable, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils_basic
from cutlass import Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineState,
    PipelineTmaAsync,
    pipeline_init_arrive,
    pipeline_init_wait,
)
from quack import copy_utils, layout_utils

from sglang.jit_kernel.flash_attn.cute import pipeline as pipeline_custom
from sglang.jit_kernel.flash_attn.cute import utils
from sglang.jit_kernel.flash_attn.cute.block_info import BlockInfo
from sglang.jit_kernel.flash_attn.cute.block_sparsity import (
    BlockSparseTensors,
)
from sglang.jit_kernel.flash_attn.cute.cute_dsl_utils import (
    assume_tensor_aligned,
)
from sglang.jit_kernel.flash_attn.cute.flash_fwd import (
    FlashAttentionForwardBase,
)
from sglang.jit_kernel.flash_attn.cute.mask import AttentionMask
from sglang.jit_kernel.flash_attn.cute.named_barrier import NamedBarrierFwd
from sglang.jit_kernel.flash_attn.cute.pack_gqa import (
    PackGQA,
    pack_gqa_layout,
)
from sglang.jit_kernel.flash_attn.cute.paged_kv import PagedKVManager
from sglang.jit_kernel.flash_attn.cute.seqlen_info import SeqlenInfoQK
from sglang.jit_kernel.flash_attn.cute.softmax import (
    Softmax,
    apply_score_mod_inner,
)
from sglang.jit_kernel.flash_attn.cute.tile_scheduler import (
    SchedulingMode,
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
    TileSchedulerProtocol,
)
from sglang.jit_kernel.flash_attn.cute.utils import AuxData


class FlashAttentionForwardSm120(FlashAttentionForwardBase):
    """SM120 warp-MMA forward kernel with TMA fused into the QK warps."""

    def __init__(self, *args, direct_single_wave: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_single_wave = direct_single_wave

    supports_learnable_sink = True
    num_dma_threads = 32
    # Relative CTA critical-path costs keyed by the complete kernel config:
    # (fixed work, work per N block, extra local-mask work). Sequence length,
    # window, packed heads, and SM count remain analytic inputs to the generic
    # LPT model below. New head dimensions can be added after calibrating only
    # these three config-local weights.
    _lpt_cost_by_config = {
        (256, 256, 32, 64): (213, 95, 95),
        (256, 256, 48, 64): (221, 99, 74),
        (256, 256, 64, 64): (212, 101, 67),
    }
    _lpt_tie_margin = 1

    @staticmethod
    def _estimate_lpt_makespan(
        tile_m: int,
        tile_n: int,
        cost: tuple[int, int, int],
        *,
        seqlen_q: int,
        seqlen_k: int,
        num_sms: int,
        num_head_kv: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        is_local: bool,
        window_size_left: int | None,
        window_size_right: int | None,
    ) -> int | None:
        """Estimate the one- or two-wave LPT critical path in relative units."""
        if (
            seqlen_q <= 0
            or seqlen_k <= 0
            or num_sms <= 0
            or num_head_kv <= 0
            or qhead_per_kvhead <= 0
        ):
            return None

        packed_q = seqlen_q * qhead_per_kvhead
        num_m_blocks = (packed_q + tile_m - 1) // tile_m
        num_ctas = num_m_blocks * num_head_kv
        # Beyond two waves the largest tile is preferable for this bounded-SMEM
        # pipeline.  Returning None also keeps this O(1).
        if num_ctas > 2 * num_sms:
            return None

        fixed_cost, n_block_cost, local_mask_cost = cost
        fixed_cost += local_mask_cost if is_local else 0
        num_k_blocks = (seqlen_k + tile_n - 1) // tile_n
        seqlen_delta = seqlen_k - seqlen_q

        def cta_cost(launch_idx: int) -> int:
            # SingleTileVarlenScheduler repeats each reversed (LPT) M block
            # across the packed KV heads before advancing to the next block.
            m_block = num_m_blocks - 1 - launch_idx // num_head_kv
            m_idx_min = m_block * tile_m // qhead_per_kvhead
            m_idx_max = (
                (m_block + 1) * tile_m + qhead_per_kvhead - 1
            ) // qhead_per_kvhead

            if is_causal or (is_local and window_size_right is not None):
                n_idx_right = m_idx_max + seqlen_delta
                if not is_causal:
                    n_idx_right += window_size_right
                n_block_max = min(
                    num_k_blocks,
                    max(0, (n_idx_right + tile_n - 1) // tile_n),
                )
            else:
                n_block_max = num_k_blocks

            n_block_min = 0
            if is_local and window_size_left is not None:
                n_idx_left = m_idx_min + seqlen_delta - window_size_left
                n_block_min = max(n_idx_left // tile_n, 0)
            num_n_blocks = max(n_block_max - n_block_min, 0)
            return fixed_cost + n_block_cost * num_n_blocks

        heaviest_cta = cta_cost(0)
        if num_ctas <= num_sms:
            return heaviest_cta
        # With at most two waves, the first tail CTA is paired with the lightest
        # CTA in the first hardware wave.  This captures the discrete tail that
        # an average-work occupancy model misses.
        wave_boundary = cta_cost(num_sms - 1) + cta_cost(num_sms)
        return max(heaviest_cta, wave_boundary)

    @staticmethod
    def _smem_usage_in_bytes(
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        Q_in_regs=False,
    ) -> int:
        """Return SMEM usage after padding head dimensions to kernel alignment."""
        head_dim = math.ceil(head_dim / 16) * 16
        head_dim_v = math.ceil(head_dim_v / 16) * 16
        element_size = 2
        smem_usage_Q = tile_m * head_dim * element_size
        smem_usage_K = tile_n * head_dim * num_stages * element_size
        smem_usage_V = tile_n * head_dim_v * num_stages * element_size
        smem_usage_QV = (
            smem_usage_Q + smem_usage_V
            if not Q_in_regs
            else max(smem_usage_Q, smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_K
        if (head_dim, head_dim_v, tile_m, tile_n) in (
            (256, 256, 32, 64),
            (256, 256, 48, 64),
            (256, 256, 64, 48),
            (256, 256, 64, 64),
        ):
            # Q is copied to QK registers before the mainloop, so its allocation
            # can hold both P stages and later the O epilogue tile.  Four FP32
            # rows hold the per-stage rescale values and final scale/LSE.
            smem_usage += 4 * tile_m * 4
        # Q/K/V/P/final full-empty mbarriers and conservative field alignment.
        return smem_usage + 2048

    @staticmethod
    @lru_cache(maxsize=4096)
    def get_fwd_tile_size(
        head_dim: int,
        head_dim_v: int,
        total_q_rows: int | None = None,
        num_sms: int | None = None,
        num_batch: int | None = None,
        seqlen_q: int | None = None,
        seqlen_k: int | None = None,
        num_head_kv: int | None = None,
        qhead_per_kvhead: int | None = None,
        is_causal: bool = False,
        is_local: bool = False,
        window_size_left: int | None = None,
        window_size_right: int | None = None,
        pack_gqa: bool = False,
    ) -> tuple[int, int]:
        """Select an SM120 tile that fits the architecture's 99 KB SMEM."""
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        if (head_dim, head_dim_v) == (256, 256):
            if (
                total_q_rows is not None
                and num_sms is not None
                and total_q_rows > 76 * num_sms
                and total_q_rows <= 92 * num_sms
            ):
                # Preserve the previous multi-batch/non-packed fallback where
                # per-sequence LPT costs are unavailable on the host.
                fallback_candidates = (
                    (48, 64, 1),
                    (64, 64, 1),
                    (64, 48, 1),
                    (32, 64, 1),
                )
            else:
                fallback_candidates = (
                    (64, 64, 1),
                    (64, 48, 1),
                    (48, 64, 1),
                    (32, 64, 1),
                )
        elif (head_dim, head_dim_v) in ((64, 64), (128, 128)):
            # The larger SM120 warp-MMA shapes are register-bound at these
            # head dimensions. M128N128 at HD64 and M128N64 at HD128 spill,
            # while M64N64 remains spill-free and dominates across wave counts.
            fallback_candidates = (
                (64, 64, 1),
                (128, 64, 1),
            ) + (((128, 128, 1),) if head_dim == 64 else ())
        else:
            fallback_candidates = (
                ((128, 128, 1),) if max(head_dim, head_dim_v) <= 64 else ()
            ) + ((128, 64, 1), (64, 64, 1))

        # The scheduling equation is independent of HD. Pipeline-specific work
        # stays in a compact cost table keyed by the complete kernel config, so
        # another HD can be enabled without duplicating the sequence/window
        # policy. Shapes without calibrated configs retain the fallback above.
        model_candidates = tuple(
            (config[2], config[3], 1)
            for config in FlashAttentionForwardSm120._lpt_cost_by_config
            if config[:2] == (head_dim, head_dim_v)
            and FlashAttentionForwardSm120._smem_usage_in_bytes(*config, 1)
            <= smem_capacity
        )
        can_model_lpt = (
            len(model_candidates) >= 2
            and num_batch == 1
            and pack_gqa
            and total_q_rows is not None
            and num_sms is not None
            and seqlen_q is not None
            and seqlen_k is not None
            and num_head_kv is not None
            and qhead_per_kvhead is not None
            and total_q_rows == seqlen_q * num_head_kv * qhead_per_kvhead
            and (is_causal or is_local)
        )
        candidates = fallback_candidates
        if can_model_lpt:
            scores = {
                candidate: FlashAttentionForwardSm120._estimate_lpt_makespan(
                    candidate[0],
                    candidate[1],
                    FlashAttentionForwardSm120._lpt_cost_by_config[
                        (head_dim, head_dim_v, candidate[0], candidate[1])
                    ],
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    num_sms=num_sms,
                    num_head_kv=num_head_kv,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_causal=is_causal,
                    is_local=is_local,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                )
                for candidate in model_candidates
            }
            valid_scores = {
                candidate: score
                for candidate, score in scores.items()
                if score is not None
            }
            if valid_scores:
                exact_best = min(valid_scores, key=valid_scores.get)
                packed_q = seqlen_q * qhead_per_kvhead

                def num_ctas(candidate: tuple[int, int, int]) -> int:
                    return ((packed_q + candidate[0] - 1) // candidate[0]) * num_head_kv

                max_model_ctas = max(
                    num_ctas(candidate) for candidate in model_candidates
                )
                if num_ctas(exact_best) == max_model_ctas:
                    # Keep a genuinely faster high-parallelism tile. Once that
                    # tile loses outright, a near tie favors fewer CTAs.
                    preferred = exact_best
                else:
                    cutoff = (
                        valid_scores[exact_best]
                        + FlashAttentionForwardSm120._lpt_tie_margin
                    )
                    near = tuple(
                        candidate
                        for candidate, score in valid_scores.items()
                        if score <= cutoff
                    )
                    preferred = min(
                        near,
                        key=lambda candidate: (
                            num_ctas(candidate),
                            -(candidate[0] * candidate[1]),
                        ),
                    )
                candidates = (preferred,) + tuple(
                    candidate
                    for candidate in fallback_candidates
                    if candidate != preferred
                )
        for tile_m, tile_n, num_stages in candidates:
            if (
                FlashAttentionForwardSm120._smem_usage_in_bytes(
                    head_dim,
                    head_dim_v,
                    tile_m,
                    tile_n,
                    num_stages,
                )
                <= smem_capacity
            ):
                return tile_m, tile_n
        raise ValueError(
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) exceeds "
            f"SM120 shared-memory capacity ({smem_capacity} bytes)"
        )

    @staticmethod
    def get_fwd_num_stages(
        head_dim: int, head_dim_v: int, tile_m: int, tile_n: int
    ) -> int:
        """Return the public pipeline specialization depth."""
        return 1

    @staticmethod
    def get_fwd_num_threads(
        head_dim: int,
        head_dim_v: int,
        tile_m: int,
        tile_n: int,
        paged_kv: bool = False,
    ) -> int:
        """Return the number of warp-MMA consumer threads for an SM120 tile."""
        if paged_kv and head_dim == 256 and head_dim_v == 256:
            # The contiguous HD256 path assigns distinct QK and PV warp sets.
            # Paged KV instead reserves one DMA warp for gather and has each
            # consumer warp own the same 16 rows through both MMA phases.
            return tile_m * 2
        config = (head_dim, head_dim_v, tile_m, tile_n)
        if config == (256, 256, 32, 64):
            return 128
        if config == (256, 256, 48, 64):
            return 192
        if config in ((256, 256, 64, 48), (256, 256, 64, 64)):
            return 256
        if config == (256, 256, 96, 48):
            return 192
        return 128

    def _uses_split_pv_warps(self) -> bool:
        """Whether dedicated QK/PV warp sets exchange P through SMEM."""
        config = (
            self.tile_hdim,
            self.tile_hdimv,
            self.tile_m,
            self.tile_n,
            self.num_threads,
        )
        return config in (
            (256, 256, 32, 64, 128),
            (256, 256, 48, 64, 192),
            (256, 256, 64, 48, 256),
            (256, 256, 64, 64, 256),
        )

    def _q_in_regs_pipeline(self) -> bool:
        """Whether Q remains resident while its SMEM allocation holds P."""
        config = (
            self.tile_hdim,
            self.tile_hdimv,
            self.tile_m,
            self.tile_n,
            self.num_threads,
        )
        return config in (
            (256, 256, 32, 64, 128),
            (256, 256, 48, 64, 192),
            (256, 256, 64, 48, 256),
            (256, 256, 64, 64, 256),
        )

    def _num_k_stages(self) -> int:
        return self.num_stages

    def _num_v_stages(self) -> int:
        return 1 if self._uses_split_pv_warps() else self.num_stages

    def _num_p_stages(self) -> int:
        return 2

    def _num_dma_threads(self) -> int:
        return 0 if self._uses_split_pv_warps() else self.num_dma_threads

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
        paged_kv=False,
    ) -> bool:
        """Check the constraints of the dedicated SM120 TMA kernel."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0 or head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_stages != FlashAttentionForwardSm120.get_fwd_num_stages(
            head_dim, head_dim_v, tile_m, tile_n
        ):
            return False
        if num_threads != FlashAttentionForwardSm120.get_fwd_num_threads(
            head_dim, head_dim_v, tile_m, tile_n, paged_kv=paged_kv
        ):
            return False
        if Q_in_regs:
            return False
        smem_usage = FlashAttentionForwardSm120._smem_usage_in_bytes(
            head_dim,
            head_dim_v,
            tile_m,
            tile_n,
            num_stages,
            Q_in_regs,
        )
        if smem_usage > utils_basic.get_smem_capacity_in_bytes("sm_120"):
            return False
        if (head_dim, head_dim_v, tile_m, tile_n, num_threads) in (
            (256, 256, 32, 64, 128),
            (256, 256, 48, 64, 192),
            (256, 256, 64, 48, 256),
            (256, 256, 64, 64, 256),
        ):
            return True
        return (tile_m * 2) % num_threads == 0

    def _get_smem_layout_atom(self):
        sQ_layout_atom = self._make_smem_layout_atom(
            self.dtype, self.tile_hdim, is_k_major=True
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = self._make_smem_layout_atom(
            self.dtype, self.tile_hdimv, is_k_major=True
        )
        sO_layout_atom = sV_layout_atom
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, None

    def _setup_attributes(self):
        super()._setup_attributes()
        if const_expr(self._uses_split_pv_warps()):
            sK_layout_atom = self._make_smem_layout_atom(
                self.dtype, self.tile_hdim, is_k_major=True
            )
            self.sK_layout = cute.tile_to_shape(
                sK_layout_atom,
                (self.tile_n, self.tile_hdim, self._num_k_stages()),
                (0, 1, 2),
            )
        sV_layout_atom = self._make_smem_layout_atom(
            self.dtype, self.tile_hdimv, is_k_major=False
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            (self.tile_hdimv, self.tile_n, self._num_v_stages()),
            (1, 0, 2),
        )
        self.sP_layout = None
        if const_expr(self._uses_split_pv_warps()):
            sP_layout_atom = self._make_smem_layout_atom(
                self.dtype, self.tile_n, is_k_major=True
            )
            self.sP_layout = cute.tile_to_shape(
                sP_layout_atom,
                (self.tile_m, self.tile_n, self._num_p_stages()),
                (0, 1, 2),
            )

    @staticmethod
    def _make_smem_layout_atom(
        dtype: type[cutlass.Numeric],
        major_dim: int,
        *,
        is_k_major: bool,
    ) -> cute.ComposedLayout:
        """Build a TMA-compatible SMEM layout for SM120 warp MMA."""
        major_mode_bits = const_expr(major_dim * dtype.width)
        if const_expr(major_mode_bits % 1024 == 0):
            contiguous_bits, swizzle_bits = 1024, 3
        elif const_expr(major_mode_bits % 512 == 0):
            contiguous_bits, swizzle_bits = 512, 2
        elif const_expr(major_mode_bits % 256 == 0):
            contiguous_bits, swizzle_bits = 256, 1
        else:
            contiguous_bits, swizzle_bits = 128, 0
        contiguous_elems = const_expr(contiguous_bits // dtype.width)
        layout = (
            cute.make_layout(
                (8, contiguous_elems),
                stride=(contiguous_elems, 1),
            )
            if const_expr(is_k_major)
            else cute.make_layout(
                (contiguous_elems, 8),
                stride=(1, contiguous_elems),
            )
        )
        return cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 4, 3),
            0,
            layout,
        )

    def _get_tiled_mma(self):
        split_pv_warps = self._uses_split_pv_warps()
        num_qk_warps = self.tile_m // 16 if split_pv_warps else self.num_threads // 32
        num_pv_warps_m = self.tile_m // 16 if split_pv_warps else self.num_threads // 32
        num_pv_warps_n = 1
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (num_qk_warps, 1, 1),
            permutation_mnk=(num_qk_warps * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (num_pv_warps_m, num_pv_warps_n, 1),
            permutation_mnk=(num_pv_warps_m * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], 1024
            ]
            for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
        ]
        mbar_Q_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_K_struct = cute.struct.MemRange[cutlass.Int64, self._num_k_stages() * 2]
        mbar_V_struct = cute.struct.MemRange[cutlass.Int64, self._num_v_stages() * 2]
        mbar_P_struct = cute.struct.MemRange[
            cutlass.Int64,
            2 * self._num_p_stages() if self._uses_split_pv_warps() else 0,
        ]
        mbar_final_struct = cute.struct.MemRange[
            cutlass.Int64, 2 if self._uses_split_pv_warps() else 0
        ]
        num_stats = 4 * self.tile_m if self._uses_split_pv_warps() else 0
        softmax_stats_struct = cute.struct.MemRange[Float32, num_stats]
        num_p_elements = (
            0
            if self._q_in_regs_pipeline() or not self._uses_split_pv_warps()
            else cute.cosize(self.sP_layout)
        )
        sP_struct = cute.struct.Align[
            cute.struct.MemRange[
                self.dtype,
                num_p_elements,
            ],
            1024,
        ]

        @cute.struct
        class SharedStorage:
            mbar_Q: mbar_Q_struct
            mbar_K: mbar_K_struct
            mbar_V: mbar_V_struct
            mbar_P: mbar_P_struct
            mbar_final: mbar_final_struct
            softmax_stats: softmax_stats_struct
            sP: sP_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        return SharedStorage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        stream: cuda.CUstream = None,
    ):
        assert blocksparse_tensors is None, "Block sparsity is not supported on SM120"
        assert (
            mPageTable is None or not self._uses_split_pv_warps()
        ), "SM120 paged KV requires the dedicated DMA-warp specialization"
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                )
            )
        )
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_qk_threads = tiled_mma_qk.size
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = (
            self.num_qk_threads if self._uses_split_pv_warps() else self.num_threads
        )
        self.num_epilogue_threads = (
            self.num_mma_threads if self._uses_split_pv_warps() else self.num_threads
        )
        self.use_tma_O = False
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        QO_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        )
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        )
        mQ, mO = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=QO_layout_transpose)
            )
            for t in (mQ, mO)
        ]
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        if const_expr(mPageTable is None):
            V_layout_transpose = (
                [1, 0, 2, 3] if const_expr(mCuSeqlensK is None) else [1, 0, 2]
            )
            mV = cute.make_tensor(
                mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
            )
        if const_expr(mLSE is not None):
            LSE_layout_transpose = (
                [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
            )
            mLSE = cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )

        if const_expr(mPageTable is None):
            tma_copy_op = cpasync.CopyBulkTensorTileG2SOp()
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                tma_copy_op,
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                tma_copy_op,
                mV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_hdimv, self.tile_n),
                1,
            )
            self.tma_copy_bytes_K = cute.size_in_bytes(
                mK.element_type, cute.select(self.sK_layout, mode=[0, 1])
            )
            self.tma_copy_bytes_V = cute.size_in_bytes(
                mV.element_type, cute.select(self.sV_layout, mode=[0, 1])
            )
        else:
            tma_atom_K = None
            tma_atom_V = None
            tma_tensor_K = mK
            tma_tensor_V = mV

        is_varlen = const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None)
        num_batch = (
            mCuSeqlensQ.shape[0] - 1
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[3])
        )
        TileScheduler = SingleTileVarlenScheduler if is_varlen else SingleTileScheduler
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=num_batch,
            num_splits=1,
            seqlen_k=0,
            headdim=mQ.shape[1],
            headdim_v=mO.shape[1],
            total_q=(
                cute.size(mQ.shape[0])
                if const_expr(mCuSeqlensQ is not None)
                else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3])
            ),
            tile_shape_mn=(self.tile_m, self.tile_n),
            lpt=(self.is_causal or self.is_local) and not self.direct_single_wave,
            direct_single_batch=self.direct_single_wave,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            is_persistent=False,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args,
            scheduling_mode=SchedulingMode.STATIC,
        )
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        window_size_left = (
            Int32(window_size_left) if window_size_left is not None else None
        )
        window_size_right = (
            Int32(window_size_right) if window_size_right is not None else None
        )
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ,
            mK,
            self.qhead_per_kvhead,
            self.pack_gqa,
            aux_data.tensors,
            mPageTable,
        )

        self.kernel(
            mQ,
            tma_tensor_K,
            tma_tensor_V,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_K,
            tma_atom_V,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            learnable_sink,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            aux_data,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads + self._num_dma_threads(), 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if const_expr(mPageTable is None):
            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)
        sP = None
        sRowScale = None
        sFinalScale = None
        sLSE = None
        if const_expr(sP_layout is not None):
            if const_expr(self._q_in_regs_pipeline()):
                sP = storage.sQ.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
            else:
                sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
            sSoftmaxStats = storage.softmax_stats.get_tensor(
                cute.make_layout((4, self.tile_m), stride=(self.tile_m, 1))
            )
            sRowScale = sSoftmaxStats
            sFinalScale = sSoftmaxStats[2, None]
            sLSE = sSoftmaxStats[3, None]

        tma_group = CooperativeGroup(Agent.Thread)
        qk_group = CooperativeGroup(
            Agent.Thread, self.num_qk_threads // cute.arch.WARP_SIZE
        )
        pv_group = CooperativeGroup(
            Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        if const_expr(mPageTable is None):
            pipeline_k = PipelineTmaAsync.create(
                num_stages=self._num_k_stages(),
                producer_group=tma_group,
                consumer_group=qk_group,
                tx_count=self.tma_copy_bytes_K,
                barrier_storage=storage.mbar_K.data_ptr(),
                defer_sync=True,
            )
            pipeline_v = PipelineTmaAsync.create(
                num_stages=self._num_v_stages(),
                producer_group=tma_group,
                consumer_group=pv_group,
                tx_count=self.tma_copy_bytes_V,
                barrier_storage=storage.mbar_V.data_ptr(),
                defer_sync=True,
            )
        else:
            dma_group = CooperativeGroup(Agent.Thread, self.num_dma_threads)
            mma_group = CooperativeGroup(Agent.Thread, self.num_threads)
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                num_stages=self._num_k_stages(),
                producer_group=dma_group,
                consumer_group=mma_group,
                barrier_storage=storage.mbar_K.data_ptr(),
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                num_stages=self._num_v_stages(),
                producer_group=dma_group,
                consumer_group=mma_group,
                barrier_storage=storage.mbar_V.data_ptr(),
                defer_sync=True,
            )
        pipeline_p = None
        pipeline_final = None
        if const_expr(sP_layout is not None):
            pipeline_p = PipelineAsync.create(
                num_stages=self._num_p_stages(),
                producer_group=CooperativeGroup(Agent.Thread, self.num_qk_threads),
                consumer_group=CooperativeGroup(Agent.Thread, self.num_mma_threads),
                barrier_storage=storage.mbar_P.data_ptr(),
                defer_sync=True,
                name="sm120_p",
            )
            pipeline_final = PipelineAsync.create(
                num_stages=1,
                producer_group=CooperativeGroup(Agent.Thread, self.num_qk_threads),
                consumer_group=CooperativeGroup(Agent.Thread, self.num_mma_threads),
                barrier_storage=storage.mbar_final.data_ptr(),
                defer_sync=True,
                name="sm120_final",
            )
        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        if work_tile.is_valid_tile:
            pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
            pipeline_init_wait(cluster_shape_mn=(1, 1))

            if const_expr(self._uses_split_pv_warps()):
                cute.experimental.iket.range_push("consumer")
                if warp_idx < self.num_qk_threads // cute.arch.WARP_SIZE:
                    self.mma_qk_pipeline_persistent(
                        mQ,
                        mK,
                        mV,
                        sQ,
                        sK,
                        sV,
                        tma_atom_K,
                        tma_atom_V,
                        sP,
                        sRowScale,
                        sFinalScale,
                        sLSE,
                        learnable_sink,
                        pipeline_k,
                        pipeline_v,
                        pipeline_p,
                        pipeline_final,
                        gmem_tiled_copy_Q,
                        tiled_mma_qk,
                        tidx,
                        softmax_scale_log2,
                        softmax_scale,
                        tile_scheduler,
                        mCuSeqlensQ,
                        mCuSeqlensK,
                        mSeqUsedQ,
                        mSeqUsedK,
                        window_size_left,
                        window_size_right,
                        aux_data,
                        fastdiv_mods,
                    )
                else:
                    self.mma_pv_pipeline_persistent(
                        mQ,
                        mK,
                        mO,
                        mLSE,
                        sV,
                        sO,
                        sP,
                        sRowScale,
                        sFinalScale,
                        sLSE,
                        pipeline_v,
                        pipeline_p,
                        pipeline_final,
                        gmem_tiled_copy_O,
                        tiled_mma_pv,
                        tidx - self.num_qk_threads,
                        tile_scheduler,
                        mCuSeqlensQ,
                        mCuSeqlensK,
                        mSeqUsedQ,
                        mSeqUsedK,
                        window_size_left,
                        window_size_right,
                    )
                cute.experimental.iket.range_pop()
            elif warp_idx == 0:
                cute.experimental.iket.range_push("producer")
                if const_expr(mPageTable is None):
                    self.load_tma_persistent(
                        mK,
                        mV,
                        sK,
                        sV,
                        tma_atom_K,
                        tma_atom_V,
                        pipeline_k,
                        pipeline_v,
                        tile_scheduler,
                        mQ,
                        mCuSeqlensQ,
                        mCuSeqlensK,
                        mSeqUsedQ,
                        mSeqUsedK,
                        window_size_left,
                        window_size_right,
                    )
                else:
                    self.load_paged_persistent(
                        mK,
                        mV,
                        mPageTable,
                        sK,
                        sV,
                        pipeline_k,
                        pipeline_v,
                        tile_scheduler,
                        mQ,
                        mCuSeqlensQ,
                        mSeqUsedQ,
                        mSeqUsedK,
                        window_size_left,
                        window_size_right,
                        tidx,
                    )
                cute.experimental.iket.range_pop()
            elif warp_idx <= self.num_threads // cute.arch.WARP_SIZE:
                cute.experimental.iket.range_push("consumer")
                self.mma_persistent(
                    mQ,
                    mK,
                    mO,
                    mLSE,
                    sQ,
                    sK,
                    sV,
                    sO,
                    sP,
                    sRowScale,
                    sLSE,
                    learnable_sink,
                    pipeline_k,
                    pipeline_v,
                    gmem_tiled_copy_Q,
                    gmem_tiled_copy_O,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    tidx - self.num_dma_threads,
                    softmax_scale_log2,
                    softmax_scale,
                    tile_scheduler,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                    mPageTable,
                    window_size_left,
                    window_size_right,
                    True,
                    aux_data,
                    fastdiv_mods,
                )
                cute.experimental.iket.range_pop()

    @cute.jit
    def load_tma_persistent(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        tile_scheduler: TileSchedulerProtocol,
        mQ: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
    ):
        producer_state_k = PipelineState(
            self._num_k_stages(), Int32(0), Int32(0), Int32(1)
        )
        producer_state_v = PipelineState(
            self._num_v_stages(), Int32(0), Int32(0), Int32(1)
        )
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=(
                mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        head_idx_kv = (
            head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        )
        producer_state_k, producer_state_v = self.load_tma(
            mK,
            mV,
            sK,
            sV,
            tma_atom_K,
            tma_atom_V,
            pipeline_k,
            pipeline_v,
            producer_state_k,
            producer_state_v,
            seqlen,
            n_block_min,
            n_block_max,
            head_idx_kv,
            batch_idx,
        )

        pipeline_k.producer_tail(producer_state_k)
        pipeline_v.producer_tail(producer_state_v)

    @cute.jit
    def load_paged_persistent(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPageTable: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        tile_scheduler: TileSchedulerProtocol,
        mQ: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        tidx: Int32,
    ):
        producer_state_k = PipelineState(
            self._num_k_stages(), Int32(0), Int32(0), Int32(1)
        )
        producer_state_v = PipelineState(
            self._num_v_stages(), Int32(0), Int32(0), Int32(1)
        )
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=(
                mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=None,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        head_idx_kv = (
            head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        )
        paged_kv_manager = PagedKVManager.create(
            mPageTable,
            mK,
            mV,
            FastDivmodDivisor(mK.shape[0]),
            batch_idx,
            head_idx_kv,
            tidx,
            seqlen.seqlen_k,
            0,
            self.tile_n,
            self.tile_hdim,
            self.tile_hdimv,
            self.num_dma_threads,
            mK.element_type,
            arch=120,
        )
        num_n_blocks = cutlass.max(n_block_max - n_block_min, 1)
        for n_tile in cutlass.range(num_n_blocks, unroll=1):
            n_block = cutlass.max(n_block_max - 1 - n_tile, n_block_min)
            paged_kv_manager.load_page_table(n_block)

            pipeline_k.producer_acquire(producer_state_k)
            paged_kv_manager.load_KV(
                n_block,
                sK[None, None, producer_state_k.index],
                "K",
            )
            cute.arch.cp_async_commit_group()
            pipeline_k.producer_commit(producer_state_k)
            producer_state_k.advance()

            pipeline_v.producer_acquire(producer_state_v)
            sV_stage = layout_utils.transpose_view(
                sV[None, None, producer_state_v.index]
            )
            paged_kv_manager.load_KV(n_block, sV_stage, "V")
            cute.arch.cp_async_commit_group()
            pipeline_v.producer_commit(producer_state_v)
            producer_state_v.advance()

        pipeline_k.producer_tail(producer_state_k)
        pipeline_v.producer_tail(producer_state_v)

    @cute.jit
    def mma_qk_pipeline_persistent(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        sP: cute.Tensor,
        sRowScale: cute.Tensor,
        sFinalScale: cute.Tensor,
        sLSE: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        pipeline_final: PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=(
                mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        head_idx_kv = (
            head_idx if const_expr(self.pack_gqa) else head_idx // self.qhead_per_kvhead
        )
        self.mma_qk_pipeline(
            mQ,
            mK,
            mV,
            sQ,
            sK,
            sV,
            tma_atom_K,
            tma_atom_V,
            sP,
            sRowScale,
            sFinalScale,
            sLSE,
            learnable_sink,
            pipeline_k,
            pipeline_v,
            pipeline_p,
            pipeline_final,
            gmem_tiled_copy_Q,
            tiled_mma_qk,
            tidx,
            softmax_scale_log2,
            softmax_scale,
            block_info,
            seqlen,
            n_block_min,
            n_block_max,
            m_block,
            head_idx,
            head_idx_kv,
            batch_idx,
            aux_data,
            fastdiv_mods,
        )

    @cute.jit
    def mma_pv_pipeline_persistent(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sV: cute.Tensor,
        sO: cute.Tensor,
        sP: cute.Tensor,
        sRowScale: cute.Tensor,
        sFinalScale: cute.Tensor,
        sLSE: cute.Tensor,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        pipeline_final: PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
    ):
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=(
                mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        self.mma_pv_pipeline(
            mO,
            mLSE,
            sV,
            sO,
            sP,
            sRowScale,
            sFinalScale,
            sLSE,
            pipeline_v,
            pipeline_p,
            pipeline_final,
            gmem_tiled_copy_O,
            tiled_mma_pv,
            tidx,
            block_info,
            seqlen,
            n_block_min,
            n_block_max,
            m_block,
            head_idx,
            batch_idx,
        )

    @cute.jit
    def _run_n_block_schedule(
        self,
        compute_one_n_block: Callable,
        role_state,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        m_block: Int32,
        mask_fn: Optional[Callable],
    ):
        if const_expr(mask_fn is not None):
            mask_fn_seqlen = partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True)
            mask_fn_no_seqlen = partial(
                mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
            )
        else:
            mask_fn_seqlen = None
            mask_fn_no_seqlen = None

        n_block = cutlass.max(n_block_max - 1, 0)
        role_state = compute_one_n_block(
            n_block,
            role_state,
            mask_fn=mask_fn_seqlen,
            is_first_n_block=True,
        )
        n_block_upper = n_block
        if const_expr(self.is_causal or self.is_local):
            n_block_min_causal_local_mask = (
                block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
            )
            for n_tile in cutlass.range(
                n_block_max - 1 - n_block_min_causal_local_mask, unroll=1
            ):
                n_block = n_block_max - 2 - n_tile
                role_state = compute_one_n_block(
                    n_block,
                    role_state,
                    mask_fn=mask_fn_seqlen,
                )
            n_block_upper = cutlass.min(n_block_upper, n_block_min_causal_local_mask)
        n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
            seqlen, m_block, n_block_min
        )
        for n_tile in cutlass.range(
            n_block_upper - n_block_min_before_local_mask, unroll=1
        ):
            role_state = compute_one_n_block(
                n_block_upper - n_tile - 1,
                role_state,
                mask_fn=mask_fn_no_seqlen,
            )
        if const_expr(self.is_local and block_info.window_size_left is not None):
            n_block_upper = cutlass.min(n_block_upper, n_block_min_before_local_mask)
            for n_tile in cutlass.range(n_block_upper - n_block_min, unroll=1):
                role_state = compute_one_n_block(
                    n_block_upper - n_tile - 1,
                    role_state,
                    mask_fn=mask_fn_no_seqlen,
                )
        return role_state

    @cute.jit
    def mma_qk_pipeline(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        sP: cute.Tensor,
        sRowScale: cute.Tensor,
        sFinalScale: cute.Tensor,
        sLSE: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        pipeline_final: PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        m_block: Int32,
        head_idx: Int32,
        head_idx_kv: Int32,
        batch_idx: Int32,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
        if const_expr(not self.pack_gqa):
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
        if const_expr(not seqlen.has_cu_seqlens_k):
            mK_cur = mK[None, None, head_idx_kv, batch_idx]
            mV_cur = mV[None, None, head_idx_kv, batch_idx]
        else:
            mK_cur = cute.domain_offset(
                (seqlen.offset_k, 0), mK[None, None, head_idx_kv]
            )
            mV_cur = cute.domain_offset(
                (0, seqlen.offset_k), mV[None, None, head_idx_kv]
            )
        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
        gV = cute.local_tile(mV_cur, (self.tile_hdimv, self.tile_n), (0, None))
        copy_K, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_K, 0, cute.make_layout(1), gK, sK
        )
        copy_V, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_V, 0, cute.make_layout(1), gV, sV
        )

        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        smem_copy_atom_qk = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_thr_copy_q = utils.make_tiled_copy_A(
            smem_copy_atom_qk, tiled_mma_qk
        ).get_slice(tidx)
        smem_thr_copy_k = utils.make_tiled_copy_B(
            smem_copy_atom_qk, tiled_mma_qk
        ).get_slice(tidx)
        tCrQ = None
        if const_expr(self._q_in_regs_pipeline()):
            tCrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        smem_store_atom_p = utils.get_smem_store_atom(120, self.dtype)
        smem_thr_store_p = cute.make_tiled_copy_C(
            smem_store_atom_p, tiled_mma_qk
        ).get_slice(tidx)
        tPsP_store = smem_thr_store_p.partition_D(sP)

        cute.experimental.iket.range_push("qk_q_load")
        gmem_thr_copy_q = gmem_tiled_copy_Q.get_slice(tidx)
        if const_expr(not self.pack_gqa):
            self.load_Q(
                gmem_thr_copy_q,
                gQ,
                sQ,
                m_block,
                seqlen=seqlen.seqlen_q,
                headdim=mQ.shape[1],
            )
        else:
            PackGQA(
                self.tile_m,
                self.tile_hdim,
                self.check_hdim_oob,
                self.qhead_per_kvhead,
            ).load_Q(
                mQ_cur,
                sQ,
                gmem_tiled_copy_Q,
                tidx,
                m_block,
                seqlen.seqlen_q,
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_Q_load_threads,
        )
        if const_expr(self._q_in_regs_pipeline()):
            tCsQ = smem_thr_copy_q.partition_S(sQ)
            tCrQ_copy_view = smem_thr_copy_q.retile(tCrQ)
            cute.copy(smem_thr_copy_q, tCsQ, tCrQ_copy_view)
            # All QK warps must finish reading Q before its allocation becomes P.
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_Q_load_threads,
            )
        cute.experimental.iket.range_pop()

        acc_shape_s = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_shape_s[0][0] * acc_shape_s[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
        mask = AttentionMask(
            self.tile_m,
            self.tile_n,
            seqlen,
            block_info.window_size_left,
            block_info.window_size_right,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        mask_fn = partial(
            mask.apply_mask,
            batch_idx=batch_idx,
            head_idx=head_idx,
            m_block=m_block,
            thr_mma=thr_mma_qk,
            mask_causal=self.is_causal,
            mask_local=self.is_local,
            aux_data=aux_data,
            fastdiv_mods=(
                fastdiv_mods if const_expr(self.mask_mod is not None) else None
            ),
        )
        compute_one_n_block = partial(
            self.compute_one_n_block_qk_pipeline,
            thr_mma_qk=thr_mma_qk,
            sQ=sQ,
            tCrQ=tCrQ,
            sK=sK,
            tPsP_store=tPsP_store,
            smem_thr_copy_q=smem_thr_copy_q,
            smem_thr_copy_k=smem_thr_copy_k,
            smem_thr_store_p=smem_thr_store_p,
            tScS_mn=tScS_mn,
            copy_K=copy_K,
            copy_V=copy_V,
            tidx=tidx,
            sRowScale=sRowScale,
            softmax=softmax,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            pipeline_p=pipeline_p,
            batch_idx=batch_idx,
            head_idx=head_idx,
            m_block=m_block,
            seqlen=seqlen,
            aux_data=aux_data,
            fastdiv_mods=fastdiv_mods,
        )
        role_state = (
            PipelineState(self._num_k_stages(), Int32(0), Int32(0), Int32(1)),
            PipelineState(self._num_v_stages(), Int32(0), Int32(0), Int32(1)),
            PipelineState(self._num_k_stages(), Int32(0), Int32(0), Int32(0)),
            PipelineState(self._num_p_stages(), Int32(0), Int32(0), Int32(1)),
        )
        cute.experimental.iket.range_push("qk_mainloop")
        producer_state_k, producer_state_v, k_state, p_state = (
            self._run_n_block_schedule(
                compute_one_n_block,
                role_state,
                block_info,
                seqlen,
                n_block_min,
                n_block_max,
                m_block,
                mask_fn,
            )
        )
        cute.experimental.iket.range_pop()

        cute.experimental.iket.range_push("qk_finalize")
        sink_val = None
        if const_expr(learnable_sink is not None):
            if const_expr(not self.pack_gqa):
                sink_val = Float32(learnable_sink[head_idx])
            else:
                sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                    row = m_block * self.tile_m + tScS_mn[r][0]
                    q_head_idx = (
                        row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                    )
                    sink_val[r] = Float32(learnable_sink[q_head_idx])
        row_scale = softmax.finalize(sink_val=sink_val)
        final_state = PipelineState(1, Int32(0), Int32(0), Int32(1))
        pipeline_final.producer_acquire(final_state)
        if tScS_mn[0][1] == 0:
            for r in cutlass.range(cute.size(row_scale), unroll_full=True):
                row = tScS_mn[r][0]
                sFinalScale[row] = row_scale[r]
                sLSE[row] = softmax.row_sum[r]
        cute.arch.fence_view_async_shared()
        pipeline_final.producer_commit(final_state)
        final_state.advance()
        cute.experimental.iket.range_pop()

        pipeline_p.producer_tail(p_state)
        pipeline_final.producer_tail(final_state)
        if tidx < cute.arch.WARP_SIZE:
            pipeline_k.producer_tail(producer_state_k)
            pipeline_v.producer_tail(producer_state_v)

    @cute.jit
    def compute_one_n_block_qk_pipeline(
        self,
        n_block: Int32,
        role_state,
        thr_mma_qk: cute.TiledMma,
        sQ: cute.Tensor,
        tCrQ: Optional[cute.Tensor],
        sK: cute.Tensor,
        tPsP_store: cute.Tensor,
        smem_thr_copy_q: cute.TiledCopy,
        smem_thr_copy_k: cute.TiledCopy,
        smem_thr_store_p: cute.TiledCopy,
        tScS_mn: cute.Tensor,
        copy_K: Callable,
        copy_V: Callable,
        tidx: Int32,
        sRowScale: cute.Tensor,
        softmax: Softmax,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
    ):
        producer_state_k, producer_state_v, k_state, p_state = role_state
        cute.experimental.iket.range_push("qk_tma_issue")
        if tidx < cute.arch.WARP_SIZE:
            pipeline_k.producer_acquire(producer_state_k)
            copy_K(
                src_idx=n_block,
                dst_idx=producer_state_k.index,
                tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state_k),
            )
            pipeline_k.producer_commit(producer_state_k)
            pipeline_v.producer_acquire(producer_state_v)
            copy_V(
                src_idx=n_block,
                dst_idx=producer_state_v.index,
                tma_bar_ptr=pipeline_v.producer_get_barrier(producer_state_v),
            )
            pipeline_v.producer_commit(producer_state_v)
        producer_state_k.advance()
        producer_state_v.advance()
        cute.experimental.iket.range_pop()

        pipeline_p.producer_acquire(p_state)

        cute.experimental.iket.range_push("qk_k_wait")
        k_wait_token = pipeline_k.consumer_try_wait(k_state)
        pipeline_k.consumer_wait(k_state, k_wait_token)
        cute.experimental.iket.range_pop()

        cute.experimental.iket.range_push("qk_mma")
        acc_shape_s = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_s = cute.make_rmem_tensor(acc_shape_s, Float32)
        acc_s.fill(0.0)
        if const_expr(self._q_in_regs_pipeline()):
            self._gemm_qk_a_in_regs(
                thr_mma_qk,
                acc_s,
                tCrQ,
                sK[None, None, k_state.index],
                smem_thr_copy_k,
            )
        else:
            self._gemm_qk_phase_local(
                thr_mma_qk,
                acc_s,
                sQ,
                sK[None, None, k_state.index],
                smem_thr_copy_q,
                smem_thr_copy_k,
            )
        pipeline_k.consumer_release(k_state)
        k_state.advance()
        cute.experimental.iket.range_pop()

        cute.experimental.iket.range_push("qk_softmax_store_p")
        if const_expr(self.score_mod is not None):
            self.apply_score_mod(
                thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_s,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
        if const_expr(mask_fn is not None):
            mask_fn(acc_s, n_block=n_block)
        row_scale = softmax.online_softmax(acc_s, is_first=is_first_n_block)
        rP = cute.make_fragment_like(acc_s, self.dtype)
        rP.store(acc_s.load().to(self.dtype))
        tOrP_qk = layout_utils.reshape_acc_to_frgA(rP)
        tPrP = smem_thr_store_p.retile(tOrP_qk)
        cute.copy(
            smem_thr_store_p,
            tPrP,
            tPsP_store[None, None, None, p_state.index],
        )
        self._publish_row_scale(row_scale, tScS_mn, sRowScale[p_state.index, None])
        cute.arch.fence_view_async_shared()
        pipeline_p.producer_commit(p_state)
        p_state.advance()
        cute.experimental.iket.range_pop()
        return producer_state_k, producer_state_v, k_state, p_state

    @cute.jit
    def mma_pv_pipeline(
        self,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sV: cute.Tensor,
        sO: cute.Tensor,
        sP: cute.Tensor,
        sRowScale: cute.Tensor,
        sFinalScale: cute.Tensor,
        sLSE: cute.Tensor,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        pipeline_final: PipelineAsync,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        acc_shape_o = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_o = cute.make_rmem_tensor(acc_shape_o, Float32)
        acc_o.fill(0.0)

        smem_copy_atom_p = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_v = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_p = utils.make_tiled_copy_A(
            smem_copy_atom_p, tiled_mma_pv
        ).get_slice(tidx)
        smem_thr_copy_v = utils.make_tiled_copy_B(
            smem_copy_atom_v, tiled_mma_pv
        ).get_slice(tidx)
        tPsP = smem_thr_copy_p.partition_S(sP)
        tOrP = thr_mma_pv.make_fragment_A(thr_mma_pv.partition_A(sP[None, None, 0]))
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        tOcO_mn = layout_utils.reshape_acc_to_mn(thr_mma_pv.partition_C(cO))
        compute_one_n_block = partial(
            self.compute_one_n_block_pv_pipeline,
            thr_mma_pv=thr_mma_pv,
            acc_o=acc_o,
            tOrP=tOrP,
            tPsP=tPsP,
            sV=sV,
            sRowScale=sRowScale,
            tOcO_mn=tOcO_mn,
            smem_thr_copy_p=smem_thr_copy_p,
            smem_thr_copy_v=smem_thr_copy_v,
            pipeline_v=pipeline_v,
            pipeline_p=pipeline_p,
        )
        role_state = (
            PipelineState(self._num_v_stages(), Int32(0), Int32(0), Int32(0)),
            PipelineState(self._num_p_stages(), Int32(0), Int32(0), Int32(0)),
        )
        cute.experimental.iket.range_push("pv_mainloop")
        v_state, p_state = self._run_n_block_schedule(
            compute_one_n_block,
            role_state,
            block_info,
            seqlen,
            n_block_min,
            n_block_max,
            m_block,
            None,
        )
        cute.experimental.iket.range_pop()

        cute.experimental.iket.range_push("pv_finalize_epilogue")
        final_state = PipelineState(1, Int32(0), Int32(0), Int32(0))
        pipeline_final.consumer_wait(final_state)
        num_rows_pv = acc_o.shape[0][0] * acc_o.shape[1]
        row_scale = cute.make_rmem_tensor(num_rows_pv, Float32)
        lse = cute.make_rmem_tensor(num_rows_pv, Float32)
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            row = tOcO_mn[r, 0][0]
            row_scale[r] = sFinalScale[row]
            lse[r] = sLSE[row]
        pipeline_final.consumer_release(final_state)
        final_state.advance()
        self._rescale_O(acc_o, row_scale)
        self.epilogue(
            acc_o,
            lse,
            mO,
            mLSE,
            sO,
            seqlen,
            gmem_tiled_copy_O,
            None,
            tiled_mma_pv,
            tidx,
            m_block,
            head_idx,
            batch_idx,
        )
        cute.experimental.iket.range_pop()

    @cute.jit
    def compute_one_n_block_pv_pipeline(
        self,
        n_block: Int32,
        role_state,
        thr_mma_pv: cute.TiledMma,
        acc_o: cute.Tensor,
        tOrP: cute.Tensor,
        tPsP: cute.Tensor,
        sV: cute.Tensor,
        sRowScale: cute.Tensor,
        tOcO_mn: cute.Tensor,
        smem_thr_copy_p: cute.TiledCopy,
        smem_thr_copy_v: cute.TiledCopy,
        pipeline_v: PipelineAsync,
        pipeline_p: PipelineAsync,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
    ):
        v_state, p_state = role_state
        cute.experimental.iket.range_push("pv_p_wait_load")
        p_wait_token = pipeline_p.consumer_try_wait(p_state)
        pipeline_p.consumer_wait(p_state, p_wait_token)
        num_rows_pv = acc_o.shape[0][0] * acc_o.shape[1]
        row_scale = cute.make_rmem_tensor(num_rows_pv, Float32)
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            row_scale[r] = sRowScale[p_state.index, tOcO_mn[r, 0][0]]
        self._rescale_O(acc_o, row_scale)
        tOrP_copy_view = smem_thr_copy_p.retile(tOrP)
        cute.copy(
            smem_thr_copy_p,
            tPsP[None, None, None, p_state.index],
            tOrP_copy_view,
        )
        pipeline_p.consumer_release(p_state)
        p_state.advance()
        cute.experimental.iket.range_pop()

        cute.experimental.iket.range_push("pv_v_wait_mma")
        v_wait_token = pipeline_v.consumer_try_wait(v_state)
        pipeline_v.consumer_wait(v_state, v_wait_token)
        self._gemm_pv_phase_local(
            thr_mma_pv,
            acc_o,
            tOrP,
            sV[None, None, v_state.index],
            smem_thr_copy_v,
        )
        pipeline_v.consumer_release(v_state)
        v_state.advance()
        cute.experimental.iket.range_pop()
        return v_state, p_state

    @cute.jit
    def mma_persistent(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sRowScale: Optional[cute.Tensor],
        sLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        tile_scheduler: TileSchedulerProtocol,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        is_qk_owner: cutlass.Constexpr[bool],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        consumer_state = PipelineState(self.num_stages, Int32(0), Int32(0), Int32(0))
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=(
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
            ),
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        m_block, head_idx, batch_idx, _ = work_tile.tile_idx
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_idx,
            seqlen_q_static=(
                mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1]
            ),
            seqlen_k_static=(
                mK.shape[0]
                if const_expr(mPageTable is None)
                else mK.shape[0] * mPageTable.shape[1]
            ),
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
        self.mma(
            mQ,
            mO,
            mLSE,
            sQ,
            sK,
            sV,
            sO,
            sP,
            sRowScale,
            sLSE,
            learnable_sink,
            pipeline_k,
            pipeline_v,
            gmem_tiled_copy_Q,
            gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tidx,
            softmax_scale_log2,
            softmax_scale,
            consumer_state,
            block_info,
            seqlen,
            n_block_min,
            n_block_max,
            m_block,
            head_idx,
            batch_idx,
            is_qk_owner,
            aux_data,
            fastdiv_mods,
        )

    @cute.jit
    def load_tma(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        producer_state_k: PipelineState,
        producer_state_v: PipelineState,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        if const_expr(not seqlen.has_cu_seqlens_k):
            mK_cur = mK[None, None, head_idx, batch_idx]
            mV_cur = mV[None, None, head_idx, batch_idx]
        else:
            mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, head_idx])
            mV_cur = cute.domain_offset((0, seqlen.offset_k), mV[None, None, head_idx])
        gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
        gV = cute.local_tile(mV_cur, (self.tile_hdimv, self.tile_n), (0, None))
        copy_K, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_K, 0, cute.make_layout(1), gK, sK
        )
        copy_V, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_V, 0, cute.make_layout(1), gV, sV
        )
        num_n_blocks = cutlass.max(n_block_max - n_block_min, 1)
        for n_tile in cutlass.range(num_n_blocks, unroll=1):
            n_block = cutlass.max(n_block_max - 1 - n_tile, n_block_min)
            cute.experimental.iket.range_push("k_acquire_issue")
            pipeline_k.producer_acquire(producer_state_k)
            copy_K(
                src_idx=n_block,
                dst_idx=producer_state_k.index,
                tma_bar_ptr=pipeline_k.producer_get_barrier(producer_state_k),
            )
            pipeline_k.producer_commit(producer_state_k)
            producer_state_k.advance()
            cute.experimental.iket.range_pop()

            cute.experimental.iket.range_push("v_acquire_issue")
            pipeline_v.producer_acquire(producer_state_v)
            copy_V(
                src_idx=n_block,
                dst_idx=producer_state_v.index,
                tma_bar_ptr=pipeline_v.producer_get_barrier(producer_state_v),
            )
            pipeline_v.producer_commit(producer_state_v)
            producer_state_v.advance()
            cute.experimental.iket.range_pop()
        return producer_state_k, producer_state_v

    @cute.jit
    def mma(
        self,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sRowScale: Optional[cute.Tensor],
        sLSE: Optional[cute.Tensor],
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        consumer_state: PipelineState,
        block_info: BlockInfo,
        seqlen: SeqlenInfoQK,
        n_block_min: Int32,
        n_block_max: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        is_qk_owner: cutlass.Constexpr[bool],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
        if const_expr(not self.pack_gqa):
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))

        split_pv_warps = self._uses_split_pv_warps()
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype
        )
        smem_thr_copy_V = utils.make_tiled_copy_B(
            smem_copy_atom_V, tiled_mma_pv
        ).get_slice(tidx)
        if const_expr(split_pv_warps):
            tOrV = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sV[None, None, 0]))
            tOsV = smem_thr_copy_V.partition_S(sV)
            smem_thr_copy_P = utils.make_tiled_copy_A(
                smem_copy_atom_QK, tiled_mma_pv
            ).get_slice(tidx)
            tPsP = smem_thr_copy_P.partition_S(sP)
            tOrP = thr_mma_pv.make_fragment_A(thr_mma_pv.partition_A(sP))
        if const_expr(not split_pv_warps or is_qk_owner):
            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            smem_thr_copy_Q = utils.make_tiled_copy_A(
                smem_copy_atom_QK, tiled_mma_qk
            ).get_slice(tidx)
            smem_thr_copy_K = utils.make_tiled_copy_B(
                smem_copy_atom_QK, tiled_mma_qk
            ).get_slice(tidx)
            if const_expr(split_pv_warps):
                tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
                tSrK = thr_mma_qk.make_fragment_B(
                    thr_mma_qk.partition_B(sK[None, None, 0])
                )
                tSsQ = smem_thr_copy_Q.partition_S(sQ)
                tSsK = smem_thr_copy_K.partition_S(sK)
                smem_store_atom_P = utils.get_smem_store_atom(120, self.dtype)
                smem_thr_store_P = cute.make_tiled_copy_C(
                    smem_store_atom_P, tiled_mma_qk
                ).get_slice(tidx)
                tPsP_store = smem_thr_store_P.partition_D(sP)

        gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
        cute.experimental.iket.range_push("q_load")
        if const_expr(not self.pack_gqa):
            self.load_Q(
                gmem_thr_copy_Q,
                gQ,
                sQ,
                m_block,
                seqlen=seqlen.seqlen_q,
                headdim=mQ.shape[1],
            )
        else:
            PackGQA(
                self.tile_m,
                self.tile_hdim,
                self.check_hdim_oob,
                self.qhead_per_kvhead,
            ).load_Q(
                mQ_cur,
                sQ,
                gmem_tiled_copy_Q,
                tidx,
                m_block,
                seqlen.seqlen_q,
            )
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier(barrier_id=1, number_of_threads=self.num_epilogue_threads)
        cute.experimental.iket.range_pop()

        if const_expr(not split_pv_warps or is_qk_owner):
            softmax = Softmax.create(
                softmax_scale_log2,
                num_rows=acc_O.shape[0][0] * acc_O.shape[1],
                softmax_scale=softmax_scale,
            )
            softmax.reset()
            if const_expr(split_pv_warps):
                mma_params = SimpleNamespace(
                    thr_mma_qk=thr_mma_qk,
                    thr_mma_pv=thr_mma_pv,
                    tSrQ=tSrQ,
                    tSrK=tSrK,
                    tOrV=tOrV,
                    acc_O=acc_O,
                    tOrP=tOrP,
                )
                smem_copy_params = SimpleNamespace(
                    smem_thr_copy_Q=smem_thr_copy_Q,
                    smem_thr_copy_K=smem_thr_copy_K,
                    smem_thr_copy_V=smem_thr_copy_V,
                    tSsQ=tSsQ,
                    tSsK=tSsK,
                    tOsV=tOsV,
                    smem_thr_store_P=smem_thr_store_P,
                    tPsP_store=tPsP_store,
                    smem_thr_copy_P=smem_thr_copy_P,
                    tPsP=tPsP,
                    sRowScale=sRowScale,
                    sLSE=sLSE,
                )
            else:
                mma_params = SimpleNamespace(
                    thr_mma_qk=thr_mma_qk,
                    thr_mma_pv=thr_mma_pv,
                    acc_O=acc_O,
                )
                smem_copy_params = SimpleNamespace(
                    smem_thr_copy_Q=smem_thr_copy_Q,
                    smem_thr_copy_K=smem_thr_copy_K,
                    smem_thr_copy_V=smem_thr_copy_V,
                    sQ=sQ,
                    sK=sK,
                    sV=sV,
                )
            mask = AttentionMask(
                self.tile_m,
                self.tile_n,
                seqlen,
                block_info.window_size_left,
                block_info.window_size_right,
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            )
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_data=aux_data,
                fastdiv_mods=(
                    fastdiv_mods if const_expr(self.mask_mod is not None) else None
                ),
            )
        else:
            softmax = None
            mma_params = SimpleNamespace(
                thr_mma_pv=thr_mma_pv,
                tOrV=tOrV,
                acc_O=acc_O,
                tOrP=tOrP,
            )
            smem_copy_params = SimpleNamespace(
                smem_thr_copy_V=smem_thr_copy_V,
                tOsV=tOsV,
                smem_thr_copy_P=smem_thr_copy_P,
                tPsP=tPsP,
                sRowScale=sRowScale,
                sLSE=sLSE,
            )
            mask_fn = None
        if const_expr(split_pv_warps):
            compute_one_n_block = (
                self.compute_one_n_block_split_pv_owner
                if const_expr(is_qk_owner)
                else self.compute_one_n_block_split_pv_helper
            )
        else:
            compute_one_n_block = self.compute_one_n_block
        if const_expr(not split_pv_warps or is_qk_owner):
            mask_fn_seqlen = partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True)
            mask_fn_no_seqlen = partial(
                mask_fn, mask_mod=self.mask_mod, mask_seqlen=False
            )
        else:
            mask_fn_seqlen = None
            mask_fn_no_seqlen = None
        cute.experimental.iket.range_push("mainloop")
        n_block = cutlass.max(n_block_max - 1, 0)
        consumer_state = compute_one_n_block(
            n_block,
            consumer_state,
            mma_params,
            smem_copy_params,
            softmax,
            pipeline_k,
            pipeline_v,
            score_mod=self.score_mod,
            batch_idx=batch_idx,
            head_idx=head_idx,
            m_block=m_block,
            seqlen=seqlen,
            aux_data=aux_data,
            fastdiv_mods=fastdiv_mods,
            mask_fn=mask_fn_seqlen,
            is_first_n_block=True,
        )
        n_block_upper = n_block
        if const_expr(self.is_causal or self.is_local):
            n_block_min_causal_local_mask = (
                block_info.get_n_block_min_causal_local_mask(
                    seqlen, m_block, n_block_min
                )
            )
            for n_tile in cutlass.range(
                n_block_max - 1 - n_block_min_causal_local_mask, unroll=1
            ):
                n_block = n_block_max - 2 - n_tile
                consumer_state = compute_one_n_block(
                    n_block,
                    consumer_state,
                    mma_params,
                    smem_copy_params,
                    softmax,
                    pipeline_k,
                    pipeline_v,
                    score_mod=self.score_mod,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=m_block,
                    seqlen=seqlen,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                    mask_fn=mask_fn_seqlen,
                )
            n_block_upper = cutlass.min(n_block_upper, n_block_min_causal_local_mask)
        n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
            seqlen, m_block, n_block_min
        )
        for n_tile in cutlass.range(
            n_block_upper - n_block_min_before_local_mask, unroll=1
        ):
            consumer_state = compute_one_n_block(
                n_block_upper - n_tile - 1,
                consumer_state,
                mma_params,
                smem_copy_params,
                softmax,
                pipeline_k,
                pipeline_v,
                score_mod=self.score_mod,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
                mask_fn=mask_fn_no_seqlen,
            )
        if const_expr(self.is_local and block_info.window_size_left is not None):
            n_block_upper = cutlass.min(n_block_upper, n_block_min_before_local_mask)
            for n_tile in cutlass.range(n_block_upper - n_block_min, unroll=1):
                consumer_state = compute_one_n_block(
                    n_block_upper - n_tile - 1,
                    consumer_state,
                    mma_params,
                    smem_copy_params,
                    softmax,
                    pipeline_k,
                    pipeline_v,
                    score_mod=self.score_mod,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=m_block,
                    seqlen=seqlen,
                    aux_data=aux_data,
                    fastdiv_mods=fastdiv_mods,
                    mask_fn=mask_fn_no_seqlen,
                )

        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("finalize_epilogue")
        if const_expr(split_pv_warps):
            if const_expr(is_qk_owner):
                sink_val = None
                if const_expr(learnable_sink is not None):
                    if const_expr(not self.pack_gqa):
                        sink_val = Float32(learnable_sink[head_idx])
                    else:
                        sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                        tScS_mn_finalize = layout_utils.reshape_acc_to_mn(
                            thr_mma_qk.partition_C(cS)
                        )
                        for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                            row = m_block * self.tile_m + tScS_mn_finalize[r][0]
                            q_head_idx = (
                                row % self.qhead_per_kvhead
                                + head_idx * self.qhead_per_kvhead
                            )
                            sink_val[r] = Float32(learnable_sink[q_head_idx])
                row_scale_qk = softmax.finalize(sink_val=sink_val)
                cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                tScS_mn_finalize = layout_utils.reshape_acc_to_mn(
                    thr_mma_qk.partition_C(cS)
                )
                if tScS_mn_finalize[0][1] == 0:
                    for r in cutlass.range(cute.size(row_scale_qk), unroll_full=True):
                        row = tScS_mn_finalize[r][0]
                        sRowScale[row] = row_scale_qk[r]
                        sLSE[row] = softmax.row_sum[r]
                cute.arch.fence_view_async_shared()
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PFull),
                number_of_threads=self.num_mma_threads,
            )
            cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
            tOcO_mn_finalize = layout_utils.reshape_acc_to_mn(
                thr_mma_pv.partition_C(cO)
            )
            num_rows_pv = acc_O.shape[0][0] * acc_O.shape[1]
            row_scale_pv = cute.make_rmem_tensor(num_rows_pv, Float32)
            lse = cute.make_rmem_tensor(num_rows_pv, Float32)
            for r in cutlass.range(cute.size(row_scale_pv), unroll_full=True):
                row = tOcO_mn_finalize[r, 0][0]
                row_scale_pv[r] = sRowScale[row]
                lse[r] = sLSE[row]
            self._rescale_O(acc_O, row_scale_pv)
        else:
            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:
                    sink_val = cute.make_rmem_tensor_like(softmax.row_max, Float32)
                    cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
                    tScS_mn_finalize = layout_utils.reshape_acc_to_mn(
                        thr_mma_qk.partition_C(cS)
                    )
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn_finalize[r][0]
                        q_head_idx = (
                            row % self.qhead_per_kvhead
                            + head_idx * self.qhead_per_kvhead
                        )
                        sink_val[r] = Float32(learnable_sink[q_head_idx])
            row_scale = softmax.finalize(sink_val=sink_val)
            softmax.rescale_O(acc_O, row_scale)
            lse = softmax.row_sum

        self.epilogue(
            acc_O,
            lse,
            mO,
            mLSE,
            sO,
            seqlen,
            gmem_tiled_copy_O,
            None,
            tiled_mma_pv,
            tidx,
            m_block,
            head_idx,
            batch_idx,
        )
        cute.experimental.iket.range_pop()
        return consumer_state

    @cute.jit
    def compute_one_n_block_split_pv_owner(
        self,
        n_block: Int32,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        score_mod: Callable | None,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """Compute QK/softmax in four owner warps, then join eight PV warps."""
        cute.experimental.iket.range_push("k_wait")
        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C(
            (self.tile_m, self.tile_n)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        k_wait_token = pipeline_k.consumer_try_wait(consumer_state)
        pipeline_k.consumer_wait(consumer_state, k_wait_token)
        cute.experimental.iket.range_pop()

        cute.experimental.iket.range_push("qk_mma")
        self._gemm_qk(
            mma_params.thr_mma_qk,
            acc_S,
            mma_params.tSrQ,
            mma_params.tSrK,
            smem_copy_params.tSsQ,
            smem_copy_params.tSsK[None, None, None, consumer_state.index],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
        )
        cute.experimental.iket.range_pop()

        cute.experimental.iket.range_push("softmax")
        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP_qk = layout_utils.reshape_acc_to_frgA(rP)
        tPrP = smem_copy_params.smem_thr_store_P.retile(tOrP_qk)
        cute.copy(
            smem_copy_params.smem_thr_store_P,
            tPrP,
            smem_copy_params.tPsP_store,
        )
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = layout_utils.reshape_acc_to_mn(mma_params.thr_mma_qk.partition_C(cS))
        self._publish_row_scale(row_scale, tScS_mn, smem_copy_params.sRowScale)
        cute.arch.fence_view_async_shared()
        cute.experimental.iket.range_pop()

        return self._compute_one_n_block_split_pv_common(
            consumer_state,
            mma_params,
            smem_copy_params,
            pipeline_k,
            pipeline_v,
            True,
        )

    @cute.jit
    def compute_one_n_block_split_pv_helper(
        self,
        n_block: Int32,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: None,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        score_mod: Callable | None,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        """Join the owner-published P tile using the four helper PV warps."""
        return self._compute_one_n_block_split_pv_common(
            consumer_state,
            mma_params,
            smem_copy_params,
            pipeline_k,
            pipeline_v,
            False,
        )

    @cute.jit
    def _compute_one_n_block_split_pv_common(
        self,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        release_k: cutlass.Constexpr[bool],
    ):
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PFull),
            number_of_threads=self.num_mma_threads,
        )
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        tOcO_mn = layout_utils.reshape_acc_to_mn(mma_params.thr_mma_pv.partition_C(cO))
        num_rows_pv = mma_params.acc_O.shape[0][0] * mma_params.acc_O.shape[1]
        row_scale_pv = cute.make_rmem_tensor(num_rows_pv, Float32)
        for r in cutlass.range(cute.size(row_scale_pv), unroll_full=True):
            row_scale_pv[r] = smem_copy_params.sRowScale[tOcO_mn[r, 0][0]]
        self._rescale_O(mma_params.acc_O, row_scale_pv)

        cute.experimental.iket.range_push("v_wait")
        v_wait_token = pipeline_v.consumer_try_wait(consumer_state)
        pipeline_v.consumer_wait(consumer_state, v_wait_token)
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("pv_mma")
        tOrP_copy_view = smem_copy_params.smem_thr_copy_P.retile(mma_params.tOrP)
        cute.copy(
            smem_copy_params.smem_thr_copy_P,
            smem_copy_params.tPsP,
            tOrP_copy_view,
        )
        self._gemm_pv(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            mma_params.tOrP,
            mma_params.tOrV,
            smem_copy_params.tOsV[None, None, None, consumer_state.index],
            smem_copy_params.smem_thr_copy_V,
        )
        pipeline_v.consumer_release(consumer_state)
        cute.experimental.iket.range_pop()

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PEmpty),
            number_of_threads=self.num_mma_threads,
        )
        if const_expr(release_k):
            pipeline_k.consumer_release(consumer_state)
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def _publish_row_scale(
        self,
        row_scale: cute.Tensor,
        row_coords: cute.Tensor,
        sRowScale: cute.Tensor,
    ):
        """Have one QK lane group publish each row without escaping state."""
        if row_coords[0][1] == 0:
            for r in cutlass.range(cute.size(row_scale), unroll_full=True):
                sRowScale[row_coords[r][0]] = row_scale[r]

    @cute.jit
    def _rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor):
        """Apply a shared-published row scale without carrying softmax state."""
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        for r in cutlass.range(cute.size(row_scale), unroll_full=True):
            acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])

    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        consumer_state: PipelineState,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        pipeline_k: PipelineAsync,
        pipeline_v: PipelineAsync,
        score_mod: Callable | None,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ):
        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C(
            (self.tile_m, self.tile_n)
        )
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        cute.experimental.iket.range_push("k_wait")
        k_wait_token = pipeline_k.consumer_try_wait(consumer_state)
        pipeline_k.consumer_wait(consumer_state, k_wait_token)
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("qk_mma")
        self._gemm_qk_phase_local(
            mma_params.thr_mma_qk,
            acc_S,
            smem_copy_params.sQ,
            smem_copy_params.sK[None, None, consumer_state.index],
            smem_copy_params.smem_thr_copy_Q,
            smem_copy_params.smem_thr_copy_K,
        )
        pipeline_k.consumer_release(consumer_state)
        cute.experimental.iket.range_pop()

        cute.experimental.iket.range_push("softmax")
        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=check_inf
        )
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)
        cute.experimental.iket.range_pop()

        cute.experimental.iket.range_push("v_wait")
        v_wait_token = pipeline_v.consumer_try_wait(consumer_state)
        pipeline_v.consumer_wait(consumer_state, v_wait_token)
        cute.experimental.iket.range_pop()
        cute.experimental.iket.range_push("pv_mma")
        self._gemm_pv_phase_local(
            mma_params.thr_mma_pv,
            mma_params.acc_O,
            tOrP,
            smem_copy_params.sV[None, None, consumer_state.index],
            smem_copy_params.smem_thr_copy_V,
        )
        pipeline_v.consumer_release(consumer_state)
        cute.experimental.iket.range_pop()
        consumer_state.advance()
        return consumer_state

    @cute.jit
    def _gemm_qk_a_in_regs(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrQ: cute.Tensor,
        sK: cute.Tensor,
        smem_thr_copy_K: cute.TiledCopy,
    ):
        """Issue QK with the full Q fragment resident across N blocks."""
        tCrK = tiled_mma.make_fragment_B(tiled_mma.partition_B(sK))
        tCsK = smem_thr_copy_K.partition_S(sK)
        tCrK_copy_view = smem_thr_copy_K.retile(tCrK)
        cute.copy(
            smem_thr_copy_K,
            tCsK[None, None, 0],
            tCrK_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tCsK.shape[2])):
            if k < cute.size(tCsK.shape[2]) - 1:
                cute.copy(
                    smem_thr_copy_K,
                    tCsK[None, None, k + 1],
                    tCrK_copy_view[None, None, k + 1],
                )
            cute.gemm(
                tiled_mma,
                acc,
                tCrQ[None, None, k],
                tCrK[None, None, k],
                acc,
            )

    @cute.jit
    def _gemm_qk_phase_local(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        smem_thr_copy_Q: cute.TiledCopy,
        smem_thr_copy_K: cute.TiledCopy,
    ):
        """Allocate Q/K fragments only for the QK phase."""
        tCrQ = tiled_mma.make_fragment_A(tiled_mma.partition_A(sQ))
        tCrK = tiled_mma.make_fragment_B(tiled_mma.partition_B(sK))
        self._gemm_qk(
            tiled_mma,
            acc,
            tCrQ,
            tCrK,
            smem_thr_copy_Q.partition_S(sQ),
            smem_thr_copy_K.partition_S(sK),
            smem_thr_copy_Q,
            smem_thr_copy_K,
        )

    @cute.jit
    def _gemm_pv_phase_local(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrP: cute.Tensor,
        sV: cute.Tensor,
        smem_thr_copy_V: cute.TiledCopy,
    ):
        """Allocate the V fragment only for the PV phase."""
        tCrV = tiled_mma.make_fragment_B(tiled_mma.partition_B(sV))
        self._gemm_pv(
            tiled_mma,
            acc,
            tCrP,
            tCrV,
            smem_thr_copy_V.partition_S(sV),
            smem_thr_copy_V,
        )

    @cute.jit
    def _gemm_qk(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrQ: cute.Tensor,
        tCrK: cute.Tensor,
        tCsQ: cute.Tensor,
        tCsK: cute.Tensor,
        smem_thr_copy_Q: cute.TiledCopy,
        smem_thr_copy_K: cute.TiledCopy,
    ):
        """Issue the SM120 QK warp-MMA mainloop."""
        tCrQ_copy_view = smem_thr_copy_Q.retile(tCrQ)
        tCrK_copy_view = smem_thr_copy_K.retile(tCrK)
        cute.copy(smem_thr_copy_Q, tCsQ[None, None, 0], tCrQ_copy_view[None, None, 0])
        cute.copy(smem_thr_copy_K, tCsK[None, None, 0], tCrK_copy_view[None, None, 0])
        for k in cutlass.range_constexpr(cute.size(tCsQ.shape[2])):
            if k < cute.size(tCsQ.shape[2]) - 1:
                cute.copy(
                    smem_thr_copy_Q,
                    tCsQ[None, None, k + 1],
                    tCrQ_copy_view[None, None, k + 1],
                )
                cute.copy(
                    smem_thr_copy_K,
                    tCsK[None, None, k + 1],
                    tCrK_copy_view[None, None, k + 1],
                )
            cute.gemm(
                tiled_mma,
                acc,
                tCrQ[None, None, k],
                tCrK[None, None, k],
                acc,
            )

    @cute.jit
    def _gemm_pv(
        self,
        tiled_mma: cute.TiledMma,
        acc: cute.Tensor,
        tCrP: cute.Tensor,
        tCrV: cute.Tensor,
        tCsV: cute.Tensor,
        smem_thr_copy_V: cute.TiledCopy,
    ):
        """Issue the SM120 PV warp-MMA mainloop."""
        tCrV_copy_view = smem_thr_copy_V.retile(tCrV)
        for k in cutlass.range_constexpr(cute.size(tCrP.shape[2])):
            cute.copy(
                smem_thr_copy_V,
                tCsV[None, None, k],
                tCrV_copy_view[None, None, k],
            )
            cute.gemm(
                tiled_mma,
                acc,
                tCrP[None, None, k],
                tCrV[None, None, k],
                acc,
            )

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)
        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
