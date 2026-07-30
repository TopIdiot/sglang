"""
Test WelmV4FusedRMSNorm (Triton) against RMSNorm (reference) for numerical consistency.

Covers all usage patterns found in welmv4.py:
  1. qk_norm: 3D input (tokens, num_heads, head_dim), no residual
  2. input_layernorm: 2D input with residual, optional residual_after_layernorm
  3. post_attention_layernorm: 2D input with residual + clone_fp32_out
"""

import itertools
import os
import unittest
from unittest import mock

import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.welmv4_op import (
    WELM_QKV_DIRECT_MAX_ROWS,
    WelmV4FusedRMSNorm,
    welm_rmsnorm_use_pdl,
)


class TestWelmRMSNormPDLDispatch(unittest.TestCase):
    _FEATURE_ON = {
        "SGLANG_WELM_V45_80A3_FUSED_PRE_ATTN": "1",
        "SGLANG_WELM_RMSNORM_PDL": "1",
    }

    def test_large_m_boundary_tracks_mk(self):
        if WELM_QKV_DIRECT_MAX_ROWS is None:
            self.skipTest("matching MK route metadata is unavailable")
        with mock.patch.dict(os.environ, self._FEATURE_ON):
            self.assertFalse(welm_rmsnorm_use_pdl(WELM_QKV_DIRECT_MAX_ROWS))
            self.assertTrue(welm_rmsnorm_use_pdl(WELM_QKV_DIRECT_MAX_ROWS + 1))

    def test_kill_switch_disables_both_routes(self):
        large_rows = (
            WELM_QKV_DIRECT_MAX_ROWS + 1
            if WELM_QKV_DIRECT_MAX_ROWS is not None
            else 16384
        )
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_WELM_V45_80A3_FUSED_PRE_ATTN": "1",
                "SGLANG_WELM_RMSNORM_PDL": "0",
            },
        ):
            self.assertFalse(welm_rmsnorm_use_pdl(1))
            self.assertFalse(welm_rmsnorm_use_pdl(large_rows))

    def test_feature_gate_disables_pdl(self):
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_WELM_V45_80A3_FUSED_PRE_ATTN": "0",
                "SGLANG_WELM_RMSNORM_PDL": "1",
            },
        ):
            self.assertFalse(welm_rmsnorm_use_pdl(1))
            self.assertFalse(welm_rmsnorm_use_pdl(16384))

    def test_missing_mk_metadata_disables_only_large_m_pdl(self):
        with mock.patch.dict(os.environ, self._FEATURE_ON), mock.patch(
            "sglang.srt.layers.welmv4_op.WELM_QKV_DIRECT_MAX_ROWS", None
        ):
            self.assertTrue(welm_rmsnorm_use_pdl(1))
            self.assertFalse(welm_rmsnorm_use_pdl(16384))


class TestWelmV4FusedRMSNormVsRMSNorm(unittest.TestCase):

    DTYPES = [torch.float16, torch.bfloat16]
    SEEDS = [0, 42]
    EPS_VALUES = [1e-6, 1e-5]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    @staticmethod
    def _reference_rmsnorm(x, weight, eps):
        """Pure PyTorch RMSNorm — matches RMSNorm.forward_native logic."""
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        x = (x * weight.float()).to(orig_dtype)
        return x

    def _make_shared_modules(self, hidden_size, eps=1e-6):
        ref = RMSNorm(hidden_size, eps=eps)
        fused = WelmV4FusedRMSNorm(hidden_size, eps=eps)
        with torch.no_grad():
            weight = torch.randn(hidden_size)
            weight = weight / weight.norm() + 1.0
            ref.weight.copy_(weight)
            fused.weight.copy_(weight)
        return ref.cuda(), fused.cuda()

    # ------------------------------------------------------------------
    # 1) No residual — 2D input (tokens, hidden_size)
    #    Mimics: final norm, or any standalone norm call
    # ------------------------------------------------------------------
    def _run_no_residual_2d(self, num_tokens, hidden_size, dtype, eps, seed):
        torch.manual_seed(seed)
        ref, fused = self._make_shared_modules(hidden_size, eps)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale

        with torch.inference_mode():
            ref_out = ref.forward_native(x.clone())
            fused_out, _ = fused.forward_cuda(x.clone())

        torch.testing.assert_close(
            fused_out,
            ref_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"no_residual_2d failed: tokens={num_tokens}, hidden={hidden_size}, "
            f"dtype={dtype}, eps={eps}, seed={seed}",
        )

    def test_no_residual_2d(self):
        num_tokens_list = [1, 7, 64, 512]
        hidden_sizes = [128, 768, 1024, 4096, 5120, 7168, 8192]
        for params in itertools.product(
            num_tokens_list, hidden_sizes, self.DTYPES, self.EPS_VALUES, self.SEEDS
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                dtype=params[2],
                eps=params[3],
                seed=params[4],
            ):
                self._run_no_residual_2d(*params)

    # ------------------------------------------------------------------
    # 2) No residual — 3D input (tokens, num_heads, head_dim)
    #    Mimics: q_norm / k_norm in Qwen2MoeAttention
    # ------------------------------------------------------------------
    def _run_no_residual_3d(self, num_tokens, num_heads, head_dim, dtype, eps, seed):
        torch.manual_seed(seed)
        ref, fused = self._make_shared_modules(head_dim, eps)

        x = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device="cuda")

        with torch.inference_mode():
            ref_out = ref.forward_native(x.clone())
            fused_out, _ = fused.forward_cuda(x.clone())

        torch.testing.assert_close(
            fused_out,
            ref_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"no_residual_3d failed: tokens={num_tokens}, heads={num_heads}, "
            f"head_dim={head_dim}, dtype={dtype}",
        )

    def test_no_residual_3d_qk_norm(self):
        """Simulates q_norm / k_norm per-head normalization."""
        num_tokens_list = [1, 16, 128]
        num_heads_list = [1, 8, 32, 64]
        head_dims = [64, 128, 192, 256]
        for params in itertools.product(
            num_tokens_list,
            num_heads_list,
            head_dims,
            self.DTYPES,
            [1e-5],
            [0],
        ):
            with self.subTest(
                num_tokens=params[0],
                num_heads=params[1],
                head_dim=params[2],
                dtype=params[3],
                eps=params[4],
                seed=params[5],
            ):
                self._run_no_residual_3d(*params)

    # ------------------------------------------------------------------
    # 3) With residual, residual_after_layernorm=False (default)
    #    Mimics: input_layernorm(hidden_states, residual)
    # ------------------------------------------------------------------
    def _run_with_residual(self, num_tokens, hidden_size, dtype, eps, seed):
        torch.manual_seed(seed)
        ref, fused = self._make_shared_modules(hidden_size, eps)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale
        residual = torch.randn_like(x) * scale

        with torch.inference_mode():
            ref_out, ref_residual = ref.forward_native(x.clone(), residual.clone())
            fused_out, fused_residual = fused.forward_cuda(
                x.clone(), residual.clone(), residual_after_layernorm=False
            )

        torch.testing.assert_close(
            fused_out,
            ref_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"with_residual output failed: tokens={num_tokens}, hidden={hidden_size}, "
            f"dtype={dtype}",
        )
        torch.testing.assert_close(
            fused_residual,
            ref_residual,
            atol=1e-2,
            rtol=1e-2,
            msg=f"with_residual residual failed: tokens={num_tokens}, hidden={hidden_size}, "
            f"dtype={dtype}",
        )

    def test_with_residual(self):
        num_tokens_list = [1, 7, 64, 512]
        hidden_sizes = [128, 768, 1024, 4096, 7168, 8192]
        for params in itertools.product(
            num_tokens_list, hidden_sizes, self.DTYPES, self.EPS_VALUES, self.SEEDS
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                dtype=params[2],
                eps=params[3],
                seed=params[4],
            ):
                self._run_with_residual(*params)

    # ------------------------------------------------------------------
    # 4) With residual, residual_after_layernorm=True
    #    Mimics: input_layernorm(hidden_states, residual,
    #            residual_after_layernorm=True)  (ppln mode)
    # ------------------------------------------------------------------
    def _run_residual_after_layernorm(
        self, num_tokens, hidden_size, dtype, eps, seed
    ):
        torch.manual_seed(seed)
        ref, fused = self._make_shared_modules(hidden_size, eps)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale
        residual = torch.randn_like(x) * scale

        with torch.inference_mode():
            ref_out, _ = ref.forward_native(x.clone(), residual.clone())
            fused_out, fused_residual = fused.forward_cuda(
                x.clone(), residual.clone(), residual_after_layernorm=True
            )

        torch.testing.assert_close(
            fused_out,
            ref_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"residual_after_layernorm output failed: tokens={num_tokens}, "
            f"hidden={hidden_size}, dtype={dtype}",
        )
        torch.testing.assert_close(
            fused_residual,
            fused_out,
            atol=0,
            rtol=0,
            msg="residual_after_layernorm: residual should equal normed output",
        )

    def test_residual_after_layernorm(self):
        num_tokens_list = [1, 7, 64, 512]
        hidden_sizes = [128, 1024, 4096, 7168]
        for params in itertools.product(
            num_tokens_list, hidden_sizes, self.DTYPES, [1e-6], self.SEEDS
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                dtype=params[2],
                eps=params[3],
                seed=params[4],
            ):
                self._run_residual_after_layernorm(*params)

    # ------------------------------------------------------------------
    # 5) residual_after_layernorm=True without existing residual
    #    (first decoder layer, residual=None)
    # ------------------------------------------------------------------
    def _run_residual_after_layernorm_no_input_residual(
        self, num_tokens, hidden_size, dtype, eps, seed
    ):
        torch.manual_seed(seed)
        ref, fused = self._make_shared_modules(hidden_size, eps)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale

        with torch.inference_mode():
            ref_out = ref.forward_native(x.clone())
            fused_out, fused_residual = fused.forward_cuda(
                x.clone(), residual=None, residual_after_layernorm=True
            )

        torch.testing.assert_close(
            fused_out,
            ref_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"residual_after_layernorm (no input residual) output failed",
        )
        torch.testing.assert_close(
            fused_residual,
            fused_out,
            atol=0,
            rtol=0,
            msg="residual_after_layernorm: residual should equal normed output "
            "even when no input residual",
        )

    def test_residual_after_layernorm_no_input_residual(self):
        num_tokens_list = [1, 32, 256]
        hidden_sizes = [128, 4096, 7168]
        for params in itertools.product(
            num_tokens_list, hidden_sizes, self.DTYPES, [1e-6], [0]
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                dtype=params[2],
                eps=params[3],
                seed=params[4],
            ):
                self._run_residual_after_layernorm_no_input_residual(*params)

    # ------------------------------------------------------------------
    # 6) clone_fp32_out=True (with residual)
    #    Mimics: post_attention_layernorm(hidden_states, residual,
    #            clone_fp32_out=True)
    # ------------------------------------------------------------------
    def _run_clone_fp32_out(self, num_tokens, hidden_size, dtype, eps, seed):
        torch.manual_seed(seed)
        ref, fused = self._make_shared_modules(hidden_size, eps)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale
        residual = torch.randn_like(x) * scale

        with torch.inference_mode():
            ref_out, ref_residual = ref.forward_native(x.clone(), residual.clone())
            fused_out, fused_residual, fp32_out = fused.forward_cuda(
                x.clone(), residual.clone(), clone_fp32_out=True
            )

        torch.testing.assert_close(
            fused_out,
            ref_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"clone_fp32_out output failed",
        )
        torch.testing.assert_close(
            fused_residual,
            ref_residual,
            atol=1e-2,
            rtol=1e-2,
            msg=f"clone_fp32_out residual failed",
        )
        self.assertEqual(fp32_out.dtype, torch.float32)
        torch.testing.assert_close(
            fp32_out,
            fused_out.float(),
            atol=1e-2,
            rtol=1e-2,
            msg="clone_fp32_out: fp32 copy should match the normed output in float32",
        )
        ref_fp32 = self._reference_rmsnorm(
            (x + residual).float(), fused.weight, eps
        )
        torch.testing.assert_close(
            fp32_out,
            ref_fp32,
            atol=1e-2,
            rtol=1e-2,
            msg="clone_fp32_out: fp32 copy should match reference fp32 computation",
        )

    def test_clone_fp32_out(self):
        num_tokens_list = [1, 7, 64, 512]
        hidden_sizes = [128, 1024, 4096, 7168]
        for params in itertools.product(
            num_tokens_list, hidden_sizes, self.DTYPES, [1e-6], self.SEEDS
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                dtype=params[2],
                eps=params[3],
                seed=params[4],
            ):
                self._run_clone_fp32_out(*params)

    # ------------------------------------------------------------------
    # 7) clone_fp32_out=True without residual
    # ------------------------------------------------------------------
    def _run_clone_fp32_out_no_residual(
        self, num_tokens, hidden_size, dtype, eps, seed
    ):
        torch.manual_seed(seed)
        ref, fused = self._make_shared_modules(hidden_size, eps)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale

        with torch.inference_mode():
            ref_out = ref.forward_native(x.clone())
            fused_out, _, fp32_out = fused.forward_cuda(
                x.clone(), clone_fp32_out=True
            )

        torch.testing.assert_close(
            fused_out,
            ref_out,
            atol=1e-2,
            rtol=1e-2,
            msg="clone_fp32_out (no residual) output failed",
        )
        self.assertEqual(fp32_out.dtype, torch.float32)
        ref_fp32 = self._reference_rmsnorm(x.float(), fused.weight, eps)
        torch.testing.assert_close(
            fp32_out,
            ref_fp32,
            atol=1e-2,
            rtol=1e-2,
            msg="clone_fp32_out (no residual): fp32 copy should match reference",
        )

    def test_clone_fp32_out_no_residual(self):
        num_tokens_list = [1, 32, 256]
        hidden_sizes = [128, 4096, 7168]
        for params in itertools.product(
            num_tokens_list, hidden_sizes, self.DTYPES, [1e-6], [0]
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                dtype=params[2],
                eps=params[3],
                seed=params[4],
            ):
                self._run_clone_fp32_out_no_residual(*params)

    # ------------------------------------------------------------------
    # 8) Non-power-of-2 hidden sizes
    # ------------------------------------------------------------------
    def test_non_power_of_2_hidden_sizes(self):
        hidden_sizes = [769, 770, 771, 5121, 5125, 8199]
        for hidden_size in hidden_sizes:
            for dtype in self.DTYPES:
                with self.subTest(hidden_size=hidden_size, dtype=dtype):
                    self._run_no_residual_2d(32, hidden_size, dtype, 1e-6, 0)
                    self._run_with_residual(32, hidden_size, dtype, 1e-6, 0)

    # ------------------------------------------------------------------
    # 9) Large batch stress test
    # ------------------------------------------------------------------
    def test_large_batch(self):
        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                self._run_no_residual_2d(4096, 4096, dtype, 1e-6, 0)
                self._run_with_residual(4096, 4096, dtype, 1e-6, 0)

    # ------------------------------------------------------------------
    # 10) Weight values edge cases: all-ones, near-zero, large
    # ------------------------------------------------------------------
    def _run_weight_edge_case(self, weight_init, hidden_size, dtype):
        eps = 1e-6
        ref = RMSNorm(hidden_size, eps=eps).cuda()
        fused = WelmV4FusedRMSNorm(hidden_size, eps=eps).cuda()

        with torch.no_grad():
            if weight_init == "ones":
                w = torch.ones(hidden_size)
            elif weight_init == "small":
                w = torch.full((hidden_size,), 1e-4)
            elif weight_init == "large":
                w = torch.full((hidden_size,), 100.0)
            elif weight_init == "random":
                w = torch.randn(hidden_size) * 2.0
            else:
                raise ValueError(f"Unknown weight_init: {weight_init}")
            ref.weight.copy_(w)
            fused.weight.copy_(w)

        x = torch.randn(32, hidden_size, dtype=dtype, device="cuda")

        with torch.inference_mode():
            ref_out = ref.forward_native(x.clone())
            fused_out, _ = fused.forward_cuda(x.clone())

        torch.testing.assert_close(
            fused_out,
            ref_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"weight_edge_case={weight_init} failed",
        )

    def test_weight_edge_cases(self):
        for weight_init in ["ones", "small", "large", "random"]:
            for hidden_size in [128, 4096]:
                for dtype in self.DTYPES:
                    with self.subTest(
                        weight_init=weight_init,
                        hidden_size=hidden_size,
                        dtype=dtype,
                    ):
                        self._run_weight_edge_case(weight_init, hidden_size, dtype)

    # ------------------------------------------------------------------
    # 11) Input magnitude edge cases: near-zero, very large
    # ------------------------------------------------------------------
    def _run_input_magnitude(self, magnitude, hidden_size, dtype):
        torch.manual_seed(0)
        ref, fused = self._make_shared_modules(hidden_size, eps=1e-6)

        if magnitude == "near_zero":
            x = torch.randn(32, hidden_size, dtype=dtype, device="cuda") * 1e-6
        elif magnitude == "large":
            x = torch.randn(32, hidden_size, dtype=dtype, device="cuda") * 100.0
        elif magnitude == "mixed":
            x = torch.randn(32, hidden_size, dtype=dtype, device="cuda")
            x[:, : hidden_size // 2] *= 1e-5
            x[:, hidden_size // 2 :] *= 100.0
        else:
            raise ValueError(f"Unknown magnitude: {magnitude}")

        with torch.inference_mode():
            ref_out = ref.forward_native(x.clone())
            fused_out, _ = fused.forward_cuda(x.clone())

        torch.testing.assert_close(
            fused_out,
            ref_out,
            atol=1e-1,
            rtol=5e-2,
            msg=f"input_magnitude={magnitude} failed",
        )

    def test_input_magnitude_edge_cases(self):
        for magnitude in ["near_zero", "large", "mixed"]:
            for hidden_size in [128, 4096]:
                for dtype in self.DTYPES:
                    with self.subTest(
                        magnitude=magnitude,
                        hidden_size=hidden_size,
                        dtype=dtype,
                    ):
                        self._run_input_magnitude(magnitude, hidden_size, dtype)

    # ------------------------------------------------------------------
    # 12) Full pipeline: residual + clone_fp32_out + residual_after_layernorm
    # ------------------------------------------------------------------
    def _run_full_combo(
        self, num_tokens, hidden_size, dtype, residual_after_layernorm
    ):
        torch.manual_seed(0)
        ref, fused = self._make_shared_modules(hidden_size, eps=1e-6)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale
        residual = torch.randn_like(x) * scale

        with torch.inference_mode():
            ref_out, ref_residual = ref.forward_native(x.clone(), residual.clone())
            fused_out, fused_residual, fp32_out = fused.forward_cuda(
                x.clone(),
                residual.clone(),
                residual_after_layernorm=residual_after_layernorm,
                clone_fp32_out=True,
            )

        torch.testing.assert_close(
            fused_out,
            ref_out,
            atol=1e-2,
            rtol=1e-2,
            msg=f"full_combo output failed (ral={residual_after_layernorm})",
        )
        if not residual_after_layernorm:
            torch.testing.assert_close(
                fused_residual,
                ref_residual,
                atol=1e-2,
                rtol=1e-2,
                msg="full_combo residual failed",
            )
        else:
            torch.testing.assert_close(
                fused_residual,
                fused_out,
                atol=0,
                rtol=0,
                msg="full_combo: residual_after_layernorm should yield "
                "residual == output",
            )
        self.assertEqual(fp32_out.dtype, torch.float32)

    def test_full_combo(self):
        for params in itertools.product(
            [1, 64, 512],
            [128, 4096, 7168],
            self.DTYPES,
            [False, True],
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                dtype=params[2],
                residual_after_layernorm=params[3],
            ):
                self._run_full_combo(*params)

    # ------------------------------------------------------------------
    # 13) Chained norm calls — simulate two consecutive decoder layers
    #     to verify residual is correctly propagated
    # ------------------------------------------------------------------
    def test_chained_layers(self):
        """Simulate residual flow across two consecutive decoder layers."""
        hidden_size = 4096
        num_tokens = 64
        eps = 1e-6

        for dtype in self.DTYPES:
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                ref1, fused1 = self._make_shared_modules(hidden_size, eps)
                ref2, fused2 = self._make_shared_modules(hidden_size, eps)

                scale = 1 / (2 * hidden_size)
                x = (
                    torch.randn(
                        num_tokens, hidden_size, dtype=dtype, device="cuda"
                    )
                    * scale
                )
                residual = None

                with torch.inference_mode():
                    # --- Reference path ---
                    if residual is None:
                        ref_h1 = ref1.forward_native(x.clone())
                        ref_r1 = x.clone().float().to(dtype)
                    else:
                        ref_h1, ref_r1 = ref1.forward_native(
                            x.clone(), residual.clone()
                        )

                    attn_out = torch.randn_like(ref_h1) * scale
                    ref_h2, ref_r2 = ref2.forward_native(
                        attn_out.clone(), ref_r1.clone()
                    )

                    # --- Fused path ---
                    fused_h1, fused_r1 = fused1.forward_cuda(x.clone())
                    fused_h2, fused_r2 = fused2.forward_cuda(
                        attn_out.clone(), fused_r1.clone()
                    )

                torch.testing.assert_close(
                    fused_h1, ref_h1, atol=1e-2, rtol=1e-2,
                    msg="chained layer 1 output mismatch",
                )
                torch.testing.assert_close(
                    fused_h2, ref_h2, atol=1e-2, rtol=1e-2,
                    msg="chained layer 2 output mismatch",
                )
                torch.testing.assert_close(
                    fused_r2, ref_r2, atol=1e-2, rtol=1e-2,
                    msg="chained layer 2 residual mismatch",
                )

    # ------------------------------------------------------------------
    # 14) Single-token input (batch_size=1)
    # ------------------------------------------------------------------
    def test_single_token(self):
        for hidden_size in [128, 4096, 7168]:
            for dtype in self.DTYPES:
                with self.subTest(hidden_size=hidden_size, dtype=dtype):
                    self._run_no_residual_2d(1, hidden_size, dtype, 1e-6, 0)
                    self._run_with_residual(1, hidden_size, dtype, 1e-6, 0)

    # ------------------------------------------------------------------
    # 15) Verify output shapes are correct
    # ------------------------------------------------------------------
    def test_output_shapes(self):
        hidden_size = 4096
        num_tokens = 32
        dtype = torch.bfloat16
        eps = 1e-6
        fused = WelmV4FusedRMSNorm(hidden_size, eps=eps).cuda()

        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
        residual = torch.randn_like(x)

        with torch.inference_mode():
            out1 = fused.forward_cuda(x.clone())
            self.assertEqual(len(out1), 2)
            self.assertEqual(out1[0].shape, (num_tokens, hidden_size))
            self.assertEqual(out1[0].dtype, dtype)

            out2 = fused.forward_cuda(x.clone(), residual.clone())
            self.assertEqual(len(out2), 2)
            self.assertEqual(out2[0].shape, (num_tokens, hidden_size))
            self.assertEqual(out2[1].shape, (num_tokens, hidden_size))

            out3 = fused.forward_cuda(
                x.clone(), residual.clone(), clone_fp32_out=True
            )
            self.assertEqual(len(out3), 3)
            self.assertEqual(out3[0].shape, (num_tokens, hidden_size))
            self.assertEqual(out3[1].shape, (num_tokens, hidden_size))
            self.assertEqual(out3[2].shape, (num_tokens, hidden_size))
            self.assertEqual(out3[2].dtype, torch.float32)

            out4 = fused.forward_cuda(
                x.clone(),
                residual.clone(),
                residual_after_layernorm=True,
            )
            self.assertEqual(len(out4), 2)

        x3d = torch.randn(num_tokens, 8, 128, dtype=dtype, device="cuda")
        fused_3d = WelmV4FusedRMSNorm(128, eps=eps).cuda()
        with torch.inference_mode():
            out5 = fused_3d.forward_cuda(x3d.clone())
            self.assertEqual(out5[0].shape, (num_tokens, 8, 128))


if __name__ == "__main__":
    unittest.main(verbosity=2)
