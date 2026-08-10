"""
Precision tests for WelmV4 fused ops (welmv4_op.py).

Each fused op is compared against the pre-fusion calling pattern used in welmv4.py,
using SGLang's own native implementations as the reference.

Model config parameters (both configs share the same values):
  hidden_size=2048, head_dim=256, num_attention_heads=24, num_key_value_heads=2,
  qk_rope_head_dim=64, qk_norm=False, k_norm=True, rms_norm_eps=1e-5,
  gated_self_attention_headwise=True, ppln=True

Tolerances: atol=2e-2 (bf16 has ~0.0078 per-ULP precision).
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.layers.welmv4_op import (
    WelmV4FusedRMSNorm,
    WelmV4InplaceRotaryEmbedding,
    inplace_sigmoid_mul,
    mmq_style_add_residual,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.models.welmv4 import (
    _pack_welm_kv_mirror_states,
    _unpack_welm_kv_mirror_states,
    _welm_init_kv_mirror_last_q_indices,
    _welm_kv_mirror_pp_key,
    _welm_scatter_kv_mirror_rows,
    _welm_select_kv_mirror_rows,
    _welm_should_sync_kv_mirror_dp_metadata,
    _welm_update_contracted_dp_metadata,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

torch.random.manual_seed(1)

# bf16 residual paths can differ from PyTorch's reference by a few output ULPs
# because the fused Triton kernel accumulates the residual add in fp32 before
# RMSNorm, while the reference native path follows PyTorch's bf16 behavior.
ATOL = 1e-3
RTOL = 1e-3
RESIDUAL_ATOL = 4e-2
RESIDUAL_RTOL = 1e-2
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Model config parameters (shared by both configs)
HIDDEN_SIZE = 2048
HEAD_DIM = 256
NUM_Q_HEADS = 24
NUM_K_HEADS = 2
QK_ROPE_HEAD_DIM = 64
RMS_NORM_EPS = 1e-5


class TestWelmKvMirrorLastQSelection(CustomTestCase):
    def test_middle_chunk_has_no_active_final_q(self):
        forward_batch = SimpleNamespace(
            extend_seq_lens=torch.tensor([4, 4], dtype=torch.int32),
            welm_kv_mirror_last_q_indices=torch.empty(0, dtype=torch.long),
            welm_kv_mirror_active_batch_indices=torch.empty(0, dtype=torch.long),
            welm_kv_mirror_output_size=2,
        )

        first_contract = _welm_init_kv_mirror_last_q_indices(forward_batch)
        self.assertTrue(first_contract)

        token_states = torch.arange(8 * 2, dtype=torch.float32).view(8, 2)
        selected = _welm_select_kv_mirror_rows(
            token_states, forward_batch, first_contract=first_contract
        )
        self.assertEqual(tuple(selected.shape), (0, 2))

        scattered = _welm_scatter_kv_mirror_rows(selected, forward_batch)
        self.assertEqual(tuple(scattered.shape), (2, 2))
        torch.testing.assert_close(scattered, torch.zeros(2, 2))

    def test_uses_scheduler_supplied_final_q_indices(self):
        forward_batch = SimpleNamespace(
            extend_seq_lens=torch.tensor([4, 4, 4], dtype=torch.int32),
            welm_kv_mirror_last_q_indices=torch.tensor([7], dtype=torch.long),
            welm_kv_mirror_active_batch_indices=torch.tensor([1], dtype=torch.long),
            welm_kv_mirror_output_size=3,
        )

        first_contract = _welm_init_kv_mirror_last_q_indices(forward_batch)
        self.assertTrue(first_contract)
        self.assertEqual(forward_batch.custom_last_index.tolist(), [7])

        token_states = torch.arange(12 * 2, dtype=torch.float32).view(12, 2)
        selected = _welm_select_kv_mirror_rows(
            token_states, forward_batch, first_contract=first_contract
        )
        torch.testing.assert_close(selected, token_states[[7]])

        scattered = _welm_scatter_kv_mirror_rows(selected, forward_batch)
        self.assertEqual(tuple(scattered.shape), (3, 2))
        torch.testing.assert_close(scattered[1], token_states[7])
        torch.testing.assert_close(scattered[0], torch.zeros(2))
        torch.testing.assert_close(scattered[2], torch.zeros(2))

        next_contract = _welm_init_kv_mirror_last_q_indices(forward_batch)
        self.assertFalse(next_contract)
        selected_again = _welm_select_kv_mirror_rows(
            scattered, forward_batch, first_contract=next_contract
        )
        torch.testing.assert_close(selected_again, token_states[[7]])


class TestWelmKvMirrorDpMetadata(CustomTestCase):
    def _make_forward_batch(self, forward_mode: ForwardMode):
        return SimpleNamespace(
            enable_welm_kv_mirror_opt=True,
            forward_mode=forward_mode,
            is_extend_in_batch=True,
            # Decode has 13 real requests but is padded to 14 rows for attn-TP=2.
            global_num_tokens_gpu=torch.tensor([8192, 14], dtype=torch.int64),
            global_num_tokens_cpu=[8192, 14],
            original_global_num_tokens_cpu=[8192, 14],
            global_num_tokens_for_logprob_gpu=torch.tensor(
                [1, 13], dtype=torch.int64
            ),
            global_num_tokens_for_logprob_cpu=[1, 13],
            global_num_reqs_cpu=[1, 13],
            global_forward_modes=[ForwardMode.EXTEND.value, forward_mode.value],
            welm_kv_mirror_contract_flags=[True, False],
            num_token_non_padded=None,
            return_logprob=False,
            scale_seq_factor=1,
            dp_padding_mode=DpPaddingMode.SUM_LEN,
            dp_local_start_pos=torch.tensor(1),
            dp_local_num_tokens=torch.tensor(14),
            global_dp_buffer_len=8206,
        )

    @mock.patch("sglang.srt.models.welmv4.is_dp_attention_enabled", return_value=True)
    def test_decode_and_idle_peers_sync_metadata(self, _):
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE):
            with self.subTest(mode=mode):
                self.assertTrue(
                    _welm_should_sync_kv_mirror_dp_metadata(
                        self._make_forward_batch(mode)
                    )
                )

        self.assertTrue(
            _welm_should_sync_kv_mirror_dp_metadata(
                self._make_forward_batch(ForwardMode.EXTEND)
            )
        )

    @mock.patch(
        "sglang.srt.models.welmv4._welm_cuda_graph_capture_active",
        return_value=False,
    )
    @mock.patch(
        "sglang.srt.model_executor.forward_batch_info.get_attention_tp_size",
        return_value=4,
    )
    @mock.patch(
        "sglang.srt.model_executor.forward_batch_info.get_attention_tp_rank",
        return_value=0,
    )
    @mock.patch(
        "sglang.srt.models.welmv4.welm_use_previous_precision",
        return_value=False,
    )
    @mock.patch("sglang.srt.models.welmv4.is_dp_attention_enabled", return_value=True)
    @mock.patch("sglang.srt.layers.dp_attention.set_is_extend_in_batch")
    @mock.patch("sglang.srt.layers.dp_attention.set_dp_buffer_len")
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_tp_size", return_value=4)
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_tp_rank", return_value=0)
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_dp_rank", return_value=0)
    def test_contracted_cpu_valid_rows_follow_uneven_attntp_split(
        self,
        _dp_rank,
        _tp_rank,
        _tp_size,
        _set_dp_buffer_len,
        _set_is_extend_in_batch,
        _is_dp_attention_enabled,
        _welm_use_previous_precision,
        _forward_batch_tp_rank,
        _forward_batch_tp_size,
        _cuda_graph_capture_active,
    ):
        forward_batch = self._make_forward_batch(ForwardMode.EXTEND)
        forward_batch.global_num_tokens_gpu = torch.tensor([5, 0], dtype=torch.int64)
        forward_batch.global_num_tokens_cpu = [5, 0]
        forward_batch.original_global_num_tokens_cpu = [5, 0]
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            [5, 0], dtype=torch.int64
        )
        forward_batch.global_num_tokens_for_logprob_cpu = [5, 0]
        forward_batch.global_num_reqs_cpu = [5, 0]
        forward_batch.global_forward_modes = [
            ForwardMode.EXTEND.value,
            ForwardMode.IDLE.value,
        ]
        forward_batch.welm_kv_mirror_contract_flags = [True, False]
        forward_batch.num_token_non_padded = torch.tensor(5, dtype=torch.int64)

        _welm_update_contracted_dp_metadata(
            forward_batch,
            5,
            contract_to_request_counts=True,
        )

        self.assertEqual(forward_batch.num_token_non_padded.item(), 2)
        self.assertEqual(forward_batch.num_token_non_padded_cpu, 2)

    @mock.patch(
        "sglang.srt.models.welmv4._welm_cuda_graph_capture_active",
        return_value=True,
    )
    @mock.patch(
        "sglang.srt.models.welmv4.welm_use_previous_precision",
        return_value=False,
    )
    @mock.patch("sglang.srt.models.welmv4.is_dp_attention_enabled", return_value=True)
    @mock.patch("sglang.srt.layers.dp_attention.set_is_extend_in_batch")
    @mock.patch("sglang.srt.layers.dp_attention.set_dp_buffer_len")
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_dp_rank", return_value=1)
    def test_decode_peer_preserves_padded_rows_in_mixed_layout(
        self,
        _,
        set_dp_buffer_len,
        set_is_extend_in_batch,
        _is_dp_attention_enabled,
        _welm_use_previous_precision,
        _cuda_graph_capture_active,
    ):
        forward_batch = self._make_forward_batch(ForwardMode.DECODE)

        _welm_update_contracted_dp_metadata(
            forward_batch,
            14,
            marker_attr="_welm_kv_mirror_contracted_dp_metadata_rows",
            contract_to_request_counts=True,
        )

        self.assertEqual(forward_batch.global_num_tokens_cpu, [1, 14])
        self.assertEqual(forward_batch.global_num_tokens_gpu.tolist(), [1, 14])
        self.assertEqual(forward_batch.global_num_tokens_for_logprob_cpu, [1, 13])
        self.assertEqual(
            forward_batch.global_num_tokens_for_logprob_gpu.tolist(), [1, 13]
        )
        self.assertEqual(forward_batch.global_dp_buffer_len, 15)
        self.assertIsNone(forward_batch.dp_local_start_pos)
        self.assertIsNone(forward_batch.dp_local_num_tokens)
        self.assertEqual(
            forward_batch._welm_kv_mirror_contracted_dp_metadata_rows, 14
        )
        set_dp_buffer_len.assert_called_once_with(15, 14, False, [1, 14])
        set_is_extend_in_batch.assert_called_once_with(True)

    @mock.patch(
        "sglang.srt.models.welmv4._welm_cuda_graph_capture_active",
        return_value=True,
    )
    @mock.patch(
        "sglang.srt.models.welmv4.welm_use_previous_precision",
        return_value=False,
    )
    @mock.patch("sglang.srt.models.welmv4.is_dp_attention_enabled", return_value=True)
    @mock.patch("sglang.srt.layers.dp_attention.set_is_extend_in_batch")
    @mock.patch("sglang.srt.layers.dp_attention.set_dp_buffer_len")
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_dp_rank", return_value=0)
    def test_extend_peer_builds_the_same_mixed_layout(
        self,
        _,
        set_dp_buffer_len,
        set_is_extend_in_batch,
        _is_dp_attention_enabled,
        _welm_use_previous_precision,
        _cuda_graph_capture_active,
    ):
        forward_batch = self._make_forward_batch(ForwardMode.EXTEND)
        forward_batch.global_forward_modes = [
            ForwardMode.EXTEND.value,
            ForwardMode.DECODE.value,
        ]

        _welm_update_contracted_dp_metadata(
            forward_batch,
            1,
            marker_attr="_welm_kv_mirror_contracted_dp_metadata_rows",
            contract_to_request_counts=True,
        )

        self.assertEqual(forward_batch.global_num_tokens_cpu, [1, 14])
        self.assertEqual(forward_batch.global_num_tokens_gpu.tolist(), [1, 14])
        self.assertEqual(forward_batch.global_num_tokens_for_logprob_cpu, [1, 13])
        self.assertEqual(
            forward_batch.global_num_tokens_for_logprob_gpu.tolist(), [1, 13]
        )
        self.assertEqual(forward_batch.global_dp_buffer_len, 15)
        set_dp_buffer_len.assert_called_once_with(15, 1, False, [1, 14])
        set_is_extend_in_batch.assert_called_once_with(True)

    @mock.patch(
        "sglang.srt.models.welmv4._welm_cuda_graph_capture_active",
        return_value=True,
    )
    @mock.patch(
        "sglang.srt.models.welmv4.welm_use_previous_precision",
        return_value=False,
    )
    @mock.patch("sglang.srt.models.welmv4.is_dp_attention_enabled", return_value=True)
    @mock.patch("sglang.srt.layers.dp_attention.set_is_extend_in_batch")
    @mock.patch("sglang.srt.layers.dp_attention.set_dp_buffer_len")
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_dp_rank", return_value=0)
    def test_proposal_graph_contraction_preserves_low_latency_deepep(
        self,
        _,
        _set_dp_buffer_len,
        set_is_extend_in_batch,
        _is_dp_attention_enabled,
        _welm_use_previous_precision,
        _cuda_graph_capture_active,
    ):
        forward_batch = self._make_forward_batch(ForwardMode.DRAFT_EXTEND)
        forward_batch._welm_force_low_latency_deepep = True

        _welm_update_contracted_dp_metadata(
            forward_batch,
            1,
            marker_attr="_welm_kv_mirror_contracted_dp_metadata_rows",
            contract_to_request_counts=True,
        )

        self.assertTrue(forward_batch.is_extend_in_batch)
        set_is_extend_in_batch.assert_called_once_with(False)

    @mock.patch(
        "sglang.srt.models.welmv4._welm_cuda_graph_capture_active",
        return_value=True,
    )
    @mock.patch(
        "sglang.srt.models.welmv4.welm_use_previous_precision",
        return_value=False,
    )
    @mock.patch("sglang.srt.models.welmv4.is_dp_attention_enabled", return_value=True)
    @mock.patch("sglang.srt.layers.dp_attention.set_is_extend_in_batch")
    @mock.patch("sglang.srt.layers.dp_attention.set_dp_buffer_len")
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_dp_rank", return_value=0)
    def test_input_logprob_extend_preserves_rows_while_peer_contracts(
        self,
        _,
        set_dp_buffer_len,
        set_is_extend_in_batch,
        _is_dp_attention_enabled,
        _welm_use_previous_precision,
        _cuda_graph_capture_active,
    ):
        forward_batch = self._make_forward_batch(ForwardMode.EXTEND)
        forward_batch.global_num_tokens_gpu = torch.tensor(
            [8192, 4096], dtype=torch.int64
        )
        forward_batch.global_num_tokens_cpu = [8192, 4096]
        forward_batch.original_global_num_tokens_cpu = [8192, 4096]
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            [1, 1], dtype=torch.int64
        )
        forward_batch.global_num_tokens_for_logprob_cpu = [1, 1]
        forward_batch.global_num_reqs_cpu = [1, 1]
        forward_batch.global_forward_modes = [
            ForwardMode.EXTEND.value,
            ForwardMode.EXTEND.value,
        ]
        forward_batch.welm_kv_mirror_contract_flags = [False, True]
        forward_batch.return_logprob = True
        forward_batch.extend_seq_lens_cpu = [8192]
        forward_batch.extend_logprob_start_lens_cpu = [8191]

        self.assertTrue(
            _welm_should_sync_kv_mirror_dp_metadata(forward_batch)
        )
        _welm_update_contracted_dp_metadata(
            forward_batch,
            8192,
            marker_attr="_welm_kv_mirror_contracted_dp_metadata_rows",
            contract_to_request_counts=True,
        )

        self.assertEqual(forward_batch.global_num_tokens_cpu, [8192, 1])
        self.assertEqual(forward_batch.global_num_tokens_gpu.tolist(), [8192, 1])
        self.assertEqual(forward_batch.global_num_tokens_for_logprob_cpu, [1, 1])
        self.assertEqual(forward_batch.global_dp_buffer_len, 8193)
        set_dp_buffer_len.assert_called_once_with(8193, 8192, False, [8192, 1])
        set_is_extend_in_batch.assert_called_once_with(True)

    @mock.patch(
        "sglang.srt.models.welmv4._welm_cuda_graph_capture_active",
        return_value=True,
    )
    @mock.patch(
        "sglang.srt.models.welmv4.welm_use_previous_precision",
        return_value=False,
    )
    @mock.patch("sglang.srt.models.welmv4.is_dp_attention_enabled", return_value=True)
    @mock.patch("sglang.srt.layers.dp_attention.set_is_extend_in_batch")
    @mock.patch("sglang.srt.layers.dp_attention.set_dp_buffer_len")
    @mock.patch("sglang.srt.layers.dp_attention.get_attention_dp_rank", return_value=1)
    def test_nonshrinking_extend_rank_still_syncs_peer_contraction(
        self,
        _,
        set_dp_buffer_len,
        set_is_extend_in_batch,
        _is_dp_attention_enabled,
        _welm_use_previous_precision,
        _cuda_graph_capture_active,
    ):
        forward_batch = self._make_forward_batch(ForwardMode.EXTEND)
        forward_batch.global_num_tokens_gpu = torch.tensor(
            [8192, 2], dtype=torch.int64
        )
        forward_batch.global_num_tokens_cpu = [8192, 2]
        forward_batch.original_global_num_tokens_cpu = [8192, 2]
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            [1, 2], dtype=torch.int64
        )
        forward_batch.global_num_tokens_for_logprob_cpu = [1, 2]
        forward_batch.global_num_reqs_cpu = [1, 2]
        forward_batch.welm_kv_mirror_contract_flags = [True, True]

        self.assertTrue(_welm_should_sync_kv_mirror_dp_metadata(forward_batch))
        _welm_update_contracted_dp_metadata(
            forward_batch,
            2,
            marker_attr="_welm_kv_mirror_contracted_dp_metadata_rows",
            contract_to_request_counts=True,
        )

        self.assertEqual(forward_batch.global_num_tokens_cpu, [1, 2])
        self.assertEqual(forward_batch.global_num_tokens_gpu.tolist(), [1, 2])
        self.assertEqual(forward_batch.global_dp_buffer_len, 3)
        set_dp_buffer_len.assert_called_once_with(3, 2, False, [1, 2])
        set_is_extend_in_batch.assert_called_once_with(True)


num_tokens_list = [127]


def setUpModule():
    """Set global server args required by RotaryEmbedding init."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


class TestWelmV4KvMirrorPPProxy(CustomTestCase):
    def test_pack_uses_distinct_layer_keys(self):
        states = {
            0: (torch.randn(2, 3), torch.randn(2, 3)),
            15: (torch.randn(1, 3), torch.randn(1, 3)),
        }
        proxy_tensors = {"hidden_states": torch.randn(2, 3)}

        _pack_welm_kv_mirror_states(proxy_tensors, states)

        expected_keys = {
            "hidden_states",
            _welm_kv_mirror_pp_key(0, "k"),
            f"{_welm_kv_mirror_pp_key(0, 'k')}.shape",
            _welm_kv_mirror_pp_key(0, "v"),
            f"{_welm_kv_mirror_pp_key(0, 'v')}.shape",
            _welm_kv_mirror_pp_key(15, "k"),
            f"{_welm_kv_mirror_pp_key(15, 'k')}.shape",
            _welm_kv_mirror_pp_key(15, "v"),
            f"{_welm_kv_mirror_pp_key(15, 'v')}.shape",
        }
        self.assertEqual(set(proxy_tensors), expected_keys)

        unpacked = _unpack_welm_kv_mirror_states(PPProxyTensors(proxy_tensors))
        self.assertEqual(set(unpacked), set(states))
        for layer_idx, (mirror_k, mirror_v) in states.items():
            torch.testing.assert_close(unpacked[layer_idx][0], mirror_k)
            torch.testing.assert_close(unpacked[layer_idx][1], mirror_v)


# =====================================================================
# Test 1: WelmV4FusedRMSNorm vs RMSNorm.forward_native
# =====================================================================
class TestWelmV4FusedRMSNorm(CustomTestCase):
    """
    Pre-fusion code used RMSNorm from sglang.srt.layers.layernorm.
    Calling patterns (from the git diff):

    DecoderLayer (hidden_size=2048, eps=1e-5):
      - input_layernorm:
          Before:  if residual is None:
                       residual = hidden_states
                       hidden_states = self.input_layernorm(hidden_states)
                   else:
                       hidden_states, residual = self.input_layernorm(hidden_states, residual)
          After:   hidden_states, residual = self.input_layernorm(
                       hidden_states, residual, residual_after_layernorm=...)

      - post_attention_layernorm:
          Before:  hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
          After:   hidden_states, residual, hidden_states_fp32 = self.post_attention_layernorm(
                       hidden_states, residual, clone_fp32_out=True)

    Attention k_norm (hidden_size=head_dim=256, eps=1e-5, k_norm=True, qk_norm=False):
      Before:  k_by_head = self.k_norm.forward_native(k_by_head)
      After:   k_by_head, _ = self.k_norm(k_by_head)
    """

    def _make_ref_and_fused(self, hidden_size, eps=RMS_NORM_EPS):
        ref = RMSNorm(hidden_size, eps=eps).to(device=DEVICE, dtype=DTYPE)
        fused = WelmV4FusedRMSNorm(hidden_size, eps=eps).to(device=DEVICE, dtype=DTYPE)
        fused.weight.data.copy_(ref.weight.data)
        return ref, fused

    # --- Case (a): no residual (first layer, residual=None) ---
    def test_no_residual(self):
        ref, fused = self._make_ref_and_fused(HIDDEN_SIZE)
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                x = torch.randn(num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                ref_out = ref.forward_native(x.clone())
                ref_residual = x.clone()
                fused_out, fused_residual = fused.forward_cuda(x.clone())
                torch.testing.assert_close(
                    fused_out, ref_out, atol=RESIDUAL_ATOL, rtol=RESIDUAL_RTOL
                )
                torch.testing.assert_close(
                    fused_residual,
                    ref_residual,
                    atol=RESIDUAL_ATOL,
                    rtol=RESIDUAL_RTOL,
                )

    # --- Case (b): with residual ---
    def test_with_residual(self):
        ref, fused = self._make_ref_and_fused(HIDDEN_SIZE)
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                x = torch.randn(num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                residual = torch.randn(
                    num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
                )
                ref_out, ref_residual = ref.forward_native(
                    x.clone(), residual=residual.clone()
                )
                fused_out, fused_residual = fused.forward_cuda(
                    x.clone(), residual=residual.clone()
                )
                torch.testing.assert_close(
                    fused_out, ref_out, atol=RESIDUAL_ATOL, rtol=RESIDUAL_RTOL
                )
                torch.testing.assert_close(
                    fused_residual,
                    ref_residual,
                    atol=RESIDUAL_ATOL,
                    rtol=RESIDUAL_RTOL,
                )

    # --- Case (c): residual_after_layernorm=True (ppln=True) ---
    # Pre-fusion: hidden_states, residual = layernorm(h, r); residual = hidden_states.clone()
    # Fused: hidden_states, residual = layernorm(h, r, residual_after_layernorm=True)
    def test_residual_after_layernorm(self):
        ref, fused = self._make_ref_and_fused(HIDDEN_SIZE)
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                x = torch.randn(num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                residual = torch.randn(
                    num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
                )
                ref_out, _ = ref.forward_native(x.clone(), residual=residual.clone())
                ref_residual = ref_out.clone()
                fused_out, fused_residual = fused.forward_cuda(
                    x.clone(),
                    residual=residual.clone(),
                    residual_after_layernorm=True,
                )
                torch.testing.assert_close(
                    fused_out, ref_out, atol=RESIDUAL_ATOL, rtol=RESIDUAL_RTOL
                )
                torch.testing.assert_close(
                    fused_residual,
                    ref_residual,
                    atol=RESIDUAL_ATOL,
                    rtol=RESIDUAL_RTOL,
                )

    # --- Case (d): clone_fp32_out=True (post_attention_layernorm for MoE router) ---
    def test_clone_fp32_out(self):
        ref, fused = self._make_ref_and_fused(HIDDEN_SIZE)
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                x = torch.randn(num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE)
                residual = torch.randn(
                    num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
                )
                ref_out, ref_residual = ref.forward_native(
                    x.clone(), residual=residual.clone()
                )
                fused_out, fused_residual, fused_fp32 = fused.forward_cuda(
                    x.clone(), residual=residual.clone(), clone_fp32_out=True
                )
                torch.testing.assert_close(
                    fused_out, ref_out, atol=RESIDUAL_ATOL, rtol=RESIDUAL_RTOL
                )
                torch.testing.assert_close(
                    fused_residual, ref_residual, atol=ATOL, rtol=RTOL
                )
                # fp32_out cast to bf16 should exactly match bf16 output
                torch.testing.assert_close(
                    fused_fp32.to(DTYPE), fused_out, atol=0, rtol=0
                )
                # fp32_out vs pre-fusion pattern (bf16 output cast to fp32)
                ref_fp32 = ref_out.to(torch.float32)
                torch.testing.assert_close(fused_fp32, ref_fp32, atol=2e-2, rtol=2e-2)

    # --- Case (e): per-head k_norm (head_dim=256, k_norm=True, qk_norm=False) ---
    # Before: k_by_head = self.k_norm.forward_native(k_by_head)
    #         k_by_head shape: (num_tokens, num_kv_heads, head_dim)
    # After:  k_by_head, _ = self.k_norm(k_by_head)
    # Reshape to 2D to match kernel's stride assumption.
    def test_per_head_k_norm(self):
        ref, fused = self._make_ref_and_fused(HEAD_DIM)
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                k_by_head = torch.randn(
                    num_tokens, NUM_K_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                ref_out = ref.forward_native(k_by_head.clone())
                x_2d = k_by_head.clone().reshape(-1, HEAD_DIM)
                fused_out_2d, _ = fused.forward_cuda(x_2d)
                fused_out = fused_out_2d.view(num_tokens, NUM_K_HEADS, HEAD_DIM)
                torch.testing.assert_close(fused_out, ref_out, atol=ATOL, rtol=RTOL)


# =====================================================================
# Test 1b: add-first residual stream vs next-layer fused RMSNorm residual path
# =====================================================================
class TestWelmV4AddFirstResidual(CustomTestCase):
    def test_add_first_matches_next_layer_rmsnorm(self):
        fused = WelmV4FusedRMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS).to(
            device=DEVICE, dtype=DTYPE
        )
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                mlp_out = torch.randn(
                    num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
                )
                residual = torch.randn(
                    num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
                )

                ref_hidden, ref_residual = fused.forward_cuda(
                    mlp_out.clone(), residual=residual.clone()
                )
                combined = mmq_style_add_residual(
                    mlp_out.contiguous(), residual.contiguous()
                )
                add_first_hidden, add_first_residual = fused.forward_cuda(
                    combined, residual=None, output_dtype=DTYPE
                )

                torch.testing.assert_close(
                    combined.to(DTYPE),
                    ref_residual,
                    atol=ATOL,
                    rtol=RTOL,
                )
                torch.testing.assert_close(
                    add_first_hidden,
                    ref_hidden,
                    atol=RESIDUAL_ATOL,
                    rtol=RESIDUAL_RTOL,
                )
                torch.testing.assert_close(
                    add_first_residual, combined, atol=0, rtol=0
                )

    def test_add_first_matches_residual_after_layernorm(self):
        fused = WelmV4FusedRMSNorm(HIDDEN_SIZE, eps=RMS_NORM_EPS).to(
            device=DEVICE, dtype=DTYPE
        )
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                mlp_out = torch.randn(
                    num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
                )
                residual = torch.randn(
                    num_tokens, HIDDEN_SIZE, device=DEVICE, dtype=DTYPE
                )

                ref_hidden, _, ref_residual = fused.forward_cuda(
                    mlp_out.clone(),
                    residual=residual.clone(),
                    residual_after_layernorm=True,
                    clone_fp32_out=True,
                )
                combined = mmq_style_add_residual(
                    mlp_out.contiguous(), residual.contiguous()
                )
                add_first_hidden, _, add_first_residual = fused.forward_cuda(
                    combined,
                    residual=None,
                    residual_after_layernorm=True,
                    clone_fp32_out=True,
                    output_dtype=DTYPE,
                )

                torch.testing.assert_close(
                    add_first_hidden,
                    ref_hidden,
                    atol=RESIDUAL_ATOL,
                    rtol=RESIDUAL_RTOL,
                )
                torch.testing.assert_close(
                    add_first_residual,
                    ref_residual,
                    atol=RESIDUAL_ATOL,
                    rtol=RESIDUAL_RTOL,
                )


# =====================================================================
# Test 2: inplace_sigmoid_mul vs torch.sigmoid(gate) * attn_output
# =====================================================================
# NOTE: 和之前的实现无法对齐，应为 triton 中的 sigmoid 算子强制限制在fp32或以上。
class TestInplaceSigmoidMul(CustomTestCase):
    """
    Pre-fusion code (gated_self_attention_headwise=True):
        gate = self.gate_proj(hidden_states)[0].unsqueeze(-1)  # (bs*seq_len, num_heads, 1)
        attn_output = attn_output.view(attn_shape[0], self.num_heads, -1)
        attn_output = attn_output * torch.sigmoid(gate)

    After fusion:
        inplace_sigmoid_mul(gate, attn_output)

    Triton kernel computes sigmoid in fp32. Pre-fusion code computed in bf16:
        attn_output = attn_output * torch.sigmoid(gate)
    Reference uses the pre-fusion bf16 path.
    Model params: num_attention_heads=24, head_dim=256.
    """

    def test_basic(self):
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                gate = torch.randn(
                    num_tokens, NUM_Q_HEADS, 1, device=DEVICE, dtype=DTYPE
                )
                attn_output = torch.randn(
                    num_tokens, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                # Reference: pre-fusion PyTorch code (bf16 path)
                ref_out = attn_output.clone() * torch.sigmoid(gate.clone())
                fused_attn = attn_output.clone()
                inplace_sigmoid_mul(gate.clone(), fused_attn)
                torch.testing.assert_close(fused_attn, ref_out, atol=ATOL, rtol=RTOL)


# =====================================================================
# Test 3: WelmV4InplaceRotaryEmbedding
# =====================================================================
class TestWelmV4InplaceRotaryEmbedding(CustomTestCase):
    """
    Pre-fusion code (on CUDA, via CustomOp dispatch -> forward_cuda):
        # self.rotary_emb was RotaryEmbedding(head_size=qk_rope_head_dim, rotary_dim=qk_rope_head_dim)
        # which on CUDA dispatches to forward_cuda -> sgl_kernel.apply_rope_with_cos_sin_cache_inplace
        q_nope, q_pe = q.view(q_by_head_shape).split([nope_dim, rope_dim], dim=-1)
        k_nope, k_pe = k.view(k_by_head_shape).split([nope_dim, rope_dim], dim=-1)
        q_pe = q_pe.reshape((*q_shape[:-1], q_shape[-1] // head_dim * rope_dim))
        k_pe = k_pe.reshape((*k_shape[:-1], k_shape[-1] // head_dim * rope_dim))
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)  # -> forward_cuda on CUDA
        q[..., nope_dim:] = q_pe;  k[..., nope_dim:] = k_pe

    After fusion:
        # self.rotary_emb is WelmV4InplaceRotaryEmbedding(head_size=head_dim, rotary_dim=qk_rope_head_dim)
        self.rotary_emb.forward_cuda(positions, q, k)

    Model params: head_dim=256, qk_rope_head_dim=64 (partial_rotary_factor=0.25),
    num_attention_heads=24, num_key_value_heads=2, is_neox_style=True.
    """

    def _make_ref_and_fused(self, max_pos=32768, base=100000):
        """
        ref: RotaryEmbedding(head_size=QK_ROPE_HEAD_DIM, rotary_dim=QK_ROPE_HEAD_DIM)
             — matches the pre-fusion rotary_emb, on CUDA uses sgl_kernel rope.
        fused: WelmV4InplaceRotaryEmbedding(head_size=HEAD_DIM, rotary_dim=QK_ROPE_HEAD_DIM)
             — the fused replacement, applies rope in-place on the tail of each head.
        """
        ref = RotaryEmbedding(
            head_size=QK_ROPE_HEAD_DIM,
            rotary_dim=QK_ROPE_HEAD_DIM,
            max_position_embeddings=max_pos,
            base=base,
            is_neox_style=True,
            dtype=DTYPE,
        ).to(DEVICE)

        fused = WelmV4InplaceRotaryEmbedding(
            head_size=HEAD_DIM,
            rotary_dim=QK_ROPE_HEAD_DIM,
            max_position_embeddings=max_pos,
            base=base,
            is_neox_style=True,
            dtype=DTYPE,
        ).to(DEVICE)

        return ref, fused

    def _ref_rope_on_tail(self, ref, positions, data, num_heads):
        """
        Reference: exactly replicate the pre-fusion calling pattern.
        Split out the rope portion, flatten to 2D, call ref.forward_cuda, put back.
        """
        num_tokens = positions.shape[0]
        nope_dim = HEAD_DIM - QK_ROPE_HEAD_DIM

        data = data.view(num_tokens, num_heads, HEAD_DIM).clone()

        # Split: extract rope portion from tail of each head
        data_pe = data[..., nope_dim:].contiguous()  # (tokens, heads, rope_dim)

        # Flatten to 2D as the pre-fusion code did:
        #   q_pe = q_pe.reshape((*q_shape[:-1], num_heads * rope_dim))
        data_pe_flat = data_pe.reshape(num_tokens, num_heads * QK_ROPE_HEAD_DIM)

        # We need a dummy counterpart for the other argument (ref expects both q and k)
        dummy = torch.zeros_like(data_pe_flat)

        # Call RotaryEmbedding.forward_cuda (sgl_kernel rope) — same as pre-fusion
        data_pe_out, _ = ref.forward_cuda(positions, data_pe_flat, dummy)

        # Reshape back and put into original data
        data_pe_out = data_pe_out.reshape(num_tokens, num_heads, QK_ROPE_HEAD_DIM)
        data[..., nope_dim:] = data_pe_out

        return data.view(num_tokens, num_heads * HEAD_DIM)

    # --- Case (a): basic partial rotation (head_dim=256, rope_dim=64) ---
    def test_basic_partial_rotation(self):
        ref, fused = self._make_ref_and_fused()
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                positions = torch.randint(
                    0, 4096, (num_tokens,), device=DEVICE, dtype=torch.int64
                )
                q = torch.randn(
                    num_tokens, NUM_Q_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                k = torch.randn(
                    num_tokens, NUM_K_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                q_ref = self._ref_rope_on_tail(ref, positions, q.clone(), NUM_Q_HEADS)
                k_ref = self._ref_rope_on_tail(ref, positions, k.clone(), NUM_K_HEADS)
                q_fused = q.clone()
                k_fused = k.clone()
                fused.forward_cuda(positions, q_fused, k_fused)
                torch.testing.assert_close(q_fused, q_ref, atol=ATOL, rtol=RTOL)
                torch.testing.assert_close(k_fused, k_ref, atol=ATOL, rtol=RTOL)

    # --- Case (b): verify nope portion unchanged ---
    def test_nope_unchanged(self):
        _, fused = self._make_ref_and_fused()
        nope_dim = HEAD_DIM - QK_ROPE_HEAD_DIM
        for num_tokens in num_tokens_list:
            with self.subTest(num_tokens=num_tokens):
                positions = torch.randint(
                    0, 4096, (num_tokens,), device=DEVICE, dtype=torch.int64
                )
                q = torch.randn(
                    num_tokens, NUM_Q_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                k = torch.randn(
                    num_tokens, NUM_K_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE
                )
                q_orig = q.clone().view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
                k_orig = k.clone().view(num_tokens, NUM_K_HEADS, HEAD_DIM)
                q_fused = q.clone()
                k_fused = k.clone()
                fused.forward_cuda(positions, q_fused, k_fused)
                q_fused_h = q_fused.view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
                k_fused_h = k_fused.view(num_tokens, NUM_K_HEADS, HEAD_DIM)
                # Non-rope part (first nope_dim elements per head) must be unchanged
                torch.testing.assert_close(
                    q_fused_h[..., :nope_dim], q_orig[..., :nope_dim], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    k_fused_h[..., :nope_dim], k_orig[..., :nope_dim], atol=0, rtol=0
                )

    # --- Case (c): with last_index (KV mirror mode) ---
    # Pre-fusion:
    #   _, k_pe = self.rotary_emb(positions, q_pe_proxy, k_pe)
    #   q_pe, _ = self.rotary_emb(positions[custom_last_index], q_pe, k_pe_proxy)
    # After fusion:
    #   self.rotary_emb.forward_cuda(positions, q, k, last_index=custom_last_index)
    def test_with_last_index(self):
        ref, fused = self._make_ref_and_fused()
        num_tokens = 64
        bs = 4
        last_index = torch.tensor([15, 31, 47, 63], device=DEVICE, dtype=torch.int64)
        positions = torch.randint(
            0, 4096, (num_tokens,), device=DEVICE, dtype=torch.int64
        )
        q = torch.randn(num_tokens, NUM_Q_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
        k = torch.randn(num_tokens, NUM_K_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
        nope_dim = HEAD_DIM - QK_ROPE_HEAD_DIM

        # Reference: k with all positions (all tokens)
        k_ref = self._ref_rope_on_tail(ref, positions, k.clone(), NUM_K_HEADS)

        # Reference: q with positions[last_index] for first bs tokens only
        # Pre-fusion: q_pe, _ = self.rotary_emb(positions[last_index], q_pe, k_pe_proxy)
        q_ref_heads = q.clone().view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
        q_pe_bs = q_ref_heads[:bs, :, nope_dim:].contiguous()
        q_pe_bs_flat = q_pe_bs.reshape(bs, NUM_Q_HEADS * QK_ROPE_HEAD_DIM)
        dummy = torch.zeros(
            bs, NUM_K_HEADS * QK_ROPE_HEAD_DIM, device=DEVICE, dtype=DTYPE
        )
        q_pe_out, _ = ref.forward_cuda(positions[last_index], q_pe_bs_flat, dummy)
        q_pe_out = q_pe_out.reshape(bs, NUM_Q_HEADS, QK_ROPE_HEAD_DIM)
        q_ref_heads[:bs, :, nope_dim:] = q_pe_out

        # Fused
        q_fused = q.clone()
        k_fused = k.clone()
        fused.forward_cuda(positions, q_fused, k_fused, last_index=last_index)

        # Compare k (all tokens)
        torch.testing.assert_close(k_fused, k_ref, atol=ATOL, rtol=RTOL)
        # Compare q (first bs tokens)
        q_fused_heads = q_fused.view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
        torch.testing.assert_close(
            q_fused_heads[:bs], q_ref_heads[:bs], atol=ATOL, rtol=RTOL
        )
        # q tokens beyond bs should be unchanged
        q_orig_heads = q.view(num_tokens, NUM_Q_HEADS, HEAD_DIM)
        torch.testing.assert_close(
            q_fused_heads[bs:], q_orig_heads[bs:], atol=0, rtol=0
        )

    def test_empty_query_with_last_index(self):
        ref, fused = self._make_ref_and_fused()
        num_tokens = 64
        positions = torch.randint(
            0, 4096, (num_tokens,), device=DEVICE, dtype=torch.int64
        )
        q = torch.empty(0, NUM_Q_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
        k = torch.randn(num_tokens, NUM_K_HEADS * HEAD_DIM, device=DEVICE, dtype=DTYPE)
        last_index = torch.empty(0, device=DEVICE, dtype=torch.int64)

        k_ref = self._ref_rope_on_tail(ref, positions, k.clone(), NUM_K_HEADS)
        q_fused = q.clone()
        k_fused = k.clone()

        fused.forward_cuda(positions, q_fused, k_fused, last_index=last_index)

        self.assertEqual(tuple(q_fused.shape), (0, NUM_Q_HEADS * HEAD_DIM))
        torch.testing.assert_close(k_fused, k_ref, atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    unittest.main()
