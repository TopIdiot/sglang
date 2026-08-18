import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.managers.overlap_utils as overlap_utils
from sglang.srt.managers.overlap_utils import FutureIndices, FutureMap
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.welmv4_mtp_draft_proposal_cuda_graph_runner import (
    WelmMTPDraftProposalCudaGraphRunner,
)


def _make_draft_input(batch_size=2, has_sampling_state=True):
    topk = 1
    draft_input = EagleDraftInput(
        hidden_states=torch.arange(batch_size * 4, dtype=torch.float32).reshape(
            batch_size, 4
        ),
        topk_p=torch.ones((batch_size, topk), dtype=torch.float32),
        topk_index=torch.arange(batch_size, dtype=torch.int64).reshape(batch_size, 1),
        bonus_tokens=torch.arange(batch_size, dtype=torch.int32),
        new_seq_lens=torch.arange(batch_size, dtype=torch.int32) + 8,
        num_tokens_per_req=1,
        num_tokens_for_logprob_per_req=1,
    )
    if has_sampling_state:
        draft_input.draft_probs = (
            torch.arange(batch_size * 2 * 4, dtype=torch.float32).reshape(
                batch_size, 2, 4
            )
            / 100
        )
        draft_input.welm_mtp_draft_topk_indices = torch.arange(
            batch_size * 2 * 3, dtype=torch.int64
        ).reshape(batch_size, 2, 3)
        draft_input.welm_mtp_draft_topk_values = (
            torch.arange(batch_size * 2 * 3, dtype=torch.float32).reshape(
                batch_size, 2, 3
            )
            / 10
        )
    return draft_input


def _attach_oe_history(draft_input):
    batch_size = int(draft_input.bonus_tokens.numel())
    draft_input.welm_mtp_oe_history_state = (
        torch.arange(batch_size * 3, dtype=torch.int64).reshape(batch_size, 3) + 1000
    )
    return draft_input


def _attach_prebuilt_verify(draft_input):
    prebuilt_verify = object()
    draft_input.welm_mtp_linear_verify_ready = True
    draft_input.welm_mtp_prebuilt_verify_input = prebuilt_verify
    draft_input.welm_mtp_prebuilt_verify_bs = len(draft_input.topk_p)
    return prebuilt_verify


class TestWelmMTPOverlapFutureMap(unittest.TestCase):
    def setUp(self):
        self._record_stream = torch.Tensor.record_stream
        torch.Tensor.record_stream = lambda _tensor, _stream: None
        self._spec_need_hidden_states = overlap_utils.spec_need_hidden_states
        overlap_utils.spec_need_hidden_states = lambda: False

    def tearDown(self):
        torch.Tensor.record_stream = self._record_stream
        overlap_utils.spec_need_hidden_states = self._spec_need_hidden_states

    def test_resolve_preserves_welm_mtp_sampling_state(self):
        # Model a decode-disaggregation pool with 4 running slots, 2
        # pre-allocation slots, and CUDA-graph padding slot 0.
        future_map = FutureMap(
            max_running_requests=4,
            chunked_prefill_size=16,
            context_len=128,
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.EAGLE,
            request_buffer_len=7,
        )
        stored = _make_draft_input()
        future_indices = future_map.alloc_future_indices(
            2, torch.tensor([6, 5], dtype=torch.int64)
        )
        future_map.store_to_map_for_new_batch(future_indices, stored)

        self.assertEqual(future_map.future_buffer_len, 52)
        self.assertEqual(future_map.welm_mtp_draft_probs_buf.shape[0], 7)
        self.assertEqual(future_map.welm_mtp_draft_topk_indices_buf.shape[0], 7)
        self.assertEqual(future_map.welm_mtp_draft_topk_values_buf.shape[0], 7)

        resolved = _make_draft_input(has_sampling_state=False)
        resolved.future_indices = future_indices
        future_map.resolve_future(SimpleNamespace(spec_info=resolved))

        torch.testing.assert_close(
            resolved.draft_probs,
            stored.draft_probs,
        )
        torch.testing.assert_close(
            resolved.welm_mtp_draft_topk_indices,
            stored.welm_mtp_draft_topk_indices,
        )
        torch.testing.assert_close(
            resolved.welm_mtp_draft_topk_values,
            stored.welm_mtp_draft_topk_values,
        )

    def test_deferred_prefill_does_not_allocate_sampling_state(self):
        future_map = FutureMap(
            max_running_requests=4,
            chunked_prefill_size=16,
            context_len=128,
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.EAGLE,
        )
        stored = _make_draft_input(has_sampling_state=False)
        stored.welm_mtp_deferred_prefill_draft = True
        stored.welm_mtp_deferred_prefill_draft_mask = torch.ones(
            (2,), dtype=torch.bool
        )
        future_indices = future_map.alloc_future_indices(
            2, torch.tensor([1, 2], dtype=torch.int64)
        )

        future_map.store_to_map_for_new_batch(future_indices, stored)

        self.assertFalse(future_map.has_welm_mtp_draft_probs_buf)
        self.assertFalse(future_map.has_welm_mtp_draft_topk_buf)
        self.assertFalse(hasattr(future_map, "welm_mtp_draft_probs_buf"))
        self.assertFalse(hasattr(future_map, "welm_mtp_draft_topk_indices_buf"))
        self.assertFalse(hasattr(future_map, "welm_mtp_draft_topk_values_buf"))

    def test_resolve_drops_sampling_state_when_mixed_with_deferred_prefill(self):
        future_map = FutureMap(
            max_running_requests=4,
            chunked_prefill_size=0,
            context_len=128,
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.EAGLE,
        )
        stored = _make_draft_input()
        future_indices = future_map.alloc_future_indices(
            2, torch.tensor([1, 3], dtype=torch.int64)
        )
        future_map.store_to_map_for_new_batch(future_indices, stored)
        missing_index = future_indices.indices[1]
        missing_req_index = future_map.req_pool_indices_buf[missing_index]
        future_map.welm_mtp_has_draft_probs_buf[missing_index] = False
        future_map.welm_mtp_has_draft_topk_buf[missing_index] = False
        future_map.welm_mtp_deferred_prefill_draft_buf[missing_index] = True
        future_map.welm_mtp_draft_probs_buf[missing_req_index].zero_()
        future_map.welm_mtp_draft_topk_indices_buf[missing_req_index].zero_()
        future_map.welm_mtp_draft_topk_values_buf[missing_req_index].zero_()

        resolved = _make_draft_input(has_sampling_state=False)
        resolved.future_indices = future_indices
        future_map.resolve_future(SimpleNamespace(spec_info=resolved))

        self.assertTrue(resolved.welm_mtp_deferred_prefill_draft)
        torch.testing.assert_close(
            resolved.welm_mtp_deferred_prefill_draft_mask,
            torch.tensor([False, True]),
        )
        self.assertIsNone(resolved.draft_probs)
        self.assertIsNone(resolved.welm_mtp_draft_topk_indices)
        self.assertIsNone(resolved.welm_mtp_draft_topk_values)

    def test_resolve_preserves_welm_mtp_oe_history_state(self):
        future_map = FutureMap(
            max_running_requests=4,
            chunked_prefill_size=0,
            context_len=128,
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.EAGLE,
        )
        stored = _attach_oe_history(_make_draft_input())
        future_indices = future_map.alloc_future_indices(
            2, torch.tensor([1, 2], dtype=torch.int64)
        )
        future_map.store_to_map_for_new_batch(future_indices, stored)

        resolved = _make_draft_input(has_sampling_state=False)
        resolved.future_indices = future_indices
        future_map.resolve_future(SimpleNamespace(spec_info=resolved))

        torch.testing.assert_close(
            resolved.welm_mtp_oe_history_state,
            stored.welm_mtp_oe_history_state,
        )

    def test_resolve_preserves_welm_mtp_oe_history_state_for_mixed_rows(self):
        future_map = FutureMap(
            max_running_requests=4,
            chunked_prefill_size=0,
            context_len=128,
            device=torch.device("cpu"),
            spec_algo=SpeculativeAlgorithm.EAGLE,
        )
        stored = _attach_oe_history(_make_draft_input())
        future_indices = future_map.alloc_future_indices(
            2, torch.tensor([1, 2], dtype=torch.int64)
        )
        future_map.store_to_map_for_new_batch(future_indices, stored)
        missing_index = future_indices.indices[1]
        future_map.welm_mtp_has_oe_history_buf[missing_index] = False
        future_map.welm_mtp_oe_history_buf[missing_index].zero_()

        resolved = _make_draft_input(has_sampling_state=False)
        resolved.welm_mtp_oe_history_state = torch.ones((2, 3), dtype=torch.int64)
        resolved.future_indices = future_indices
        future_map.resolve_future(SimpleNamespace(spec_info=resolved))

        self.assertTrue(hasattr(resolved, "welm_mtp_oe_history_state"))
        torch.testing.assert_close(
            resolved.welm_mtp_oe_history_state[0],
            stored.welm_mtp_oe_history_state[0],
        )
        torch.testing.assert_close(
            resolved.welm_mtp_oe_history_state[1],
            torch.zeros_like(resolved.welm_mtp_oe_history_state[1]),
        )

    def test_filter_then_merge_same_bs_invalidates_prebuilt_verify(self):
        draft_input = _make_draft_input()
        original_bs = len(draft_input.topk_p)
        _attach_prebuilt_verify(draft_input)

        draft_input.filter_batch(
            torch.tensor([1], dtype=torch.int64), has_been_filtered=False
        )
        draft_input.merge_batch(_make_draft_input(batch_size=1))

        self.assertEqual(len(draft_input.topk_p), original_bs)
        self.assertFalse(draft_input.welm_mtp_linear_verify_ready)
        self.assertIsNone(draft_input.welm_mtp_prebuilt_verify_input)
        self.assertEqual(draft_input.welm_mtp_prebuilt_verify_bs, -1)

    def test_overlap_filter_invalidates_prebuilt_verify(self):
        draft_input = _make_draft_input()
        draft_input.future_indices = FutureIndices(
            indices=torch.tensor([10, 11], dtype=torch.int64)
        )
        _attach_prebuilt_verify(draft_input)

        draft_input.filter_batch(torch.tensor([0], dtype=torch.int64))

        self.assertFalse(draft_input.welm_mtp_linear_verify_ready)
        self.assertIsNone(draft_input.welm_mtp_prebuilt_verify_input)
        self.assertEqual(draft_input.welm_mtp_prebuilt_verify_bs, -1)

    def test_merge_invalidates_prebuilt_verify(self):
        draft_input = _make_draft_input()
        _attach_prebuilt_verify(draft_input)

        draft_input.merge_batch(_make_draft_input(batch_size=1))

        self.assertFalse(draft_input.welm_mtp_linear_verify_ready)
        self.assertIsNone(draft_input.welm_mtp_prebuilt_verify_input)
        self.assertEqual(draft_input.welm_mtp_prebuilt_verify_bs, -1)

    def test_overlap_does_not_return_runner_owned_prebuilt_verify(self):
        draft_input = _make_draft_input()
        prebuilt_verify = _attach_prebuilt_verify(draft_input)
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker._is_welmv4_mtp_draft_model = Mock(
            side_effect=RuntimeError("continued past prebuilt verify")
        )
        model_worker_batch = SimpleNamespace(
            spec_info=draft_input,
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            seq_lens=torch.ones((2,), dtype=torch.int32),
        )

        self.assertIs(worker.draft(model_worker_batch), prebuilt_verify)

        draft_input.future_indices = FutureIndices(
            indices=torch.tensor([10, 11], dtype=torch.int64)
        )
        with self.assertRaisesRegex(RuntimeError, "continued past prebuilt verify"):
            worker.draft(model_worker_batch)

    def test_overlap_rebuilds_linear_verify_inputs_from_resolved_rows(self):
        runner = WelmMTPDraftProposalCudaGraphRunner.__new__(
            WelmMTPDraftProposalCudaGraphRunner
        )
        runner.linear_verify_prepare = True
        runner._fused_linear_graph_outputs_base = True
        runner.sample_draft = True
        runner.max_bs = 4
        runner.num_tokens_per_bs = 4
        runner.buffers = SimpleNamespace(
            linear_verify_mask=torch.empty((4 * 4 * 4,), dtype=torch.bool),
            linear_verify_positions=torch.empty((4 * 4,), dtype=torch.int64),
            linear_verify_retrieve_index=torch.empty((4, 4), dtype=torch.int64),
            linear_verify_retrieve_next_token=torch.empty((4, 4), dtype=torch.int64),
            linear_verify_retrieve_next_sibling=torch.empty((4, 4), dtype=torch.int64),
            linear_verify_tokens=torch.empty((4 * 4,), dtype=torch.int64),
            linear_verify_hash_seq_lens=torch.empty((4,), dtype=torch.int32),
        )
        draft_input = _make_draft_input()
        draft_input.draft_proposal_tokens = torch.zeros((2, 3), dtype=torch.int64)
        draft_input.welm_mtp_linear_verify_ready = True
        seq_lens = torch.tensor([8, 9], dtype=torch.int32)

        builder_path = (
            "sglang.srt.speculative.welmv4_mtp_staging."
            "build_welm_mtp_linear_verify_inputs"
        )
        with patch(builder_path) as build_linear_verify:
            runner.build_linear_verify_inputs(draft_input, seq_lens)
            build_linear_verify.assert_not_called()

        draft_input.future_indices = FutureIndices(
            indices=torch.tensor([10, 11], dtype=torch.int64)
        )
        with patch(builder_path) as build_linear_verify:
            runner.build_linear_verify_inputs(draft_input, seq_lens)
            build_linear_verify.assert_called_once()


class TestWelmMTPPrefillDefer(unittest.TestCase):
    @staticmethod
    def _worker():
        return EagleDraftWorker.__new__(EagleDraftWorker)

    @staticmethod
    def _req(fill_len, origin_len, *, is_chunked=1):
        return SimpleNamespace(
            rid=f"req-{fill_len}-{origin_len}",
            is_chunked=is_chunked,
            fill_ids=list(range(fill_len)),
            origin_input_ids=list(range(origin_len)),
            output_ids=[],
        )

    def test_defer_intermediate_chunk(self):
        worker = self._worker()
        batch = SimpleNamespace(reqs=[self._req(fill_len=8192, origin_len=11333)])

        self.assertTrue(worker._should_defer_welmv4_mtp_prefill_draft(batch))

    def test_do_not_defer_final_chunk(self):
        worker = self._worker()
        batch = SimpleNamespace(reqs=[self._req(fill_len=11333, origin_len=11333)])

        self.assertFalse(worker._should_defer_welmv4_mtp_prefill_draft(batch))

    def test_mixed_intermediate_and_completed_prefill_rows_use_row_mask(self):
        worker = self._worker()
        batch = SimpleNamespace(
            reqs=[
                self._req(fill_len=8192, origin_len=11333),
                self._req(fill_len=10, origin_len=10, is_chunked=0),
            ]
        )

        self.assertFalse(worker._should_defer_welmv4_mtp_prefill_draft(batch))
        torch.testing.assert_close(
            worker._get_welmv4_mtp_deferred_prefill_mask(batch),
            torch.tensor([True, False]),
        )


if __name__ == "__main__":
    unittest.main()
