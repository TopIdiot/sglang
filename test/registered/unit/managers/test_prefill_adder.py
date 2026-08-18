import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.context_parallel import build_cp_prefill_split_spec
from sglang.srt.managers import schedule_policy
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefResult,
    EvictResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.cp_sharded_allocator import CPShardedKVPoolAllocator
from sglang.srt.models.welm_deferred_mirror import build_welm_deferred_prefill_span
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="stage-b", runner_config="1-gpu-small")
register_amd_ci(est_time=2, suite="stage-b-test-1-gpu-small-amd")


class TestPrefillAdder(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        self.mock_tree_cache = self.create_tree_cache()
        self.mock_token_allocator = self.create_token_allocator()

    def create_tree_cache(
        self,
        *,
        full_evictable_size: int = 0,
        swa_evictable_size: int = 0,
        evictable_size: int = 0,
    ) -> MagicMock:
        tree_cache = MagicMock()
        tree_cache.full_evictable_size.return_value = full_evictable_size
        tree_cache.swa_evictable_size.return_value = swa_evictable_size
        tree_cache.evictable_size.return_value = evictable_size
        tree_cache.disable = False
        tree_cache.supports_mamba.return_value = False
        tree_cache.inc_lock_ref.return_value = IncLockRefResult()
        tree_cache.dec_lock_ref.return_value = DecLockRefResult()
        return tree_cache

    def create_token_allocator(
        self,
        *,
        full_available_size: int = 0,
        swa_available_size: int = 0,
        available_size: int = 0,
    ) -> MagicMock:
        allocator = MagicMock()
        allocator.full_available_size.return_value = full_available_size
        allocator.swa_available_size.return_value = swa_available_size
        allocator.available_size.return_value = available_size
        return allocator

    def create_running_batch(self, reqs=None) -> MagicMock:
        batch = MagicMock()
        batch.reqs = list(reqs or [])
        batch.release_req.return_value = None
        batch.filter_batch.return_value = None
        return batch

    def create_server_args(
        self, *, schedule_low_priority_values_first: bool
    ) -> MagicMock:
        server_args = MagicMock()
        server_args.schedule_low_priority_values_first = (
            schedule_low_priority_values_first
        )
        return server_args

    def create_mock_req(self, rid, priority, max_new_tokens, output_len=0, wait_time=0):
        req = MagicMock(spec=Req)
        req.rid = str(rid)
        req.priority = priority
        req.extend_input_len = 0
        req.extend_logprob_start_len = 0
        req.output_ids = [0] * output_len
        req.sampling_params = SimpleNamespace(max_new_tokens=max_new_tokens)
        req.time_stats = SimpleNamespace(wait_queue_entry_time=wait_time)
        req.finished.return_value = False
        return req

    def create_adder(self, running_batch, **kwargs):
        defaults = dict(
            page_size=1,
            tree_cache=self.mock_tree_cache,
            token_to_kv_pool_allocator=self.mock_token_allocator,
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=10000,
            rem_chunk_tokens=None,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )
        defaults.update(kwargs)
        return PrefillAdder(**defaults)

    def create_deferred_full_hit_req(self, token_ids=None):
        req = Req.__new__(Req)
        req.rid = "deferred-full-hit"
        req.origin_input_ids = list(token_ids or [11, 22, 33, 44, 55])
        req.output_ids = []
        req.welm_deferred_prefill_span = build_welm_deferred_prefill_span(
            req.origin_input_ids
        )
        req.prefix_indices = torch.tensor([101, 205, 309, 413])[
            : req.welm_deferred_prefill_span.committed_kv_len
        ]
        req.extend_input_len = 0
        req.attn_cp_prefill_split_spec = None
        req.host_hit_length = 0
        req.storage_hit_length = 0
        req.cached_tokens = 0
        req.cached_tokens_device = 0
        req.cached_tokens_host = 0
        req.cached_tokens_storage = 0
        req.already_computed = 0
        req.retracted_stain = False
        req._cache_breakdown_computed = False
        req.last_node = object()
        req.cache_protected_len = req.welm_deferred_prefill_span.committed_kv_len
        req.req_pool_idx = None
        req.kv_committed_len = 0
        req.kv_allocated_len = 0
        req.swa_uuid_for_lock = None
        req.sampling_params = SimpleNamespace(
            ignore_eos=False,
            max_new_tokens=1,
        )
        return req

    def test_deferred_full_hit_reuses_logical_kv_without_page_allocation(self):
        req_to_token_pool = MagicMock()
        req_to_token_pool.alloc.side_effect = lambda reqs: [7]
        self.mock_tree_cache.req_to_token_pool = req_to_token_pool
        adder = self.create_adder(self.create_running_batch())
        req = self.create_deferred_full_hit_req()

        result = adder.add_one_req(req, False, None)

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [])
        self.assertEqual(adder.completed_without_forward_reqs, [req])
        self.assertEqual(req.req_pool_idx, 7)
        self.assertEqual(req.kv_committed_len, 4)
        self.assertEqual(req.kv_allocated_len, 4)
        self.assertEqual(req.cached_tokens, 4)
        self.assertEqual(req.cached_tokens_device, 4)
        self.assertEqual(req.cached_tokens_host, 0)
        self.assertEqual(req.cached_tokens_storage, 0)
        self.assertEqual(req.already_computed, 4)
        req_to_token_pool.write.assert_called_once()
        write_indices, write_values = req_to_token_pool.write.call_args.args
        self.assertEqual(write_indices, (7, slice(0, 4)))
        self.assertTrue(torch.equal(write_values, req.prefix_indices))
        self.mock_tree_cache.inc_lock_ref.assert_called_once_with(req.last_node)
        self.mock_token_allocator.alloc_extend.assert_not_called()
        self.assertEqual(adder.next_attn_cp_owner_rotation, 0)

    def test_deferred_zero_page_completion_allocates_no_kv_slot(self):
        req_to_token_pool = MagicMock()
        req_to_token_pool.alloc.return_value = [9]
        self.mock_tree_cache.req_to_token_pool = req_to_token_pool
        adder = self.create_adder(self.create_running_batch())
        req = self.create_deferred_full_hit_req([77])

        result = adder.add_one_req(req, False, None)

        self.assertEqual(result, AddReqResult.CONTINUE)
        self.assertEqual(adder.completed_without_forward_reqs, [req])
        self.assertEqual(req.kv_committed_len, 0)
        self.assertEqual(req.kv_allocated_len, 0)
        req_to_token_pool.write.assert_not_called()
        self.mock_token_allocator.alloc_extend.assert_not_called()

    def test_deferred_full_hit_releases_req_slot_when_locking_fails(self):
        req_to_token_pool = MagicMock()
        req_to_token_pool.alloc.return_value = [7]
        self.mock_tree_cache.req_to_token_pool = req_to_token_pool
        self.mock_tree_cache.inc_lock_ref.side_effect = RuntimeError("lock failed")
        adder = self.create_adder(self.create_running_batch())
        req = self.create_deferred_full_hit_req()

        with self.assertRaisesRegex(RuntimeError, "lock failed"):
            adder.add_one_req(req, False, None)

        req_to_token_pool.free.assert_called_once_with(req)
        self.assertEqual(adder.completed_without_forward_reqs, [])

    def test_preempt_success_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req, mock_server_args)

        self.assertTrue(success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 175)  # 50 + 75 + 100 - 50 = 175
        running_batch.release_req.assert_called_once()

    def test_preempt_passes_kv_cache_offload_owner(self):
        running_req = self.create_mock_req("run", priority=0, max_new_tokens=50)
        running_batch = self.create_running_batch([running_req])
        offload_owner = object()
        adder = self.create_adder(
            running_batch,
            kv_cache_offload_owner=offload_owner,
        )
        self.mock_token_allocator.full_available_size.return_value = 50
        self.mock_token_allocator.available_size.return_value = 50
        server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )

        assert adder.preempt_to_schedule(
            self.create_mock_req("new", priority=1, max_new_tokens=49),
            server_args,
        )

        running_batch.release_req.assert_called_once_with(
            0,
            0,
            server_args,
            kv_cache_offload_owner=offload_owner,
        )

    def test_preempt_success_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=49)

        success = adder.preempt_to_schedule(new_req, mock_server_args)

        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 125)  # 50 + 75 + 100 - 100 = 125
        running_batch.release_req.assert_called_once()

    def test_preempt_fail_low_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=2, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_fail_high_priority_values_first(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = (
            225  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 225

        new_req_fail_by_priority_check = self.create_mock_req(
            "new1", priority=0, max_new_tokens=49
        )

        success_by_priority_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_priority_check)

        new_req_fail_by_priority_check = self.create_mock_req(
            "new2", priority=-1, max_new_tokens=110
        )
        success_by_capacity_check = adder.preempt_to_schedule(
            new_req_fail_by_priority_check, mock_server_args
        )
        self.assertFalse(success_by_capacity_check)

    def test_preempt_skip_already_preempted_request(self):
        params = [
            ("req_prio_0", 0, 50),
            ("req_prio_1", 1, 75),
            ("req_prio_2", 2, 100),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=False
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 225)

        self.mock_token_allocator.full_available_size.return_value = 225
        self.mock_token_allocator.available_size.return_value = 225

        # New request preempts req_prio_0
        first_req = self.create_mock_req(
            "new_req_prio_1", priority=1, max_new_tokens=49
        )
        first_success = adder.preempt_to_schedule(first_req, mock_server_args)
        self.assertTrue(first_success)
        self.assertIn(running_reqs[0], adder.preempt_list)
        self.assertEqual(adder.rem_total_token_offset, 175)
        running_batch.release_req.assert_called_once()

        # Second call needs more tokens than currently free, so it would need to
        # preempt req_prio_0 again if already-preempted requests were not filtered out.
        second_req = self.create_mock_req(
            "second_new_req_prio_1", priority=1, max_new_tokens=76
        )
        second_success = adder.preempt_to_schedule(second_req, mock_server_args)

        self.assertFalse(second_success)
        self.assertEqual(adder.rem_total_token_offset, 175)
        self.assertEqual(adder.preempt_list.count(running_reqs[0]), 1)
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_once(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=75)

        success = adder.preempt_to_schedule(new_req, mock_server_args)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertEqual(
            adder.rem_total_token_offset, 375
        )  # 50 + 75 + 100 + 125 + 125 - 100 = 375
        running_batch.release_req.assert_called_once()

    def test_preempt_success_low_priority_values_first_exact_twice(self):
        params = [
            ("run1", 0, 50),
            ("run2", 1, 75),
            ("run3", 2, 100),
            ("run4", 2, 125),
            ("run4", 2, 125),
        ]
        running_reqs = [
            self.create_mock_req(rid, priority, max_new_tokens)
            for rid, priority, max_new_tokens in params
        ]
        mock_server_args = self.create_server_args(
            schedule_low_priority_values_first=True
        )
        running_batch = self.create_running_batch(running_reqs)
        adder = self.create_adder(running_batch)

        self.assertEqual(adder.rem_total_token_offset, 475)

        self.mock_token_allocator.full_available_size.return_value = (
            475  # full occupation of GRam
        )
        self.mock_token_allocator.available_size.return_value = 475

        new_req = self.create_mock_req("new1", priority=1, max_new_tokens=200)

        success = adder.preempt_to_schedule(new_req, mock_server_args)
        self.assertTrue(success)
        self.assertIn(running_reqs[2], adder.preempt_list)
        self.assertIn(running_reqs[3], adder.preempt_list)
        self.assertEqual(
            adder.rem_total_token_offset, 250
        )  # 50 + 75 + 100 + 125 + 125 - 100 - 125 = 250
        self.assertEqual(running_batch.release_req.call_count, 2)

    def test_mixed_chunk_prefill_budgets(self):
        self.mock_token_allocator.available_size.return_value = 1000

        decode_reqs = [
            self.create_mock_req(f"decode_{i}", priority=0, max_new_tokens=50)
            for i in range(8)
        ]
        running_batch = self.create_running_batch(decode_reqs)

        adder = self.create_adder(
            running_batch,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            num_mixed_decode_tokens=len(decode_reqs),
        )

        self.assertEqual(adder.rem_input_tokens, 192)  # 200 - 8
        self.assertEqual(adder.rem_chunk_tokens, 56)  # 64 - 8
        self.assertEqual(adder.rem_total_token_offset, 408)  # 8 + 8 * 50
        self.assertEqual(adder.cur_rem_token_offset, 8)
        self.assertEqual(adder.budget_state(), AddReqResult.CONTINUE)

        # Add a prefill that exactly consumes the chunk budget
        req1 = self.create_mock_req("req1", priority=0, max_new_tokens=64)
        req1.extend_input_len = 56
        req1.host_hit_length = 0
        req1.prefix_indices = []
        req1.fill_ids = list(range(56))
        req1.last_node = MagicMock()
        req1.sampling_params.ignore_eos = False

        result1 = adder.add_one_req(
            req1, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder.can_run_list), 1)
        self.assertEqual(adder.rem_chunk_tokens, 0)  # 56 - 56
        self.assertEqual(adder.rem_input_tokens, 136)  # 192 - 56
        self.assertEqual(result1, AddReqResult.OTHER)

        # 3 decode requests finished
        remaining_decode_reqs = decode_reqs[3:]
        running_batch2 = self.create_running_batch(remaining_decode_reqs)

        adder2 = self.create_adder(
            running_batch2,
            rem_input_tokens=200,
            rem_chunk_tokens=64,
            num_mixed_decode_tokens=len(remaining_decode_reqs),
        )

        self.assertEqual(adder2.rem_input_tokens, 195)  # 200 - 5
        self.assertEqual(adder2.rem_chunk_tokens, 59)  # 64 - 5
        self.assertEqual(adder2.rem_total_token_offset, 255)  # 5 + 5 * 50
        self.assertEqual(adder2.budget_state(), AddReqResult.CONTINUE)

        # Same prefill no longer exhausts the chunk budget
        req2 = self.create_mock_req("req2", priority=0, max_new_tokens=64)
        req2.extend_input_len = 56
        req2.host_hit_length = 0
        req2.prefix_indices = []
        req2.fill_ids = list(range(56))
        req2.last_node = MagicMock()
        req2.sampling_params.ignore_eos = False

        result2 = adder2.add_one_req(
            req2, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 1)
        self.assertEqual(adder2.rem_chunk_tokens, 3)  # 59 - 56 = 3 remaining
        self.assertEqual(result2, AddReqResult.CONTINUE)

        # Fit last small prefill request
        req3 = self.create_mock_req("req3", priority=0, max_new_tokens=16)
        req3.extend_input_len = 3
        req3.host_hit_length = 0
        req3.prefix_indices = []
        req3.fill_ids = list(range(3))
        req3.last_node = MagicMock()
        req3.sampling_params.ignore_eos = False

        result3 = adder2.add_one_req(
            req3, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(len(adder2.can_run_list), 2)
        self.assertEqual(adder2.rem_chunk_tokens, 0)  # 3 - 3 = 0
        self.assertEqual(result3, AddReqResult.OTHER)

    def _build_hybrid_swa_chunked_req(
        self,
        *,
        page_size,
        rem_swa,
        rem_chunk=2048,
        extend_input_len=500,
        is_hybrid_swa=True,
        full_available=100_000,
    ):
        self.mock_token_allocator.swa_available_size.return_value = rem_swa
        self.mock_token_allocator.full_available_size.return_value = full_available
        self.mock_token_allocator.available_size.return_value = full_available
        self.mock_tree_cache.sliding_window_size = 128
        adder = self.create_adder(
            self.create_running_batch(),
            page_size=page_size,
            rem_chunk_tokens=rem_chunk,
        )
        adder.is_hybrid_swa = is_hybrid_swa

        req = self.create_mock_req("chunked", priority=0, max_new_tokens=128)
        req.extend_input_len = extend_input_len
        req.prefix_indices = []
        req.fill_ids = list(range(extend_input_len))
        req.set_extend_input_len = MagicMock()
        return adder, req

    def test_add_chunked_req_hybrid_swa_reserves_page_for_alloc_extend(self):
        # alloc_extend needs extend_num_tokens + page_size per request. If the
        # scheduler hands out all of rem_swa_tokens, alloc_extend cannot get its
        # extra page and OOMs. With the fix, extend_input_len must cap at
        # rem_swa_tokens - page_size so the page is reserved.
        PAGE_SIZE = 64
        REM_SWA = 100
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE, rem_swa=REM_SWA
        )

        result = adder.add_chunked_req(req)

        self.assertIs(result, req)  # truncated → chunked prefill continues
        req.set_extend_input_len.assert_called_once()
        new_len = req.set_extend_input_len.call_args.args[0]
        self.assertLessEqual(new_len + PAGE_SIZE, REM_SWA)
        self.assertEqual(new_len, REM_SWA - PAGE_SIZE)

    def test_add_chunked_req_hybrid_swa_defers_when_swa_below_page(self):
        # When rem_swa_tokens <= page_size there is no room to serve even the
        # reservation, so the chunked req must be deferred (returned unchanged)
        # instead of falling back to rem_chunk_tokens and bypassing SWA budget.
        PAGE_SIZE = 64
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE, rem_swa=PAGE_SIZE
        )
        original_len = req.extend_input_len

        result = adder.add_chunked_req(req)

        self.assertIs(result, req)
        req.set_extend_input_len.assert_not_called()
        self.assertEqual(req.extend_input_len, original_len)
        self.assertEqual(len(adder.can_run_list), 0)

    def test_swa_budget_for_req(self):
        cases = [
            # (extend, rem_chunk, window, page, expected, label)
            (64, None, 128, 16, 128 + 16, "no_cap_floor_active"),
            (200, None, 256, 32, 256 + 32, "no_cap_floor_active_other_dims"),
            (300, None, 128, 16, 300 + 16, "no_cap_floor_inactive"),
            (200, 50, 64, 8, 64 + 8, "cap_binds_then_floor"),
            (300, 500, 64, 64, 300 + 64, "cap_does_not_bind"),
            (0, None, 128, 16, 128 + 16, "extend_zero_floor_only"),
        ]
        for extend, rem_chunk, window, page, expected, label in cases:
            with self.subTest(label=label):
                self.mock_tree_cache.sliding_window_size = window
                adder = self.create_adder(
                    self.create_running_batch(),
                    page_size=page,
                    rem_chunk_tokens=rem_chunk,
                )
                self.assertEqual(adder._swa_budget_for_req(extend), expected)

    def test_add_chunked_req_non_hybrid_no_swa_reservation(self):
        # Non-hybrid path: the SWA-pool reservation must NOT apply, otherwise
        # the fix would regress non-SWA models.
        PAGE_SIZE = 16
        adder, req = self._build_hybrid_swa_chunked_req(
            page_size=PAGE_SIZE,
            rem_swa=10,
            rem_chunk=500,
            extend_input_len=200,
            is_hybrid_swa=False,
            full_available=300,
        )

        result = adder.add_chunked_req(req)
        self.assertIsNone(result)
        req.set_extend_input_len.assert_called_once_with(200)
        self.assertIn(req, adder.can_run_list)

    def _create_cp_sharded_allocator(
        self,
        *,
        physical_size=1000,
        logical_size=2000,
        cp_size=2,
        page_size=1,
        cp_kv_chunk_size=4,
    ):
        allocator_cls = (
            TokenToKVPoolAllocator
            if page_size == 1
            else PagedTokenToKVPoolAllocator
        )
        base_kwargs = dict(
            size=physical_size,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        if page_size > 1:
            base_kwargs["page_size"] = page_size
        base = allocator_cls(**base_kwargs)
        return CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=cp_size,
            cp_kv_chunk_size=cp_kv_chunk_size,
            logical_size=logical_size,
        )

    def _create_prefill_req(self, *, input_len, extend_len, max_new_tokens=0):
        req = self.create_mock_req(
            f"req_{input_len}_{extend_len}",
            priority=0,
            max_new_tokens=max_new_tokens,
        )
        req.origin_input_ids = [0] * input_len
        req.output_ids = []
        req.extend_input_len = extend_len
        req.host_hit_length = 0
        req.prefix_indices = []
        req.fill_ids = list(range(extend_len))
        req.last_node = MagicMock()
        req.attn_cp_owner_rotation = None
        req.sampling_params.ignore_eos = False
        req.set_extend_input_len = MagicMock(
            side_effect=lambda value: setattr(req, "extend_input_len", value)
        )
        return req

    def test_attncp_long_context_reserve_only_activates_for_long_requests(self):
        allocator = self._create_cp_sharded_allocator()
        long_running_req = self._create_prefill_req(input_len=70000, extend_len=0)
        short_running_req = self._create_prefill_req(input_len=32768, extend_len=0)

        short_adder = self.create_adder(
            self.create_running_batch([short_running_req]),
            token_to_kv_pool_allocator=allocator,
            rem_chunk_tokens=8,
        )
        self.assertEqual(short_adder.rem_total_tokens, 2000)

        long_adder = self.create_adder(
            self.create_running_batch([long_running_req]),
            token_to_kv_pool_allocator=allocator,
            rem_chunk_tokens=8,
        )
        # reserve = rem_chunk_tokens * 16 * cp_size
        self.assertEqual(long_adder.rem_total_tokens, 2000 - 8 * 16 * 2)

    def test_attncp_long_context_candidate_is_throttled_before_near_oom(self):
        allocator = self._create_cp_sharded_allocator(
            physical_size=130, logical_size=260
        )
        adder = self.create_adder(
            self.create_running_batch(),
            token_to_kv_pool_allocator=allocator,
            rem_chunk_tokens=8,
        )
        long_req = self._create_prefill_req(input_len=70000, extend_len=8)

        result = adder.add_one_req(
            long_req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(len(adder.can_run_list), 0)

    def test_attncp_short_context_is_not_throttled_by_long_context_reserve(self):
        allocator = self._create_cp_sharded_allocator(
            physical_size=130, logical_size=260
        )
        adder = self.create_adder(
            self.create_running_batch(),
            token_to_kv_pool_allocator=allocator,
            rem_chunk_tokens=8,
        )
        short_req = self._create_prefill_req(input_len=32768, extend_len=8)

        result = adder.add_one_req(
            short_req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertEqual(result, AddReqResult.OTHER)
        self.assertEqual(adder.can_run_list, [short_req])

    def test_attncp_owner_rotation_is_assigned_round_robin_once(self):
        allocator = self._create_cp_sharded_allocator(cp_size=4)
        adder = self.create_adder(
            self.create_running_batch(),
            token_to_kv_pool_allocator=allocator,
            next_attn_cp_owner_rotation=0,
        )
        reqs = [
            self._create_prefill_req(input_len=8, extend_len=8) for _ in range(5)
        ]

        rotations = [adder._ensure_cp_sharded_owner_rotation(req) for req in reqs]

        self.assertEqual(rotations, [0, 1, 2, 3, 0])
        self.assertEqual(adder._ensure_cp_sharded_owner_rotation(reqs[1]), 1)
        self.assertEqual(adder.next_attn_cp_owner_rotation, 5)

    def test_attncp_phase1_rejects_second_prefill_request(self):
        allocator = self._create_cp_sharded_allocator()
        adder = self.create_adder(
            self.create_running_batch(),
            token_to_kv_pool_allocator=allocator,
            rem_chunk_tokens=16,
        )
        first = self._create_prefill_req(input_len=8, extend_len=8)
        second = self._create_prefill_req(input_len=8, extend_len=8)

        first_result = adder.add_one_req(
            first, has_chunked_req=False, truncation_align_size=None
        )
        second_result = adder.add_one_req(
            second, has_chunked_req=False, truncation_align_size=None
        )

        self.assertIn(first_result, (AddReqResult.CONTINUE, AddReqResult.OTHER))
        self.assertEqual(second_result, AddReqResult.OTHER)
        self.assertEqual(adder.can_run_list, [first])
        self.assertIsNone(second.attn_cp_owner_rotation)
        self.assertTrue(adder.prefill_request_limit_reached())

    def test_attncp_chunked_prefill_spec_uses_final_truncated_interval(self):
        allocator = self._create_cp_sharded_allocator(cp_size=2)
        adder = self.create_adder(
            self.create_running_batch(),
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            rem_chunk_tokens=4,
        )
        req = self._create_prefill_req(input_len=8, extend_len=8)

        result = adder.add_one_req(
            req, has_chunked_req=False, truncation_align_size=None
        )

        self.assertIn(result, (AddReqResult.CONTINUE, AddReqResult.OTHER))
        spec = req.attn_cp_prefill_split_spec
        self.assertEqual(spec.extend_start, 0)
        self.assertEqual(spec.extend_len, 4)
        self.assertEqual(spec.owner_rotation, req.attn_cp_owner_rotation)
        self.assertEqual(tuple(block.logical_start for block in spec.blocks), (0,))
        self.assertEqual(tuple(block.token_count for block in spec.blocks), (4,))

    def test_attncp_page_size_one_keeps_legacy_prefill_without_split_spec(self):
        allocator = self._create_cp_sharded_allocator(
            physical_size=32,
            logical_size=64,
            cp_size=2,
            page_size=1,
            cp_kv_chunk_size=4,
        )
        adder = self.create_adder(
            self.create_running_batch(),
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            rem_chunk_tokens=8,
        )
        req = self._create_prefill_req(input_len=8, extend_len=8)

        result = adder.add_one_req(
            req,
            has_chunked_req=False,
            truncation_align_size=None,
        )

        self.assertIn(result, (AddReqResult.CONTINUE, AddReqResult.OTHER))
        self.assertIsNone(req.attn_cp_prefill_split_spec)

    def test_attncp_final_chunk_spec_precedes_lock_and_capacity_check(self):
        allocator = self._create_cp_sharded_allocator(cp_size=2)
        tree_cache = self.create_tree_cache()
        adder = self.create_adder(
            self.create_running_batch(),
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            rem_chunk_tokens=4,
        )
        req = self._create_prefill_req(input_len=8, extend_len=8)
        req.attn_cp_prefill_split_spec = None
        events = []
        original_build = schedule_policy.build_cp_prefill_split_spec

        def record_build(**kwargs):
            events.append(("build", kwargs["extend_start"], kwargs["extend_len"]))
            return original_build(**kwargs)

        def record_lock(_last_node):
            events.append(("lock",))
            return IncLockRefResult()

        def reject_capacity(*_args, **_kwargs):
            events.append(("capacity",))
            return False

        tree_cache.inc_lock_ref.side_effect = record_lock
        with patch.object(
            schedule_policy,
            "build_cp_prefill_split_spec",
            side_effect=record_build,
        ), patch.object(
            adder,
            "_cp_sharded_can_allocate_extend",
            side_effect=reject_capacity,
        ):
            result = adder.add_one_req(
                req, has_chunked_req=False, truncation_align_size=None
            )

        self.assertEqual(result, AddReqResult.NO_TOKEN)
        self.assertEqual(events[0], ("build", 0, 4))
        self.assertLess(events.index(("build", 0, 4)), events.index(("lock",)))
        self.assertLess(events.index(("build", 0, 4)), events.index(("capacity",)))
        self.assertIsNone(req.attn_cp_prefill_split_spec)
        self.assertEqual(adder.can_run_list, [])

    def test_attncp_chunk_continuation_failure_does_not_retain_spec(self):
        allocator = self._create_cp_sharded_allocator(cp_size=2)
        adder = self.create_adder(
            self.create_running_batch(),
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            rem_chunk_tokens=4,
        )
        req = self._create_prefill_req(input_len=12, extend_len=8)
        req.prefix_indices = list(range(4))
        req.fill_ids = list(range(12))
        req.attn_cp_owner_rotation = 1
        req.attn_cp_prefill_split_spec = object()
        events = []
        original_build = schedule_policy.build_cp_prefill_split_spec

        def record_build(**kwargs):
            events.append(("build", kwargs["extend_start"], kwargs["extend_len"]))
            return original_build(**kwargs)

        def reject_capacity(*_args, **_kwargs):
            events.append(("capacity",))
            return False

        with patch.object(
            schedule_policy,
            "build_cp_prefill_split_spec",
            side_effect=record_build,
        ), patch.object(
            adder,
            "_cp_sharded_can_allocate_extend",
            side_effect=reject_capacity,
        ):
            remaining = adder.add_chunked_req(req)

        self.assertIs(remaining, req)
        self.assertEqual(events, [("build", 4, 4), ("capacity",)])
        self.assertIsNone(req.attn_cp_prefill_split_spec)
        self.assertEqual(adder.can_run_list, [])

    def test_attncp_chunk_continuation_replaces_previous_interval_spec(self):
        allocator = self._create_cp_sharded_allocator(cp_size=2)
        adder = self.create_adder(
            self.create_running_batch(),
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            rem_chunk_tokens=4,
        )
        req = self._create_prefill_req(input_len=12, extend_len=8)
        req.prefix_indices = list(range(4))
        req.fill_ids = list(range(12))
        req.attn_cp_owner_rotation = 1
        req.attn_cp_prefill_split_spec = object()

        remaining = adder.add_chunked_req(req)

        self.assertIs(remaining, req)
        spec = req.attn_cp_prefill_split_spec
        self.assertEqual((spec.extend_start, spec.extend_len), (4, 4))
        self.assertEqual(spec.owner_rotation, 1)

    def test_attncp_unaligned_prefix_fails_before_cache_lock(self):
        allocator = self._create_cp_sharded_allocator(cp_size=2)
        tree_cache = self.create_tree_cache()
        adder = self.create_adder(
            self.create_running_batch(),
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            rem_chunk_tokens=None,
        )
        req = self._create_prefill_req(input_len=6, extend_len=4)
        req.prefix_indices = list(range(2))
        req.fill_ids = list(range(6))

        with self.assertRaisesRegex(ValueError, "recorded leading-page owner"):
            adder.add_one_req(
                req, has_chunked_req=False, truncation_align_size=None
            )

        tree_cache.inc_lock_ref.assert_not_called()
        self.assertEqual(adder.can_run_list, [])

    def test_attncp_partial_prefix_uses_recorded_leading_page_owner(self):
        allocator = self._create_cp_sharded_allocator(
            physical_size=32,
            logical_size=64,
            cp_size=2,
            page_size=4,
            cp_kv_chunk_size=8,
        )
        prefix_spec = build_cp_prefill_split_spec(
            extend_start=0,
            extend_len=6,
            cp_size=2,
            page_size=4,
            owner_rotation=0,
        )
        prefix = allocator.alloc_extend_with_logical(
            prefix_lens=torch.tensor([0], dtype=torch.int64),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([6], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([6], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64),
            extend_num_tokens=6,
            split_spec=prefix_spec,
        )
        tree_cache = self.create_tree_cache()
        adder = self.create_adder(
            self.create_running_batch(),
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            rem_chunk_tokens=None,
        )
        req = self._create_prefill_req(input_len=8, extend_len=2)
        req.prefix_indices = prefix.logical_slots
        req.fill_ids = list(range(8))
        req.attn_cp_owner_rotation = 1

        result = adder.add_one_req(
            req,
            has_chunked_req=False,
            truncation_align_size=None,
        )

        self.assertIn(result, (AddReqResult.CONTINUE, AddReqResult.OTHER))
        spec = req.attn_cp_prefill_split_spec
        self.assertEqual((spec.extend_start, spec.extend_len), (6, 2))
        self.assertEqual(len(spec.blocks), 1)
        self.assertEqual(spec.blocks[0].owner_rank, 1)
        self.assertEqual(spec.page_demand(4), (0, 0))
        self.assertEqual(adder.cp_sharded_token_offsets, [0, 0])

    def test_attncp_hicache_hit_fails_before_lock_or_load_back(self):
        allocator = self._create_cp_sharded_allocator(cp_size=2)
        tree_cache = self.create_tree_cache()
        tree_cache.init_load_back.return_value = (torch.arange(4), MagicMock())
        adder = self.create_adder(
            self.create_running_batch(),
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            rem_chunk_tokens=None,
        )
        req = self._create_prefill_req(input_len=8, extend_len=8)
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.host_hit_length = 4
        req.best_match_node = MagicMock()
        req.attn_cp_prefill_split_spec = None

        with self.assertRaisesRegex(ValueError, "HiCache"):
            adder.add_one_req(
                req, has_chunked_req=False, truncation_align_size=None
            )

        tree_cache.inc_lock_ref.assert_not_called()
        tree_cache.init_load_back.assert_not_called()
        self.assertIsNone(req.attn_cp_prefill_split_spec)
        self.assertEqual(adder.can_run_list, [])

    def test_attncp_hicache_hit_fails_before_ignore_eos_dispatch(self):
        allocator = self._create_cp_sharded_allocator(cp_size=2)
        tree_cache = self.create_tree_cache()
        tree_cache.disable = True
        adder = self.create_adder(
            self.create_running_batch(),
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            rem_chunk_tokens=None,
        )
        req = self._create_prefill_req(input_len=8, extend_len=8)
        req.host_hit_length = 4
        req.sampling_params.ignore_eos = True
        req.attn_cp_prefill_split_spec = None

        with patch.object(adder, "add_one_req_ignore_eos") as add_ignore_eos:
            with self.assertRaisesRegex(ValueError, "HiCache"):
                adder.add_one_req(
                    req, has_chunked_req=False, truncation_align_size=None
                )

        add_ignore_eos.assert_not_called()
        self.assertIsNone(req.attn_cp_prefill_split_spec)

    def test_attncp_admission_evicts_for_deficient_owner_without_collective(self):
        allocator = self._create_cp_sharded_allocator(
            physical_size=4,
            logical_size=8,
        )
        cached = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 4, dtype=torch.int64)
        )
        tree_cache = self.create_tree_cache(evictable_size=4)

        def evict(_params):
            allocator.free(cached.logical_slots)
            return EvictResult(num_tokens_evicted=4)

        tree_cache.evict.side_effect = evict
        adder = self.create_adder(
            self.create_running_batch(),
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
        )
        req = self._create_prefill_req(input_len=4, extend_len=4)

        with patch.object(torch.distributed, "all_reduce") as all_reduce:
            result = adder.add_one_req(
                req,
                has_chunked_req=False,
                truncation_align_size=None,
            )

        self.assertIn(result, (AddReqResult.CONTINUE, AddReqResult.OTHER))
        self.assertEqual(adder.can_run_list, [req])
        tree_cache.evict.assert_called_once()
        all_reduce.assert_not_called()

    def test_attncp_admission_uses_explicit_split_page_demand(self):
        allocator = self._create_cp_sharded_allocator(
            physical_size=8,
            logical_size=32,
            cp_size=2,
            page_size=4,
            cp_kv_chunk_size=8,
        )
        cached = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 8, dtype=torch.int64)
        )
        self.assertEqual(
            allocator.physical_load_snapshot().allocated_tokens,
            (8, 0),
        )
        tree_cache = self.create_tree_cache(evictable_size=8)

        def evict(_params):
            allocator.free(cached.logical_slots)
            return EvictResult(num_tokens_evicted=8)

        tree_cache.evict.side_effect = evict
        adder = self.create_adder(
            self.create_running_batch(),
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=allocator,
            page_size=4,
            rem_chunk_tokens=8,
        )
        req = self._create_prefill_req(input_len=8, extend_len=8)
        req.attn_cp_owner_rotation = 1

        with patch.object(torch.distributed, "all_reduce") as all_reduce:
            result = adder.add_one_req(
                req,
                has_chunked_req=False,
                truncation_align_size=None,
            )

        self.assertIn(result, (AddReqResult.CONTINUE, AddReqResult.OTHER))
        self.assertEqual(req.attn_cp_prefill_split_spec.page_demand(4), (1, 1))
        self.assertEqual(adder.cp_sharded_token_offsets, [4, 4])
        tree_cache.evict.assert_called_once()
        all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main()
