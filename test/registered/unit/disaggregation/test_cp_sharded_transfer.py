import threading
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

import sglang.srt.disaggregation.mooncake.conn as mooncake_conn
from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferInfo,
    TransferKVChunk,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import resolve_cp_sharded_transfer_page_runs
from sglang.srt.managers import schedule_batch as schedule_batch_module
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.cp_sharded_allocator import CPShardedKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.models.welm_deferred_mirror import WelmPDExecutionMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _DummySWAKVPool(BaseSWAKVPool):
    def __init__(self):
        self.full_kv_pool = None
        self.swa_kv_pool = None
        self.full_to_swa_index_mapping = None

    def register_mapping(self, mapping):
        self.full_to_swa_index_mapping = mapping

    def translate_loc_from_full_to_swa(self, kv_indices):
        return self.full_to_swa_index_mapping[kv_indices]

    def invalidate_loc_cache(self):
        pass

    def set_swa_loc(self, loc):
        pass

    def get_state_buf_infos(self):
        return [], [], []

    def get_key_buffer(self, layer_id):
        raise NotImplementedError()

    def get_value_buffer(self, layer_id):
        raise NotImplementedError()

    def get_kv_buffer(self, layer_id):
        raise NotImplementedError()

    def set_kv_buffer(self, layer, loc, cache_k, cache_v):
        raise NotImplementedError()


class TestCPShardedTransfer(unittest.TestCase):
    def test_decode_prealloc_rejects_prefill_owner_layout(self):
        base = PagedTokenToKVPoolAllocator(
            size=16,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
        )

        with self.assertRaisesRegex(RuntimeError, "Decode owner layout"):
            DecodePreallocQueue(
                req_to_token_pool=None,
                token_to_kv_pool_allocator=allocator,
                draft_token_to_kv_pool=None,
                welm_mtp_kv_mirror_state_buffers=None,
                req_to_metadata_buffer_idx_allocator=None,
                metadata_buffers=None,
                scheduler=None,
                transfer_queue=None,
                tree_cache=None,
                gloo_group=None,
                tp_rank=0,
                tp_size=1,
                dp_size=1,
                gpu_id=0,
                bootstrap_port=0,
                max_total_num_tokens=16,
                pp_rank=0,
                num_reserved_decode_tokens=0,
                transfer_backend=None,
            )

    def test_mooncake_metadata_advertises_decode_cp_owner_layout(self):
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.bootstrap_infos = [{"is_dummy": False}]
        receiver.bootstrap_room = 7
        receiver.session_id = "session:1"
        receiver.required_dst_info_num = 1
        receiver.kv_mgr = SimpleNamespace(
            enable_staging=False,
            is_cp_sharded_kv=True,
            local_ip="127.0.0.1",
            rank_port=12345,
            record_failure=MagicMock(),
            update_status=MagicMock(),
        )
        socket = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(socket, nullcontext())
        )

        receiver.send_metadata(
            np.array([11, 0, 12], dtype=np.int32),
            aux_index=3,
            state_indices=[],
            decode_prefix_len=0,
        )

        message = socket.send_multipart.call_args.args[0]
        self.assertEqual(message[9], b"1")
        decoded = TransferInfo.from_zmq(message)
        self.assertTrue(decoded.dst_is_cp_sharded_kv)

    def test_mooncake_registration_uses_runtime_attention_tp_rank(self):
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.bootstrap_infos = [{"is_dummy": False}]
        receiver.session_id = "session:1"
        receiver.kv_mgr = SimpleNamespace(
            kv_args=SimpleNamespace(
                engine_rank=3,
                kv_data_ptrs=[100],
                aux_data_ptrs=[200],
                state_data_ptrs=[],
                state_item_lens=[],
                state_dim_per_tensor=[],
                kv_item_lens=[64],
            ),
            attn_tp_rank=1,
            attn_tp_size=2,
            local_ip="127.0.0.1",
            rank_port=12345,
            enable_staging=False,
            server_args=SimpleNamespace(enable_hisparse=False),
        )
        socket = MagicMock()
        receiver._connect_to_bootstrap_server = MagicMock(
            return_value=(socket, nullcontext())
        )

        receiver._register_kv_args()

        message = socket.send_multipart.call_args.args[0]
        self.assertEqual(message[7], b"1")

    def test_decode_transfer_poll_converges_across_attntp_and_cp_groups(self):
        class _FakeCPAllocator:
            pass

        poller = MagicMock()
        req = SimpleNamespace(bootstrap_host="prefill", rid="req")
        queue = object.__new__(DecodeTransferQueue)
        queue.queue = [SimpleNamespace(req=req, kv_receiver=poller)]
        queue.enable_staging = False
        queue.gloo_group = object()
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
            token_to_kv_pool_allocator=_FakeCPAllocator(),
            server_args=SimpleNamespace(disaggregation_transfer_backend="mooncake"),
        )

        with (
            patch(
                "sglang.srt.disaggregation.decode.CPShardedKVPoolAllocator",
                _FakeCPAllocator,
            ),
            patch(
                "sglang.srt.disaggregation.decode."
                "poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Transferring],
            ) as cp_poll,
            patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                return_value=[KVPoll.Transferring],
            ) as legacy_poll,
        ):
            transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        cp_poll.assert_called_once_with(
            [poller],
            queue.scheduler.attn_cp_cpu_group,
            queue.scheduler.attn_tp_cpu_group,
        )
        legacy_poll.assert_not_called()

    def test_decode_transfer_poll_preserves_legacy_group_for_non_cp(self):
        poller = MagicMock()
        req = SimpleNamespace(bootstrap_host="prefill", rid="req")
        queue = object.__new__(DecodeTransferQueue)
        queue.queue = [SimpleNamespace(req=req, kv_receiver=poller)]
        queue.enable_staging = False
        queue.gloo_group = object()
        queue.scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=object(),
            server_args=SimpleNamespace(disaggregation_transfer_backend="mooncake"),
        )

        with (
            patch(
                "sglang.srt.disaggregation.decode."
                "poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Transferring],
            ) as cp_poll,
            patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                return_value=[KVPoll.Transferring],
            ) as legacy_poll,
        ):
            transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        legacy_poll.assert_called_once_with([poller], queue.gloo_group)
        cp_poll.assert_not_called()

    def test_decode_prealloc_handshake_converges_across_attntp_and_cp_groups(self):
        class _FakeCPAllocator:
            pass

        poller = MagicMock()
        req = SimpleNamespace(
            rid="req",
            bootstrap_room=7,
            bootstrap_host="prefill",
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=poller,
            waiting_for_input=False,
        )
        queue = object.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.gloo_group = object()
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=object(),
            attn_tp_cpu_group=object(),
            token_to_kv_pool_allocator=_FakeCPAllocator(),
            server_args=SimpleNamespace(disaggregation_transfer_backend="mooncake"),
        )

        with (
            patch(
                "sglang.srt.disaggregation.decode.CPShardedKVPoolAllocator",
                _FakeCPAllocator,
            ),
            patch(
                "sglang.srt.disaggregation.decode."
                "poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Bootstrapping],
            ) as cp_poll,
            patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                return_value=[KVPoll.Bootstrapping],
            ) as legacy_poll,
        ):
            queue._update_handshake_waiters()

        cp_poll.assert_called_once_with(
            [poller],
            queue.scheduler.attn_cp_cpu_group,
            queue.scheduler.attn_tp_cpu_group,
        )
        legacy_poll.assert_not_called()

    def test_decode_cp_resume_waits_for_owner_physical_capacity(self):
        class _FakeCPAllocator:
            pass

        allocator = _FakeCPAllocator()
        allocator.page_size = 16
        allocator.alloc_size_per_rank_for_range = MagicMock(
            return_value=(49152, 46384)
        )
        req = SimpleNamespace(
            rid="req",
            is_retracted=True,
            load_kv_cache=MagicMock(),
        )
        queue = object.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool_allocator = allocator
        queue.tree_cache = object()
        queue.retracted_queue = [req]
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: 1)
        queue._uses_swa_tail_prealloc = lambda: False
        queue._allocatable_token_budgets = MagicMock(return_value=108240)
        queue._prealloc_required_tokens = MagicMock(
            return_value=(95536, 95536)
        )
        queue._prealloc_kv_lens = MagicMock(return_value=(95524, 95524))
        queue._pre_alloc = MagicMock()

        with patch(
            "sglang.srt.disaggregation.decode.CPShardedKVPoolAllocator",
            _FakeCPAllocator,
        ), patch(
            "sglang.srt.disaggregation.decode.ensure_cp_sharded_kv_capacity",
            return_value=False,
        ) as ensure_capacity:
            resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, [])
        self.assertEqual(queue.retracted_queue, [req])
        self.assertTrue(req.is_retracted)
        queue._pre_alloc.assert_not_called()
        req.load_kv_cache.assert_not_called()
        allocator.alloc_size_per_rank_for_range.assert_called_once_with(0, 95524)
        ensure_capacity.assert_called_once_with(
            allocator=allocator,
            tree_cache=queue.tree_cache,
            demand_tokens=(49152, 46384),
        )

    def test_decode_cp_initial_prealloc_waits_for_owner_physical_capacity(self):
        req = MagicMock()
        req.rid = "req"
        req.origin_input_ids = list(range(8))
        req.output_ids = []
        req.finished_reason = None
        req.cache_protected_len = 0
        req.sampling_params.max_new_tokens = 16
        decode_req = SimpleNamespace(req=req, waiting_for_input=True)

        queue = object.__new__(DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = lambda: False
        queue._allocatable_token_budgets = MagicMock(return_value=1000)
        queue._ensure_cp_sharded_prompt_capacity = MagicMock(
            return_value=False
        )
        queue._pre_alloc = MagicMock()
        queue.transfer_queue = SimpleNamespace(queue=[], enable_staging=False)
        queue.tree_cache = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
        queue.token_to_kv_pool = MagicMock()
        queue.token_to_kv_pool_allocator = MagicMock(page_size=4)
        queue.scheduler = SimpleNamespace(
            running_batch=SimpleNamespace(reqs=[]),
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
            enable_hisparse=False,
            waiting_queue=[],
            last_batch=None,
            server_args=SimpleNamespace(
                disaggregation_decode_enable_radix_cache=False
            ),
            stream_output=MagicMock(),
        )

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [])
        self.assertEqual(queue.queue, [decode_req])
        queue._ensure_cp_sharded_prompt_capacity.assert_called_once_with(
            prefix_len=0, fill_len=8
        )
        queue._pre_alloc.assert_not_called()

    def test_decode_cp_keeps_original_extend_allocation_for_swa(self):
        class _FakeSWAKVPool:
            pass

        class _FakeCPAllocator:
            page_size = 16

            def alloc_extend_swa_tail(self):
                raise AssertionError("CP decode must not use SWA tail allocation")

        queue = object.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool = _FakeSWAKVPool()
        queue.token_to_kv_pool_allocator = _FakeCPAllocator()

        with patch(
            "sglang.srt.disaggregation.decode.SWAKVPool", _FakeSWAKVPool
        ), patch(
            "sglang.srt.disaggregation.decode.CPShardedKVPoolAllocator",
            _FakeCPAllocator,
        ):
            self.assertFalse(queue._uses_swa_tail_prealloc())

    def test_decode_cp_prealloc_uses_independent_round_robin_layout(self):
        page_size = 4
        seq_len = 32

        for cp_rank in range(2):
            with self.subTest(cp_rank=cp_rank):
                base = SWATokenToKVPoolAllocator(
                    size=64,
                    size_swa=64,
                    page_size=page_size,
                    dtype=torch.float32,
                    device="cpu",
                    kvcache=_DummySWAKVPool(),
                    need_sort=False,
                )
                allocator = CPShardedKVPoolAllocator(
                    base,
                    cp_rank=cp_rank,
                    cp_size=2,
                    cp_kv_chunk_size=8,
                    logical_size=128,
                    logical_full_size=128,
                    logical_swa_size=128,
                    use_decode_owner_layout=True,
                )
                queue = object.__new__(DecodePreallocQueue)
                queue.token_to_kv_pool_allocator = allocator
                queue.req_to_token_pool = ReqToTokenPool(
                    size=2,
                    max_context_len=64,
                    device="cpu",
                    enable_memory_saver=False,
                )
                queue.scheduler = SimpleNamespace(
                    server_args=SimpleNamespace(
                        disaggregation_decode_enable_radix_cache=False
                    ),
                    enable_hisparse=False,
                )
                queue.tree_cache = MagicMock()
                queue._uses_swa_tail_prealloc = lambda: False
                req = SimpleNamespace(
                    req_pool_idx=None,
                    is_chunked=0,
                    kv_committed_len=0,
                    origin_input_ids=list(range(seq_len)),
                    output_ids=[],
                    rid=f"req-{cp_rank}",
                    # Decode must not consume Prefill's request rotation.
                    attn_cp_owner_rotation=1,
                    set_extend_input_len=MagicMock(),
                )

                physical_dst = queue._pre_alloc(req)
                logical_slots = queue.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :seq_len
                ].to(dtype=torch.int64)

                self.assertEqual(logical_slots.tolist(), list(range(4, 36)))
                self.assertNotEqual(physical_dst.tolist(), logical_slots.tolist())
                self.assertEqual(req.attn_cp_owner_rotation, 1)
                self.assertEqual(
                    physical_dst.ne(0).reshape(-1, page_size)[:, 0].tolist(),
                    [True, True, False, False, True, True, False, False]
                    if cp_rank == 0
                    else [False, False, True, True, False, False, True, True],
                )

                expected_physical = allocator.logical_slots_to_physical(
                    logical_slots
                )
                base.get_cpu_copy = MagicMock(return_value="cpu-copy")
                base.load_cpu_copy = MagicMock()
                self.assertEqual(allocator.get_cpu_copy(logical_slots), "cpu-copy")
                base.get_cpu_copy.assert_called_once()
                torch.testing.assert_close(
                    base.get_cpu_copy.call_args.args[0], expected_physical
                )
                allocator.load_cpu_copy("cpu-copy", logical_slots)
                base.load_cpu_copy.assert_called_once()
                self.assertEqual(base.load_cpu_copy.call_args.args[0], "cpu-copy")
                torch.testing.assert_close(
                    base.load_cpu_copy.call_args.args[1], expected_physical
                )

                chunk_cache = object.__new__(ChunkCache)
                chunk_cache.req_to_token_pool = queue.req_to_token_pool
                chunk_cache.token_to_kv_pool_allocator = allocator
                chunk_cache.scale_seq_factor = 1
                req.pop_committed_kv_cache = MagicMock(return_value=seq_len)
                chunk_cache.cache_finished_req(req)
                queue.req_to_token_pool.free(req)
                self.assertEqual(allocator.available_size(), 128)
                self.assertEqual(base.full_attn_allocator.available_size(), 64)
                self.assertEqual(base.swa_attn_allocator.available_size(), 64)

                resumed_dst = queue._pre_alloc(req)
                resumed_logical = queue.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :seq_len
                ].to(dtype=torch.int64)
                self.assertEqual(resumed_logical.numel(), seq_len)
                self.assertEqual(torch.unique(resumed_logical).numel(), seq_len)
                self.assertTrue(torch.all(resumed_logical > 0).item())
                self.assertEqual(
                    resumed_dst.ne(0).reshape(-1, page_size)[:, 0].tolist(),
                    physical_dst.ne(0).reshape(-1, page_size)[:, 0].tolist(),
                )

                base.load_cpu_copy.reset_mock()
                allocator.load_cpu_copy("cpu-copy", resumed_logical)
                base.load_cpu_copy.assert_called_once()
                torch.testing.assert_close(
                    base.load_cpu_copy.call_args.args[1],
                    allocator.logical_slots_to_physical(resumed_logical),
                )

                chunk_cache.cache_finished_req(req)
                queue.req_to_token_pool.free(req)
                self.assertEqual(allocator.available_size(), 128)
                self.assertEqual(base.full_attn_allocator.available_size(), 64)
                self.assertEqual(base.swa_attn_allocator.available_size(), 64)

    def test_deferred_decode_cp_preallocates_only_committed_prefix_with_fragmentation(
        self,
    ):
        page_size = 4
        prompt = list(range(17))
        base = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=page_size,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        base.free_pages = torch.tensor(
            [7, 2, 9, 4, 1, 3, 5, 8, 10, 12, 14, 16, 6, 11, 13, 15],
            dtype=torch.int64,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
            logical_size=128,
            use_decode_owner_layout=True,
        )
        allocator.logical_free_pages = torch.tensor(
            [9, 2, 12, 5, 11, 3, 10, 4, 1, 6, 7, 8, 13, 14, 15, 16],
            dtype=torch.int64,
        )
        queue = object.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool_allocator = allocator
        queue.req_to_token_pool = ReqToTokenPool(
            size=2,
            max_context_len=64,
            device="cpu",
            enable_memory_saver=False,
        )
        queue.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                welm_kv_mirror_pd_mode=(
                    WelmPDExecutionMode.DEFERRED_LAST_PROMPT.value
                ),
                disaggregation_decode_enable_radix_cache=False,
            ),
            enable_hisparse=False,
        )
        queue.tree_cache = MagicMock()
        queue._uses_swa_tail_prealloc = lambda: False
        req = SimpleNamespace(
            req_pool_idx=None,
            is_chunked=0,
            kv_committed_len=0,
            origin_input_ids=prompt,
            output_ids=[],
            welm_deferred_decode_state=(
                schedule_batch_module.WelmDeferredDecodeState.from_prompt_tokens(
                    prompt
                )
            ),
            rid="deferred-fragmented",
            set_extend_input_len=MagicMock(),
        )

        physical_dst = queue._pre_alloc(req)
        committed_len = len(prompt) - 1
        logical_slots = queue.req_to_token_pool.req_to_token[
            req.req_pool_idx, :committed_len
        ].to(dtype=torch.int64)

        self.assertEqual(req.kv_allocated_len, committed_len)
        self.assertEqual(req.kv_committed_len, committed_len)
        self.assertEqual(req.fill_ids, prompt[:-1])
        self.assertEqual(logical_slots.numel(), committed_len)
        self.assertEqual(torch.unique(logical_slots).numel(), committed_len)
        self.assertFalse(torch.equal(physical_dst, logical_slots))
        torch.testing.assert_close(
            physical_dst,
            allocator.logical_slots_to_physical(logical_slots),
        )

    @staticmethod
    def _make_sender(allocator, logical_page_count, *, state_types=None):
        manager = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            kv_args=SimpleNamespace(state_types=state_types or []),
            add_transfer_request=MagicMock(),
        )
        sender = object.__new__(MooncakeKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = 7
        sender.curr_idx = 0
        sender.num_kv_indices = logical_page_count
        sender.aux_index = 3
        sender._transfer_num_kv_indices = 0
        sender._transfer_num_state_indices = 0
        return sender, manager

    @staticmethod
    def _make_transfer_worker_manager(room):
        manager = object.__new__(MooncakeKVManager)
        manager.enable_staging = False
        manager.bootstrap_port = 8998
        manager.routing_attn_tp_rank = 0
        manager.routing_attn_cp_size = 2
        manager.routing_attn_cp_rank = 0
        manager.pp_size = 1
        manager.pp_rank = 0
        manager.session_lock = threading.Lock()
        manager.failed_sessions = set()
        manager.is_mla_backend = False
        manager.is_cp_sharded_kv = True
        manager.attn_tp_size = 4
        manager.request_status = {room: KVPoll.Transferring}
        manager.req_to_decode_prefix_len = {room: 0}
        manager.send_aux = MagicMock(return_value=0)
        manager.sync_status_to_decode_endpoint = MagicMock()
        manager.record_failure = MagicMock()

        def update_status(update_room, status):
            manager.request_status[update_room] = status

        manager.update_status = MagicMock(side_effect=update_status)
        manager.check_status = lambda check_room: manager.request_status[check_room]

        transfer_infos = {}
        manager.decode_kv_args_table = {}
        for idx in range(2):
            session_id = f"session:{100 + idx}"
            transfer_infos[session_id] = TransferInfo(
                room=room,
                endpoint="127.0.0.1",
                dst_port=9000 + idx,
                mooncake_session_id=session_id,
                dst_kv_indices=np.array([11, 12], dtype=np.int32),
                dst_aux_index=idx,
                dst_state_indices=[],
                required_dst_info_num=2,
                is_dummy=False,
            )
            manager.decode_kv_args_table[session_id] = SimpleNamespace(
                dst_aux_ptrs=[0]
            )
        manager.transfer_infos = {room: transfer_infos}
        return manager

    @staticmethod
    def _run_one_transfer_chunk(manager, chunk):
        class _OneShotQueue:
            def __init__(self):
                self.used = False

            def get(self):
                if self.used:
                    raise RuntimeError("stop test worker")
                self.used = True
                return chunk

        manager.transfer_worker(_OneShotQueue(), MagicMock())

    def test_mooncake_staging_rejects_cp_sharded_owner_runs(self):
        with self.assertRaisesRegex(ValueError, "staging.*sharded-kv"):
            mooncake_conn.validate_mooncake_staging_configuration(
                enable_staging=True,
                is_cp_sharded_kv=True,
            )

    def test_fragmented_logical_and_physical_pages_preserve_destination_runs(self):
        base = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        # Keep the pool fragmented so a four-page allocation uses the listed pages.
        base.free_pages = torch.tensor(
            [7, 2, 9, 4, 1, 3, 5, 8, 10, 12, 14, 16, 6, 11, 13, 15],
            dtype=torch.int64,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
            logical_size=64,
        )
        allocator.logical_free_pages = torch.tensor(
            [9, 2, 12, 5, 11, 3, 10, 4, 1, 6, 7, 8, 13, 14, 15, 16],
            dtype=torch.int64,
        )
        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 32, dtype=torch.int64)
        )
        logical_pages = (allocation.logical_slots[::4] // 4).numpy()

        np.testing.assert_array_equal(
            logical_pages,
            np.array([9, 2, 12, 5, 11, 3, 10, 4]),
        )

        runs = resolve_cp_sharded_transfer_page_runs(
            allocator,
            logical_pages,
            slice(11, 19),
        )

        self.assertEqual(len(runs), 2)
        np.testing.assert_array_equal(runs[0][0], np.array([7, 2]))
        self.assertEqual(runs[0][1], slice(11, 13))
        np.testing.assert_array_equal(runs[1][0], np.array([9, 4]))
        self.assertEqual(runs[1][1], slice(17, 19))

    def test_radix_prefix_hit_branch_sends_cached_prefix_and_new_tail(self):
        base = PagedTokenToKVPoolAllocator(
            size=64,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
            logical_size=128,
        )
        first_path = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 32, dtype=torch.int64)
        )
        tree = RadixCache.create_simulated(
            mock_allocator=allocator,
            page_size=4,
        )
        first_tokens = list(range(32))
        tree.insert(
            InsertParams(
                key=RadixKey(first_tokens),
                value=first_path.logical_slots,
            )
        )

        cached_prefix = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(first_tokens[:24]))
        ).device_indices
        new_tail = allocator.alloc_for_positions_with_logical(
            torch.arange(24, 32, dtype=torch.int64)
        )
        second_path = torch.cat((cached_prefix, new_tail.logical_slots))
        second_tokens = first_tokens[:24] + list(range(100, 108))
        tree.insert(
            InsertParams(
                key=RadixKey(second_tokens),
                value=second_path,
            )
        )

        first_match = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(first_tokens))
        ).device_indices
        second_match = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(second_tokens))
        ).device_indices
        self.assertEqual(first_match[:24].tolist(), second_match[:24].tolist())
        self.assertNotEqual(first_match[24:].tolist(), second_match[24:].tolist())

        logical_pages = (second_match[::4] // 4).numpy()
        sender, manager = self._make_sender(allocator, len(logical_pages))
        req = SimpleNamespace(
            start_send_idx=0,
            fill_ids=second_tokens,
            origin_input_ids=second_tokens,
            req_pool_idx=0,
            disagg_kv_sender=sender,
        )
        metadata_buffers = SimpleNamespace(set_buf=MagicMock())
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=SimpleNamespace(
                req_to_token=second_match.reshape(1, -1)
            ),
            disagg_metadata_buffers=metadata_buffers,
            disagg_prefill_bootstrap_queue=SimpleNamespace(kv_manager=manager),
        )
        SchedulerDisaggregationPrefillMixin.send_kv_chunk(
            scheduler,
            req,
            last_chunk=True,
        )

        calls = manager.add_transfer_request.call_args_list
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0].args[1], np.array([1, 2]))
        self.assertEqual(calls[0].args[2], slice(0, 2))
        self.assertFalse(calls[0].args[3])
        np.testing.assert_array_equal(calls[1].args[1], np.array([5, 6]))
        self.assertEqual(calls[1].args[2], slice(6, 8))
        self.assertTrue(calls[1].args[3])
        self.assertEqual(req.start_send_idx, 32)
        metadata_buffers.set_buf.assert_called_once_with(req)

    def test_owner_without_local_pages_in_last_chunk_enqueues_completion(self):
        base = PagedTokenToKVPoolAllocator(
            size=32,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
            logical_size=64,
        )
        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 16, dtype=torch.int64)
        )
        logical_pages = (allocation.logical_slots[::4] // 4).numpy()
        sender, manager = self._make_sender(
            allocator,
            len(logical_pages),
            state_types=[StateType.WELM_MTP_MIRROR],
        )

        sender.send(logical_pages[:2])
        sender.send(logical_pages[2:], [allocation.logical_slots.numpy()])

        calls = manager.add_transfer_request.call_args_list
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0].args[1], np.array([1, 2]))
        self.assertEqual(calls[0].args[2], slice(0, 2))
        self.assertFalse(calls[0].args[3])
        self.assertEqual(calls[1].args[1].size, 0)
        self.assertEqual(calls[1].args[2], slice(2, 2))
        self.assertTrue(calls[1].args[3])
        self.assertEqual(calls[1].kwargs["aux_index"], 3)
        np.testing.assert_array_equal(
            calls[1].kwargs["state_indices"][0],
            allocation.physical_write_slots[:8].numpy(),
        )
        np.testing.assert_array_equal(
            calls[1].kwargs["state_index_positions"][0],
            np.arange(8),
        )

    def test_empty_last_chunk_completes_all_decode_destinations(self):
        room = 7
        manager = self._make_transfer_worker_manager(room)

        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(2, 2),
            is_last_chunk=True,
            prefill_aux_index=3,
            state_indices=None,
            state_index_positions=None,
        )

        with self.assertRaisesRegex(RuntimeError, "stop test worker"):
            self._run_one_transfer_chunk(manager, chunk)

        self.assertEqual(manager.send_aux.call_count, 2)
        manager.update_status.assert_called_once_with(room, KVPoll.Success)
        self.assertEqual(manager.sync_status_to_decode_endpoint.call_count, 2)
        self.assertNotIn(room, manager.transfer_infos)
        self.assertNotIn(room, manager.req_to_decode_prefix_len)

    def test_empty_last_chunk_propagates_state_transfer_failure(self):
        room = 7
        manager = self._make_transfer_worker_manager(room)
        manager.maybe_send_extra = MagicMock(return_value=-1)
        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([], dtype=np.int32),
            index_slice=slice(2, 2),
            is_last_chunk=True,
            prefill_aux_index=3,
            state_indices=[np.array([5], dtype=np.int32)],
            state_index_positions=[np.array([0], dtype=np.int64)],
        )

        with self.assertRaisesRegex(RuntimeError, "stop test worker"):
            self._run_one_transfer_chunk(manager, chunk)

        self.assertEqual(manager.maybe_send_extra.call_count, 2)
        self.assertEqual(manager.send_aux.call_count, 2)
        manager.update_status.assert_called_once_with(room, KVPoll.Failed)
        self.assertEqual(manager.sync_status_to_decode_endpoint.call_count, 2)

    def test_transfer_converts_prefill_zigzag_run_to_decode_round_robin(self):
        room = 7
        manager = self._make_transfer_worker_manager(room)
        infos = list(manager.transfer_infos[room].values())
        infos[0].dst_kv_indices = np.array(
            [11, 12, 0, 0, 15, 16, 0, 0], dtype=np.int32
        )
        infos[1].dst_kv_indices = np.array(
            [0, 0, 21, 22, 0, 0, 25, 26], dtype=np.int32
        )
        for info in infos:
            manager.decode_kv_args_table[info.mooncake_session_id] = SimpleNamespace(
                dst_aux_ptrs=[0],
                dst_attn_tp_size=manager.attn_tp_size,
                enable_hisparse=False,
                dst_kv_ptrs=[100, 200],
            )
        manager.send_kvcache = MagicMock(return_value=0)
        manager.maybe_send_extra = MagicMock(return_value=0)

        # Prefill CP2 zigzag with rotation=1 owns page ordinals [2, 6) on
        # this rank. Decode CP2 round-robin splits those positions across both
        # destination ranks: [2, 4) -> D1 and [4, 6) -> D0.
        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([101, 102, 103, 104], dtype=np.int32),
            index_slice=slice(2, 6),
            is_last_chunk=True,
            prefill_aux_index=3,
            state_indices=None,
            state_index_positions=None,
        )

        with self.assertRaisesRegex(RuntimeError, "stop test worker"):
            self._run_one_transfer_chunk(manager, chunk)

        calls_by_session = {
            call.args[0]: (call.args[1], call.args[3])
            for call in manager.send_kvcache.call_args_list
        }
        src, dst = calls_by_session[infos[0].mooncake_session_id]
        np.testing.assert_array_equal(src, np.array([103, 104]))
        np.testing.assert_array_equal(dst, np.array([15, 16]))
        src, dst = calls_by_session[infos[1].mooncake_session_id]
        np.testing.assert_array_equal(src, np.array([101, 102]))
        np.testing.assert_array_equal(dst, np.array([21, 22]))

    def test_tp_replica_to_decode_cp_owner_uses_direct_page_copy(self):
        room = 7
        manager = self._make_transfer_worker_manager(room)
        manager.is_cp_sharded_kv = False
        manager.attn_tp_size = 4
        manager.attn_tp_rank = 0
        manager.kv_args = SimpleNamespace(
            # Global engine rank is not an attention-TP coordinate under CP.
            engine_rank=3,
            total_kv_head_num=2,
            kv_item_lens=[64],
        )

        session_id = "session:100"
        req = manager.transfer_infos[room][session_id]
        req.dst_kv_indices = np.array([11, 0, 12, 0], dtype=np.int32)
        req.required_dst_info_num = 1
        req.dst_is_cp_sharded_kv = True
        manager.transfer_infos[room] = {session_id: req}
        manager.decode_kv_args_table[session_id] = SimpleNamespace(
            dst_aux_ptrs=[0],
            dst_attn_tp_size=2,
            dst_tp_rank=0,
            dst_kv_item_len=64,
            enable_hisparse=False,
            dst_kv_ptrs=[100, 200],
        )
        manager.send_kvcache = MagicMock(return_value=0)
        manager.send_kvcache_slice = MagicMock(return_value=0)
        manager.maybe_send_extra = MagicMock(return_value=0)

        chunk = TransferKVChunk(
            room=room,
            prefill_kv_indices=np.array([101, 102, 103, 104], dtype=np.int32),
            index_slice=slice(0, 4),
            is_last_chunk=True,
            prefill_aux_index=3,
            state_indices=None,
            state_index_positions=None,
        )

        with self.assertRaisesRegex(RuntimeError, "stop test worker"):
            self._run_one_transfer_chunk(manager, chunk)

        manager.send_kvcache_slice.assert_not_called()
        manager.send_kvcache.assert_called_once()
        call = manager.send_kvcache.call_args
        np.testing.assert_array_equal(call.args[1], np.array([101, 103]))
        np.testing.assert_array_equal(call.args[3], np.array([11, 12]))

    def test_cp_state_transfer_skips_decode_non_owner_slots(self):
        manager = object.__new__(MooncakeKVManager)
        manager.attn_tp_size = 2
        manager.attn_tp_rank = 0
        manager.is_mla_backend = False
        manager.is_cp_sharded_kv = True
        manager.kv_args = SimpleNamespace(
            state_types=[StateType.WELM_MTP_MIRROR],
            state_data_ptrs=[[100, 200]],
            state_item_lens=[[16, 16]],
            state_dim_per_tensor=[[]],
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        target = SimpleNamespace(
            dst_attn_tp_size=2,
            dst_state_data_ptrs=[[300, 400]],
            dst_state_item_lens=[[16, 16]],
            dst_state_dim_per_tensor=[[]],
        )
        source = [np.array([5, 6], dtype=np.int32)]
        source_positions = [np.array([2, 3], dtype=np.int64)]

        non_owner_req = SimpleNamespace(
            mooncake_session_id="session:100",
            dst_state_indices=[[11, 12, 0, 0]],
        )
        self.assertEqual(
            manager.maybe_send_extra(
                non_owner_req,
                source,
                MagicMock(),
                target,
                source_positions,
            ),
            0,
        )
        manager._send_kvcache_generic.assert_not_called()

        owner_req = SimpleNamespace(
            mooncake_session_id="session:101",
            dst_state_indices=[[0, 0, 21, 22]],
        )
        self.assertEqual(
            manager.maybe_send_extra(
                owner_req,
                source,
                MagicMock(),
                target,
                source_positions,
            ),
            0,
        )
        call_kwargs = manager._send_kvcache_generic.call_args.kwargs
        np.testing.assert_array_equal(
            call_kwargs["prefill_data_indices"], np.array([5, 6])
        )
        np.testing.assert_array_equal(
            call_kwargs["dst_data_indices"], np.array([21, 22])
        )

    def test_mooncake_sender_pulls_each_owner_run_and_state_positions(self):
        base = SWATokenToKVPoolAllocator(
            size=64,
            size_swa=64,
            page_size=4,
            dtype=torch.float32,
            device="cpu",
            kvcache=_DummySWAKVPool(),
            need_sort=False,
        )
        allocator = CPShardedKVPoolAllocator(
            base,
            cp_rank=0,
            cp_size=2,
            cp_kv_chunk_size=8,
            logical_size=128,
        )
        allocation = allocator.alloc_for_positions_with_logical(
            torch.arange(0, 32, dtype=torch.int64)
        )
        logical_pages = (allocation.logical_slots[::4] // 4).numpy()

        sender, manager = self._make_sender(
            allocator,
            len(logical_pages),
            state_types=[StateType.SWA, StateType.WELM_MTP_MIRROR],
        )

        sender.send(
            logical_pages,
            [logical_pages, allocation.logical_slots.numpy()],
        )

        calls = manager.add_transfer_request.call_args_list
        self.assertEqual(len(calls), 2)
        np.testing.assert_array_equal(calls[0].args[1], np.array([1, 2]))
        self.assertEqual(calls[0].args[2], slice(0, 2))
        self.assertFalse(calls[0].args[3])
        np.testing.assert_array_equal(calls[1].args[1], np.array([3, 4]))
        self.assertEqual(calls[1].args[2], slice(6, 8))
        self.assertTrue(calls[1].args[3])
        np.testing.assert_array_equal(
            calls[1].kwargs["state_index_positions"][0],
            np.array([0, 1, 6, 7]),
        )
        np.testing.assert_array_equal(
            calls[1].kwargs["state_index_positions"][1],
            np.array(list(range(8)) + list(range(24, 32))),
        )

    def test_tp2_to_tp4_replicates_equal_swa_and_mirror_state_items(self):
        manager = object.__new__(MooncakeKVManager)
        manager.attn_tp_size = 2
        manager.attn_tp_rank = 0
        manager.is_mla_backend = False
        manager.is_cp_sharded_kv = False
        manager.kv_args = SimpleNamespace(
            engine_rank=0,
            total_kv_head_num=2,
            state_types=[StateType.SWA, StateType.WELM_MTP_MIRROR],
            state_data_ptrs=[[100, 200], [300, 400]],
            state_item_lens=[[64, 64], [16, 16]],
            state_dim_per_tensor=[[], []],
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        req = SimpleNamespace(
            mooncake_session_id="session:1",
            dst_state_indices=[[11, 12], [21, 22]],
        )
        target = SimpleNamespace(
            dst_tp_rank=1,
            dst_attn_tp_size=4,
            dst_state_data_ptrs=[[500, 600], [700, 800]],
            dst_state_item_lens=[[64, 64], [16, 16]],
            dst_state_dim_per_tensor=[[], []],
        )

        ret = manager.maybe_send_extra(
            req,
            [np.array([1, 2]), np.array([3, 4])],
            MagicMock(),
            target,
        )

        self.assertEqual(ret, 0)
        self.assertEqual(manager._send_kvcache_generic.call_count, 2)
        first_call, second_call = manager._send_kvcache_generic.call_args_list
        self.assertEqual(first_call.kwargs["item_lens"], [64, 64])
        np.testing.assert_array_equal(
            first_call.kwargs["prefill_data_indices"], np.array([1, 2])
        )
        np.testing.assert_array_equal(
            first_call.kwargs["dst_data_indices"], np.array([11, 12])
        )
        self.assertEqual(second_call.kwargs["item_lens"], [16, 16])

    def test_tp4_to_tp2_maps_equal_state_items_by_kv_head_replica(self):
        manager = object.__new__(MooncakeKVManager)
        manager.attn_tp_size = 4
        manager.attn_tp_rank = 0
        manager.is_mla_backend = False
        manager.is_cp_sharded_kv = False
        manager.kv_args = SimpleNamespace(
            engine_rank=3,
            total_kv_head_num=2,
            state_types=[StateType.SWA, StateType.WELM_MTP_MIRROR],
            state_data_ptrs=[[100, 200], [300, 400]],
            state_item_lens=[[64, 64], [16, 16]],
            state_dim_per_tensor=[[], []],
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        req = SimpleNamespace(
            mooncake_session_id="session:1",
            dst_state_indices=[[11, 12], [21, 22]],
            dst_is_cp_sharded_kv=False,
        )
        target = SimpleNamespace(
            dst_tp_rank=0,
            dst_attn_tp_size=2,
            dst_state_data_ptrs=[[500, 600], [700, 800]],
            dst_state_item_lens=[[64, 64], [16, 16]],
            dst_state_dim_per_tensor=[[], []],
        )

        ret = manager.maybe_send_extra(
            req,
            [np.array([1, 2]), np.array([3, 4])],
            MagicMock(),
            target,
        )

        self.assertEqual(ret, 0)
        self.assertEqual(manager._send_kvcache_generic.call_count, 2)

    def test_tp4_to_tp2_rejects_misrouted_state_replica(self):
        manager = object.__new__(MooncakeKVManager)
        manager.attn_tp_size = 4
        manager.attn_tp_rank = 2
        manager.is_mla_backend = False
        manager.is_cp_sharded_kv = False
        manager.kv_args = SimpleNamespace(
            engine_rank=2,
            total_kv_head_num=2,
            state_types=[StateType.SWA],
            state_data_ptrs=[[100, 200]],
            state_item_lens=[[64, 64]],
            state_dim_per_tensor=[[]],
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        req = SimpleNamespace(
            mooncake_session_id="session:1",
            dst_state_indices=[[11, 12]],
            dst_is_cp_sharded_kv=False,
        )
        target = SimpleNamespace(
            dst_tp_rank=0,
            dst_attn_tp_size=2,
            dst_state_data_ptrs=[[500, 600]],
            dst_state_item_lens=[[64, 64]],
            dst_state_dim_per_tensor=[[]],
        )

        with self.assertRaisesRegex(RuntimeError, "target is routed"):
            manager.maybe_send_extra(
                req,
                [np.array([1, 2])],
                MagicMock(),
                target,
            )

        manager._send_kvcache_generic.assert_not_called()

    def test_tp2_to_tp4_rejects_non_replicated_state_layout(self):
        manager = object.__new__(MooncakeKVManager)
        manager.attn_tp_size = 2
        manager.attn_tp_rank = 0
        manager.is_mla_backend = False
        manager.is_cp_sharded_kv = False
        manager.kv_args = SimpleNamespace(
            engine_rank=0,
            total_kv_head_num=2,
            state_types=[StateType.SWA],
            state_data_ptrs=[[100, 200]],
            state_item_lens=[[64, 64]],
            state_dim_per_tensor=[[]],
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        req = SimpleNamespace(
            mooncake_session_id="session:1",
            dst_state_indices=[[11, 12]],
        )
        target = SimpleNamespace(
            dst_tp_rank=1,
            dst_attn_tp_size=4,
            dst_state_data_ptrs=[[500, 600]],
            dst_state_item_lens=[[32, 32]],
            dst_state_dim_per_tensor=[[]],
        )

        with self.assertRaisesRegex(RuntimeError, "state item layout"):
            manager.maybe_send_extra(
                req,
                [np.array([1, 2])],
                MagicMock(),
                target,
            )

        manager._send_kvcache_generic.assert_not_called()

    def test_same_tp_rejects_non_matching_state_layout(self):
        manager = object.__new__(MooncakeKVManager)
        manager.attn_tp_size = 2
        manager.attn_tp_rank = 0
        manager.is_mla_backend = False
        manager.is_cp_sharded_kv = False
        manager.kv_args = SimpleNamespace(
            state_types=[StateType.SWA],
            state_data_ptrs=[[100, 200]],
            state_item_lens=[[64, 64]],
            state_dim_per_tensor=[[]],
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        req = SimpleNamespace(
            mooncake_session_id="session:1",
            dst_state_indices=[[11, 12]],
        )
        target = SimpleNamespace(
            dst_tp_rank=0,
            dst_attn_tp_size=2,
            dst_state_data_ptrs=[[500]],
            dst_state_item_lens=[[32]],
            dst_state_dim_per_tensor=[[]],
        )

        with self.assertRaisesRegex(RuntimeError, "state item layout"):
            manager.maybe_send_extra(
                req,
                [np.array([1, 2])],
                MagicMock(),
                target,
            )

        manager._send_kvcache_generic.assert_not_called()

    def test_tp2_to_tp4_rejects_misrouted_state_target(self):
        manager = object.__new__(MooncakeKVManager)
        manager.attn_tp_size = 2
        manager.attn_tp_rank = 1
        manager.is_mla_backend = False
        manager.is_cp_sharded_kv = False
        manager.kv_args = SimpleNamespace(
            engine_rank=1,
            total_kv_head_num=2,
            state_types=[StateType.SWA],
            state_data_ptrs=[[100, 200]],
            state_item_lens=[[64, 64]],
            state_dim_per_tensor=[[]],
        )
        manager._send_kvcache_generic = MagicMock(return_value=0)
        req = SimpleNamespace(
            mooncake_session_id="session:1",
            dst_state_indices=[[11, 12]],
        )
        target = SimpleNamespace(
            dst_tp_rank=1,
            dst_attn_tp_size=4,
            dst_state_data_ptrs=[[500, 600]],
            dst_state_item_lens=[[64, 64]],
            dst_state_dim_per_tensor=[[]],
        )

        with self.assertRaisesRegex(RuntimeError, "target is routed"):
            manager.maybe_send_extra(
                req,
                [np.array([1, 2])],
                MagicMock(),
                target,
            )

        manager._send_kvcache_generic.assert_not_called()

    def test_state_replication_rejects_unsupported_topologies(self):
        for state_type, dst_attn_tp_size, error in (
            (StateType.NSA, 4, "different TP sizes"),
            (StateType.SWA, 1, "target is routed"),
        ):
            with self.subTest(
                state_type=state_type,
                dst_attn_tp_size=dst_attn_tp_size,
            ):
                manager = object.__new__(MooncakeKVManager)
                manager.attn_tp_size = 2
                manager.attn_tp_rank = 0
                manager.is_mla_backend = False
                manager.is_cp_sharded_kv = False
                manager.kv_args = SimpleNamespace(
                    engine_rank=0,
                    total_kv_head_num=2,
                    state_types=[state_type],
                    state_data_ptrs=[[100, 200]],
                    state_item_lens=[[64, 64]],
                    state_dim_per_tensor=[[]],
                )
                manager._send_kvcache_generic = MagicMock(return_value=0)
                req = SimpleNamespace(
                    mooncake_session_id="session:1",
                    dst_state_indices=[[11, 12]],
                )
                target = SimpleNamespace(
                    dst_tp_rank=0,
                    dst_attn_tp_size=dst_attn_tp_size,
                    dst_state_data_ptrs=[[500, 600]],
                    dst_state_item_lens=[[64, 64]],
                    dst_state_dim_per_tensor=[[]],
                )

                with self.assertRaisesRegex(RuntimeError, error):
                    manager.maybe_send_extra(
                        req,
                        [np.array([1, 2])],
                        MagicMock(),
                        target,
                    )

                manager._send_kvcache_generic.assert_not_called()


if __name__ == "__main__":
    unittest.main()
