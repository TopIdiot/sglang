from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.disaggregation import decode as decode_module
from sglang.srt.disaggregation import (
    decode_schedule_batch_mixin as decode_batch_mixin_module,
)
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.welm_deferred_protocol import (
    WelmDeferredCompletion,
)
from sglang.srt.managers import schedule_batch as schedule_batch_module
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.models.welm_deferred_mirror import (
    DeferredDecodeInputKind,
    WelmPDExecutionMode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _deferred_server_args(**kwargs):
    values = {
        "welm_kv_mirror_pd_mode": WelmPDExecutionMode.DEFERRED_LAST_PROMPT.value,
        "disaggregation_transfer_backend": "fake",
        "disaggregation_decode_enable_radix_cache": False,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def _deferred_state(prompt_token_ids):
    return schedule_batch_module.WelmDeferredDecodeState.from_prompt_tokens(
        prompt_token_ids
    )


def _time_stats():
    return SimpleNamespace(
        decode_transfer_queue_entry_time=None,
        wait_queue_entry_time=None,
        set_decode_transfer_queue_entry_time=MagicMock(),
        set_wait_queue_entry_time=MagicMock(),
    )


def _deferred_req(prompt_token_ids, *, bootstrap_host=None):
    state = _deferred_state(prompt_token_ids)
    return SimpleNamespace(
        rid="req-0",
        origin_input_ids=list(prompt_token_ids),
        output_ids=[],
        fill_ids=list(prompt_token_ids[:-1]),
        welm_deferred_decode_state=state,
        kv_allocated_len=state.committed_kv_len,
        kv_committed_len=state.committed_kv_len,
        bootstrap_host=bootstrap_host,
        bootstrap_port=8998,
        bootstrap_room=7,
        return_logprob=False,
        cached_tokens=0,
        cached_tokens_device=0,
        cached_tokens_host=0,
        cached_tokens_storage=0,
        finished_reason=None,
        time_stats=_time_stats(),
    )


def _completion_for(req):
    state = req.welm_deferred_decode_state
    return WelmDeferredCompletion(
        committed_kv_len=state.committed_kv_len,
        seed_position=state.seed_position,
        seed_token_id=state.seed_token_id,
        input_kind=state.input_kind,
    )


def test_deferred_decode_state_enforces_linear_lifecycle():
    phase = schedule_batch_module.WelmDeferredDecodePhase
    state = _deferred_state([11, 12, 13])

    assert state.phase is phase.PREFILL_PENDING
    assert state.committed_kv_len == 2
    assert state.seed_position == 2
    assert state.seed_token_id == 13
    assert state.input_kind is DeferredDecodeInputKind.TOKEN_ID
    assert state.committed_token_ids([11, 12, 13]) == [11, 12]

    state.transition_to(phase.READY)
    state.transition_to(phase.INFLIGHT)
    state.transition_to(phase.CONSUMED)

    with pytest.raises(RuntimeError, match="invalid WeLM deferred decode transition"):
        state.transition_to(phase.READY)


def test_decode_prealloc_admission_creates_prefill_pending_state():
    req = SimpleNamespace(
        rid="req-0",
        origin_input_ids=[11, 12, 13],
        output_ids=[],
        welm_deferred_decode_state=None,
        bootstrap_host=None,
    )
    receiver = MagicMock()
    decode_req = SimpleNamespace(req=req, kv_receiver=receiver)
    queue = object.__new__(decode_module.DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(server_args=_deferred_server_args())
    queue._check_if_req_exceed_kv_capacity = MagicMock(return_value=False)
    queue._create_receiver_and_enqueue = MagicMock(return_value=decode_req)

    queue.add(req)

    assert (
        req.welm_deferred_decode_state.phase
        is schedule_batch_module.WelmDeferredDecodePhase.PREFILL_PENDING
    )
    assert req.welm_deferred_decode_state.committed_kv_len == 2
    receiver.init.assert_called_once_with(0)


def test_decode_prealloc_admission_rejects_malformed_lifecycle_state():
    req = SimpleNamespace(
        rid="req-0",
        origin_input_ids=[11, 12, 13],
        output_ids=[],
        welm_deferred_decode_state=object(),
        bootstrap_host=None,
    )
    queue = object.__new__(decode_module.DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(server_args=_deferred_server_args())
    queue._check_if_req_exceed_kv_capacity = MagicMock(return_value=False)
    queue._create_receiver_and_enqueue = MagicMock()

    with pytest.raises(RuntimeError, match="invalid lifecycle state"):
        queue.add(req)

    queue._create_receiver_and_enqueue.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_embeds", [[0.0]], "input embeddings"),
        ("positional_embed_overrides", object(), "positional embedding overrides"),
        ("multimodal_inputs", object(), "multimodal inputs"),
    ],
)
def test_decode_prealloc_rejects_non_token_seed_before_receiver_creation(
    field, value, message, monkeypatch
):
    req = SimpleNamespace(
        rid="req-0",
        origin_input_ids=[11, 12, 13],
        output_ids=[],
        welm_deferred_decode_state=None,
        bootstrap_host=None,
        return_logprob=False,
    )
    setattr(req, field, value)
    queue = object.__new__(decode_module.DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(
        server_args=_deferred_server_args(),
        stream_output=MagicMock(),
    )
    queue._check_if_req_exceed_kv_capacity = MagicMock(return_value=False)
    queue._create_receiver_and_enqueue = MagicMock()
    prepare_abort = MagicMock()
    monkeypatch.setattr(decode_module, "prepare_abort", prepare_abort)

    queue.add(req)

    prepare_abort.assert_called_once()
    assert message in prepare_abort.call_args.args[1]
    queue.scheduler.stream_output.assert_called_once_with([req], False)
    queue._check_if_req_exceed_kv_capacity.assert_not_called()
    queue._create_receiver_and_enqueue.assert_not_called()
    assert req.welm_deferred_decode_state is None


@pytest.mark.parametrize(
    "prompt_token_ids,expected_full_len",
    [([41], 0), (list(range(10)), 9)],
)
def test_prealloc_kv_lens_use_committed_prompt_without_seed(
    prompt_token_ids, expected_full_len
):
    queue = object.__new__(decode_module.DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(server_args=_deferred_server_args())
    queue._uses_swa_tail_prealloc = lambda: False
    req = _deferred_req(prompt_token_ids)

    assert queue._prealloc_kv_lens(req) == (
        expected_full_len,
        expected_full_len,
    )


def test_swa_prealloc_uses_committed_prompt_tail_without_seed():
    queue = object.__new__(decode_module.DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(
        server_args=_deferred_server_args(),
        sliding_window_size=5,
    )
    queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=4)
    queue._uses_swa_tail_prealloc = lambda: True
    req = _deferred_req(list(range(10)))

    assert queue._prealloc_kv_lens(req) == (9, 5)


def _real_paged_prealloc_queue(*, page_size=4):
    allocator = PagedTokenToKVPoolAllocator(
        size=64,
        page_size=page_size,
        dtype=torch.float32,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )

    def alloc_extend_cpu(
        *,
        prefix_lens,
        prefix_lens_cpu,
        seq_lens,
        seq_lens_cpu,
        last_loc,
        extend_num_tokens,
        num_new_pages=None,
    ):
        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=allocator.device
        )
        alloc_extend_naive(
            prefix_lens_cpu,
            seq_lens_cpu,
            last_loc,
            allocator.free_pages,
            out_indices,
            allocator.page_size,
            allocator.device,
        )
        if num_new_pages is None:
            num_new_pages = int(
                (
                    (seq_lens_cpu + allocator.page_size - 1)
                    // allocator.page_size
                    - (prefix_lens_cpu + allocator.page_size - 1)
                    // allocator.page_size
                )
                .sum()
                .item()
            )
        if num_new_pages > len(allocator.free_pages):
            return None
        allocator.free_pages = allocator.free_pages[num_new_pages:]
        return out_indices

    allocator.alloc_extend = alloc_extend_cpu
    queue = object.__new__(decode_module.DecodePreallocQueue)
    queue.token_to_kv_pool_allocator = allocator
    queue.req_to_token_pool = ReqToTokenPool(
        size=2,
        max_context_len=64,
        device="cpu",
        enable_memory_saver=False,
    )
    queue.scheduler = SimpleNamespace(
        server_args=_deferred_server_args(),
        enable_hisparse=False,
    )
    queue.tree_cache = MagicMock()
    queue._uses_swa_tail_prealloc = lambda: False
    return queue, allocator


def _real_prealloc_req(prompt_token_ids):
    req = _deferred_req(prompt_token_ids)
    req.req_pool_idx = None
    req.is_chunked = 0
    req.rid = "real-prealloc"
    req.set_extend_input_len = MagicMock()
    return req


def test_one_token_prompt_preallocates_zero_decode_pages():
    queue, allocator = _real_paged_prealloc_queue()
    req = _real_prealloc_req([41])
    available_before = allocator.available_size()

    physical_dst = queue._pre_alloc(req)

    assert physical_dst.numel() == 0
    assert req.kv_allocated_len == 0
    assert req.kv_committed_len == 0
    assert req.fill_ids == []
    assert allocator.available_size() == available_before


def test_full_radix_hit_preallocates_no_new_decode_pages():
    queue, allocator = _real_paged_prealloc_queue()
    req = _real_prealloc_req(list(range(9)))
    prefix_indices = allocator.alloc(8)
    assert prefix_indices is not None
    available_before = allocator.available_size()

    physical_dst = queue._pre_alloc(
        req,
        prefix_indices=prefix_indices,
        prefix_len=8,
    )

    assert physical_dst.numel() == 0
    assert allocator.available_size() == available_before
    torch.testing.assert_close(
        queue.req_to_token_pool.req_to_token[req.req_pool_idx, :8].to(torch.int64),
        prefix_indices.to(torch.int64),
    )


@pytest.mark.parametrize("committed_len", [0, 2, 16])
def test_retracted_seed_releases_each_paged_allocation_once(committed_len):
    page_size = 16
    prompt = list(range(committed_len + 1))
    queue, allocator = _real_paged_prealloc_queue(page_size=page_size)
    req = _real_prealloc_req(prompt)
    initial_available = allocator.available_size()

    queue._pre_alloc(req)
    if committed_len % page_size == 0:
        seed_page = allocator.alloc(page_size)
        assert seed_page is not None
        seed_loc = seed_page[0]
    else:
        seed_loc = (
            queue.req_to_token_pool.req_to_token[
                req.req_pool_idx, committed_len - 1
            ].to(torch.int64)
            + 1
        )
    queue.req_to_token_pool.req_to_token[
        req.req_pool_idx, committed_len
    ] = seed_loc

    req.kv_allocated_len = committed_len + 1
    req.kv_committed_len = committed_len
    req.kv_committed_freed = False
    req.kv_overallocated_freed = False
    req.cache_protected_len = 0
    req.extra_key = None
    req.priority = 0
    req.mamba_pool_idx = None

    def pop_committed():
        assert not req.kv_committed_freed
        req.kv_committed_freed = True
        return req.kv_committed_len

    def pop_overallocated():
        assert not req.kv_overallocated_freed
        req.kv_overallocated_freed = True
        return req.kv_committed_len, req.kv_allocated_len

    req.pop_committed_kv_cache = pop_committed
    req.pop_overallocated_kv_cache = pop_overallocated

    tree_cache = RadixCache.create_simulated(
        mock_allocator=allocator,
        page_size=page_size,
    )
    tree_cache.req_to_token_pool = queue.req_to_token_pool
    req.last_node = tree_cache.root_node
    server_args = SimpleNamespace(
        page_size=page_size,
        speculative_algorithm=None,
        strip_thinking_cache=False,
    )

    with patch(
        "sglang.srt.mem_cache.common.get_global_server_args",
        return_value=server_args,
    ):
        release_kv_cache(
            req,
            tree_cache,
            is_insert=False,
            allow_uncommitted_tail=True,
        )

    assert req.req_pool_idx is None
    assert allocator.available_size() == initial_available
    assert len(torch.unique(allocator.free_pages)) == len(allocator.free_pages)


def test_transfer_commit_caches_only_page_aligned_committed_prefix():
    prompt = list(range(7))
    queue, allocator = _real_paged_prealloc_queue()
    req = _real_prealloc_req(prompt)
    tree_cache = RadixCache.create_simulated(
        mock_allocator=allocator,
        page_size=allocator.page_size,
    )
    tree_cache.req_to_token_pool = queue.req_to_token_pool
    queue.tree_cache = tree_cache
    req.extra_key = None
    req.priority = 0
    req.cache_protected_len = 0
    req.last_node = tree_cache.root_node

    queue._pre_alloc(req)
    decode_req = SimpleNamespace(
        req=req,
        kv_receiver=MagicMock(),
        metadata_buffer_index=0,
    )
    transfer_queue = object.__new__(decode_module.DecodeTransferQueue)
    transfer_queue.metadata_buffers = SimpleNamespace(
        bootstrap_room=torch.zeros((1, 1), dtype=torch.uint64),
        get_welm_deferred_completion=MagicMock(return_value=_completion_for(req)),
        get_cached_token_stats=MagicMock(
            return_value=torch.tensor([6, 4, 1, 1], dtype=torch.int32)
        ),
    )
    transfer_queue.scheduler = SimpleNamespace(server_args=_deferred_server_args())
    transfer_queue.tree_cache = tree_cache
    transfer_queue.tp_rank = 0

    with patch(
        "sglang.srt.disaggregation.decode.hicache_timing_enabled",
        return_value=False,
    ):
        assert transfer_queue._commit_transfer_to_req(decode_req)

    match = tree_cache.match_prefix(
        MatchPrefixParams(key=RadixKey(prompt[:-1]))
    )
    assert len(match.device_indices) == 4
    assert len(req.prefix_indices) == 6
    assert req.fill_ids == prompt[:-1]
    assert req.output_ids == []


@pytest.mark.parametrize("matched_len", [8, 6])
def test_prefix_match_queries_only_committed_prompt_tokens(matched_len):
    prompt_token_ids = list(range(9))
    req = _deferred_req(prompt_token_ids)
    result = SimpleNamespace(
        device_indices=torch.arange(matched_len, dtype=torch.int64),
        last_device_node=object(),
    )
    queue = object.__new__(decode_module.DecodePreallocQueue)
    queue.scheduler = SimpleNamespace(server_args=_deferred_server_args())
    queue.tree_cache = SimpleNamespace(
        supports_mamba=lambda: False,
        inc_lock_ref=MagicMock(),
    )

    with patch(
        "sglang.srt.disaggregation.decode.match_prefix_for_req",
        return_value=result,
    ) as match_prefix:
        prefix_indices, prefix_len = queue._match_prefix_and_lock(req)

    assert prefix_indices.tolist() == list(range(matched_len))
    assert prefix_len == matched_len
    assert match_prefix.call_args.args[2] == prompt_token_ids[:-1]
    queue.tree_cache.inc_lock_ref.assert_called_once_with(result.last_device_node)


@pytest.mark.parametrize(
    "prompt_len,matched_len,expected_prefix_len",
    [(9, 8, 8), (10, 6, 4)],
)
def test_pop_preallocated_uses_committed_length_for_full_and_partial_page_hits(
    prompt_len, matched_len, expected_prefix_len
):
    req = _deferred_req(list(range(prompt_len)))
    req.priority = 0
    req.cache_protected_len = 0
    req.sampling_params = SimpleNamespace(max_new_tokens=16)
    decode_req = SimpleNamespace(
        req=req,
        waiting_for_input=True,
        kv_receiver=MagicMock(),
        metadata_buffer_index=-1,
    )

    queue = object.__new__(decode_module.DecodePreallocQueue)
    queue.queue = [decode_req]
    queue.pending_reqs = []
    queue.retracted_queue = []
    queue.num_reserved_decode_tokens = 0
    queue._resolve_pending_reqs = MagicMock()
    queue._update_handshake_waiters = MagicMock()
    queue._uses_swa_tail_prealloc = lambda: False
    queue._allocatable_token_budgets = MagicMock(return_value=1000)
    queue._ensure_cp_sharded_prompt_capacity = MagicMock(return_value=True)
    queue._match_prefix_and_lock = MagicMock(
        return_value=(torch.arange(matched_len, dtype=torch.int64), matched_len)
    )
    queue._pre_alloc = MagicMock(return_value=torch.empty(0, dtype=torch.int64))
    queue.transfer_queue = SimpleNamespace(queue=[], enable_staging=False)
    queue.tree_cache = MagicMock()
    queue.req_to_token_pool = MagicMock()
    queue.req_to_token_pool.available_size.return_value = 1
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
    queue.req_to_metadata_buffer_idx_allocator.alloc.return_value = 3
    queue.token_to_kv_pool = MagicMock()
    queue.token_to_kv_pool_allocator = MagicMock(page_size=4)
    queue.scheduler = SimpleNamespace(
        running_batch=SimpleNamespace(reqs=[]),
        enable_priority_scheduling=False,
        schedule_low_priority_values_first=False,
        enable_hisparse=False,
        waiting_queue=[],
        last_batch=None,
        server_args=_deferred_server_args(
            disaggregation_decode_enable_radix_cache=True
        ),
        stream_output=MagicMock(),
    )

    with patch(
        "sglang.srt.disaggregation.decode.hicache_timing_enabled",
        return_value=False,
    ):
        preallocated, failed = queue.pop_preallocated()

    assert preallocated == [decode_req]
    assert failed == []
    queue._ensure_cp_sharded_prompt_capacity.assert_called_once_with(
        prefix_len=expected_prefix_len,
        fill_len=prompt_len - 1,
    )
    pre_alloc_args = queue._pre_alloc.call_args.args
    assert pre_alloc_args[0] is req
    assert pre_alloc_args[1].tolist() == list(range(expected_prefix_len))
    assert pre_alloc_args[2] == expected_prefix_len
    decode_req.kv_receiver.send_metadata.assert_called_once()
    assert decode_req.kv_receiver.send_metadata.call_args.kwargs[
        "decode_prefix_len"
    ] == expected_prefix_len


def test_valid_transfer_commits_ready_seed_without_legacy_output_payload():
    req = _deferred_req([11, 12, 13], bootstrap_host="prefill")
    receiver = MagicMock()
    decode_req = SimpleNamespace(
        req=req,
        kv_receiver=receiver,
        metadata_buffer_index=3,
    )
    metadata_buffers = SimpleNamespace(
        bootstrap_room=torch.tensor([[0], [0], [0], [7]], dtype=torch.uint64),
        get_welm_deferred_completion=MagicMock(return_value=_completion_for(req)),
        get_cached_token_stats=MagicMock(
            return_value=torch.tensor([11, 7, 3, 1], dtype=torch.int32)
        ),
        get_buf=MagicMock(
            side_effect=AssertionError(
                "deferred transfer must not read legacy output metadata"
            )
        ),
    )
    queue = object.__new__(decode_module.DecodeTransferQueue)
    queue.metadata_buffers = metadata_buffers
    queue.scheduler = SimpleNamespace(server_args=_deferred_server_args())
    queue.tree_cache = object()
    queue.tp_rank = 0

    with (
        patch(
            "sglang.srt.disaggregation.decode.maybe_cache_unfinished_req"
        ) as cache_req,
        patch(
            "sglang.srt.disaggregation.decode.hicache_timing_enabled",
            return_value=False,
        ),
    ):
        should_remove = queue._commit_transfer_to_req(decode_req)

    assert should_remove
    assert req.output_ids == []
    assert req.fill_ids == [11, 12]
    assert req.cached_tokens == 11
    assert req.cached_tokens_device == 7
    assert req.cached_tokens_host == 3
    assert req.cached_tokens_storage == 1
    assert (
        req.welm_deferred_decode_state.phase
        is schedule_batch_module.WelmDeferredDecodePhase.READY
    )
    cache_req.assert_called_once_with(req, queue.tree_cache)
    metadata_buffers.get_buf.assert_not_called()
    receiver.clear.assert_called_once_with()
    assert decode_req.kv_receiver is None
    req.time_stats.set_wait_queue_entry_time.assert_called_once_with()


def test_transfer_waits_for_bootstrap_room_before_marking_ready():
    req = _deferred_req([11, 12, 13], bootstrap_host="prefill")
    decode_req = SimpleNamespace(
        req=req,
        kv_receiver=MagicMock(),
        metadata_buffer_index=0,
    )
    metadata_buffers = SimpleNamespace(
        bootstrap_room=torch.zeros((1, 1), dtype=torch.uint64),
        get_welm_deferred_completion=MagicMock(return_value=_completion_for(req)),
    )
    queue = object.__new__(decode_module.DecodeTransferQueue)
    queue.metadata_buffers = metadata_buffers
    queue.scheduler = SimpleNamespace(
        server_args=_deferred_server_args(disaggregation_transfer_backend="mooncake")
    )

    assert not queue._commit_transfer_to_req(decode_req)
    assert (
        req.welm_deferred_decode_state.phase
        is schedule_batch_module.WelmDeferredDecodePhase.PREFILL_PENDING
    )
    metadata_buffers.get_welm_deferred_completion.assert_not_called()
    decode_req.kv_receiver.clear.assert_not_called()


def test_fake_transfer_builds_completion_from_local_typed_state():
    req = _deferred_req(
        [11, 12, 13], bootstrap_host=decode_module.FAKE_BOOTSTRAP_HOST
    )
    decode_req = SimpleNamespace(
        req=req,
        kv_receiver=MagicMock(),
        metadata_buffer_index=0,
    )
    metadata_buffers = SimpleNamespace(
        bootstrap_room=torch.zeros((1, 1), dtype=torch.uint64),
        get_welm_deferred_completion=MagicMock(
            side_effect=AssertionError(
                "fake transfer has no wire completion metadata"
            )
        ),
        get_cached_token_stats=MagicMock(
            side_effect=AssertionError(
                "fake transfer has no wire cache statistics"
            )
        ),
    )
    queue = object.__new__(decode_module.DecodeTransferQueue)
    queue.metadata_buffers = metadata_buffers
    queue.scheduler = SimpleNamespace(
        server_args=_deferred_server_args(
            disaggregation_transfer_backend="mooncake"
        )
    )
    queue.tree_cache = object()
    queue.tp_rank = 0

    with (
        patch("sglang.srt.disaggregation.decode.maybe_cache_unfinished_req"),
        patch(
            "sglang.srt.disaggregation.decode.hicache_timing_enabled",
            return_value=False,
        ),
    ):
        assert queue._commit_transfer_to_req(decode_req)

    assert (
        req.welm_deferred_decode_state.phase
        is schedule_batch_module.WelmDeferredDecodePhase.READY
    )
    metadata_buffers.get_welm_deferred_completion.assert_not_called()
    metadata_buffers.get_cached_token_stats.assert_not_called()


@pytest.mark.parametrize("failure_kind", ["missing", "mismatch", "duplicate"])
def test_invalid_transfer_aborts_and_releases_decode_allocation_once(failure_kind):
    req = _deferred_req([11, 12, 13], bootstrap_host="prefill")
    if failure_kind == "duplicate":
        req.welm_deferred_decode_state.transition_to(
            schedule_batch_module.WelmDeferredDecodePhase.READY
        )

    completion = _completion_for(req)
    if failure_kind == "mismatch":
        completion = WelmDeferredCompletion(
            committed_kv_len=completion.committed_kv_len,
            seed_position=completion.seed_position,
            seed_token_id=completion.seed_token_id + 1,
            input_kind=completion.input_kind,
        )

    metadata_get = MagicMock(return_value=completion)
    if failure_kind == "missing":
        metadata_get.side_effect = RuntimeError(
            "WeLM deferred completion buffer is not enabled"
        )

    receiver = MagicMock()
    receiver.poll.return_value = KVPoll.Success
    decode_req = SimpleNamespace(
        req=req,
        kv_receiver=receiver,
        metadata_buffer_index=3,
    )
    queue = object.__new__(decode_module.DecodeTransferQueue)
    queue.queue = [decode_req]
    queue.enable_staging = False
    queue.gloo_group = object()
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.metadata_buffers = SimpleNamespace(
        bootstrap_room=torch.tensor(
            [[0], [0], [0], [7]], dtype=torch.uint64
        ),
        get_welm_deferred_completion=metadata_get,
    )
    queue.tree_cache = object()
    queue.tp_rank = 0
    queue.scheduler = SimpleNamespace(
        server_args=_deferred_server_args(
            disaggregation_transfer_backend="mooncake"
        ),
        token_to_kv_pool_allocator=object(),
        stream_output=MagicMock(),
        enable_hisparse=False,
        enable_metrics=False,
    )

    with (
        patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Success],
        ),
        patch("sglang.srt.disaggregation.decode.release_kv_cache") as release,
    ):
        transferred = queue.pop_transferred()

    assert transferred == []
    assert req.finished_reason is not None
    release.assert_called_once_with(req, queue.tree_cache, is_insert=False)
    receiver.clear.assert_called_once_with()
    queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
    assert queue.queue == []


def test_legacy_transfer_still_appends_prefill_output_token():
    req = SimpleNamespace(
        rid="legacy",
        origin_input_ids=[1, 2],
        output_ids=[],
        welm_deferred_decode_state=None,
        bootstrap_host=None,
        bootstrap_room=7,
        return_logprob=False,
        cached_tokens=0,
        cached_tokens_device=0,
        cached_tokens_host=0,
        cached_tokens_storage=0,
        time_stats=_time_stats(),
    )
    decode_req = SimpleNamespace(
        req=req,
        kv_receiver=MagicMock(),
        metadata_buffer_index=0,
    )
    metadata_buffers = SimpleNamespace(
        get_buf=MagicMock(
            return_value=(
                torch.tensor([42]),
                torch.tensor([1, 2, 3, 4]),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.int64),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.int64),
                torch.zeros(1),
                torch.zeros(1, dtype=torch.int64),
                torch.zeros(1),
                torch.tensor([7], dtype=torch.uint64),
            )
        ),
        get_welm_deferred_completion=MagicMock(),
    )
    queue = object.__new__(decode_module.DecodeTransferQueue)
    queue.metadata_buffers = metadata_buffers
    queue.scheduler = SimpleNamespace(
        server_args=SimpleNamespace(
            welm_kv_mirror_pd_mode=WelmPDExecutionMode.LEGACY.value,
            disaggregation_transfer_backend="fake",
        )
    )
    queue.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    queue.tp_rank = 0

    with patch(
        "sglang.srt.disaggregation.decode.hicache_timing_enabled",
        return_value=False,
    ):
        assert queue._commit_transfer_to_req(decode_req)

    assert req.output_ids == [42]
    metadata_buffers.get_welm_deferred_completion.assert_not_called()


def test_ready_deferred_seed_is_not_consumed_by_legacy_prebuilt_path():
    req = _deferred_req([11, 12, 13])
    req.welm_deferred_decode_state.transition_to(
        schedule_batch_module.WelmDeferredDecodePhase.READY
    )
    scheduler = SimpleNamespace(
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        waiting_queue=[req],
        enable_priority_scheduling=False,
        running_batch=SimpleNamespace(batch_size=lambda: 0),
        req_to_token_pool=SimpleNamespace(size=8),
        max_running_requests=8,
        server_args=_deferred_server_args(),
        _should_isolate_welm_mtp_prebuilt=lambda: False,
    )

    batch = decode_module.SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(
        scheduler
    )

    assert batch is None
    assert scheduler.waiting_queue == [req]


@pytest.mark.parametrize("method_name", ["prepare_for_prebuilt", "process_prebuilt"])
def test_deferred_seed_fails_fast_if_it_reaches_prebuilt(method_name):
    req = _deferred_req([11, 12, 13])
    batch = SimpleNamespace(reqs=[req])
    method = getattr(
        decode_batch_mixin_module.ScheduleBatchDisaggregationDecodeMixin,
        method_name,
    )

    with pytest.raises(RuntimeError, match="deferred decode seed.*PREBUILT"):
        if method_name == "prepare_for_prebuilt":
            method(batch)
        else:
            method(batch, _deferred_server_args(), MagicMock())
