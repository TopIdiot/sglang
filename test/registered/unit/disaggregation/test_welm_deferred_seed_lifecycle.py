from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.disaggregation import decode as decode_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    _assert_no_welm_deferred_decode_seed,
)
from sglang.srt.managers import schedule_batch as schedule_batch_module
from sglang.srt.managers import (
    scheduler_output_processor_mixin as output_processor_module,
)
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.models.welm_deferred_mirror import WelmPDExecutionMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _server_args():
    return SimpleNamespace(
        welm_kv_mirror_pd_mode=(
            WelmPDExecutionMode.DEFERRED_LAST_PROMPT.value
        ),
        disaggregation_decode_enable_radix_cache=False,
    )


def _state(prompt_token_ids, phase):
    state = schedule_batch_module.WelmDeferredDecodeState.from_prompt_tokens(
        list(prompt_token_ids)
    )
    if phase is schedule_batch_module.WelmDeferredDecodePhase.PREFILL_PENDING:
        return state
    state.transition_to(schedule_batch_module.WelmDeferredDecodePhase.READY)
    if phase is schedule_batch_module.WelmDeferredDecodePhase.READY:
        return state
    state.transition_to(schedule_batch_module.WelmDeferredDecodePhase.INFLIGHT)
    if phase is schedule_batch_module.WelmDeferredDecodePhase.INFLIGHT:
        return state
    state.transition_to(schedule_batch_module.WelmDeferredDecodePhase.CONSUMED)
    return state


def _decode_abort_scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.waiting_queue = []
    scheduler.enable_hicache_storage = False
    scheduler.tree_cache = object()
    scheduler.send_to_tokenizer = SimpleNamespace(send_output=MagicMock())
    scheduler.stream_output = MagicMock()
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.grammar_manager = SimpleNamespace(abort_requests=MagicMock())
    scheduler.enable_priority_scheduling = False
    scheduler.enable_hisparse = False
    scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
        queue=[], retracted_queue=[]
    )
    scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
    scheduler.running_batch = SimpleNamespace(reqs=[])
    scheduler.cur_batch = None
    return scheduler


@pytest.mark.parametrize("prompt_token_ids", ([13], [11, 12, 13]))
def test_inflight_retract_restores_ready_and_marks_seed_slot_overallocated(
    prompt_token_ids,
):
    state = _state(
        prompt_token_ids,
        schedule_batch_module.WelmDeferredDecodePhase.INFLIGHT,
    )
    req = Req.__new__(Req)
    req.rid = "seed"
    req.origin_input_ids = prompt_token_ids
    req.output_ids = []
    req.welm_deferred_decode_state = state
    req.kv_committed_len = len(prompt_token_ids)
    req.kv_allocated_len = len(prompt_token_ids)
    req.offload_kv_cache = MagicMock()
    req.reset_for_retract = MagicMock()

    batch = ScheduleBatch(
        reqs=[req],
        tree_cache=object(),
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
    )
    captured = {}

    def capture_release(
        released_req,
        tree_cache,
        is_insert,
        allow_uncommitted_tail=False,
    ):
        captured["phase"] = released_req.welm_deferred_decode_state.phase
        captured["committed"] = released_req.kv_committed_len
        captured["allocated"] = released_req.kv_allocated_len
        captured["is_insert"] = is_insert
        captured["allow_uncommitted_tail"] = allow_uncommitted_tail

    with (
        patch(
            "sglang.srt.managers.schedule_batch.release_kv_cache",
            side_effect=capture_release,
        ),
        patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
    ):
        batch.release_req(
            0,
            remaing_req_count=0,
            server_args=SimpleNamespace(disaggregation_mode="decode"),
        )

    req.offload_kv_cache.assert_called_once()
    req.reset_for_retract.assert_called_once()
    assert captured == {
        "phase": schedule_batch_module.WelmDeferredDecodePhase.READY,
        "committed": len(prompt_token_ids) - 1,
        "allocated": len(prompt_token_ids),
        "is_insert": False,
        "allow_uncommitted_tail": True,
    }


@pytest.mark.parametrize(
    "phase",
    (
        schedule_batch_module.WelmDeferredDecodePhase.PREFILL_PENDING,
        schedule_batch_module.WelmDeferredDecodePhase.READY,
    ),
)
def test_retract_rejects_deferred_state_that_is_not_running(phase):
    req = Req.__new__(Req)
    req.origin_input_ids = [11, 12, 13]
    req.output_ids = []
    req.welm_deferred_decode_state = _state(req.origin_input_ids, phase)
    req.kv_committed_len = 2
    req.kv_allocated_len = 2

    with pytest.raises(RuntimeError, match="requires INFLIGHT or CONSUMED"):
        req.prepare_for_retract()


def test_release_kv_cache_requires_explicit_uncommitted_tail_permission():
    def make_req():
        req = SimpleNamespace(
            req_pool_idx=0,
            kv_committed_len=2,
            kv_allocated_len=3,
            kv_committed_freed=False,
            kv_overallocated_freed=False,
            mamba_pool_idx=None,
        )

        def pop_committed():
            req.kv_committed_freed = True
            return req.kv_committed_len

        def pop_overallocated():
            req.kv_overallocated_freed = True
            return req.kv_committed_len, req.kv_allocated_len

        req.pop_committed_kv_cache = pop_committed
        req.pop_overallocated_kv_cache = pop_overallocated
        return req

    def make_tree_cache():
        req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11, 12]], dtype=torch.int64),
            free=MagicMock(),
        )
        allocator = SimpleNamespace(free=MagicMock())
        tree_cache = SimpleNamespace(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            supports_mamba=lambda: False,
        )
        tree_cache.cache_finished_req = MagicMock(
            side_effect=lambda req, is_insert: req.pop_committed_kv_cache()
        )
        return tree_cache

    server_args = SimpleNamespace(
        page_size=1,
        speculative_algorithm=None,
        strip_thinking_cache=False,
    )
    with patch(
        "sglang.srt.mem_cache.common.get_global_server_args",
        return_value=server_args,
    ):
        with pytest.raises(AssertionError, match="Unexpected overallocated"):
            release_kv_cache(make_req(), make_tree_cache(), is_insert=False)

        req = make_req()
        tree_cache = make_tree_cache()
        release_kv_cache(
            req,
            tree_cache,
            is_insert=False,
            allow_uncommitted_tail=True,
        )

    torch.testing.assert_close(
        tree_cache.token_to_kv_pool_allocator.free.call_args.args[0],
        torch.tensor([12], dtype=torch.int64),
    )
    tree_cache.req_to_token_pool.free.assert_called_once_with(req)


def test_prefill_pending_prealloc_abort_cleans_receiver_and_pending_alias_once():
    req = SimpleNamespace(
        rid="seed",
        return_logprob=False,
        finished_reason=None,
        welm_deferred_decode_state=_state(
            [11, 12, 13],
            schedule_batch_module.WelmDeferredDecodePhase.PREFILL_PENDING,
        ),
    )
    receiver = MagicMock()
    decode_req = decode_module.DecodeRequest(req=req, kv_receiver=receiver)
    scheduler = _decode_abort_scheduler()
    queue = decode_module.DecodePreallocQueue.__new__(
        decode_module.DecodePreallocQueue
    )
    queue.queue = [decode_req]
    queue.pending_reqs = [decode_req]
    queue.retracted_queue = []
    queue.scheduler = scheduler
    queue._resolve_pending_reqs = MagicMock()
    queue._update_handshake_waiters = MagicMock()
    queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
    queue._allocatable_token_budgets = MagicMock(return_value=0)
    scheduler.disagg_decode_prealloc_queue = queue

    scheduler.abort_request(AbortReq(rid=req.rid))
    scheduler.abort_request(AbortReq(rid=req.rid))
    preallocated, failed = queue.pop_preallocated()

    assert preallocated == []
    assert failed == [decode_req]
    receiver.abort.assert_called_once_with()
    receiver.clear.assert_called_once_with()
    assert decode_req.kv_receiver is None
    assert queue.queue == []
    assert queue.pending_reqs == []
    scheduler.stream_output.assert_called_once_with([req], False)


def test_prefill_pending_transfer_abort_releases_owned_resources_once():
    req = SimpleNamespace(
        rid="seed",
        bootstrap_host="prefill",
        bootstrap_room=7,
        return_logprob=False,
        finished_reason=None,
        welm_deferred_decode_state=_state(
            [11, 12, 13],
            schedule_batch_module.WelmDeferredDecodePhase.PREFILL_PENDING,
        ),
    )
    receiver = MagicMock()
    decode_req = decode_module.DecodeRequest(
        req=req,
        kv_receiver=receiver,
        metadata_buffer_index=3,
    )
    scheduler = _decode_abort_scheduler()
    scheduler.server_args = SimpleNamespace(
        disaggregation_transfer_backend="mooncake"
    )
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.stream_output = MagicMock()
    scheduler.enable_hisparse = False
    scheduler.enable_metrics = False
    queue = decode_module.DecodeTransferQueue.__new__(
        decode_module.DecodeTransferQueue
    )
    queue.queue = [decode_req]
    queue.enable_staging = False
    queue.gloo_group = object()
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.scheduler = scheduler
    queue.tree_cache = object()
    queue.tp_rank = 0
    scheduler.disagg_decode_transfer_queue = queue

    with (
        patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[decode_module.KVPoll.Failed],
        ),
        patch("sglang.srt.disaggregation.decode.release_kv_cache") as release,
    ):
        scheduler.abort_request(AbortReq(rid=req.rid))
        scheduler.abort_request(AbortReq(rid=req.rid))
        transferred = queue.pop_transferred()

    assert transferred == []
    receiver.abort.assert_called_once_with()
    receiver.clear.assert_called_once_with()
    assert decode_req.kv_receiver is None
    release.assert_called_once_with(req, queue.tree_cache, is_insert=False)
    queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
    scheduler.stream_output.assert_called_once_with([req], False)
    assert queue.queue == []


def test_ready_waiting_abort_releases_decode_allocation_once():
    req = SimpleNamespace(
        rid="seed",
        mamba_pool_idx=None,
        welm_deferred_decode_state=_state(
            [11, 12, 13], schedule_batch_module.WelmDeferredDecodePhase.READY
        ),
    )
    scheduler = _decode_abort_scheduler()
    scheduler.waiting_queue = [req]

    with patch(
        "sglang.srt.managers.scheduler.release_kv_cache"
    ) as release:
        scheduler.abort_request(AbortReq(rid=req.rid))
        scheduler.abort_request(AbortReq(rid=req.rid))

    release.assert_called_once_with(req, scheduler.tree_cache)
    scheduler.send_to_tokenizer.send_output.assert_called_once()
    assert scheduler.waiting_queue == []


@pytest.mark.parametrize(
    "phase",
    (
        schedule_batch_module.WelmDeferredDecodePhase.INFLIGHT,
        schedule_batch_module.WelmDeferredDecodePhase.CONSUMED,
    ),
)
def test_running_abort_stays_on_standard_decode_finish_path(phase):
    req = SimpleNamespace(
        rid="seed",
        to_finish=None,
        finished=lambda: False,
        welm_deferred_decode_state=_state([11, 12, 13], phase),
    )
    scheduler = _decode_abort_scheduler()
    scheduler.running_batch = SimpleNamespace(reqs=[req])

    with patch(
        "sglang.srt.managers.scheduler.release_kv_cache"
    ) as release:
        scheduler.abort_request(AbortReq(rid=req.rid))
        first_finish = req.to_finish
        scheduler.abort_request(AbortReq(rid=req.rid))

    assert isinstance(first_finish, schedule_batch_module.FINISH_ABORT)
    assert isinstance(req.to_finish, schedule_batch_module.FINISH_ABORT)
    release.assert_not_called()


def test_consumed_state_uses_ordinary_retracted_decode_fill_semantics():
    state = _state(
        [11, 12, 13], schedule_batch_module.WelmDeferredDecodePhase.CONSUMED
    )
    req = SimpleNamespace(
        origin_input_ids=[11, 12, 13],
        output_ids=[42, 43],
        welm_deferred_decode_state=state,
    )

    assert decode_module._decode_transfer_fill_ids(req, _server_args()) == [
        11,
        12,
        13,
        42,
    ]
    assert decode_module._decode_transfer_fill_len(req, _server_args()) == 4


def test_consumed_state_may_enter_legacy_prebuilt_resume_path():
    state = _state(
        [11, 12, 13], schedule_batch_module.WelmDeferredDecodePhase.CONSUMED
    )
    req = SimpleNamespace(
        rid="consumed",
        origin_input_ids=[11, 12, 13],
        output_ids=[42],
        welm_deferred_decode_state=state,
        kv_committed_len=3,
        fill_ids=[11, 12, 13],
        prefix_indices=torch.empty((0,), dtype=torch.int64),
        init_next_round_input=MagicMock(),
        set_extend_input_len=MagicMock(),
        time_stats=SimpleNamespace(set_forward_entry_time=MagicMock()),
    )
    scheduler = SimpleNamespace(
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        waiting_queue=[req],
        enable_priority_scheduling=False,
        running_batch=SimpleNamespace(batch_size=lambda: 0),
        req_to_token_pool=SimpleNamespace(size=4),
        max_running_requests=4,
        server_args=_server_args(),
        tree_cache=object(),
        token_to_kv_pool_allocator=object(),
        model_config=object(),
        enable_overlap=False,
        spec_algorithm=object(),
        future_map=object(),
        _should_isolate_welm_mtp_prebuilt=lambda: False,
    )
    new_batch = MagicMock()

    with patch.object(
        ScheduleBatch,
        "init_new",
        return_value=new_batch,
    ):
        result = (
            decode_module.SchedulerDisaggregationDecodeMixin
            .get_new_prebuilt_batch(scheduler)
        )

    assert result is new_batch
    assert scheduler.waiting_queue == []
    req.init_next_round_input.assert_called_once_with(scheduler.tree_cache)
    new_batch.prepare_for_prebuilt.assert_called_once()
    new_batch.process_prebuilt.assert_called_once_with(
        scheduler.server_args, scheduler.future_map
    )


def test_deferred_seed_scheduler_leaves_consumed_request_for_prebuilt():
    req = SimpleNamespace(
        rid="consumed",
        welm_deferred_decode_state=_state(
            [11, 12, 13],
            schedule_batch_module.WelmDeferredDecodePhase.CONSUMED,
        ),
        finished=lambda: False,
        is_retracted=False,
    )
    scheduler = SimpleNamespace(
        server_args=_server_args(),
        disaggregation_mode=DisaggregationMode.DECODE,
        waiting_queue=[req],
        enable_priority_scheduling=False,
        running_batch=SimpleNamespace(reqs=[]),
        req_to_token_pool=SimpleNamespace(size=4),
        max_running_requests=4,
    )

    result = Scheduler.get_new_welm_deferred_seed_batch(scheduler)

    assert result is None
    assert scheduler.waiting_queue == [req]


def test_only_unconsumed_deferred_state_is_rejected_from_prebuilt():
    consumed = SimpleNamespace(
        rid="consumed",
        welm_deferred_decode_state=_state(
            [1], schedule_batch_module.WelmDeferredDecodePhase.CONSUMED
        ),
    )
    _assert_no_welm_deferred_decode_seed([consumed])

    ready = SimpleNamespace(
        rid="ready",
        welm_deferred_decode_state=_state(
            [1], schedule_batch_module.WelmDeferredDecodePhase.READY
        ),
    )
    with pytest.raises(RuntimeError, match="legacy PREBUILT"):
        _assert_no_welm_deferred_decode_seed([ready])


def test_first_deferred_output_uses_standard_logprob_grammar_and_streaming_once(
    monkeypatch,
):
    state = _state(
        [11, 12, 13], schedule_batch_module.WelmDeferredDecodePhase.INFLIGHT
    )
    grammar = MagicMock()
    grammar.is_terminated.return_value = False
    req = Req.__new__(Req)
    req.rid = "seed"
    req.origin_input_ids = [11, 12, 13]
    req.output_ids = []
    req.welm_deferred_decode_state = state
    req.return_logprob = True
    req.output_token_logprobs_val = []
    req.output_token_logprobs_idx = []
    req.output_top_logprobs_val = []
    req.output_top_logprobs_idx = []
    req.output_token_ids_logprobs_val = []
    req.output_token_ids_logprobs_idx = []
    req.top_logprobs_num = 0
    req.token_ids_logprob = None
    req.return_hidden_states = False
    req.grammar = grammar
    req.is_retracted = False
    req.finished = lambda: False
    req.to_finish = None
    req.sampling_params = SimpleNamespace(max_new_tokens=8)
    req._check_token_based_finish = lambda _tokens: False
    req._check_vocab_boundary_finish = lambda _tokens: False
    req._check_str_based_finish = lambda: False
    req.time_stats = SimpleNamespace(
        set_last_decode_finish_time=MagicMock(),
        last_forward_entry_time=None,
        forward_entry_time=None,
        last_decode_finish_time=None,
    )

    batch = ScheduleBatch(
        reqs=[req],
        return_logprob=True,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        enable_overlap=False,
    )
    result = GenerationBatchResult(
        logits_output=SimpleNamespace(
            next_token_logprobs=torch.tensor([-0.25]),
            next_token_top_logprobs_val=[],
            next_token_top_logprobs_idx=[],
            next_token_token_ids_logprobs_val=[],
            next_token_token_ids_logprobs_idx=[],
            hidden_states=None,
        ),
        next_token_ids=torch.tensor([42], dtype=torch.int64),
        can_run_cuda_graph=False,
    )
    scheduler = SchedulerOutputProcessorMixin()
    scheduler.enable_overlap = False
    scheduler.enable_overlap_mlx = False
    scheduler.num_generated_tokens = 0
    scheduler.enable_metrics = False
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        free_group_begin=MagicMock(),
        free_group_end=MagicMock(),
    )
    scheduler._maybe_update_reasoning_tokens = MagicMock()
    scheduler._mamba_prefix_cache_update = MagicMock()
    scheduler._handle_finished_req = MagicMock()
    scheduler.stream_output = MagicMock()
    scheduler.forward_ct_decode = 0
    scheduler.report_decode_stats = MagicMock()
    monkeypatch.setattr(
        output_processor_module, "decode_scheduler_profile_enabled", lambda: False
    )
    monkeypatch.setattr(
        output_processor_module, "hicache_timing_enabled", lambda: False
    )

    scheduler.process_batch_result_decode(batch, result)

    assert req.output_ids == [42]
    assert state.phase is schedule_batch_module.WelmDeferredDecodePhase.CONSUMED
    assert req.output_token_logprobs_val == [-0.25]
    assert req.output_token_logprobs_idx == [42]
    grammar.accept_token.assert_called_once_with(42)
    scheduler.stream_output.assert_called_once_with([req], True)
