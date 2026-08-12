from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    Req,
    ScheduleBatch,
    WelmDeferredDecodePhase,
    WelmDeferredDecodeState,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    WelmDeferredPrefillCompletion,
)
from sglang.srt.models.welm_deferred_mirror import (
    WelmPDExecutionMode,
    build_welm_deferred_prefill_span,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def _req(token_ids=(11, 22, 33, 44)):
    req = Req.__new__(Req)
    req.rid = "req"
    req.origin_input_ids = list(token_ids)
    req.output_ids = []
    req.return_logprob = False
    req.logprob_start_len = -1
    req.return_hidden_states = False
    req.return_routed_experts = False
    req.return_indexer_topk = False
    req.input_embeds = None
    req.positional_embed_overrides = None
    req.multimodal_inputs = None
    req.grammar = MagicMock()
    req.sampling_params = SimpleNamespace(max_new_tokens=17)
    req.welm_deferred_prefill_span = None
    req.welm_deferred_decode_state = None
    req.kv_committed_len = 0
    req.kv_allocated_len = 0
    req.attn_cp_prefill_split_spec = None
    req.time_stats = SimpleNamespace(
        set_wait_queue_entry_time=MagicMock(),
        set_prefill_finished_time=MagicMock(),
        set_prefill_transfer_queue_entry_time=MagicMock(),
    )
    return req


def _scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.enable_overlap = False
    scheduler.tree_cache = object()
    scheduler.server_args = SimpleNamespace(
        welm_kv_mirror_pd_mode=WelmPDExecutionMode.DEFERRED_LAST_PROMPT.value
    )
    scheduler.waiting_queue = []
    scheduler._set_or_validate_priority = MagicMock(return_value=True)
    scheduler._abort_on_queued_limit = MagicMock(return_value=False)
    scheduler._prefetch_kvcache = MagicMock()
    scheduler.stream_output = MagicMock()
    return scheduler


def _ready_full_hit_req(scheduler):
    req = _req([11, 22, 33])
    req.finished_reason = None
    req.to_finish = None
    req.req_pool_idx = 7
    req.mamba_pool_idx = None
    req.priority = 1
    req.time_stats.trace_ctx = MagicMock()
    req.time_stats.wait_queue_entry_time = 1
    Scheduler._add_request_to_queue(scheduler, req)
    scheduler.waiting_queue.clear()
    req.extend_input_len = 0
    req.kv_committed_len = 2
    req.kv_allocated_len = 2
    Scheduler.process_deferred_prefill_without_forward(scheduler, [req])
    return req


def test_monolithic_admission_attaches_span_and_state_before_prefetch():
    scheduler = _scheduler()
    req = _req()
    observed = {}

    def capture_prefetch(prefetched_req):
        observed["span"] = prefetched_req.welm_deferred_prefill_span
        observed["state"] = prefetched_req.welm_deferred_decode_state

    scheduler._prefetch_kvcache.side_effect = capture_prefetch

    Scheduler._add_request_to_queue(scheduler, req)

    span = req.welm_deferred_prefill_span
    state = req.welm_deferred_decode_state
    assert observed == {"span": span, "state": state}
    assert isinstance(state, WelmDeferredDecodeState)
    assert state.phase is WelmDeferredDecodePhase.PREFILL_PENDING
    assert state.committed_kv_len == 3
    assert state.seed_position == 3
    assert state.seed_token_id == 44
    assert req.sampling_params.max_new_tokens == 17
    assert scheduler.waiting_queue == [req]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("return_hidden_states", True, "hidden states"),
        ("sampling_params", SimpleNamespace(max_new_tokens=0), "zero-generation"),
    ],
)
def test_monolithic_admission_rejects_unsupported_payload_before_prefetch(
    field, value, message
):
    scheduler = _scheduler()
    req = _req()
    setattr(req, field, value)

    with patch.object(scheduler_module, "prepare_abort") as prepare_abort:
        Scheduler._add_request_to_queue(scheduler, req)

    prepare_abort.assert_called_once()
    assert message in prepare_abort.call_args.args[1]
    assert prepare_abort.call_args.kwargs["status_code"] is HTTPStatus.BAD_REQUEST
    scheduler.stream_output.assert_called_once_with([req], req.return_logprob)
    scheduler._prefetch_kvcache.assert_not_called()
    assert scheduler.waiting_queue == []


def test_monolithic_session_continuation_is_rejected_before_session_lookup():
    scheduler = _scheduler()

    class SessionLookupGuard:
        def __contains__(self, _session_id):
            raise AssertionError("session controller must not be queried")

        get = MagicMock()

    scheduler.session_controller = SessionLookupGuard()
    scheduler.tokenizer = object()
    scheduler.model_config = SimpleNamespace(vocab_size=128)
    recv_req = SimpleNamespace(
        rid="session-req",
        input_text="prompt",
        input_ids=[11, 22],
        sampling_params=SamplingParams(max_new_tokens=8),
        session_params=SimpleNamespace(id="session-id"),
        http_worker_ipc=None,
        time_stats=APIServerReqTimeStats(),
    )

    with patch.object(scheduler_module, "prepare_abort") as prepare_abort:
        Scheduler.handle_generate_request(scheduler, recv_req)

    scheduler.session_controller.get.assert_not_called()
    prepare_abort.assert_called_once()
    assert "session continuation" in prepare_abort.call_args.args[1]
    assert prepare_abort.call_args.kwargs["status_code"] is HTTPStatus.BAD_REQUEST
    scheduler.stream_output.assert_called_once()


def test_monolithic_full_hit_transitions_ready_without_transfer():
    scheduler = _scheduler()
    scheduler.send_kv_chunk = MagicMock()
    scheduler.disagg_prefill_inflight_queue = []
    req = _req([77])
    req.extend_input_len = 0
    Scheduler._add_request_to_queue(scheduler, req)
    scheduler.waiting_queue.clear()

    Scheduler.process_deferred_prefill_without_forward(scheduler, [req])

    assert req.welm_deferred_decode_state.phase is WelmDeferredDecodePhase.READY
    assert scheduler.waiting_queue == [req]
    scheduler.send_kv_chunk.assert_not_called()
    assert scheduler.disagg_prefill_inflight_queue == []


def test_monolithic_inflight_retraction_reenters_prefill_after_kv_release():
    scheduler = _scheduler()
    req = _ready_full_hit_req(scheduler)
    scheduler.waiting_queue.clear()
    state = req.welm_deferred_decode_state
    state.transition_to(WelmDeferredDecodePhase.INFLIGHT)
    req.kv_committed_len = state.committed_kv_len + 1
    req.kv_allocated_len = state.committed_kv_len + 1
    req.offload_kv_cache = MagicMock()
    req.reset_for_retract = MagicMock()

    batch = ScheduleBatch(reqs=[req], tree_cache=scheduler.tree_cache)

    def release_row(released_req, *_args, **_kwargs):
        released_req.req_pool_idx = None

    with (
        patch(
            "sglang.srt.managers.schedule_batch.release_kv_cache",
            side_effect=release_row,
        ),
        patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
    ):
        batch.release_req(
            0,
            remaing_req_count=0,
            server_args=SimpleNamespace(disaggregation_mode="null"),
        )

    assert req.req_pool_idx is None
    assert state.phase is WelmDeferredDecodePhase.PREFILL_PENDING

    Scheduler._add_request_to_queue(scheduler, req, is_retracted=True)
    scheduler.enable_priority_scheduling = False
    scheduler.req_to_token_pool = SimpleNamespace(size=8)
    scheduler.max_running_requests = 8
    scheduler.running_batch = ScheduleBatch(reqs=[])

    assert Scheduler.get_new_welm_deferred_seed_batch(scheduler) is None
    assert scheduler.waiting_queue == [req]


@pytest.mark.parametrize("removal", ("abort", "timeout", "priority"))
def test_monolithic_ready_full_hit_queue_removal_releases_allocation(removal):
    scheduler = _scheduler()
    scheduler.enable_hicache_storage = False
    scheduler.enable_hierarchical_cache = False
    scheduler.send_to_tokenizer = SimpleNamespace(send_output=MagicMock())
    scheduler.grammar_manager = SimpleNamespace(abort_requests=MagicMock())
    scheduler.running_batch = ScheduleBatch(reqs=[])
    scheduler.cur_batch = None
    req = _ready_full_hit_req(scheduler)

    with patch.object(scheduler_module, "release_kv_cache") as release:
        if removal == "abort":
            Scheduler.abort_request(scheduler, AbortReq(rid=req.rid))
        elif removal == "timeout":
            req.time_stats.wait_queue_entry_time = 1
            with (
                patch.object(
                    scheduler_module.envs.SGLANG_REQ_WAITING_TIMEOUT,
                    "get",
                    return_value=10,
                ),
                patch.object(scheduler_module.time, "perf_counter", return_value=20),
            ):
                Scheduler._abort_on_waiting_timeout(scheduler)
        else:
            scheduler.max_queued_requests = 1
            scheduler.enable_priority_scheduling = True
            scheduler.schedule_low_priority_values_first = False
            incoming = _req([41, 42])
            incoming.rid = "higher-priority"
            incoming.priority = 2
            incoming.time_stats.trace_ctx = MagicMock()
            assert Scheduler._abort_on_queued_limit(scheduler, incoming) is False

    release.assert_called_once_with(req, scheduler.tree_cache, is_insert=False)
    assert scheduler.waiting_queue == []


def test_monolithic_main_loop_admits_ready_seed_into_running_batch():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_fpm = False
    scheduler._abort_on_waiting_timeout = MagicMock()
    scheduler._abort_on_running_timeout = MagicMock()
    scheduler.dllm_config = None
    scheduler.dllm_manager = None
    scheduler.enable_hisparse = False
    scheduler.last_batch = None
    scheduler.chunked_req = None
    scheduler.running_batch = ScheduleBatch(reqs=[])
    scheduler.require_mlp_sync = False
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    scheduler.server_args = SimpleNamespace(
        welm_kv_mirror_pd_mode=WelmPDExecutionMode.DEFERRED_LAST_PROMPT.value,
        speculative_skip_dp_mlp_sync=True,
    )
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    seed_batch = ScheduleBatch(reqs=[_req()])
    scheduler.get_new_welm_deferred_seed_batch = MagicMock(
        return_value=seed_batch
    )
    scheduler.get_new_batch_prefill = MagicMock(return_value=None)
    scheduler.maybe_prepare_mlp_sync_batch = MagicMock(
        side_effect=lambda batch, **_kwargs: batch
    )
    scheduler._maybe_prepare_ngram_embedding = MagicMock(
        side_effect=lambda batch: batch
    )
    scheduler.update_running_batch = MagicMock(side_effect=lambda batch: batch)

    with patch.object(scheduler_module, "set_schedule_time_batch"):
        batch = Scheduler.get_next_batch_to_run(scheduler)

    assert batch is seed_batch
    assert scheduler.running_batch is seed_batch
    scheduler.update_running_batch.assert_called_once_with(seed_batch)


def _deferred_completion_req(rid, token_ids, *, final):
    req = _req(token_ids)
    req.rid = rid
    scheduler = _scheduler()
    Scheduler._add_request_to_queue(scheduler, req)
    req.fill_ids = list(token_ids[:-1] if final else token_ids[:-2])
    req.is_chunked = 0 if final else 1
    req.kv_committed_len = len(req.fill_ids)
    req.kv_allocated_len = len(req.fill_ids)
    req.time_stats.set_last_chunked_prefill_finish_time = MagicMock()
    return req


def _deferred_completion_batch(reqs, final_mask):
    return SimpleNamespace(
        reqs=reqs,
        forward_mode=ForwardMode.EXTEND,
        welm_deferred_prefill=True,
        welm_deferred_prefill_final_mask=tuple(final_mask),
        return_logprob=False,
        return_hidden_states=False,
        spec_info=None,
        attn_cp_prefill_split_specs=None,
        prefill_stats=None,
        dp_cooperation_info=None,
    )


def _deferred_completion_result(size):
    return GenerationBatchResult(
        logits_output=None,
        next_token_ids=torch.full((size,), -1, dtype=torch.int64),
        welm_deferred_prefill_completion=WelmDeferredPrefillCompletion(),
    )


def test_monolithic_completion_patches_only_final_seed_before_future_store():
    intermediate = _deferred_completion_req(
        "intermediate", [11, 12, 13, 14], final=False
    )
    final = _deferred_completion_req("final", [21, 22, 23], final=True)
    batch = _deferred_completion_batch([intermediate, final], [False, True])
    result = _deferred_completion_result(2)
    scheduler = _scheduler()

    cache_observations = []
    with patch.object(
        scheduler_module,
        "maybe_cache_unfinished_req",
        side_effect=lambda req, tree_cache: cache_observations.append(
            (req, tree_cache, req.welm_deferred_decode_state.phase)
        ),
    ):
        handled = Scheduler._prepare_welm_deferred_prefill_result_for_decode(
            scheduler, batch, result
        )

    assert handled is True
    assert cache_observations == [
        (final, scheduler.tree_cache, WelmDeferredDecodePhase.PREFILL_PENDING)
    ]
    assert result.next_token_ids.tolist() == [-1, 23]
    assert (
        intermediate.welm_deferred_decode_state.phase
        is WelmDeferredDecodePhase.PREFILL_PENDING
    )
    assert final.welm_deferred_decode_state.phase is WelmDeferredDecodePhase.READY


def test_overlap_run_batch_patches_seed_before_future_map_store():
    req = _deferred_completion_req("final", [21, 22, 23], final=True)
    batch = _deferred_completion_batch([req], [True])
    worker_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(copy_for_forward=lambda: object()),
        seq_lens=[2],
    )
    batch.get_model_worker_batch = lambda: worker_batch
    batch.is_spec_v2 = False
    result = _deferred_completion_result(1)
    future_indices = SimpleNamespace(indices=torch.tensor([7], dtype=torch.int64))
    future_map = MagicMock()
    future_map.alloc_future_indices.return_value = future_indices

    operation_order = []
    active_stream = {"name": None}

    class NamedStreamContext:
        def __init__(self, name):
            self.name = name
            self.previous = None

        def __enter__(self):
            self.previous = active_stream["name"]
            active_stream["name"] = self.name

        def __exit__(self, *_args):
            active_stream["name"] = self.previous

    def cache_prefix(cached_req, tree_cache):
        assert cached_req is req
        assert tree_cache is scheduler.tree_cache
        assert active_stream["name"] == "schedule"
        assert (
            req.welm_deferred_decode_state.phase
            is WelmDeferredDecodePhase.PREFILL_PENDING
        )
        operation_order.append("cache")

    def assert_seed_ready(_indices, stored_result):
        operation_order.append("future")
        assert operation_order == ["cache", "future"]
        assert active_stream["name"] == "forward"
        assert stored_result.next_token_ids.tolist() == [23]
        assert req.welm_deferred_decode_state.phase is WelmDeferredDecodePhase.READY

    future_map.store_to_map.side_effect = assert_seed_ready
    scheduler = _scheduler()
    scheduler.forward_ct = 0
    scheduler._profile_batch_predicate = MagicMock()
    scheduler.forward_sleep_time = None
    scheduler.is_generation = True
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    scheduler.enable_overlap = True
    scheduler.record_batch_in_overlap = MagicMock()
    scheduler.future_map = future_map
    scheduler.forward_stream_ctx = NamedStreamContext("forward")
    scheduler.forward_stream = SimpleNamespace(name="forward", wait_stream=MagicMock())
    scheduler.schedule_stream = SimpleNamespace(name="schedule")
    scheduler.model_worker = SimpleNamespace(
        forward_batch_generation=MagicMock(return_value=result)
    )
    scheduler.device_module = SimpleNamespace(
        Event=MagicMock(return_value=MagicMock()),
        StreamContext=lambda stream: NamedStreamContext(stream.name),
    )
    scheduler.server_args.enable_dp_attention = False

    with patch.object(
        scheduler_module,
        "maybe_cache_unfinished_req",
        side_effect=cache_prefix,
    ):
        Scheduler.run_batch(scheduler, batch)

    future_map.store_to_map.assert_called_once_with(future_indices, result)
    assert batch.output_ids.tolist() == [-7]


def test_non_overlap_run_batch_uses_seed_as_decode_input():
    req = _deferred_completion_req("final", [31, 32, 33], final=True)
    batch = _deferred_completion_batch([req], [True])
    batch.get_model_worker_batch = lambda: SimpleNamespace()
    batch.is_spec_v2 = False
    result = _deferred_completion_result(1)
    scheduler = _scheduler()
    scheduler.forward_ct = 0
    scheduler._profile_batch_predicate = MagicMock()
    scheduler.forward_sleep_time = None
    scheduler.is_generation = True
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    scheduler.enable_overlap = False
    scheduler.enable_pdmux = False
    scheduler.model_worker = SimpleNamespace(
        forward_batch_generation=MagicMock(return_value=result)
    )
    scheduler.update_cache_from_scheduler = MagicMock()
    scheduler.server_args.enable_dp_attention = False

    with patch.object(scheduler_module, "maybe_cache_unfinished_req") as cache_prefix:
        Scheduler.run_batch(scheduler, batch)

    cache_prefix.assert_called_once_with(req, scheduler.tree_cache)
    assert batch.output_ids.tolist() == [33]
    assert req.welm_deferred_decode_state.phase is WelmDeferredDecodePhase.READY


def test_completion_uses_frozen_final_mask_after_chunk_state_advances():
    req = _deferred_completion_req("final", [21, 22, 23], final=True)
    req.is_chunked = 1
    batch = _deferred_completion_batch([req], [True])
    result = _deferred_completion_result(1)
    scheduler = _scheduler()

    with patch.object(scheduler_module, "maybe_cache_unfinished_req") as cache_prefix:
        Scheduler._prepare_welm_deferred_prefill_result_for_decode(
            scheduler, batch, result
        )

    cache_prefix.assert_called_once_with(req, scheduler.tree_cache)
    assert result.next_token_ids.tolist() == [23]
    assert req.welm_deferred_decode_state.phase is WelmDeferredDecodePhase.READY


def test_cache_publication_failure_does_not_publish_seed():
    req = _deferred_completion_req("final", [21, 22, 23], final=True)
    batch = _deferred_completion_batch([req], [True])
    result = _deferred_completion_result(1)
    scheduler = _scheduler()

    with (
        patch.object(
            scheduler_module,
            "maybe_cache_unfinished_req",
            side_effect=RuntimeError("cache publication failed"),
        ),
        pytest.raises(RuntimeError, match="cache publication failed"),
    ):
        Scheduler._prepare_welm_deferred_prefill_result_for_decode(
            scheduler, batch, result
        )

    assert result.next_token_ids.tolist() == [-1]
    assert (
        req.welm_deferred_decode_state.phase is WelmDeferredDecodePhase.PREFILL_PENDING
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda batch, _result: setattr(
                batch, "welm_deferred_prefill_final_mask", (True,)
            ),
            "final mask",
        ),
        (
            lambda batch, _result: setattr(batch, "forward_mode", ForwardMode.DECODE),
            "EXTEND",
        ),
        (
            lambda batch, _result: setattr(batch.reqs[1], "fill_ids", []),
            "committed span",
        ),
        (
            lambda batch, _result: batch.reqs[
                1
            ].welm_deferred_decode_state.transition_to(WelmDeferredDecodePhase.READY),
            "PREFILL_PENDING",
        ),
        (
            lambda batch, _result: setattr(
                batch.reqs[1].welm_deferred_decode_state, "seed_token_id", 999
            ),
            "seed token",
        ),
        (
            lambda _batch, result: setattr(result, "delay_sample_func", lambda: None),
            "delayed sampling",
        ),
        (
            lambda _batch, result: setattr(
                result, "next_token_ids", torch.tensor([-1], dtype=torch.int64)
            ),
            "request-aligned",
        ),
    ],
)
def test_monolithic_completion_validation_is_atomic(mutate, message):
    first = _deferred_completion_req("first", [11, 12], final=True)
    second = _deferred_completion_req("second", [21, 22], final=True)
    batch = _deferred_completion_batch([first, second], [True, True])
    result = _deferred_completion_result(2)
    scheduler = _scheduler()
    mutate(batch, result)

    with pytest.raises(RuntimeError, match=message):
        Scheduler._prepare_welm_deferred_prefill_result_for_decode(
            scheduler, batch, result
        )

    assert result.next_token_ids.tolist() in ([-1, -1], [-1])
    assert (
        first.welm_deferred_decode_state.phase
        is WelmDeferredDecodePhase.PREFILL_PENDING
    )


def _metadata_req(rid, prompt, *, scheduled_len, with_state=True):
    span = build_welm_deferred_prefill_span(prompt)
    state = WelmDeferredDecodeState.from_prefill_span(span) if with_state else None
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(prompt),
        output_ids=[],
        fill_ids=list(prompt[:scheduled_len]),
        welm_deferred_prefill_span=span,
        welm_deferred_decode_state=state,
        attn_cp_prefill_split_spec=None,
        stream=False,
        grammar=None,
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        is_prefill_only=False,
        lora_id=None,
        dllm_block_offset=0,
        _scale_seq_factor=1,
    )


def _metadata_batch(reqs):
    return ScheduleBatch.init_new(
        reqs,
        SimpleNamespace(device="cpu"),
        object(),
        object(),
        SimpleNamespace(vocab_size=128),
        enable_overlap=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
    )


def test_uniform_pending_rows_freeze_request_aligned_final_mask():
    intermediate = _metadata_req("intermediate", [1, 2, 3, 4], scheduled_len=2)
    final = _metadata_req("final", [5, 6, 7, 8], scheduled_len=3)

    batch = _metadata_batch([intermediate, final])

    assert batch.welm_deferred_prefill is True
    assert batch.welm_deferred_prefill_final_mask == (False, True)
    snapshot = batch.copy()
    assert snapshot.welm_deferred_prefill is True
    assert snapshot.welm_deferred_prefill_final_mask == (False, True)


def test_prefill_role_span_without_decode_state_is_still_marked():
    batch = _metadata_batch(
        [_metadata_req("prefill", [1, 2, 3], scheduled_len=2, with_state=False)]
    )

    assert batch.welm_deferred_prefill is True
    assert batch.welm_deferred_prefill_final_mask == (True,)


def test_pending_deferred_and_ordinary_rows_cannot_share_local_batch():
    pending = _metadata_req("pending", [1, 2, 3], scheduled_len=2)
    ordinary = _metadata_req("ordinary", [4, 5, 6], scheduled_len=2)
    ordinary.welm_deferred_prefill_span = None
    ordinary.welm_deferred_decode_state = None

    with pytest.raises(RuntimeError, match="mixed.*deferred Prefill"):
        _metadata_batch([pending, ordinary])


def test_only_uniform_marker_crosses_worker_and_forward_boundaries(monkeypatch):
    req = _metadata_req("req", [1, 2, 3], scheduled_len=2)
    batch = _metadata_batch([req])
    batch.forward_mode = ForwardMode.EXTEND
    batch.input_ids = torch.tensor([1, 2], dtype=torch.int64)
    batch.req_pool_indices = torch.tensor([0], dtype=torch.int64)
    batch.seq_lens = torch.tensor([2], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([2], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([2], dtype=torch.int32)
    batch.out_cache_loc = torch.tensor([10, 11], dtype=torch.int64)
    batch.seq_lens_sum = 2
    batch.extend_num_tokens = 2
    batch.extend_lens = [2]
    batch.prefix_lens = [0]
    batch.extend_logprob_start_lens = [2]

    worker_batch = batch.get_model_worker_batch()

    assert worker_batch.welm_deferred_prefill is True
    assert "welm_deferred_prefill_final_mask" not in ModelWorkerBatch.__dataclass_fields__

    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_batch_info.compute_position",
        lambda *_args, **_kwargs: (
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
        lambda *_args, **_kwargs: False,
    )
    runner = SimpleNamespace(
        req_to_token_pool=object(),
        token_to_kv_pool=object(),
        token_to_kv_pool_allocator=object(),
        attn_backend=object(),
        device=torch.device("cpu"),
        server_args=SimpleNamespace(
            enable_welm_kv_mirror_opt=True,
            attention_backend="fa3",
            enable_lora=False,
        ),
        use_ngram_embedding=False,
        model_is_mrope=False,
        is_hybrid_swa=False,
        supports_attn_cp_prefill_runtime=False,
    )

    forward_batch = ForwardBatch.init_new(worker_batch, runner)

    assert forward_batch.welm_deferred_prefill is True
    assert "welm_deferred_prefill_final_mask" not in ForwardBatch.__dataclass_fields__
