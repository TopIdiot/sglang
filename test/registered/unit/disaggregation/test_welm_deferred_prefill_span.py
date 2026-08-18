from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import sglang.srt.disaggregation.prefill as prefill_module
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.welm_deferred_protocol import (
    WelmDeferredCompletion,
)
from sglang.srt.disaggregation.fake.conn import FakeKVSender
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    WelmDeferredDecodePhase,
    WelmDeferredDecodeState,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.models.welm_deferred_mirror import (
    WelmDeferredPrefillSpan,
    build_welm_deferred_prefill_span,
    prepare_welm_deferred_prefill_span,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_req(token_ids, *, deferred: bool) -> Req:
    req = Req.__new__(Req)
    req.rid = "req"
    req.dllm_config = None
    req.origin_input_ids = list(token_ids)
    req.output_ids = []
    req.fill_ids = []
    req.prefix_indices = torch.empty((0,), dtype=torch.int64)
    req.attn_cp_prefill_split_spec = None
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.top_logprobs_num = 0
    req.token_ids_logprob = None
    req.return_hidden_states = False
    req.return_routed_experts = False
    req.return_indexer_topk = False
    req.input_embeds = None
    req.stream = False
    req.grammar = None
    req.sampling_params = SimpleNamespace(max_new_tokens=8)
    req.positional_embed_overrides = None
    req.extra_key = None
    req.is_retracted = False
    req.retracted_stain = False
    req.multimodal_inputs = None
    req._scale_seq_factor = 1
    req.welm_deferred_prefill_span = (
        build_welm_deferred_prefill_span(req.origin_input_ids) if deferred else None
    )
    return req


def _tree_cache(prefix_len: int):
    node = object()
    match_result = SimpleNamespace(
        device_indices=torch.arange(prefix_len, dtype=torch.int64),
        last_device_node=node,
        last_host_node=node,
        best_match_node=node,
        host_hit_length=0,
        mamba_branching_seqlen=None,
        cache_protected_len=prefix_len,
    )
    return SimpleNamespace(
        supports_mamba=lambda: False,
        scale_seq_factor=1,
        match_prefix=MagicMock(return_value=match_result),
    )


def test_prefill_span_commits_prompt_without_last_token():
    span = build_welm_deferred_prefill_span([11, 22, 33, 44])

    assert span == WelmDeferredPrefillSpan(
        prompt_len=4,
        committed_kv_len=3,
        seed_position=3,
        seed_token_id=44,
    )


def test_prefill_span_supports_single_token_zero_page_completion():
    span = build_welm_deferred_prefill_span([77])

    assert span.committed_kv_len == 0
    assert span.seed_position == 0
    assert span.seed_token_id == 77


def test_prefill_span_rejects_empty_prompt():
    with pytest.raises(ValueError, match="empty prompt"):
        build_welm_deferred_prefill_span([])


def test_req_preserves_origin_but_exposes_only_committed_prefill_tokens():
    req = _make_req([11, 22, 33, 44], deferred=True)

    assert req.origin_input_ids == [11, 22, 33, 44]
    assert req.prefill_kv_token_ids() == [11, 22, 33]
    assert req.prefill_kv_len() == 3


def test_consumed_deferred_request_restores_ordinary_reprefill_tokens():
    req = _make_req([11, 22, 33, 44], deferred=True)
    state = WelmDeferredDecodeState.from_prefill_span(
        req.welm_deferred_prefill_span
    )
    state.transition_to(WelmDeferredDecodePhase.READY)
    state.transition_to(WelmDeferredDecodePhase.INFLIGHT)
    state.transition_to(WelmDeferredDecodePhase.CONSUMED)
    req.welm_deferred_decode_state = state
    req.output_ids = [55, 66]

    assert req.prefill_kv_token_ids() == [11, 22, 33, 44, 55, 66]
    assert req.prefill_kv_len() == 6
    assert req._compute_max_prefix_len(6) == 5


def test_bootstrap_prepares_deferred_span_before_capacity_check():
    queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
    queue.kv_manager = SimpleNamespace(welm_deferred_mirror_capability=object())
    queue.scheduler = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_none=lambda: True)
    )
    queue.max_total_num_tokens = 3
    req = _make_req([11, 22, 33, 44], deferred=False)

    assert queue._prepare_deferred_req(req)

    assert req.welm_deferred_prefill_span.committed_kv_len == 3
    assert vars(req).get("welm_deferred_decode_state") is None
    assert queue._check_if_req_exceed_kv_capacity(req) is False


def test_bootstrap_rejects_single_token_deferred_mtp_prompt(monkeypatch):
    queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
    queue.kv_manager = SimpleNamespace(welm_deferred_mirror_capability=object())
    queue.scheduler = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        stream_output=MagicMock(),
    )
    req = _make_req([77], deferred=False)
    req.time_stats = SimpleNamespace(
        trace_ctx=SimpleNamespace(abort=MagicMock())
    )
    prepare_abort = MagicMock()
    monkeypatch.setattr(prefill_module, "prepare_abort", prepare_abort)

    assert queue._prepare_deferred_req(req) is False

    prepare_abort.assert_called_once()
    assert "at least two prompt tokens" in prepare_abort.call_args.args[1]
    queue.scheduler.stream_output.assert_called_once_with([req], req.return_logprob)


def test_bootstrap_allows_single_token_deferred_without_mtp():
    queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
    queue.kv_manager = SimpleNamespace(welm_deferred_mirror_capability=object())
    queue.scheduler = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_none=lambda: True)
    )
    req = _make_req([77], deferred=False)

    assert queue._prepare_deferred_req(req)
    assert req.welm_deferred_prefill_span.committed_kv_len == 0


def test_topology_neutral_span_preparation_rejects_changed_prompt():
    req = _make_req([11, 22, 33, 44], deferred=False)
    prepare_welm_deferred_prefill_span(req)
    req.origin_input_ids[-1] = 55

    with pytest.raises(RuntimeError, match="span changed"):
        prepare_welm_deferred_prefill_span(req)


def test_deferred_output_logprob_is_preserved_but_not_requested_from_prefill():
    req = _make_req([11, 22, 33, 44], deferred=True)
    req.return_logprob = True
    req.logprob_start_len = len(req.origin_input_ids)

    with patch(
        "sglang.srt.managers.schedule_batch.get_global_server_args",
        return_value=SimpleNamespace(speculative_algorithm=None),
    ):
        batch = ScheduleBatch.init_new(
            [req],
            SimpleNamespace(device="cpu"),
            object(),
            object(),
            SimpleNamespace(),
            enable_overlap=False,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )

    assert req.return_logprob is True
    assert Req.requires_prompt_logprobs(req) is False
    assert req.requires_logprob_payload() is False
    assert batch.return_logprob is False


def test_non_deferred_output_logprob_keeps_existing_prefill_payload_semantics():
    req = _make_req([11, 22, 33, 44], deferred=False)
    req.return_logprob = True
    req.logprob_start_len = len(req.origin_input_ids)

    with patch(
        "sglang.srt.managers.schedule_batch.get_global_server_args",
        return_value=SimpleNamespace(speculative_algorithm=None),
    ):
        batch = ScheduleBatch.init_new(
            [req],
            SimpleNamespace(device="cpu"),
            object(),
            object(),
            SimpleNamespace(),
            enable_overlap=False,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )

    assert req.requires_logprob_payload() is True
    assert batch.return_logprob is True


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"sampling_params": SimpleNamespace(max_new_tokens=0)}, "zero-generation"),
        (
            {"return_logprob": True, "logprob_start_len": 0},
            "prompt logprobs",
        ),
        ({"return_hidden_states": True}, "hidden states"),
        ({"return_routed_experts": True}, "routed experts"),
        ({"return_indexer_topk": True}, "indexer top-k"),
        ({"input_embeds": [[0.0]]}, "input embeddings"),
        ({"positional_embed_overrides": object()}, "positional embedding overrides"),
        ({"multimodal_inputs": object()}, "multimodal inputs"),
    ],
)
def test_deferred_prefill_rejects_unsupported_request_payloads(
    updates, message, monkeypatch
):
    queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
    queue.kv_manager = SimpleNamespace(welm_deferred_mirror_capability=object())
    queue.scheduler = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        stream_output=MagicMock(),
    )
    req = _make_req([11, 22, 33, 44], deferred=False)
    req.time_stats = SimpleNamespace(
        trace_ctx=SimpleNamespace(abort=MagicMock())
    )
    for name, value in updates.items():
        setattr(req, name, value)
    prepare_abort = MagicMock()
    monkeypatch.setattr(prefill_module, "prepare_abort", prepare_abort)

    assert queue._prepare_deferred_req(req) is False

    prepare_abort.assert_called_once()
    assert message in prepare_abort.call_args.args[1]
    queue.scheduler.stream_output.assert_called_once_with([req], req.return_logprob)
    assert req.welm_deferred_prefill_span is None


def test_deferred_rejection_stops_before_sender_or_capacity_work(monkeypatch):
    queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
    queue.kv_manager = SimpleNamespace(welm_deferred_mirror_capability=object())
    queue.scheduler = SimpleNamespace(
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        stream_output=MagicMock(),
    )
    queue._check_if_req_exceed_kv_capacity = MagicMock()
    queue._process_req = MagicMock()
    queue.queue = []
    req = _make_req([11, 22, 33, 44], deferred=False)
    req.sampling_params.max_new_tokens = 0
    req.time_stats = SimpleNamespace(
        trace_ctx=SimpleNamespace(abort=MagicMock())
    )
    prepare_abort = MagicMock()
    get_kv_class = MagicMock()
    monkeypatch.setattr(prefill_module, "prepare_abort", prepare_abort)
    monkeypatch.setattr(prefill_module, "get_kv_class", get_kv_class)

    queue.add(req, num_kv_heads=2)

    prepare_abort.assert_called_once()
    queue._check_if_req_exceed_kv_capacity.assert_not_called()
    queue._process_req.assert_not_called()
    get_kv_class.assert_not_called()
    assert queue.queue == []


def test_fake_sender_sends_last_zero_page_completion():
    sender = FakeKVSender.__new__(FakeKVSender)

    assert sender.should_send_kv_chunk(num_pages=0, last_chunk=True)
    assert not sender.should_send_kv_chunk(num_pages=0, last_chunk=False)


def test_deferred_req_allows_full_committed_prefix_hit():
    req = _make_req([11, 22, 33, 44], deferred=True)
    tree_cache = _tree_cache(prefix_len=3)

    req.init_next_round_input(tree_cache)

    match_params = tree_cache.match_prefix.call_args.args[0]
    assert match_params.key.token_ids == [11, 22, 33]
    assert req.origin_input_ids == [11, 22, 33, 44]
    assert req.fill_ids == [11, 22, 33]
    assert req.extend_input_len == 0
    assert len(req.prefix_indices) == 3


def test_deferred_req_matches_split_radix_nodes_and_extends_committed_suffix():
    tree_cache = RadixCache.create_simulated()
    tree_cache.insert(
        InsertParams(
            key=RadixKey(token_ids=[11, 22, 99]),
            value=torch.tensor([101, 205, 999], dtype=torch.int64),
        )
    )
    tree_cache.insert(
        InsertParams(
            key=RadixKey(token_ids=[11, 22, 33]),
            value=torch.tensor([101, 205, 309], dtype=torch.int64),
        )
    )
    req = _make_req([11, 22, 33, 44, 55], deferred=True)

    req.init_next_round_input(tree_cache)

    assert req.origin_input_ids == [11, 22, 33, 44, 55]
    assert req.fill_ids == [11, 22, 33, 44]
    assert req.prefix_indices.tolist() == [101, 205, 309]
    assert req.extend_input_len == 1


def test_legacy_req_still_reserves_last_input_token_for_forward():
    req = _make_req([11, 22, 33, 44], deferred=False)
    tree_cache = _tree_cache(prefix_len=3)

    req.init_next_round_input(tree_cache)

    match_params = tree_cache.match_prefix.call_args.args[0]
    assert match_params.key.token_ids == [11, 22, 33]
    assert req.fill_ids == [11, 22, 33, 44]
    assert req.extend_input_len == 1


def _bootstrap_req(token_ids, *, decode_prefix_len: int):
    req = SimpleNamespace(
        rid="req",
        origin_input_ids=list(token_ids),
        output_ids=[],
        welm_deferred_prefill_span=build_welm_deferred_prefill_span(token_ids),
        disagg_kv_sender=MagicMock(),
        time_stats=MagicMock(),
        metadata_buffer_index=None,
    )
    req.disagg_kv_sender.pop_decode_prefix_len.return_value = decode_prefix_len
    return req


def _bootstrap_queue(req):
    queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
    queue.queue = [req]
    queue.scheduler = SimpleNamespace(
        attn_cp_cpu_group=object(),
        attn_tp_cpu_group=object(),
        enable_metrics=False,
    )
    queue.tp_rank = 0
    queue.token_to_kv_pool = SimpleNamespace(page_size=16)
    queue.req_to_metadata_buffer_idx_allocator = MagicMock()
    queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1
    queue.req_to_metadata_buffer_idx_allocator.alloc.return_value = 5
    return queue


def test_bootstrap_uses_committed_length_and_accepts_zero_page_full_hit():
    req = _bootstrap_req(range(17), decode_prefix_len=16)
    queue = _bootstrap_queue(req)

    with patch.object(
        prefill_module,
        "poll_and_all_reduce_attn_cp_tp_group",
        return_value=[KVPoll.WaitingForInput],
    ):
        ready = queue.pop_bootstrapped()

    assert ready == [req]
    assert req.start_send_idx == 16
    req.disagg_kv_sender.init.assert_called_once_with(0, 5)


def test_bootstrap_rejects_decode_prefix_beyond_committed_length():
    req = _bootstrap_req([11, 22, 33, 44], decode_prefix_len=4)
    queue = _bootstrap_queue(req)

    with (
        patch.object(
            prefill_module,
            "poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.WaitingForInput],
        ),
        pytest.raises(RuntimeError, match="decode_prefix_len.*committed_kv_len"),
    ):
        queue.pop_bootstrapped()

    queue.req_to_metadata_buffer_idx_allocator.alloc.assert_not_called()
    req.disagg_kv_sender.init.assert_not_called()


def test_zero_forward_completion_enters_transfer_without_output_token():
    token_ids = [11, 22, 33, 44]
    req = SimpleNamespace(
        origin_input_ids=token_ids,
        output_ids=[],
        extend_input_len=0,
        welm_deferred_prefill_span=build_welm_deferred_prefill_span(token_ids),
        time_stats=MagicMock(),
    )
    scheduler = SimpleNamespace(
        disaggregation_mode=DisaggregationMode.PREFILL,
        disagg_prefill_inflight_queue=[],
        send_kv_chunk=MagicMock(),
    )

    Scheduler.process_deferred_prefill_without_forward(scheduler, [req])

    assert req.output_ids == []
    scheduler.send_kv_chunk.assert_called_once_with(req, last_chunk=True)
    assert scheduler.disagg_prefill_inflight_queue == [req]
    req.time_stats.set_prefill_finished_time.assert_called_once_with()
    req.time_stats.set_prefill_transfer_queue_entry_time.assert_called_once_with()


def _send_fixture(*, deferred: bool):
    token_ids = [11, 22, 33, 44, 55, 66]
    req = SimpleNamespace(
        rid="req",
        origin_input_ids=token_ids,
        output_ids=[] if deferred else [99],
        fill_ids=token_ids[:-1] if deferred else token_ids,
        welm_deferred_prefill_span=(
            build_welm_deferred_prefill_span(token_ids) if deferred else None
        ),
        start_send_idx=0,
        req_pool_idx=0,
        metadata_buffer_index=3,
        bootstrap_room=1234,
        cached_tokens=5,
        cached_tokens_device=3,
        cached_tokens_host=1,
        cached_tokens_storage=1,
        disagg_kv_sender=MagicMock(),
    )
    req.disagg_kv_sender.should_send_kv_chunk.return_value = True
    req_to_token = torch.tensor(
        [[16, 17, 18, 19, 40, 41, 42, 43]], dtype=torch.int32
    )
    metadata = MagicMock()
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(page_size=4),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        disagg_metadata_buffers=metadata,
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            kv_manager=SimpleNamespace(
                kv_args=SimpleNamespace(state_types=[]),
            )
        ),
    )
    return scheduler, req, metadata


def test_last_chunk_transfers_only_committed_pages_and_typed_completion():
    scheduler, req, metadata = _send_fixture(deferred=True)

    prefill_module.SchedulerDisaggregationPrefillMixin.send_kv_chunk(
        scheduler, req, last_chunk=True
    )

    sent_pages, sent_state = req.disagg_kv_sender.send.call_args.args
    np.testing.assert_array_equal(sent_pages, np.array([4, 10], dtype=np.int32))
    assert sent_state == []
    assert req.start_send_idx == 5
    metadata.set_buf.assert_not_called()
    metadata.set_cached_token_stats.assert_called_once_with(req)
    metadata.set_bootstrap_room.assert_called_once_with(3, 1234)
    metadata.set_welm_deferred_completion.assert_called_once_with(
        3,
        WelmDeferredCompletion(
            committed_kv_len=5,
            seed_position=5,
            seed_token_id=66,
        ),
    )


def test_legacy_last_chunk_keeps_existing_metadata_path():
    scheduler, req, metadata = _send_fixture(deferred=False)

    prefill_module.SchedulerDisaggregationPrefillMixin.send_kv_chunk(
        scheduler, req, last_chunk=True
    )

    metadata.set_buf.assert_called_once_with(req)
    metadata.set_welm_deferred_completion.assert_not_called()
