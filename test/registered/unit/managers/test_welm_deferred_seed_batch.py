from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import overlap_utils
from sglang.srt.managers import schedule_batch as schedule_batch_module
from sglang.srt.managers import scheduler_dp_attn_mixin
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.models.welm_deferred_mirror import WelmPDExecutionMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def test_deferred_seed_state_is_not_duplicated_in_batch_metadata():
    for batch_type in (ScheduleBatch, ModelWorkerBatch, ForwardBatch):
        assert "welm_deferred_seed_mask" not in batch_type.__dataclass_fields__


def _server_args(disaggregation_mode=DisaggregationMode.DECODE):
    return SimpleNamespace(
        welm_kv_mirror_pd_mode=(
            WelmPDExecutionMode.DEFERRED_LAST_PROMPT.value
        ),
        disaggregation_mode=disaggregation_mode.value,
        disaggregation_decode_enable_radix_cache=False,
    )


def _spec_algorithm_none():
    return SimpleNamespace(is_none=lambda: True)


def _seed_req(rid: str, prompt_token_ids):
    state = schedule_batch_module.WelmDeferredDecodeState.from_prompt_tokens(
        list(prompt_token_ids)
    )
    state.transition_to(schedule_batch_module.WelmDeferredDecodePhase.READY)
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=list(prompt_token_ids),
        output_ids=[],
        fill_ids=list(prompt_token_ids[:-1]),
        welm_deferred_decode_state=state,
        req_pool_idx=int(rid.rsplit("-", 1)[-1]),
        kv_allocated_len=state.committed_kv_len,
        kv_committed_len=state.committed_kv_len,
        decode_batch_idx=0,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        stream=False,
        grammar=None,
        multimodal_inputs=None,
        lora_id=None,
        dllm_block_offset=0,
        input_embeds=None,
        attn_cp_prefill_split_spec=None,
        _scale_seq_factor=1,
        is_prefill_only=False,
        time_stats=SimpleNamespace(set_forward_entry_time=MagicMock()),
        finished=lambda: False,
    )


def _normal_req(rid: str, *, finished=False):
    return SimpleNamespace(
        rid=rid,
        output_ids=[91],
        welm_deferred_decode_state=None,
        attn_cp_prefill_split_spec=None,
        finished=lambda: finished,
    )


def _scheduler(
    waiting_queue,
    running_reqs=(),
    *,
    capacity=4,
    disaggregation_mode=DisaggregationMode.DECODE,
):
    running_batch = SimpleNamespace(
        reqs=list(running_reqs),
        batch_size=lambda: len(running_batch.reqs),
    )
    return SimpleNamespace(
        grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
        waiting_queue=list(waiting_queue),
        enable_priority_scheduling=False,
        running_batch=running_batch,
        req_to_token_pool=SimpleNamespace(size=capacity, device="cpu"),
        token_to_kv_pool_allocator=object(),
        tree_cache=object(),
        model_config=SimpleNamespace(is_encoder_decoder=False, vocab_size=128),
        enable_overlap=False,
        spec_algorithm=_spec_algorithm_none(),
        max_running_requests=capacity,
        server_args=_server_args(disaggregation_mode),
        disaggregation_mode=disaggregation_mode,
    )


@pytest.mark.parametrize(
    "disaggregation_mode",
    [DisaggregationMode.DECODE, DisaggregationMode.NULL],
)
def test_seed_only_admission_builds_standard_decode_row_metadata(
    disaggregation_mode,
):
    req = _seed_req("seed-0", [11, 12, 13])
    scheduler = _scheduler([req], disaggregation_mode=disaggregation_mode)
    sampling_info = MagicMock()

    with patch(
        "sglang.srt.managers.schedule_batch."
        "SamplingBatchInfo.from_schedule_batch",
        return_value=sampling_info,
    ):
        batch = Scheduler.get_new_welm_deferred_seed_batch(scheduler)

    assert batch.forward_mode is None
    assert batch.reqs == [req]
    assert batch.output_ids.tolist() == [13]
    assert batch.seq_lens.tolist() == [2]
    assert batch.seq_lens_cpu.tolist() == [2]
    assert batch.orig_seq_lens.tolist() == [2]
    assert batch.seq_lens_sum == 2
    assert batch.req_pool_indices.tolist() == [req.req_pool_idx]
    assert batch.sampling_info is sampling_info
    assert scheduler.waiting_queue == []
    assert (
        req.welm_deferred_decode_state.phase
        is schedule_batch_module.WelmDeferredDecodePhase.READY
    )


def test_overlap_seed_oe_history_starts_before_last_prompt_token():
    req = _seed_req("seed-0", [10, 11, 12, 13])
    req._overlap_decode_count = 0
    batch = _decode_batch(
        [req],
        output_ids=[13],
        seq_lens=[2],
    )
    batch.enable_overlap = True
    server_args = SimpleNamespace(
        prepare_n_gram_inputs=False,
        enable_mamba_extra_buffer=lambda: False,
    )

    with (
        patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.srt.managers.schedule_batch.build_router_replay_decode_batch",
            return_value=(None, None),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            return_value=torch.tensor([37], dtype=torch.int64),
        ),
    ):
        batch.prepare_for_decode()

    context = schedule_batch_module.OverEncodingContext.from_decode_hash_kernel(
        [req],
        enable_overlap=True,
        history_width=3,
    )

    assert context.hash_prefixes == [[12], [11], [10]]
    assert req._overlap_decode_count == 0


def test_seed_decode_preserves_output_logprob_and_sampling_configuration():
    req = _seed_req("seed-0", [11, 12, 13])
    req.return_logprob = True
    req.top_logprobs_num = 2
    req.token_ids_logprob = [7, 9]
    req.sampling_params = SimpleNamespace(
        temperature=0.0,
        seed=1234,
        logit_bias={42: 3.0},
    )
    scheduler = _scheduler([req])
    sampling_info = MagicMock()

    with patch(
        "sglang.srt.managers.schedule_batch."
        "SamplingBatchInfo.from_schedule_batch",
        return_value=sampling_info,
    ) as build_sampling_info:
        batch = Scheduler.get_new_welm_deferred_seed_batch(scheduler)

    assert batch.return_logprob is True
    assert batch.top_logprobs_nums == [2]
    assert batch.token_ids_logprobs == [[7, 9]]
    assert batch.sampling_info is sampling_info
    assert req.sampling_params.temperature == 0.0
    assert req.sampling_params.seed == 1234
    assert req.sampling_params.logit_bias == {42: 3.0}
    build_sampling_info.assert_called_once_with(batch, 128)


def test_seed_admission_waits_for_capacity_then_uses_freed_slot():
    req = _seed_req("seed-0", [11, 12, 13])
    normal = _normal_req("normal", finished=False)
    scheduler = _scheduler([req], [normal], capacity=1)

    assert Scheduler.get_new_welm_deferred_seed_batch(scheduler) is None
    assert scheduler.waiting_queue == [req]

    normal.finished = lambda: True
    with patch(
        "sglang.srt.managers.schedule_batch."
        "SamplingBatchInfo.from_schedule_batch",
        return_value=MagicMock(),
    ):
        batch = Scheduler.get_new_welm_deferred_seed_batch(scheduler)

    assert batch.reqs == [req]
    assert scheduler.waiting_queue == []


def test_seed_admission_fills_all_available_decode_rows():
    seeds = [
        _seed_req("seed-0", [1]),
        _seed_req("seed-1", [2, 3]),
        _seed_req("seed-2", [4, 5, 6]),
    ]
    scheduler = _scheduler(seeds, [_normal_req("normal")], capacity=3)

    with patch(
        "sglang.srt.managers.schedule_batch."
        "SamplingBatchInfo.from_schedule_batch",
        return_value=MagicMock(),
    ):
        batch = Scheduler.get_new_welm_deferred_seed_batch(scheduler)

    assert [req.rid for req in batch.reqs] == ["seed-0", "seed-1"]
    assert batch.output_ids.tolist() == [1, 3]
    assert batch.seq_lens.tolist() == [0, 1]
    assert [req.rid for req in scheduler.waiting_queue] == ["seed-2"]


def test_monolithic_seed_selector_ignores_ordinary_and_pending_prefill_rows():
    ordinary = _normal_req("ordinary")
    pending = _seed_req("seed-1", [11, 12])
    pending.welm_deferred_decode_state.phase = (
        schedule_batch_module.WelmDeferredDecodePhase.PREFILL_PENDING
    )
    ready = _seed_req("seed-2", [21, 22, 23])
    scheduler = _scheduler(
        [ordinary, pending, ready],
        disaggregation_mode=DisaggregationMode.NULL,
    )

    with patch(
        "sglang.srt.managers.schedule_batch."
        "SamplingBatchInfo.from_schedule_batch",
        return_value=MagicMock(),
    ):
        batch = Scheduler.get_new_welm_deferred_seed_batch(scheduler)

    assert batch.reqs == [ready]
    assert scheduler.waiting_queue == [ordinary, pending]


def test_seed_admission_applies_priority_before_capacity_limit():
    first = _seed_req("seed-0", [1])
    second = _seed_req("seed-1", [2])
    scheduler = _scheduler([first, second], capacity=1)
    scheduler.enable_priority_scheduling = True
    scheduler.policy = SimpleNamespace(
        calc_priority=lambda waiting, _running: waiting.reverse()
    )

    with patch(
        "sglang.srt.managers.schedule_batch."
        "SamplingBatchInfo.from_schedule_batch",
        return_value=MagicMock(),
    ):
        batch = Scheduler.get_new_welm_deferred_seed_batch(scheduler)

    assert batch.reqs == [second]
    assert scheduler.waiting_queue == [first]


def test_disagg_decode_scheduler_runs_seed_only_batch_immediately():
    req = _seed_req("seed-0", [11, 12, 13])
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = _server_args()
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.server_args.enable_welm_kv_mirror_opt = True
    scheduler.spec_algorithm = SimpleNamespace(
        is_none=lambda: True,
        is_eagle=lambda: False,
    )
    scheduler.running_batch = ScheduleBatch(reqs=[])
    scheduler.waiting_queue = [req]
    scheduler.grammar_manager = SimpleNamespace(
        has_waiting_grammars=lambda: False
    )
    scheduler.enable_priority_scheduling = False
    scheduler.req_to_token_pool = SimpleNamespace(size=4, device="cpu")
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.tree_cache = object()
    scheduler.model_config = SimpleNamespace(
        is_encoder_decoder=False,
        vocab_size=128,
    )
    scheduler.enable_overlap = False
    scheduler.max_running_requests = 4
    scheduler.chunked_req = None
    scheduler.maybe_prepare_mlp_sync_batch = lambda batch: batch

    def prepare_running(batch):
        batch.prepare_for_decode()
        return batch

    scheduler.update_running_batch = prepare_running
    global_args = SimpleNamespace(
        prepare_n_gram_inputs=False,
        enable_mamba_extra_buffer=lambda: False,
    )

    with (
        patch(
            "sglang.srt.managers.schedule_batch."
            "SamplingBatchInfo.from_schedule_batch",
            return_value=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=False)
            ),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=global_args,
        ),
        patch(
            "sglang.srt.managers.schedule_batch.build_router_replay_decode_batch",
            return_value=(None, None),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            return_value=torch.tensor([37], dtype=torch.int64),
        ),
        patch("sglang.srt.disaggregation.decode.set_schedule_time_batch"),
    ):
        batch = scheduler.get_next_disagg_decode_batch_to_run()

    assert batch.forward_mode is ForwardMode.DECODE
    assert batch.input_ids.tolist() == [13]
    assert scheduler.waiting_queue == []


def test_disagg_decode_scheduler_mixes_seed_with_running_decode_rows():
    normal = _normal_req("normal")
    normal.req_pool_idx = 7
    normal.kv_allocated_len = 5
    normal.kv_committed_len = 5
    normal.decode_batch_idx = 0
    seed = _seed_req("seed-1", [11, 12, 13])

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = _server_args()
    scheduler.disaggregation_mode = DisaggregationMode.DECODE
    scheduler.server_args.enable_welm_kv_mirror_opt = True
    scheduler.spec_algorithm = SimpleNamespace(
        is_none=lambda: True,
        is_eagle=lambda: False,
    )
    scheduler.running_batch = _decode_batch(
        [normal], output_ids=[91], seq_lens=[5]
    )
    scheduler.waiting_queue = [seed]
    scheduler.grammar_manager = SimpleNamespace(
        has_waiting_grammars=lambda: False
    )
    scheduler.enable_priority_scheduling = False
    scheduler.req_to_token_pool = SimpleNamespace(size=4, device="cpu")
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.tree_cache = object()
    scheduler.model_config = SimpleNamespace(
        is_encoder_decoder=False,
        vocab_size=128,
    )
    scheduler.enable_overlap = False
    scheduler.max_running_requests = 4
    scheduler.chunked_req = None
    scheduler.maybe_prepare_mlp_sync_batch = lambda batch: batch
    scheduler.update_running_batch = lambda batch: (
        batch.prepare_for_decode() or batch
    )
    global_args = SimpleNamespace(
        prepare_n_gram_inputs=False,
        enable_mamba_extra_buffer=lambda: False,
    )

    with (
        patch(
            "sglang.srt.managers.schedule_batch."
            "SamplingBatchInfo.from_schedule_batch",
            return_value=SimpleNamespace(
                penalizer_orchestrator=SimpleNamespace(is_required=False)
            ),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=global_args,
        ),
        patch(
            "sglang.srt.managers.schedule_batch.build_router_replay_decode_batch",
            return_value=(None, None),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            return_value=torch.tensor([51, 52], dtype=torch.int64),
        ),
        patch("sglang.srt.disaggregation.decode.set_schedule_time_batch"),
    ):
        batch = scheduler.get_next_disagg_decode_batch_to_run()

    assert batch.forward_mode is ForwardMode.DECODE
    assert batch.input_ids.tolist() == [91, 13]
    assert batch.seq_lens.tolist() == [6, 3]
    assert normal.kv_committed_len == 6
    assert seed.kv_committed_len == 3
    assert (
        seed.welm_deferred_decode_state.phase
        is schedule_batch_module.WelmDeferredDecodePhase.INFLIGHT
    )


@pytest.mark.parametrize(
    ("prompt_token_ids", "committed_len", "seed_token_id"),
    [([41], 0, 41), ([11, 12, 13], 2, 13)],
)
def test_prepare_for_decode_runs_seed_through_normal_decode_allocation(
    prompt_token_ids, committed_len, seed_token_id
):
    req = _seed_req("seed-0", prompt_token_ids)
    batch = ScheduleBatch(
        reqs=[req],
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        tree_cache=object(),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        spec_algorithm=_spec_algorithm_none(),
        device="cpu",
        output_ids=torch.tensor([seed_token_id], dtype=torch.int64),
        req_pool_indices=torch.tensor([req.req_pool_idx], dtype=torch.int64),
        seq_lens=torch.tensor([committed_len], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([committed_len], dtype=torch.int64),
        orig_seq_lens=torch.tensor([committed_len], dtype=torch.int32),
        seq_lens_sum=committed_len,
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=False)
        ),
    )
    server_args = SimpleNamespace(
        prepare_n_gram_inputs=False,
        enable_mamba_extra_buffer=lambda: False,
    )

    with (
        patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.srt.managers.schedule_batch.build_router_replay_decode_batch",
            return_value=(None, None),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            return_value=torch.tensor([37], dtype=torch.int64),
        ),
    ):
        batch.prepare_for_decode()

    assert batch.forward_mode is ForwardMode.DECODE
    assert batch.input_ids.tolist() == [seed_token_id]
    assert batch.seq_lens.tolist() == [committed_len + 1]
    assert batch.seq_lens_cpu.tolist() == [committed_len + 1]
    assert batch.orig_seq_lens.tolist() == [committed_len + 1]
    assert batch.seq_lens_sum == committed_len + 1
    assert batch.out_cache_loc.tolist() == [37]
    assert req.kv_allocated_len == committed_len + 1
    assert req.kv_committed_len == committed_len + 1
    assert req.decode_batch_idx == 1
    assert (
        req.welm_deferred_decode_state.phase
        is schedule_batch_module.WelmDeferredDecodePhase.INFLIGHT
    )


def test_prefill_batch_restores_seed_logprob_metadata_before_decode():
    req = _seed_req("seed-0", [11, 12, 13])
    req.return_logprob = True
    req.top_logprobs_num = 2
    req.token_ids_logprob = [7, 9]
    batch = ScheduleBatch(
        reqs=[req],
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        tree_cache=object(),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        spec_algorithm=_spec_algorithm_none(),
        device="cpu",
        output_ids=torch.tensor([13], dtype=torch.int64),
        req_pool_indices=torch.tensor([req.req_pool_idx], dtype=torch.int64),
        seq_lens=torch.tensor([2], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([2], dtype=torch.int64),
        orig_seq_lens=torch.tensor([2], dtype=torch.int32),
        seq_lens_sum=2,
        return_logprob=False,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=False)
        ),
    )
    server_args = SimpleNamespace(
        prepare_n_gram_inputs=False,
        enable_mamba_extra_buffer=lambda: False,
    )

    with (
        patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.srt.managers.schedule_batch.build_router_replay_decode_batch",
            return_value=(None, None),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            return_value=torch.tensor([37], dtype=torch.int64),
        ),
    ):
        batch.prepare_for_decode()

    assert batch.return_logprob is True
    assert batch.top_logprobs_nums == [2]
    assert batch.token_ids_logprobs == [[7, 9]]


def test_prepare_for_decode_rejects_prefill_pending_seed_row():
    req = _seed_req("seed-0", [11, 12, 13])
    req.welm_deferred_decode_state.phase = (
        schedule_batch_module.WelmDeferredDecodePhase.PREFILL_PENDING
    )
    batch = ScheduleBatch(
        reqs=[req],
        model_config=SimpleNamespace(is_encoder_decoder=False),
        spec_algorithm=_spec_algorithm_none(),
        output_ids=torch.tensor([13]),
        seq_lens=torch.tensor([2]),
        seq_lens_cpu=torch.tensor([2]),
        orig_seq_lens=torch.tensor([2]),
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=False)
        ),
    )

    with pytest.raises(RuntimeError, match="before Prefill completion"):
        batch.prepare_for_decode()


def _decode_batch(reqs, *, output_ids, seq_lens):
    sampling_info = MagicMock()
    sampling_info.penalizer_orchestrator.is_required = False
    return ScheduleBatch(
        reqs=list(reqs),
        model_config=SimpleNamespace(is_encoder_decoder=False, vocab_size=128),
        spec_algorithm=_spec_algorithm_none(),
        device="cpu",
        forward_mode=ForwardMode.DECODE,
        input_ids=torch.tensor(output_ids, dtype=torch.int64),
        output_ids=torch.tensor(output_ids, dtype=torch.int64),
        req_pool_indices=torch.arange(len(reqs), dtype=torch.int64),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        orig_seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        out_cache_loc=torch.arange(100, 100 + len(reqs), dtype=torch.int64),
        seq_lens_sum=sum(seq_lens),
        sampling_info=sampling_info,
    )


def test_deferred_seed_remains_dp_cuda_graph_eligible():
    req = _seed_req("seed-0", [11, 12, 13])
    batch = _decode_batch(
        [req], output_ids=[13], seq_lens=[3]
    )
    info = SimpleNamespace(
        num_tokens=1,
        num_tokens_for_logprob=1,
        is_extend_in_batch=False,
        tbo_split_seq_index=None,
        global_forward_mode=ForwardMode.DECODE,
        global_forward_modes=[ForwardMode.DECODE.value],
        global_num_reqs=[1],
        has_cache_hit_extend=False,
        welm_kv_mirror_contract_flags=[False],
        welm_deferred_prefill_flags=[False],
        welm_mtp_global_prefill_num_tokens=[0],
        global_has_non_greedy_sampling=False,
        global_needs_top_p_sampling=False,
        can_cuda_graph=True,
        global_num_tokens=[1],
        global_num_tokens_for_logprob=[1],
        has_router_replay=False,
    )

    scheduler_dp_attn_mixin._update_gather_batch(
        batch,
        info,
        require_mlp_tp_gather=False,
    )

    assert batch.can_run_dp_cuda_graph
    assert batch.welm_deferred_prefill_flags == [False]


def test_overlap_resolution_leaves_non_negative_seed_token_untouched():
    input_ids = torch.tensor([13, -1], dtype=torch.int64)
    future_token_ids = torch.tensor([0, 77], dtype=torch.int64)

    overlap_utils._resolve_future_token_ids_native(input_ids, future_token_ids)

    assert input_ids.tolist() == [13, 77]


def test_overlap_mixed_batch_advances_seed_lifecycle_across_two_iterations():
    normal = _normal_req("normal")
    normal.req_pool_idx = 7
    normal.kv_allocated_len = 5
    normal.kv_committed_len = 5
    normal.decode_batch_idx = 0
    seed = _seed_req("seed-1", [11, 12, 13])
    seed.finished_reason = None
    seed.to_finish = None
    seed.sampling_params = SimpleNamespace(max_new_tokens=4)
    seed._check_token_based_finish = lambda _: False
    seed._check_vocab_boundary_finish = lambda _: False
    seed._check_str_based_finish = lambda: False

    batch = _decode_batch(
        [normal, seed], output_ids=[91, 13], seq_lens=[5, 2]
    )
    batch.enable_overlap = True
    server_args = SimpleNamespace(
        prepare_n_gram_inputs=False,
        enable_mamba_extra_buffer=lambda: False,
    )

    with (
        patch(
            "sglang.srt.managers.schedule_batch.get_global_server_args",
            return_value=server_args,
        ),
        patch(
            "sglang.srt.managers.schedule_batch.build_router_replay_decode_batch",
            return_value=(None, None),
        ),
        patch(
            "sglang.srt.managers.schedule_batch.alloc_for_decode",
            side_effect=[
                torch.tensor([51, 52], dtype=torch.int64),
                torch.tensor([53, 54], dtype=torch.int64),
            ],
        ),
    ):
        batch.prepare_for_decode()
        first_input_ids = batch.input_ids.clone()

        seed.output_ids = [42]
        schedule_batch_module.Req.check_finished(seed)

        batch.output_ids = torch.tensor([-1, -2], dtype=torch.int64)
        batch.prepare_for_decode()

    overlap_utils._resolve_future_token_ids_native(
        batch.input_ids,
        torch.tensor([0, 92, 42], dtype=torch.int64),
    )

    assert first_input_ids.tolist() == [91, 13]
    assert batch.input_ids.tolist() == [92, 42]
    assert (
        seed.welm_deferred_decode_state.phase
        is schedule_batch_module.WelmDeferredDecodePhase.CONSUMED
    )


def test_first_generated_token_marks_deferred_seed_consumed():
    req = schedule_batch_module.Req.__new__(schedule_batch_module.Req)
    req.welm_deferred_decode_state = _seed_req(
        "seed-0", [11, 12, 13]
    ).welm_deferred_decode_state
    req.welm_deferred_decode_state.transition_to(
        schedule_batch_module.WelmDeferredDecodePhase.INFLIGHT
    )
    req.output_ids = [42]
    req.finished_reason = None
    req.to_finish = None
    req.sampling_params = SimpleNamespace(max_new_tokens=4)
    req.grammar = None
    req._check_token_based_finish = lambda _: False
    req._check_vocab_boundary_finish = lambda _: False
    req._check_str_based_finish = lambda: False

    req.check_finished()

    assert (
        req.welm_deferred_decode_state.phase
        is schedule_batch_module.WelmDeferredDecodePhase.CONSUMED
    )
