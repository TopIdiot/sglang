from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.context_parallel.prefill_layout import (
    build_cp_prefill_split_spec,
)
from sglang.srt.managers import schedule_policy, scheduler as scheduler_module
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    WelmDeferredDecodePhase,
    WelmDeferredDecodeState,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    CacheAwarePolicy,
    PrefillAdder,
    SchedulePolicy,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.mem_cache.cp_sharded_residency import CPLogicalOwnerPlan
from sglang.srt.models.welm_deferred_mirror import build_welm_deferred_prefill_span


def _make_prefill_adder(*, nsa_in_seq_cp: bool) -> PrefillAdder:
    tree_cache = SimpleNamespace(
        disable=True,
        supports_mamba=lambda: False,
    )
    running_batch = SimpleNamespace(reqs=[])

    with patch.object(
        schedule_policy,
        "is_nsa_prefill_cp_in_seq_split",
        return_value=nsa_in_seq_cp,
    ):
        return PrefillAdder(
            page_size=1,
            tree_cache=tree_cache,
            token_to_kv_pool_allocator=object(),
            running_batch=running_batch,
            new_token_ratio=1.0,
            rem_input_tokens=4096,
            rem_chunk_tokens=1024,
        )


def test_nsa_in_seq_cp_keeps_single_request_prefill_limit():
    adder = _make_prefill_adder(nsa_in_seq_cp=True)
    adder.can_run_list.append(object())
    req = SimpleNamespace(sampling_params=SimpleNamespace(ignore_eos=True))

    with patch.object(
        PrefillAdder, "add_one_req_ignore_eos", return_value=AddReqResult.CONTINUE
    ) as add_ignore_eos:
        assert adder.add_one_req(req, False, None) == AddReqResult.OTHER
        add_ignore_eos.assert_not_called()


def test_non_cp_prefill_does_not_force_single_request_limit():
    # The helper uses a plain allocator, so only NSA in-seq CP should impose
    # its single-request limit here.
    adder = _make_prefill_adder(nsa_in_seq_cp=False)
    adder.can_run_list.append(object())
    req = SimpleNamespace(sampling_params=SimpleNamespace(ignore_eos=True))

    with patch.object(
        PrefillAdder, "add_one_req_ignore_eos", return_value=AddReqResult.CONTINUE
    ) as add_ignore_eos:
        assert adder.add_one_req(req, False, None) == AddReqResult.CONTINUE
        add_ignore_eos.assert_called_once_with(req)


def _split_spec(start: int, length: int, rotation: int = 0):
    return build_cp_prefill_split_spec(
        extend_start=start,
        extend_len=length,
        cp_size=2,
        page_size=4,
        owner_rotation=rotation,
    )


def _req_with_spec(rid: str, spec):
    return SimpleNamespace(
        rid=rid,
        attn_cp_prefill_split_spec=spec,
        return_logprob=False,
        stream=False,
        grammar=None,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        is_prefill_only=False,
        lora_id=None,
        dllm_block_offset=0,
        origin_input_ids=[1, 2, 3, 4],
        output_ids=[],
        fill_ids=[1, 2, 3, 4],
    )


def test_req_interval_rebuild_clears_stale_split_spec():
    req = Req.__new__(Req)
    req.attn_cp_prefill_split_spec = _split_spec(0, 4)
    req.prefix_indices = torch.arange(4)
    req.fill_ids = [1, 2, 3, 4, 5, 6]
    req.logprob_start_len = -1
    req._scale_seq_factor = 1
    req.retracted_stain = False

    req.set_extend_input_len(2)

    assert req.attn_cp_prefill_split_spec is None


def _make_sharded_prefill_ordering_fixture(
    *,
    model_is_encoder_decoder=False,
    enable_hierarchical_cache=False,
    enable_hicache_storage=False,
    schedule_policy_name=None,
):
    match_result = SimpleNamespace(
        device_indices=torch.empty((0,), dtype=torch.int64),
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
        host_hit_length=0,
        mamba_branching_seqlen=0,
        cache_protected_len=0,
    )
    tree_cache = SimpleNamespace(
        disable=False,
        match_prefix=MagicMock(return_value=match_result),
        check_hicache_events=MagicMock(),
        check_prefetch_progress=MagicMock(return_value=True),
        pop_prefetch_loaded_tokens=MagicMock(return_value=0),
        ready_to_load_host_cache=MagicMock(return_value=None),
        supports_mamba=lambda: False,
        scale_seq_factor=1,
    )

    def make_waiting_req(rid):
        req = Req.__new__(Req)
        req.rid = rid
        req.dllm_config = None
        req.origin_input_ids = [1, 2, 3, 4]
        req.output_ids = []
        req.prefix_indices = torch.empty((0,), dtype=torch.int64)
        req.attn_cp_prefill_split_spec = None
        req.session = None
        req.return_logprob = False
        req.logprob_start_len = -1
        req.positional_embed_overrides = None
        req.extra_key = None
        req.is_retracted = False
        req.retracted_stain = False
        req.multimodal_inputs = None
        req._scale_seq_factor = 1
        req.init_next_round_input = MagicMock(wraps=req.init_next_round_input)
        return req

    first = make_waiting_req("r0")
    second = make_waiting_req("r1")

    adder = SimpleNamespace(
        cp_sharded_allocator=object(),
        can_run_list=[],
        completed_without_forward_reqs=[],
        prefill_request_limit_reached=MagicMock(return_value=False),
        preempt_list=[],
        new_chunked_req=None,
        next_attn_cp_owner_rotation=1,
    )

    def add_one_req(req, **_kwargs):
        adder.can_run_list.append(req)
        return AddReqResult.CONTINUE

    adder.add_one_req = MagicMock(side_effect=add_one_req)
    new_batch = SimpleNamespace(prepare_for_extend=MagicMock())
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
    scheduler.enable_hierarchical_cache = enable_hierarchical_cache
    scheduler.enable_priority_preemption = False
    scheduler.is_hybrid_swa = False
    scheduler.running_batch = SimpleNamespace(
        batch_is_full=False,
        reqs=[],
        is_empty=lambda: True,
    )
    scheduler.waiting_queue = [first, second]
    scheduler.chunked_req = None
    scheduler.get_num_allocatable_reqs = MagicMock(return_value=2)
    scheduler.policy = (
        SimpleNamespace(calc_priority=MagicMock())
        if schedule_policy_name is None
        else SchedulePolicy(
            schedule_policy_name,
            tree_cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
    )
    scheduler.chunked_prefill_size = 8
    scheduler.enable_dynamic_chunking = False
    scheduler.page_size = 4
    scheduler.token_to_kv_pool_allocator = object()
    scheduler.tree_cache = tree_cache
    scheduler.new_token_ratio = 1.0
    scheduler.max_prefill_tokens = 128
    scheduler.is_mixed_chunk = False
    scheduler.priority_scheduling_preemption_threshold = 0
    scheduler.max_prefill_bs = 0
    scheduler.max_running_requests = 8
    scheduler.server_args = SimpleNamespace(
        prefill_max_requests=None,
        attn_cp_mode="sharded-kv",
    )
    scheduler.attn_cp_size = 2
    scheduler.dllm_config = None
    scheduler.truncation_align_size = None
    scheduler.next_attn_cp_owner_rotation = 0
    scheduler.enable_lora = False
    scheduler.disaggregation_mode = None
    scheduler.enable_hicache_storage = enable_hicache_storage
    scheduler.req_to_token_pool = object()
    scheduler.can_run_list = None
    scheduler.spec_algorithm = object()
    scheduler.model_config = SimpleNamespace(
        is_encoder_decoder=model_is_encoder_decoder,
    )
    scheduler.enable_overlap = False
    scheduler.enable_priority_scheduling = False
    scheduler._get_num_pending_tokens = MagicMock(return_value=0)

    return scheduler, adder, new_batch, first, second, tree_cache


def _run_sharded_prefill_ordering_fixture(scheduler, adder, new_batch):
    with (
        patch.object(scheduler_module, "PrefillAdder", return_value=adder),
        patch.object(
            scheduler_module.ScheduleBatch, "init_new", return_value=new_batch
        ),
        patch.object(
            scheduler_module.PrefillStats, "from_adder", return_value=None
        ),
        patch.object(scheduler_module, "set_time_batch"),
    ):
        return Scheduler._get_new_batch_prefill_raw(
            scheduler,
            prefill_delayer_single_pass=None,
        )


def test_scheduler_sharded_prefill_stops_before_second_prefix_match():
    scheduler, adder, new_batch, first, second, tree_cache = (
        _make_sharded_prefill_ordering_fixture()
    )

    result = _run_sharded_prefill_ordering_fixture(scheduler, adder, new_batch)

    assert result is new_batch
    first.init_next_round_input.assert_called_once_with(tree_cache)
    second.init_next_round_input.assert_not_called()
    tree_cache.match_prefix.assert_called_once()
    assert tree_cache.match_prefix.call_args.args[0].req is first
    adder.add_one_req.assert_called_once()


def test_scheduler_completes_deferred_full_hit_without_gpu_batch():
    scheduler, adder, new_batch, first, _, tree_cache = (
        _make_sharded_prefill_ordering_fixture()
    )
    scheduler.waiting_queue = [first]
    first.welm_deferred_prefill_span = build_welm_deferred_prefill_span(
        first.origin_input_ids
    )
    scheduler.process_deferred_prefill_without_forward = MagicMock()

    def complete_without_forward(req, **_kwargs):
        adder.completed_without_forward_reqs.append(req)
        return AddReqResult.CONTINUE

    adder.add_one_req = MagicMock(side_effect=complete_without_forward)

    result = _run_sharded_prefill_ordering_fixture(scheduler, adder, new_batch)

    assert result is None
    assert scheduler.waiting_queue == []
    scheduler.process_deferred_prefill_without_forward.assert_called_once_with([first])
    new_batch.prepare_for_extend.assert_not_called()


@pytest.mark.parametrize("pending_first", [True, False])
def test_scheduler_separates_deferred_pending_from_consumed_reprefill(pending_first):
    scheduler, adder, new_batch, first, second, _ = (
        _make_sharded_prefill_ordering_fixture()
    )
    scheduler.attn_cp_size = 1
    scheduler.server_args.attn_cp_mode = "none"
    adder.cp_sharded_allocator = None

    pending, consumed = (first, second) if pending_first else (second, first)
    for req in (pending, consumed):
        req.welm_deferred_prefill_span = build_welm_deferred_prefill_span(
            req.origin_input_ids
        )
        req.welm_deferred_decode_state = WelmDeferredDecodeState.from_prefill_span(
            req.welm_deferred_prefill_span
        )
    consumed.welm_deferred_decode_state.transition_to(WelmDeferredDecodePhase.READY)
    consumed.welm_deferred_decode_state.transition_to(WelmDeferredDecodePhase.INFLIGHT)
    consumed.welm_deferred_decode_state.transition_to(WelmDeferredDecodePhase.CONSUMED)
    consumed.output_ids = [5]

    result = _run_sharded_prefill_ordering_fixture(scheduler, adder, new_batch)

    admitted, skipped = first, second
    assert result is new_batch
    assert adder.can_run_list == [admitted]
    assert scheduler.waiting_queue == [skipped]
    admitted.init_next_round_input.assert_called_once()
    skipped.init_next_round_input.assert_not_called()


def test_cache_aware_policy_uses_deferred_committed_token_ids():
    req = Req.__new__(Req)
    req.rid = "deferred"
    req.origin_input_ids = [11, 22, 33, 44]
    req.output_ids = []
    req.welm_deferred_prefill_span = build_welm_deferred_prefill_span(
        req.origin_input_ids
    )
    req.extra_key = None
    req.prefix_indices = torch.arange(64)
    policy = SchedulePolicy.__new__(SchedulePolicy)
    policy.tree_cache = object()
    policy.waiting_queue_radix_tree = MagicMock()
    policy.waiting_queue_radix_tree.reset.return_value = None
    match_result = SimpleNamespace(device_indices=req.prefix_indices)

    with patch.object(
        schedule_policy,
        "match_prefix_for_req",
        return_value=match_result,
    ) as match_prefix:
        policy._compute_prefix_matches([req], CacheAwarePolicy.LPM)

    assert match_prefix.call_args.args[2] == [11, 22, 33]


@pytest.mark.parametrize(
    ("model_is_encoder_decoder", "enable_hierarchical_cache", "enable_hicache_storage"),
    [
        pytest.param(True, False, False, id="encoder_decoder"),
        pytest.param(False, True, False, id="hierarchical_cache"),
        pytest.param(False, False, True, id="hicache_storage"),
    ],
)
def test_scheduler_sharded_prefill_rejects_unsupported_modes_before_prefix_work(
    model_is_encoder_decoder,
    enable_hierarchical_cache,
    enable_hicache_storage,
):
    scheduler, adder, new_batch, first, second, tree_cache = (
        _make_sharded_prefill_ordering_fixture(
            model_is_encoder_decoder=model_is_encoder_decoder,
            enable_hierarchical_cache=enable_hierarchical_cache,
            enable_hicache_storage=enable_hicache_storage,
        )
    )

    with pytest.raises(ValueError):
        _run_sharded_prefill_ordering_fixture(scheduler, adder, new_batch)

    first.init_next_round_input.assert_not_called()
    second.init_next_round_input.assert_not_called()
    tree_cache.match_prefix.assert_not_called()
    adder.add_one_req.assert_not_called()


def test_scheduler_sharded_prefill_rejects_cache_aware_policy_before_prefix_probe():
    scheduler, adder, new_batch, first, second, tree_cache = (
        _make_sharded_prefill_ordering_fixture(schedule_policy_name="lpm")
    )

    with pytest.raises(ValueError, match="cache-aware"):
        _run_sharded_prefill_ordering_fixture(scheduler, adder, new_batch)

    first.init_next_round_input.assert_not_called()
    second.init_next_round_input.assert_not_called()
    tree_cache.match_prefix.assert_not_called()
    adder.add_one_req.assert_not_called()


def test_schedule_batch_split_specs_filter_merge_and_copy_in_request_order():
    first_spec = _split_spec(0, 4)
    second_spec = _split_spec(4, 4, rotation=1)
    req0 = _req_with_spec("r0", first_spec)
    req1 = _req_with_spec("r1", second_spec)
    req2 = _req_with_spec("r2", None)

    batch = ScheduleBatch(
        reqs=[req0, req1],
        model_config=SimpleNamespace(is_encoder_decoder=False),
        device="cpu",
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([4, 8]),
        seq_lens_cpu=torch.tensor([4, 8]),
        orig_seq_lens=torch.tensor([4, 8]),
        sampling_info=MagicMock(),
        attn_cp_prefill_split_specs=(first_spec, second_spec),
    )

    snapshot = batch.copy()
    batch.filter_batch(keep_indices=[1])

    assert batch.attn_cp_prefill_split_specs == (second_spec,)
    assert snapshot.attn_cp_prefill_split_specs == (first_spec, second_spec)

    other = ScheduleBatch(
        reqs=[req2],
        model_config=SimpleNamespace(is_encoder_decoder=False),
        req_pool_indices=torch.tensor([2]),
        seq_lens=torch.tensor([9]),
        seq_lens_cpu=torch.tensor([9]),
        orig_seq_lens=torch.tensor([9]),
        seq_lens_sum=9,
        sampling_info=MagicMock(),
        attn_cp_prefill_split_specs=None,
    )
    batch.seq_lens_sum = 8
    batch.merge_batch(other)

    assert batch.attn_cp_prefill_split_specs == (second_spec, None)


def test_schedule_batch_lifecycle_uses_current_request_specs_only():
    first_spec = _split_spec(0, 4)
    second_spec = _split_spec(4, 4, rotation=1)
    req0 = _req_with_spec("r0", first_spec)
    req1 = _req_with_spec("r1", second_spec)
    batch = ScheduleBatch(
        reqs=[req0, req1],
        model_config=SimpleNamespace(is_encoder_decoder=False),
        device="cpu",
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([4, 8]),
        seq_lens_cpu=torch.tensor([4, 8]),
        orig_seq_lens=torch.tensor([4, 8]),
        sampling_info=MagicMock(),
        attn_cp_prefill_split_specs=(first_spec, second_spec),
    )

    # Completion/retraction clears the shared Req while an overlap snapshot can
    # still hold the old batch tuple.
    req0.attn_cp_prefill_split_spec = None
    snapshot = batch.copy()
    assert snapshot.attn_cp_prefill_split_specs == (None, second_spec)

    batch.filter_batch(keep_indices=[0])
    assert batch.attn_cp_prefill_split_specs is None

    other_req = _req_with_spec("r2", None)
    other = ScheduleBatch(
        reqs=[other_req],
        model_config=SimpleNamespace(is_encoder_decoder=False),
        req_pool_indices=torch.tensor([2]),
        seq_lens=torch.tensor([9]),
        seq_lens_cpu=torch.tensor([9]),
        orig_seq_lens=torch.tensor([9]),
        seq_lens_sum=9,
        sampling_info=MagicMock(),
        attn_cp_prefill_split_specs=(second_spec,),
    )
    batch.seq_lens_sum = 4
    batch.merge_batch(other)

    assert batch.attn_cp_prefill_split_specs is None


def test_split_specs_propagate_as_immutable_request_aligned_tuples():
    first_spec = _split_spec(0, 4)
    second_spec = _split_spec(4, 4, rotation=1)
    reqs = [_req_with_spec("r0", first_spec), _req_with_spec("r1", second_spec)]
    batch = ScheduleBatch(
        reqs=reqs,
        forward_mode=ForwardMode.EXTEND,
        input_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([4, 8]),
        seq_lens_cpu=torch.tensor([4, 8]),
        orig_seq_lens=torch.tensor([4, 8]),
        out_cache_loc=torch.arange(8),
        seq_lens_sum=12,
        extend_num_tokens=8,
        extend_lens=[4, 4],
        prefix_lens=[0, 4],
        extend_logprob_start_lens=[0, 0],
        attn_cp_prefill_split_specs=(first_spec, second_spec),
    )
    batch._get_welm_kv_mirror_last_q_indices = MagicMock(
        return_value=(None, None, None)
    )

    worker_batch = batch.get_model_worker_batch()

    assert worker_batch.attn_cp_prefill_split_specs == (first_spec, second_spec)
    assert isinstance(worker_batch.attn_cp_prefill_split_specs, tuple)

    model_runner = SimpleNamespace(
        req_to_token_pool=object(),
        token_to_kv_pool=object(),
        attn_backend=object(),
        device="cpu",
        server_args=SimpleNamespace(
            enable_welm_kv_mirror_opt=False,
            attention_backend="torch_native",
            enable_lora=False,
        ),
        use_ngram_embedding=False,
        model_is_mrope=False,
        is_hybrid_swa=False,
    )
    with patch(
        "sglang.srt.model_executor.forward_batch_info.compute_position",
        return_value=(torch.arange(8), torch.tensor([0, 4])),
    ), patch(
        "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
        return_value=False,
    ):
        forward_batch = ForwardBatch.init_new(worker_batch, model_runner)

    assert forward_batch.attn_cp_prefill_split_specs == (first_spec, second_spec)
    assert isinstance(forward_batch.attn_cp_prefill_split_specs, tuple)
    assert forward_batch.attn_cp_prefill_runtime_layout is None


@pytest.mark.parametrize("is_hybrid_swa", [False, True])
def test_forward_batch_materializes_prefill_cp_runtime_without_rewriting_full_rows(
    is_hybrid_swa,
):
    spec = _split_spec(0, 8)
    req = _req_with_spec("r0", spec)
    input_ids = torch.arange(8, dtype=torch.int64) + 10
    out_cache_loc = torch.tensor([100, 101, 102, 103, 0, 0, 0, 0])
    batch = ScheduleBatch(
        reqs=[req],
        forward_mode=ForwardMode.EXTEND,
        input_ids=input_ids,
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([8]),
        seq_lens_cpu=torch.tensor([8]),
        orig_seq_lens=torch.tensor([8]),
        out_cache_loc=out_cache_loc,
        seq_lens_sum=8,
        extend_num_tokens=8,
        extend_lens=[8],
        prefix_lens=[0],
        extend_logprob_start_lens=[0],
        attn_cp_prefill_split_specs=(spec,),
    )
    batch._get_welm_kv_mirror_last_q_indices = MagicMock(
        return_value=(None, None, None)
    )
    worker_batch = batch.get_model_worker_batch()
    allocator = SimpleNamespace(cp_rank=0, cp_size=2)
    allocator.owner_plan_for_logical_slots = lambda slots: CPLogicalOwnerPlan(
        owner_ranks=torch.empty((0,), dtype=torch.int64),
        per_rank_counts=(0, 0),
        rank_packed_to_logical=torch.empty((0,), dtype=torch.int64),
    )
    allocator.logical_slots_to_physical = lambda slots: slots + 1000
    allocator.translate_loc_from_full_to_swa = lambda slots: torch.where(
        slots.ne(0), slots + 2000, torch.zeros_like(slots)
    )
    model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.empty((1, 8), dtype=torch.int64)
        ),
        token_to_kv_pool=object(),
        token_to_kv_pool_allocator=allocator,
        attn_backend=object(),
        device="cpu",
        supports_attn_cp_prefill_runtime=True,
        _prefill_cp_kv_ipc_transport=object(),
        server_args=SimpleNamespace(
            enable_welm_kv_mirror_opt=False,
            attention_backend="fa3",
            enable_lora=False,
        ),
        use_ngram_embedding=False,
        model_is_mrope=False,
        is_hybrid_swa=is_hybrid_swa,
    )

    with patch(
        "sglang.srt.model_executor.forward_batch_info.compute_position",
        return_value=(torch.arange(8, dtype=torch.int32), torch.tensor([0])),
    ), patch(
        "sglang.srt.model_executor.forward_batch_info.enable_num_token_non_padded",
        return_value=False,
    ):
        forward_batch = ForwardBatch.init_new(worker_batch, model_runner)

    runtime = forward_batch.attn_cp_prefill_runtime_layout
    assert runtime is not None
    assert torch.equal(forward_batch.input_ids, input_ids)
    assert torch.equal(forward_batch.positions, torch.arange(8, dtype=torch.int32))
    assert torch.equal(forward_batch.out_cache_loc, out_cache_loc)
    assert torch.equal(runtime.local_input_ids, input_ids[:4])
    assert torch.equal(runtime.local_positions, forward_batch.positions[:4])
    assert torch.equal(runtime.local_out_cache_loc, out_cache_loc[:4])
    gather_plan = forward_batch.attn_cp_prefill_kv_gather_plan
    assert gather_plan is not None
    assert gather_plan.prefix.sizes == (0, 0)
    assert gather_plan.extend.sizes == spec.per_rank_tokens
    source_push_plan = forward_batch.attn_cp_prefill_kv_source_push_plan
    assert source_push_plan is not None
    assert source_push_plan.source_mask == 0b11
    assert source_push_plan.prefix.source_rows.numel() == 0
    assert source_push_plan.extend.source_rows.tolist() == [0, 1, 2, 3]
    assert source_push_plan.extend.destination_rows.tolist() == [0, 1, 2, 3]


def test_decode_does_not_materialize_prefill_cp_runtime():
    spec = _split_spec(0, 4)
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
        input_ids=torch.tensor([1]),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([5]),
        out_cache_loc=torch.tensor([100]),
        seq_lens_sum=5,
        positions=torch.tensor([4]),
        attn_cp_prefill_split_specs=(spec,),
    )
    model_runner = SimpleNamespace(supports_attn_cp_prefill_runtime=True)

    forward_batch._materialize_attn_cp_prefill_runtime(model_runner)

    assert forward_batch.attn_cp_prefill_runtime_layout is None


def test_page_size_one_phase2_prefill_fails_instead_of_dense_fallback():
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor([1]),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
        out_cache_loc=torch.tensor([100]),
        seq_lens_sum=1,
        positions=torch.tensor([0]),
        attn_cp_prefill_split_specs=None,
    )
    model_runner = SimpleNamespace(
        supports_attn_cp_prefill_runtime=True,
        token_to_kv_pool_allocator=SimpleNamespace(
            cp_rank=0,
            cp_size=2,
            page_size=1,
        ),
    )

    with pytest.raises(NotImplementedError, match="refusing dense fallback"):
        forward_batch._materialize_attn_cp_prefill_runtime(model_runner)


def test_decode_mode_clears_request_and_batch_split_specs():
    spec = _split_spec(0, 4)
    req = _req_with_spec("r0", spec)
    batch = ScheduleBatch(
        reqs=[req],
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=False)
        ),
        attn_cp_prefill_split_specs=(spec,),
    )

    batch.prepare_for_decode()

    assert req.attn_cp_prefill_split_spec is None
    assert batch.attn_cp_prefill_split_specs is None


def test_encoder_decoder_rejects_phase2_spec_before_preparation():
    spec = _split_spec(0, 4)
    req = _req_with_spec("r0", spec)
    batch = ScheduleBatch(
        reqs=[req],
        model_config=SimpleNamespace(is_encoder_decoder=True),
        attn_cp_prefill_split_specs=(spec,),
    )
    batch.prepare_encoder_info_extend = MagicMock()

    with pytest.raises(ValueError, match="encoder-decoder"):
        batch.prepare_for_extend()

    batch.prepare_encoder_info_extend.assert_not_called()


def test_prefill_result_completion_clears_request_split_spec():
    spec = _split_spec(0, 4)
    req = SimpleNamespace(
        rid="r0",
        attn_cp_prefill_split_spec=spec,
        is_retracted=False,
        is_chunked=1,
        time_stats=MagicMock(),
        finished=lambda: False,
    )
    batch = SimpleNamespace(
        reqs=[req],
        attn_cp_prefill_split_specs=(spec,),
        return_logprob=False,
        decoding_reqs=None,
        prefill_stats=None,
        dp_cooperation_info=None,
        fpm_start_time=0,
    )
    result = SimpleNamespace(
        copy_done=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=SimpleNamespace(),
        next_token_ids=torch.tensor([1]),
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
        draft_continuation_state=None,
        can_run_cuda_graph=False,
        welm_deferred_prefill_completion=None,
    )
    scheduler = SimpleNamespace(
        is_generation=True,
        stream_output=MagicMock(),
        report_prefill_stats=MagicMock(),
    )

    SchedulerOutputProcessorMixin.process_batch_result_prefill(
        scheduler, batch, result
    )

    assert req.attn_cp_prefill_split_spec is None
    assert batch.attn_cp_prefill_split_specs is None


def test_stale_prefill_completion_preserves_next_chunk_split_spec():
    completed_spec = _split_spec(0, 4)
    next_chunk_spec = _split_spec(4, 4)
    req = SimpleNamespace(
        rid="r0",
        attn_cp_prefill_split_spec=next_chunk_spec,
        is_retracted=False,
        is_chunked=1,
        time_stats=MagicMock(),
        finished=lambda: False,
    )
    completed_batch = SimpleNamespace(
        reqs=[req],
        attn_cp_prefill_split_specs=(completed_spec,),
        return_logprob=False,
        decoding_reqs=None,
        prefill_stats=None,
        dp_cooperation_info=None,
        fpm_start_time=0,
    )
    result = SimpleNamespace(
        copy_done=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        logits_output=SimpleNamespace(),
        next_token_ids=torch.tensor([0]),
        extend_input_len_per_req=None,
        extend_logprob_start_len_per_req=None,
        draft_continuation_state=None,
        can_run_cuda_graph=False,
        welm_deferred_prefill_completion=None,
    )
    scheduler = SimpleNamespace(
        is_generation=True,
        stream_output=MagicMock(),
        report_prefill_stats=MagicMock(),
    )

    SchedulerOutputProcessorMixin.process_batch_result_prefill(
        scheduler, completed_batch, result
    )

    assert req.attn_cp_prefill_split_spec is next_chunk_spec
    assert completed_batch.attn_cp_prefill_split_specs is None
