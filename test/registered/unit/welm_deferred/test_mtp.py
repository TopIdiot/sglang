import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.speculative.welmv4_mtp_kv import (
    should_use_welm_mtp_direct_kv,
    should_use_welm_mtp_lightweight_prefill,
    should_use_welm_mtp_storage_draft_kv,
)
from sglang.test.ci.ci_register import register_cpu_ci
from mtp_test_utils import (
    build_eagle_worker as _build_eagle_worker,
    model_config as _model_config,
    worker_args as _worker_args,
)

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


_LEGACY_MIRROR_ENV = "SGLANG_WELM_MTP_LEGACY_MIRROR_STATE"


def test_missing_spec_v2_draft_worker_fails_unless_lightweight():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.draft_worker = SimpleNamespace(
        draft_worker=None,
        _is_welm_mtp_lightweight_prefill=False,
    )
    scheduler.spec_algorithm = SimpleNamespace(
        is_ngram=lambda: False,
        supports_spec_v2=lambda: True,
    )
    scheduler.enable_overlap = True

    with pytest.raises(RuntimeError, match="missing its Draft worker"):
        Scheduler._get_draft_model_runner(scheduler)


def test_mtp_direct_kv_gate_selects_all_default_execution_modes():
    target_config = _model_config()
    assert should_use_welm_mtp_direct_kv(
        _worker_args(disaggregation_mode="null"), target_config
    )
    for mode in ("legacy", "deferred-last-prompt"):
        assert should_use_welm_mtp_direct_kv(
            _worker_args(
                disaggregation_mode="decode",
                welm_kv_mirror_pd_mode=mode,
            ),
            target_config,
        )
        assert should_use_welm_mtp_storage_draft_kv(
            _worker_args(
                disaggregation_mode="prefill",
                welm_kv_mirror_pd_mode=mode,
            ),
            target_config,
        )
        assert should_use_welm_mtp_lightweight_prefill(
            _worker_args(
                disaggregation_mode="prefill",
                welm_kv_mirror_pd_mode=mode,
            ),
            target_config,
        )

    with pytest.raises(RuntimeError, match="topk=1"):
        should_use_welm_mtp_direct_kv(
            _worker_args(disaggregation_mode="null", speculative_eagle_topk=2),
            target_config,
        )
    with pytest.raises(RuntimeError, match="AttnCP"):
        should_use_welm_mtp_direct_kv(
            _worker_args(disaggregation_mode="null", attn_cp_size=2), target_config
        )
    for unsupported_backend in ("flashinfer", "triton"):
        with pytest.raises(RuntimeError, match="requires FA3"):
            should_use_welm_mtp_direct_kv(
                _worker_args(
                    disaggregation_mode="null", attention_backend=unsupported_backend
                ),
                target_config,
            )


def test_legacy_mirror_compatibility_is_explicit_and_pd_legacy_only():
    target_config = _model_config()
    with patch.dict(os.environ, {_LEGACY_MIRROR_ENV: "1"}):
        assert not should_use_welm_mtp_direct_kv(
            _worker_args(
                disaggregation_mode="decode",
                welm_kv_mirror_pd_mode="legacy",
            ),
            target_config,
        )
        assert not should_use_welm_mtp_storage_draft_kv(
            _worker_args(
                disaggregation_mode="prefill",
                welm_kv_mirror_pd_mode="legacy",
            ),
            target_config,
        )
        with pytest.raises(RuntimeError, match="P/D legacy"):
            should_use_welm_mtp_direct_kv(
                _worker_args(disaggregation_mode="null"), target_config
            )
        with pytest.raises(RuntimeError, match="P/D legacy"):
            should_use_welm_mtp_storage_draft_kv(
                _worker_args(disaggregation_mode="prefill"), target_config
            )
        with pytest.raises(RuntimeError, match="P/D legacy"):
            should_use_welm_mtp_direct_kv(
                _worker_args(disaggregation_mode="decode"),
                _model_config(architectures=["Qwen2MoeForCausalLM"]),
            )


def test_mtp_direct_kv_rejects_pre_attention_v2_until_supported():
    with patch.dict(
        os.environ,
        {"SGLANG_WELM_V45_80A3_FUSED_PRE_ATTN": "1"},
    ):
        with pytest.raises(RuntimeError, match="Pre-Attention V2"):
            should_use_welm_mtp_direct_kv(
                _worker_args(disaggregation_mode="null"), _model_config()
            )


@pytest.mark.parametrize("disaggregation_mode", ["null", "decode"])
def test_mtp_direct_kv_binds_before_single_target_and_draft_graph_capture(
    disaggregation_mode,
):
    events = []
    args = _worker_args(disaggregation_mode=disaggregation_mode)
    req_pool = object()
    allocator = object()
    target_pool = SimpleNamespace(page_size=16)
    bind = MagicMock(side_effect=lambda *_args, **_kwargs: events.append("bind"))
    model_config = _model_config()
    model_config.context_len = 4096
    target_runner = SimpleNamespace(
        model_config=model_config,
        model=SimpleNamespace(model=SimpleNamespace(bind_mtp_direct_kv=bind)),
        token_to_kv_pool=target_pool,
        token_to_kv_pool_allocator=allocator,
        init_welm_mtp_direct_kv_graphs=MagicMock(
            side_effect=lambda: events.append("target_graph")
        ),
    )
    target = SimpleNamespace(
        model_runner=target_runner,
        model_config=model_config,
        get_memory_pool=MagicMock(return_value=(req_pool, allocator)),
    )

    draft_runner = SimpleNamespace(
        model_config=_model_config(
            architectures=["WeLMV4MoeForCausalLMNextN"]
        ),
        model=SimpleNamespace(model=object()),
        token_to_kv_pool=SimpleNamespace(page_size=16),
        token_to_kv_pool_allocator=allocator,
        tp_group=object(),
    )
    draft = MagicMock(draft_runner=draft_runner)
    draft.draft_runner = draft_runner
    draft.req_to_token_pool = req_pool
    draft.init_cuda_graphs.side_effect = lambda: events.append("draft_graph")

    with (
        patch(
            "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker",
            return_value=draft,
        ) as draft_cls,
        patch(
            "sglang.srt.speculative.eagle_worker_v2._get_plan_stream",
            return_value=(object(), object()),
        ),
    ):
        worker = _build_eagle_worker(args, target_worker=target)

    assert events == ["bind", "target_graph", "draft_graph"]
    assert worker.draft_worker is draft
    assert worker.welm_mtp_direct_kv_enabled
    assert draft.welm_mtp_direct_kv_enabled
    assert draft_cls.call_args.kwargs["defer_cuda_graph_capture"]
    bind.assert_called_once_with(
        draft_runner.model.model,
        target_kv_pool=target_pool,
        draft_kv_pool=draft_runner.token_to_kv_pool,
    )


def test_mtp_direct_kv_rejects_non_welm_nextn_draft_before_binding():
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

    req_pool = object()
    allocator = object()
    bind = MagicMock()
    target_runner = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(bind_mtp_direct_kv=bind)),
        token_to_kv_pool=SimpleNamespace(page_size=16),
        token_to_kv_pool_allocator=allocator,
        init_welm_mtp_direct_kv_graphs=MagicMock(),
    )
    draft_runner = SimpleNamespace(
        model_config=_model_config(architectures=["Qwen3ForCausalLM"]),
        model=SimpleNamespace(model=object()),
        token_to_kv_pool=SimpleNamespace(page_size=16),
        token_to_kv_pool_allocator=allocator,
        tp_group=object(),
    )
    draft_worker = MagicMock(draft_runner=draft_runner)
    draft_worker.draft_runner = draft_runner
    draft_worker.req_to_token_pool = req_pool
    worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
    worker._target_worker = SimpleNamespace(model_runner=target_runner)
    worker._draft_worker = draft_worker
    worker.req_to_token_pool = req_pool

    with pytest.raises(RuntimeError, match="WeLM NextN Draft"):
        worker._initialize_welm_mtp_direct_kv()

    bind.assert_not_called()


@pytest.mark.parametrize("mode", ["legacy", "deferred-last-prompt"])
def test_lightweight_prefill_binds_storage_only_draft_pool_by_default(mode):
    args = _worker_args(
        disaggregation_mode="prefill", welm_kv_mirror_pd_mode=mode
    )
    events = []
    req_pool = object()
    allocator = object()
    target_pool = SimpleNamespace(page_size=16)
    draft_pool = SimpleNamespace(page_size=16)
    draft_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["WeLMV4MoeForCausalLMNextN"])
    )
    bind = MagicMock(side_effect=lambda *_args, **_kwargs: events.append("bind"))
    model_config = _model_config()
    model_config.context_len = 4096
    target_runner = SimpleNamespace(
        model_config=model_config,
        welm_mtp_storage_draft_model_config=draft_config,
        model=SimpleNamespace(
            model=SimpleNamespace(bind_mtp_storage_draft_kv=bind)
        ),
        token_to_kv_pool=target_pool,
        token_to_kv_pool_allocator=allocator,
        create_storage_only_kv_pool=MagicMock(return_value=draft_pool),
        init_welm_mtp_direct_kv_graphs=MagicMock(
            side_effect=lambda: events.append("target_graph")
        ),
    )
    target = SimpleNamespace(
        model_runner=target_runner,
        model_config=model_config,
        get_memory_pool=MagicMock(return_value=(req_pool, allocator)),
    )

    worker = _build_eagle_worker(args, target_worker=target)

    assert getattr(worker, "welm_mtp_storage_draft_kv_pool", None) is draft_pool
    assert worker.welm_mtp_direct_kv_enabled
    assert events == ["bind", "target_graph"]
    target_runner.create_storage_only_kv_pool.assert_called_once_with(draft_config)
    bind.assert_called_once_with(
        draft_config,
        draft_kv_pool=draft_pool,
    )


@pytest.mark.parametrize("page_size", [1, 16])
def test_storage_only_draft_swa_pool_reuses_target_mapping(page_size):
    from sglang.srt.model_executor.model_runner import ModelRunner

    runner = ModelRunner.__new__(ModelRunner)
    runner.full_max_total_num_tokens = 64
    runner.swa_max_total_num_tokens = 32
    runner.page_size = page_size
    runner.kv_cache_dtype = torch.bfloat16
    runner.device = "cpu"
    runner.server_args = SimpleNamespace(
        enable_memory_saver=False,
        enable_pdmux=False,
    )
    mapping = torch.arange(65, dtype=torch.int64)
    runner.token_to_kv_pool_allocator = SimpleNamespace(
        full_to_swa_index_mapping=mapping
    )
    draft_config = SimpleNamespace(
        is_hybrid_swa=True,
        full_attention_layer_ids=[],
        swa_attention_layer_ids=[0],
        head_dim=128,
        v_head_dim=128,
        get_num_kv_heads=lambda tp_size: 1,
    )

    with (
        patch(
            "sglang.srt.model_executor.model_runner_kv_cache_mixin.get_attention_tp_size",
            return_value=4,
        ),
        patch(
            "sglang.srt.model_executor.model_runner_kv_cache_mixin.SWAKVPool"
        ) as pool_cls,
    ):
        pool = runner.create_storage_only_kv_pool(draft_config)

    assert pool is pool_cls.return_value
    pool_cls.assert_called_once_with(
        size=64,
        size_swa=32,
        page_size=page_size,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=128,
        v_head_dim=128,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[],
        enable_kvcache_transpose=False,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=True,
        enable_kv_cache_copy=True,
    )
    pool.register_mapping.assert_called_once_with(mapping)


def _oe_history_worker():
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker._should_use_welmv4_mtp_oe_hash_kernel = lambda: True
    worker._welmv4_mtp_oe_hash_config = lambda: ((2, 3), (1, 1), 3)
    return worker


def test_mtp_root_only_oe_history_updates_only_seed_rows():
    worker = _oe_history_worker()
    worker._welmv4_mtp_oe_prefix_width = lambda: 2
    worker._init_welmv4_mtp_oe_history_from_context = MagicMock(
        return_value=torch.tensor([[70, 71, 72]], dtype=torch.int64)
    )
    original_history = torch.tensor(
        [[10, 11, 12], [0, 0, 0]], dtype=torch.int64
    )
    draft_input = SimpleNamespace(
        bonus_tokens=torch.tensor([9, 20], dtype=torch.int64),
        welm_mtp_root_only_verify_mask=torch.tensor([False, True]),
        welm_mtp_oe_history_state=original_history,
    )
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(origin_input_ids=[1, 2, 9], output_ids=[]),
            SimpleNamespace(origin_input_ids=[3, 4, 20], output_ids=[]),
        ]
    )

    worker._prepare_welmv4_mtp_root_only_oe_history(draft_input, batch)

    torch.testing.assert_close(
        draft_input.welm_mtp_oe_history_state,
        torch.tensor([[10, 11, 12], [70, 71, 72]]),
    )
    torch.testing.assert_close(original_history[1], torch.zeros(3, dtype=torch.int64))
    init_call = worker._init_welmv4_mtp_oe_history_from_context.call_args
    assert init_call.args[0] is None
    assert init_call.kwargs["prefix_rows"] == [[4], [3]]
    torch.testing.assert_close(
        init_call.kwargs["first_token_ids"], torch.tensor([20])
    )


def _root_mirror_worker():
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

    worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
    worker.speculative_num_draft_tokens = 4
    worker.welm_mtp_kv_mirror_state_buffers = {
        "48.k": torch.full((32, 2), -1, dtype=torch.bfloat16),
        "48.v": torch.full((32, 2), -2, dtype=torch.bfloat16),
    }
    req_to_token = torch.zeros((2, 8), dtype=torch.int64)
    req_to_token[0, 2] = 10
    req_to_token[1, 3] = 20
    worker.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    return worker


def test_root_only_direct_kv_commit_only_validates_canonical_root_slot():
    worker = _root_mirror_worker()
    worker.welm_mtp_direct_kv_enabled = True
    worker.welm_mtp_kv_mirror_state_buffers = None
    root_slot = 16
    worker.req_to_token_pool.req_to_token[1, 3] = root_slot
    batch = SimpleNamespace(
        seq_lens=torch.tensor([2, 3], dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        out_cache_loc=torch.tensor(
            [10, 11, 12, 13, root_slot, root_slot + 1, root_slot + 2, root_slot + 3]
        ),
    )
    verify_input = SimpleNamespace(
        welm_mtp_root_only_verify_mask=torch.tensor([False, True])
    )

    worker._commit_welmv4_mtp_root_only_mirror_states(
        batch,
        verify_input,
        torch.tensor([[0, 1, -1, -1], [4, -1, -1, -1]]),
        None,
    )


def test_pd_decode_direct_kv_bootstrap_does_not_require_mirror_payload():
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.server_args = SimpleNamespace(enable_welm_kv_mirror_opt=True)
    worker.welm_mtp_direct_kv_enabled = True
    worker.draft_runner = object()
    worker._build_welmv4_mtp_pd_prefill_mirror_states = MagicMock(return_value=None)
    worker._should_use_welmv4_mtp_oe_hash_kernel = lambda: False
    worker._run_welmv4_mtp_merged_extend_draft = MagicMock()
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        seq_lens=torch.tensor([2], dtype=torch.int32),
        extend_seq_lens=[2],
        extend_prefix_lens=[0],
        input_ids=torch.tensor([1, 2], dtype=torch.int32),
        is_extend_in_batch=True,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        oe_context=None,
    )
    draft_input = EagleDraftInput(
        bonus_tokens=torch.tensor([3], dtype=torch.int64),
        new_seq_lens=batch.seq_lens,
    )
    draft_input.welm_mtp_deferred_prefill_draft = True
    draft_input.welm_mtp_deferred_prefill_draft_mask = torch.ones(
        (1,), dtype=torch.bool
    )
    original_model_specific_states = draft_input.model_specific_states

    with patch(
        "sglang.srt.speculative.eagle_worker_v2.ForwardBatch.init_new",
        return_value=SimpleNamespace(return_logprob=True),
    ):
        worker._materialize_welmv4_mtp_deferred_prefill_draft(batch, draft_input)

    worker._build_welmv4_mtp_pd_prefill_mirror_states.assert_not_called()
    worker._run_welmv4_mtp_merged_extend_draft.assert_called_once()
    assert draft_input.model_specific_states is original_model_specific_states
    assert not draft_input.welm_mtp_deferred_prefill_draft
    assert draft_input.welm_mtp_deferred_prefill_draft_mask is None


def _continuation_worker():
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    worker.device = torch.device("cpu")
    worker.speculative_num_draft_tokens = 4
    worker.draft_runner = SimpleNamespace(
        model_config=SimpleNamespace(vocab_size=1024)
    )
    worker.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor(
            [
                [10, 11, 12, 13, 14, 15],
                [20, 21, 22, 23, 24, 25],
                [30, 31, 32, 33, 34, 35],
            ],
            dtype=torch.int64,
        )
    )
    worker._should_use_welmv4_mtp_oe_hash_kernel = lambda: False
    return worker


class _NoRootPredicateTensor(torch.Tensor):
    def any(self, *args, **kwargs):
        raise AssertionError("root-only worker read a device predicate")


def _continuation_inputs():
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.utils import DraftContinuationState, GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.eagle_info import EagleDraftInput

    next_draft_input = EagleDraftInput(
        bonus_tokens=torch.tensor([70, 71, 72], dtype=torch.int32),
        new_seq_lens=torch.tensor([3, 4, 5], dtype=torch.int32),
    )
    hidden_states = torch.arange(12 * 3, dtype=torch.float32).reshape(12, 3)
    batch_result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=torch.zeros((12, 8)),
            hidden_states=hidden_states,
            model_specific_states=None,
        ),
        next_token_ids=torch.arange(100, 112, dtype=torch.int64),
        accept_lens=torch.tensor([2, 1, 3], dtype=torch.int32),
        spec_accept_index=torch.tensor(
            [[0, 1, -1, -1], [4, -1, -1, -1], [8, 9, 10, -1]],
            dtype=torch.int64,
        ),
        welm_mtp_accepted_draft_token_ids=torch.tensor(
            [[1, 2, -1, -1], [3, -1, -1, -1], [4, 5, 6, -1]],
            dtype=torch.int64,
        ),
        draft_continuation_state=DraftContinuationState(next_draft_input),
        speculative_num_draft_tokens=4,
    )
    reqs = [
        SimpleNamespace(origin_input_ids=[1, 2, 10], output_ids=[], req_pool_idx=0),
        SimpleNamespace(
            origin_input_ids=[20, 21, 22, 23], output_ids=[], req_pool_idx=1
        ),
        SimpleNamespace(
            origin_input_ids=[30, 31, 32, 33, 34], output_ids=[], req_pool_idx=2
        ),
    ]
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        input_ids=torch.arange(12, dtype=torch.int64),
        req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
        seq_lens=torch.tensor([2, 3, 4], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([2, 3, 4], dtype=torch.int32),
        seq_lens_sum=9,
        out_cache_loc=torch.tensor(
            [12, 13, 14, 15, 23, 24, 25, 26, 34, 35, 36, 37],
            dtype=torch.int64,
        ),
        reqs=reqs,
        sampling_info=None,
        spec_info=SimpleNamespace(
            draft_token=torch.tensor(
                [10, 0, 0, 0, 23, 0, 0, 0, 34, 0, 0, 0],
                dtype=torch.int64,
            )
        ),
        oe_context=None,
        lora_ids=[None, None, None],
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        orig_seq_lens=None,
        global_num_tokens=None,
        global_num_tokens_for_logprob=None,
        global_num_reqs=None,
        global_forward_modes=None,
        welm_mtp_global_prefill_num_tokens=None,
    )
    return batch, batch_result


def test_ordinary_continuation_subset_remaps_dense_verify_rows():
    worker = _continuation_worker()
    batch, batch_result = _continuation_inputs()
    mirror_k = torch.arange(12 * 2, dtype=torch.bfloat16).reshape(12, 2)
    mirror_v = mirror_k + 100
    batch_result.logits_output.model_specific_states = {
        "welm_kv_mirror_states": {48: (mirror_k, mirror_v)}
    }

    subset_batch, subset_result = (
        worker._subset_welmv4_mtp_continuation(
            batch, batch_result, [0, 2]
        )
    )

    torch.testing.assert_close(
        subset_batch.seq_lens, torch.tensor([2, 4], dtype=torch.int32)
    )
    torch.testing.assert_close(
        subset_batch.out_cache_loc,
        torch.tensor([12, 13, 14, 15, 34, 35, 36, 37]),
    )
    torch.testing.assert_close(
        subset_result.spec_accept_index,
        torch.tensor([[0, 1, -1, -1], [4, 5, 6, -1]]),
    )
    torch.testing.assert_close(
        subset_result.next_token_ids,
        torch.tensor([100, 101, 102, 103, 108, 109, 110, 111]),
    )
    torch.testing.assert_close(
        subset_result.logits_output.hidden_states,
        batch_result.logits_output.hidden_states[[0, 1, 2, 3, 8, 9, 10, 11]],
    )
    subset_k, subset_v = subset_result.logits_output.model_specific_states[
        "welm_kv_mirror_states"
    ][48]
    token_rows = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
    torch.testing.assert_close(subset_k, mirror_k[token_rows])
    torch.testing.assert_close(subset_v, mirror_v[token_rows])
    assert subset_result.draft_continuation_state.draft_input.bonus_tokens.tolist() == [
        70,
        72,
    ]


def test_direct_continuation_subset_rejects_mirror_payload():
    worker = _continuation_worker()
    worker.welm_mtp_direct_kv_enabled = True
    batch, batch_result = _continuation_inputs()
    mirror = torch.zeros((12, 2), dtype=torch.bfloat16)
    batch_result.logits_output.model_specific_states = {
        "welm_kv_mirror_states": {48: (mirror, mirror)}
    }

    with pytest.raises(RuntimeError, match="payload-free"):
        worker._subset_welmv4_mtp_continuation(batch, batch_result, [0, 2])


def test_root_only_continuation_runs_ordinary_and_seed_paths_once():
    worker = _continuation_worker()
    worker.topk = 1
    worker.speculative_num_steps = 3
    worker._is_welmv4_mtp_draft_model = lambda: True
    batch, batch_result = _continuation_inputs()
    batch.welm_mtp_root_only_rows = [False, True, False]
    destination = batch_result.draft_continuation_state.draft_input
    destination.welm_mtp_root_only_verify_mask = torch.tensor(
        [False, True, False]
    ).as_subclass(_NoRootPredicateTensor)
    calls = []
    graph_topk_p = torch.empty((2, 1), dtype=torch.float32)
    graph_topk_index = torch.empty((2, 1), dtype=torch.int64)
    graph_hidden_states = torch.empty((2, 1), dtype=torch.float32)
    graph_oe_history = torch.empty((2, 2), dtype=torch.int64)

    def run_continuation(subset_batch, subset_result):
        calls.append(
            (
                [req.origin_input_ids for req in subset_batch.reqs],
                subset_result.accept_lens.tolist(),
                subset_batch.forward_mode,
            )
        )
        draft_input = subset_result.draft_continuation_state.draft_input
        row_values = torch.tensor(
            [[float(10 * (req.req_pool_idx + 1))] for req in subset_batch.reqs]
        )
        rows = row_values.shape[0]
        graph_topk_p[:rows].copy_(row_values / 100)
        graph_topk_index[:rows].copy_(row_values)
        graph_hidden_states[:rows].copy_(row_values)
        graph_oe_history[:rows].copy_(row_values.to(torch.int64).expand(-1, 2))
        draft_input.topk_p = graph_topk_p[:rows]
        draft_input.topk_index = graph_topk_index[:rows]
        draft_input.hidden_states = graph_hidden_states[:rows]
        draft_input.welm_mtp_oe_history_state = graph_oe_history[:rows]

    worker._draft_extend_for_decode_unpartitioned = run_continuation
    worker._materialize_welmv4_mtp_deferred_prefill_draft = MagicMock(
        side_effect=AssertionError("seed must use ordinary continuation")
    )

    worker._draft_extend_for_decode(batch, batch_result)

    assert calls == [
        (
            [[1, 2, 10], [30, 31, 32, 33, 34]],
            [2, 3],
            batch.forward_mode,
        ),
        ([[20, 21, 22, 23]], [1], batch.forward_mode),
    ]
    torch.testing.assert_close(
        destination.topk_index, torch.tensor([[10], [20], [30]])
    )
    torch.testing.assert_close(
        destination.hidden_states, torch.tensor([[10.0], [20.0], [30.0]])
    )
    torch.testing.assert_close(
        destination.welm_mtp_oe_history_state,
        torch.tensor([[10, 10], [20, 20], [30, 30]]),
    )
    assert destination.welm_mtp_root_only_verify_mask is None


def test_root_only_continuation_dp_counts_are_rank_coordinated():
    worker = _continuation_worker()
    batch, _ = _continuation_inputs()
    batch.global_num_reqs = [3, 1]
    batch.welm_mtp_global_root_only_num_reqs = [1, 1]
    batch.welm_mtp_global_root_only_num_tokens = [1, 1]

    with (
        patch(
            "sglang.srt.speculative.eagle_worker_v2.is_dp_attention_enabled",
            return_value=True,
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.get_attention_dp_rank",
            return_value=0,
        ),
    ):
        counts = worker._get_welmv4_mtp_continuation_partition_counts(
            batch, ordinary_rows=2, seed_rows=1
        )

    assert counts == ([2, 0], [1, 1])


def test_root_only_continuation_dp_requires_global_partition_counts():
    worker = _continuation_worker()
    worker.topk = 1
    worker.speculative_num_steps = 3
    worker._is_welmv4_mtp_draft_model = lambda: True
    batch, batch_result = _continuation_inputs()
    batch.global_num_reqs = [3, 1]
    batch.welm_mtp_root_only_rows = [False, True, False]
    batch_result.draft_continuation_state.draft_input.welm_mtp_root_only_verify_mask = (
        torch.tensor([False, True, False])
    )

    with (
        patch(
            "sglang.srt.speculative.eagle_worker_v2.is_dp_attention_enabled",
            return_value=True,
        ),
        pytest.raises(RuntimeError, match="synchronized partition counts"),
    ):
        worker._draft_extend_for_decode(batch, batch_result)


def test_root_only_continuation_dp_rank_without_seed_joins_both_phases():
    worker = _continuation_worker()
    worker.topk = 1
    worker.speculative_num_steps = 3
    worker._is_welmv4_mtp_draft_model = lambda: True
    batch, batch_result = _continuation_inputs()
    batch.global_num_reqs = [3, 1]
    batch.welm_mtp_global_root_only_num_reqs = [0, 1]
    batch.welm_mtp_global_root_only_num_tokens = [0, 1]
    calls = []

    def run_continuation(subset_batch, result):
        calls.append(
            (
                subset_batch.forward_mode,
                subset_batch.is_extend_in_batch,
                subset_batch.all_extend_in_batch,
                subset_batch.welm_kv_mirror_contract_flags,
            )
        )
        draft_input = result.draft_continuation_state.draft_input
        rows = len(subset_batch.reqs)
        draft_input.topk_p = torch.ones((rows, 1))
        draft_input.topk_index = torch.tensor([[10], [20], [30]])[:rows]
        draft_input.hidden_states = torch.ones((rows, 1))

    worker._get_welmv4_mtp_continuation_partition_counts = lambda *args, **kwargs: (
        [3, 0],
        [0, 1],
    )
    worker._draft_extend_for_decode_unpartitioned = run_continuation
    worker._materialize_welmv4_mtp_deferred_prefill_draft = MagicMock(
        side_effect=AssertionError("seed must use ordinary continuation")
    )

    with (
        patch(
            "sglang.srt.speculative.eagle_worker_v2.is_dp_attention_enabled",
            return_value=True,
        ),
        patch(
            "sglang.srt.speculative.eagle_worker_v2.get_attention_dp_rank",
            return_value=0,
        ),
    ):
        worker._draft_extend_for_decode(batch, batch_result)

    assert calls[0][0] is batch.forward_mode
    assert calls[1][0].is_idle()
    assert calls[1][1:] == (False, False, [False, False])
    torch.testing.assert_close(
        batch_result.draft_continuation_state.draft_input.topk_index,
        torch.tensor([[10], [20], [30]]),
    )
