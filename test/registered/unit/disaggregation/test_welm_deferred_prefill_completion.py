from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from sglang.srt.disaggregation import prefill as prefill_module
from sglang.srt.managers import tp_worker as tp_worker_module
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.model_runner import ModelRunnerOutput
from sglang.srt.models import welmv4 as welmv4_model
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _completion():
    return forward_batch_info.WelmDeferredPrefillCompletion()


def test_generation_result_preserves_legacy_positional_field_order():
    logits_output = object()
    pp_proxy = object()
    next_token_ids = torch.tensor([7], dtype=torch.int64)

    result = GenerationBatchResult(logits_output, pp_proxy, next_token_ids)

    assert result.logits_output is logits_output
    assert result.pp_hidden_states_proxy_tensors is pp_proxy
    assert result.next_token_ids is next_token_ids
    assert result.welm_deferred_prefill_completion is None


@pytest.mark.parametrize("omit_final_output", [True, False])
def test_deferred_prefill_model_returns_typed_completion_without_logits(
    omit_final_output,
):
    class FakeBaseModel(nn.Module):
        scale_seq_times = 0

        def forward(self, *_args, **_kwargs):
            return torch.ones((2, 4), dtype=torch.bfloat16)

    model = welmv4_model.WeLMV4MoeForCausalLM.__new__(
        welmv4_model.WeLMV4MoeForCausalLM
    )
    nn.Module.__init__(model)
    model.model = FakeBaseModel()
    model.deferred_execution = SimpleNamespace(
        omit_final_output=omit_final_output,
        role=(
            welmv4_model.WelmDeferredExecutionRole.PREFILL
            if omit_final_output
            else welmv4_model.WelmDeferredExecutionRole.MONOLITHIC
        ),
    )
    model.pp_group = SimpleNamespace(is_last_rank=True)
    model.logits_processor = MagicMock(
        side_effect=AssertionError("deferred Prefill must not compute logits")
    )
    model.lm_head = MagicMock(
        side_effect=AssertionError("deferred Prefill must not invoke LM head")
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda **_kwargs: True),
        welm_deferred_prefill=True,
        spec_info=None,
        enable_welm_kv_mirror_opt=False,
    )

    output = model(
        torch.tensor([1, 2], dtype=torch.int64),
        torch.tensor([0, 1], dtype=torch.int64),
        forward_batch,
    )

    assert isinstance(output, forward_batch_info.WelmDeferredPrefillCompletion)
    model.logits_processor.assert_not_called()
    model.lm_head.assert_not_called()


def test_post_forward_sync_restores_batch_without_slicing_completion():
    original_forward_mode = SimpleNamespace(is_extend=lambda **_kwargs: True)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda **_kwargs: False),
        batch_size=4,
        _original_forward_mode=original_forward_mode,
        _original_batch_size=1,
        spec_info=None,
        welm_deferred_prefill=True,
        seq_lens_sum=2,
    )

    forward_batch_info.ForwardBatch.post_forward_mlp_sync_batch(
        forward_batch,
        _completion(),
    )

    assert forward_batch.forward_mode is original_forward_mode
    assert forward_batch.batch_size == 1


def test_post_forward_sync_rejects_unmarked_deferred_completion():
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda **_kwargs: True),
        batch_size=1,
        spec_info=None,
        welm_deferred_prefill=False,
    )

    with pytest.raises(RuntimeError, match="explicitly marked"):
        forward_batch_info.ForwardBatch.post_forward_mlp_sync_batch(
            forward_batch,
            _completion(),
        )


def test_tp_worker_skips_sampler_and_returns_only_internal_bookkeeping_ids(monkeypatch):
    completion = _completion()
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.tensor([11, 12], dtype=torch.int64),
    )
    model_worker_batch = SimpleNamespace(
        hicache_consumer_index=None,
        return_logprob=False,
        return_hidden_states=False,
        seq_lens=[1, 1],
        input_ids=forward_batch.input_ids,
        sampling_info=SimpleNamespace(grammars=None),
        is_prefill_only=False,
    )
    monkeypatch.setattr(
        tp_worker_module.ForwardBatch,
        "init_new",
        lambda *_args, **_kwargs: forward_batch,
    )
    sample = MagicMock(side_effect=AssertionError("sampler must not run"))
    worker = SimpleNamespace(
        set_hicache_consumer=lambda *_args: None,
        is_dllm=lambda: False,
        pp_group=SimpleNamespace(is_last_rank=True),
        model_runner=SimpleNamespace(
            forward=lambda *_args, **_kwargs: ModelRunnerOutput(
                logits_output=completion,
                can_run_graph=False,
            ),
            sample=sample,
        ),
        enable_overlap=False,
        enable_spec=False,
    )

    result = tp_worker_module.TpModelWorker.forward_batch_generation(
        worker,
        model_worker_batch,
    )

    assert result.logits_output is None
    assert result.welm_deferred_prefill_completion is completion
    assert torch.equal(result.next_token_ids, torch.zeros(2, dtype=torch.int64))
    sample.assert_not_called()


def _request(*, chunked: int):
    return SimpleNamespace(
        rid="req-0",
        is_chunked=chunked,
        output_ids=[],
        return_logprob=False,
        grammar=MagicMock(),
        welm_deferred_prefill_span=object(),
        time_stats=SimpleNamespace(
            set_prefill_finished_time=MagicMock(),
            set_prefill_transfer_queue_entry_time=MagicMock(),
            set_last_chunked_prefill_finish_time=MagicMock(),
            last_forward_entry_time=None,
            forward_entry_time=None,
            prefill_finished_time=None,
        ),
        tmp_end_idx=16,
    )


def _batch(req):
    return SimpleNamespace(
        reqs=[req],
        return_logprob=False,
        return_hidden_states=False,
        prefill_stats=None,
        dp_cooperation_info=None,
        forward_mode="extend",
    )


def _scheduler(*, overlap: bool):
    scheduler = prefill_module.SchedulerDisaggregationPrefillMixin()
    scheduler.enable_overlap = overlap
    scheduler.tree_cache = object()
    scheduler.disagg_prefill_inflight_queue = []
    scheduler.send_kv_chunk = MagicMock()
    scheduler.report_prefill_stats = MagicMock()
    scheduler.tp_rank = 0
    scheduler.pp_rank = 0
    return scheduler


def _result():
    return GenerationBatchResult(
        logits_output=None,
        next_token_ids=torch.tensor([987654321], dtype=torch.int64),
        welm_deferred_prefill_completion=_completion(),
    )


def test_prefill_completion_transfers_kv_without_output_or_grammar(monkeypatch):
    req = _request(chunked=0)
    batch = _batch(req)
    scheduler = _scheduler(overlap=False)
    cache_req = MagicMock()
    monkeypatch.setattr(prefill_module, "maybe_cache_unfinished_req", cache_req)
    monkeypatch.setattr(prefill_module, "hicache_timing_enabled", lambda: False)

    scheduler.process_batch_result_disagg_prefill(batch, _result())

    assert req.output_ids == []
    req.grammar.accept_token.assert_not_called()
    cache_req.assert_called_once_with(req, scheduler.tree_cache)
    scheduler.send_kv_chunk.assert_called_once_with(req, last_chunk=True)
    assert scheduler.disagg_prefill_inflight_queue == [req]
    req.time_stats.set_prefill_transfer_queue_entry_time.assert_called_once_with()


def test_nonfinal_chunk_cannot_emit_deferred_completion(monkeypatch):
    req = _request(chunked=1)
    batch = _batch(req)
    scheduler = _scheduler(overlap=True)
    monkeypatch.setattr(prefill_module, "hicache_timing_enabled", lambda: False)

    scheduler.process_batch_result_disagg_prefill(batch, _result())

    assert req.is_chunked == 0
    assert req.output_ids == []
    assert scheduler.disagg_prefill_inflight_queue == []
    scheduler.send_kv_chunk.assert_called_once_with(
        req,
        last_chunk=False,
        end_idx=16,
    )
    req.time_stats.set_last_chunked_prefill_finish_time.assert_called_once_with()


def test_overlap_bookkeeping_ids_stay_out_of_request_and_wire(monkeypatch):
    req = _request(chunked=0)
    batch = _batch(req)
    scheduler = _scheduler(overlap=True)
    result = _result()
    result.copy_done = MagicMock()
    future_map = FutureMap(
        max_running_requests=4,
        chunked_prefill_size=16,
        context_len=64,
        device=torch.device("cpu"),
        spec_algo=SimpleNamespace(is_none=lambda: True),
    )
    future_indices = future_map.alloc_future_indices(1)
    monkeypatch.setattr(prefill_module, "maybe_cache_unfinished_req", MagicMock())
    monkeypatch.setattr(prefill_module, "hicache_timing_enabled", lambda: False)

    future_map.store_to_map(future_indices, result)
    result.copy_to_cpu(return_logprob=False, return_hidden_states=False)
    scheduler.process_batch_result_disagg_prefill(batch, result)

    assert future_map.token_ids_buf[future_indices.indices].tolist() == [987654321]
    assert req.output_ids == []
    req.grammar.accept_token.assert_not_called()
    scheduler.send_kv_chunk.assert_called_once_with(req, last_chunk=True)


@pytest.mark.parametrize(
    "return_logprob,return_hidden_states",
    [(True, False), (False, True)],
)
def test_deferred_prefill_completion_rejects_output_payload_requests(
    return_logprob, return_hidden_states
):
    result = _result()
    result.copy_done = MagicMock()

    with pytest.raises(RuntimeError, match="does not support.*output payload"):
        result.copy_to_cpu(
            return_logprob=return_logprob,
            return_hidden_states=return_hidden_states,
        )
