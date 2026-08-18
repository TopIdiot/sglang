from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers import schedule_batch as schedule_batch_module
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.penaltylib.frequency_penalty import (
    BatchedFrequencyPenalizer,
)
from sglang.srt.sampling.penaltylib.min_new_tokens import (
    BatchedMinNewTokensPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.penaltylib.presence_penalty import (
    BatchedPresencePenalizer,
)
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

VOCAB_SIZE = 32


class _PenaltyBatch:
    def __init__(self, reqs):
        self.reqs = reqs
        self.device = "cpu"


def _penalty_req():
    return SimpleNamespace(
        sampling_params=SimpleNamespace(
            frequency_penalty=1.5,
            presence_penalty=0.5,
            repetition_penalty=2.0,
            min_new_tokens=2,
            stop_token_ids={3},
        ),
        tokenizer=SimpleNamespace(
            additional_stop_token_ids=None,
            eos_token_id=2,
        ),
    )


def _penalty_orchestrator():
    batch = _PenaltyBatch([_penalty_req(), _penalty_req()])
    orchestrator = BatchedPenalizerOrchestrator(
        VOCAB_SIZE,
        batch,
        {
            BatchedFrequencyPenalizer,
            BatchedMinNewTokensPenalizer,
            BatchedPresencePenalizer,
            BatchedRepetitionPenalizer,
        },
    )
    orchestrator._test_batch = batch
    return orchestrator


def test_row_active_mask_skips_all_stateful_penalties_for_seed_row():
    orchestrator = _penalty_orchestrator()

    orchestrator.cumulate_output_tokens(
        torch.tensor([5, 6], dtype=torch.int64),
        row_active_mask=torch.tensor([True, False]),
    )

    frequency = orchestrator.penalizers[BatchedFrequencyPenalizer]
    presence = orchestrator.penalizers[BatchedPresencePenalizer]
    repetition = orchestrator.penalizers[BatchedRepetitionPenalizer]
    min_new_tokens = orchestrator.penalizers[BatchedMinNewTokensPenalizer]
    assert frequency.cumulated_frequency_penalties[0, 5].item() == 1.5
    assert torch.count_nonzero(frequency.cumulated_frequency_penalties[1]) == 0
    assert presence.cumulated_presence_penalties[0, 5].item() == 0.5
    assert torch.count_nonzero(presence.cumulated_presence_penalties[1]) == 0
    assert repetition.cumulated_repetition_penalties[0, 5].item() == 2.0
    assert torch.all(repetition.cumulated_repetition_penalties[1] == 1)
    assert min_new_tokens.len_output_tokens.flatten().tolist() == [1, 0]


def test_all_active_mask_matches_legacy_unmasked_penalty_updates():
    unmasked = _penalty_orchestrator()
    masked = _penalty_orchestrator()
    output_ids = torch.tensor([5, 6], dtype=torch.int64)

    unmasked.cumulate_output_tokens(output_ids)
    masked.cumulate_output_tokens(
        output_ids,
        row_active_mask=torch.ones((2,), dtype=torch.bool),
    )

    for penalizer_type in (
        BatchedFrequencyPenalizer,
        BatchedPresencePenalizer,
        BatchedRepetitionPenalizer,
        BatchedMinNewTokensPenalizer,
    ):
        lhs = unmasked.penalizers[penalizer_type]
        rhs = masked.penalizers[penalizer_type]
        for name, value in vars(lhs).items():
            if torch.is_tensor(value):
                assert torch.equal(value, getattr(rhs, name))


def _decode_req(rid, token_id, deferred_state=None):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[token_id],
        output_ids=[token_id] if deferred_state is None else [],
        welm_deferred_decode_state=deferred_state,
        req_pool_idx=0 if deferred_state is None else 1,
        kv_allocated_len=1 if deferred_state is None else 0,
        kv_committed_len=1 if deferred_state is None else 0,
        decode_batch_idx=0,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        input_embeds=None,
        attn_cp_prefill_split_spec=None,
        _scale_seq_factor=1,
    )


def test_prepare_for_decode_masks_ready_and_inflight_seed_but_cumulates_y0():
    state = schedule_batch_module.WelmDeferredDecodeState.from_prompt_tokens([13])
    state.transition_to(schedule_batch_module.WelmDeferredDecodePhase.READY)
    normal = _decode_req("normal", 91)
    seed = _decode_req("seed", 13, deferred_state=state)
    orchestrator = MagicMock(is_required=True)
    batch = ScheduleBatch(
        reqs=[normal, seed],
        model_config=SimpleNamespace(is_encoder_decoder=False),
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        device="cpu",
        enable_overlap=True,
        output_ids=torch.tensor([91, 13], dtype=torch.int64),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=torch.tensor([1, 0], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([1, 0], dtype=torch.int64),
        orig_seq_lens=torch.tensor([1, 0], dtype=torch.int32),
        seq_lens_sum=1,
        sampling_info=SimpleNamespace(penalizer_orchestrator=orchestrator),
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
            side_effect=[
                torch.tensor([20, 21], dtype=torch.int64),
                torch.tensor([22, 23], dtype=torch.int64),
                torch.tensor([24, 25], dtype=torch.int64),
            ],
        ),
    ):
        batch.prepare_for_decode()
        first_call = orchestrator.cumulate_output_tokens.call_args

        orchestrator.cumulate_output_tokens.reset_mock()
        normal.output_ids = [92]
        batch.output_ids = torch.tensor([-1, -2], dtype=torch.int64)
        batch.prepare_for_decode()
        inflight_call = orchestrator.cumulate_output_tokens.call_args

        orchestrator.cumulate_output_tokens.reset_mock()
        seed.output_ids = [42]
        state.transition_to(schedule_batch_module.WelmDeferredDecodePhase.CONSUMED)
        batch.output_ids = torch.tensor([-1, -2], dtype=torch.int64)
        batch.prepare_for_decode()
        consumed_call = orchestrator.cumulate_output_tokens.call_args

    assert first_call.args[0].tolist() == [91, 13]
    assert first_call.kwargs["row_active_mask"].tolist() == [True, False]
    assert inflight_call.args[0].tolist() == [92, 13]
    assert inflight_call.kwargs["row_active_mask"].tolist() == [True, False]
    assert consumed_call.args[0].tolist() == [92, 42]
    assert consumed_call.kwargs["row_active_mask"] is None


def test_spec_v2_prepare_reuses_deferred_penalty_mask_before_inflight():
    state = schedule_batch_module.WelmDeferredDecodeState.from_prompt_tokens([13])
    state.transition_to(schedule_batch_module.WelmDeferredDecodePhase.READY)
    normal = _decode_req("normal", 91)
    seed = _decode_req("seed", 13, deferred_state=state)
    spec_info = MagicMock()
    batch = ScheduleBatch(
        reqs=[normal, seed],
        model_config=SimpleNamespace(is_encoder_decoder=False),
        spec_algorithm=SimpleNamespace(
            is_none=lambda: False,
            supports_spec_v2=lambda: True,
        ),
        spec_info=spec_info,
        device="cpu",
        enable_overlap=True,
        output_ids=torch.tensor([91, 13], dtype=torch.int64),
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=torch.tensor([1, 0], dtype=torch.int64),
        seq_lens_cpu=torch.tensor([1, 0], dtype=torch.int64),
        orig_seq_lens=torch.tensor([1, 0], dtype=torch.int32),
        seq_lens_sum=1,
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=MagicMock(is_required=True)
        ),
    )

    batch.prepare_for_decode()

    spec_info.prepare_for_decode.assert_called_once()
    args, kwargs = spec_info.prepare_for_decode.call_args
    assert args == (batch,)
    assert kwargs["row_active_mask"].tolist() == [True, False]
    assert state.phase is schedule_batch_module.WelmDeferredDecodePhase.INFLIGHT
