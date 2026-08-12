from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def test_dp_extend_preserves_output_only_logprob_start():
    input_ids = list(range(8))
    req = SimpleNamespace(
        rid="output-only-logprob",
        _scale_seq_factor=4,
        prefix_indices=[],
        fill_ids=input_ids,
        origin_input_ids=input_ids,
        extend_input_len=len(input_ids),
        logprob_start_len=len(input_ids),
        seqlen=len(input_ids),
        retract_replay_skips_logprob=lambda: False,
    )
    batch = ScheduleBatch(reqs=[req], return_logprob=True, is_prefill_only=False)

    batch.prepare_extend_metadata_for_dp_sync()

    assert req.extend_logprob_start_len == req.extend_input_len
    assert batch.extend_logprob_start_lens == [req.extend_input_len]
