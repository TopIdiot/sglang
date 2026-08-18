from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.utils import setup_state_kv_args
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from mtp_test_utils import (
    build_eagle_worker,
    model_config,
    worker_args,
)

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

@pytest.mark.parametrize("mode", ["legacy", "deferred-last-prompt"])
def test_lightweight_prefill_skips_draft_execution_resources(mode):
    with patch(
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker"
    ) as draft_cls:
        worker = build_eagle_worker(worker_args(welm_kv_mirror_pd_mode=mode))

    draft_cls.assert_not_called()
    assert worker.draft_worker is None
    assert worker._is_welm_mtp_lightweight_prefill
    assert worker.plan_stream is None
    assert worker.num_new_pages_per_topk is None
    assert worker.extend_lens is None


def test_lightweight_worker_runs_target_prefill_and_rejects_draft_paths():
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    worker = build_eagle_worker(worker_args(welm_kv_mirror_pd_mode="legacy"))
    expected_output = SimpleNamespace(
        logits_output=SimpleNamespace(
            hidden_states=torch.zeros((1, 4)),
            model_specific_states=None,
        ),
        next_token_ids=torch.tensor([7]),
        draft_continuation_state=None,
        welm_deferred_prefill_completion=None,
    )
    worker.target_worker.forward_batch_generation.return_value = expected_output
    prefill_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        is_extend_in_batch=False,
        seq_lens=torch.tensor([1]),
        extend_seq_lens=[1],
        out_cache_loc=torch.tensor([1]),
    )

    with patch.object(
        EagleDraftWorker,
        "_copy_welmv4_mtp_pd_prefill_mirror_states",
        side_effect=AssertionError("direct Prefill must not call raw mirror copy"),
    ):
        assert worker.forward_batch_generation(prefill_batch) is expected_output
    assert prefill_batch.capture_hidden_mode is CaptureHiddenMode.LAST
    assert expected_output.draft_continuation_state is not None
    assert (
        expected_output.draft_continuation_state.draft_input
        is prefill_batch.spec_info
    )
    worker.target_worker.forward_batch_generation.assert_called_once_with(prefill_batch)

    decode_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        is_extend_in_batch=False,
    )
    with pytest.raises(RuntimeError, match="role=prefill.*decode"):
        worker.forward_batch_generation(decode_batch)
    with pytest.raises(RuntimeError, match="role=prefill.*verify"):
        worker.verify(SimpleNamespace())


def test_deferred_mtp_decode_transfer_includes_execution_draft_pool():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = worker_args(disaggregation_mode="decode")
    scheduler.model_config = model_config()
    scheduler._welm_mtp_legacy_mirror_state_enabled = False
    execution_pool = object()
    draft_config = object()
    scheduler._get_execution_draft_kv_pool = MagicMock(
        return_value=(execution_pool, draft_config)
    )

    transfer_pool, transfer_config = Scheduler._get_disaggregation_draft_kv_pool(
        scheduler
    )

    assert transfer_pool is execution_pool
    assert transfer_config is draft_config


def test_deferred_mtp_prefill_transfer_includes_storage_draft_pool():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = worker_args(disaggregation_mode="prefill")
    scheduler.model_config = model_config()
    scheduler._welm_mtp_legacy_mirror_state_enabled = False
    storage_pool = object()
    draft_config = object()
    scheduler.draft_worker = SimpleNamespace(
        welm_mtp_storage_draft_kv_pool=storage_pool
    )
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            welm_mtp_storage_draft_model_config=draft_config
        )
    )
    scheduler._get_execution_draft_kv_pool = MagicMock(return_value=(None, None))

    transfer_pool, transfer_config = Scheduler._get_disaggregation_draft_kv_pool(
        scheduler
    )

    assert transfer_pool is storage_pool
    assert transfer_config is draft_config


def test_swa_transfer_descriptor_appends_draft_state_to_target():
    target_pool = SWAKVPool.__new__(SWAKVPool)
    target_pool.get_state_buf_infos = lambda: ([11], [12], [13])
    draft_pool = SWAKVPool.__new__(SWAKVPool)
    draft_pool.get_state_buf_infos = lambda: ([21], [22], [23])
    kv_args = SimpleNamespace()

    setup_state_kv_args(
        kv_args,
        token_to_kv_pool=target_pool,
        draft_token_to_kv_pool=draft_pool,
    )

    assert kv_args.state_types == [StateType.SWA]
    assert kv_args.state_data_ptrs == [[11, 21]]
    assert kv_args.state_data_lens == [[12, 22]]
    assert kv_args.state_item_lens == [[13, 23]]


def test_eagle_kv_offload_owner_saves_and_loads_target_and_draft():
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

    worker = EAGLEWorkerV2.__new__(EAGLEWorkerV2)
    worker.token_to_kv_pool_allocator = MagicMock()
    worker.token_to_kv_pool_allocator.get_cpu_copy.return_value = "target-cpu"
    draft_pool = MagicMock()
    draft_pool.get_cpu_copy.return_value = "draft-cpu"
    worker._draft_worker = SimpleNamespace(
        draft_runner=SimpleNamespace(token_to_kv_pool=draft_pool)
    )
    indices = torch.tensor([2, 5], dtype=torch.int64)

    payload = worker.get_cpu_copy(indices, mamba_indices=None)
    worker.load_cpu_copy(payload, indices, mamba_indices=None)

    assert payload == {"target": "target-cpu", "draft": "draft-cpu"}
    worker.token_to_kv_pool_allocator.get_cpu_copy.assert_called_once_with(
        indices, mamba_indices=None
    )
    draft_pool.get_cpu_copy.assert_called_once_with(indices)
    worker.token_to_kv_pool_allocator.load_cpu_copy.assert_called_once_with(
        "target-cpu", indices, mamba_indices=None
    )
    draft_pool.load_cpu_copy.assert_called_once_with("draft-cpu", indices)
