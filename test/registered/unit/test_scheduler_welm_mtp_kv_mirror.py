from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def test_null_mode_skips_welm_mtp_kv_mirror_state_buffers():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(enable_welm_kv_mirror_opt=True)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler._get_draft_model_runner = MagicMock()

    with patch.object(
        scheduler_module, "make_welmv4_mtp_kv_mirror_state_buffers"
    ) as make_buffers:
        Scheduler._init_welm_mtp_kv_mirror_state_buffers(scheduler)

    assert scheduler.welm_mtp_kv_mirror_state_buffers is None
    scheduler._get_draft_model_runner.assert_not_called()
    make_buffers.assert_not_called()


@pytest.mark.parametrize(
    "disaggregation_mode",
    [DisaggregationMode.PREFILL, DisaggregationMode.DECODE],
)
def test_pd_modes_initialize_welm_mtp_kv_mirror_state_buffers(
    disaggregation_mode,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(enable_welm_kv_mirror_opt=True)
    scheduler.disaggregation_mode = disaggregation_mode
    draft_runner = object()
    scheduler._get_draft_model_runner = MagicMock(return_value=draft_runner)

    token_to_kv_pool = SimpleNamespace(size=128, page_size=8)
    get_kvcache = MagicMock(return_value=token_to_kv_pool)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=get_kvcache)

    inner_draft_worker = SimpleNamespace()
    scheduler.draft_worker = SimpleNamespace(draft_worker=inner_draft_worker)
    buffers = {"48.k": object(), "48.v": object()}

    with patch.object(
        scheduler_module,
        "make_welmv4_mtp_kv_mirror_state_buffers",
        return_value=buffers,
    ) as make_buffers:
        Scheduler._init_welm_mtp_kv_mirror_state_buffers(scheduler)

    scheduler._get_draft_model_runner.assert_called_once_with()
    get_kvcache.assert_called_once_with()
    make_buffers.assert_called_once_with(draft_runner, 136)
    assert scheduler.welm_mtp_kv_mirror_state_buffers is buffers
    assert inner_draft_worker.welm_mtp_kv_mirror_state_buffers is buffers
