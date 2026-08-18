from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


@pytest.mark.parametrize(
    ("disaggregation_mode", "uses_draft_runner"),
    [
        (DisaggregationMode.PREFILL, False),
        (DisaggregationMode.DECODE, True),
    ],
)
def test_legacy_pd_compatibility_initializes_welm_mtp_kv_mirror_state_buffers(
    disaggregation_mode,
    uses_draft_runner,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(
        enable_welm_kv_mirror_opt=True,
        welm_kv_mirror_pd_mode="legacy",
    )
    scheduler.disaggregation_mode = disaggregation_mode
    scheduler._welm_mtp_legacy_mirror_state_enabled = True
    draft_config = SimpleNamespace(
        dtype=object(),
        hf_config=SimpleNamespace(
            architectures=["WeLMV4MoeForCausalLMNextN"]
        ),
    )
    draft_runner = SimpleNamespace(
        model_config=draft_config,
        device="cuda:1",
        tp_size=4,
    )
    scheduler._get_draft_model_runner = MagicMock(
        return_value=draft_runner if uses_draft_runner else None
    )
    scheduler.model_config = SimpleNamespace(
        dtype=object(),
        hf_config=SimpleNamespace(
            architectures=["WeLMV4MoeForCausalLM"],
            num_nextn_predict_layers=1,
        ),
    )
    scheduler.device = "cuda:0"
    scheduler.attn_tp_group = SimpleNamespace(world_size=4)
    scheduler.token_to_kv_pool_allocator = object()

    inner_draft_worker = SimpleNamespace() if uses_draft_runner else None
    scheduler.draft_worker = SimpleNamespace(
        draft_worker=inner_draft_worker,
        _is_welm_mtp_lightweight_prefill=not uses_draft_runner,
    )
    buffers = {"48.k": object(), "48.v": object()}
    spec = object()

    with (
        patch.object(
            scheduler_module,
            "build_welmv4_mtp_kv_mirror_buffer_spec",
            return_value=spec,
        ) as build_spec,
        patch.object(
            scheduler_module,
            "get_welmv4_mtp_kv_mirror_max_rows",
            return_value=136,
        ) as get_max_rows,
        patch.object(
            scheduler_module,
            "allocate_welmv4_mtp_kv_mirror_state_buffers",
            return_value=buffers,
        ) as allocate_buffers,
    ):
        Scheduler._init_welm_mtp_kv_mirror_state_buffers(scheduler)

    scheduler._get_draft_model_runner.assert_called_once_with()
    source_config = draft_config if uses_draft_runner else scheduler.model_config
    build_spec.assert_called_once_with(
        model_config=source_config,
        attention_tp_size=4,
        dtype=source_config.dtype,
        device="cuda:1" if uses_draft_runner else "cuda:0",
    )
    get_max_rows.assert_called_once_with(scheduler.token_to_kv_pool_allocator)
    allocate_buffers.assert_called_once_with(spec, max_rows=136)
    assert scheduler.welm_mtp_kv_mirror_state_buffers is buffers
    assert scheduler.draft_worker.welm_mtp_kv_mirror_state_buffers is buffers
    if inner_draft_worker is not None:
        assert inner_draft_worker.welm_mtp_kv_mirror_state_buffers is buffers


def test_legacy_monolithic_skips_welm_mtp_kv_mirror_state_buffers():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(
        enable_welm_kv_mirror_opt=True,
        welm_kv_mirror_pd_mode="legacy",
    )
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler._welm_mtp_legacy_mirror_state_enabled = False
    scheduler._get_draft_model_runner = MagicMock()

    with patch.object(
        scheduler_module, "allocate_welmv4_mtp_kv_mirror_state_buffers"
    ) as allocate_buffers:
        Scheduler._init_welm_mtp_kv_mirror_state_buffers(scheduler)

    assert scheduler.welm_mtp_kv_mirror_state_buffers is None
    scheduler._get_draft_model_runner.assert_not_called()
    allocate_buffers.assert_not_called()
