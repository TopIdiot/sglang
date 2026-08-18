import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import (
    Scheduler,
    _should_process_cache_hit_extend_before_schedule,
)
from sglang.srt.managers.scheduler_dp_attn_mixin import (
    MLPSyncBatchInfo,
    _has_cache_hit_extend,
    _has_non_greedy_sampling,
    _needs_top_p_sampling,
    _will_contract_welm_kv_mirror,
    prepare_mlp_sync_batch_raw,
)


class _ForwardMode:
    def __init__(self, *, is_extend: bool):
        self._is_extend = is_extend

    def is_extend(self) -> bool:
        return self._is_extend

    def is_extend_without_speculative(self) -> bool:
        return self._is_extend

    def is_decode(self) -> bool:
        return not self._is_extend

    def is_decode_or_idle(self) -> bool:
        return not self._is_extend

    def is_prebuilt(self) -> bool:
        return False


def _batch(
    *,
    is_extend: bool,
    cached_tokens: int = 0,
    has_cache_hit_extend_in_batch: bool = False,
    return_logprob: bool = False,
    extend_len: int = 1,
    logprob_start_len: int = 1,
    decoding_cached_tokens=None,
    welm_deferred_prefill: bool = False,
    welm_deferred_prefill_flags=None,
):
    reqs = [
        SimpleNamespace(
            cached_tokens=cached_tokens,
            sampling_params=SimpleNamespace(top_k=1, top_p=1.0),
        )
    ]
    decoding_reqs = None
    if decoding_cached_tokens is not None:
        decoding_reqs = [
            SimpleNamespace(
                cached_tokens=decoding_cached_tokens,
                sampling_params=SimpleNamespace(top_k=1, top_p=1.0),
            )
        ]
        reqs.extend(decoding_reqs)

    batch = SimpleNamespace(
        forward_mode=_ForwardMode(is_extend=is_extend),
        is_extend_in_batch=is_extend,
        has_cache_hit_extend_in_batch=has_cache_hit_extend_in_batch,
        reqs=reqs,
        decoding_reqs=decoding_reqs,
        return_logprob=return_logprob,
        extend_lens=[extend_len],
        extend_logprob_start_lens=[logprob_start_len],
        is_spec_v2=False,
        has_grammar=False,
        welm_deferred_prefill=welm_deferred_prefill,
        welm_deferred_prefill_flags=welm_deferred_prefill_flags,
        extend_num_tokens=extend_len,
        spec_info=None,
        router_replay_topk_ids=None,
        device=torch.device("cpu"),
    )
    batch.batch_size = lambda: len(batch.reqs)
    batch.has_router_replay = lambda: False
    return batch


def _scheduler(
    *,
    attn_cp_size: int = 2,
    attn_cp_mode: str = "sharded-kv",
    last_batch=None,
    has_result: bool = True,
):
    return SimpleNamespace(
        require_mlp_sync=False,
        attn_cp_size=attn_cp_size,
        server_args=SimpleNamespace(attn_cp_mode=attn_cp_mode),
        result_queue=[object()] if has_result else [],
        last_batch=last_batch,
    )


class TestSchedulerOverlap(unittest.TestCase):
    def test_sampling_intent_is_available_before_sampling_info_is_built(self):
        batch = SimpleNamespace(
            sampling_info=None,
            reqs=[
                SimpleNamespace(sampling_params=SimpleNamespace(top_k=1000, top_p=0.95))
            ],
        )

        self.assertTrue(_has_non_greedy_sampling(batch))
        self.assertTrue(_needs_top_p_sampling(batch))

        batch.reqs[0].sampling_params.top_k = 1
        batch.reqs[0].sampling_params.top_p = 1.0
        self.assertFalse(_has_non_greedy_sampling(batch))
        self.assertFalse(_needs_top_p_sampling(batch))

    def test_only_extend_batches_publish_cache_hit_flag(self):
        self.assertFalse(
            _has_cache_hit_extend(_batch(is_extend=False, cached_tokens=8192))
        )
        self.assertTrue(
            _has_cache_hit_extend(_batch(is_extend=True, cached_tokens=8192))
        )

    def test_mixed_batch_only_checks_extend_requests_for_cache_hits(self):
        self.assertFalse(
            _has_cache_hit_extend(
                _batch(
                    is_extend=True,
                    cached_tokens=0,
                    decoding_cached_tokens=8192,
                )
            )
        )
        self.assertTrue(
            _has_cache_hit_extend(
                _batch(
                    is_extend=True,
                    cached_tokens=8192,
                    decoding_cached_tokens=0,
                )
            )
        )

    def test_dp_cache_hit_extend_decision_uses_global_flag(self):
        # A decode request keeps its prefix-cache accounting after prefill.  It
        # must not make only that DP rank process the pending result early.
        decode_batch = _batch(is_extend=False, cached_tokens=8192)
        self.assertFalse(
            _should_process_cache_hit_extend_before_schedule(
                decode_batch,
                has_pending_result=True,
                require_mlp_sync=True,
            )
        )

        # If any peer actually has a cache-hit extend, all DP ranks receive the
        # synchronized flag and take the early-processing path together.
        decode_batch.has_cache_hit_extend_in_batch = True
        self.assertTrue(
            _should_process_cache_hit_extend_before_schedule(
                decode_batch,
                has_pending_result=True,
                require_mlp_sync=True,
            )
        )

    def test_deferred_prefill_uses_native_cache_hit_overlap_rules(self):
        self.assertFalse(
            _should_process_cache_hit_extend_before_schedule(
                _batch(is_extend=True, welm_deferred_prefill=True),
                has_pending_result=True,
                require_mlp_sync=False,
            )
        )
        self.assertTrue(
            _should_process_cache_hit_extend_before_schedule(
                _batch(
                    is_extend=True,
                    cached_tokens=8192,
                    welm_deferred_prefill=True,
                ),
                has_pending_result=True,
                require_mlp_sync=False,
            )
        )

    def test_dp_deferred_prefill_keeps_native_overlap(self):
        peer_deferred_batch = _batch(
            is_extend=False,
            welm_deferred_prefill_flags=[False, True],
        )

        self.assertFalse(
            _should_process_cache_hit_extend_before_schedule(
                peer_deferred_batch,
                has_pending_result=True,
                require_mlp_sync=True,
            )
        )

    def test_mlp_sync_packs_cache_hit_extend_in_existing_flags_word(self):
        info = MLPSyncBatchInfo(
            dp_size=2,
            tp_size=1,
            cp_size=1,
            num_tokens=1,
            num_tokens_for_logprob=1,
            num_reqs=1,
            can_cuda_graph=False,
            is_extend_in_batch=True,
            local_can_run_tbo=True,
            local_forward_mode=2,
            has_router_replay=True,
            has_cache_hit_extend=True,
            will_contract_welm_kv_mirror=True,
            local_has_non_greedy_sampling=True,
            local_needs_top_p_sampling=True,
        )

        # All booleans share the already-gathered flags element, so the fix
        # does not add a collective or increase its payload.
        row = info._get_local_tensor(device="cpu")
        self.assertEqual(row[6].item(), 55)

        info._finish_parse(row.unsqueeze(0))
        self.assertTrue(info.global_has_non_greedy_sampling)
        self.assertTrue(info.global_needs_top_p_sampling)

    def test_mlp_sync_packs_deferred_prefill_in_existing_flags_word(self):
        info = MLPSyncBatchInfo(
            dp_size=2,
            tp_size=1,
            cp_size=1,
            num_tokens=16,
            num_tokens_for_logprob=1,
            num_reqs=1,
            can_cuda_graph=False,
            is_extend_in_batch=True,
            local_can_run_tbo=False,
            local_forward_mode=2,
            has_router_replay=True,
            has_cache_hit_extend=True,
            will_contract_welm_kv_mirror=True,
            is_welm_deferred_prefill=True,
        )

        local_record = info._get_local_tensor(device="cpu")
        info.is_welm_deferred_prefill = False
        baseline_record = info._get_local_tensor(device="cpu")

        self.assertEqual(local_record.numel(), baseline_record.numel())
        self.assertEqual(local_record[6].item(), baseline_record[6].item() | 8)

    def test_mlp_sync_packs_root_only_mtp_partition_counts(self):
        info = MLPSyncBatchInfo(
            dp_size=2,
            tp_size=1,
            cp_size=1,
            num_tokens=1,
            num_tokens_for_logprob=1,
            num_reqs=1,
            can_cuda_graph=True,
            is_extend_in_batch=False,
            local_can_run_tbo=True,
            local_forward_mode=1,
            welm_mtp_root_only_num_reqs=1,
            welm_mtp_root_only_num_tokens=8,
        )

        local_record = info._get_local_tensor(device="cpu")
        self.assertEqual(local_record[11].item(), (8 << 32) | 1)

        peer_record = local_record.clone()
        peer_record[11] = 0
        info._finish_parse(torch.stack((local_record, peer_record)))
        self.assertNotIn("is_welm_mtp_root_only", info.__dataclass_fields__)
        self.assertFalse(hasattr(info, "welm_mtp_root_only_flags"))
        self.assertEqual(info.welm_mtp_global_root_only_num_reqs, [1, 0])
        self.assertEqual(info.welm_mtp_global_root_only_num_tokens, [8, 0])

    @mock.patch("sglang.srt.managers.scheduler_dp_attn_mixin.get_tp_group")
    @mock.patch("torch.distributed.all_gather_into_tensor")
    def test_mlp_sync_unpacks_request_aligned_deferred_flags(
        self, all_gather_into_tensor, get_tp_group
    ):
        info = MLPSyncBatchInfo(
            dp_size=2,
            tp_size=1,
            cp_size=1,
            num_tokens=16,
            num_tokens_for_logprob=1,
            num_reqs=1,
            can_cuda_graph=False,
            is_extend_in_batch=True,
            local_can_run_tbo=False,
            local_forward_mode=2,
            is_welm_deferred_prefill=True,
        )

        def gather(output, local, group):
            del group
            records = output.view(2, 1, local.numel())
            records[0, 0].copy_(local)
            records[1, 0].copy_(local)
            records[1, 0, 6] = 0

        all_gather_into_tensor.side_effect = gather
        get_tp_group.return_value.active_ranks_cpu = torch.ones(2, dtype=torch.int32)

        info.all_gather(device="cpu", group=object())

        self.assertEqual(info.welm_deferred_prefill_flags, [True, False])

    @mock.patch("sglang.srt.managers.scheduler_dp_attn_mixin.TboDPAttentionPreparer")
    @mock.patch.object(MLPSyncBatchInfo, "all_gather", autospec=True)
    def test_deferred_prefill_disables_local_tbo_before_gather(
        self, all_gather, tbo_preparer_cls
    ):
        captured_local_tbo = []

        def gather(info, *, device, group):
            del device, group
            captured_local_tbo.append(info.local_can_run_tbo)
            info.global_num_tokens = [info.num_tokens, 1]
            info.global_num_tokens_for_logprob = [info.num_tokens_for_logprob, 1]
            info.global_num_reqs = [info.num_reqs, 1]
            info.global_forward_modes = [info.local_forward_mode] * 2
            info.welm_kv_mirror_contract_flags = [False, False]
            info.welm_deferred_prefill_flags = [info.is_welm_deferred_prefill, False]
            info.welm_mtp_global_prefill_num_tokens = [0, 0]
            info.global_has_non_greedy_sampling = False
            info.global_needs_top_p_sampling = False
            info.tp0_info = torch.zeros((2, 8), dtype=torch.int64)

        all_gather.side_effect = gather
        tbo_preparer = tbo_preparer_cls.return_value
        tbo_preparer.prepare_all_gather.return_value = (True, 2)
        tbo_preparer.compute_output.return_value = (None, None)
        tp_group = SimpleNamespace(
            cpu_group=object(), device_group=object(), device="cpu"
        )

        for deferred, is_extend, expected in (
            (True, True, False),
            (False, False, True),
        ):
            with self.subTest(deferred=deferred):
                prepare_mlp_sync_batch_raw(
                    _batch(
                        is_extend=is_extend,
                        welm_deferred_prefill=deferred,
                    ),
                    dp_size=2,
                    attn_tp_size=1,
                    attn_cp_size=1,
                    tp_group=tp_group,
                    get_idle_batch=lambda: None,
                    disable_cuda_graph=False,
                    require_mlp_tp_gather=True,
                    disable_overlap_schedule=True,
                    offload_tags=set(),
                )
                self.assertIs(captured_local_tbo[-1], expected)

    def test_input_logprob_extend_does_not_publish_contract_flag(self):
        self.assertFalse(
            _will_contract_welm_kv_mirror(
                _batch(
                    is_extend=True,
                    return_logprob=True,
                    extend_len=1,
                    logprob_start_len=0,
                )
            )
        )
        self.assertTrue(
            _will_contract_welm_kv_mirror(
                _batch(
                    is_extend=True,
                    return_logprob=True,
                    extend_len=1,
                    logprob_start_len=1,
                )
            )
        )

    def test_attncp_sharded_kv_serializes_prefill_after_decode(self):
        scheduler = _scheduler(last_batch=_batch(is_extend=False))

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertTrue(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=True)
                )
            )

    def test_attncp_sharded_kv_serializes_decode_after_prefill(self):
        scheduler = _scheduler(last_batch=_batch(is_extend=True))

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertTrue(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=False)
                )
            )

    def test_attncp_sharded_kv_serializes_consecutive_prefills(self):
        scheduler = _scheduler(last_batch=_batch(is_extend=True))

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertTrue(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=True)
                )
            )

    def test_attncp_sharded_kv_keeps_decode_decode_overlap(self):
        scheduler = _scheduler(last_batch=_batch(is_extend=False))

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertFalse(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=False)
                )
            )

    def test_non_attncp_does_not_disable_decode_prefill_boundary(self):
        scheduler = _scheduler(
            attn_cp_size=1, attn_cp_mode="none", last_batch=_batch(is_extend=False)
        )

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertFalse(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=True)
                )
            )

    def test_attncp_sharded_kv_requires_pending_result_to_disable_overlap(self):
        scheduler = _scheduler(last_batch=_batch(is_extend=False), has_result=False)

        with envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.override(False):
            self.assertFalse(
                Scheduler.is_disable_overlap_for_batch(
                    scheduler, _batch(is_extend=True)
                )
            )


if __name__ == "__main__":
    unittest.main()
