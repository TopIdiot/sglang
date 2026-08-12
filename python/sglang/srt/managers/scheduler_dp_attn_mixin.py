from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.metrics_collector import DPCooperationInfo
from sglang.srt.utils.common import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.managers.scheduler import Scheduler


_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()
_WELM_MTP_PREFILL_INFO_SHIFT = 32
_WELM_MTP_PREFILL_INFO_REQ_MASK = (1 << _WELM_MTP_PREFILL_INFO_SHIFT) - 1
_MLP_SYNC_FLAG_ROUTER_REPLAY = 1 << 0
_MLP_SYNC_FLAG_CACHE_HIT_EXTEND = 1 << 1
_MLP_SYNC_FLAG_WELM_KV_MIRROR_CONTRACT = 1 << 2
_MLP_SYNC_FLAG_WELM_DEFERRED_PREFILL = 1 << 3

# WeLM fused sched-sync (SGLANG_WELM_FUSED_SCHED_SYNC): the mlp-sync row is
# widened so one all_gather can also carry the per-round scheduling intent
# that previously required separate collectives (recv broadcasts + the
# dp-spec-prefill all_reduce). Words 0-7 keep their classic meaning; classic
# gathers simply carry zeros in the new words.
_MLP_SYNC_ROW_WORDS = 12
_WELM_FUSED_SCHED_SYNC_SCHEMA = 1

# Intent word (row index 8) bits. Any nonzero intent on any rank sends the
# whole round down the classic scheduling path.
_FUSED_INTENT_NEEDS_CLASSIC = 1 << 0  # prefill/chunked/grammar/prefill-only work
_FUSED_INTENT_PENDING_WORK = 1 << 1  # leader polled unbroadcast work requests
_FUSED_INTENT_PENDING_CONTROL = 1 << 2  # leader polled unbroadcast control msgs
_FUSED_INTENT_MAY_RETRACT = 1 << 3  # decode memory check failed / test retract


def _pack_welm_mtp_prefill_info(prefill_num_tokens: int, num_reqs: int) -> int:
    return (
        int(prefill_num_tokens) << _WELM_MTP_PREFILL_INFO_SHIFT
    ) | int(num_reqs)


def _unpack_welm_mtp_prefill_info(value: int) -> tuple[int, int]:
    value = int(value)
    return (
        value >> _WELM_MTP_PREFILL_INFO_SHIFT,
        value & _WELM_MTP_PREFILL_INFO_REQ_MASK,
    )


def _has_cache_hit_extend(batch: Optional[ScheduleBatch]) -> bool:
    if batch is None or not batch.forward_mode.is_extend():
        return False

    # mix_with_running() appends decode requests after the extend requests.
    # Their cached-token accounting must not turn a mixed batch into a
    # cache-hit extend batch.
    num_decoding_reqs = len(batch.decoding_reqs or ())
    extend_reqs = (
        batch.reqs[:-num_decoding_reqs] if num_decoding_reqs else batch.reqs
    )
    return any(getattr(req, "cached_tokens", 0) > 0 for req in extend_reqs)


def _will_contract_welm_kv_mirror(batch: Optional[ScheduleBatch]) -> bool:
    if batch is None or not batch.forward_mode.is_extend_without_speculative():
        return False
    if batch.welm_deferred_prefill:
        return False
    if not batch.return_logprob:
        return True
    return not any(
        int(extend_len) - int(start_len) > 0
        for extend_len, start_len in zip(
            batch.extend_lens,
            batch.extend_logprob_start_lens,
        )
    )


@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    cp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    num_reqs: int
    can_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int
    has_router_replay: bool = False
    has_cache_hit_extend: bool = False
    will_contract_welm_kv_mirror: bool = False
    is_welm_deferred_prefill: bool = False
    welm_mtp_prefill_num_tokens: int = 0
    # WeLM fused sched-sync extras (zero on classic gathers)
    fused_intent: int = 0
    fused_stamp: int = 0

    # some gathered elements
    tp0_info: torch.Tensor = None
    all_rows: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    global_num_reqs: list[int] = None
    global_forward_modes: list[int] = None
    welm_kv_mirror_contract_flags: list[bool] = None
    welm_deferred_prefill_flags: list[bool] = None
    welm_mtp_global_prefill_num_tokens: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None
    dp_cooperation_info: Optional[DPCooperationInfo] = None

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                int(self.can_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
                (
                    int(self.has_router_replay) * _MLP_SYNC_FLAG_ROUTER_REPLAY
                    | int(self.has_cache_hit_extend)
                    * _MLP_SYNC_FLAG_CACHE_HIT_EXTEND
                    | int(self.will_contract_welm_kv_mirror)
                    * _MLP_SYNC_FLAG_WELM_KV_MIRROR_CONTRACT
                    | int(self.is_welm_deferred_prefill)
                    * _MLP_SYNC_FLAG_WELM_DEFERRED_PREFILL
                ),
                _pack_welm_mtp_prefill_info(
                    self.welm_mtp_prefill_num_tokens,
                    self.num_reqs,
                ),
                self.fused_intent,
                self.fused_stamp,
                _WELM_FUSED_SCHED_SYNC_SCHEMA,
                0,  # reserved
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                0,  # num_tokens
                0,  # num_tokens_for_logprob
                1,  # can_cuda_graph
                0,  # is_extend_in_batch
                1,  # local_can_run_tbo
                ForwardMode.IDLE.value,  # local_forward_mode
                0,  # packed flags
                _pack_welm_mtp_prefill_info(0, 0),  # welm_mtp_prefill_info
                0,  # fused_intent
                self.fused_stamp,  # keep the stamp assert happy on substitution
                _WELM_FUSED_SCHED_SYNC_SCHEMA,
                0,  # reserved
            ],
            device=device,
            dtype=dtype,
        )

    def all_gather(self, device, group: torch.distributed.ProcessGroup):
        local_info_tensor = self._get_local_tensor(device=device)
        global_info_tensor = torch.empty(
            (self.dp_size, self.tp_size * self.cp_size, _MLP_SYNC_ROW_WORDS),
            dtype=torch.int64,
            device=device,
        )

        torch.distributed.all_gather_into_tensor(
            global_info_tensor.flatten(),
            local_info_tensor,
            group=group,
        )
        if device == "cpu":
            tp_active_ranks = get_tp_group().active_ranks_cpu
        else:
            tp_active_ranks = get_tp_group().active_ranks

        # Set fallback values for inactive ranks
        tp_info = global_info_tensor.view(
            self.dp_size * self.tp_size * self.cp_size, _MLP_SYNC_ROW_WORDS
        )
        tp_info[tp_active_ranks == 0] = self._get_fallback_tensor(device=device)
        self.all_rows = tp_info

        self._finish_parse(global_info_tensor[:, 0, :])

    def gather_hierarchical(
        self,
        leader_group: torch.distributed.ProcessGroup,
        attn_tp_cpu_group: torch.distributed.ProcessGroup,
        attn_tp_src: int,
        is_leader: bool,
    ):
        """Two-stage transport for the fused sched-sync gather.

        Within an attn-TP group every rank's row is mirrored (the flat path
        already consumes only the tp0 rows), so the dp-group leaders exchange
        one row per group over a small cross-node gloo group (dp_size-1 ring
        hops instead of world_size-1), then fan the (dp_size, W) result out
        over the intra-node attn-TP cpu group. Callers must run the stamp
        assert on all_rows AFTER this returns so leaders and members raise
        together on desync instead of hanging in the broadcast.

        Only valid when elastic EP is off (no inactive-rank fallback rows) —
        guaranteed by the fused sched-sync init qualification.
        """
        rows = torch.empty((self.dp_size, _MLP_SYNC_ROW_WORDS), dtype=torch.int64)
        if is_leader:
            local_info_tensor = self._get_local_tensor(device="cpu")
            torch.distributed.all_gather_into_tensor(
                rows.flatten(),
                local_info_tensor,
                group=leader_group,
            )
        if self.tp_size * self.cp_size > 1:
            torch.distributed.broadcast(
                rows, src=attn_tp_src, group=attn_tp_cpu_group
            )
        self.all_rows = rows
        self._finish_parse(rows)

    def _finish_parse(self, tp0_info: torch.Tensor):
        self.tp0_info = tp0_info
        # Perform only one Device-to-Host (D2H) memory copy
        cpu_data = tp0_info[:, [0, 1, 2, 3, 5, 6, 7]].cpu()
        self.global_num_tokens = cpu_data[:, 0].tolist()
        self.global_num_tokens_for_logprob = cpu_data[:, 1].tolist()
        self.can_cuda_graph = bool(cpu_data[:, 2].min().item())
        self.is_extend_in_batch = bool(cpu_data[:, 3].max().item())
        self.global_forward_modes = cpu_data[:, 4].tolist()
        packed_flags = cpu_data[:, 5].tolist()
        self.has_router_replay = any(
            value & _MLP_SYNC_FLAG_ROUTER_REPLAY for value in packed_flags
        )
        self.has_cache_hit_extend = any(
            value & _MLP_SYNC_FLAG_CACHE_HIT_EXTEND for value in packed_flags
        )
        self.welm_kv_mirror_contract_flags = [
            bool(value & _MLP_SYNC_FLAG_WELM_KV_MIRROR_CONTRACT)
            for value in packed_flags
        ]
        self.welm_deferred_prefill_flags = [
            bool(value & _MLP_SYNC_FLAG_WELM_DEFERRED_PREFILL)
            for value in packed_flags
        ]
        prefill_info = [
            _unpack_welm_mtp_prefill_info(value) for value in cpu_data[:, 6].tolist()
        ]
        self.welm_mtp_global_prefill_num_tokens = [
            item[0] for item in prefill_info
        ]
        self.global_num_reqs = [item[1] for item in prefill_info]
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(tp0_info[:, 5].tolist())


def _ensure_router_replay_gather_inputs(batch: ScheduleBatch, num_tokens: int):
    if batch.router_replay_topk_ids is not None:
        return

    cfg = batch.model_config.hf_text_config
    batch.router_replay_topk_ids = torch.zeros(
        (
            num_tokens,
            getattr(cfg, "num_hidden_layers"),
            getattr(cfg, "num_experts_per_tok"),
        ),
        dtype=torch.int32,
        device=batch.device,
    )
    batch.router_replay_mask = torch.zeros(
        (num_tokens,), dtype=torch.bool, device=batch.device
    )


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_all_gather=False,
):
    # TODO: handle the case when moe_dense_tp_size != 1
    if not require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
    if not skip_all_gather:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode
        batch.global_forward_modes = mlp_sync_info.global_forward_modes
        batch.global_num_reqs = mlp_sync_info.global_num_reqs
        batch.has_cache_hit_extend_in_batch = mlp_sync_info.has_cache_hit_extend
        batch.welm_kv_mirror_contract_flags = (
            mlp_sync_info.welm_kv_mirror_contract_flags
        )
        batch.welm_deferred_prefill_flags = (
            mlp_sync_info.welm_deferred_prefill_flags
        )
        batch.welm_mtp_global_prefill_num_tokens = (
            mlp_sync_info.welm_mtp_global_prefill_num_tokens
        )

    # Check forward mode for cuda graph
    batch.can_run_dp_cuda_graph = mlp_sync_info.can_cuda_graph

    if require_mlp_tp_gather and mlp_sync_info.has_router_replay:
        _ensure_router_replay_gather_inputs(batch, mlp_sync_info.num_tokens)


def _is_welm_mtp_intermediate_prefill_chunk(req) -> bool:
    if getattr(req, "is_chunked", 0) <= 0:
        return False

    fill_ids = getattr(req, "fill_ids", None)
    origin_input_ids = getattr(req, "origin_input_ids", None)
    if fill_ids is None or origin_input_ids is None:
        return True

    output_ids = getattr(req, "output_ids", ())
    return len(fill_ids) < len(origin_input_ids) + len(output_ids)


def _get_welm_mtp_prefill_num_tokens(local_batch: Optional[ScheduleBatch]) -> int:
    if local_batch is None or not local_batch.forward_mode.is_extend():
        return 0

    reqs = getattr(local_batch, "reqs", None)
    if not reqs:
        return 0

    if all(_is_welm_mtp_intermediate_prefill_chunk(req) for req in reqs):
        return 0

    return int(local_batch.extend_num_tokens or 0)


def compute_local_mlp_sync_info(
    local_batch: Optional[ScheduleBatch],
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    disable_cuda_graph: bool,
):
    """Compute this rank's mlp-sync row from local scheduler state.

    Shared by the classic prepare_mlp_sync_batch_raw path and the WeLM fused
    sched-sync fast path so both contribute bit-identical rows.
    Returns (mlp_sync_info, tbo_preparer); no collective is issued here.
    """
    # Check if other DP workers have running batches
    if local_batch is None or local_batch.forward_mode.is_prebuilt():
        num_tokens = 0
        num_tokens_for_logprob = 0
        num_reqs = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
        num_reqs = local_batch.batch_size()
    else:
        num_tokens = local_batch.extend_num_tokens
        num_reqs = local_batch.batch_size()
        if local_batch.return_logprob:
            scale = getattr(local_batch, "scale_seq_factor", 1) or 1
            num_tokens_for_logprob = sum(
                # We should have at least 1 token for sample in every case.
                max(extend_len // scale - logprob_start_len, 1)
                for logprob_start_len, extend_len in zip(
                    local_batch.extend_logprob_start_lens,
                    local_batch.extend_lens,
                )
            )
        else:
            num_tokens_for_logprob = local_batch.batch_size()

    can_cuda_graph = (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    ) and not disable_cuda_graph

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch
    has_router_replay = (
        local_batch is not None
        and (
            local_batch.router_replay_topk_ids is not None
            or local_batch.has_router_replay()
        )
    )
    has_cache_hit_extend = _has_cache_hit_extend(local_batch)
    will_contract_welm_kv_mirror = _will_contract_welm_kv_mirror(local_batch)
    is_welm_deferred_prefill = bool(
        local_batch is not None and local_batch.welm_deferred_prefill
    )
    welm_mtp_prefill_num_tokens = _get_welm_mtp_prefill_num_tokens(local_batch)

    tbo_preparer = TboDPAttentionPreparer()
    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)
    if is_welm_deferred_prefill:
        local_can_run_tbo = False

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        num_reqs=num_reqs,
        can_cuda_graph=can_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
        has_router_replay=has_router_replay,
        has_cache_hit_extend=has_cache_hit_extend,
        will_contract_welm_kv_mirror=will_contract_welm_kv_mirror,
        is_welm_deferred_prefill=is_welm_deferred_prefill,
        welm_mtp_prefill_num_tokens=welm_mtp_prefill_num_tokens,
    )
    return mlp_sync_info, tbo_preparer


def apply_gathered_mlp_sync(
    local_batch: Optional[ScheduleBatch],
    mlp_sync_info: MLPSyncBatchInfo,
    tbo_preparer: TboDPAttentionPreparer,
    get_idle_batch: Callable[[], ScheduleBatch],
    require_mlp_tp_gather: bool,
    skip_all_gather: bool,
):
    """Apply gathered mlp-sync metadata to the local batch.

    Mirrors the post-gather half of the classic prepare_mlp_sync_batch_raw:
    TBO output, idle-batch creation for ranks without work, and the
    _update_gather_batch field propagation.
    """
    if not skip_all_gather:
        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info[:, 4:6],
            )
        )

    need_idle_batch = skip_all_gather or max(mlp_sync_info.global_num_tokens) > 0
    if need_idle_batch:
        batch_to_gather = local_batch
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()
        _update_gather_batch(
            batch_to_gather, mlp_sync_info, require_mlp_tp_gather, skip_all_gather
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
):
    mlp_sync_info, tbo_preparer = compute_local_mlp_sync_info(
        local_batch,
        dp_size=dp_size,
        attn_tp_size=attn_tp_size,
        attn_cp_size=attn_cp_size,
        disable_cuda_graph=disable_cuda_graph,
    )

    skip_all_gather = envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()
    if len(offload_tags) == 0 and (
        disable_overlap_schedule
        or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    if not skip_all_gather:
        mlp_sync_info.all_gather(device=device, group=group)

    return apply_gathered_mlp_sync(
        local_batch,
        mlp_sync_info,
        tbo_preparer,
        get_idle_batch=get_idle_batch,
        require_mlp_tp_gather=require_mlp_tp_gather,
        skip_all_gather=skip_all_gather,
    )


class SchedulerDPAttnMixin:
    def prepare_mlp_sync_batch(self: Scheduler, local_batch: ScheduleBatch):
        return prepare_mlp_sync_batch_raw(
            local_batch,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.attn_tp_size,
            attn_cp_size=self.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            disable_overlap_schedule=self.server_args.disable_overlap_schedule,
            offload_tags=self.offload_tags,
        )

    def welm_fused_sched_sync_round(
        self: Scheduler, pending_work_count: int, pending_control_count: int
    ):
        """Single fused CPU all_gather for a steady decode round.

        Collapses the per-round recv broadcasts (empty in steady state), the
        dp-spec-prefill all_reduce, and the decode mlp-sync all_gather into
        one collective. Pre-gather work is read-only or idempotent
        (filter_batch); every side-effecting step (retract, admission,
        control handling) is deferred until the gathered consensus authorizes
        the fast path.

        Returns (True, batch) when every rank is on a plain decode/idle
        round; batch may be None (fully idle fleet). Returns (False, None)
        when any rank raised an intent bit — the caller must then run the
        classic scheduling path, which re-resolves the round from scratch
        (nothing gathered here has been applied).
        """
        rb = self.running_batch
        initial_bs = rb.batch_size()
        if not rb.is_empty():
            # Same filter update_running_batch would run (idempotent), hoisted
            # so the gathered num_tokens reflects the post-filter batch size.
            rb.filter_batch(v1_spec_info_filtered=True)
            if rb.batch_size() < initial_bs:
                rb.batch_is_full = False

        candidate = None
        if not rb.is_empty() and not rb.is_prefill_only:
            candidate = rb

        intent = 0
        if pending_work_count > 0:
            intent |= _FUSED_INTENT_PENDING_WORK
        if pending_control_count > 0:
            intent |= _FUSED_INTENT_PENDING_CONTROL
        if (
            self.chunked_req is not None
            or len(self.waiting_queue) > 0
            # Grammar-queue requests are promoted into waiting_queue inside
            # get_new_batch_prefill; count them here so the fast path stays
            # collective-safe (same lesson as the dp-spec-prefill consensus).
            or self.grammar_manager.has_waiting_grammars()
            or (not rb.is_empty() and rb.is_prefill_only)
        ):
            intent |= _FUSED_INTENT_NEEDS_CLASSIC
        if candidate is not None and (
            not self._check_decode_mem(candidate)
            or (
                envs.SGLANG_TEST_RETRACT.get()
                and self.forward_ct % envs.SGLANG_TEST_RETRACT_INTERVAL.get() == 0
            )
        ):
            # Retraction may fire in update_running_batch: it would change the
            # batch size after we gathered it, so route through the classic
            # path where retract happens before the decode mlp-sync gather.
            # Deliberately unconditional: check_decode_mem may evict from the
            # radix cache, and that side effect must stay identical on every
            # rank of an attn-TP group, so it must not be gated on
            # leader-only intent bits such as pending work/control.
            intent |= _FUSED_INTENT_MAY_RETRACT

        mlp_sync_info, tbo_preparer = compute_local_mlp_sync_info(
            candidate,
            dp_size=self.server_args.dp_size,
            attn_tp_size=self.attn_tp_size,
            attn_cp_size=self.attn_cp_size,
            disable_cuda_graph=self.server_args.disable_cuda_graph,
        )
        mlp_sync_info.fused_intent = intent
        mlp_sync_info.fused_stamp = self.forward_ct

        mlp_sync_info.gather_hierarchical(
            leader_group=self._welm_fused_leader_group,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            attn_tp_src=self.attn_tp_group.ranks[0],
            is_leader=(self.attn_tp_rank == 0 and self.attn_cp_rank == 0),
        )

        rows = mlp_sync_info.all_rows
        if not (
            bool((rows[:, 9] == self.forward_ct).all())
            and bool((rows[:, 10] == _WELM_FUSED_SCHED_SYNC_SCHEMA).all())
        ):
            # A rank skipped or double-entered a collective. Failing loudly
            # here beats the silent cross-rank hang this used to become.
            raise RuntimeError(
                "[welm] fused sched-sync desync detected "
                f"(local forward_ct={self.forward_ct}, "
                f"schema={_WELM_FUSED_SCHED_SYNC_SCHEMA}); gathered rows: "
                f"{rows.tolist()}"
            )

        if bool(rows[:, 8].any()):
            return False, None

        # Fast path: every rank is on a plain decode (or idle) round. Finish
        # the non-retract remainder of update_running_batch — the pre-gather
        # _check_decode_mem consensus guarantees the retract branch is dead —
        # then apply the gathered metadata exactly like the classic path.
        if candidate is not None:
            self.new_token_ratio = max(
                self.new_token_ratio - self.new_token_ratio_decay,
                self.min_new_token_ratio,
            )
            candidate.prepare_for_decode()

        batch = apply_gathered_mlp_sync(
            candidate,
            mlp_sync_info,
            tbo_preparer,
            get_idle_batch=self.get_idle_batch,
            require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
            skip_all_gather=False,
        )
        return True, batch

    def maybe_prepare_mlp_sync_batch(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to prepare MLP sync batch for DP attention.
        Should be called after get_new_batch_prefill().

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.require_mlp_sync for prepare_mlp_sync_batch decision
        """
        if need_sync if need_sync is not None else self.require_mlp_sync:
            batch = self.prepare_mlp_sync_batch(batch)
        return batch

    def get_idle_batch(self: Scheduler) -> ScheduleBatch:
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch
