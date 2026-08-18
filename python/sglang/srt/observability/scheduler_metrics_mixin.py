from __future__ import annotations

import dataclasses
import logging
import tempfile
import time
from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from sglang.srt.disaggregation.kv_events import EventPublisherFactory, KVEventBatch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    DisaggregationMetrics,
    GetLoadsReqInput,
    GetLoadsReqOutput,
    LoRAMetrics,
    MemoryMetrics,
    QueueMetrics,
    SpeculativeMetrics,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    QueueCount,
    SchedulerMetricsCollector,
    SchedulerStats,
    compute_routing_key_stats,
)
from sglang.srt.utils.device_timer import DeviceTimer
from sglang.srt.utils.scheduler_status_logger import SchedulerStatusLogger

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder
    from sglang.srt.managers.scheduler import EmbeddingBatchResult, Scheduler

logger = logging.getLogger(__name__)

RECORD_STEP_TIME = envs.SGLANG_RECORD_STEP_TIME.get()
LOG_FORWARD_ITERS = envs.SGLANG_LOG_FORWARD_ITERS.get()
ENABLE_METRICS_DEVICE_TIMER = envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.get()


@dataclasses.dataclass
class PrefillStats:
    """Stats for logging prefill batch metrics."""

    log_input_tokens: int
    log_hit_tokens: int
    new_token_ratio: float
    num_running_reqs: QueueCount
    num_new_seqs: int  # len(can_run_list)
    num_pending_tokens: int = 0

    @classmethod
    def from_adder(
        cls,
        adder: PrefillAdder,
        running_reqs: List[Req],
        enable_priority_scheduling: bool = False,
        num_pending_tokens: int = 0,
    ):
        return cls(
            log_input_tokens=adder.log_input_tokens,
            log_hit_tokens=adder.log_hit_tokens,
            new_token_ratio=adder.new_token_ratio,
            num_running_reqs=QueueCount.from_reqs(
                running_reqs, enable_priority_scheduling
            ),
            num_new_seqs=len(adder.can_run_list),
            num_pending_tokens=num_pending_tokens,
        )


@dataclasses.dataclass
class KvMetrics:
    request_active_slots: int = 0
    request_total_slots: int = 0
    kv_active_blocks: int = 0
    kv_total_blocks: int = 0
    num_requests_waiting: int = 0
    gpu_cache_usage_perc: float = 0.0
    gpu_prefix_cache_hit_rate: float = 0.0
    data_parallel_rank: int = 0


class SchedulerMetricsMixin:
    enable_fpm: bool = False

    def init_metrics(
        self: Scheduler, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ):
        # Basic stats
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.last_decode_stats_tic = time.perf_counter()
        self.last_prefill_stats_tic = time.perf_counter()
        self.last_gen_throughput: float = 0.0
        self.last_input_throughput: float = 0.0
        self.step_time_dict = defaultdict(list)  # Dict[batch size -> step time]
        self.stats = SchedulerStats()
        self._graph_backend_label = {
            "cpu": "cpu graph",
            "npu": "npu graph",
            "musa": "musa graph",
        }.get(getattr(self, "device", ""), "cuda graph")

        # Cumulative spec-decoding counters (reset every decode_log_interval).
        # Each update adds (num_correct_drafts + bs, bs).
        # `*_accept_tokens` = drafts + bonus; `*_correct_drafts` = drafts-only.
        self.spec_num_accept_tokens = 0  # per-log-interval
        self.spec_num_forward_ct = 0
        self.spec_total_num_accept_tokens = 0  # lifetime
        self.spec_total_num_forward_ct = 0

        # For PD disaggregation
        self.kv_transfer_speed_gb_s: float = 0.0
        self.kv_transfer_latency_ms: float = 0.0

        # Metrics
        self.enable_metrics = self.server_args.enable_metrics
        self.is_stats_logging_rank = self.attn_tp_rank == 0
        self.current_scheduler_metrics_enabled = self.enable_metrics and (
            self.is_stats_logging_rank
            or self.server_args.enable_metrics_for_all_schedulers
        )
        self.enable_mfu_metrics = self.server_args.enable_mfu_metrics
        if self.enable_mfu_metrics:
            self._init_estimated_perf_constants()
            self._mfu_log_flops = 0.0
            self._mfu_log_read_bytes = 0.0
            self._mfu_log_write_bytes = 0.0

        if self.enable_metrics:
            engine_type = DisaggregationMode.to_engine_type(
                self.server_args.disaggregation_mode
            )

            labels = {
                "model_name": self.server_args.served_model_name,
                "engine_type": engine_type,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "moe_ep_rank": self.moe_ep_rank,
            }
            if self.enable_priority_scheduling:
                labels["priority"] = ""
            if dp_rank is not None:
                labels["dp_rank"] = dp_rank
            if self.server_args.extra_metric_labels:
                labels.update(self.server_args.extra_metric_labels)
            self.metrics_collector = SchedulerMetricsCollector(
                labels=labels,
                enable_lora=self.enable_lora,
                enable_hierarchical_cache=self.enable_hierarchical_cache,
                enable_streaming_session=self.server_args.enable_streaming_session,
                server_args=self.server_args,
            )

        self.fwd_occupancy = float("nan")

        if ENABLE_METRICS_DEVICE_TIMER:
            self._device_timer_window_batch_count = 0
            self._device_timer_window_gpu_time = 0.0
            self._device_timer_window_start = None

            def _wrap_execution_reporter(**kwargs):
                self._device_timer_window_gpu_time += kwargs["t"]
                if self.enable_metrics:
                    self.metrics_collector.increment_forward_execution_seconds(**kwargs)

            self.forward_pass_device_timer = DeviceTimer(
                reporter=_wrap_execution_reporter,
            )

        self.init_kv_events(self.server_args.kv_events_config)

        self._init_fpm()

        self.scheduler_status_logger = SchedulerStatusLogger.maybe_create(
            enable_metrics=self.enable_metrics
        )

    def install_device_timer_on_runners(self: Scheduler):
        if not hasattr(self, "forward_pass_device_timer"):
            return
        timer = self.forward_pass_device_timer
        self.tp_worker.model_runner.device_timer = timer
        if self.draft_worker is not None:
            dw = getattr(self.draft_worker, "draft_worker", None)
            if dw is not None:
                if hasattr(dw, "draft_runner"):
                    dw.draft_runner.device_timer = timer
                for r in getattr(dw, "draft_runner_list", []):
                    r.device_timer = timer

    def init_kv_events(self: Scheduler, kv_events_config: Optional[str]):
        self.enable_kv_cache_events = bool(
            kv_events_config and self.attn_tp_rank == 0 and self.attn_cp_rank == 0
        )

        if self.enable_kv_cache_events:
            self.kv_event_publisher = EventPublisherFactory.create(
                kv_events_config, self.attn_dp_rank
            )

    def _init_fpm(self: Scheduler):
        """Initialize Forward Pass Metrics (FPM) publisher if configured."""
        self.enable_fpm = False
        if (
            self.server_args.enable_forward_pass_metrics
            and self.attn_tp_rank == 0
            and self.pp_rank == self.pp_size - 1
        ):
            from sglang.srt.observability.forward_pass_metrics import (
                _FpmPublisherThread,
            )

            self._fpm_dp_rank = self.dp_rank if self.dp_rank is not None else 0
            self._fpm_worker_id = self.server_args.forward_pass_metrics_worker_id
            base_endpoint = self.server_args.forward_pass_metrics_ipc_name
            if base_endpoint is None:
                ipc_path = tempfile.NamedTemporaryFile(delete=False).name
                base_endpoint = f"ipc://{ipc_path}"
                self.server_args.forward_pass_metrics_ipc_name = base_endpoint
            endpoint = f"{base_endpoint}.{self._fpm_dp_rank}"
            self._fpm_publisher = _FpmPublisherThread(
                endpoint,
                worker_id=self._fpm_worker_id,
                dp_rank=self._fpm_dp_rank,
            )
            self._fpm_gpu_time_acc = 0.0

            def _fpm_device_timer_reporter(t, **_kwargs):
                self._fpm_gpu_time_acc += t

            if hasattr(self, "forward_pass_device_timer"):
                self.forward_pass_device_timer.add_reporter(_fpm_device_timer_reporter)
            else:
                self.forward_pass_device_timer = DeviceTimer(
                    reporter=_fpm_device_timer_reporter,
                )
            self._fpm_uses_device_timer = True
            self.enable_fpm = True
            logger.info(
                "FPM: ZMQ PUB bound on %s (dp_rank=%d, device_timer=%s)",
                endpoint,
                self._fpm_dp_rank,
                self._fpm_uses_device_timer,
            )

    def _build_scheduled_request_metrics(self: Scheduler, batch: ScheduleBatch):
        from sglang.srt.observability.forward_pass_metrics import (
            ScheduledRequestMetrics,
            WelfordAccumulator,
        )

        num_prefill_requests = 0
        sum_prefill_tokens = 0
        sum_prefill_kv_tokens = 0
        prefill_lengths = WelfordAccumulator()

        if batch.forward_mode.is_mixed():
            decode_req_ids = {id(req) for req in batch.decoding_reqs or []}
            prefill_reqs = [req for req in batch.reqs if id(req) not in decode_req_ids]
        elif batch.forward_mode.is_extend():
            prefill_reqs = batch.reqs
        else:
            prefill_reqs = []

        if prefill_reqs:
            stats = batch.prefill_stats
            for req in prefill_reqs:
                prefill_lengths.add(len(req.origin_input_ids))
            num_prefill_requests = stats.num_new_seqs if stats else len(prefill_reqs)
            sum_prefill_tokens = stats.log_input_tokens if stats else 0
            sum_prefill_kv_tokens = sum(len(req.prefix_indices) for req in prefill_reqs)

        decode_kv = WelfordAccumulator()
        if batch.forward_mode.is_mixed():
            for req in batch.decoding_reqs or []:
                decode_kv.add(req.seqlen)
        elif batch.forward_mode.is_decode():
            for sl in batch.seq_lens_cpu:
                decode_kv.add(int(sl))

        return ScheduledRequestMetrics(
            num_prefill_requests=num_prefill_requests,
            sum_prefill_tokens=sum_prefill_tokens,
            var_prefill_length=prefill_lengths.variance(),
            sum_prefill_kv_tokens=sum_prefill_kv_tokens,
            num_decode_requests=decode_kv.count,
            sum_decode_kv_tokens=decode_kv.total,
            var_decode_kv_tokens=decode_kv.variance(),
        )

    def _build_queued_request_metrics(self: Scheduler):
        from sglang.srt.observability.forward_pass_metrics import (
            QueuedRequestMetrics,
            WelfordAccumulator,
        )

        prefill_q = WelfordAccumulator()
        decode_q = WelfordAccumulator()
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in self.disagg_prefill_bootstrap_queue.queue:
                prefill_q.add(len(req.origin_input_ids))
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            for req in self.disagg_decode_prealloc_queue.queue:
                decode_q.add(req.seqlen)
            for req in self.disagg_decode_transfer_queue.queue:
                decode_q.add(req.seqlen)
        else:
            for req in self.waiting_queue:
                if len(req.output_ids) > 0:
                    decode_q.add(req.seqlen)
                else:
                    prefill_q.add(len(req.origin_input_ids))

        return QueuedRequestMetrics(
            num_prefill_requests=prefill_q.count,
            sum_prefill_tokens=prefill_q.total,
            var_prefill_length=prefill_q.variance(),
            num_decode_requests=decode_q.count,
            sum_decode_kv_tokens=decode_q.total,
            var_decode_kv_tokens=decode_q.variance(),
        )

    def update_spec_metrics(self: Scheduler, bs: int, num_correct_drafts: int):
        self.spec_num_accept_tokens += num_correct_drafts + bs
        self.spec_num_forward_ct += bs

        # Bonus tokens updated elsewhere
        self.num_generated_tokens += num_correct_drafts

    def _init_estimated_perf_constants(self: Scheduler) -> None:
        model_config = self.model_config
        hf_text_config = model_config.hf_text_config

        hidden_size = float(model_config.hidden_size)
        num_layers = int(
            getattr(model_config, "num_attention_layers", None)
            or getattr(model_config, "num_hidden_layers", 0)
        )
        head_dim = float(getattr(model_config, "head_dim", 0))
        num_attn_heads = float(model_config.get_num_attention_heads(self.tp_size))
        num_kv_heads = float(model_config.get_num_kv_heads(self.tp_size))
        tp_size = max(1, int(getattr(self, "tp_size", 1)))
        intermediate_size = getattr(hf_text_config, "intermediate_size", None)
        if intermediate_size is None:
            intermediate_size = getattr(hf_text_config, "ffn_hidden_size", 0)
        intermediate_size = float(intermediate_size)
        vocab_size = float(getattr(model_config, "vocab_size", 0))

        dtype_num_bytes = getattr(model_config.dtype, "itemsize", None)
        if dtype_num_bytes is None:
            dtype_num_bytes = 2
        # Keep this estimator lightweight and consistent with current server dtype.
        # KV cache quantization-aware bytes can be added in a follow-up.
        act_bytes = float(dtype_num_bytes)
        w_bytes = float(dtype_num_bytes)
        cache_bytes = float(dtype_num_bytes)

        # Linear-layer FLOPs per token on one GPU.
        self._mfu_num_layers = num_layers
        self._mfu_attn_q_flops = 2.0 * hidden_size * num_attn_heads * head_dim
        self._mfu_attn_kv_flops = 2.0 * hidden_size * (2.0 * num_kv_heads * head_dim)
        self._mfu_attn_o_flops = 2.0 * (num_attn_heads * head_dim) * hidden_size

        is_welmv4 = getattr(hf_text_config, "model_type", "") == "welmv4_moe"
        self._mfu_attn_gate_flops = (
            2.0 * hidden_size * num_attn_heads if is_welmv4 else 0.0
        )

        moe_intermediate_size = getattr(hf_text_config, "moe_intermediate_size", None)
        shared_expert_intermediate_size = getattr(
            hf_text_config, "shared_expert_intermediate_size", None
        )
        has_shared_expert_gate = bool(
            getattr(hf_text_config, "has_shared_expert_gate", True)
        )
        num_experts_per_tok = getattr(hf_text_config, "num_experts_per_tok", None)
        num_experts = getattr(hf_text_config, "num_experts", None)
        if (
            is_welmv4
            and moe_intermediate_size is not None
            and shared_expert_intermediate_size is not None
            and num_experts_per_tok is not None
            and num_experts is not None
        ):
            # WeLM MoE has a replicated router and TP-sharded routed/shared experts.
            router_flops = 2.0 * hidden_size * float(num_experts)
            routed_moe_flops = (
                6.0
                * hidden_size
                * float(num_experts_per_tok)
                * float(moe_intermediate_size)
                / tp_size
            )
            shared_moe_flops = (
                6.0 * hidden_size * float(shared_expert_intermediate_size) / tp_size
            )
            shared_gate_flops = 2.0 * hidden_size if has_shared_expert_gate else 0.0
            self._mfu_mlp_flops = (
                router_flops
                + routed_moe_flops
                + shared_moe_flops
                + shared_gate_flops
            )
        else:
            self._mfu_mlp_flops = (
                6.0 * hidden_size * intermediate_size if intermediate_size > 0 else 0.0
            )

        self._mfu_lm_head_flops = (
            2.0 * hidden_size * vocab_size / tp_size if vocab_size > 0 else 0.0
        )
        self._mfu_q_o_gate_flops = (
            self._mfu_attn_q_flops
            + self._mfu_attn_o_flops
            + self._mfu_attn_gate_flops
        )
        self._mfu_linear_flops_per_layer = (
            self._mfu_q_o_gate_flops
            + self._mfu_attn_kv_flops
            + self._mfu_mlp_flops
        )
        self._linear_flops_per_token = max(
            0.0, self._mfu_linear_flops_per_layer * num_layers
        )

        # Attention dot-product FLOPs coefficient per token-context pair on one GPU.
        # attn_qk + attn_av = 4 * q_heads * context * head_dim
        self._mfu_attn_ctx_flops = 4.0 * num_attn_heads * head_dim
        self._attn_dot_flops_coeff = self._mfu_attn_ctx_flops * num_layers
        self._mfu_sliding_windows = self._init_mfu_sliding_windows(num_layers)
        self._mfu_kv_mirror_layers = self._init_mfu_kv_mirror_layers(num_layers)
        self._mfu_enable_welm_kv_mirror_opt = bool(
            getattr(self.server_args, "enable_welm_kv_mirror_opt", False)
        )

        # KV cache bytes (write one K and one V vector per generated token).
        self._kv_cache_bytes_per_token_per_layer = (
            2.0 * num_kv_heads * head_dim * cache_bytes
        )
        self._kv_cache_bytes_per_token = (
            2.0 * num_layers * num_kv_heads * head_dim * cache_bytes
        )

        # Weight read bytes per token.
        self._weight_read_bytes_per_token = (
            hidden_size
            * head_dim
            * (num_attn_heads + 2.0 * num_kv_heads)
            * w_bytes
            * num_layers
            + hidden_size * head_dim * num_attn_heads * w_bytes * num_layers
            + (
                3.0 * hidden_size * intermediate_size * w_bytes * num_layers
                if intermediate_size > 0
                else 0.0
            )
        )

        # Activation movement bytes per token (coarse approximation).
        self._qkv_act_bytes_per_token = (
            hidden_size * act_bytes * num_layers
            + (num_attn_heads + 2.0 * num_kv_heads) * head_dim * act_bytes * num_layers
            + head_dim * num_attn_heads * act_bytes * num_layers
            + hidden_size * act_bytes * num_layers
        )
        self._ffn_act_bytes_per_token = (
            3.0 * intermediate_size * act_bytes * num_layers
            if intermediate_size > 0
            else 0.0
        )

        # Prefill reads Q/K/V activations from on-device memory.
        self._prefill_attn_act_read_per_token = (
            (num_attn_heads + 2.0 * num_kv_heads) * head_dim * act_bytes * num_layers
        )

        # Decode reads Q from activation memory; K/V reads are from KV cache.
        self._decode_q_read_bytes_per_token = (
            num_attn_heads * head_dim * act_bytes * num_layers
        )

    def _mfu_get_config_value(self: Scheduler, name: str, default=None):
        for cfg in (
            getattr(self.model_config, "hf_text_config", None),
            getattr(self.model_config, "hf_config", None),
            self.model_config,
        ):
            if cfg is not None and hasattr(cfg, name):
                value = getattr(cfg, name)
                if value is not None:
                    return value
        return default

    @staticmethod
    def _mfu_to_int_list(value) -> Optional[List[int]]:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (int, float)):
            return [int(value)]
        return [int(x) for x in value]

    @classmethod
    def _mfu_get_int_list_attr(cls, obj, *names: str) -> Optional[List[int]]:
        for name in names:
            value = getattr(obj, name, None)
            if value is None:
                continue
            return cls._mfu_to_int_list(value)
        return None

    @staticmethod
    def _mfu_window_plus_current(window, max_position_embeddings: int) -> Optional[int]:
        if window is None:
            return None
        window = int(window)
        if window <= 0 or window >= max_position_embeddings:
            return None
        return window + 1

    def _init_mfu_sliding_windows(
        self: Scheduler, num_layers: int
    ) -> Tuple[Optional[int], ...]:
        max_position_embeddings = int(
            self._mfu_get_config_value("max_position_embeddings", 10**18)
        )
        layerwise = self._mfu_get_config_value("sliding_window_size_layerwise", None)
        if layerwise is not None and not isinstance(layerwise, (int, float, str)):
            windows = list(layerwise)
            return tuple(
                self._mfu_window_plus_current(
                    windows[i] if i < len(windows) else None,
                    max_position_embeddings,
                )
                for i in range(num_layers)
            )

        scalar_window = None
        for key in ("sliding_window_size", "sliding_window", "window_size"):
            scalar_window = self._mfu_get_config_value(key, None)
            if scalar_window is not None:
                break
        window = self._mfu_window_plus_current(scalar_window, max_position_embeddings)
        return tuple(window for _ in range(num_layers))

    def _init_mfu_kv_mirror_layers(self: Scheduler, num_layers: int) -> set[int]:
        mirror_layers = self._mfu_get_config_value("kv_mirror_layers", []) or []
        imitated_layers = self._mfu_get_config_value("kv_mirror_imitated_layers", []) or []
        valid_layers: set[int] = set()
        for mirror, imitated in zip(mirror_layers, imitated_layers):
            mirror = int(mirror)
            imitated = int(imitated)
            if 0 <= mirror < num_layers and 0 <= imitated < num_layers:
                valid_layers.add(mirror)
        return valid_layers

    def _mfu_prefill_items(
        self: Scheduler, batch: Optional[ScheduleBatch], num_tokens: int, num_seqs: int
    ) -> List[Tuple[int, int]]:
        if batch is not None:
            extend_lens = self._mfu_get_int_list_attr(
                batch, "extend_lens", "extend_seq_lens_cpu", "extend_seq_lens"
            )
            prefix_lens = self._mfu_get_int_list_attr(
                batch, "prefix_lens", "extend_prefix_lens_cpu", "extend_prefix_lens"
            )
            seq_lens_value = getattr(batch, "seq_lens_cpu", None)
            if seq_lens_value is None:
                seq_lens_value = getattr(batch, "seq_lens", None)
            seq_lens = self._mfu_to_int_list(seq_lens_value)
            chunked_prefill_size = int(
                getattr(self.server_args, "chunked_prefill_size", 0) or 0
            )
            if not extend_lens and seq_lens and len(seq_lens) == max(1, num_seqs):
                inferred_extend_lens = None
                if len(seq_lens) == 1 and num_tokens <= seq_lens[0]:
                    inferred_extend_lens = [num_tokens]
                elif sum(seq_lens) == num_tokens:
                    inferred_extend_lens = list(seq_lens)
                else:
                    per_seq_len, remainder = divmod(num_tokens, len(seq_lens))
                    can_infer_uniform_extend = (
                        remainder == 0
                        and per_seq_len > 0
                        and all(seq_len >= per_seq_len for seq_len in seq_lens)
                        and (
                            (
                                chunked_prefill_size > 0
                                and per_seq_len == chunked_prefill_size
                            )
                            or len(set(seq_lens)) == 1
                        )
                    )
                    if can_infer_uniform_extend:
                        inferred_extend_lens = [per_seq_len] * len(seq_lens)

                if inferred_extend_lens is not None:
                    return [
                        (
                            max(0, int(seq_len) - int(extend_len)),
                            max(0, int(extend_len)),
                        )
                        for seq_len, extend_len in zip(seq_lens, inferred_extend_lens)
                        if int(extend_len) > 0
                    ]
            if extend_lens and sum(extend_lens) == num_tokens:
                if prefix_lens is None or len(prefix_lens) != len(extend_lens):
                    prefix_lens = [0] * len(extend_lens)
                if seq_lens is not None and len(seq_lens) == len(extend_lens):
                    for i, (seq_len, extend_len) in enumerate(zip(seq_lens, extend_lens)):
                        derived_prefix_len = int(seq_len) - int(extend_len)
                        if derived_prefix_len > prefix_lens[i]:
                            prefix_lens[i] = derived_prefix_len
                reqs = getattr(batch, "reqs", None) or []
                if reqs and len(reqs) == len(extend_lens):
                    for i, (req, extend_len) in enumerate(zip(reqs, extend_lens)):
                        if prefix_lens[i] > 0:
                            continue
                        extend_batch_idx = int(getattr(req, "extend_batch_idx", 0) or 0)
                        if extend_batch_idx <= 1:
                            continue
                        origin_len = len(getattr(req, "origin_input_ids", []) or [])
                        if origin_len <= extend_len:
                            continue
                        chunk_size = (
                            chunked_prefill_size
                            if chunked_prefill_size > 0
                            else extend_len
                        )
                        inferred_prefix_len = (extend_batch_idx - 1) * chunk_size
                        prefix_lens[i] = min(
                            max(0, inferred_prefix_len),
                            max(0, origin_len - int(extend_len)),
                        )
                return [
                    (max(0, int(prefix_len)), max(0, int(extend_len)))
                    for prefix_len, extend_len in zip(prefix_lens, extend_lens)
                    if int(extend_len) > 0
                ]

        num_seqs = max(1, int(num_seqs))
        base_len, remainder = divmod(num_tokens, num_seqs)
        return [
            (0, base_len + (1 if idx < remainder else 0))
            for idx in range(num_seqs)
            if base_len + (1 if idx < remainder else 0) > 0
        ]

    @staticmethod
    def _mfu_prefill_context_sum(
        prefix_len: int, extend_len: int, window_plus_current: Optional[int]
    ) -> float:
        if extend_len <= 0:
            return 0.0
        if window_plus_current is None:
            return extend_len * prefix_len + extend_len * (extend_len + 1) / 2.0

        uncapped = min(extend_len, max(0, window_plus_current - prefix_len))
        return (
            uncapped * prefix_len
            + uncapped * (uncapped + 1) / 2.0
            + (extend_len - uncapped) * window_plus_current
        )

    @staticmethod
    def _mfu_last_context(
        prefix_len: int, extend_len: int, window_plus_current: Optional[int]
    ) -> float:
        if extend_len <= 0:
            return 0.0
        context = prefix_len + extend_len
        if window_plus_current is not None:
            context = min(context, window_plus_current)
        return float(context)

    def _estimate_prefill_perf_from_batch(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
        num_tokens: int,
        num_seqs: int,
    ) -> Tuple[float, float, float]:
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        items = self._mfu_prefill_items(batch, tokens, num_seqs)
        active_seqs = len(items)
        enable_kv_mirror = (
            self._mfu_enable_welm_kv_mirror_opt and len(self._mfu_kv_mirror_layers) > 0
        )
        first_mirror_layer = (
            min(self._mfu_kv_mirror_layers) if enable_kv_mirror else None
        )

        flops = tokens * self._mfu_num_layers * self._mfu_attn_kv_flops
        for layer_idx in range(self._mfu_num_layers):
            window = self._mfu_sliding_windows[layer_idx]
            full_context_sum = sum(
                self._mfu_prefill_context_sum(prefix_len, extend_len, window)
                for prefix_len, extend_len in items
            )

            q_rows = tokens
            context_sum = full_context_sum
            if first_mirror_layer is not None and layer_idx > first_mirror_layer:
                q_rows = active_seqs
                context_sum = sum(
                    self._mfu_last_context(prefix_len, extend_len, window)
                    for prefix_len, extend_len in items
                )
            flops += q_rows * self._mfu_q_o_gate_flops
            flops += context_sum * self._mfu_attn_ctx_flops

            mlp_rows = tokens
            if first_mirror_layer is not None and layer_idx >= first_mirror_layer:
                mlp_rows = active_seqs
            flops += mlp_rows * self._mfu_mlp_flops

        flops += active_seqs * self._mfu_lm_head_flops

        read_bytes = (
            tokens * self._weight_read_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._prefill_attn_act_read_per_token
        )
        write_bytes = (
            tokens * self._kv_cache_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._ffn_act_bytes_per_token
        )
        return flops, read_bytes, write_bytes

    def _estimate_prefill_perf(
        self: Scheduler, num_tokens: int
    ) -> Tuple[float, float, float]:
        return self._estimate_prefill_perf_from_batch(None, num_tokens, 1)

    def _estimate_decode_perf(
        self: Scheduler, batch: ScheduleBatch, num_tokens: int
    ) -> Tuple[float, float, float]:
        tokens = max(0, int(num_tokens))
        if tokens == 0:
            return 0.0, 0.0, 0.0

        seq_lens = self._mfu_to_int_list(getattr(batch, "seq_lens_cpu", None)) or [
            1
        ] * tokens
        context_scale = tokens / len(seq_lens) if seq_lens else 0.0
        flops = tokens * (
            self._mfu_linear_flops_per_layer * self._mfu_num_layers
            + self._mfu_lm_head_flops
        )
        context_by_layer = 0.0
        for layer_idx in range(self._mfu_num_layers):
            window = self._mfu_sliding_windows[layer_idx]
            layer_context = 0.0
            for seq_len in seq_lens:
                layer_context += (
                    min(seq_len, window) if window is not None else float(seq_len)
                )
            layer_context *= context_scale
            context_by_layer += layer_context
            flops += self._mfu_attn_ctx_flops * layer_context

        read_bytes = (
            tokens * self._weight_read_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._decode_q_read_bytes_per_token
            + context_by_layer * self._kv_cache_bytes_per_token_per_layer
        )
        write_bytes = (
            tokens * self._kv_cache_bytes_per_token
            + tokens * self._qkv_act_bytes_per_token
            + tokens * self._ffn_act_bytes_per_token
        )
        return flops, read_bytes, write_bytes

    def reset_metrics(self: Scheduler):
        self.forward_ct_decode = 0
        self.num_generated_tokens = 0
        self.spec_num_accept_tokens = 0
        self.spec_num_forward_ct = 0
        self.spec_total_num_accept_tokens = 0
        self.spec_total_num_forward_ct = 0

    def report_prefill_stats(
        self: Scheduler,
        batch: Optional[ScheduleBatch],
        prefill_stats: PrefillStats,
        can_run_cuda_graph: bool,
        dp_cooperation_info: Optional[DPCooperationInfo] = None,
    ):
        if (
            not self.is_stats_logging_rank
            and not self.current_scheduler_metrics_enabled
        ):
            return

        now = time.perf_counter()
        gap_latency = now - self.last_prefill_stats_tic
        self.last_prefill_stats_tic = now
        self.last_input_throughput = (
            prefill_stats.log_input_tokens / gap_latency if gap_latency > 0 else 0.0
        )

        pool_stats = self.get_pool_stats()
        token_usage_msg = ", ".join(pool_stats.get_prefill_usage_msg_parts()) + ", "

        self.stats.new_token_ratio = prefill_stats.new_token_ratio
        batch_iter = (
            batch.forward_iter
            if batch is not None and batch.forward_iter is not None
            else self.forward_ct
        )
        iter_msg = f" [{batch_iter}]" if LOG_FORWARD_ITERS else ""

        msg = (
            f"Prefill batch{iter_msg}, "
            f"#new-seq: {prefill_stats.num_new_seqs}, "
            f"#new-token: {prefill_stats.log_input_tokens}, "
            f"#cached-token: {prefill_stats.log_hit_tokens}, "
            f"{token_usage_msg}"
            f"#running-req: {prefill_stats.num_running_reqs.total}, "
            f"#queue-req: {len(self.waiting_queue)}, "
            f"#pending-token: {prefill_stats.num_pending_tokens}, "
        )

        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            msg += f"#bootstrap-req: {len(self.disagg_prefill_bootstrap_queue.queue)}, "
            msg += f"#inflight-req: {len(self.disagg_prefill_inflight_queue)}, "

        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            msg += f"waiting-image-req: {len(self.mm_receiver.waiting_list)}, "

        msg += f"{self._graph_backend_label}: {can_run_cuda_graph}, "
        msg += f"input throughput (token/s): {self.last_input_throughput:.2f}"

        if self.enable_mfu_metrics and gap_latency > 0:
            flops, _, _ = self._estimate_prefill_perf_from_batch(
                batch, prefill_stats.log_input_tokens, prefill_stats.num_new_seqs
            )
            tflops_per_s = flops / gap_latency / 1e12
            msg += f", est. prefill TFLOPS/s (per GPU): {tflops_per_s:.2f}"

        if ENABLE_METRICS_DEVICE_TIMER:
            msg += f", fwd occupancy: {self.fwd_occupancy:.2f}%"

        if self.is_stats_logging_rank:
            logger.info(msg)
        if self.current_scheduler_metrics_enabled:
            self.metrics_collector.increment_prefill_cuda_graph_pass(
                value=can_run_cuda_graph
            )
            self.metrics_collector.increment_realtime_tokens(
                prefill_compute_tokens=prefill_stats.log_input_tokens,
                prefill_cache_tokens=prefill_stats.log_hit_tokens,
                dp_cooperation_info=dp_cooperation_info,
            )
            if self.enable_mfu_metrics:
                flops, read_bytes, write_bytes = self._estimate_prefill_perf_from_batch(
                    batch, prefill_stats.log_input_tokens, prefill_stats.num_new_seqs
                )
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )

            priority_enabled = self.enable_priority_scheduling
            total_tokens = prefill_stats.log_input_tokens + prefill_stats.log_hit_tokens
            cache_hit_rate = (
                prefill_stats.log_hit_tokens / total_tokens if total_tokens > 0 else 0.0
            )

            # Basics
            self.stats.num_running_reqs = prefill_stats.num_running_reqs
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            self.stats.cache_hit_rate = cache_hit_rate

            # Memory pool usage ratios / Absolute token counts
            pool_stats.update_scheduler_stats(self.stats)

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_bootstrap_queue.queue, priority_enabled
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_inflight_queue, priority_enabled
                )
                self.stats.kv_transfer_speed_gb_s = self.kv_transfer_speed_gb_s
                self.stats.kv_transfer_latency_ms = self.kv_transfer_latency_ms
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_transfer_queue.queue, priority_enabled
                )

            # Utilization / LoRA / HiCache
            self.calculate_utilization()
            self.stats.fwd_occupancy = self.fwd_occupancy
            self.update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def report_decode_stats(
        self: Scheduler,
        can_run_cuda_graph: bool,
        running_batch: ScheduleBatch = None,
        num_correct_drafts: int = 0,
    ):
        batch = running_batch or self.running_batch

        decode_tokens = batch.batch_size() + num_correct_drafts
        mfu_perf = None
        if self.enable_mfu_metrics:
            mfu_perf = self._estimate_decode_perf(batch, decode_tokens)
            flops, read_bytes, write_bytes = mfu_perf
            self._mfu_log_flops += flops
            self._mfu_log_read_bytes += read_bytes
            self._mfu_log_write_bytes += write_bytes

        # Every-iteration work: realtime token counting + status logger
        if self.current_scheduler_metrics_enabled:
            self.metrics_collector.increment_realtime_tokens(
                # TODO unify this w/ the bumping logic in `Scheduler.num_generated_tokens` accumulator
                decode_tokens=decode_tokens,
                dp_cooperation_info=batch.dp_cooperation_info,
            )
            if self.enable_mfu_metrics:
                flops, read_bytes, write_bytes = mfu_perf
                self.metrics_collector.increment_estimated_perf(
                    num_flops_per_gpu=flops,
                    num_read_bytes_per_gpu=read_bytes,
                    num_write_bytes_per_gpu=write_bytes,
                )

            if x := self.scheduler_status_logger:
                x.maybe_dump(batch, self.waiting_queue)

        # Periodic work: log + heavy metrics at decode_log_interval
        if self.forward_ct_decode % self.server_args.decode_log_interval != 0:
            return
        if (
            not self.is_stats_logging_rank
            and not self.current_scheduler_metrics_enabled
        ):
            return

        gap_latency = time.perf_counter() - self.last_decode_stats_tic
        self.last_decode_stats_tic = time.perf_counter()
        self.last_gen_throughput = self.num_generated_tokens / gap_latency

        self.num_generated_tokens = 0
        num_running_reqs = len(batch.reqs)

        pool_stats = self.get_pool_stats()
        token_usage_msg = ", ".join(pool_stats.get_decode_usage_msg_parts()) + ", "

        if RECORD_STEP_TIME:
            self.step_time_dict[num_running_reqs].append(
                gap_latency / self.server_args.decode_log_interval
            )

        batch_iter = (
            batch.forward_iter
            if batch is not None and batch.forward_iter is not None
            else self.forward_ct
        )
        iter_msg = f" [{batch_iter}]" if LOG_FORWARD_ITERS else ""
        msg = f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"

        if self.spec_algorithm.is_none():
            spec_accept_length = 0
            spec_accept_rate = 0
        else:
            if self.spec_num_forward_ct == 0:
                spec_accept_length = 0
                spec_accept_rate = 0
            else:
                spec_accept_length = (
                    self.spec_num_accept_tokens / self.spec_num_forward_ct
                )
                num_correct_drafts = (
                    self.spec_num_accept_tokens - self.spec_num_forward_ct
                )
                if self.server_args.speculative_num_draft_tokens:
                    draft_per_round = (
                        self.server_args.speculative_num_draft_tokens - 1
                    )
                else:
                    draft_per_round = self.server_args.speculative_num_steps or 0
                total_draft_tokens = self.spec_num_forward_ct * draft_per_round
                spec_accept_rate = (
                    num_correct_drafts / total_draft_tokens
                    if total_draft_tokens > 0
                    else 0
                )
            self.spec_total_num_accept_tokens += self.spec_num_accept_tokens
            self.spec_total_num_forward_ct += self.spec_num_forward_ct
            self.spec_num_accept_tokens = self.spec_num_forward_ct = 0
            msg += f"accept len: {spec_accept_length:.2f}, accept rate: {spec_accept_rate:.2f}, "
        cache_hit_rate = 0.0

        if self.disaggregation_mode == DisaggregationMode.DECODE:
            msg += f"pre-allocated usage: {self.disagg_decode_prealloc_queue.num_tokens_pre_allocated / self.max_total_num_tokens:.2f}, "
            msg += f"#prealloc-req: {len(self.disagg_decode_prealloc_queue.queue)}, "
            msg += f"#transfer-req: {len(self.disagg_decode_transfer_queue.queue)}, "
            msg += f"#retracted-req: {len(self.disagg_decode_prealloc_queue.retracted_queue)}, "

        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
        ):
            msg += f"waiting-image-req: {len(self.mm_receiver.waiting_list)}, "

        msg += (
            f"{self._graph_backend_label}: {can_run_cuda_graph}, "
            f"gen throughput (token/s): {self.last_gen_throughput:.2f}, "
            f"#queue-req: {len(self.waiting_queue)}"
        )

        if self.enable_mfu_metrics and gap_latency > 0:
            flops_per_s = self._mfu_log_flops / gap_latency
            read_bytes_per_s = self._mfu_log_read_bytes / gap_latency
            write_bytes_per_s = self._mfu_log_write_bytes / gap_latency
            tflops_per_s = flops_per_s / 1e12
            read_gb_per_s = read_bytes_per_s / 1e9
            write_gb_per_s = write_bytes_per_s / 1e9
            msg += (
                f", est. decode TFLOPS/s (per GPU): {tflops_per_s:.2f}, "
                f"est. read BW (GB/s per GPU): {read_gb_per_s:.2f}, "
                f"est. write BW (GB/s per GPU): {write_gb_per_s:.2f}"
            )
            self._mfu_log_flops = 0.0
            self._mfu_log_read_bytes = 0.0
            self._mfu_log_write_bytes = 0.0

        if ENABLE_METRICS_DEVICE_TIMER:
            msg += f", fwd occupancy: {self.fwd_occupancy:.2f}%"

        if self.is_stats_logging_rank:
            logger.info(msg)
        if self.current_scheduler_metrics_enabled:
            priority_enabled = self.enable_priority_scheduling

            # Basics
            self.stats.num_running_reqs = QueueCount.from_reqs(
                batch.reqs, priority_enabled
            )
            self.stats.num_queue_reqs = QueueCount.from_reqs(
                self.waiting_queue, priority_enabled
            )
            self.stats.num_grammar_queue_reqs = len(self.grammar_manager)
            self.stats.gen_throughput = self.last_gen_throughput
            self.stats.cache_hit_rate = cache_hit_rate
            self.stats.decode_sum_seq_lens = batch.seq_lens_cpu.sum().item()

            # Memory pool usage ratios / Absolute token counts
            pool_stats.update_scheduler_stats(self.stats)

            # Speculative decoding
            self.stats.spec_accept_length = spec_accept_length
            self.stats.spec_accept_rate = spec_accept_rate

            # Retract
            self.stats.num_retracted_reqs = self.num_retracted_reqs
            self.stats.num_paused_reqs = self.num_paused_reqs
            self.num_retracted_reqs = self.num_paused_reqs = 0

            # PD disaggregation
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.stats.num_prefill_bootstrap_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_bootstrap_queue.queue, priority_enabled
                )
                self.stats.num_prefill_inflight_queue_reqs = QueueCount.from_reqs(
                    self.disagg_prefill_inflight_queue, priority_enabled
                )
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                self.stats.num_decode_prealloc_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_prealloc_queue.queue, priority_enabled
                )
                self.stats.num_decode_transfer_queue_reqs = QueueCount.from_reqs(
                    self.disagg_decode_transfer_queue.queue, priority_enabled
                )

            # Streaming session metrics
            self.stats.num_streaming_sessions = self._streaming_session_count()
            self.stats.streaming_session_held_tokens = self._session_held_tokens()

            # Routing key metrics
            # (to reduce the overhead, we only compute this when all requests have routing_key)
            if all(r.routing_key is not None for r in batch.reqs):
                running_routing_keys = [r.routing_key for r in batch.reqs]
                waiting_routing_keys = [r.routing_key for r in self.waiting_queue]
                (
                    self.stats.num_unique_running_routing_keys,
                    self.stats.routing_key_running_req_counts,
                ) = compute_routing_key_stats(running_routing_keys)
                _, self.stats.routing_key_all_req_counts = compute_routing_key_stats(
                    running_routing_keys + waiting_routing_keys
                )

            # Utilization / LoRA / HiCache
            self.calculate_utilization()
            self.stats.fwd_occupancy = self.fwd_occupancy
            self.update_lora_metrics()
            self._log_hicache_stats()
            self.metrics_collector.log_stats(self.stats)
            self._emit_kv_metrics()
        self._publish_kv_events()

    def log_batch_result_stats(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        if not self.enable_metrics:
            return
        if not isinstance(result, GenerationBatchResult):
            return

        if (m := result.expert_distribution_metrics) is not None:
            self.metrics_collector.increment_eplb_balancedness(
                forward_mode=batch.forward_mode.name.lower(),
                balancedness=m.eplb_balancedness.item(),
            )

    def _emit_kv_metrics(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        kv_metrics = KvMetrics()
        kv_metrics.request_active_slots = self.stats.num_running_reqs.total
        kv_metrics.request_total_slots = self.max_running_requests
        kv_metrics.kv_active_blocks = int(
            self.stats.token_usage * self.max_total_num_tokens
        )
        kv_metrics.kv_total_blocks = self.max_total_num_tokens
        kv_metrics.num_requests_waiting = self.stats.num_queue_reqs.total
        kv_metrics.gpu_cache_usage_perc = self.stats.token_usage
        kv_metrics.gpu_prefix_cache_hit_rate = self.stats.cache_hit_rate
        kv_metrics.data_parallel_rank = self.dp_rank if self.dp_rank is not None else 0

        if not self.send_metrics_from_scheduler.closed:
            self.send_metrics_from_scheduler.send_pyobj(kv_metrics)

    def _publish_kv_events(self: Scheduler):
        if not self.enable_kv_cache_events:
            return

        events = self.tree_cache.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

    def _emit_forward_pass_metrics(
        self: Scheduler,
        batch: ScheduleBatch,
        result=None,
    ):
        """Emit per-iteration ForwardPassMetrics over ZMQ PUB.

        Prefers GPU-accurate timing from DeviceTimer (which wraps
        model_runner.forward / cuda_graph.replay via PR #24197).
        Falls back to monotonic clock when DeviceTimer is not enabled.
        """
        if not self.enable_fpm:
            return

        from sglang.srt.observability.forward_pass_metrics import (
            ForwardPassMetrics,
        )

        if self._fpm_uses_device_timer:
            self.forward_pass_device_timer._report()
            wall_time = self._fpm_gpu_time_acc
            self._fpm_gpu_time_acc = 0.0
            if wall_time == 0.0:
                return
        else:
            wall_time = max(0.0, time.monotonic() - batch.fpm_start_time)

        fpm = ForwardPassMetrics(
            worker_id=self._fpm_worker_id,
            dp_rank=self._fpm_dp_rank,
            wall_time=wall_time,
            scheduled_requests=self._build_scheduled_request_metrics(batch),
            queued_requests=self._build_queued_request_metrics(),
        )
        self._fpm_publisher.publish(fpm)

    def _shutdown_fpm(self: Scheduler):
        """Shut down the FPM publisher thread."""
        if self.enable_fpm:
            self._fpm_publisher.shutdown()

    def _log_hicache_stats(self: Scheduler):
        """Populate HiCache host-tier stats on self.stats.

        These are pushed to Prometheus by SchedulerMetricsCollector.log_stats().
        """
        if not self.enable_hierarchical_cache:
            return

        host_pool = getattr(self.tree_cache, "token_to_kv_pool_host", None) or getattr(
            self.tree_cache, "full_kv_pool_host", None
        )
        assert host_pool is not None, "Host pool not found"
        self.stats.hicache_host_used_tokens = (
            host_pool.size - host_pool.available_size()
        )
        self.stats.hicache_host_total_tokens = host_pool.size

    def update_lora_metrics(self: Scheduler):
        """Update LoRA pool metrics for monitoring and autoscaling."""
        if not self.enable_lora:
            return

        try:
            # Get LoRA memory pool stats
            lora_manager = self.tp_worker.model_runner.lora_manager
            if lora_manager is None or lora_manager.memory_pool is None:
                return

            mem_pool = lora_manager.memory_pool
            slots_total = mem_pool.max_loras_per_batch

            # Calculate active adapters from running batch
            # This gives a true measure of current load for autoscaling purposes
            active_lora_ids = set()

            # For PP mode, check all running micro batches
            if hasattr(self, "running_mbs") and self.running_mbs:
                for batch in self.running_mbs:
                    if batch and hasattr(batch, "reqs"):
                        for req in batch.reqs:
                            if hasattr(req, "lora_id") and req.lora_id is not None:
                                active_lora_ids.add(req.lora_id)
            # For normal mode, check running_batch
            elif hasattr(self, "running_batch") and self.running_batch:
                if hasattr(self.running_batch, "reqs"):
                    for req in self.running_batch.reqs:
                        if hasattr(req, "lora_id") and req.lora_id is not None:
                            active_lora_ids.add(req.lora_id)

            # Count active adapters (excluding None for base model)
            slots_used = len(active_lora_ids)
            utilization = slots_used / slots_total if slots_total > 0 else 0.0

            # Update stats
            self.stats.lora_pool_slots_used = slots_used
            self.stats.lora_pool_slots_total = slots_total
            self.stats.lora_pool_utilization = utilization

        except Exception as e:
            logger.warning(f"Failed to update LoRA metrics: {e}")

    def calculate_utilization(self: Scheduler):
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.stats.utilization = -1
        else:
            max_under_slo = getattr(self, "max_running_requests_under_SLO", None)
            if max_under_slo is not None and max_under_slo > 0:
                self.stats.utilization = max(
                    self.stats.num_running_reqs.total / max_under_slo,
                    self.stats.token_usage / 0.9,
                )

    def _get_num_pending_tokens(self: Scheduler, chunk_deduct: int = 0) -> int:
        """Get the total number of tokens pending prefill.

        This includes tokens from waiting queue requests plus remaining tokens
        from the currently chunked request.

        Args:
            chunk_deduct: extra tokens to subtract from the chunked request's
                remaining count. At batch-scheduling time the current chunk
                has been planned but ``prefix_indices`` does not yet include it,
                so callers pass ``extend_input_len`` here. At load-reporting
                time ``prefix_indices`` is already up-to-date, so the default
                0 is correct.
        """
        num_pending_tokens = sum(req.seqlen for req in self.waiting_queue)
        if self.chunked_req is not None:
            req = self.chunked_req
            num_pending_tokens += req.seqlen - len(req.prefix_indices) - chunk_deduct
        return num_pending_tokens

    def get_loads(self: Scheduler, req: GetLoadsReqInput = None) -> GetLoadsReqOutput:
        """
        Get comprehensive load metrics for /v1/loads endpoint.

        Args:
            req: Request containing include list and optional dp_rank filter

        Returns:
            GetLoadsReqOutput with core metrics and optional detailed sections
        """
        if req is None:
            req = GetLoadsReqInput()

        include = set(req.include) if req.include else {"core"}
        include_all = "all" in include

        num_running_reqs = len(self.running_batch.reqs)

        waiting_queues = [self.waiting_queue]
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            waiting_queues.append(self.disagg_prefill_bootstrap_queue.queue)
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            waiting_queues.append(self.disagg_decode_prealloc_queue.queue)
            waiting_queues.append(self.disagg_decode_transfer_queue.queue)
            waiting_queues.append(self.disagg_decode_prealloc_queue.retracted_queue)

        num_waiting_reqs = sum(len(queue) for queue in waiting_queues)
        num_used_tokens, kv_token_usage = self.get_pool_stats().get_kv_token_stats()
        num_total_tokens = num_used_tokens + sum(
            req.seqlen for queue in waiting_queues for req in queue
        )

        memory = None
        if include_all or "memory" in include:
            try:
                memory = MemoryMetrics(
                    weight_gb=round(
                        self.tp_worker.model_runner.weight_load_mem_usage, 3
                    ),
                    kv_cache_gb=round(
                        self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 3
                    ),
                    graph_gb=round(self.tp_worker.model_runner.graph_mem_usage, 3),
                    token_capacity=int(self.max_total_num_tokens),
                )
            except AttributeError as e:
                logger.debug(f"Memory metrics not available: {e}")

        speculative = None
        if include_all or "spec" in include:
            if not self.spec_algorithm.is_none() and self.spec_total_num_forward_ct > 0:
                speculative = SpeculativeMetrics(
                    accept_length=(
                        self.spec_total_num_accept_tokens
                        / self.spec_total_num_forward_ct
                    ),
                    accept_rate=self.stats.spec_accept_rate,
                )

        lora = None
        if include_all or "lora" in include:
            if hasattr(self, "lora_scheduler") and self.lora_scheduler is not None:
                lora = LoRAMetrics(
                    slots_used=self.stats.lora_pool_slots_used,
                    slots_total=self.stats.lora_pool_slots_total,
                    utilization=self.stats.lora_pool_utilization,
                )

        disaggregation = None
        if include_all or "disagg" in include:
            mode_str = "null"
            prefill_bootstrap = 0
            prefill_inflight = 0
            decode_prealloc = 0
            decode_transfer = 0
            decode_retracted = 0

            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                mode_str = "prefill"
                prefill_bootstrap = len(self.disagg_prefill_bootstrap_queue.queue)
                prefill_inflight = len(self.disagg_prefill_inflight_queue)
            elif self.disaggregation_mode == DisaggregationMode.DECODE:
                mode_str = "decode"
                decode_prealloc = len(self.disagg_decode_prealloc_queue.queue)
                decode_transfer = len(self.disagg_decode_transfer_queue.queue)
                decode_retracted = len(
                    self.disagg_decode_prealloc_queue.retracted_queue
                )

            disaggregation = DisaggregationMetrics(
                mode=mode_str,
                prefill_bootstrap_queue_reqs=prefill_bootstrap,
                prefill_inflight_queue_reqs=prefill_inflight,
                decode_prealloc_queue_reqs=decode_prealloc,
                decode_transfer_queue_reqs=decode_transfer,
                decode_retracted_queue_reqs=decode_retracted,
                kv_transfer_speed_gb_s=self.stats.kv_transfer_speed_gb_s,
                kv_transfer_latency_ms=self.stats.kv_transfer_latency_ms,
            )

        queues = None
        if include_all or "queues" in include:
            queues = QueueMetrics(
                waiting=len(self.waiting_queue),
                grammar=self.stats.num_grammar_queue_reqs,
                paused=self.stats.num_paused_reqs,
                retracted=self.stats.num_retracted_reqs,
            )

        return GetLoadsReqOutput(
            dp_rank=self.dp_rank,
            timestamp=time.time(),
            num_running_reqs=num_running_reqs,
            num_waiting_reqs=num_waiting_reqs,
            num_used_tokens=num_used_tokens,
            num_total_tokens=num_total_tokens,
            max_total_num_tokens=self.max_total_num_tokens,
            token_usage=round(kv_token_usage, 4),
            gen_throughput=round(self.stats.gen_throughput, 2),
            cache_hit_rate=round(self.stats.cache_hit_rate, 4),
            utilization=round(self.stats.utilization, 4),
            max_running_requests=self.max_running_requests,
            memory=memory,
            speculative=speculative,
            lora=lora,
            disaggregation=disaggregation,
            queues=queues,
        )

    def update_device_timer(self: Scheduler):
        if not ENABLE_METRICS_DEVICE_TIMER:
            return
        self.forward_pass_device_timer._report()
        now = time.perf_counter()
        if self._device_timer_window_batch_count == 0:
            self._device_timer_window_start = now
            self._device_timer_window_gpu_time = 0.0
            cpu_time = 0
            self.fwd_occupancy = float("nan")
        else:
            cpu_time = now - self._device_timer_window_start
            self.fwd_occupancy = min(
                self._device_timer_window_gpu_time / cpu_time * 100, 100
            )
        # ratio = self._device_timer_window_gpu_time / cpu_time if cpu_time > 0 else float("nan")
        # print(f"{self._device_timer_window_batch_count=} {self.fwd_occupancy=}, {self._device_timer_window_gpu_time=}, {cpu_time=}, {ratio=}")
        self._device_timer_window_batch_count += 1
        if (
            self._device_timer_window_batch_count
            >= self.server_args.decode_log_interval
        ):
            self._device_timer_window_batch_count = 0

    def reset_device_timer_window(self: Scheduler):
        if ENABLE_METRICS_DEVICE_TIMER:
            self._device_timer_window_batch_count = 0
            self.fwd_occupancy = float("nan")
