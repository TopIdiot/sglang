from __future__ import annotations

import atexit
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import yarn_get_mscale
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpecInput

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


_LOCAL_Q_HEADS = 6
_LOCAL_KV_HEADS = 1
_HEAD_DIM = 256
_PAGE_SIZE = 16
_CUDA_GRAPH_WORKSPACE_BYTES_ENV = "SGLANG_MK_DECODE_CUDA_GRAPH_WORKSPACE_BYTES"
_DEFAULT_CUDA_GRAPH_WORKSPACE_BYTES = 128 * 1024 * 1024

logger = logging.getLogger(__name__)


@dataclass
class MkDecodeAttentionMetadata:
    page_ids_by_window_left: dict[int, torch.Tensor]
    workspace_by_window_left: dict[int, object]
    token_counts_cpu: torch.Tensor


@dataclass
class _CudaGraphWorkspaceAllocation:
    offset: int
    stride: int
    size: int


def _parse_capacity_bytes(value: str | None, default: int) -> int:
    if value is None or not value.strip():
        return int(default)
    text = value.strip().lower()
    multipliers = (
        ("gib", 1024**3),
        ("gb", 1024**3),
        ("g", 1024**3),
        ("mib", 1024**2),
        ("mb", 1024**2),
        ("m", 1024**2),
        ("kib", 1024),
        ("kb", 1024),
        ("k", 1024),
        ("b", 1),
    )
    multiplier = 1
    for suffix, suffix_multiplier in multipliers:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            multiplier = suffix_multiplier
            break
    try:
        capacity = int(text) * multiplier
    except ValueError as exc:
        raise ValueError(
            f"{_CUDA_GRAPH_WORKSPACE_BYTES_ENV} must be an integer byte count "
            "or use a k/m/g suffix"
        ) from exc
    if capacity <= 0:
        raise ValueError(f"{_CUDA_GRAPH_WORKSPACE_BYTES_ENV} must be positive")
    return int(capacity)


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


class MkDecodeAttentionBackend(AttentionBackend):
    """WeLMv4.5 mk decode-attention backend.

    The mk workspace is the attention metadata. The backend only supports decode;
    use SGLang's hybrid backend with a regular prefill backend.
    """

    def __init__(self, model_runner: "ModelRunner"):
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.max_context_len = int(model_runner.model_config.context_len)
        self.page_size = int(model_runner.server_args.page_size)
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.forward_metadata: Optional[MkDecodeAttentionMetadata] = None
        self._timing_path = os.environ.get("SGLANG_MK_DECODE_PLAN_TIMING_PATH")
        self._timing_file = None
        self._timing_records = []
        if self._timing_path:
            atexit.register(self._flush_timing_records)
        self._shape_dump_path = os.environ.get("SGLANG_MK_DECODE_SHAPE_DUMP_PATH")
        self._shape_dump_file = None
        self._shape_dump_records = []
        self._shape_dump_count = 0
        self._shape_dump_limit = int(
            os.environ.get("SGLANG_MK_DECODE_SHAPE_DUMP_LIMIT", "256")
        )
        if self._shape_dump_path:
            atexit.register(self._flush_shape_dump_records)
        self._workspace_by_signature: dict[
            tuple[int, int, int, bool, float], object
        ] = {}
        self._cuda_graph_workspace_capacity_bytes = _parse_capacity_bytes(
            os.environ.get(_CUDA_GRAPH_WORKSPACE_BYTES_ENV),
            _DEFAULT_CUDA_GRAPH_WORKSPACE_BYTES,
        )
        self._cuda_graph_workspace_buffer: Optional[torch.Tensor] = None
        self._cuda_graph_workspace_allocations: dict[
            tuple[int, int, int, bool, float], _CudaGraphWorkspaceAllocation
        ] = {}
        self._cuda_graph_workspace_next_offset = 0
        self._cuda_graph_workspace_frozen_signatures: set[
            tuple[int, int, int, bool, float]
        ] = set()
        self._page_ids_buffer_by_signature: dict[
            tuple[int, int, int], torch.Tensor
        ] = {}
        self._cuda_graph_seq_len_fill_value = None
        self._cuda_graph_capture_state = None

        self._validate_runner()
        self._sample_layer_id_by_window_left = self._collect_window_layer_samples()
        self._decode_window_lefts = tuple(sorted(self._sample_layer_id_by_window_left))
        self._sm_scale_by_window_left = self._collect_window_sm_scales()

        try:
            from mk.kernels.decode_attention_welmv45 import (  # noqa: F401
                decode_attention_welmv45_init_workspace,
                decode_attention_welmv45_run,
            )
        except Exception as exc:  # pragma: no cover - import path/env specific
            raise RuntimeError(
                "mk_decode_attention backend requires the mk package with "
                "mk.kernels.decode_attention_welmv45 available."
            ) from exc

    def _record_timing(self, record: dict) -> None:
        if not self._timing_path:
            return
        if self._timing_file is None:
            path = self._timing_path
            if path.endswith(os.sep) or os.path.isdir(path):
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, f"mk_decode_plan_timing_{os.getpid()}.jsonl")
            else:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                root, ext = os.path.splitext(path)
                path = f"{root}_{os.getpid()}{ext or '.jsonl'}"
            self._timing_file = path
        self._timing_records.append(record)
        if len(self._timing_records) >= 16:
            self._flush_timing_records()

    def _flush_timing_records(self) -> None:
        if not self._timing_records or self._timing_file is None:
            return
        with open(self._timing_file, "a", encoding="utf-8") as f:
            for record in self._timing_records:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._timing_records.clear()

    def _record_shape_dump(self, record: dict) -> None:
        if not self._shape_dump_path:
            return
        if self._shape_dump_limit >= 0 and self._shape_dump_count >= self._shape_dump_limit:
            return
        if self._shape_dump_file is None:
            path = self._shape_dump_path
            if path.endswith(os.sep) or os.path.isdir(path):
                os.makedirs(path, exist_ok=True)
                path = os.path.join(path, f"mk_decode_shape_dump_{os.getpid()}.jsonl")
            else:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                root, ext = os.path.splitext(path)
                path = f"{root}_{os.getpid()}{ext or '.jsonl'}"
            self._shape_dump_file = path
        self._shape_dump_count += 1
        self._shape_dump_records.append(record)
        if len(self._shape_dump_records) >= 4:
            self._flush_shape_dump_records()

    def _flush_shape_dump_records(self) -> None:
        if not self._shape_dump_records or self._shape_dump_file is None:
            return
        with open(self._shape_dump_file, "a", encoding="utf-8") as f:
            for record in self._shape_dump_records:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._shape_dump_records.clear()

    def _validate_runner(self) -> None:
        if self.page_size != _PAGE_SIZE:
            raise ValueError(
                f"mk_decode_attention requires page_size={_PAGE_SIZE}, "
                f"got {self.page_size}."
            )
        model_config = self.model_runner.model_config
        num_q_heads = model_config.num_attention_heads
        tp_size = getattr(self.model_runner, "tp_size", 1)
        if hasattr(model_config, "get_num_kv_heads"):
            local_kv_heads = model_config.get_num_kv_heads(tp_size)
        else:
            num_kv_heads = getattr(model_config, "num_key_value_heads", None)
            local_kv_heads = (
                num_kv_heads // tp_size if num_kv_heads is not None else None
            )
        local_q_heads = num_q_heads // tp_size
        if (
            local_q_heads != _LOCAL_Q_HEADS
            or local_kv_heads != _LOCAL_KV_HEADS
            or model_config.head_dim != _HEAD_DIM
        ):
            raise ValueError(
                "mk_decode_attention only supports local "
                f"q_heads={_LOCAL_Q_HEADS}, kv_heads={_LOCAL_KV_HEADS}, "
                f"head_dim={_HEAD_DIM}; got q_heads={local_q_heads}, "
                f"kv_heads={local_kv_heads}, head_dim={model_config.head_dim}."
            )

    def _collect_window_layer_samples(self) -> dict[int, int]:
        samples: dict[int, int] = {}
        layerwise = getattr(
            self.model_runner.model_config.hf_config,
            "sliding_window_size_layerwise",
            [],
        )
        num_layers = int(self.model_runner.model_config.num_hidden_layers)
        for layer_id in range(num_layers):
            value = int(layerwise[layer_id]) if layer_id < len(layerwise) else -1
            window_left = (
                value
                if value > 0 and value + 1 < self.max_context_len
                else -1
            )
            samples.setdefault(window_left, layer_id)
        if not samples:
            samples[-1] = 0
        return samples

    def _default_sm_scale(self) -> float:
        scale = 1.0 / math.sqrt(float(_HEAD_DIM))
        rope_scaling = getattr(self.model_runner.model_config.hf_config, "rope_scaling", None)
        if not isinstance(rope_scaling, dict) or rope_scaling.get("type") != "yarn":
            return scale
        mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
        if rope_scaling.get("apply_softmax_scale", False) and mscale_all_dim:
            mscale = yarn_get_mscale(
                float(rope_scaling["factor"]),
                float(mscale_all_dim),
            )
            scale *= mscale * mscale
        return float(scale)

    def _collect_window_sm_scales(self) -> dict[int, float]:
        default_scale = self._default_sm_scale()
        model = getattr(self.model_runner, "model", None)
        layers = getattr(getattr(model, "model", model), "layers", None)
        scales: dict[int, float] = {}
        for window_left, layer_id in self._sample_layer_id_by_window_left.items():
            scale = default_scale
            try:
                layer = layers[layer_id]
                attn = getattr(layer, "self_attn", None)
                radix_attn = getattr(attn, "attn", None)
                scale = float(getattr(radix_attn, "scaling", scale))
            except Exception:  # noqa: BLE001
                pass
            scales[window_left] = scale
        return scales

    def _normalize_token_counts_cpu(
        self, seq_lens_cpu: torch.Tensor, bs: int
    ) -> torch.Tensor:
        if seq_lens_cpu is None:
            raise RuntimeError("mk_decode_attention requires CPU seq_lens metadata.")
        token_counts = seq_lens_cpu[:bs]
        if token_counts.device.type != "cpu":
            raise RuntimeError("mk_decode_attention requires seq_lens_cpu on CPU.")
        if token_counts.dtype != torch.int32:
            token_counts = token_counts.to(dtype=torch.int32)
        if not token_counts.is_contiguous():
            token_counts = token_counts.contiguous()
        return token_counts

    def _page_ids_signature(
        self, bs: int, max_pages: int, window_left: int
    ) -> tuple[int, int, int]:
        return (int(bs), int(max_pages), int(window_left))

    def _workspace_signature(
        self,
        bs: int,
        max_pages: int,
        window_left: int,
        has_sinks: bool,
        sm_scale: float,
    ) -> tuple[int, int, int, bool, float]:
        return (
            int(bs),
            int(max_pages),
            int(window_left),
            bool(has_sinks),
            float(sm_scale),
        )

    @staticmethod
    def _format_bytes(value: int) -> str:
        value = int(value)
        if value >= 1024**2:
            return f"{value / 1024**2:.3f} MiB"
        if value >= 1024:
            return f"{value / 1024:.3f} KiB"
        return f"{value} B"

    def _ensure_cuda_graph_workspace_buffer(self) -> torch.Tensor:
        buffer = self._cuda_graph_workspace_buffer
        if buffer is None:
            buffer = torch.empty(
                (self._cuda_graph_workspace_capacity_bytes,),
                dtype=torch.uint8,
                device=self.device,
            )
            self._cuda_graph_workspace_buffer = buffer
        return buffer

    def _allocate_cuda_graph_workspace_slice(
        self,
        signature: tuple[int, int, int, bool, float],
        required_stride: int,
        required_size: int,
    ) -> _CudaGraphWorkspaceAllocation:
        offset = _align_up(self._cuda_graph_workspace_next_offset, 256)
        end = offset + int(required_size)
        capacity = int(self._cuda_graph_workspace_capacity_bytes)
        if end > capacity:
            raise RuntimeError(
                "mk_decode_attention CUDA graph workspace capacity exceeded: "
                f"signature={signature}, required={self._format_bytes(required_size)}, "
                f"used={self._format_bytes(self._cuda_graph_workspace_next_offset)}, "
                f"capacity={self._format_bytes(capacity)}. Increase "
                f"{_CUDA_GRAPH_WORKSPACE_BYTES_ENV}."
            )
        allocation = _CudaGraphWorkspaceAllocation(
            offset=offset,
            stride=int(required_stride),
            size=int(required_size),
        )
        self._cuda_graph_workspace_allocations[signature] = allocation
        self._cuda_graph_workspace_next_offset = end
        return allocation

    def _install_cuda_graph_workspace(
        self,
        signature: tuple[int, int, int, bool, float],
        workspace: object,
    ) -> bool:
        """Pin graph workspace storage to this backend's persistent GPU buffer.

        Returns whether the workspace GPU pointer or per-layer stride changed after
        mk initialized the workspace, which means the caller must run init again to
        copy the CPU template into the graph-owned storage.
        """

        required_stride = int(workspace.workspace_size)
        required_size = required_stride * int(workspace.num_layers)
        if required_stride <= 0 or required_size <= 0:
            raise RuntimeError(
                "mk_decode_attention CUDA graph workspace requires a non-empty "
                f"workspace, got stride={required_stride}, size={required_size}."
            )

        buffer = self._ensure_cuda_graph_workspace_buffer()
        allocation = self._cuda_graph_workspace_allocations.get(signature)
        if allocation is None:
            allocation = self._allocate_cuda_graph_workspace_slice(
                signature, required_stride, required_size
            )
        elif required_stride > allocation.stride:
            if signature in self._cuda_graph_workspace_frozen_signatures:
                raise RuntimeError(
                    "mk_decode_attention CUDA graph workspace signature requires "
                    "a larger stride after CUDA graph capture: "
                    f"signature={signature}, captured_stride={allocation.stride}, "
                    f"required_stride={required_stride}. Increase the graph bucket "
                    "capacity or disable mk_decode_attention CUDA graph replay."
                )
            allocation = self._allocate_cuda_graph_workspace_slice(
                signature, required_stride, required_size
            )

        old_workspace = getattr(workspace, "workspace", None)
        old_workspace_ptr = (
            int(old_workspace.data_ptr())
            if isinstance(old_workspace, torch.Tensor)
            else 0
        )
        old_stride = int(getattr(workspace, "workspace_size", 0))

        workspace.workspace_size = int(allocation.stride)
        workspace.workspace = buffer[allocation.offset : allocation.offset + allocation.size]
        cpu_workspace = getattr(workspace, "cpu_workspace", None)
        if (
            not isinstance(cpu_workspace, torch.Tensor)
            or cpu_workspace.device.type != "cpu"
            or int(cpu_workspace.numel()) < int(allocation.stride)
        ):
            workspace.cpu_workspace = torch.empty(
                (int(allocation.stride),),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )

        return (
            old_workspace_ptr != int(workspace.workspace.data_ptr())
            or old_stride != int(allocation.stride)
        )

    def _freeze_current_cuda_graph_workspace_signatures(self) -> None:
        metadata = self.forward_metadata
        if metadata is None:
            return
        for workspace in metadata.workspace_by_window_left.values():
            signature = self._workspace_signature(
                int(workspace.num_sub_tasks),
                int(workspace.max_pages),
                int(workspace.window_left),
                bool(workspace.has_sinks),
                float(workspace.sm_scale),
            )
            self._cuda_graph_workspace_frozen_signatures.add(signature)

    def _get_page_ids_buffer(
        self,
        *,
        bs: int,
        max_pages: int,
        window_left: int,
    ) -> torch.Tensor:
        signature = self._page_ids_signature(bs, max_pages, window_left)
        page_ids = self._page_ids_buffer_by_signature.get(signature)
        if page_ids is None:
            page_ids = torch.empty(
                (bs, max_pages),
                dtype=torch.int32,
                device=self.device,
            )
            self._page_ids_buffer_by_signature[signature] = page_ids
        return page_ids

    def _build_page_ids(
        self,
        *,
        req_pool_indices: torch.Tensor,
        max_pages: int,
        window_left: int,
        token_to_kv_pool,
        use_swa_mapping: bool,
        raw_bs: int | None = None,
        timing_record: dict | None = None,
    ) -> torch.Tensor:
        begin_ns = time.perf_counter_ns() if timing_record is not None else 0
        bs = int(req_pool_indices.shape[0])
        page_ids = self._get_page_ids_buffer(
            bs=bs,
            max_pages=max_pages,
            window_left=window_left,
        )
        token_slots = self.req_to_token[
            req_pool_indices,
            : max_pages * self.page_size : self.page_size,
        ]
        if use_swa_mapping:
            token_slots = token_to_kv_pool.translate_loc_from_full_to_swa(token_slots)
        page_ids.copy_(torch.div(token_slots, self.page_size, rounding_mode="floor"))
        if raw_bs is not None and int(raw_bs) < bs:
            page_ids[int(raw_bs) :].zero_()
        if timing_record is not None:
            timing_record["build_page_ids_ms"] = (
                time.perf_counter_ns() - begin_ns
            ) / 1.0e6
        return page_ids

    def _window_uses_swa_mapping(self, token_to_kv_pool, window_left: int) -> bool:
        if window_left < 0 or not hasattr(
            token_to_kv_pool, "translate_loc_from_full_to_swa"
        ):
            return False
        sample_layer_id = self._sample_layer_id_by_window_left[window_left]
        if hasattr(token_to_kv_pool, "is_swa_layer"):
            return bool(token_to_kv_pool.is_swa_layer(sample_layer_id))
        layers_mapping = getattr(token_to_kv_pool, "layers_mapping", None)
        if layers_mapping is not None:
            _, kind = layers_mapping[sample_layer_id]
            return bool(kind is True or kind == "swa")
        return True

    def _summarize_mk_plan_task_pressure(
        self,
        *,
        workspace=None,
        token_counts_cpu: torch.Tensor | None = None,
        workspace_token_counts_cpu: torch.Tensor | None = None,
        max_pages: int | None = None,
        window_left: int = -1,
        forced_num_splits: int | None = None,
    ) -> dict:
        if workspace is not None:
            plan = getattr(workspace, "_plan", None)
            split_plan = getattr(plan, "split_plan", None)
            if plan is None or split_plan is None:
                return {}
            return self._summarize_split_plan_task_pressure(
                split_plan=split_plan,
                worker_count=int(plan.worker_count),
                max_splits=int(plan.max_splits),
                batch=int(workspace.num_sub_tasks),
            )
        if token_counts_cpu is None or max_pages is None:
            return {}
        try:
            from mk.kernels.decode_attention_welmv45 import _make_split_plan

            if forced_num_splits is None and workspace_token_counts_cpu is not None:
                workspace_plan = _make_split_plan(
                    token_counts=workspace_token_counts_cpu,
                    max_pages=int(max_pages),
                    worker_count=torch.cuda.get_device_properties(
                        self.device
                    ).multi_processor_count,
                    num_splits=None,
                    split_kv_chunk_size=None,
                    window_left=int(window_left),
                    fma_max_tokens=1024,
                    include_split_token_counts=False,
                )
                forced_num_splits = int(workspace_plan.max_splits)
            plan = _make_split_plan(
                token_counts=token_counts_cpu,
                max_pages=int(max_pages),
                worker_count=torch.cuda.get_device_properties(
                    self.device
                ).multi_processor_count,
                num_splits=forced_num_splits,
                split_kv_chunk_size=None,
                window_left=int(window_left),
                fma_max_tokens=1024,
                include_split_token_counts=False,
            )
            assert plan.split_plan is not None
            return self._summarize_split_plan_task_pressure(
                split_plan=plan.split_plan,
                worker_count=int(plan.worker_count),
                max_splits=int(plan.max_splits),
                batch=int(token_counts_cpu.numel()),
            )
        except Exception as exc:
            return {
                "task_pressure_error_type": type(exc).__name__,
                "task_pressure_error": str(exc)[:512],
            }

    @staticmethod
    def _summarize_split_plan_task_pressure(
        *,
        split_plan: torch.Tensor,
        worker_count: int,
        max_splits: int,
        batch: int,
    ) -> dict:
        from mk.kernels.decode_attention_welmv45 import (
            _MAX_MERGE_WORKERS,
            _PLAN_ACTIVE_SPLITS_FIELD,
            _PLAN_MERGE_WORKER_FIELD,
            _PLAN_WORKER_FIELD,
        )

        records = split_plan.detach().cpu()
        split_counts = [0] * worker_count
        direct_counts = [0] * worker_count
        merge_counts = [0] * worker_count
        worker_tokens = [0] * worker_count
        request_seen = [False] * batch
        request_merge_workers = [0] * batch

        for row in records.tolist():
            request = int(row[0])
            if 0 <= request < batch and not request_seen[request]:
                request_seen[request] = True
                request_merge_workers[request] = int(row[_PLAN_MERGE_WORKER_FIELD])
            worker = int(row[_PLAN_WORKER_FIELD])
            if (
                max_splits == 1
                and 0 <= worker < worker_count
                and int(row[_PLAN_ACTIVE_SPLITS_FIELD]) == 1
            ):
                direct_counts[worker] += 1
            elif 0 <= worker < worker_count and int(row[1]) >= 0:
                split_counts[worker] += 1
                worker_tokens[worker] += int(row[4])

        if max_splits > 1:
            for request in range(batch):
                if not request_seen[request]:
                    continue
                merge_worker = request_merge_workers[request]
                if 0 <= merge_worker < worker_count:
                    merge_counts[merge_worker] += 1

        total_counts = [
            split_counts[i] + direct_counts[i] + merge_counts[i]
            for i in range(worker_count)
        ]
        max_tasks = max(total_counts) if total_counts else 0
        max_worker = total_counts.index(max_tasks) if total_counts else -1
        max_split_records = int(batch) * int(max_splits)
        merge_worker_count = (
            min(batch, worker_count, _MAX_MERGE_WORKERS)
            if max_splits > 1
            else 0
        )
        max_local_tasks_capacity = (
            (max_split_records + worker_count - 1) // worker_count
            + 2
            + (
                (batch + merge_worker_count - 1) // merge_worker_count
                if merge_worker_count
                else 0
            )
            + 1
        )
        return {
            "task_capacity": int(max_local_tasks_capacity),
            "task_max_worker": int(max_worker),
            "task_max_count": int(max_tasks),
            "task_max_split_count": int(split_counts[max_worker])
            if max_worker >= 0
            else 0,
            "task_max_direct_count": int(direct_counts[max_worker])
            if max_worker >= 0
            else 0,
            "task_max_merge_count": int(merge_counts[max_worker])
            if max_worker >= 0
            else 0,
            "task_max_split_tokens": int(worker_tokens[max_worker])
            if max_worker >= 0
            else 0,
            "task_capacity_margin": int(max_local_tasks_capacity - max_tasks),
            "task_over_capacity": bool(max_tasks > max_local_tasks_capacity),
            "task_top_workers": [
                {
                    "worker": int(worker),
                    "total": int(total_counts[worker]),
                    "split": int(split_counts[worker]),
                    "direct": int(direct_counts[worker]),
                    "merge": int(merge_counts[worker]),
                    "split_tokens": int(worker_tokens[worker]),
                }
                for worker in sorted(
                    range(worker_count),
                    key=lambda idx: (total_counts[idx], split_counts[idx], idx),
                    reverse=True,
                )[:8]
            ],
        }

    def _init_one_metadata(
        self,
        *,
        bs: int,
        planner_token_counts_cpu: torch.Tensor,
        workspace_token_counts_cpu: torch.Tensor,
        req_pool_indices: torch.Tensor,
        token_to_kv_pool,
        window_left: int,
        has_sinks: bool,
        sm_scale: float,
        preserve_workspace_plan_capacity: bool,
        raw_bs: int | None,
        timing_record: dict | None = None,
        dump_page_ids: bool = False,
        use_cuda_graph_workspace: bool = False,
    ) -> tuple[torch.Tensor, object]:
        from mk.kernels.decode_attention_welmv45 import (
            decode_attention_welmv45_init_workspace,
        )

        max_seq_len = max(int(workspace_token_counts_cpu.max().item()), 1)
        max_pages = max((max_seq_len + self.page_size - 1) // self.page_size, 1)
        page_ids = self._build_page_ids(
            req_pool_indices=req_pool_indices,
            max_pages=max_pages,
            window_left=window_left,
            token_to_kv_pool=token_to_kv_pool,
            use_swa_mapping=self._window_uses_swa_mapping(
                token_to_kv_pool, window_left
            ),
            raw_bs=raw_bs,
            timing_record=timing_record,
        )
        sample_layer_id = self._sample_layer_id_by_window_left[window_left]
        key_cache = token_to_kv_pool.get_key_buffer(sample_layer_id)
        num_cache_pages = int(key_cache.shape[0]) // self.page_size
        if timing_record is not None:
            timing_record["max_pages"] = int(max_pages)
            timing_record["num_cache_pages"] = int(num_cache_pages)
            timing_record["sample_layer_id"] = int(sample_layer_id)
            if dump_page_ids:
                timing_record["page_ids"] = page_ids.detach().cpu().tolist()
        signature = self._workspace_signature(
            bs,
            max_pages,
            window_left,
            has_sinks,
            sm_scale,
        )
        existing_workspace = (
            self._workspace_by_signature.get(signature)
            if preserve_workspace_plan_capacity
            else None
        )
        forced_num_splits = None
        if preserve_workspace_plan_capacity and existing_workspace is not None:
            forced_num_splits = int(existing_workspace._plan.max_splits)
        if timing_record is not None:
            timing_record["forced_num_splits"] = (
                int(forced_num_splits) if forced_num_splits is not None else None
            )
        allow_workspace_resize = not (
            use_cuda_graph_workspace
            and existing_workspace is not None
            and signature in self._cuda_graph_workspace_frozen_signatures
        )
        init_begin_ns = time.perf_counter_ns() if timing_record is not None else 0
        workspace = decode_attention_welmv45_init_workspace(
            page_ids,
            planner_token_counts_cpu,
            num_cache_pages=num_cache_pages,
            num_layers=int(self.model_runner.model_config.num_hidden_layers),
            has_sinks=has_sinks,
            sm_scale=sm_scale,
            window_size=(window_left, 0) if window_left >= 0 else None,
            num_splits=forced_num_splits,
            workspace=existing_workspace,
            allow_resize=allow_workspace_resize,
        )
        if use_cuda_graph_workspace:
            needs_reinit = self._install_cuda_graph_workspace(signature, workspace)
            if needs_reinit:
                workspace = decode_attention_welmv45_init_workspace(
                    page_ids,
                    planner_token_counts_cpu,
                    num_cache_pages=num_cache_pages,
                    num_layers=int(self.model_runner.model_config.num_hidden_layers),
                    has_sinks=has_sinks,
                    sm_scale=sm_scale,
                    window_size=(window_left, 0) if window_left >= 0 else None,
                    num_splits=forced_num_splits,
                    workspace=workspace,
                    allow_resize=allow_workspace_resize,
                )
                self._install_cuda_graph_workspace(signature, workspace)
        if timing_record is not None:
            timing_record["workspace_init_ms"] = (
                time.perf_counter_ns() - init_begin_ns
            ) / 1.0e6
            timing_record["num_split_records"] = int(
                workspace._plan.num_split_records
            )
            timing_record["max_splits"] = int(workspace._plan.max_splits)
            timing_record["chunk_pages"] = int(workspace._plan.chunk_pages)
            timing_record.update(
                self._summarize_mk_plan_task_pressure(workspace=workspace)
            )
        self._workspace_by_signature[signature] = workspace
        return page_ids, workspace

    def _init_metadata_common(
        self,
        *,
        bs: int,
        seq_lens_cpu: torch.Tensor,
        workspace_seq_lens_cpu: torch.Tensor | None = None,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        token_to_kv_pool,
        raw_bs: int | None = None,
        timing_phase: str | None = None,
        use_cuda_graph_workspace: bool = False,
    ) -> None:
        total_begin_ns = (
            time.perf_counter_ns()
            if self._timing_path and timing_phase is not None
            else 0
        )
        dump_shapes = bool(self._shape_dump_path and timing_phase is not None)
        if forward_mode.is_idle():
            self.forward_metadata = None
            return
        if not forward_mode.is_decode():
            raise RuntimeError(
                "mk_decode_attention only supports decode. Set "
                "--prefill-attention-backend to another backend and "
                "--decode-attention-backend mk_decode_attention."
            )
        token_counts_cpu = self._normalize_token_counts_cpu(seq_lens_cpu, bs)
        raw_bs = bs if raw_bs is None else max(0, min(int(raw_bs), bs))
        if raw_bs < bs:
            token_counts_cpu = token_counts_cpu.clone()
            token_counts_cpu[raw_bs:] = 0
        preserve_workspace_plan_capacity = workspace_seq_lens_cpu is not None and (
            workspace_seq_lens_cpu is not seq_lens_cpu
        )
        workspace_token_counts_cpu = (
            self._normalize_token_counts_cpu(workspace_seq_lens_cpu, bs)
            if workspace_seq_lens_cpu is not None
            else token_counts_cpu
        )
        page_ids_by_window_left: dict[int, torch.Tensor] = {}
        workspace_by_window_left: dict[int, object] = {}
        timing_windows = [] if total_begin_ns or dump_shapes else None
        for window_left in self._decode_window_lefts:
            timing_record = (
                {"window_left": int(window_left)} if timing_windows is not None else None
            )
            sm_scale = self._sm_scale_by_window_left[window_left]
            try:
                page_ids, workspace = self._init_one_metadata(
                    bs=bs,
                    planner_token_counts_cpu=token_counts_cpu,
                    workspace_token_counts_cpu=workspace_token_counts_cpu,
                    req_pool_indices=req_pool_indices,
                    token_to_kv_pool=token_to_kv_pool,
                    window_left=window_left,
                    has_sinks=True,
                    sm_scale=sm_scale,
                    preserve_workspace_plan_capacity=preserve_workspace_plan_capacity,
                    raw_bs=raw_bs,
                    timing_record=timing_record,
                    dump_page_ids=dump_shapes,
                    use_cuda_graph_workspace=use_cuda_graph_workspace,
                )
            except Exception as exc:
                pressure_summary = (
                    self._summarize_mk_plan_task_pressure(
                        token_counts_cpu=token_counts_cpu,
                        workspace_token_counts_cpu=workspace_token_counts_cpu,
                        max_pages=(
                            int(timing_record["max_pages"])
                            if timing_record is not None
                            and "max_pages" in timing_record
                            else None
                        ),
                        window_left=int(window_left),
                        forced_num_splits=(
                            int(timing_record["forced_num_splits"])
                            if timing_record is not None
                            and timing_record.get("forced_num_splits") is not None
                            else None
                        ),
                    )
                    if timing_record is not None
                    else {}
                )
                active_counts = token_counts_cpu[:raw_bs]
                workspace_counts = workspace_token_counts_cpu[:bs]
                logger.exception(
                    "mk_decode_attention metadata init failed: %s",
                    json.dumps(
                        {
                            "phase": timing_phase,
                            "bs": int(bs),
                            "raw_bs": int(raw_bs),
                            "seq_min": int(active_counts.min().item())
                            if raw_bs > 0
                            else 0,
                            "seq_max": int(active_counts.max().item())
                            if raw_bs > 0
                            else 0,
                            "workspace_seq_min": int(workspace_counts.min().item())
                            if bs > 0
                            else 0,
                            "workspace_seq_max": int(workspace_counts.max().item())
                            if bs > 0
                            else 0,
                            "window_left": int(window_left),
                            "preserve_workspace_plan_capacity": bool(
                                preserve_workspace_plan_capacity
                            ),
                            "forced_num_splits": (
                                int(timing_record["forced_num_splits"])
                                if timing_record is not None
                                and timing_record.get("forced_num_splits") is not None
                                else None
                            ),
                            "token_counts": token_counts_cpu.tolist(),
                            "workspace_token_counts": workspace_token_counts_cpu.tolist(),
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:512],
                            **pressure_summary,
                        },
                        separators=(",", ":"),
                    ),
                )
                if self._timing_path and timing_phase is not None:
                    error_record = {
                        "event": "metadata_init_error",
                        "phase": timing_phase,
                        "pid": os.getpid(),
                        "bs": int(bs),
                        "raw_bs": int(raw_bs),
                        "seq_min": int(active_counts.min().item())
                        if raw_bs > 0
                        else 0,
                        "seq_max": int(active_counts.max().item())
                        if raw_bs > 0
                        else 0,
                        "workspace_seq_min": int(workspace_counts.min().item())
                        if bs > 0
                        else 0,
                        "workspace_seq_max": int(workspace_counts.max().item())
                        if bs > 0
                        else 0,
                        "window_left": int(window_left),
                        "preserve_workspace_plan_capacity": bool(
                            preserve_workspace_plan_capacity
                        ),
                        "token_counts": token_counts_cpu.tolist(),
                        "workspace_token_counts": workspace_token_counts_cpu.tolist(),
                        "page_size": int(self.page_size),
                        "head_dim": int(_HEAD_DIM),
                        "local_q_heads": int(_LOCAL_Q_HEADS),
                        "local_kv_heads": int(_LOCAL_KV_HEADS),
                        "num_layers": int(
                            self.model_runner.model_config.num_hidden_layers
                        ),
                        "sm_scale": float(sm_scale),
                        "window": timing_record,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1024],
                    }
                    error_record.update(pressure_summary)
                    self._record_timing(error_record)
                if dump_shapes:
                    active_counts = token_counts_cpu[:raw_bs]
                    shape_record = {
                        "event": "metadata_init_error",
                        "monotonic_ns": time.perf_counter_ns(),
                        "phase": timing_phase,
                        "pid": os.getpid(),
                        "record_index": int(self._shape_dump_count),
                        "bs": int(bs),
                        "raw_bs": int(raw_bs),
                        "seq_min": int(active_counts.min().item())
                        if raw_bs > 0
                        else 0,
                        "seq_max": int(active_counts.max().item())
                        if raw_bs > 0
                        else 0,
                        "page_size": int(self.page_size),
                        "head_dim": int(_HEAD_DIM),
                        "local_q_heads": int(_LOCAL_Q_HEADS),
                        "local_kv_heads": int(_LOCAL_KV_HEADS),
                        "num_layers": int(
                            self.model_runner.model_config.num_hidden_layers
                        ),
                        "sm_scale": float(sm_scale),
                        "token_counts": token_counts_cpu.tolist(),
                        "workspace_token_counts": workspace_token_counts_cpu.tolist(),
                        "preserve_workspace_plan_capacity": bool(
                            preserve_workspace_plan_capacity
                        ),
                        "windows": [timing_record],
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1024],
                    }
                    shape_record.update(pressure_summary)
                    self._record_shape_dump(shape_record)
                raise
            page_ids_by_window_left[window_left] = page_ids
            workspace_by_window_left[window_left] = workspace
            if timing_record is not None:
                timing_windows.append(timing_record)
        self.forward_metadata = MkDecodeAttentionMetadata(
            page_ids_by_window_left=page_ids_by_window_left,
            workspace_by_window_left=workspace_by_window_left,
            token_counts_cpu=token_counts_cpu,
        )
        if total_begin_ns:
            active_token_counts = token_counts_cpu[:raw_bs]
            self._record_timing(
                {
                    "monotonic_ns": time.perf_counter_ns(),
                    "phase": timing_phase,
                    "pid": os.getpid(),
                    "bs": int(bs),
                    "raw_bs": int(raw_bs),
                    "seq_min": int(active_token_counts.min().item())
                    if raw_bs > 0
                    else 0,
                    "seq_max": int(active_token_counts.max().item())
                    if raw_bs > 0
                    else 0,
                    "total_ms": (time.perf_counter_ns() - total_begin_ns) / 1.0e6,
                    "windows": timing_windows,
                }
            )
        if dump_shapes:
            active_token_counts = token_counts_cpu[:raw_bs]
            self._record_shape_dump(
                {
                    "monotonic_ns": time.perf_counter_ns(),
                    "phase": timing_phase,
                    "pid": os.getpid(),
                    "record_index": int(self._shape_dump_count),
                    "bs": int(bs),
                    "raw_bs": int(raw_bs),
                    "seq_min": int(active_token_counts.min().item())
                    if raw_bs > 0
                    else 0,
                    "seq_max": int(active_token_counts.max().item())
                    if raw_bs > 0
                    else 0,
                    "page_size": int(self.page_size),
                    "head_dim": int(_HEAD_DIM),
                    "local_q_heads": int(_LOCAL_Q_HEADS),
                    "local_kv_heads": int(_LOCAL_KV_HEADS),
                    "num_layers": int(self.model_runner.model_config.num_hidden_layers),
                    "sm_scale": float(sm_scale),
                    "token_counts": token_counts_cpu.tolist(),
                    "workspace_token_counts": workspace_token_counts_cpu.tolist(),
                    "windows": timing_windows,
                }
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._init_metadata_common(
            bs=forward_batch.batch_size,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            workspace_seq_lens_cpu=forward_batch.seq_lens_cpu,
            req_pool_indices=forward_batch.req_pool_indices,
            forward_mode=forward_batch.forward_mode,
            token_to_kv_pool=forward_batch.token_to_kv_pool,
            raw_bs=forward_batch.batch_size,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # Keep one backend-owned CUDA buffer for all captured mk workspaces so
        # graph replay never observes a changed workspace base allocation.
        self._ensure_cuda_graph_workspace_buffer()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        fill_value = getattr(self, "_cuda_graph_seq_len_fill_value", None)
        if fill_value is None:
            raise RuntimeError(
                "mk_decode_attention CUDA graph capture requires a sequence length "
                "bucket from the CUDA graph runner."
            )
        seq_lens_cpu = torch.full(
            (bs,),
            int(fill_value),
            dtype=torch.int32,
            device="cpu",
        )
        self._init_metadata_common(
            bs=bs,
            seq_lens_cpu=seq_lens_cpu,
            workspace_seq_lens_cpu=seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            forward_mode=forward_mode,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            raw_bs=bs,
            use_cuda_graph_workspace=True,
        )
        self._cuda_graph_capture_state = (
            bs,
            seq_lens_cpu,
            req_pool_indices,
            forward_mode,
        )

    def on_after_cuda_graph_warmup(self):
        if self._cuda_graph_capture_state is None:
            return
        bs, seq_lens_cpu, req_pool_indices, forward_mode = (
            self._cuda_graph_capture_state
        )
        self._init_metadata_common(
            bs=bs,
            seq_lens_cpu=seq_lens_cpu,
            workspace_seq_lens_cpu=seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            forward_mode=forward_mode,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            raw_bs=bs,
            use_cuda_graph_workspace=True,
        )
        self._freeze_current_cuda_graph_workspace_signatures()

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        fill_value = getattr(self, "_cuda_graph_seq_len_fill_value", None)
        workspace_seq_lens_cpu = None
        if fill_value is not None:
            workspace_seq_lens_cpu = torch.full(
                (bs,),
                int(fill_value),
                dtype=torch.int32,
                device="cpu",
            )
        replay_forward_batch = getattr(self, "_replay_forward_batch", None)
        raw_bs = int(getattr(replay_forward_batch, "batch_size", bs))
        self._init_metadata_common(
            bs=bs,
            seq_lens_cpu=seq_lens_cpu,
            workspace_seq_lens_cpu=workspace_seq_lens_cpu,
            req_pool_indices=req_pool_indices,
            forward_mode=forward_mode,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            raw_bs=raw_bs,
            timing_phase="replay",
            use_cuda_graph_workspace=True,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.max_context_len

    def _window_left_for_layer(self, layer: RadixAttention) -> int:
        window_left = int(layer.sliding_window_size)
        return (
            window_left
            if window_left > 0 and window_left + 1 < self.max_context_len
            else -1
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        sinks: Optional[torch.Tensor] = None,
    ):
        from mk.kernels.decode_attention_welmv45 import decode_attention_welmv45_run

        if self.forward_metadata is None:
            raise RuntimeError("mk_decode_attention metadata is not initialized.")
        if sinks is None:
            raise RuntimeError("mk_decode_attention requires attention sinks.")
        if layer.logit_cap:
            raise RuntimeError("mk_decode_attention does not support logit_cap.")
        if (
            layer.tp_q_head_num != _LOCAL_Q_HEADS
            or layer.tp_k_head_num != _LOCAL_KV_HEADS
        ):
            raise RuntimeError("mk_decode_attention layer head shape mismatch.")
        if layer.qk_head_dim != _HEAD_DIM or layer.v_head_dim != _HEAD_DIM:
            raise RuntimeError("mk_decode_attention layer head_dim mismatch.")

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

        window_left = self._window_left_for_layer(layer)
        workspace = self.forward_metadata.workspace_by_window_left[window_left]
        page_ids = self.forward_metadata.page_ids_by_window_left[window_left]
        key_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        value_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        key_cache = key_cache.view(-1, self.page_size, _LOCAL_KV_HEADS, _HEAD_DIM)
        value_cache = value_cache.view(-1, self.page_size, _LOCAL_KV_HEADS, _HEAD_DIM)
        query = q.contiguous().view(-1, _LOCAL_Q_HEADS, _HEAD_DIM)
        output, _ = decode_attention_welmv45_run(
            workspace,
            layer_id=int(layer.layer_id),
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            page_ids=page_ids,
            sinks=sinks,
        )
        return output.view(-1, _LOCAL_Q_HEADS * _HEAD_DIM)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        raise RuntimeError(
            "mk_decode_attention only supports decode; use a separate prefill backend."
        )
