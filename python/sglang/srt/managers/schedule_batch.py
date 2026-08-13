from __future__ import annotations

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import ceil_align, is_pin_memory_available

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.

TODO(lmzheng): ModelWorkerBatch seems a bit redundant and we consider removing it in the future.
"""

import copy
import dataclasses
import logging
import re
from concurrent.futures import Future
from enum import Enum, auto
from functools import lru_cache
from http import HTTPStatus
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.context_parallel import CPPrefillSplitSpec
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, DisaggregationMode
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.dllm.mixin.req import ReqDllmMixin
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    MatchPrefixParams,
    zero_match_result,
)
from sglang.srt.mem_cache.common import (
    alloc_for_decode,
    alloc_for_extend,
    evict_from_tree_cache,
    release_kv_cache,
)
from sglang.srt.mem_cache.cp_sharded_allocator import unwrap_cp_sharded_allocator
from sglang.srt.mem_cache.hisparse_memory_pool import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.models.welm_deferred_mirror import (
    DeferredDecodeInputKind,
    WelmDeferredPrefillSpan,
    build_welm_deferred_prefill_span,
)
from sglang.srt.models.welm_perf_opt import (
    get_welm_oe_hash_config,
    should_use_welm_oe_hash_kernel,
)
from sglang.srt.observability.metrics_collector import (
    DPCooperationInfo,
    SchedulerMetricsCollector,
)
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, get_global_server_args
from sglang.srt.utils import flatten_nested_list
from sglang.srt.utils.cuda_ipc_transport_utils import CudaIpcTensorTransportProxy

if TYPE_CHECKING:
    from typing import Any, Dict

    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats
    from sglang.srt.session.session_controller import Session
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

# Constant used as the base offset for MM (multimodal) pad values.
# This ensures pad_values don't overlap with valid text token IDs.
MM_PAD_SHIFT_VALUE = 1_000_000

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def sanity_check_mm_pad_shift_value(vocab_size: int) -> None:
    if vocab_size > MM_PAD_SHIFT_VALUE:
        raise ValueError(
            f"Model vocab_size ({vocab_size}) exceeds MM_PAD_SHIFT_VALUE ({MM_PAD_SHIFT_VALUE}). "
            f"MM pad_values may overlap with valid token IDs. "
            f"Please increase MM_PAD_SHIFT_VALUE in schedule_batch.py."
        )


def _compute_pad_value(hash: int) -> int:
    """Compute pad value from hash."""
    return MM_PAD_SHIFT_VALUE + (hash % (1 << 30))


def validate_router_replay_experts(
    routed_experts: Any,
    *,
    num_layers: int,
    num_experts_per_tok: int,
    num_logical_routed_experts: int,
    min_router_seq_len: int = 0,
    rid: Optional[str] = None,
) -> Optional[torch.Tensor]:
    if routed_experts is None:
        return None

    req_msg = f" for request {rid}" if rid is not None else ""
    try:
        tensor = torch.as_tensor(routed_experts)
    except Exception as exc:
        raise ValueError(f"Invalid routed_experts{req_msg}: {exc}") from exc

    if tensor.dim() != 3:
        raise ValueError(
            f"Invalid routed_experts{req_msg}: expected rank-3 tensor "
            "[router_seq_len, num_layers, num_experts_per_tok], "
            f"got shape {tuple(tensor.shape)}."
        )
    if tensor.shape[1] != num_layers:
        raise ValueError(
            f"Invalid routed_experts{req_msg}: num_layers mismatch, "
            f"expected {num_layers}, got {tensor.shape[1]}."
        )
    if tensor.shape[2] != num_experts_per_tok:
        raise ValueError(
            f"Invalid routed_experts{req_msg}: num_experts_per_tok mismatch, "
            f"expected {num_experts_per_tok}, got {tensor.shape[2]}."
        )
    if tensor.shape[0] < min_router_seq_len:
        raise ValueError(
            f"Invalid routed_experts{req_msg}: router replay trace is too short, "
            f"expected at least {min_router_seq_len} token positions, "
            f"got {tensor.shape[0]}."
        )

    tensor = tensor.to(device="cpu", dtype=torch.int32)
    if tensor.numel() > 0 and not getattr(
        tensor, "_sglang_router_replay_trusted", False
    ):
        if bool((tensor < -1).any()):
            min_id = int(tensor.min().item())
            raise ValueError(
                f"Invalid routed_experts{req_msg}: expert id out of range "
                f"[0, {num_logical_routed_experts}) or -1 for masked layers; "
                f"min={min_id}."
            )
        invalid_mask = tensor < 0
        partially_masked = invalid_mask.any(dim=2) & ~invalid_mask.all(dim=2)
        if bool(partially_masked.any()):
            raise ValueError(
                f"Invalid routed_experts{req_msg}: expert id out of range; "
                "-1 mask values must cover all experts for a token/layer."
            )
        valid_tensor = tensor[tensor >= 0]
        if valid_tensor.numel() > 0:
            min_id = int(valid_tensor.min().item())
            max_id = int(valid_tensor.max().item())
        else:
            min_id = max_id = -1
        if valid_tensor.numel() > 0 and max_id >= num_logical_routed_experts:
            raise ValueError(
                f"Invalid routed_experts{req_msg}: expert id out of range "
                f"[0, {num_logical_routed_experts}); min={min_id}, max={max_id}."
            )

    return tensor


def _has_router_replay(req: Any) -> bool:
    return getattr(req, "router_replay_experts", None) is not None


def _check_router_replay_batch_consistency(reqs: List[Any]) -> bool:
    replay_flags = [_has_router_replay(req) for req in reqs]
    if any(replay_flags) and not all(replay_flags):
        raise ValueError("Cannot mix router replay requests with non-replay requests.")
    return any(replay_flags)


def _router_replay_short_trace_error(
    req: Any, start: int, end: int, router_seq_len: int
) -> ValueError:
    rid = getattr(req, "rid", None)
    req_msg = f" for request {rid}" if rid is not None else ""
    return ValueError(
        f"router replay trace{req_msg} is too short: need token positions "
        f"[{start}:{end}), but trace length is {router_seq_len}."
    )


def _is_router_replay_decode_overrun(req: Any, trace_pos: int) -> bool:
    sampling_params = getattr(req, "sampling_params", None)
    max_new_tokens = getattr(sampling_params, "max_new_tokens", None)
    origin_input_ids = getattr(req, "origin_input_ids", None)
    if max_new_tokens is None or origin_input_ids is None:
        return False

    max_forward_trace_len = len(origin_input_ids) + max(max_new_tokens - 1, 0)
    return trace_pos >= max_forward_trace_len


def build_router_replay_extend_batch(
    reqs: List[Any],
    logical_prefix_lens: List[int],
    device: Union[str, torch.device],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not _check_router_replay_batch_consistency(reqs):
        return None, None

    slices = []
    for req, logical_prefix_len in zip(reqs, logical_prefix_lens):
        trace = req.router_replay_experts
        start = logical_prefix_len
        end = logical_prefix_len + req.extend_input_len
        if end > trace.shape[0]:
            raise _router_replay_short_trace_error(req, start, end, trace.shape[0])
        slices.append(trace[start:end])

    if len(slices) == 0:
        return None, None
    if sum(s.shape[0] for s in slices) == 0:
        ref = reqs[0].router_replay_experts
        topk_ids = torch.empty((0, ref.shape[1], ref.shape[2]), dtype=torch.int32)
    else:
        topk_ids = torch.cat(slices, dim=0)
    topk_ids = topk_ids.to(device=device, non_blocking=True)
    # PERF: hoist per-layer cleanup (originally done num_layers times per step in
    # `_apply_router_replay_from_forward_batch`) to per-step. Concretely:
    #   - replace -1 sentinels (input region after push_traces_to_redis.py) with 0,
    #     so `scores.gather(1, forced_ids)` in `_gather_router_replay_weights` is
    #     always safe regardless of `mask` value.
    #   - tighten mask: a row is only valid if EVERY (layer, top_k) entry is >= 0.
    # This eliminates 5 per-layer kernel launches (ge×2, all, where, zeros_like,
    # mask &) per decode step in topk.py, ~240 fewer kernels per step at 48-layer
    # 80B / ~465 fewer at 93-layer 600B.
    if topk_ids.numel() > 0:
        valid_per_row = (topk_ids >= 0).all(dim=tuple(range(1, topk_ids.ndim)))
        # Use clamp_min_ over torch.where(...zeros_like...): single in-place op.
        topk_ids = topk_ids.clamp_min_(0)
    else:
        valid_per_row = torch.ones(
            (topk_ids.shape[0],), dtype=torch.bool, device=device
        )
    return topk_ids, valid_per_row


def build_router_replay_decode_batch(
    reqs: List[Any],
    device: Union[str, torch.device],
    trace_positions: Optional[List[int]] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not _check_router_replay_batch_consistency(reqs):
        return None, None
    if trace_positions is not None and len(trace_positions) != len(reqs):
        raise ValueError(
            "router replay decode trace_positions length must match request count."
        )

    slices = []
    all_valid = True
    has_overrun = False
    overrun_indices: List[int] = []
    for i, req in enumerate(reqs):
        trace = req.router_replay_experts
        trace_pos = (
            trace_positions[i] if trace_positions is not None else req.seqlen - 1
        )
        if trace_pos >= trace.shape[0]:
            if _is_router_replay_decode_overrun(req, trace_pos):
                slices.append(torch.zeros_like(trace[:1]))
                all_valid = False
                has_overrun = True
                overrun_indices.append(i)
                continue
            raise _router_replay_short_trace_error(
                req, trace_pos, trace_pos + 1, trace.shape[0]
            )
        slices.append(trace[trace_pos : trace_pos + 1])

    if len(slices) == 0:
        return None, None
    topk_ids = torch.cat(slices, dim=0).to(device=device, non_blocking=True)
    # PERF: avoid `torch.tensor([True]*N, dtype=bool, device=cuda)` which forces
    # a sync H->D copy on the hot path (~6-14 ms per decode step on TP=8 DP=4
    # 80B with c=32). The common case is "all-valid" (every req still has trace
    # data left), so allocate the mask directly on the GPU. Only fall back to
    # an H->D copy when at least one request has overrun.
    if not has_overrun:
        mask = torch.ones((len(slices),), dtype=torch.bool, device=device)
    else:
        # Build the bool mask on CPU (cheap), then async-copy to GPU.
        cpu_mask = torch.ones((len(slices),), dtype=torch.bool)
        if overrun_indices:
            cpu_mask[overrun_indices] = False
        mask = cpu_mask.to(device=device, non_blocking=True)
    # PERF: hoist per-layer cleanup to per-step (see extend variant above for
    # rationale). For decode the per-row count equals batch size, so this loop
    # body runs once per decode step regardless of num_layers.
    if topk_ids.numel() > 0:
        valid_per_row = (topk_ids >= 0).all(dim=tuple(range(1, topk_ids.ndim)))
        mask = mask & valid_per_row
        topk_ids = topk_ids.clamp_min_(0)
    return topk_ids, mask


class BaseFinishReason:
    def __init__(self, is_error: bool = False):
        self.is_error = is_error

    def to_json(self):
        raise NotImplementedError()


class FINISH_MATCHED_TOKEN(BaseFinishReason):
    def __init__(self, matched: Union[int, List[int]]):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISH_MATCHED_STR(BaseFinishReason):
    def __init__(self, matched: str):
        super().__init__()
        self.matched = matched

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
        }


class FINISHED_MATCHED_REGEX(BaseFinishReason):
    def __init__(self, matched: str, matched_text: Optional[str] = None):
        super().__init__()
        self.matched = matched
        self.matched_text = matched_text

    def to_json(self):
        return {
            "type": "stop",  # to match OpenAI API's return value
            "matched": self.matched,
            "matched_text": self.matched_text,
        }


class FINISH_LENGTH(BaseFinishReason):
    def __init__(self, length: int):
        super().__init__()
        self.length = length

    def to_json(self):
        return {
            "type": "length",  # to match OpenAI API's return value
            "length": self.length,
        }


class FINISH_ABORT(BaseFinishReason):
    def __init__(self, message=None, status_code=None, err_type=None):
        super().__init__(is_error=True)
        self.message = message or "Aborted"
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return {
            "type": "abort",
            "message": self.message,
            "status_code": self.status_code,
            "err_type": self.err_type,
        }


class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()

    @staticmethod
    def from_str(modality_str: str):
        try:
            return Modality[modality_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid modality string: {modality_str}. Valid modalities are: {[m.name for m in Modality]}"
            )

    @staticmethod
    def all():
        return [Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]


class MultimodalInputFormat(Enum):
    NORMAL = auto()
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()


@dataclasses.dataclass
class MultimodalDataItem:
    """
    One MultimodalDataItem represents a single multimodal input (one image, one video, or one audio).
    For example, if there are 3 images and 1 audio, there will be 4 MultimodalDataItems.

    Each item has its own hash and pad_value, enabling per-image RadixAttention caching.

    We put the common fields first and the model-specific fields in model_specific_data.
    """

    modality: Modality
    hash: int = None
    pad_value: int = None
    offsets: Optional[list] = None

    format: MultimodalInputFormat = MultimodalInputFormat.NORMAL

    # the raw features returned by processor, e.g. pixel_values or audio_features
    feature: Union[torch.Tensor, np.ndarray] = None
    # the precomputed embeddings, passed as final encoder embeddings
    # One and only one of the feature and precomputed_embeddings will be empty
    precomputed_embeddings: Optional[Union[torch.Tensor, np.ndarray]] = None

    # Model-specific data stored in a dictionary
    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __getattr__(self, name: str):
        if (
            "model_specific_data" in self.__dict__
            and name in self.__dict__["model_specific_data"]
        ):
            return self.__dict__["model_specific_data"][name]
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

    def __setitem__(self, key: str, value: Any):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.model_specific_data[key] = value

    def set(self, key: str, value: Any):
        self.__setitem__(key, value)

    @staticmethod
    def is_empty_list(l):
        if l is None:
            return True
        return len([item for item in flatten_nested_list(l) if item is not None]) == 0

    def set_pad_value(self):
        """
        Set the pad value after first hashing the data
        """
        if self.pad_value is not None:
            return

        from sglang.srt.managers.mm_utils import hash_feature

        if envs.SGLANG_MM_SKIP_COMPUTE_HASH.get():
            import uuid

            self.hash = uuid.uuid4().int
            self.pad_value = _compute_pad_value(self.hash)
            return
        if self.hash is None:
            if self.feature is not None:
                hashed_feature = self.feature
            else:
                hashed_feature = self.precomputed_embeddings
            self.hash = hash_feature(hashed_feature)
        assert self.hash is not None
        self.pad_value = _compute_pad_value(self.hash)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality

    def is_audio(self):
        return self.modality == Modality.AUDIO

    def is_image(self):
        return self.modality == Modality.IMAGE

    def is_video(self):
        return self.modality == Modality.VIDEO

    def is_valid(self) -> bool:
        return self.is_image() or self.is_video() or self.is_audio()

    def validate(self):
        ...
        # TODO

    def is_precomputed_embedding(self):
        return self.format == MultimodalInputFormat.PRECOMPUTED_EMBEDDING

    @staticmethod
    def from_dict(obj: dict):
        kwargs = dict(obj)
        modality = kwargs.pop("modality")
        if isinstance(modality, str):
            modality = Modality[modality]
        ret = MultimodalDataItem(modality=modality, **kwargs)
        ret.validate()
        return ret

    def reconstruct(self):
        if not isinstance(self.feature, CudaIpcTensorTransportProxy):
            return

        reconstruct_device = torch.cuda.current_device()
        if isinstance(self.feature, CudaIpcTensorTransportProxy):
            self.feature = self.feature.reconstruct_on_target_device(reconstruct_device)
        if isinstance(self.precomputed_embeddings, CudaIpcTensorTransportProxy):
            self.precomputed_embeddings = (
                self.precomputed_embeddings.reconstruct_on_target_device(
                    reconstruct_device
                )
            )
        for extra_key in self.model_specific_data:
            if isinstance(
                self.model_specific_data[extra_key], CudaIpcTensorTransportProxy
            ):
                extra_data = self.model_specific_data[
                    extra_key
                ].reconstruct_on_target_device(reconstruct_device)
                self.model_specific_data[extra_key] = extra_data


@dataclasses.dataclass
class MultimodalProcessorOutput:
    """Raw output from multimodal processors, before pad/hash computation.

    This is the typed replacement for the dict previously returned by
    ``BaseMultimodalProcessor.process_mm_data_async``.  Unlike
    ``MultimodalInputs``, items here do NOT carry pad_value or hash yet.
    """

    mm_items: List[MultimodalDataItem]
    input_ids: Optional[List[int]] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    # for transformers-compatibility
    token_type_ids: Optional[torch.Tensor] = None

    @staticmethod
    def from_dict(d: dict) -> "MultimodalProcessorOutput":
        return MultimodalProcessorOutput(
            mm_items=d["mm_items"],
            input_ids=d.get("input_ids"),
            im_token_id=d.get("im_token_id"),
            im_start_id=d.get("im_start_id"),
            im_end_id=d.get("im_end_id"),
            slice_start_id=d.get("slice_start_id"),
            slice_end_id=d.get("slice_end_id"),
            video_token_id=d.get("video_token_id"),
            audio_token_id=d.get("audio_token_id"),
            audio_start_id=d.get("audio_start_id"),
            audio_end_id=d.get("audio_end_id"),
            mrope_positions=d.get("mrope_positions"),
            mrope_position_delta=d.get("mrope_position_delta"),
            vision_position_ids=d.get("vision_position_ids"),
            media_nums_per_sample=d.get("media_nums_per_sample"),
            visible_frame_counts=d.get("visible_frame_counts"),
        )


@dataclasses.dataclass
class MultimodalInputs:
    """The multimodal data related inputs."""

    # items of data
    mm_items: List[MultimodalDataItem]
    image_pad_len: Optional[list] = None
    num_image_tokens: Optional[int] = None

    # image
    im_token_id: Optional[int] = None
    im_start_id: Optional[int] = None
    im_end_id: Optional[int] = None
    slice_start_id: Optional[int] = None
    slice_end_id: Optional[int] = None

    # video
    video_token_id: Optional[int] = None

    # audio
    audio_token_id: Optional[int] = None
    audio_start_id: Optional[int] = None
    audio_end_id: Optional[int] = None

    # QWen2-VL related
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None
    mrope_position_delta_repeated_cache: Optional[torch.Tensor] = None

    # Moss-VL related
    vision_position_ids: Optional[torch.Tensor] = None
    media_nums_per_sample: Optional[List[int]] = None
    visible_frame_counts: Optional[torch.Tensor] = None

    def release_features(self):
        """Release feature tensors to free GPU memory."""
        for item in self.mm_items:
            item.feature = None

    @staticmethod
    def from_processor_output(obj: "MultimodalProcessorOutput"):
        mm_items = obj.mm_items
        for mm_item in mm_items:
            mm_item.reconstruct()

        ret = MultimodalInputs(
            mm_items=mm_items,
        )

        assert isinstance(ret.mm_items, list)
        ret.mm_items = [item for item in ret.mm_items if item.is_valid()]

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            # Multi-modal feature hashing optimization:
            # When SGLANG_MM_BUFFER_SIZE_MB > 0, we temporarily move feature tensors to GPU
            # for faster hash computation, while avoiding OOM issues.
            from sglang.srt.managers.mm_utils import (
                init_feature_buffer,
                is_feature_buffer_initialized,
                reset_buffer_offset,
                try_add_to_buffer,
            )

            device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
            if not is_feature_buffer_initialized():
                init_feature_buffer(device)
            reset_buffer_offset()
            for item in ret.mm_items:
                if item.feature is not None:
                    if isinstance(item.feature, torch.Tensor):
                        item.feature = try_add_to_buffer(item.feature)

        for item in ret.mm_items:
            item.set_pad_value()

        if envs.SGLANG_MM_BUFFER_SIZE_MB.get() > 0:
            for item in ret.mm_items:
                if item.feature is not None:
                    item.feature = item.feature.to("cpu", non_blocking=True)

        optional_args = [
            "mrope_positions",
            "mrope_position_delta",
            "im_token_id",
            "im_start_id",
            "im_end_id",
            "video_token_id",
            "slice_start_id",
            "slice_end_id",
            "audio_start_id",
            "audio_end_id",
            "audio_token_id",
            "vision_position_ids",
            "media_nums_per_sample",
            "visible_frame_counts",
        ]
        for arg in optional_args:
            val = getattr(obj, arg, None)
            if val is not None:
                setattr(ret, arg, val)

        return ret

    def contains_image_inputs(self) -> bool:
        return any(item.is_image() for item in self.mm_items)

    def contains_video_inputs(self) -> bool:
        return any(item.is_video() for item in self.mm_items)

    def contains_audio_inputs(self) -> bool:
        return any(item.is_audio() for item in self.mm_items)

    def contains_mm_input(self) -> bool:
        return any(True for item in self.mm_items if item.is_valid())

    def merge(self, other: MultimodalInputs):
        """
        merge image inputs when requests are being merged
        """

        # args needed to be merged
        optional_args = [
            "mm_items",
            "image_pad_len",
        ]
        for arg in optional_args:
            self_arg = getattr(self, arg, None)
            if self_arg is not None:
                setattr(self, arg, self_arg + getattr(other, arg))

        mrope_positions = self.mrope_positions
        if mrope_positions is not None:
            if other.mrope_positions is None:
                self.mrope_positions = mrope_positions
            else:
                self.mrope_positions = torch.cat(
                    [self.mrope_positions, other.mrope_positions], dim=1
                )

        mrope_position_delta = self.mrope_position_delta
        if mrope_position_delta is not None:
            if other.mrope_position_delta is None:
                self.mrope_position_delta = mrope_position_delta
            else:
                self.mrope_position_delta = torch.cat(
                    [self.mrope_position_delta, other.mrope_position_delta], dim=0
                )

        for key, val in other.__dict__.items():
            if "_id" in key:
                # set token_ids
                if getattr(self, key, None) is None:
                    setattr(self, key, getattr(other, key, None))
        # other args would be kept intact


@dataclasses.dataclass
class HashInputIdsBuffer:
    """CPU boundary tokens used by the WeLM OE hash kernel.

    ``prefixes[lag - 1][segment_idx]`` is the token before a decode/prefill
    segment starts. The current segment tokens still come from forward input_ids.
    """

    prefixes: List[List[int]]


@dataclasses.dataclass
class OverEncodingContext:
    """Context data for WeLM OE CUDA hash-kernel inputs."""

    NUM_GRAMS: ClassVar[int] = 3

    input_ids_buffer: Optional[HashInputIdsBuffer] = None
    legacy_prefixes: Optional[List[List[int]]] = None

    @classmethod
    def from_extend_hash_kernel(
        cls,
        reqs,
        logical_prefix_lens,
        history_width: int,
        legacy_history_width: Optional[int] = None,
    ) -> "OverEncodingContext":
        """Build an extend context whose OE hashes are produced on GPU.

        The normal forward ``input_ids`` tensor contains the current segment.
        ``prefixes`` only carries the few tokens before each segment boundary
        that the hash kernel cannot read from that tensor.
        """
        prefixes = cls._extend_prefixes(reqs, logical_prefix_lens, history_width)
        legacy_prefixes = None
        if legacy_history_width is not None and legacy_history_width > history_width:
            legacy_prefixes = cls._extend_prefixes(
                reqs, logical_prefix_lens, legacy_history_width
            )
        return cls(
            input_ids_buffer=HashInputIdsBuffer(prefixes),
            legacy_prefixes=legacy_prefixes,
        )

    @staticmethod
    def _extend_prefixes(reqs, logical_prefix_lens, history_width: int):
        prefixes = []
        for lag in range(1, history_width + 1):
            row = []
            for r, lpl in zip(reqs, logical_prefix_lens):
                pos = lpl - lag
                row.append(r.fill_ids[pos] if pos >= 0 else 0)
            prefixes.append(row)
        return prefixes

    @classmethod
    def from_decode_hash_kernel(
        cls,
        reqs,
        enable_overlap,
        history_width: int,
        legacy_history_width: Optional[int] = None,
    ) -> "OverEncodingContext":
        """Build a decode context whose OE hashes are produced on GPU."""
        offsets = []
        if enable_overlap:
            for r in reqs:
                dc = r._overlap_decode_count
                offsets.append(1 if len(r.output_ids) > dc else 0)
                r._overlap_decode_count = dc + 1
        else:
            offsets = [1] * len(reqs)

        prefixes = cls._decode_prefixes(reqs, offsets, history_width)
        legacy_prefixes = None
        if legacy_history_width is not None and legacy_history_width > history_width:
            legacy_prefixes = cls._decode_prefixes(reqs, offsets, legacy_history_width)
        return cls(
            input_ids_buffer=HashInputIdsBuffer(prefixes),
            legacy_prefixes=legacy_prefixes,
        )

    @classmethod
    def _decode_prefixes(cls, reqs, offsets, history_width: int):
        prefixes = []
        for lag in range(1, history_width + 1):
            row = []
            for r, offset in zip(reqs, offsets):
                row.append(cls._decode_history_token(r, lag + offset))
            prefixes.append(row)
        return prefixes

    @staticmethod
    def _decode_history_token(req, shift: int) -> int:
        total_len = len(req.origin_input_ids) + len(req.output_ids)
        pos = total_len - shift
        if pos < 0:
            return 0
        if pos < len(req.origin_input_ids):
            return req.origin_input_ids[pos]
        return req.output_ids[pos - len(req.origin_input_ids)]

    def merge_buffer(self, other: "OverEncodingContext"):
        """Merge over-encoding state from *other* into this instance."""
        self_prefixes = self.hash_prefixes
        other_prefixes = other.hash_prefixes
        if self_prefixes is None or other_prefixes is None:
            raise ValueError("WeLM OE contexts must contain CUDA hash prefixes.")
        if len(self_prefixes) != len(other_prefixes):
            raise ValueError(
                "Cannot merge WeLM OE hash prefixes with different history "
                f"widths: {len(self_prefixes)} vs {len(other_prefixes)}."
            )
        self.input_ids_buffer = HashInputIdsBuffer(
            [lhs + rhs for lhs, rhs in zip(self_prefixes, other_prefixes)]
        )
        if self.legacy_prefixes is not None and other.legacy_prefixes is not None:
            if len(self.legacy_prefixes) != len(other.legacy_prefixes):
                raise ValueError(
                    "Cannot merge WeLM OE legacy prefixes with different history "
                    f"widths: {len(self.legacy_prefixes)} vs "
                    f"{len(other.legacy_prefixes)}."
                )
            self.legacy_prefixes = [
                lhs + rhs
                for lhs, rhs in zip(self.legacy_prefixes, other.legacy_prefixes)
            ]

    def __bool__(self) -> bool:
        return self.input_ids_buffer is not None

    @property
    def hash_prefixes(self) -> Optional[List[List[int]]]:
        if isinstance(self.input_ids_buffer, HashInputIdsBuffer):
            return self.input_ids_buffer.prefixes
        return None


class WelmDeferredDecodePhase(str, Enum):
    PREFILL_PENDING = "prefill_pending"
    READY = "ready"
    INFLIGHT = "inflight"
    CONSUMED = "consumed"


@dataclasses.dataclass
class WelmDeferredDecodeState:
    phase: WelmDeferredDecodePhase
    committed_kv_len: int
    seed_position: int
    seed_token_id: int
    input_kind: DeferredDecodeInputKind

    @classmethod
    def from_prompt_tokens(
        cls, prompt_token_ids: List[int]
    ) -> "WelmDeferredDecodeState":
        span = build_welm_deferred_prefill_span(prompt_token_ids)
        return cls.from_prefill_span(span)

    @classmethod
    def from_prefill_span(
        cls, span: WelmDeferredPrefillSpan
    ) -> "WelmDeferredDecodeState":
        return cls(
            phase=WelmDeferredDecodePhase.PREFILL_PENDING,
            committed_kv_len=span.committed_kv_len,
            seed_position=span.seed_position,
            seed_token_id=span.seed_token_id,
            input_kind=DeferredDecodeInputKind.TOKEN_ID,
        )

    def validate_prompt_tokens(self, prompt_token_ids: List[int]) -> None:
        if len(prompt_token_ids) != self.committed_kv_len + 1:
            raise RuntimeError(
                "WeLM deferred decode prompt length changed after admission: "
                f"expected {self.committed_kv_len + 1}, "
                f"got {len(prompt_token_ids)}"
            )
        if self.seed_position != self.committed_kv_len:
            raise RuntimeError(
                "WeLM deferred decode seed position must equal committed KV length"
            )
        if int(prompt_token_ids[self.seed_position]) != self.seed_token_id:
            raise RuntimeError(
                "WeLM deferred decode seed token changed after admission"
            )
        if self.input_kind is not DeferredDecodeInputKind.TOKEN_ID:
            raise RuntimeError(
                "WeLM deferred decode currently requires TOKEN_ID seed input"
            )

    def validate_prefill_span(
        self,
        span: WelmDeferredPrefillSpan,
        prompt_token_ids: List[int],
    ) -> None:
        span.validate_prompt_tokens(prompt_token_ids)
        self.validate_prompt_tokens(prompt_token_ids)
        if (
            self.committed_kv_len,
            self.seed_position,
            self.seed_token_id,
        ) != (
            span.committed_kv_len,
            span.seed_position,
            span.seed_token_id,
        ):
            raise RuntimeError(
                "WeLM deferred lifecycle state does not match Prefill span"
            )

    def committed_token_ids(self, prompt_token_ids: List[int]) -> List[int]:
        self.validate_prompt_tokens(prompt_token_ids)
        return list(prompt_token_ids[: self.committed_kv_len])

    def transition_to(self, next_phase: WelmDeferredDecodePhase) -> None:
        expected_next = {
            WelmDeferredDecodePhase.PREFILL_PENDING: WelmDeferredDecodePhase.READY,
            WelmDeferredDecodePhase.READY: WelmDeferredDecodePhase.INFLIGHT,
            WelmDeferredDecodePhase.INFLIGHT: WelmDeferredDecodePhase.CONSUMED,
        }.get(self.phase)
        if next_phase is not expected_next:
            raise RuntimeError(
                "invalid WeLM deferred decode transition: "
                f"{self.phase.value} -> {next_phase.value}"
            )
        self.phase = next_phase

    def rollback_inflight_to_ready(self) -> None:
        if self.phase is not WelmDeferredDecodePhase.INFLIGHT:
            raise RuntimeError(
                "invalid WeLM deferred decode rollback: "
                f"expected inflight, got {self.phase.value}"
            )
        self.phase = WelmDeferredDecodePhase.READY

    def rollback_ready_to_prefill_pending(self) -> None:
        if self.phase is not WelmDeferredDecodePhase.READY:
            raise RuntimeError(
                "invalid WeLM deferred Prefill rollback: "
                f"expected ready, got {self.phase.value}"
            )
        self.phase = WelmDeferredDecodePhase.PREFILL_PENDING


def is_welm_deferred_prefill_pending(req: "Req") -> bool:
    span = getattr(req, "welm_deferred_prefill_span", None)
    state = getattr(req, "welm_deferred_decode_state", None)
    if state is not None and not isinstance(state, WelmDeferredDecodeState):
        raise RuntimeError("WeLM deferred Prefill batch has invalid lifecycle state")
    if (
        state is not None
        and state.phase is WelmDeferredDecodePhase.PREFILL_PENDING
        and span is None
    ):
        raise RuntimeError(
            "WeLM deferred Prefill state is missing its immutable span"
        )
    return span is not None and (
        state is None or state.phase is WelmDeferredDecodePhase.PREFILL_PENDING
    )


class Req(ReqDllmMixin):
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: List[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        dllm_config: Optional[DllmConfig] = None,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        origin_input_ids_unpadded: Optional[Tuple[int]] = None,
        lora_id: Optional[str] = None,
        input_embeds: Optional[List[List[float]]] = None,
        positional_embed_overrides: Optional[PositionalEmbeds] = None,
        token_type_ids: List[int] = None,
        session: Optional[Session] = None,
        custom_logit_processor: Optional[str] = None,
        require_reasoning: bool = False,
        return_hidden_states: bool = False,
        return_routed_experts: bool = False,
        routed_experts_start_len: int = 0,
        router_replay_experts: Optional[torch.Tensor] = None,
        forced_decode_token_ids: Optional[List[int]] = None,
        return_indexer_topk: bool = False,
        eos_token_ids: Optional[Set[int]] = None,
        bootstrap_host: Optional[str] = None,
        bootstrap_port: Optional[int] = None,
        bootstrap_room: Optional[int] = None,
        disagg_mode: Optional[DisaggregationMode] = None,
        routed_dp_rank: Optional[int] = None,
        disagg_prefill_dp_rank: Optional[int] = None,
        vocab_size: Optional[int] = None,
        priority: Optional[int] = None,
        metrics_collector: Optional[SchedulerMetricsCollector] = None,
        extra_key: Optional[str] = None,
        routing_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        http_worker_ipc: Optional[str] = None,
        time_stats: Optional[
            Union[APIServerReqTimeStats, DPControllerReqTimeStats]
        ] = None,
        return_pooled_hidden_states: bool = False,
        multi_item_delimiter_indices: Optional[List[int]] = None,
    ):
        # Input and output info
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids_unpadded = (
            origin_input_ids_unpadded
            if origin_input_ids_unpadded
            else origin_input_ids  # Before image padding
        )
        self.origin_input_ids = origin_input_ids
        # Each decode stage's output ids
        self.output_ids = []
        # Tracks how many decode steps this request has been through (overlap mode)
        self._overlap_decode_count = 0
        # fill_ids = origin_input_ids + output_ids. Updated if chunked.
        self.fill_ids = []
        self.welm_deferred_prefill_span: Optional[WelmDeferredPrefillSpan] = None
        self.welm_deferred_decode_state: Optional[WelmDeferredDecodeState] = None
        self.session = session
        self.input_embeds = input_embeds
        self.positional_embed_overrides = positional_embed_overrides
        self.multi_item_delimiter_indices = multi_item_delimiter_indices

        # For req-level memory management
        self.kv_committed_len = 0
        self.kv_allocated_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False

        # for corss-endoder model
        self.token_type_ids = token_type_ids

        # The length of KV that have been removed in swa cache.
        # SWA KV cache eviction behavior differs by cache type:
        # - Radix cache: KV in range [cache_protected_len, swa_evicted_seqlen) is freed manually in
        #   `ScheduleBatch.maybe_evict_swa`; KV in range [0, cache_protected_len) is freed during radix cache eviction.
        # - Chunk cache: KV in range [0, swa_evicted_seqlen) is freed manually in `ScheduleBatch.maybe_evict_swa`.
        self.swa_evicted_seqlen = 0

        # The index of the extend / decode batch
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # For multi-http worker
        self.http_worker_ipc = http_worker_ipc

        # Require reasoning for the request
        self.require_reasoning = require_reasoning

        # State indicating whether the reasoning phase has finished (only meaningful when require_reasoning is True)
        self._is_reasoning_over = False
        self.reasoning_tokens = 0

        # Sampling info
        if isinstance(sampling_params.custom_params, dict):
            sampling_params = copy.copy(sampling_params)
            sampling_params.custom_params = sampling_params.custom_params | {
                "__req__": self
            }
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor
        self.return_hidden_states = return_hidden_states

        # extra key for classifying the request (e.g. cache_salt)
        if lora_id is not None:
            extra_key = (
                extra_key or ""
            ) + lora_id  # lora_id is concatenated to the extra key

        self.extra_key = extra_key
        self.lora_id = lora_id
        self.routing_key = routing_key

        # Memory pool info
        self.req_pool_idx: Optional[int] = None
        self.mamba_pool_idx: Optional[torch.Tensor] = None  # shape (1)
        self.mamba_ping_pong_track_buffer: Optional[torch.Tensor] = None  # shape (2)
        self.mamba_next_track_idx: Optional[int] = None  # 0 or 1
        self.mamba_last_track_seqlen: Optional[int] = (
            None  # seq len of the last cached mamba state
        )
        # the branching point seqlen to track mamba state. If set, given by prefix match,
        # it will be the tracked seqlen in the ping pong buffer for the right prefill pass.
        self.mamba_branching_seqlen: Optional[int] = None

        # Check finish
        self.tokenizer = None
        self.finished_reason: Optional[BaseFinishReason] = None
        # finished position (in output_ids), used when checking stop conditions with speculative decoding
        self.finished_len = None
        # Whether this request has finished output
        self.finished_output = None
        # If we want to abort the request in the middle of the event loop,
        # set to_finish instead of directly setting finished_reason.
        # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
        self.to_finish: Optional[BaseFinishReason] = None
        self.stream = stream
        self.eos_token_ids = eos_token_ids
        self.vocab_size = vocab_size
        self.priority = priority

        # For incremental decoding
        # ----- | --------- read_ids -------|
        # ----- |   surr_ids  |
        # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
        # ----- ^ ----------- ^ ----------- ^
        # ----- 1 ----------- 2 ----------- 3
        # 1: surr_offset
        # 2: read_offset
        # 3: last token
        self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
        self.read_offset = None
        self.decoded_text = ""

        # For multimodal inputs
        self.multimodal_inputs: Optional[MultimodalInputs] = None

        # Prefix info
        # The indices to kv cache for the shared prefix.
        self.prefix_indices: torch.Tensor = torch.empty((0,), dtype=torch.int64)
        # Stable per-request owner phase for sharded-KV context parallelism.
        self.attn_cp_owner_rotation: Optional[int] = None
        # Immutable ownership plan for the currently scheduled prefill interval.
        self.attn_cp_prefill_split_spec: Optional[CPPrefillSplitSpec] = None
        # Number of tokens to run prefill.
        self.extend_input_len = 0
        # The relative logprob_start_len in an extend batch
        self.extend_logprob_start_len = 0
        # TODO(ispobock): rename to last_device_node
        self.last_node: Any = None
        self.last_host_node: Any = None
        self.best_match_node: Any = None
        self.host_hit_length = 0
        # Tokens loaded from storage backend (L3) during prefetch for this request
        self.storage_hit_length = 0
        # The node to lock until for swa radix tree lock ref
        self.swa_uuid_for_lock: Optional[int] = None
        # Whether the prefill-time SWA tree lock has been released early
        self.swa_prefix_lock_released: bool = False
        # The prefix length that is inserted into the tree cache
        self.cache_protected_len: int = 0
        # Physical-to-logical KV scale factor, set by init_next_round_input.
        self._scale_seq_factor: int = 1

        # Whether or not if it is chunked. It increments whenever
        # it is chunked, and decrement whenever chunked request is
        # processed.
        self.is_chunked = 0

        # For retraction
        self.is_retracted = False
        # Indicates if the req has ever been retracted.
        self.retracted_stain = False

        # Incremental streamining
        self.send_token_offset: int = 0
        self.send_decode_id_offset: int = 0
        # TODO (Byron): send_output_token_logprobs_offset and send_decode_id_offset can be different in disaggregation mode
        # because the decode server does not have the first output token logprobs
        self.send_output_token_logprobs_offset: int = 0

        # Logprobs (arguments)
        self.return_logprob = return_logprob
        # Start index to compute logprob from.
        self.logprob_start_len = 0
        self.top_logprobs_num = top_logprobs_num
        self.token_ids_logprob = token_ids_logprob
        self.temp_scaled_logprobs = False
        self.top_p_normalized_logprobs = False

        # Logprobs (return values)
        # True means the input logprob has been already sent to detokenizer.
        self.input_logprob_sent: bool = False
        self.input_token_logprobs_val: Optional[List[float]] = None
        self.input_token_logprobs_idx: Optional[List[int]] = None
        self.input_top_logprobs_val: Optional[List[float]] = None
        self.input_top_logprobs_idx: Optional[List[int]] = None
        self.input_token_ids_logprobs_val: Optional[List[float]] = None
        self.input_token_ids_logprobs_idx: Optional[List[int]] = None
        # Temporary holder to store input_token_logprobs.
        self.input_token_logprobs: Optional[List[Tuple[int]]] = None
        self.temp_input_top_logprobs_val: Optional[List[torch.Tensor]] = None
        self.temp_input_top_logprobs_idx: Optional[List[int]] = None
        self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
        self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

        if return_logprob:
            # shape: (bs, 1)
            self.output_token_logprobs_val = []
            self.output_token_logprobs_idx = []
            # shape: (bs, k)
            self.output_top_logprobs_val = []
            self.output_top_logprobs_idx = []
            # Can contain either lists or GPU tensors (delayed copy optimization for prefill-only scoring)
            self.output_token_ids_logprobs_val: List[
                Union[List[float], torch.Tensor]
            ] = []
            self.output_token_ids_logprobs_idx = []
        else:
            self.output_token_logprobs_val = self.output_token_logprobs_idx = (
                self.output_top_logprobs_val
            ) = self.output_top_logprobs_idx = self.output_token_ids_logprobs_val = (
                self.output_token_ids_logprobs_idx
            ) = None
        self.hidden_states: List[List[float]] = []
        self.hidden_states_tensor = None  # Note: use tensor instead of list to transfer hidden_states when PD + MTP
        self.output_topk_p = None
        self.output_topk_index = None

        # capture routed experts
        self.return_routed_experts = return_routed_experts
        self.routed_experts_start_len = routed_experts_start_len
        self.router_replay_experts = router_replay_experts
        self.forced_decode_token_ids = forced_decode_token_ids
        self.routed_experts: Optional[torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, topk)
        )

        self.return_indexer_topk = return_indexer_topk
        self.indexer_topk: Optional[torch.Tensor] = (
            None  # cpu tensor: shape (seqlen, num_indexer_layers, index_topk)
        )
        # Customized info
        self.customized_info: Optional[Dict[str, List[Any]]] = None

        # Embedding (return values)
        self.embedding = None

        # Constrained decoding
        self.grammar_key: Optional[Tuple[str, str]] = None
        self.grammar: Optional[Union[BaseGrammarObject, Future[BaseGrammarObject]]] = (
            None
        )
        self.grammar_wait_ct = 0

        # The number of cached tokens that were already cached in the KV cache
        self.cached_tokens = 0
        self.already_computed = 0

        # Detailed breakdown of cached tokens by source (for HiCache)
        self.cached_tokens_device = 0  # Tokens from device cache (GPU)
        self.cached_tokens_host = 0  # Tokens from host cache (CPU memory)
        self.cached_tokens_storage = 0  # Tokens from L3 storage backend
        self._cache_breakdown_computed = (
            False  # Track if breakdown was already computed
        )

        # Per-request count of verification forward passes.
        self.spec_verify_ct = 0

        # Per-request count of accepted draft tokens (excludes the bonus token).
        self.spec_num_correct_drafts = 0

        # Acceptance histogram for speculative decoding.
        # List index = number of accepted tokens in a step, List value = count of steps with that many accepted tokens.
        # Example: histogram[0] = 5 means 5 steps with 0 accepted tokens, histogram[3] = 10 means 10 steps with 3 accepted tokens.
        self.spec_correct_drafts_histogram: List[int] = []

        # The number of times this request has been retracted / preempted.
        self.retraction_count = 0
        self.retraction_mb_id = None

        # For observability
        self.metrics_collector = metrics_collector
        if time_stats is not None:
            self.time_stats = SchedulerReqTimeStats.new_from_obj(time_stats)
        else:
            self.time_stats = SchedulerReqTimeStats(disagg_mode=disagg_mode)
        self.time_stats.set_metrics_collector(metrics_collector)
        self.time_stats.set_scheduler_recv_time()
        self.has_log_time_stats: bool = False

        # For disaggregation
        self.bootstrap_host: str = bootstrap_host
        self.bootstrap_port: Optional[int] = bootstrap_port
        self.bootstrap_room: Optional[int] = bootstrap_room
        self.skip_radix_cache_insert = bootstrap_host == FAKE_BOOTSTRAP_HOST
        self.disagg_kv_sender: Optional[BaseKVSender] = None

        self.routed_dp_rank: Optional[int] = routed_dp_rank
        self.disagg_prefill_dp_rank: Optional[int] = disagg_prefill_dp_rank

        # the start index of the sent kv cache
        # We want to send it chunk by chunk for chunked prefill.
        # After every chunk forward, we do the following:
        # kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])
        # start_send_idx = len(req.fill_ids)
        self.start_send_idx: int = 0

        # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
        # This is because kv is not ready in `process_prefill_chunk`.
        # We use `tmp_end_idx` to store the end index of the kv cache to send.
        self.tmp_end_idx: int = -1
        self.metadata_buffer_index: int = -1

        # For Matryoshka embeddings
        self.dimensions = dimensions

        # Whether to return pooled hidden states (pre-head transformer output)
        self.return_pooled_hidden_states = return_pooled_hidden_states
        self.pooled_hidden_state = None

        # For diffusion LLM
        self.init_diffusion_llm(dllm_config)

        # For hisparse
        self.hisparse_staging = False

    @property
    def seqlen(self) -> int:
        """Get the current sequence length of the request."""
        return len(self.origin_input_ids) + len(self.output_ids)

    @property
    def is_prefill_only(self) -> bool:
        """Check if this request is prefill-only (no token generation needed)."""
        # NOTE: when spec is enabled, prefill_only optimizations are disabled

        spec_alg = get_global_server_args().speculative_algorithm
        return self.sampling_params.max_new_tokens == 0 and spec_alg is None

    @property
    def output_ids_through_stop(self) -> List[int]:
        """Get the output ids through the stop condition. Stop position is included."""
        if self.finished_len is not None:
            return self.output_ids[: self.finished_len]
        return self.output_ids

    def _cache_commit_len(self) -> int:
        # Report only the prompt prefix so thinking + answer fall into the
        # overallocated range and are reclaimed by release_kv_cache. #22373.
        if get_global_server_args().strip_thinking_cache and self.reasoning_tokens > 0:
            return min(self.kv_committed_len, len(self.origin_input_ids))
        return self.kv_committed_len

    def pop_committed_kv_cache(self) -> int:
        """Return the length of committed KV cache and mark them as freed."""
        assert (
            not self.kv_committed_freed
        ), f"Committed KV cache already freed ({self.kv_committed_len=})"
        self.kv_committed_freed = True
        return self._cache_commit_len()

    def pop_overallocated_kv_cache(self) -> Tuple[int, int]:
        """Return the range of over-allocated KV cache and mark them as freed."""

        # NOTE: This function is called when there is over-allocation of KV cache.
        # Over-allocation: we allocate more KV cache than the committed length.
        # e.g., speculative decoding may allocate more KV cache than actually used.
        assert (
            not self.kv_overallocated_freed
        ), f"Overallocated KV cache already freed, {self.kv_committed_len=}, {self.kv_allocated_len=}"
        self.kv_overallocated_freed = True
        return self._cache_commit_len(), self.kv_allocated_len

    def update_spec_correct_drafts_histogram(self, num_correct_drafts: int):
        """Update the speculative decoding acceptance histogram.

        Args:
            num_correct_drafts: Number of correct draft tokens (no bonus) in this step.
        """
        if len(self.spec_correct_drafts_histogram) <= num_correct_drafts:
            self.spec_correct_drafts_histogram.extend(
                [0] * (num_correct_drafts - len(self.spec_correct_drafts_histogram) + 1)
            )
        self.spec_correct_drafts_histogram[num_correct_drafts] += 1

    def extend_image_inputs(self, image_inputs):
        if self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)

    def finished(self) -> bool:
        # Whether request reached finished condition
        return self.finished_reason is not None

    def init_next_round_input(
        self,
        tree_cache: Optional[BasePrefixCache] = None,
        cow_mamba: Optional[bool] = None,
    ):
        self.attn_cp_prefill_split_spec = None
        if self.is_dllm():
            self._init_fill_ids_for_dllm()
            self.determine_dllm_phase()
        else:
            self.fill_ids = self.prefill_kv_token_ids()

        input_len = len(self.fill_ids)

        # Streaming sessions reuse committed KV from the session slot, so
        # custom logprob_start_len is not supported — override to -1.
        if (
            self.session is not None
            and self.session.streaming
            and self.return_logprob
            and self.logprob_start_len >= 0
        ):
            logger.warning(
                "logprob_start_len=%d is not supported for streaming sessions "
                "and will be ignored (rid=%s). Only new-token logprobs are returned.",
                self.logprob_start_len,
                self.rid,
            )
            self.logprob_start_len = -1

        max_prefix_len = self._compute_max_prefix_len(input_len)
        token_ids = self.fill_ids[:max_prefix_len]

        # Disable prefix caching when embed overrides are present: same token IDs
        # with different override vectors must not share cached KV values.
        if self.positional_embed_overrides is not None:
            max_prefix_len = 0
            token_ids = []

        if tree_cache is not None:
            if cow_mamba is None:
                cow_mamba = tree_cache.supports_mamba()
            match_result = tree_cache.match_prefix(
                MatchPrefixParams(
                    key=RadixKey(token_ids=token_ids, extra_key=self.extra_key),
                    req=self,
                    cow_mamba=cow_mamba,
                )
            )
            if envs.SGLANG_RADIX_FORCE_MISS.get():
                match_result = zero_match_result(tree_cache, match_result)
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.best_match_node,
                self.host_hit_length,
                self.mamba_branching_seqlen,
            ) = (
                match_result.device_indices,
                match_result.last_device_node,
                match_result.last_host_node,
                match_result.best_match_node,
                match_result.host_hit_length,
                match_result.mamba_branching_seqlen,
            )
            if match_result.cache_protected_len is not None:
                self.cache_protected_len = match_result.cache_protected_len
            else:
                self.cache_protected_len = len(self.prefix_indices)

            if self.is_dllm():
                self._update_block_offset_for_dllm()

            self._scale_seq_factor = getattr(tree_cache, "scale_seq_factor", 1)

        if (
            self.is_retracted
            and self.multimodal_inputs is not None
            and self.multimodal_inputs.mrope_positions is not None
        ):
            from sglang.srt.managers.mm_utils import (
                extend_mrope_positions_for_retracted_request,
            )

            self.multimodal_inputs.mrope_positions = (
                extend_mrope_positions_for_retracted_request(
                    self.multimodal_inputs.mrope_positions, len(self.output_ids)
                )
            )

        self.set_extend_input_len(
            len(self.fill_ids) - len(self.prefix_indices) // self._scale_seq_factor
        )

    def _compute_max_prefix_len(self, input_len: int) -> int:
        deferred_state = getattr(self, "welm_deferred_decode_state", None)
        if (
            getattr(self, "welm_deferred_prefill_span", None) is not None
            and not (
                isinstance(deferred_state, WelmDeferredDecodeState)
                and deferred_state.phase is WelmDeferredDecodePhase.CONSUMED
            )
        ):
            return input_len
        # The matched length is at most 1 less than the input length to enable logprob computation.
        max_prefix_len = input_len - 1
        if (
            self.return_logprob
            and self.logprob_start_len >= 0
            and not self.retract_replay_skips_logprob()
        ):
            # A retract replay with stored logprobs recomputes none of them, so
            # its prefix match may also cover the generated tokens.
            max_prefix_len = min(max_prefix_len, self.logprob_start_len)
        return max(max_prefix_len, 0)

    def prefill_kv_token_ids(self) -> List[int]:
        span = getattr(self, "welm_deferred_prefill_span", None)
        state = getattr(self, "welm_deferred_decode_state", None)
        if span is None or (
            isinstance(state, WelmDeferredDecodeState)
            and state.phase is WelmDeferredDecodePhase.CONSUMED
        ):
            return self.origin_input_ids + self.output_ids
        return span.committed_token_ids(self.origin_input_ids)

    def prefill_kv_len(self) -> int:
        span = getattr(self, "welm_deferred_prefill_span", None)
        state = getattr(self, "welm_deferred_decode_state", None)
        if span is None or (
            isinstance(state, WelmDeferredDecodeState)
            and state.phase is WelmDeferredDecodePhase.CONSUMED
        ):
            return len(self.origin_input_ids) + len(self.output_ids)
        return span.committed_kv_len

    def update_cached_token_stats(self, prefix_len: int, computed_len: int) -> None:
        if self.retracted_stain:
            return

        self.cached_tokens += prefix_len - self.already_computed
        if not self._cache_breakdown_computed:
            host_total = self.host_hit_length
            storage_portion = min(host_total, self.storage_hit_length)
            self.cached_tokens_device = max(0, len(self.prefix_indices) - host_total)
            self.cached_tokens_host = host_total - storage_portion
            self.cached_tokens_storage = storage_portion
            self._cache_breakdown_computed = True

        self.already_computed = computed_len

    def requires_prompt_logprobs(self) -> bool:
        return (
            self.return_logprob
            and 0 <= self.logprob_start_len < len(self.origin_input_ids)
        )

    def requires_logprob_payload(self) -> bool:
        deferred_state = getattr(self, "welm_deferred_decode_state", None)
        if isinstance(deferred_state, WelmDeferredDecodeState):
            return (
                self.return_logprob
                and deferred_state.phase is not WelmDeferredDecodePhase.PREFILL_PENDING
            )
        return self.return_logprob and (
            getattr(self, "welm_deferred_prefill_span", None) is None
            or Req.requires_prompt_logprobs(self)
        )

    # Based on https://github.com/vllm-project/vllm/blob/7a64d24aad69e4d2548aa0bf528d9fe63428ab01/vllm/transformers_utils/detokenizer.py#L194-L313
    def init_incremental_detokenize(self):
        first_iter = self.surr_offset is None or self.read_offset is None

        output_ids = self.output_ids_through_stop

        if first_iter:
            self.read_offset = len(self.origin_input_ids_unpadded)
            self.surr_offset = max(
                self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0
            )
            self.surr_and_decode_ids = (
                self.origin_input_ids_unpadded[self.surr_offset :] + output_ids
            )
            self.cur_decode_ids_len = len(output_ids)
        else:
            self.surr_and_decode_ids.extend(output_ids[self.cur_decode_ids_len :])
            self.cur_decode_ids_len = len(output_ids)

        return self.surr_and_decode_ids, self.read_offset - self.surr_offset

    def tail_str(self) -> str:
        # Check stop strings and stop regex patterns together
        if (
            len(self.sampling_params.stop_strs) == 0
            and len(self.sampling_params.stop_regex_strs) == 0
        ):
            return ""

        max_len_tail_str = max(
            self.sampling_params.stop_str_max_len + 1,
            self.sampling_params.stop_regex_max_len + 1,
        )

        tail_len = min(max_len_tail_str, len(self.output_ids))
        return self.tokenizer.decode(self.output_ids[-tail_len:])

    def check_match_stop_str_prefix(self) -> bool:
        """
        Check if the suffix of tail_str overlaps with any stop_str prefix
        """
        if not self.sampling_params.stop_strs:
            return False

        tail_str = self.tail_str()

        # Early return if tail_str is empty
        if not tail_str:
            return False

        for stop_str in self.sampling_params.stop_strs:
            if not stop_str:
                continue
            # Check if stop_str is contained in tail_str (fastest check first)
            if stop_str in tail_str:
                return True

            # Check if tail_str suffix matches stop_str prefix
            # Only check if stop_str is not empty, it's for stream output
            min_len = min(len(tail_str), len(stop_str))
            for i in range(1, min_len + 1):
                if tail_str[-i:] == stop_str[:i]:
                    return True

        return False

    def _check_token_based_finish(self, new_accepted_tokens: List[int]) -> bool:
        if self.sampling_params.ignore_eos:
            return False

        # Check stop token ids
        matched_eos = False

        for i, token_id in enumerate(new_accepted_tokens):
            if self.sampling_params.stop_token_ids:
                matched_eos |= token_id in self.sampling_params.stop_token_ids
            if self.eos_token_ids:
                matched_eos |= token_id in self.eos_token_ids
            if self.tokenizer is not None:
                matched_eos |= token_id == self.tokenizer.eos_token_id
                if self.tokenizer.additional_stop_token_ids:
                    matched_eos |= token_id in self.tokenizer.additional_stop_token_ids
            if matched_eos:
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=token_id)
                matched_pos = len(self.output_ids) - len(new_accepted_tokens) + i
                self.finished_len = matched_pos + 1
                return True

        return False

    def _check_str_based_finish(self):
        if (
            len(self.sampling_params.stop_strs) > 0
            or len(self.sampling_params.stop_regex_strs) > 0
        ):
            tail_str = self.tail_str()

            # Check stop strings
            if len(self.sampling_params.stop_strs) > 0:
                for stop_str in self.sampling_params.stop_strs:
                    if stop_str in tail_str or stop_str in self.decoded_text:
                        self.finished_reason = FINISH_MATCHED_STR(matched=stop_str)
                        return True

            # Check stop regex
            if len(self.sampling_params.stop_regex_strs) > 0:
                for stop_regex_str in self.sampling_params.stop_regex_strs:
                    match = re.search(stop_regex_str, tail_str)
                    if match:
                        self.finished_reason = FINISHED_MATCHED_REGEX(
                            matched=stop_regex_str,
                            matched_text=match.group(0),
                        )
                        return True

        return False

    def _check_vocab_boundary_finish(self, new_accepted_tokens: List[int] = None):
        for i, token_id in enumerate(new_accepted_tokens):
            if token_id > self.vocab_size or token_id < 0:
                offset = len(self.output_ids) - len(new_accepted_tokens) + i
                if self.sampling_params.stop_token_ids:
                    self.output_ids[offset] = next(
                        iter(self.sampling_params.stop_token_ids)
                    )
                if self.eos_token_ids:
                    self.output_ids[offset] = next(iter(self.eos_token_ids))
                self.finished_reason = FINISH_MATCHED_STR(matched="NaN happened")
                self.finished_len = offset + 1
                return True

        return False

    def check_finished(self, new_accepted_len: int = 1):
        deferred_state = self.welm_deferred_decode_state
        if (
            isinstance(deferred_state, WelmDeferredDecodeState)
            and deferred_state.phase is WelmDeferredDecodePhase.INFLIGHT
        ):
            if new_accepted_len != 1 or len(self.output_ids) != 1:
                raise RuntimeError(
                    "WeLM deferred seed must produce exactly one first output "
                    "before becoming ordinary decode"
                )
            deferred_state.transition_to(WelmDeferredDecodePhase.CONSUMED)

        if self.finished():
            return

        if self.to_finish:
            self.finished_reason = self.to_finish
            self.to_finish = None
            return

        if len(self.output_ids) >= self.sampling_params.max_new_tokens:
            self.finished_reason = FINISH_LENGTH(
                length=self.sampling_params.max_new_tokens
            )
            self.finished_len = self.sampling_params.max_new_tokens
            return

        if self.grammar is not None:
            if self.grammar.is_terminated():
                self.finished_reason = FINISH_MATCHED_TOKEN(matched=self.output_ids[-1])
                return

        new_accepted_tokens = self.output_ids[-new_accepted_len:]

        if self._check_token_based_finish(new_accepted_tokens):
            return

        if self._check_vocab_boundary_finish(new_accepted_tokens):
            return

        if self._check_str_based_finish():
            return

    def prepare_for_retract(self) -> bool:
        """Restore deferred seed state before releasing a decode row.

        Returns whether release_kv_cache must reclaim one uncommitted KV tail.
        """
        state = self.welm_deferred_decode_state
        if state is None:
            return False
        if not isinstance(state, WelmDeferredDecodeState):
            raise RuntimeError(
                "decode retraction received invalid WeLM deferred lifecycle state"
            )
        if state.phase is WelmDeferredDecodePhase.CONSUMED:
            return False
        if state.phase is not WelmDeferredDecodePhase.INFLIGHT:
            raise RuntimeError(
                "WeLM deferred decode retraction requires INFLIGHT or CONSUMED "
                f"state, got {state.phase.value}"
            )

        state.validate_prompt_tokens(self.origin_input_ids)
        if self.output_ids:
            raise RuntimeError(
                "WeLM deferred seed cannot be retracted after producing output"
            )
        expected_allocated_len = state.committed_kv_len + 1
        if (
            self.kv_committed_len != expected_allocated_len
            or self.kv_allocated_len != expected_allocated_len
        ):
            raise RuntimeError(
                "WeLM deferred seed KV accounting changed before retraction: "
                f"committed={self.kv_committed_len}, "
                f"allocated={self.kv_allocated_len}, "
                f"expected={expected_allocated_len}"
            )

        state.rollback_inflight_to_ready()
        self.kv_committed_len = state.committed_kv_len
        return True

    def reset_for_retract(self):
        # Increment retraction count before resetting other state. We should not reset this
        # since we are tracking the total number of retractions for each request.
        self.retraction_count += 1

        self.prefix_indices = torch.empty((0,), dtype=torch.int64)
        self.attn_cp_prefill_split_spec = None
        self.routed_experts = None
        self.indexer_topk = None
        self.last_node = None
        self.cache_protected_len = 0
        self.swa_uuid_for_lock = None
        self.swa_prefix_lock_released = False
        self.extend_input_len = 0
        self.is_retracted = True
        self.retracted_stain = True
        self.input_token_logprobs = None
        self.temp_input_top_logprobs_val = None
        self.temp_input_top_logprobs_idx = None
        self.extend_logprob_start_len = 0
        self.is_chunked = 0
        self.mamba_pool_idx = None
        self.mamba_ping_pong_track_buffer = None
        self.mamba_next_track_idx = None
        self.mamba_last_track_seqlen = None
        self.mamba_branching_seqlen = None
        self.already_computed = 0
        self.kv_allocated_len = 0
        self.kv_committed_len = 0
        self.kv_committed_freed = False
        self.kv_overallocated_freed = False
        self.swa_evicted_seqlen = 0
        self.extend_batch_idx = 0
        self.decode_batch_idx = 0

        # When using input_embeds, we cannot easily mix the original input embeddings
        # with the newly generated output token IDs during re-prefill of retracted request.
        # output_ids will have no use, but will lead to wrong size cache indexes.
        # Therefore, we discard the generated output_ids and restart prefill and generation
        # to ensure shape consistency in KV cache.
        if self.input_embeds is not None:
            self.output_ids = []

    def retract_replay_skips_logprob(self) -> bool:
        # A retraction victim was in the running decode batch, so its input
        # logprobs were finalized and every generated token's logprob is already
        # stored on the request; add_input_logprob_return_values drops any
        # recomputed replay values. Mirroring that discard condition here makes
        # skipping the recomputation exactly behavior-preserving.
        return self.retracted_stain and (
            not self.return_logprob or self.input_token_logprobs_val is not None
        )

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        # Copies over both the kv cache and mamba state if available
        self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(
            token_indices, mamba_indices=self.mamba_pool_idx
        )

    def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[
            self.req_pool_idx, : self.seqlen - 1
        ]
        # Loads both the kv cache and mamba state if exists
        token_to_kv_pool_allocator.load_cpu_copy(
            self.kv_cache_cpu, token_indices, mamba_indices=self.mamba_pool_idx
        )
        del self.kv_cache_cpu

    def log_time_stats(self):
        # If overlap schedule, we schedule one decode batch ahead so this gets called twice.
        if self.has_log_time_stats:
            return

        bootstrap_info = (
            f", bootstrap_room={self.bootstrap_room}"
            if self.bootstrap_room is not None
            else ""
        )
        prefix = (
            f"ReqTimeStats("
            f"rid={self.rid}{bootstrap_info}, "
            f"input_len={len(self.origin_input_ids)}, "
            f"cached_input_len={self.cached_tokens}, "
            f"output_len={len(self.output_ids)}, "
            f"type={self.time_stats.disagg_mode_str()})"
        )
        logger.info(f"{prefix}: {self.time_stats.convert_to_duration()}")
        self.has_log_time_stats = True

    def set_extend_input_len(self, extend_input_len: int):
        # Setting extend_input_len and computing the relative logprob_start_len in an extend batch
        #
        # Key variables:
        # - logprob_start_len: Absolute position in full sequence where logprob computation begins
        # - extend_logprob_start_len: Relative position within current extend batch where logprob computation begins
        # - extend_input_len: Number of tokens that need to be processed in this extend batch
        self.attn_cp_prefill_split_spec = None
        self.extend_input_len = extend_input_len
        scale = getattr(self, "_scale_seq_factor", 1) or 1
        prefix_len = len(self.prefix_indices) // scale
        if self.logprob_start_len == -1:
            logprob_start_len = len(self.fill_ids) - 1
        else:
            # logprob_start_len should be at least the length of the prefix indices
            logprob_start_len = max(self.logprob_start_len, prefix_len)
        self.extend_logprob_start_len = min(
            logprob_start_len - prefix_len,
            self.extend_input_len,
        )
        if self.retract_replay_skips_logprob():
            # Skip the replayed region so a retracted request's re-prefill does
            # not materialize [replay_len, vocab_size] logits for values the
            # scheduler would drop anyway; _get_pruned_states still keeps the
            # sampling position.
            self.extend_logprob_start_len = self.extend_input_len

    def set_finish_with_abort(self, error_msg: str):
        if get_tensor_model_parallel_rank() == 0:
            logger.error(f"{error_msg}, {self.rid=}")
        self.multimodal_inputs = None
        self.grammar = None
        self.origin_input_ids = [0]  # set it to one token to skip the long prefill
        self.return_logprob = False
        self.logprob_start_len = -1
        self.to_finish = FINISH_ABORT(
            error_msg, HTTPStatus.BAD_REQUEST, "BadRequestError"
        )

    def update_reasoning_tokens(self, token_id, think_end_id):
        if self._is_reasoning_over:
            return

        if not isinstance(token_id, list):
            token_id = [token_id]

        try:
            end_pos = token_id.index(think_end_id)
            self.reasoning_tokens += end_pos + 1
            self._is_reasoning_over = True
        except ValueError:
            self.reasoning_tokens += len(token_id)

    def __repr__(self):
        return (
            f"Req(rid={self.rid}, "
            f"input_ids={self.origin_input_ids}, output_ids={self.output_ids}, "
            f"{self.grammar=}, "
            f"{self.sampling_params=})"
        )


@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """Store all information of a batch on the scheduler."""

    # Request, memory pool, and cache
    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None
    is_hybrid_swa: bool = False

    # Batch configs
    model_config: ModelConfig = None
    forward_mode: ForwardMode = None
    enable_overlap: bool = False
    # Tell whether the current running batch is full so that we can skip
    # the check of whether to prefill new requests.
    # This is an optimization to reduce the overhead of the prefill check.
    batch_is_full: bool = False

    # For chunked prefill in PP
    chunked_req: Optional[Req] = None

    # Sampling info
    sampling_info: SamplingBatchInfo = None

    # Batched arguments to model runner
    input_ids: torch.Tensor = None  # shape: [b], int64
    input_embeds: torch.Tensor = None  # shape: [b, hidden_size], float32
    # Token replacement embeddings and absolute positions (optional).
    replace_embeds: Optional[torch.Tensor] = None
    replace_positions: Optional[torch.Tensor] = None
    ne_token_table: torch.Tensor = None
    token_type_ids: torch.Tensor = None  # shape: [b], int64
    req_pool_indices: torch.Tensor = None  # shape: [b], int64
    seq_lens: torch.Tensor = None  # shape: [b], int64
    seq_lens_cpu: torch.Tensor = None  # shape: [b], int64
    # The output locations of the KV cache
    out_cache_loc: torch.Tensor = None  # shape: [b], int64
    output_ids: torch.Tensor = None  # shape: [b], int64

    # For hybrid GDN prefix cache
    mamba_track_indices: torch.Tensor = None  # shape: [b], int64
    mamba_track_mask: torch.Tensor = None  # shape: [b], bool
    mamba_track_seqlens: torch.Tensor = None  # shape: [b], int64

    # For multimodal inputs
    multimodal_inputs: Optional[List] = None

    # The sum of all sequence lengths
    seq_lens_sum: int = None
    # The original sequence lengths, Qwen-1M related
    orig_seq_lens: torch.Tensor = None  # shape: [b], int32

    # For DP attention
    inner_idle_batch: Optional[ScheduleBatch] = None
    global_num_tokens: Optional[List[int]] = None
    global_num_tokens_for_logprob: Optional[List[int]] = None
    global_num_reqs: Optional[List[int]] = None
    global_forward_modes: Optional[List[int]] = None
    welm_kv_mirror_contract_flags: Optional[List[bool]] = None
    welm_deferred_prefill_flags: Optional[List[bool]] = None
    welm_mtp_global_prefill_num_tokens: Optional[List[int]] = None
    global_has_non_greedy_sampling: bool = False
    global_needs_top_p_sampling: bool = False
    is_extend_in_batch: bool = False
    has_cache_hit_extend_in_batch: bool = False
    all_extend_in_batch: bool = False
    can_run_dp_cuda_graph: bool = False
    tbo_split_seq_index: Optional[int] = None
    global_forward_mode: Optional[ForwardMode] = None

    # For processing logprobs
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # For logits and logprob post processing
    temp_scaled_logprobs: bool = False
    top_p_normalized_logprobs: bool = False

    # For extend and mixed chunekd prefill
    prefix_lens: List[int] = None
    extend_lens: List[int] = None
    extend_num_tokens: Optional[int] = None
    decoding_reqs: List[Req] = None
    extend_logprob_start_lens: List[int] = None
    attn_cp_prefill_split_specs: Optional[
        Tuple[Optional[CPPrefillSplitSpec], ...]
    ] = None
    # It comes empty list if logprob is not required.
    extend_input_logprob_token_ids: Optional[torch.Tensor] = None

    # For encoder-decoder architectures
    encoder_cached: Optional[List[bool]] = None
    encoder_lens: Optional[torch.Tensor] = None
    encoder_lens_cpu: Optional[List[int]] = None
    encoder_out_cache_loc: Optional[torch.Tensor] = None

    # For matryoshka embeddings
    dimensions: Optional[list[int]] = None

    # Whether to return pooled hidden states (pre-head transformer output)
    return_pooled_hidden_states: bool = False

    # For split prefill
    split_index: int = 0
    split_prefill_finished: bool = False
    split_forward_count: int = 1
    split_forward_batch: ForwardBatch = None
    seq_lens_cpu_cache: torch.Tensor = None

    # Forward-pass metrics
    fpm_start_time: float = 0.0

    # Stream
    has_stream: bool = False

    # Has grammar
    has_grammar: bool = False

    # Device
    device: str = "cuda"

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None
    # Local speculative execution state for this batch. In spec-v2 the
    # scheduler installs GenerationBatchResult.draft_continuation_state.draft_input
    # here for the next step; it is not a public generation result or PP output.
    spec_info: Optional[SpecInput] = None

    # Whether to return hidden states
    return_hidden_states: bool = False

    # Whether to return captured experts
    return_routed_experts: bool = False

    return_indexer_topk: bool = False

    # MoE router replay tensors aligned to current forward rows.
    router_replay_topk_ids: Optional[torch.Tensor] = None
    router_replay_mask: Optional[torch.Tensor] = None

    # Whether this batch is prefill-only (no token generation needed)
    is_prefill_only: bool = False

    # Multi-item scoring delimiter indices (set during prepare_for_extend)
    multi_item_delimiter_indices: Optional[List[torch.Tensor]] = None

    # hicache pointer for synchronizing data loading from CPU to GPU
    hicache_consumer_index: int = -1

    oe_context: Optional[OverEncodingContext] = None
    scale_seq_factor: int = 1

    # WeLM deferred Prefill metadata. The request-aligned final mask remains
    # scheduler-local; only the uniform marker crosses into model execution.
    welm_deferred_prefill: bool = False
    welm_deferred_prefill_final_mask: Optional[Tuple[bool, ...]] = None

    # Diffusion LLM
    dllm_config: Optional[DllmConfig] = None

    # Metrics
    dp_cooperation_info: Optional[DPCooperationInfo] = None
    prefill_stats: Optional[PrefillStats] = None
    forward_iter: Optional[int] = None

    # HiSparse
    hisparse_coordinator: Optional[HiSparseCoordinator] = None

    @classmethod
    def init_new(
        cls,
        reqs: List[Req],
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tree_cache: BasePrefixCache,
        model_config: ModelConfig,
        enable_overlap: bool,
        spec_algorithm: SpeculativeAlgorithm,
        chunked_req: Optional[Req] = None,
        dllm_config: Optional[DllmConfig] = None,
    ):
        return_logprob = any(Req.requires_logprob_payload(req) for req in reqs)

        is_hybrid_swa = False
        unwrapped_allocator = unwrap_cp_sharded_allocator(token_to_kv_pool_allocator)
        if isinstance(
            unwrapped_allocator,
            (SWATokenToKVPoolAllocator, DeepSeekV4HiSparseTokenToKVPoolAllocator),
        ):
            is_hybrid_swa = True

        attn_cp_prefill_split_specs = tuple(
            req.attn_cp_prefill_split_spec for req in reqs
        )
        if not any(spec is not None for spec in attn_cp_prefill_split_specs):
            attn_cp_prefill_split_specs = None

        batch = cls(
            reqs=reqs,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            is_hybrid_swa=is_hybrid_swa,
            model_config=model_config,
            enable_overlap=enable_overlap,
            return_logprob=return_logprob,
            has_stream=any(req.stream for req in reqs),
            has_grammar=any(req.grammar for req in reqs),
            device=req_to_token_pool.device,
            spec_algorithm=spec_algorithm,
            return_hidden_states=any(req.return_hidden_states for req in reqs),
            return_routed_experts=any(req.return_routed_experts for req in reqs),
            return_indexer_topk=any(req.return_indexer_topk for req in reqs),
            is_prefill_only=all(req.is_prefill_only for req in reqs),
            chunked_req=chunked_req,
            dllm_config=dllm_config,
            attn_cp_prefill_split_specs=attn_cp_prefill_split_specs,
        )
        batch._freeze_welm_deferred_prefill_metadata()
        return batch

    def _freeze_welm_deferred_prefill_metadata(self) -> None:
        deferred_rows = [is_welm_deferred_prefill_pending(req) for req in self.reqs]

        if any(deferred_rows) and not all(deferred_rows):
            raise RuntimeError(
                "mixed deferred Prefill and ordinary rows are not supported"
            )
        self.welm_deferred_prefill = bool(deferred_rows and all(deferred_rows))
        if not self.welm_deferred_prefill:
            self.welm_deferred_prefill_final_mask = None
            return

        final_mask = []
        for req in self.reqs:
            span = req.welm_deferred_prefill_span
            span.validate_prompt_tokens(req.origin_input_ids)
            scheduled_len = len(req.fill_ids)
            if scheduled_len > span.committed_kv_len:
                raise RuntimeError(
                    "WeLM deferred Prefill chunk exceeds its committed span: "
                    f"rid={req.rid}, scheduled_len={scheduled_len}, "
                    f"committed_kv_len={span.committed_kv_len}"
                )
            final_mask.append(scheduled_len == span.committed_kv_len)
        self.welm_deferred_prefill_final_mask = tuple(final_mask)

    def _clear_welm_deferred_prefill_metadata(self) -> None:
        self.welm_deferred_prefill = False
        self.welm_deferred_prefill_final_mask = None

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def is_dllm(self):
        return self.dllm_config is not None

    def _get_scale_seq_factor(self) -> int:
        if not self.reqs:
            return 1

        scale = max(getattr(req, "_scale_seq_factor", 1) or 1 for req in self.reqs)
        for req in self.reqs:
            req_scale = getattr(req, "_scale_seq_factor", 1) or 1
            if req_scale != scale:
                raise ValueError(
                    f"Cannot batch requests with different scale_seq_factor: "
                    f"batch_scale={scale}, req_scale={req_scale}, rid={req.rid}"
                )
        return scale

    def has_router_replay(self) -> bool:
        return any(_has_router_replay(req) for req in self.reqs)

    def prepare_encoder_info_extend(self, input_ids: List[int], seq_lens: List[int]):
        _pin = is_pin_memory_available(self.device)
        self.encoder_lens_cpu = []
        self.encoder_cached = []

        for req in self.reqs:
            im = req.multimodal_inputs
            if im is None or im.num_image_tokens is None:
                # No image input
                self.encoder_lens_cpu.append(0)
                self.encoder_cached.append(True)
            else:
                self.encoder_lens_cpu.append(im.num_image_tokens)
                self.encoder_cached.append(
                    self.forward_mode.is_decode()
                    or len(req.prefix_indices) >= im.num_image_tokens
                )

        self.encoder_lens = torch.tensor(
            self.encoder_lens_cpu, dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        # Strip encoder infos
        pt = 0
        decoder_out_cache_loc = []
        encoder_out_cache_loc = []
        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            seq_lens[i] -= encoder_len

            if len(req.prefix_indices) < encoder_len:
                # NOTE: the encoder part should be considered as a whole
                assert len(req.prefix_indices) == 0
                input_ids[i] = input_ids[i][encoder_len:]
                encoder_out_cache_loc.append(self.out_cache_loc[pt : pt + encoder_len])
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt + encoder_len : pt + req.extend_input_len]
                )
                self.extend_lens[i] -= encoder_len
                self.extend_num_tokens -= encoder_len
            else:
                decoder_out_cache_loc.append(
                    self.out_cache_loc[pt : pt + req.extend_input_len]
                )
                self.prefix_lens[i] -= encoder_len

            pt += req.extend_input_len

        # Reassign
        self.input_ids = torch.tensor(
            sum(input_ids, []), dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=_pin).to(
            self.device, non_blocking=True
        )
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)

        if not decoder_out_cache_loc:
            self.out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.out_cache_loc = torch.cat(decoder_out_cache_loc)

        if not encoder_out_cache_loc:
            self.encoder_out_cache_loc = torch.zeros(0, dtype=torch.int64).to(
                self.device, non_blocking=True
            )
        else:
            self.encoder_out_cache_loc = torch.cat(encoder_out_cache_loc)

        assert (
            len(self.out_cache_loc) == self.extend_num_tokens
        ), f"Expected {len(self.out_cache_loc)}, got {self.extend_num_tokens}"

        if self.extend_input_logprob_token_ids is not None:
            new_token_ids_parts = []
            offset = 0
            for i, req in enumerate(self.reqs):
                encoder_len = self.encoder_lens_cpu[i]
                old_start_len = self.extend_logprob_start_lens[i]
                old_contribution = req.extend_input_len - old_start_len

                if len(req.prefix_indices) < encoder_len:
                    tokens_to_strip = max(0, encoder_len - old_start_len)
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset + tokens_to_strip : offset + old_contribution
                        ]
                    )
                    self.extend_logprob_start_lens[i] = max(
                        0, old_start_len - encoder_len
                    )
                else:
                    new_token_ids_parts.append(
                        self.extend_input_logprob_token_ids[
                            offset : offset + old_contribution
                        ]
                    )

                offset += old_contribution

            if new_token_ids_parts:
                self.extend_input_logprob_token_ids = torch.cat(new_token_ids_parts)
            else:
                self.extend_input_logprob_token_ids = None

        for i, req in enumerate(self.reqs):
            encoder_len = self.encoder_lens_cpu[i]
            if encoder_len == 0:
                continue
            if len(req.prefix_indices) < encoder_len:
                req.extend_input_len -= encoder_len
                req.extend_logprob_start_len = max(
                    0, req.extend_logprob_start_len - encoder_len
                )
            req.logprob_start_len = max(req.logprob_start_len, encoder_len)

    def prepare_for_extend(self):
        self.forward_mode = ForwardMode.EXTEND

        self._refresh_attn_cp_prefill_split_specs()
        if (
            self.attn_cp_prefill_split_specs is not None
            and self.model_config.is_encoder_decoder
        ):
            raise ValueError(
                "Phase 2 sharded-KV prefill does not support encoder-decoder batches"
            )

        if self.is_dllm():
            # For DLLM, we use a separate forward mode
            self.forward_mode = ForwardMode.DLLM_EXTEND

        self.scale_seq_factor = self._get_scale_seq_factor()

        # Init tensors
        reqs = self.reqs
        scale = self.scale_seq_factor
        logical_prefix_lens = [len(r.prefix_indices) // scale for r in reqs]
        input_ids = [r.fill_ids[lpl:] for r, lpl in zip(reqs, logical_prefix_lens)]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        orig_seq_lens = [max(len(r.fill_ids), len(r.origin_input_ids)) for r in reqs]
        prefix_lens = logical_prefix_lens
        extend_lens = [r.extend_input_len for r in reqs]
        expanded_seq_lens = [s * scale for s in seq_lens]

        # For matryoshka embeddings
        if self.model_config.is_matryoshka and any(
            r.dimensions is not None for r in reqs
        ):
            self.dimensions = [
                r.dimensions if r.dimensions else self.model_config.hidden_size
                for r in reqs
            ]

        # OR across the batch so ForwardBatch matches a single fused forward; requests
        # that did not ask for PHS still skip attaching it in the output processor.
        self.return_pooled_hidden_states = any(
            r.return_pooled_hidden_states for r in reqs
        )

        token_type_ids = [
            r.token_type_ids for r in reqs if r.token_type_ids is not None
        ]

        _pin = is_pin_memory_available(self.device)
        input_ids_tensor = torch.tensor(
            list(chain.from_iterable(input_ids)), dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        self.oe_context = None
        if get_global_server_args().prepare_n_gram_inputs:
            oe_grams, _ = get_welm_oe_hash_config(self.model_config)
            if not oe_grams:
                raise RuntimeError(
                    "WeLM OE requires oe_grams for the CUDA hash kernel."
                )
            if torch.device(self.device).type != "cuda":
                raise RuntimeError("WeLM OE requires the CUDA hash kernel.")
            if not should_use_welm_oe_hash_kernel(self.model_config):
                raise RuntimeError(
                    "WeLM OE CUDA hash kernel is not supported by this model config."
                )
            self.oe_context = OverEncodingContext.from_extend_hash_kernel(
                reqs,
                logical_prefix_lens,
                max(oe_grams) - 1,
                legacy_history_width=max(oe_grams),
            )
        seq_lens_tensor = torch.tensor(
            expanded_seq_lens, dtype=torch.int64, pin_memory=_pin
        ).to(self.device, non_blocking=True)
        seq_lens_cpu = torch.tensor(expanded_seq_lens, dtype=torch.int64)
        orig_seq_lens_tensor = torch.tensor(
            orig_seq_lens, dtype=torch.int32, pin_memory=_pin
        ).to(self.device, non_blocking=True)

        token_type_ids_tensor = None
        if len(token_type_ids) > 0:
            token_type_ids_tensor = torch.tensor(
                sum(token_type_ids, []), dtype=torch.int64, pin_memory=_pin
            ).to(self.device, non_blocking=True)

        # Set batch fields needed by alloc_for_extend
        self.prefix_lens = prefix_lens
        self.extend_lens = extend_lens
        self.seq_lens = seq_lens_tensor
        self.seq_lens_cpu = seq_lens_cpu
        self.extend_num_tokens = extend_num_tokens * scale

        # Allocate memory
        out_cache_loc, req_pool_indices_tensor, req_pool_indices = alloc_for_extend(
            self
        )

        # Set fields
        input_embeds = []
        all_replace_embeds: List[torch.Tensor] = []
        all_replace_positions: List[int] = []
        has_replace_embeds = False
        input_id_pointer = 0
        input_id_lens = [len(input_id) for input_id in input_ids]
        extend_input_logprob_token_ids = []
        multimodal_inputs = []
        mamba_track_mask_cpu = []
        mamba_track_indices_cpu = []
        mamba_track_seqlens_cpu = []

        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.req_pool_idx = req_pool_indices[i]
            assert seq_len - pre_len == req.extend_input_len, (
                f"seq_len={seq_len}, pre_len={pre_len}, "
                f"extend_input_len={req.extend_input_len}, "
                f"len(prefix_indices)={len(req.prefix_indices)}, "
                f"batch_scale={scale}, req_scale={getattr(req, '_scale_seq_factor', 1)}, "
                f"is_chunked={req.is_chunked}, rid={req.rid}"
            )

            req.extend_batch_idx += 1

            # update req-level memory management fields
            req.kv_committed_len = seq_len
            req.kv_allocated_len = seq_len

            # If input_embeds are available, store them
            if req.input_embeds is not None:
                # Slice to match extend_input_len — PrefillAdder truncates
                # fill_ids/extend_input_len on chunk overflow but not input_embeds.
                input_embeds.extend(
                    req.input_embeds[pre_len : pre_len + req.extend_input_len]
                )

            if req.positional_embed_overrides is not None:
                # Override positions are absolute in the full sequence.
                # Convert to extend-tensor coordinates by subtracting pre_len,
                # then skip any that fall within the cached prefix.
                embeds_to_add = []
                for embed_idx, pos in enumerate(
                    req.positional_embed_overrides.positions
                ):
                    extend_pos = pos - pre_len
                    if extend_pos < 0 or extend_pos >= req.extend_input_len:
                        continue  # Outside current extend chunk, skip
                    embeds_to_add.append((embed_idx, input_id_pointer + extend_pos))
                if embeds_to_add:
                    has_replace_embeds = True
                    indices, positions = zip(*embeds_to_add)
                    all_replace_embeds.append(
                        req.positional_embed_overrides.embeds[list(indices)]
                    )
                    all_replace_positions.extend(positions)
            input_id_pointer += input_id_lens[i]

            multimodal_inputs.append(req.multimodal_inputs)

            # This also owns the first-chunk HiCache source breakdown.
            req.update_cached_token_stats(pre_len, seq_len)
            req.is_retracted = False

            if get_global_server_args().enable_mamba_extra_buffer():
                self._mamba_radix_cache_v2_req_prepare_for_extend(
                    req,
                    mamba_track_mask_cpu,
                    mamba_track_indices_cpu,
                    mamba_track_seqlens_cpu,
                )

            if self.return_logprob:
                # Find input logprob token ids.
                # First, find a global index within origin_input_ids and slide it by 1
                # to compute input logprobs. It is because you need the next token
                # to compute input logprobs. E.g., (chunk size 2)
                #
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [1, 2]
                # extend_input_logprob_token_id = [2, 3]
                #
                # Note that it can also overflow. In this case, we pad it with 0.
                # input_logprobs = [1, 2, 3, 4]
                # fill_ids = [3, 4]
                # extend_input_logprob_token_id = [4, 0]
                global_start_idx, global_end_idx = (
                    pre_len,
                    len(req.fill_ids),
                )
                if req.logprob_start_len == -1:
                    logprob_start_len = len(req.origin_input_ids)
                else:
                    logprob_start_len = req.logprob_start_len
                # Apply logprob_start_len
                if global_start_idx < logprob_start_len:
                    global_start_idx = logprob_start_len

                if req.retract_replay_skips_logprob():
                    # The replay computes no input logprobs (see
                    # set_extend_input_len), so contribute no token ids to keep
                    # the flat list aligned with the pruned logprob rows.
                    logprob_token_ids = []
                else:
                    logprob_token_ids = req.origin_input_ids[
                        global_start_idx + 1 : global_end_idx + 1
                    ]
                extend_input_logprob_token_ids.extend(logprob_token_ids)

                # We will need req.extend_input_len - req.extend_logprob_start_len number of
                # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
                extend_input_logprob_token_ids.extend(
                    [0]
                    * (
                        req.extend_input_len
                        - req.extend_logprob_start_len
                        - len(logprob_token_ids)
                    )
                )

        if self.return_logprob:
            extend_input_logprob_token_ids = torch.tensor(
                extend_input_logprob_token_ids
            )
            # Clamp placeholder or out-of-range token IDs (e.g., multimodal hashes)
            # so they stay within the vocab boundary before being sent to GPU.
            extend_input_logprob_token_ids.clamp_(0, self.model_config.vocab_size - 1)
        else:
            extend_input_logprob_token_ids = None

        if has_replace_embeds:
            replace_embeds_tensor = torch.cat(all_replace_embeds, dim=0).to(
                self.device, non_blocking=True
            )
            replace_positions_tensor = torch.tensor(
                all_replace_positions, dtype=torch.long, device=self.device
            )
        else:
            replace_embeds_tensor = None
            replace_positions_tensor = None

        self.input_ids = input_ids_tensor
        self.req_pool_indices = req_pool_indices_tensor
        self.orig_seq_lens = orig_seq_lens_tensor
        self.out_cache_loc = out_cache_loc
        self.input_embeds = (
            torch.tensor(input_embeds, pin_memory=_pin).to(
                self.device, non_blocking=True
            )
            if input_embeds
            else None
        )
        self.replace_embeds = replace_embeds_tensor
        self.replace_positions = replace_positions_tensor
        for mm_input in multimodal_inputs:
            if mm_input is None:
                continue
            if isinstance(mm_input.vision_position_ids, torch.Tensor):
                mm_input.vision_position_ids = mm_input.vision_position_ids.to(
                    self.device, non_blocking=True
                )
            if isinstance(mm_input.visible_frame_counts, torch.Tensor):
                mm_input.visible_frame_counts = mm_input.visible_frame_counts.to(
                    self.device, non_blocking=True
                )
        self.multimodal_inputs = multimodal_inputs
        self.token_type_ids = token_type_ids_tensor
        self.seq_lens_sum = sum(expanded_seq_lens)

        # Pre-compute delimiter indices as CPU tensors for MIS.
        # When --enable-mis is on, every request in the batch is expected to
        # carry delimiter indices (the score endpoint always produces MIS-structured
        # requests). Consumers index this list without None-checking.
        if get_global_server_args().enable_mis and any(
            r.multi_item_delimiter_indices is not None for r in reqs
        ):
            assert all(
                r.multi_item_delimiter_indices is not None for r in reqs
            ), "MIS batch must have delimiter indices on every request"
            self.multi_item_delimiter_indices = [
                torch.tensor(r.multi_item_delimiter_indices, dtype=torch.int64)
                for r in reqs
            ]
        else:
            self.multi_item_delimiter_indices = None

        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.prefix_lens = [p * scale for p in prefix_lens]
        self.extend_lens = [e * scale for e in extend_lens]
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        if get_global_server_args().enable_mamba_extra_buffer():
            self.mamba_track_indices = torch.tensor(
                mamba_track_indices_cpu,
                dtype=torch.int64,
                device=self.device,
            )
            self.mamba_track_mask = torch.tensor(
                mamba_track_mask_cpu,
                dtype=torch.bool,
                device=self.device,
            )
            self.mamba_track_seqlens = torch.tensor(
                mamba_track_seqlens_cpu,
                dtype=torch.int64,
                device=self.device,
            )

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_extend(input_ids, seq_lens)

        self.router_replay_topk_ids, self.router_replay_mask = (
            build_router_replay_extend_batch(
                reqs, logical_prefix_lens=logical_prefix_lens, device=self.device
            )
        )

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def _mamba_radix_cache_v2_req_prepare_for_extend(
        self,
        req: Req,
        mamba_track_mask_cpu: List[bool],
        mamba_track_indices_cpu: List[int],
        mamba_track_seqlens_cpu: List[int],
    ):
        def _force_track_h(i: int) -> int:
            assert i % FLA_CHUNK_SIZE == 0
            # There are 3 cases for mamba_track_seqlen passed to mamba_track_seqlens_cpu:
            # 1) aligned with FLA_CHUNK_SIZE-> retrieve from last_recurrent_state
            #    a) is the last position -> retrieve from last_recurrent_state
            #    b) is NOT the last position -> retrieve from h
            # 2) unaligned with FLA_CHUNK_SIZE -> retrieve from h
            # Currently, the math calculation only supports case 1a and 2. So for 1b, we need to add 1
            # to force the math calculation to retrieve the correct mamba state from h.
            return i + 1

        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        mask = req.extend_input_len >= mamba_cache_chunk_size
        mamba_track_mask_cpu.append(mask)
        mamba_track_indices_cpu.append(
            req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx].item()
        )
        mamba_track_seqlen = -1
        if mask:
            # mamba_track_seqlen is used to calculate the indices to track in
            # hybrid_linear_attn_backend's _init_track_ssm_indices. Due to the
            # fact that the ssm state between aligned and non-aligned are retrieved differently,
            # if 1) last pos and 2) is aligned, then retrieved from the last_recurrent_state,
            # otherwise retrieved from h (i.e. unaligned).
            # We need to pass the non-aligned seqlen to the calculation. Even though
            # we pass in mamba_track_seqlen, the actual tracked seqlen is mamba_last_track_seqlen.
            mamba_track_seqlen = len(req.prefix_indices) + req.extend_input_len

            # mamba_track_seqlen_aligned/mamba_last_track_seqlen is actual tracked seqlen. Used to pass to
            # mamba radix cache to track which seqlen this mamba state should store at.
            mamba_track_seqlen_aligned = (
                len(req.prefix_indices)
                + (req.extend_input_len // mamba_cache_chunk_size)
                * mamba_cache_chunk_size
            )

            # mamba_track_fla_chunk_aligned is the aligned seqlen based on FLA_CHUNK_SIZE
            # If mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned, which can be true when
            # page_size > FLA_CHUNK_SIZE, we need to force the math calculation to retrieve the correct mamba state from h
            # by _force_track_h()
            mamba_track_fla_chunk_aligned = (
                len(req.prefix_indices)
                + (req.extend_input_len // FLA_CHUNK_SIZE) * FLA_CHUNK_SIZE
            )
            if mamba_track_fla_chunk_aligned != mamba_track_seqlen_aligned:
                # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                mamba_track_seqlen = _force_track_h(mamba_track_seqlen_aligned)

            req.mamba_next_track_idx = (
                self.req_to_token_pool.get_mamba_ping_pong_other_idx(
                    req.mamba_next_track_idx
                )
            )
            if req.mamba_branching_seqlen is not None:
                # track branching point in this forward if the branching point
                # is within the current extend batch.
                branching_seqlen_aligned_mask = (
                    req.mamba_branching_seqlen - len(req.prefix_indices)
                ) % mamba_cache_chunk_size == 0
                if (
                    req.mamba_branching_seqlen > len(req.prefix_indices)
                    and req.mamba_branching_seqlen < mamba_track_seqlen
                    and branching_seqlen_aligned_mask
                ):
                    # We want to track mamba_track_seqlen_aligned, and it's not the last position,
                    # so we need to add 1 to the seqlen to retrieve the correct mamba state from h.
                    # See _force_track_h() for more details.
                    mamba_track_seqlen = _force_track_h(req.mamba_branching_seqlen)
                    mamba_track_seqlen_aligned = req.mamba_branching_seqlen
            req.mamba_last_track_seqlen = mamba_track_seqlen_aligned
        mamba_track_seqlens_cpu.append(mamba_track_seqlen)

    def prepare_extend_metadata_for_dp_sync(self):
        # DP-attention spec scheduling needs extend sizes before the global MLP
        # sync, but must not launch local CUDA prefill prep before every DP rank
        # has reached that sync point.
        self.forward_mode = ForwardMode.EXTEND
        self.scale_seq_factor = self._get_scale_seq_factor()

        reqs = self.reqs
        scale = self.scale_seq_factor
        logical_prefix_lens = [len(r.prefix_indices) // scale for r in reqs]
        extend_num_tokens = sum(
            len(r.fill_ids) - lpl for r, lpl in zip(reqs, logical_prefix_lens)
        )
        extend_lens = [r.extend_input_len for r in reqs]

        for req, pre_len in zip(reqs, logical_prefix_lens):
            if req.retract_replay_skips_logprob():
                # Keep in sync with set_extend_input_len: a retracted request's
                # replay skips its whole logprob region. Check this before the
                # pre_len comparison because such a replay's prefix match may
                # cover positions past logprob_start_len.
                req.extend_logprob_start_len = req.extend_input_len
            elif req.logprob_start_len >= pre_len:
                if req.logprob_start_len == len(req.origin_input_ids):
                    req.extend_logprob_start_len = req.extend_input_len
                else:
                    req.extend_logprob_start_len = min(
                        req.logprob_start_len - pre_len,
                        req.extend_input_len,
                        req.seqlen - 1,
                    )
            else:
                req.extend_logprob_start_len = 0

        self.prefix_lens = [p * scale for p in logical_prefix_lens]
        self.extend_lens = [e * scale for e in extend_lens]
        self.extend_num_tokens = extend_num_tokens * scale
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]

    def prepare_for_split_prefill(self):
        self.prepare_for_extend()
        # For split prefill, we need to set the forward mode to SPLIT_PREFILL
        self.forward_mode = ForwardMode.SPLIT_PREFILL

    def mix_with_running(self, running_batch: "ScheduleBatch"):
        self.forward_mode = ForwardMode.MIXED
        running_bs = running_batch.batch_size()
        scale = self._get_scale_seq_factor()

        for req in running_batch.reqs:
            req.fill_ids = req.origin_input_ids + req.output_ids
            req.set_extend_input_len(1)

        input_ids = torch.cat([self.input_ids, running_batch.input_ids])
        out_cache_loc = torch.cat([self.out_cache_loc, running_batch.out_cache_loc])

        self.merge_batch(running_batch)
        self.input_ids = input_ids
        self.out_cache_loc = out_cache_loc

        # For overlap scheduler, the output_ids has one step delay
        delta = 0 if self.enable_overlap else -1

        # NOTE: prefix_indices is what has been cached, but we don't cache each decode step
        self.prefix_lens.extend(
            [
                (len(r.origin_input_ids) + len(r.output_ids) + delta) * scale
                for r in running_batch.reqs
            ]
        )
        self.extend_lens.extend([scale] * running_bs)
        self.extend_num_tokens += running_bs * scale
        # TODO (lianmin): Revisit this. It should be seq_len - 1
        self.extend_logprob_start_lens.extend([0] * running_bs)
        self.is_prefill_only = False

    def new_tokens_required_next_decode(
        self, selected_indices: Optional[List[int]] = None
    ):
        page_size = self.token_to_kv_pool_allocator.page_size
        scale = self._get_scale_seq_factor()
        requests = (
            self.reqs
            if selected_indices is None
            else [self.reqs[i] for i in selected_indices]
        )

        if self.spec_algorithm.is_none():
            return sum(
                ceil_align(r.kv_committed_len + scale, page_size)
                - ceil_align(r.kv_committed_len, page_size)
                for r in requests
            )

        if self.is_spec_v2:
            return self._new_tokens_required_next_decode_spec_v2(requests, page_size)

        server_args = get_global_server_args()
        len_per_topk = server_args.speculative_num_steps or 1
        spec_topk = server_args.speculative_eagle_topk or 1
        spec_tokens = server_args.speculative_num_draft_tokens

        if page_size > 1 and spec_topk > 1:
            # last partial page and ceil alignment
            len_per_topk = ceil_align(len_per_topk + page_size, page_size)
            spec_tokens = ceil_align(spec_tokens, page_size)
        elif page_size > 1:
            # only page alignment
            len_per_topk = ceil_align(len_per_topk, page_size)
            spec_tokens = ceil_align(spec_tokens, page_size)

        num_tokens = max(len_per_topk * spec_topk, spec_tokens) * len(requests)
        return num_tokens

    def _new_tokens_required_next_decode_spec_v2(self, requests, page_size):
        """Tight estimate matching eagle_info_v2.prepare_for_decode allocation."""
        from sglang.srt.managers.utils import get_alloc_len_per_decode

        alloc_len = get_alloc_len_per_decode()
        total = 0
        for r in requests:
            x = max(0, r.kv_committed_len + 2 * alloc_len - r.kv_allocated_len)
            cur = r.kv_allocated_len
            nxt = cur + x
            total += ceil_align(nxt, page_size) - ceil_align(cur, page_size)
        return total

    def check_decode_mem(self, selected_indices: Optional[List[int]] = None):
        num_tokens = self.new_tokens_required_next_decode(selected_indices)
        evict_from_tree_cache(self.tree_cache, num_tokens)
        return self.token_to_kv_pool_allocator.available_size() >= num_tokens

    def retract_all(self, server_args: ServerArgs):
        retracted_reqs = self.reqs
        for idx in range(len(self.reqs)):
            self.release_req(idx, len(self.reqs) - idx, server_args)

        self.filter_batch(retracted_reqs)
        return retracted_reqs

    def retract_decode(
        self,
        server_args: ServerArgs,
        check_decode_mem: Optional[Callable[[Optional[List[int]]], bool]] = None,
    ) -> Tuple[List[Req], float, List[Req]]:
        """Retract the decoding requests when there is not enough memory."""
        check_decode_mem = check_decode_mem or self.check_decode_mem
        sorted_indices = list(range(len(self.reqs)))

        # TODO(lsyin): improve retraction policy for radix cache
        # For spec decoding, filter_batch API can only filter
        # requests from the back, so we can only retract from the back.
        # TODO(sang): Clean up finish path and support better retract
        # policy.
        if not server_args.speculative_algorithm:
            use_deterministic_cp_tiebreaker = server_args.attn_cp_mode == "sharded-kv"
            sorted_indices.sort(
                key=lambda i: (
                    len(self.reqs[i].output_ids),
                    -len(self.reqs[i].origin_input_ids),
                    (str(self.reqs[i].rid) if use_deterministic_cp_tiebreaker else ""),
                ),
                reverse=True,
            )

        retracted_reqs = []
        first_iter = True
        while first_iter or (not check_decode_mem(selected_indices=sorted_indices)):
            if len(sorted_indices) == 1:
                # Always keep at least one request
                break

            first_iter = False
            idx = sorted_indices.pop()
            req = self.reqs[idx]
            retracted_reqs.append(req)
            # release memory and don't insert into the tree because we need the space instantly
            self.release_req(idx, len(sorted_indices), server_args)

        reqs_to_abort: List[Req] = []
        if len(sorted_indices) <= 1 and not check_decode_mem(
            selected_indices=sorted_indices
        ):
            # Even the last remaining request cannot fit in memory.
            # Instead of crashing the scheduler, gracefully abort it.
            last_idx = sorted_indices.pop()
            last_req = self.reqs[last_idx]
            last_req.to_finish = FINISH_ABORT(
                "Out of memory even after retracting all other requests "
                "in the decode batch. Aborting the last request.",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            reqs_to_abort.append(last_req)
            self.release_req(last_idx, 0, server_args)
            logger.warning(
                "retract_decode: aborted last request %s due to OOM", last_req.rid
            )

        self.filter_batch(keep_indices=sorted_indices)

        # Reqs in batch are filtered
        total_decoded_tokens = sum(len(r.output_ids) for r in self.reqs)
        total_max_new_tokens = sum(r.sampling_params.max_new_tokens for r in self.reqs)

        new_estimate_ratio = (
            total_decoded_tokens
            + envs.SGLANG_RETRACT_DECODE_STEPS.get() * len(self.reqs)
        ) / (
            total_max_new_tokens + 1
        )  # avoid zero division
        new_estimate_ratio = min(1.0, new_estimate_ratio)

        return retracted_reqs, new_estimate_ratio, reqs_to_abort

    def release_req(self, idx: int, remaing_req_count: int, server_args: ServerArgs):
        req = self.reqs[idx]
        allow_uncommitted_tail = req.prepare_for_retract()

        if self.hisparse_coordinator is not None and not req.finished():
            self.hisparse_coordinator.retract_req(req)

        if server_args.disaggregation_mode == "decode":
            req.offload_kv_cache(
                self.req_to_token_pool, self.token_to_kv_pool_allocator
            )
        # TODO (csy): for preempted requests, we may want to insert into the tree
        if allow_uncommitted_tail:
            release_kv_cache(
                req,
                self.tree_cache,
                is_insert=False,
                allow_uncommitted_tail=True,
            )
        else:
            release_kv_cache(req, self.tree_cache, is_insert=False)
        if allow_uncommitted_tail and server_args.disaggregation_mode != "decode":
            req.welm_deferred_decode_state.rollback_ready_to_prefill_pending()
        # NOTE(lsyin): we should use the newly evictable memory instantly.
        num_tokens = remaing_req_count * envs.SGLANG_RETRACT_DECODE_STEPS.get()
        evict_from_tree_cache(self.tree_cache, num_tokens)

        req.reset_for_retract()

    def prepare_encoder_info_decode(self):
        # Reset the encoder cached status
        self.encoder_cached = [True] * len(self.reqs)

    def prepare_for_idle(self):
        self.forward_mode = ForwardMode.IDLE
        self._clear_welm_deferred_prefill_metadata()
        self._clear_attn_cp_prefill_split_specs(clear_reqs=True)
        self.input_ids = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens = torch.empty(0, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.empty(0, dtype=torch.int64)
        self.orig_seq_lens = torch.empty(0, dtype=torch.int32, device=self.device)
        self.out_cache_loc = torch.empty(0, dtype=torch.int64, device=self.device)
        self.req_pool_indices = torch.empty(0, dtype=torch.int64, device=self.device)
        self.router_replay_topk_ids = None
        self.router_replay_mask = None
        self.seq_lens_sum = 0
        self.extend_num_tokens = 0
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def prepare_for_welm_deferred_seed_decode(self):
        """Populate ready deferred seeds as ordinary decode rows."""
        if not self.reqs:
            raise RuntimeError("WeLM deferred seed batch must not be empty")

        seed_token_ids = []
        committed_lens = []
        req_pool_indices = []
        for req in self.reqs:
            state = vars(req).get("welm_deferred_decode_state")
            if not isinstance(state, WelmDeferredDecodeState):
                raise RuntimeError(
                    "WeLM deferred seed admission is missing lifecycle state"
                )
            if state.phase is not WelmDeferredDecodePhase.READY:
                raise RuntimeError(
                    "WeLM deferred seed admission requires READY state: "
                    f"rid={req.rid}, phase={state.phase.value}"
                )
            state.validate_prompt_tokens(req.origin_input_ids)
            if req.output_ids:
                raise RuntimeError(
                    "WeLM deferred seed admission requires an empty output"
                )
            if req.kv_allocated_len != state.committed_kv_len:
                raise RuntimeError(
                    "WeLM deferred seed allocation length mismatch: "
                    f"rid={req.rid}, allocated={req.kv_allocated_len}, "
                    f"committed={state.committed_kv_len}"
                )
            if req.kv_committed_len != state.committed_kv_len:
                raise RuntimeError(
                    "WeLM deferred seed committed length mismatch: "
                    f"rid={req.rid}, request={req.kv_committed_len}, "
                    f"state={state.committed_kv_len}"
                )
            if req.req_pool_idx is None:
                raise RuntimeError(
                    f"WeLM deferred seed {req.rid} has no decode request slot"
                )

            seed_token_ids.append(state.seed_token_id)
            committed_lens.append(state.committed_kv_len)
            req_pool_indices.append(req.req_pool_idx)

        self.output_ids = torch.tensor(
            seed_token_ids, dtype=torch.int64, device=self.device
        )
        self.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=self.device
        )
        self.seq_lens = torch.tensor(
            committed_lens, dtype=torch.int64, device=self.device
        )
        self.seq_lens_cpu = torch.tensor(committed_lens, dtype=torch.int64)
        self.orig_seq_lens = torch.tensor(
            committed_lens, dtype=torch.int32, device=self.device
        )
        self.seq_lens_sum = sum(committed_lens)
        if self.return_logprob:
            self.top_logprobs_nums = [req.top_logprobs_num for req in self.reqs]
            self.token_ids_logprobs = [req.token_ids_logprob for req in self.reqs]
        self.multimodal_inputs = [req.multimodal_inputs for req in self.reqs]
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self, self.model_config.vocab_size
        )

    @property
    def is_spec_v2(self):
        # FIXME: finally deprecate is_spec_v2
        ret = self.enable_overlap and not self.spec_algorithm.is_none()
        assert not ret or self.spec_algorithm.supports_spec_v2()
        return ret

    def prepare_for_decode(self):
        scale = self._get_scale_seq_factor()
        self.forward_mode = ForwardMode.DECODE
        self._clear_welm_deferred_prefill_metadata()
        bs = len(self.reqs)
        deferred_seed_states = []
        deferred_seed_requires_logprob = False
        deferred_penalty_active_mask = []
        has_inactive_deferred_penalty_row = False
        for req in self.reqs:
            state = vars(req).get("welm_deferred_decode_state")
            if state is None:
                deferred_penalty_active_mask.append(True)
                continue
            if not isinstance(state, WelmDeferredDecodeState):
                raise RuntimeError(
                    "decode batch received invalid WeLM deferred lifecycle state"
                )
            if state.phase is WelmDeferredDecodePhase.PREFILL_PENDING:
                raise RuntimeError(
                    "WeLM deferred seed entered decode before Prefill completion"
                )
            is_ready_seed = state.phase is WelmDeferredDecodePhase.READY
            penalty_active = state.phase is WelmDeferredDecodePhase.CONSUMED
            deferred_penalty_active_mask.append(penalty_active)
            has_inactive_deferred_penalty_row |= not penalty_active
            if is_ready_seed:
                state.validate_prompt_tokens(req.origin_input_ids)
                if req.output_ids:
                    raise RuntimeError(
                        "WeLM deferred seed entered decode with generated output"
                    )
                if req.kv_committed_len != state.committed_kv_len:
                    raise RuntimeError(
                        "WeLM deferred seed committed length changed before decode"
                    )
                if self.enable_overlap:
                    # The seed is the current input but still lives in the prompt.
                    req._overlap_decode_count = -1
                deferred_seed_states.append(state)
                deferred_seed_requires_logprob |= req.return_logprob

        if deferred_seed_states:
            self.return_logprob |= deferred_seed_requires_logprob
            self.top_logprobs_nums = (
                [req.top_logprobs_num for req in self.reqs]
                if self.return_logprob
                else None
            )
            self.token_ids_logprobs = (
                [req.token_ids_logprob for req in self.reqs]
                if self.return_logprob
                else None
            )

        # Decode embeds the last output token via embed_tokens; clear the stale
        # prefill-time tensor so it doesn't leak into ForwardBatch.
        self.input_embeds = None
        self._clear_attn_cp_prefill_split_specs(clear_reqs=True)

        # Clear context parallel metadata - CP is only for prefill, not decode
        if hasattr(self, "attn_cp_metadata") and self.attn_cp_metadata is not None:
            self.attn_cp_metadata = None

        if self.is_spec_v2:
            # TODO(spec-v2): all spec v2 should go through this path
            draft_input: EagleDraftInput = self.spec_info
            draft_input.prepare_for_decode(self)

        if not self.spec_algorithm.is_none():
            # if spec decoding is used, the decode batch is prepared inside
            # `forward_batch_speculative_generation` after running draft models.
            return

        if self.sampling_info.penalizer_orchestrator.is_required:
            penalty_active_mask = (
                torch.tensor(
                    deferred_penalty_active_mask,
                    dtype=torch.bool,
                    device=self.device,
                )
                if has_inactive_deferred_penalty_row
                else None
            )
            if self.enable_overlap:
                # TODO: this can be slow, optimize this.
                delayed_output_ids = torch.tensor(
                    [
                        (
                            req.output_ids[-1]
                            if len(req.output_ids)
                            else req.origin_input_ids[-1]
                        )
                        for req in self.reqs
                    ],
                    dtype=torch.int64,
                    device=self.device,
                )
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    delayed_output_ids,
                    row_active_mask=penalty_active_mask,
                )
            else:
                self.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                    self.output_ids.to(torch.int64),
                    row_active_mask=penalty_active_mask,
                )

        # Update fields
        # Coerce to int64: torch sampling helpers (sampling_from_probs_torch /
        # top_k_top_p_min_p_sampling_from_probs_torch) return int32 token ids,
        # but downstream kernels enforce int64 (e.g. DeepSeek-V4 hash_topk).
        self.input_ids = self.output_ids.to(torch.int64)
        self.output_ids = None
        # Router replay trace_pos must equal the position whose MoE forward
        # is being executed in THIS decode step. Capture writes routing for
        # the token at sequence position `in_len + k` (slot req_to_token[p]).
        # `seq_lens_cpu` is incremented by `scale` LATER in this function
        # (see `self.seq_lens_cpu.add_(scale)` below), so its current value
        # is exactly the position about to be processed.
        trace_positions = (self.seq_lens_cpu // scale).tolist()
        self.router_replay_topk_ids, self.router_replay_mask = (
            build_router_replay_decode_batch(
                self.reqs, device=self.device, trace_positions=trace_positions
            )
        )

        self.oe_context = None
        if get_global_server_args().prepare_n_gram_inputs:
            oe_grams, _ = get_welm_oe_hash_config(self.model_config)
            if not oe_grams:
                raise RuntimeError(
                    "WeLM OE requires oe_grams for the CUDA hash kernel."
                )
            if torch.device(self.device).type != "cuda":
                raise RuntimeError("WeLM OE requires the CUDA hash kernel.")
            if not should_use_welm_oe_hash_kernel(self.model_config):
                raise RuntimeError(
                    "WeLM OE CUDA hash kernel is not supported by this model config."
                )
            self.oe_context = OverEncodingContext.from_decode_hash_kernel(
                self.reqs,
                self.enable_overlap,
                max(oe_grams) - 1,
                legacy_history_width=max(oe_grams),
            )

        if self.model_config.is_encoder_decoder:
            self.prepare_encoder_info_decode()

        # Allocate memory
        self.out_cache_loc = alloc_for_decode(self, token_per_req=scale)

        # Update req-level memory management fields
        for req in self.reqs:
            req.decode_batch_idx += 1
            req.kv_committed_len += 1
            req.kv_allocated_len += 1
        locs = self.seq_lens.clone()

        # Update seq_lens after allocation
        if self.enable_overlap:
            # Do not use in-place operations in the overlap mode
            self.seq_lens = self.seq_lens + scale
            self.seq_lens_cpu = self.seq_lens_cpu + scale
            self.orig_seq_lens = self.orig_seq_lens + 1
        else:
            # A faster in-place version
            self.seq_lens.add_(scale)
            self.seq_lens_cpu.add_(scale)
            self.orig_seq_lens.add_(1)
        self.seq_lens_sum += bs * scale

        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.map_last_loc_to_buffer(
                self.seq_lens,
                self.out_cache_loc,
                self.req_pool_indices,
                self.seq_lens_cpu,
            )

        if get_global_server_args().enable_mamba_extra_buffer():
            if len(self.reqs) == 0:
                self.mamba_track_indices = torch.empty(
                    (0,), dtype=torch.int64, device=self.device
                )
            else:
                # already on device
                all_buffers = torch.stack(
                    [req.mamba_ping_pong_track_buffer for req in self.reqs]
                )
                idx = (
                    torch.tensor(
                        [req.mamba_next_track_idx for req in self.reqs],
                        dtype=torch.int64,
                        pin_memory=True,
                    )
                    .unsqueeze(1)
                    .to(device=all_buffers.device, non_blocking=True)
                )
                self.mamba_track_indices = (
                    torch.gather(all_buffers, 1, idx).squeeze(1).to(torch.int64)
                )

            # async H2D
            self.mamba_track_mask = (
                (self.seq_lens_cpu % get_global_server_args().mamba_track_interval == 0)
                .pin_memory()
                .to(device=self.device, non_blocking=True)
            )

        if scale > 1:
            self.extend_num_tokens = bs * scale
            self.extend_lens = [scale] * bs
            self.prefix_lens = locs.tolist()
            self.extend_logprob_start_lens = [0] * bs
            self.extend_input_logprob_token_ids = None

        for state in deferred_seed_states:
            state.transition_to(WelmDeferredDecodePhase.INFLIGHT)

    def maybe_evict_swa(self):
        if self.tree_cache.supports_swa():
            sliding_window_size = self.tree_cache.sliding_window_size
            server_args = get_global_server_args()

            for idx, req in enumerate(self.reqs):
                if self.forward_mode.is_decode():
                    if req.decode_batch_idx % sliding_window_size == 1:
                        self._evict_swa(req, req.seqlen - 1)
                elif self.forward_mode.is_extend() and self.tree_cache.is_chunk_cache():
                    pre_len = self.prefix_lens[idx]
                    if self.enable_overlap:
                        if req.extend_batch_idx < 2:
                            continue
                        pre_len = (
                            pre_len - server_args.chunked_prefill_size
                            if server_args.chunked_prefill_size > 0
                            else pre_len
                        )
                    self._evict_swa(req, pre_len)

    def _evict_swa(self, req: Req, pre_len: int):
        assert self.tree_cache.supports_swa(), "prefix cache must support swa"
        if hasattr(self.tree_cache, "evict_swa"):
            self.tree_cache.evict_swa(req, pre_len)
            return
        sliding_window_size = self.tree_cache.sliding_window_size

        if req.cache_protected_len % self.tree_cache.page_size != 0:
            return

        new_swa_evicted_seqlen = (
            (pre_len - req.cache_protected_len) // sliding_window_size
        ) * sliding_window_size
        new_swa_evicted_seqlen = max(new_swa_evicted_seqlen, req.swa_evicted_seqlen)
        if new_swa_evicted_seqlen <= req.swa_evicted_seqlen:
            return

        free_slots = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, req.swa_evicted_seqlen : new_swa_evicted_seqlen
        ]
        self.token_to_kv_pool_allocator.free_swa(free_slots)
        req.swa_evicted_seqlen = new_swa_evicted_seqlen

    def maybe_wait_verify_done(self):
        if self.is_spec_v2:
            draft_input: EagleDraftInput = self.spec_info
            if draft_input.verify_done is not None:
                draft_input.verify_done.synchronize()

    def filter_batch(
        self,
        chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
        keep_indices: Optional[List[int]] = None,
        # FIXME(lsyin): deprecate this API after spec v1 is deprecated
        v1_spec_info_filtered: Optional[bool] = False,
    ):
        # FIXME(lsyin): used here to get the correct seq_lens
        # The batch has been launched but we need it verified to get correct next batch info
        self.maybe_wait_verify_done()

        if keep_indices is None:
            if isinstance(chunked_req_to_exclude, Req):
                chunked_req_to_exclude = [chunked_req_to_exclude]
            elif chunked_req_to_exclude is None:
                chunked_req_to_exclude = []
            keep_indices = [
                i
                for i in range(len(self.reqs))
                if not self.reqs[i].finished()
                and self.reqs[i] not in chunked_req_to_exclude
            ]

        prebuilt_filter_state = None
        if (
            self.forward_mode is not None
            and self.forward_mode.is_prebuilt()
            and keep_indices
            and len(keep_indices) < len(self.reqs)
        ):
            extend_lens = list(self.extend_lens)
            packed_num_tokens = sum(extend_lens)
            if self.input_ids is None or self.input_ids.numel() != packed_num_tokens:
                raise RuntimeError(
                    "Cannot filter PREBUILT batch with invalid packed input_ids: "
                    f"input_tokens={None if self.input_ids is None else self.input_ids.numel()}, "
                    f"extend_tokens={packed_num_tokens}."
                )
            if (
                self.out_cache_loc is None
                or self.out_cache_loc.numel() != packed_num_tokens
            ):
                raise RuntimeError(
                    "Cannot filter PREBUILT batch with invalid packed KV slots: "
                    f"kv_slots={None if self.out_cache_loc is None else self.out_cache_loc.numel()}, "
                    f"extend_tokens={packed_num_tokens}."
                )

            input_chunks = self.input_ids.split(extend_lens)
            cache_chunks = self.out_cache_loc.split(extend_lens)
            prebuilt_filter_state = {
                "input_ids": torch.cat([input_chunks[i] for i in keep_indices]),
                "out_cache_loc": torch.cat(
                    [cache_chunks[i] for i in keep_indices]
                ),
                "prefix_lens": [self.prefix_lens[i] for i in keep_indices],
                "extend_lens": [extend_lens[i] for i in keep_indices],
                "extend_logprob_start_lens": [
                    self.extend_logprob_start_lens[i] for i in keep_indices
                ],
            }

            if self.oe_context is not None:
                hash_prefixes = self.oe_context.hash_prefixes
                prebuilt_filter_state["oe_hash_prefixes"] = (
                    None
                    if hash_prefixes is None
                    else [[row[i] for i in keep_indices] for row in hash_prefixes]
                )
                legacy_prefixes = self.oe_context.legacy_prefixes
                prebuilt_filter_state["oe_legacy_prefixes"] = (
                    None
                    if legacy_prefixes is None
                    else [[row[i] for i in keep_indices] for row in legacy_prefixes]
                )

        if keep_indices is None or len(keep_indices) == 0:
            # Filter out all requests
            self.reqs = []
            self.spec_info = None
            self.attn_cp_prefill_split_specs = None
            return

        if len(keep_indices) == len(self.reqs):
            # No need to filter
            self._refresh_attn_cp_prefill_split_specs()
            return

        keep_indices_device = torch.tensor(
            keep_indices,
            dtype=torch.int64,
            pin_memory=is_pin_memory_available(self.device),
        ).to(self.device, non_blocking=True)

        if self.model_config.is_encoder_decoder:
            self.encoder_lens = self.encoder_lens[keep_indices_device]
            self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

        self.reqs = [self.reqs[i] for i in keep_indices]
        self._refresh_attn_cp_prefill_split_specs()
        if self.multimodal_inputs is not None:
            self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
        self.req_pool_indices = self.req_pool_indices[keep_indices_device]
        self.seq_lens = self.seq_lens[keep_indices_device]
        self.seq_lens_cpu = self.seq_lens_cpu[keep_indices]
        self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
        self.out_cache_loc = None
        self.seq_lens_sum = self.seq_lens.sum().item()

        if self.output_ids is not None:
            self.output_ids = self.output_ids[keep_indices_device]

        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.router_replay_topk_ids = None
        self.router_replay_mask = None
        self.return_logprob = any(
            Req.requires_logprob_payload(req) for req in self.reqs
        )
        if self.return_logprob:
            self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
            self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
        else:
            self.top_logprobs_nums = None
            self.token_ids_logprobs = None

        self.has_stream = any(req.stream for req in self.reqs)
        self.has_grammar = any(req.grammar for req in self.reqs)

        self.sampling_info.filter_batch(keep_indices, keep_indices_device)
        # NOTE: spec_info filtered before batch filtering only happens in:
        # - Spec v1's verify phase
        # - Only for decode batch (running_batch)
        has_been_filtered = v1_spec_info_filtered and not self.is_spec_v2

        if self.spec_info:
            self.spec_info.filter_batch(
                new_indices=keep_indices_device,
                has_been_filtered=has_been_filtered,
            )

        if prebuilt_filter_state is not None:
            self.input_ids = prebuilt_filter_state["input_ids"]
            self.out_cache_loc = prebuilt_filter_state["out_cache_loc"]
            self.prefix_lens = prebuilt_filter_state["prefix_lens"]
            self.extend_lens = prebuilt_filter_state["extend_lens"]
            self.extend_num_tokens = sum(self.extend_lens)
            self.extend_logprob_start_lens = prebuilt_filter_state[
                "extend_logprob_start_lens"
            ]
            if self.oe_context is not None:
                oe_hash_prefixes = prebuilt_filter_state["oe_hash_prefixes"]
                self.oe_context.input_ids_buffer = (
                    None
                    if oe_hash_prefixes is None
                    else HashInputIdsBuffer(oe_hash_prefixes)
                )
                self.oe_context.legacy_prefixes = prebuilt_filter_state[
                    "oe_legacy_prefixes"
                ]

    def merge_batch(self, other: "ScheduleBatch"):
        # In the regular scheduler path:
        # 1) self is always prefill, whose seq_lens is not a future
        # 2) other is always decode, which is finished in previous step
        # so verify_done is already synced and this is a no-op.
        # In disagg decode + overlap, merge_batch can be called before
        # filter_batch, so running_batch.seq_lens may still be a forward_stream
        # future. Synchronize here to avoid a cross-stream data race.
        self.maybe_wait_verify_done()

        # Penalizer orchestrator must be merged before Batch.reqs is merged. This is because
        # orchestrator.merge() depends on Batch.reqs during preparation of each penalizers, so it
        # needs to be called with pre-merged Batch.reqs.
        self.sampling_info.merge_batch(other.sampling_info)

        # Encoder-decoder infos
        if self.model_config.is_encoder_decoder:
            self.encoder_lens = torch.cat([self.encoder_lens, other.encoder_lens])
            self.encoder_lens_cpu.extend(other.encoder_lens_cpu)
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.out_cache_loc = None
        self.seq_lens_sum += other.seq_lens_sum
        if self.output_ids is not None:
            self.output_ids = torch.cat([self.output_ids, other.output_ids])
        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        if self.return_logprob and other.return_logprob:
            self.top_logprobs_nums.extend(other.top_logprobs_nums)
            self.token_ids_logprobs.extend(other.token_ids_logprobs)
        elif self.return_logprob:
            self.top_logprobs_nums.extend([0] * len(other.reqs))
            self.token_ids_logprobs.extend([None] * len(other.reqs))
        elif other.return_logprob:
            self.top_logprobs_nums = [0] * len(self.reqs) + other.top_logprobs_nums
            self.token_ids_logprobs = [None] * len(self.reqs) + other.token_ids_logprobs
        self.reqs.extend(other.reqs)
        self._refresh_attn_cp_prefill_split_specs()
        if self.multimodal_inputs is not None:
            self.multimodal_inputs.extend(other.multimodal_inputs)

        self.return_logprob |= other.return_logprob
        self.has_stream |= other.has_stream
        self.has_grammar |= other.has_grammar
        self.return_hidden_states |= other.return_hidden_states
        self.is_prefill_only = self.is_prefill_only and other.is_prefill_only
        self.router_replay_topk_ids = None
        self.router_replay_mask = None

        if self.spec_info:
            self.spec_info.merge_batch(other.spec_info)
        if self.oe_context and other.oe_context:
            self.oe_context.merge_buffer(other.oe_context)

    def get_model_worker_batch(
        self, seq_lens_cpu_cache: Optional[torch.Tensor] = None
    ) -> ModelWorkerBatch:
        scale = self._get_scale_seq_factor()
        if (
            self.forward_mode.is_decode_or_idle()
            and scale <= 1
            and not getattr(self.spec_info, "welm_mtp_deferred_prefill_draft", False)
        ):
            extend_seq_lens = extend_prefix_lens = extend_logprob_start_lens = None
            welm_kv_mirror_last_q_indices = None
            welm_kv_mirror_active_batch_indices = None
            welm_kv_mirror_output_size = None
        else:
            extend_seq_lens = self.extend_lens
            extend_prefix_lens = self.prefix_lens
            extend_logprob_start_lens = self.extend_logprob_start_lens
            (
                welm_kv_mirror_last_q_indices,
                welm_kv_mirror_active_batch_indices,
                welm_kv_mirror_output_size,
            ) = self._get_welm_kv_mirror_last_q_indices()

        if self.sampling_info:
            if self.has_grammar:
                self.sampling_info.grammars = [req.grammar for req in self.reqs]
            else:
                self.sampling_info.grammars = None

        seq_lens_cpu = (
            seq_lens_cpu_cache if seq_lens_cpu_cache is not None else self.seq_lens_cpu
        )

        return ModelWorkerBatch(
            forward_mode=self.forward_mode,
            input_ids=self.input_ids,
            req_pool_indices=self.req_pool_indices,
            seq_lens=self.seq_lens,
            orig_seq_lens=self.orig_seq_lens,
            out_cache_loc=self.out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=self.seq_lens_sum,
            return_logprob=self.return_logprob,
            top_logprobs_nums=self.top_logprobs_nums,
            token_ids_logprobs=self.token_ids_logprobs,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            global_num_reqs=self.global_num_reqs,
            global_forward_modes=self.global_forward_modes,
            welm_kv_mirror_contract_flags=self.welm_kv_mirror_contract_flags,
            welm_deferred_prefill_flags=self.welm_deferred_prefill_flags,
            welm_mtp_global_prefill_num_tokens=(
                self.welm_mtp_global_prefill_num_tokens
            ),
            global_has_non_greedy_sampling=self.global_has_non_greedy_sampling,
            global_needs_top_p_sampling=self.global_needs_top_p_sampling,
            is_extend_in_batch=self.is_extend_in_batch,
            all_extend_in_batch=self.all_extend_in_batch,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            tbo_split_seq_index=self.tbo_split_seq_index,
            global_forward_mode=self.global_forward_mode,
            extend_num_tokens=self.extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            attn_cp_prefill_split_specs=self._current_attn_cp_prefill_split_specs(),
            welm_kv_mirror_last_q_indices=welm_kv_mirror_last_q_indices,
            welm_kv_mirror_active_batch_indices=welm_kv_mirror_active_batch_indices,
            welm_kv_mirror_output_size=welm_kv_mirror_output_size,
            router_replay_topk_ids=self.router_replay_topk_ids,
            router_replay_mask=self.router_replay_mask,
            multimodal_inputs=self.multimodal_inputs,
            encoder_cached=self.encoder_cached,
            encoder_lens=self.encoder_lens,
            encoder_lens_cpu=self.encoder_lens_cpu,
            encoder_out_cache_loc=self.encoder_out_cache_loc,
            chunked_req=self.chunked_req,
            lora_ids=[req.lora_id for req in self.reqs],
            sampling_info=self.sampling_info,
            input_embeds=self.input_embeds,
            replace_embeds=self.replace_embeds,
            replace_positions=self.replace_positions,
            ne_token_table=self.ne_token_table,
            token_type_ids=self.token_type_ids,
            spec_algorithm=self.spec_algorithm,
            spec_info=self.spec_info,
            hicache_consumer_index=self.hicache_consumer_index,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self.return_hidden_states
                else (
                    getattr(
                        self.spec_info, "capture_hidden_mode", CaptureHiddenMode.NULL
                    )
                    if self.spec_info
                    else CaptureHiddenMode.NULL
                )
            ),
            extend_input_logprob_token_ids=self.extend_input_logprob_token_ids,
            is_prefill_only=self.is_prefill_only,
            multi_item_delimiter_indices=self.multi_item_delimiter_indices,
            dimensions=self.dimensions,
            return_pooled_hidden_states=self.return_pooled_hidden_states,
            dllm_block_offsets=[req.dllm_block_offset for req in self.reqs],
            dllm_config=self.dllm_config,
            reqs=self.reqs,
            has_grammar=self.has_grammar,
            dp_cooperation_info=self.dp_cooperation_info,
            prefill_stats=self.prefill_stats,
            oe_context=self.oe_context,
            scale_seq_factor=scale,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
            forward_iter=self.forward_iter,
            welm_deferred_prefill=self.welm_deferred_prefill,
        )

    def _get_welm_kv_mirror_last_q_indices(
        self,
    ) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[int]]:
        if not self.extend_lens:
            return None, None, None

        last_q_indices: List[int] = []
        active_batch_indices: List[int] = []
        offset = 0
        for batch_idx, (req, extend_len) in enumerate(zip(self.reqs, self.extend_lens)):
            if self.forward_mode.is_draft_extend():
                if extend_len > 0:
                    last_q_indices.append(offset + extend_len - 1)
                    active_batch_indices.append(batch_idx)
                offset += extend_len
                continue

            committed_seq_len = len(req.origin_input_ids) + max(
                len(req.output_ids) - 1, 0
            )
            if extend_len > 0 and len(req.fill_ids) >= committed_seq_len:
                last_q_indices.append(offset + extend_len - 1)
                active_batch_indices.append(batch_idx)
            offset += extend_len

        return last_q_indices, active_batch_indices, len(self.extend_lens)

    def copy(self):
        # Only contain fields that will be used by process_batch_result.
        # Shallow-copy the reqs list so that in-place mutations (filter_batch,
        # merge_batch) on the original don't corrupt this snapshot.
        return ScheduleBatch(
            reqs=self.reqs[:],
            req_to_token_pool=self.req_to_token_pool,
            req_pool_indices=self.req_pool_indices,
            model_config=self.model_config,
            forward_mode=self.forward_mode,
            out_cache_loc=self.out_cache_loc,
            return_logprob=self.return_logprob,
            decoding_reqs=self.decoding_reqs,
            spec_algorithm=self.spec_algorithm,
            global_num_tokens=self.global_num_tokens,
            global_num_tokens_for_logprob=self.global_num_tokens_for_logprob,
            can_run_dp_cuda_graph=self.can_run_dp_cuda_graph,
            all_extend_in_batch=self.all_extend_in_batch,
            is_extend_in_batch=self.is_extend_in_batch,
            has_cache_hit_extend_in_batch=self.has_cache_hit_extend_in_batch,
            is_prefill_only=self.is_prefill_only,
            router_replay_topk_ids=self.router_replay_topk_ids,
            router_replay_mask=self.router_replay_mask,
            attn_cp_prefill_split_specs=self._current_attn_cp_prefill_split_specs(),
            seq_lens_cpu=self.seq_lens_cpu,
            enable_overlap=self.enable_overlap,
            mamba_track_indices=self.mamba_track_indices,
            mamba_track_mask=self.mamba_track_mask,
            mamba_track_seqlens=self.mamba_track_seqlens,
            dp_cooperation_info=self.dp_cooperation_info,
            prefill_stats=self.prefill_stats,
            fpm_start_time=self.fpm_start_time,
            forward_iter=self.forward_iter,
            welm_deferred_prefill=self.welm_deferred_prefill,
            welm_deferred_prefill_final_mask=(
                self.welm_deferred_prefill_final_mask
            ),
        )

    def _clear_attn_cp_prefill_split_specs(self, *, clear_reqs: bool) -> None:
        self.attn_cp_prefill_split_specs = None
        if clear_reqs:
            for req in self.reqs:
                req.attn_cp_prefill_split_spec = None

    def _current_attn_cp_prefill_split_specs(
        self,
    ) -> Optional[Tuple[Optional[CPPrefillSplitSpec], ...]]:
        specs = tuple(req.attn_cp_prefill_split_spec for req in self.reqs)
        return specs if any(spec is not None for spec in specs) else None

    def _refresh_attn_cp_prefill_split_specs(self) -> None:
        self.attn_cp_prefill_split_specs = (
            self._current_attn_cp_prefill_split_specs()
        )

    def maybe_evict_swa(self):
        if self.tree_cache.supports_swa():
            sliding_window_size = self.tree_cache.sliding_window_size
            server_args = get_global_server_args()

            release_leaf_lock = (
                envs.SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW.get()
                and hasattr(self.tree_cache, "dec_swa_lock_only")
            )

            # Eviction_interval: trade-off between SWA token waste and eviction overhead
            page_size = self.tree_cache.page_size
            eviction_interval = max(
                page_size,
                int(
                    sliding_window_size
                    * envs.SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER.get()
                ),
            )
            eviction_interval = (eviction_interval // page_size) * page_size
            for idx, req in enumerate(self.reqs):
                if self.forward_mode.is_decode():
                    # We set evict_swa condition here with two reasons:
                    # 1. In overlap scheduler, we cannot evict swa when req.decode_batch_idx == 0 since the prev extend batch is still running.
                    # 2. Evict swa every eviction_interval tokens to reduce the overhead.
                    if req.decode_batch_idx % eviction_interval == 1:
                        self._evict_swa(req, req.seqlen - 1)

                    # Once the decode position has moved past the sliding window,
                    # the SWA portion of the prefill-time tree lock is no longer
                    # needed by this request. Convert it from protected to
                    # evictable so SWA LRU can reclaim it under pressure.
                    if (
                        release_leaf_lock
                        and not req.swa_prefix_lock_released
                        and req.swa_uuid_for_lock is not None
                        and req.last_node is not None
                        and req.decode_batch_idx >= sliding_window_size
                    ):
                        self.tree_cache.dec_swa_lock_only(
                            req.last_node, req.swa_uuid_for_lock
                        )
                        req.swa_prefix_lock_released = True
                elif self.forward_mode.is_extend() and self.tree_cache.is_chunk_cache():
                    pre_len = self.prefix_lens[idx]
                    if self.enable_overlap:
                        # In chunked prefill case, when the second extend batch is scheduling, the first extend batch is still running, so we cannot evict swa tokens
                        if req.extend_batch_idx < 2:
                            continue
                        else:
                            pre_len = (
                                pre_len - server_args.chunked_prefill_size
                                if server_args.chunked_prefill_size > 0
                                else pre_len
                            )
                            self._evict_swa(req, pre_len)
                    else:
                        self._evict_swa(req, pre_len)

    def _evict_swa(self, req: Req, pre_len: int):
        assert self.tree_cache.supports_swa(), "prefix cache must support swa"
        if hasattr(self.tree_cache, "evict_swa"):
            self.tree_cache.evict_swa(req, pre_len)
            return
        sliding_window_size = self.tree_cache.sliding_window_size

        # For swa radix cache, we need to evict the tokens that are not in the tree cache and also not in the sliding window
        assert (
            req.cache_protected_len % self.tree_cache.page_size == 0
        ), "cache_protected_len must be page aligned"
        req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, req.cache_protected_len)

        # Subtract an extra page_size so the eviction frontier never reaches the
        # radix tree insert boundary (page_floor(seq_len)). This keeps at least one
        # page of non-evicted SWA KV for the tree to store as a non-tombstone node,
        # preserving cache reuse in multi-turn scenarios. Without this, leaf nodes
        # may become tombstoned, causing SWA memory leak.
        # See also: _insert_helper case 3 in swa_radix_cache.py (defensive counterpart).
        new_swa_evicted_seqlen = max(
            req.swa_evicted_seqlen,
            pre_len - sliding_window_size - self.tree_cache.page_size,
        )

        if self.tree_cache.page_size > 1:
            new_swa_evicted_seqlen = (
                new_swa_evicted_seqlen // self.tree_cache.page_size
            ) * self.tree_cache.page_size

        if new_swa_evicted_seqlen > req.swa_evicted_seqlen:
            free_slots = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, req.swa_evicted_seqlen : new_swa_evicted_seqlen
            ]
            self.token_to_kv_pool_allocator.free_swa(free_slots)
            req.swa_evicted_seqlen = new_swa_evicted_seqlen

    def __str__(self):
        return (
            f"ScheduleBatch(forward_mode={self.forward_mode.name if self.forward_mode else 'None'}, "
            f"#req={(len(self.reqs))})"
        )


@dataclasses.dataclass
class ModelWorkerBatch:
    # The forward mode
    forward_mode: ForwardMode
    # The input ids
    input_ids: torch.Tensor
    # The indices of requests in the req_to_token_pool
    req_pool_indices: torch.Tensor
    # The sequence length
    seq_lens: torch.Tensor
    # The indices of output tokens in the token_to_kv_pool_allocator
    out_cache_loc: torch.Tensor
    # The sequence length tensor on CPU
    seq_lens_cpu: Optional[torch.Tensor]
    seq_lens_sum: int

    # For logprob
    return_logprob: bool
    top_logprobs_nums: Optional[List[int]]
    token_ids_logprobs: Optional[List[List[int]]]

    # For DP attention
    global_num_tokens: Optional[List[int]]
    global_num_tokens_for_logprob: Optional[List[int]]
    global_num_reqs: Optional[List[int]]
    global_forward_modes: Optional[List[int]]
    welm_kv_mirror_contract_flags: Optional[List[bool]]
    welm_deferred_prefill_flags: Optional[List[bool]]
    welm_mtp_global_prefill_num_tokens: Optional[List[int]]
    global_has_non_greedy_sampling: bool
    global_needs_top_p_sampling: bool
    is_extend_in_batch: bool
    all_extend_in_batch: bool
    can_run_dp_cuda_graph: bool
    tbo_split_seq_index: Optional[int]
    global_forward_mode: Optional[ForwardMode]

    # For extend
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[List[int]]
    extend_prefix_lens: Optional[List[int]]
    extend_logprob_start_lens: Optional[List[int]]
    attn_cp_prefill_split_specs: Optional[
        Tuple[Optional[CPPrefillSplitSpec], ...]
    ]
    welm_kv_mirror_last_q_indices: Optional[List[int]]
    welm_kv_mirror_active_batch_indices: Optional[List[int]]
    welm_kv_mirror_output_size: Optional[int]
    extend_input_logprob_token_ids: Optional[torch.Tensor]
    router_replay_topk_ids: Optional[torch.Tensor]
    router_replay_mask: Optional[torch.Tensor]
    # For multimodal
    multimodal_inputs: Optional[List[MultimodalInputs]]

    # For encoder-decoder
    encoder_cached: Optional[List[bool]]
    encoder_lens: Optional[torch.Tensor]
    encoder_lens_cpu: Optional[List[int]]
    encoder_out_cache_loc: Optional[torch.Tensor]

    # For LoRA
    lora_ids: Optional[List[str]]

    # Sampling info
    sampling_info: SamplingBatchInfo

    # Uniform model-execution marker; request-aligned finality stays in scheduler.
    welm_deferred_prefill: bool = False

    # The original sequence lengths, Qwen-1M related
    orig_seq_lens: Optional[torch.Tensor] = None

    # The input Embeds
    input_embeds: Optional[torch.Tensor] = None
    replace_embeds: Optional[torch.Tensor] = None
    replace_positions: Optional[torch.Tensor] = None

    # token table for ngram embedding
    ne_token_table: Optional[torch.Tensor] = None

    # For corss-encoder model
    token_type_ids: Optional[torch.Tensor] = None

    # Speculative decoding
    spec_algorithm: SpeculativeAlgorithm = None

    spec_info: Optional[SpecInput] = None

    # If set, the output of the batch contains the hidden states of the run.
    capture_hidden_mode: CaptureHiddenMode = None
    hicache_consumer_index: int = -1

    # For matryoshka embeddings
    dimensions: Optional[list[int]] = None

    # Whether to return pooled hidden states (pre-head transformer output)
    return_pooled_hidden_states: bool = False

    # Whether this batch is prefill-only (no token generation needed)
    is_prefill_only: bool = False

    # Pre-computed delimiter indices for multi-item scoring (CPU tensors, one per request)
    multi_item_delimiter_indices: Optional[List[torch.Tensor]] = None

    # Diffusion LLM
    dllm_block_offsets: Optional[List[int]] = None
    dllm_config: Optional[DllmConfig] = None

    # For constrained decoding
    # FIXME(lsyin): remove this after fully overlap grammar
    reqs: Optional[List[Req]] = None
    chunked_req: Optional[Req] = None
    has_grammar: bool = False

    # Over Encoding
    oe_context: Optional[OverEncodingContext] = None
    scale_seq_factor: int = 1

    # Metrics
    dp_cooperation_info: Optional[DPCooperationInfo] = None
    prefill_stats: Optional[PrefillStats] = None

    # For hidden states before normal
    return_hidden_states_before_norm: bool = False

    # For mamba state tracking
    mamba_track_indices: Optional[torch.Tensor] = None  # shape: [b], int64
    mamba_track_mask: Optional[torch.Tensor] = None  # shape: [b], bool
    mamba_track_seqlens: Optional[torch.Tensor] = None  # shape: [b], int64

    # Scheduler-wide forward order, shared by every TP/CP rank.
    forward_iter: Optional[int] = None
