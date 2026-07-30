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
"""Run the model with cuda graph and torch.compile."""

from __future__ import annotations

import bisect
import contextlib
import gc
import inspect
import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import tqdm
from torch.profiler import ProfilerActivity, profile

from sglang.srt.batch_overlap.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    graph_capture,
    set_pdmux_status,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_cp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.utils import (
    get_deepep_mode,
    get_moe_a2a_backend,
    should_record_nolora_graph,
)
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.managers.schedule_batch import (
    HashInputIdsBuffer,
    OverEncodingContext,
)
from sglang.srt.model_executor.forward_batch_context import set_current_forward_batch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    NgramEmbeddingInfo,
    PPProxyTensors,
    compute_local_num_token_non_padded,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.models.welm_perf_opt import (
    fill_welm_oe_hash_inputs,
    get_welm_oe_hash_config,
    should_use_welm_oe_hash_kernel,
)
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx, get_stream_groups
from sglang.srt.utils import (
    empty_context,
    get_available_gpu_memory,
    get_bool_env_var,
    is_hip,
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_compile
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

_is_hip = is_hip()

if not _is_hip:
    from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
        BreakableCUDAGraph,
        BreakableCUDAGraphCapture,
        eager_on_graph,
    )

logger = logging.getLogger(__name__)

WELM_KV_MIRROR_PP_KEY_PREFIX = "welm_kv_mirror"


def _is_welm_v4_model_config(model_config) -> bool:
    for config in (
        getattr(model_config, "hf_config", None),
        getattr(model_config, "hf_text_config", None),
    ):
        if config is None:
            continue
        if getattr(config, "model_type", None) == "welmv4_moe":
            return True
        archs = getattr(config, "architectures", None) or []
        if any(str(arch).startswith("WeLMV4") for arch in archs):
            return True
    return False


def _welm_kv_mirror_packed_len(numel: int) -> int:
    try:
        tp_size = get_attention_tp_size()
    except AssertionError:
        tp_size = 1
    if tp_size <= 1:
        return numel

    pad_len = 1
    while (numel + pad_len) % tp_size == 0:
        pad_len += 1
    return numel + pad_len


if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_has_foreach_copy = hasattr(torch, "_foreach_copy_")


def _grouped_foreach_copy_(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> None:
    """Call torch._foreach_copy_ grouped by (dst_dtype, src_dtype) pairs."""

    def foreach_copy(dsts: List[torch.Tensor], srcs: List[torch.Tensor]) -> None:
        if _has_foreach_copy:
            torch._foreach_copy_(dsts, srcs)
        else:
            for dst, src in zip(dsts, srcs):
                dst.copy_(src)

    groups: Dict[Tuple[torch.dtype, torch.dtype], Tuple[List, List]] = {}
    for dst, src in zip(dsts, srcs):
        key = (dst.dtype, src.dtype)
        if key not in groups:
            groups[key] = ([], [])
        groups[key][0].append(dst)
        groups[key][1].append(src)
    for group_dsts, group_srcs in groups.values():
        foreach_copy(group_dsts, group_srcs)


@dataclass
class DecodeInputBuffers(ForwardInputBuffers):

    input_ids: torch.Tensor
    input_embeds: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    out_cache_loc: torch.Tensor
    out_cache_loc_swa: Optional[torch.Tensor]
    positions: torch.Tensor
    mrope_positions: torch.Tensor
    num_token_non_padded: torch.Tensor
    custom_mask: torch.Tensor
    next_token_logits_buffer: torch.Tensor
    mamba_track_indices: Optional[torch.Tensor]
    mamba_track_mask: Optional[torch.Tensor]
    global_num_tokens_gpu: torch.Tensor
    global_num_tokens_for_logprob_gpu: torch.Tensor
    encoder_lens: Optional[torch.Tensor]
    pp_proxy_tensors: Optional[Dict[str, torch.Tensor]]
    ngram_embedding_info: Optional["NgramEmbeddingInfo"]
    welm_oe_decode_hashed_inputs: Optional[torch.Tensor]
    # Persistent output buffer for the mk fused decode embedding-GEMM kernel
    # under cuda graph capture. Sized to mk's max supported batch (32) so the
    # buckets bs ∈ {1,2,...,32} can each take a slice that's the EXACT same
    # data_ptr() the captured graph baked in. ``None`` when the runner won't
    # use the prepared mk path (env off, mk missing, OE not configured, etc.).
    welm_oe_fused_decode_output: Optional[torch.Tensor]
    router_replay_topk_ids: torch.Tensor
    router_replay_mask: torch.Tensor
    router_replay_local_topk_ids: torch.Tensor
    router_replay_local_mask: torch.Tensor
    scale_seq_factor: int = 1

    @classmethod
    def create(
        cls,
        *,
        device: torch.device,
        max_bs: int,
        max_num_token: int,
        hidden_size: int,
        vocab_size: int,
        dtype: torch.dtype,
        dp_size: int,
        pp_size: int,
        is_encoder_decoder: bool,
        require_mlp_tp_gather: bool,
        seq_len_fill_value: int,
        encoder_len_fill_value: int,
        num_tokens_per_bs: int,
        cache_loc_dtype: torch.dtype,
        enable_mamba_track: bool,
        welm_oe_decode_hash_num_branches: int,
        scale_seq_factor: int,
        router_replay_num_layers: int = 0,
        router_replay_top_k: int = 0,
        kv_mirror_layers: Optional[List[int]] = None,
        kv_mirror_tensor_size: Optional[int] = None,
        ne_token_table: Optional[torch.Tensor] = None,
        is_hybrid_swa: bool = False,
        hc_hidden_size: Optional[int] = None,
        pp_residual_dtype: Optional[torch.dtype] = None,
        welm_oe_fused_decode_output_max_bs: int = 0,
        welm_oe_fused_decode_output_hidden_size: int = 0,
    ) -> "DecodeInputBuffers":
        with torch.device(device):
            welm_oe_decode_hashed_inputs = (
                torch.zeros(
                    (welm_oe_decode_hash_num_branches, max_num_token),
                    dtype=torch.int64,
                )
                if welm_oe_decode_hash_num_branches > 0
                else None
            )
            # Persistent output for the mk fused decode + all-reduce kernel
            # under cuda graph capture. Cap rows at mk's max batch (32);
            # buckets > 32 use the eager captured fused path which doesn't
            # touch this buffer. dtype is bf16 because mk requires it.
            welm_oe_fused_decode_output = (
                torch.zeros(
                    (
                        welm_oe_fused_decode_output_max_bs,
                        welm_oe_fused_decode_output_hidden_size,
                    ),
                    dtype=torch.bfloat16,
                )
                if welm_oe_fused_decode_output_max_bs > 0
                and welm_oe_fused_decode_output_hidden_size > 0
                else None
            )
            input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
            input_embeds = torch.zeros((max_num_token, hidden_size), dtype=dtype)
            req_pool_indices = torch.zeros((max_bs,), dtype=torch.int64)
            seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int32)
            out_cache_loc = torch.zeros((max_num_token,), dtype=cache_loc_dtype)
            out_cache_loc_swa = (
                torch.zeros((max_num_token,), dtype=torch.int32)
                if is_hybrid_swa
                else None
            )
            positions = torch.zeros((max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, max_num_token), dtype=torch.int64)
            num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            custom_mask = torch.ones(
                (max_bs * seq_len_fill_value + max_num_token) * num_tokens_per_bs,
                dtype=torch.bool,
            )
            next_token_logits_buffer = torch.zeros(
                (max_num_token, vocab_size),
                dtype=torch.float,
            )
            mamba_track_indices = (
                torch.zeros((max_bs,), dtype=torch.int64)
                if enable_mamba_track
                else None
            )
            mamba_track_mask = (
                torch.zeros((max_bs,), dtype=torch.bool) if enable_mamba_track else None
            )

            if pp_size > 1:
                # mHC (e.g. DSV4) flattens residual into hidden_states (size = hc_hidden_size).
                is_mhc = hc_hidden_size is not None
                hs = hc_hidden_size if is_mhc else hidden_size
                pp_proxy_tensors = {
                    "hidden_states": torch.zeros((max_bs, hs), dtype=dtype),
                }
                if not is_mhc:
                    pp_proxy_tensors["residual"] = torch.zeros(
                        (max_bs, hidden_size), dtype=pp_residual_dtype or dtype
                    )
                if kv_mirror_layers and kv_mirror_tensor_size:
                    for layer_idx in kv_mirror_layers:
                        pp_proxy_tensors[
                            f"{WELM_KV_MIRROR_PP_KEY_PREFIX}.{layer_idx}.k"
                        ] = torch.zeros((max_bs, kv_mirror_tensor_size), dtype=dtype)
                        pp_proxy_tensors[
                            f"{WELM_KV_MIRROR_PP_KEY_PREFIX}.{layer_idx}.v"
                        ] = torch.zeros((max_bs, kv_mirror_tensor_size), dtype=dtype)
            else:
                pp_proxy_tensors = None

            if is_encoder_decoder:
                encoder_lens = torch.full(
                    (max_bs,), encoder_len_fill_value, dtype=torch.int32
                )
            else:
                encoder_lens = None

            if require_mlp_tp_gather:
                global_num_tokens_gpu = torch.zeros((dp_size,), dtype=torch.int32)
                global_num_tokens_for_logprob_gpu = torch.zeros(
                    (dp_size,), dtype=torch.int32
                )
            else:
                global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                global_num_tokens_for_logprob_gpu = torch.zeros((1,), dtype=torch.int32)

            router_shape = (
                max_num_token,
                router_replay_num_layers,
                router_replay_top_k,
            )
            router_replay_topk_ids = torch.zeros(router_shape, dtype=torch.int32)
            router_replay_mask = torch.zeros((max_num_token,), dtype=torch.bool)
            router_replay_local_topk_ids = torch.zeros(router_shape, dtype=torch.int32)
            router_replay_local_mask = torch.zeros((max_num_token,), dtype=torch.bool)

            ngram_embedding_info = (
                NgramEmbeddingInfo(
                    token_table=ne_token_table,
                    column_starts=torch.zeros([max_bs], dtype=torch.int32),
                    req_lens=torch.ones([max_bs], dtype=torch.int32),
                    out_column_starts=torch.zeros([max_bs], dtype=torch.int32),
                    out_req_lens=torch.ones([max_bs], dtype=torch.int32),
                )
                if ne_token_table is not None
                else None
            )

        # Keep seq_lens_cpu as a true CPU tensor, like the old implementation.
        seq_lens_cpu = torch.full(
            (max_bs,),
            seq_len_fill_value,
            dtype=torch.int32,
            device="cpu",
        )

        return cls(
            input_ids=input_ids,
            input_embeds=input_embeds,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            out_cache_loc_swa=out_cache_loc_swa,
            positions=positions,
            mrope_positions=mrope_positions,
            num_token_non_padded=num_token_non_padded,
            custom_mask=custom_mask,
            next_token_logits_buffer=next_token_logits_buffer,
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            encoder_lens=encoder_lens,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
            pp_proxy_tensors=pp_proxy_tensors,
            ngram_embedding_info=ngram_embedding_info,
            welm_oe_decode_hashed_inputs=welm_oe_decode_hashed_inputs,
            welm_oe_fused_decode_output=welm_oe_fused_decode_output,
            router_replay_topk_ids=router_replay_topk_ids,
            router_replay_mask=router_replay_mask,
            router_replay_local_topk_ids=router_replay_local_topk_ids,
            router_replay_local_mask=router_replay_local_mask,
            scale_seq_factor=scale_seq_factor,
        )

    def populate_from_forward_batch(
        self,
        *,
        forward_batch: ForwardBatch,
        raw_bs: int,
        raw_num_token: int,
        bs: int,
        seq_len_padding_value: int,
        require_gathered_buffer: bool,
        num_tokens_per_bs: int,
        nsa_enable_prefill_cp: bool,
        enable_num_token_non_padded_flag: bool,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        if bs != raw_bs:
            self.seq_lens.fill_(seq_len_padding_value)
            self.out_cache_loc.zero_()
            # Padded SWA indices left over from a previous replay would point
            # into real SWA slots, so set_kv_buffer on padded tokens would
            # corrupt active requests' KV. Zero the whole buffer so padded
            # positions map to the sentinel slot (matches piecewise runner).
            if self.out_cache_loc_swa is not None:
                self.out_cache_loc_swa.zero_()
            if self.mamba_track_indices is not None:
                self.mamba_track_indices.zero_()
            if self.mamba_track_mask is not None:
                self.mamba_track_mask.fill_(False)

        # Build batched copy lists for all GPU tensors.
        is_scale_seq = self.scale_seq_factor > 1
        raw_model_input = raw_bs if is_scale_seq else raw_num_token
        dsts = [
            self.input_ids[:raw_model_input],
            self.req_pool_indices[:raw_bs],
            self.seq_lens[:raw_bs],
            self.out_cache_loc[:raw_num_token],
            self.positions[:raw_num_token],
        ]
        srcs = [
            forward_batch.input_ids,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.out_cache_loc,
            forward_batch.positions,
        ]

        if self.ngram_embedding_info is not None:
            ngram_embedding_info = forward_batch.ngram_embedding_info
            self.ngram_embedding_info.column_starts[:raw_bs].copy_(
                ngram_embedding_info.column_starts
            )
            self.ngram_embedding_info.req_lens[:raw_bs].copy_(
                ngram_embedding_info.req_lens
            )

        if (
            self.mamba_track_indices is not None
            and forward_batch.mamba_track_indices is not None
        ):
            dsts.append(self.mamba_track_indices[:raw_bs])
            srcs.append(forward_batch.mamba_track_indices)
        if (
            self.mamba_track_mask is not None
            and forward_batch.mamba_track_mask is not None
        ):
            dsts.append(self.mamba_track_mask[:raw_bs])
            srcs.append(forward_batch.mamba_track_mask)

        if self.encoder_lens is not None and forward_batch.encoder_lens is not None:
            dsts.append(self.encoder_lens[:raw_bs])
            srcs.append(forward_batch.encoder_lens)

        if forward_batch.mrope_positions is not None:
            dsts.append(self.mrope_positions[:, :raw_num_token])
            srcs.append(forward_batch.mrope_positions)

        if require_gathered_buffer:
            self.global_num_tokens_gpu.fill_(bs * num_tokens_per_bs)
            self.global_num_tokens_for_logprob_gpu.fill_(bs * num_tokens_per_bs)

        if enable_num_token_non_padded_flag:
            if require_gathered_buffer and not nsa_enable_prefill_cp:
                num_tokens_per_dp = bs * num_tokens_per_bs
                local = compute_local_num_token_non_padded(
                    global_num_token_non_padded=forward_batch.num_token_non_padded,
                    num_tokens_per_dp=num_tokens_per_dp,
                )
                dsts.append(self.num_token_non_padded)
                srcs.append(local)
            else:
                dsts.append(self.num_token_non_padded)
                srcs.append(forward_batch.num_token_non_padded)

        # Pipeline-parallel proxy tensors.
        if pp_proxy_tensors is not None and self.pp_proxy_tensors is not None:
            for buf in self.pp_proxy_tensors.values():
                buf.zero_()
            for key, buf in self.pp_proxy_tensors.items():
                src = pp_proxy_tensors.tensors.get(key, None)
                if src is None:
                    continue
                shape = pp_proxy_tensors.tensors.get(f"{key}.shape")
                if (
                    shape is not None
                    and key.startswith(f"{WELM_KV_MIRROR_PP_KEY_PREFIX}.")
                ):
                    dim = _welm_kv_mirror_packed_len(math.prod(shape))
                    src = src.view(-1)[:dim]
                    dsts.append(buf.view(-1)[:dim])
                    srcs.append(src)
                    continue
                dim = src.shape[0]
                dsts.append(buf[:dim])
                srcs.append(src)

        # SWA cache location (int32, separate from the int64 batch above).
        if (
            self.out_cache_loc_swa is not None
            and forward_batch.out_cache_loc_swa is not None
        ):
            dsts.append(self.out_cache_loc_swa[:raw_num_token])
            srcs.append(forward_batch.out_cache_loc_swa[:raw_num_token])

        # Batch all GPU copies, grouped by dtype pair.
        _grouped_foreach_copy_(dsts, srcs)
        static_num_token = bs * num_tokens_per_bs
        if require_gathered_buffer:
            copy_router_replay_to_cuda_graph_buffers(
                dst_topk_ids=self.router_replay_local_topk_ids,
                dst_mask=self.router_replay_local_mask,
                src_topk_ids=forward_batch.router_replay_topk_ids,
                src_mask=forward_batch.router_replay_mask,
                raw_num_token=raw_num_token,
                static_num_token=static_num_token,
            )

            # NOTE(replay+dp-attn): use dp_gather_REPLICATE rather than
            # dp_gather_partial here. router_replay_topk_ids/mask are
            # replicated across attn-TP ranks within a DP rank, so the all-
            # reduce that backs dp_gather would otherwise SUM identical TP
            # copies (e.g. expert id 510 + 510 = 1020) and produce
            # out-of-range expert ids that crash `scores.gather(1,
            # forced_ids)` in _gather_router_replay_weights.  Replicate semantics
            # contribute only from attn_tp_rank==0 so the all-reduce returns a
            # single copy.
            from sglang.srt.layers.dp_attention import dp_gather_replicate

            global_static_num_token = (
                static_num_token * self.global_num_tokens_gpu.numel()
            )
            old_global_num_tokens_gpu = forward_batch.global_num_tokens_gpu
            old_dp_padding_mode = forward_batch.dp_padding_mode
            old_dp_local_start_pos = forward_batch.dp_local_start_pos
            old_dp_local_num_tokens = forward_batch.dp_local_num_tokens
            try:
                forward_batch.global_num_tokens_gpu = self.global_num_tokens_gpu
                forward_batch.dp_padding_mode = (
                    DpPaddingMode.get_default_mode_in_cuda_graph()
                )
                forward_batch.dp_local_start_pos = None
                forward_batch.dp_local_num_tokens = None
                # Zero the slice BEFORE the all-reduce: dp_gather_replicate
                # sums across the TP group, so any stale non-zero data on
                # non-rank-0 attn_tp ranks would otherwise be summed in.
                self.router_replay_topk_ids[:global_static_num_token].zero_()
                dp_gather_replicate(
                    self.router_replay_topk_ids[:global_static_num_token],
                    self.router_replay_local_topk_ids[:static_num_token],
                    forward_batch,
                )
                global_mask_i32 = torch.zeros(
                    (global_static_num_token,),
                    dtype=torch.int32,
                    device=self.router_replay_mask.device,
                )
                dp_gather_replicate(
                    global_mask_i32,
                    self.router_replay_local_mask[:static_num_token].to(torch.int32),
                    forward_batch,
                )
                self.router_replay_mask[:global_static_num_token].copy_(
                    global_mask_i32.to(torch.bool)
                )
            finally:
                forward_batch.global_num_tokens_gpu = old_global_num_tokens_gpu
                forward_batch.dp_padding_mode = old_dp_padding_mode
                forward_batch.dp_local_start_pos = old_dp_local_start_pos
                forward_batch.dp_local_num_tokens = old_dp_local_num_tokens
            if self.router_replay_mask.shape[0] > global_static_num_token:
                self.router_replay_mask[global_static_num_token:].fill_(False)
        else:
            copy_router_replay_to_cuda_graph_buffers(
                dst_topk_ids=self.router_replay_topk_ids,
                dst_mask=self.router_replay_mask,
                src_topk_ids=forward_batch.router_replay_topk_ids,
                src_mask=forward_batch.router_replay_mask,
                raw_num_token=raw_num_token,
                static_num_token=static_num_token,
            )

        # CPU tensor copy (cannot be batched with GPU tensors).
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(seq_len_padding_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)


def copy_router_replay_to_cuda_graph_buffers(
    *,
    dst_topk_ids: torch.Tensor,
    dst_mask: torch.Tensor,
    src_topk_ids: Optional[torch.Tensor],
    src_mask: Optional[torch.Tensor],
    raw_num_token: int,
    static_num_token: int,
) -> None:
    if src_topk_ids is not None:
        dst_topk_ids[:raw_num_token].copy_(src_topk_ids)
    if src_mask is not None:
        dst_mask[:raw_num_token].copy_(src_mask)
    else:
        dst_mask[:raw_num_token].fill_(False)
    if static_num_token > raw_num_token:
        dst_mask[raw_num_token:static_num_token].fill_(False)


# Detect whether the current forward pass is in capture mode
is_capture_mode = False
# When capturing dual MoE backends, tracks which variant is being captured.
# None = not dual, "lora" = capturing lora variant, "nolora" = capturing nolora variant.
_capture_lora_variant: Optional[str] = None


def get_is_capture_mode():
    return is_capture_mode


def compile_in_capture_mode(func):
    if get_is_capture_mode():
        return torch.compile(func)
    return func


def get_capture_lora_variant() -> Optional[str]:
    """Return the lora variant being captured, or None if not in dual capture."""
    return _capture_lora_variant


def _set_capture_lora_variant(variant: Optional[str]):
    global _capture_lora_variant
    _capture_lora_variant = variant


@contextmanager
def model_capture_mode():
    global is_capture_mode
    is_capture_mode = True

    yield

    is_capture_mode = False


@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()


def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, MultiPlatformOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode=os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
                ),
                dynamic=_is_hip and get_bool_env_var("SGLANG_TORCH_DYNAMIC_SHAPE"),
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config():
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future

    # FIXME: tmp workaround
    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024

    monkey_patch_torch_compile()


def get_batch_sizes_to_capture(model_runner: ModelRunner, num_tokens_per_bs=1):
    server_args = model_runner.server_args
    capture_bs = server_args.cuda_graph_bs
    num_max_requests = model_runner.req_to_token_pool.size

    mul_base = 1
    if server_args.enable_two_batch_overlap:
        mul_base *= 2
        num_tokens_per_bs = 1  # tbo not test, set num_tokens_per_bs to 1

    if require_gathered_buffer(server_args):
        mul_base *= get_attention_tp_size()

    # Sharded-KV CP keeps each decode request as a full-Q request and only shards
    # KV residency. Its CUDA graph bucket should therefore stay aligned with the
    # original request batch size; otherwise a single-request decode is replayed
    # through a bs=2 graph when attn_cp_size=2, which can change WeLM MoE/OE
    # numerics around tie tokens.
    skip_cp_bucket_alignment = server_args.attn_cp_mode == "sharded-kv"
    if not skip_cp_bucket_alignment and mul_base % get_attention_cp_size() != 0:
        mul_base *= get_attention_cp_size()

    # pad `num_max_requests` to avoid being filtered out
    num_max_requests = (num_max_requests + mul_base - 1) // mul_base * mul_base
    if max(capture_bs) > num_max_requests:
        # In some cases (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs += [num_max_requests]

    # Model input token count = bs * num_tokens_per_bs; must be a multiple of attn_tp_size.
    capture_bs = [bs for bs in capture_bs if bs * num_tokens_per_bs % mul_base == 0]
    capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]
    capture_bs = list(sorted(set(capture_bs)))

    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs]
        if server_args.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


def _default_make_graph_key(
    bs, stream_idx=None, variant_label=None, seq_len_bucket=None
):
    """Build a graph dict key from batch size, stream index, and lora variant.

    Standalone function so it can be used by CudaGraphRunner.capture() even when
    called on subclasses (e.g. EAGLEDraftCudaGraphRunner) that don't inherit from
    CudaGraphRunner and thus lack the method.
    """
    key = bs if stream_idx is None else f"{stream_idx}_{bs}"
    if seq_len_bucket is not None:
        key = f"{key}_s{int(seq_len_bucket)}"
    if variant_label is not None:
        key = f"{variant_label}_{key}"
    return key


class CudaGraphRunner:
    """A CudaGraphRunner runs the forward pass of a model with cuda graph and torch.compile."""

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        attn_backend=None,
        speculative_num_steps: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        # Parse args
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.use_ngram_embedding = model_runner.use_ngram_embedding
        if self.use_ngram_embedding:
            hf_config = model_runner.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size
        self.enable_pdmux = model_runner.server_args.enable_pdmux

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()
        self.record_nolora_graph = should_record_nolora_graph()

        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm = self.dllm_config is not None
        self.attn_backend = attn_backend or model_runner.attn_backend
        self.speculative_num_steps = (
            model_runner.server_args.speculative_num_steps
            if speculative_num_steps is None
            else speculative_num_steps
        )
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
            if speculative_num_draft_tokens is None
            else speculative_num_draft_tokens
        )

        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = 1
        if model_runner.spec_algorithm.is_speculative():
            if self.model_runner.is_draft_worker:
                # DFLASH draft workers reuse this runner for TARGET_VERIFY mode.
                if not self.model_runner.spec_algorithm.is_dflash():
                    raise RuntimeError("This should not happen")
            self.capture_forward_mode = ForwardMode.TARGET_VERIFY
            self.num_tokens_per_bs = self.speculative_num_draft_tokens
        elif self.is_dllm:
            self.capture_forward_mode = ForwardMode.DLLM_EXTEND
            self.num_tokens_per_bs = self.dllm_config.block_size

        self.scale_seq_factor = (
            getattr(model_runner.model_config.hf_config, "scale_seq_times", 0) + 1
        )
        if self.scale_seq_factor > 1 and self.num_tokens_per_bs == 1:
            self.num_tokens_per_bs = self.scale_seq_factor
            self.capture_forward_mode = ForwardMode.TARGET_VERIFY

        # Batch sizes to capture
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(
            model_runner, self.num_tokens_per_bs
        )
        log_info_on_rank0(logger, f"Capture cuda graph bs {self.capture_bs}")
        if KTRANSFORMERS_AVAILABLE:
            KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        self.enable_seq_len_graph_buckets = (
            self.dllm_config is None
            and self.capture_forward_mode.is_decode()
            and getattr(self.attn_backend, "is_attn_cp_sharded_kv", False)
        )
        self.seq_len_fill_values_by_bs = self._build_seq_len_fill_value_buckets_by_bs()
        self.seq_len_fill_value = max(
            max(values) for values in self.seq_len_fill_values_by_bs.values()
        )
        if self.enable_seq_len_graph_buckets:
            self.attn_backend.cuda_graph_max_seq_len = self.seq_len_fill_value
            bucket_desc = ", ".join(
                f"{bs}:{values}"
                for bs, values in sorted(self.seq_len_fill_values_by_bs.items())
            )
            log_info_on_rank0(
                logger, f"AttnCP cuda graph seq buckets by bs {{{bucket_desc}}}"
            )
        elif getattr(self.attn_backend, "is_attn_cp_sharded_kv", False):
            self.attn_backend.cuda_graph_max_seq_len = self.seq_len_fill_value
        self.attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

        # Init PDMux if needed
        self.maybe_init_pdmux()

        # Non-zero encoder length ensures cross-attention kernels are captured in the graph.
        self.encoder_len_fill_value = (
            getattr(model_runner.model_config.hf_config, "max_source_positions", 0)
            if self.is_encoder_decoder
            else 0
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        if self.model_runner.server_args.enable_lora:
            # Phase 2 of LoRA CUDA graph init: dense LoRA batch metadata.
            # Phase 1 (MoE buffers) was handled earlier in ModelRunner via
            # lora_manager.init_cuda_graph_moe_buffers().
            self.model_runner.lora_manager.init_cuda_graph_batch_info(
                max_bs_in_cuda_graph=self.max_bs,
                num_tokens_per_bs=self.num_tokens_per_bs,
            )

        enable_mamba_track = (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and self.model_runner.spec_algorithm.is_none()
        )

        if self.require_gathered_buffer:
            assert self.require_mlp_tp_gather or self.require_attn_tp_gather

        hf_config = self.model_runner.model_config.hf_config
        kv_mirror_layers = []
        kv_mirror_tensor_size = None
        if self.pp_size > 1 and self.model_runner.server_args.enable_welm_kv_mirror_opt:
            num_hidden_layers = getattr(hf_config, "num_hidden_layers", 0)
            kv_mirror_layers = [
                int(layer_idx)
                for layer_idx in getattr(hf_config, "kv_mirror_layers", [])
                if 0 <= int(layer_idx) < num_hidden_layers
            ]
            if kv_mirror_layers:
                num_attention_heads = getattr(hf_config, "num_attention_heads")
                num_key_value_heads = getattr(hf_config, "num_key_value_heads")
                head_dim = getattr(
                    hf_config,
                    "head_dim",
                    self.model_runner.model_config.hidden_size // num_attention_heads,
                )
                kv_mirror_tensor_size = (
                    max(1, num_key_value_heads // self.attn_tp_size) * head_dim
                )

        self.welm_oe_decode_hash_config = None
        welm_oe_decode_hash_num_branches = 0
        # scale_seq captures as TARGET_VERIFY even without speculative decoding
        # because one input token expands into multiple KV writes. Its OE hash
        # inputs are still decode-style, bs-sized inputs.
        is_scale_seq_pseudo_tv = (
            self.scale_seq_factor > 1
            and self.capture_forward_mode.is_target_verify()
            and not self.model_runner.spec_algorithm.is_speculative()
        )
        if (
            self.model_runner.server_args.prepare_n_gram_inputs
            and (
                self.capture_forward_mode.is_decode()
                or (
                    self.capture_forward_mode.is_target_verify()
                    and self.model_runner.spec_algorithm.is_speculative()
                )
                or is_scale_seq_pseudo_tv
            )
            and should_use_welm_oe_hash_kernel(
                self.model_runner.model_config
            )
        ):
            oe_grams, oe_vocab_sizes = get_welm_oe_hash_config(
                self.model_runner.model_config
            )
            if oe_grams and len(oe_grams) == len(oe_vocab_sizes):
                self.welm_oe_decode_hash_config = (oe_grams, oe_vocab_sizes)
                welm_oe_decode_hash_num_branches = len(oe_vocab_sizes)

        # Eligibility for the mk graph-friendly fused decode path. Discovery
        # only runs the structural shape check (no ForwardBatch needed); the
        # per-bucket prepare happens after buffers are allocated. Gates that
        # short-circuit eligibility BEFORE we allocate the static output
        # buffer keep memory clean for non-WeLM models.
        self._welm_oe_fused_config: Optional[Any] = None
        self._welm_oe_prepared_handles: Dict[int, Any] = {}
        self._welm_oe_fused_eligible = False
        welm_oe_fused_max_bs = 0
        welm_oe_fused_hidden_size = 0
        _welm_eligibility_gates = {
            "welm_oe_decode_hash_config_set": self.welm_oe_decode_hash_config is not None,
            "num_tokens_per_bs_eq_1": self.num_tokens_per_bs == 1,
            "no_dp_attention": not self.model_runner.server_args.enable_dp_attention,
            "no_pdmux": not getattr(
                self.model_runner.server_args, "enable_pdmux", False
            ),
            "decode_capture_mode": bool(self.capture_forward_mode.is_decode()),
        }
        if all(_welm_eligibility_gates.values()):
            # Lazy import — keeps cuda_graph_runner.py independent of mk for
            # non-WeLM models even when this file is imported eagerly.
            from sglang.srt.models.welm_perf_opt import (
                discover_welm_oe_fused_decode_modules,
                should_use_welm_oe_fused_decode_gemm,
            )

            if should_use_welm_oe_fused_decode_gemm():
                try:
                    self._welm_oe_fused_config = (
                        discover_welm_oe_fused_decode_modules(
                            self.model_runner.model
                        )
                    )
                except Exception as exc:  # defensive: discovery failures fall back
                    logger.warning(
                        "WeLM OE fused-decode discovery failed: %r; staying on the "
                        "eager captured path.",
                        exc,
                    )
                    self._welm_oe_fused_config = None
                if self._welm_oe_fused_config is not None:
                    self._welm_oe_fused_eligible = True
                    welm_oe_fused_max_bs = 32
                    welm_oe_fused_hidden_size = (
                        self._welm_oe_fused_config.input_hidden_size
                    )
                    logger.info(
                        "WeLM OE fused-decode: eligibility passed; will "
                        "prepare per-bucket mk handles for cuda graph capture "
                        "(input_hidden_size=%d, hash_hidden_size=%d, "
                        "world_size=%d, partition_size=%d).",
                        self._welm_oe_fused_config.input_hidden_size,
                        self._welm_oe_fused_config.hash_hidden_size,
                        self._welm_oe_fused_config.world_size,
                        self._welm_oe_fused_config.embedding_partition_size,
                    )
                else:
                    logger.info(
                        "WeLM OE fused-decode: discover returned None "
                        "(unsupported topology); staying on the eager "
                        "captured path."
                    )
        else:
            failing = [k for k, v in _welm_eligibility_gates.items() if not v]
            if (
                # Only log when the env is set — non-WeLM deployments don't
                # need to see this.
                self.welm_oe_decode_hash_config is not None
            ):
                logger.info(
                    "WeLM OE fused-decode: cuda-graph prepared path skipped "
                    "(failing gates: %s).",
                    ", ".join(failing) or "none",
                )
        self.buffers: DecodeInputBuffers = DecodeInputBuffers.create(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            hidden_size=self.model_runner.model_config.hidden_size,
            vocab_size=self.model_runner.model_config.vocab_size,
            dtype=self.model_runner.model_config.dtype,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            is_encoder_decoder=self.is_encoder_decoder,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            seq_len_fill_value=self.seq_len_fill_value,
            encoder_len_fill_value=self.encoder_len_fill_value,
            num_tokens_per_bs=self.num_tokens_per_bs,
            cache_loc_dtype=self._cache_loc_dtype(),
            enable_mamba_track=enable_mamba_track,
            welm_oe_decode_hash_num_branches=welm_oe_decode_hash_num_branches,
            scale_seq_factor=self.scale_seq_factor,
            router_replay_num_layers=getattr(
                self.model_runner.model_config.hf_text_config,
                "num_hidden_layers",
                0,
            ),
            router_replay_top_k=getattr(
                self.model_runner.model_config.hf_text_config,
                "num_experts_per_tok",
                0,
            ),
            kv_mirror_layers=kv_mirror_layers,
            kv_mirror_tensor_size=kv_mirror_tensor_size,
            ne_token_table=(
                model_runner.token_table if self.use_ngram_embedding else None
            ),
            is_hybrid_swa=model_runner.is_hybrid_swa,
            hc_hidden_size=getattr(
                self.model_runner.model_config, "hc_hidden_size", None
            ),
            pp_residual_dtype=(
                torch.float32
                if _is_welm_v4_model_config(self.model_runner.model_config)
                else None
            ),
            welm_oe_fused_decode_output_max_bs=welm_oe_fused_max_bs,
            welm_oe_fused_decode_output_hidden_size=welm_oe_fused_hidden_size,
        )
        self.buffers.share_buffers()

        from sglang.srt.models.welm_v45_80a3_fused_pre_attn_config import (
            welm_v45_80a3_fused_pre_attn_enabled,
        )

        welm_qkv_prepared = 0
        if welm_v45_80a3_fused_pre_attn_enabled():
            from sglang.srt.models.welm_v45_80a3_fused_pre_attn import (
                prepare_welm_v45_80a3_fused_pre_attn_cuda_graphs,
            )

            welm_qkv_prepared = prepare_welm_v45_80a3_fused_pre_attn_cuda_graphs(
                self.model_runner,
                self.buffers,
                self.capture_bs,
                self.num_tokens_per_bs,
            )
        if welm_qkv_prepared:
            logger.info(
                "Prepared %d WeLM v4.5 80A3 fused pre-attention CUDA graph "
                "handles.",
                welm_qkv_prepared,
            )

        # Build per-bucket Prepared mk handles BEFORE capture. mk's prepare
        # does symm-mem rendezvous + workspace allocation; both must happen
        # outside ``torch.cuda.graph(...)`` capture.
        if (
            self._welm_oe_fused_eligible
            and self._welm_oe_fused_config is not None
            and self.buffers.welm_oe_fused_decode_output is not None
        ):
            from sglang.srt.models.welm_perf_opt import (
                prepare_welm_oe_fused_decode_handle,
            )

            for bs in self.capture_bs:
                if not (1 <= int(bs) <= 32):
                    continue
                static_input_ids = self.buffers.input_ids[: int(bs)]
                static_output = self.buffers.welm_oe_fused_decode_output[: int(bs)]
                handle = prepare_welm_oe_fused_decode_handle(
                    config=self._welm_oe_fused_config,
                    static_input_ids=static_input_ids,
                    static_output=static_output,
                    runtime_batch_size=int(bs),
                )
                if handle is not None:
                    self._welm_oe_prepared_handles[int(bs)] = handle
            if not self._welm_oe_prepared_handles:
                # Every bucket got rejected by mk (e.g. unsupported shape) —
                # disable eligibility so capture/replay don't try to dispatch
                # to a non-existent handle.
                self._welm_oe_fused_eligible = False

        self.tbo_plugin = TboCudaGraphRunnerPlugin()

        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _build_seq_len_fill_value_buckets_by_bs(self) -> Dict[int, List[int]]:
        if self.dllm_config is not None:
            fill_value = int(self.dllm_config.block_size)
            return {int(bs): [fill_value] for bs in self.capture_bs}

        fill_value = int(self.attn_backend.get_cuda_graph_seq_len_fill_value())
        if not self.enable_seq_len_graph_buckets:
            return {int(bs): [fill_value] for bs in self.capture_bs}

        max_context_len = int(
            getattr(self.attn_backend, "max_context_len", fill_value) or fill_value
        )
        max_seq_cap = max(1, min(fill_value, max_context_len))
        max_total_num_tokens = int(
            getattr(self.model_runner, "max_total_num_tokens", 0) or 0
        )
        default_small_bucket = int(
            getattr(self.model_runner.server_args, "max_prefill_tokens", None)
            or max_seq_cap
        )

        page_size = max(1, int(getattr(self.attn_backend, "page_size", 1) or 1))
        is_attn_cp_sharded_kv = bool(
            getattr(self.attn_backend, "is_attn_cp_sharded_kv", False)
        )
        full_context_fallback_bs = self._build_full_context_fallback_batch_sizes()
        skew_bucket_bs = self._build_skew_bucket_batch_sizes(
            full_context_fallback_bs
        )
        values: Dict[int, List[int]] = {}
        prev_bs = 0
        # AttnCP sharded-KV selects a CUDA graph by max(seq_lens), while KV
        # residency is constrained by sum(seq_lens). Long-context eval batches
        # can therefore be arbitrarily skewed. A sparse, geometrically spaced
        # batch frontier captures the full context length, guaranteeing a graph
        # for every supported batch and sequence length with at most roughly 2x
        # batch padding. Add the common 2:1-skew bucket only immediately after
        # each frontier point, where it avoids the largest padding jump without
        # adding one whole-model graph for every captured batch size.
        for bs in sorted(int(x) for x in self.capture_bs):
            bs_int = max(1, int(bs))
            min_real_bs = bs_int if self.disable_padding else prev_bs + 1
            skew_real_bs = max(1, (min_real_bs + 1) // 2)
            if max_total_num_tokens > 0 and not getattr(
                self.attn_backend, "cuda_graph_max_seq_len_is_explicit", False
            ):
                cap = min(max_seq_cap, max_total_num_tokens // max(1, min_real_bs))
                skew_cap = min(
                    max_seq_cap,
                    max_total_num_tokens // max(1, skew_real_bs),
                )
            else:
                cap = max_seq_cap
                skew_cap = max_seq_cap
            cap = self._align_seq_len_bucket(max(1, cap), page_size, max_seq_cap)
            skew_cap = self._align_seq_len_bucket(
                max(1, skew_cap), page_size, max_seq_cap
            )
            buckets = [min(default_small_bucket, cap)]
            while buckets[-1] * 2 < cap:
                buckets.append(buckets[-1] * 2)
            if buckets[-1] != cap:
                buckets.append(cap)
            if (
                is_attn_cp_sharded_kv
                and bs_int in skew_bucket_bs
                and skew_cap > cap
            ):
                buckets.append(skew_cap)
            if is_attn_cp_sharded_kv and bs_int in full_context_fallback_bs:
                buckets.append(max_seq_cap)
            values[bs_int] = sorted(
                {
                    self._align_seq_len_bucket(max(1, x), page_size, max_seq_cap)
                    for x in buckets
                }
            )
            prev_bs = bs_int
        return values

    def _build_full_context_fallback_batch_sizes(self) -> set[int]:
        """Choose a sparse batch frontier that covers all padded batch sizes.

        Each successive fallback graph is the largest captured batch no more
        than twice the previous one. Runtime selection may move to a larger
        batch graph when the closest graph lacks the requested sequence cap.
        """
        capture_bs = sorted({max(1, int(bs)) for bs in self.capture_bs})
        if not capture_bs:
            return set()
        if self.disable_padding:
            return set(capture_bs)

        fallback_bs = {capture_bs[0]}
        current = capture_bs[0]
        while current < capture_bs[-1]:
            within_bound = [bs for bs in capture_bs if current < bs <= 2 * current]
            if within_bound:
                next_bs = within_bound[-1]
            else:
                next_bs = next(bs for bs in capture_bs if bs > current)
            fallback_bs.add(next_bs)
            current = next_bs
        return fallback_bs

    def _build_skew_bucket_batch_sizes(
        self, full_context_fallback_bs: set[int]
    ) -> set[int]:
        """Place skew buckets only at costly full-frontier transitions.

        The first captured batch after a full-context frontier point represents
        the narrow range where falling through to the next frontier would add
        the most batch padding. Frontier points themselves already have the
        full-context graph, so they do not need a separate skew bucket.
        """
        if self.disable_padding:
            return set()

        capture_bs = sorted({max(1, int(bs)) for bs in self.capture_bs})
        skew_bucket_bs = {
            capture_bs[index + 1]
            for index, bs in enumerate(capture_bs[:-1])
            if bs in full_context_fallback_bs
        }
        return skew_bucket_bs - full_context_fallback_bs

    @staticmethod
    def _align_seq_len_bucket(
        seq_len: int, page_size: int, max_seq_cap: Optional[int] = None
    ) -> int:
        if page_size <= 1:
            return int(seq_len)
        aligned = max(
            1, ((int(seq_len) + page_size - 1) // page_size) * page_size
        )
        if max_seq_cap is not None:
            aligned = min(aligned, int(max_seq_cap))
        return aligned

    def _seq_len_fill_values_for_bs(self, bs: int) -> List[int]:
        return self.seq_len_fill_values_by_bs.get(int(bs), [self.seq_len_fill_value])

    def _select_seq_len_fill_value_for_bs(
        self, bs: int, max_seq_len: Optional[int] = None
    ) -> Optional[int]:
        buckets = self._seq_len_fill_values_for_bs(bs)
        if max_seq_len is None:
            return max(buckets)
        index = bisect.bisect_left(buckets, int(max_seq_len))
        if index >= len(buckets):
            return None
        return buckets[index]

    def _select_graph_shape(
        self, cuda_graph_bs: int, max_seq_len: Optional[int] = None
    ) -> Tuple[Optional[int], Optional[int]]:
        """Select the closest captured (batch, sequence-cap) graph shape.

        Sequence buckets are not necessarily present at every batch size. If
        the closest padded batch cannot cover a long sequence, continue to a
        larger batch on the full-context fallback frontier.
        """
        requested_bs = int(cuda_graph_bs)
        if self.disable_padding:
            candidate_bs = [requested_bs] if requested_bs in self.capture_bs else []
        else:
            index = bisect.bisect_left(self.capture_bs, requested_bs)
            candidate_bs = self.capture_bs[index:]

        for graph_bs in candidate_bs:
            seq_len_bucket = self._select_seq_len_fill_value_for_bs(
                graph_bs,
                max_seq_len if self.enable_seq_len_graph_buckets else None,
            )
            if seq_len_bucket is not None:
                return int(graph_bs), int(seq_len_bucket)
        return None, None

    def _prepare_seq_lens_for_capture(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_len_fill_value: int,
    ) -> None:
        """Materialize the graph shape without making every row long.

        One row establishes the static maximum sequence dimension. Empty rows
        are safe for AttnCP's zero dummy slot and keep full-context fallback
        graph capture proportional to one long sequence instead of bs long
        sequences.
        """
        if self.enable_seq_len_graph_buckets:
            seq_lens.zero_()
            seq_lens_cpu.zero_()
            seq_lens[0] = int(seq_len_fill_value)
            seq_lens_cpu[0] = int(seq_len_fill_value)
        else:
            seq_lens.fill_(int(seq_len_fill_value))
            seq_lens_cpu.fill_(int(seq_len_fill_value))

    def _graph_seq_len_key(self, seq_len_fill_value: Optional[int]) -> Optional[int]:
        return int(seq_len_fill_value) if self.enable_seq_len_graph_buckets else None

    def maybe_init_pdmux(self):
        if self.enable_pdmux:
            self.stream_groups = get_stream_groups()
            for attn_backend in self.model_runner.decode_attn_backend_group:
                if getattr(attn_backend, "is_attn_cp_sharded_kv", False):
                    attn_backend.cuda_graph_max_seq_len = self.seq_len_fill_value
                attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

    def _cache_loc_dtype(self):
        return torch.int64

    def _make_graph_key(
        self, bs, stream_idx=None, variant_label=None, seq_len_bucket=None
    ):
        """Build a graph dict key from batch size, stream index, and lora variant."""
        return _default_make_graph_key(bs, stream_idx, variant_label, seq_len_bucket)

    def _resolve_lora_variant(self, forward_batch: ForwardBatch):
        """Return the variant label for the given batch, or None if dual backends are off."""
        if not getattr(self, "record_nolora_graph", False):
            return None
        if forward_batch.lora_ids is not None and any(
            uid is not None for uid in forward_batch.lora_ids
        ):
            return "lora"
        return "nolora"

    def can_run(self, forward_batch: ForwardBatch):
        # Disable for token embedding overrides (dynamic per-request)
        if forward_batch.replace_embeds is not None:
            return False
        stream_idx = get_current_stream_idx() if self.enable_pdmux else None
        attn_backend = (
            self.model_runner.decode_attn_backend_group[stream_idx]
            if stream_idx is not None
            else self.attn_backend
        )
        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                or self.model_runner.spec_algorithm.is_dflash()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            cuda_graph_bs = forward_batch.batch_size

        max_seq_len = None
        if self.enable_seq_len_graph_buckets:
            seq_lens_cpu = forward_batch.seq_lens_cpu
            if hasattr(seq_lens_cpu, "max"):
                max_seq_len = int(seq_lens_cpu.max().item())
            else:
                max_seq_len = int(max(seq_lens_cpu))
        graph_bs, seq_len_bucket = self._select_graph_shape(
            cuda_graph_bs, max_seq_len
        )

        if forward_batch.return_logprob and getattr(
            attn_backend, "is_attn_cp_sharded_kv", False
        ):
            return False

        variant_label = self._resolve_lora_variant(forward_batch)
        graph_key = (
            self._make_graph_key(
                graph_bs,
                stream_idx,
                variant_label,
                self._graph_seq_len_key(seq_len_bucket),
            )
            if graph_bs is not None and seq_len_bucket is not None
            else None
        )
        is_bs_supported = graph_key in self.graphs if graph_key is not None else False

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )
        is_tbo_supported = (
            forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
        )

        is_ngram_supported = (
            (
                forward_batch.batch_size * self.num_tokens_per_bs
                == forward_batch.input_ids.numel()
            )
            if self.model_runner.spec_algorithm.is_ngram()
            else True
        )

        return (
            is_bs_supported
            and is_encoder_lens_supported
            and is_tbo_supported
            and capture_hidden_mode_matches
            and is_ngram_supported
        )

    def _init_profile_context_and_memory_record(self):
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )
        torch.cuda.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        torch.cuda.memory._dump_snapshot(f"cuda_graph_runner_memory_usage.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by CUDA Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=10
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=10
            )
            + "\n\nMemory Usage is saved to cuda_graph_runner_memory_usage.pickle\n"
        )
        logger.info(log_message)

    def capture(self) -> None:
        profile_context = empty_context()
        if self.enable_profile_cuda_graph:
            profile_context = self._init_profile_context_and_memory_record()

        def _capture_one_stream(stream_idx: Optional[int] = None):
            avail_mem = get_available_gpu_memory(
                self.model_runner.device,
                self.model_runner.gpu_id,
                empty_cache=False,
            )
            # Reverse the order to enable better memory sharing across cuda graphs.
            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_bs)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_bs)
            )
            # When record_nolora_graph is set, capture each batch size twice:
            # once with LoRA hooks and once without.
            lora_variants = (
                [("lora", True), ("nolora", False)]
                if getattr(self, "record_nolora_graph", False)
                else [(None, None)]
            )
            for i, bs in enumerate(capture_range):
                seq_len_buckets = list(reversed(self._seq_len_fill_values_for_bs(bs)))
                for seq_len_fill_value in seq_len_buckets:
                    if get_tensor_model_parallel_rank() == 0:
                        avail_mem = get_available_gpu_memory(
                            self.model_runner.device,
                            self.model_runner.gpu_id,
                            empty_cache=False,
                        )
                        capture_range.set_description(
                            "Capturing batches "
                            f"({bs=} seq_cap={seq_len_fill_value} "
                            f"{avail_mem=:.2f} GB)"
                        )

                    for variant_label, variant_has_lora in lora_variants:
                        _set_capture_lora_variant(variant_label)
                        with patch_model(
                            self.model_runner.model,
                            bs in self.compile_bs,
                            num_tokens=bs * self.num_tokens_per_bs,
                            tp_group=self.model_runner.tp_group,
                        ) as forward:
                            (
                                graph,
                                output_buffers,
                            ) = self.capture_one_batch_size(
                                bs, forward, stream_idx, seq_len_fill_value
                            )
                            key = _default_make_graph_key(
                                bs,
                                stream_idx,
                                variant_label,
                                self._graph_seq_len_key(seq_len_fill_value),
                            )
                            self.graphs[key] = graph
                            self.output_buffers[key] = output_buffers

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
            if not self.enable_pdmux:
                with graph_capture() as graph_capture_context, profile_context as prof:
                    self.stream = graph_capture_context.stream
                    _capture_one_stream()
            else:
                set_pdmux_status(False)
                for i, sg in enumerate(self.stream_groups):
                    with (
                        graph_capture(stream=sg[1]) as graph_capture_context,
                        profile_context as prof,
                    ):
                        self.stream = graph_capture_context.stream
                        _capture_one_stream(i)

        _set_capture_lora_variant(None)

        if self.enable_profile_cuda_graph:
            self._post_process_after_profile(prof)

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        if self.model_runner.server_args.debug_cuda_graph:
            assert (
                envs.SGLANG_USE_BREAKABLE_CUDA_GRAPH.get()
            ), "Breakable CUDA graph is not enabled in debug mode"

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.model_runner.server_args.enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

        if envs.SGLANG_USE_BREAKABLE_CUDA_GRAPH.get():
            if memory_saver_adapter.enabled:
                raise NotImplementedError(
                    "Breakable CUDA graph is not compatible with memory saver mode"
                )
            graph_ctx = BreakableCUDAGraphCapture
        else:
            graph_ctx = (
                partial(memory_saver_adapter.cuda_graph, tag=GPU_MEMORY_TYPE_CUDA_GRAPH)
                if memory_saver_adapter.enabled
                else self.device_module.graph
            )

        if self.model_runner.server_args.debug_cuda_graph:
            captured_fn = eager_on_graph(True)(run_once_fn)
        else:
            captured_fn = run_once_fn

        with graph_ctx(cuda_graph=graph, pool=pool, stream=stream):
            out = captured_fn()
        return out

    def _create_device_graph(self):
        if envs.SGLANG_USE_BREAKABLE_CUDA_GRAPH.get():
            if _is_hip:
                raise RuntimeError("Breakable CUDA graph is not supported on ROCm/HIP")
            return BreakableCUDAGraph()
        return torch.cuda.CUDAGraph()

    def capture_one_batch_size(
        self,
        bs: int,
        forward: Callable,
        stream_idx: Optional[int] = None,
        seq_len_fill_value: Optional[int] = None,
    ):
        buffers: DecodeInputBuffers = self.buffers
        graph = self._create_device_graph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs
        model_input_len = bs if self.scale_seq_factor > 1 else num_tokens
        seq_len_fill_value = (
            self._select_seq_len_fill_value_for_bs(bs)
            if seq_len_fill_value is None
            else int(seq_len_fill_value)
        )
        assert seq_len_fill_value is not None

        # Graph inputs
        # For scale_seq, the model takes bs input tokens and internally
        # expands to bs*scale hidden states. input_ids must be bs, not num_tokens.
        input_ids = buffers.input_ids[:model_input_len]
        req_pool_indices = buffers.req_pool_indices[:bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        self._prepare_seq_lens_for_capture(
            seq_lens, seq_lens_cpu, seq_len_fill_value
        )
        out_cache_loc = buffers.out_cache_loc[:num_tokens]
        positions = buffers.positions[:num_tokens]
        if self.is_encoder_decoder:
            encoder_lens = buffers.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = buffers.mrope_positions[:, :num_tokens]
        # For scale_seq, the model contracts hidden_states from num_tokens
        # back to bs before logits computation.
        logits_buffer_len = model_input_len if self.scale_seq_factor > 1 else num_tokens
        next_token_logits_buffer = buffers.next_token_logits_buffer[:logits_buffer_len]

        # Adjust for attention TP if needed (matching replay path in
        # populate_from_forward_batch).
        buffers.num_token_non_padded[...] = num_tokens
        if (
            enable_num_token_non_padded()
            and self.require_gathered_buffer
            and not self.nsa_enable_prefill_cp
        ):
            local = compute_local_num_token_non_padded(
                global_num_token_non_padded=buffers.num_token_non_padded,
                num_tokens_per_dp=num_tokens,
            )
            buffers.num_token_non_padded.copy_(local)

        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None
        router_replay_num_tokens = num_tokens
        if self.require_gathered_buffer:
            router_replay_num_tokens = num_tokens * buffers.global_num_tokens_gpu.numel()

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.model_runner.server_args.enable_lora:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            lora_ids = [None] * bs
        else:
            lora_ids = None

        # mamba state tracking
        mamba_track_indices = (
            buffers.mamba_track_indices[:bs]
            if buffers.mamba_track_indices is not None
            else None
        )
        mamba_track_mask = (
            buffers.mamba_track_mask[:bs]
            if buffers.mamba_track_mask is not None
            else None
        )

        if stream_idx is None:
            attn_backend = self.attn_backend
        else:
            assert self.enable_pdmux
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=None,  # Prefill only
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            router_replay_topk_ids=buffers.router_replay_topk_ids[
                :router_replay_num_tokens
            ],
            router_replay_mask=buffers.router_replay_mask[:router_replay_num_tokens],
            global_forward_mode=self.capture_forward_mode,
            lora_ids=lora_ids,
            scale_seq_factor=self.scale_seq_factor,
        )
        if self.model_runner.server_args.prepare_n_gram_inputs:
            use_welm_oe_decode_hash = (
                self.welm_oe_decode_hash_config is not None
                and buffers.welm_oe_decode_hashed_inputs is not None
            )
            oe_capture_token = model_input_len
            if use_welm_oe_decode_hash:
                forward_batch.oe_context = OverEncodingContext(
                    input_ids_buffer=HashInputIdsBuffer([])
                )
                forward_batch.welm_oe_decode_hashed_inputs = (
                    buffers.welm_oe_decode_hashed_inputs[:, :oe_capture_token]
                )
            else:
                raise RuntimeError(
                    "WeLM OE CUDA graph requires decode hash buffers. "
                    "Materialized n-gram fallback is no longer supported."
                )

        # mk fused decode + all-reduce: when a Prepared handle exists for
        # this batch bucket, attach it to forward_batch alongside the
        # persistent (bs, input_hidden_size) bf16 output buffer that mk
        # baked into its captured launch. The model code in welm_perf_opt
        # short-circuits to ``handle.launch()`` and returns the output
        # tensor directly; the captured graph reads this exact slice.
        prepared_handle = (
            self._welm_oe_prepared_handles.get(int(bs))
            if self._welm_oe_fused_eligible
            else None
        )
        if (
            prepared_handle is not None
            and buffers.welm_oe_fused_decode_output is not None
            and num_tokens <= 32
        ):
            forward_batch.welm_oe_fused_prepared = prepared_handle
            forward_batch.welm_oe_fused_output = (
                buffers.welm_oe_fused_decode_output[:num_tokens]
            )
            use_welm_oe_fused_prepared = True
        else:
            forward_batch.welm_oe_fused_prepared = None
            forward_batch.welm_oe_fused_output = None
            use_welm_oe_fused_prepared = False

        # HiSparse: set coordinator so the hisparse code path is captured into the graph
        forward_batch.hisparse_coordinator = self.model_runner.hisparse_coordinator
        if forward_batch.hisparse_coordinator is not None:
            forward_batch.hisparse_coordinator.num_real_reqs.fill_(bs)

        if buffers.ngram_embedding_info is not None:
            forward_batch.ngram_embedding_info = buffers.ngram_embedding_info.slice(bs)

        self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

        if lora_ids is not None:
            self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

        # Attention backend
        attn_backend._cuda_graph_seq_len_fill_value = seq_len_fill_value
        try:
            attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_batch.forward_mode,
                forward_batch.spec_info,
            )
        finally:
            attn_backend._cuda_graph_seq_len_fill_value = None

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            if use_welm_oe_decode_hash:
                forward_batch.welm_oe_decode_hashed_inputs = (
                    buffers.welm_oe_decode_hashed_inputs[:, :oe_capture_token]
                )
            else:
                forward_batch.welm_oe_decode_hashed_inputs = None
            if use_welm_oe_fused_prepared:
                forward_batch.welm_oe_fused_prepared = prepared_handle
                forward_batch.welm_oe_fused_output = (
                    buffers.welm_oe_fused_decode_output[:num_tokens]
                )
            else:
                forward_batch.welm_oe_fused_prepared = None
                forward_batch.welm_oe_fused_output = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.pp_size > 1
                and "pp_proxy_tensors" in inspect.signature(forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if (
                self.model_runner.spec_algorithm.is_dflash()
                and self.model_runner.is_draft_worker
                and "input_embeds" in inspect.signature(forward).parameters
            ):
                kwargs["input_embeds"] = buffers.input_embeds[:num_tokens]

            with set_current_forward_batch(forward_batch):
                logits_output_or_pp_proxy_tensors = forward(
                    input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
            return logits_output_or_pp_proxy_tensors

        self.deepep_adapter.capture(is_extend_in_batch=False)

        # swa_loc must be set before capture so that set_kv_buffer's
        # Python branch (if self.swa_loc is not None) takes the fast path,
        # and the graph records GPU ops using this buffer instead of the
        # per-layer translate_loc_from_full_to_swa fallback.
        if self.buffers.out_cache_loc_swa is not None:
            self.model_runner.token_to_kv_pool.set_swa_loc(
                self.buffers.out_cache_loc_swa[:num_tokens]
            )

        for _ in range(2):
            self.device_module.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()
            attn_backend.on_after_cuda_graph_warmup()

        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())

        out = self._capture_graph(
            graph, get_global_graph_memory_pool(), stream, run_once
        )

        return graph, out

    def recapture_if_needed(self, forward_batch: ForwardBatch):

        # If the required capture_hidden_mode changes, we need to recapture the graph

        # These are the different factors that can influence the capture_hidden_mode
        capture_hidden_mode_required_by_forward_batch = (
            forward_batch.capture_hidden_mode
        )
        capture_hidden_mode_required_by_spec_info = (
            getattr(forward_batch.spec_info, "capture_hidden_mode", None)
            or CaptureHiddenMode.NULL
        )
        capture_hidden_mode_required_for_returning_hidden_states = (
            CaptureHiddenMode.FULL
            if self.model_runner.server_args.enable_return_hidden_states
            else CaptureHiddenMode.NULL
        )

        # Determine the highest capture_hidden_mode required
        # (If we have FULL, we can emulate LAST or NULL)
        # (If we have LAST, we can emulate NULL)
        required_capture_hidden_mode = max(
            capture_hidden_mode_required_by_forward_batch,
            capture_hidden_mode_required_by_spec_info,
            capture_hidden_mode_required_for_returning_hidden_states,
        )

        # If the current hidden mode is no longer aligned with the required hidden mode, we need to set it to what is required and re-capture
        if self.capture_hidden_mode != required_capture_hidden_mode:
            self.capture_hidden_mode = required_capture_hidden_mode
            self.capture()

    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        buffers = self.buffers
        self.recapture_if_needed(forward_batch)

        raw_bs = forward_batch.batch_size
        raw_num_token = raw_bs * self.num_tokens_per_bs

        # Pad
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            requested_graph_bs = (
                max_num_tokens / self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                or self.model_runner.spec_algorithm.is_dflash()
                else max_num_tokens
            )
        else:
            requested_graph_bs = raw_bs
        if self.enable_seq_len_graph_buckets:
            if forward_batch.seq_lens_cpu is not None:
                max_seq_len = int(forward_batch.seq_lens_cpu.max().item())
            else:
                max_seq_len = int(forward_batch.seq_lens.max().item())
        else:
            max_seq_len = None
        bs, seq_len_fill_value = self._select_graph_shape(
            requested_graph_bs, max_seq_len
        )
        if bs is None or seq_len_fill_value is None:
            raise RuntimeError(
                "CUDA graph replay requested without a captured graph shape: "
                f"requested_bs={requested_graph_bs}, max_seq_len={max_seq_len}"
            )
        seq_len_padding_value = (
            0 if self.enable_seq_len_graph_buckets else seq_len_fill_value
        )

        buffers.populate_from_forward_batch(
            forward_batch=forward_batch,
            raw_bs=raw_bs,
            raw_num_token=raw_num_token,
            bs=bs,
            seq_len_padding_value=seq_len_padding_value,
            require_gathered_buffer=self.require_gathered_buffer,
            num_tokens_per_bs=self.num_tokens_per_bs,
            nsa_enable_prefill_cp=self.nsa_enable_prefill_cp,
            enable_num_token_non_padded_flag=enable_num_token_non_padded(),
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if (
            self.model_runner.spec_algorithm.is_dflash()
            and self.model_runner.is_draft_worker
            and forward_batch.input_embeds is not None
        ):
            buffers.input_embeds[:raw_num_token].copy_(forward_batch.input_embeds)
            # Padded tokens aren't read, so skip zeroing them.
        cached_welm_oe_hash_inputs = getattr(
            forward_batch, "welm_oe_decode_hashed_inputs", None
        )
        # In scale_seq mode, forward_batch.input_ids and OE hash inputs are
        # bs-sized, while raw_num_token/static_num_token describe the expanded
        # KV-write view. Keep hash buffer copies aligned with input_ids.
        is_scale_seq_replay = self.scale_seq_factor > 1
        oe_active_token = raw_bs if is_scale_seq_replay else raw_num_token
        oe_static_token = bs if is_scale_seq_replay else bs * self.num_tokens_per_bs
        # Precedence branch: if the runner has a prepared mk fused-decode
        # handle for this bucket, refresh its prefix buffer (host-side write
        # + stream-ordered H2D copy) and dispatch the prepared handle to the
        # captured graph. Skips the welm_oe_decode_hashed_inputs fill — mk
        # computes its own hashed lookups internally.
        prepared_handle = (
            self._welm_oe_prepared_handles.get(int(bs))
            if self._welm_oe_fused_eligible
            else None
        )
        if (
            prepared_handle is not None
            and buffers.welm_oe_fused_decode_output is not None
            and raw_num_token <= 32
            and bs <= 32
        ):
            from sglang.srt.models.welm_perf_opt import (
                build_welm_oe_fused_prefix_list,
            )

            prefix_rows = build_welm_oe_fused_prefix_list(
                forward_batch=forward_batch,
                runtime_batch_size=int(bs),
                max_history=int(prepared_handle.max_history),
            )
            try:
                prepared_handle.set_prefixes(prefix_rows)
            except Exception as exc:
                # Hard failure mid-replay would crash the request; degrade by
                # disabling the prepared handle for this step. The captured
                # graph still reads stale prefix bytes — but since the graph
                # was captured with the prepared path, falling back here
                # produces wrong outputs. Surface loudly.
                logger.error(
                    "WeLM OE fused decode set_prefixes failed at replay: %r",
                    exc,
                )
                raise
            forward_batch.welm_oe_fused_prepared = prepared_handle
            forward_batch.welm_oe_fused_output = (
                buffers.welm_oe_fused_decode_output[:bs]
            )
        elif (
            self.welm_oe_decode_hash_config is not None
            and buffers.welm_oe_decode_hashed_inputs is not None
            and cached_welm_oe_hash_inputs is not None
        ):
            buffers.welm_oe_decode_hashed_inputs[:, :oe_active_token].copy_(
                cached_welm_oe_hash_inputs[:, :oe_active_token]
            )
            if oe_active_token < oe_static_token:
                buffers.welm_oe_decode_hashed_inputs[
                    :, oe_active_token:oe_static_token
                ].zero_()
        elif (
            self.welm_oe_decode_hash_config is not None
            and buffers.welm_oe_decode_hashed_inputs is not None
            and forward_batch.oe_context is not None
            and forward_batch.oe_context.hash_prefixes is not None
        ):
            oe_grams, oe_vocab_sizes = self.welm_oe_decode_hash_config
            fill_welm_oe_hash_inputs(
                forward_batch.input_ids[:oe_active_token],
                buffers.welm_oe_decode_hashed_inputs[:, :oe_active_token],
                forward_batch,
                oe_grams,
                oe_vocab_sizes,
                self.model_runner.model_config.vocab_size,
            )
            if oe_active_token < oe_static_token:
                buffers.welm_oe_decode_hashed_inputs[
                    :, oe_active_token:oe_static_token
                ].zero_()
        if self.enable_two_batch_overlap:
            self.tbo_plugin.replay_prepare(
                forward_mode=self.capture_forward_mode,
                bs=bs,
                num_token_non_padded=len(forward_batch.input_ids),
                spec_info=forward_batch.spec_info,
            )
        if forward_batch.forward_mode.is_idle() and forward_batch.spec_info is not None:
            forward_batch.spec_info.custom_mask = buffers.custom_mask
        # Attention backend
        if self.enable_pdmux:
            stream_idx = get_current_stream_idx()
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]
        else:
            attn_backend = self.attn_backend
        # FIXME: implicit channel for backends (dsv4) that need forward_batch
        # in replay metadata prep. Should become a real param on the interface.
        attn_backend._replay_forward_batch = forward_batch
        attn_backend._cuda_graph_seq_len_fill_value = seq_len_fill_value
        try:
            attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                buffers.req_pool_indices[:bs],
                buffers.seq_lens[:bs],
                forward_batch.seq_lens_sum + (bs - raw_bs) * seq_len_padding_value,
                buffers.encoder_lens[:bs] if self.is_encoder_decoder else None,
                self.capture_forward_mode,
                forward_batch.spec_info,
                seq_lens_cpu=buffers.seq_lens_cpu[:bs],
            )
        finally:
            attn_backend._replay_forward_batch = None
            attn_backend._cuda_graph_seq_len_fill_value = None

        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs
        self.seq_len_bucket = seq_len_fill_value

        if self.model_runner.hisparse_coordinator is not None:
            self.model_runner.hisparse_coordinator.num_real_reqs.fill_(raw_bs)

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        self.deepep_adapter.replay()

        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)
            if (
                self.model_runner.spec_algorithm.is_dflash()
                and self.model_runner.is_draft_worker
                and forward_batch.input_embeds is not None
            ):
                self.buffers.input_embeds[: self.raw_num_token].copy_(
                    forward_batch.input_embeds
                )

        # Replay
        variant_label = self._resolve_lora_variant(forward_batch)
        stream_idx = get_current_stream_idx() if self.enable_pdmux else None
        graph_key = self._make_graph_key(
            self.bs,
            stream_idx,
            variant_label,
            self._graph_seq_len_key(getattr(self, "seq_len_bucket", None)),
        )
        ctx = (
            self.model_runner.device_timer.wrap(
                metadata={
                    "category": forward_batch.forward_mode.name.lower(),
                }
            )
            if self.model_runner.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            self.graphs[graph_key].replay()

        output = self.output_buffers[graph_key]

        if isinstance(output, LogitsProcessorOutput):
            # For scale_seq, the model internally contracts hidden_states
            # from (bs*scale, D) to (bs, D) before logits. The effective
            # logits output size is raw_bs, not raw_num_token.
            logits_num_token = (
                self.raw_bs if self.scale_seq_factor > 1 else self.raw_num_token
            )
            if self.is_dllm:
                next_token_logits = None
                full_logits = (
                    output.full_logits[:logits_num_token]
                    if output.full_logits is not None
                    else None
                )
            else:
                full_logits = None
                next_token_logits = (
                    output.next_token_logits[:logits_num_token]
                    if output.next_token_logits is not None
                    else None
                )
            model_specific_states = output.model_specific_states
            if model_specific_states is not None:
                model_specific_states = dict(model_specific_states)
                kv_mirror_states = model_specific_states.get("welm_kv_mirror_states")
                if isinstance(kv_mirror_states, dict):
                    model_specific_states["welm_kv_mirror_states"] = {
                        layer_idx: tuple(
                            (
                                tensor[: self.raw_num_token]
                                if (
                                    isinstance(tensor, torch.Tensor)
                                    and tensor.shape
                                    and tensor.shape[0] >= self.raw_num_token
                                )
                                else tensor
                            )
                            for tensor in tensors
                        )
                        for layer_idx, tensors in kv_mirror_states.items()
                    }

            return LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
                model_specific_states=model_specific_states,
                customized_info=output.customized_info,
            )
        else:
            assert isinstance(output, PPProxyTensors)
            tensors = {}
            for key, value in output.tensors.items():
                if key.endswith(".shape"):
                    continue
                if not isinstance(value, torch.Tensor):
                    tensors[key] = value
                    continue

                shape = output.tensors.get(f"{key}.shape")
                if shape is not None and key.startswith(
                    f"{WELM_KV_MIRROR_PP_KEY_PREFIX}."
                ):
                    replay_shape = (self.raw_num_token, *tuple(shape[1:]))
                    packed_len = _welm_kv_mirror_packed_len(math.prod(replay_shape))
                    tensors[key] = value[:packed_len].clone()
                    tensors[f"{key}.shape"] = replay_shape
                else:
                    tensors[key] = value[: self.raw_num_token].clone()
            return PPProxyTensors(tensors)

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        if (
            self.model_runner.spec_algorithm.is_eagle()
            or self.model_runner.spec_algorithm.is_standalone()
        ):
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            if self.model_runner.is_draft_worker:
                raise RuntimeError("This should not happen.")
            else:

                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.model_runner.spec_algorithm.is_standalone()
                    else CaptureHiddenMode.FULL
                )
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.buffers.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    retrieve_cum_len=None,
                    spec_steps=self.speculative_num_steps,
                    topk=self.model_runner.server_args.speculative_eagle_topk,
                    draft_token_num=self.speculative_num_draft_tokens,
                    capture_hidden_mode=capture_mode,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )
        elif self.model_runner.spec_algorithm.is_dflash():
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput
            from sglang.srt.speculative.dflash_utils import (
                resolve_dflash_verify_mask_policy,
            )

            # Avoid enabling custom-mask modes during graph capture for backends that
            # can express DFLASH verify via their built-in causal path.
            _, build_custom_mask = resolve_dflash_verify_mask_policy(
                self.model_runner.attn_backend
            )
            spec_info = DFlashVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=self.model_runner.server_args.speculative_num_draft_tokens,
                custom_mask=(
                    None
                    if (self.model_runner.is_draft_worker or not build_custom_mask)
                    else self.buffers.custom_mask
                ),
                capture_hidden_mode=(
                    CaptureHiddenMode.NULL
                    if self.model_runner.is_draft_worker
                    else CaptureHiddenMode.FULL
                ),
            )

        elif self.model_runner.spec_algorithm.is_ngram():
            from sglang.srt.speculative.ngram_info import NgramVerifyInput

            spec_info = NgramVerifyInput(
                draft_token=None,
                tree_mask=self.buffers.custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                draft_token_num=self.num_tokens_per_bs,
            )
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        return spec_info


CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Possible solutions:\n"
    "1. set --mem-fraction-static to a smaller value (e.g., 0.8 or 0.7)\n"
    "2. set --cuda-graph-max-bs to a smaller value (e.g., 16)\n"
    "3. disable torch compile by not using --enable-torch-compile\n"
    "4. disable CUDA graph by --disable-cuda-graph. (Not recommended. Huge performance loss)\n"
    "Open an issue on GitHub https://github.com/sgl-project/sglang/issues/new/choose \n"
)


class DeepEPCudaGraphRunnerAdapter:
    def __init__(self):
        # Record DeepEP mode used during capture to ensure replay consistency
        self._captured_deepep_mode = None

    def capture(self, is_extend_in_batch: bool):
        if not get_moe_a2a_backend().is_deepep():
            return
        self._captured_deepep_mode = get_deepep_mode().resolve(
            is_extend_in_batch=is_extend_in_batch
        )
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)

    def replay(self):
        if not get_moe_a2a_backend().is_deepep():
            return
        assert self._captured_deepep_mode is not None
        DeepEPBuffer.set_dispatch_mode(self._captured_deepep_mode)
