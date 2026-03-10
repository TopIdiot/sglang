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

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""

import gc
import logging
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import (
    RotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
    yarn_get_mscale,
)
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.layers.welmv4_op import (
    WelmV4FusedRMSNorm,
    WelmV4InplaceRotaryEmbedding,
    inplace_sigmoid_mul,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args

# from sglang.srt.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.utils import add_prefix, is_cuda, make_layers

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()


def log(*args):
    if torch.cuda.current_device() == 0:
        print(*args)


def hash_input_ids_vectorized(input_ids: torch.Tensor) -> torch.Tensor:
    ids = input_ids.to(torch.int64)
    result = ids * 2654435761
    result = result & 0xFFFFFFFF
    return result.to(input_ids.dtype)


class KVMirrorManager:
    """
    Manager for kv mirror algorithm
    """

    activations_dict_kv = dict()

    @staticmethod
    def set_kv_activation(layer_number, kv_activation):
        KVMirrorManager.activations_dict_kv[layer_number] = kv_activation

    @staticmethod
    def get_kv_activation(layer_number, clear=False):
        assert (
            layer_number in KVMirrorManager.activations_dict_kv
        ), f"layer {layer_number} not in activations_dict_kv, only layers {KVMirrorManager.activations_dict_kv.keys()} are existing"
        kv_activation = KVMirrorManager.activations_dict_kv.pop(layer_number)
        if clear:
            KVMirrorManager.activations_dict_kv.clear()
        return kv_activation


def remove_all_references(obj: Any, verbose: bool = False) -> int:
    removed_count = 0
    referrers = gc.get_referrers(obj)

    if verbose:
        print(f"找到 {len(referrers)} 个引用者")

    for ref in referrers:
        try:
            # 跳过 gc.get_referrers 返回的列表本身
            if ref is referrers:
                continue

            # 1. 如果是字典，查找并删除键值对
            if isinstance(ref, dict):
                keys_to_delete = []
                for key, value in ref.items():
                    if value is obj:
                        keys_to_delete.append(key)

                for key in keys_to_delete:
                    try:
                        del ref[key]
                        removed_count += 1
                        if verbose:
                            print(f"  从字典中删除键: {key}")
                    except (TypeError, KeyError):
                        pass  # 可能是只读字典

            # 2. 如果是列表，尝试替换或删除元素
            elif isinstance(ref, list):
                try:
                    indices = [i for i, item in enumerate(ref) if item is obj]
                    # 从后往前删除，避免索引变化
                    for i in reversed(indices):
                        ref.pop(i)
                        removed_count += 1
                        if verbose:
                            print(f"  从列表中删除索引: {i}")
                except (TypeError, AttributeError):
                    pass  # 可能是只读列表

            # 3. 如果是对象，尝试删除属性
            elif hasattr(ref, "__dict__"):
                attrs_to_delete = []
                for attr_name, attr_value in ref.__dict__.items():
                    if attr_value is obj:
                        attrs_to_delete.append(attr_name)

                for attr_name in attrs_to_delete:
                    try:
                        delattr(ref, attr_name)
                        removed_count += 1
                        if verbose:
                            print(
                                f"  从对象 {type(ref).__name__} 中删除属性: {attr_name}"
                            )
                    except (AttributeError, TypeError):
                        pass

            # 4. 如果是元组或不可变类型，无法删除
            elif isinstance(ref, (tuple, frozenset)):
                if verbose:
                    print(f"  跳过不可变类型: {type(ref).__name__}")

        except Exception as e:
            if verbose:
                print(f"  处理引用时出错: {type(ref).__name__} - {e}")
            continue

    # 强制垃圾回收
    gc.collect()

    if verbose:
        remaining_refs = sys.getrefcount(obj) - 1
        print(f"删除后剩余引用数: {remaining_refs}")

    return removed_count


class LayerManager:
    decoder_layer = dict()
    num_nextn_predict_layers: int = 0
    num_target_layers: int = 0
    num_nextn_predict_layer_idx: List[int] = []

    @staticmethod
    def set_decoder_layer(layer_idx, decoder_layer):
        LayerManager.decoder_layer[layer_idx] = decoder_layer

    @staticmethod
    def post_init(kv_mirror_layers, kv_mirror_imitated_layers, is_nextn=False):

        if is_nextn:
            LayerManager.num_nextn_predict_layer_idx = kv_mirror_layers
        for mirror_layer_id in kv_mirror_layers:
            if mirror_layer_id >= len(LayerManager.decoder_layer):
                continue
            imitated_layer_id = kv_mirror_imitated_layers[
                kv_mirror_layers.index(mirror_layer_id)
            ]
            mirror_layer_attn = LayerManager.decoder_layer[mirror_layer_id].self_attn
            imitated_layer_attn = LayerManager.decoder_layer[
                imitated_layer_id
            ].self_attn

            mirror_qkv_proj_weight = mirror_layer_attn.qkv_proj.weight
            mirror_qkv_proj_bias = getattr(mirror_layer_attn.qkv_proj, "bias", None)
            imitated_qkv_proj_weight = imitated_layer_attn.qkv_proj.weight
            imitated_qkv_proj_bias = getattr(imitated_layer_attn.qkv_proj, "bias", None)
            assert (mirror_qkv_proj_bias is not None) == (
                imitated_qkv_proj_bias is not None
            )

            mirror_layer_attn.qkv_proj_weight = mirror_qkv_proj_weight[
                : mirror_layer_attn.q_size, :
            ].clone()
            imitated_layer_attn.qkv_proj_weight = torch.concat(
                [
                    imitated_qkv_proj_weight,
                    mirror_qkv_proj_weight[mirror_layer_attn.q_size :, :],
                ],
                dim=0,
            )
            if mirror_qkv_proj_bias is not None:
                mirror_layer_attn.qkv_proj_bias = mirror_qkv_proj_bias[
                    : mirror_layer_attn.q_size
                ].clone()
                imitated_layer_attn.qkv_proj_bias = torch.concat(
                    [
                        imitated_qkv_proj_bias,
                        mirror_qkv_proj_bias[mirror_layer_attn.q_size :],
                    ],
                    dim=0,
                )
            else:
                imitated_layer_attn.qkv_proj_bias = None
                mirror_layer_attn.qkv_proj_bias = None

            # if mirror_layer_id not in LayerManager.num_nextn_predict_layer_idx:
            #     remove_all_references(mirror_layer_attn.qkv_proj)
            #     remove_all_references(mirror_qkv_proj_weight)
            #     remove_all_references(mirror_qkv_proj_bias)
            #     remove_all_references(imitated_layer_attn.qkv_proj)
            #     remove_all_references(imitated_qkv_proj_weight)
            #     remove_all_references(imitated_qkv_proj_bias)
        torch.cuda.empty_cache()


class WeLMV4MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


def expert_bias_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    expert_bias: torch.Tensor,
    renormalize: bool = False,
    score_func: str = "sigmoid",
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    if score_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1).type_as(gating_output)
    else:
        scores = torch.sigmoid(gating_output).type_as(gating_output)

    scores_for_routing = scores + expert_bias
    _, indices = torch.topk(scores_for_routing, topk, dim=-1)
    topk_scores = torch.gather(scores, dim=1, index=indices).type_as(scores)

    return topk_scores, indices


def sigmoid_routing_function(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # if softmax, then use qwen3 moe's routing function
    scores = torch.sigmoid(gating_output).type_as(gating_output)
    scores_for_routing = scores
    if correction_bias is not None:
        scores += correction_bias
    _, indices = torch.topk(scores, topk, dim=-1)
    topk_scores = torch.gather(scores, dim=1, index=indices).type_as(scores)
    return topk_scores, indices


class WeLMV4MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.expert_bias = torch.nn.Parameter(
            torch.zeros((config.num_experts), dtype=torch.float32)
        )
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.router_score_func = (
            config.router_score_func
            if hasattr(config, "router_score_func")
            else "softmax"
        )
        if config.moe_routing_type == "expert_bias":
            from functools import partial

            custom_routing_function = partial(
                expert_bias_routing,
                expert_bias=self.expert_bias,
                score_func=self.router_score_func,
            )
            self.custom_routing_function = custom_routing_function
        else:
            if self.router_score_func == "softmax":
                self.custom_routing_function = None
            elif self.router_score_func == "sigmoid":
                self.custom_routing_function = sigmoid_routing_function
            else:
                raise ValueError(f"Unknown router_score_func: {self.router_score_func}")

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            layer_id=self.layer_id,
            renormalize=config.norm_topk_prob,
            custom_routing_function=self.custom_routing_function,
        )

        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.gate.weight.data = self.gate.weight.to(torch.float32)
        if config.shared_expert_intermediate_size > 0:
            self.shared_expert = WeLMV4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
            )
        else:
            self.shared_expert = None

        self.shared_expert_gate = None
        has_shared_expert_gate = getattr(
            config, "has_shared_expert_gate", True
        )  # default to true since qwen2_moe always has it
        if has_shared_expert_gate:
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp32: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
                )
        router_logits = F.linear(hidden_states_fp32, self.gate.weight)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(hidden_states, topk_output)
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1 and not use_reduce_scatter:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(num_tokens, hidden_dim)


# WelmV4InplaceRotaryEmbedding
class WeLMV4LinearScalingRotaryEmbedding(WelmV4InplaceRotaryEmbedding):
    """WelmV4 rotary embedding with linear scaling, using WelmV4's inplace Triton
    kernel so that RoPE is correctly applied to the *tail* of head_dim (partial rotary).
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        max_len = self.max_position_embeddings * self.scaling_factor
        t = torch.arange(max_len, dtype=torch.float)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


class WeLMV4MoeYarnScalingRotaryEmbedding(WelmV4InplaceRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        compress: float = 0,
        max_position: int = 40 * 4096,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.compress = compress
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.max_position = max_position
        inv_freq_extra = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        inv_freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq_extra", inv_freq_extra, persistent=False)
        self.register_buffer("inv_freq_inter", inv_freq_inter, persistent=False)

        self.cos_sin_cache = self._update_cos_sin_cache(self.max_position)

    def _update_cos_sin_cache(self, seqlen: int):
        """Update cos/sin cache with YaRN scaling"""
        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2, dtype=torch.float32
        ).to(device=self.inv_freq_inter.device)

        inv_freq = (
            self.inv_freq_inter * (1 - inv_freq_mask)
            + self.inv_freq_extra * inv_freq_mask
        )

        seq = (
            torch.arange(seqlen, device=self.inv_freq_extra.device, dtype=torch.float32)
            * self.compress
        )

        freqs = torch.outer(seq, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        _cos_cached = (torch.cos(freqs) * _mscale).to(torch.float32)
        _sin_cached = (torch.sin(freqs) * _mscale).to(torch.float32)
        cache = torch.cat((_cos_cached, _sin_cached), dim=-1)
        return cache


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    compress: float = 1.0,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if rope_scaling is None:
        raise ValueError(f"Please set RoPE scaling")
    else:
        scaling_type = rope_scaling["type"]

        if scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = WeLMV4LinearScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )

        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            if max_position != int(original_max_position * scaling_factor):
                raise ValueError(
                    f"max_position ({max_position}) != original_max_position ({original_max_position}) * scaling_factor ({scaling_factor})"
                )
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in (
                    "extrapolation_factor",
                    "attn_factor",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                )
            }
            rotary_emb = WeLMV4MoeYarnScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
                compress=compress,
                max_position=max_position,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


class WeLMV4MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        compress: float = 1.0,
        max_position_embeddings: int = 8192,
        qkv_bias: int = True,
        out_bias: int = False,
        qk_norm: bool = False,
        k_norm: bool = False,
        qk_rope_head_dim: int = 0,
        qk_norm_eps: float = 1e-5,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
        kv_mirror_layers=[],
        kv_mirror_imitated_layers=[],
        sliding_window_size_layerwise=[],
        enable_attn_sink_layerwise=[],
        layer_idx: Optional[int] = None,
        o_norm=False,
        total_layer_num: int = 1,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.compress = compress
        self.max_position_embeddings = max_position_embeddings
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_norm = qk_norm
        self.only_k_norm = k_norm
        self.use_o_norm = o_norm
        self.total_layer_num = total_layer_num
        self.o_norm = RMSNorm(self.hidden_size) if self.use_o_norm else nn.Identity()

        self.q_norm = (
            WelmV4FusedRMSNorm(self.head_dim, eps=qk_norm_eps)
            if self.qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            WelmV4FusedRMSNorm(self.head_dim, eps=qk_norm_eps)
            if self.qk_norm or self.only_k_norm
            else nn.Identity()
        )

        self.kv_mirror_layers = kv_mirror_layers
        self.kv_mirror_imitated_layers = kv_mirror_imitated_layers
        self.layer_idx = layer_idx
        print(
            "self.layer_idx:{}".format(layer_idx),
            "self.kv_mirror_layers:",
            self.kv_mirror_layers,
            "self.kv_mirror_imitated_layers:",
            self.kv_mirror_imitated_layers,
            flush=True,
        )
        if len(sliding_window_size_layerwise) > layer_idx:
            self.sliding_window_size = sliding_window_size_layerwise[layer_idx]
        else:
            self.sliding_window_size = -1
        print(
            "self.layer_idx:{}".format(layer_idx),
            "self.sliding_window_size:",
            self.sliding_window_size,
            flush=True,
        )
        if len(enable_attn_sink_layerwise) > layer_idx:
            self.enable_attention_sink = enable_attn_sink_layerwise[layer_idx]
        else:
            self.enable_attention_sink = False
        print(
            "self.layer_idx:{}".format(layer_idx),
            "self.enable_attention_sink:",
            self.enable_attention_sink,
            flush=True,
        )
        if self.enable_attention_sink == True:
            self.attn_sink = nn.Parameter(
                torch.empty(self.num_heads), requires_grad=False
            )
        else:
            self.attn_sink = None

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=out_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=True,
            prefix=add_prefix("o_proj", prefix),
        )
        if rope_scaling is None:
            rope_scaling = {"type": "linear", "factor": 1 / self.compress}
        else:
            assert self.compress == 1.0, "Compress must be 1.0 for custom rope scaling."
            if rope_scaling["type"] == "yarn":
                mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
                apply_softmax_scale = rope_scaling.get("apply_softmax_scale", False)
                scaling_factor = rope_scaling["factor"]
                if apply_softmax_scale and mscale_all_dim:
                    mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                    self.scaling = self.scaling * mscale * mscale

        self.rotary_emb = get_rope(
            # self.qk_rope_head_dim,
            self.head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            compress=self.compress,
            rope_scaling=rope_scaling,
        )

        self.rotary_emb_orig = get_rope(
            self.qk_rope_head_dim,
            # self.head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            compress=self.compress,
            rope_scaling=rope_scaling,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            sliding_window_size=self.sliding_window_size,
        )
        self.gated_self_attention_headwise = True
        if self.gated_self_attention_headwise:
            self.gate_proj = ColumnParallelLinear(
                hidden_size,
                self.total_num_heads,
                bias=False,
            )
        self.attn.is_kv_mirror = self.layer_idx in self.kv_mirror_layers
        self.kv_mirror_layer_idx = (
            layer_idx if not is_nextn else layer_idx + len(LayerManager.decoder_layer)
        )
        if get_global_server_args().speculative_algorithm is not None:
            self.need_clear_kv_cache = (
                self.layer_idx == LayerManager.num_nextn_predict_layers - 1
            )
        else:
            self.need_clear_kv_cache = (
                self.layer_idx == LayerManager.num_target_layers - 1
            )
        self.is_nextn = is_nextn

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.kv_mirror_layer_idx in self.kv_mirror_imitated_layers:
            if hasattr(self, "qkv_proj_weight"):
                qkv = F.linear(hidden_states, self.qkv_proj_weight, self.qkv_proj_bias)
                q, k, v, mirror_k, mirror_v = qkv.split(
                    [
                        self.q_size,
                        self.kv_size,
                        self.kv_size,
                        self.kv_size,
                        self.kv_size,
                    ],
                    dim=-1,
                )
                KVMirrorManager.set_kv_activation(
                    self.kv_mirror_layer_idx, (mirror_k, mirror_v)
                )
            else:
                qkv, _ = self.qkv_proj(hidden_states)
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        elif self.kv_mirror_layer_idx in self.kv_mirror_layers:
            if (
                self.kv_mirror_layer_idx in LayerManager.num_nextn_predict_layer_idx
                and not forward_batch.forward_mode.is_extend_without_speculative()
            ):
                qkv, _ = self.qkv_proj(hidden_states)
                q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            else:
                mirror_layer_number = self.kv_mirror_imitated_layers[
                    self.kv_mirror_layers.index(self.kv_mirror_layer_idx)
                ]
                if (
                    forward_batch.enable_kv_mirror
                    and forward_batch.forward_mode.is_extend_without_speculative()
                    and not hasattr(forward_batch, "custom_last_index")
                ):
                    forward_batch.custom_last_index = (
                        torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                    )
                    hidden_states = hidden_states[forward_batch.custom_last_index]
                k, v = KVMirrorManager.get_kv_activation(
                    mirror_layer_number, clear=self.need_clear_kv_cache
                )
                q = F.linear(hidden_states, self.qkv_proj_weight, self.qkv_proj_bias)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q_shape = q.shape
        k_shape = k.shape

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        if self.qk_norm:
            q_by_head, _ = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        # for welmv4, qk_norm is false and only_k_norm is true
        # fuse: implement a high precision and fused k_norm
        if self.qk_norm or self.only_k_norm:
            k_by_head, _ = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)

        qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        if qk_nope_head_dim > 0:
            if (
                forward_batch.enable_kv_mirror
                and forward_batch.forward_mode.is_extend_without_speculative()
                and self.kv_mirror_layer_idx in self.kv_mirror_layers
            ):
                # optim: mirror rope embedding optim @kavioyu
                # q_pe_proxy = torch.empty(
                #     (k_pe.shape[0],) + q_pe.shape[1:],
                #     dtype=q_pe.dtype,
                #     device=q_pe.device,
                # )
                # k_pe_proxy = torch.empty(
                #     (q_pe.shape[0],) + k_pe.shape[1:],
                #     dtype=k_pe.dtype,
                #     device=k_pe.device,
                # )
                # _, k_pe = self.rotary_emb(positions, q_pe_proxy, k_pe)
                # q_pe, _ = self.rotary_emb(
                #     positions[forward_batch.custom_last_index], q_pe, k_pe_proxy
                # )
                self.rotary_emb.forward_cuda(
                    positions, q, k, last_index=forward_batch.custom_last_index
                )
            else:
                self.rotary_emb.forward_cuda(positions, q, k)
            q = q.view(q_shape)
            k = k.view(k_shape)
        else:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch, sinks=self.attn_sink)
        if self.gated_self_attention_headwise:
            attn_shape = attn_output.shape
            gate = self.gate_proj(hidden_states)[0].unsqueeze(
                -1
            )  # (bs * seq_len, num_heads, 1)
            # fuse: implement a fused sigmoid mul, maybe could use torch.compile
            attn_output = attn_output.view(attn_shape[0], self.num_heads, -1)
            inplace_sigmoid_mul(gate, attn_output)
            attn_output = attn_output.view(attn_shape)

        output, _ = self.o_proj(attn_output)
        if self.use_o_norm:
            output = self.o_norm(output)
        return output


class WeLMV4MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        if getattr(config, "qkv_bias", None) is not None:
            qkv_bias = getattr(config, "qkv_bias")
        elif getattr(config, "qkv_proj_bias", None) is not None:
            qkv_bias = getattr(config, "qkv_proj_bias")
        else:
            qkv_bias = True
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        qk_norm = getattr(config, "qk_norm", False)
        k_norm = getattr(config, "k_norm", False)
        out_bias = getattr(config, "out_proj_bias", False)
        head_dim = getattr(
            config, "head_dim", self.hidden_size // config.num_attention_heads
        )
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", head_dim)

        self.kv_mirror_layers = getattr(config, "kv_mirror_layers", [])
        self.kv_mirror_imitated_layers = getattr(
            config, "kv_mirror_imitated_layers", []
        )
        self.sliding_window_size_layerwise = getattr(
            config, "sliding_window_size_layerwise", []
        )
        self.enable_attn_sink_layerwise = getattr(
            config, "enable_attn_sink_layerwise", []
        )
        self.ppln = getattr(config, "ppln", False)
        o_norm = getattr(config, "o_norm", False)
        self.prenorm_layer_idx = getattr(config, "prenorm_layer_idx", [])
        print(
            "self.ppln:",
            self.ppln,
            "o_norm:",
            o_norm,
            "self.prenorm_layer_idx:",
            self.prenorm_layer_idx,
        )
        total_layer_num = config.num_hidden_layers

        self.self_attn = WeLMV4MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            qk_norm=qk_norm,
            k_norm=k_norm,
            qk_rope_head_dim=qk_rope_head_dim,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            prefix=add_prefix("self_attn", prefix),
            kv_mirror_layers=self.kv_mirror_layers,
            kv_mirror_imitated_layers=self.kv_mirror_imitated_layers,
            sliding_window_size_layerwise=self.sliding_window_size_layerwise,
            enable_attn_sink_layerwise=self.enable_attn_sink_layerwise,
            layer_idx=layer_id,
            o_norm=o_norm and layer_id not in self.prenorm_layer_idx,
            total_layer_num=total_layer_num,
            is_nextn=is_nextn,
        )
        LayerManager.num_nextn_predict_layers = getattr(
            config, "num_nextn_predict_layers", 0
        )
        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Qwen2MoE all layers are sparse (include nextn layers)
        self.is_layer_sparse = True
        is_previous_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=True,
        )

        if self.is_layer_sparse:
            self.mlp = WeLMV4MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = WeLMV4MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        # self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = RMSNorm(
        #     config.hidden_size, eps=config.rms_norm_eps
        # )
        self.input_layernorm = WelmV4FusedRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = WelmV4FusedRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )
        LayerManager.set_decoder_layer(self.self_attn.kv_mirror_layer_idx, self)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # mtp 这里需要prenorm这里处理的对吗？需要确认一下@kavioyu
        # fuse: input layer norm and copy residual
        residual_after_layernorm = (
            self.ppln and self.layer_id not in self.prenorm_layer_idx
        )
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual, residual_after_layernorm=residual_after_layernorm
        )
        # if residual is None:
        #     residual = hidden_states
        #     hidden_states = self.input_layernorm(hidden_states)
        # else:
        #     hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.ppln and self.layer_id not in self.prenorm_layer_idx:
            residual = hidden_states.clone().to(
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        if (
            forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
            and self.layer_id == self.kv_mirror_layers[-1]
        ):
            residual = residual[forward_batch.custom_last_index]
        # fuse: implement a post_attention_layernorm with cast to float32 to avoid cast before moe router
        # hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, residual, hidden_states_fp32 = self.post_attention_layernorm(
            hidden_states, residual, clone_fp32_out=True
        )
        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        hidden_states = self.mlp(
            hidden_states, hidden_states_fp32, forward_batch, use_reduce_scatter
        )
        return hidden_states, residual


class WeLMV4MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = WeLMV4MoeDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        self.oe_dim = config.oe_dim
        self.oe_grams = config.oe_grams
        self.oe_vocab_sizes = config.oe_vocab_sizes

        if len(self.oe_vocab_sizes) > 0:
            self.oe_embed = nn.ModuleList(
                [
                    VocabParallelEmbedding(
                        self.oe_vocab_sizes[i],
                        self.oe_dim,
                    )
                    for i in range(len(self.oe_vocab_sizes))
                ]
            )
            self.oe_gate_up_proj = ReplicatedLinear(
                self.oe_dim * len(self.oe_vocab_sizes),
                config.hidden_size,
                bias=False,
                quant_config=None,
            )

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to WeLMV4MoeDecoderLayer
        decoder_layer_type = decoder_layer_type or WeLMV4MoeDecoderLayer
        LayerManager.num_target_layers = config.num_hidden_layers
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds

            if len(self.oe_grams) > 0:
                input_ids_ngram = []
                input_ids_ngram_tmp = input_ids
                if getattr(forward_batch, "n_gram_input_ids", None) is not None:
                    input_ids_gram_n = [
                        forward_batch.n_gram_input_ids.input_ids_gram2,
                        forward_batch.n_gram_input_ids.input_ids_gram3,
                        forward_batch.n_gram_input_ids.input_ids_gram4,
                    ]
                else:
                    zero_ids = torch.zeros_like(input_ids)
                    input_ids_gram_n = [zero_ids, zero_ids, zero_ids]
                for g in range(1, max(self.oe_grams)):
                    input_ids_ngram_tmp = input_ids_ngram_tmp + input_ids_gram_n[
                        g - 1
                    ] * (self.vocab_size**g)
                    input_ids_ngram.append(
                        hash_input_ids_vectorized(input_ids_ngram_tmp)
                    )

                emb_ngram = []
                for i, vs in enumerate(self.oe_vocab_sizes):
                    input_ids_ngram_hashed_tmp = (
                        input_ids_ngram[self.oe_grams[i] - 2] % vs
                    )
                    emb_ngram_tmp = self.oe_embed[i](input_ids_ngram_hashed_tmp)
                    emb_ngram.append(emb_ngram_tmp)
                emb_new, _ = self.oe_gate_up_proj(torch.cat(emb_ngram, dim=-1))
                hidden_states = (hidden_states + emb_new) / 2.0

            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        if forward_batch.can_run_tbo:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                if i in self.layers_to_capture:
                    aux_hidden_states.append(
                        hidden_states + residual
                        if residual is not None
                        else hidden_states
                    )
                with get_global_expert_distribution_recorder().with_current_layer(i):
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions, hidden_states, forward_batch, residual
                    )
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class WeLMV4MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self.model = WeLMV4MoeModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            alt_stream=alt_stream,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

        # decoder layer
        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                num_target_layers = LayerManager.num_target_layers
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        if is_nextn:
            # nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            next_layer_prefixes = [
                f"model.layers.{i+num_target_layers}" for i in range(num_nextn_layers)
            ]
            nextn_spec_weight_names = [
                "shared_head.norm",
                "eh_proj",
                "enorm",
                "hnorm",
            ]

        for name, loaded_weight in weights:
            if not is_nextn:
                if hasattr(self.config, "num_nextn_predict_layers"):
                    num_nextn_layers = self.config.num_nextn_predict_layers
                    if num_nextn_layers > 0 and name.startswith("model.layers"):
                        name_list = name.split(".")
                        if (
                            len(name_list) >= 3
                            and int(name_list[2]) >= self.config.num_hidden_layers
                        ):
                            continue
            else:
                flag = False
                matched_prefix = None
                for next_layer_prefix in next_layer_prefixes:
                    if name.startswith(next_layer_prefix):
                        flag = True
                        matched_prefix = next_layer_prefix
                        break
                if not flag:
                    continue
                # if not name.startswith(nextn_layer_prefix):
                #     continue
                # Use shared head and embed weights from target model
                if "shared_head.head" in name or "embed_tokens" in name:
                    continue

                is_decoder = True
                # For nextn specific weights
                for weight_name in nextn_spec_weight_names:
                    if weight_name in name:
                        name = name.replace(matched_prefix, "model")
                        is_decoder = False
                        break
                # For decoder layer weights
                if is_decoder:
                    weight_suffix = int(next_layer_prefix.split(".")[-1])
                    name = name.replace(
                        matched_prefix,
                        f"model.decoder_layers.{weight_suffix-num_target_layers}",
                    )
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "self_attn.gate_proj" in name:
                    continue
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                if name == "model.oe_gate_up_proj.weight":
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        if "attn_sink" in name:
                            start = get_attention_tp_rank() * param.numel()
                            param.data.copy_(
                                loaded_weight[start : start + param.numel()]
                            )
                        else:
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
        self.post_init_after_load_weights(is_nextn=is_nextn)

    def get_embed_and_head(self):
        return [
            self.model.embed_tokens,
            self.model.oe_embed,
            self.model.oe_gate_up_proj,
        ], self.lm_head

    def set_embed_and_head(self, embed, head):
        self.model.embed_tokens = embed[0]
        self.model.oe_embed = embed[1]
        self.model.oe_gate_up_proj = embed[2]
        self.lm_head = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def post_init_after_load_weights(self, is_nextn=False):
        total_kv_mirror_layers = getattr(self.model.config, "kv_mirror_layers", [])
        total_kv_mirror_imitated_layers = getattr(
            self.model.config, "kv_mirror_imitated_layers", []
        )
        if is_nextn:
            kv_mirror_layer_ids = [
                decoder.self_attn.kv_mirror_layer_idx
                for decoder in self.model.decoder_layers
            ]
            kv_mirror_imitated_layers = total_kv_mirror_imitated_layers[
                : len(kv_mirror_layer_ids)
            ]
        else:
            kv_mirror_layer_ids = [
                decoder_layer.self_attn.kv_mirror_layer_idx
                for decoder_layer in self.model.layers
            ]
            kv_mirror_layer_ids = [
                layer_id
                for layer_id in total_kv_mirror_layers
                if layer_id in kv_mirror_layer_ids
            ]  # keep the order of total_kv_mirror_layers
            kv_mirror_imitated_layers = total_kv_mirror_imitated_layers[
                -len(kv_mirror_layer_ids) :
            ]
        LayerManager.post_init(
            kv_mirror_layer_ids, kv_mirror_imitated_layers, is_nextn=is_nextn
        )

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.set_eagle3_layers_to_capture(
                [
                    2,
                    num_layers // 2,
                    num_layers - 3,
                ]
            )  # Specific layers for EAGLE3 support
        else:
            self.model.set_eagle3_layers_to_capture([val + 1 for val in layer_ids])


EntryClass = WeLMV4MoeForCausalLM
