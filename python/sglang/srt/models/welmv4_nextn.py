"""Inference-only WeLMV4 NextN Speculative Decoding."""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.welmv4 import (
    WeLMV4MoeDecoderLayer,
    WeLMV4MoeForCausalLM,
    hash_input_ids_vectorized,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, is_npu

logger = logging.getLogger(__name__)


_is_cuda = is_cuda()
_is_npu = is_npu()


class WeLMV4ModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = None
        self.oe_embed = None
        self.oe_gate_up_proj = None
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

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None

        layer_name = "decoder"
        if _is_npu and (
            get_global_server_args().speculative_draft_model_path
            == get_global_server_args().model_path
        ):
            layer_name = "layers." + str(config.num_hidden_layers)

        self.decoder_layers = nn.ModuleList(
            [
                WeLMV4MoeDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    is_nextn=True,
                    prefix=add_prefix(layer_name, prefix),
                    alt_stream=self.alt_stream,
                )
                for i in range(config.num_nextn_predict_layers)
            ]
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if len(self.oe_grams) > 0:
            input_ids_ngram = []
            input_ids_ngram_tmp = input_ids
            max_n = max(self.oe_grams)
            if getattr(forward_batch, "n_gram_input_ids", None) is not None:
                input_ids_gram_n = []
                for n in range(2, max_n + 1):
                    gram = forward_batch.n_gram_input_ids.get_gram(n)
                    input_ids_gram_n.append(
                        gram if gram is not None else torch.zeros_like(input_ids)
                    )
            else:
                zero_ids = torch.zeros_like(input_ids)
                input_ids_gram_n = [zero_ids for _ in range(max_n - 1)]
            for g in range(1, max_n):
                input_ids_ngram_tmp = input_ids_ngram_tmp + input_ids_gram_n[g - 1] * (
                    self.vocab_size**g
                )
                input_ids_ngram.append(hash_input_ids_vectorized(input_ids_ngram_tmp))

            emb_ngram = []
            for i, vs in enumerate(self.oe_vocab_sizes):
                input_ids_ngram_hashed_tmp = input_ids_ngram[self.oe_grams[i] - 2] % vs
                emb_ngram_tmp = self.oe_embed[i](input_ids_ngram_hashed_tmp)
                emb_ngram.append(emb_ngram_tmp)
            emb_new, _ = self.oe_gate_up_proj(torch.cat(emb_ngram, dim=-1))
            hidden_states = (hidden_states + emb_new) / 2.0

        if (
            forward_batch.enable_kv_mirror
            and forward_batch.forward_mode.is_extend_without_speculative()
        ):
            forward_batch.custom_last_index = (
                torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            )
            hidden_states = hidden_states[forward_batch.custom_last_index]

        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )

        residual = None
        with get_global_expert_distribution_recorder().disable_this_region():
            for layer in self.decoder_layers:
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    forward_batch,
                    residual,
                )

        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)
            else:
                hidden_states = self.shared_head.norm(hidden_states)
        return hidden_states


class WeLMV4MoeForCausalLMNextN(WeLMV4MoeForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        # if not set, model load will be broken in DeepseekV3ForCausalLM load_weights()
        self.pp_group = get_pp_group()

        self.model = WeLMV4ModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = None
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)


EntryClass = WeLMV4MoeForCausalLMNextN
