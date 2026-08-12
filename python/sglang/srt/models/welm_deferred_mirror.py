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

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Optional, Sequence


class WelmPDExecutionMode(str, Enum):
    LEGACY = "legacy"
    DEFERRED_LAST_PROMPT = "deferred-last-prompt"


class WelmDeferredExecutionRole(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MONOLITHIC = "monolithic"


def is_welm_deferred_mirror_enabled(server_args: Any) -> bool:
    return (
        getattr(server_args, "welm_kv_mirror_pd_mode", "legacy")
        == WelmPDExecutionMode.DEFERRED_LAST_PROMPT.value
    )


class DeferredDecodeInputKind(IntEnum):
    TOKEN_ID = 1
    EMBEDDING = 2


@dataclass(frozen=True)
class WelmDeferredMirrorPair:
    source_layer: int
    target_layer: int


@dataclass(frozen=True)
class WelmDeferredMirrorPlan:
    num_hidden_layers: int
    execution_end_layer: int
    pairs: tuple[WelmDeferredMirrorPair, ...]
    fingerprint: str


_WELM_DEFERRED_MODEL_EXECUTION_ATTR = "_sglang_welm_deferred_model_execution"


@dataclass(frozen=True)
class WelmDeferredModelExecution:
    role: WelmDeferredExecutionRole
    plan: WelmDeferredMirrorPlan

    def __post_init__(self) -> None:
        try:
            role = WelmDeferredExecutionRole(self.role)
        except ValueError as exc:
            raise ValueError(
                "WeLM deferred model role must be prefill or decode, or monolithic"
            ) from exc
        object.__setattr__(self, "role", role)

    @property
    def logical_num_layers(self) -> int:
        return self.plan.num_hidden_layers

    @property
    def execution_end_layer(self) -> int:
        return (
            self.plan.execution_end_layer
            if self.role is WelmDeferredExecutionRole.PREFILL
            else self.plan.num_hidden_layers
        )

    @property
    def prefill_execution_end_layer(self) -> int:
        return (
            self.plan.num_hidden_layers
            if self.role is WelmDeferredExecutionRole.DECODE
            else self.plan.execution_end_layer
        )

    @property
    def omitted_layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self.execution_end_layer, self.logical_num_layers))

    @property
    def omit_final_output(self) -> bool:
        return self.role is WelmDeferredExecutionRole.PREFILL


def bind_welm_deferred_model_execution(
    config: Any,
    plan: WelmDeferredMirrorPlan,
    *,
    role: WelmDeferredExecutionRole | str,
) -> WelmDeferredModelExecution:
    execution = WelmDeferredModelExecution(role=role, plan=plan)
    setattr(config, _WELM_DEFERRED_MODEL_EXECUTION_ATTR, execution)
    return execution


def get_welm_deferred_model_execution(
    config: Any,
) -> Optional[WelmDeferredModelExecution]:
    return getattr(config, _WELM_DEFERRED_MODEL_EXECUTION_ATTR, None)


@dataclass(frozen=True)
class WelmDeferredPrefillSpan:
    prompt_len: int
    committed_kv_len: int
    seed_position: int
    seed_token_id: int

    def validate_prompt_tokens(self, prompt_token_ids: Sequence[int]) -> None:
        if len(prompt_token_ids) != self.prompt_len:
            raise RuntimeError(
                "WeLM deferred prompt length changed after span construction: "
                f"expected {self.prompt_len}, got {len(prompt_token_ids)}"
            )
        if int(prompt_token_ids[self.seed_position]) != self.seed_token_id:
            raise RuntimeError("WeLM deferred seed token changed after span construction")

    def committed_token_ids(self, prompt_token_ids: Sequence[int]) -> list[int]:
        self.validate_prompt_tokens(prompt_token_ids)
        return list(prompt_token_ids[: self.committed_kv_len])


def build_welm_deferred_prefill_span(
    prompt_token_ids: Sequence[int],
) -> WelmDeferredPrefillSpan:
    prompt_len = len(prompt_token_ids)
    if prompt_len == 0:
        raise ValueError("WeLM deferred prefill does not support an empty prompt")

    seed_position = prompt_len - 1
    seed_token_id = int(prompt_token_ids[seed_position])
    if seed_token_id < 0:
        raise ValueError("WeLM deferred seed_token_id must be non-negative")
    return WelmDeferredPrefillSpan(
        prompt_len=prompt_len,
        committed_kv_len=seed_position,
        seed_position=seed_position,
        seed_token_id=seed_token_id,
    )


def get_welm_deferred_request_unsupported_reason(req: Any) -> Optional[str]:
    sampling_params = getattr(req, "sampling_params", None)
    if getattr(sampling_params, "max_new_tokens", None) == 0:
        return "zero-generation requests"

    requires_prompt_logprobs = getattr(req, "requires_prompt_logprobs", None)
    if callable(requires_prompt_logprobs) and requires_prompt_logprobs():
        return "prompt logprobs"
    if getattr(req, "return_hidden_states", False):
        return "hidden states"
    if getattr(req, "return_routed_experts", False):
        return "routed experts"
    if getattr(req, "return_indexer_topk", False):
        return "indexer top-k"
    if getattr(req, "input_embeds", None) is not None:
        return "input embeddings"
    if getattr(req, "positional_embed_overrides", None) is not None:
        return "positional embedding overrides"
    if getattr(req, "multimodal_inputs", None) is not None:
        return "multimodal inputs"
    return None


def prepare_welm_deferred_prefill_span(req: Any) -> WelmDeferredPrefillSpan:
    unsupported_reason = get_welm_deferred_request_unsupported_reason(req)
    if unsupported_reason is not None:
        raise ValueError(
            f"WeLM deferred mirror does not support {unsupported_reason}"
        )

    span = build_welm_deferred_prefill_span(req.origin_input_ids)
    existing = getattr(req, "welm_deferred_prefill_span", None)
    if existing is not None:
        if existing != span:
            raise RuntimeError("WeLM deferred prefill span changed after admission")
        return existing
    req.welm_deferred_prefill_span = span
    return span


def _canonical_plan_payload(
    num_hidden_layers: int,
    execution_end_layer: int,
    pairs: tuple[WelmDeferredMirrorPair, ...],
) -> bytes:
    payload = {
        "num_hidden_layers": num_hidden_layers,
        "execution_end_layer": execution_end_layer,
        "pairs": [
            {
                "source_layer": pair.source_layer,
                "target_layer": pair.target_layer,
            }
            for pair in pairs
        ],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def build_welm_deferred_mirror_plan(config: Any) -> WelmDeferredMirrorPlan:
    num_hidden_layers = int(config.num_hidden_layers)
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")

    target_layers = tuple(int(layer) for layer in config.kv_mirror_layers)
    source_layers = tuple(
        int(layer) for layer in config.kv_mirror_imitated_layers
    )
    if len(target_layers) != len(source_layers):
        raise ValueError(
            "kv_mirror_layers and kv_mirror_imitated_layers must have the same length"
        )

    base_pairs = []
    for target_layer, source_layer in zip(target_layers, source_layers):
        if source_layer < 0 or source_layer >= num_hidden_layers:
            raise ValueError(
                f"mirror source layer {source_layer} is outside the base model"
            )
        if target_layer < 0:
            raise ValueError(f"mirror target layer {target_layer} must be non-negative")
        if target_layer >= num_hidden_layers:
            continue
        base_pairs.append(
            WelmDeferredMirrorPair(
                source_layer=source_layer,
                target_layer=target_layer,
            )
        )

    if not base_pairs:
        raise ValueError("deferred mirror mode requires at least one base mirror target")

    base_pairs.sort(key=lambda pair: pair.target_layer)
    targets = [pair.target_layer for pair in base_pairs]
    if len(targets) != len(set(targets)):
        raise ValueError("deferred mirror plan contains a duplicate target layer")

    execution_end_layer = targets[0]
    expected_targets = list(range(execution_end_layer, num_hidden_layers))
    if targets != expected_targets:
        raise ValueError(
            "base mirror targets must form a contiguous suffix ending at "
            f"layer {num_hidden_layers - 1}; got {targets}"
        )

    invalid_sources = [
        pair.source_layer
        for pair in base_pairs
        if pair.source_layer >= execution_end_layer
    ]
    if invalid_sources:
        raise ValueError(
            "every mirror source must be before execution_end_layer "
            f"{execution_end_layer}; got {invalid_sources}"
        )

    pairs = tuple(base_pairs)
    fingerprint = hashlib.sha256(
        _canonical_plan_payload(num_hidden_layers, execution_end_layer, pairs)
    ).hexdigest()
    return WelmDeferredMirrorPlan(
        num_hidden_layers=num_hidden_layers,
        execution_end_layer=execution_end_layer,
        pairs=pairs,
        fingerprint=fingerprint,
    )


def _resolve_attention_backends(server_args: Any) -> tuple[Optional[str], Optional[str]]:
    default_backend = getattr(server_args, "attention_backend", None)
    prefill_backend = (
        getattr(server_args, "prefill_attention_backend", None) or default_backend
    )
    decode_backend = (
        getattr(server_args, "decode_attention_backend", None) or default_backend
    )
    return prefill_backend, decode_backend


def resolve_welm_deferred_mirror_plan(
    server_args: Any,
    model_config: Any,
    *,
    use_previous_precision: bool,
) -> Optional[WelmDeferredMirrorPlan]:
    raw_mode = getattr(server_args, "welm_kv_mirror_pd_mode", "legacy")
    try:
        mode = WelmPDExecutionMode(raw_mode)
    except ValueError as exc:
        choices = ", ".join(item.value for item in WelmPDExecutionMode)
        raise ValueError(
            f"unknown WeLM mirror P/D mode {raw_mode!r}; choose from {choices}"
        ) from exc

    if mode is WelmPDExecutionMode.LEGACY:
        return None

    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(model_config, "hf_text_config", None)
    architectures = tuple(getattr(hf_config, "architectures", ()) or ())
    if architectures != ("WeLMV4MoeForCausalLM",) or getattr(
        model_config, "is_multimodal", False
    ):
        raise ValueError(
            "--enable-kv-mirror-deferred currently requires "
            "the language-only WeLMV4MoeForCausalLM architecture"
        )

    if not getattr(server_args, "enable_welm_kv_mirror_opt", False):
        raise ValueError(
            "--enable-kv-mirror-deferred requires "
            "--enable-welm-kv-mirror-opt"
        )

    role = getattr(server_args, "disaggregation_mode", "null")
    if role not in {"null", "prefill", "decode"}:
        raise ValueError(
            "deferred WeLM mirror mode requires --disaggregation-mode "
            "null, prefill, or decode"
        )
    if role in {"prefill", "decode"} and getattr(
        server_args, "disaggregation_transfer_backend", None
    ) != "mooncake":
        raise ValueError(
            "deferred WeLM mirror mode currently requires "
            "--disaggregation-transfer-backend mooncake"
        )
    if getattr(server_args, "enable_mixed_chunk", False):
        raise ValueError(
            "deferred WeLM mirror mode does not support mixed chunk "
            "(--enable-mixed-chunk)"
        )

    prefill_backend, decode_backend = _resolve_attention_backends(server_args)
    if prefill_backend != "fa3" or decode_backend != "fa3":
        raise ValueError(
            "deferred WeLM mirror mode currently requires FA3 for both prefill "
            f"and decode; got prefill={prefill_backend!r}, decode={decode_backend!r}"
        )
    if getattr(server_args, "pp_size", 1) != 1:
        raise ValueError(
            "deferred WeLM mirror mode does not support pipeline parallelism"
        )
    if getattr(server_args, "speculative_algorithm", None) is not None:
        raise ValueError(
            "deferred WeLM mirror mode does not support speculative decoding"
        )
    if getattr(server_args, "enable_hierarchical_cache", False):
        raise ValueError("deferred WeLM mirror mode does not support HiCache")
    if getattr(server_args, "disaggregation_decode_enable_offload_kvcache", False):
        raise ValueError("deferred WeLM mirror mode does not support KV offload")
    if getattr(server_args, "enable_lora", False):
        raise ValueError("deferred WeLM mirror mode does not support LoRA")
    if getattr(server_args, "enable_suffix_parallel", False):
        raise ValueError("deferred WeLM mirror mode does not support suffix parallel")
    scale_seq_times = max(
        getattr(hf_config, "scale_seq_times", 0) or 0,
        getattr(text_config, "scale_seq_times", 0) or 0,
    )
    if scale_seq_times > 0:
        raise ValueError("deferred WeLM mirror mode does not support Scale-Seq")
    if getattr(server_args, "kv_cache_dtype", "auto") not in {
        "auto",
        "bf16",
        "bfloat16",
    }:
        raise ValueError(
            "deferred WeLM mirror mode requires a BF16 KV cache dtype"
        )
    if use_previous_precision:
        raise ValueError(
            "deferred WeLM mirror mode does not support previous-precision execution"
        )

    if (
        role == "decode"
        and getattr(server_args, "enable_dp_attention", False)
        and getattr(server_args, "attn_cp_mode", "none") == "sharded-kv"
        and getattr(server_args, "attn_cp_size", 1) > 1
    ):
        raise ValueError(
            "deferred WeLM mirror mode does not support Decode DP attention "
            "combined with sharded-KV CP"
        )

    if text_config is None:
        raise ValueError("WeLM model config is missing hf_text_config")
    return build_welm_deferred_mirror_plan(text_config)
