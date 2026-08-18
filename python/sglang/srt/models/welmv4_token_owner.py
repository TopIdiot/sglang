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

import enum
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.distributed.parallel_state import get_attn_tp_group, get_tp_group
from sglang.srt.layers.communicator import TokenOwnerLayout
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_cp_size,
    get_attention_tp_size,
    is_dp_attention_enabled,
    is_suffix_parallel_enabled,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.mk_moe_router import (
    MkMoeRouterMode,
    get_mk_moe_router_mode,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.welmv4_op import welm_use_previous_precision
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import get_global_server_args


_WELM_TOKEN_OWNER_PREFILL_MODES = frozenset(
    (
        ForwardMode.EXTEND,
        ForwardMode.MIXED,
        ForwardMode.SPLIT_PREFILL,
        ForwardMode.DLLM_EXTEND,
    )
)


def welm_token_owner_forward_phase(forward_batch) -> str:
    forward_mode = forward_batch.forward_mode
    if getattr(forward_batch, "welm_mtp_variable_decode_extend", False):
        if forward_mode not in (ForwardMode.EXTEND, ForwardMode.IDLE):
            raise RuntimeError(
                "WeLM MTP variable decode extend has incompatible forward mode "
                f"{forward_mode!r}"
            )
        return "decode"
    if forward_mode in (
        ForwardMode.DECODE,
        ForwardMode.IDLE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.DRAFT_EXTEND,
        ForwardMode.DRAFT_EXTEND_V2,
    ):
        return "decode"
    if forward_mode in _WELM_TOKEN_OWNER_PREFILL_MODES:
        return "prefill"
    raise RuntimeError(
        f"WeLM token-owner does not support forward mode {forward_mode!r}"
    )


def welm_token_owner_enabled_for_forward(
    forward_batch, *, decode_token_owner_enabled: bool
) -> bool:
    if decode_token_owner_enabled:
        return True
    if welm_token_owner_forward_phase(forward_batch) == "prefill":
        return True
    global_forward_modes = getattr(forward_batch, "global_forward_modes", None) or ()
    return any(
        ForwardMode(mode) in _WELM_TOKEN_OWNER_PREFILL_MODES
        for mode in global_forward_modes
    )


def needs_input_logprobs(forward_batch) -> bool:
    if not getattr(forward_batch, "return_logprob", False):
        return False
    extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    start_lens = getattr(forward_batch, "extend_logprob_start_lens_cpu", None)
    return bool(
        extend_lens is not None
        and start_lens is not None
        and any(
            int(extend_len) - int(start_len) > 0
            for extend_len, start_len in zip(extend_lens, start_lens)
        )
    )


def validate_welm_token_owner_capability(
    *,
    enabled: bool,
    dp_attention_enabled: bool,
    dp_size: int,
    attn_cp_size: int,
    attn_tp_size: int,
    pp_size: int,
    global_tp_size: int,
    global_tp_rank: int,
    attn_tp_rank: int,
    moe_a2a_backend: str,
    use_previous_precision: bool,
    suffix_parallel_enabled: bool,
    tbo_enabled: bool,
    router_replay_enabled: bool,
    mk_router_enabled: bool,
    speculative_enabled: bool,
    speculative_moe_a2a_backend: Optional[str],
    deepep_mode: str,
    disaggregation_mode: str,
    attn_tp_input_scattered: bool,
    activation_dump_enabled: bool = False,
    routed_expert_capture_enabled: bool = False,
    expert_distribution_recorder_enabled: bool = False,
    pdmux_enabled: bool = False,
) -> bool:
    if not enabled:
        return False
    requirements = (
        (attn_cp_size == 1, "currently requires AttnCP1"),
        (attn_tp_size > 1, "currently requires AttnTP>1"),
        (pp_size == 1, "currently requires PP1"),
        (
            moe_a2a_backend in ("none", "deepep"),
            "currently requires MoE A2A backend none or DeepEP",
        ),
        (not use_previous_precision, "does not support previous-precision"),
        (not suffix_parallel_enabled, "does not support suffix parallel"),
        (not tbo_enabled, "does not support TBO"),
        (not router_replay_enabled, "does not support Router Replay"),
        (not mk_router_enabled, "does not support MK MoE Router"),
        (not attn_tp_input_scattered, "does not compose with AttnTP input scatter"),
        (not pdmux_enabled, "does not support PDMux"),
        (
            not activation_dump_enabled,
            "activation dump does not support owner-local routing",
        ),
        (
            not routed_expert_capture_enabled,
            "routed-expert capture does not support owner-local routing",
        ),
        (
            not expert_distribution_recorder_enabled,
            "expert distribution recorder does not support owner-local routing",
        ),
    )
    for supported, message in requirements:
        if not supported:
            raise NotImplementedError(f"WeLM token-owner {message}")
    if global_tp_size % attn_tp_size != 0:
        raise RuntimeError("Global TP size must be divisible by AttnTP size")
    if not dp_attention_enabled:
        pure_tp_requirements = (
            (dp_size == 1, "pure-TP requires DP=1"),
            (
                global_tp_size == attn_tp_size,
                "pure-TP requires a single owner group",
            ),
        )
        for supported, message in pure_tp_requirements:
            if not supported:
                raise NotImplementedError(f"WeLM token-owner {message}")
    if attn_tp_rank != global_tp_rank % attn_tp_size:
        raise RuntimeError(
            "Global TP and AttnTP ranks do not use DP-major rank ordering"
        )
    if (
        moe_a2a_backend == "deepep"
        and deepep_mode == "low_latency"
        and disaggregation_mode != "decode"
    ):
        raise NotImplementedError(
            "WeLM token-owner DeepEP low_latency mode is decode-only; "
            "use auto for non-disaggregated prefill/decode"
        )
    if speculative_enabled:
        speculative_backend = speculative_moe_a2a_backend or moe_a2a_backend
        valid_backend = (
            moe_a2a_backend == speculative_backend == "none"
            or moe_a2a_backend == speculative_backend == "deepep"
        )
        if not valid_backend:
            raise NotImplementedError(
                "WeLM token-owner speculative decoding requires Global TP MoE "
                "or matching DeepEP target and draft backends"
            )
        if moe_a2a_backend == "deepep" and deepep_mode not in (
            "auto",
            "low_latency",
        ):
            raise NotImplementedError(
                "WeLM token-owner speculative DeepEP requires deepep-mode "
                "auto or low_latency"
            )
    return True


def welm_token_owner_enabled(*, pp_size: int, activation_dump_enabled: bool) -> bool:
    server_args = get_global_server_args()
    enabled = bool(getattr(server_args, "enable_token_owner", False))
    if not enabled:
        return False
    global_tp_group = get_tp_group()
    attn_tp_group = get_attn_tp_group()
    return validate_welm_token_owner_capability(
        enabled=True,
        dp_attention_enabled=is_dp_attention_enabled(),
        dp_size=server_args.dp_size,
        attn_cp_size=get_attention_cp_size(),
        attn_tp_size=get_attention_tp_size(),
        pp_size=pp_size,
        global_tp_size=global_tp_group.world_size,
        global_tp_rank=global_tp_group.rank_in_group,
        attn_tp_rank=attn_tp_group.rank_in_group,
        moe_a2a_backend=get_moe_a2a_backend().value,
        use_previous_precision=welm_use_previous_precision(),
        suffix_parallel_enabled=is_suffix_parallel_enabled(),
        tbo_enabled=server_args.enable_two_batch_overlap,
        router_replay_enabled=server_args.enable_moe_router_replay,
        mk_router_enabled=get_mk_moe_router_mode() is not MkMoeRouterMode.OFF,
        speculative_enabled=server_args.speculative_algorithm is not None,
        speculative_moe_a2a_backend=server_args.speculative_moe_a2a_backend,
        deepep_mode=server_args.deepep_mode,
        disaggregation_mode=server_args.disaggregation_mode,
        attn_tp_input_scattered=server_args.enable_attn_tp_input_scattered,
        activation_dump_enabled=activation_dump_enabled,
        routed_expert_capture_enabled=server_args.enable_return_routed_experts,
        expert_distribution_recorder_enabled=(
            server_args.expert_distribution_recorder_mode is not None
        ),
        pdmux_enabled=server_args.enable_pdmux,
    )


def kv_mirror_owner_sizes(
    *,
    original_token_count: int,
    owner_count: int,
    last_q_indices: Sequence[int],
    active_batch_indices: Sequence[int],
    output_size: int,
) -> tuple[int, ...]:
    original_token_count = int(original_token_count)
    output_size = int(output_size)
    last_q_indices = tuple(map(int, last_q_indices))
    active_batch_indices = tuple(map(int, active_batch_indices))
    if original_token_count < 0 or output_size < 0 or owner_count <= 0:
        raise ValueError("invalid KV mirror owner dimensions")
    if len(last_q_indices) != len(active_batch_indices):
        raise RuntimeError("KV mirror last-Q and active-request metadata must align")
    if any(not 0 <= index < output_size for index in active_batch_indices) or any(
        current >= following
        for current, following in zip(
            active_batch_indices, active_batch_indices[1:]
        )
    ):
        raise RuntimeError("KV mirror active-request indices must be ordered")
    if any(not 0 <= index < original_token_count for index in last_q_indices):
        raise RuntimeError("KV mirror last-Q index is outside the original rows")

    original_layout = TokenOwnerLayout.balanced(
        valid_token_count=original_token_count,
        owner_count=owner_count,
        local_owner_rank=0,
    )
    owner_ends = (*original_layout.owner_offsets[1:], original_token_count)
    active_owners = []
    owner = 0
    for last_q_index in last_q_indices:
        while owner < owner_count - 1 and last_q_index >= owner_ends[owner]:
            owner += 1
        active_owners.append(owner)
    if any(
        current > following
        for current, following in zip(active_owners, active_owners[1:])
    ):
        raise RuntimeError("KV mirror survivor owners are not monotonic")

    boundaries = [0]
    survivor = 0
    for next_owner in range(1, owner_count):
        while survivor < len(active_owners) and active_owners[survivor] < next_owner:
            survivor += 1
        boundaries.append(
            active_batch_indices[survivor]
            if survivor < len(active_batch_indices)
            else output_size
        )
    boundaries.append(output_size)
    return tuple(
        following - current
        for current, following in zip(boundaries, boundaries[1:])
    )


def _validate_rows(tensor: torch.Tensor, expected: int, name: str) -> None:
    rows = 0 if tensor.ndim == 0 else tensor.shape[0]
    if rows != expected:
        raise RuntimeError(f"{name} has {rows} rows, expected {expected}")


@dataclass(frozen=True)
class GlobalTPRouterContext:
    gathers_hidden_states = True

    layout: TokenOwnerLayout
    global_tp_group: object
    local_layout: Optional[TokenOwnerLayout] = None
    attn_tp_group: Optional[object] = None

    def __post_init__(self):
        if (
            self.global_tp_group.world_size != len(self.layout.owner_sizes)
            or self.global_tp_group.rank_in_group != self.layout.local_owner_rank
        ):
            raise RuntimeError("token-owner Router group does not match its layout")
        if self.local_layout is not None and (
            self.attn_tp_group is None
            or self.attn_tp_group.world_size != len(self.local_layout.owner_sizes)
            or self.attn_tp_group.rank_in_group
            != self.local_layout.local_owner_rank
        ):
            raise RuntimeError("token-owner proxy group does not match its layout")

    def local_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        layout = self.local_layout or self.layout
        _validate_rows(tensor, layout.local_valid_rows, "owner-local Router input")
        return tensor

    def prepare_moe_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_output: StandardTopKOutput,
    ) -> tuple[torch.Tensor, StandardTopKOutput]:
        self.local_rows(hidden_states)
        if not isinstance(topk_output, StandardTopKOutput):
            raise RuntimeError("token-owner requires Standard TopK output")
        self.local_rows(topk_output.topk_weights)
        self.local_rows(topk_output.topk_ids)
        inputs = [hidden_states, topk_output.topk_weights, topk_output.topk_ids]
        if self.local_layout is not None:
            local_size = len(self.local_layout.owner_sizes)
            group_start = (self.layout.local_owner_rank // local_size) * local_size
            if (
                self.layout.owner_sizes[group_start : group_start + local_size]
                != self.local_layout.owner_sizes
            ):
                inputs = self.attn_tp_group.all_gatherv(
                    inputs, sizes=list(self.local_layout.owner_sizes)
                )
                if not isinstance(inputs, list) or len(inputs) != 3:
                    raise RuntimeError(
                        "token-owner proxy gather returned invalid output"
                    )
                if self.local_layout.local_owner_rank != 0:
                    inputs = [tensor[:0] for tensor in inputs]

        gathered = self.global_tp_group.all_gatherv(
            inputs, sizes=list(self.layout.owner_sizes)
        )
        if not isinstance(gathered, list) or len(gathered) != 3:
            raise RuntimeError(
                "token-owner expert-input gather returned invalid output"
            )
        full_hidden, full_weights, full_ids = gathered
        for tensor, name in (
            (full_hidden, "global expert input"),
            (full_weights, "global top-k weights"),
            (full_ids, "global top-k ids"),
        ):
            _validate_rows(tensor, self.layout.valid_token_count, name)
        return full_hidden, StandardTopKOutput(
            topk_weights=full_weights,
            topk_ids=full_ids,
            router_logits=topk_output.router_logits.new_empty(
                (self.layout.valid_token_count, 0)
            ),
        )


@dataclass(frozen=True)
class DeepEPRouterContext:
    gathers_hidden_states = False

    layout: TokenOwnerLayout
    valid_local_mask: torch.Tensor

    def __post_init__(self):
        if (
            self.valid_local_mask.dtype != torch.bool
            or self.valid_local_mask.ndim != 1
            or self.valid_local_mask.shape[0] != self.layout.local_valid_rows
        ):
            raise RuntimeError("invalid DeepEP token-owner valid-row mask")

    def local_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        _validate_rows(
            tensor, self.layout.local_valid_rows, "DeepEP owner-local Router input"
        )
        return tensor

    def prepare_moe_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_output: StandardTopKOutput,
    ) -> tuple[torch.Tensor, StandardTopKOutput]:
        self.local_rows(hidden_states)
        if not isinstance(topk_output, StandardTopKOutput):
            raise RuntimeError("DeepEP token-owner requires Standard TopK output")
        self.local_rows(topk_output.topk_weights)
        self.local_rows(topk_output.topk_ids)
        if self.valid_local_mask.device != topk_output.topk_ids.device:
            raise RuntimeError(
                "DeepEP token-owner valid-row mask is on the wrong device"
            )
        invalid = ~self.valid_local_mask[:, None]
        return hidden_states, StandardTopKOutput(
            topk_weights=topk_output.topk_weights.masked_fill(invalid, 0),
            topk_ids=topk_output.topk_ids.masked_fill(invalid, -1),
            router_logits=topk_output.router_logits,
        )


class TokenOwnerLayerIdentity(enum.IntEnum):
    PRE_MIRROR = 0
    MIRROR_BOUNDARY = 1
    MIRROR_TAIL = 2


class TokenOwnerTransition(enum.Enum):
    NONE = enum.auto()
    CONTRACT_KEEP_OWNER = enum.auto()
    CONTRACT_EXIT_OWNER = enum.auto()


def resolve_token_owner_transition(
    forward_batch,
    *,
    has_mirror_boundary: bool,
    exit_owner: bool,
) -> TokenOwnerTransition:
    if (
        not has_mirror_boundary
        or not getattr(forward_batch, "enable_welm_kv_mirror_opt", False)
        or getattr(forward_batch, "welm_mtp_merge_kv_fill_draft", False)
        or getattr(forward_batch, "welm_mtp_variable_decode_extend", False)
    ):
        return TokenOwnerTransition.NONE

    flags = getattr(forward_batch, "welm_kv_mirror_contract_flags", None)
    modes = getattr(forward_batch, "global_forward_modes", None)
    if flags is None:
        mode = getattr(forward_batch, "forward_mode", None)
        contracts = bool(
            mode is not None
            and ForwardMode(mode).is_extend_without_speculative()
            and not needs_input_logprobs(forward_batch)
        )
        flags = (contracts,)
        modes = (mode,)
    else:
        flags = tuple(map(bool, flags))
        if modes is None or len(modes) != len(flags):
            raise RuntimeError(
                "WeLM Prefill mirror Token Owner requires one forward mode per "
                "contraction flag"
            )
    if not any(flags):
        return TokenOwnerTransition.NONE
    if any(
        contracts and not ForwardMode(mode).is_extend_without_speculative()
        for mode, contracts in zip(modes, flags, strict=True)
    ):
        raise RuntimeError(
            "WeLM Prefill mirror Token Owner only supports ordinary Prefill"
        )
    return (
        TokenOwnerTransition.CONTRACT_EXIT_OWNER
        if exit_owner
        else TokenOwnerTransition.CONTRACT_KEEP_OWNER
    )


def classify_token_owner_layers(
    *,
    local_start_layer: int,
    local_end_layer: int,
    execution_start_layer: int,
    execution_end_layer: int,
    mirror_targets: Sequence[int],
) -> tuple[Optional[int], tuple[TokenOwnerLayerIdentity, ...]]:
    if not (
        0 <= execution_start_layer <= execution_end_layer
        and execution_start_layer <= local_start_layer <= local_end_layer
        and local_end_layer <= execution_end_layer
    ):
        raise ValueError("invalid token-owner layer execution range")
    boundary = next(
        (
            target
            for target in sorted(set(map(int, mirror_targets)))
            if execution_start_layer <= target < execution_end_layer
        ),
        None,
    )
    identities = tuple(
        TokenOwnerLayerIdentity.PRE_MIRROR
        if boundary is None or layer_id < boundary
        else TokenOwnerLayerIdentity.MIRROR_BOUNDARY
        if layer_id == boundary
        else TokenOwnerLayerIdentity.MIRROR_TAIL
        for layer_id in range(local_start_layer, local_end_layer)
    )
    return boundary, identities


@dataclass(frozen=True, slots=True)
class TokenOwnerLayerState:
    transitions_rows: bool = False
    owner_input: bool = False
    owner_output: bool = False
    input_local_layout: Optional[TokenOwnerLayout] = None
    output_local_layout: Optional[TokenOwnerLayout] = None
    output_global_layout: Optional[TokenOwnerLayout] = None
    transport_local_layout: Optional[TokenOwnerLayout] = None
    local_contracts_rows: bool = False
    survivor_source_rows: tuple[int, ...] = ()
    survivor_output_rows: tuple[int, ...] = ()
    router_context: Optional[GlobalTPRouterContext | DeepEPRouterContext] = None


@dataclass(frozen=True, slots=True)
class TokenOwnerForwardPlan:
    transition: TokenOwnerTransition
    uses_dp_transport: bool
    layer_states: tuple[
        TokenOwnerLayerState,
        TokenOwnerLayerState,
        TokenOwnerLayerState,
    ]
    post_global_num_tokens_cpu: tuple[int, ...]
    post_global_num_tokens_for_logprob_cpu: Optional[tuple[int, ...]]
    post_global_dp_buffer_len: Optional[int]
    post_local_num_token_non_padded: Optional[int]
    post_dp_padding_mode: Optional[DpPaddingMode]
    row_alignment: int

    def state_for(self, identity: TokenOwnerLayerIdentity) -> TokenOwnerLayerState:
        return self.layer_states[int(identity)]


@dataclass(slots=True)
class _CountStaging:
    tensor: torch.Tensor
    ready_event: torch.cuda.Event


class WeLMTokenOwnerRuntime:
    def __init__(
        self,
        *,
        attn_tp_group=None,
        global_tp_group=None,
        disable_prefill_mirror_token_owner: bool = False,
    ):
        self.attn_tp_group = (
            get_attn_tp_group() if attn_tp_group is None else attn_tp_group
        )
        self.global_tp_group = (
            get_tp_group() if global_tp_group is None else global_tp_group
        )
        self.disable_prefill_mirror_token_owner = (
            disable_prefill_mirror_token_owner
        )
        self._forward_batch = None
        self._plan: Optional[TokenOwnerForwardPlan] = None
        self._count_staging: dict[str, list[_CountStaging]] = {}

    def begin_forward(
        self,
        forward_batch,
        *,
        single_group_physical_rows: Optional[int],
        transition: TokenOwnerTransition,
        device: torch.device,
    ) -> TokenOwnerForwardPlan:
        self._forward_batch = forward_batch
        owner_group_count = self._owner_group_count()
        if single_group_physical_rows is not None:
            if owner_group_count != 1:
                raise RuntimeError(
                    "single-group physical rows require one token-owner group"
                )
            pre_counts = (int(single_group_physical_rows),)
        else:
            counts = getattr(forward_batch, "global_num_tokens_cpu", None)
            if counts is None:
                raise RuntimeError(
                    "WeLM token-owner requires CPU global token counts"
                )
            pre_counts = tuple(map(int, counts))
        if not pre_counts or any(count < 0 for count in pre_counts):
            raise RuntimeError("WeLM token-owner received invalid global token counts")
        if len(pre_counts) != owner_group_count:
            raise RuntimeError(
                "WeLM token-owner group sizes do not match token counts"
            )
        flags_source = getattr(forward_batch, "welm_kv_mirror_contract_flags", None)
        if flags_source is not None and len(flags_source) != owner_group_count:
            raise RuntimeError(
                "WeLM token-owner mirror contraction requires one flag per owner "
                "group"
            )

        if transition is TokenOwnerTransition.NONE:
            post_counts = pre_counts
            post_logprob_counts = self._optional_counts(
                forward_batch, "global_num_tokens_for_logprob_cpu", len(pre_counts)
            )
            row_alignment = 1
        else:
            if single_group_physical_rows is not None:
                original_counts = pre_counts
                flags = (True,)
                output_size = getattr(
                    forward_batch, "welm_kv_mirror_output_size", None
                )
                if output_size is None:
                    raise RuntimeError(
                        "WeLM token-owner mirror contraction requires scheduler "
                        "output size"
                    )
                request_counts = (int(output_size),)
            else:
                original_counts = self._required_counts(
                    forward_batch,
                    "original_global_num_tokens_cpu",
                    owner_group_count,
                )
                if flags_source is None:
                    raise RuntimeError(
                        "WeLM token-owner mirror contraction requires one flag per "
                        "owner group"
                    )
                flags = tuple(map(bool, flags_source))
                request_counts = self._required_counts(
                    forward_batch, "global_num_reqs_cpu", owner_group_count
                )
            row_alignment = (
                self.attn_tp_group.world_size
                if transition is TokenOwnerTransition.CONTRACT_EXIT_OWNER
                and get_moe_a2a_backend().is_deepep()
                else 1
            )
            post_counts = tuple(
                self._ceil_align(request_count, row_alignment)
                if contracts
                else pre_count
                for pre_count, request_count, contracts in zip(
                    pre_counts, request_counts, flags, strict=True
                )
            )
            if any(
                contracts
                and request_count > pre_count
                for pre_count, request_count, contracts in zip(
                    pre_counts, request_counts, flags, strict=True
                )
            ):
                raise RuntimeError(
                    "WeLM Prefill mirror survivor count exceeds the input row "
                    "domain"
                )
            current_logprob_counts = self._optional_counts(
                forward_batch, "global_num_tokens_for_logprob_cpu", owner_group_count
            )
            post_logprob_counts = tuple(
                request_count
                if contracts
                else (
                    current_logprob_counts[index]
                    if current_logprob_counts is not None
                    else pre_counts[index]
                )
                for index, (request_count, contracts) in enumerate(
                    zip(request_counts, flags, strict=True)
                )
            )

        local_group = self._owner_group_index()
        pre_local_layout = TokenOwnerLayout.balanced(
            valid_token_count=pre_counts[local_group],
            owner_count=self.attn_tp_group.world_size,
            local_owner_rank=self.attn_tp_group.rank_in_group,
        )
        pre_global_layout = self._global_layout(pre_counts, None)
        pre_context = self._router_context(
            pre_local_layout, pre_global_layout, device=device, valid_mask=None
        )
        pre_state = TokenOwnerLayerState(
            owner_input=True,
            owner_output=True,
            input_local_layout=pre_local_layout,
            output_local_layout=pre_local_layout,
            output_global_layout=pre_global_layout,
            router_context=pre_context,
        )

        if transition is TokenOwnerTransition.NONE:
            states = (pre_state, pre_state, pre_state)
        else:
            local_contracts = flags[local_group]
            keep_owner = transition is TokenOwnerTransition.CONTRACT_KEEP_OWNER
            post_local_layout = TokenOwnerLayout.balanced(
                valid_token_count=post_counts[local_group],
                owner_count=self.attn_tp_group.world_size,
                local_owner_rank=self.attn_tp_group.rank_in_group,
            )
            if keep_owner and local_contracts:
                post_local_layout = TokenOwnerLayout.from_owner_sizes(
                    kv_mirror_owner_sizes(
                        original_token_count=original_counts[local_group],
                        owner_count=self.attn_tp_group.world_size,
                        last_q_indices=self._required_indices(
                            forward_batch, "welm_kv_mirror_last_q_indices_cpu"
                        ),
                        active_batch_indices=self._required_indices(
                            forward_batch,
                            "welm_kv_mirror_active_batch_indices_cpu",
                        ),
                        output_size=post_counts[local_group],
                    ),
                    local_owner_rank=self.attn_tp_group.rank_in_group,
                )
            source_rows, output_rows = self._survivor_mapping(
                forward_batch,
                pre_local_layout,
                post_local_layout,
                owner_local=keep_owner,
                enabled=local_contracts,
            )
            if keep_owner:
                post_global_layout = self._global_layout(
                    post_counts,
                    flags,
                    single_group_local_layout=(
                        post_local_layout if owner_group_count == 1 else None
                    ),
                )
                post_context = self._router_context(
                    post_local_layout,
                    post_global_layout,
                    device=device,
                    valid_mask=self._build_valid_mask(
                        post_local_layout, output_rows, device=device
                    )
                    if local_contracts
                    else None,
                )
                boundary = TokenOwnerLayerState(
                    transitions_rows=True,
                    owner_input=True,
                    owner_output=True,
                    input_local_layout=pre_local_layout,
                    output_local_layout=post_local_layout,
                    output_global_layout=post_global_layout,
                    local_contracts_rows=local_contracts,
                    survivor_source_rows=source_rows,
                    survivor_output_rows=output_rows,
                    router_context=post_context,
                )
                tail = TokenOwnerLayerState(
                    owner_input=True,
                    owner_output=True,
                    input_local_layout=post_local_layout,
                    output_local_layout=post_local_layout,
                    output_global_layout=post_global_layout,
                    router_context=post_context,
                )
            else:
                transport_local_layout = (
                    post_local_layout
                    if single_group_physical_rows is not None and row_alignment > 1
                    else None
                )
                boundary = TokenOwnerLayerState(
                    transitions_rows=True,
                    owner_input=True,
                    input_local_layout=pre_local_layout,
                    transport_local_layout=transport_local_layout,
                    local_contracts_rows=local_contracts,
                    survivor_source_rows=source_rows,
                    survivor_output_rows=output_rows,
                )
                tail = TokenOwnerLayerState(
                    transport_local_layout=transport_local_layout,
                )
            states = (pre_state, boundary, tail)

        post_padding_mode = getattr(forward_batch, "dp_padding_mode", None)
        if transition is not TokenOwnerTransition.NONE and owner_group_count > 1:
            post_padding_mode = DpPaddingMode.SUM_LEN
        post_local_num_token_non_padded = None
        if transition is not TokenOwnerTransition.NONE:
            real_local_rows = (
                post_logprob_counts[local_group]
                if post_logprob_counts is not None
                else post_counts[local_group]
            )
            local_real_start = post_local_layout.owner_offsets[
                post_local_layout.local_owner_rank
            ]
            post_local_num_token_non_padded = min(
                max(real_local_rows - local_real_start, 0),
                post_local_layout.local_valid_rows,
            )
        if post_padding_mode is DpPaddingMode.SUM_LEN:
            post_global_dp_buffer_len = sum(post_counts)
        elif post_padding_mode is DpPaddingMode.MAX_LEN:
            post_global_dp_buffer_len = max(post_counts) * len(post_counts)
        elif (
            transition is not TokenOwnerTransition.NONE
            and getattr(forward_batch, "global_num_tokens_gpu", None) is not None
        ):
            post_global_dp_buffer_len = sum(post_counts)
        else:
            post_global_dp_buffer_len = None
        plan = TokenOwnerForwardPlan(
            transition=transition,
            uses_dp_transport=single_group_physical_rows is None,
            layer_states=states,
            post_global_num_tokens_cpu=post_counts,
            post_global_num_tokens_for_logprob_cpu=post_logprob_counts,
            post_global_dp_buffer_len=post_global_dp_buffer_len,
            post_local_num_token_non_padded=post_local_num_token_non_padded,
            post_dp_padding_mode=post_padding_mode,
            row_alignment=row_alignment,
        )
        self._plan = plan
        return plan

    def state_for(
        self,
        forward_batch,
        identity: TokenOwnerLayerIdentity,
    ) -> TokenOwnerLayerState:
        if self._forward_batch is not forward_batch:
            raise RuntimeError("token-owner plan belongs to a different forward batch")
        if self._plan is None:
            raise RuntimeError("token-owner forward plan has not been built")
        return self._plan.state_for(identity)

    def publish_boundary_transport(
        self,
        forward_batch,
        boundary_state: TokenOwnerLayerState,
    ) -> None:
        plan = self._require_boundary_state(forward_batch, boundary_state)
        if boundary_state.local_contracts_rows:
            output_size = getattr(forward_batch, "kv_mirror_output_size", None)
            expected_size = plan.post_global_num_tokens_cpu[
                self._owner_group_index()
            ]
            if output_size is None or int(output_size) != expected_size:
                raise RuntimeError(
                    "WeLM Token Owner mirror output rows do not match the forward "
                    "plan"
                )
        counts = list(plan.post_global_num_tokens_cpu)
        logprob_counts = (
            list(plan.post_global_num_tokens_for_logprob_cpu)
            if plan.post_global_num_tokens_for_logprob_cpu is not None
            else None
        )

        if plan.uses_dp_transport:
            self._copy_counts(
                getattr(forward_batch, "global_num_tokens_gpu", None),
                counts,
                staging_slot="tokens",
            )
            if logprob_counts is not None:
                self._copy_counts(
                    getattr(
                        forward_batch, "global_num_tokens_for_logprob_gpu", None
                    ),
                    logprob_counts,
                    staging_slot="logprobs",
                )
            forward_batch.global_num_tokens_cpu = counts
            if logprob_counts is not None:
                forward_batch.global_num_tokens_for_logprob_cpu = logprob_counts
            forward_batch.dp_padding_mode = plan.post_dp_padding_mode
            forward_batch.global_dp_buffer_len = plan.post_global_dp_buffer_len
            forward_batch.dp_local_start_pos = None
            forward_batch.dp_local_num_tokens = None

        local_non_padded = plan.post_local_num_token_non_padded
        non_padded_tensor = getattr(forward_batch, "num_token_non_padded", None)
        if local_non_padded is not None:
            if non_padded_tensor is not None:
                if plan.uses_dp_transport:
                    forward_batch.num_token_non_padded = non_padded_tensor.new_tensor(
                        local_non_padded
                    )
                else:
                    non_padded_tensor.fill_(local_non_padded)
            forward_batch.num_token_non_padded_cpu = local_non_padded

        if plan.uses_dp_transport and plan.post_global_dp_buffer_len is not None:
            local_group = self._owner_group_index()
            local_buffer_len = (
                max(counts)
                if plan.post_dp_padding_mode is DpPaddingMode.MAX_LEN
                else counts[local_group]
            )
            set_dp_buffer_len(
                plan.post_global_dp_buffer_len,
                local_buffer_len,
                plan.post_dp_padding_mode is DpPaddingMode.MAX_LEN,
                counts,
            )
        if plan.uses_dp_transport:
            set_is_extend_in_batch(
                forward_batch.is_extend_in_batch
                and not getattr(
                    forward_batch, "_welm_force_low_latency_deepep", False
                )
            )

    def contract_boundary_residual(
        self,
        residual: torch.Tensor,
        boundary_state: TokenOwnerLayerState,
        *,
        output_scattered: bool,
    ) -> torch.Tensor:
        plan = self._require_boundary_state(self._forward_batch, boundary_state)

        input_layout = boundary_state.input_local_layout
        expected_input_rows = (
            input_layout.local_valid_rows
            if boundary_state.owner_output
            else input_layout.valid_token_count
        )
        _validate_rows(residual, expected_input_rows, "mirror boundary residual")

        if boundary_state.local_contracts_rows:
            output_rows = (
                boundary_state.output_local_layout.local_valid_rows
                if boundary_state.owner_output
                else plan.post_global_num_tokens_cpu[
                    self._owner_group_index()
                ]
            )
            output = residual.new_zeros((output_rows, *residual.shape[1:]))
            if boundary_state.survivor_source_rows:
                source = torch.tensor(
                    boundary_state.survivor_source_rows,
                    dtype=torch.long,
                    device=residual.device,
                )
                destination = torch.tensor(
                    boundary_state.survivor_output_rows,
                    dtype=torch.long,
                    device=residual.device,
                )
                output.index_copy_(0, destination, residual.index_select(0, source))
        else:
            output = residual

        if output_scattered:
            output = output.tensor_split(self.attn_tp_group.world_size)[
                self.attn_tp_group.rank_in_group
            ]
        return output

    def _require_boundary_state(
        self,
        forward_batch,
        boundary_state: TokenOwnerLayerState,
    ) -> TokenOwnerForwardPlan:
        if self._forward_batch is not forward_batch or self._plan is None:
            raise RuntimeError("token-owner plan belongs to a different forward batch")
        if (
            boundary_state
            is not self._plan.state_for(TokenOwnerLayerIdentity.MIRROR_BOUNDARY)
            or self._plan.transition is TokenOwnerTransition.NONE
        ):
            raise RuntimeError("invalid Token Owner mirror boundary state")
        return self._plan

    def _copy_counts(
        self,
        destination: Optional[torch.Tensor],
        counts: list[int],
        *,
        staging_slot: str,
    ) -> None:
        if destination is None:
            return
        if destination.numel() != len(counts):
            raise RuntimeError("Token Owner DP count tensor has the wrong size")
        if destination.device.type != "cuda":
            destination.copy_(torch.as_tensor(counts, dtype=destination.dtype))
            return

        entries = self._count_staging.setdefault(staging_slot, [])
        entry = next(
            (
                candidate
                for candidate in entries
                if candidate.tensor.dtype == destination.dtype
                and candidate.tensor.numel() == destination.numel()
                and candidate.ready_event.query()
            ),
            None,
        )
        if entry is None:
            entry = _CountStaging(
                tensor=torch.empty(
                    destination.numel(), dtype=destination.dtype, pin_memory=True
                ),
                ready_event=torch.cuda.Event(),
            )
            entries.append(entry)
        entry.tensor.copy_(torch.as_tensor(counts, dtype=destination.dtype))
        destination.copy_(entry.tensor, non_blocking=True)
        entry.ready_event.record(torch.cuda.current_stream(destination.device))

    @staticmethod
    def _ceil_align(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    @staticmethod
    def _required_counts(forward_batch, name: str, expected: int) -> tuple[int, ...]:
        values = getattr(forward_batch, name, None)
        if values is None or len(values) != expected:
            raise RuntimeError(
                f"WeLM token-owner requires {expected} values in {name}"
            )
        values = tuple(map(int, values))
        if any(value < 0 for value in values):
            raise RuntimeError(f"WeLM token-owner received invalid {name}")
        return values

    @staticmethod
    def _optional_counts(
        forward_batch, name: str, expected: int
    ) -> Optional[tuple[int, ...]]:
        values = getattr(forward_batch, name, None)
        if values is None:
            return None
        if len(values) != expected:
            raise RuntimeError(
                f"WeLM token-owner requires {expected} values in {name}"
            )
        values = tuple(map(int, values))
        if any(value < 0 for value in values):
            raise RuntimeError(f"WeLM token-owner received invalid {name}")
        return values

    @staticmethod
    def _required_indices(forward_batch, name: str) -> tuple[int, ...]:
        values = getattr(forward_batch, name, None)
        if values is None:
            raise RuntimeError(f"WeLM token-owner requires CPU metadata {name}")
        return tuple(map(int, values))

    def _global_layout(
        self,
        counts: tuple[int, ...],
        contract_flags: Optional[tuple[bool, ...]],
        *,
        single_group_local_layout: Optional[TokenOwnerLayout] = None,
    ) -> TokenOwnerLayout:
        owner_count = self.attn_tp_group.world_size
        owner_sizes = tuple(
            size
            for group_index, count in enumerate(counts)
            for size in (
                single_group_local_layout.owner_sizes
                if single_group_local_layout is not None
                else (count, *(0 for _ in range(owner_count - 1)))
                if contract_flags is not None and contract_flags[group_index]
                else TokenOwnerLayout.balanced(
                    valid_token_count=count,
                    owner_count=owner_count,
                    local_owner_rank=0,
                ).owner_sizes
            )
        )
        return TokenOwnerLayout.from_owner_sizes(
            owner_sizes,
            local_owner_rank=self.global_tp_group.rank_in_group,
        )

    def _survivor_mapping(
        self,
        forward_batch,
        pre_layout: TokenOwnerLayout,
        post_layout: TokenOwnerLayout,
        *,
        owner_local: bool,
        enabled: bool,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if not enabled:
            return (), ()
        source_rows = self._required_indices(
            forward_batch, "welm_kv_mirror_last_q_indices_cpu"
        )
        output_rows = self._required_indices(
            forward_batch, "welm_kv_mirror_active_batch_indices_cpu"
        )
        if len(source_rows) != len(output_rows):
            raise RuntimeError(
                "KV mirror last-Q and active-request metadata must align"
            )
        if not owner_local:
            if any(not 0 <= row < pre_layout.valid_token_count for row in source_rows):
                raise RuntimeError("KV mirror last-Q index is outside the original rows")
            if any(not 0 <= row < post_layout.valid_token_count for row in output_rows):
                raise RuntimeError("KV mirror output index is outside contracted rows")
            return source_rows, output_rows

        source_start = pre_layout.owner_offsets[pre_layout.local_owner_rank]
        source_end = source_start + pre_layout.local_valid_rows
        output_start = post_layout.owner_offsets[post_layout.local_owner_rank]
        output_end = output_start + post_layout.local_valid_rows
        local_source = []
        local_output = []
        for source, output in zip(source_rows, output_rows, strict=True):
            if source_start <= source < source_end:
                if not output_start <= output < output_end:
                    raise RuntimeError(
                        "KV mirror survivor is outside its original owner segment"
                    )
                local_source.append(source - source_start)
                local_output.append(output - output_start)
        return tuple(local_source), tuple(local_output)

    @staticmethod
    def _build_valid_mask(
        layout: TokenOwnerLayout,
        output_rows: tuple[int, ...],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros(layout.local_valid_rows, dtype=torch.bool, device=device)
        if output_rows:
            mask[
                torch.tensor(output_rows, dtype=torch.long, device=device)
            ] = True
        return mask

    def _router_context(
        self,
        local_layout: TokenOwnerLayout,
        global_layout: TokenOwnerLayout,
        *,
        device: torch.device,
        valid_mask: Optional[torch.Tensor],
    ) -> Optional[GlobalTPRouterContext | DeepEPRouterContext]:
        if get_moe_a2a_backend().is_deepep():
            return (
                DeepEPRouterContext(local_layout, valid_mask)
                if valid_mask is not None
                else None
            )
        return GlobalTPRouterContext(
            layout=global_layout,
            local_layout=local_layout,
            global_tp_group=self.global_tp_group,
            attn_tp_group=self.attn_tp_group,
        )

    def _owner_group_count(self) -> int:
        global_size = self.global_tp_group.world_size
        local_size = self.attn_tp_group.world_size
        if global_size % local_size != 0:
            raise RuntimeError(
                "Global TP size is not divisible by token-owner group size"
            )
        return global_size // local_size

    def _owner_group_index(self) -> int:
        return self.global_tp_group.rank_in_group // self.attn_tp_group.world_size
