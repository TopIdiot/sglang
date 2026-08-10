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

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.distributed.parallel_state import get_attn_tp_group, get_tp_group
from sglang.srt.layers.communicator import TokenOwnerLayout
from sglang.srt.layers.dp_attention import (
    get_attention_cp_size,
    get_attention_tp_size,
    is_dp_attention_enabled,
    is_suffix_parallel_enabled,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.mk_moe_router import (
    MkMoeRouterMode,
    get_mk_moe_router_mode,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.welmv4_op import welm_use_previous_precision
from sglang.srt.server_args import get_global_server_args


def validate_welm_token_owner_capability(
    *,
    enabled: bool,
    dp_attention_enabled: bool,
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
) -> bool:
    if not enabled:
        return False
    requirements = (
        (dp_attention_enabled, "currently requires DP Attention"),
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


class WeLMTokenOwnerRuntime:
    def __init__(self, *, attn_tp_group=None, global_tp_group=None):
        self.attn_tp_group = (
            get_attn_tp_group() if attn_tp_group is None else attn_tp_group
        )
        self.global_tp_group = (
            get_tp_group() if global_tp_group is None else global_tp_group
        )
        self._forward_batch = None
        self.invalidate()

    def begin_forward(self, forward_batch) -> None:
        self._forward_batch = forward_batch
        self.invalidate()

    def invalidate(self) -> None:
        self._layouts = None
        self._mirror_mapping = None
        self._valid_mask = None

    def _ensure_batch(self, forward_batch) -> None:
        if self._forward_batch is not forward_batch:
            self.begin_forward(forward_batch)

    def _dp_rank(self) -> int:
        return self.global_tp_group.rank_in_group // self.attn_tp_group.world_size

    def _original_local_layout(self, forward_batch) -> TokenOwnerLayout:
        original_counts = getattr(
            forward_batch, "original_global_num_tokens_cpu", None
        )
        dp_rank = self._dp_rank()
        if original_counts is None or not 0 <= dp_rank < len(original_counts):
            raise RuntimeError(
                "WeLM token-owner mirror contraction requires original DP counts"
            )
        return TokenOwnerLayout.balanced(
            valid_token_count=int(original_counts[dp_rank]),
            owner_count=self.attn_tp_group.world_size,
            local_owner_rank=self.attn_tp_group.rank_in_group,
        )

    def _compute_layouts(self, forward_batch):
        counts_source = getattr(forward_batch, "global_num_tokens_cpu", None)
        if counts_source is None:
            raise RuntimeError("WeLM token-owner requires CPU global token counts")
        counts = tuple(map(int, counts_source))
        if not counts or any(count < 0 for count in counts):
            raise RuntimeError("WeLM token-owner received invalid global token counts")

        attn_tp_size = self.attn_tp_group.world_size
        global_tp_rank = self.global_tp_group.rank_in_group
        if self.global_tp_group.world_size != len(counts) * attn_tp_size:
            raise RuntimeError("WeLM token-owner group sizes do not match token counts")
        if self.attn_tp_group.rank_in_group != global_tp_rank % attn_tp_size:
            raise RuntimeError("WeLM token-owner rank ordering changed at runtime")

        preserve_owner = hasattr(
            forward_batch, "_welm_kv_mirror_contracted_dp_metadata_rows"
        ) and not getattr(forward_batch, "welm_mtp_merge_kv_fill_draft", False)
        flags = getattr(forward_batch, "welm_kv_mirror_contract_flags", None)
        if preserve_owner:
            if flags is None or len(flags) != len(counts):
                raise RuntimeError(
                    "WeLM token-owner mirror contraction requires one flag per DP rank"
                )
            flags = tuple(map(bool, flags))
        else:
            flags = (False,) * len(counts)

        dp_rank = self._dp_rank()
        if flags[dp_rank]:
            original_counts = getattr(
                forward_batch, "original_global_num_tokens_cpu", None
            )
            if original_counts is None or len(original_counts) != len(counts):
                raise RuntimeError(
                    "WeLM token-owner mirror contraction requires original DP counts"
                )
            owner_sizes = kv_mirror_owner_sizes(
                original_token_count=original_counts[dp_rank],
                owner_count=attn_tp_size,
                last_q_indices=getattr(
                    forward_batch, "welm_kv_mirror_last_q_indices_cpu", ()
                ),
                active_batch_indices=getattr(
                    forward_batch, "welm_kv_mirror_active_batch_indices_cpu", ()
                ),
                output_size=counts[dp_rank],
            )
            local_layout = TokenOwnerLayout.from_owner_sizes(
                owner_sizes,
                local_owner_rank=self.attn_tp_group.rank_in_group,
            )
        else:
            local_layout = TokenOwnerLayout.balanced(
                valid_token_count=counts[dp_rank],
                owner_count=attn_tp_size,
                local_owner_rank=self.attn_tp_group.rank_in_group,
            )

        global_owner_sizes = tuple(
            owner_size
            for count, contracted in zip(counts, flags, strict=True)
            for owner_size in (
                (count, *(0 for _ in range(attn_tp_size - 1)))
                if contracted
                else TokenOwnerLayout.balanced(
                    valid_token_count=count,
                    owner_count=attn_tp_size,
                    local_owner_rank=0,
                ).owner_sizes
            )
        )
        return local_layout, TokenOwnerLayout.from_owner_sizes(
            global_owner_sizes,
            local_owner_rank=global_tp_rank,
        )

    def _get_layouts(self, forward_batch):
        self._ensure_batch(forward_batch)
        if self._layouts is None:
            self._layouts = self._compute_layouts(forward_batch)
        return self._layouts

    def local_layout(self, forward_batch) -> TokenOwnerLayout:
        return self._get_layouts(forward_batch)[0]

    def global_layout(self, forward_batch) -> TokenOwnerLayout:
        return self._get_layouts(forward_batch)[1]

    def local_contraction_active(self, forward_batch) -> bool:
        self._ensure_batch(forward_batch)
        if not hasattr(forward_batch, "_welm_kv_mirror_contracted_dp_metadata_rows"):
            return False
        if getattr(forward_batch, "welm_mtp_merge_kv_fill_draft", False):
            return False
        flags = getattr(forward_batch, "welm_kv_mirror_contract_flags", None)
        counts = getattr(forward_batch, "global_num_tokens_cpu", None)
        if flags is None or counts is None or len(flags) != len(counts):
            raise RuntimeError(
                "WeLM token-owner mirror contraction requires one flag per DP rank"
            )
        return bool(flags[self._dp_rank()])

    def _get_mirror_mapping(self, forward_batch):
        self._ensure_batch(forward_batch)
        if self._mirror_mapping is not None:
            return self._mirror_mapping
        layout = self.local_layout(forward_batch)
        original_layout = self._original_local_layout(forward_batch)
        last_q_indices = getattr(
            forward_batch, "welm_kv_mirror_last_q_indices_cpu", None
        )
        active_batch_indices = getattr(
            forward_batch, "welm_kv_mirror_active_batch_indices_cpu", None
        )
        if last_q_indices is None or active_batch_indices is None or len(
            last_q_indices
        ) != len(active_batch_indices):
            raise RuntimeError(
                "WeLM token-owner mirror contraction requires aligned CPU row metadata"
            )
        source_start = original_layout.owner_offsets[original_layout.local_owner_rank]
        source_end = source_start + original_layout.local_valid_rows
        output_start = layout.owner_offsets[layout.local_owner_rank]
        output_end = output_start + layout.local_valid_rows
        mapping = []
        for source_index, output_index in zip(
            last_q_indices, active_batch_indices, strict=True
        ):
            source_index = int(source_index)
            if source_start <= source_index < source_end:
                output_index = int(output_index)
                if not output_start <= output_index < output_end:
                    raise RuntimeError(
                        "KV mirror survivor is outside its original owner segment"
                    )
                mapping.append(
                    (source_index - source_start, output_index - output_start)
                )
        self._mirror_mapping = (layout, original_layout, tuple(mapping))
        return self._mirror_mapping

    def valid_local_mask(self, forward_batch, *, device) -> torch.Tensor:
        self._ensure_batch(forward_batch)
        if self._valid_mask is not None:
            if self._valid_mask.device != device:
                raise RuntimeError("token-owner valid-row mask device changed")
            return self._valid_mask
        layout = self.local_layout(forward_batch)
        mask = torch.ones(layout.local_valid_rows, dtype=torch.bool, device=device)
        if self.local_contraction_active(forward_batch):
            mask.zero_()
            _, _, mapping = self._get_mirror_mapping(forward_batch)
            if mapping:
                mask[
                    torch.tensor(
                        [output for _, output in mapping],
                        dtype=torch.long,
                        device=device,
                    )
                ] = True
        self._valid_mask = mask
        return mask

    def align_kv_mirror_residual(self, residual, forward_batch):
        if not self.local_contraction_active(forward_batch):
            return residual
        layout, original_layout, mapping = self._get_mirror_mapping(forward_batch)
        if residual.shape[0] != original_layout.local_valid_rows:
            raise RuntimeError(
                "pre-contraction residual rows do not match the original owner"
            )
        contracted = residual.new_zeros(
            (layout.local_valid_rows, *residual.shape[1:])
        )
        if mapping:
            source, output = zip(*mapping, strict=True)
            contracted.index_copy_(
                0,
                torch.tensor(output, dtype=torch.long, device=residual.device),
                residual.index_select(
                    0, torch.tensor(source, dtype=torch.long, device=residual.device)
                ),
            )
        return contracted

    def build_router_context(self, forward_batch, *, device, use_deepep):
        if use_deepep:
            if not self.local_contraction_active(forward_batch):
                return None
            return DeepEPRouterContext(
                layout=self.local_layout(forward_batch),
                valid_local_mask=self.valid_local_mask(
                    forward_batch,
                    device=device,
                ),
            )
        return GlobalTPRouterContext(
            layout=self.global_layout(forward_batch),
            local_layout=self.local_layout(forward_batch),
            global_tp_group=self.global_tp_group,
            attn_tp_group=self.attn_tp_group,
        )
