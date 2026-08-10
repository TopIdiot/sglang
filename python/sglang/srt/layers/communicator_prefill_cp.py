"""Layer communication for persistent-token sharded-KV prefill."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.distributed.parallel_state import (
    get_attn_cp_group,
    get_attn_tp_group,
    get_tp_group,
)
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.topk import StandardTopKOutput


def pair_lane_sizes(token_count: int) -> tuple[int, int]:
    token_count = int(token_count)
    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    return ((token_count + 1) // 2, token_count // 2)


def global_tp_destination_sizes(
    cp_token_counts: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        lane_count
        for token_count in cp_token_counts
        for lane_count in pair_lane_sizes(token_count)
    )


@dataclass(frozen=True)
class PrefillCPRouterContext:
    gathers_hidden_states = False

    global_tp_group: object
    destination_sizes: tuple[int, ...]
    owner_start: int
    owner_count: int
    global_token_count: int

    @staticmethod
    def _validate_rows(tensor: torch.Tensor, expected: int, name: str) -> None:
        if tensor.ndim == 0 or tensor.shape[0] != expected:
            rows = 0 if tensor.ndim == 0 else tensor.shape[0]
            raise RuntimeError(f"{name} has {rows} rows, expected {expected}")

    def local_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        self._validate_rows(tensor, self.global_token_count, "global router input")
        return tensor.narrow(0, self.owner_start, self.owner_count)

    def gather_routing_metadata(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_rows(topk_weights, self.owner_count, "owner top-k weights")
        self._validate_rows(topk_ids, self.owner_count, "owner top-k ids")
        if topk_weights.shape != topk_ids.shape:
            raise RuntimeError("owner top-k weights and ids must have matching shapes")
        if topk_weights.device != topk_ids.device:
            raise RuntimeError("owner top-k weights and ids must be on the same device")
        if self.global_token_count == 0:
            return topk_weights, topk_ids

        gathered = self.global_tp_group.all_gatherv(
            [topk_weights, topk_ids], sizes=list(self.destination_sizes)
        )
        if not isinstance(gathered, list) or len(gathered) != 2:
            raise RuntimeError(
                "prefill CP router metadata gather returned an invalid result"
            )
        full_weights, full_ids = gathered
        self._validate_rows(
            full_weights, self.global_token_count, "global top-k weights"
        )
        self._validate_rows(full_ids, self.global_token_count, "global top-k ids")
        if full_weights.shape != full_ids.shape:
            raise RuntimeError("global top-k weights and ids must have matching shapes")
        if (
            full_weights.dtype != topk_weights.dtype
            or full_ids.dtype != topk_ids.dtype
        ):
            raise RuntimeError("router metadata gather changed a tensor dtype")
        return full_weights, full_ids

    def prepare_moe_inputs(
        self,
        hidden_states: torch.Tensor,
        topk_output: StandardTopKOutput,
    ) -> tuple[torch.Tensor, StandardTopKOutput]:
        if not isinstance(topk_output, StandardTopKOutput):
            raise RuntimeError(
                "owner-local prefill CP routing requires Standard TopK output"
            )
        full_weights, full_ids = self.gather_routing_metadata(
            topk_output.topk_weights,
            topk_output.topk_ids,
        )
        return hidden_states, StandardTopKOutput(
            topk_weights=full_weights,
            topk_ids=full_ids,
            router_logits=topk_output.router_logits.new_empty(
                (self.global_token_count, 0)
            ),
        )


class PrefillCPLayerCommunicator:
    def __init__(
        self,
        *,
        input_layernorm,
        post_attention_layernorm,
        cp_group=None,
        attn_tp_group=None,
        global_tp_group=None,
        use_ep_dispatch: bool | None = None,
        fused_attntp2_norm_runner=None,
    ):
        self.input_layernorm = input_layernorm
        self.post_attention_layernorm = post_attention_layernorm
        self.cp_group = cp_group if cp_group is not None else get_attn_cp_group()
        self.attn_tp_group = (
            attn_tp_group if attn_tp_group is not None else get_attn_tp_group()
        )
        self.global_tp_group = (
            global_tp_group if global_tp_group is not None else get_tp_group()
        )
        self.use_ep_dispatch = (
            not get_moe_a2a_backend().is_none()
            if use_ep_dispatch is None
            else bool(use_ep_dispatch)
        )
        self.fused_attntp2_norm_runner = fused_attntp2_norm_runner

    def _counts(self, layout) -> tuple[int, ...]:
        counts = tuple(int(count) for count in layout.active_tokens_per_cp_rank())
        if any(count < 0 for count in counts):
            raise RuntimeError("prefill CP token counts must be non-negative")
        if layout.cp_rank < 0 or layout.cp_rank >= len(counts):
            raise RuntimeError("prefill CP rank is outside the active count vector")
        if counts[layout.cp_rank] != layout.active_local_tokens:
            raise RuntimeError("prefill CP local token count does not match its layout")
        return counts

    def _validate_topology(self, layout) -> tuple[tuple[int, ...], int]:
        counts = self._counts(layout)
        if (
            self.cp_group.world_size != len(counts)
            or self.cp_group.rank_in_group != layout.cp_rank
            or self.attn_tp_group.world_size != 2
            or self.global_tp_group.world_size != 2 * len(counts)
        ):
            raise RuntimeError("prefill CP communicator topology does not match layout")
        lane = self.attn_tp_group.rank_in_group
        if lane not in (0, 1):
            raise RuntimeError("prefill CP attention-TP lane must be 0 or 1")
        expected_tp_rank = 2 * layout.cp_rank + lane
        if self.global_tp_group.rank_in_group != expected_tp_rank:
            raise RuntimeError("prefill CP global-TP rank does not match CP/lane order")
        return counts, lane

    def validate_mlp(self, mlp) -> None:
        if not self.use_ep_dispatch:
            validate_local_router = getattr(mlp, "validate_local_router", None)
            if validate_local_router is None:
                raise RuntimeError(
                    "prefill CP token-owner routing requires an MoE router validator"
                )
            validate_local_router()
            return
        experts = getattr(mlp, "experts", None)
        if experts is None:
            raise RuntimeError("prefill CP EP dispatch requires an MoE expert module")
        moe_tp_size = int(getattr(experts, "moe_tp_size", 0))
        moe_ep_size = int(getattr(experts, "moe_ep_size", 0))
        if moe_tp_size != 1:
            raise NotImplementedError(
                "prefill CP EP dispatch currently requires MoE-TP1"
            )
        if moe_ep_size != self.global_tp_group.world_size:
            raise NotImplementedError(
                "prefill CP EP dispatch currently requires "
                f"EP{self.global_tp_group.world_size}"
            )

        shared_expert = getattr(mlp, "shared_expert", None)
        if shared_expert is None:
            return
        if (
            getattr(shared_expert.gate_up_proj, "tp_size", None) != 1
            or getattr(shared_expert.down_proj, "tp_size", None) != 1
        ):
            raise RuntimeError(
                "prefill CP EP dispatch requires a replicated shared expert"
            )

    @staticmethod
    def _validate_rows(tensor: torch.Tensor, expected: int, name: str) -> None:
        if tensor.shape[0] != expected:
            raise RuntimeError(
                f"{name} has {tensor.shape[0]} rows, expected {expected}"
            )

    def gather_global_tp_input(self, hidden_states: torch.Tensor, layout):
        counts, _ = self._validate_topology(layout)
        self._validate_rows(
            hidden_states, layout.active_local_tokens, "CP-local hidden"
        )
        if sum(counts) == 0:
            return hidden_states
        return self.cp_group.all_gatherv(
            hidden_states, sizes=list(counts)
        )[0]

    def build_router_context(self, forward_batch) -> PrefillCPRouterContext:
        if self.use_ep_dispatch:
            raise RuntimeError(
                "prefill CP token-owner router is not used with EP dispatch"
            )
        layout = self._layout_from_batch(forward_batch)
        counts, lane = self._validate_topology(layout)
        if getattr(forward_batch, "router_replay_topk_ids", None) is not None:
            raise NotImplementedError(
                "owner-local prefill CP routing does not support Router Replay"
            )

        destination_sizes = global_tp_destination_sizes(counts)
        global_tp_rank = self.global_tp_group.rank_in_group
        owner_count = destination_sizes[global_tp_rank]
        owner_start = sum(destination_sizes[:global_tp_rank])
        local_lane_sizes = pair_lane_sizes(counts[layout.cp_rank])
        if owner_count != local_lane_sizes[lane]:
            raise RuntimeError("prefill CP router owner mapping is inconsistent")
        return PrefillCPRouterContext(
            global_tp_group=self.global_tp_group,
            destination_sizes=destination_sizes,
            owner_start=owner_start,
            owner_count=owner_count,
            global_token_count=sum(counts),
        )

    def reduce_scatter_global_tp_output(self, partial: torch.Tensor, layout):
        counts, _ = self._validate_topology(layout)
        self._validate_rows(partial, sum(counts), "global-TP partial")
        if sum(counts) == 0:
            return partial
        destination_sizes = global_tp_destination_sizes(counts)
        return self.global_tp_group.reduce_scatterv(
            partial, sizes=list(destination_sizes)
        )

    def restore_attn_pair(self, local_part: torch.Tensor, layout):
        counts, lane = self._validate_topology(layout)
        pair_sizes = pair_lane_sizes(counts[layout.cp_rank])
        self._validate_rows(local_part, pair_sizes[lane], "attention-TP lane part")
        if sum(pair_sizes) == 0:
            return local_part
        return self.attn_tp_group.all_gatherv(
            local_part, sizes=list(pair_sizes)
        )[0]

    def scatter_ep_input(self, hidden_states: torch.Tensor, layout):
        counts, lane = self._validate_topology(layout)
        local_count = counts[layout.cp_rank]
        self._validate_rows(hidden_states, local_count, "EP pair input")
        lane_sizes = pair_lane_sizes(local_count)
        start = 0 if lane == 0 else lane_sizes[0]
        return hidden_states.narrow(0, start, lane_sizes[lane])

    def gather_ep_output(self, hidden_states: torch.Tensor, layout):
        return self.restore_attn_pair(hidden_states, layout)

    @staticmethod
    def _layout_from_batch(forward_batch):
        layout = getattr(
            forward_batch, "attn_cp_prefill_runtime_layout", None
        )
        if layout is None:
            raise RuntimeError("prefill CP communicator requires a runtime layout")
        return layout

    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch,
        residual_after_layernorm: bool = False,
        **_kwargs,
    ):
        layout = self._layout_from_batch(forward_batch)
        self._validate_topology(layout)
        self._validate_rows(
            hidden_states, layout.active_local_tokens, "layer input hidden"
        )
        if hidden_states.shape[0] == 0:
            residual = (
                hidden_states.to(torch.float32)
                if residual_after_layernorm
                else hidden_states
            )
        elif residual_after_layernorm:
            hidden_states, _, residual = self.input_layernorm(
                hidden_states,
                residual,
                residual_after_layernorm=True,
                clone_fp32_out=True,
            )
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual
            )
        return hidden_states, residual

    def prepare_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch,
        cache=None,
    ):
        del cache
        layout = self._layout_from_batch(forward_batch)
        self._validate_rows(
            hidden_states, layout.active_local_tokens, "attention output"
        )
        if hidden_states.shape[0] == 0:
            normalized = hidden_states
            residual = hidden_states
        else:
            normalized, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        if self.use_ep_dispatch:
            return self.scatter_ep_input(normalized, layout), residual
        return self.gather_global_tp_input(normalized, layout), residual

    def prepare_mlp_fused_attntp2(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch,
        *,
        o_norm,
    ):
        runner = self.fused_attntp2_norm_runner
        if runner is None:
            raise RuntimeError(
                "Prefill CP AttnTP2 fused IPC norm runner is unavailable"
            )

        layout = self._layout_from_batch(forward_batch)
        self._validate_topology(layout)
        self._validate_rows(
            hidden_states, layout.active_local_tokens, "attention partial"
        )
        self._validate_rows(
            residual, layout.active_local_tokens, "attention residual"
        )
        if hidden_states.shape[0] == 0:
            normalized = hidden_states
        else:
            normalized, residual, _ = runner.forward_prefill_cp(
                hidden_states,
                residual,
                o_norm.weight,
                self.post_attention_layernorm.weight,
                o_norm.eps,
                self.post_attention_layernorm.eps,
                lane_rotation=(
                    layout.spec.owner_rotation % self.attn_tp_group.world_size
                ),
            )
        if self.use_ep_dispatch:
            return self.scatter_ep_input(normalized, layout), residual
        return self.gather_global_tp_input(normalized, layout), residual

    def postprocess_layer(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        forward_batch,
    ):
        layout = self._layout_from_batch(forward_batch)
        if self.use_ep_dispatch:
            return self.gather_ep_output(hidden_states, layout), residual
        local_part = self.reduce_scatter_global_tp_output(hidden_states, layout)
        hidden_states = self.restore_attn_pair(local_part, layout)
        return hidden_states, residual

    def should_use_reduce_scatter(self, forward_batch) -> bool:
        return self._layout_from_batch(forward_batch) is not None

    def has_active_mlp_tokens(self, forward_batch) -> bool:
        layout = self._layout_from_batch(forward_batch)
        return sum(self._counts(layout)) != 0
