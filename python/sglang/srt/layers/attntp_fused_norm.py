"""Runtime selection for fused AttnTP reduction and WeLM norms."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
import logging
from typing import Mapping

import torch

from sglang.jit_kernel.attntp_fused_norm.ipc import (
    MAX_DECODE_ROWS,
    OutputMode,
)
from sglang.jit_kernel.attntp_fused_norm.candidates import (
    run_prepared_production_candidate_out,
)
from sglang.jit_kernel.attntp_fused_norm.tuning import (
    AttnTPFusedNormCandidate,
    AttnTPFusedNormRegistry,
    AttnTPFusedNormWorkloadKey,
)

_PHASES = ("prefill", "decode")
_TOPOLOGIES = ("tp", "dp", "cp")
_ARENA_FAMILIES = ("direct_symm", "tile_pipeline")
_DIRECT_FAMILIES = ("ipc_owner_pull", "ipc_source_push", "local_fused")
_PREFILL_CP_ATTNTP2_COMM_NAME = "prefill_cp_attntp2_fused_norm"

logger = logging.getLogger(__name__)


def attntp_fused_norm_communicator_name(phase: str, topology: str) -> str:
    if phase not in _PHASES:
        raise ValueError(f"phase must be one of {_PHASES}")
    if topology not in _TOPOLOGIES:
        raise ValueError(f"topology must be one of {_TOPOLOGIES}")
    if phase == "prefill" and topology == "cp":
        return _PREFILL_CP_ATTNTP2_COMM_NAME
    return f"attntp_fused_norm_{phase}_{topology}"


def max_prefill_cp_local_tokens(
    *,
    max_global_tokens: int,
    cp_size: int,
    page_size: int,
) -> int:
    """Bound one rotated-zigzag owner's rows for a single prefill request."""
    max_global_tokens = int(max_global_tokens)
    cp_size = int(cp_size)
    page_size = int(page_size)
    if max_global_tokens < 0:
        raise ValueError("max_global_tokens must be non-negative")
    if cp_size <= 0:
        raise ValueError("cp_size must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if max_global_tokens == 0:
        return 0

    max_new_pages = (max_global_tokens + page_size - 1) // page_size
    if max_new_pages <= cp_size:
        max_owner_pages = 1
    elif max_new_pages < 2 * cp_size:
        max_owner_pages = 2
    else:
        pages_per_block = (max_new_pages + 2 * cp_size - 1) // (2 * cp_size)
        max_owner_pages = 2 * pages_per_block

    # A cache-hit extend may begin in an existing partial page whose owner is
    # independent of the rotated owner assigned to newly allocated pages.
    bound = max_owner_pages * page_size + page_size - 1
    return min(max_global_tokens, bound)


def get_or_create_attntp_fused_norm_manager(
    *,
    group,
    phase: str,
    topology: str,
    hidden_size: int,
    max_rows: int,
) -> AttnTPFusedNormManager:
    name = attntp_fused_norm_communicator_name(phase, topology)
    existing = group.get_graph_capture_communicator(name)
    if existing is not None:
        if not isinstance(existing, AttnTPFusedNormManager):
            raise RuntimeError(f"{name} is already owned by {type(existing).__name__}")
        if (
            existing.phase != phase
            or existing.topology != topology
            or existing.attn_tp_size != group.world_size
            or existing.hidden_size != hidden_size
            or existing.max_rows != max_rows
        ):
            raise RuntimeError(
                "Existing autotuned AttnTP fused norm manager has an "
                "incompatible topology or capacity"
            )
        return existing

    if group.world_size not in (2, 4, 8):
        raise RuntimeError(
            "Distributed AttnTP fused norm requires AttnTP size 2, 4, or 8"
        )
    if group.device.type != "cuda":
        raise RuntimeError(
            "Distributed AttnTP fused norm requires an NVIDIA CUDA device"
        )
    manager = AttnTPFusedNormManager(
        phase=phase,
        topology=topology,
        attn_tp_size=group.world_size,
        hidden_size=hidden_size,
        max_rows=max_rows,
    )
    group.register_graph_capture_communicator(name, manager)
    return manager


def get_prefill_cp_attntp_fused_norm_manager(
    *,
    group,
    max_global_tokens: int,
    cp_size: int,
    page_size: int,
    hidden_size: int,
) -> AttnTPFusedNormManager:
    if group.world_size != 2:
        raise RuntimeError("Prefill CP fused norm currently requires AttnTP2")
    max_local_tokens = max_prefill_cp_local_tokens(
        max_global_tokens=max_global_tokens,
        cp_size=cp_size,
        page_size=page_size,
    )
    if max_local_tokens <= 0:
        raise RuntimeError("Prefill CP fused norm requires positive capacity")
    return get_or_create_attntp_fused_norm_manager(
        group=group,
        phase="prefill",
        topology="cp",
        hidden_size=hidden_size,
        max_rows=max_local_tokens,
    )


@dataclass
class PreparedAttnTPFusedNormRunner:
    """Uniform runtime adapter around one already-prepared kernel winner."""

    phase: str
    candidate: AttnTPFusedNormCandidate
    runner: object
    capacity: int
    topology: str = "tp"
    _actual_rows: torch.Tensor | None = field(init=False, repr=False)
    _lane_rotation: torch.Tensor | None = field(init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError(f"phase must be one of {_PHASES}")
        if self.topology not in _TOPOLOGIES:
            raise ValueError(f"topology must be one of {_TOPOLOGIES}")
        if not isinstance(self.candidate, AttnTPFusedNormCandidate):
            raise ValueError("candidate must be an AttnTPFusedNormCandidate")
        if self.candidate.output_mode is not OutputMode.REPLICATED:
            raise ValueError(
                "current production prepared runners require replicated output"
            )
        if self.candidate.family not in (*_ARENA_FAMILIES, *_DIRECT_FAMILIES):
            raise ValueError(
                f"unsupported fused AttnTP norm family: {self.candidate.family}"
            )
        if type(self.capacity) is not int or self.capacity <= 0:
            raise ValueError("prepared runner capacity must be positive")
        device = torch.device(getattr(self.runner, "device"))
        if self.phase == "prefill":
            self._actual_rows = torch.empty((1,), dtype=torch.int32, device=device)
            self._lane_rotation = torch.zeros((1,), dtype=torch.int32, device=device)
        else:
            self._actual_rows = None
            self._lane_rotation = None

    @property
    def resource_identity(self) -> int:
        return id(getattr(self.runner, "_resource", self.runner))

    def capture(self):
        if self._closed:
            raise RuntimeError("prepared fused AttnTP norm runner is closed")
        return self.runner.capture()

    def close(self) -> None:
        if self._closed:
            return
        self.runner.close()
        self._closed = True

    def forward(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.phase == "prefill" and self.topology == "cp":
            raise RuntimeError(
                "Prefill CP must use forward_prefill_cp with lane_rotation"
            )
        return self._forward(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            o_norm_eps,
            post_norm_eps,
            lane_rotation=None,
        )

    def forward_prefill_cp(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
        *,
        lane_rotation: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.phase != "prefill" or self.topology != "cp":
            raise RuntimeError(
                "forward_prefill_cp requires a Prefill CP prepared runner"
            )
        return self._forward(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            o_norm_eps,
            post_norm_eps,
            lane_rotation=lane_rotation,
        )

    def _forward(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
        *,
        lane_rotation: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(partial, dtype=torch.bfloat16)
        residual_out = torch.empty_like(residual, dtype=torch.float32)
        self._forward_out(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            output,
            residual_out,
            o_norm_eps,
            post_norm_eps,
            lane_rotation=lane_rotation,
        )
        return output, residual_out

    def _forward_out(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        output: torch.Tensor,
        residual_out: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
        *,
        lane_rotation: int | None,
    ) -> None:
        if self._closed:
            raise RuntimeError("prepared fused AttnTP norm runner is closed")
        if partial.ndim != 2 or residual.shape != partial.shape:
            raise ValueError("partial and residual must have the same 2D shape")
        if output.shape != partial.shape or output.dtype != torch.bfloat16:
            raise ValueError("output must be BF16 with the same shape as partial")
        if residual_out.shape != residual.shape or residual_out.dtype != torch.float32:
            raise ValueError(
                "residual_out must be FP32 with the same shape as residual"
            )
        rows = partial.shape[0]
        if rows <= 0 or rows > self.capacity:
            raise RuntimeError(
                f"fused AttnTP norm supports [1, {self.capacity}] rows, got {rows}"
            )
        if self.phase == "prefill":
            if self._actual_rows is None or self._lane_rotation is None:
                raise RuntimeError("Prefill metadata tensors are unavailable")
            self._actual_rows.fill_(rows)
            if self.topology == "cp":
                if type(lane_rotation) is not int or lane_rotation < 0:
                    raise ValueError("lane_rotation must be a non-negative integer")
                self._lane_rotation.fill_(lane_rotation)
            elif lane_rotation is not None:
                raise ValueError("only Prefill CP accepts lane_rotation")
            actual_rows = self._actual_rows
            rotation_tensor = self._lane_rotation
        else:
            if lane_rotation is not None:
                raise ValueError("Decode does not accept lane_rotation")
            actual_rows = None
            rotation_tensor = None

        run_prepared_production_candidate_out(
            phase=self.phase,
            topology=self.topology,
            candidate=self.candidate,
            runner=self.runner,
            partial=partial,
            residual=residual,
            o_norm_weight=o_norm_weight,
            post_norm_weight=post_norm_weight,
            output=output,
            residual_out=residual_out,
            actual_rows=actual_rows,
            lane_rotation=rotation_tensor,
            o_norm_eps=o_norm_eps,
            post_norm_eps=post_norm_eps,
        )


class AttnTPFusedNormManager:
    """Fail-fast selector for prepared fused AttnTP norm winners."""

    def __init__(
        self,
        *,
        phase: str,
        topology: str,
        attn_tp_size: int,
        hidden_size: int,
        max_rows: int | None = None,
    ) -> None:
        if phase not in _PHASES:
            raise ValueError(f"phase must be one of {_PHASES}")
        if topology not in _TOPOLOGIES:
            raise ValueError(f"topology must be one of {_TOPOLOGIES}")
        if attn_tp_size not in (1, 2, 4, 8):
            raise ValueError("attn_tp_size must be one of (1, 2, 4, 8)")
        if hidden_size not in (2048, 4096):
            raise ValueError("hidden_size must be 2048 or 4096")
        if max_rows is not None and (type(max_rows) is not int or max_rows <= 0):
            raise ValueError("max_rows must be a positive integer or None")
        self.phase = phase
        self.topology = topology
        self.attn_tp_size = attn_tp_size
        self.hidden_size = hidden_size
        self.max_rows = max_rows
        self.registry: AttnTPFusedNormRegistry | None = None
        self.prepared: Mapping[
            AttnTPFusedNormWorkloadKey,
            PreparedAttnTPFusedNormRunner,
        ] = {}
        self.resources: tuple[object, ...] = ()
        self.manifest_path: str | None = None
        self._closed = False

    def install(
        self,
        *,
        registry: AttnTPFusedNormRegistry,
        prepared: Mapping[
            AttnTPFusedNormWorkloadKey,
            PreparedAttnTPFusedNormRunner,
        ],
        resources: tuple[object, ...] = (),
    ) -> None:
        if self._closed:
            raise RuntimeError("fused AttnTP norm manager is closed")
        if self.registry is not None:
            raise RuntimeError("fused AttnTP norm manager is already installed")
        missing = set(registry.winners) - set(prepared)
        unexpected = set(prepared) - set(registry.winners)
        if missing or unexpected:
            raise RuntimeError(
                "fused AttnTP norm prepared runner set does not match registry: "
                f"missing={sorted(key.encode() for key in missing)}, "
                f"unexpected={sorted(key.encode() for key in unexpected)}"
            )
        for key, runner in prepared.items():
            if (
                key.phase != self.phase
                or key.topology != self.topology
                or key.attn_tp_size != self.attn_tp_size
                or key.hidden_size != self.hidden_size
                or runner.phase != key.phase
                or runner.topology != key.topology
            ):
                raise RuntimeError(
                    "prepared fused AttnTP norm key does not match manager: "
                    f"{key.encode()}"
                )
            if runner.candidate != registry.winners[key].candidate:
                raise RuntimeError(
                    "prepared fused AttnTP norm candidate does not match winner: "
                    f"{key.encode()}"
                )
        unique_resources = []
        seen_resources = set()
        for resource in resources:
            if not callable(getattr(resource, "close", None)):
                raise ValueError("fused AttnTP norm resources must be closeable")
            resource_id = id(resource)
            if resource_id in seen_resources:
                continue
            seen_resources.add(resource_id)
            unique_resources.append(resource)
        self.registry = registry
        self.prepared = dict(prepared)
        self.resources = tuple(unique_resources)

    def install_winners(
        self,
        winners,
        *,
        group,
        device,
    ) -> None:
        """Prepare selected JIT runners over one shared production resource set."""

        from sglang.jit_kernel.attntp_fused_norm.candidates import (
            prepare_production_candidate,
        )
        from sglang.jit_kernel.attntp_fused_norm.resources import (
            build_production_resource_set,
        )

        if self._closed:
            raise RuntimeError("fused AttnTP norm manager is closed")
        if self.registry is not None:
            raise RuntimeError("fused AttnTP norm manager is already installed")
        selected = dict(winners)
        if not selected:
            raise ValueError("fused AttnTP norm winners must not be empty")
        if (
            self.max_rows is not None
            and max(key.row_bucket for key in selected) > self.max_rows
        ):
            raise RuntimeError(
                "fused AttnTP norm winner exceeds manager capacity: "
                f"{max(key.row_bucket for key in selected)} > {self.max_rows}"
            )

        registry = AttnTPFusedNormRegistry.build(
            required_keys=tuple(selected),
            winners=selected,
        )
        resource_set = build_production_resource_set(
            selected,
            group=group,
            device=device,
        )
        prepared = {}
        try:
            for key in registry.winners:
                winner = registry.winners[key]
                tile_layout = (
                    resource_set.tile_layout_for(key, winner.candidate)
                    if winner.candidate.family == "tile_pipeline"
                    else None
                )
                raw_runner = prepare_production_candidate(
                    key,
                    winner.candidate,
                    group=group,
                    device=device,
                    resource=resource_set.resource_for(winner.candidate),
                    tile_arena_layout=tile_layout,
                )
                try:
                    prepared[key] = PreparedAttnTPFusedNormRunner(
                        phase=key.phase,
                        candidate=winner.candidate,
                        runner=raw_runner,
                        capacity=key.row_bucket,
                        topology=key.topology,
                    )
                except Exception:
                    raw_runner.close()
                    raise
            self.install(
                registry=registry,
                prepared=prepared,
                resources=resource_set.resources,
            )
        except Exception:
            for runner in reversed(tuple(prepared.values())):
                runner.close()
            resource_set.close()
            raise

    def capture(self):
        if self.registry is None:
            raise RuntimeError("fused AttnTP norm manager is not installed")
        stack = ExitStack()
        seen_resources = set()
        for runner in self.prepared.values():
            if runner.resource_identity in seen_resources:
                continue
            seen_resources.add(runner.resource_identity)
            stack.enter_context(runner.capture())
        return stack

    def close(self) -> None:
        if self._closed:
            return
        seen_runners = set()
        for prepared in self.prepared.values():
            runner_id = id(prepared.runner)
            if runner_id in seen_runners:
                continue
            seen_runners.add(runner_id)
            prepared.close()
        for resource in self.resources:
            resource.close()
        self._closed = True

    def forward(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
        *,
        execution: str = "eager",
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if self.phase == "prefill" and self.topology == "cp":
            raise RuntimeError(
                "Prefill CP must use forward_prefill_cp with lane_rotation"
            )
        return self._forward(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            o_norm_eps,
            post_norm_eps,
            execution=execution,
            lane_rotation=None,
        )

    def forward_prefill_cp(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
        *,
        lane_rotation: int,
        execution: str = "eager",
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if self.phase != "prefill" or self.topology != "cp":
            raise RuntimeError("forward_prefill_cp requires a Prefill CP manager")
        if (
            type(lane_rotation) is not int
            or lane_rotation < 0
            or lane_rotation >= self.attn_tp_size
        ):
            raise ValueError("lane_rotation must be inside the attention-TP group")
        return self._forward(
            partial,
            residual,
            o_norm_weight,
            post_norm_weight,
            o_norm_eps,
            post_norm_eps,
            execution=execution,
            lane_rotation=lane_rotation,
        )

    def _forward(
        self,
        partial: torch.Tensor,
        residual: torch.Tensor,
        o_norm_weight: torch.Tensor,
        post_norm_weight: torch.Tensor,
        o_norm_eps: float,
        post_norm_eps: float,
        *,
        execution: str,
        lane_rotation: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        if self._closed:
            raise RuntimeError("fused AttnTP norm manager is closed")
        if self.registry is None:
            raise RuntimeError("fused AttnTP norm manager is not installed")
        if partial.ndim != 2 or residual.shape != partial.shape:
            raise ValueError("partial and residual must have the same 2D shape")
        if partial.shape[1] != self.hidden_size:
            raise ValueError(
                "fused AttnTP norm hidden size mismatch: "
                f"{partial.shape[1]} != {self.hidden_size}"
            )
        rows = partial.shape[0]
        if self.max_rows is not None and rows > self.max_rows:
            raise RuntimeError(
                "fused AttnTP norm runtime rows exceed manager capacity: "
                f"{rows} > {self.max_rows}"
            )
        if rows == 0:
            return (
                partial.new_empty(partial.shape, dtype=torch.bfloat16),
                residual.new_empty(residual.shape, dtype=torch.float32),
                None,
            )
        prepared_capacity = max(runner.capacity for runner in self.prepared.values())
        if self.phase == "decode" and rows > prepared_capacity:
            output = torch.empty_like(partial, dtype=torch.bfloat16)
            residual_out = torch.empty_like(residual, dtype=torch.float32)
            for start in range(0, rows, prepared_capacity):
                end = min(start + prepared_capacity, rows)
                chunk_rows = end - start
                key, _ = self.registry.resolve_runtime(
                    phase=self.phase,
                    topology=self.topology,
                    attn_tp_size=self.attn_tp_size,
                    hidden_size=self.hidden_size,
                    rows=chunk_rows,
                    execution=execution,
                )
                try:
                    runner = self.prepared[key]
                except KeyError as error:
                    raise RuntimeError(
                        "fused AttnTP norm prepared runner is missing for "
                        f"{key.encode()}"
                    ) from error
                runner._forward_out(
                    partial[start:end],
                    residual[start:end],
                    o_norm_weight,
                    post_norm_weight,
                    output[start:end],
                    residual_out[start:end],
                    o_norm_eps,
                    post_norm_eps,
                    lane_rotation=None,
                )
            return output, residual_out, None
        key, _ = self.registry.resolve_runtime(
            phase=self.phase,
            topology=self.topology,
            attn_tp_size=self.attn_tp_size,
            hidden_size=self.hidden_size,
            rows=partial.shape[0],
            execution=execution,
        )
        try:
            runner = self.prepared[key]
        except KeyError as error:
            raise RuntimeError(
                f"fused AttnTP norm prepared runner is missing for {key.encode()}"
            ) from error
        if self.phase == "prefill" and self.topology == "cp":
            output, residual_out = runner.forward_prefill_cp(
                partial,
                residual,
                o_norm_weight,
                post_norm_weight,
                o_norm_eps,
                post_norm_eps,
                lane_rotation=lane_rotation,
            )
        else:
            output, residual_out = runner.forward(
                partial,
                residual,
                o_norm_weight,
                post_norm_weight,
                o_norm_eps,
                post_norm_eps,
            )
        return output, residual_out, None


def registered_attntp_fused_norm_managers(group):
    managers = []
    for phase in _PHASES:
        for topology in _TOPOLOGIES:
            name = attntp_fused_norm_communicator_name(phase, topology)
            manager = group.get_graph_capture_communicator(name)
            if manager is None:
                continue
            if not isinstance(manager, AttnTPFusedNormManager):
                raise RuntimeError(
                    "Autotuned AttnTP fused norm expected a deferred manager "
                    f"for {name}, found {type(manager).__name__}"
                )
            managers.append((name, manager))
    return tuple(managers)


def prepare_attntp_fused_norm_before_kv_pool(model_runner):
    """Autotune and install deferred fused-norm managers before KV sizing."""

    if not bool(getattr(model_runner, "enable_attntp_fused_norm", False)):
        return None

    from sglang.jit_kernel.attntp_fused_norm.tuning import (
        PRODUCTION_SEARCH_PROFILE,
        AttnTPFusedNormRegistry,
        AttnTPFusedNormWorkloadKey,
        attntp_production_manifest_path,
        build_production_manifest_identity,
        build_production_row_buckets,
        load_or_tune_attntp_manifest,
        run_local_production_autotune,
    )
    from sglang.srt.distributed import (
        get_attn_tp_group,
        get_world_group,
    )
    from sglang.srt.environ import envs

    attn_tp_group = get_attn_tp_group()
    tuning_device = torch.device(attn_tp_group.device)
    if tuning_device.type == "cuda" and tuning_device.index is None:
        raise RuntimeError("AttnTP fused norm requires a concrete per-rank CUDA device")

    world_group = get_world_group()
    node_rank = int(getattr(model_runner.server_args, "node_rank", 0))
    managers = registered_attntp_fused_norm_managers(attn_tp_group)
    local_signature = tuple(
        (
            name,
            manager.phase,
            manager.topology,
            manager.attn_tp_size,
            manager.hidden_size,
            manager.max_rows,
        )
        for name, manager in managers
    )
    discovery = tuple(
        world_group.all_gather_object(
            (
                node_rank,
                int(world_group.local_rank),
                local_signature,
            )
        )
    )
    local_discovery = tuple(payload for payload in discovery if payload[0] == node_rank)
    if not local_discovery:
        raise RuntimeError("AttnTP fused norm startup found no node-local participants")
    signatures = {payload[2] for payload in local_discovery}
    if signatures == {()}:
        raise RuntimeError(
            "AttnTP fused norm is enabled but no deferred manager was created "
            "on any node-local rank"
        )
    if len(signatures) != 1:
        raise RuntimeError(
            "AttnTP fused norm manager sets differ across "
            f"node-local ranks: {local_discovery}"
        )

    prepared_managers = []
    for communicator_name, manager in managers:
        if manager.registry is not None:
            raise RuntimeError(
                "Autotuned AttnTP fused norm manager was prepared more than once"
            )
        if manager.max_rows is None:
            raise RuntimeError(
                "Autotuned AttnTP fused norm manager has no row capacity"
            )
        execution = "graph" if manager.phase == "decode" else "eager"
        tuning_capacity = (
            min(manager.max_rows, MAX_DECODE_ROWS)
            if manager.phase == "decode"
            else manager.max_rows
        )
        required_keys = tuple(
            AttnTPFusedNormWorkloadKey.current(
                phase=manager.phase,
                topology=manager.topology,
                attn_tp_size=manager.attn_tp_size,
                hidden_size=manager.hidden_size,
                row_bucket=row_bucket,
                execution=execution,
            )
            for row_bucket in build_production_row_buckets(tuning_capacity)
        )
        identity = build_production_manifest_identity(
            required_keys,
            device=tuning_device,
            topology_extra={
                "communicator": communicator_name,
                "cp_size": int(getattr(model_runner.server_args, "attn_cp_size", 1)),
                "dp_size": int(getattr(model_runner.server_args, "dp_size", 1)),
                "page_size": int(getattr(model_runner.server_args, "page_size", 1)),
            },
            profile=PRODUCTION_SEARCH_PROFILE,
        )
        manifest_path = attntp_production_manifest_path(
            envs.SGLANG_CACHE_DIR.get(),
            identity,
        )
        identity_token = manifest_path.stem
        participant_state = {}

        def synchronize_cache(local_cached):
            gathered = tuple(
                world_group.all_gather_object(
                    (
                        node_rank,
                        identity_token,
                        int(world_group.local_rank),
                        local_cached,
                    )
                )
            )
            members = sorted(
                (
                    (group_index, local_rank, cached)
                    for group_index, (
                        member_node_rank,
                        token,
                        local_rank,
                        cached,
                    ) in enumerate(gathered)
                    if member_node_rank == node_rank and token == identity_token
                ),
                key=lambda item: item[1],
            )
            member_indices = [
                group_index for group_index, _local_rank, _cached in members
            ]
            if world_group.rank_in_group not in member_indices:
                raise RuntimeError(
                    "AttnTP fused norm autotune lost the current rank from "
                    "its node-local participant group"
                )
            participant_state["index"] = member_indices.index(world_group.rank_in_group)
            participant_state["count"] = len(member_indices)
            return next(
                (
                    cached
                    for _index, _local_rank, cached in members
                    if cached is not None
                ),
                None,
            )

        def gather_compilation(local_status):
            gathered = tuple(
                world_group.all_gather_object((node_rank, identity_token, local_status))
            )
            return tuple(
                status
                for member_node_rank, token, status in gathered
                if member_node_rank == node_rank and token == identity_token
            )

        def local_tune():
            if not participant_state:
                raise RuntimeError(
                    "AttnTP fused norm autotune participants were not synchronized"
                )
            return run_local_production_autotune(
                required_keys,
                participant_index=participant_state["index"],
                participant_count=participant_state["count"],
                compilation_gather_fn=gather_compilation,
                attn_tp_group=attn_tp_group,
                device=tuning_device,
                profile=PRODUCTION_SEARCH_PROFILE,
            )

        def gather_winners(local_winners):
            gathered = tuple(
                world_group.all_gather_object(
                    (node_rank, identity_token, local_winners)
                )
            )
            return tuple(
                winners
                for member_node_rank, token, winners in gathered
                if member_node_rank == node_rank and token == identity_token
            )

        winners = load_or_tune_attntp_manifest(
            manifest_path,
            identity,
            local_tune_fn=local_tune,
            synchronize_cache_fn=synchronize_cache,
            gather_fn=gather_winners,
            publish=world_group.local_rank == 0,
        )
        registry = AttnTPFusedNormRegistry.build(
            required_keys=required_keys,
            winners=winners,
        )
        manager.install_winners(
            registry.winners,
            group=attn_tp_group.cpu_group,
            device=tuning_device,
        )
        manager.manifest_path = str(manifest_path)
        logger.info(
            "Installed autotuned AttnTP fused norm winners: manifest=%s, workloads=%d",
            manifest_path,
            len(required_keys),
        )
        prepared_managers.append(manager)
    return tuple(prepared_managers)
