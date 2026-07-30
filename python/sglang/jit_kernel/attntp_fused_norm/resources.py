"""Reusable collective resources for fused AttnTP norm kernels."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def _run_rank_coordinated_init_stage(
    *,
    group,
    stage: str,
    operation: Callable[[], T],
    cleanup: Callable[[], None] | None = None,
    result_key: Callable[[T], object] | None = None,
) -> T:
    """Run one startup stage and fail it consistently across all ranks."""

    local_result = None
    local_error = None
    try:
        local_result = operation()
        local_result_key = result_key(local_result) if result_key is not None else None
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
        local_result_key = None

    payload = (stage, local_error, local_result_key)
    gathered = [None] * dist.get_world_size(group=group)
    try:
        dist.all_gather_object(gathered, payload, group=group)
    except Exception:
        if cleanup is not None:
            cleanup()
        raise

    stage_mismatches = tuple(
        (rank, rank_payload)
        for rank, rank_payload in enumerate(gathered)
        if not isinstance(rank_payload, (tuple, list))
        or len(rank_payload) != 3
        or rank_payload[0] != stage
    )
    rank_errors = tuple(
        (rank, rank_payload[1])
        for rank, rank_payload in enumerate(gathered)
        if isinstance(rank_payload, (tuple, list))
        and len(rank_payload) == 3
        and rank_payload[0] == stage
        and rank_payload[1] is not None
    )
    result_keys = (
        tuple(rank_payload[2] for rank_payload in gathered)
        if result_key is not None and not stage_mismatches and not rank_errors
        else ()
    )
    results_disagree = bool(result_keys) and any(
        key != result_keys[0] for key in result_keys[1:]
    )
    if stage_mismatches or rank_errors or results_disagree:
        cleanup_error = None
        if cleanup is not None:
            try:
                cleanup()
            except Exception as error:
                cleanup_error = f"{type(error).__name__}: {error}"
        details = []
        if stage_mismatches:
            details.append(f"stage mismatch={stage_mismatches}")
        if rank_errors:
            details.append(
                ", ".join(f"rank{rank}={error}" for rank, error in rank_errors)
            )
        if results_disagree:
            details.append(f"rank results disagree={result_keys}")
        if cleanup_error is not None:
            details.append(f"local cleanup failed={cleanup_error}")
        raise RuntimeError(f"AttnTP {stage} failed: {'; '.join(details)}")

    return local_result


def _validate_collective_shape(
    *,
    attn_tp_size: int,
    hidden_size: int | None = None,
) -> None:
    if type(attn_tp_size) is not int or attn_tp_size not in (1, 2, 4, 8):
        raise ValueError("attn_tp_size must be one of (1, 2, 4, 8)")
    if hidden_size is not None and (
        type(hidden_size) is not int or hidden_size not in (2048, 4096)
    ):
        raise ValueError("hidden_size must be one of (2048, 4096)")


class AttnTPIPCResource:
    """One capacity-sized CustomAllReduceV2 resource for an AttnTP group."""

    def __init__(
        self,
        *,
        group,
        device: torch.device,
        attn_tp_size: int,
        max_rows: int,
        max_pull_size: int,
        max_push_size: int,
        source_push_capacity: int | None = None,
        synchronize_fn: Callable[..., None] | None = None,
    ) -> None:
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        def validate_local() -> torch.device:
            _validate_collective_shape(attn_tp_size=attn_tp_size)
            self._validate_max_rows(max_rows)
            self._validate_sizes(max_pull_size, max_push_size)
            validated_device = torch.device(device)
            if validated_device.type != "cuda":
                raise ValueError("AttnTP IPC resource requires a CUDA device")
            if attn_tp_size == 1:
                raise ValueError("AttnTP1 does not require an IPC resource")
            if dist.get_world_size(group=group) != attn_tp_size:
                raise ValueError("process-group size must match AttnTP size")
            return validated_device

        self.device = _run_rank_coordinated_init_stage(
            group=group,
            stage="IPC argument validation",
            operation=validate_local,
            result_key=lambda validated_device: validated_device.type,
        )

        supported = _run_rank_coordinated_init_stage(
            group=group,
            stage="IPC capability check",
            operation=lambda: CustomAllReduceV2.is_supported(
                group=group,
                device=self.device,
            ),
            result_key=bool,
        )
        if not supported:
            raise RuntimeError(
                "AttnTP IPC resource requires enabled CUDA IPC CustomAllReduceV2"
            )

        communicator = None

        def prepare_local_communicator():
            nonlocal communicator
            communicator = CustomAllReduceV2.create_deferred(
                group,
                self.device,
                max_pull_size=max_pull_size,
                max_push_size=max_push_size,
            )
            return communicator

        def abort_communicator() -> None:
            if communicator is not None:
                communicator.abort()

        communicator = _run_rank_coordinated_init_stage(
            group=group,
            stage="IPC local prepare",
            operation=prepare_local_communicator,
            cleanup=abort_communicator,
        )
        _run_rank_coordinated_init_stage(
            group=group,
            stage="IPC handle exchange",
            operation=communicator.finalize,
            cleanup=abort_communicator,
        )
        _run_rank_coordinated_init_stage(
            group=group,
            stage="IPC resource binding",
            operation=lambda: self._initialize_from_communicator(
                communicator=communicator,
                device=self.device,
                attn_tp_size=attn_tp_size,
                max_rows=max_rows,
                max_pull_size=max_pull_size,
                max_push_size=max_push_size,
                source_push_capacity=source_push_capacity,
            ),
            cleanup=abort_communicator,
        )
        if synchronize_fn is None:
            _run_rank_coordinated_init_stage(
                group=group,
                stage="IPC CUDA synchronization",
                operation=lambda: torch.cuda.synchronize(self.device),
                cleanup=abort_communicator,
            )
            dist.barrier(group=group)
        else:
            _run_rank_coordinated_init_stage(
                group=group,
                stage="IPC custom synchronization",
                operation=lambda: synchronize_fn(group=group, device=self.device),
                cleanup=abort_communicator,
            )

    @classmethod
    def from_communicator(
        cls,
        *,
        communicator,
        device: torch.device,
        attn_tp_size: int,
        max_rows: int,
        max_pull_size: int,
        max_push_size: int,
        source_push_capacity: int | None = None,
    ) -> AttnTPIPCResource:
        resource = cls.__new__(cls)
        resource._initialize_from_communicator(
            communicator=communicator,
            device=device,
            attn_tp_size=attn_tp_size,
            max_rows=max_rows,
            max_pull_size=max_pull_size,
            max_push_size=max_push_size,
            source_push_capacity=source_push_capacity,
        )
        return resource

    @staticmethod
    def _validate_max_rows(max_rows: int) -> None:
        if type(max_rows) is not int or max_rows <= 0:
            raise ValueError("max_rows must be a positive integer")

    @staticmethod
    def _validate_sizes(max_pull_size: int, max_push_size: int) -> None:
        for name, value in (
            ("max_pull_size", max_pull_size),
            ("max_push_size", max_push_size),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if max_pull_size == 0 and max_push_size == 0:
            raise ValueError("IPC resource must allocate pull or push capacity")

    def _initialize_from_communicator(
        self,
        *,
        communicator,
        device: torch.device,
        attn_tp_size: int,
        max_rows: int,
        max_pull_size: int,
        max_push_size: int,
        source_push_capacity: int | None,
    ) -> None:
        _validate_collective_shape(attn_tp_size=attn_tp_size)
        self._validate_max_rows(max_rows)
        self._validate_sizes(max_pull_size, max_push_size)
        if communicator is None or getattr(communicator, "disabled", True):
            raise RuntimeError("IPC communicator must be enabled")
        self.device = torch.device(device)
        self.attn_tp_size = attn_tp_size
        self.max_rows = max_rows
        self.max_pull_size = max_pull_size
        self.max_push_size = max_push_size
        if source_push_capacity is None:
            source_push_capacity = max_rows if max_push_size > 0 else 0
        if (
            type(source_push_capacity) is not int
            or source_push_capacity < 0
            or source_push_capacity > max_rows
            or (max_push_size == 0 and source_push_capacity != 0)
            or (max_push_size > 0 and source_push_capacity == 0)
        ):
            raise ValueError(
                "source_push_capacity must match the allocated push arena"
            )
        self.source_push_capacity = source_push_capacity
        self.communicator = communicator
        self.rank = int(getattr(communicator, "rank", 0))
        self._closed = False

    def require(
        self,
        *,
        attn_tp_size: int,
        min_rows: int,
        min_pull_size: int,
        min_push_size: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("AttnTP IPC resource is closed")
        _validate_collective_shape(attn_tp_size=attn_tp_size)
        if type(min_rows) is not int or min_rows <= 0:
            raise ValueError("min_rows must be a positive integer")
        for name, value in (
            ("min_pull_size", min_pull_size),
            ("min_push_size", min_push_size),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if attn_tp_size != self.attn_tp_size:
            raise ValueError("IPC resource AttnTP size does not match")
        if min_rows > self.max_rows:
            raise ValueError(
                f"required row capacity {min_rows} exceeds IPC resource "
                f"row capacity {self.max_rows}"
            )
        if min_pull_size > self.max_pull_size:
            raise ValueError(
                f"required pull capacity {min_pull_size} exceeds IPC resource "
                f"pull capacity {self.max_pull_size}"
            )
        if min_push_size > self.max_push_size:
            raise ValueError(
                f"required push capacity {min_push_size} exceeds IPC resource "
                f"push capacity {self.max_push_size}"
            )

    def capture(self):
        if self._closed:
            raise RuntimeError("AttnTP IPC resource is closed")
        return self.communicator.capture()

    def close(self) -> None:
        if self._closed:
            return
        self.communicator.close()
        self.communicator.disabled = True
        self._closed = True


class AttnTPSymmetricMemoryResource:
    """One maximum-capacity SymmetricMemory arena for an AttnTP group."""

    def __init__(
        self,
        *,
        group,
        device: torch.device,
        attn_tp_size: int,
        hidden_size: int,
        capacity: int,
        total_bytes: int,
        direct_signal_offset_bytes: int = 0,
        direct_signal_slots: int = 0,
    ) -> None:
        self._reset_local_state()
        validation = {}

        def validate_local():
            _validate_collective_shape(
                attn_tp_size=attn_tp_size,
                hidden_size=hidden_size,
            )
            if attn_tp_size == 1:
                raise ValueError("AttnTP1 does not require SymmetricMemory")
            if type(capacity) is not int or capacity <= 0:
                raise ValueError("symmetric resource capacity must be positive")
            input_bytes = capacity * hidden_size * torch.bfloat16.itemsize
            if (
                type(total_bytes) is not int
                or total_bytes < input_bytes
                or total_bytes % torch.bfloat16.itemsize != 0
            ):
                raise ValueError(
                    "symmetric resource total_bytes must cover aligned BF16 input"
                )
            if (
                type(direct_signal_offset_bytes) is not int
                or direct_signal_offset_bytes < 0
                or type(direct_signal_slots) is not int
                or direct_signal_slots < 0
            ):
                raise ValueError("direct Symm signal region must be non-negative")
            if direct_signal_slots and direct_signal_offset_bytes < input_bytes:
                raise ValueError("direct Symm signal region overlaps input payload")

            validated_device = torch.device(device)
            if validated_device.type != "cuda":
                raise ValueError("AttnTP SymmetricMemory resource requires CUDA")
            if dist.get_world_size(group=group) != attn_tp_size:
                raise ValueError("process-group size must match AttnTP size")
            rank = dist.get_rank(group=group)
            torch.cuda.set_device(validated_device)
            try:
                from torch.distributed import _symmetric_memory as torch_symm_mem
            except ImportError as error:
                raise RuntimeError("PyTorch SymmetricMemory is unavailable") from error

            validation.update(
                device=validated_device,
                rank=rank,
                torch_symm_mem=torch_symm_mem,
                input_bytes=input_bytes,
                input_elements=capacity * hidden_size,
                storage_elements=total_bytes // torch.bfloat16.itemsize,
            )
            return (
                validated_device.type,
                attn_tp_size,
                hidden_size,
                capacity,
                total_bytes,
                direct_signal_offset_bytes,
                direct_signal_slots,
            )

        _run_rank_coordinated_init_stage(
            group=group,
            stage="SymmetricMemory argument validation",
            operation=validate_local,
            result_key=lambda result: result,
        )
        self.device = validation["device"]
        self.rank = validation["rank"]
        self.attn_tp_size = attn_tp_size
        self.hidden_size = hidden_size
        self.capacity = capacity
        self.total_bytes = total_bytes
        self.direct_signal_offset_bytes = direct_signal_offset_bytes
        self.direct_signal_slots = direct_signal_slots

        def allocate_local_storage() -> None:
            self._storage = validation["torch_symm_mem"].empty(
                validation["storage_elements"],
                dtype=torch.bfloat16,
                device=self.device,
            )
            self._input = self._storage[: validation["input_elements"]].view(
                capacity,
                hidden_size,
            )

        _run_rank_coordinated_init_stage(
            group=group,
            stage="SymmetricMemory local allocation",
            operation=allocate_local_storage,
            cleanup=self._release_local_state,
        )

        group_name = getattr(group, "group_name", group)

        def rendezvous() -> None:
            self._handle = validation["torch_symm_mem"].rendezvous(
                self._storage,
                group_name,
            )

        _run_rank_coordinated_init_stage(
            group=group,
            stage="SymmetricMemory rendezvous",
            operation=rendezvous,
            cleanup=self._release_local_state,
        )
        _run_rank_coordinated_init_stage(
            group=group,
            stage="SymmetricMemory peer validation",
            operation=self._bind_peer_state,
            cleanup=self._release_local_state,
        )
        _run_rank_coordinated_init_stage(
            group=group,
            stage="SymmetricMemory device barrier",
            operation=lambda: self._handle.barrier(channel=0),
            cleanup=self._release_local_state,
        )

        def initialize_local_arena() -> None:
            self.signal_pad.zero_()
            if total_bytes > validation["input_bytes"]:
                self._storage[validation["input_elements"] :].zero_()
            torch.cuda.synchronize(self.device)

        _run_rank_coordinated_init_stage(
            group=group,
            stage="SymmetricMemory arena initialization",
            operation=initialize_local_arena,
            cleanup=self._release_local_state,
        )
        dist.barrier(group=group)
        self._closed = False

    def _reset_local_state(self) -> None:
        self._closed = True
        self._handle = None
        self._input = None
        self._storage = None
        self.signal_pad = None
        self.peer_pointer = 0
        self.peer_signal_pointer = 0
        self.pointer_table = 0
        self.multicast_pointer = 0

    def _bind_peer_state(self) -> None:
        buffer_ptrs = list(self._handle.buffer_ptrs)
        if len(buffer_ptrs) != self.attn_tp_size:
            raise RuntimeError(
                "SymmetricMemory returned an unexpected peer-pointer table"
            )
        if int(buffer_ptrs[self.rank]) != self._storage.data_ptr():
            raise RuntimeError("SymmetricMemory local pointer does not match the arena")
        self.pointer_table = int(self._handle.buffer_ptrs_dev)
        if self.pointer_table == 0:
            raise RuntimeError(
                "SymmetricMemory device buffer-pointer table is unavailable"
            )
        self.multicast_pointer = int(self._handle.multicast_ptr)
        if self.attn_tp_size == 2:
            self.peer_pointer = int(buffer_ptrs[1 - self.rank])
            if self.peer_pointer == 0:
                raise RuntimeError("SymmetricMemory peer pointer is unavailable")
        else:
            self.peer_pointer = 0

        self.signal_pad = self._handle.get_signal_pad(self.rank)
        if (
            self.signal_pad.dtype != torch.uint32
            or self.signal_pad.ndim != 1
            or not self.signal_pad.is_contiguous()
            or self.signal_pad.numel() < 2
        ):
            raise RuntimeError("SymmetricMemory signal pad has an invalid layout")
        signal_pad_ptrs = list(self._handle.signal_pad_ptrs)
        if len(signal_pad_ptrs) != self.attn_tp_size:
            raise RuntimeError(
                "SymmetricMemory returned an unexpected signal-pointer table"
            )
        self.peer_signal_pointer = (
            int(signal_pad_ptrs[1 - self.rank]) if self.attn_tp_size == 2 else 0
        )
        if self.attn_tp_size == 2 and self.peer_signal_pointer == 0:
            raise RuntimeError("SymmetricMemory peer signal pad is unavailable")

    def _release_local_state(self) -> None:
        self._handle = None
        self._input = None
        self._storage = None
        self.signal_pad = None
        self.peer_pointer = 0
        self.peer_signal_pointer = 0
        self.pointer_table = 0
        self.multicast_pointer = 0

    @property
    def storage(self) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("AttnTP SymmetricMemory resource is closed")
        return self._storage

    def require(
        self,
        *,
        attn_tp_size: int,
        hidden_size: int,
        capacity: int,
        required_bytes: int,
    ) -> None:
        if self._closed:
            raise RuntimeError("AttnTP SymmetricMemory resource is closed")
        _validate_collective_shape(
            attn_tp_size=attn_tp_size,
            hidden_size=hidden_size,
        )
        if attn_tp_size != self.attn_tp_size:
            raise ValueError("symmetric resource AttnTP size does not match")
        if hidden_size != self.hidden_size:
            raise ValueError("symmetric resource hidden size does not match")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("requested symmetric capacity must be positive")
        if capacity > self.capacity:
            raise ValueError(
                f"requested capacity {capacity} exceeds symmetric resource "
                f"capacity {self.capacity}"
            )
        if type(required_bytes) is not int or required_bytes <= 0:
            raise ValueError("required symmetric bytes must be positive")
        if required_bytes > self.total_bytes:
            raise ValueError(
                f"required bytes {required_bytes} exceed symmetric resource "
                f"bytes {self.total_bytes}"
            )

    def require_direct_symm(self, *, capacity: int) -> None:
        input_bytes = capacity * self.hidden_size * torch.bfloat16.itemsize
        if self.attn_tp_size >= 4:
            if self.multicast_pointer == 0:
                raise RuntimeError(
                    f"AttnTP{self.attn_tp_size} direct Symm requires multicast"
                )
            if (
                self.direct_signal_offset_bytes < input_bytes
                or self.direct_signal_slots < 2
            ):
                raise ValueError(
                    "symmetric resource has no compatible direct signal region"
                )

    def input_view_for(self, rows: int):
        if self._closed:
            raise RuntimeError("AttnTP SymmetricMemory resource is closed")
        if type(rows) is not int or rows <= 0:
            raise ValueError("symmetric input rows must be positive")
        if rows > self.capacity:
            raise ValueError(
                f"requested rows {rows} exceeds symmetric resource capacity "
                f"{self.capacity}"
            )
        return self._input[:rows]

    def capture(self):
        if self._closed:
            raise RuntimeError("AttnTP SymmetricMemory resource is closed")
        return nullcontext()

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        self._closed = True
        self._release_local_state()


@dataclass
class AttnTPFusedNormResourceSet:
    resources: tuple[object, ...]
    resources_by_family: Mapping[str, object]
    tile_layouts: Mapping[tuple[object, str], object]
    _closed: bool = field(default=False, init=False, repr=False)

    def resource_for(self, candidate):
        if self._closed:
            raise RuntimeError("production resource set is closed")
        try:
            return self.resources_by_family[candidate.family]
        except KeyError as error:
            raise RuntimeError(
                "production resource is missing for candidate family "
                f"{candidate.family}"
            ) from error

    def tile_layout_for(self, workload, candidate):
        if self._closed:
            raise RuntimeError("production resource set is closed")
        try:
            return self.tile_layouts[(workload, candidate.stable_id)]
        except KeyError as error:
            raise RuntimeError(
                "production Tile arena layout is missing for "
                f"{workload.encode()} and {candidate.stable_id}"
            ) from error

    def close(self) -> None:
        if self._closed:
            return
        for resource in reversed(self.resources):
            resource.close()
        self._closed = True


def build_candidate_resource_set(
    entries,
    *,
    group,
    device: torch.device,
) -> AttnTPFusedNormResourceSet:
    """Allocate at most one IPC and one Symm resource for candidate runners."""

    from sglang.jit_kernel.attntp_fused_norm.ipc import (
        AttnTPNormSpec,
        NormInternalPrecision,
        OutputMode,
        required_decode_owner_pull_buffer_bytes,
        required_decode_source_push_buffer_bytes,
        required_owner_pull_buffer_bytes,
        required_source_push_buffer_bytes,
    )
    from sglang.jit_kernel.attntp_fused_norm.candidates import (
        tile_tuning_from_production_candidate,
    )
    from sglang.jit_kernel.attntp_fused_norm.tile import (
        TilePipelineControlKey,
        shared_tile_pipeline_arena_layout,
    )
    from sglang.jit_kernel.attntp_fused_norm.symm import (
        _direct_symm_resource_layout,
    )

    candidate_entries = tuple(entries)
    if not candidate_entries:
        raise ValueError("production resource set requires at least one winner")
    first_key = candidate_entries[0][0]
    expected_shape = (
        first_key.phase,
        first_key.topology,
        first_key.attn_tp_size,
        first_key.hidden_size,
        first_key.output_mode,
    )
    for key, candidate in candidate_entries:
        if (
            key.phase,
            key.topology,
            key.attn_tp_size,
            key.hidden_size,
            key.output_mode,
        ) != expected_shape:
            raise ValueError(
                "production resource candidates must share one phase and topology"
            )
        if candidate.output_mode is not key.output_mode:
            raise ValueError("candidate output mode does not match workload key")
    if first_key.output_mode is not OutputMode.REPLICATED:
        raise ValueError("current production resources require replicated output")

    spec = AttnTPNormSpec(
        attn_tp_size=first_key.attn_tp_size,
        hidden_size=first_key.hidden_size,
        output_mode=first_key.output_mode,
        internal_precision=NormInternalPrecision.FULL_FP32,
    )
    device = torch.device(device)
    resources = []
    resources_by_family = {}
    tile_layouts = {}
    try:
        ipc_entries = tuple(
            (key, candidate)
            for key, candidate in candidate_entries
            if candidate.family in ("ipc_owner_pull", "ipc_source_push")
        )
        if ipc_entries:
            ipc_capacity = max(key.row_bucket for key, _candidate in ipc_entries)
            owner_pull_capacity = max(
                (
                    key.row_bucket
                    for key, candidate in ipc_entries
                    if candidate.family == "ipc_owner_pull"
                ),
                default=0,
            )
            source_push_capacity = max(
                (
                    key.row_bucket
                    for key, candidate in ipc_entries
                    if candidate.family == "ipc_source_push"
                ),
                default=0,
            )
            max_pull_size = 0
            if owner_pull_capacity:
                max_pull_size = (
                    required_owner_pull_buffer_bytes(owner_pull_capacity, spec)
                    if first_key.phase == "prefill"
                    else required_decode_owner_pull_buffer_bytes(
                        owner_pull_capacity, spec
                    )
                )
            max_push_size = 0
            if source_push_capacity:
                max_push_size = (
                    required_source_push_buffer_bytes(source_push_capacity, spec)
                    if first_key.phase == "prefill"
                    else required_decode_source_push_buffer_bytes(
                        source_push_capacity, spec
                    )
                )
            ipc_resource = AttnTPIPCResource(
                group=group,
                device=device,
                attn_tp_size=first_key.attn_tp_size,
                max_rows=ipc_capacity,
                max_pull_size=max_pull_size,
                max_push_size=max_push_size,
                source_push_capacity=source_push_capacity,
            )
            resources.append(ipc_resource)
            if owner_pull_capacity:
                resources_by_family["ipc_owner_pull"] = ipc_resource
            if source_push_capacity:
                resources_by_family["ipc_source_push"] = ipc_resource

        symm_entries = tuple(
            (key, candidate)
            for key, candidate in candidate_entries
            if candidate.family in ("direct_symm", "tile_pipeline")
        )
        if symm_entries:
            symm_capacity = max(key.row_bucket for key, _candidate in symm_entries)
            input_bytes = (
                symm_capacity * first_key.hidden_size * torch.bfloat16.itemsize
            )
            has_direct = any(
                candidate.family == "direct_symm" for _key, candidate in symm_entries
            )
            if has_direct:
                direct_total, signal_offset, signal_slots = (
                    _direct_symm_resource_layout(
                        device=device,
                        spec=spec,
                        rows=symm_capacity,
                    )
                )
            else:
                direct_total, signal_offset, signal_slots = (
                    input_bytes,
                    0,
                    0,
                )

            tile_entries = tuple(
                (key, candidate)
                for key, candidate in symm_entries
                if candidate.family == "tile_pipeline"
            )
            shared_layout = None
            if tile_entries:
                control_keys = {
                    TilePipelineControlKey(
                        capacity=key.row_bucket,
                        rows_per_tile=(
                            tile_tuning_from_production_candidate(
                                candidate
                            ).shape.rows_per_tile
                        ),
                        ring_stages=(
                            tile_tuning_from_production_candidate(
                                candidate
                            ).shape.ring_stages
                        ),
                    )
                    for key, candidate in tile_entries
                }
                shared_layout = shared_tile_pipeline_arena_layout(
                    capacity=symm_capacity,
                    spec=spec,
                    control_keys=control_keys,
                    reserved_bytes_after_input=direct_total - input_bytes,
                )
                total_bytes = max(direct_total, shared_layout.total_bytes)
            else:
                total_bytes = direct_total

            symm_resource = AttnTPSymmetricMemoryResource(
                group=group,
                device=device,
                attn_tp_size=first_key.attn_tp_size,
                hidden_size=first_key.hidden_size,
                capacity=symm_capacity,
                total_bytes=total_bytes,
                direct_signal_offset_bytes=signal_offset,
                direct_signal_slots=signal_slots,
            )
            resources.append(symm_resource)
            if has_direct:
                resources_by_family["direct_symm"] = symm_resource
            if shared_layout is not None:
                resources_by_family["tile_pipeline"] = symm_resource
                for key, candidate in tile_entries:
                    tuning = tile_tuning_from_production_candidate(candidate)
                    control_key = TilePipelineControlKey(
                        capacity=key.row_bucket,
                        rows_per_tile=tuning.shape.rows_per_tile,
                        ring_stages=tuning.shape.ring_stages,
                    )
                    tile_layouts[(key, candidate.stable_id)] = shared_layout.layout_for(
                        control_key
                    )

        supported = set(resources_by_family)
        unexpected = {
            candidate.family for _key, candidate in candidate_entries
        } - supported
        if unexpected:
            raise ValueError(
                "unsupported production resource families: "
                f"{tuple(sorted(unexpected))}"
            )
        return AttnTPFusedNormResourceSet(
            resources=tuple(resources),
            resources_by_family=MappingProxyType(resources_by_family),
            tile_layouts=MappingProxyType(tile_layouts),
        )
    except Exception:
        for resource in reversed(resources):
            resource.close()
        raise


def build_production_resource_set(
    winners,
    *,
    group,
    device: torch.device,
) -> AttnTPFusedNormResourceSet:
    """Allocate shared resources for the selected production winners."""

    entries = tuple(winners.items())
    if not entries:
        raise ValueError("production resource set requires at least one winner")
    candidates = []
    for key, winner in entries:
        if winner.row_bucket != key.row_bucket:
            raise ValueError("winner row bucket does not match workload key")
        if winner.candidate.output_mode is not key.output_mode:
            raise ValueError("winner output mode does not match workload key")
        candidates.append((key, winner.candidate))
    return build_candidate_resource_set(
        candidates,
        group=group,
        device=device,
    )
