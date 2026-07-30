from contextlib import nullcontext

import pytest
import torch

from sglang.jit_kernel.attntp_fused_norm import resources as resources_module
from sglang.jit_kernel.attntp_fused_norm.ipc import (
    AttnTPNormSpec,
    DecodeAttnTPFusedIPCNormRunner,
    DecodeCommunicationAlgorithm,
    NormInternalPrecision,
    OutputMode,
    PrefillAttnTPFusedIPCNormRunner,
    PrefillCommunicationAlgorithm,
    required_owner_pull_buffer_bytes,
    required_source_push_buffer_bytes,
)
from sglang.jit_kernel.attntp_fused_norm.resources import (
    AttnTPIPCResource,
    AttnTPSymmetricMemoryResource,
    build_candidate_resource_set,
    build_production_resource_set,
)
from sglang.jit_kernel.attntp_fused_norm.tuning import (
    AttnTPFusedNormCandidate,
    AttnTPFusedNormWinner,
    AttnTPFusedNormWorkloadKey,
)
from sglang.jit_kernel.attntp_fused_norm.symm import (
    PrefillAttnTPFusedSymmNormRunner,
)
from sglang.jit_kernel.attntp_fused_norm.tile import (
    PrefillAttnTPReplicatedTilePipelineRunner,
    TilePipelineControlKey,
    shared_tile_pipeline_arena_layout,
)
from sglang.srt.distributed.device_communicators import custom_all_reduce_v2


def test_ipc_resource_enforces_capacity_and_closes_communicator_once() -> None:
    communicator = _FakeCommunicator()
    resource = AttnTPIPCResource.from_communicator(
        communicator=communicator,
        device=torch.device("cuda:0"),
        attn_tp_size=2,
        max_rows=128,
        max_pull_size=4096,
        max_push_size=8192,
    )

    resource.require(
        attn_tp_size=2,
        min_rows=128,
        min_pull_size=1024,
        min_push_size=8192,
    )
    with pytest.raises(ValueError, match="push capacity"):
        resource.require(
            attn_tp_size=2,
            min_rows=128,
            min_pull_size=0,
            min_push_size=8193,
        )

    resource.close()
    resource.close()

    assert communicator.close_calls == 1
    assert communicator.disabled


def test_rank_coordinated_prepare_failure_skips_finalize_and_cleans_local_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = object()
    finalized = []
    cleaned = []

    monkeypatch.setattr(resources_module.dist, "get_world_size", lambda group: 2)

    def report_peer_prepare_failure(gathered, local_payload, *, group) -> None:
        assert group == "attntp-cpu-group"
        gathered[:] = (
            local_payload,
            (
                "IPC local prepare",
                "RuntimeError: peer CUDA allocation failed",
                None,
            ),
        )

    monkeypatch.setattr(
        resources_module.dist,
        "all_gather_object",
        report_peer_prepare_failure,
    )

    with pytest.raises(
        RuntimeError,
        match="IPC local prepare.*rank1=RuntimeError: peer CUDA allocation failed",
    ):
        local = resources_module._run_rank_coordinated_init_stage(
            group="attntp-cpu-group",
            stage="IPC local prepare",
            operation=lambda: prepared,
            cleanup=lambda: cleaned.append(prepared),
        )
        finalized.append(local)

    assert finalized == []
    assert cleaned == [prepared]


def test_deferred_custom_all_reduce_does_not_exchange_handles_until_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = _FakeCustomAllReduceObject()
    exchanged = []
    monkeypatch.setattr(custom_all_reduce_v2.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(custom_all_reduce_v2.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "get_custom_all_reduce_cls",
        lambda: lambda **_kwargs: obj,
    )

    communicator = custom_all_reduce_v2.CustomAllReduceV2.create_deferred(
        "attntp-cpu-group",
        torch.device("cuda:0"),
        max_pull_size=4096,
        max_push_size=8192,
    )
    monkeypatch.setattr(
        communicator,
        "_share_list",
        lambda handles: exchanged.append(tuple(handles)) or [[101], [202]],
    )

    assert obj.share_storage_calls == 1
    assert obj.post_init_calls == []
    assert exchanged == []
    assert communicator.disabled

    communicator.finalize()

    assert exchanged == [(101,)]
    assert obj.post_init_calls == [[101, 202]]
    assert not communicator.disabled
    communicator.abort()


def test_custom_all_reduce_default_constructor_still_finalizes_eagerly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = _FakeCustomAllReduceObject()
    monkeypatch.setattr(custom_all_reduce_v2.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(custom_all_reduce_v2.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        custom_all_reduce_v2.CustomAllReduceV2,
        "is_supported",
        staticmethod(lambda **_kwargs: True),
    )
    monkeypatch.setattr(
        custom_all_reduce_v2.CustomAllReduceV2,
        "_share_list",
        lambda _self, _handles: [[101], [202]],
    )
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "get_custom_all_reduce_cls",
        lambda: lambda **_kwargs: obj,
    )

    communicator = custom_all_reduce_v2.CustomAllReduceV2(
        "attntp-cpu-group",
        torch.device("cuda:0"),
        max_pull_size=4096,
        max_push_size=8192,
    )

    assert obj.post_init_calls == [[101, 202]]
    assert not communicator.disabled
    communicator.abort()


def test_deferred_custom_all_reduce_abort_is_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = _FakeCustomAllReduceObject()
    monkeypatch.setattr(custom_all_reduce_v2.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(custom_all_reduce_v2.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        custom_all_reduce_v2,
        "get_custom_all_reduce_cls",
        lambda: lambda **_kwargs: obj,
    )
    monkeypatch.setattr(
        custom_all_reduce_v2.dist,
        "barrier",
        lambda **_kwargs: pytest.fail("abort must not enter a collective"),
    )
    communicator = custom_all_reduce_v2.CustomAllReduceV2.create_deferred(
        "attntp-cpu-group",
        torch.device("cuda:0"),
        max_pull_size=4096,
        max_push_size=8192,
    )

    communicator.abort()
    communicator.abort()

    assert obj.free_ipc_handles_calls == 1
    assert obj.free_storage_calls == 1
    assert communicator.disabled


def test_symmetric_peer_allocation_failure_skips_rendezvous_and_releases_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symmetric_memory = _FakeSymmetricMemory()
    monkeypatch.setattr(resources_module.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(resources_module.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(
        torch.distributed,
        "_symmetric_memory",
        symmetric_memory,
        raising=False,
    )

    def report_peer_allocation_failure(gathered, local_payload, *, group) -> None:
        assert group == "attntp-cpu-group"
        if local_payload[0] == "SymmetricMemory local allocation":
            gathered[:] = (
                local_payload,
                (
                    local_payload[0],
                    "RuntimeError: peer CUDA allocation failed",
                    None,
                ),
            )
        else:
            gathered[:] = (local_payload, local_payload)

    monkeypatch.setattr(
        resources_module.dist,
        "all_gather_object",
        report_peer_allocation_failure,
    )
    resource = object.__new__(AttnTPSymmetricMemoryResource)

    with pytest.raises(
        RuntimeError,
        match=(
            "SymmetricMemory local allocation.*"
            "rank1=RuntimeError: peer CUDA allocation failed"
        ),
    ):
        resource.__init__(
            group="attntp-cpu-group",
            device=torch.device("cuda:0"),
            attn_tp_size=2,
            hidden_size=2048,
            capacity=1,
            total_bytes=2048 * torch.bfloat16.itemsize,
        )

    assert symmetric_memory.empty_calls == 1
    assert symmetric_memory.rendezvous_calls == 0
    assert resource._storage is None
    assert resource._input is None


def test_symmetric_resource_maximum_capacity_serves_smaller_views() -> None:
    resource = object.__new__(AttnTPSymmetricMemoryResource)
    resource.attn_tp_size = 2
    resource.hidden_size = 8
    resource.capacity = 16
    resource.total_bytes = 16 * 8 * torch.bfloat16.itemsize
    resource.device = torch.device("cpu")
    resource._closed = False
    resource._input = torch.empty((16, 8), dtype=torch.bfloat16)

    small = resource.input_view_for(5)
    full = resource.input_view_for(16)

    assert small.shape == (5, 8)
    assert full.shape == (16, 8)
    assert small.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="exceeds symmetric resource capacity"):
        resource.input_view_for(17)


def test_symmetric_resource_close_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = object.__new__(AttnTPSymmetricMemoryResource)
    resource.device = torch.device("cuda:0")
    resource._closed = False
    resource._handle = object()
    resource._input = object()
    resource._storage = object()
    resource.signal_pad = object()
    resource.peer_pointer = 1
    resource.peer_signal_pointer = 2
    resource.pointer_table = 3
    resource.multicast_pointer = 4
    synchronized = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: synchronized.append(device),
    )

    resource.close()
    resource.close()

    assert synchronized == [torch.device("cuda:0")]
    assert resource._storage is None
    assert resource.pointer_table == 0


def test_shared_tile_arena_reuses_payload_but_isolates_ticket_state() -> None:
    spec = _replicated_full_fp32_spec()
    small_key = TilePipelineControlKey(
        capacity=1024,
        rows_per_tile=4,
        ring_stages=8,
    )
    large_key = TilePipelineControlKey(
        capacity=4096,
        rows_per_tile=4,
        ring_stages=8,
    )
    plan = shared_tile_pipeline_arena_layout(
        capacity=4096,
        spec=spec,
        control_keys=(small_key, large_key),
    )
    small = plan.layout_for(small_key)
    large = plan.layout_for(large_key)

    assert plan.input_bytes == 4096 * 2048 * torch.bfloat16.itemsize
    assert small.input_bytes == 1024 * 2048 * torch.bfloat16.itemsize
    assert large.input_bytes == plan.input_bytes
    assert small.reduced_offset_bytes >= plan.input_bytes
    assert large.reduced_offset_bytes >= plan.input_bytes
    assert small.control_offset_bytes != large.control_offset_bytes
    assert small.kernel_done_offset_bytes != large.kernel_done_offset_bytes
    assert small.total_bytes <= plan.total_bytes
    assert large.total_bytes <= plan.total_bytes


def test_shared_tile_arena_starts_after_direct_symm_signal_region() -> None:
    spec = _replicated_full_fp32_spec()
    key = TilePipelineControlKey(4096, 4, 8)
    direct_signal_bytes = 8192
    plan = shared_tile_pipeline_arena_layout(
        capacity=4096,
        spec=spec,
        control_keys=(key,),
        reserved_bytes_after_input=direct_signal_bytes,
    )
    layout = plan.layout_for(key)

    assert layout.reduced_offset_bytes >= plan.input_bytes + direct_signal_bytes
    assert layout.control_offset_bytes > layout.reduced_offset_bytes


def test_prepared_tile_configs_share_resource_without_owning_it() -> None:
    spec = _replicated_full_fp32_spec()
    small_key = TilePipelineControlKey(1024, 4, 8)
    large_key = TilePipelineControlKey(4096, 4, 8)
    plan = shared_tile_pipeline_arena_layout(
        capacity=4096,
        spec=spec,
        control_keys=(small_key, large_key),
    )
    resource = _FakeSymmetricResource(capacity=4096)

    small = PrefillAttnTPReplicatedTilePipelineRunner.from_resource(
        resource=resource,
        spec=spec,
        capacity=1024,
        arena_layout=plan.layout_for(small_key),
        rows_per_tile=4,
        ring_stages=8,
    )
    large = PrefillAttnTPReplicatedTilePipelineRunner.from_resource(
        resource=resource,
        spec=spec,
        capacity=4096,
        arena_layout=plan.layout_for(large_key),
        rows_per_tile=4,
        ring_stages=8,
    )

    assert small._resource is resource
    assert large._resource is resource
    assert small._input.payload is large._input.payload
    assert small.arena_layout.control_offset_bytes != (
        large.arena_layout.control_offset_bytes
    )

    small.close()
    large.close()
    assert resource.close_calls == 0


def test_prepared_ipc_algorithms_use_fixed_source_push_arena_capacity() -> None:
    spec = AttnTPNormSpec(
        attn_tp_size=2,
        hidden_size=2048,
        output_mode=OutputMode.REPLICATED,
        internal_precision=NormInternalPrecision.FULL_FP32,
    )
    resource = _FakeIPCResource()

    source_push = PrefillAttnTPFusedIPCNormRunner.from_resource(
        resource=resource,
        spec=spec,
        capacity=16,
        algorithm=PrefillCommunicationAlgorithm.SOURCE_PUSH,
    )
    owner_pull = PrefillAttnTPFusedIPCNormRunner.from_resource(
        resource=resource,
        spec=spec,
        capacity=4096,
        algorithm=PrefillCommunicationAlgorithm.OWNER_PULL,
    )

    assert source_push._resource is resource
    assert owner_pull._resource is resource
    assert source_push.communicator is owner_pull.communicator
    assert source_push.capacity == 16
    assert source_push.arena_capacity == resource.source_push_capacity

    decode_source_push = DecodeAttnTPFusedIPCNormRunner.from_resource(
        resource=resource,
        spec=spec,
        max_rows=16,
        algorithm=DecodeCommunicationAlgorithm.SOURCE_PUSH,
    )
    assert decode_source_push.max_rows == 16
    assert decode_source_push.arena_rows == resource.source_push_capacity

    source_push.close()
    owner_pull.close()
    decode_source_push.close()
    assert resource.close_calls == 0


def test_prepared_direct_symm_configs_share_one_payload_resource() -> None:
    spec = _replicated_full_fp32_spec()
    resource = _FakeSymmetricResource(capacity=4096)

    block_128 = PrefillAttnTPFusedSymmNormRunner.from_resource(
        resource=resource,
        spec=spec,
        capacity=4096,
        block_size=128,
    )
    block_256 = PrefillAttnTPFusedSymmNormRunner.from_resource(
        resource=resource,
        spec=spec,
        capacity=4096,
        block_size=256,
    )

    assert block_128._resource is resource
    assert block_256._resource is resource
    assert block_128._input.payload is block_256._input.payload

    block_128.close()
    block_256.close()
    assert resource.close_calls == 0


def test_production_resource_set_allocates_one_ipc_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_1k = _production_key(1024)
    key_4k = _production_key(4096)
    candidate = AttnTPFusedNormCandidate(
        family="ipc_source_push",
        output_mode=OutputMode.REPLICATED,
        parameters=(
            ("block_size", 256),
            ("blocks_per_sm", 4),
            ("rows_per_tile", 4),
            ("signal_backoff", 64),
        ),
    )
    winners = {
        key: AttnTPFusedNormWinner(
            candidate=candidate,
            row_bucket=key.row_bucket,
            median_ms=0.1,
            p90_ms=0.11,
        )
        for key in (key_1k, key_4k)
    }
    allocations = []
    resource = object()
    monkeypatch.setattr(
        "sglang.jit_kernel.attntp_fused_norm.resources.AttnTPIPCResource",
        lambda **kwargs: allocations.append(kwargs) or resource,
    )

    resource_set = build_production_resource_set(
        winners,
        group="attntp-cpu-group",
        device=torch.device("cuda:3"),
    )

    assert resource_set.resources == (resource,)
    assert resource_set.resource_for(candidate) is resource
    assert len(allocations) == 1
    assert allocations[0]["group"] == "attntp-cpu-group"
    assert allocations[0]["device"] == torch.device("cuda:3")
    assert allocations[0]["max_rows"] == 4096
    assert allocations[0]["max_pull_size"] == 0
    assert allocations[0]["max_push_size"] > 0


def test_production_resource_set_allocates_pull_only_for_owner_pull(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_1k = _production_key(1024)
    key_4k = _production_key(4096)
    candidate = AttnTPFusedNormCandidate(
        family="ipc_owner_pull",
        output_mode=OutputMode.REPLICATED,
        parameters=(
            ("block_size", 256),
            ("blocks_per_sm", 4),
            ("rows_per_tile", 0),
            ("signal_backoff", 0),
        ),
    )
    winners = {
        key: AttnTPFusedNormWinner(
            candidate=candidate,
            row_bucket=key.row_bucket,
            median_ms=0.1,
            p90_ms=0.11,
        )
        for key in (key_1k, key_4k)
    }
    allocations = []
    resource = object()
    monkeypatch.setattr(
        "sglang.jit_kernel.attntp_fused_norm.resources.AttnTPIPCResource",
        lambda **kwargs: allocations.append(kwargs) or resource,
    )

    resource_set = build_production_resource_set(
        winners,
        group="attntp-cpu-group",
        device=torch.device("cuda:3"),
    )

    assert resource_set.resources == (resource,)
    assert resource_set.resource_for(candidate) is resource
    assert len(allocations) == 1
    assert allocations[0]["max_rows"] == 4096
    assert allocations[0]["max_pull_size"] > 0
    assert allocations[0]["max_push_size"] == 0


def test_candidate_resource_set_shares_one_resource_across_tuning_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _production_key(4096)
    candidates = tuple(
        AttnTPFusedNormCandidate(
            family="ipc_source_push",
            output_mode=OutputMode.REPLICATED,
            parameters=(
                ("block_size", block_size),
                ("blocks_per_sm", 4),
                ("rows_per_tile", 4),
                ("signal_backoff", 64),
            ),
        )
        for block_size in (128, 256)
    )
    allocations = []
    resource = _FakeIPCResource()
    monkeypatch.setattr(
        "sglang.jit_kernel.attntp_fused_norm.resources.AttnTPIPCResource",
        lambda **kwargs: allocations.append(kwargs) or resource,
    )

    resource_set = build_candidate_resource_set(
        tuple((key, candidate) for candidate in candidates),
        group="attntp-cpu-group",
        device=torch.device("cuda:3"),
    )

    assert len(allocations) == 1
    assert resource_set.resources == (resource,)
    assert all(
        resource_set.resource_for(candidate) is resource for candidate in candidates
    )
    resource_set.close()
    resource_set.close()
    assert resource.close_calls == 1


def test_candidate_resource_set_shares_one_ipc_resource_for_pull_and_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_key = _production_key(4096)
    owner_key = _production_key(64)
    source_push = AttnTPFusedNormCandidate(
        family="ipc_source_push",
        output_mode=OutputMode.REPLICATED,
        parameters=(
            ("block_size", 256),
            ("blocks_per_sm", 4),
            ("rows_per_tile", 4),
            ("signal_backoff", 64),
        ),
    )
    owner_pull = AttnTPFusedNormCandidate(
        family="ipc_owner_pull",
        output_mode=OutputMode.REPLICATED,
        parameters=(
            ("block_size", 256),
            ("blocks_per_sm", 4),
            ("rows_per_tile", 0),
            ("signal_backoff", 0),
        ),
    )
    allocations = []
    resource = _FakeIPCResource()
    monkeypatch.setattr(
        "sglang.jit_kernel.attntp_fused_norm.resources.AttnTPIPCResource",
        lambda **kwargs: allocations.append(kwargs) or resource,
    )

    resource_set = build_candidate_resource_set(
        ((source_key, source_push), (owner_key, owner_pull)),
        group="attntp-cpu-group",
        device=torch.device("cuda:3"),
    )

    spec = _replicated_full_fp32_spec()
    assert len(allocations) == 1
    assert allocations[0]["max_rows"] == 4096
    assert allocations[0]["max_pull_size"] == required_owner_pull_buffer_bytes(
        64, spec
    )
    assert allocations[0]["max_push_size"] == required_source_push_buffer_bytes(
        4096, spec
    )
    assert resource_set.resources == (resource,)
    assert resource_set.resource_for(source_push) is resource
    assert resource_set.resource_for(owner_pull) is resource


class _FakeCommunicator:
    def __init__(self) -> None:
        self.obj = object()
        self.disabled = False
        self.close_calls = 0

    def capture(self):
        return nullcontext()

    def close(self) -> None:
        self.close_calls += 1


class _FakeCustomAllReduceObject:
    def __init__(self) -> None:
        self.share_storage_calls = 0
        self.post_init_calls = []
        self.free_ipc_handles_calls = 0
        self.free_storage_calls = 0

    def share_storage(self):
        self.share_storage_calls += 1
        return 101

    def post_init(self, handles) -> None:
        self.post_init_calls.append(handles)

    def free_ipc_handles(self) -> None:
        self.free_ipc_handles_calls += 1

    def free_storage(self) -> None:
        self.free_storage_calls += 1


class _FakeSymmetricMemory:
    def __init__(self) -> None:
        self.empty_calls = 0
        self.rendezvous_calls = 0

    def empty(self, elements, *, dtype, device):
        self.empty_calls += 1
        assert dtype == torch.bfloat16
        assert device == torch.device("cuda:0")
        return torch.empty(elements, dtype=dtype)

    def rendezvous(self, _storage, _group_name):
        self.rendezvous_calls += 1
        pytest.fail("rendezvous must not run after a peer allocation failure")


class _FakeIPCResource:
    def __init__(self) -> None:
        self.device = torch.device("cuda:0")
        self.rank = 0
        self.attn_tp_size = 2
        self.max_rows = 4096
        self.source_push_capacity = 4096
        self.communicator = _FakeCommunicator()
        self.close_calls = 0
        self._closed = False

    def require(
        self,
        *,
        attn_tp_size: int,
        min_rows: int,
        min_pull_size: int,
        min_push_size: int,
    ) -> None:
        assert attn_tp_size == self.attn_tp_size
        assert min_rows <= self.max_rows
        assert min_pull_size >= 0
        assert min_push_size >= 0

    def capture(self):
        return self.communicator.capture()

    def close(self) -> None:
        self.close_calls += 1


class _FakeView:
    def __init__(self, payload, rows: int) -> None:
        self.payload = payload
        self.rows = rows


class _FakeSymmetricResource:
    def __init__(self, *, capacity: int) -> None:
        self.device = torch.device("cuda:0")
        self.rank = 0
        self.attn_tp_size = 2
        self.hidden_size = 2048
        self.capacity = capacity
        self.total_bytes = 1 << 30
        self.pointer_table = 1234
        self.multicast_pointer = 0
        self.direct_signal_offset_bytes = 0
        self.direct_signal_slots = 0
        self.peer_pointer = 4321
        self.signal_pad = object()
        self.peer_signal_pointer = 5678
        self.close_calls = 0
        self._closed = False
        self._payload = object()

    def require(
        self,
        *,
        attn_tp_size: int,
        hidden_size: int,
        capacity: int,
        required_bytes: int,
    ) -> None:
        assert attn_tp_size == self.attn_tp_size
        assert hidden_size == self.hidden_size
        assert capacity <= self.capacity
        assert required_bytes <= self.total_bytes

    def input_view_for(self, rows: int):
        return _FakeView(self._payload, rows)

    def require_direct_symm(self, *, capacity: int) -> None:
        assert capacity <= self.capacity

    def capture(self):
        return nullcontext()

    def close(self) -> None:
        self.close_calls += 1


def _replicated_full_fp32_spec() -> AttnTPNormSpec:
    return AttnTPNormSpec(
        attn_tp_size=2,
        hidden_size=2048,
        output_mode=OutputMode.REPLICATED,
        internal_precision=NormInternalPrecision.FULL_FP32,
    )


def _production_key(rows: int) -> AttnTPFusedNormWorkloadKey:
    return AttnTPFusedNormWorkloadKey.current(
        phase="prefill",
        topology="cp",
        attn_tp_size=2,
        hidden_size=2048,
        row_bucket=rows,
        execution="eager",
    )
