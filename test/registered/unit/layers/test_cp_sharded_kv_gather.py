"""CPU contract tests for compact Phase 2 sharded-KV prefill gathers."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.attention.cp_sharded_kv as cp_sharded_kv
from sglang.srt.context_parallel import (
    build_cp_prefill_split_spec,
    contract_cp_prefill_runtime_to_last_q,
    materialize_cp_prefill_runtime_layout,
)
from sglang.srt.distributed.parallel_state import (
    GroupCoordinator,
    should_enable_attn_tp_pynccl,
)
from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator
from sglang.srt.layers.attention.cp_sharded_kv import (
    CPShardedKVPageTableResolver,
    build_cp_prefill_kv_gather_plan,
    build_cp_prefill_kv_source_push_plan,
    build_cp_prefill_swa_gather_plan,
    gather_cp_prefill_kv,
    gather_cp_prefill_swa_kv,
    restore_rank_packed_rows,
)
from sglang.srt.mem_cache.cp_sharded_allocator import CPShardedKVPoolAllocator
from sglang.srt.mem_cache.cp_sharded_residency import CPLogicalOwnerPlan
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


@dataclass
class _FakeAllocator:
    owner_ranks: torch.Tensor
    cp_rank: int
    cp_size: int

    def owner_plan_for_logical_slots(self, logical_slots):
        assert logical_slots.shape == self.owner_ranks.shape
        counts = torch.bincount(self.owner_ranks, minlength=self.cp_size).tolist()
        return CPLogicalOwnerPlan(
            owner_ranks=self.owner_ranks,
            per_rank_counts=tuple(int(count) for count in counts),
            rank_packed_to_logical=torch.argsort(self.owner_ranks, stable=True),
        )

    def logical_slots_to_physical(self, logical_slots):
        return logical_slots + 1000


class _FakeCPGroup:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.rank_in_group = 0
        self.world_size = 2

    def all_gatherv(self, tensors, sizes):
        self.calls.append((tensors, sizes))
        return self.outputs.pop(0)


class _FakeAllToAllVGroup:
    def __init__(self, rank, outputs):
        self.rank_in_group = rank
        self.outputs = list(outputs)
        self.calls = []

    def all_to_allv(self, tensors, send_sizes, recv_sizes):
        self.calls.append((tensors, tuple(send_sizes), tuple(recv_sizes)))
        return self.outputs.pop(0)


class _FakePyNccl:
    def __init__(self):
        self.available = True
        self.disabled = True
        self.group_start_count = 0
        self.group_end_count = 0
        self.gather_calls = []
        self.reduce_scatter_calls = []
        self.send_calls = []
        self.recv_calls = []

    @contextmanager
    def change_state(self, enable):
        assert enable
        old_disabled = self.disabled
        self.disabled = False
        try:
            yield
        finally:
            self.disabled = old_disabled

    def group_start(self):
        self.group_start_count += 1

    def group_end(self):
        self.group_end_count += 1

    def all_gather(self, output, input_, sizes):
        self.gather_calls.append((output, input_, sizes))

    def reduce_scatter(self, output, input_, sizes):
        self.reduce_scatter_calls.append((output, input_, sizes))

    def send(self, tensor, dst):
        self.send_calls.append((tensor, dst))

    def recv(self, tensor, src):
        self.recv_calls.append((tensor, src))
        tensor.fill_(src)


class _FakeNcclLibrary:
    def __init__(self):
        self.broadcasts = []
        self.reduces = []

    def ncclGroupStart(self):
        pass

    def ncclGroupEnd(self):
        pass

    def ncclBroadcast(self, _send, _recv, count, _dtype, root, _comm, _stream):
        self.broadcasts.append((count, root))

    def ncclReduce(
        self, _send, _recv, count, _dtype, _op, root, _comm, _stream
    ):
        self.reduces.append((count, root))


def _runtime(spec, cp_rank: int):
    out_cache_loc = torch.zeros(spec.extend_len, dtype=torch.int64)
    for block in spec.local_blocks(cp_rank):
        start = block.logical_start - spec.extend_start
        end = start + block.token_count
        out_cache_loc[start:end] = torch.arange(start, end) + 2000
    return materialize_cp_prefill_runtime_layout(
        spec=spec,
        cp_rank=cp_rank,
        input_ids=torch.arange(spec.extend_len),
        positions=torch.arange(
            spec.extend_start,
            spec.extend_start + spec.extend_len,
            dtype=torch.int32,
        ),
        out_cache_loc=out_cache_loc,
    )


def test_short_empty_q_rank_can_use_bounded_full_cp_kv_participation():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=16,
        cp_size=4,
        page_size=16,
        owner_rotation=0,
    )

    assert cp_sharded_kv.should_use_full_cp_kv_collective(
        _runtime(spec, cp_rank=0), page_size=16
    )


def test_long_prefix_with_short_extend_keeps_selective_cp_kv_participation():
    spec = build_cp_prefill_split_spec(
        extend_start=64,
        extend_len=16,
        cp_size=4,
        page_size=16,
        owner_rotation=0,
    )

    assert not cp_sharded_kv.should_use_full_cp_kv_collective(
        _runtime(spec, cp_rank=0), page_size=16
    )


def test_contracted_q_does_not_make_a_fully_owned_split_use_full_collective():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=64,
        cp_size=4,
        page_size=16,
        owner_rotation=0,
    )
    runtime = contract_cp_prefill_runtime_to_last_q(_runtime(spec, cp_rank=0))

    assert all(count > 0 for count in spec.per_rank_tokens)
    assert any(count == 0 for count in runtime.active_tokens_per_cp_rank())
    assert not cp_sharded_kv.should_use_full_cp_kv_collective(
        runtime, page_size=16
    )


def test_gather_plan_uses_residency_owners_for_mixed_prefix():
    spec = build_cp_prefill_split_spec(
        extend_start=8,
        extend_len=24,
        cp_size=3,
        page_size=4,
        owner_rotation=1,
    )
    runtime = _runtime(spec, cp_rank=1)
    prefix_logical_slots = torch.tensor([11, 12, 21, 22, 31, 32, 41, 42])
    prefix_owners = torch.tensor([2, 0, 1, 2, 0, 1, 2, 0])
    allocator = _FakeAllocator(prefix_owners, cp_rank=1, cp_size=3)

    plan = build_cp_prefill_kv_gather_plan(
        prefix_logical_slots=prefix_logical_slots,
        runtime_layout=runtime,
        allocator=allocator,
    )

    assert plan.prefix.sizes == (3, 2, 3)
    assert torch.equal(
        plan.prefix.local_physical_slots,
        torch.tensor([1021, 1032]),
    )
    assert torch.equal(
        plan.prefix.rank_packed_to_logical,
        torch.tensor([1, 4, 7, 2, 5, 0, 3, 6]),
    )
    assert plan.prefix.logical_token_count == 8
    assert plan.extend.sizes == spec.per_rank_tokens
    assert torch.equal(
        plan.extend.local_physical_slots,
        runtime.local_out_cache_loc,
    )
    expected_extend_map = torch.tensor(
        [
            logical_index - spec.extend_start
            for rank in range(3)
            for block in spec.local_blocks(rank)
            for logical_index in range(
                block.logical_start,
                block.logical_start + block.token_count,
            )
        ]
    )
    assert torch.equal(plan.extend.rank_packed_to_logical, expected_extend_map)
    assert plan.extend.logical_token_count == spec.extend_len


def test_rank_packed_rows_restore_exact_logical_order():
    spec = build_cp_prefill_split_spec(
        extend_start=8,
        extend_len=24,
        cp_size=3,
        page_size=4,
        owner_rotation=1,
    )
    runtime = _runtime(spec, cp_rank=1)
    prefix_slots = torch.arange(8) + 10
    owners = torch.tensor([2, 0, 1, 2, 0, 1, 2, 0])
    plan = build_cp_prefill_kv_gather_plan(
        prefix_logical_slots=prefix_slots,
        runtime_layout=runtime,
        allocator=_FakeAllocator(owners, cp_rank=1, cp_size=3),
    )
    logical_rows = torch.arange(8, dtype=torch.float32).view(8, 1, 1)
    packed_rows = logical_rows.index_select(
        0, plan.prefix.rank_packed_to_logical
    )

    restored = restore_rank_packed_rows(packed_rows, plan.prefix)

    assert torch.equal(restored, logical_rows)


def test_swa_prefix_slots_translate_from_owner_local_full_slots():
    allocator = object.__new__(CPShardedKVPoolAllocator)
    allocator.base_allocator = SimpleNamespace(
        translate_loc_from_full_to_swa=lambda slots: slots + 100
    )
    resolver = CPShardedKVPageTableResolver(allocator)

    resolved = resolver.resolve_swa_slots(torch.tensor([0, 8, 9]))

    assert resolved.tolist() == [0, 108, 109]


def test_gather_plan_accepts_empty_prefix_and_zero_local_extend():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=1,
        cp_size=4,
        page_size=4,
        owner_rotation=0,
    )
    runtime = _runtime(spec, cp_rank=3)
    allocator = _FakeAllocator(
        torch.empty((0,), dtype=torch.int64), cp_rank=3, cp_size=4
    )

    plan = build_cp_prefill_kv_gather_plan(
        prefix_logical_slots=torch.empty((0,), dtype=torch.int64),
        runtime_layout=runtime,
        allocator=allocator,
    )

    assert plan.prefix.sizes == (0, 0, 0, 0)
    assert plan.prefix.local_physical_slots.shape == (0,)
    assert plan.prefix.rank_packed_to_logical.shape == (0,)
    assert plan.extend.sizes == (1, 0, 0, 0)
    assert plan.extend.local_physical_slots.shape == (0,)
    assert plan.extend.rank_packed_to_logical.tolist() == [0]


def test_source_push_plan_uses_physical_prefix_and_owner_local_extend_rows():
    spec = build_cp_prefill_split_spec(
        extend_start=8,
        extend_len=24,
        cp_size=3,
        page_size=4,
        owner_rotation=1,
    )
    runtime = _runtime(spec, cp_rank=1)
    prefix_owners = torch.tensor([2, 0, 1, 2, 0, 1, 2, 0])
    gather_plan = build_cp_prefill_kv_gather_plan(
        prefix_logical_slots=torch.tensor([11, 12, 21, 22, 31, 32, 41, 42]),
        runtime_layout=runtime,
        allocator=_FakeAllocator(prefix_owners, cp_rank=1, cp_size=3),
    )

    plan = build_cp_prefill_kv_source_push_plan(
        gather_plan=gather_plan,
        cp_rank=1,
    )

    assert plan.source_mask == 0b111
    assert plan.prefix.source_rows.dtype == torch.int32
    assert plan.prefix.destination_rows.dtype == torch.int32
    assert plan.prefix.source_rows.tolist() == [1021, 1032]
    assert plan.prefix.destination_rows.tolist() == [2, 5]
    assert plan.extend.source_rows.tolist() == list(
        range(gather_plan.extend.sizes[1])
    )
    extend_offset = sum(gather_plan.extend.sizes[:1])
    expected_extend_destinations = (
        gather_plan.extend.rank_packed_to_logical.narrow(
            0,
            extend_offset,
            gather_plan.extend.sizes[1],
        )
        + gather_plan.prefix.logical_token_count
    )
    assert (
        plan.extend.destination_rows.tolist()
        == expected_extend_destinations.tolist()
    )
    assert plan.logical_token_count == 32


def test_source_push_plan_keeps_global_source_mask_when_local_rank_is_empty():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=1,
        cp_size=4,
        page_size=4,
        owner_rotation=0,
    )
    runtime = _runtime(spec, cp_rank=3)
    gather_plan = build_cp_prefill_kv_gather_plan(
        prefix_logical_slots=torch.empty((0,), dtype=torch.int64),
        runtime_layout=runtime,
        allocator=_FakeAllocator(
            torch.empty((0,), dtype=torch.int64), cp_rank=3, cp_size=4
        ),
    )

    plan = build_cp_prefill_kv_source_push_plan(
        gather_plan=gather_plan,
        cp_rank=3,
    )

    assert plan.source_mask == 0b0001
    assert plan.prefix.source_rows.numel() == 0
    assert plan.extend.source_rows.numel() == 0
    assert plan.logical_token_count == 1


def test_prefill_kv_uses_exactly_two_coalesced_gathers():
    spec = build_cp_prefill_split_spec(
        extend_start=4,
        extend_len=8,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    runtime = _runtime(spec, cp_rank=0)
    prefix_slots = torch.tensor([11, 12, 21, 22])
    prefix_owners = torch.tensor([1, 0, 1, 0])
    plan = build_cp_prefill_kv_gather_plan(
        prefix_logical_slots=prefix_slots,
        runtime_layout=runtime,
        allocator=_FakeAllocator(prefix_owners, cp_rank=0, cp_size=2),
    )

    prefix_k_logical = torch.arange(4, dtype=torch.float32).view(4, 1, 1)
    prefix_v_logical = prefix_k_logical + 100
    extend_k_logical = torch.arange(8, dtype=torch.float32).view(8, 1, 1) + 10
    extend_v_logical = extend_k_logical + 100
    prefix_k_packed = prefix_k_logical.index_select(
        0, plan.prefix.rank_packed_to_logical
    )
    prefix_v_packed = prefix_v_logical.index_select(
        0, plan.prefix.rank_packed_to_logical
    )
    extend_k_packed = extend_k_logical.index_select(
        0, plan.extend.rank_packed_to_logical
    )
    extend_v_packed = extend_v_logical.index_select(
        0, plan.extend.rank_packed_to_logical
    )
    group = _FakeCPGroup(
        [(prefix_k_packed, prefix_v_packed), (extend_k_packed, extend_v_packed)]
    )
    local_prefix_k = prefix_k_logical[[1, 3]]
    local_prefix_v = prefix_v_logical[[1, 3]]
    local_extend_k = extend_k_logical[:4]
    local_extend_v = extend_v_logical[:4]

    full_k, full_v = gather_cp_prefill_kv(
        plan=plan,
        local_prefix_k=local_prefix_k,
        local_prefix_v=local_prefix_v,
        local_extend_k=local_extend_k,
        local_extend_v=local_extend_v,
        cp_group=group,
        destination_ranks=(0, 1),
    )

    assert len(group.calls) == 2
    assert group.calls[0][0][0] is local_prefix_k
    assert group.calls[0][0][1] is local_prefix_v
    assert group.calls[0][1] == list(plan.prefix.sizes)
    assert group.calls[1][0][0] is local_extend_k
    assert group.calls[1][0][1] is local_extend_v
    assert group.calls[1][1] == list(plan.extend.sizes)
    assert torch.equal(full_k, torch.cat((prefix_k_logical, extend_k_logical)))
    assert torch.equal(full_v, torch.cat((prefix_v_logical, extend_v_logical)))


def _empty_prefix_plan(spec, cp_rank, runtime):
    return build_cp_prefill_kv_gather_plan(
        prefix_logical_slots=torch.empty((0,), dtype=torch.int64),
        runtime_layout=runtime,
        allocator=_FakeAllocator(
            torch.empty((0,), dtype=torch.int64),
            cp_rank=cp_rank,
            cp_size=len(spec.per_rank_tokens),
        ),
    )


def test_swa_compact_plan_matches_32k_cp4_zigzag_payload():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=32768,
        cp_size=4,
        page_size=16,
        owner_rotation=0,
    )
    compact_rows = []
    remote_rows = 0

    for cp_rank in range(4):
        runtime = _runtime(spec, cp_rank)
        gather_plan = _empty_prefix_plan(spec, cp_rank, runtime)

        swa_plan = build_cp_prefill_swa_gather_plan(
            plan=gather_plan,
            runtime_layout=runtime,
            window_left=512,
        )

        compact_rows.append(swa_plan.compact_token_count)
        remote_rows += sum(swa_plan.extend.recv_sizes) - swa_plan.extend.recv_sizes[
            cp_rank
        ]
        assert swa_plan.prefix.send_sizes == (0, 0, 0, 0)
        assert swa_plan.prefix.recv_sizes == (0, 0, 0, 0)
        assert swa_plan.block_k_lengths in (
            (4096, 4608),
            (4608, 4608),
        )
        assert swa_plan.block_cu_seqlens_k.tolist() == [
            [0, length] for length in swa_plan.block_k_lengths
        ]

    assert compact_rows == [8704, 9216, 9216, 9216]
    assert remote_rows == 3072


def test_swa_compact_plan_contracts_to_last_q_window():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=32768,
        cp_size=4,
        page_size=16,
        owner_rotation=2,
    )
    last_owner = spec.blocks[-1].owner_rank

    for cp_rank in range(4):
        full_runtime = _runtime(spec, cp_rank)
        runtime = contract_cp_prefill_runtime_to_last_q(full_runtime)
        gather_plan = _empty_prefix_plan(spec, cp_rank, full_runtime)

        swa_plan = build_cp_prefill_swa_gather_plan(
            plan=gather_plan,
            runtime_layout=runtime,
            window_left=512,
        )

        if cp_rank == last_owner:
            assert swa_plan.compact_token_count == 513
            assert swa_plan.block_k_lengths == (513,)
            assert swa_plan.block_cu_seqlens_k.tolist() == [[0, 513]]
            assert sum(swa_plan.extend.recv_sizes) == 513
            assert swa_plan.extend.recv_sizes[cp_rank] == 513
        else:
            assert swa_plan.compact_token_count == 0
            assert swa_plan.block_k_lengths == ()
            assert swa_plan.block_cu_seqlens_k.shape == (0, 2)
            assert sum(swa_plan.extend.recv_sizes) == 0


def _local_segment_rows(logical_rows, segment, cp_rank):
    offset = sum(segment.sizes[:cp_rank])
    local_logical = segment.rank_packed_to_logical.narrow(
        0, offset, segment.sizes[cp_rank]
    )
    return logical_rows.index_select(0, local_logical)


def _compact_logical_rows(runtime, window_left):
    return torch.cat(
        [
            torch.arange(
                max(0, block.logical_start - window_left),
                block.visible_kv_end,
                dtype=torch.float32,
            ).view(-1, 1, 1)
            for block in runtime.q_blocks
        ],
        dim=0,
    )


def test_swa_compact_gather_cold_prefill_skips_prefix_exchange():
    spec = build_cp_prefill_split_spec(
        extend_start=0,
        extend_len=16,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    cp_rank = 0
    runtime = _runtime(spec, cp_rank)
    gather_plan = _empty_prefix_plan(spec, cp_rank, runtime)
    swa_plan = build_cp_prefill_swa_gather_plan(
        plan=gather_plan,
        runtime_layout=runtime,
        window_left=2,
    )
    extend_k_logical = torch.arange(16, dtype=torch.float32).view(16, 1, 1)
    extend_v_logical = extend_k_logical + 100
    local_extend_k = _local_segment_rows(
        extend_k_logical, gather_plan.extend, cp_rank
    )
    local_extend_v = _local_segment_rows(
        extend_v_logical, gather_plan.extend, cp_rank
    )
    expected_k = _compact_logical_rows(runtime, window_left=2)
    expected_v = expected_k + 100
    group = _FakeAllToAllVGroup(
        cp_rank,
        [
            (
                expected_k.index_select(
                    0, swa_plan.extend.recv_packed_to_compact
                ),
                expected_v.index_select(
                    0, swa_plan.extend.recv_packed_to_compact
                ),
            )
        ],
    )

    compact_k, compact_v = gather_cp_prefill_swa_kv(
        plan=swa_plan,
        packed_prefix_k=torch.empty((0, 1, 1)),
        packed_prefix_v=torch.empty((0, 1, 1)),
        packed_extend_k=local_extend_k.index_select(
            0, swa_plan.extend.local_send_indices
        ),
        packed_extend_v=local_extend_v.index_select(
            0, swa_plan.extend.local_send_indices
        ),
        cp_group=group,
    )

    assert len(group.calls) == 1
    assert group.calls[0][1] == swa_plan.extend.send_sizes
    assert group.calls[0][2] == swa_plan.extend.recv_sizes
    assert torch.equal(compact_k, expected_k)
    assert torch.equal(compact_v, expected_v)


def test_swa_compact_gather_restores_fragmented_prefix_by_logical_row():
    spec = build_cp_prefill_split_spec(
        extend_start=8,
        extend_len=8,
        cp_size=2,
        page_size=4,
        owner_rotation=0,
    )
    cp_rank = 0
    runtime = _runtime(spec, cp_rank)
    prefix_owners = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])
    gather_plan = build_cp_prefill_kv_gather_plan(
        prefix_logical_slots=torch.arange(8) + 10,
        runtime_layout=runtime,
        allocator=_FakeAllocator(prefix_owners, cp_rank=cp_rank, cp_size=2),
    )
    swa_plan = build_cp_prefill_swa_gather_plan(
        plan=gather_plan,
        runtime_layout=runtime,
        window_left=4,
    )
    prefix_k_logical = torch.arange(8, dtype=torch.float32).view(8, 1, 1)
    prefix_v_logical = prefix_k_logical + 100
    extend_k_logical = torch.arange(8, 16, dtype=torch.float32).view(8, 1, 1)
    extend_v_logical = extend_k_logical + 100
    expected_k = _compact_logical_rows(runtime, window_left=4)
    expected_v = expected_k + 100
    group = _FakeAllToAllVGroup(
        cp_rank,
        [
            (
                expected_k.index_select(
                    0, swa_plan.prefix.recv_packed_to_compact
                ),
                expected_v.index_select(
                    0, swa_plan.prefix.recv_packed_to_compact
                ),
            ),
            (
                expected_k.index_select(
                    0, swa_plan.extend.recv_packed_to_compact
                ),
                expected_v.index_select(
                    0, swa_plan.extend.recv_packed_to_compact
                ),
            ),
        ],
    )

    compact_k, compact_v = gather_cp_prefill_swa_kv(
        plan=swa_plan,
        packed_prefix_k=_local_segment_rows(
            prefix_k_logical, gather_plan.prefix, cp_rank
        ).index_select(0, swa_plan.prefix.local_send_indices),
        packed_prefix_v=_local_segment_rows(
            prefix_v_logical, gather_plan.prefix, cp_rank
        ).index_select(0, swa_plan.prefix.local_send_indices),
        packed_extend_k=_local_segment_rows(
            extend_k_logical, gather_plan.extend, cp_rank
        ).index_select(0, swa_plan.extend.local_send_indices),
        packed_extend_v=_local_segment_rows(
            extend_v_logical, gather_plan.extend, cp_rank
        ).index_select(0, swa_plan.extend.local_send_indices),
        cp_group=group,
    )

    assert len(group.calls) == 2
    assert torch.equal(compact_k, expected_k)
    assert torch.equal(compact_v, expected_v)


def _coordinator(world_size, rank, pynccl_comm):
    return SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        pynccl_comm=pynccl_comm,
        use_symmetric_memory=lambda *_args, **_kwargs: nullcontext(),
    )


def test_all_gatherv_world_size_one_does_not_require_pynccl():
    coordinator = _coordinator(world_size=1, rank=0, pynccl_comm=None)
    tensor = torch.arange(3)

    outputs = GroupCoordinator.all_gatherv(coordinator, [tensor], sizes=[3])

    assert len(outputs) == 1
    assert outputs[0] is tensor


def test_all_gatherv_all_zero_returns_empty_without_nccl_group():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=4, rank=2, pynccl_comm=pynccl)
    k = torch.empty((0, 2, 3))
    v = torch.empty((0, 2, 3))

    outputs = GroupCoordinator.all_gatherv(
        coordinator, [k, v], sizes=[0, 0, 0, 0]
    )

    assert [tuple(output.shape) for output in outputs] == [(0, 2, 3), (0, 2, 3)]
    assert pynccl.group_start_count == 0
    assert pynccl.group_end_count == 0
    assert pynccl.gather_calls == []


def test_all_gatherv_accepts_unequal_sizes_with_empty_local_rank():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=3, rank=1, pynccl_comm=pynccl)
    tensor = torch.empty((0, 2))

    outputs = GroupCoordinator.all_gatherv(
        coordinator, [tensor], sizes=[2, 0, 1]
    )

    assert tuple(outputs[0].shape) == (3, 2)
    assert pynccl.group_start_count == 1
    assert pynccl.group_end_count == 1
    assert len(pynccl.gather_calls) == 1
    assert pynccl.gather_calls[0][2] == [2, 0, 1]
    assert pynccl.disabled


def test_gatherv_to_ranks_skips_rank_with_no_input_or_output_dependency():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=4, rank=3, pynccl_comm=pynccl)

    outputs = GroupCoordinator.gatherv_to_ranks(
        coordinator,
        [torch.empty((0, 2)), torch.empty((0, 2))],
        sizes=[2, 0, 0, 0],
        dst_ranks=[0],
    )

    assert outputs is None
    assert pynccl.group_start_count == 0
    assert pynccl.group_end_count == 0
    assert pynccl.send_calls == []
    assert pynccl.recv_calls == []


def test_gatherv_to_ranks_source_only_rank_sends_coalesced_tensors():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=4, rank=2, pynccl_comm=pynccl)
    k = torch.tensor([[20.0, 21.0]])
    v = k + 100

    outputs = GroupCoordinator.gatherv_to_ranks(
        coordinator,
        [k, v],
        sizes=[2, 0, 1, 0],
        dst_ranks=[0, 1],
    )

    assert outputs is None
    assert pynccl.group_start_count == 1
    assert pynccl.group_end_count == 1
    assert [(tensor, dst) for tensor, dst in pynccl.send_calls] == [
        (k, 0),
        (k, 1),
        (v, 0),
        (v, 1),
    ]
    assert pynccl.recv_calls == []
    assert pynccl.disabled


def test_gatherv_to_ranks_destination_receives_only_nonempty_sources():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=4, rank=1, pynccl_comm=pynccl)

    outputs = GroupCoordinator.gatherv_to_ranks(
        coordinator,
        [torch.empty((0, 2)), torch.empty((0, 2))],
        sizes=[2, 0, 1, 0],
        dst_ranks=[0, 1],
    )

    assert [tuple(output.shape) for output in outputs] == [(3, 2), (3, 2)]
    assert [src for _tensor, src in pynccl.recv_calls] == [0, 2, 0, 2]
    assert torch.equal(outputs[0][:2], torch.zeros((2, 2)))
    assert torch.equal(outputs[0][2:], torch.full((1, 2), 2.0))
    assert pynccl.send_calls == []
    assert pynccl.group_start_count == 1
    assert pynccl.group_end_count == 1
    assert pynccl.disabled


def test_gatherv_to_ranks_single_local_consumer_uses_no_nccl():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=4, rank=0, pynccl_comm=pynccl)
    k = torch.tensor([[1.0, 2.0]])
    v = k + 100

    outputs = GroupCoordinator.gatherv_to_ranks(
        coordinator,
        [k, v],
        sizes=[1, 0, 0, 0],
        dst_ranks=[0],
    )

    assert torch.equal(outputs[0], k)
    assert torch.equal(outputs[1], v)
    assert pynccl.group_start_count == 0
    assert pynccl.group_end_count == 0
    assert pynccl.send_calls == []
    assert pynccl.recv_calls == []


def test_all_to_allv_coalesces_asymmetric_kv_exchange():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=4, rank=1, pynccl_comm=pynccl)
    k = torch.arange(6, dtype=torch.float32).view(3, 2)
    v = k + 100

    outputs = GroupCoordinator.all_to_allv(
        coordinator,
        [k, v],
        send_sizes=[2, 0, 1, 0],
        recv_sizes=[1, 0, 2, 0],
    )

    assert [tuple(output.shape) for output in outputs] == [(3, 2), (3, 2)]
    assert [src for _tensor, src in pynccl.recv_calls] == [0, 2, 0, 2]
    assert [(tensor.shape[0], dst) for tensor, dst in pynccl.send_calls] == [
        (2, 0),
        (1, 2),
        (2, 0),
        (1, 2),
    ]
    assert torch.equal(outputs[0][:1], torch.zeros((1, 2)))
    assert torch.equal(outputs[0][1:], torch.full((2, 2), 2.0))
    assert pynccl.group_start_count == 1
    assert pynccl.group_end_count == 1
    assert pynccl.disabled


def test_all_to_allv_self_copy_uses_no_nccl():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=3, rank=1, pynccl_comm=pynccl)
    k = torch.arange(4, dtype=torch.float32).view(2, 2)
    v = k + 100

    outputs = GroupCoordinator.all_to_allv(
        coordinator,
        [k, v],
        send_sizes=[0, 2, 0],
        recv_sizes=[0, 2, 0],
    )

    assert torch.equal(outputs[0], k)
    assert torch.equal(outputs[1], v)
    assert pynccl.group_start_count == 0
    assert pynccl.group_end_count == 0
    assert pynccl.send_calls == []
    assert pynccl.recv_calls == []


def test_all_to_allv_all_zero_uses_no_nccl():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=3, rank=2, pynccl_comm=pynccl)
    inputs = [torch.empty((0, 2)), torch.empty((0, 2))]

    outputs = GroupCoordinator.all_to_allv(
        coordinator,
        inputs,
        send_sizes=[0, 0, 0],
        recv_sizes=[0, 0, 0],
    )

    assert [tuple(output.shape) for output in outputs] == [(0, 2), (0, 2)]
    assert pynccl.group_start_count == 0
    assert pynccl.group_end_count == 0


@pytest.mark.parametrize(
    ("send_sizes", "recv_sizes", "message"),
    [
        ([1, 0], [1, 0, 0], "send_sizes"),
        ([1, -1, 0], [1, 0, 0], "send_sizes"),
        ([1, 0, 0], [1, 0], "recv_sizes"),
        ([1, 0, 0], [1, -1, 0], "recv_sizes"),
    ],
)
def test_all_to_allv_rejects_invalid_size_vectors(
    send_sizes, recv_sizes, message
):
    coordinator = _coordinator(world_size=3, rank=0, pynccl_comm=_FakePyNccl())

    with pytest.raises(ValueError, match=message):
        GroupCoordinator.all_to_allv(
            coordinator,
            [torch.empty((1, 2))],
            send_sizes=send_sizes,
            recv_sizes=recv_sizes,
        )


def test_pynccl_unequal_all_gather_skips_zero_count_roots():
    communicator = PyNcclCommunicator.__new__(PyNcclCommunicator)
    communicator.disabled = False
    communicator.device = torch.device("cpu")
    communicator.nccl = _FakeNcclLibrary()
    communicator.comm = object()
    communicator._resolve_stream = lambda: SimpleNamespace(cuda_stream=0)

    communicator.all_gather(
        torch.empty((3, 2), dtype=torch.float32),
        torch.empty((0, 2), dtype=torch.float32),
        sizes=[2, 0, 1],
    )

    assert [root for _count, root in communicator.nccl.broadcasts] == [0, 2]
    assert [count for count, _root in communicator.nccl.broadcasts] == [4, 2]


def test_reduce_scatterv_world_size_one_does_not_require_pynccl():
    coordinator = _coordinator(world_size=1, rank=0, pynccl_comm=None)
    tensor = torch.arange(6).view(3, 2)

    output = GroupCoordinator.reduce_scatterv(coordinator, tensor, sizes=[3])

    assert output is tensor


def test_reduce_scatterv_all_zero_returns_empty_without_nccl():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=4, rank=2, pynccl_comm=pynccl)
    tensor = torch.empty((0, 2))

    output = GroupCoordinator.reduce_scatterv(
        coordinator, tensor, sizes=[0, 0, 0, 0]
    )

    assert output.shape == (0, 2)
    assert pynccl.reduce_scatter_calls == []


def test_reduce_scatterv_equal_sizes_uses_native_collective():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=4, rank=2, pynccl_comm=pynccl)
    tensor = torch.arange(16, dtype=torch.float32).view(8, 2)

    output = GroupCoordinator.reduce_scatterv(
        coordinator, tensor, sizes=[2, 2, 2, 2]
    )

    assert output.shape == (2, 2)
    assert len(pynccl.reduce_scatter_calls) == 1
    assert pynccl.reduce_scatter_calls[0][2] is None
    assert pynccl.disabled


def test_reduce_scatterv_accepts_zero_sized_destination():
    pynccl = _FakePyNccl()
    coordinator = _coordinator(world_size=3, rank=1, pynccl_comm=pynccl)
    tensor = torch.arange(6, dtype=torch.float32).view(3, 2)

    output = GroupCoordinator.reduce_scatterv(
        coordinator, tensor, sizes=[2, 0, 1]
    )

    assert output.shape == (0, 2)
    assert len(pynccl.reduce_scatter_calls) == 1
    assert pynccl.reduce_scatter_calls[0][2] == [2, 0, 1]
    assert pynccl.disabled


def test_pynccl_unequal_reduce_scatter_skips_zero_count_roots():
    communicator = PyNcclCommunicator.__new__(PyNcclCommunicator)
    communicator.disabled = False
    communicator.device = torch.device("cpu")
    communicator.nccl = _FakeNcclLibrary()
    communicator.comm = object()
    communicator._resolve_stream = lambda: SimpleNamespace(cuda_stream=0)

    communicator.reduce_scatter(
        torch.empty((0, 2), dtype=torch.float32),
        torch.empty((3, 2), dtype=torch.float32),
        sizes=[2, 0, 1],
    )

    assert [root for _count, root in communicator.nccl.reduces] == [0, 2]
    assert [count for count, _root in communicator.nccl.reduces] == [4, 2]


def test_attention_tp_group_enables_pynccl_for_context_parallel():
    assert should_enable_attn_tp_pynccl(
        attn_cp_size=4,
        sync_token_ids=False,
        enable_symm_mem=False,
    )
    assert not should_enable_attn_tp_pynccl(
        attn_cp_size=1,
        sync_token_ids=False,
        enable_symm_mem=False,
    )
    assert should_enable_attn_tp_pynccl(
        attn_cp_size=1,
        sync_token_ids=False,
        enable_symm_mem=False,
        enable_token_owner=True,
    )
