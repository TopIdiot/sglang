from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def _load_candidate_module():
    try:
        return importlib.import_module("sglang.jit_kernel.attntp_fused_norm.ipc")
    except ModuleNotFoundError as error:
        pytest.fail(f"production AttnTP kernel module is missing: {error}")


def test_internal_precision_contract() -> None:
    module = _load_candidate_module()
    default = module.AttnTPNormSpec(
        attn_tp_size=2,
        hidden_size=2048,
        output_mode=module.OutputMode.TOKEN_SCATTERED,
    )

    assert default.internal_precision is module.NormInternalPrecision.REFERENCE_BF16

    explicit = module.AttnTPNormSpec(
        attn_tp_size=2,
        hidden_size=2048,
        output_mode=module.OutputMode.TOKEN_SCATTERED,
        internal_precision=module.NormInternalPrecision.FULL_FP32,
    )
    assert explicit.internal_precision is module.NormInternalPrecision.FULL_FP32

    with pytest.raises(ValueError, match="internal_precision"):
        module.AttnTPNormSpec(
            attn_tp_size=2,
            hidden_size=2048,
            output_mode=module.OutputMode.TOKEN_SCATTERED,
            internal_precision="full_fp32",
        )


def test_all_jit_factories_specialize_internal_precision() -> None:
    module = _load_candidate_module()
    factory_names = (
        "_jit_prefill_attntp_fused_ipc_norm_module",
        "_jit_local_attntp_fused_norm_module",
        "_jit_prefill_attntp_source_push_norm_module",
        "_jit_prefill_attntp_owner_pull_norm_module",
        "_jit_decode_attntp_source_push_norm_module",
        "_jit_decode_attntp_owner_pull_norm_module",
        "_jit_decode_attntp_local_norm_module",
    )

    for factory_name in factory_names:
        parameters = inspect.signature(getattr(module, factory_name)).parameters
        assert "internal_precision" in parameters, factory_name


def test_legacy_prefill_factory_rejects_full_fp32() -> None:
    module = _load_candidate_module()

    with pytest.raises(NotImplementedError, match="legacy.*reference_bf16"):
        module._jit_prefill_attntp_fused_ipc_norm_module(
            1,
            "peer",
            4,
            module.NormInternalPrecision.FULL_FP32,
        )


def test_prefill_out_api_has_only_consumed_outputs() -> None:
    module = _load_candidate_module()
    parameters = list(
        inspect.signature(module.fused_prefill_attntp_norm_out).parameters
    )

    assert parameters[:9] == [
        "custom_ar",
        "partial",
        "residual",
        "o_norm_weight",
        "post_norm_weight",
        "output",
        "residual_out",
        "o_norm_eps",
        "post_norm_eps",
    ]
    assert "fp32_output" not in parameters
    assert "internal_precision" in parameters
    assert (
        inspect.signature(module.fused_prefill_attntp_norm_out)
        .parameters["internal_precision"]
        .default
        is module.NormInternalPrecision.REFERENCE_BF16
    )


def test_allocating_api_returns_two_outputs() -> None:
    module = _load_candidate_module()
    annotation = inspect.signature(module.fused_prefill_attntp_norm).return_annotation

    assert annotation == "tuple[torch.Tensor, torch.Tensor]"


def test_benchmark_package_is_not_imported_by_production_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    forbidden_import = "benchmark.kernels.attntp_fused_ipc_norm_v2"
    offenders = []
    for path in (repo_root / "python" / "sglang" / "srt").rglob("*.py"):
        if forbidden_import in path.read_text():
            offenders.append(path.relative_to(repo_root).as_posix())

    assert offenders == []


def test_prefill_runner_exposes_capture_context() -> None:
    module = _load_candidate_module()

    parameters = list(
        inspect.signature(module.PrefillAttnTPFusedIPCNormRunner.capture).parameters
    )

    assert parameters == ["self"]


def test_prefill_runner_exposes_capacity_query() -> None:
    module = _load_candidate_module()

    parameters = list(
        inspect.signature(
            module.PrefillAttnTPFusedIPCNormRunner.output_capacity_for
        ).parameters
    )

    assert parameters == ["self", "capacity"]


def test_ipc_initialization_waits_for_device_before_group(monkeypatch) -> None:
    module = _load_candidate_module()
    calls = []
    group = object()
    device = torch.device("cuda:0")

    monkeypatch.setattr(
        module.torch.cuda,
        "synchronize",
        lambda selected_device: calls.append(("device", selected_device)),
    )
    monkeypatch.setattr(
        module.dist,
        "barrier",
        lambda *, group: calls.append(("group", group)),
    )

    module._synchronize_ipc_communicator_initialization(
        group=group,
        device=device,
    )

    assert calls == [("device", device), ("group", group)]


@pytest.mark.parametrize("runner_kind", ["prefill", "decode"])
def test_ipc_runners_synchronize_after_communicator_init(
    monkeypatch,
    runner_kind: str,
) -> None:
    module = _load_candidate_module()
    custom_ar_module = importlib.import_module(
        "sglang.srt.distributed.device_communicators.custom_all_reduce_v2"
    )
    calls = []
    group = object()
    device = torch.device("cuda:0")

    class _FakeCustomAllReduceV2:
        disabled = False
        obj = object()

        def __init__(self, *args, **kwargs) -> None:
            calls.append(("communicator", args, kwargs))

    monkeypatch.setattr(
        custom_ar_module,
        "CustomAllReduceV2",
        _FakeCustomAllReduceV2,
    )
    monkeypatch.setattr(
        module.dist,
        "get_world_size",
        lambda *, group: 2,
    )
    monkeypatch.setattr(module.dist, "get_rank", lambda *, group: 0)
    monkeypatch.setattr(
        module,
        "_synchronize_ipc_communicator_initialization",
        lambda *, group, device: calls.append(("synchronize", group, device)),
    )
    spec = module.AttnTPNormSpec(
        attn_tp_size=2,
        hidden_size=2048,
        output_mode=module.OutputMode.TOKEN_SCATTERED,
    )

    if runner_kind == "prefill":
        module.PrefillAttnTPFusedIPCNormRunner(
            group=group,
            device=device,
            spec=spec,
            capacity=17,
            algorithm=module.PrefillCommunicationAlgorithm.SOURCE_PUSH,
        )
    else:
        module.DecodeAttnTPFusedIPCNormRunner(
            group=group,
            device=device,
            spec=spec,
            max_rows=17,
            algorithm=module.DecodeCommunicationAlgorithm.SOURCE_PUSH,
        )

    assert [call[0] for call in calls] == ["communicator", "synchronize"]
    assert calls[1] == ("synchronize", group, device)


def test_prefill_runner_capacity_query_uses_maximum() -> None:
    module = _load_candidate_module()
    runner = object.__new__(module.PrefillAttnTPFusedIPCNormRunner)
    runner.capacity = 257
    runner.spec = module.AttnTPNormSpec(
        attn_tp_size=2,
        hidden_size=2048,
        output_mode=module.OutputMode.TOKEN_SCATTERED,
    )

    assert runner.output_capacity_for(17) == 9
    assert runner.output_capacity_for(257) == 129
    with pytest.raises(ValueError, match="exceeds runner maximum"):
        runner.output_capacity_for(258)


def test_decode_runner_has_separate_fixed_row_api() -> None:
    module = _load_candidate_module()

    init_parameters = list(
        inspect.signature(module.DecodeAttnTPFusedIPCNormRunner.__init__).parameters
    )
    run_parameters = list(
        inspect.signature(module.DecodeAttnTPFusedIPCNormRunner.run_out).parameters
    )

    assert init_parameters == [
        "self",
        "group",
        "device",
        "spec",
        "max_rows",
        "algorithm",
        "block_size",
        "signal_backoff",
        "blocks_per_sm",
    ]
    assert run_parameters == [
        "self",
        "partial",
        "residual",
        "o_norm_weight",
        "post_norm_weight",
        "output",
        "residual_out",
        "o_norm_eps",
        "post_norm_eps",
    ]
    assert list(
        inspect.signature(module.DecodeAttnTPFusedIPCNormRunner.capture).parameters
    ) == ["self"]


def test_decode_communication_algorithm_is_independent_from_prefill() -> None:
    module = _load_candidate_module()

    assert [algorithm.value for algorithm in module.DecodeCommunicationAlgorithm] == [
        "source_push",
        "owner_pull",
    ]
    assert module.DecodeCommunicationAlgorithm is not (
        module.PrefillCommunicationAlgorithm
    )


@pytest.mark.parametrize("attn_tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "output_mode_name",
    ["REPLICATED", "SINGLE_CONTRIBUTOR", "TOKEN_SCATTERED"],
)
def test_decode_output_capacity_contract(
    attn_tp_size: int,
    output_mode_name: str,
) -> None:
    module = _load_candidate_module()
    rows = 17
    output_mode = getattr(module.OutputMode, output_mode_name)
    spec = module.AttnTPNormSpec(
        attn_tp_size=attn_tp_size,
        hidden_size=2048,
        output_mode=output_mode,
    )

    expected = (
        (rows + attn_tp_size - 1) // attn_tp_size
        if output_mode is module.OutputMode.TOKEN_SCATTERED
        else rows
    )

    assert module.decode_output_capacity(rows, spec) == expected


@pytest.mark.parametrize("attn_tp_size", [2, 4, 8])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize(
    "output_mode_name",
    ["REPLICATED", "SINGLE_CONTRIBUTOR", "TOKEN_SCATTERED"],
)
def test_decode_workspace_contract(
    attn_tp_size: int,
    hidden_size: int,
    output_mode_name: str,
) -> None:
    module = _load_candidate_module()
    max_rows = 256
    output_mode = getattr(module.OutputMode, output_mode_name)
    spec = module.AttnTPNormSpec(
        attn_tp_size=attn_tp_size,
        hidden_size=hidden_size,
        output_mode=output_mode,
    )
    slot_rows = (
        (max_rows + attn_tp_size - 1) // attn_tp_size
        if output_mode is module.OutputMode.TOKEN_SCATTERED
        else max_rows
    )
    payload_bytes = ((slot_rows * hidden_size * 2 + 127) // 128) * 128
    signal_bytes = ((slot_rows * 4 + 127) // 128) * 128

    assert (
        module.required_decode_source_push_buffer_bytes(max_rows, spec)
        == payload_bytes + signal_bytes
    )
    assert (
        module.required_decode_owner_pull_buffer_bytes(max_rows, spec)
        == ((max_rows * hidden_size * 2 + 127) // 128) * 128
    )


def test_decode_runner_row_query_enforces_supported_range() -> None:
    module = _load_candidate_module()
    runner = object.__new__(module.DecodeAttnTPFusedIPCNormRunner)
    runner.max_rows = 256
    runner.spec = module.AttnTPNormSpec(
        attn_tp_size=4,
        hidden_size=2048,
        output_mode=module.OutputMode.TOKEN_SCATTERED,
    )

    assert runner.output_capacity_for(1) == 1
    assert runner.output_capacity_for(256) == 64
    for invalid_rows in (0, 257):
        with pytest.raises(ValueError, match="Decode rows"):
            runner.output_capacity_for(invalid_rows)


@pytest.mark.parametrize("attn_tp_size", [1, 2, 4, 8])
def test_balanced_ranges_cover_rows_with_rotation(attn_tp_size: int) -> None:
    module = _load_candidate_module()
    row_counts = (0, 1, attn_tp_size - 1, attn_tp_size, attn_tp_size + 1, 17)

    for total_rows in row_counts:
        for owner_start in range(attn_tp_size):
            ordered_ranks = tuple(
                (owner_start + index) % attn_tp_size for index in range(attn_tp_size)
            )
            ranges = {
                rank: module.balanced_row_range(
                    total_rows=total_rows,
                    rank=rank,
                    attn_tp_size=attn_tp_size,
                    owner_start=owner_start,
                )
                for rank in range(attn_tp_size)
            }

            cursor = 0
            counts = []
            for rank in ordered_ranks:
                offset, count = ranges[rank]
                assert offset == cursor
                assert count >= 0
                cursor += count
                counts.append(count)
            assert cursor == total_rows
            assert max(counts, default=0) - min(counts, default=0) <= 1


def test_output_rows_follow_selected_ownership_mode() -> None:
    module = _load_candidate_module()
    total_rows = 11
    attn_tp_size = 4
    owner_start = 2

    replicated = [
        module.output_rows_for_rank(
            total_rows=total_rows,
            rank=rank,
            attn_tp_size=attn_tp_size,
            owner_start=owner_start,
            output_mode=module.OutputMode.REPLICATED,
        )
        for rank in range(attn_tp_size)
    ]
    contributor = [
        module.output_rows_for_rank(
            total_rows=total_rows,
            rank=rank,
            attn_tp_size=attn_tp_size,
            owner_start=owner_start,
            output_mode=module.OutputMode.SINGLE_CONTRIBUTOR,
        )
        for rank in range(attn_tp_size)
    ]
    scattered = [
        module.output_rows_for_rank(
            total_rows=total_rows,
            rank=rank,
            attn_tp_size=attn_tp_size,
            owner_start=owner_start,
            output_mode=module.OutputMode.TOKEN_SCATTERED,
        )
        for rank in range(attn_tp_size)
    ]

    assert replicated == [total_rows] * attn_tp_size
    assert contributor == [0, 0, total_rows, 0]
    assert scattered == [3, 2, 3, 3]
    assert sum(scattered) == total_rows


@pytest.mark.parametrize("attn_tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize(
    "output_mode_name",
    ["REPLICATED", "SINGLE_CONTRIBUTOR", "TOKEN_SCATTERED"],
)
def test_kernel_spec_accepts_supported_matrix(
    attn_tp_size: int,
    hidden_size: int,
    output_mode_name: str,
) -> None:
    module = _load_candidate_module()
    spec = module.AttnTPNormSpec(
        attn_tp_size=attn_tp_size,
        hidden_size=hidden_size,
        output_mode=getattr(module.OutputMode, output_mode_name),
        dtype=torch.bfloat16,
    )

    assert hash(spec) == hash(spec)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attn_tp_size": 3}, "AttnTP size"),
        ({"hidden_size": 1024}, "hidden size"),
        ({"dtype": torch.float16}, "BF16"),
        ({"output_mode": "replicated"}, "OutputMode"),
    ],
)
def test_kernel_spec_rejects_unsupported_values(kwargs, message: str) -> None:
    module = _load_candidate_module()
    arguments = {
        "attn_tp_size": 2,
        "hidden_size": 2048,
        "output_mode": module.OutputMode.REPLICATED,
        "dtype": torch.bfloat16,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        module.AttnTPNormSpec(**arguments)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total_rows": -1}, "non-negative"),
        ({"rank": 2}, "rank"),
        ({"attn_tp_size": 3}, "AttnTP size"),
        ({"owner_start": 2}, "owner_start"),
    ],
)
def test_balanced_range_rejects_invalid_arguments(kwargs, message: str) -> None:
    module = _load_candidate_module()
    arguments = {
        "total_rows": 17,
        "rank": 0,
        "attn_tp_size": 2,
        "owner_start": 0,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        module.balanced_row_range(**arguments)


@pytest.mark.parametrize("attn_tp_size", [2, 4, 8])
@pytest.mark.parametrize("hidden_size", [2048, 4096])
@pytest.mark.parametrize(
    "output_mode_name",
    ["REPLICATED", "SINGLE_CONTRIBUTOR", "TOKEN_SCATTERED"],
)
@pytest.mark.parametrize(
    "internal_precision_name",
    ["REFERENCE_BF16", "FULL_FP32"],
)
def test_source_push_capacity_contract(
    attn_tp_size: int,
    hidden_size: int,
    output_mode_name: str,
    internal_precision_name: str,
) -> None:
    module = _load_candidate_module()
    capacity = 257
    output_mode = getattr(module.OutputMode, output_mode_name)
    spec = module.AttnTPNormSpec(
        attn_tp_size=attn_tp_size,
        hidden_size=hidden_size,
        output_mode=output_mode,
        internal_precision=getattr(
            module.NormInternalPrecision,
            internal_precision_name,
        ),
    )

    output_capacity = module.source_push_output_capacity(capacity, spec)
    expected_capacity = (
        capacity
        if output_mode
        in (
            module.OutputMode.REPLICATED,
            module.OutputMode.SINGLE_CONTRIBUTOR,
        )
        else (capacity + attn_tp_size - 1) // attn_tp_size
    )
    buffer_bytes = module.required_source_push_buffer_bytes(capacity, spec)
    owner_capacity = (
        capacity
        if output_mode is module.OutputMode.SINGLE_CONTRIBUTOR
        else (capacity + attn_tp_size - 1) // attn_tp_size
    )
    source_payload_bytes = owner_capacity * hidden_size * 2
    signal_bytes = ((owner_capacity * 4 + 127) // 128) * 128
    expected_buffer_bytes = source_payload_bytes + signal_bytes
    if output_mode is module.OutputMode.REPLICATED:
        gather_element_bytes = 4 if internal_precision_name == "FULL_FP32" else 2
        gather_payload_bytes = owner_capacity * hidden_size * gather_element_bytes
        expected_buffer_bytes += gather_payload_bytes + signal_bytes

    assert output_capacity == expected_capacity
    assert buffer_bytes % 128 == 0
    assert buffer_bytes == expected_buffer_bytes
    assert (
        module.required_owner_pull_buffer_bytes(capacity, spec)
        == ((capacity * hidden_size * 2 + 127) // 128) * 128
    )


def test_source_push_signal_layout_is_candidate_independent() -> None:
    source = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/jit_kernel/csrc/distributed/attntp_fused_norm"
        / "prefill_attntp_fused_ipc_norm.cuh"
    ).read_text()

    assert "const uint32_t arena_signal_slots = owner_capacity;" in source
    assert "static_cast<uint64_t>(arena_signal_slots) * sizeof(uint32_t)" in source


@pytest.mark.parametrize(
    ("attn_tp_size", "output_mode_name", "message"),
    [(1, "TOKEN_SCATTERED", "requires AttnTP size")],
)
def test_source_push_rejects_unimplemented_modes(
    attn_tp_size: int,
    output_mode_name: str,
    message: str,
) -> None:
    module = _load_candidate_module()
    spec = module.AttnTPNormSpec(
        attn_tp_size=attn_tp_size,
        hidden_size=2048,
        output_mode=getattr(module.OutputMode, output_mode_name),
    )

    with pytest.raises(NotImplementedError, match=message):
        module.source_push_output_capacity(17, spec)


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_source_push_rejects_invalid_capacity(capacity) -> None:
    module = _load_candidate_module()
    spec = module.AttnTPNormSpec(
        attn_tp_size=2,
        hidden_size=2048,
        output_mode=module.OutputMode.TOKEN_SCATTERED,
    )

    with pytest.raises(ValueError, match="positive integer"):
        module.source_push_output_capacity(capacity, spec)
