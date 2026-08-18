from types import SimpleNamespace

import pytest
import torch

import sglang.srt.layers.communicator as communicator
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.communicator import (
    CommunicateSimpleFn,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
    TokenOwnerLayout,
)


@pytest.fixture
def patch_scatter_mode_dependencies(monkeypatch):
    def apply(*, moe_backend_is_none: bool):
        monkeypatch.setattr(
            communicator,
            "get_moe_a2a_backend",
            lambda: SimpleNamespace(is_none=lambda: moe_backend_is_none),
        )
        monkeypatch.setattr(
            communicator,
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            lambda: False,
        )
        monkeypatch.setattr(
            communicator,
            "is_enable_moe_cp_allgather",
            lambda: False,
        )

    return apply


def build_sparse_modes(*, enable_token_owner: bool) -> LayerScatterModes:
    return LayerScatterModes.init_new(
        num_layers=3,
        layer_id=1,
        is_layer_sparse=True,
        is_previous_layer_sparse=True,
        is_next_layer_sparse=True,
        enable_token_owner=enable_token_owner,
    )


@pytest.mark.parametrize(
    (
        "enable_token_owner",
        "moe_backend_is_none",
        "expected_router",
        "expected_mlp",
        "expected_residual",
        "expected_output",
    ),
    [
        (
            False,
            True,
            ScatterMode.FULL,
            ScatterMode.FULL,
            ScatterMode.TP_ATTN_FULL,
            ScatterMode.TP_ATTN_FULL,
        ),
        (
            True,
            True,
            ScatterMode.SCATTERED,
            ScatterMode.FULL,
            ScatterMode.SCATTERED,
            ScatterMode.SCATTERED,
        ),
        (
            True,
            False,
            ScatterMode.SCATTERED,
            ScatterMode.SCATTERED,
            ScatterMode.SCATTERED,
            ScatterMode.SCATTERED,
        ),
    ],
)
def test_sparse_layer_scatter_modes(
    patch_scatter_mode_dependencies,
    enable_token_owner,
    moe_backend_is_none,
    expected_router,
    expected_mlp,
    expected_residual,
    expected_output,
):
    patch_scatter_mode_dependencies(moe_backend_is_none=moe_backend_is_none)

    modes = build_sparse_modes(enable_token_owner=enable_token_owner)

    assert modes.router_mode == expected_router
    assert modes.mlp_mode == expected_mlp
    assert modes.middle_residual_mode == expected_residual
    assert modes.layer_output_mode == expected_output


@pytest.mark.parametrize("owner_count", [2, 4, 8])
def test_token_owner_layout_balances_odd_rows(owner_count):
    layout = TokenOwnerLayout.balanced(
        valid_token_count=owner_count * 2 + 1,
        owner_count=owner_count,
        local_owner_rank=owner_count - 1,
    )

    assert layout.owner_sizes == (3, *(2 for _ in range(owner_count - 1)))
    assert layout.owner_offsets == tuple(
        sum(layout.owner_sizes[:rank]) for rank in range(owner_count)
    )
    assert layout.local_valid_rows == 2


def test_token_owner_layout_supports_empty_lane():
    layout = TokenOwnerLayout.from_owner_sizes(
        (1, 0, 0, 0),
        local_owner_rank=2,
    )

    assert layout.valid_token_count == 1
    assert layout.owner_offsets == (0, 1, 1, 1)
    assert layout.local_valid_rows == 0


def test_token_owner_layout_slices_local_rows_without_copy():
    layout = TokenOwnerLayout.from_owner_sizes((2, 1, 3), local_owner_rank=1)
    full = torch.arange(12).view(6, 2)

    local = layout.local_rows(full)

    assert local.tolist() == [[4, 5]]
    assert local.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"valid_token_count": -1, "owner_count": 2, "local_owner_rank": 0},
            "non-negative",
        ),
        ({"valid_token_count": 1, "owner_count": 0, "local_owner_rank": 0}, "positive"),
        (
            {"valid_token_count": 1, "owner_count": 2, "local_owner_rank": 2},
            "local_owner_rank",
        ),
    ],
)
def test_token_owner_layout_rejects_invalid_balanced_layout(kwargs, message):
    with pytest.raises(ValueError, match=message):
        TokenOwnerLayout.balanced(**kwargs)


def _fake_group(*, world_size, rank_in_group):
    return SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank_in_group,
        pynccl_comm=None,
        device_group=object(),
    )


@pytest.mark.parametrize("sizes", ([1, 2], [1, 1]))
def test_all_gatherv_requires_pynccl(sizes):
    group = _fake_group(world_size=2, rank_in_group=0)
    local = torch.tensor([[10.0]])

    with pytest.raises(RuntimeError, match="pynccl communicator"):
        GroupCoordinator.all_gatherv(group, local, sizes=sizes)


@pytest.mark.parametrize("sizes", ([1, 2], [2, 2]))
def test_reduce_scatterv_requires_pynccl(sizes):
    group = _fake_group(world_size=2, rank_in_group=1)
    packed = torch.arange(sum(sizes), dtype=torch.float32).view(-1, 1)

    with pytest.raises(RuntimeError, match="pynccl communicator"):
        GroupCoordinator.reduce_scatterv(group, packed, sizes=sizes)


def test_owner_rows_all_gatherv_before_attention(monkeypatch):
    layout = TokenOwnerLayout.from_owner_sizes((2, 1), local_owner_rank=1)
    local = torch.tensor([[3.0, 4.0]])

    class FakeGroup:
        def all_gatherv(self, inputs, sizes):
            assert len(inputs) == 1 and inputs[0] is local
            assert sizes == [2, 1]
            return [torch.tensor([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])]

    monkeypatch.setattr(communicator, "get_attention_tp_group", FakeGroup)

    gathered = CommunicateSimpleFn._scattered_to_tp_attn_full(
        local,
        forward_batch=SimpleNamespace(),
        context=SimpleNamespace(attn_tp_size=2),
        token_owner_layout=layout,
    )

    assert gathered.tolist() == [[1.0, 2.0], [5.0, 6.0], [3.0, 4.0]]


def test_partial_o_is_reduced_before_o_norm_and_residual_norm(monkeypatch):
    layout = TokenOwnerLayout.from_owner_sizes((2, 1), local_owner_rank=1)
    partial_o = torch.tensor([[1.0], [2.0], [3.0]])
    full_residual = torch.tensor([[10.0], [20.0], [30.0]])
    calls = []

    class FakeGroup:
        def reduce_scatterv(self, input_, sizes):
            calls.append(("reduce_scatterv", input_.clone(), tuple(sizes)))
            return torch.tensor([[7.0]])

    class ONorm:
        def __call__(self, hidden_states):
            calls.append(("o_norm", hidden_states.clone()))
            return hidden_states + 1, None

    class PostAttentionNorm:
        def __call__(self, hidden_states, residual):
            calls.append(
                ("post_attention_norm", hidden_states.clone(), residual.clone())
            )
            return hidden_states + residual, hidden_states + residual

    monkeypatch.setattr(communicator, "get_attention_tp_group", FakeGroup)

    hidden_states, residual = (
        CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual(
            partial_o,
            full_residual,
            forward_batch=SimpleNamespace(),
            layernorm=PostAttentionNorm(),
            context=SimpleNamespace(attn_tp_size=2, attn_tp_rank=1),
            residual_input_mode=ScatterMode.TP_ATTN_FULL,
            token_owner_layout=layout,
            pre_layernorm=ONorm(),
        )
    )

    assert [call[0] for call in calls] == [
        "reduce_scatterv",
        "o_norm",
        "post_attention_norm",
    ]
    assert calls[0][2] == (2, 1)
    assert calls[2][2].tolist() == [[30.0]]
    assert hidden_states.tolist() == [[38.0]]
    assert residual.tolist() == [[38.0]]


def test_replicated_scatter_reduces_partial_o_before_norms(monkeypatch):
    partial_o = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    local_residual = torch.tensor([[30.0], [40.0]])
    calls = []

    def reduce_scatter(output, input_):
        calls.append(("reduce_scatter", input_.clone()))
        output.copy_(torch.tensor([[7.0], [8.0]]))

    class ONorm:
        def __call__(self, hidden_states):
            calls.append(("o_norm", hidden_states.clone()))
            return hidden_states + 1

    class PostAttentionNorm:
        def __call__(self, hidden_states, residual):
            calls.append(("post_attention_norm", hidden_states.clone()))
            return hidden_states + residual, hidden_states + residual

    monkeypatch.setattr(
        communicator, "attn_tp_reduce_scatter_tensor", reduce_scatter
    )

    hidden_states, residual = (
        CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual(
            partial_o,
            local_residual,
            forward_batch=SimpleNamespace(),
            layernorm=PostAttentionNorm(),
            context=SimpleNamespace(attn_tp_size=2, attn_tp_rank=1),
            residual_input_mode=ScatterMode.SCATTERED,
            pre_layernorm=ONorm(),
        )
    )

    assert [call[0] for call in calls] == [
        "reduce_scatter",
        "o_norm",
        "post_attention_norm",
    ]
    assert hidden_states.tolist() == [[38.0], [49.0]]
    assert residual.tolist() == [[38.0], [49.0]]


def test_global_tp_expert_partial_reduces_directly_to_owner(monkeypatch):
    layout = TokenOwnerLayout.from_owner_sizes((2, 1, 1, 0), local_owner_rank=1)
    partial = torch.arange(8, dtype=torch.float32).view(4, 2)
    residual = torch.tensor([[20.0, 21.0]])

    class FakeGroup:
        def reduce_scatterv(self, input_, sizes):
            assert input_ is partial
            assert sizes == [2, 1, 1, 0]
            return torch.tensor([[8.0, 9.0]])

    monkeypatch.setattr(communicator, "get_tp_group", FakeGroup)

    hidden_states, output_residual = (
        CommunicateSummableTensorPairFn._reduce_scatter_to_owner(
            partial,
            residual,
            forward_batch=SimpleNamespace(),
            context=SimpleNamespace(),
            token_owner_layout=layout,
        )
    )

    assert hidden_states.tolist() == [[8.0, 9.0]]
    assert output_residual is residual


def test_global_tp_proxy_output_returns_to_original_local_owner(monkeypatch):
    global_layout = TokenOwnerLayout.from_owner_sizes(
        (3, 0, 2, 2),
        local_owner_rank=1,
    )
    local_layout = TokenOwnerLayout.from_owner_sizes(
        (2, 1),
        local_owner_rank=1,
    )
    partial = torch.arange(14, dtype=torch.float32).view(7, 2)
    residual = torch.tensor([[20.0, 21.0]])

    class GlobalGroup:
        def reduce_scatterv(self, input_, sizes):
            assert input_ is partial
            assert sizes == [3, 0, 2, 2]
            return partial.new_empty((0, 2))

    class LocalGroup:
        def reduce_scatterv(self, input_, sizes):
            assert input_.tolist() == [[0.0, 0.0]] * 3
            assert sizes == [2, 1]
            return torch.tensor([[8.0, 9.0]])

    monkeypatch.setattr(communicator, "get_tp_group", GlobalGroup)
    monkeypatch.setattr(communicator, "get_attention_tp_group", LocalGroup)

    hidden_states, output_residual = (
        CommunicateSummableTensorPairFn._reduce_scatter_to_owner(
            partial,
            residual,
            forward_batch=SimpleNamespace(),
            context=SimpleNamespace(),
            token_owner_layout=global_layout,
            local_token_owner_layout=local_layout,
        )
    )

    assert hidden_states.tolist() == [[8.0, 9.0]]
    assert output_residual is residual


def test_final_global_partial_restores_only_inside_local_attn_tp(monkeypatch):
    global_layout = TokenOwnerLayout.from_owner_sizes(
        (2, 1, 1, 0),
        local_owner_rank=1,
    )
    local_layout = TokenOwnerLayout.from_owner_sizes((2, 1), local_owner_rank=1)
    partial = torch.arange(8, dtype=torch.float32).view(4, 2)
    residual = torch.tensor([[20.0, 21.0]])

    class GlobalGroup:
        def reduce_scatterv(self, input_, sizes):
            assert input_ is partial
            assert sizes == [2, 1, 1, 0]
            return torch.tensor([[8.0, 9.0]])

    class LocalGroup:
        def all_gatherv(self, input_, sizes):
            assert input_.tolist() == [[28.0, 30.0]]
            assert sizes == [2, 1]
            return [
                torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [28.0, 30.0]]
                )
            ]

    monkeypatch.setattr(communicator, "get_tp_group", GlobalGroup)
    monkeypatch.setattr(communicator, "get_attention_tp_group", LocalGroup)

    hidden_states, output_residual = (
        CommunicateSummableTensorPairFn._reduce_scatter_to_owner_and_gather(
            partial,
            residual,
            forward_batch=SimpleNamespace(),
            context=SimpleNamespace(),
            token_owner_layout=global_layout,
            local_token_owner_layout=local_layout,
        )
    )

    assert hidden_states.shape == (3, 2)
    assert output_residual is None


def test_deepep_final_owner_rows_use_variable_local_gather(monkeypatch):
    local_layout = TokenOwnerLayout.from_owner_sizes((2, 1), local_owner_rank=1)
    hidden_states = torch.tensor([[8.0, 9.0]])
    residual = torch.tensor([[20.0, 21.0]])

    class LocalGroup:
        def all_gatherv(self, input_, sizes):
            assert input_.tolist() == [[28.0, 30.0]]
            assert sizes == [2, 1]
            return [torch.arange(6, dtype=torch.float32).view(3, 2)]

    monkeypatch.setattr(communicator, "get_attention_tp_group", LocalGroup)

    output, output_residual = CommunicateSummableTensorPairFn._gather(
        hidden_states,
        residual,
        forward_batch=SimpleNamespace(),
        context=SimpleNamespace(),
        local_token_owner_layout=local_layout,
    )

    assert output.shape == (3, 2)
    assert output_residual is None


def test_layer_communicator_routes_explicit_local_and_global_owner_layouts():
    local_layout = TokenOwnerLayout.from_owner_sizes((2, 1), local_owner_rank=1)
    global_layout = TokenOwnerLayout.from_owner_sizes(
        (2, 1, 1, 0),
        local_owner_rank=1,
    )

    communicator_instance = LayerCommunicator.__new__(LayerCommunicator)
    communicator_instance.input_layernorm = lambda hidden_states: hidden_states
    communicator_instance.post_attention_layernorm = object()
    communicator_instance.pre_mlp_norm = object()
    communicator_instance.qkv_latent_func = None
    communicator_instance.allow_reduce_scatter = True
    communicator_instance._context = SimpleNamespace()

    def prepare_attention(hidden_states, token_owner_layout, **kwargs):
        assert token_owner_layout is local_layout
        return hidden_states

    def prepare_router(token_owner_layout, pre_layernorm, **kwargs):
        assert token_owner_layout is local_layout
        assert pre_layernorm is communicator_instance.pre_mlp_norm
        return kwargs["hidden_states"], kwargs["residual"]

    def postprocess(token_owner_layout, local_token_owner_layout, **kwargs):
        assert token_owner_layout is global_layout
        assert local_token_owner_layout is local_layout
        return kwargs["hidden_states"], kwargs["residual"]

    communicator_instance._communicate_simple_fn = prepare_attention
    communicator_instance._communicate_with_all_reduce_and_layer_norm_fn = (
        prepare_router
    )
    communicator_instance._communicate_summable_tensor_pair_fn = postprocess

    forward_batch = SimpleNamespace()
    hidden_states = torch.tensor([[1.0], [2.0], [3.0]])
    hidden_states, residual = communicator_instance.prepare_attn(
        hidden_states,
        None,
        forward_batch,
        token_owner_layout=local_layout,
    )
    hidden_states, residual = communicator_instance.prepare_mlp(
        hidden_states,
        residual,
        forward_batch,
        token_owner_layout=local_layout,
    )
    output, output_residual = communicator_instance.postprocess_layer(
        hidden_states,
        residual,
        forward_batch,
        token_owner_global_layout=global_layout,
        token_owner_local_layout=local_layout,
    )

    assert output is hidden_states
    assert output_residual is residual


def test_layer_communicator_transition_gathers_hidden_and_residual_together():
    local_layout = TokenOwnerLayout.from_owner_sizes((2, 1), local_owner_rank=1)

    communicator_instance = LayerCommunicator.__new__(LayerCommunicator)
    communicator_instance.input_layernorm = (
        lambda hidden_states, residual, post_residual_addition=None: (
            hidden_states + 1,
            residual + 2,
        )
    )
    communicator_instance.qkv_latent_func = None
    communicator_instance._context = SimpleNamespace()
    calls = 0

    def gather(hidden_states, token_owner_layout, **kwargs):
        nonlocal calls
        calls += 1
        assert token_owner_layout is local_layout
        assert isinstance(hidden_states, tuple)
        hidden_states, residual = hidden_states
        assert hidden_states.tolist() == [[2.0]]
        assert residual.tolist() == [[4.0]]
        return (
            torch.tensor([[10.0], [11.0], [2.0]]),
            torch.tensor([[20.0], [21.0], [4.0]]),
        )

    communicator_instance._communicate_simple_fn = gather

    hidden_states, residual = communicator_instance.prepare_attn(
        torch.tensor([[1.0]]),
        torch.tensor([[2.0]]),
        SimpleNamespace(),
        gather_token_owner_residual=True,
        token_owner_layout=local_layout,
    )

    assert calls == 1
    assert hidden_states.tolist() == [[10.0], [11.0], [2.0]]
    assert residual.tolist() == [[20.0], [21.0], [4.0]]


def test_layer_communicator_routes_non_owner_transport_layout():
    communicator_instance = LayerCommunicator.__new__(LayerCommunicator)
    communicator_instance.allow_reduce_scatter = True
    communicator_instance._context = SimpleNamespace()
    layout = TokenOwnerLayout.from_owner_sizes((2, 1), local_owner_rank=1)

    def gather(local_token_owner_layout, **kwargs):
        assert local_token_owner_layout is layout
        return kwargs["hidden_states"], kwargs["residual"]

    communicator_instance._communicate_summable_tensor_pair_fn = gather
    hidden_states = torch.tensor([[1.0]])
    residual = torch.tensor([[2.0]])

    output, output_residual = communicator_instance.postprocess_layer(
        hidden_states,
        residual,
        SimpleNamespace(),
        transport_local_layout=layout,
    )

    assert output is hidden_states
    assert output_residual is residual


def test_layer_communicator_preserves_legacy_postprocess_signature_without_layout():
    communicator_instance = LayerCommunicator.__new__(LayerCommunicator)
    communicator_instance.allow_reduce_scatter = True
    communicator_instance._context = SimpleNamespace()

    def legacy_postprocess(
        hidden_states,
        residual,
        forward_batch,
        context,
        allow_reduce_scatter,
    ):
        assert allow_reduce_scatter
        return hidden_states, residual

    communicator_instance._communicate_summable_tensor_pair_fn = legacy_postprocess
    hidden_states = torch.tensor([[1.0]])
    residual = torch.tensor([[2.0]])

    output, output_residual = communicator_instance.postprocess_layer(
        hidden_states,
        residual,
        SimpleNamespace(),
    )

    assert output is hidden_states
    assert output_residual is residual
