import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

_FEATURE_ENV = "SGLANG_WELM_V45_80A3_FUSED_PRE_ATTN"
_ORIGINAL_FEATURE_ENV = os.environ.get(_FEATURE_ENV)
os.environ[_FEATURE_ENV] = "1"
try:
    pre_attn_v2 = importlib.import_module(
        "sglang.srt.models.welm_v45_80a3_h2048_hd256_pre_attn_v2"
    )
finally:
    if _ORIGINAL_FEATURE_ENV is None:
        os.environ.pop(_FEATURE_ENV, None)
    else:
        os.environ[_FEATURE_ENV] = _ORIGINAL_FEATURE_ENV

from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=20, suite="stage-a-test-1-gpu-small")


def test_pre_attn_v2_is_opt_in(monkeypatch):
    enabled = pre_attn_v2.welm_v45_80a3_h2048_hd256_pre_attn_v2_enabled
    monkeypatch.delenv(_FEATURE_ENV, raising=False)
    assert not enabled()
    monkeypatch.setenv(_FEATURE_ENV, "true")
    assert enabled()


def test_disabled_feature_does_not_import_mk_or_v2_module():
    env = os.environ.copy()
    env.pop(_FEATURE_ENV, None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import sglang.srt.models.welmv4; "
                "import sglang.srt.model_executor.cuda_graph_runner; "
                "loaded=set(sys.modules); "
                "assert 'sglang.srt.models."
                "welm_v45_80a3_h2048_hd256_pre_attn_v2' not in loaded; "
                "assert not any(n == 'mk' or n.startswith('mk.') for n in loaded)"
            ),
        ],
        check=True,
        env=env,
    )


def test_cuda_graph_preparation_uses_swa_cache_location():
    regular_loc = torch.tensor([1, 2], dtype=torch.int64)
    swa_loc = torch.tensor([3, 4], dtype=torch.int32)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    calls = []

    class Pool:
        @staticmethod
        def is_swa_layer(layer_idx):
            return layer_idx == 1

    class Module:
        def __init__(self, layer_idx, result):
            self.layer_idx = layer_idx
            self.result = result

        def prepare_welm_v45_80a3_h2048_hd256_pre_attn_v2_cuda_graph(
            self, rows, graph_positions, cache_loc, pool
        ):
            calls.append((self.layer_idx, rows, graph_positions, cache_loc, pool))
            return self.result

    modules = [Module(0, True), Module(1, False), nn.Linear(1, 1)]
    pool = Pool()
    runner = SimpleNamespace(
        token_to_kv_pool=pool,
        model=SimpleNamespace(modules=lambda: modules),
    )
    buffers = SimpleNamespace(
        positions=positions,
        out_cache_loc=regular_loc,
        out_cache_loc_swa=swa_loc,
    )

    prepared = pre_attn_v2.prepare_welm_v45_80a3_h2048_hd256_pre_attn_v2_cuda_graphs(
        runner, buffers, capture_bs=[1], num_tokens_per_bs=2
    )

    assert prepared == 1
    assert calls == [
        (0, 2, positions, regular_loc, pool),
        (1, 2, positions, swa_loc, pool),
    ]


def test_cuda_graph_preparation_falls_back_above_tuned_max(caplog):
    rows_seen = []

    class Module:
        layer_idx = 0

        @staticmethod
        def prepare_welm_v45_80a3_h2048_hd256_pre_attn_v2_cuda_graph(rows, *_args):
            rows_seen.append(rows)
            return rows <= pre_attn_v2._WELM_V45_80A3_H2048_HD256_PRE_ATTN_V2_MAX_ROWS

    runner = SimpleNamespace(
        token_to_kv_pool=object(),
        model=SimpleNamespace(modules=lambda: [Module()]),
    )
    buffers = SimpleNamespace(
        positions=object(), out_cache_loc=object(), out_cache_loc_swa=None
    )
    prepared = pre_attn_v2.prepare_welm_v45_80a3_h2048_hd256_pre_attn_v2_cuda_graphs(
        runner,
        buffers,
        capture_bs=[4095, 4096],
        num_tokens_per_bs=4,
    )

    assert prepared == 1
    assert rows_seen == [16380, 16384]
    assert "M<=16383" in caplog.text
    assert "M=16384" in caplog.text


def test_nextn_cuda_graph_preparation_uses_contracted_q_and_full_kv_rows():
    calls = []
    packed_kv = torch.empty((33, 512), dtype=torch.bfloat16)
    raw_kv = (packed_kv[:, :256], packed_kv[:, 256:])

    class Module:
        @staticmethod
        def _mk_nextn_mirror_key():
            return 48

        @staticmethod
        def prepare_welm_v45_80a3_h2048_hd256_nextn_pre_attn_v2_cuda_graph(
            q_rows, kv_rows, graph_raw_kv, positions, cache_loc, pool
        ):
            calls.append((q_rows, kv_rows, graph_raw_kv, positions, cache_loc, pool))
            return 1

    pool = object()
    positions = torch.arange(32, dtype=torch.int64)
    cache_loc = torch.arange(32, dtype=torch.int64)
    runner = SimpleNamespace(
        token_to_kv_pool=pool,
        model=SimpleNamespace(modules=lambda: [Module(), nn.Linear(1, 1)]),
    )
    buffers = SimpleNamespace(positions=positions, out_cache_loc=cache_loc)

    prepared = (
        pre_attn_v2.prepare_welm_v45_80a3_h2048_hd256_nextn_pre_attn_v2_cuda_graphs(
            runner,
            buffers,
            capture_bs=[1, 8],
            num_tokens_per_bs=4,
            mirror_kv_states={48: raw_kv},
        )
    )

    assert prepared == 2
    assert [(call[0], call[1]) for call in calls] == [(1, 4), (8, 32)]
    assert all(call[2] == raw_kv for call in calls)
    assert all(
        call[3] is positions and call[4] is cache_loc and call[5] is pool
        for call in calls
    )


def test_nextn_cuda_graph_preparation_uses_swa_cache_location():
    calls = []
    packed_kv = torch.empty((5, 512), dtype=torch.bfloat16)
    raw_kv = (packed_kv[:, :256], packed_kv[:, 256:])
    regular_loc = torch.arange(4, dtype=torch.int64)
    swa_loc = torch.arange(4, dtype=torch.int32)

    class Pool:
        @staticmethod
        def is_swa_layer(layer_idx):
            return layer_idx == 1

    class Module:
        def __init__(self, layer_idx):
            self.layer_idx = layer_idx

        @staticmethod
        def _mk_nextn_mirror_key():
            return 48

        def prepare_welm_v45_80a3_h2048_hd256_nextn_pre_attn_v2_cuda_graph(
            self, _q_rows, _kv_rows, _raw_kv, _positions, cache_loc, _pool
        ):
            calls.append((self.layer_idx, cache_loc))
            return 1

    pool = Pool()
    runner = SimpleNamespace(
        token_to_kv_pool=pool,
        model=SimpleNamespace(modules=lambda: [Module(0), Module(1)]),
    )
    buffers = SimpleNamespace(
        positions=torch.arange(4, dtype=torch.int64),
        out_cache_loc=regular_loc,
        out_cache_loc_swa=swa_loc,
    )

    prepared = (
        pre_attn_v2.prepare_welm_v45_80a3_h2048_hd256_nextn_pre_attn_v2_cuda_graphs(
            runner,
            buffers,
            capture_bs=[1],
            num_tokens_per_bs=4,
            mirror_kv_states={48: raw_kv},
        )
    )

    assert prepared == 2
    assert calls == [(0, regular_loc), (1, swa_loc)]


def test_backend_owned_kv_paths_stay_on_baseline():
    # ForwardMode.is_context_parallel_extend() also returns True for an ordinary
    # EXTEND batch. The mode alone must not disable V2 when no CP runtime layout
    # was materialized.
    assert not pre_attn_v2._requires_attention_backend_kv_write(
        SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    )
    assert pre_attn_v2._requires_attention_backend_kv_write(
        SimpleNamespace(attn_cp_prefill_runtime_layout=object())
    )
    assert pre_attn_v2._requires_attention_backend_kv_write(
        SimpleNamespace(attn_backend=SimpleNamespace(fa_skip_kv_cache=True))
    )
    assert not pre_attn_v2._requires_attention_backend_kv_write(SimpleNamespace())


class _Prepared:
    def __init__(self, q, mirror_outputs=()):
        self.q = q
        self.mirror_outputs = mirror_outputs
        self.plan = SimpleNamespace(name="fake_v2_plan")
        self.launch_count = 0
        self.rebound_calls = []

    def launch(self):
        self.launch_count += 1
        return self.q

    def launch_rebound(self, hidden_states, *args, **kwargs):
        self.launch_count += 1
        self.rebound_calls.append((hidden_states, args, kwargs))
        return self.q


class _IdlePrepared(_Prepared):
    def __init__(self, q, k, v):
        super().__init__(q)
        self.k = k
        self.v = v

    def launch(self):
        self.launch_count += 1
        return self.q, self.k, self.v

    def launch_rebound(self, hidden_states):
        self.launch_count += 1
        self.rebound_calls.append((hidden_states, (), {}))
        return self.q, self.k, self.v


class _FakePool:
    def __init__(self, rows):
        self.key = torch.empty((rows + 8, 1, 256), dtype=torch.bfloat16)
        self.value = torch.empty_like(self.key)

    def get_key_buffer(self, _layer_idx):
        return self.key

    def get_value_buffer(self, _layer_idx):
        return self.value


class _FakeAttention(pre_attn_v2.WeLMV45_80A3H2048HD256PreAttnV2Mixin):
    def __init__(self, kind, prepared):
        self.kind = kind
        self.prepared = prepared
        self.layer_idx = 0
        self.need_clear_kv_cache = False

    def _mk_projection_kind(self):
        return self.kind

    def _mk_mirror_source_keys(self):
        return (3,) if self.kind == "mirror_source" else ()

    def _mk_mirror_consumer_key(self):
        return 3 if self.kind == "mirror_consumer" else None

    def _mk_graph_dump_enabled(self):
        return False

    def _mk_v2_layer_contract(self, kind, rows):
        return kind == self.kind and 0 < rows <= 16383

    def _mk_cache_loc_for_rows(self, forward_batch, rows):
        return forward_batch.out_cache_loc

    @staticmethod
    def _mk_v2_io_contract(*_args):
        return True

    def _prepare_v2(self, *_args, raw_k=None, raw_v=None, **_kwargs):
        self.seen_raw_kv = (raw_k, raw_v)
        return self.prepared

    def _mk_prepare_indexed_mtp_v2_inputs(self, *_args):
        return getattr(self, "indexed_inputs", None)


def test_indexed_swa_uses_forward_batch_cache_locations_before_pool_state():
    direct_swa_loc = torch.tensor([3, 4, 5, 6], dtype=torch.int32)
    stale_pool_loc = torch.tensor([30, 40, 50, 60], dtype=torch.int32)

    class Pool:
        swa_loc = stale_pool_loc

        @staticmethod
        def is_swa_layer(_layer_idx):
            return True

    attention = _FakeAttention("indexed_mtp", prepared=None)
    batch = SimpleNamespace(
        token_to_kv_pool=Pool(),
        out_cache_loc=torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        out_cache_loc_swa=direct_swa_loc,
        welm_mtp_merge_kv_fill_draft=False,
        welm_kv_mirror_contracted=False,
    )

    select_cache_loc = (
        pre_attn_v2.WeLMV45_80A3H2048HD256PreAttnV2Mixin._mk_cache_loc_for_rows
    )
    assert select_cache_loc(attention, batch, rows=4) is direct_swa_loc


@pytest.mark.parametrize("kind", ["standard", "mirror_source", "mirror_consumer"])
def test_eager_v2_routes_update_mirror_state_and_return_shape_only_kv(
    monkeypatch, kind
):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    rows = 2
    q = torch.arange(rows * 1536, dtype=torch.bfloat16).view(rows, 1536)
    mirror_storage = torch.empty((rows, 512), dtype=torch.bfloat16)
    mirror_kv = (mirror_storage[:, :256], mirror_storage[:, 256:])
    prepared = _Prepared(
        q,
        mirror_outputs=(mirror_kv,) if kind == "mirror_source" else (),
    )
    attn = _FakeAttention(kind, prepared)
    pool = _FakePool(rows)
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        out_cache_loc=torch.arange(rows, dtype=torch.int64),
    )
    states = {3: mirror_kv} if kind == "mirror_consumer" else {}
    hidden = torch.empty((rows, 2048), dtype=torch.bfloat16)
    positions = torch.arange(rows, dtype=torch.int64)

    result = attn._try_mk_h2048_hd256_pre_attn_v2(
        positions, hidden, forward_batch, states
    )

    assert result is not None and result.kv_cache_written
    assert prepared.launch_count == 1
    assert result.q is q and result.hidden_states is hidden
    assert result.k.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
    assert result.v.untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
    if kind == "mirror_source":
        assert states[3] == mirror_kv
    elif kind == "mirror_consumer":
        assert 3 not in states
        assert attn.seen_raw_kv == mirror_kv


def test_graph_consumer_rebinds_dynamic_inputs_without_staging_copies(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    rows = 2
    q = torch.empty((rows, 1536), dtype=torch.bfloat16)
    prepared = _Prepared(q)
    attn = _FakeAttention("mirror_consumer", prepared)
    static_input = torch.full((rows, 2048), -1, dtype=torch.bfloat16)
    static_kv = torch.full((rows, 512), -2, dtype=torch.bfloat16)
    attn._mk_h2048_hd256_pre_attn_v2_graph_ops = {
        ("mirror_consumer", rows): {
            "kind": "mirror_consumer",
            "input": static_input,
            "prepared": prepared,
            "raw_kv": static_kv,
            "raw_k": static_kv[:, :256],
            "raw_v": static_kv[:, 256:],
        }
    }
    dynamic_kv = torch.randn((rows, 512), dtype=torch.bfloat16)
    states = {3: (dynamic_kv[:, :256], dynamic_kv[:, 256:])}
    hidden = torch.randn((rows, 2048), dtype=torch.bfloat16)
    batch = SimpleNamespace()

    result = attn._try_mk_h2048_hd256_pre_attn_v2(
        torch.arange(rows, dtype=torch.int64), hidden, batch, states
    )

    assert result is not None
    assert len(prepared.rebound_calls) == 1
    rebound_hidden, rebound_args, rebound_kwargs = prepared.rebound_calls[0]
    assert rebound_hidden is hidden and rebound_args == ()
    assert rebound_kwargs["raw_k"].data_ptr() == dynamic_kv[:, :256].data_ptr()
    assert rebound_kwargs["raw_v"].data_ptr() == dynamic_kv[:, 256:].data_ptr()
    torch.testing.assert_close(static_input, torch.full_like(static_input, -1))
    torch.testing.assert_close(static_kv, torch.full_like(static_kv, -2))
    assert 3 not in states


def test_idle_nextn_v2_returns_full_qkv_without_claiming_cache_write(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    rows = 2
    q = torch.empty((rows, 1536), dtype=torch.bfloat16)
    k = torch.empty((rows, 256), dtype=torch.bfloat16)
    v = torch.empty_like(k)
    prepared = _IdlePrepared(q, k, v)
    attn = _FakeAttention("idle_mtp", prepared)
    hidden = torch.empty((rows, 2048), dtype=torch.bfloat16)

    result = attn._try_mk_h2048_hd256_pre_attn_v2(
        torch.arange(rows, dtype=torch.int64), hidden, SimpleNamespace(), {}
    )

    assert result is not None
    assert result.q is q and result.k is k and result.v is v
    assert not result.kv_cache_written
    assert prepared.launch_count == 1


def test_indexed_nextn_v2_returns_gathered_hidden_and_keeps_mirror_until_last(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    rows = 4
    q = torch.empty((rows, 1536), dtype=torch.bfloat16)
    gathered_hidden = torch.randn((rows, 2048), dtype=torch.bfloat16)
    prepared = _Prepared(q)
    prepared.gathered_hidden_states = gathered_hidden
    attn = _FakeAttention("indexed_mtp", prepared)
    raw_kv = torch.randn((rows, 512), dtype=torch.bfloat16)
    attn.indexed_inputs = {
        "raw_k": raw_kv[:, :256],
        "raw_v": raw_kv[:, 256:],
        "q_indices": torch.arange(rows, dtype=torch.int64),
        "kv_indices": torch.arange(rows, dtype=torch.int64),
        "q_positions": torch.arange(rows, dtype=torch.int64),
        "kv_positions": torch.arange(rows, dtype=torch.int64),
    }
    pool = _FakePool(rows)
    batch = SimpleNamespace(
        token_to_kv_pool=pool,
        out_cache_loc=torch.arange(rows, dtype=torch.int64),
    )
    states = {3: (raw_kv[:, :256], raw_kv[:, 256:])}

    result = attn._try_mk_h2048_hd256_pre_attn_v2(
        torch.arange(rows, dtype=torch.int64),
        torch.empty((rows, 2048), dtype=torch.bfloat16),
        batch,
        states,
    )

    assert result is not None and result.kv_cache_written
    assert result.hidden_states is gathered_hidden
    assert 3 in states
    assert attn.seen_raw_kv == (
        attn.indexed_inputs["raw_k"],
        attn.indexed_inputs["raw_v"],
    )


def test_indexed_nextn_graph_rebinds_inputs_without_staging_copies(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    q_rows = 2
    kv_rows = 8
    source_kv = torch.randn((kv_rows + 1, 512), dtype=torch.bfloat16)
    q = torch.empty((q_rows, 1536), dtype=torch.bfloat16)
    gathered_hidden = torch.empty((q_rows, 2048), dtype=torch.bfloat16)
    prepared = _Prepared(q)
    prepared.gathered_hidden_states = gathered_hidden
    attn = _FakeAttention("indexed_mtp", prepared)
    attn.indexed_inputs = {
        "raw_k": source_kv[:, :256],
        "raw_v": source_kv[:, 256:],
        "q_indices": torch.arange(q_rows, dtype=torch.int64),
        "kv_indices": torch.arange(kv_rows, dtype=torch.int64),
        "q_positions": torch.arange(q_rows, dtype=torch.int64),
        "kv_positions": torch.arange(kv_rows, dtype=torch.int64),
    }
    static_input = torch.full((q_rows, 2048), -1, dtype=torch.bfloat16)
    graph_cache_loc = torch.full((kv_rows,), -2, dtype=torch.int64)
    attn._mk_h2048_hd256_pre_attn_v2_graph_ops = {
        ("indexed_mtp", q_rows): {
            "kind": "indexed_mtp",
            "input": static_input,
            "prepared": prepared,
            "raw_kv": None,
            "raw_k": attn.indexed_inputs["raw_k"],
            "raw_v": attn.indexed_inputs["raw_v"],
            "q_indices": torch.empty((q_rows,), dtype=torch.int64),
            "kv_indices": torch.empty((kv_rows,), dtype=torch.int64),
            "q_positions": torch.empty((q_rows,), dtype=torch.int64),
            "kv_positions": torch.empty((kv_rows,), dtype=torch.int64),
            "cache_loc": graph_cache_loc,
            "project_hidden": gathered_hidden,
            "project_hidden_from_kernel": True,
        }
    }
    dynamic_cache_loc = torch.arange(10, 10 + kv_rows, dtype=torch.int64)
    batch = SimpleNamespace(out_cache_loc=dynamic_cache_loc)
    hidden = torch.randn((q_rows, 2048), dtype=torch.bfloat16)

    result = attn._try_mk_h2048_hd256_pre_attn_v2(
        torch.arange(q_rows, dtype=torch.int64), hidden, batch, {3: object()}
    )

    assert result is not None and prepared.launch_count == 1
    assert len(prepared.rebound_calls) == 1
    rebound_hidden, rebound_args, rebound_kwargs = prepared.rebound_calls[0]
    assert rebound_hidden is hidden and rebound_kwargs == {}
    expected_rebound_args = (
        attn.indexed_inputs["raw_k"],
        attn.indexed_inputs["raw_v"],
        attn.indexed_inputs["q_indices"],
        attn.indexed_inputs["kv_indices"],
        attn.indexed_inputs["q_positions"],
        attn.indexed_inputs["kv_positions"],
        dynamic_cache_loc,
    )
    assert all(
        actual is expected
        for actual, expected in zip(rebound_args, expected_rebound_args)
    )
    torch.testing.assert_close(static_input, torch.full_like(static_input, -1))
    torch.testing.assert_close(graph_cache_loc, torch.full_like(graph_cache_loc, -2))


def test_nextn_index_plan_preserves_independent_q_and_mirrored_kv_indices():
    welmv4 = importlib.import_module("sglang.srt.models.welmv4")
    projection = object.__new__(welmv4.NextnMirrorQProjection)
    nn.Module.__init__(projection)
    projection.mirror_layer_idx = None
    projection.imitated_layer_idx = 3
    attn = SimpleNamespace(qkv_proj=projection)
    hidden = torch.empty((2, 2048), dtype=torch.bfloat16)
    raw_kv = torch.empty((2, 512), dtype=torch.bfloat16)

    class Mode:
        @staticmethod
        def is_decode():
            return False

        @staticmethod
        def is_draft_extend(include_v2=False):
            return include_v2

        @staticmethod
        def is_extend_without_speculative():
            return False

    identity_indices = torch.arange(16, dtype=torch.int64)
    batch = SimpleNamespace(
        forward_mode=Mode(),
        spec_info=SimpleNamespace(
            mirrored_kv_indices=torch.tensor([1, 0], dtype=torch.int64)
        ),
        enable_welm_kv_mirror_opt=False,
        welm_mtp_merge_kv_fill_draft=False,
        welm_deferred_prefill=False,
        welm_mtp_identity_indices=identity_indices,
    )
    result = welmv4.Qwen2MoeAttention._mk_prepare_indexed_mtp_v2_inputs(
        attn,
        torch.tensor([7, 8], dtype=torch.int64),
        hidden,
        batch,
        {3: (raw_kv[:, :256], raw_kv[:, 256:])},
    )

    assert result is not None
    assert result["q_indices"].untyped_storage().data_ptr() == identity_indices.data_ptr()
    torch.testing.assert_close(result["q_indices"], torch.tensor([0, 1]))
    torch.testing.assert_close(result["kv_indices"], torch.tensor([1, 0]))
    torch.testing.assert_close(result["q_positions"], torch.tensor([7, 8]))
    torch.testing.assert_close(result["kv_positions"], torch.tensor([7, 8]))


def test_v2_only_mk_symbols_are_available():
    from mk import kernels

    assert hasattr(
        kernels,
        "prepare_welm_v45_80a3_h2048_hd256_standard_qkv_optimized_v2",
    )
    assert not hasattr(kernels, "prepare_welm_v45_80a3_fused_pre_attn")
