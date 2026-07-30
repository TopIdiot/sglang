import builtins
import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

# This file exercises the opt-in implementation. Disabled-import behavior is
# verified in a clean subprocess below.
_FEATURE_ENV = "SGLANG_WELM_V45_80A3_FUSED_PRE_ATTN"
_ORIGINAL_FEATURE_ENV = os.environ.get(_FEATURE_ENV)
os.environ[_FEATURE_ENV] = "1"
try:
    welmv4 = importlib.import_module("sglang.srt.models.welmv4")
    mk_fusion = importlib.import_module(
        "sglang.srt.models.welm_v45_80a3_fused_pre_attn"
    )
finally:
    if _ORIGINAL_FEATURE_ENV is None:
        os.environ.pop(_FEATURE_ENV, None)
    else:
        os.environ[_FEATURE_ENV] = _ORIGINAL_FEATURE_ENV

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="stage-a-test-1-gpu-small")


def test_welm_mk_fused_qkv_is_opt_in(monkeypatch):
    monkeypatch.delenv(mk_fusion._WELM_V45_80A3_FUSED_PRE_ATTN_ENV, raising=False)
    assert not mk_fusion._welm_v45_80a3_fused_pre_attn_enabled()

    monkeypatch.setenv(mk_fusion._WELM_V45_80A3_FUSED_PRE_ATTN_ENV, "1")
    assert mk_fusion._welm_v45_80a3_fused_pre_attn_enabled()


def test_disabled_feature_does_not_import_mk():
    env = os.environ.copy()
    env.pop(mk_fusion._WELM_V45_80A3_FUSED_PRE_ATTN_ENV, None)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import sglang.srt.models.welmv4; "
                "import sglang.srt.model_executor.cuda_graph_runner; "
                "loaded = set(sys.modules); "
                "assert 'sglang.srt.models.welm_v45_80a3_fused_pre_attn' "
                "not in loaded; "
                "assert not any(name == 'mk' or name.startswith('mk.') "
                "for name in loaded)"
            ),
        ],
        check=True,
        env=env,
    )


class _ForwardMode:
    @staticmethod
    def is_decode():
        return True


class MHATokenToKVPool:
    def __init__(self, key_cache, value_cache):
        self.key_cache = key_cache
        self.value_cache = value_cache

    def get_key_buffer(self, _layer_idx):
        return self.key_cache

    def get_value_buffer(self, _layer_idx):
        return self.value_cache


def test_prepare_welm_v45_80a3_fused_pre_attn_cuda_graphs_uses_swa_cache_loc():
    regular_cache_loc = torch.tensor([1, 2], dtype=torch.int64)
    swa_cache_loc = torch.tensor([3, 4], dtype=torch.int32)
    positions = torch.tensor([0, 1], dtype=torch.int64)
    calls = []

    class Pool:
        @staticmethod
        def is_swa_layer(layer_idx):
            return layer_idx == 1

    class Module:
        def __init__(self, layer_idx, dense_result):
            self.layer_idx = layer_idx
            self.dense_result = dense_result

        def prepare_welm_v45_80a3_fused_pre_attn_cuda_graph(
            self, rows, graph_positions, cache_loc, pool
        ):
            calls.append(("dense", self.layer_idx, rows, cache_loc, pool))
            assert graph_positions is positions
            return self.dense_result

    modules = [Module(0, True), Module(1, False), nn.Linear(1, 1)]
    runner = SimpleNamespace(
        token_to_kv_pool=Pool(),
        model=SimpleNamespace(modules=lambda: modules),
    )
    buffers = SimpleNamespace(
        positions=positions,
        out_cache_loc=regular_cache_loc,
        out_cache_loc_swa=swa_cache_loc,
    )

    prepared = mk_fusion.prepare_welm_v45_80a3_fused_pre_attn_cuda_graphs(
        runner, buffers, capture_bs=[1], num_tokens_per_bs=2
    )

    assert prepared == 1
    assert calls == [
        ("dense", 0, 2, regular_cache_loc, runner.token_to_kv_pool),
        ("dense", 1, 2, swa_cache_loc, runner.token_to_kv_pool),
    ]


def test_prepare_welm_v45_80a3_fused_pre_attn_cuda_graphs_covers_large_m_capture_shapes():
    from sglang.srt.server_args import ServerArgs

    rows_seen = []

    class Module:
        layer_idx = 0

        @staticmethod
        def prepare_welm_v45_80a3_fused_pre_attn_cuda_graph(rows, *_args):
            rows_seen.append(rows)
            return True

    runner = SimpleNamespace(
        token_to_kv_pool=object(),
        model=SimpleNamespace(modules=lambda: [Module()]),
    )
    buffers = SimpleNamespace(
        positions=object(),
        out_cache_loc=object(),
        out_cache_loc_swa=None,
    )

    capture_args = SimpleNamespace(
        disable_cuda_graph_padding=False,
        speculative_algorithm="EAGLE",
        cuda_graph_max_bs=512,
    )
    capture_bs = ServerArgs._generate_cuda_graph_batch_sizes(capture_args)
    prepared = mk_fusion.prepare_welm_v45_80a3_fused_pre_attn_cuda_graphs(
        runner,
        buffers,
        capture_bs=capture_bs,
        num_tokens_per_bs=4,
    )

    assert max(capture_bs) == 512
    assert prepared == len(capture_bs)
    assert rows_seen == [batch_size * 4 for batch_size in capture_bs]
    assert {512, 640, 1024, 2048}.issubset(rows_seen)
    assert max(rows_seen) <= mk_fusion._WELM_MK_MAX_FUSED_ROWS


def test_prepare_welm_v45_80a3_fused_pre_attn_cuda_graphs_warns_above_mk_limit(caplog):
    runner = SimpleNamespace(
        token_to_kv_pool=object(),
        model=SimpleNamespace(modules=lambda: []),
    )
    buffers = SimpleNamespace(
        positions=object(),
        out_cache_loc=object(),
        out_cache_loc_swa=None,
    )

    mk_fusion.prepare_welm_v45_80a3_fused_pre_attn_cuda_graphs(
        runner,
        buffers,
        capture_bs=[4096, 4097],
        num_tokens_per_bs=4,
    )

    assert "1 capture shape(s)" in caplog.text
    assert "batch size 4097 (16388 rows)" in caplog.text


def test_welm_mk_fusion_preserves_backend_owned_kv_write_paths():
    context_parallel_mode = SimpleNamespace(is_context_parallel_extend=lambda: True)
    assert mk_fusion._requires_attention_backend_kv_write(
        SimpleNamespace(forward_mode=context_parallel_mode)
    )
    assert mk_fusion._requires_attention_backend_kv_write(
        SimpleNamespace(attn_cp_prefill_runtime_layout=object())
    )
    assert mk_fusion._requires_attention_backend_kv_write(
        SimpleNamespace(attn_backend=SimpleNamespace(fa_skip_kv_cache=True))
    )
    assert not mk_fusion._requires_attention_backend_kv_write(SimpleNamespace())


def test_welm_mk_loader_surfaces_attribute_error(monkeypatch):
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "mk.errors":
            raise AttributeError("mk module initialization bug")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(mk_fusion, "_WELM_MK_FUSED_QKV_FN", None)
    monkeypatch.setattr(mk_fusion, "_WELM_MK_FUSED_QKV_IMPORT_FAILED", False)
    monkeypatch.setattr(builtins, "__import__", broken_import)
    with pytest.raises(AttributeError, match="mk module initialization bug"):
        mk_fusion._load_welm_mk_fused_qkv()


def test_welm_mk_dense_cuda_graph_preserves_backend_owned_kv_write(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    handle = SimpleNamespace(launch=lambda: pytest.fail("must use backend KV write"))
    attn = SimpleNamespace(
        _mk_fused_qkv_graph_handles={
            1: {
                "input": torch.empty((1, 2048), dtype=torch.bfloat16),
                "q": torch.empty((1, 512), dtype=torch.bfloat16),
                "k": torch.empty((1, 256), dtype=torch.bfloat16),
                "v": torch.empty((1, 256), dtype=torch.bfloat16),
                "handle": handle,
            }
        }
    )
    result = welmv4.Qwen2MoeAttention._try_mk_fused_qkv_knorm_rope_kv_write(
        attn,
        torch.arange(1, dtype=torch.int64),
        torch.zeros((1, 2048), dtype=torch.bfloat16),
        SimpleNamespace(attn_backend=SimpleNamespace(fa_skip_kv_cache=True)),
    )
    assert result is None


def _make_attention(projection=None):
    attn = welmv4.Qwen2MoeAttention.__new__(welmv4.Qwen2MoeAttention)
    nn.Module.__init__(attn)
    attn.is_nextn = False
    attn._welm_v45_80a3_fused_pre_attn_model_contract = True
    attn.suffix_parallel = False
    attn.scale_seq_attn_per_suffix = False
    attn.scale_seq_factor = 1
    attn.qk_norm = False
    attn.only_k_norm = True
    attn.head_dim = 256
    attn.qk_rope_head_dim = 64
    attn.q_size = 512
    attn.kv_size = 256
    attn.layer_idx = 0
    attn.qkv_proj = projection
    attn.k_norm = SimpleNamespace(
        weight=torch.ones((256,), dtype=torch.bfloat16), eps=1e-5
    )
    attn.rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.ones((64, 64), dtype=torch.float32)
    )
    return attn


def test_fused_pre_attn_rejects_non_v45_80a3_model_contract(monkeypatch):
    class FakeProjection:
        bias = None
        weight = torch.zeros((1024, 2048), dtype=torch.bfloat16)

    attn = _make_attention(FakeProjection())
    attn._welm_v45_80a3_fused_pre_attn_model_contract = False
    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(
        mk_fusion,
        "_load_welm_mk_fused_qkv",
        lambda: pytest.fail("non-80A3 model must not load the kernel"),
    )

    result = attn._try_mk_fused_qkv_knorm_rope_kv_write(
        torch.arange(1, dtype=torch.int64),
        torch.zeros((1, 2048), dtype=torch.bfloat16),
        SimpleNamespace(),
    )
    assert result is None


@pytest.mark.parametrize("rows", [513, 640, 1024, 2048, 16384])
def test_welm_mk_graph_prepare_and_capture_cover_large_m(monkeypatch, rows):
    class FakeProjection:
        bias = None
        weight = torch.zeros((1024, 2048), dtype=torch.bfloat16)

    class Handle:
        def __init__(self):
            self.launch_count = 0

        def prepare_workspace(self):
            pytest.fail("large-M graph launch must capture workspace initialization")

        def launch(self):
            self.launch_count += 1

    attn = _make_attention(FakeProjection())
    positions = torch.arange(rows, dtype=torch.int64)
    cache_loc = torch.arange(rows, dtype=torch.int32)
    key_cache = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    pool = MHATokenToKVPool(key_cache, key_cache.clone())
    expected = (object(), object(), object())
    handle = Handle()

    def prepare_stub(*args, **_kwargs):
        assert tuple(args[0].shape) == (rows, 2048)
        assert args[5].shape == (rows,)
        assert args[9].shape == (rows,)
        return (*expected, handle)

    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(welmv4, "_WELM_GRAPH_DUMP_ENABLED", False)
    monkeypatch.setattr(welmv4, "StandardQkvProjection", FakeProjection)
    monkeypatch.setattr(mk_fusion, "_prepare_welm_qkv", prepare_stub)

    assert attn.prepare_welm_v45_80a3_fused_pre_attn_cuda_graph(
        rows, positions, cache_loc, pool
    )
    entry = attn._mk_fused_qkv_graph_handles[rows]
    assert not entry["reusable_workspace"]

    hidden = torch.ones((rows, 2048), dtype=torch.bfloat16)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    result = attn._try_mk_fused_qkv_knorm_rope_kv_write(
        positions, hidden, SimpleNamespace()
    )
    assert all(actual is wanted for actual, wanted in zip(result, expected))
    assert torch.equal(entry["input"], hidden)
    assert handle.launch_count == 1


def test_welm_mk_graph_prepare_rejects_rows_above_mk_limit(monkeypatch):
    class FakeProjection:
        bias = None
        weight = torch.zeros((1024, 2048), dtype=torch.bfloat16)

    rows = mk_fusion._WELM_MK_MAX_FUSED_ROWS + 1
    attn = _make_attention(FakeProjection())
    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(welmv4, "_WELM_GRAPH_DUMP_ENABLED", False)
    monkeypatch.setattr(welmv4, "StandardQkvProjection", FakeProjection)
    monkeypatch.setattr(
        mk_fusion,
        "_prepare_welm_qkv",
        lambda *_args, **_kwargs: pytest.fail("unsupported rows must not prepare"),
    )

    key_cache = torch.zeros((1, 1, 256), dtype=torch.bfloat16)
    assert not attn.prepare_welm_v45_80a3_fused_pre_attn_cuda_graph(
        rows,
        torch.arange(rows, dtype=torch.int64),
        torch.arange(rows, dtype=torch.int32),
        MHATokenToKVPool(key_cache, key_cache.clone()),
    )


@pytest.mark.parametrize("rows", [2, 1025])
def test_welm_mk_fused_qkv_dispatches_and_slices_packed_weight(monkeypatch, rows):
    class FakeProjection:
        def __init__(self):
            self.weight = torch.randn((1024, 2048), dtype=torch.bfloat16)
            self.bias = None

    projection = FakeProjection()
    attn = welmv4.Qwen2MoeAttention.__new__(welmv4.Qwen2MoeAttention)
    nn.Module.__init__(attn)
    attn.is_nextn = False
    attn._welm_v45_80a3_fused_pre_attn_model_contract = True
    attn.suffix_parallel = False
    attn.scale_seq_attn_per_suffix = False
    attn.qk_norm = False
    attn.only_k_norm = True
    attn.head_dim = 256
    attn.qk_rope_head_dim = 64
    attn.q_size = 512
    attn.kv_size = 256
    attn.layer_idx = 3
    attn.qkv_proj = projection
    attn.k_norm = SimpleNamespace(
        weight=torch.ones((256,), dtype=torch.bfloat16), eps=1e-5
    )
    attn.rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.ones((16, 64), dtype=torch.float32)
    )

    hidden = torch.randn((rows, 2048), dtype=torch.bfloat16)
    positions = torch.arange(rows, dtype=torch.int64)
    key_cache = torch.zeros((rows + 16, 1, 256), dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(),
        out_cache_loc=torch.arange(rows, dtype=torch.int64),
        token_to_kv_pool=MHATokenToKVPool(key_cache, value_cache),
    )
    expected = (
        torch.empty((rows, 512), dtype=torch.bfloat16),
        torch.empty((rows, 256), dtype=torch.bfloat16),
        torch.empty((rows, 256), dtype=torch.bfloat16),
    )

    def fused_stub(*args, **kwargs):
        assert args[0] is hidden
        assert torch.equal(args[1], projection.weight[:512])
        assert torch.equal(args[2], projection.weight[512:768])
        assert torch.equal(args[3], projection.weight[768:1024])
        assert args[7] is key_cache
        assert args[8] is value_cache
        assert args[9] is forward_batch.out_cache_loc
        assert kwargs == {
            "q_bias": None,
            "k_bias": None,
            "v_bias": None,
            "k_norm_eps": 1e-5,
            "head_dim": 256,
        }
        return expected

    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(welmv4, "_WELM_GRAPH_DUMP_ENABLED", False)
    monkeypatch.setattr(welmv4, "StandardQkvProjection", FakeProjection)
    monkeypatch.setattr(welmv4, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4, "_welm_should_contract_kv_mirror", lambda _: False)
    monkeypatch.setattr(mk_fusion, "_load_welm_mk_fused_qkv", lambda: fused_stub)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    assert (
        attn._try_mk_fused_qkv_knorm_rope_kv_write(positions, hidden, forward_batch)
        is expected
    )


def test_welm_mk_fused_qkv_uses_prepared_handle_during_cuda_graph_capture(
    monkeypatch,
):
    class FakeProjection:
        bias = None
        weight = torch.zeros((1024, 2048), dtype=torch.bfloat16)

    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(welmv4, "_WELM_GRAPH_DUMP_ENABLED", False)
    monkeypatch.setattr(welmv4, "StandardQkvProjection", FakeProjection)
    monkeypatch.setattr(welmv4, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4, "_welm_should_contract_kv_mirror", lambda _: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    static_input = torch.empty((1, 2048), dtype=torch.bfloat16)
    expected = (
        torch.empty((1, 512), dtype=torch.bfloat16),
        torch.empty((1, 256), dtype=torch.bfloat16),
        torch.empty((1, 256), dtype=torch.bfloat16),
    )

    class Handle:
        rebound_calls = []

        def launch(self):
            pytest.fail("small-M graph capture must reuse the prepared workspace")

        def launch_prepared(self):
            pytest.fail("small-M graph capture must bind the graph input directly")

        def launch_prepared_rebound(self, rebindings):
            self.rebound_calls.append(rebindings)

    handle = Handle()
    attn = SimpleNamespace(
        _welm_v45_80a3_fused_pre_attn_model_contract=True,
        _mk_fused_qkv_graph_handles={
            1: {
                "input": static_input,
                "q": expected[0],
                "k": expected[1],
                "v": expected[2],
                "handle": handle,
                "reusable_workspace": True,
            }
        }
    )
    out_cache_loc = torch.zeros((1,), dtype=torch.int64)
    hidden = torch.zeros((1, 2048), dtype=torch.bfloat16)
    result = welmv4.Qwen2MoeAttention._try_mk_fused_qkv_knorm_rope_kv_write(
        attn,
        torch.arange(1, dtype=torch.int64),
        hidden,
        SimpleNamespace(
            forward_mode=_ForwardMode(),
            out_cache_loc=out_cache_loc,
        ),
    )
    assert result == expected
    assert len(handle.rebound_calls) == 1
    assert handle.rebound_calls[0][0] is hidden


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m", [1, 64])
def test_welm_mk_fused_qkv_reusable_eager_direct_and_rebound_launches(monkeypatch, m):
    hidden = torch.zeros((m, 2048), device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros((m,), device="cuda", dtype=torch.int64)
    cache_loc = torch.ones((m,), device="cuda", dtype=torch.int64)
    key_cache = torch.zeros((8, 1, 256), device="cuda", dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    q_weight = torch.empty((512, 2048), device="cuda", dtype=torch.bfloat16)
    k_weight = torch.empty((256, 2048), device="cuda", dtype=torch.bfloat16)
    v_weight = torch.empty_like(k_weight)
    k_norm_weight = torch.ones((256,), device="cuda", dtype=torch.bfloat16)
    cos_sin_cache = torch.ones((8, 64), device="cuda", dtype=torch.float32)
    expected = (
        torch.empty((m, 512), device="cuda", dtype=torch.bfloat16),
        torch.empty((m, 256), device="cuda", dtype=torch.bfloat16),
        torch.empty((m, 256), device="cuda", dtype=torch.bfloat16),
    )

    class Handle:
        def __init__(self):
            self.calls = []

        def prepare_workspace(self, *, stream):
            self.calls.append(("prepare", stream))

        def launch_prepared(self, *, stream):
            self.calls.append(("direct", stream))

        def launch_prepared_rebound(self, rebindings, *, stream):
            self.calls.append(("rebound", rebindings, stream))

    handle = Handle()
    prepare_calls = []

    def prepare_stub(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return (*expected, handle)

    monkeypatch.setattr(mk_fusion, "_prepare_welm_qkv", prepare_stub)
    attn = SimpleNamespace(k_norm=SimpleNamespace(eps=1e-5), head_dim=256)

    def launch(
        current_hidden,
        current_positions,
        current_cache_loc,
        current_q_weight=q_weight,
    ):
        return mk_fusion.WeLMV45_80A3FusedPreAttnMixin._try_mk_fused_qkv_reusable_eager(
            attn,
            current_hidden,
            current_q_weight,
            k_weight,
            v_weight,
            k_norm_weight,
            current_positions,
            cos_sin_cache,
            key_cache,
            value_cache,
            current_cache_loc,
            q_bias=None,
            k_bias=None,
            v_bias=None,
        )

    stream = int(torch.cuda.current_stream().cuda_stream)
    assert all(
        actual is wanted
        for actual, wanted in zip(launch(hidden, positions, cache_loc), expected)
    )
    assert all(
        actual is wanted
        for actual, wanted in zip(launch(hidden, positions, cache_loc), expected)
    )

    rebound_hidden = hidden.clone()
    rebound_positions = positions.clone()
    rebound_cache_loc = cache_loc.clone()
    assert all(
        actual is wanted
        for actual, wanted in zip(
            launch(rebound_hidden, rebound_positions, rebound_cache_loc), expected
        )
    )

    assert len(prepare_calls) == 1
    assert handle.calls[:3] == [
        ("prepare", stream),
        ("direct", stream),
        ("direct", stream),
    ]
    kind, rebindings, rebound_stream = handle.calls[3]
    assert kind == "rebound"
    assert rebound_stream == stream
    assert set(rebindings) == {0, 8, 15}
    assert rebindings[0] is rebound_hidden
    assert rebindings[8] is rebound_positions
    assert rebindings[15] is rebound_cache_loc

    replacement_q_weight = q_weight.clone()
    launch(hidden, positions, cache_loc, replacement_q_weight)
    assert len(prepare_calls) == 2
    assert handle.calls[-2:] == [("prepare", stream), ("direct", stream)]


def test_welm_mk_fused_qkv_covers_previous_precision(monkeypatch):
    class FakeProjection:
        bias = None
        weight = torch.zeros((1024, 2048), dtype=torch.bfloat16)

    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(welmv4, "_WELM_GRAPH_DUMP_ENABLED", False)
    monkeypatch.setattr(welmv4, "StandardQkvProjection", FakeProjection)
    monkeypatch.setattr(welmv4, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4, "welm_use_previous_precision", lambda: True)
    attn = _make_attention(FakeProjection())
    key_cache = torch.zeros((16, 1, 256), dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    expected = (object(), object(), object())
    calls = 0

    def fused_stub(*args, **kwargs):
        nonlocal calls
        calls += 1
        return expected

    monkeypatch.setattr(mk_fusion, "_load_welm_mk_fused_qkv", lambda: fused_stub)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    result = welmv4.Qwen2MoeAttention._try_mk_fused_qkv_knorm_rope_kv_write(
        attn,
        torch.arange(1, dtype=torch.int64),
        torch.zeros((1, 2048), dtype=torch.bfloat16),
        SimpleNamespace(
            forward_mode=_ForwardMode(),
            out_cache_loc=torch.zeros((1,), dtype=torch.int64),
            token_to_kv_pool=MHATokenToKVPool(key_cache, value_cache),
        ),
    )
    assert result is expected
    assert calls == 1


def test_welm_mk_fused_qkv_covers_dp_prefill_bias_int32_swa(monkeypatch):
    class FakeProjection:
        weight = torch.zeros((1024, 2048), dtype=torch.bfloat16)
        bias = torch.arange(1024, dtype=torch.bfloat16)

    class SwaPool(MHATokenToKVPool):
        def is_swa_layer(self, _layer_idx):
            return True

    attn = _make_attention(FakeProjection())
    hidden = torch.zeros((2, 2048), dtype=torch.bfloat16)
    positions = torch.arange(2, dtype=torch.int64)
    key_cache = torch.zeros((16, 1, 256), dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    pool = SwaPool(key_cache, value_cache)
    swa_loc = torch.tensor([3, 5], dtype=torch.int32)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: False),
        out_cache_loc=torch.tensor([11, 13], dtype=torch.int64),
        out_cache_loc_swa=swa_loc,
        token_to_kv_pool=pool,
    )
    expected = (object(), object(), object())

    def fused_stub(*args, **kwargs):
        assert args[9] is swa_loc
        assert torch.equal(kwargs["q_bias"], attn.qkv_proj.bias[:512])
        assert torch.equal(kwargs["k_bias"], attn.qkv_proj.bias[512:768])
        assert torch.equal(kwargs["v_bias"], attn.qkv_proj.bias[768:1024])
        return expected

    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(welmv4, "_WELM_GRAPH_DUMP_ENABLED", False)
    monkeypatch.setattr(welmv4, "StandardQkvProjection", FakeProjection)
    monkeypatch.setattr(welmv4, "is_dp_attention_enabled", lambda: True)
    monkeypatch.setattr(mk_fusion, "_load_welm_mk_fused_qkv", lambda: fused_stub)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    result = attn._try_mk_fused_qkv_knorm_rope_kv_write(
        positions, hidden, forward_batch
    )
    assert result is expected


def test_welm_mk_cache_loc_covers_mtp_and_mirror():
    attn = _make_attention()
    key_cache = torch.zeros((16, 1, 256), dtype=torch.bfloat16)
    pool = MHATokenToKVPool(key_cache, key_cache.clone())
    base = torch.arange(8, dtype=torch.int64)
    forward_batch = SimpleNamespace(
        token_to_kv_pool=pool,
        out_cache_loc=base,
    )
    forward_batch.welm_mtp_merge_kv_fill_draft = True
    forward_batch.welm_mtp_kv_fill_cache_loc = torch.tensor(
        [9, 10, 11], dtype=torch.int32
    )
    assert (
        attn._mk_cache_loc_for_rows(forward_batch, 3)
        is forward_batch.welm_mtp_kv_fill_cache_loc
    )

    forward_batch.welm_mtp_merge_kv_fill_draft = False
    forward_batch.welm_kv_mirror_contracted = True
    forward_batch.custom_last_cache_loc = torch.tensor([13, 15], dtype=torch.int64)
    assert (
        attn._mk_cache_loc_for_rows(forward_batch, 2)
        is forward_batch.custom_last_cache_loc
    )


def test_welm_mk_fused_qkv_routes_mirror_source_and_consumer(monkeypatch):
    class FakeSourceProjection:
        mirror_layer_indices = [48]
        weight = torch.zeros((1536, 2048), dtype=torch.bfloat16)
        bias = None

    class FakeConsumerProjection:
        mirror_layer_idx = 48
        imitated_layer_idx = 0
        weight = torch.zeros((512, 2048), dtype=torch.bfloat16)
        bias = None

    rows = 4
    hidden = torch.zeros((rows, 2048), dtype=torch.bfloat16)
    positions = torch.arange(rows, dtype=torch.int64)
    key_cache = torch.zeros((16, 1, 256), dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(),
        out_cache_loc=torch.arange(rows, dtype=torch.int32),
        token_to_kv_pool=MHATokenToKVPool(key_cache, value_cache),
    )
    source_result = (
        torch.empty((rows, 512), dtype=torch.bfloat16),
        torch.empty((rows, 256), dtype=torch.bfloat16),
        torch.empty((rows, 256), dtype=torch.bfloat16),
    )
    raw_mirror = (
        torch.empty((rows, 256), dtype=torch.bfloat16),
        torch.empty((rows, 256), dtype=torch.bfloat16),
    )
    consumer_result = (
        torch.empty((rows, 512), dtype=torch.bfloat16),
        torch.empty((rows, 256), dtype=torch.bfloat16),
        raw_mirror[1],
    )

    def source_stub(*args, **kwargs):
        assert args[0] is hidden
        assert args[1] is source_attn.qkv_proj.weight
        assert kwargs["packed_bias"] is None
        return (*source_result, (raw_mirror,))

    def consumer_stub(*args, **kwargs):
        assert args[0] is hidden
        assert args[1] is consumer_attn.qkv_proj.weight
        assert args[2] is raw_mirror[0]
        assert args[3] is raw_mirror[1]
        assert kwargs["q_bias"] is None
        return consumer_result

    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(welmv4, "_WELM_GRAPH_DUMP_ENABLED", False)
    monkeypatch.setattr(welmv4, "ImitateQkvMultiBankKvProjection", FakeSourceProjection)
    monkeypatch.setattr(welmv4, "MirrorQProjection", FakeConsumerProjection)
    monkeypatch.setattr(welmv4, "_welm_should_contract_kv_mirror", lambda _: False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        mk_fusion,
        "_load_welm_mk_mirror_fns",
        lambda: {"source": source_stub, "consumer": consumer_stub},
    )

    source_attn = _make_attention(FakeSourceProjection())
    source_attn.need_clear_kv_cache = False
    mirror_states = {}
    assert (
        source_attn._try_mk_fused_qkv_knorm_rope_kv_write(
            positions, hidden, forward_batch, mirror_states
        )
        == source_result
    )
    assert mirror_states == {48: raw_mirror}

    consumer_attn = _make_attention(FakeConsumerProjection())
    consumer_attn.need_clear_kv_cache = False
    assert (
        consumer_attn._try_mk_fused_qkv_knorm_rope_kv_write(
            positions, hidden, forward_batch, mirror_states
        )
        == consumer_result
    )
    assert mirror_states == {}


def test_welm_mk_graph_prepare_only_falls_back_for_config_errors(monkeypatch):
    class FakeMKConfigError(RuntimeError):
        pass

    class FakeProjection:
        weight = torch.zeros((1024, 2048), dtype=torch.bfloat16)
        bias = None

    attn = _make_attention(FakeProjection())
    cache_loc = torch.tensor([2, 5], dtype=torch.int32)
    positions = torch.arange(2, dtype=torch.int64)
    key_cache = torch.zeros((16, 1, 256), dtype=torch.bfloat16)
    pool = MHATokenToKVPool(key_cache, key_cache.clone())
    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(welmv4, "StandardQkvProjection", FakeProjection)
    monkeypatch.setattr(
        mk_fusion,
        "_is_mk_config_error",
        lambda exc: isinstance(exc, FakeMKConfigError),
    )
    monkeypatch.setattr(
        mk_fusion,
        "_prepare_welm_qkv",
        lambda *args, **kwargs: (_ for _ in ()).throw(FakeMKConfigError("shape")),
    )
    assert not attn.prepare_welm_v45_80a3_fused_pre_attn_cuda_graph(
        2, positions, cache_loc, pool
    )

    for error in (
        ValueError("value bug"),
        AttributeError("attribute bug"),
        ImportError("kernel-internal import bug"),
    ):
        monkeypatch.setattr(
            mk_fusion,
            "_prepare_welm_qkv",
            lambda *args, error=error, **kwargs: (_ for _ in ()).throw(error),
        )
        with pytest.raises(type(error), match=str(error)):
            attn.prepare_welm_v45_80a3_fused_pre_attn_cuda_graph(
                2, positions, cache_loc, pool
            )

    monkeypatch.setattr(
        mk_fusion,
        "_prepare_welm_qkv",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            mk_fusion._MKCapabilityUnavailable("missing symbol")
        ),
    )
    assert not attn.prepare_welm_v45_80a3_fused_pre_attn_cuda_graph(
        2, positions, cache_loc, pool
    )

    class WorkspaceHandle:
        supports_dynamic_input_rebind = True

        @staticmethod
        def prepare_workspace():
            raise FakeMKConfigError("workspace config")

    monkeypatch.setattr(
        mk_fusion,
        "_prepare_welm_qkv",
        lambda *args, **kwargs: (object(), object(), object(), WorkspaceHandle()),
    )
    assert not attn.prepare_welm_v45_80a3_fused_pre_attn_cuda_graph(
        2, positions, cache_loc, pool
    )


def test_welm_mk_fused_qkv_config_error_is_cached_per_configuration(monkeypatch):
    class FakeMKConfigError(RuntimeError):
        pass

    class FakeProjection:
        bias = None
        weight = torch.zeros((1024, 2048), dtype=torch.bfloat16)

    attn = welmv4.Qwen2MoeAttention.__new__(welmv4.Qwen2MoeAttention)
    nn.Module.__init__(attn)
    attn.is_nextn = False
    attn._welm_v45_80a3_fused_pre_attn_model_contract = True
    attn.suffix_parallel = False
    attn.scale_seq_attn_per_suffix = False
    attn.qk_norm = False
    attn.only_k_norm = True
    attn.head_dim = 256
    attn.qk_rope_head_dim = 64
    attn.q_size = 512
    attn.kv_size = 256
    attn.layer_idx = 0
    attn.qkv_proj = FakeProjection()
    attn.k_norm = SimpleNamespace(
        weight=torch.ones((256,), dtype=torch.bfloat16), eps=1e-5
    )
    attn.rotary_emb = SimpleNamespace(
        cos_sin_cache=torch.ones((16, 64), dtype=torch.float32)
    )

    hidden = torch.zeros((1, 2048), dtype=torch.bfloat16)
    positions = torch.zeros((1,), dtype=torch.int64)
    key_cache = torch.zeros((16, 1, 256), dtype=torch.bfloat16)
    value_cache = torch.zeros_like(key_cache)
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(),
        out_cache_loc=torch.ones((1,), dtype=torch.int64),
        token_to_kv_pool=MHATokenToKVPool(key_cache, value_cache),
    )

    calls = 0

    def fused_stub(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise FakeMKConfigError("unsupported stride")

    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(welmv4, "_WELM_GRAPH_DUMP_ENABLED", False)
    monkeypatch.setattr(mk_fusion, "_WELM_MK_FUSED_QKV_CONFIG_ERROR", FakeMKConfigError)
    monkeypatch.setattr(welmv4, "StandardQkvProjection", FakeProjection)
    monkeypatch.setattr(welmv4, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4, "_welm_should_contract_kv_mirror", lambda _: False)
    monkeypatch.setattr(mk_fusion, "_load_welm_mk_fused_qkv", lambda: fused_stub)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    assert (
        attn._try_mk_fused_qkv_knorm_rope_kv_write(positions, hidden, forward_batch)
        is None
    )
    assert (
        attn._try_mk_fused_qkv_knorm_rope_kv_write(positions, hidden, forward_batch)
        is None
    )
    assert calls == 1

    hidden_rows2 = torch.zeros((2, 2048), dtype=torch.bfloat16)
    positions_rows2 = torch.zeros((2,), dtype=torch.int64)
    forward_batch_rows2 = SimpleNamespace(
        forward_mode=_ForwardMode(),
        out_cache_loc=torch.ones((2,), dtype=torch.int64),
        token_to_kv_pool=forward_batch.token_to_kv_pool,
    )
    assert (
        attn._try_mk_fused_qkv_knorm_rope_kv_write(
            positions_rows2, hidden_rows2, forward_batch_rows2
        )
        is None
    )
    assert calls == 2


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("runtime attribute bug"),
        ImportError("kernel-internal import bug"),
    ],
)
def test_welm_mk_fused_qkv_surfaces_unexpected_runtime_errors(monkeypatch, error):
    class FakeProjection:
        bias = None
        weight = torch.zeros((1024, 2048), dtype=torch.bfloat16)

    attn = _make_attention(FakeProjection())
    hidden = torch.zeros((1, 2048), dtype=torch.bfloat16)
    positions = torch.zeros((1,), dtype=torch.int64)
    key_cache = torch.zeros((16, 1, 256), dtype=torch.bfloat16)
    forward_batch = SimpleNamespace(
        forward_mode=_ForwardMode(),
        out_cache_loc=torch.ones((1,), dtype=torch.int64),
        token_to_kv_pool=MHATokenToKVPool(key_cache, key_cache.clone()),
    )

    def fused_stub(*args, **kwargs):
        raise error

    monkeypatch.setattr(mk_fusion, "_WELM_V45_80A3_FUSED_PRE_ATTN_ENABLED", True)
    monkeypatch.setattr(welmv4, "_WELM_GRAPH_DUMP_ENABLED", False)
    monkeypatch.setattr(welmv4, "StandardQkvProjection", FakeProjection)
    monkeypatch.setattr(welmv4, "is_dp_attention_enabled", lambda: False)
    monkeypatch.setattr(welmv4, "_welm_should_contract_kv_mirror", lambda _: False)
    monkeypatch.setattr(mk_fusion, "_load_welm_mk_fused_qkv", lambda: fused_stub)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    with pytest.raises(type(error), match=str(error)):
        attn._try_mk_fused_qkv_knorm_rope_kv_write(positions, hidden, forward_batch)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
