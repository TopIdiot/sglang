import types

import pytest
import torch

from sglang.srt.state_capturer.base import BaseDeviceCache
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    encode_routed_experts_payload,
    extract_routed_experts_from_meta_info,
)
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _apply_router_replay_from_forward_batch,
    apply_router_replay_topk_override,
    fused_topk_torch_native,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import (
    build_router_replay_decode_batch,
    build_router_replay_extend_batch,
    validate_router_replay_experts,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_dp_attn_mixin import (
    MLPSyncBatchInfo,
    _update_gather_batch,
)
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.model_executor.cuda_graph_runner import (
    DecodeInputBuffers,
    copy_router_replay_to_cuda_graph_buffers,
)
from sglang.srt.model_executor.forward_batch_context import set_current_forward_batch
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


def make_trace(num_tokens=5, num_layers=3, top_k=2):
    return torch.tensor(
        [
            [
                [token_pos * 10000 + layer_id * 100 + topk_idx for topk_idx in range(top_k)]
                for layer_id in range(num_layers)
            ]
            for token_pos in range(num_tokens)
        ],
        dtype=torch.int32,
    )


def make_req(rid, trace, *, extend_input_len=0, seqlen=0):
    return types.SimpleNamespace(
        rid=rid,
        router_replay_experts=trace,
        extend_input_len=extend_input_len,
        seqlen=seqlen,
    )


def make_routed_experts_capturer(num_tokens=32, num_layers=3, top_k=2):
    server_args = ServerArgs(model_path="dummy")
    server_args.chunked_prefill_size = num_tokens
    server_args.dp_size = 1
    set_global_server_args_for_scheduler(server_args)
    capturer = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    capturer.num_fused_shared_experts = 0
    capturer.topk_size = top_k
    capturer.host_cache = types.SimpleNamespace(
        buffer=torch.full((num_tokens, num_layers, top_k), -1, dtype=torch.int32)
    )
    capturer.device_cache = BaseDeviceCache(
        max_batch_size=num_tokens,
        num_layers=num_layers,
        topk_size=top_k,
        device="cpu",
        name="test_routed_experts",
    )
    return capturer


def test_generate_req_input_accepts_single_rank3_routed_experts():
    trace = make_trace(num_tokens=2).tolist()
    req = GenerateReqInput(
        input_ids=[1, 2],
        sampling_params={"max_new_tokens": 1},
        routed_experts=trace,
    )

    req.normalize_batch_and_arguments()

    assert req.is_single
    assert req.routed_experts == trace


def test_generate_req_input_requires_batch_replay_all_or_none():
    trace = make_trace(num_tokens=2).tolist()
    req = GenerateReqInput(
        input_ids=[[1, 2], [3, 4]],
        sampling_params={"max_new_tokens": 1},
        routed_experts=[trace, None],
    )

    with pytest.raises(ValueError, match="all items.*routed_experts"):
        req.normalize_batch_and_arguments()


def test_generate_req_input_slices_batch_routed_experts_per_item():
    traces = [
        make_trace(num_tokens=2).tolist(),
        (make_trace(num_tokens=2) + 7).tolist(),
    ]
    req = GenerateReqInput(
        input_ids=[[1, 2], [3, 4]],
        sampling_params={"max_new_tokens": 1},
        routed_experts=traces,
    )

    req.normalize_batch_and_arguments()

    assert not req.is_single
    assert req[0].routed_experts == traces[0]
    assert req[1].routed_experts == traces[1]


def test_generate_req_input_slices_batch_forced_decode_tokens_per_item():
    forced = [[10, 11, 12], [20, 21, 22]]
    req = GenerateReqInput(
        input_ids=[[1, 2], [3, 4]],
        sampling_params={"max_new_tokens": 3},
        forced_decode_token_ids=forced,
    )

    req.normalize_batch_and_arguments()

    assert not req.is_single
    assert req[0].forced_decode_token_ids == forced[0]
    assert req[1].forced_decode_token_ids == forced[1]


def test_tokenizer_forced_decode_tokens_fill_max_new_tokens():
    req = GenerateReqInput(
        input_ids=[1, 2],
        sampling_params={},
        forced_decode_token_ids=[10, 11, 12],
    )
    tokenizer_manager = types.SimpleNamespace(
        context_len=16,
        num_reserved_tokens=0,
        validate_total_tokens=True,
        model_config=types.SimpleNamespace(vocab_size=100),
        server_args=types.SimpleNamespace(
            allow_auto_truncate=False,
            enable_return_hidden_states=False,
            enable_moe_router_replay=False,
            enable_custom_logit_processor=False,
        ),
    )

    TokenizerManager._validate_one_request(tokenizer_manager, req, req.input_ids)

    assert req.sampling_params["max_new_tokens"] == 3


def test_tokenizer_forced_decode_tokens_reject_max_new_tokens_mismatch():
    req = GenerateReqInput(
        input_ids=[1, 2],
        sampling_params={"max_new_tokens": 2},
        forced_decode_token_ids=[10, 11, 12],
    )
    tokenizer_manager = types.SimpleNamespace(
        context_len=16,
        num_reserved_tokens=0,
        validate_total_tokens=True,
        model_config=types.SimpleNamespace(vocab_size=100),
        server_args=types.SimpleNamespace(
            allow_auto_truncate=False,
            enable_return_hidden_states=False,
            enable_moe_router_replay=False,
            enable_custom_logit_processor=False,
        ),
    )

    with pytest.raises(ValueError, match="length must match"):
        TokenizerManager._validate_one_request(tokenizer_manager, req, req.input_ids)


def test_tokenizer_resolves_remote_dense_router_replay_payload():
    values = make_trace(num_tokens=3, num_layers=2, top_k=2).numpy()

    class Store:
        def get(self, ref):
            assert ref["key"] == "trace-key"
            return values.tobytes()

    tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
    tokenizer_manager.routed_experts_store = Store()
    payload = {
        "format": "remote_dense",
        "values": {
            "format": "remote",
            "backend": "redis",
            "key": "trace-key",
            "encoding": "raw_tensor_bytes",
            "shape": list(values.shape),
            "dtype": "int32",
        },
        "shape": list(values.shape),
        "dtype": "int32",
    }

    resolved = TokenizerManager._resolve_router_replay_experts(
        tokenizer_manager, payload
    )

    assert resolved.dtype == torch.int32
    assert resolved.shape == values.shape
    assert resolved._sglang_router_replay_trusted
    assert torch.equal(resolved[0, 0], torch.tensor(values[0, 0], dtype=torch.int32))
    assert torch.equal(resolved, torch.tensor(values, dtype=torch.int32))


def test_tokenizer_rejects_remote_masked_router_replay_payload():
    tokenizer_manager = TokenizerManager.__new__(TokenizerManager)

    with pytest.raises(ValueError, match="Unsupported routed_experts payload format"):
        TokenizerManager._resolve_router_replay_experts(
            tokenizer_manager, {"format": "remote_masked_dense", "values": {}}
        )


def test_tokenizer_rejects_dense_router_replay_payload_without_values():
    tokenizer_manager = TokenizerManager.__new__(TokenizerManager)

    with pytest.raises(ValueError, match="dense routed_experts requires a values"):
        TokenizerManager._resolve_router_replay_experts(
            tokenizer_manager, {"format": "dense", "shape": [1, 2, 2]}
        )


def test_scheduler_forced_decode_tokens_override_after_sampling():
    req = types.SimpleNamespace(output_ids=[10, 11], forced_decode_token_ids=[1, 2, 3])

    token_id = SchedulerOutputProcessorMixin.maybe_apply_forced_decode_token(
        None, req, 99
    )

    assert token_id == 3


def test_scheduler_forced_decode_tokens_patch_batch_output_ids_tensor():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    req0 = types.SimpleNamespace(output_ids=[], forced_decode_token_ids=[7, 8])
    req1 = types.SimpleNamespace(output_ids=[3], forced_decode_token_ids=[9, 10])
    scheduler = types.SimpleNamespace(enable_overlap=False)
    batch = types.SimpleNamespace(
        output_ids=torch.tensor([99, 88, 77], dtype=torch.int64, device=device)
    )

    updates = [
        (0, req0, SchedulerOutputProcessorMixin.get_forced_decode_token_id(None, req0)),
        (1, req1, SchedulerOutputProcessorMixin.get_forced_decode_token_id(None, req1)),
    ]
    SchedulerOutputProcessorMixin.maybe_patch_batch_output_ids_with_forced_tokens(
        scheduler, batch, updates
    )

    assert batch.output_ids.tolist() == [7, 10, 77]


def test_scheduler_forced_decode_tokens_patch_overlap_future_map():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    req0 = types.SimpleNamespace(output_ids=[], forced_decode_token_ids=[7])
    req1 = types.SimpleNamespace(output_ids=[3], forced_decode_token_ids=[9, 10])
    future_map = types.SimpleNamespace(
        token_ids_buf=torch.full((8,), -1, dtype=torch.int64, device=device)
    )
    scheduler = types.SimpleNamespace(enable_overlap=True, future_map=future_map)
    batch = types.SimpleNamespace(
        output_ids=torch.tensor([-2, -5, -6], dtype=torch.int64, device=device)
    )

    updates = [
        (0, req0, SchedulerOutputProcessorMixin.get_forced_decode_token_id(None, req0)),
        (1, req1, SchedulerOutputProcessorMixin.get_forced_decode_token_id(None, req1)),
    ]
    SchedulerOutputProcessorMixin.maybe_patch_batch_output_ids_with_forced_tokens(
        scheduler, batch, updates
    )

    assert batch.output_ids.tolist() == [-2, -5, -6]
    assert future_map.token_ids_buf[[2, 5]].tolist() == [7, 10]


def test_tokenizer_rejects_router_replay_without_server_flag():
    trace = make_trace(num_tokens=2).tolist()
    req = GenerateReqInput(
        input_ids=[1, 2],
        sampling_params={"max_new_tokens": 1},
        routed_experts=trace,
    )
    tokenizer_manager = types.SimpleNamespace(
        context_len=16,
        num_reserved_tokens=0,
        validate_total_tokens=True,
        server_args=types.SimpleNamespace(
            allow_auto_truncate=False,
            enable_return_hidden_states=False,
            enable_moe_router_replay=False,
            enable_custom_logit_processor=False,
        ),
    )

    with pytest.raises(ValueError, match="--enable-moe-router-replay"):
        TokenizerManager._validate_one_request(tokenizer_manager, req, req.input_ids)


def test_tokenizer_allows_return_routed_experts_without_router_replay_flag():
    req = GenerateReqInput(
        input_ids=[1, 2],
        sampling_params={"max_new_tokens": 1},
        return_routed_experts=True,
    )
    tokenizer_manager = types.SimpleNamespace(
        context_len=16,
        num_reserved_tokens=0,
        validate_total_tokens=True,
        server_args=types.SimpleNamespace(
            allow_auto_truncate=False,
            enable_return_hidden_states=False,
            enable_return_routed_experts=True,
            enable_moe_router_replay=False,
            enable_custom_logit_processor=False,
        ),
    )

    TokenizerManager._validate_one_request(tokenizer_manager, req, req.input_ids)


def test_scheduler_rejects_router_replay_without_server_flag():
    def set_finish_with_abort(message):
        req.abort_msg = message

    req = types.SimpleNamespace(
        router_replay_experts=None,
        abort_msg=None,
        set_finish_with_abort=set_finish_with_abort,
    )
    recv_req = types.SimpleNamespace(router_replay_experts=make_trace(num_tokens=2))
    queued_reqs = []
    scheduler = types.SimpleNamespace(
        server_args=types.SimpleNamespace(enable_moe_router_replay=False),
        init_req_max_new_tokens=lambda req: setattr(
            req, "max_new_tokens_initialized", True
        ),
        _add_request_to_queue=lambda req: queued_reqs.append(req),
    )

    ok = Scheduler._validate_router_replay_request(scheduler, req, recv_req)

    assert not ok
    assert "--enable-moe-router-replay" in req.abort_msg
    assert req.max_new_tokens_initialized
    assert queued_reqs == [req]
    assert req.router_replay_experts is None


def test_validate_router_replay_experts_converts_to_cpu_int32():
    trace = make_trace(num_tokens=4, num_layers=3, top_k=2).to(torch.int64)

    out = validate_router_replay_experts(
        trace.tolist(),
        num_layers=3,
        num_experts_per_tok=2,
        num_logical_routed_experts=10_000_000,
        rid="req-ok",
    )

    assert out.dtype == torch.int32
    assert out.device.type == "cpu"
    assert torch.equal(out, trace.to(torch.int32))


def test_validate_router_replay_experts_rejects_short_prompt_trace():
    trace = make_trace(num_tokens=2, num_layers=3, top_k=2)

    with pytest.raises(ValueError, match="too short"):
        validate_router_replay_experts(
            trace,
            num_layers=3,
            num_experts_per_tok=2,
            num_logical_routed_experts=10_000_000,
            min_router_seq_len=3,
            rid="req-short-prompt",
        )


@pytest.mark.parametrize(
    ("trace", "match"),
    [
        ([[1, 2], [3, 4]], "rank-3"),
        (make_trace(num_layers=2).tolist(), "num_layers"),
        (make_trace(top_k=1).tolist(), "num_experts_per_tok"),
        ((make_trace() - 1).tolist(), "out of range"),
        ((make_trace() + 10_000_000).tolist(), "out of range"),
    ],
)
def test_validate_router_replay_experts_rejects_bad_shapes_and_ids(trace, match):
    with pytest.raises(ValueError, match=match):
        validate_router_replay_experts(
            trace,
            num_layers=3,
            num_experts_per_tok=2,
            num_logical_routed_experts=10_000_000,
            rid="req-bad",
        )


def test_validate_router_replay_experts_allows_kv_mirror_masked_layers():
    trace = make_trace(num_tokens=2, num_layers=3, top_k=2)
    trace[:, 1, :] = -1

    out = validate_router_replay_experts(
        trace,
        num_layers=3,
        num_experts_per_tok=2,
        num_logical_routed_experts=10_000_000,
        rid="req-kv-mirror",
    )

    assert torch.equal(out, trace)


def test_validate_router_replay_experts_rejects_negative_ids_below_mask_value():
    trace = make_trace(num_tokens=2, num_layers=3, top_k=2)
    trace[0, 1, 0] = -2

    with pytest.raises(ValueError, match="-1 for masked layers"):
        validate_router_replay_experts(
            trace,
            num_layers=3,
            num_experts_per_tok=2,
            num_logical_routed_experts=10_000_000,
            rid="req-negative",
        )


def test_build_router_replay_extend_batch_uses_logical_offsets():
    trace = make_trace(num_tokens=5)
    req = make_req("prefill", trace, extend_input_len=2)

    topk_ids, mask = build_router_replay_extend_batch(
        [req], logical_prefix_lens=[2], device="cpu"
    )

    assert torch.equal(topk_ids.cpu(), trace[2:4])
    assert torch.equal(mask.cpu(), torch.tensor([True, True]))


def test_build_router_replay_extend_batch_supports_chunked_prefill_offsets():
    trace = make_trace(num_tokens=6)
    req = make_req("chunk", trace, extend_input_len=2)

    first_ids, first_mask = build_router_replay_extend_batch(
        [req], logical_prefix_lens=[0], device="cpu"
    )
    second_ids, second_mask = build_router_replay_extend_batch(
        [req], logical_prefix_lens=[2], device="cpu"
    )

    assert torch.equal(first_ids.cpu(), trace[0:2])
    assert torch.equal(second_ids.cpu(), trace[2:4])
    assert torch.equal(first_mask.cpu(), torch.tensor([True, True]))
    assert torch.equal(second_mask.cpu(), torch.tensor([True, True]))


def test_build_router_replay_decode_batch_uses_seqlen_minus_one():
    trace = make_trace(num_tokens=5)
    req = make_req("decode", trace, seqlen=4)

    topk_ids, mask = build_router_replay_decode_batch([req], device="cpu")

    assert torch.equal(topk_ids.cpu(), trace[3:4])
    assert torch.equal(mask.cpu(), torch.tensor([True]))


def test_build_router_replay_decode_batch_uses_explicit_trace_positions():
    trace = make_trace(num_tokens=5)
    req = make_req("decode-overlap", trace, seqlen=4)

    topk_ids, mask = build_router_replay_decode_batch(
        [req], device="cpu", trace_positions=[4]
    )

    assert torch.equal(topk_ids.cpu(), trace[4:5])
    assert torch.equal(mask.cpu(), torch.tensor([True]))


def test_return_routed_experts_capturer_overwrites_kv_slots_dense():
    capturer = make_routed_experts_capturer(num_tokens=32, num_layers=3, top_k=2)
    slots = torch.tensor([10, 11, 12, 13, 14, 15], dtype=torch.long)
    forward_batch = types.SimpleNamespace(
        out_cache_loc=slots,
    )

    layer0 = torch.tensor(
        [[0, 1], [10, 11], [20, 21], [30, 31], [40, 41], [50, 51]],
        dtype=torch.int32,
    )
    layer1 = layer0 + 100
    layer2 = layer0 + 200

    with set_current_forward_batch(forward_batch):
        capturer.capture(0, layer0)
        capturer.capture(1, layer1)
        capturer.capture(2, layer2)

    capturer.on_forward_end(
        forward_batch,
        can_run_graph=False,
        cuda_graph_batch=None,
        no_copy_to_cpu=False,
    )
    expected = torch.stack([layer0, layer1, layer2], dim=1)
    assert torch.equal(capturer.host_cache.buffer[slots], expected)

    router_rows = slots.numel()
    req_to_token_pool = types.SimpleNamespace(req_to_token=slots.view(1, router_rows))
    routed = capturer.get_topk(
        req_pool_idx=0,
        seqlen=router_rows + 1,
        req_to_token_pool=req_to_token_pool,
    )

    assert routed["schema_version"] == 3
    assert routed["format"] == "dense"
    assert routed["shape"] == [6, 3, 2]
    assert routed["missing_value"] == -1
    assert routed["invalid_cache_locs"] == 0
    assert "valid_mask" not in routed
    assert torch.equal(routed["values"], expected)


def test_return_routed_experts_overlap_output_keeps_device_values_until_copy():
    capturer = make_routed_experts_capturer(num_tokens=16, num_layers=2, top_k=2)
    forward_batch = types.SimpleNamespace(
        out_cache_loc=torch.tensor([4, 5], dtype=torch.long),
    )
    layer0 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    with set_current_forward_batch(forward_batch):
        capturer.capture(0, layer0)

    output = capturer.on_forward_end(
        forward_batch,
        can_run_graph=False,
        cuda_graph_batch=None,
        no_copy_to_cpu=True,
    )

    assert torch.equal(output.topk[:, 0, :], layer0)
    output.copy_to_cpu()
    output.finalize()
    assert torch.equal(
        capturer.host_cache.buffer[forward_batch.out_cache_loc, 0], layer0
    )


def test_routed_experts_dense_payload_encoding_round_trip():
    payload = {
        "schema_version": 3,
        "format": "dense",
        "values": make_trace(num_tokens=2),
        "shape": [2, 3, 2],
        "missing_value": -1,
    }

    encoded = encode_routed_experts_payload(payload)
    assert isinstance(encoded["values"], str)

    decoded = extract_routed_experts_from_meta_info(
        {"meta_info": {"routed_experts": encoded}}
    )
    assert decoded.tolist() == payload["values"].numpy().tolist()


def test_build_router_replay_decode_batch_masks_overlap_overrun_after_max_new_tokens():
    trace = make_trace(num_tokens=2)
    req = make_req("decode-overlap-overrun", trace, seqlen=2)
    req.origin_input_ids = [11]
    req.sampling_params = types.SimpleNamespace(max_new_tokens=2)

    topk_ids, mask = build_router_replay_decode_batch(
        [req], device="cpu", trace_positions=[2]
    )

    assert torch.equal(topk_ids.cpu(), torch.zeros_like(trace[:1]))
    assert torch.equal(mask.cpu(), torch.tensor([False]))


def test_build_router_replay_batch_rejects_short_trace():
    trace = make_trace(num_tokens=2)
    req = make_req("short", trace, extend_input_len=3)

    with pytest.raises(ValueError, match="replay trace.*too short"):
        build_router_replay_extend_batch([req], logical_prefix_lens=[0], device="cpu")


def test_build_router_replay_batch_rejects_mixed_replay_requests():
    trace = make_trace(num_tokens=2)
    replay_req = make_req("replay", trace, extend_input_len=1)
    normal_req = make_req("normal", None, extend_input_len=1)

    with pytest.raises(ValueError, match="mix.*router replay"):
        build_router_replay_extend_batch(
            [replay_req, normal_req], logical_prefix_lens=[0, 0], device="cpu"
        )


def test_topk_override_forces_masked_rows_and_gathers_softmax_weights():
    router_logits = torch.tensor(
        [[1.0, 4.0, 2.0, 3.0], [5.0, 1.0, 4.0, 0.0]], dtype=torch.float32
    )
    hidden_states = torch.ones((2, 4), dtype=torch.float32)
    topk_config = TopKConfig(top_k=2, renormalize=True, scoring_func="softmax")
    topk_weights, topk_ids = fused_topk_torch_native(
        hidden_states, router_logits, topk=2, renormalize=True
    )
    forced_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    mask = torch.tensor([True, False])

    out_weights, out_ids = apply_router_replay_topk_override(
        router_logits=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        forced_ids=forced_ids,
        mask=mask,
        topk_config=topk_config,
    )

    expected_scores = router_logits.softmax(dim=-1).gather(1, forced_ids.long())
    expected_scores = expected_scores / expected_scores.sum(dim=-1, keepdim=True)
    assert torch.equal(out_ids[0], forced_ids[0])
    assert torch.equal(out_ids[1], topk_ids[1])
    torch.testing.assert_close(out_weights[0], expected_scores[0])
    torch.testing.assert_close(out_weights[1], topk_weights[1])


def test_topk_override_gathers_sigmoid_weights_without_renormalize():
    router_logits = torch.tensor([[1.0, -2.0, 3.0, 0.5]], dtype=torch.float32)
    topk_config = TopKConfig(top_k=2, renormalize=False, scoring_func="sigmoid")
    topk_weights = torch.tensor([[0.9, 0.8]], dtype=torch.float32)
    topk_ids = torch.tensor([[2, 0]], dtype=torch.int32)
    forced_ids = torch.tensor([[1, 3]], dtype=torch.int32)
    mask = torch.tensor([True])

    out_weights, out_ids = apply_router_replay_topk_override(
        router_logits=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        forced_ids=forced_ids,
        mask=mask,
        topk_config=topk_config,
    )

    expected = router_logits.sigmoid().gather(1, forced_ids.long())
    assert torch.equal(out_ids, forced_ids)
    torch.testing.assert_close(out_weights, expected)


def test_router_replay_topk_override_skips_kv_mirror_masked_layer_rows():
    router_logits = torch.tensor(
        [[1.0, 4.0, 2.0, 3.0], [5.0, 1.0, 4.0, 0.0]], dtype=torch.float32
    )
    hidden_states = torch.ones((2, 4), dtype=torch.float32)
    topk_config = TopKConfig(top_k=2, renormalize=True, scoring_func="softmax")
    topk_weights, topk_ids = fused_topk_torch_native(
        hidden_states, router_logits, topk=2, renormalize=True
    )
    replay_topk_ids = torch.tensor(
        [
            [[0, 2], [-1, -1]],
            [[1, 3], [2, 0]],
        ],
        dtype=torch.int32,
    )
    forward_batch = types.SimpleNamespace(
        router_replay_topk_ids=replay_topk_ids,
        router_replay_mask=torch.tensor([True, True]),
    )

    with set_current_forward_batch(forward_batch):
        layer0_weights, layer0_ids = _apply_router_replay_from_forward_batch(
            router_logits=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            topk_config=topk_config,
            layer_id=0,
        )
        layer1_weights, layer1_ids = _apply_router_replay_from_forward_batch(
            router_logits=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            topk_config=topk_config,
            layer_id=1,
        )

    assert torch.equal(layer0_ids, replay_topk_ids[:, 0, :].to(layer0_ids.dtype))
    assert torch.equal(layer1_ids[0], topk_ids[0])
    assert torch.equal(layer1_ids[1], replay_topk_ids[1, 1, :].to(layer1_ids.dtype))
    torch.testing.assert_close(layer1_weights[0], topk_weights[0])


def test_router_replay_topk_override_contracts_prefill_to_kv_mirror_last_q():
    router_logits = torch.tensor(
        [[1.0, 4.0, 2.0, 3.0], [5.0, 1.0, 4.0, 0.0]], dtype=torch.float32
    )
    hidden_states = torch.ones((2, 4), dtype=torch.float32)
    topk_config = TopKConfig(top_k=2, renormalize=True, scoring_func="softmax")
    topk_weights, topk_ids = fused_topk_torch_native(
        hidden_states, router_logits, topk=2, renormalize=True
    )
    replay_topk_ids = torch.tensor(
        [
            [[0, 1]],
            [[1, 2]],
            [[2, 3]],
            [[3, 0]],
            [[0, 2]],
        ],
        dtype=torch.int32,
    )
    forward_batch = types.SimpleNamespace(
        router_replay_topk_ids=replay_topk_ids,
        router_replay_mask=torch.ones((5,), dtype=torch.bool),
        custom_last_index=torch.tensor([1, 4], dtype=torch.long),
    )

    with set_current_forward_batch(forward_batch):
        out_weights, out_ids = _apply_router_replay_from_forward_batch(
            router_logits=router_logits,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            topk_config=topk_config,
            layer_id=0,
        )

    assert torch.equal(out_ids, replay_topk_ids[[1, 4], 0, :].to(out_ids.dtype))
    expected_scores = router_logits.softmax(dim=-1).gather(1, out_ids.long())
    expected_scores = expected_scores / expected_scores.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(out_weights, expected_scores.to(out_weights.dtype))


def test_cuda_graph_replay_copy_clears_padding_mask():
    src_ids = make_trace(num_tokens=2)
    src_mask = torch.tensor([True, True])
    dst_ids = torch.full((4, 3, 2), -1, dtype=torch.int32)
    dst_mask = torch.ones((4,), dtype=torch.bool)

    copy_router_replay_to_cuda_graph_buffers(
        dst_topk_ids=dst_ids,
        dst_mask=dst_mask,
        src_topk_ids=src_ids,
        src_mask=src_mask,
        raw_num_token=2,
        static_num_token=4,
    )

    assert torch.equal(dst_ids[:2], src_ids)
    assert torch.equal(dst_mask, torch.tensor([True, True, False, False]))


def test_cuda_graph_replay_buffers_use_global_capacity_for_mlp_sync():
    buffers = DecodeInputBuffers.create(
        device=torch.device("cpu"),
        max_bs=2,
        max_num_token=4,
        hidden_size=8,
        vocab_size=16,
        dtype=torch.float32,
        dp_size=4,
        pp_size=1,
        is_encoder_decoder=False,
        require_mlp_tp_gather=True,
        seq_len_fill_value=1,
        encoder_len_fill_value=0,
        num_tokens_per_bs=1,
        cache_loc_dtype=torch.int64,
        enable_mamba_track=False,
        welm_oe_decode_hash_num_branches=0,
        scale_seq_factor=1,
        router_replay_num_layers=3,
        router_replay_top_k=2,
    )

    assert buffers.router_replay_topk_ids.shape == (16, 3, 2)
    assert buffers.router_replay_mask.shape == (16,)
    assert buffers.router_replay_local_topk_ids.shape == (4, 3, 2)
    assert buffers.router_replay_local_mask.shape == (4,)


def test_mlp_sync_adds_empty_replay_buffers_on_idle_dp_rank():
    batch = types.SimpleNamespace(
        global_num_tokens=None,
        global_num_tokens_for_logprob=None,
        is_extend_in_batch=False,
        tbo_split_seq_index=None,
        global_forward_mode=None,
        can_run_dp_cuda_graph=False,
        router_replay_topk_ids=None,
        router_replay_mask=None,
        model_config=types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(
                num_hidden_layers=3,
                num_experts_per_tok=2,
            )
        ),
        device=torch.device("cpu"),
    )
    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=4,
        tp_size=1,
        cp_size=1,
        num_tokens=0,
        num_tokens_for_logprob=0,
        can_cuda_graph=False,
        is_extend_in_batch=False,
        local_can_run_tbo=True,
        local_forward_mode=0,
        has_router_replay=True,
        global_num_tokens=[0, 1, 0, 0],
        global_num_tokens_for_logprob=[0, 1, 0, 0],
    )

    _update_gather_batch(batch, mlp_sync_info, require_mlp_tp_gather=True)

    assert batch.router_replay_topk_ids.shape == (0, 3, 2)
    assert batch.router_replay_topk_ids.dtype == torch.int32
    assert batch.router_replay_mask.shape == (0,)
    assert batch.router_replay_mask.dtype == torch.bool


def test_mlp_sync_local_tensor_includes_router_replay_flag():
    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=4,
        tp_size=1,
        cp_size=1,
        num_tokens=2,
        num_tokens_for_logprob=2,
        can_cuda_graph=True,
        is_extend_in_batch=False,
        local_can_run_tbo=True,
        local_forward_mode=1,
        has_router_replay=True,
    )

    local_tensor = mlp_sync_info._get_local_tensor(device="cpu")
    fallback_tensor = mlp_sync_info._get_fallback_tensor(device="cpu")

    assert local_tensor.tolist() == [2, 2, 1, 0, 1, 1, 1]
    assert fallback_tensor.tolist() == [0, 0, 1, 0, 1, 4, 0]
