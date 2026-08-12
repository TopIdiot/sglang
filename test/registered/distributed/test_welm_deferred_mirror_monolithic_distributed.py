"""Distributed control-flow coverage for monolithic deferred Prefill."""

from __future__ import annotations

import multiprocessing as mp
import socket
from contextlib import nullcontext
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import welmv4
from sglang.srt.models.welm_deferred_mirror import WelmDeferredExecutionRole
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=15, suite="stage-a-test-cpu")

_WORLD_SIZE = 2


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _CollectiveLayer(nn.Module):
    def __init__(self, layer_id: int, trace: list[tuple[int, int, bool]]) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.trace = trace
        self.kv_mirror_layers = ()

    def forward(
        self,
        _positions,
        hidden_states,
        _forward_batch,
        residual,
        kv_mirror_states,
    ):
        suffix_collective = self.layer_id >= 2
        self.trace.append(
            (self.layer_id, int(hidden_states.shape[0]), suffix_collective)
        )
        if suffix_collective:
            participation = torch.tensor([1], dtype=torch.int64)
            dist.all_reduce(participation)
            assert participation.item() == _WORLD_SIZE
        if hidden_states.shape[0] != 0:
            hidden_states = hidden_states + self.layer_id + 1
        return hidden_states, residual, kv_mirror_states


def _make_model(trace: list[tuple[int, int, bool]]) -> welmv4.Qwen2MoeModel:
    model = welmv4.Qwen2MoeModel.__new__(welmv4.Qwen2MoeModel)
    nn.Module.__init__(model)
    model.token_owner_runtime = None
    model.config = SimpleNamespace(num_hidden_layers=4)
    model.deferred_execution = SimpleNamespace(
        role=WelmDeferredExecutionRole.MONOLITHIC,
        prefill_execution_end_layer=2,
        omit_final_output=False,
    )
    model.execution_end_layer = 4
    model.start_layer = 0
    model.end_layer = 4
    model.layers = nn.ModuleList([_CollectiveLayer(i, trace) for i in range(4)])
    model.layers_to_capture = []
    model.mk_moe_router = None
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.embed_tokens = object()
    model.oe_grams = []
    model.oe_vocab_sizes = []
    model.vocab_size = 32
    model.scale_seq_times = 0
    model.norm = mock.MagicMock(side_effect=lambda hidden: hidden)
    model.norm.weight = torch.ones(4, dtype=torch.bfloat16)
    return model


def _run_case(rank: int, flags: list[bool], nondeferred_rows: int = 3):
    local_deferred = flags[rank]
    local_rows = 2 if local_deferred else nondeferred_rows
    counts = [2 if deferred else nondeferred_rows for deferred in flags]
    logprob_counts = [1 if deferred else nondeferred_rows for deferred in flags]
    trace: list[tuple[int, int, bool]] = []
    model = _make_model(trace)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        welm_deferred_prefill=local_deferred,
        welm_deferred_prefill_flags=flags,
        global_num_tokens_cpu=list(counts),
        global_num_tokens_gpu=torch.tensor(counts, dtype=torch.int64),
        global_num_tokens_for_logprob_cpu=list(logprob_counts),
        global_num_tokens_for_logprob_gpu=torch.tensor(
            logprob_counts, dtype=torch.int64
        ),
        global_dp_buffer_len=sum(counts),
        dp_padding_mode=DpPaddingMode.SUM_LEN,
        dp_local_start_pos=torch.tensor(rank),
        dp_local_num_tokens=torch.tensor(local_rows),
        num_token_non_padded=None,
        scale_seq_factor=1,
        can_run_tbo=False,
        spec_info=None,
        spec_algorithm=None,
        capture_hidden_mode=SimpleNamespace(need_capture=lambda: False),
        model_specific_states=None,
        attn_cp_prefill_runtime_layout=None,
    )

    with (
        mock.patch.object(
            welmv4,
            "welm_embeddings",
            return_value=torch.zeros((local_rows, 4), dtype=torch.bfloat16),
        ),
        mock.patch.object(welmv4, "welm_use_previous_precision", return_value=False),
        mock.patch.object(welmv4, "is_dp_attention_enabled", return_value=True),
        mock.patch.object(welmv4, "_welm_cuda_graph_capture_active", return_value=True),
        mock.patch.object(
            welmv4, "_welm_should_contract_kv_mirror", return_value=False
        ),
        mock.patch.object(welmv4, "_set_welm_kv_mirror_states"),
        mock.patch.object(
            welmv4,
            "get_global_expert_distribution_recorder",
            return_value=SimpleNamespace(
                with_current_layer=lambda _layer: nullcontext()
            ),
        ),
        mock.patch(
            "sglang.srt.layers.dp_attention.get_attention_dp_rank",
            return_value=rank,
        ),
        mock.patch("sglang.srt.layers.dp_attention.set_dp_buffer_len"),
        mock.patch("sglang.srt.layers.dp_attention.set_is_extend_in_batch"),
    ):
        output = model(
            torch.arange(local_rows, dtype=torch.int64),
            torch.arange(local_rows, dtype=torch.int64),
            forward_batch,
        )

    return {
        "trace": trace,
        "output_rows": int(output.shape[0]),
        "output_value": None if output.numel() == 0 else float(output[0, 0]),
        "global_counts": forward_batch.global_num_tokens_cpu,
        "global_logprob_counts": forward_batch.global_num_tokens_for_logprob_cpu,
        "global_buffer_len": forward_batch.global_dp_buffer_len,
    }


def _run_rank(rank: int, port: int, result_queue) -> None:
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=30),
        )
        mixed = _run_case(rank, [True, False])
        dist.barrier()
        idle_peer = _run_case(rank, [True, False], nondeferred_rows=0)
        dist.barrier()
        all_deferred = _run_case(rank, [True, True])
        dist.barrier()
        result_queue.put(
            (
                rank,
                "ok",
                {"mixed": mixed, "idle_peer": idle_peer, "all": all_deferred},
            )
        )
    except Exception as exc:  # pragma: no cover - subprocess diagnostics
        result_queue.put((rank, "error", repr(exc)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_deferred_dp_suffix_collectives_complete_without_deadlock():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_run_rank, args=(rank, port, result_queue))
        for rank in range(_WORLD_SIZE)
    ]
    for process in processes:
        process.start()

    try:
        results = {}
        for _ in range(_WORLD_SIZE):
            rank, status, payload = result_queue.get(timeout=60)
            assert status == "ok", f"rank {rank} failed: {payload}"
            results[rank] = payload
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

    assert results[0]["mixed"] == {
        "trace": [(0, 2, False), (1, 2, False), (2, 0, True), (3, 0, True)],
        "output_rows": 0,
        "output_value": None,
        "global_counts": [0, 3],
        "global_logprob_counts": [0, 3],
        "global_buffer_len": 3,
    }
    assert results[1]["mixed"] == {
        "trace": [(0, 3, False), (1, 3, False), (2, 3, True), (3, 3, True)],
        "output_rows": 3,
        "output_value": 10.0,
        "global_counts": [0, 3],
        "global_logprob_counts": [0, 3],
        "global_buffer_len": 3,
    }
    for rank in range(_WORLD_SIZE):
        assert results[rank]["idle_peer"] == {
            "trace": [
                (0, 2 if rank == 0 else 0, False),
                (1, 2 if rank == 0 else 0, False),
            ],
            "output_rows": 0,
            "output_value": None,
            "global_counts": [0, 0],
            "global_logprob_counts": [0, 0],
            "global_buffer_len": 0,
        }
        assert results[rank]["all"] == {
            "trace": [(0, 2, False), (1, 2, False)],
            "output_rows": 0,
            "output_value": None,
            "global_counts": [0, 0],
            "global_logprob_counts": [0, 0],
            "global_buffer_len": 0,
        }
