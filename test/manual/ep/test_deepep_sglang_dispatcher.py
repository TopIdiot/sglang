"""Hopper BF16 smoke test for SGLang's dispatcher against an installed DeepEP.

Unlike the upstream-style DeepEP tests in this directory, this test exercises
the arguments, output adapters, and event handling used by SGLang itself.

Example (single 8-GPU node):

    WORLD_SIZE=1 RANK=0 python test_deepep_sglang_dispatcher.py
"""

import argparse
import gc
import os

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import set_is_extend_in_batch
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPBuffer,
    DeepEPDispatcher,
    DeepEPLLCombineInput,
    DeepEPNormalCombineInput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import DeepEPMode, MoeRunnerBackend
from sglang.test.test_deepep_utils import calc_diff, init_dist


NUM_PROCESSES = 8
HIDDEN_SIZE = 7168


def _new_dispatcher(
    group,
    *,
    mode,
    num_experts,
    router_topk,
    async_finish=False,
    return_recv_hook=False,
):
    return DeepEPDispatcher(
        group=group,
        router_topk=router_topk,
        num_experts=num_experts,
        num_local_experts=num_experts // group.size(),
        hidden_size=HIDDEN_SIZE,
        params_dtype=torch.bfloat16,
        deepep_mode=mode,
        async_finish=async_finish,
        return_recv_hook=return_recv_hook,
    )


def _test_normal(rank, group):
    num_tokens, num_experts, router_topk = 128, 256, 1
    hidden_states = torch.randn(
        (num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    topk_ids = (
        torch.arange(num_tokens, dtype=torch.int64, device="cuda") * group.size()
        + rank
    ).remainder(num_experts)[:, None]
    topk_weights = torch.ones(
        (num_tokens, router_topk), dtype=torch.float32, device="cuda"
    )
    topk_output = StandardTopKOutput(topk_weights, topk_ids, torch.empty(0))

    # The smoke test does not run a grouped GEMM. Keeping this flag enabled
    # selects the production SGLang combine path and its aligned DeepEP layout.
    deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM = True
    moe_utils.MOE_RUNNER_BACKEND = MoeRunnerBackend.DEEP_GEMM

    for async_finish in (False, True):
        dispatcher = _new_dispatcher(
            group,
            mode=DeepEPMode.NORMAL,
            num_experts=num_experts,
            router_topk=router_topk,
            async_finish=async_finish,
        )
        with envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True):
            dispatched = dispatcher.dispatch(hidden_states, topk_output)
        assert dispatched.hidden_states_scale is None
        combined = dispatcher.combine(
            DeepEPNormalCombineInput(
                dispatched.hidden_states,
                dispatched.topk_ids,
                dispatched.topk_weights,
            )
        )
        diff = calc_diff(combined, hidden_states)
        assert diff < 1e-5, (async_finish, diff)
        if rank == 0:
            print(f"[sglang normal BF16] async={async_finish}: passed", flush=True)


def _test_low_latency(rank, group):
    num_tokens, num_experts, router_topk = 128, 288, 8
    hidden_states = torch.randn(
        (num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    scores = torch.randn(
        (num_tokens, num_experts), dtype=torch.float32, device="cuda"
    )
    topk_ids = torch.topk(scores, router_topk, dim=-1, sorted=True).indices
    topk_weights = torch.rand(
        (num_tokens, router_topk), dtype=torch.float32, device="cuda"
    )
    topk_output = StandardTopKOutput(topk_weights, topk_ids, torch.empty(0))
    expected = hidden_states * topk_weights.sum(dim=1, keepdim=True)

    moe_utils.MOE_RUNNER_BACKEND = MoeRunnerBackend.DEEP_GEMM
    for return_recv_hook in (False, True):
        dispatcher = _new_dispatcher(
            group,
            mode=DeepEPMode.LOW_LATENCY,
            num_experts=num_experts,
            router_topk=router_topk,
            return_recv_hook=return_recv_hook,
        )
        dispatcher.set_quant_config({"bf16_dispatch": True})
        dispatched = dispatcher.dispatch(hidden_states, topk_output)
        assert dispatched.hidden_states_scale is None
        combined = dispatcher.combine(
            DeepEPLLCombineInput(
                dispatched.hidden_states,
                dispatched.topk_ids,
                dispatched.topk_weights,
            )
        )
        diff = calc_diff(combined, expected)
        assert diff < 1e-5, (return_recv_hook, diff)
        if rank == 0:
            print(
                f"[sglang low-latency BF16] recv_hook={return_recv_hook}: passed",
                flush=True,
            )


def _test_auto(rank, group):
    """Exercise the normal-to-low-latency transition on one shared buffer."""
    num_tokens, num_experts, router_topk = 128, 256, 1
    dispatcher = _new_dispatcher(
        group,
        mode=DeepEPMode.AUTO,
        num_experts=num_experts,
        router_topk=router_topk,
        async_finish=True,
        return_recv_hook=True,
    )
    dispatcher.set_quant_config({"bf16_dispatch": True})

    hidden_states = torch.randn(
        (num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    topk_ids = (
        torch.arange(num_tokens, dtype=torch.int64, device="cuda") * group.size()
        + rank
    ).remainder(num_experts)[:, None]
    topk_weights = torch.ones(
        (num_tokens, router_topk), dtype=torch.float32, device="cuda"
    )
    topk_output = StandardTopKOutput(topk_weights, topk_ids, torch.empty(0))

    deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM = True
    moe_utils.MOE_RUNNER_BACKEND = MoeRunnerBackend.DEEP_GEMM
    set_is_extend_in_batch(True)
    with envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True):
        dispatched = dispatcher.dispatch(hidden_states, topk_output)
    combined = dispatcher.combine(
        DeepEPNormalCombineInput(
            dispatched.hidden_states,
            dispatched.topk_ids,
            dispatched.topk_weights,
        )
    )
    assert calc_diff(combined, hidden_states) < 1e-5

    hidden_states = torch.randn(
        (num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    topk_weights = torch.rand(
        (num_tokens, router_topk), dtype=torch.float32, device="cuda"
    )
    topk_output = StandardTopKOutput(topk_weights, topk_ids, torch.empty(0))
    set_is_extend_in_batch(False)
    dispatched = dispatcher.dispatch(hidden_states, topk_output)
    combined = dispatcher.combine(
        DeepEPLLCombineInput(
            dispatched.hidden_states,
            dispatched.topk_ids,
            dispatched.topk_weights,
        )
    )
    expected = hidden_states * topk_weights
    assert calc_diff(combined, expected) < 1e-5
    if rank == 0:
        print("[sglang auto BF16] normal->low-latency: passed", flush=True)


def test_loop(local_rank, num_local_ranks, mode):
    rank, _, group = init_dist(local_rank, num_local_ranks)
    major, _ = torch.cuda.get_device_capability()
    assert major == 9, "This integration smoke test is currently scoped to Hopper"
    torch.manual_seed(rank)
    set_is_extend_in_batch(mode != "low_latency")

    try:
        if mode == "normal":
            _test_normal(rank, group)
        elif mode == "low_latency":
            _test_low_latency(rank, group)
        else:
            _test_auto(rank, group)
        group.barrier()
    finally:
        # Release DeepEP's process-group reference before c10d teardown.
        DeepEPBuffer._buffer = None
        gc.collect()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("normal", "low_latency", "auto", "all"), default="all"
    )
    args = parser.parse_args()

    num_processes = int(os.getenv("DEEPEP_TEST_NUM_PROCESSES", NUM_PROCESSES))
    modes = (
        ("normal", "low_latency", "auto") if args.mode == "all" else (args.mode,)
    )
    base_port = int(os.getenv("MASTER_PORT", "8361"))
    for index, mode in enumerate(modes):
        os.environ["MASTER_PORT"] = str(base_port + index)
        torch.multiprocessing.spawn(
            test_loop, args=(num_processes, mode), nprocs=num_processes
        )


if __name__ == "__main__":
    main()
