from types import SimpleNamespace
from unittest.mock import MagicMock

import torch


def model_config(**hf_overrides):
    values = {
        "architectures": ["WeLMV4MoeForCausalLM"],
        "num_hidden_layers": 48,
        "num_nextn_predict_layers": 2,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "kv_mirror_layers": [48, 49, 47],
        "kv_mirror_imitated_layers": [0, 4, 1],
    }
    values.update(hf_overrides)
    hf_config = SimpleNamespace(**values)
    return SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=hf_config,
        head_dim=hf_config.head_dim,
        dtype=torch.bfloat16,
        get_num_kv_heads=lambda tp_size: max(
            1, hf_config.num_key_value_heads // tp_size
        ),
    )


def worker_args(**overrides):
    values = {
        "speculative_eagle_topk": 1,
        "speculative_num_steps": 3,
        "speculative_num_draft_tokens": 4,
        "speculative_algorithm": "EAGLE",
        "device": "cpu",
        "page_size": 16,
        "disaggregation_mode": "prefill",
        "enable_welm_kv_mirror_opt": True,
        "welm_kv_mirror_pd_mode": "deferred-last-prompt",
        "attn_cp_size": 1,
        "attention_backend": "fa3",
        "prefill_attention_backend": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_target_worker():
    worker = MagicMock()
    worker.model_runner.model_config = model_config()
    worker.model_runner.model_config.context_len = 4096
    worker.model_config = worker.model_runner.model_config
    worker.get_memory_pool.return_value = (MagicMock(), MagicMock())
    return worker


def build_eagle_worker(server_args, target_worker=None):
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

    return EAGLEWorkerV2(
        server_args,
        gpu_id=0,
        tp_rank=0,
        dp_rank=0,
        moe_ep_rank=0,
        attn_cp_rank=0,
        moe_dp_rank=0,
        nccl_port=12345,
        target_worker=target_worker or make_target_worker(),
    )
