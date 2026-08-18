"""Inference-only WeLMV4 NextN Speculative Decoding."""

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

import sglang.srt.models.welmv4 as welmv4_module
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import (
    get_attention_dp_rank,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.moe.mk_moe_router import (
    MkMoeRouterMode,
    WelmV45_80A3MkMoeRouterAdapter,
    get_mk_moe_router_mode,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.welm_perf_opt import (
    compute_welm_local_embedding,
    compute_welm_oe_embedding,
)
from sglang.srt.models.welmv4 import (
    Qwen2MoeDecoderLayer,
    Qwen2MoeSparseMoeBlock,
    _welm_token_owner_enabled_for_forward,
    _welm_token_owner_physical_rows,
    WelmV4FusedRMSNorm,
    WeLMV4MoeForCausalLM,
    _get_welm_kv_mirror_states,
    _welm_idle_has_contracting_peer,
    _welm_init_kv_mirror_last_q_indices,
    _welm_prepare_kv_mirror_logits_states,
    _welm_scatter_kv_mirror_rows,
    _welm_select_kv_mirror_rows,
    _welm_should_contract_kv_mirror,
    _welm_token_owner_enabled,
    _welm_update_contracted_dp_metadata,
    welm_nextn_local_to_hf_name,
    welm_use_previous_precision,
)
from sglang.srt.models.welmv4_token_owner import (
    TokenOwnerTransition,
    WeLMTokenOwnerRuntime,
)
from sglang.srt.server_args import (
    get_global_server_args,
    get_welm_decode_token_owner_enabled,
)
from sglang.srt.utils import add_prefix, is_cuda, is_npu

logger = logging.getLogger(__name__)


def _welm_mtp_trace(message: str) -> None:
    if os.environ.get("SGLANG_WELM_MTP_TRACE", "0") == "1":
        print(f"[WELM_MTP_TRACE pid={os.getpid()}] {message}", flush=True)


_is_cuda = is_cuda()
_is_npu = is_npu()
_MTP_DUMP_PASS = 0
_MTP_DUMP_PASS_ACTIVE = False
_MTP_GRAPH_DUMP_PASS_ACTIVE = False
_MTP_DUMP_WRITTEN = set()
_MTP_GRAPH_DUMP_CALL_INDEX = 0
_MTP_GRAPH_DUMP_CURRENT_PREFIX = None
_MTP_GRAPH_DUMP_PREV_NAME_PREFIX = ""
_MTP_DUMP_CONTEXT_ENV = "SGLANG_DUMP_MTP_ACTIVATIONS_CONTEXT"
_MTP_TRUE_ENV_VALUES = {"1", "true", "on", "yes"}
_MTP_FUSE_EMBEDDING_ALLREDUCE_ENV = "SGLANG_WELM_MTP_FUSE_EMBEDDING_ALLREDUCE"
_MTP_DUMP_ENABLED = (
    os.environ.get("SGLANG_DUMP_MTP_ACTIVATIONS", "0").strip().lower()
    in _MTP_TRUE_ENV_VALUES
)


def _welm_prepare_mtp_pruned_logits_metadata(
    forward_batch: ForwardBatch,
    local_num_tokens: int,
) -> LogitsMetadata:
    """Build request-row logits metadata without changing the layer layout."""
    metadata = LogitsMetadata.from_forward_batch(forward_batch)
    metadata.welm_kv_mirror_contracted = True
    if not is_dp_attention_enabled():
        if getattr(
            forward_batch,
            "welm_token_owner_enabled",
            get_global_server_args().enable_token_owner,
        ):
            forward_batch._welm_mtp_contracted_dp_metadata_rows = int(
                local_num_tokens
            )
        return metadata

    counts = getattr(
        forward_batch, "_welm_mtp_contract_global_num_tokens_cpu", None
    )
    gpu_counts = metadata.global_num_tokens_for_logprob_gpu
    if counts is None or gpu_counts is None or len(counts) != gpu_counts.numel():
        raise RuntimeError("Missing DP row counts for pruned WeLM MTP logits.")
    if gpu_counts is metadata.global_num_tokens_gpu:
        raise RuntimeError("WeLM MTP layer and logits DP counts must not alias.")
    counts = [int(count) for count in counts]
    dp_rank = get_attention_dp_rank()
    if counts[dp_rank] != int(local_num_tokens):
        raise RuntimeError(
            "WeLM MTP logits row-count mismatch: "
            f"local_rows={local_num_tokens}, dp_rank={dp_rank}, counts={counts}."
        )

    if all(count == counts[0] for count in counts):
        gpu_counts.fill_(counts[0])
    else:
        for index, count in enumerate(counts):
            gpu_counts[index] = count
    metadata.global_num_tokens_gpu = gpu_counts
    metadata.global_num_tokens_for_logprob_cpu = counts
    metadata.global_dp_buffer_len = sum(counts)
    metadata.dp_local_start_pos = None
    metadata.dp_local_num_tokens = None
    forward_batch._welm_mtp_contracted_dp_metadata_rows = int(local_num_tokens)
    return metadata


def _reset_mtp_graph_dump_call_index() -> None:
    if not _MTP_DUMP_ENABLED:
        return
    global _MTP_GRAPH_DUMP_CALL_INDEX, _MTP_GRAPH_DUMP_CURRENT_PREFIX
    global _MTP_GRAPH_DUMP_PREV_NAME_PREFIX
    _MTP_GRAPH_DUMP_CALL_INDEX = 0
    _MTP_GRAPH_DUMP_CURRENT_PREFIX = None
    _MTP_GRAPH_DUMP_PREV_NAME_PREFIX = ""
    welmv4_module._welm_set_graph_dump_name_prefix(None)


def _cuda_graph_capture_active() -> bool:
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if get_is_capture_mode():
            return True
    except Exception:
        pass
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def _mtp_dump_enabled() -> bool:
    return _MTP_DUMP_ENABLED


def _mtp_dump_dir() -> Path:
    root = os.environ.get("SGLANG_DUMP_MTP_ACTIVATIONS_DIR", "./sglang_mtp_dump")
    rank = os.environ.get("RANK", "0")
    path = Path(root) / f"Rank{rank}_pid{os.getpid()}" / f"Pass{_MTP_DUMP_PASS:05d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _current_mtp_dump_context() -> str:
    return os.environ.get(_MTP_DUMP_CONTEXT_ENV, "").strip()


def _write_mtp_pass_metadata(
    dump_dir: Path,
    *,
    pass_id: int,
    context: str,
    graph_step: Optional[str] = None,
    first_dim_limit: Optional[int] = None,
) -> None:
    if not _MTP_DUMP_ENABLED:
        return
    metadata = {
        "pass_id": pass_id,
        "context": context,
        "graph_step": graph_step,
        "first_dim_limit": first_dim_limit,
        "pid": os.getpid(),
        "rank": os.environ.get("RANK", "0"),
    }
    torch.save(metadata, dump_dir / "mtp_pass_metadata.pt")


def _start_mtp_dump_pass() -> None:
    if not _MTP_DUMP_ENABLED:
        return
    global _MTP_DUMP_PASS_ACTIVE, _MTP_GRAPH_DUMP_PASS_ACTIVE
    global _MTP_GRAPH_DUMP_CALL_INDEX, _MTP_GRAPH_DUMP_CURRENT_PREFIX
    global _MTP_GRAPH_DUMP_PREV_NAME_PREFIX
    capture_active = _cuda_graph_capture_active()
    _MTP_DUMP_PASS_ACTIVE = not capture_active
    _MTP_GRAPH_DUMP_PASS_ACTIVE = capture_active
    if _MTP_GRAPH_DUMP_PASS_ACTIVE:
        _MTP_GRAPH_DUMP_CURRENT_PREFIX = f"graph_step_{_MTP_GRAPH_DUMP_CALL_INDEX:05d}."
        _MTP_GRAPH_DUMP_CALL_INDEX += 1
        _MTP_GRAPH_DUMP_PREV_NAME_PREFIX = (
            welmv4_module._welm_set_graph_dump_name_prefix(
                _MTP_GRAPH_DUMP_CURRENT_PREFIX
            )
        )
    else:
        _MTP_GRAPH_DUMP_CURRENT_PREFIX = None
        _MTP_GRAPH_DUMP_PREV_NAME_PREFIX = ""
    if _MTP_DUMP_PASS_ACTIVE or _MTP_GRAPH_DUMP_PASS_ACTIVE:
        _MTP_DUMP_WRITTEN.clear()
        dump_dir = _mtp_dump_dir() if _MTP_DUMP_PASS_ACTIVE else None
        if dump_dir is not None:
            _write_mtp_pass_metadata(
                dump_dir,
                pass_id=_MTP_DUMP_PASS,
                context=_current_mtp_dump_context(),
            )
            os.environ["SGLANG_DUMP_MTP_ACTIVATIONS_PROCESS_DIR"] = str(dump_dir)


def _finish_mtp_dump_pass() -> None:
    if not _MTP_DUMP_ENABLED:
        return
    global _MTP_DUMP_PASS, _MTP_DUMP_PASS_ACTIVE, _MTP_GRAPH_DUMP_PASS_ACTIVE
    global _MTP_GRAPH_DUMP_CURRENT_PREFIX
    global _MTP_GRAPH_DUMP_PREV_NAME_PREFIX
    if _MTP_DUMP_PASS_ACTIVE or _MTP_GRAPH_DUMP_PASS_ACTIVE:
        if _MTP_GRAPH_DUMP_PASS_ACTIVE:
            welmv4_module._welm_set_graph_dump_name_prefix(
                _MTP_GRAPH_DUMP_PREV_NAME_PREFIX
            )
        if _MTP_DUMP_PASS_ACTIVE:
            _MTP_DUMP_PASS += 1
        _MTP_DUMP_WRITTEN.clear()
        os.environ.pop("SGLANG_DUMP_MTP_ACTIVATIONS_PROCESS_DIR", None)
    _MTP_DUMP_PASS_ACTIVE = False
    _MTP_GRAPH_DUMP_PASS_ACTIVE = False
    _MTP_GRAPH_DUMP_CURRENT_PREFIX = None
    _MTP_GRAPH_DUMP_PREV_NAME_PREFIX = ""


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _dump_tensor(name: str, value) -> None:
    if not _MTP_DUMP_ENABLED:
        return
    if isinstance(value, torch.Tensor):
        safe_name = _safe_name(name)
        capture_active = _cuda_graph_capture_active()
        if not capture_active and name in _MTP_DUMP_WRITTEN:
            return
        if capture_active:
            welmv4_module._welm_graph_dump_tensor(safe_name, value)
            return
        torch.save(value.detach().cpu(), _mtp_dump_dir() / f"{safe_name}.pt")
        _MTP_DUMP_WRITTEN.add(name)


def _flush_mtp_graph_dump_pass(
    context: str,
    first_dim_limit: Optional[int] = None,
) -> list[Path]:
    global _MTP_DUMP_PASS
    if not _MTP_DUMP_ENABLED:
        return []
    dump_dir = _mtp_dump_dir()
    saved = welmv4_module._welm_flush_graph_dump_buffers(
        context, dump_dir, first_dim_limit=first_dim_limit
    )
    if not saved:
        return []

    graph_step_files = []
    for path in dump_dir.glob("graph_step_*.pt"):
        stem = path.name[: -len(".pt")]
        prefix, sep, rest = stem.partition(".")
        if not sep or not prefix.startswith("graph_step_") or not rest:
            continue
        graph_step_files.append((prefix, rest, path))

    if not graph_step_files:
        _write_mtp_pass_metadata(
            dump_dir,
            pass_id=_MTP_DUMP_PASS,
            context=context,
            first_dim_limit=first_dim_limit,
        )
        _MTP_DUMP_PASS += 1
        return [dump_dir]

    written_dirs = []
    step_order = {
        step: idx for idx, step in enumerate(sorted({x[0] for x in graph_step_files}))
    }
    for step, rest, path in graph_step_files:
        pass_id = _MTP_DUMP_PASS + step_order[step]
        step_dir = dump_dir.parent / f"Pass{pass_id:05d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        if step_dir not in written_dirs:
            written_dirs.append(step_dir)
        path.replace(step_dir / f"{rest}.pt")
        _write_mtp_pass_metadata(
            step_dir,
            pass_id=pass_id,
            context=context,
            graph_step=step,
            first_dim_limit=first_dim_limit,
        )
    _MTP_DUMP_PASS += len(step_order)
    return written_dirs


class MTPProjector(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True)
        self.ln_f = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if welm_use_previous_precision()
            else WelmV4FusedRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        )


class WeLMV4ModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        token_owner_runtime: Optional[WeLMTokenOwnerRuntime] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_physical_mtp_layers = int(config.num_nextn_predict_layers)
        self.num_nextn_predict_layers = self.num_physical_mtp_layers
        self.token_owner_runtime = token_owner_runtime
        self.decode_token_owner_enabled = bool(
            token_owner_runtime is not None
            and get_welm_decode_token_owner_enabled(get_global_server_args())
        )

        self.embed_tokens = None
        self.oe_embed = None
        self.oe_gate_up_proj = None
        self.oe_dim = config.oe_dim
        self.oe_grams = config.oe_grams
        self.oe_vocab_sizes = config.oe_vocab_sizes
        shared_embeddings_enabled = (
            get_global_server_args().welm_shared_embedding_policy != "disabled"
        )
        self.fuse_embedding_allreduce = (
            os.environ.get(_MTP_FUSE_EMBEDDING_ALLREDUCE_ENV, "0").strip().lower()
            in _MTP_TRUE_ENV_VALUES
        )

        if len(self.oe_vocab_sizes) > 0:
            if shared_embeddings_enabled and get_pp_group().is_first_rank:
                from sglang.srt.layers.full_vocab_shared_embedding import (
                    FullVocabSharedEmbedding,
                )

                self.oe_embed = nn.ModuleList(
                    [
                        FullVocabSharedEmbedding(
                            key=f"model.oe_embed.{i}.weight",
                            num_embeddings=self.oe_vocab_sizes[i],
                            embedding_dim=self.oe_dim,
                        )
                        for i in range(len(self.oe_vocab_sizes))
                    ]
                )
            elif not shared_embeddings_enabled:
                self.oe_embed = nn.ModuleList(
                    [
                        VocabParallelEmbedding(
                            self.oe_vocab_sizes[i],
                            self.oe_dim,
                            use_attn_tp_group=is_dp_attention_enabled(),
                            padding_size=get_global_server_args().welm_vocab_padding_size,
                        )
                        for i in range(len(self.oe_vocab_sizes))
                    ]
                )
            self.oe_gate_up_proj = ReplicatedLinear(
                self.oe_dim * len(self.oe_vocab_sizes),
                config.hidden_size,
                bias=False,
                quant_config=None,
            )

        self.projectors = nn.ModuleList(
            [MTPProjector(config) for _ in range(self.num_physical_mtp_layers)]
        )
        self.alt_stream = torch.cuda.Stream() if _is_cuda else None

        layer_name = "decoder"
        if _is_npu and (
            get_global_server_args().speculative_draft_model_path
            == get_global_server_args().model_path
        ):
            layer_name = "layers." + str(config.num_hidden_layers)

        self.decoder_layers = nn.ModuleList(
            [
                Qwen2MoeDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    is_nextn=True,
                    prefix=add_prefix(layer_name, prefix),
                    alt_stream=self.alt_stream,
                    token_owner_runtime=token_owner_runtime,
                )
                for i in range(self.num_physical_mtp_layers)
            ]
        )
        self.mk_moe_router = None
        mk_moe_router_mode = get_mk_moe_router_mode()
        if mk_moe_router_mode is not MkMoeRouterMode.OFF:
            moe_blocks = [
                layer.mlp
                for layer in self.decoder_layers
                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
            ]
            # A NextN forward executes one physical MTP layer, so every layer
            # can reuse the same workspace slot after begin_forward resets it.
            self.mk_moe_router = WelmV45_80A3MkMoeRouterAdapter(
                mode=mk_moe_router_mode,
                config=config,
                server_args=get_global_server_args(),
                slot_count=1,
                use_previous_precision=welm_use_previous_precision(),
                use_mxfp8=any(block.use_mxfp8 for block in moe_blocks),
            )
            for block in moe_blocks:
                block.set_mk_moe_router(self.mk_moe_router, 0)
        if self.fuse_embedding_allreduce:
            logger.info(
                "%s=1: MTP base-token and OE partials will share one TP "
                "all-reduce per draft step.",
                _MTP_FUSE_EMBEDDING_ALLREDUCE_ENV,
            )

    def _can_fuse_embedding_allreduce(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> bool:
        return (
            self.fuse_embedding_allreduce
            and not welm_use_previous_precision()
            and len(self.oe_grams) > 0
            and self.oe_embed is not None
            and self.oe_gate_up_proj is not None
            and input_ids.numel() > 0
            and not forward_batch.forward_mode.is_idle()
        )

    def _apply_base_embedding(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, bool]:
        """Compute base embedding and report whether it is still TP-local."""

        fuse_allreduce = self._can_fuse_embedding_allreduce(input_ids, forward_batch)
        if fuse_allreduce:
            return compute_welm_local_embedding(self.embed_tokens, input_ids), True
        return self.embed_tokens(input_ids), False

    def _get_physical_step_idx(self, step_idx: int) -> int:
        if step_idx < 0:
            raise RuntimeError(
                f"WeLM MTP logical step index must be non-negative, got {step_idx}."
            )
        if self.num_physical_mtp_layers <= 0:
            raise RuntimeError(
                "WeLM MTP requires at least one physical MTP layer, got "
                f"{self.num_physical_mtp_layers}."
            )
        if (
            self.num_physical_mtp_layers != 1
            and step_idx >= self.num_physical_mtp_layers
        ):
            raise RuntimeError(
                "WeLM MTP logical step index is out of physical layer range: "
                f"{step_idx} vs {self.num_physical_mtp_layers}."
            )
        return step_idx % self.num_physical_mtp_layers

    def _get_projector(self, step_idx: int) -> MTPProjector:
        return self.projectors[self._get_physical_step_idx(step_idx)]

    def _apply_oe_embedding(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        hashed_inputs: Optional[torch.Tensor] = None,
        base_hidden_is_local: bool = False,
    ) -> torch.Tensor:
        if (
            len(self.oe_grams) == 0
            or input_ids.numel() == 0
            or forward_batch.forward_mode.is_idle()
        ):
            if base_hidden_is_local:
                raise RuntimeError(
                    "A TP-local WeLM base embedding cannot bypass OE fusion."
                )
            return hidden_states

        if hashed_inputs is None:
            if getattr(forward_batch, "welm_oe_decode_hashed_inputs", None) is None:
                raise RuntimeError("WeLMV4 MTP OE requires cached CUDA hash inputs.")
            return compute_welm_oe_embedding(
                input_ids=input_ids,
                forward_batch=forward_batch,
                base_hidden_states=hidden_states,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=self.oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=self.oe_embed,
                oe_proj_module=self.oe_gate_up_proj,
                fuse_base_embedding_allreduce=base_hidden_is_local,
            )

        from sglang.srt.speculative.welm_mtp_draft_ngram_hash import (
            launch_deferred_welm_mtp_draft_ngram_hash,
        )

        launch_deferred_welm_mtp_draft_ngram_hash(forward_batch)
        had_attr = hasattr(forward_batch, "welm_oe_decode_hashed_inputs")
        previous = getattr(forward_batch, "welm_oe_decode_hashed_inputs", None)
        forward_batch.welm_oe_decode_hashed_inputs = hashed_inputs
        try:
            return compute_welm_oe_embedding(
                input_ids=input_ids,
                forward_batch=forward_batch,
                base_hidden_states=hidden_states,
                oe_grams=self.oe_grams,
                oe_vocab_sizes=self.oe_vocab_sizes,
                vocab_size=self.vocab_size,
                oe_embed_modules=self.oe_embed,
                oe_proj_module=self.oe_gate_up_proj,
                fuse_base_embedding_allreduce=base_hidden_is_local,
            )
        finally:
            if had_attr:
                forward_batch.welm_oe_decode_hashed_inputs = previous
            else:
                delattr(forward_batch, "welm_oe_decode_hashed_inputs")

    def _build_merged_query_embedding_only(
        self,
        forward_batch: ForwardBatch,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not getattr(forward_batch, "welm_mtp_merge_kv_fill_draft", False):
            return None
        if not _welm_should_contract_kv_mirror(forward_batch):
            return None

        _welm_init_kv_mirror_last_q_indices(forward_batch)
        query_input_ids = getattr(forward_batch, "welm_mtp_query_input_ids", None)
        if query_input_ids is None:
            raise RuntimeError("Missing merged WeLMV4 MTP query input ids.")
        query_input_ids = query_input_ids.to(device=device, dtype=torch.int64)

        active_indices = getattr(forward_batch, "kv_mirror_active_batch_indices", None)
        output_size = getattr(forward_batch, "kv_mirror_output_size", None)
        if (
            active_indices is not None
            and output_size is not None
            and query_input_ids.shape[0] == output_size
            and active_indices.numel() != output_size
        ):
            query_input_ids = query_input_ids[active_indices]

        expected_rows = forward_batch.custom_last_index.numel()
        if query_input_ids.shape[0] != expected_rows:
            raise RuntimeError(
                "Merged WeLMV4 MTP query input shape does not match active "
                f"last-query rows: {query_input_ids.shape[0]} vs {expected_rows}."
            )

        query_hidden, query_hidden_is_local = self._apply_base_embedding(
            query_input_ids, forward_batch
        )
        query_hashed_inputs = getattr(
            forward_batch, "welm_mtp_query_oe_hashed_inputs", None
        )
        if (
            query_hashed_inputs is not None
            and active_indices is not None
            and output_size is not None
            and query_hashed_inputs.shape[1] == output_size
            and active_indices.numel() != output_size
        ):
            query_hashed_inputs = query_hashed_inputs[:, active_indices]
        if query_hashed_inputs is not None and query_hashed_inputs.shape[1] != int(
            query_input_ids.numel()
        ):
            raise RuntimeError(
                "Merged WeLMV4 MTP query OE hash shape mismatch: "
                f"{query_hashed_inputs.shape[1]} vs {query_input_ids.numel()}."
            )
        return self._apply_oe_embedding(
            query_input_ids,
            query_hidden,
            forward_batch,
            hashed_inputs=query_hashed_inputs,
            base_hidden_is_local=query_hidden_is_local,
        )

    def _maybe_override_merged_query_embedding(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not getattr(forward_batch, "welm_mtp_merge_kv_fill_draft", False):
            return hidden_states
        if _welm_should_contract_kv_mirror(forward_batch):
            return hidden_states

        query_input_ids = getattr(forward_batch, "welm_mtp_query_input_ids", None)
        if query_input_ids is None:
            raise RuntimeError("Missing merged WeLMV4 MTP query input ids.")
        query_input_ids = query_input_ids.to(
            device=hidden_states.device, dtype=torch.int64
        )
        custom_last_index = getattr(forward_batch, "custom_last_index", None)
        if custom_last_index is None:
            if forward_batch.extend_seq_lens is None:
                if (
                    forward_batch.forward_mode.is_idle()
                    and hidden_states.shape[0] == 0
                    and query_input_ids.numel() == 0
                ):
                    custom_last_index = torch.empty(
                        (0,), dtype=torch.long, device=hidden_states.device
                    )
                    forward_batch.custom_last_index = custom_last_index
                else:
                    raise RuntimeError(
                        "Merged WeLMV4 MTP query embedding override requires "
                        "custom_last_index or extend_seq_lens."
                    )
            else:
                custom_last_index = (
                    torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                )
                forward_batch.custom_last_index = custom_last_index
                out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
                if out_cache_loc is not None:
                    forward_batch.custom_last_cache_loc = out_cache_loc[
                        custom_last_index
                    ]

        expected_rows = custom_last_index.numel()
        if query_input_ids.shape[0] != expected_rows:
            raise RuntimeError(
                "Merged WeLMV4 MTP query input shape does not match active "
                f"last-query rows: {query_input_ids.shape[0]} vs {expected_rows}."
            )

        query_hidden, query_hidden_is_local = self._apply_base_embedding(
            query_input_ids, forward_batch
        )
        query_hidden = self._apply_oe_embedding(
            query_input_ids,
            query_hidden,
            forward_batch,
            hashed_inputs=getattr(
                forward_batch, "welm_mtp_query_oe_hashed_inputs", None
            ),
            base_hidden_is_local=query_hidden_is_local,
        )
        hidden_states = hidden_states.clone()
        hidden_states[custom_last_index] = query_hidden.to(hidden_states.dtype)
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        forward_batch.welm_token_owner_enabled = bool(
            self.token_owner_runtime is not None
            and _welm_token_owner_enabled_for_forward(
                forward_batch,
                decode_token_owner_enabled=self.decode_token_owner_enabled,
            )
        )
        mtp_step_idx = int(getattr(forward_batch, "mtp_step_idx", 0))
        if not getattr(forward_batch, "kv_fill_only", False) and mtp_step_idx == 0:
            _start_mtp_dump_pass()

        _dump_tensor(f"model.mtp.{mtp_step_idx}.input_ids", input_ids)
        _dump_tensor(f"model.mtp.{mtp_step_idx}.positions", positions)
        _dump_tensor(
            f"model.mtp.{mtp_step_idx}.main_hidden_in",
            forward_batch.spec_info.hidden_states,
        )

        hidden_states = self._build_merged_query_embedding_only(
            forward_batch, input_ids.device
        )
        needs_merged_query_override = hidden_states is None
        if hidden_states is None:
            base_hidden_is_local = False
            if input_embeds is None:
                hidden_states, base_hidden_is_local = self._apply_base_embedding(
                    input_ids, forward_batch
                )
            else:
                hidden_states = input_embeds
            hidden_states = self._apply_oe_embedding(
                input_ids,
                hidden_states,
                forward_batch,
                base_hidden_is_local=base_hidden_is_local,
            )
        if hidden_states is None:
            raise RuntimeError("WeLM MTP embedding path returned no hidden states")
        if needs_merged_query_override:
            hidden_states = self._maybe_override_merged_query_embedding(
                hidden_states, forward_batch
            )
        _dump_tensor(f"model.mtp.{mtp_step_idx}.embedding", hidden_states)

        main_hidden_states = forward_batch.spec_info.hidden_states
        if (
            _welm_should_contract_kv_mirror(forward_batch)
            and main_hidden_states is not None
        ):
            _welm_init_kv_mirror_last_q_indices(forward_batch)
            first_contract = (
                hidden_states.shape[0] != forward_batch.kv_mirror_output_size
            )
            hidden_states = _welm_select_kv_mirror_rows(
                hidden_states, forward_batch, first_contract=first_contract
            )
            # Zero-pad the selected rows back to the attn_tp-aligned output
            # domain (kv_mirror_output_size), matching the main model; the
            # per-layer attn-TP collectives below require aligned row counts.
            hidden_states = _welm_scatter_kv_mirror_rows(hidden_states, forward_batch)
            main_first_contract = (
                main_hidden_states.shape[0] != forward_batch.kv_mirror_output_size
            )
            main_hidden_states = _welm_select_kv_mirror_rows(
                main_hidden_states,
                forward_batch,
                first_contract=main_first_contract,
            )
            main_hidden_states = _welm_scatter_kv_mirror_rows(
                main_hidden_states, forward_batch
            )
            _welm_update_contracted_dp_metadata(
                forward_batch,
                hidden_states.shape[0],
                marker_attr="_welm_mtp_contracted_dp_metadata_rows",
                contract_to_request_counts=True,
                logprob_local_num_tokens=max(
                    hidden_states.shape[0]
                    - getattr(forward_batch, "_welm_kv_mirror_row_pad", 0),
                    0,
                ),
            )
        elif (
            forward_batch.forward_mode.is_idle()
            and forward_batch.is_extend_in_batch
            and main_hidden_states is not None
            and _welm_idle_has_contracting_peer(forward_batch)
        ):
            # Contract at model entry only when non-idle peers contract at
            # entry too (KV-mirror prefill rounds). In merged draft-extend
            # rounds the decode ranks contract after the layer, so an idle
            # rank contracting here would size the layer collectives with
            # request counts while its peers still use the padded counts.
            _welm_update_contracted_dp_metadata(
                forward_batch,
                hidden_states.shape[0],
                marker_attr="_welm_mtp_contracted_dp_metadata_rows",
                contract_to_request_counts=True,
            )

        if forward_batch.welm_token_owner_enabled:
            self.token_owner_runtime.begin_forward(
                forward_batch,
                single_group_physical_rows=(
                    _welm_token_owner_physical_rows(
                        forward_batch,
                        post_pruning=True,
                    )
                    if not is_dp_attention_enabled()
                    else None
                ),
                transition=TokenOwnerTransition.NONE,
                device=hidden_states.device,
            )

        needs_empty_dp_collectives = (
            is_dp_attention_enabled()
            and getattr(forward_batch, "global_num_tokens_gpu", None) is not None
        )
        if hidden_states.shape[0] == 0 and not needs_empty_dp_collectives:
            return hidden_states, hidden_states

        proj = self._get_projector(mtp_step_idx)
        if hidden_states.shape[0] > 0:
            main_rows = (
                None if main_hidden_states is None else int(main_hidden_states.shape[0])
            )
            if main_rows != int(hidden_states.shape[0]):
                raise RuntimeError(
                    "WeLM MTP projector inputs must share the rank-local row "
                    f"layout: embedding rows={int(hidden_states.shape[0])}, "
                    f"main hidden rows={main_rows}, "
                    f"mode={forward_batch.forward_mode}, "
                    f"mtp_step={mtp_step_idx}, "
                    "global_num_tokens="
                    f"{getattr(forward_batch, 'global_num_tokens_cpu', None)}."
                )
            enorm_output = proj.enorm(hidden_states)
            hnorm_output = proj.hnorm(main_hidden_states)
            _dump_tensor(f"model.mtp.{mtp_step_idx}.enorm", enorm_output)
            _dump_tensor(f"model.mtp.{mtp_step_idx}.hnorm", hnorm_output)
            hidden_states = proj.eh_proj(
                torch.cat((enorm_output, hnorm_output), dim=-1)
            )
            _dump_tensor(f"model.mtp.{mtp_step_idx}.projector_out", hidden_states)

        residual = None
        kv_mirror_states = _get_welm_kv_mirror_states(forward_batch)
        final_experts_output = None
        final_shared_output = None
        physical_step_idx = self._get_physical_step_idx(mtp_step_idx)
        layer = self.decoder_layers[physical_step_idx]
        if self.mk_moe_router is not None:
            self.mk_moe_router.begin_forward(
                forward_batch=forward_batch,
                num_tokens=hidden_states.shape[0],
                allow_fused_router=not _MTP_DUMP_ENABLED,
            )
        _welm_mtp_trace(
            "nextn_before_layer "
            f"step={mtp_step_idx} mode={forward_batch.forward_mode} "
            f"hidden_shape={tuple(hidden_states.shape)} "
            f"main_hidden_shape={None if main_hidden_states is None else tuple(main_hidden_states.shape)} "
            f"global_tokens={getattr(forward_batch, 'global_num_tokens_cpu', None)}"
        )
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states, residual, kv_mirror_states = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                kv_mirror_states,
            )
            final_experts_output = getattr(layer, "final_mlp_experts_output", None)
            final_shared_output = getattr(layer, "final_mlp_shared_output", None)
        _welm_mtp_trace(
            "nextn_after_layer "
            f"step={mtp_step_idx} mode={forward_batch.forward_mode} "
            f"hidden_shape={tuple(hidden_states.shape)} "
            f"residual_shape={None if residual is None else tuple(residual.shape)} "
            f"global_tokens={getattr(forward_batch, 'global_num_tokens_cpu', None)}"
        )

        if hidden_states.shape[0] == 0 and not needs_empty_dp_collectives:
            return hidden_states, hidden_states

        ln_f = proj.ln_f
        hidden_states_for_next_mtp = None
        if not forward_batch.forward_mode.is_idle() or needs_empty_dp_collectives:
            _welm_mtp_trace(
                "nextn_before_ln_f "
                f"step={mtp_step_idx} mode={forward_batch.forward_mode} "
                f"hidden_shape={tuple(hidden_states.shape)} "
                f"residual_shape={None if residual is None else tuple(residual.shape)}"
            )
            if residual is not None:
                if welm_use_previous_precision():
                    hidden_states_for_next_mtp = (
                        hidden_states.float() + residual.float()
                    ).to(ln_f.weight.dtype)
                    hidden_states, _ = ln_f(hidden_states, residual)
                else:
                    # Component outputs are captured before the MLP all-reduce.
                    # Rebuilding from them is only equivalent when there is no
                    # tensor-parallel reduction to apply.
                    can_rebuild_final_mlp = (
                        final_experts_output is not None
                        and getattr(layer.mlp, "tp_size", 1) == 1
                        and not is_dp_attention_enabled()
                    )
                    if can_rebuild_final_mlp:
                        hidden_states = final_experts_output.float() + residual.float()
                        if final_shared_output is not None:
                            hidden_states = hidden_states + final_shared_output.float()
                        hidden_states_for_next_mtp = hidden_states.to(ln_f.weight.dtype)
                        hidden_states, _ = ln_f(hidden_states_for_next_mtp)
                    else:
                        hidden_states, hidden_states_for_next_mtp = ln_f(
                            hidden_states, residual, output_dtype=ln_f.weight.dtype
                        )
                        if hidden_states_for_next_mtp.dtype != ln_f.weight.dtype:
                            hidden_states_for_next_mtp = hidden_states_for_next_mtp.to(
                                ln_f.weight.dtype
                            )
            else:
                hidden_states_for_next_mtp = hidden_states.to(ln_f.weight.dtype)
                norm_output = ln_f(hidden_states_for_next_mtp)
                hidden_states = (
                    norm_output[0] if isinstance(norm_output, tuple) else norm_output
                )
            _dump_tensor(
                f"model.mtp.{mtp_step_idx}.decoder.output",
                hidden_states_for_next_mtp,
            )
            _welm_mtp_trace(
                "nextn_after_ln_f "
                f"step={mtp_step_idx} mode={forward_batch.forward_mode} "
                f"hidden_shape={tuple(hidden_states.shape)} "
                f"next_hidden_shape={None if hidden_states_for_next_mtp is None else tuple(hidden_states_for_next_mtp.shape)}"
            )
        _dump_tensor(f"model.mtp.{mtp_step_idx}.ln_f", hidden_states)
        return hidden_states, hidden_states_for_next_mtp


class WeLMV4MoeForCausalLMNextN(WeLMV4MoeForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        # if not set, model load will be broken in DeepseekV3ForCausalLM load_weights()
        self.pp_group = get_pp_group()
        enable_token_owner = _welm_token_owner_enabled(pp_size=self.pp_group.world_size)
        self.token_owner_runtime = (
            WeLMTokenOwnerRuntime() if enable_token_owner else None
        )

        self.model = WeLMV4ModelNextN(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            token_owner_runtime=self.token_owner_runtime,
        )
        self.lm_head = None
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        mtp_step_idx = getattr(forward_batch, "mtp_step_idx", 0)
        started_dump = (
            _MTP_DUMP_ENABLED
            and not getattr(forward_batch, "kv_fill_only", False)
            and mtp_step_idx == 0
        )
        try:
            hidden_states, hidden_states_for_next_mtp = self.model(
                input_ids, positions, forward_batch
            )
            aux_hidden_states = (
                [hidden_states_for_next_mtp]
                if hidden_states_for_next_mtp is not None
                else None
            )
            logits_metadata = forward_batch
            _welm_mtp_trace(
                "wrapper_branch "
                f"contract={_welm_should_contract_kv_mirror(forward_batch)} "
                f"flags={getattr(forward_batch, 'welm_kv_mirror_contract_flags', None)} "
                f"local_flag={getattr(forward_batch, '_welm_local_kv_mirror_contract_enabled', None)} "
                f"mode={forward_batch.forward_mode} "
                f"hidden={tuple(hidden_states.shape)} "
                f"custom_last={None if getattr(forward_batch, 'custom_last_index', None) is None else int(forward_batch.custom_last_index.numel())} "
                f"merge={getattr(forward_batch, 'welm_mtp_merge_kv_fill_draft', False)} "
                f"contracted={getattr(forward_batch, 'welm_kv_mirror_contracted', False)}"
            )
            if _welm_should_contract_kv_mirror(forward_batch):
                hidden_states, aux_hidden_states = (
                    _welm_prepare_kv_mirror_logits_states(
                        hidden_states, aux_hidden_states, forward_batch
                    )
                )
                # Strip the attn-TP alignment pad rows before logits, like the
                # main model; the next MTP step re-pads at its entry. Token
                # counts stay in the padded domain (attn-TP collectives need
                # the alignment) -- the entry metadata update already set the
                # un-padded logprob counts for the logits gather.
                row_pad = getattr(forward_batch, "_welm_kv_mirror_row_pad", 0)
                if (
                    row_pad > 0
                    and hidden_states.shape[0] == forward_batch.kv_mirror_output_size
                ):
                    num_real_rows = forward_batch.kv_mirror_output_size - row_pad
                    hidden_states = hidden_states[:num_real_rows]
                    if aux_hidden_states is not None:
                        aux_hidden_states = [
                            hidden[:num_real_rows] for hidden in aux_hidden_states
                        ]
            elif getattr(forward_batch, "welm_mtp_merge_kv_fill_draft", False):
                custom_last_index = getattr(forward_batch, "custom_last_index", None)
                if custom_last_index is not None and (
                    hidden_states.shape[0] != custom_last_index.numel()
                    # Idle ranks carry zero rows on both sides but must still
                    # contract here, in the same phase as their decode peers,
                    # to keep the logits collectives size-matched.
                    or forward_batch.forward_mode.is_idle()
                ):
                    if aux_hidden_states is None:
                        raise RuntimeError(
                            "Merged WeLM MTP full-layout logits pruning requires "
                            "the full hidden state for the next MTP step."
                        )
                    hidden_states = hidden_states[custom_last_index]
                    logits_metadata = _welm_prepare_mtp_pruned_logits_metadata(
                        forward_batch,
                        hidden_states.shape[0],
                    )
            logits_output = self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                logits_metadata,
                aux_hidden_states,
            )
            if _MTP_DUMP_ENABLED:
                _dump_tensor(
                    f"model.mtp.{mtp_step_idx}.logits",
                    logits_output.next_token_logits,
                )
            return logits_output
        finally:
            if started_dump:
                _finish_mtp_dump_pass()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)

    def get_hf_weight_name(self, local_name: str) -> Optional[str]:
        return welm_nextn_local_to_hf_name(self.config, local_name)


EntryClass = WeLMV4MoeForCausalLMNextN
