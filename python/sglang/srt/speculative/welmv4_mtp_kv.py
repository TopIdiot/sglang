from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs


# Keep direct-pool selection and its temporary mirror-state compatibility path
# out of the generic speculative utilities.
WELMV4_KV_MIRROR_STATES_KEY = "welm_kv_mirror_states"


@dataclass(frozen=True)
class WelmMTPKVMirrorBufferSpec:
    mirror_layers: tuple[int, ...]
    tensor_size: int
    dtype: torch.dtype
    device: torch.device


def is_welmv4_mtp_target_config(model_config) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", ()) or ()
    return (
        tuple(architectures) == ("WeLMV4MoeForCausalLM",)
        and int(getattr(hf_config, "num_nextn_predict_layers", 0) or 0) > 0
    )


def is_welmv4_mtp_pd(server_args, target_model_config) -> bool:
    algorithm = getattr(server_args, "speculative_algorithm", None)
    return (
        getattr(server_args, "disaggregation_mode", "null") in ("prefill", "decode")
        and bool(getattr(server_args, "enable_welm_kv_mirror_opt", False))
        and isinstance(algorithm, str)
        and algorithm.upper() == "EAGLE"
        and is_welmv4_mtp_target_config(target_model_config)
    )


def should_use_welm_mtp_lightweight_prefill(
    server_args,
    target_model_config,
) -> bool:
    return (
        getattr(server_args, "disaggregation_mode", "null") == "prefill"
        and is_welmv4_mtp_pd(server_args, target_model_config)
    )


def should_use_welm_mtp_legacy_mirror_state(
    server_args,
    target_model_config,
) -> bool:
    if not envs.SGLANG_WELM_MTP_LEGACY_MIRROR_STATE.get():
        return False
    if (
        not is_welmv4_mtp_pd(server_args, target_model_config)
        or getattr(server_args, "welm_kv_mirror_pd_mode", "legacy") != "legacy"
    ):
        raise RuntimeError(
            "SGLANG_WELM_MTP_LEGACY_MIRROR_STATE=1 only supports WeLM MTP "
            "P/D legacy mode"
        )
    return True


def should_use_welm_mtp_direct_kv(server_args, target_model_config) -> bool:
    algorithm = getattr(server_args, "speculative_algorithm", None)
    disaggregation_mode = getattr(server_args, "disaggregation_mode", "null")
    legacy_mirror_state = should_use_welm_mtp_legacy_mirror_state(
        server_args, target_model_config
    )
    selected = (
        bool(getattr(server_args, "enable_welm_kv_mirror_opt", False))
        and isinstance(algorithm, str)
        and algorithm.upper() == "EAGLE"
        and is_welmv4_mtp_target_config(target_model_config)
        and disaggregation_mode in ("null", "decode")
        and not legacy_mirror_state
    )
    if not selected:
        return False
    _validate_welm_mtp_direct_kv_args(server_args, target_model_config)
    return True


def should_use_welm_mtp_storage_draft_kv(
    server_args, target_model_config
) -> bool:
    legacy_mirror_state = should_use_welm_mtp_legacy_mirror_state(
        server_args, target_model_config
    )
    selected = (
        should_use_welm_mtp_lightweight_prefill(server_args, target_model_config)
        and not legacy_mirror_state
    )
    if not selected:
        return False
    _validate_welm_mtp_direct_kv_args(server_args, target_model_config)
    return True


def _validate_welm_mtp_direct_kv_args(
    server_args, target_model_config
) -> None:
    from sglang.srt.models.welm_v45_80a3_h2048_hd256_pre_attn_v2_config import (
        welm_v45_80a3_h2048_hd256_pre_attn_v2_enabled,
    )

    if welm_v45_80a3_h2048_hd256_pre_attn_v2_enabled():
        raise RuntimeError(
            "WeLM MTP direct K/V does not support Pre-Attention V2 yet"
        )
    if int(getattr(server_args, "speculative_eagle_topk", 0) or 0) != 1:
        raise RuntimeError("WeLM MTP direct K/V currently requires topk=1")
    if int(getattr(server_args, "attn_cp_size", 1) or 1) != 1:
        raise RuntimeError("WeLM MTP direct K/V does not support AttnCP yet")
    prefill_backend = (
        getattr(server_args, "prefill_attention_backend", None)
        or getattr(server_args, "attention_backend", None)
    )
    if prefill_backend != "fa3":
        raise RuntimeError(
            "WeLM MTP direct K/V requires FA3 as the prefill "
            f"attention backend, got {prefill_backend!r}"
        )
    if bool(getattr(server_args, "enable_suffix_parallel", False)):
        raise RuntimeError("WeLM MTP direct K/V does not support suffix parallel")
    hf_config = getattr(target_model_config, "hf_config", None)
    if int(getattr(hf_config, "scale_seq_times", 0) or 0) > 0:
        raise RuntimeError("WeLM MTP direct K/V does not support Scale-Seq")


def build_welmv4_mtp_kv_mirror_buffer_spec(
    *,
    model_config,
    attention_tp_size: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> WelmMTPKVMirrorBufferSpec:
    if attention_tp_size <= 0:
        raise ValueError("attention_tp_size must be positive")

    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        raise ValueError("WeLM MTP mirror buffers require a Hugging Face config")

    num_hidden_layers = int(
        getattr(
            hf_config,
            "num_target_hidden_layers",
            getattr(hf_config, "num_hidden_layers", 0),
        )
        or 0
    )
    num_nextn_predict_layers = int(
        getattr(hf_config, "num_nextn_predict_layers", 0) or 0
    )
    mirror_layers = list(getattr(hf_config, "kv_mirror_layers", ()) or ())
    imitated_layers = list(
        getattr(hf_config, "kv_mirror_imitated_layers", ()) or ()
    )
    if len(mirror_layers) != len(imitated_layers):
        raise ValueError(
            "kv_mirror_layers and kv_mirror_imitated_layers must have the same length"
        )

    valid_layers = tuple(
        sorted(
            int(mirror)
            for mirror, imitated in zip(mirror_layers, imitated_layers)
            if num_hidden_layers
            <= int(mirror)
            < num_hidden_layers + num_nextn_predict_layers
            and 0 <= int(imitated) < num_hidden_layers
        )
    )
    expected_layers = tuple(
        range(num_hidden_layers, num_hidden_layers + num_nextn_predict_layers)
    )
    if num_hidden_layers <= 0 or not expected_layers or valid_layers != expected_layers:
        raise ValueError(
            "WeLM MTP config must define the complete NextN mirror layer set"
        )

    head_dim = int(
        getattr(model_config, "head_dim", getattr(hf_config, "head_dim", 0)) or 0
    )
    tensor_size = int(model_config.get_num_kv_heads(attention_tp_size)) * head_dim
    if tensor_size <= 0:
        raise ValueError("WeLM MTP mirror tensor size must be positive")

    return WelmMTPKVMirrorBufferSpec(
        mirror_layers=valid_layers,
        tensor_size=tensor_size,
        dtype=dtype,
        device=torch.device(device),
    )


def allocate_welmv4_mtp_kv_mirror_state_buffers(
    spec: WelmMTPKVMirrorBufferSpec,
    *,
    max_rows: int,
) -> Dict[str, torch.Tensor]:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    return {
        f"{layer_idx}.{component}": torch.zeros(
            (max_rows, spec.tensor_size), dtype=spec.dtype, device=spec.device
        )
        for layer_idx in spec.mirror_layers
        for component in ("k", "v")
    }


def get_welmv4_mtp_kv_mirror_max_rows(token_to_kv_pool_allocator) -> int:
    size_full = int(
        getattr(
            token_to_kv_pool_allocator,
            "size_full",
            token_to_kv_pool_allocator.size,
        )
    )
    return size_full + int(token_to_kv_pool_allocator.page_size)


def get_welmv4_mtp_kv_mirror_state_buf_infos(
    buffers: Optional[Dict[str, torch.Tensor]],
) -> Tuple[List[int], List[int], List[int]]:
    if not buffers:
        return [], [], []

    ordered_buffers = [buffers[key] for key in sorted(buffers)]
    return (
        [tensor.data_ptr() for tensor in ordered_buffers],
        [tensor.nbytes for tensor in ordered_buffers],
        [tensor[0].nbytes for tensor in ordered_buffers],
    )


def build_welmv4_mtp_model_specific_states(
    buffers: Optional[Dict[str, torch.Tensor]],
    rows: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
) -> Optional[Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor]]]]:
    if not buffers:
        return None

    kv_mirror_states = {}
    for key, tensor in buffers.items():
        layer_idx, suffix = key.split(".", 1)
        pair = kv_mirror_states.setdefault(int(layer_idx), [None, None])
        if indices is None:
            if rows is None:
                raise ValueError(
                    "rows is required when building WeLM MTP mirror states "
                    "without explicit indices."
                )
            state_tensor = tensor[:rows]
        else:
            state_tensor = tensor.index_select(
                0, indices.to(device=tensor.device, dtype=torch.long)
            )
        pair[0 if suffix == "k" else 1] = state_tensor

    kv_mirror_states = {
        layer_idx: (k, v)
        for layer_idx, (k, v) in kv_mirror_states.items()
        if k is not None and v is not None
    }
    if not kv_mirror_states:
        return None
    return {WELMV4_KV_MIRROR_STATES_KEY: kv_mirror_states}


def copy_welmv4_mtp_kv_mirror_states_to_buffers(
    dst_buffers: Optional[Dict[str, torch.Tensor]],
    model_specific_states,
    rows: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
) -> None:
    if not dst_buffers:
        return

    if indices is None:
        if rows is None:
            raise ValueError(
                "rows is required when copying WeLM MTP mirror states without "
                "explicit indices."
            )
        dst_indices = None
        copy_rows = int(rows)
    else:
        if indices.ndim != 1 or indices.dtype is not torch.long:
            raise RuntimeError(
                "WeLM MTP mirror destination indices must be one-dimensional int64"
            )
        dst_indices = indices
        copy_rows = int(dst_indices.numel())
    if copy_rows < 0:
        raise RuntimeError("WeLM MTP mirror row count must be non-negative")

    kv_mirror_states = (model_specific_states or {}).get(WELMV4_KV_MIRROR_STATES_KEY)
    if not isinstance(kv_mirror_states, dict):
        raise RuntimeError("WeLM MTP mirror state layer set is missing")

    try:
        expected_layers = {int(key.split(".", 1)[0]) for key in dst_buffers}
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("invalid WeLM MTP mirror destination key") from exc
    expected_keys = {
        f"{layer_idx}.{suffix}"
        for layer_idx in expected_layers
        for suffix in ("k", "v")
    }
    if set(dst_buffers) != expected_keys:
        raise RuntimeError("WeLM MTP mirror destination K/V pair is incomplete")
    if set(kv_mirror_states) != expected_layers:
        raise RuntimeError(
            "WeLM MTP mirror state layer set mismatch: "
            f"expected={sorted(expected_layers)}, got={sorted(kv_mirror_states)}"
        )

    copy_ops = []
    for layer_idx in sorted(expected_layers):
        tensors = kv_mirror_states[layer_idx]
        if not isinstance(tensors, (tuple, list)) or len(tensors) != 2:
            raise RuntimeError(
                f"WeLM MTP mirror K/V pair is invalid for layer {layer_idx}"
            )
        for suffix, src in zip(("k", "v"), tensors, strict=True):
            dst = dst_buffers[f"{layer_idx}.{suffix}"]
            if not isinstance(dst, torch.Tensor) or dst.ndim != 2:
                raise RuntimeError(
                    f"WeLM MTP mirror destination {layer_idx}.{suffix} must be 2D"
                )
            if not isinstance(src, torch.Tensor) or src.ndim != 2:
                raise RuntimeError(
                    f"WeLM MTP mirror source {layer_idx}.{suffix} must be 2D"
                )
            if src.shape[0] < copy_rows:
                raise RuntimeError(
                    f"WeLM MTP mirror source {layer_idx}.{suffix} row count "
                    f"{src.shape[0]} is smaller than {copy_rows}"
                )
            if src.shape[1] != dst.shape[1]:
                raise RuntimeError(
                    f"WeLM MTP mirror source {layer_idx}.{suffix} width "
                    f"{src.shape[1]} does not match {dst.shape[1]}"
                )
            if src.dtype != dst.dtype:
                raise RuntimeError(
                    f"WeLM MTP mirror source {layer_idx}.{suffix} dtype "
                    f"{src.dtype} does not match {dst.dtype}"
                )
            if src.device != dst.device:
                raise RuntimeError(
                    f"WeLM MTP mirror source {layer_idx}.{suffix} device "
                    f"{src.device} does not match {dst.device}"
                )
            if dst_indices is not None and dst_indices.device != dst.device:
                raise RuntimeError(
                    "WeLM MTP mirror destination indices device "
                    f"{dst_indices.device} does not match {dst.device}"
                )
            if dst_indices is None and copy_rows > dst.shape[0]:
                raise RuntimeError(
                    f"WeLM MTP mirror destination {layer_idx}.{suffix} row count "
                    f"{dst.shape[0]} is smaller than {copy_rows}"
                )
            copy_ops.append((dst, src[:copy_rows]))

    for dst, src in copy_ops:
        if dst_indices is None:
            dst[:copy_rows].copy_(src)
        else:
            dst.index_copy_(0, dst_indices, src)


def slice_welmv4_mtp_kv_mirror_states(model_specific_states, indices=None):
    if not model_specific_states:
        return None

    kv_mirror_states = model_specific_states.get(WELMV4_KV_MIRROR_STATES_KEY)
    if not isinstance(kv_mirror_states, dict):
        return None

    sliced_states = {}
    for layer_idx, tensors in kv_mirror_states.items():
        if not isinstance(tensors, (tuple, list)) or len(tensors) != 2:
            continue
        k, v = tensors
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            continue
        if indices is None:
            sliced_states[int(layer_idx)] = (k, v)
        else:
            sliced_states[int(layer_idx)] = (k[indices], v[indices])

    if not sliced_states:
        return None

    ret = dict(model_specific_states)
    ret[WELMV4_KV_MIRROR_STATES_KEY] = sliced_states
    return ret
