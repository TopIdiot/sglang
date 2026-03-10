"""N-gram operations for speculative decoding.

Provides JIT-compiled CUDA kernels that were previously in ``prc_custom_ops``:

- ``build_ngram_with_tree``: build n-gram input IDs from a tree structure
- ``build_ngram_with_target_verify``: build n-gram input IDs for target verification
- ``assign_ngram_input_ids_draft_extend_after_decode``: assign n-gram input IDs
  during draft-extend-after-decode
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ngram_ops_module() -> Module:
    return load_jit(
        "ngram_ops",
        cuda_files=["ngram_ops.cuh"],
        cuda_wrappers=[
            ("build_ngram_with_tree", "build_ngram_with_tree"),
            ("build_ngram_with_target_verify", "build_ngram_with_target_verify"),
            (
                "assign_ngram_input_ids_draft_extend_after_decode",
                "assign_ngram_input_ids_draft_extend_after_decode",
            ),
        ],
    )


def build_ngram_with_tree(
    ngram_input_ids: torch.Tensor,
    parent_list: torch.Tensor,
    token_list: torch.Tensor,
    current_parrent_list: torch.Tensor,
    buffer: torch.Tensor,
    buffer_size: int,
    gram_n: int,
    topk: int,
    i: int,
) -> None:
    """Build n-gram input IDs from a tree structure.

    Args:
        ngram_input_ids: Output tensor for n-gram input IDs.
        parent_list: Parent indices in the tree.
        token_list: Token IDs in the tree.
        current_parrent_list: Current parent list for the latest layer.
        buffer: N-gram input IDs buffer.
        buffer_size: Size of the buffer per sequence.
        gram_n: N-gram order (2, 3, or 4).
        topk: Top-k value.
        i: Current tree layer index.
    """
    module = _jit_ngram_ops_module()
    module.build_ngram_with_tree(
        ngram_input_ids,
        parent_list,
        token_list,
        current_parrent_list,
        buffer,
        buffer_size,
        gram_n,
        topk,
        i,
    )


def build_ngram_with_target_verify(
    ngram_input_ids: torch.Tensor,
    buffer: torch.Tensor,
    draft_token_ids: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    gram_n: int,
    draft_token_num: int,
    buffer_size: int,
) -> None:
    """Build n-gram input IDs for target verification.

    Args:
        ngram_input_ids: Output tensor for n-gram input IDs.
        buffer: N-gram input IDs buffer.
        draft_token_ids: Draft token IDs.
        tree_mask: Boolean tree attention mask.
        positions: Position IDs.
        seq_lens: Sequence lengths.
        gram_n: N-gram order (2, 3, or 4).
        draft_token_num: Number of draft tokens.
        buffer_size: Size of the buffer per sequence.
    """
    module = _jit_ngram_ops_module()
    module.build_ngram_with_target_verify(
        ngram_input_ids,
        buffer,
        draft_token_ids,
        tree_mask,
        positions,
        seq_lens,
        gram_n,
        draft_token_num,
        buffer_size,
    )


def assign_ngram_input_ids_draft_extend_after_decode(
    input_ids: torch.Tensor,
    buffer: torch.Tensor,
    input_ids_gram: torch.Tensor,
    accept_length: torch.Tensor,
    gram_n: int,
    buffer_size: int,
    update_buffer: bool = False,
) -> None:
    """Assign n-gram input IDs during draft-extend-after-decode.

    Args:
        input_ids: Input token IDs (int64).
        buffer: N-gram input IDs buffer (int64).
        input_ids_gram: Output n-gram input IDs (int64).
        accept_length: Accepted lengths per sequence (int32).
        gram_n: N-gram order (2, 3, or 4).
        buffer_size: Size of the buffer per sequence (must be < 10).
        update_buffer: Whether to update the buffer (default: False).
    """
    module = _jit_ngram_ops_module()
    module.assign_ngram_input_ids_draft_extend_after_decode(
        input_ids,
        buffer,
        input_ids_gram,
        accept_length,
        gram_n,
        buffer_size,
        int(update_buffer),
    )
