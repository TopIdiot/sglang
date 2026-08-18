from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_is_cuda = is_cuda()
_is_hip = is_hip()


def _resolve_future_token_ids_native(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


if _is_cuda or _is_hip:
    from sglang.jit_kernel.resolve_future_token_ids import (
        resolve_future_token_ids_cuda,
    )

    _resolve_future_token_ids = resolve_future_token_ids_cuda
else:
    _resolve_future_token_ids = _resolve_future_token_ids_native


@dataclass
class FutureIndices:
    indices: torch.Tensor
    interval: Optional[slice] = None


class FutureMap:
    def __init__(
        self,
        max_running_requests: int,
        chunked_prefill_size: int,
        context_len: int,
        device: torch.device,
        spec_algo: Optional[SpeculativeAlgorithm] = None,
        request_buffer_len: Optional[int] = None,
    ):
        # FIXME: the calculation of future_limit and future_buffer_len maybe too conservative
        self.future_ct = 0

        # Circular buffer layout (wraps in this order):
        # Running decode batch -> Prefill chunk 1 -> ... -> Prefill chunk N
        # A running decode batch's result will be resolved after all prefill chunks are done.
        # reserve `max_num_chunks` extra future slots on top of `max_running_requests * 3`.
        max_num_chunks = (
            (context_len + chunked_prefill_size - 1) // chunked_prefill_size
            if chunked_prefill_size
            else 0
        )
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        # Adding 2 * max_running_requests to future_limit ensures the buffer is sufficiently large.
        self.future_buffer_len = self.future_limit + 2 * max_running_requests
        # The large stochastic-draft tensors are request state, so keep one
        # copy per actual request-pool row instead of reserving a copy for every
        # chunked-prefill slot in the future ring.  A regular request pool has
        # max_running_requests + 1 rows (including CUDA-graph padding slot 0),
        # while a disaggregated decode pool also has pre-allocation rows.
        self.request_buffer_len = (
            max_running_requests + 1
            if request_buffer_len is None
            else request_buffer_len
        )
        if self.request_buffer_len < max_running_requests + 1:
            raise ValueError(
                "request_buffer_len must cover all running request slots and "
                f"CUDA-graph padding: got {self.request_buffer_len}, expected "
                f"at least {max_running_requests + 1}."
            )
        self.device = device
        self.spec_algo = spec_algo

        if self.spec_algo.is_none():
            # For non-speculative decoding, we only need to store the token ids.
            self.buf_initialized = True
            self.token_ids_buf = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )
        else:
            # For speculative decoding, we lazily initialize the buffers
            # This is to make the shape derivation easier.
            self.buf_initialized = False
            self.req_pool_indices_buf = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )

    def _lazy_init_buf(self, draft_input: EagleDraftInput):
        self.buf_initialized = True

        # Get a reference for each tensor
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        bonus_tokens0 = draft_input.bonus_tokens[0]
        new_seq_lens0 = draft_input.new_seq_lens[0]

        self.topk_p_buf = torch.empty(
            (self.future_buffer_len, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.future_buffer_len, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        self.bonus_tokens_buf = torch.empty(
            (self.future_buffer_len, *bonus_tokens0.shape),
            dtype=bonus_tokens0.dtype,
            device=self.device,
        )
        self.new_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )
        self.welm_mtp_deferred_prefill_draft_buf = torch.empty(
            (self.future_buffer_len,),
            dtype=torch.bool,
            device=self.device,
        )
        self.welm_mtp_has_draft_probs_buf = torch.empty(
            (self.future_buffer_len,),
            dtype=torch.bool,
            device=self.device,
        )
        self.welm_mtp_has_draft_topk_buf = torch.empty(
            (self.future_buffer_len,),
            dtype=torch.bool,
            device=self.device,
        )
        self.welm_mtp_has_draft_proposal_buf = torch.empty(
            (self.future_buffer_len,),
            dtype=torch.bool,
            device=self.device,
        )
        self.welm_mtp_has_oe_history_buf = torch.empty(
            (self.future_buffer_len,),
            dtype=torch.bool,
            device=self.device,
        )

        if spec_need_hidden_states():
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.future_buffer_len, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

        self.has_welm_mtp_draft_probs_buf = False
        self.has_welm_mtp_draft_topk_buf = False
        self.has_welm_mtp_draft_proposal_buf = False
        self.has_welm_mtp_oe_history_buf = False
        self._ensure_welm_mtp_draft_probs_buf(draft_input)
        self._ensure_welm_mtp_draft_topk_buf(draft_input)
        self._ensure_welm_mtp_draft_proposal_buf(draft_input)
        self._ensure_welm_mtp_oe_history_buf(draft_input)

    def _ensure_welm_mtp_draft_probs_buf(self, draft_input: EagleDraftInput):
        if getattr(self, "has_welm_mtp_draft_probs_buf", False):
            return

        draft_probs = getattr(draft_input, "draft_probs", None)
        if draft_probs is None:
            return

        draft_probs0 = draft_probs[0]
        self.welm_mtp_draft_probs_buf = torch.empty(
            (self.request_buffer_len, *draft_probs0.shape),
            dtype=draft_probs0.dtype,
            device=self.device,
        )
        self.has_welm_mtp_draft_probs_buf = True

    def _ensure_welm_mtp_draft_topk_buf(self, draft_input: EagleDraftInput):
        if getattr(self, "has_welm_mtp_draft_topk_buf", False):
            return

        draft_topk_indices = getattr(draft_input, "welm_mtp_draft_topk_indices", None)
        draft_topk_values = getattr(draft_input, "welm_mtp_draft_topk_values", None)
        if draft_topk_indices is None or draft_topk_values is None:
            return

        draft_topk_indices0 = draft_topk_indices[0]
        draft_topk_values0 = draft_topk_values[0]
        self.welm_mtp_draft_topk_indices_buf = torch.empty(
            (self.request_buffer_len, *draft_topk_indices0.shape),
            dtype=draft_topk_indices0.dtype,
            device=self.device,
        )
        self.welm_mtp_draft_topk_values_buf = torch.empty(
            (self.request_buffer_len, *draft_topk_values0.shape),
            dtype=draft_topk_values0.dtype,
            device=self.device,
        )
        self.has_welm_mtp_draft_topk_buf = True

    def _ensure_welm_mtp_draft_proposal_buf(self, draft_input: EagleDraftInput):
        if getattr(self, "has_welm_mtp_draft_proposal_buf", False):
            return

        parent_list = getattr(draft_input, "draft_proposal_parent_list", None)
        top_scores_index = getattr(
            draft_input, "draft_proposal_top_scores_index", None
        )
        draft_tokens = getattr(draft_input, "draft_proposal_tokens", None)
        if parent_list is None or top_scores_index is None or draft_tokens is None:
            return

        parent_list0 = parent_list[0]
        top_scores_index0 = top_scores_index[0]
        draft_tokens0 = draft_tokens[0]
        self.welm_mtp_draft_proposal_parent_list_buf = torch.empty(
            (self.future_buffer_len, *parent_list0.shape),
            dtype=parent_list0.dtype,
            device=self.device,
        )
        self.welm_mtp_draft_proposal_top_scores_index_buf = torch.empty(
            (self.future_buffer_len, *top_scores_index0.shape),
            dtype=top_scores_index0.dtype,
            device=self.device,
        )
        self.welm_mtp_draft_proposal_tokens_buf = torch.empty(
            (self.future_buffer_len, *draft_tokens0.shape),
            dtype=draft_tokens0.dtype,
            device=self.device,
        )
        self.has_welm_mtp_draft_proposal_buf = True

    def _ensure_welm_mtp_oe_history_buf(self, draft_input: EagleDraftInput):
        if getattr(self, "has_welm_mtp_oe_history_buf", False):
            return

        history_state = getattr(draft_input, "welm_mtp_oe_history_state", None)
        if history_state is None:
            return

        history_state0 = history_state[0]
        self.welm_mtp_oe_history_buf = torch.empty(
            (self.future_buffer_len, *history_state0.shape),
            dtype=history_state0.dtype,
            device=self.device,
        )
        self.has_welm_mtp_oe_history_buf = True

    def alloc_future_indices(
        self, bs: int, req_pool_indices: Optional[torch.Tensor] = None
    ) -> FutureIndices:
        """Update the circular buffer pointer and allocate future indices."""
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        start = cur_future_ct + 1
        end = cur_future_ct + 1 + bs
        indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
        if not self.spec_algo.is_none():
            if req_pool_indices is None or req_pool_indices.numel() != bs:
                num_req_pool_indices = (
                    None if req_pool_indices is None else req_pool_indices.numel()
                )
                raise ValueError(
                    "Speculative overlap requires one request pool index per "
                    f"future row, got bs={bs} and "
                    f"req_pool_indices={num_req_pool_indices}."
                )
            self.req_pool_indices_buf[start:end] = req_pool_indices
        return FutureIndices(indices=indices, interval=slice(start, end))

    def resolve_future(self, model_worker_batch: ModelWorkerBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(model_worker_batch.input_ids, self.token_ids_buf)
        else:
            # TODO(lsyin): write future indices into spec_info.future_indices
            draft_input: EagleDraftInput = model_worker_batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = draft_input.future_indices.indices
            req_pool_indices = self.req_pool_indices_buf[indices]
            # The indices tensor was allocated on the default stream but is
            # used here on the forward stream. Meanwhile, the old spec_info
            # holding this tensor will lose all Python references (replaced at
            # model_worker_batch.spec_info and batch.spec_info), so the
            # caching allocator (torch GC) could reclaim the memory before
            # the GPU finishes reading it.
            indices.record_stream(torch.get_device_module(self.device).current_stream())
            draft_input.topk_p = self.topk_p_buf[indices]
            draft_input.topk_index = self.topk_index_buf[indices]
            draft_input.bonus_tokens = self.bonus_tokens_buf[indices]
            draft_input.new_seq_lens = self.new_seq_lens_buf[indices]
            if hasattr(self, "welm_mtp_deferred_prefill_draft_buf"):
                draft_input.welm_mtp_deferred_prefill_draft_mask = (
                    self.welm_mtp_deferred_prefill_draft_buf[indices]
                )
                draft_input.welm_mtp_deferred_prefill_draft = (
                    bool(draft_input.welm_mtp_deferred_prefill_draft_mask.any().item())
                    if draft_input.welm_mtp_deferred_prefill_draft_mask.numel() > 0
                    else False
                )
            if spec_need_hidden_states():
                draft_input.hidden_states = self.hidden_states_buf[indices]
            has_draft_probs = False
            if (
                getattr(self, "has_welm_mtp_draft_probs_buf", False)
                and hasattr(self, "welm_mtp_has_draft_probs_buf")
            ):
                draft_probs_flags = self.welm_mtp_has_draft_probs_buf[indices]
                has_draft_probs = (
                    bool(draft_probs_flags.all().item())
                    if draft_probs_flags.numel() > 0
                    else False
                )
            draft_input.draft_probs = (
                self.welm_mtp_draft_probs_buf[req_pool_indices]
                if has_draft_probs
                else None
            )
            has_draft_topk = False
            if (
                getattr(self, "has_welm_mtp_draft_topk_buf", False)
                and hasattr(self, "welm_mtp_has_draft_topk_buf")
            ):
                draft_topk_flags = self.welm_mtp_has_draft_topk_buf[indices]
                has_draft_topk = (
                    bool(draft_topk_flags.all().item())
                    if draft_topk_flags.numel() > 0
                    else False
                )
            if has_draft_topk:
                draft_input.welm_mtp_draft_topk_indices = (
                    self.welm_mtp_draft_topk_indices_buf[req_pool_indices]
                )
                draft_input.welm_mtp_draft_topk_values = (
                    self.welm_mtp_draft_topk_values_buf[req_pool_indices]
                )
            else:
                draft_input.welm_mtp_draft_topk_indices = None
                draft_input.welm_mtp_draft_topk_values = None
            has_draft_proposal = False
            if (
                getattr(self, "has_welm_mtp_draft_proposal_buf", False)
                and hasattr(self, "welm_mtp_has_draft_proposal_buf")
            ):
                draft_proposal_flags = self.welm_mtp_has_draft_proposal_buf[indices]
                has_draft_proposal = (
                    bool(draft_proposal_flags.all().item())
                    if draft_proposal_flags.numel() > 0
                    else False
                )
            if has_draft_proposal:
                draft_input.draft_proposal_parent_list = (
                    self.welm_mtp_draft_proposal_parent_list_buf[indices]
                )
                draft_input.draft_proposal_top_scores_index = (
                    self.welm_mtp_draft_proposal_top_scores_index_buf[indices]
                )
                draft_input.draft_proposal_tokens = (
                    self.welm_mtp_draft_proposal_tokens_buf[indices]
                )
            else:
                draft_input.draft_proposal_parent_list = None
                draft_input.draft_proposal_top_scores_index = None
                draft_input.draft_proposal_tokens = None
            has_oe_history = False
            if hasattr(self, "welm_mtp_has_oe_history_buf"):
                oe_history_flags = self.welm_mtp_has_oe_history_buf[indices]
                has_oe_history = (
                    bool(oe_history_flags.any().item())
                    if oe_history_flags.numel() > 0
                    else False
                )
            if (
                getattr(self, "has_welm_mtp_oe_history_buf", False)
                and has_oe_history
            ):
                draft_input.welm_mtp_oe_history_state = self.welm_mtp_oe_history_buf[
                    indices
                ]
            elif hasattr(draft_input, "welm_mtp_oe_history_state"):
                delattr(draft_input, "welm_mtp_oe_history_state")

    def is_empty_slice(self, s: slice) -> bool:
        start, stop, step = s.indices(self.future_buffer_len)
        if step > 0:
            return start >= stop
        else:
            return start <= stop

    def store_to_map(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ):
        if self.spec_algo.is_none():
            intv = future_indices.interval
            self.token_ids_buf[intv] = batch_result.next_token_ids
        else:
            draft_continuation_state = batch_result.draft_continuation_state
            assert draft_continuation_state is not None
            draft_input: EagleDraftInput = draft_continuation_state.draft_input
            self.store_to_map_for_new_batch(future_indices, draft_input)

    def store_to_map_for_new_batch(
        self, future_indices: FutureIndices, draft_input: EagleDraftInput
    ):
        intv = future_indices.interval
        if self.is_empty_slice(intv):
            # idle indices in dp attention do not need store info
            return

        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)

        req_pool_indices = self.req_pool_indices_buf[intv]

        self.topk_p_buf[intv] = draft_input.topk_p
        self.topk_index_buf[intv] = draft_input.topk_index
        self.bonus_tokens_buf[intv] = draft_input.bonus_tokens
        self.new_seq_lens_buf[intv] = draft_input.new_seq_lens
        deferred_mask = getattr(
            draft_input, "welm_mtp_deferred_prefill_draft_mask", None
        )
        if hasattr(self, "welm_mtp_deferred_prefill_draft_buf"):
            if deferred_mask is None:
                self.welm_mtp_deferred_prefill_draft_buf[intv] = bool(
                    getattr(draft_input, "welm_mtp_deferred_prefill_draft", False)
                )
            else:
                self.welm_mtp_deferred_prefill_draft_buf[intv] = deferred_mask
        if hasattr(self, "welm_mtp_has_draft_probs_buf"):
            has_draft_probs = getattr(draft_input, "draft_probs", None) is not None
            if has_draft_probs and deferred_mask is not None:
                self.welm_mtp_has_draft_probs_buf[intv] = ~deferred_mask
            else:
                self.welm_mtp_has_draft_probs_buf[intv] = has_draft_probs
        if hasattr(self, "welm_mtp_has_draft_topk_buf"):
            has_draft_topk = (
                getattr(draft_input, "welm_mtp_draft_topk_indices", None) is not None
                and getattr(draft_input, "welm_mtp_draft_topk_values", None)
                is not None
            )
            if has_draft_topk and deferred_mask is not None:
                self.welm_mtp_has_draft_topk_buf[intv] = ~deferred_mask
            else:
                self.welm_mtp_has_draft_topk_buf[intv] = has_draft_topk
        if hasattr(self, "welm_mtp_has_draft_proposal_buf"):
            has_draft_proposal = (
                getattr(draft_input, "draft_proposal_parent_list", None) is not None
                and getattr(draft_input, "draft_proposal_top_scores_index", None)
                is not None
                and getattr(draft_input, "draft_proposal_tokens", None) is not None
            )
            if has_draft_proposal and deferred_mask is not None:
                self.welm_mtp_has_draft_proposal_buf[intv] = ~deferred_mask
            else:
                self.welm_mtp_has_draft_proposal_buf[intv] = has_draft_proposal
        if hasattr(self, "welm_mtp_has_oe_history_buf"):
            self.welm_mtp_has_oe_history_buf[intv] = (
                getattr(draft_input, "welm_mtp_oe_history_state", None) is not None
            )
        if spec_need_hidden_states():
            self.hidden_states_buf[intv] = draft_input.hidden_states
        self._ensure_welm_mtp_draft_probs_buf(draft_input)
        if getattr(self, "has_welm_mtp_draft_probs_buf", False):
            if getattr(draft_input, "draft_probs", None) is not None:
                self.welm_mtp_draft_probs_buf[req_pool_indices] = (
                    draft_input.draft_probs
                )
        self._ensure_welm_mtp_draft_topk_buf(draft_input)
        if getattr(self, "has_welm_mtp_draft_topk_buf", False):
            has_draft_topk = (
                draft_input.welm_mtp_draft_topk_indices is not None
                and draft_input.welm_mtp_draft_topk_values is not None
            )
            if has_draft_topk:
                self.welm_mtp_draft_topk_indices_buf[req_pool_indices] = (
                    draft_input.welm_mtp_draft_topk_indices
                )
                self.welm_mtp_draft_topk_values_buf[req_pool_indices] = (
                    draft_input.welm_mtp_draft_topk_values
                )
        self._ensure_welm_mtp_draft_proposal_buf(draft_input)
        if getattr(self, "has_welm_mtp_draft_proposal_buf", False):
            has_draft_proposal = (
                draft_input.draft_proposal_parent_list is not None
                and draft_input.draft_proposal_top_scores_index is not None
                and draft_input.draft_proposal_tokens is not None
            )
            if not has_draft_proposal:
                self.welm_mtp_draft_proposal_parent_list_buf[intv].zero_()
                self.welm_mtp_draft_proposal_top_scores_index_buf[intv].zero_()
                self.welm_mtp_draft_proposal_tokens_buf[intv].zero_()
            else:
                self.welm_mtp_draft_proposal_parent_list_buf[intv] = (
                    draft_input.draft_proposal_parent_list
                )
                self.welm_mtp_draft_proposal_top_scores_index_buf[intv] = (
                    draft_input.draft_proposal_top_scores_index
                )
                self.welm_mtp_draft_proposal_tokens_buf[intv] = (
                    draft_input.draft_proposal_tokens
                )
        self._ensure_welm_mtp_oe_history_buf(draft_input)
        if getattr(self, "has_welm_mtp_oe_history_buf", False):
            history_state = getattr(draft_input, "welm_mtp_oe_history_state", None)
            if history_state is None:
                self.welm_mtp_oe_history_buf[intv].zero_()
            else:
                self.welm_mtp_oe_history_buf[intv] = history_state
