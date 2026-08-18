from __future__ import annotations

import logging
from http import HTTPStatus
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.server_args import ServerArgs


def _assert_no_welm_deferred_decode_seed(reqs) -> None:
    from sglang.srt.managers.schedule_batch import (
        WelmDeferredDecodePhase,
        WelmDeferredDecodeState,
    )

    deferred_rids = []
    for req in reqs:
        state = vars(req).get("welm_deferred_decode_state")
        if (
            isinstance(state, WelmDeferredDecodeState)
            and state.phase is not WelmDeferredDecodePhase.CONSUMED
        ):
            deferred_rids.append(req.rid)
    if deferred_rids:
        raise RuntimeError(
            "WeLM deferred decode seed must not enter the legacy PREBUILT path: "
            f"rids={deferred_rids}"
        )


class ScheduleBatchDisaggregationDecodeMixin:
    def _is_welm_mtp_pd_decode_prebuilt(self: ScheduleBatch, server_args: ServerArgs):
        return (
            server_args.disaggregation_mode == "decode"
            and server_args.enable_welm_kv_mirror_opt
            and self.spec_algorithm.is_eagle()
        )

    def prepare_for_prebuilt(self: ScheduleBatch):
        """
        Prepare a prebuilt extend by populate metadata
        Adapted from .prepare_for_extend().
        """
        _assert_no_welm_deferred_decode_seed(self.reqs)

        self.forward_mode = ForwardMode.PREBUILT
        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []
        pre_lens = []
        req_pool_indices = []

        # Pre-calculate total size
        total_size = sum(req.extend_input_len for req in reqs)
        out_cache_loc = torch.empty(total_size, dtype=torch.int64, device=self.device)

        # Fill the tensor in one pass
        offset = 0
        for i, req in enumerate(reqs):
            req_pool_indices.append(req.req_pool_idx)

            chunk = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                : req.extend_input_len
            ]
            assert (
                offset + req.extend_input_len <= total_size
            ), f"Exceeds total size: offset={offset}, req.extend_input_len={req.extend_input_len}, total_size={total_size}"
            out_cache_loc[offset : offset + req.extend_input_len] = chunk
            offset += req.extend_input_len

            pre_len = len(req.prefix_indices)
            seq_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
            seq_lens.append(seq_len)
            if len(req.output_ids) == 0:
                assert (
                    seq_len - pre_len == req.extend_input_len
                ), f"seq_len={seq_len}, pre_len={pre_len}, req.extend_input_len={req.extend_input_len}"

            if not req.retracted_stain:
                req.cached_tokens += pre_len - req.already_computed
                req.already_computed = seq_len
            req.is_retracted = False
            pre_lens.append(pre_len)
            req.extend_logprob_start_len = 0

        extend_input_logprob_token_ids = None

        # Set fields
        self.input_ids = torch.tensor(
            sum(input_ids, []), dtype=torch.int32, device=self.device
        )
        self.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=self.device
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        self.orig_seq_lens = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_input_len for r in reqs]
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids
        self.multimodal_inputs = [r.multimodal_inputs for r in reqs]

        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
        if self._is_welm_mtp_pd_decode_prebuilt(server_args):
            from sglang.srt.managers.schedule_batch import OverEncodingContext
            from sglang.srt.models.welm_perf_opt import (
                get_welm_oe_hash_config,
                should_use_welm_oe_hash_kernel,
            )

            if should_use_welm_oe_hash_kernel(self.model_config):
                oe_grams, _ = get_welm_oe_hash_config(self.model_config)
                if not oe_grams:
                    raise RuntimeError(
                        "WeLM OE requires oe_grams for the CUDA hash kernel."
                    )
                self.oe_context = OverEncodingContext.from_extend_hash_kernel(
                    reqs,
                    self.prefix_lens,
                    max(oe_grams) - 1,
                    legacy_history_width=max(oe_grams),
                )

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def process_prebuilt(
        self: ScheduleBatch,
        server_args: ServerArgs,
        future_map: FutureMap,
    ):
        """Assign the buffered last input id to schedule batch"""
        _assert_no_welm_deferred_decode_seed(self.reqs)
        self.output_ids = []
        for req in self.reqs:
            self.output_ids.append(req.output_ids[-1])
            maybe_cache_unfinished_req(req, self.tree_cache)
            if req.grammar is not None:
                # FIXME: this try-except block is for handling unexpected xgrammar issue.
                try:
                    # if it is not None, then the grammar is from a retracted request, and we should not
                    # accept the token as it's already accepted
                    if req.grammar.current_token is None:
                        req.grammar.accept_token(req.output_ids[-1])
                except ValueError as e:
                    from sglang.srt.managers.schedule_batch import FINISH_ABORT

                    # Grammar accept_token can raise ValueError if the token is not in the grammar.
                    # This can happen if the grammar is not set correctly or the token is invalid.
                    # Use to_finish (not finished_reason) so that process_batch_result_prebuilt
                    # handles the release via check_finished -> release_kv_cache in one place.
                    error_message = f"Grammar accept_token failed for req {req.rid} with token {req.output_ids[-1]}: {e}"
                    req.to_finish = FINISH_ABORT(
                        error_message, HTTPStatus.INTERNAL_SERVER_ERROR
                    )
                req.grammar.finished = req.finished()
        self.output_ids = torch.tensor(self.output_ids, device=self.device)

        # Simulate the eagle run.
        if self.spec_algorithm.is_eagle():
            is_welm_mtp_pd_decode = self._is_welm_mtp_pd_decode_prebuilt(server_args)
            num_states = (
                server_args.speculative_num_steps
                if is_welm_mtp_pd_decode and server_args.speculative_eagle_topk == 1
                else server_args.speculative_eagle_topk
            )
            if server_args.enable_multi_layer_eagle:
                num_states *= server_args.speculative_num_steps

            hidden_states_list = [req.hidden_states_tensor for req in self.reqs]
            hidden_states = torch.stack(hidden_states_list, dim=0).to(self.device)

            # local import to avoid circular import
            from sglang.srt.speculative.eagle_info import EagleDraftInput

            if is_welm_mtp_pd_decode:
                topk_index = (
                    self.output_ids.to(dtype=torch.int64)
                    .view(-1, 1)
                    .expand(-1, num_states)
                    .contiguous()
                )
                topk_p = torch.ones(
                    (len(self.reqs), num_states),
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                topk_p = torch.stack(
                    [
                        torch.as_tensor(
                            req.output_topk_p[:num_states],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        for req in self.reqs
                    ],
                    dim=0,
                )
                topk_index = torch.stack(
                    [
                        torch.as_tensor(
                            req.output_topk_index[:num_states],
                            device=self.device,
                            dtype=torch.int64,
                        )
                        for req in self.reqs
                    ],
                    dim=0,
                )

            spec_info = EagleDraftInput(
                topk_p=topk_p,
                topk_index=topk_index,
                hidden_states=hidden_states,
                bonus_tokens=self.output_ids,
                new_seq_lens=self.seq_lens,
            )
            if is_welm_mtp_pd_decode:
                # local import to avoid circular import
                from sglang.srt.disaggregation.decode import _is_fake_transfer

                if server_args.speculative_eagle_topk == 1 and all(
                    _is_fake_transfer(req, server_args) for req in self.reqs
                ):
                    # Fake-transfer requests (request-level 2.2.2.2 bootstrap or
                    # global fake backend): no prefill peer produced draft KV or
                    # mirror states, and the deferred draft prefill deadlocks
                    # under dp-attention (idle DP ranks never join the
                    # out-of-band extend's collectives). The prebuilt
                    # topk_p/topk_index above already satisfy the linear
                    # draft-proposal contract, so enter steady-state verify
                    # directly. get_new_prebuilt_batch keeps prebuilt batches
                    # uniform in fakeness, so all() cannot see a mixed batch.
                    spec_info.welm_mtp_deferred_prefill_draft = False
                    spec_info.welm_mtp_deferred_prefill_draft_mask = None
                else:
                    spec_info.welm_mtp_deferred_prefill_draft = True
                    spec_info.welm_mtp_deferred_prefill_draft_mask = torch.ones(
                        (len(self.reqs),),
                        dtype=torch.bool,
                        device=self.device,
                    )
                spec_info.num_tokens_per_req = 1
                spec_info.num_tokens_for_logprob_per_req = 1
            else:
                spec_info.prepare_for_extend(self)
            spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
            if self.enable_overlap:
                spec_info.future_indices = future_map.alloc_future_indices(
                    len(self.seq_lens), self.req_pool_indices
                )
                future_map.store_to_map_for_new_batch(
                    spec_info.future_indices, spec_info
                )
            self.spec_info = spec_info
