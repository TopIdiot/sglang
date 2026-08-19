"""Token-id based tool-call detector for WeLM-v4 models.

WeLM-v4 emits GLM-style tool calls, but each structural marker
(``<tool_call>``, ``</tool_call>``, ``<arg_key>``, ...) is a single added-token
id. This detector locates those markers by token id, so prose that merely
quotes ``<tool_call>`` cannot trigger a false tool-call boundary. If the
tokenizer cannot resolve the markers, the detector falls back to the existing
text-based GLM parser.
"""

import json
import logging
from enum import Enum
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.glm4_moe_detector import (
    Glm4MoeDetector,
    get_argument_type,
    parse_arguments,
)
from sglang.srt.function_call.utils import (
    coerce_union_literal,
    get_argument_types,
    still_possible,
)

logger = logging.getLogger(__name__)


def _resolve_token_id(tokenizer, token: str) -> Optional[int]:
    """Resolve a WeLM-v4 control token to a single vocabulary id."""
    if tokenizer is None:
        return None
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if token_id is None:
        return None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        return None
    return token_id


class _IncrementalDecoder:
    """Decode a growing id list into stable text deltas."""

    def __init__(
        self,
        tokenizer,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.prefix_offset = 0
        self.read_offset = 0
        self.emitted = ""

    def _decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        return self.tokenizer.decode(
            ids,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )

    def step(self, ids: List[int]) -> str:
        prefix_text = self._decode(ids[self.prefix_offset : self.read_offset])
        new_text = self._decode(ids[self.prefix_offset :])
        if len(new_text) <= len(prefix_text) or new_text.endswith("\ufffd"):
            return ""
        delta = new_text[len(prefix_text) :]
        self.prefix_offset = self.read_offset
        self.read_offset = len(ids)
        self.emitted += delta
        return delta

    def flush(self, ids: List[int]) -> str:
        full = self._decode(ids)
        if len(full) <= len(self.emitted):
            return ""
        delta = full[len(self.emitted) :]
        self.emitted = full
        self.prefix_offset = len(ids)
        self.read_offset = len(ids)
        return delta


def trim_matched_stop_for_id_parser(
    output: Optional[List[int]],
    finished_reason: Optional[dict],
    no_stop_trim: bool,
):
    if output is None or no_stop_trim or not finished_reason:
        return output
    matched = finished_reason.get("matched_text") or finished_reason.get("matched")
    if not matched:
        return output
    if isinstance(matched, str) and isinstance(output, list):
        return None
    if isinstance(matched, int) and isinstance(output, list) and output:
        return output[:-1]
    return output


def _collect_token_ids(*values) -> set[int]:
    token_ids = set()
    for value in values:
        if value is None:
            continue
        try:
            token_ids.update(value)
        except TypeError:
            token_ids.add(value)
    return token_ids


def filter_id_based_stream_stop(
    request,
    tokenizer,
    model_config,
    delta: str,
    delta_ids: Optional[List[int]],
    choice_logprobs: Optional[dict],
    finish_reason: Optional[dict],
    skip_special_tokens: bool,
    spaces_between_special_tokens: bool,
) -> tuple[str, Optional[List[int]], Optional[dict]]:
    if request.no_stop_trim:
        return delta, delta_ids, choice_logprobs

    finish_reason = finish_reason or {}
    matched = finish_reason.get("matched_text") or finish_reason.get("matched")
    if isinstance(matched, str):
        stop_token_ids = _collect_token_ids(
            getattr(request, "stop_token_ids", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "additional_stop_token_ids", None),
            getattr(model_config, "hf_eos_token_id", None),
        )
        if not delta_ids or delta_ids[-1] not in stop_token_ids:
            raise ValueError(
                "WeLM-v4 ID parser received a string finish reason without "
                "a terminal stop token ID"
            )
        stop_pos = len(delta_ids) - 1
    elif not delta_ids:
        return delta, delta_ids, choice_logprobs
    elif isinstance(matched, int):
        stop_token_ids = {matched}
    elif request.ignore_eos:
        return delta, delta_ids, choice_logprobs
    else:
        stop_token_ids = _collect_token_ids(
            getattr(request, "stop_token_ids", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "additional_stop_token_ids", None),
            getattr(model_config, "hf_eos_token_id", None),
        )

    if not isinstance(matched, str):
        stop_pos = next(
            (i for i, token_id in enumerate(delta_ids) if token_id in stop_token_ids),
            None,
        )
        if stop_pos is None:
            return delta, delta_ids, choice_logprobs

    filtered_ids = delta_ids[:stop_pos]
    return (
        (
            tokenizer.decode(
                filtered_ids,
                skip_special_tokens=skip_special_tokens,
                spaces_between_special_tokens=spaces_between_special_tokens,
            )
            if filtered_ids
            else ""
        ),
        filtered_ids,
        # Keep logprobs for the raw sampled token sequence, including the stop
        # token. Only the visible delta and the ids consumed by parsers are
        # stop-trimmed. This matches non-streaming and text-based parser paths.
        choice_logprobs,
    )


class _StreamPhase(str, Enum):
    OUTSIDE = "outside"
    NAME = "name"
    KEY = "key"
    WAIT_VALUE = "wait_value"
    VALUE = "value"
    BETWEEN_ARGS = "between_args"
    DRAINING = "draining"


class WelmV4StreamingParseError(ValueError):
    """A malformed streamed tool call that can no longer be retracted."""


class WelmV4ToolDetector(BaseFormatDetector):
    """GLM XML tool-call detector driven by WeLM-v4 control-token ids."""

    accepts_token_ids = True
    filter_id_based_stream_stop = staticmethod(filter_id_based_stream_stop)

    def __init__(
        self,
        tokenizer=None,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ):
        super().__init__()
        self.bot_token = "<tool_call>"
        self.eot_token = "</tool_call>"
        self._fallback = Glm4MoeDetector()
        self.tokenizer = None
        self.skip_special_tokens = True
        self.spaces_between_special_tokens = True
        self.bot_id = None
        self.eot_id = None
        self.arg_key_id = None
        self.arg_key_end_id = None
        self.arg_value_id = None
        self.arg_value_end_id = None
        self.id_capable = False
        self.reparse_content_ids: bool = False

        self._normal_ids: List[int] = []
        self._normal_decoder: Optional[_IncrementalDecoder] = None
        self._phase = _StreamPhase.OUTSIDE
        self._active_tool_index: Optional[int] = None
        self._active_block_ids: List[int] = []
        self._name_ids: List[int] = []
        self._key_ids: List[int] = []
        self._separator_ids: List[int] = []
        self._current_key: Optional[str] = None
        self._current_value_type: str = "string"
        self._current_value_types: set = set()
        self._value_undecided = False
        self._current_value_ids: List[int] = []
        self._current_value_decoder: Optional[_IncrementalDecoder] = None
        self._arg_count = 0
        self._pending_post_tool_whitespace_ids: Optional[List[int]] = None
        self.skip_unstreamed_arg_backfill = False
        self.configure_tokenizer(
            tokenizer,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    def configure_tokenizer(
        self,
        tokenizer,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self._normal_ids = []
        self._normal_decoder = None
        self._pending_post_tool_whitespace_ids = None
        self.skip_unstreamed_arg_backfill = False
        self._reset_streaming_tool_state()
        self.bot_id = _resolve_token_id(tokenizer, self.bot_token)
        self.eot_id = _resolve_token_id(tokenizer, self.eot_token)
        self.arg_key_id = _resolve_token_id(tokenizer, "<arg_key>")
        self.arg_key_end_id = _resolve_token_id(tokenizer, "</arg_key>")
        self.arg_value_id = _resolve_token_id(tokenizer, "<arg_value>")
        self.arg_value_end_id = _resolve_token_id(tokenizer, "</arg_value>")
        self.id_capable = all(
            token_id is not None
            for token_id in [
                self.bot_id,
                self.eot_id,
                self.arg_key_id,
                self.arg_key_end_id,
                self.arg_value_id,
                self.arg_value_end_id,
            ]
        )
        self.accepts_token_ids = self.id_capable

    def _reset_streaming_tool_state(self) -> None:
        self._phase = _StreamPhase.OUTSIDE
        self._active_tool_index = None
        self._active_block_ids = []
        self._name_ids = []
        self._key_ids = []
        self._separator_ids = []
        self._current_key = None
        self._current_value_type = "string"
        self._current_value_types = set()
        self._value_undecided = False
        self._current_value_ids = []
        self._current_value_decoder = None
        self._arg_count = 0

    @staticmethod
    def trim_matched_stop(output, finished_reason: Optional[dict], no_stop_trim: bool):
        return trim_matched_stop_for_id_parser(output, finished_reason, no_stop_trim)

    def _new_decoder(self) -> _IncrementalDecoder:
        return _IncrementalDecoder(
            self.tokenizer,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )

    def _decode(
        self,
        ids: List[int],
        *,
        skip_special_tokens: Optional[bool] = None,
    ) -> str:
        if not ids:
            return ""
        kwargs = {
            "skip_special_tokens": (
                self.skip_special_tokens
                if skip_special_tokens is None
                else skip_special_tokens
            ),
            "spaces_between_special_tokens": self.spaces_between_special_tokens,
        }
        return self.tokenizer.decode(ids, **kwargs)

    def _reset_after_streaming_error(self) -> None:
        self.skip_unstreamed_arg_backfill = True
        self.current_tool_name_sent = False
        self._normal_ids = []
        self._pending_post_tool_whitespace_ids = None
        self._normal_decoder = self._new_decoder()
        self._reset_streaming_tool_state()

    def _streaming_error_result(
        self,
        *,
        index: int,
        ids: List[int],
        in_tool_before: bool,
        active_block_before: List[int],
        normal_chunks: List[str],
        calls: List[ToolCallItem],
        active_call_base: int,
        new_text: str,
    ) -> StreamingParseResult:
        pending_ids = self._pending_post_tool_whitespace_ids or []
        self._pending_post_tool_whitespace_ids = None
        if in_tool_before:
            calls = calls[:active_call_base]
            raw_ids = pending_ids + [
                *([self.bot_id] if self.bot_id is not None else []),
                *active_block_before,
                *ids[index:],
            ]
        else:
            raw_ids = pending_ids + ids[index:]

        try:
            raw_text = self._decode(raw_ids, skip_special_tokens=False)
        except Exception:
            logger.exception("Failed to decode WeLM-v4 parser fallback token ids")
            # new_text is the authoritative text for this entire input chunk.
            # Do not prepend already decoded chunks or their prefix is duplicated.
            fallback_text = new_text
        else:
            fallback_text = "".join(normal_chunks) + raw_text
        self._reset_after_streaming_error()
        return StreamingParseResult(normal_text=fallback_text, calls=calls)

    def has_tool_call(self, text: str, token_ids: Optional[List[int]] = None) -> bool:
        if self.id_capable and token_ids is not None:
            return self.bot_id in (token_ids or [])
        return self._fallback.has_tool_call(text)

    def detect_and_parse(
        self,
        text: str,
        tools: List[Tool],
        token_ids: Optional[List[int]] = None,
    ) -> StreamingParseResult:
        if not self.id_capable or token_ids is None:
            return self._fallback.detect_and_parse(text, tools)

        ids = list(token_ids or [])
        try:
            first = ids.index(self.bot_id)
        except ValueError:
            return StreamingParseResult(normal_text=self._decode(ids), calls=[])

        normal_ids = ids[:first]
        calls: List[ToolCallItem] = []
        pending_gap_ids: List[int] = []
        i = first
        while i < len(ids):
            try:
                close = ids.index(self.eot_id, i + 1)
            except ValueError:
                normal_ids.extend(pending_gap_ids)
                normal_ids.extend(ids[i:])
                break
            block = ids[i + 1 : close]
            try:
                parsed_calls = self._parse_block_ids(block, tools)
            except Exception as e:
                logger.error("WeLM-v4 tool block parse error: %s", e, exc_info=True)
                parsed_calls = None
            if parsed_calls:
                pending_gap_ids.clear()
                calls.extend(parsed_calls)
            else:
                normal_ids.extend(pending_gap_ids)
                pending_gap_ids.clear()
                normal_ids.extend(ids[i : close + 1])
            i = close + 1
            try:
                next_tool = ids.index(self.bot_id, i)
            except ValueError:
                next_tool = len(ids)
            gap_ids = ids[i:next_tool]
            if parsed_calls and not self._decode(gap_ids).strip():
                if next_tool < len(ids):
                    pending_gap_ids = list(gap_ids)
            else:
                normal_ids.extend(gap_ids)
            i = next_tool

        return StreamingParseResult(normal_text=self._decode(normal_ids), calls=calls)

    def _parse_block_ids(
        self, block: List[int], tools: List[Tool]
    ) -> Optional[List[ToolCallItem]]:
        markers = {
            self.bot_id,
            self.arg_key_id,
            self.arg_key_end_id,
            self.arg_value_id,
            self.arg_value_end_id,
        }

        def next_marker(start: int) -> int:
            return next(
                (
                    index
                    for index in range(start, len(block))
                    if block[index] in markers
                ),
                len(block),
            )

        name_end = next_marker(0)
        if name_end < len(block) and block[name_end] != self.arg_key_id:
            return None
        func_name = self._decode(block[:name_end]).removesuffix("\n")

        pairs: List[tuple] = []
        j = name_end
        while (
            self.arg_key_id is not None
            and j < len(block)
            and block[j] == self.arg_key_id
        ):
            key_end = next_marker(j + 1)
            if key_end == len(block) or block[key_end] != self.arg_key_end_id:
                return None
            val_start = next_marker(key_end + 1)
            if (
                val_start == len(block)
                or block[val_start] != self.arg_value_id
                or self._decode(block[key_end + 1 : val_start]).strip()
            ):
                return None
            val_end = next_marker(val_start + 1)
            if val_end == len(block) or block[val_end] != self.arg_value_end_id:
                return None
            key = self._decode(block[j + 1 : key_end])
            value = self._decode(block[val_start + 1 : val_end])
            pairs.append((key, value))
            j = val_end + 1
            next_key = next_marker(j)
            if self._decode(block[j:next_key]).strip():
                return None
            j = next_key
            if j < len(block) and block[j] != self.arg_key_id:
                return None

        arguments = self._parse_argument_pairs(pairs, func_name, tools)
        return self.parse_base_json({"name": func_name, "parameters": arguments}, tools)

    def _parse_argument_pairs(
        self, pairs: List[tuple], func_name: str, tools: List[Tool]
    ):
        arguments = {}
        for key, value in pairs:
            arg_type = get_argument_type(func_name, key, tools)
            if arg_type == "string":
                # A union that also permits a non-string type may reinterpret a
                # bare literal; a pure string keeps the rendered text verbatim.
                coerced, was_coerced = coerce_union_literal(
                    value, get_argument_types(func_name, key, tools)
                )
                arguments[key] = coerced if was_coerced else value
                continue
            parsed_value, is_valid_json = parse_arguments(value, arg_type)
            if arg_type is None:
                arguments[key] = (
                    parsed_value
                    if is_valid_json and isinstance(parsed_value, (dict, list))
                    else value
                )
            else:
                arguments[key] = parsed_value if is_valid_json else value
        return arguments

    def _is_known_or_forwarded_tool(self, func_name: str, tools: List[Tool]) -> bool:
        if envs.SGLANG_FORWARD_UNKNOWN_TOOLS.get():
            return True
        return func_name in self._get_tool_indices(tools)

    def _ensure_tool_tracking(self, tool_index: int, name: str) -> None:
        while len(self.streamed_args_for_tool) <= tool_index:
            self.streamed_args_for_tool.append("")
        while len(self.prev_tool_call_arr) <= tool_index:
            self.prev_tool_call_arr.append({})
        self.prev_tool_call_arr[tool_index] = {"name": name, "arguments": {}}

    def _append_arg_delta(self, calls: List[ToolCallItem], delta: str) -> None:
        if not delta or self._active_tool_index is None:
            return
        tool_index = self._active_tool_index
        calls.append(ToolCallItem(tool_index=tool_index, name=None, parameters=delta))
        self.streamed_args_for_tool[tool_index] += delta

    def _try_emit_name(self, tools: List[Tool], calls: List[ToolCallItem]) -> bool:
        func_name = self._decode(self._name_ids).removesuffix("\n")
        if not func_name or not self._is_known_or_forwarded_tool(func_name, tools):
            self._phase = _StreamPhase.DRAINING
            return False

        self.current_tool_id += 1
        self._active_tool_index = self.current_tool_id
        self.current_tool_name_sent = True
        self._pending_post_tool_whitespace_ids = None
        self._ensure_tool_tracking(self._active_tool_index, func_name)
        calls.append(
            ToolCallItem(
                tool_index=self._active_tool_index,
                name=func_name,
                parameters="",
            )
        )
        return True

    def _current_tool_name(self) -> Optional[str]:
        if self._active_tool_index is None:
            return None
        if self._active_tool_index >= len(self.prev_tool_call_arr):
            return None
        return self.prev_tool_call_arr[self._active_tool_index].get("name")

    def _structural_token_ids(self) -> set[int]:
        return {
            self.bot_id,
            self.eot_id,
            self.arg_key_id,
            self.arg_key_end_id,
            self.arg_value_id,
            self.arg_value_end_id,
        }

    def _separator_is_whitespace(self) -> bool:
        return not self._decode(self._separator_ids).strip()

    def _malformed_stream(self, message: str) -> None:
        if not self.current_tool_name_sent:
            self._phase = _StreamPhase.DRAINING
            return

        self._reset_after_streaming_error()
        raise WelmV4StreamingParseError(message)

    def _begin_arg_value(self, tools: List[Tool], calls: List[ToolCallItem]) -> None:
        if self._active_tool_index is None or self._current_key is None:
            self._phase = _StreamPhase.DRAINING
            return

        func_name = self._current_tool_name() or ""
        self._current_value_type = (
            get_argument_type(func_name, self._current_key, tools) or "string"
        )
        self._current_value_types = get_argument_types(
            func_name, self._current_key, tools
        )
        self._value_undecided = self._current_value_type == "string" and bool(
            self._current_value_types - {"string"}
        )
        self._current_value_ids = []
        self._current_value_decoder = self._new_decoder()

        prefix = "{" if self._arg_count == 0 else ", "
        prefix += json.dumps(self._current_key, ensure_ascii=False) + ": "
        # An undecided union may still resolve to a bare literal, so the opening
        # quote is withheld until the value is known to be a string.
        if self._current_value_type == "string" and not self._value_undecided:
            prefix += '"'
        self._append_arg_delta(calls, prefix)
        self._phase = _StreamPhase.VALUE

    def _append_value_span(
        self, token_ids: List[int], calls: List[ToolCallItem]
    ) -> None:
        if self._current_value_decoder is None:
            self._current_value_decoder = self._new_decoder()
        self._current_value_ids.extend(token_ids)
        delta = self._current_value_decoder.step(self._current_value_ids)
        if not delta:
            return
        if self._value_undecided:
            buffered = self._current_value_decoder.emitted
            if still_possible(buffered, self._current_value_types):
                return
            # No literal candidate survives this prefix: commit to string and
            # flush everything held back so far, then resume token-by-token.
            self._value_undecided = False
            self._append_arg_delta(
                calls, '"' + json.dumps(buffered, ensure_ascii=False)[1:-1]
            )
            return
        if self._current_value_type == "string":
            delta = json.dumps(delta, ensure_ascii=False)[1:-1]
        self._append_arg_delta(calls, delta)

    def _end_arg_value(self, calls: List[ToolCallItem]) -> None:
        if self._active_tool_index is None:
            self._phase = _StreamPhase.DRAINING
            return
        pending = self._current_value_decoder.flush(self._current_value_ids)
        if self._value_undecided:
            raw_value = self._current_value_decoder.emitted
            coerced, was_coerced = coerce_union_literal(
                raw_value, self._current_value_types
            )
            self._append_arg_delta(
                calls,
                json.dumps(coerced if was_coerced else raw_value, ensure_ascii=False),
            )
        else:
            if pending:
                if self._current_value_type == "string":
                    pending = json.dumps(pending, ensure_ascii=False)[1:-1]
                self._append_arg_delta(calls, pending)
            if self._current_value_type == "string":
                self._append_arg_delta(calls, '"')
        self._arg_count += 1
        self._current_key = None
        self._current_value_type = "string"
        self._current_value_types = set()
        self._value_undecided = False
        self._current_value_ids = []
        self._current_value_decoder = None
        self._separator_ids = []
        self._phase = _StreamPhase.BETWEEN_ARGS

    def _finalize_active_tool_with_tools(
        self, tools: List[Tool], calls: List[ToolCallItem]
    ) -> None:
        if self._active_tool_index is None:
            self._reset_streaming_tool_state()
            return

        tool_index = self._active_tool_index
        if self._arg_count == 0:
            self._append_arg_delta(calls, "{}")
        else:
            self._append_arg_delta(calls, "}")

        parsed_calls = self._parse_block_ids(self._active_block_ids, tools)
        if parsed_calls:
            try:
                final_args = json.loads(parsed_calls[0].parameters)
                self.prev_tool_call_arr[tool_index]["arguments"] = final_args
            except json.JSONDecodeError:
                logger.debug("Failed to parse final WeLM-v4 arguments JSON")
        else:
            try:
                self.prev_tool_call_arr[tool_index]["arguments"] = json.loads(
                    self.streamed_args_for_tool[tool_index]
                )
            except json.JSONDecodeError:
                logger.debug("Failed to parse streamed WeLM-v4 arguments JSON")

        self.current_tool_name_sent = False
        self._reset_streaming_tool_state()
        self._pending_post_tool_whitespace_ids = []

    def _handle_tool_start(self) -> None:
        self._reset_streaming_tool_state()
        self._phase = _StreamPhase.NAME
        self._active_block_ids = []

    def _handle_outside_span(
        self, token_ids: List[int], normal_chunks: List[str]
    ) -> None:
        if not token_ids:
            return
        if self._pending_post_tool_whitespace_ids is not None:
            candidate_ids = self._pending_post_tool_whitespace_ids + token_ids
            if not self._decode(candidate_ids).strip():
                self._pending_post_tool_whitespace_ids = candidate_ids
                return
            token_ids = candidate_ids
            self._pending_post_tool_whitespace_ids = None
        self._normal_ids.extend(token_ids)
        normal_chunks.append(self._normal_decoder.step(self._normal_ids))

    def _handle_outside_token(self, tid: int, normal_chunks: List[str]) -> None:
        if tid == self.bot_id:
            self._handle_tool_start()
            return

        if self.skip_unstreamed_arg_backfill and tid in {
            self.eot_id,
            self.arg_key_id,
            self.arg_key_end_id,
            self.arg_value_id,
            self.arg_value_end_id,
        }:
            normal_chunks.append(self._decode([tid], skip_special_tokens=False))
            return

        self._handle_outside_span([tid], normal_chunks)

    def _emit_active_tool_as_text(self, normal_chunks: List[str]) -> None:
        pending_ids = self._pending_post_tool_whitespace_ids or []
        self._pending_post_tool_whitespace_ids = None
        raw_ids = pending_ids + [self.bot_id, *self._active_block_ids, self.eot_id]
        normal_chunks.append(
            self._decode(
                raw_ids,
                skip_special_tokens=False,
            )
        )
        self._reset_streaming_tool_state()

    def _handle_draining_token(self, tid: int, normal_chunks: List[str]) -> None:
        if tid == self.eot_id:
            if self._active_tool_index is None:
                self._emit_active_tool_as_text(normal_chunks)
            else:
                self._reset_streaming_tool_state()

    def _handle_name_token(
        self,
        tid: int,
        tools: List[Tool],
        calls: List[ToolCallItem],
        normal_chunks: List[str],
    ) -> None:
        if tid == self.arg_key_id:
            if self._try_emit_name(tools, calls):
                self._phase = _StreamPhase.KEY
                self._key_ids = []
            return

        if tid == self.eot_id:
            if self._try_emit_name(tools, calls):
                self._finalize_active_tool_with_tools(tools, calls)
            else:
                self._emit_active_tool_as_text(normal_chunks)
            return

        if tid in self._structural_token_ids():
            self._malformed_stream(
                "Malformed WeLM-v4 tool call: unexpected control token in tool name"
            )
            return

        self._name_ids.append(tid)

    def _handle_key_token(self, tid: int) -> None:
        if tid == self.arg_key_end_id:
            self._current_key = self._decode(self._key_ids)
            self._key_ids = []
            self._separator_ids = []
            self._phase = _StreamPhase.WAIT_VALUE
            return

        if tid in self._structural_token_ids():
            self._malformed_stream(
                "Malformed WeLM-v4 tool call: argument key is not closed"
            )
            return

        self._key_ids.append(tid)

    def _handle_wait_value_token(
        self, tid: int, tools: List[Tool], calls: List[ToolCallItem]
    ) -> None:
        if tid == self.arg_value_id:
            if not self._separator_is_whitespace():
                self._malformed_stream(
                    "Malformed WeLM-v4 tool call: non-whitespace text before argument value"
                )
                return
            self._begin_arg_value(tools, calls)
            return

        if tid in self._structural_token_ids():
            self._malformed_stream(
                "Malformed WeLM-v4 tool call: argument value is missing"
            )
            return

        self._separator_ids.append(tid)

    def _handle_between_args_token(
        self,
        tid: int,
        tools: List[Tool],
        calls: List[ToolCallItem],
    ) -> None:
        if tid == self.arg_key_id:
            if not self._separator_is_whitespace():
                self._malformed_stream(
                    "Malformed WeLM-v4 tool call: non-whitespace text between arguments"
                )
                return
            self._separator_ids = []
            self._phase = _StreamPhase.KEY
            self._key_ids = []
            return

        if tid == self.eot_id:
            if not self._separator_is_whitespace():
                self._malformed_stream(
                    "Malformed WeLM-v4 tool call: non-whitespace text after arguments"
                )
                return
            self._finalize_active_tool_with_tools(tools, calls)
            return

        if tid in self._structural_token_ids():
            self._malformed_stream(
                "Malformed WeLM-v4 tool call: unexpected control token between arguments"
            )
            return

        self._separator_ids.append(tid)

    def _handle_value_token(self, tid: int, calls: List[ToolCallItem]) -> None:
        if tid == self.arg_value_end_id:
            self._end_arg_value(calls)
            return

        if tid in self._structural_token_ids():
            self._malformed_stream(
                "Malformed WeLM-v4 tool call: argument value is not closed"
            )
            return

        self._append_value_span([tid], calls)

    def _dispatch_stream_span(
        self, token_ids: List[int], calls: List[ToolCallItem], normal_chunks: List[str]
    ) -> None:
        if self._phase is _StreamPhase.OUTSIDE:
            self._handle_outside_span(token_ids, normal_chunks)
        elif self._phase is _StreamPhase.NAME:
            self._name_ids.extend(token_ids)
        elif self._phase is _StreamPhase.KEY:
            self._key_ids.extend(token_ids)
        elif self._phase in {_StreamPhase.WAIT_VALUE, _StreamPhase.BETWEEN_ARGS}:
            self._separator_ids.extend(token_ids)
        elif self._phase is _StreamPhase.VALUE:
            self._append_value_span(token_ids, calls)

    def _dispatch_stream_token(
        self,
        tid: int,
        tools: List[Tool],
        calls: List[ToolCallItem],
        normal_chunks: List[str],
    ) -> None:
        if self._phase is _StreamPhase.OUTSIDE:
            self._handle_outside_token(tid, normal_chunks)
        elif self._phase is _StreamPhase.DRAINING:
            self._handle_draining_token(tid, normal_chunks)
        elif self._phase is _StreamPhase.NAME:
            self._handle_name_token(tid, tools, calls, normal_chunks)
        elif self._phase is _StreamPhase.KEY:
            self._handle_key_token(tid)
        elif self._phase is _StreamPhase.WAIT_VALUE:
            self._handle_wait_value_token(tid, tools, calls)
        elif self._phase is _StreamPhase.VALUE:
            self._handle_value_token(tid, calls)
        elif self._phase is _StreamPhase.BETWEEN_ARGS:
            self._handle_between_args_token(tid, tools, calls)

    def parse_streaming_increment(
        self,
        new_text: str,
        tools: List[Tool],
        token_ids: Optional[List[int]] = None,
    ) -> StreamingParseResult:
        reparse_content_ids = self.reparse_content_ids
        if not self.id_capable or token_ids is None:
            if reparse_content_ids:
                raise WelmV4StreamingParseError(
                    "WeLM-v4 tool-call parser reparse lost token ids"
                )
            return self._fallback.parse_streaming_increment(new_text, tools)

        if self._normal_decoder is None:
            self._normal_decoder = self._new_decoder()

        normal_chunks: List[str] = []
        calls: List[ToolCallItem] = []

        ids = list(token_ids or [])
        structural_ids = self._structural_token_ids()
        active_call_base = 0
        index = 0
        while index < len(ids):
            is_control = ids[index] in structural_ids
            end = index + 1
            if not is_control:
                while end < len(ids) and ids[end] not in structural_ids:
                    end += 1
            span = ids[index:end]
            in_tool_before = self._phase is not _StreamPhase.OUTSIDE
            active_block_before = list(self._active_block_ids)
            if not in_tool_before and is_control and span[0] == self.bot_id:
                active_call_base = len(calls)
            try:
                if is_control:
                    self._dispatch_stream_token(span[0], tools, calls, normal_chunks)
                else:
                    self._dispatch_stream_span(span, calls, normal_chunks)
            except WelmV4StreamingParseError:
                raise
            except Exception as exc:
                logger.exception("Error in WeLM-v4 token-id streaming tool parser")
                if self.current_tool_name_sent:
                    self._reset_after_streaming_error()
                    raise WelmV4StreamingParseError(
                        "WeLM-v4 tool-call parsing failed after the tool name was streamed"
                    ) from exc
                return self._streaming_error_result(
                    index=index,
                    ids=ids,
                    in_tool_before=in_tool_before,
                    active_block_before=active_block_before,
                    normal_chunks=normal_chunks,
                    calls=calls,
                    active_call_base=active_call_base,
                    new_text=new_text,
                )
            if in_tool_before and not (is_control and span[0] == self.eot_id):
                self._active_block_ids.extend(span)
            index = end

        return StreamingParseResult(normal_text="".join(normal_chunks), calls=calls)

    def finish_content_id_reparse(self) -> str:
        if self._phase is _StreamPhase.OUTSIDE:
            self._pending_post_tool_whitespace_ids = None
            return ""
        if self.current_tool_name_sent:
            self._reset_after_streaming_error()
            raise WelmV4StreamingParseError(
                "WeLM-v4 tool call ended after irreversible output was streamed"
            )

        try:
            pending_ids = self._pending_post_tool_whitespace_ids or []
            self._pending_post_tool_whitespace_ids = None
            raw_ids = pending_ids + [self.bot_id, *self._active_block_ids]
            raw_text = self._decode(
                raw_ids,
                skip_special_tokens=False,
            )
        except Exception as exc:
            self._reset_after_streaming_error()
            raise WelmV4StreamingParseError(
                "WeLM-v4 parser could not recover incomplete content token ids"
            ) from exc
        self._reset_after_streaming_error()
        return raw_text

    def supports_structural_tag(self) -> bool:
        return self._fallback.supports_structural_tag()

    def get_structural_tag(self, *args, **kwargs):
        return self._fallback.get_structural_tag(*args, **kwargs)

    def structure_info(self):
        return self._fallback.structure_info()
