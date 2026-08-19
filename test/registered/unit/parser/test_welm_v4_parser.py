import json
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.utils import (
    coerce_union_literal,
    infer_json_schema_types,
    infer_type_from_json_schema,
    still_possible,
)
from sglang.srt.function_call.welm_v4_detector import (
    WelmV4StreamingParseError,
    WelmV4ToolDetector,
    filter_id_based_stream_stop,
    trim_matched_stop_for_id_parser,
)
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.parser.welm_v4_detector import WelmV4ReasoningDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="stage-a-test-cpu")


class FakeWelmTokenizer:
    """Tokenizer stub with dedicated ids for WeLM control tokens."""

    CONTROL = {
        "<think>": 1001,
        "</think>": 1002,
        "<tool_call>": 1003,
        "</tool_call>": 1004,
        "<arg_key>": 1005,
        "</arg_key>": 1006,
        "<arg_value>": 1007,
        "</arg_value>": 1008,
        "<|im_end|>": 1009,
    }
    # Match real ChatML-style tokenizers where only <|im_end|> is special:true.
    SPECIAL_TRUE_IDS = {1009}
    unk_token_id = 0
    eos_token_id = 1009

    def __init__(self):
        self._id2tok = {v: k for k, v in self.CONTROL.items()}

    def convert_tokens_to_ids(self, token):
        return self.CONTROL.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        # Keep ordinary text distinct from added-token ids.
        return [ord(c) for c in text]

    def decode(
        self, ids, skip_special_tokens=False, spaces_between_special_tokens=True
    ):
        out = []
        for i in ids:
            if skip_special_tokens and i in self.SPECIAL_TRUE_IDS:
                continue
            if i in self._id2tok:
                out.append(self._id2tok[i])
            else:
                out.append(chr(i))
        return "".join(out)


class CountingWelmTokenizer(FakeWelmTokenizer):
    def __init__(self):
        super().__init__()
        self.decode_inputs = []

    def decode(
        self, ids, skip_special_tokens=False, spaces_between_special_tokens=True
    ):
        self.decode_inputs.append(list(ids))
        return super().decode(
            ids,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )


class FakeByteFallbackWelmTokenizer(FakeWelmTokenizer):
    BYTE_IDS = {2001: b"\xf0", 2002: b"\x9f", 2003: b"\x99", 2004: b"\x82"}

    def decode(
        self, ids, skip_special_tokens=False, spaces_between_special_tokens=True
    ):
        output = bytearray()
        for token_id in ids:
            if skip_special_tokens and token_id in self.SPECIAL_TRUE_IDS:
                continue
            if token_id in self.BYTE_IDS:
                output.extend(self.BYTE_IDS[token_id])
            elif token_id in self._id2tok:
                output.extend(self._id2tok[token_id].encode())
            else:
                output.extend(chr(token_id).encode())
        return output.decode(errors="replace")


def enc(text):
    return [ord(c) for c in text]


C = FakeWelmTokenizer.CONTROL


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather",
                parameters={
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "number"},
                    },
                    "required": ["city"],
                },
            ),
        ),
    ]


class TestWelmV4Reasoning(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()

    def _det(self, **kw):
        return WelmV4ReasoningDetector(tokenizer=self.tok, **kw)

    def test_non_stream_basic(self):
        det = self._det(force_reasoning=True)
        ids = enc("let me think") + [C["</think>"]] + enc("the answer")
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "let me think")
        self.assertEqual(res.normal_text, "the answer")
        self.assertEqual(det.remaining_token_ids, enc("the answer"))

    def test_non_stream_preserves_answer_whitespace(self):
        det = self._det(force_reasoning=True)
        ids = enc("reason") + [C["</think>"]] + enc(" answer ")

        res = det.detect_and_parse("", ids)

        self.assertEqual(res.normal_text, " answer ")

    def test_non_stream_safe_text_reuse_avoids_answer_decode(self):
        tok = CountingWelmTokenizer()
        answer_ids = enc("long answer text")
        ids = enc("r") + [C["</think>"]] + answer_ids
        reuse = WelmV4ReasoningDetector(tokenizer=tok, force_reasoning=True)
        result = reuse.detect_and_parse("r</think>long answer text", ids)
        self.assertEqual(result.normal_text, "long answer text")
        self.assertFalse(any(call == answer_ids for call in tok.decode_inputs))

        tok.decode_inputs.clear()
        fallback = WelmV4ReasoningDetector(tokenizer=tok, force_reasoning=True)
        result = fallback.detect_and_parse("unaligned text", ids)
        self.assertEqual(result.normal_text, "long answer text")
        self.assertIn(answer_ids, tok.decode_inputs)

    def test_non_stream_strips_leading_think(self):
        det = self._det(force_reasoning=True)
        ids = [C["<think>"]] + enc("hmm") + [C["</think>"]] + enc("done")
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "hmm")
        self.assertEqual(res.normal_text, "done")

    def test_non_stream_lookalike_in_content_is_ignored(self):
        det = self._det(force_reasoning=True)
        ids = (
            enc("discussing the </think> tag here")
            + [C["</think>"]]
            + enc("real answer")
        )
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "discussing the </think> tag here")
        self.assertEqual(res.normal_text, "real answer")

    def test_non_stream_no_end_token_force_reasoning(self):
        det = self._det(force_reasoning=True)
        ids = enc("still thinking")
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "still thinking")
        self.assertEqual(res.normal_text, "")

    def test_thinking_true_false_and_adaptive_outputs(self):
        """Test generated output splits for thinking on/off/adaptive modes."""
        det = self._det(force_reasoning=False)

        res = det.detect_and_parse("", enc("reason") + [C["</think>"]] + enc("answer"))
        self.assertEqual(res.reasoning_text, "reason")
        self.assertEqual(res.normal_text, "answer")

        det = self._det(force_reasoning=False)
        res = det.detect_and_parse("", [C["</think>"]] + enc("answer"))
        self.assertEqual(res.reasoning_text, "")
        self.assertEqual(res.normal_text, "answer")

        det = self._det(force_reasoning=False)
        res = det.detect_and_parse(
            "", [C["<think>"]] + enc("maybe") + [C["</think>"]] + enc("ok")
        )
        self.assertEqual(res.reasoning_text, "maybe")
        self.assertEqual(res.normal_text, "ok")

        det = self._det(force_reasoning=False)
        res = det.detect_and_parse("", [C["</think>"]] + enc("direct"))
        self.assertEqual(res.reasoning_text, "")
        self.assertEqual(res.normal_text, "direct")

    def test_stream_token_by_token(self):
        det = self._det(force_reasoning=True, stream_reasoning=True)
        seq = enc("reason") + [C["</think>"]] + enc("ans")
        reasoning, normal, remaining = "", "", []
        for tid in seq:
            res = det.parse_streaming_increment("", [tid])
            reasoning += res.reasoning_text
            normal += res.normal_text
            remaining += det.remaining_token_ids
        self.assertEqual(reasoning, "reason")
        self.assertEqual(normal, "ans")
        self.assertEqual(remaining, enc("ans"))

    def test_stream_span_decode_and_handoff_chunk_matrix(self):
        seq = enc("reason") + [C["</think>"]] + enc("answer")
        for chunk_size in (1, 2, 4, 8, 16):
            with self.subTest(chunk_size=chunk_size):
                tok = CountingWelmTokenizer()
                det = WelmV4ReasoningDetector(tokenizer=tok, force_reasoning=True)
                det.handoff_content_ids = True
                reasoning, normal, remaining = "", "", []
                for start in range(0, len(seq), chunk_size):
                    result = det.parse_streaming_increment(
                        "",
                        seq[start : start + chunk_size],
                    )
                    reasoning += result.reasoning_text
                    normal += result.normal_text
                    remaining.extend(det.remaining_token_ids or [])
                self.assertEqual((reasoning, normal), ("reason", ""))
                self.assertEqual(remaining, enc("answer"))
                self.assertIsNone(det._answer_decoder)

        tok = CountingWelmTokenizer()
        det = WelmV4ReasoningDetector(tokenizer=tok, force_reasoning=True)
        result = det.parse_streaming_increment("", enc("abcdefghijklmnop"))
        self.assertEqual(result.reasoning_text, "abcdefghijklmnop")
        self.assertLessEqual(len(tok.decode_inputs), 2)

    def test_handoff_preserves_unicode_byte_fallback_for_tool_reparse(self):
        tok = FakeByteFallbackWelmTokenizer()
        reasoning = WelmV4ReasoningDetector(tokenizer=tok, force_reasoning=True)
        tool = WelmV4ToolDetector(tokenizer=tok)
        reasoning.handoff_content_ids = True
        tool.reparse_content_ids = True
        emoji_ids = list(tok.BYTE_IDS)

        first = reasoning.parse_streaming_increment(
            "", enc("reason") + [C["</think>"], emoji_ids[0]]
        )
        first_tool = tool.parse_streaming_increment(
            "", _tools(), reasoning.remaining_token_ids
        )
        second = reasoning.parse_streaming_increment("", emoji_ids[1:] + enc("Z"))
        second_tool = tool.parse_streaming_increment(
            "", _tools(), reasoning.remaining_token_ids
        )

        self.assertEqual(first.reasoning_text, "reason")
        self.assertEqual(second.normal_text, "")
        self.assertEqual(first_tool.normal_text + second_tool.normal_text, "🙂Z")

    def test_random_mtp_handoff_parallel_tools_is_chunk_invariant(self):
        def tool_call(city):
            return (
                [C["<tool_call>"]]
                + enc("get_weather\n")
                + [C["<arg_key>"]]
                + enc("city")
                + [C["</arg_key>"]]
                + [C["<arg_value>"]]
                + enc(city)
                + [C["</arg_value>"], C["</tool_call>"]]
            )

        normal_prefix = "正文中文 "
        normal_between = " literal <tool_call> "
        sequence = (
            enc("reason")
            + [C["</think>"]]
            + enc(normal_prefix)
            + tool_call("Paris")
            + enc(normal_between)
            + tool_call("Tokyo")
            + [C["<|im_end|>"]]
        )

        for seed in range(24):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                reasoning = self._det(force_reasoning=True)
                tool = WelmV4ToolDetector(tokenizer=self.tok)
                reasoning.handoff_content_ids = True
                tool.reparse_content_ids = True
                reasoning_text = ""
                normal_text = ""
                calls = []
                offset = 0
                while offset < len(sequence):
                    size = rng.randint(1, 16)
                    chunk = sequence[offset : offset + size]
                    offset += size
                    reasoning_result = reasoning.parse_streaming_increment("", chunk)
                    reasoning_text += reasoning_result.reasoning_text
                    tool_result = tool.parse_streaming_increment(
                        "", _tools(), reasoning.remaining_token_ids
                    )
                    normal_text += tool_result.normal_text
                    calls.extend(tool_result.calls)

                self.assertEqual(tool.finish_content_id_reparse(), "")
                names = [call.name for call in calls if call.name]
                arguments = {}
                for call in calls:
                    if call.name is None:
                        arguments.setdefault(call.tool_index, "")
                        arguments[call.tool_index] += call.parameters
                self.assertEqual(reasoning_text, "reason")
                self.assertEqual(normal_text, normal_prefix + normal_between)
                self.assertEqual(names, ["get_weather", "get_weather"])
                self.assertEqual(
                    [
                        json.loads(arguments[index])["city"]
                        for index in sorted(arguments)
                    ],
                    ["Paris", "Tokyo"],
                )

    def test_stream_lookalike_in_content(self):
        det = self._det(force_reasoning=True, stream_reasoning=True)
        seq = enc("a </think> b") + [C["</think>"]] + enc("final")
        reasoning, normal = "", ""
        for tid in seq:
            res = det.parse_streaming_increment("", [tid])
            reasoning += res.reasoning_text
            normal += res.normal_text
        self.assertEqual(reasoning, "a </think> b")
        self.assertEqual(normal, "final")

    def test_non_stream_trailing_eos_not_in_output(self):
        det = self._det(force_reasoning=True)
        ids = enc("reason") + [C["</think>"]] + enc("answer") + [C["<|im_end|>"]]
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "reason")
        self.assertEqual(res.normal_text, "answer")

    def test_stream_trailing_eos_not_in_output(self):
        det = self._det(force_reasoning=True, stream_reasoning=True)
        seq = enc("r") + [C["</think>"]] + enc("a") + [C["<|im_end|>"]]
        reasoning, normal = "", ""
        for tid in seq:
            res = det.parse_streaming_increment("", [tid])
            reasoning += res.reasoning_text
            normal += res.normal_text
        self.assertEqual(reasoning, "r")
        self.assertEqual(normal, "a")

    def test_stream_no_stream_reasoning_emits_at_end(self):
        det = self._det(force_reasoning=True, stream_reasoning=False)
        seq = enc("reason") + [C["</think>"]] + enc("ans")
        reasoning, normal = "", ""
        for tid in seq:
            res = det.parse_streaming_increment("", [tid])
            reasoning += res.reasoning_text
            normal += res.normal_text
        self.assertEqual(reasoning, "reason")
        self.assertEqual(normal, "ans")

    def test_stream_answer_leading_think_token_does_not_reopen_reasoning(self):
        det = self._det(
            force_reasoning=True, stream_reasoning=True, skip_special_tokens=False
        )
        first = [C["<think>"]] + enc("reason") + [C["</think>"]] + enc("answer")
        second = [C["<think>"]] + enc(" text")
        reasoning, normal = "", ""
        for chunk in [first, second]:
            res = det.parse_streaming_increment("", chunk)
            reasoning += res.reasoning_text
            normal += res.normal_text

        self.assertEqual(reasoning, "reason")
        self.assertEqual(normal, "answer<think> text")

    def test_stream_reasoning_decode_error_falls_back_and_continues(self):
        class FailingTokenizer(FakeWelmTokenizer):
            FAIL_ID = 424242

            def decode(
                self,
                ids,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            ):
                if self.FAIL_ID in ids:
                    raise ValueError("injected decode failure")
                return super().decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                )

        tok = FailingTokenizer()
        det = WelmV4ReasoningDetector(tokenizer=tok, force_reasoning=True)

        first = det.parse_streaming_increment("bad", [tok.FAIL_ID])
        second = det.parse_streaming_increment("", enc(" ok") + [C["</think>"]])
        third = det.parse_streaming_increment("", enc("answer"))

        self.assertEqual(first.reasoning_text, "bad")
        self.assertEqual(second.reasoning_text, " ok")
        self.assertEqual(third.normal_text, "answer")

    def test_stream_answer_decode_error_falls_back_to_normal_text(self):
        class FailingTokenizer(FakeWelmTokenizer):
            FAIL_ID = 424243

            def decode(
                self,
                ids,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            ):
                if self.FAIL_ID in ids:
                    raise ValueError("injected decode failure")
                return super().decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                )

        tok = FailingTokenizer()
        det = WelmV4ReasoningDetector(tokenizer=tok, force_reasoning=True)

        first = det.parse_streaming_increment("", enc("r") + [C["</think>"]])
        second = det.parse_streaming_increment("bad", [tok.FAIL_ID])

        self.assertEqual(first.reasoning_text, "r")
        self.assertEqual(second.normal_text, "bad")
        self.assertEqual(det.remaining_token_ids, [tok.FAIL_ID])


class TestJsonSchemaTypeInference(CustomTestCase):
    def test_local_ref_and_json_pointer_escaping(self):
        root = {
            "$defs": {
                "request/model": {"type": "object"},
                "array~model": {"type": "array"},
            }
        }
        self.assertEqual(
            infer_type_from_json_schema({"$ref": "#/$defs/request~1model"}, root),
            "object",
        )
        self.assertEqual(
            infer_type_from_json_schema({"$ref": "#/$defs/array~0model"}, root),
            "array",
        )

    def test_unresolved_remote_and_cyclic_refs_return_none(self):
        root = {
            "$defs": {
                "cycle_a": {"$ref": "#/$defs/cycle_b"},
                "cycle_b": {"$ref": "#/$defs/cycle_a"},
            }
        }
        for schema in (
            {"$ref": "#/$defs/missing"},
            {"$ref": "https://example.com/schema.json"},
            {"$ref": "#/$defs/cycle_a"},
        ):
            with self.subTest(schema=schema):
                self.assertIsNone(infer_type_from_json_schema(schema, root))
                self.assertEqual(infer_json_schema_types(schema, root), set())

    def test_full_type_set_covers_unions_and_aliases(self):
        root = {"$defs": {"MaybeText": {"type": ["string", "null"]}}}
        cases = [
            ({"type": "string"}, {"string"}),
            ({"type": ["string", "null"]}, {"string", "null"}),
            ({"type": "str"}, {"string"}),
            ({"type": "bool"}, {"boolean"}),
            ({"type": "string", "nullable": True}, {"string", "null"}),
            (
                {"anyOf": [{"type": "string"}, {"type": "number"}]},
                {"string", "number"},
            ),
            (
                {"oneOf": [{"type": "object"}, {"type": "null"}]},
                {"object", "null"},
            ),
            ({"enum": ["a", None, 1]}, {"string", "null", "integer", "number"}),
            ({"$ref": "#/$defs/MaybeText"}, {"string", "null"}),
            ({"properties": {"a": {"type": "string"}}}, {"object"}),
            ({"items": {"type": "string"}}, {"array"}),
            ({}, set()),
        ]
        for schema, expected in cases:
            with self.subTest(schema=schema):
                self.assertEqual(infer_json_schema_types(schema, root), expected)

    def test_db_style_type_names_resolve_to_standard_types(self):
        cases = [
            ("varchar", "string"),
            ("varchar(255)", "string"),
            ("char", "string"),
            ("text", "string"),
            ("uuid", "string"),
            ("timestamp", "string"),
            ("STRING", "string"),
            ("bigint", "integer"),
            ("int32", "integer"),
            ("uint8", "integer"),
            ("float64", "number"),
            ("decimal(10,2)", "number"),
            ("bool", "boolean"),
            ("list[str]", "array"),
            ("tuple", "array"),
            ("dict[str, int]", "object"),
            ("map", "object"),
            ("none", "null"),
        ]
        for raw, expected in cases:
            with self.subTest(type=raw):
                self.assertEqual(infer_type_from_json_schema({"type": raw}), expected)
                self.assertEqual(infer_json_schema_types({"type": raw}), {expected})

    def test_unrecognized_type_names_are_kept_verbatim(self):
        # A prefix only matches on a token boundary, so these must not be
        # mistaken for "int"/"list"/"num".
        for raw in ("internal", "list_price", "numbering", "MyType"):
            with self.subTest(type=raw):
                self.assertEqual(infer_type_from_json_schema({"type": raw}), raw)

    def test_db_style_type_arrays_keep_nullability(self):
        self.assertEqual(
            infer_type_from_json_schema({"type": ["varchar", "none"]}), "string"
        )
        self.assertEqual(
            infer_json_schema_types({"type": ["varchar", "none"]}), {"string", "null"}
        )

    def test_enum_list_wins_over_non_standard_type(self):
        # ``{"type": "enum"}`` is not valid JSON Schema, so the members decide.
        cases = [
            ({"type": "enum", "enum": [1, 2, 3]}, "integer", {"integer", "number"}),
            ({"type": "enum", "enum": ["a", None]}, "string", {"string", "null"}),
            ({"type": "MyType", "enum": ["a", "b"]}, "string", {"string"}),
            # No usable enum list: fall back to resolving the type name.
            ({"type": "enum"}, "string", {"string"}),
            ({"type": "enum", "enum": []}, "string", {"string"}),
            ({"type": "enum", "enum": "abc"}, "string", {"string"}),
        ]
        for schema, expected_type, expected_types in cases:
            with self.subTest(schema=schema):
                self.assertEqual(infer_type_from_json_schema(schema), expected_type)
                self.assertEqual(infer_json_schema_types(schema), expected_types)

    def test_standard_type_still_wins_over_enum_list(self):
        # Declared standard types are authoritative; only bogus ones defer.
        for schema, expected in (
            ({"type": "string", "enum": ["a", None]}, {"string"}),
            ({"type": "integer", "enum": [1, 2]}, {"integer"}),
        ):
            with self.subTest(schema=schema):
                self.assertEqual(infer_json_schema_types(schema), expected)


class TestUnionLiteralCoercion(CustomTestCase):
    def test_pure_string_and_empty_type_set_never_coerce(self):
        for types in (set(), {"string"}):
            for raw in ("null", "true", "5", '{"a":1}'):
                with self.subTest(types=types, raw=raw):
                    self.assertEqual(coerce_union_literal(raw, types), (raw, False))

    def test_special_types_win_over_string(self):
        cases = [
            ("null", {"string", "null"}, None),
            ("true", {"string", "boolean"}, True),
            ("false", {"string", "boolean"}, False),
            ("5", {"string", "number"}, 5),
            ("1e3", {"string", "number"}, 1000.0),
            ('{"a":1}', {"string", "object"}, {"a": 1}),
            ("[1,2]", {"string", "array"}, [1, 2]),
        ]
        for raw, types, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(coerce_union_literal(raw, types), (expected, True))

    def test_strict_matching_keeps_raw_text(self):
        cases = [
            (" null ", {"string", "null"}),  # tojson never pads
            ("NULL", {"string", "null"}),  # tojson emits lowercase
            ("none", {"string", "null"}),
            ("", {"string", "null"}),
            ("nullable", {"string", "null"}),
            ("yes", {"string", "boolean"}),
            ("abc", {"string", "number"}),
            ("1.5", {"string", "integer"}),  # parsed type outside the type set
            ('"hi"', {"string", "null"}),  # already-quoted string keeps its quotes
            ("NaN", {"string", "number"}),  # not valid JSON per RFC 8259
            ("Infinity", {"string", "number"}),
        ]
        for raw, types in cases:
            with self.subTest(raw=raw, types=types):
                self.assertEqual(coerce_union_literal(raw, types), (raw, False))


class TestStreamingLiteralPrefix(CustomTestCase):
    def test_empty_buffer_is_always_undecided(self):
        self.assertTrue(still_possible("", {"string", "null"}))

    def test_prefix_of_a_candidate_keeps_buffering(self):
        for literal, types in (
            ("null", {"string", "null"}),
            ("true", {"string", "boolean"}),
            ("false", {"string", "boolean"}),
        ):
            for length in range(1, len(literal) + 1):
                with self.subTest(buf=literal[:length]):
                    self.assertTrue(still_possible(literal[:length], types))
        for buf in ("-", "1", "1.2", "1e", "1e-3"):
            with self.subTest(buf=buf):
                self.assertTrue(still_possible(buf, {"string", "number"}))
        self.assertTrue(still_possible('{"a"', {"string", "object"}))
        self.assertTrue(still_possible("[1", {"string", "array"}))

    def test_first_diverging_character_ends_buffering(self):
        cases = [
            ("H", {"string", "null"}),
            ("no", {"string", "null"}),
            ("nulla", {"string", "null"}),
            (" ", {"string", "null"}),
            ("true ", {"string", "boolean"}),
            ("1a", {"string", "number"}),
            ("{", {"string", "array"}),
            ("[", {"string", "object"}),
            ("null", {"string", "boolean"}),
        ]
        for buf, types in cases:
            with self.subTest(buf=buf, types=types):
                self.assertFalse(still_possible(buf, types))


class TestWelmV4Tool(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()
        self.tools = _tools()

    def _det(self):
        return WelmV4ToolDetector(tokenizer=self.tok)

    def _call_ids(self, name, args):
        ids = [C["<tool_call>"]] + enc(name)
        for k, v in args:
            ids += enc("\n") + [C["<arg_key>"]] + enc(k) + [C["</arg_key>"]]
            ids += enc("\n") + [C["<arg_value>"]] + enc(v) + [C["</arg_value>"]]
        ids += enc("\n") + [C["</tool_call>"]]
        return ids

    def _stream_ids(self, det, tools, seq, chunk_size=1):
        normal, calls = "", []
        for i in range(0, len(seq), chunk_size):
            res = det.parse_streaming_increment("", tools, seq[i : i + chunk_size])
            normal += res.normal_text
            calls += res.calls
        return normal, calls

    def _merged_args_by_index(self, calls):
        args_by_index = {}
        for call in calls:
            if call.name is None:
                args_by_index.setdefault(call.tool_index, "")
                args_by_index[call.tool_index] += call.parameters
        return args_by_index

    def test_non_stream_basic(self):
        det = self._det()
        ids = enc("Here you go:") + self._call_ids(
            "get_weather", [("city", "Beijing"), ("days", "3")]
        )
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(res.normal_text, "Here you go:")
        self.assertEqual(len(res.calls), 1)
        self.assertEqual(res.calls[0].name, "get_weather")
        args = json.loads(res.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(args["days"], 3)

    def test_local_ref_argument_is_object_in_streaming_and_non_streaming(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="docs_server_docs",
                    parameters={
                        "$defs": {
                            "DynamicModel": {
                                "type": "object",
                                "properties": {"action": {"type": "string"}},
                            }
                        },
                        "type": "object",
                        "properties": {"request": {"$ref": "#/$defs/DynamicModel"}},
                    },
                ),
            )
        ]
        ids = self._call_ids(
            "docs_server_docs",
            [("request", '{"action":"read_content"}')],
        )

        result = self._det().detect_and_parse("", tools, ids)

        arguments = json.loads(result.calls[0].parameters)
        self.assertEqual(arguments["request"], {"action": "read_content"})

        for chunk_size in (1, 4, len(ids)):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream_ids(
                    self._det(), tools, ids, chunk_size=chunk_size
                )
                streamed = json.loads(self._merged_args_by_index(calls)[0])
                self.assertEqual(normal, "")
                self.assertEqual(streamed["request"], {"action": "read_content"})

    def test_non_stream_explicit_string_preserves_json_looking_value(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="string_tool",
                    parameters={
                        "type": "object",
                        "properties": {"request": {"type": "string"}},
                    },
                ),
            )
        ]
        raw_value = '{"action":"read_content"}'
        ids = self._call_ids("string_tool", [("request", raw_value)])

        result = self._det().detect_and_parse("", tools, ids)

        arguments = json.loads(result.calls[0].parameters)
        self.assertEqual(arguments["request"], raw_value)

    def test_non_stream_unresolved_refs_use_conservative_fallback(self):
        cases = [
            ({"$ref": "#/$defs/missing"}, {}),
            ({"$ref": "https://example.com/schema.json"}, {}),
            (
                {"$ref": "#/$defs/cycle"},
                {"$defs": {"cycle": {"$ref": "#/$defs/cycle"}}},
            ),
        ]
        for property_schema, root_extra in cases:
            with self.subTest(property_schema=property_schema):
                parameters = {
                    **root_extra,
                    "type": "object",
                    "properties": {
                        "payload": property_schema,
                        "scalar": property_schema,
                    },
                }
                tools = [
                    Tool(
                        type="function",
                        function=Function(name="fallback_tool", parameters=parameters),
                    )
                ]
                ids = self._call_ids(
                    "fallback_tool",
                    [("payload", '{"ok":true}'), ("scalar", "123")],
                )

                result = self._det().detect_and_parse("", tools, ids)

                arguments = json.loads(result.calls[0].parameters)
                self.assertEqual(arguments["payload"], {"ok": True})
                self.assertEqual(arguments["scalar"], "123")

    def test_stream_plain_span_uses_constant_decode_calls(self):
        tok = CountingWelmTokenizer()
        det = WelmV4ToolDetector(tokenizer=tok)
        result = det.parse_streaming_increment("", self.tools, enc("abcdefghijklmnop"))
        self.assertEqual(result.normal_text, "abcdefghijklmnop")
        self.assertLessEqual(len(tok.decode_inputs), 2)

    def test_non_stream_incomplete_tool_call_remains_content(self):
        incomplete = [C["<tool_call>"]] + enc("get_weather")
        detector = self._det()
        result = detector.detect_and_parse("", self.tools, incomplete)
        self.assertEqual(result.normal_text, "<tool_call>get_weather")

    def test_reparse_malformed_before_name_recovers_actual_ids(self):
        malformed = (
            [C["<tool_call>"], C["<arg_value>"]] + enc("junk") + [C["</tool_call>"]]
        )
        detector = self._det()
        detector.reparse_content_ids = True
        result = detector.parse_streaming_increment("", self.tools, malformed)

        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, self.tok.decode(malformed))
        self.assertEqual(detector.finish_content_id_reparse(), "")

    def test_reparse_closed_unknown_tool_recovers_actual_ids(self):
        invalid = [C["<tool_call>"]] + enc("unknown_tool") + [C["</tool_call>"]]
        detector = self._det()
        detector.reparse_content_ids = True
        result = detector.parse_streaming_increment("", self.tools, invalid)

        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, self.tok.decode(invalid))
        self.assertEqual(detector.finish_content_id_reparse(), "")

    def test_reparse_incomplete_before_name_does_not_fabricate_tool_end(self):
        incomplete = [C["<tool_call>"]] + enc("unknown_tool")
        det = self._det()
        det.reparse_content_ids = True
        result = det.parse_streaming_increment("", self.tools, incomplete)

        self.assertEqual(result.normal_text, "")
        recovered = det.finish_content_id_reparse()
        self.assertEqual(recovered, self.tok.decode(incomplete))
        self.assertNotIn("</tool_call>", recovered)

    def test_reparse_incomplete_after_name_is_irreversible(self):
        incomplete = (
            [C["<tool_call>"]] + enc("get_weather") + [C["<arg_key>"]] + enc("city")
        )
        det = self._det()
        det.reparse_content_ids = True
        result = det.parse_streaming_increment("", self.tools, incomplete)

        self.assertEqual([call.name for call in result.calls], ["get_weather"])
        with self.assertRaises(WelmV4StreamingParseError):
            det.finish_content_id_reparse()

    def test_reparse_incomplete_recovery_decode_error_is_structured(self):
        class FailingRawDecodeTokenizer(FakeWelmTokenizer):
            def decode(
                self,
                ids,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            ):
                if not skip_special_tokens and ids[:1] == [C["<tool_call>"]]:
                    raise RuntimeError("injected raw decode failure")
                return super().decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                )

        det = WelmV4ToolDetector(tokenizer=FailingRawDecodeTokenizer())
        det.reparse_content_ids = True
        det.parse_streaming_increment(
            "", self.tools, [C["<tool_call>"]] + enc("unknown_tool")
        )

        with self.assertRaises(WelmV4StreamingParseError):
            det.finish_content_id_reparse()

    def test_reparse_orphan_control_ids_remain_content(self):
        for orphan, expected in (
            (C["<arg_key>"], "<arg_key>"),
            (C["</tool_call>"], "</tool_call>"),
        ):
            detector = self._det()
            result = detector.detect_and_parse("", self.tools, [orphan])
            self.assertEqual(result.normal_text, expected)
            detector = self._det()
            detector.reparse_content_ids = True
            result = detector.parse_streaming_increment("", self.tools, [orphan])
            self.assertEqual(result.normal_text, expected)

    def test_non_stream_lookalike_in_content_is_ignored(self):
        det = self._det()
        ids = enc("To call a tool you write <tool_call> like this.")
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(res.calls, [])
        self.assertEqual(
            res.normal_text, "To call a tool you write <tool_call> like this."
        )

    def test_non_stream_multiple_calls(self):
        det = self._det()
        ids = (
            self._call_ids("get_weather", [("city", "A")])
            + enc("\n")
            + self._call_ids("get_weather", [("city", "B")])
        )
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(len(res.calls), 2)
        self.assertEqual(json.loads(res.calls[0].parameters)["city"], "A")
        self.assertEqual(json.loads(res.calls[1].parameters)["city"], "B")

    def test_parallel_tool_separator_is_not_content(self):
        seq = (
            self._call_ids("get_weather", [("city", "A")])
            + enc("\n")
            + self._call_ids("get_weather", [("city", "B")])
        )

        non_stream = self._det().detect_and_parse("", self.tools, seq)
        self.assertEqual(non_stream.normal_text, "")
        self.assertEqual(
            [json.loads(call.parameters)["city"] for call in non_stream.calls],
            ["A", "B"],
        )

        for chunk_size in (1, 2, 7, len(seq)):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream_ids(
                    self._det(), self.tools, seq, chunk_size=chunk_size
                )
                self.assertEqual(normal, "")
                self.assertEqual(
                    [call.name for call in calls if call.name],
                    ["get_weather", "get_weather"],
                )
                args_by_index = self._merged_args_by_index(calls)
                self.assertEqual(
                    [
                        json.loads(args_by_index[index])["city"]
                        for index in sorted(args_by_index)
                    ],
                    ["A", "B"],
                )

    def test_post_tool_real_text_preserves_separator(self):
        seq = self._call_ids("get_weather", [("city", "A")]) + enc("\nDone")

        non_stream = self._det().detect_and_parse("", self.tools, seq)
        self.assertEqual(non_stream.normal_text, "\nDone")

        for chunk_size in (1, 3, len(seq)):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream_ids(
                    self._det(), self.tools, seq, chunk_size=chunk_size
                )
                self.assertEqual(normal, "\nDone")
                self.assertEqual(
                    [call.name for call in calls if call.name], ["get_weather"]
                )

    def test_prefix_is_preserved_while_inter_tool_separator_is_dropped(self):
        prefix = "prefix\n"
        seq = (
            enc(prefix)
            + self._call_ids("get_weather", [("city", "A")])
            + enc("\n")
            + self._call_ids("get_weather", [("city", "B")])
        )

        non_stream = self._det().detect_and_parse("", self.tools, seq)
        self.assertEqual(non_stream.normal_text, prefix)

        for chunk_size in (1, 5, len(seq)):
            with self.subTest(chunk_size=chunk_size):
                normal, calls = self._stream_ids(
                    self._det(), self.tools, seq, chunk_size=chunk_size
                )
                self.assertEqual(normal, prefix)
                self.assertEqual(
                    [call.name for call in calls if call.name],
                    ["get_weather", "get_weather"],
                )

    def test_post_tool_separator_is_preserved_for_fallback_content(self):
        fallback_cases = {
            "unknown": self._call_ids("python", [("code", "1")]),
            "malformed": (
                [C["<tool_call>"], C["<arg_value>"]] + enc("junk") + [C["</tool_call>"]]
            ),
            "incomplete": [C["<tool_call>"]] + enc("unknown_tool"),
        }
        first_call = self._call_ids("get_weather", [("city", "A")])

        for name, fallback_ids in fallback_cases.items():
            seq = first_call + enc("\n") + fallback_ids
            expected = "\n" + self.tok.decode(fallback_ids)
            with self.subTest(mode="non_stream", case=name):
                result = self._det().detect_and_parse("", self.tools, seq)
                self.assertEqual(result.normal_text, expected)
                self.assertEqual(len(result.calls), 1)

            for chunk_size in (1, len(seq)):
                with self.subTest(mode="stream", case=name, chunk_size=chunk_size):
                    detector = self._det()
                    detector.reparse_content_ids = True
                    normal, calls = self._stream_ids(
                        detector, self.tools, seq, chunk_size=chunk_size
                    )
                    if name == "incomplete":
                        normal += detector.finish_content_id_reparse()
                    self.assertEqual(normal, expected)
                    self.assertEqual(
                        [call.name for call in calls if call.name], ["get_weather"]
                    )

    def test_trailing_post_tool_whitespace_is_dropped(self):
        first_call = self._call_ids("get_weather", [("city", "A")])

        for suffix in ([C["<|im_end|>"]], enc("\n") + [C["<|im_end|>"]]):
            seq = first_call + suffix
            with self.subTest(mode="non_stream", suffix=suffix):
                result = self._det().detect_and_parse("", self.tools, seq)
                self.assertEqual(result.normal_text, "")
                self.assertEqual(len(result.calls), 1)

            for chunk_size in (1, len(seq)):
                with self.subTest(mode="stream", suffix=suffix, chunk_size=chunk_size):
                    detector = self._det()
                    normal, calls = self._stream_ids(
                        detector, self.tools, seq, chunk_size=chunk_size
                    )
                    self.assertEqual(normal, "")
                    self.assertEqual(
                        [call.name for call in calls if call.name], ["get_weather"]
                    )
                    self.assertEqual(detector._normal_ids, [])

    def test_non_stream_preserves_text_around_tool_calls(self):
        det = self._det()
        ids = (
            enc(" before ")
            + self._call_ids("get_weather", [("city", "A")])
            + enc(" between ")
            + self._call_ids("get_weather", [("city", "B")])
            + enc(" after ")
        )

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(res.normal_text, " before  between  after ")
        self.assertEqual(len(res.calls), 2)

    def test_non_stream_incomplete_value_falls_back_to_content(self):
        det = self._det()
        ids = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Beijing")
            + [C["</tool_call>"]]
        )

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(res.calls, [])
        self.assertEqual(res.normal_text, self.tok.decode(ids))

    def test_non_stream_does_not_parse_tool_without_tool_end(self):
        det = self._det()
        ids = self._call_ids("get_weather", [("city", "Paris")])
        ids.pop()

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(res.calls, [])
        self.assertEqual(res.normal_text, self.tok.decode(ids))

    def test_non_stream_incomplete_value_before_next_arg_falls_back_to_content(self):
        det = self._det()
        ids = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Beijing")
            + [C["<arg_key>"]]
            + enc("days")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("3")
            + [C["</arg_value>"], C["</tool_call>"]]
        )

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(res.calls, [])
        self.assertEqual(res.normal_text, self.tok.decode(ids))

    def test_argument_key_whitespace_is_preserved(self):
        det = self._det()
        ids = self._call_ids("get_weather", [(" city ", "Beijing")])

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(json.loads(res.calls[0].parameters), {" city ": "Beijing"})

    def test_non_stream_string_value_is_not_rewritten(self):
        det = self._det()
        ids = self._call_ids("get_weather", [("city", "true")])

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(json.loads(res.calls[0].parameters), {"city": "true"})

    def test_non_stream_unknown_tool_falls_back_to_content(self):
        det = self._det()
        ids = enc("prefix") + self._call_ids("python", [("code", "1")]) + enc("suffix")

        res = det.detect_and_parse("", self.tools, ids)

        self.assertEqual(res.calls, [])
        self.assertEqual(res.normal_text, self.tok.decode(ids))

    def test_stream_token_by_token(self):
        det = self._det()
        seq = enc("ok") + self._call_ids("get_weather", [("city", "Paris")])
        normal, calls = "", []
        for tid in seq:
            res = det.parse_streaming_increment("", self.tools, [tid])
            normal += res.normal_text
            calls += res.calls
        self.assertEqual(normal, "ok")
        names = [c.name for c in calls if c.name]
        self.assertIn("get_weather", names)
        merged = "".join(c.parameters for c in calls if c.parameters)
        self.assertIn("Paris", merged)

    def test_stream_parser_error_falls_back_to_exact_raw_block(self):
        det = self._det()
        original_dispatch = det._dispatch_stream_span

        def dispatch_or_fail(token_ids, calls, normal_chunks):
            if ord("!") in token_ids:
                raise RuntimeError("injected parser failure")
            return original_dispatch(token_ids, calls, normal_chunks)

        det._dispatch_stream_span = dispatch_or_fail
        seq = [C["<tool_call>"]] + enc("get_weather!")

        res = det.parse_streaming_increment("", self.tools, seq)

        self.assertEqual(res.calls, [])
        self.assertEqual(res.normal_text, self.tok.decode(seq))
        self.assertNotIn("</tool_call>", res.normal_text)
        self.assertTrue(det.skip_unstreamed_arg_backfill)

    def test_stream_parser_and_fallback_decode_error_do_not_duplicate_text(self):
        class FailingTokenizer(FakeWelmTokenizer):
            FAIL_ID = 424244

            def decode(
                self,
                ids,
                skip_special_tokens=False,
                spaces_between_special_tokens=True,
            ):
                if self.FAIL_ID in ids:
                    raise ValueError("injected decode failure")
                return super().decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                )

        tok = FailingTokenizer()
        det = WelmV4ToolDetector(tokenizer=tok)
        original_dispatch = det._dispatch_stream_span

        def dispatch_or_fail(token_ids, calls, normal_chunks):
            if tok.FAIL_ID in token_ids:
                raise RuntimeError("injected parser failure")
            return original_dispatch(token_ids, calls, normal_chunks)

        det._dispatch_stream_span = dispatch_or_fail
        res = det.parse_streaming_increment("A?", self.tools, enc("A") + [tok.FAIL_ID])

        self.assertEqual(res.normal_text, "A?")
        self.assertEqual(res.calls, [])

    def test_stream_parser_error_after_second_call_committed_aborts_chunk(self):
        det = self._det()
        original_dispatch = det._dispatch_stream_span

        def dispatch_or_fail(token_ids, calls, normal_chunks):
            if ord("!") in token_ids:
                raise RuntimeError("injected parser failure")
            return original_dispatch(token_ids, calls, normal_chunks)

        det._dispatch_stream_span = dispatch_or_fail
        good_call = self._call_ids("get_weather", [("city", "Paris")])
        bad_call = [C["<tool_call>"]] + enc("get_weather") + [C["<arg_key>"]] + enc("!")

        with self.assertRaises(WelmV4StreamingParseError):
            det.parse_streaming_increment("", self.tools, good_call + bad_call)
        self.assertTrue(det.skip_unstreamed_arg_backfill)

    def test_stream_parser_error_after_partial_arg_raises(self):
        det = self._det()
        original_dispatch = det._dispatch_stream_span

        def dispatch_or_fail(token_ids, calls, normal_chunks):
            if ord("!") in token_ids:
                raise RuntimeError("injected parser failure")
            return original_dispatch(token_ids, calls, normal_chunks)

        det._dispatch_stream_span = dispatch_or_fail
        first_chunk = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Par")
        )
        first = det.parse_streaming_increment("", self.tools, first_chunk)

        self.assertIn('{"city": "Par', "".join(c.parameters for c in first.calls))
        with self.assertRaises(WelmV4StreamingParseError):
            det.parse_streaming_increment("", self.tools, enc("!"))
        self.assertTrue(det.skip_unstreamed_arg_backfill)

    def test_stream_parser_error_keeps_later_tool_end_visible_as_text(self):
        class SpecialControlTokenizer(FakeWelmTokenizer):
            SPECIAL_TRUE_IDS = set(FakeWelmTokenizer.CONTROL.values())

        tok = SpecialControlTokenizer()
        det = WelmV4ToolDetector(tokenizer=tok)
        original_dispatch = det._dispatch_stream_span

        def dispatch_or_fail(token_ids, calls, normal_chunks):
            if ord("!") in token_ids:
                raise RuntimeError("injected parser failure")
            return original_dispatch(token_ids, calls, normal_chunks)

        det._dispatch_stream_span = dispatch_or_fail
        first_chunk = [C["<tool_call>"]] + enc("get_weather") + enc("!")

        first = det.parse_streaming_increment("", self.tools, first_chunk)
        second = det.parse_streaming_increment("", self.tools, [C["</tool_call>"]])

        self.assertIn("<tool_call>get_weather!", first.normal_text)
        self.assertEqual(second.normal_text, "</tool_call>")

    def test_non_stream_trailing_eos_not_in_output(self):
        det = self._det()
        ids = (
            enc("Here:")
            + self._call_ids("get_weather", [("city", "X")])
            + [C["<|im_end|>"]]
        )
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(res.normal_text, "Here:")
        self.assertEqual(len(res.calls), 1)

    def test_stream_lookalike_not_triggered(self):
        det = self._det()
        seq = enc("write <tool_call> in prose")
        normal, calls = "", []
        for tid in seq:
            res = det.parse_streaming_increment("", self.tools, [tid])
            normal += res.normal_text
            calls += res.calls
        self.assertEqual(calls, [])
        self.assertEqual(normal, "write <tool_call> in prose")

    def test_stream_separator_accepts_exact_non_stream_whitespace_rule(self):
        det = self._det()
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + enc(" \t\r\n")
            + [C["<arg_value>"]]
            + enc("Paris")
            + [C["</arg_value>"]]
            + enc("\n\t ")
            + [C["</tool_call>"]]
        )

        non_stream = det.detect_and_parse("", self.tools, seq)
        self.assertEqual(json.loads(non_stream.calls[0].parameters), {"city": "Paris"})

        stream_det = self._det()
        normal, calls = self._stream_ids(stream_det, self.tools, seq, chunk_size=3)
        self.assertEqual(normal, "")
        self.assertEqual(
            json.loads("".join(c.parameters for c in calls if c.name is None)),
            {"city": "Paris"},
        )

    def test_stream_malformed_before_name_falls_back_for_all_chunk_sizes(self):
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_value>"]]
            + enc("junk")
            + [C["</tool_call>"]]
        )

        for chunk_size in range(1, len(seq) + 1):
            with self.subTest(chunk_size=chunk_size):
                det = self._det()
                normal, calls = self._stream_ids(
                    det, self.tools, seq, chunk_size=chunk_size
                )
                self.assertEqual(calls, [])
                self.assertEqual(normal, self.tok.decode(seq))

    def test_stream_malformed_after_name_raises_for_all_chunk_sizes(self):
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + enc("JUNK")
            + [C["<arg_value>"]]
            + enc("Paris")
            + [C["</arg_value>"], C["</tool_call>"]]
        )

        non_stream = self._det().detect_and_parse("", self.tools, seq)
        self.assertEqual(non_stream.calls, [])
        self.assertEqual(non_stream.normal_text, self.tok.decode(seq))

        for chunk_size in range(1, len(seq) + 1):
            with self.subTest(chunk_size=chunk_size):
                det = self._det()
                raised = False
                for i in range(0, len(seq), chunk_size):
                    try:
                        det.parse_streaming_increment(
                            "", self.tools, seq[i : i + chunk_size]
                        )
                    except WelmV4StreamingParseError:
                        raised = True
                        break
                self.assertTrue(raised)
                self.assertTrue(det.skip_unstreamed_arg_backfill)

    def test_stream_non_whitespace_between_arguments_raises(self):
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Paris")
            + [C["</arg_value>"]]
            + enc("JUNK")
            + [C["<arg_key>"]]
            + enc("days")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("3")
            + [C["</arg_value>"], C["</tool_call>"]]
        )

        det = self._det()
        with self.assertRaises(WelmV4StreamingParseError):
            det.parse_streaming_increment("", self.tools, seq)

    def test_actual_tool_start_inside_value_is_malformed(self):
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("A")
            + [C["<tool_call>"]]
            + enc("B")
            + [C["</arg_value>"], C["</tool_call>"]]
        )

        non_stream = self._det().detect_and_parse("", self.tools, seq)
        self.assertEqual(non_stream.calls, [])
        self.assertEqual(non_stream.normal_text, self.tok.decode(seq))

        stream_det = self._det()
        with self.assertRaises(WelmV4StreamingParseError):
            stream_det.parse_streaming_increment("", self.tools, seq)

    def test_stream_empty_args(self):
        det = self._det()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_time",
                    parameters={"type": "object", "properties": {}},
                ),
            )
        ]
        seq = self._call_ids("get_time", [])
        calls = []
        for tid in seq:
            res = det.parse_streaming_increment("", tools, [tid])
            calls += res.calls
        self.assertEqual([c.name for c in calls if c.name], ["get_time"])
        arg_deltas = [c.parameters for c in calls if c.name is None]
        self.assertEqual(arg_deltas, ["{}"])

    def test_stream_arg_value_lookalike_not_triggered(self):
        det = self._det()
        value = "literal <arg_key>x</arg_key><arg_value>y</arg_value>"
        seq = self._call_ids("get_weather", [("city", value)])
        calls = []
        for tid in seq:
            res = det.parse_streaming_increment("", self.tools, [tid])
            calls += res.calls
        merged = "".join(c.parameters for c in calls if c.name is None)
        args = json.loads(merged)
        self.assertEqual(args["city"], value)

    def test_stream_multiple_calls_token_id_fsm(self):
        det = self._det()
        seq = (
            self._call_ids("get_weather", [("city", "A")])
            + enc("\n")
            + self._call_ids("get_weather", [("city", "B")])
        )
        calls = []
        for tid in seq:
            res = det.parse_streaming_increment("", self.tools, [tid])
            calls += res.calls
        self.assertEqual(
            [c.name for c in calls if c.name], ["get_weather", "get_weather"]
        )
        args_by_index = self._merged_args_by_index(calls)
        arg_deltas = [json.loads(args_by_index[i]) for i in sorted(args_by_index)]
        self.assertEqual([args["city"] for args in arg_deltas], ["A", "B"])

    def test_stream_long_string_arguments_incrementally(self):
        det = self._det()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="write_file",
                    parameters={
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        }
                    },
                ),
            )
        ]
        content = "\n".join(f"line {i}: print('hello')" for i in range(200))
        seq = self._call_ids(
            "write_file", [("path", "/tmp/a.py"), ("content", content)]
        )

        normal, calls = self._stream_ids(det, tools, seq, chunk_size=17)
        self.assertEqual(normal, "")
        self.assertEqual([c.name for c in calls if c.name], ["write_file"])
        arg_deltas = [c.parameters for c in calls if c.name is None]
        self.assertGreater(len(arg_deltas), 10)
        merged = "".join(arg_deltas)
        self.assertIn("line 10", merged)
        args = json.loads(merged)
        self.assertEqual(args["content"], content)

    def test_stream_string_escapes_are_stable(self):
        det = self._det()
        value = 'quote " backslash \\ newline\nunicode 尾'
        _, calls = self._stream_ids(
            det, self.tools, self._call_ids("get_weather", [("city", value)])
        )
        args = json.loads("".join(c.parameters for c in calls if c.name is None))
        self.assertEqual(args["city"], value)

    def test_stream_multiple_argument_types(self):
        det = self._det()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="submit",
                    parameters={
                        "properties": {
                            "title": {"type": "string"},
                            "count": {"type": "number"},
                            "enabled": {"type": "boolean"},
                            "meta": {"type": "object"},
                            "items": {"type": "array"},
                        }
                    },
                ),
            )
        ]
        seq = self._call_ids(
            "submit",
            [
                ("title", "job"),
                ("count", "3"),
                ("enabled", "true"),
                ("meta", '{"a":1}'),
                ("items", '["x","y"]'),
            ],
        )
        _, calls = self._stream_ids(det, tools, seq, chunk_size=9)
        args = json.loads("".join(c.parameters for c in calls if c.name is None))
        self.assertEqual(args["title"], "job")
        self.assertEqual(args["count"], 3)
        self.assertEqual(args["enabled"], True)
        self.assertEqual(args["meta"], {"a": 1})
        self.assertEqual(args["items"], ["x", "y"])

    def test_stream_optional_string_schema_stays_string(self):
        det = self._det()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="label",
                    parameters={
                        "properties": {
                            "value": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                        }
                    },
                ),
            )
        ]
        _, calls = self._stream_ids(
            det, tools, self._call_ids("label", [("value", "true")]), chunk_size=2
        )
        args = json.loads("".join(c.parameters for c in calls if c.name is None))
        self.assertEqual(args["value"], "true")

    def test_stream_mtp_chunk_matches_token_by_token(self):
        seq = self._call_ids("get_weather", [("city", "A"), ("days", "5")])

        det_token = self._det()
        _, calls_token = self._stream_ids(det_token, self.tools, seq, chunk_size=1)

        det_chunk = self._det()
        _, calls_chunk = self._stream_ids(det_chunk, self.tools, seq, chunk_size=11)

        self.assertEqual(
            [c.name for c in calls_token if c.name],
            [c.name for c in calls_chunk if c.name],
        )
        self.assertEqual(
            json.loads("".join(c.parameters for c in calls_token if c.name is None)),
            json.loads("".join(c.parameters for c in calls_chunk if c.name is None)),
        )

    def test_stream_result_is_chunk_size_invariant(self):
        seq = self._call_ids(
            "get_weather",
            [("city", "Paris"), ("days", "7")],
        )

        expected_names = None
        expected_args = None
        for chunk_size in range(1, len(seq) + 1):
            det = self._det()
            normal, calls = self._stream_ids(
                det, self.tools, seq, chunk_size=chunk_size
            )
            names = [c.name for c in calls if c.name]
            args = json.loads("".join(c.parameters for c in calls if c.name is None))
            self.assertEqual(normal, "")
            if expected_names is None:
                expected_names = names
                expected_args = args
            self.assertEqual(names, expected_names)
            self.assertEqual(args, expected_args)

    def test_stream_unknown_tool_falls_back_to_content(self):
        det = self._det()
        seq = (
            enc("prefix")
            + self._call_ids("python", [("code", "print(1)")])
            + enc("suffix")
        )

        normal, calls = self._stream_ids(det, self.tools, seq)

        self.assertEqual(calls, [])
        self.assertEqual(normal, self.tok.decode(seq))

    def test_unknown_tool_forwarding_enabled(self):
        seq = self._call_ids("python", [("code", "print(1)")])

        with patch.dict(os.environ, {"SGLANG_FORWARD_UNKNOWN_TOOLS": "true"}):
            non_stream = self._det().detect_and_parse("", self.tools, seq)
            _, stream_calls = self._stream_ids(self._det(), self.tools, seq)

        self.assertEqual([call.name for call in non_stream.calls], ["python"])
        self.assertEqual(
            [call.name for call in stream_calls if call.name],
            ["python"],
        )
        self.assertEqual(
            json.loads(non_stream.calls[0].parameters),
            {"code": "print(1)"},
        )

    def test_stream_unknown_tool_fallback_preserves_special_token_text(self):
        class SpecialControlTokenizer(FakeWelmTokenizer):
            SPECIAL_TRUE_IDS = set(FakeWelmTokenizer.CONTROL.values())

        tok = SpecialControlTokenizer()
        det = WelmV4ToolDetector(tokenizer=tok)
        seq = enc("prefix") + self._call_ids("python", [("code", "print(1)")])

        normal, calls = self._stream_ids(det, self.tools, seq)

        self.assertEqual(calls, [])
        self.assertEqual(normal, tok.decode(seq, skip_special_tokens=False))
        self.assertIn("<tool_call>", normal)
        self.assertIn("</tool_call>", normal)

    def test_stream_missing_value_end_raises_after_name_was_sent(self):
        det = self._det()
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Beijing")
            + [C["</tool_call>"]]
        )

        with self.assertRaises(WelmV4StreamingParseError):
            self._stream_ids(det, self.tools, seq)
        self.assertTrue(det.skip_unstreamed_arg_backfill)

    def test_stream_without_tool_end_does_not_finalize_arguments(self):
        det = self._det()
        seq = self._call_ids("get_weather", [("city", "Paris")])
        seq.pop()

        _, calls = self._stream_ids(det, self.tools, seq)
        merged = "".join(c.parameters for c in calls if c.name is None)

        self.assertEqual(merged, '{"city": "Paris"')
        self.assertEqual(det.prev_tool_call_arr[0]["arguments"], {})

    def test_stream_truncated_value_is_not_completed_or_rewritten(self):
        det = self._det()
        seq = (
            [C["<tool_call>"]]
            + enc("get_weather")
            + [C["<arg_key>"]]
            + enc("city")
            + [C["</arg_key>"]]
            + [C["<arg_value>"]]
            + enc("Par")
        )

        _, calls = self._stream_ids(det, self.tools, seq)
        merged = "".join(c.parameters for c in calls if c.name is None)

        self.assertEqual(merged, '{"city": "Par')
        self.assertEqual(det.prev_tool_call_arr[0]["arguments"], {})

    def test_stream_duplicate_key_uses_last_value(self):
        det = self._det()
        seq = self._call_ids("get_weather", [("city", "A"), ("city", "B")])

        _, calls = self._stream_ids(det, self.tools, seq, chunk_size=4)
        merged = "".join(c.parameters for c in calls if c.name is None)

        self.assertEqual(json.loads(merged), {"city": "B"})

    def test_stream_schema_extra_key_is_preserved(self):
        det = self._det()
        seq = self._call_ids("get_weather", [("city", "A"), ("unit", "celsius")])

        _, calls = self._stream_ids(det, self.tools, seq, chunk_size=5)
        merged = "".join(c.parameters for c in calls if c.name is None)

        self.assertEqual(json.loads(merged), {"city": "A", "unit": "celsius"})

    def test_stream_raw_object_no_corrupt_suffix(self):
        det = self._det()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="submit",
                    parameters={"properties": {"meta": {"type": "object"}}},
                ),
            )
        ]
        _, calls = self._stream_ids(
            det, tools, self._call_ids("submit", [("meta", '{"a":1}')]), chunk_size=3
        )
        merged = "".join(c.parameters for c in calls if c.name is None)
        self.assertEqual(json.loads(merged), {"meta": {"a": 1}})
        self.assertNotIn("}}}", merged)


# Chat templates render string values verbatim and everything else through
# tojson, so a bare `null`/`true`/`5` in the output is ambiguous and only the
# schema can resolve it. Every WeLM path must agree on the resolution: the
# id-based main path, the Glm4MoeDetector fallback, and streaming.
BARE_LITERAL_CASES = [
    # (label, schema, rendered value, expected argument)
    ("pure_string_null", {"type": "string"}, "null", "null"),
    ("pure_string_bool", {"type": "string"}, "true", "true"),
    ("pure_string_exponent", {"type": "string"}, "1e3", "1e3"),
    ("pure_string_trailing_zero", {"type": "string"}, "1.230", "1.230"),
    ("pure_string_nan", {"type": "string"}, "NaN", "NaN"),
    ("pure_string_object_literal", {"type": "string"}, '{"a":1}', '{"a":1}'),
    ("pure_string_array_literal", {"type": "string"}, "[1,2]", "[1,2]"),
    ("nullable_null", {"type": ["string", "null"]}, "null", None),
    ("nullable_text", {"type": ["string", "null"]}, "hello", "hello"),
    ("nullable_padded_null", {"type": ["string", "null"]}, " null ", " null "),
    ("nullable_uppercase_null", {"type": ["string", "null"]}, "NULL", "NULL"),
    ("nullable_empty", {"type": ["string", "null"]}, "", ""),
    ("nullable_null_prefix", {"type": ["string", "null"]}, "nullable x", "nullable x"),
    ("nullable_quoted_text", {"type": ["string", "null"]}, '"hi"', '"hi"'),
    ("openapi_nullable", {"type": "string", "nullable": True}, "null", None),
    ("enum_with_null", {"enum": ["a", None]}, "null", None),
    ("enum_with_null_member", {"enum": ["a", None]}, "a", "a"),
    (
        "union_bool_true",
        {"anyOf": [{"type": "string"}, {"type": "boolean"}]},
        "true",
        True,
    ),
    (
        "union_bool_false",
        {"anyOf": [{"type": "string"}, {"type": "boolean"}]},
        "false",
        False,
    ),
    ("union_number", {"anyOf": [{"type": "string"}, {"type": "number"}]}, "5", 5),
    (
        "union_number_exponent",
        {"anyOf": [{"type": "string"}, {"type": "number"}]},
        "1e3",
        1000.0,
    ),
    (
        "union_number_unparsable",
        {"anyOf": [{"type": "string"}, {"type": "number"}]},
        "abc",
        "abc",
    ),
    (
        "union_number_nan_stays_string",
        {"anyOf": [{"type": "string"}, {"type": "number"}]},
        "NaN",
        "NaN",
    ),
    (
        "union_object",
        {"anyOf": [{"type": "string"}, {"type": "object"}]},
        '{"a":1}',
        {"a": 1},
    ),
    ("ref_to_nullable_string", {"$ref": "#/$defs/MaybeText"}, "null", None),
    ("plain_number", {"type": "number"}, "5", 5),
    ("nullable_number", {"type": ["number", "null"]}, "null", None),
]


class TestWelmV4BareJsonLiteralArguments(CustomTestCase):
    """Bare JSON literals in argument values, across all three WeLM paths."""

    def setUp(self):
        self.tok = FakeWelmTokenizer()

    @staticmethod
    def _tools(schema):
        return [
            Tool(
                type="function",
                function=Function(
                    name="f",
                    parameters={
                        "$defs": {"MaybeText": {"type": ["string", "null"]}},
                        "type": "object",
                        "properties": {"x": schema},
                    },
                ),
            )
        ]

    @staticmethod
    def _ids(value):
        return (
            [C["<tool_call>"]]
            + enc("f\n")
            + [C["<arg_key>"]]
            + enc("x")
            + [C["</arg_key>"]]
            + enc("\n")
            + [C["<arg_value>"]]
            + enc(value)
            + [C["</arg_value>"], C["</tool_call>"]]
        )

    @staticmethod
    def _text(value):
        return (
            f"<tool_call>f\n<arg_key>x</arg_key>\n"
            f"<arg_value>{value}</arg_value>\n</tool_call>"
        )

    def _main_path(self, schema, value):
        detector = WelmV4ToolDetector(tokenizer=self.tok)
        result = detector.detect_and_parse("", self._tools(schema), self._ids(value))
        return json.loads(result.calls[0].parameters)["x"]

    def _fallback_path(self, schema, value):
        # No tokenizer means no control-token ids, so detect_and_parse delegates
        # to Glm4MoeDetector. This is what /parse_function_call exercises.
        detector = WelmV4ToolDetector(tokenizer=None)
        result = detector.detect_and_parse(self._text(value), self._tools(schema))
        return json.loads(result.calls[0].parameters)["x"]

    def _stream_path(self, schema, value, chunk_size):
        detector = WelmV4ToolDetector(tokenizer=self.tok)
        tools = self._tools(schema)
        ids = self._ids(value)
        merged = ""
        for i in range(0, len(ids), chunk_size):
            result = detector.parse_streaming_increment(
                "", tools, ids[i : i + chunk_size]
            )
            for call in result.calls:
                if call.name is None:
                    merged += call.parameters
        return json.loads(merged)["x"], detector

    def test_decision_table_all_paths_agree(self):
        for label, schema, value, expected in BARE_LITERAL_CASES:
            with self.subTest(case=label):
                self.assertEqual(self._main_path(schema, value), expected)
                self.assertEqual(self._fallback_path(schema, value), expected)
                for chunk_size in (1, 4, len(self._ids(value))):
                    with self.subTest(chunk_size=chunk_size):
                        streamed, _ = self._stream_path(schema, value, chunk_size)
                        self.assertEqual(streamed, expected)

    def test_db_style_string_types_are_quoted_on_every_path(self):
        """Without alias resolution these stream as bare text and break JSON."""
        for type_name in ("varchar", "varchar(255)", "str", "text", "enum", "uuid"):
            for value in ("hello", "N/A", "", "unknown"):
                with self.subTest(type=type_name, value=value):
                    schema = {"type": type_name}
                    self.assertEqual(self._main_path(schema, value), value)
                    self.assertEqual(self._fallback_path(schema, value), value)
                    streamed, _ = self._stream_path(schema, value, 1)
                    self.assertEqual(streamed, value)

    def test_db_style_nullable_type_array_recovers_json_null(self):
        schema = {"type": ["varchar", "none"]}
        self.assertIsNone(self._main_path(schema, "null"))
        self.assertIsNone(self._fallback_path(schema, "null"))
        self.assertIsNone(self._stream_path(schema, "null", 1)[0])
        # A non-literal value on the same schema stays a string.
        self.assertEqual(
            self._stream_path(schema, "nullable text", 1)[0], "nullable text"
        )

    def test_streamed_args_match_backfill_expectation(self):
        """serving_chat backfills by prefix-matching these two states."""
        for label, schema, value, _ in BARE_LITERAL_CASES:
            with self.subTest(case=label):
                _, detector = self._stream_path(schema, value, 1)
                expected_call = json.dumps(
                    detector.prev_tool_call_arr[0]["arguments"], ensure_ascii=False
                )
                self.assertEqual(detector.streamed_args_for_tool[0], expected_call)

    def test_undecided_union_releases_buffer_at_shortest_prefix(self):
        """A union value must not block streaming beyond the decidable prefix."""
        schema = {"type": ["string", "null"]}
        expectations = [
            ("Hello world, a long sentence", 1),
            ("note: something", 2),
            ("nullable and a long paragraph", 5),
        ]
        for value, expected_delay in expectations:
            with self.subTest(value=value):
                detector = WelmV4ToolDetector(tokenizer=self.tok)
                tools = self._tools(schema)
                ids = self._ids(value)
                value_start = ids.index(C["<arg_value>"])
                merged = ""
                prefix_len = None
                first_delta_at = None
                for pos, token_id in enumerate(ids):
                    result = detector.parse_streaming_increment("", tools, [token_id])
                    for call in result.calls:
                        if call.name is None:
                            merged += call.parameters
                    if pos == value_start:
                        prefix_len = len(merged)
                    elif (
                        prefix_len is not None
                        and first_delta_at is None
                        and len(merged) > prefix_len
                    ):
                        first_delta_at = pos - value_start
                self.assertEqual(first_delta_at, expected_delay)

    def test_unresolved_ref_keeps_value_verbatim_on_main_path(self):
        # An unresolvable $ref yields no type at all, so the value never reaches
        # the union rule. The main path keeps it verbatim; the Glm4MoeDetector
        # fallback still parses untyped values, a pre-existing divergence that
        # the union fix deliberately leaves alone.
        schema = {"$ref": "#/$defs/Missing"}
        self.assertEqual(self._main_path(schema, "null"), "null")
        for chunk_size in (1, 4):
            with self.subTest(chunk_size=chunk_size):
                streamed, _ = self._stream_path(schema, "null", chunk_size)
                self.assertEqual(streamed, "null")

    def test_undecided_union_literal_buffers_until_value_end(self):
        detector = WelmV4ToolDetector(tokenizer=self.tok)
        tools = self._tools({"type": ["string", "null"]})
        ids = self._ids("null")
        value_end = ids.index(C["</arg_value>"])
        merged = ""
        at_value_end = None
        for pos, token_id in enumerate(ids):
            result = detector.parse_streaming_increment("", tools, [token_id])
            for call in result.calls:
                if call.name is None:
                    merged += call.parameters
            if pos == value_end - 1:
                at_value_end = merged
        self.assertEqual(at_value_end, '{"x": ')
        self.assertEqual(merged, '{"x": null}')


class TestTrimMatchedStopForIdParser(CustomTestCase):
    def test_int_stop_trims_last_id(self):
        out = trim_matched_stop_for_id_parser([1, 2, 3], {"matched": 3}, False)
        self.assertEqual(out, [1, 2])

    def test_int_stop_no_stop_trim_keeps(self):
        out = trim_matched_stop_for_id_parser([1, 2, 3], {"matched": 3}, True)
        self.assertEqual(out, [1, 2, 3])

    def test_str_stop_drops_unreliable_ids(self):
        out = trim_matched_stop_for_id_parser([1, 2, 3], {"matched": "<stop>"}, False)
        self.assertIsNone(out)

    def test_regex_stop_drops_unreliable_ids(self):
        out = trim_matched_stop_for_id_parser(
            [1, 2, 3],
            {"matched": r"\d+", "matched_text": "123"},
            False,
        )
        self.assertIsNone(out)

    def test_no_finished_reason_noop(self):
        self.assertEqual(trim_matched_stop_for_id_parser([1, 2], None, False), [1, 2])
        self.assertEqual(trim_matched_stop_for_id_parser("ab", {}, False), "ab")

    def test_no_matched_noop(self):
        out = trim_matched_stop_for_id_parser([1, 2], {"matched": None}, False)
        self.assertEqual(out, [1, 2])

    def test_type_mismatch_noop(self):
        self.assertEqual(
            trim_matched_stop_for_id_parser("abc", {"matched": 3}, False), "abc"
        )
        self.assertEqual(
            trim_matched_stop_for_id_parser([1, 2], {"matched": "x"}, False), None
        )

    def test_empty_list_int_stop_noop(self):
        self.assertEqual(trim_matched_stop_for_id_parser([], {"matched": 3}, False), [])


class TestWelmV4ToolDecodeFlags(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()
        self.tools = _tools()

    def test_skip_true_drops_special_in_content(self):
        det = WelmV4ToolDetector(tokenizer=self.tok, skip_special_tokens=True)
        ids = enc("ab") + [C["<|im_end|>"]] + enc("cd")
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(res.normal_text, "abcd")

    def test_skip_false_keeps_special_in_content(self):
        det = WelmV4ToolDetector(tokenizer=self.tok, skip_special_tokens=False)
        ids = enc("ab") + [C["<|im_end|>"]] + enc("cd")
        res = det.detect_and_parse("", self.tools, ids)
        self.assertEqual(res.normal_text, "ab<|im_end|>cd")

    def test_stream_skip_false_keeps_special(self):
        det = WelmV4ToolDetector(tokenizer=self.tok, skip_special_tokens=False)
        seq = enc("ab") + [C["<|im_end|>"]] + enc("cd")
        normal = ""
        for tid in seq:
            res = det.parse_streaming_increment("", self.tools, [tid])
            normal += res.normal_text
        self.assertEqual(normal, "ab<|im_end|>cd")

    def test_detector_decode_flags_default_and_settable(self):
        parser = FunctionCallParser(self.tools, "welm-v4")
        parser.configure_tokenizer(self.tok)
        self.assertTrue(parser.detector.skip_special_tokens)
        self.assertTrue(parser.detector.spaces_between_special_tokens)
        parser.detector.skip_special_tokens = False
        parser.detector.spaces_between_special_tokens = False
        self.assertFalse(parser.detector.skip_special_tokens)
        self.assertFalse(parser.detector.spaces_between_special_tokens)


class TestWelmV4ReasoningDecodeFlags(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()

    def test_skip_false_keeps_special_in_answer(self):
        det = WelmV4ReasoningDetector(
            tokenizer=self.tok, force_reasoning=True, skip_special_tokens=False
        )
        ids = enc("re") + [C["</think>"]] + enc("an") + [C["<|im_end|>"]] + enc("d")
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "re")
        self.assertEqual(res.normal_text, "an<|im_end|>d")

    def test_skip_true_drops_special_in_answer(self):
        det = WelmV4ReasoningDetector(
            tokenizer=self.tok, force_reasoning=True, skip_special_tokens=True
        )
        ids = enc("re") + [C["</think>"]] + enc("an") + [C["<|im_end|>"]] + enc("d")
        res = det.detect_and_parse("", ids)
        self.assertEqual(res.reasoning_text, "re")
        self.assertEqual(res.normal_text, "and")


class TestWelmV4Wrappers(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()
        self.tools = _tools()

    def test_reasoning_parser_wrapper(self):
        parser = ReasoningParser(
            model_type="welm-v4",
            stream_reasoning=False,
            force_reasoning=True,
        )
        parser.configure_tokenizer(self.tok)
        ids = enc("think") + [C["</think>"]] + enc("answer")
        reasoning, normal = parser.parse_non_stream("", token_ids=ids)
        self.assertEqual(reasoning, "think")
        self.assertEqual(normal, "answer")
        self.assertEqual(parser.remaining_token_ids, enc("answer"))

    def test_reasoning_parser_wrapper_forwards_decode_flags(self):
        parser = ReasoningParser(
            model_type="welm-v4",
            stream_reasoning=False,
            force_reasoning=True,
        )
        parser.configure_tokenizer(self.tok, skip_special_tokens=False)
        ids = enc("think") + [C["</think>"]] + enc("answer") + [C["<|im_end|>"]]
        reasoning, normal = parser.parse_non_stream("", token_ids=ids)
        self.assertEqual(reasoning, "think")
        self.assertEqual(normal, "answer<|im_end|>")

    def test_reasoning_parser_default_matches_deepseek_r1_contract(self):
        parser = ReasoningParser(
            model_type="welm-v4",
            stream_reasoning=False,
            force_reasoning=False,
        )
        parser.configure_tokenizer(self.tok)
        self.assertEqual(parser.detector.reasoning_default, "always")
        self.assertTrue(parser.detector.thinks_internally)
        self.assertEqual(
            parser.detector.think_excluded_tokens,
            ["<tool_call>", "</tool_call>", "<|im_end|>", "<|endoftext|>"],
        )
        reasoning, normal = parser.parse_non_stream("", token_ids=enc("reasoning"))
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(normal, "")

    def test_function_call_parser_wrapper(self):
        parser = FunctionCallParser(self.tools, "welm-v4")
        parser.configure_tokenizer(self.tok)
        ids = [C["<tool_call>"]] + enc("get_weather")
        ids += enc("\n") + [C["<arg_key>"]] + enc("city") + [C["</arg_key>"]]
        ids += enc("\n") + [C["<arg_value>"]] + enc("Tokyo") + [C["</arg_value>"]]
        ids += enc("\n") + [C["</tool_call>"]]
        self.assertTrue(parser.has_tool_call("", token_ids=ids))
        normal, calls = parser.parse_non_stream("", token_ids=ids)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters)["city"], "Tokyo")

    def test_mixed_id_and_text_parsers_keep_text_routing(self):
        welm_reasoning = ReasoningParser(
            "welm-v4", stream_reasoning=True, force_reasoning=True
        )
        welm_reasoning.configure_tokenizer(self.tok)
        text_tool = FunctionCallParser(self.tools, "glm45")
        reasoning, normal = welm_reasoning.parse_stream_chunk(
            "",
            token_ids=enc("reason") + [C["</think>"]] + enc("answer"),
        )
        routed_normal, calls = text_tool.parse_stream_chunk(
            normal,
            token_ids=welm_reasoning.remaining_token_ids,
        )
        self.assertEqual(reasoning, "reason")
        self.assertEqual(routed_normal, "answer")
        self.assertEqual(calls, [])

        text_reasoning = ReasoningParser(
            "qwen3", stream_reasoning=True, force_reasoning=False
        )
        welm_tool = FunctionCallParser(self.tools, "welm-v4")
        welm_tool.configure_tokenizer(self.tok)
        reasoning, normal = text_reasoning.parse_stream_chunk(
            "<think>reason</think>answer",
            token_ids=enc("ignored"),
        )
        routed_normal, calls = welm_tool.parse_stream_chunk(normal, token_ids=None)
        self.assertEqual(reasoning, "reason")
        self.assertEqual(routed_normal, "answer")
        self.assertEqual(calls, [])

    def test_registered_in_maps(self):
        self.assertIn("welm-v4", ReasoningParser.DetectorMap)
        self.assertIn("welm-v4", FunctionCallParser.ToolCallParserEnum)

    def test_token_id_capability_api(self):
        self.assertTrue(ReasoningParser.accepts_token_ids_for("welm-v4"))
        self.assertFalse(ReasoningParser.accepts_token_ids_for("deepseek-r1"))
        self.assertFalse(ReasoningParser.accepts_token_ids_for(None))

        reasoning_parser = ReasoningParser(
            model_type="welm-v4",
            stream_reasoning=False,
            force_reasoning=True,
        )
        self.assertFalse(reasoning_parser.accepts_token_ids)
        self.assertIsNone(reasoning_parser.remaining_token_ids)
        reasoning_parser.configure_tokenizer(self.tok)
        self.assertTrue(reasoning_parser.accepts_token_ids)

        self.assertTrue(FunctionCallParser.accepts_token_ids_for("welm-v4"))
        self.assertFalse(FunctionCallParser.accepts_token_ids_for("glm"))
        self.assertFalse(FunctionCallParser.accepts_token_ids_for(None))

        tool_parser = FunctionCallParser(self.tools, "welm-v4")
        self.assertFalse(tool_parser.accepts_token_ids)
        tool_parser.configure_tokenizer(self.tok)
        self.assertTrue(tool_parser.accepts_token_ids)


class TestWelmV4IdBasedStreamStopFilter(CustomTestCase):
    def setUp(self):
        self.tok = FakeWelmTokenizer()
        self.model_config = SimpleNamespace(hf_eos_token_id={C["<|im_end|>"]})

    def _request(self, **kwargs):
        defaults = dict(
            no_stop_trim=False,
            ignore_eos=False,
            stop_token_ids=None,
            skip_special_tokens=False,
            chat_template_kwargs={},
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    @staticmethod
    def _logprobs(tokens):
        return {
            "content": [
                {"token": token, "bytes": [], "logprob": 0.0, "top_logprobs": []}
                for token in tokens
            ]
        }

    def _filter(self, req, delta, delta_ids, tokens, finish_reason=None):
        return filter_id_based_stream_stop(
            req,
            self.tok,
            self.model_config,
            delta,
            delta_ids,
            self._logprobs(tokens) if tokens is not None else None,
            finish_reason,
            req.skip_special_tokens,
            req.chat_template_kwargs.get("spaces_between_special_tokens", True),
        )

    def test_filters_token_stops_before_id_parser_but_preserves_raw_logprobs(self):
        cases = [
            (
                self._request(),
                "A<|im_end|>",
                enc("A") + [C["<|im_end|>"]],
                ["A", "<|im_end|>"],
                None,
                "A",
                enc("A"),
                ["A", "<|im_end|>"],
            ),
            (
                self._request(),
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
                None,
                "",
                [],
                ["<|im_end|>"],
            ),
            (
                self._request(stop_token_ids=[ord("|")]),
                "A|B",
                enc("A|B"),
                ["A", "|", "B"],
                None,
                "A",
                enc("A"),
                ["A", "|", "B"],
            ),
            (
                self._request(),
                "A<|im_end|>B",
                enc("A") + [C["<|im_end|>"]] + enc("B"),
                ["A", "<|im_end|>", "B"],
                None,
                "A",
                enc("A"),
                ["A", "<|im_end|>", "B"],
            ),
            (
                self._request(no_stop_trim=True),
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
                None,
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
            ),
            (
                self._request(ignore_eos=True, stop_token_ids=[C["<|im_end|>"]]),
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
                None,
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
            ),
            (
                self._request(ignore_eos=True),
                "<|im_end|>",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
                {"type": "stop", "matched": C["<|im_end|>"]},
                "",
                [],
                ["<|im_end|>"],
            ),
            (
                self._request(skip_special_tokens=True),
                "",
                [C["<|im_end|>"]],
                ["<|im_end|>"],
                None,
                "",
                [],
                ["<|im_end|>"],
            ),
        ]

        for req, delta, ids, tokens, finish, exp_delta, exp_ids, exp_tokens in cases:
            with self.subTest(
                no_stop_trim=req.no_stop_trim,
                ignore_eos=req.ignore_eos,
                finish_reason=finish,
            ):
                delta, delta_ids, logprobs = self._filter(
                    req, delta, ids, tokens, finish
                )
                self.assertEqual(delta, exp_delta)
                self.assertEqual(delta_ids, exp_ids)
                if exp_tokens is None:
                    self.assertIsNone(logprobs)
                else:
                    self.assertEqual(
                        [item["token"] for item in logprobs["content"]], exp_tokens
                    )

    def test_additional_stop_token_is_hidden_from_parser_but_kept_in_logprobs(self):
        additional_stop_id = ord("#")
        self.tok.additional_stop_token_ids = [additional_stop_id]

        delta, delta_ids, logprobs = self._filter(
            self._request(),
            "A#",
            enc("A#"),
            ["A", "#"],
        )

        self.assertEqual(delta, "A")
        self.assertEqual(delta_ids, enc("A"))
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "#"],
        )

    def test_hf_eos_token_is_hidden_from_parser_but_kept_in_logprobs(self):
        hf_eos_id = ord("~")
        self.model_config.hf_eos_token_id = {hf_eos_id}

        delta, delta_ids, logprobs = self._filter(
            self._request(),
            "A~",
            enc("A~"),
            ["A", "~"],
        )

        self.assertEqual(delta, "A")
        self.assertEqual(delta_ids, enc("A"))
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "~"],
        )

    def test_non_stop_special_token_keeps_ids_and_logprobs(self):
        special_id = 1010
        self.tok.SPECIAL_TRUE_IDS = {*self.tok.SPECIAL_TRUE_IDS, special_id}
        self.tok._id2tok[special_id] = "<special>"
        req = self._request(skip_special_tokens=True)
        delta = self.tok.decode([special_id], skip_special_tokens=True)

        delta, delta_ids, logprobs = self._filter(
            req,
            delta,
            [special_id],
            ["<special>"],
        )

        self.assertEqual(delta, "")
        self.assertEqual(delta_ids, [special_id])
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["<special>"],
        )

    def test_string_finish_trims_terminal_stop_id_even_when_eos_is_ignored(self):
        delta, delta_ids, logprobs = self._filter(
            self._request(ignore_eos=True),
            "A<|im_end|>",
            enc("A") + [C["<|im_end|>"]],
            ["A", "<|im_end|>"],
            {"type": "stop", "matched": "NaN happened"},
        )

        self.assertEqual(delta, "A")
        self.assertEqual(delta_ids, enc("A"))
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "<|im_end|>"],
        )

    def test_string_finish_without_terminal_stop_id_fails(self):
        with self.assertRaisesRegex(ValueError, "terminal stop token ID"):
            self._filter(
                self._request(stop_token_ids=[ord("|")]),
                "A|B",
                enc("A|B"),
                ["A", "|", "B"],
                {"type": "stop", "matched": "unexpected"},
            )

    def test_reasoning_parser_dispatches_stream_stop_filter(self):
        delta, delta_ids, logprobs = ReasoningParser.filter_id_based_stream_stop(
            "welm-v4",
            self._request(),
            self.tok,
            self.model_config,
            "A<|im_end|>",
            enc("A") + [C["<|im_end|>"]],
            self._logprobs(["A", "<|im_end|>"]),
            None,
            False,
            True,
        )

        self.assertEqual(delta, "A")
        self.assertEqual(delta_ids, enc("A"))
        self.assertEqual(
            [item["token"] for item in logprobs["content"]],
            ["A", "<|im_end|>"],
        )


class MissingControlTokenizer(FakeWelmTokenizer):
    def convert_tokens_to_ids(self, token):
        if token in {
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<arg_key>",
            "</arg_key>",
            "<arg_value>",
            "</arg_value>",
        }:
            return self.unk_token_id
        return super().convert_tokens_to_ids(token)


class TestWelmV4Fallback(CustomTestCase):
    def test_reasoning_falls_back_to_text_parser(self):
        tok = MissingControlTokenizer()
        parser = ReasoningParser("welm-v4", stream_reasoning=False)
        parser.configure_tokenizer(tok)
        reasoning, normal = parser.parse_non_stream(
            "<think>why</think>answer", token_ids=enc("whyanswer")
        )
        self.assertEqual(reasoning, "why")
        self.assertEqual(normal, "answer")
        self.assertIsNone(parser.remaining_token_ids)

    def test_reasoning_stream_falls_back_to_text_parser(self):
        tok = MissingControlTokenizer()
        parser = ReasoningParser("welm-v4", stream_reasoning=True)
        parser.configure_tokenizer(tok)
        reasoning, normal = "", ""
        for chunk in ["<think>", "why", "</think>", "answer"]:
            reasoning_delta, normal_delta = parser.parse_stream_chunk(
                chunk, token_ids=enc("not-used")
            )
            reasoning += reasoning_delta
            normal += normal_delta
        self.assertEqual(reasoning, "why")
        self.assertEqual(normal, "answer")
        self.assertIsNone(parser.remaining_token_ids)

    def test_tool_falls_back_to_text_parser(self):
        tok = MissingControlTokenizer()
        parser = FunctionCallParser(_tools(), "welm-v4")
        parser.configure_tokenizer(tok)
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n"
            "<arg_value>Tokyo</arg_value>\n"
            "</tool_call>"
        )
        normal, calls = parser.parse_non_stream(text, token_ids=enc("not-used"))
        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters)["city"], "Tokyo")

    def test_tool_stream_falls_back_to_text_parser(self):
        tok = MissingControlTokenizer()
        parser = FunctionCallParser(_tools(), "welm-v4")
        parser.configure_tokenizer(tok)
        normal, calls = "", []
        for chunk in [
            "<tool_call>get_weather\n",
            "<arg_key>city</arg_key>\n",
            "<arg_value>Tokyo</arg_value>\n",
            "</tool_call>",
        ]:
            normal_delta, call_delta = parser.parse_stream_chunk(
                chunk, token_ids=enc("not-used")
            )
            normal += normal_delta
            calls += call_delta
        self.assertEqual(normal, "")
        self.assertTrue(any(call.name == "get_weather" for call in calls))
        self.assertIn("Tokyo", "".join(call.parameters for call in calls))


if __name__ == "__main__":
    unittest.main()
