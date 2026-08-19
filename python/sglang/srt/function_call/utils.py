import json
from json import JSONDecodeError, JSONDecoder
from json.decoder import WHITESPACE
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import orjson
import partial_json_parser
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice


def _find_common_prefix(s1: str, s2: str) -> str:
    prefix = ""
    min_length = min(len(s1), len(s2))
    for i in range(0, min_length):
        if s1[i] == s2[i]:
            prefix += s1[i]
        else:
            break
    return prefix


def _partial_json_loads(input_str: str, flags: Allow) -> Tuple[Any, int]:
    """
    Parse incomplete or partial JSON strings commonly encountered during streaming.

    Args:
        input_str (str): The potentially incomplete JSON string to parse.
        flags (Allow): Bitwise flags controlling what types of partial data are allowed.
            Common flags include:
            - Allow.STR: Allow partial strings (e.g., '"hello wo' -> 'hello wo')
            - Allow.OBJ: Allow partial objects (e.g., '{"key":' -> {'key': None})
            - Allow.ARR: Allow partial arrays (e.g., '[1, 2,' -> [1, 2])
            - Allow.ALL: Allow all types of partial data

    Returns:
        Tuple[Any, int]: A tuple containing:
            - parsed_object: The Python object parsed from the JSON
            - consumed_length: Number of characters consumed from input_str
    """
    try:
        return (partial_json_parser.loads(input_str, flags), len(input_str))
    except (JSONDecodeError, IndexError) as e:
        msg = getattr(e, "msg", str(e))
        if "Extra data" in msg or "pop from empty list" in msg:
            start = WHITESPACE.match(input_str, 0).end()
            obj, end = JSONDecoder().raw_decode(input_str, start)
            return obj, end
        raise


def _is_complete_json(input_str: str) -> bool:
    try:
        orjson.loads(input_str)
        return True
    except JSONDecodeError:
        return False


def _get_tool_schema_defs(tools: List[Tool]) -> dict:
    """
    Get consolidated $defs from all tools, validating for conflicts.

    Args:
        tools: List of tools to process

    Returns:
        Dictionary of consolidated $defs from all tools

    Raises:
        ValueError: If conflicting $defs are found
    """
    all_defs = {}
    for tool in tools:
        if tool.function.parameters is None:
            continue
        defs = tool.function.parameters.get("$defs", {})
        for def_name, def_schema in defs.items():
            if def_name in all_defs and all_defs[def_name] != def_schema:
                raise ValueError(
                    f"Tool definition '{def_name}' has "
                    "multiple schemas, which is not "
                    "supported."
                )
            else:
                all_defs[def_name] = def_schema
    return all_defs


def _get_tool_schema(tool: Tool) -> dict:
    return {
        "properties": {
            "name": {"type": "string", "enum": [tool.function.name]},
            "parameters": (
                tool.function.parameters
                if tool.function.parameters
                else {"type": "object", "properties": {}}
            ),
        },
        "required": ["name", "parameters"],
    }


def _resolve_local_json_schema_ref(
    ref: str, root_schema: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if ref == "#":
        return root_schema
    if not ref.startswith("#/"):
        return None

    resolved: Any = root_schema
    for part in ref[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(resolved, dict) or key not in resolved:
            return None
        resolved = resolved[key]
    return resolved if isinstance(resolved, dict) else None


# Tool schemas exported from DB/ORM tooling carry database type names
# ("varchar", "int32", "list[str]") rather than JSON Schema ones. Resolving them
# here keeps type-directed parsing working; the schema itself is never mutated,
# so request validation and the rendered prompt stay untouched.
_STANDARD_JSON_SCHEMA_TYPES = frozenset(
    {"null", "boolean", "object", "array", "number", "string", "integer"}
)

_JSON_SCHEMA_TYPE_ALIASES = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "uuid": "string",
    "date": "string",
    "datetime": "string",
    "time": "string",
    "timestamp": "string",
    "binary": "string",
    "blob": "string",
    "bytea": "string",
    "bytes": "string",
    "varbinary": "string",
    "bool": "boolean",
    "bigint": "integer",
    "smallint": "integer",
    "tinyint": "integer",
    "double": "number",
    "decimal": "number",
    "real": "number",
    "numeric": "number",
    "arr": "array",
    "tuple": "array",
    "set": "array",
    "map": "object",
    "none": "null",
}

# A prefix only matches when it spans the whole token or is followed by a
# non-identifier char, so "int" does not swallow "internal" and "list" does not
# swallow "list_price".
_TYPE_PREFIX_BOUNDARY_CHARS = frozenset("0123456789[<( \t")
_JSON_SCHEMA_TYPE_PREFIXES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("int", "uint", "long", "short", "unsigned"), "integer"),
    (("num", "float"), "number"),
    (("list",), "array"),
    (("dict",), "object"),
)


def _matches_type_prefix(base: str, prefixes: Tuple[str, ...]) -> bool:
    for prefix in prefixes:
        if base == prefix:
            return True
        if (
            len(base) > len(prefix)
            and base.startswith(prefix)
            and base[len(prefix)] in _TYPE_PREFIX_BOUNDARY_CHARS
        ):
            return True
    return False


def _resolve_standard_json_schema_type(type_name: Any) -> Optional[str]:
    """Standard JSON Schema type behind a possibly non-standard type name.

    Returns None when nothing standard can be recognized, so callers can fall
    back to the verbatim value and let downstream logic surface the oddity.
    """
    if not isinstance(type_name, str):
        return None
    if type_name in _STANDARD_JSON_SCHEMA_TYPES:
        return type_name
    # ``split("(", 1)[0]`` strips parameters such as ``varchar(255)``.
    base = type_name.split("(", 1)[0].strip().lower()
    if base in _STANDARD_JSON_SCHEMA_TYPES:
        return base
    aliased = _JSON_SCHEMA_TYPE_ALIASES.get(base)
    if aliased is not None:
        return aliased
    for prefixes, target in _JSON_SCHEMA_TYPE_PREFIXES:
        if _matches_type_prefix(base, prefixes):
            return target
    return None


def _enum_overrides_type(schema: Dict[str, Any]) -> bool:
    """Whether an ``enum`` list should win over a non-standard ``type``.

    ``{"type": "enum", "enum": [1, 2, 3]}`` is not valid JSON Schema; such a
    ``type`` is a DB/ORM artifact and says nothing reliable about the values,
    while the enum members describe them exactly.
    """
    enum_values = schema.get("enum")
    if not isinstance(enum_values, list) or not enum_values:
        return False
    type_value = schema.get("type")
    entries = type_value if isinstance(type_value, list) else [type_value]
    return any(
        isinstance(entry, str) and entry not in _STANDARD_JSON_SCHEMA_TYPES
        for entry in entries
    )


def infer_type_from_json_schema(
    schema: Dict[str, Any],
    root_schema: Optional[Dict[str, Any]] = None,
    _visited_refs: Optional[set[str]] = None,
) -> Optional[str]:
    """
    Infer the primary type of a parameter from JSON Schema.

    Supports complex JSON Schema structures including:
    - Direct type field (including type arrays)
    - anyOf/oneOf: parameter can be any of multiple types
    - enum: parameter must be one of enum values
    - allOf: parameter must satisfy all type definitions
    - properties: inferred as object type
    - items: inferred as array type

    Args:
        schema: JSON Schema definition
        root_schema: Complete schema used to resolve local ``$ref`` pointers

    Returns:
        Inferred type ('string', 'number', 'object', 'array', etc.) or None
    """
    if not isinstance(schema, dict):
        return None
    if root_schema is None:
        root_schema = schema
    visited_refs = _visited_refs or set()

    # Priority 1: Direct type field (including type arrays)
    if "type" in schema and not _enum_overrides_type(schema):
        type_value = schema["type"]
        if isinstance(type_value, str):
            return _resolve_standard_json_schema_type(type_value) or type_value
        elif isinstance(type_value, list) and type_value:
            # Handle type arrays: return first non-null type
            resolved = [_resolve_standard_json_schema_type(t) or t for t in type_value]
            non_null_types = [t for t in resolved if t != "null"]
            if non_null_types:
                return non_null_types[0]
            return "string"  # If only null, default to string

    # Priority 2: Resolve local JSON Schema references
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in visited_refs:
            return None
        resolved_schema = _resolve_local_json_schema_ref(ref, root_schema)
        if resolved_schema is None:
            return None
        return infer_type_from_json_schema(
            resolved_schema, root_schema, visited_refs | {ref}
        )

    # Priority 3: Handle anyOf/oneOf
    if "anyOf" in schema or "oneOf" in schema:
        schemas = schema.get("anyOf") or schema.get("oneOf")
        types = []

        if isinstance(schemas, list):
            for sub_schema in schemas:
                inferred_type = infer_type_from_json_schema(
                    sub_schema, root_schema, visited_refs
                )
                if inferred_type:
                    types.append(inferred_type)

            if types:
                # If all types are the same, return unified type
                if len(set(types)) == 1:
                    return types[0]
                # When types differ, prioritize string (safest)
                if "string" in types:
                    return "string"
                # Otherwise return first type
                return types[0]

    # Priority 4: Handle enum (infer type from enum values)
    if "enum" in schema and isinstance(schema["enum"], list):
        if not schema["enum"]:
            return "string"

        # Infer type from enum values
        enum_types = set()
        for value in schema["enum"]:
            if value is None:
                enum_types.add("null")
            elif isinstance(value, bool):
                enum_types.add("boolean")
            elif isinstance(value, int):
                enum_types.add("integer")
            elif isinstance(value, float):
                enum_types.add("number")
            elif isinstance(value, str):
                enum_types.add("string")
            elif isinstance(value, list):
                enum_types.add("array")
            elif isinstance(value, dict):
                enum_types.add("object")

        # If type is uniform, return that type
        if len(enum_types) == 1:
            return enum_types.pop()
        # Mixed types, prioritize string
        return "string"

    # Priority 5: Handle allOf (must satisfy all types)
    if "allOf" in schema and isinstance(schema["allOf"], list):
        schemas = schema["allOf"]
        for sub_schema in schemas:
            inferred_type = infer_type_from_json_schema(
                sub_schema, root_schema, visited_refs
            )
            if inferred_type and inferred_type != "string":
                return inferred_type
        return "string"

    # Priority 6: Infer object type
    if "properties" in schema:
        return "object"

    # Priority 7: Infer array type
    if "items" in schema:
        return "array"

    return None


# Chat templates render string argument values verbatim (no quotes) and every
# other type through ``tojson``. A bare ``null``/``true``/``5`` in the output is
# therefore ambiguous: it may be the JSON literal, or a string that happens to
# spell it. The schema is the only disambiguator, so the helpers below expose
# the *full* set of permitted types instead of a single primary type.

_NUMBER_START_CHARS = frozenset("-0123456789")
_NUMBER_BODY_CHARS = frozenset("0123456789.eE+-")


def _normalize_json_schema_type(type_name: Any) -> Optional[str]:
    resolved = _resolve_standard_json_schema_type(type_name)
    if resolved is not None:
        return resolved
    if not isinstance(type_name, str):
        return None
    return type_name.strip().lower() or None


def _json_type_names(value: Any) -> Set[str]:
    """JSON Schema type names that ``value`` satisfies."""
    if value is None:
        return {"null"}
    if isinstance(value, bool):
        return {"boolean"}
    if isinstance(value, int):
        return {"integer", "number"}
    if isinstance(value, float):
        return {"number"}
    if isinstance(value, str):
        return {"string"}
    if isinstance(value, list):
        return {"array"}
    if isinstance(value, dict):
        return {"object"}
    return set()


def infer_json_schema_types(
    schema: Dict[str, Any],
    root_schema: Optional[Dict[str, Any]] = None,
    _visited_refs: Optional[set] = None,
) -> Set[str]:
    """Infer every type a parameter is allowed to take from its JSON Schema.

    Mirrors the traversal of :func:`infer_type_from_json_schema` (type arrays,
    ``$ref``, ``anyOf``/``oneOf``/``allOf``, ``enum``, ``properties``,
    ``items``) but returns the union of all permitted types rather than a
    single primary type. Also honours the OpenAPI ``nullable: true`` extension.

    Returns:
        Set of normalized type names, or an empty set when nothing is declared.
    """
    if not isinstance(schema, dict):
        return set()
    if root_schema is None:
        root_schema = schema
    visited_refs = _visited_refs or set()

    types: Set[str] = set()
    if schema.get("nullable") is True:
        types.add("null")

    if "type" in schema and not _enum_overrides_type(schema):
        type_value = schema["type"]
        declared: Set[str] = set()
        if isinstance(type_value, str):
            normalized = _normalize_json_schema_type(type_value)
            if normalized:
                declared.add(normalized)
        elif isinstance(type_value, list):
            for entry in type_value:
                normalized = _normalize_json_schema_type(entry)
                if normalized:
                    declared.add(normalized)
        if declared:
            return types | declared

    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in visited_refs:
            return types
        resolved_schema = _resolve_local_json_schema_ref(ref, root_schema)
        if resolved_schema is None:
            return types
        return types | infer_json_schema_types(
            resolved_schema, root_schema, visited_refs | {ref}
        )

    for combiner in ("anyOf", "oneOf", "allOf"):
        sub_schemas = schema.get(combiner)
        if isinstance(sub_schemas, list):
            for sub_schema in sub_schemas:
                types |= infer_json_schema_types(sub_schema, root_schema, visited_refs)

    if isinstance(schema.get("enum"), list):
        for value in schema["enum"]:
            types |= _json_type_names(value)

    if not types:
        if "properties" in schema:
            types.add("object")
        elif "items" in schema:
            types.add("array")

    return types


def get_argument_types(
    func_name: str, arg_key: str, defined_tools: List[Tool]
) -> Set[str]:
    """Full set of schema-permitted types for one argument of one tool.

    Companion to the detectors' ``get_argument_type`` (primary type only).
    Returns an empty set when the tool, the property, or its type is undeclared.
    """
    for tool in defined_tools:
        if tool.function.name != func_name:
            continue
        parameters = tool.function.parameters or {}
        if not isinstance(parameters, dict):
            return set()
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            return set()
        arg_schema = properties.get(arg_key)
        if not isinstance(arg_schema, dict):
            return set()
        return infer_json_schema_types(arg_schema, parameters)
    return set()


def _reject_json_constant(name: str) -> Any:
    # NaN/Infinity/-Infinity are a Python extension, not valid JSON (RFC 8259).
    # Accepting them here would let a union-typed argument produce a payload
    # that strict client-side parsers (orjson, JS JSON.parse) reject.
    raise ValueError(f"{name} is not a valid JSON literal")


def coerce_union_literal(raw: str, types: Set[str]) -> Tuple[Any, bool]:
    """Recover a bare JSON literal from a union-typed argument value.

    Applies only when the schema permits a non-string type. Pure ``string``
    parameters always keep the raw text, because the template writes string
    values verbatim and re-encoding would lose the original bytes.

    Args:
        raw: Verbatim argument text as rendered by the model.
        types: Result of :func:`infer_json_schema_types` for this argument.

    Returns:
        Tuple of (value, coerced). ``coerced`` is False when the caller should
        keep ``raw`` unchanged.
    """
    if not types or not (types - {"string"}):
        return raw, False

    # ``json.loads`` tolerates surrounding whitespace, but ``tojson`` never
    # emits any, so " null " can only have come from a string.
    if raw != raw.strip():
        return raw, False

    try:
        parsed = json.loads(raw, parse_constant=_reject_json_constant)
    except (JSONDecodeError, ValueError):
        return raw, False

    # A parsed string means the raw text was already quoted; keeping it verbatim
    # preserves those quotes, which are part of the string value itself.
    if (_json_type_names(parsed) - {"string"}) & types:
        return parsed, True
    return raw, False


def still_possible(buf: str, types: Set[str]) -> bool:
    """Whether ``buf`` can still be the prefix of a non-string JSON literal.

    Used by streaming parsers to buffer a union-typed value only until the
    shortest decidable prefix is reached, then resume token-by-token output.
    """
    if not buf:
        return True

    if "null" in types and "null".startswith(buf):
        return True
    if "boolean" in types and ("true".startswith(buf) or "false".startswith(buf)):
        return True
    if types & {"number", "integer"}:
        if buf[0] in _NUMBER_START_CHARS and all(
            char in _NUMBER_BODY_CHARS for char in buf
        ):
            return True
    if "object" in types and buf[0] == "{":
        return True
    if "array" in types and buf[0] == "[":
        return True
    return False


def get_json_schema_constraint(
    tools: List[Tool],
    tool_choice: Union[ToolChoice, Literal["required"]],
    parallel_tool_calls: bool = True,
) -> Optional[dict]:
    """
    Get the JSON schema constraint for the specified tool choice.

    Args:
        tool_choice: The tool choice specification
        parallel_tool_calls: If False, constrain to exactly one tool call (maxItems=1)

    Returns:
        JSON schema dict, or None if no valid tools found
    """

    if isinstance(tool_choice, ToolChoice):
        # For specific function choice, return the user's parameters schema directly
        fn_name = tool_choice.function.name
        for tool in tools:
            if tool.function.name == fn_name:
                schema = {
                    "type": "array",
                    "minItems": 1,
                    "items": _get_tool_schema(tool),
                }
                if not parallel_tool_calls:
                    schema["maxItems"] = 1
                return schema
        return None
    elif tool_choice == "required":
        json_schema = {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": [_get_tool_schema(tool) for tool in tools],
            },
        }
        if not parallel_tool_calls:
            json_schema["maxItems"] = 1
        json_schema_defs = _get_tool_schema_defs(tools)
        if json_schema_defs:
            json_schema["$defs"] = json_schema_defs
        return json_schema

    return None
