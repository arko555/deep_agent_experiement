"""Argument validation against a tool's declared parameter schema.

Every tool call carries a payload the model produced, and that payload has to
match the tool's parameter spec or the call is wrong. This module is where
that is enforced.

The reason it exists is a specific gap. LangChain validates a tool call only
when ``args_schema`` is a Pydantic model, which is what ``@tool`` produces for
our built-ins. An MCP server, by contrast, describes its tools with a raw JSON
Schema **dict**, and LangChain does not validate against a dict at all: a
wrong-typed argument, an unknown key, and a missing required argument all pass
straight through to the server. ``normalize_schema`` closes that gap by turning
a JSON Schema into a real model, so MCP tools validate exactly like built-ins.

Rejection, not coercion. A payload that does not match the spec is refused with
a message naming the offending field. ``subagents.run_tool_loop`` already turns
an exception into a ``ToolMessage`` and gives the sub-agent another turn, so the
model sees precisely what was wrong and can correct it. Quietly repairing a
payload would hand the tool something the model never asked for.
"""

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

logger = logging.getLogger(__name__)

# Unknown keys are rejected rather than dropped. A payload the tool never
# declared is a mistake by the model, and dropping it silently would run the
# tool with less than the caller asked for — the same silent corruption this
# module exists to prevent. The error names the field, so the sub-agent can
# correct it on its next turn.
_STRICT_CONFIG = ConfigDict(extra="forbid")


def _python_type(json_type: Any) -> Any:
    """Map a JSON Schema type name to a Python annotation.

    Anything unrecognized becomes ``Any`` — an unmodelled field accepts
    whatever it is given instead of rejecting a value the server would have
    accepted itself.
    """
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }.get(json_type, Any)


def _field_annotation(prop: dict[str, Any]) -> Any:
    """Build the annotation for one property, recursing into nested objects.

    A nested ``object`` with its own ``properties`` becomes a nested model, so
    a nested payload is checked field by field rather than accepted wholesale.
    Arrays are annotated ``List[item_type]`` when the item type is known.
    """
    json_type = prop.get("type")

    if json_type == "object":
        nested = prop.get("properties")
        if isinstance(nested, dict) and nested:
            return _model_from_properties(
                nested, prop.get("required") or []
            )
        return dict

    if json_type == "array":
        items = prop.get("items")
        if isinstance(items, dict):
            item_type = _python_type(items.get("type"))
            if item_type is not Any:
                return list[item_type]  # type: ignore[valid-type]
        return list

    # A union type (e.g. ["string", "null"]) takes its first concrete member.
    if isinstance(json_type, list):
        for candidate in json_type:
            resolved = _python_type(candidate)
            if resolved is not Any:
                return resolved

    enum = prop.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(v, str) for v in enum):
        return Literal[tuple(enum)]  # type: ignore[valid-type]

    return _python_type(json_type)


def _model_from_properties(
    properties: dict[str, Any], required: list[str]
) -> type[BaseModel]:
    """Build a Pydantic model from JSON Schema ``properties`` + ``required``."""
    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            fields[name] = (Any, None)
            continue
        annotation = _field_annotation(prop)
        if name in required:
            fields[name] = (annotation, Field(...))
        else:
            # Nullable with a None default so omitting it validates, but the
            # default is stripped again by `prune_unset` — a default in the
            # model is a means of accepting an omitted field, not a statement
            # that the tool wants a null there.
            fields[name] = (annotation | None, None)

    model: type[BaseModel] = create_model(  # type: ignore[call-overload]
        "ToolArgs", __config__=_STRICT_CONFIG, **fields
    )
    return model


def normalize_schema(schema: Any) -> Any:
    """Return a Pydantic model for *schema*, or the schema unchanged.

    MCP supplies a raw JSON Schema dict; ``@tool`` already supplies a model.
    A dict is converted so ``BaseTool.invoke`` validates it. Anything else —
    an existing model, ``None``, a malformed dict — is returned as-is, so this
    is safe to call on any tool and never turns a working tool into a broken
    one.
    """
    if schema is None:
        return None
    # Already a model class: @tool built it, and LangChain will validate it.
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema
    if not isinstance(schema, dict):
        logger.debug("Unrecognized args_schema type %s; leaving unchanged",
                     type(schema).__name__)
        return schema

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        # A no-arg tool legitimately has an empty properties map. An empty
        # model still rejects unexpected arguments, which is correct.
        if properties is None and schema.get("type") == "object":
            return _model_from_properties({}, [])
        logger.warning(
            "Tool schema has no usable 'properties' (%r); arguments will not "
            "be validated.", schema.get("type"),
        )
        return schema

    required = schema.get("required")
    if not isinstance(required, list):
        required = []
    return _model_from_properties(properties, [r for r in required if isinstance(r, str)])


def prune_unset(validated: dict[str, Any]) -> dict[str, Any]:
    """Drop keys that are ``None`` because the caller omitted them.

    langchain-core's ``BaseTool._parse_input`` passes along *every* field that
    has a default, so a normalized schema makes each absent optional arrive at
    the tool as an explicit ``None``. Left alone, an MCP server would receive
    ``{"employee_id": "WD-1", "include_terminated": null}`` for a call that
    declared no such argument — a payload that does not match the spec, which
    is the exact failure this module exists to prevent.

    A field the schema genuinely allows to be null keeps its ``None``; a caller
    that omits it and a caller that passes null are then indistinguishable
    here, which is safe: both are valid for a nullable field.

    Recurses into nested objects, whose models carry the same defaults.
    """
    return {
        k: prune_unset(v) if isinstance(v, dict) else v
        for k, v in validated.items()
        if v is not None
    }


def validate_args(schema: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Validate *args* against *schema*; return them, or raise ValueError.

    The error message names the field and what was expected, because it is fed
    straight back to the model as a ``ToolMessage`` — a vague "invalid
    arguments" gives it nothing to correct itself with.

    Returns the validated arguments with omitted optionals pruned, so the
    result is exactly the payload the tool declared.
    """
    model = normalize_schema(schema)
    if model is None or not (isinstance(model, type) and issubclass(model, BaseModel)):
        # No usable schema: the tool's own call path is the only check there is.
        return args

    try:
        validated = model(**(args or {}))
    except Exception as e:
        raise ValueError(_format_error(e, args)) from e
    return prune_unset(validated.model_dump())


def _format_error(exc: Exception, args: dict[str, Any]) -> str:
    """Render a Pydantic validation error as guidance for the model."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            details = errors()
        except Exception:
            details = []
        if details:
            parts = []
            for err in details[:5]:
                field = ".".join(str(p) for p in err.get("loc", ())) or "<root>"
                parts.append(f"{field}: {err.get('msg', 'invalid')}")
            supplied = ", ".join(sorted(args)) or "none"
            return (
                "Invalid arguments for this tool. "
                + "; ".join(parts)
                + f". (supplied: {supplied})"
            )
    return f"Invalid arguments for this tool: {exc}"


def tool_spec_for(tool: Any) -> Any:
    """Best-effort read of a LangChain tool's declared argument schema."""
    return getattr(tool, "args_schema", None)


class ValidatedTool:
    """Mixin that validates a tool's raw input against its declared schema.

    Normalizing ``args_schema`` to a Pydantic model is not on its own enough.
    langchain-core validates in ``_parse_input``, and ``_to_args_and_kwargs``
    short-circuits past that entirely when the model has no fields
    ("StructuredTool with no args") — so a no-arg tool discards whatever it is
    handed, and a declared-only field with an unexpected type can slip through
    on paths that never reach the parse.

    Validating here, on the raw input, makes the guarantee independent of which
    langchain path a call arrives by. The error is a ``ValueError``, which
    ``subagents.run_tool_loop`` already turns into a ``ToolMessage`` so the
    sub-agent can correct the payload on its next turn.
    """

    def _validate_raw_input(self, tool_input: Any) -> None:
        if not isinstance(tool_input, dict):
            return
        validate_args(self.args_schema, tool_input)  # type: ignore[attr-defined]
