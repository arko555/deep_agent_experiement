"""Tests for tool-argument validation.

The load-bearing test here is :class:`TestMCPToolRejectsBadPayload`. It covers
the reason ``validation.py`` exists: langchain-core does not validate a tool
call against a ``dict`` args_schema, and MCP describes every one of its tools
with a raw JSON Schema dict. Before normalization, a wrong-typed argument, an
unknown key, and a missing required argument all reached the MCP server
untouched.
"""

import pytest
from pydantic import BaseModel

from src.services.tools_integration.mcp_client import _to_sync_tool as client_wrap
from src.services.tools_integration.validation import (
    normalize_schema,
    validate_args,
)


SEARCH_EMPLOYEE = {
    "type": "object",
    "properties": {
        "employee_id": {"type": "string", "description": "Workday worker ID."},
        "include_terminated": {"type": "boolean"},
        "fields": {
            "type": "array",
            "items": {"type": "string"},
        },
        "status": {"type": "string", "enum": ["active", "terminated"]},
    },
    "required": ["employee_id"],
}


class _FakeMCPTool:
    """Stand-in for a ``MultiServerMCPClient`` tool.

    ``ainvoke`` records what the server would actually have received, so a test
    can assert the payload never got that far.
    """

    def __init__(self, name="workday_search_employee", args_schema=SEARCH_EMPLOYEE):
        self.name = name
        self.description = "Search Workday employees."
        self.args_schema = args_schema
        self.received: list[dict] = []

    async def ainvoke(self, args):
        self.received.append(args)
        return "ok"


class TestNormalizeSchema:

    def test_dict_schema_becomes_a_pydantic_model(self):
        model = normalize_schema(SEARCH_EMPLOYEE)
        assert isinstance(model, type) and issubclass(model, BaseModel)
        assert set(model.model_fields) == {
            "employee_id",
            "include_terminated",
            "fields",
            "status",
        }

    def test_existing_model_passes_through_untouched(self):
        class Existing(BaseModel):
            a: str

        assert normalize_schema(Existing) is Existing

    def test_none_stays_none(self):
        assert normalize_schema(None) is None

    def test_unmodellable_schema_is_returned_unchanged(self):
        """A tool with a broken schema must stay callable, just unvalidated.

        Returning the dict unchanged means langchain-core treats it as a plain
        dict schema (no validation), which is the pre-existing behavior — not a
        new failure mode.
        """
        weird = {"type": "string"}
        assert normalize_schema(weird) is weird

    def test_required_field_has_no_default(self):
        model = normalize_schema(SEARCH_EMPLOYEE)
        assert model.model_fields["employee_id"].is_required()
        assert not model.model_fields["include_terminated"].is_required()


class TestValidateArgs:

    def test_valid_payload_round_trips(self):
        args = {"employee_id": "WD-1", "include_terminated": True}
        assert validate_args(SEARCH_EMPLOYEE, args) == {
            "employee_id": "WD-1",
            "include_terminated": True,
        }

    def test_omitted_optionals_are_not_invented(self):
        """An absent optional must stay absent.

        The model needs a default for the field to validate when omitted, but
        the payload sent to the tool must not acquire keys the caller never
        passed — a nullable ``include_terminated`` is not what the spec asked
        for.
        """
        assert validate_args(SEARCH_EMPLOYEE, {"employee_id": "WD-1"}) == {
            "employee_id": "WD-1"
        }

    def test_wrong_type_is_rejected(self):
        with pytest.raises(ValueError, match="employee_id"):
            validate_args(SEARCH_EMPLOYEE, {"employee_id": 12345})

    def test_missing_required_is_rejected(self):
        with pytest.raises(ValueError, match="employee_id"):
            validate_args(SEARCH_EMPLOYEE, {})

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ValueError):
            validate_args(SEARCH_EMPLOYEE, {"employee_id": "WD-1", "bogus": 1})

    def test_enum_violation_is_rejected(self):
        with pytest.raises(ValueError, match="status"):
            validate_args(SEARCH_EMPLOYEE, {"employee_id": "WD-1", "status": "nope"})

    def test_array_item_type_is_checked(self):
        with pytest.raises(ValueError, match="fields"):
            validate_args(SEARCH_EMPLOYEE, {"employee_id": "WD-1", "fields": [1, 2]})

    def test_nested_object_is_validated_field_by_field(self):
        schema = {
            "type": "object",
            "properties": {
                "window": {
                    "type": "object",
                    "properties": {"from": {"type": "string"},
                                   "to": {"type": "string"}},
                    "required": ["from"],
                }
            },
            "required": ["window"],
        }
        ok = validate_args(schema, {"window": {"from": "2026-01-01"}})
        assert ok["window"] == {"from": "2026-01-01"}

        with pytest.raises(ValueError, match=r"window\.from"):
            validate_args(schema, {"window": {"to": "2026-02-01"}})

    def test_error_names_the_field_and_lists_what_was_sent(self):
        """The message is fed back to the model as a ToolMessage.

        It has to be actionable enough for the sub-agent to correct itself,
        so it names the field, the expectation, and the supplied keys.
        """
        with pytest.raises(ValueError) as excinfo:
            validate_args(SEARCH_EMPLOYEE, {"employee_id": 1, "extra": "x"})
        message = str(excinfo.value)
        assert "employee_id" in message
        assert "supplied: employee_id, extra" in message

    def test_no_schema_is_a_passthrough(self):
        assert validate_args(None, {"anything": 1}) == {"anything": 1}


class TestMCPToolRejectsBadPayload:
    """The regression: a bad payload to an MCP tool must not reach the server."""

    def test_wrapping_produces_a_validating_schema(self):
        tool = client_wrap(_FakeMCPTool())
        model = normalize_schema(tool.args_schema)
        assert isinstance(model, type) and issubclass(model, BaseModel)

    def test_wrong_typed_argument_never_reaches_the_server(self):
        mcp_tool = _FakeMCPTool()
        tool = client_wrap(mcp_tool)

        # ValueError, not a pydantic error: `run_tool_loop` catches Exception
        # either way, but the message is what the model reads back, and
        # validation.py renders it in model-facing terms.
        with pytest.raises(ValueError, match="employee_id"):
            tool.invoke({"employee_id": 12345})

        assert mcp_tool.received == []

    def test_missing_required_argument_never_reaches_the_server(self):
        mcp_tool = _FakeMCPTool()
        tool = client_wrap(mcp_tool)

        with pytest.raises(ValueError, match="employee_id"):
            tool.invoke({})

        assert mcp_tool.received == []

    def test_valid_argument_does_reach_the_server(self, monkeypatch):
        """Normalization must not break the happy path.

        The wrapper's func drives an async MCP call, so the coroutine is
        stubbed out — what is under test is the payload, not the transport.
        """
        mcp_tool = _FakeMCPTool()
        tool = client_wrap(mcp_tool)

        async def fake_ainvoke(args):
            mcp_tool.received.append(args)
            return "ok"

        mcp_tool.ainvoke = fake_ainvoke

        result = tool.invoke({"employee_id": "WD-1"})

        assert result == "ok"
        assert mcp_tool.received == [{"employee_id": "WD-1"}]

    def test_no_arg_tool_stays_callable(self, monkeypatch):
        """A server may omit its schema for a no-arg tool.

        Such a tool must still be invocable with no arguments, and must still
        refuse an argument it never declared.
        """
        mcp_tool = _FakeMCPTool(name="workday_ping", args_schema=None)
        tool = client_wrap(mcp_tool)

        async def fake_ainvoke(args):
            mcp_tool.received.append(args)
            return "pong"

        mcp_tool.ainvoke = fake_ainvoke

        assert tool.invoke({}) == "pong"

        # langchain-core short-circuits a fieldless model in
        # `_to_args_and_kwargs` and discards the input without parsing it, so
        # this only raises because `MCPTool.run` validates the raw input.
        with pytest.raises(ValueError, match="unexpected"):
            tool.invoke({"unexpected": 1})

        assert mcp_tool.received == [{}]


class TestSubagentSeesTheError:
    """A rejected call must come back as a ToolMessage the model can act on."""

    def test_run_tool_loop_feeds_the_validation_error_back(self, monkeypatch):
        """The loop's own error handling is the repair mechanism.

        ``run_tool_loop`` catches the exception and returns it as a
        ``ToolMessage``, so the sub-agent gets another turn and can correct the
        payload. This is only true if validation *raises* — hence the
        bad-payload-first model below.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        from src.services.agent_orchestrator import subagents

        seen: list[str] = []

        class OneBadCallThenAnswer:
            """Emits one malformed call, then answers without tool calls."""

            def __init__(self):
                self._step = 0

            def bind_tools(self, tools, **kwargs):
                return self

            def invoke(self, messages, **kwargs):
                self._step += 1
                for m in messages:
                    if isinstance(m, ToolMessage):
                        seen.append(str(m.content))
                if self._step == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "workday_search_employee",
                            "args": {"employee_id": 999},
                            "id": "call-1",
                        }],
                    )
                return AIMessage(content="Recovered.")

        monkeypatch.setattr(
            "src.services.agent_orchestrator.agent_factory.get_model",
            lambda: OneBadCallThenAnswer(),
        )

        mcp_tool = _FakeMCPTool()
        tool = client_wrap(mcp_tool)

        final_text, _usage, _writes = subagents.run_tool_loop(
            system_prompt="You are the hr sub-agent.",
            description="Find employee 999.",
            tools=[tool],
        )

        # The malformed payload never reached the server...
        assert mcp_tool.received == []
        # ...it came back to the model as a message naming the bad field...
        assert any("employee_id" in s for s in seen)
        # ...and the loop continued to a final answer rather than raising.
        assert "Recovered." in final_text


class TestMCPResultUnwrapping:
    """A sub-agent must receive text, not MCP protocol structure."""

    def test_text_blocks_are_joined(self):
        from src.services.tools_integration.mcp_client import _unwrap_mcp_result

        blocks = [
            {"type": "text", "text": "line one", "id": "a"},
            {"type": "text", "text": "line two", "id": "b"},
        ]
        assert _unwrap_mcp_result(blocks) == "line one\nline two"

    def test_single_text_block_becomes_a_plain_string(self):
        from src.services.tools_integration.mcp_client import _unwrap_mcp_result

        assert _unwrap_mcp_result([{"type": "text", "text": "ok"}]) == "ok"

    def test_non_text_blocks_are_named_not_dropped(self):
        """Silently discarding content would misrepresent the server's answer."""
        from src.services.tools_integration.mcp_client import _unwrap_mcp_result

        blocks = [{"type": "image", "data": "..."}, {"type": "text", "text": "see above"}]
        out = _unwrap_mcp_result(blocks)
        assert "see above" in out
        assert "image" in out

    def test_already_unwrapped_values_pass_through(self):
        from src.services.tools_integration.mcp_client import _unwrap_mcp_result

        assert _unwrap_mcp_result("plain") == "plain"
        assert _unwrap_mcp_result(None) is None
        assert _unwrap_mcp_result([]) == []
        # Not content blocks: left alone rather than mangled.
        assert _unwrap_mcp_result([1, 2]) == [1, 2]
