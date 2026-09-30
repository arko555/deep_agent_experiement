"""End-to-end tests against a real MCP server over stdio.

`test_validation.py` proves the contract with hand-built doubles, which is fast
but proves less: it assumes the shape MCP actually hands back. These tests run
an actual `FastMCP` server as a subprocess, so the schema, the name prefix, and
the transport are all genuine.

Skipped, not failed, when the `mcp` package is unavailable — the payload
contract must not depend on a test-only dependency being installed.
"""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="the mcp package is needed to run a real server")

from src.services.tools_integration.mcp_client import clear_mcp_tools_cache, load_mcp_tools

SERVER = Path(__file__).parent / "fixtures" / "mcp_server.py"


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Configure a real stdio MCP server and clear the tool cache around it."""
    receipt = tmp_path / "receipts.jsonl"
    config = {
        "fixture": {
            "command": sys.executable,
            "args": [str(SERVER), str(receipt)],
            "transport": "stdio",
        }
    }
    monkeypatch.setenv("MCP_SERVERS", json.dumps(config))
    clear_mcp_tools_cache()
    yield receipt
    clear_mcp_tools_cache()


def _receipts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class TestAgainstRealMCPServer:

    def test_tools_load_with_the_server_name_prefix(self, server):
        """`tool_name_prefix=True` namespaces tools as `<server>_<tool>`.

        This is what lets a SKILL.md claim a whole server with `fixture*`, and
        it is also how a tool is recognizably an MCP tool.
        """
        tools = load_mcp_tools()
        assert "fixture_search_employee" in tools
        assert "fixture_ping" in tools

    def test_the_real_schema_is_normalized_to_a_model(self, server):
        """MCP's raw JSON Schema dict must not reach `args_schema` as a dict.

        langchain-core skips validation for a dict schema, so passing one
        through verbatim is exactly the bug this work fixes.
        """
        from pydantic import BaseModel

        tool = load_mcp_tools()["fixture_search_employee"]
        assert isinstance(tool.args_schema, type)
        assert issubclass(tool.args_schema, BaseModel)
        assert "employee_id" in tool.args_schema.model_fields

    def test_a_valid_call_reaches_the_server(self, server):
        tool = load_mcp_tools()["fixture_search_employee"]
        assert tool.invoke({"employee_id": "WD-1"}) == "employee:WD-1"

        received = _receipts(server)
        assert received == [{
            "tool": "search_employee",
            "args": {"employee_id": "WD-1", "include_terminated": False},
        }]

    def test_a_wrong_typed_argument_never_reaches_the_server(self, server):
        """The motivating regression, against a genuine MCP server.

        The server itself is typed (`employee_id: str`), so without
        normalization the bad value would be sent over the wire and fail
        wherever the server happens to interpret it.
        """
        tool = load_mcp_tools()["fixture_search_employee"]

        with pytest.raises(ValueError, match="employee_id"):
            tool.invoke({"employee_id": 12345})

        assert _receipts(server) == []

    def test_a_missing_required_argument_never_reaches_the_server(self, server):
        tool = load_mcp_tools()["fixture_search_employee"]

        with pytest.raises(ValueError, match="employee_id"):
            tool.invoke({})

        assert _receipts(server) == []

    def test_an_undeclared_argument_never_reaches_the_server(self, server):
        tool = load_mcp_tools()["fixture_search_employee"]

        with pytest.raises(ValueError):
            tool.invoke({"employee_id": "WD-1", "bogus": 1})

        assert _receipts(server) == []

    def test_a_no_arg_tool_is_callable_and_still_refuses_arguments(self, server):
        """A real server's no-arg tool: callable, but not a free-for-all.

        langchain-core discards input entirely for a fieldless model, so this
        only holds because `MCPTool.run` validates the raw input.
        """
        tool = load_mcp_tools()["fixture_ping"]

        assert tool.invoke({}) == "pong"

        with pytest.raises(ValueError):
            tool.invoke({"unexpected": 1})

        assert _receipts(server) == [{"tool": "ping", "args": {}}]

    def test_dispatch_routes_an_mcp_tool_by_its_kind(self, server):
        """`registry.dispatch` must reach the MCP tool, not a local function.

        The point of `kind` is that the call site does not know or care which
        transport a tool uses; this asserts the routing works on a real one.
        """
        from src.services.tools_integration.registry import ToolRegistry
        from src.types import ToolKind

        tools = load_mcp_tools()
        registry = ToolRegistry()
        for name, tool in tools.items():
            registry.register_builtin(name, tool, allowed_roles=("*",))

        assert registry.kind_of("fixture_search_employee") == ToolKind.MCP
        assert registry.dispatch(
            "fixture_search_employee", {"employee_id": "WD-2"}
        ) == "employee:WD-2"

        with pytest.raises(ValueError, match="employee_id"):
            registry.dispatch("fixture_search_employee", {"employee_id": 99})

        assert [r["args"]["employee_id"] for r in _receipts(server)] == ["WD-2"]

    def test_the_subagent_tool_loop_dispatches_over_mcp(self, server):
        """The full path: a sub-agent's ReAct loop calling a real MCP tool.

        This is the seam that matters — the loop is where a model's payload
        first meets a transport, and where a rejected payload has to come back
        as a message the sub-agent can correct rather than a crash.
        """
        from langchain_core.messages import AIMessage, ToolMessage

        from src.services.agent_orchestrator import agent_factory, subagents

        tool = load_mcp_tools()["fixture_search_employee"]
        seen: list[str] = []

        class Model:
            def __init__(self):
                self.step = 0

            def bind_tools(self, tools, **kwargs):
                return self

            def invoke(self, messages, **kwargs):
                self.step += 1
                for m in messages:
                    if isinstance(m, ToolMessage):
                        seen.append(str(m.content))
                if self.step == 1:
                    # Wrong type on purpose: the loop must not crash on it.
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "fixture_search_employee",
                            "args": {"employee_id": 999},
                            "id": "c1",
                        }],
                    )
                if self.step == 2:
                    return AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "fixture_search_employee",
                            "args": {"employee_id": "WD-3"},
                            "id": "c2",
                        }],
                    )
                return AIMessage(content="Found the employee.")

        agent_factory.get_model = lambda: Model()  # type: ignore[assignment]
        subagents._TOOL_REGISTRY = None

        final, _usage, _writes = subagents.run_tool_loop(
            system_prompt="You are the hr sub-agent.",
            description="Find employee WD-3.",
            tools=[tool],
        )

        # The bad call was refused and explained...
        assert any("employee_id" in s for s in seen)
        # ...the corrected call went through...
        assert [r["args"]["employee_id"] for r in _receipts(server)] == ["WD-3"]
        # ...and the loop finished rather than raising.
        assert "Found the employee." in final
