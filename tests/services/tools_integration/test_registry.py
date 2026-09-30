"""Tests for ToolRegistry (Phase 2.1)."""

import json
import logging
from typing import Optional

import pytest
from pydantic import ConfigDict, create_model

from src.services.tools_integration.registry import (
    ToolRegistry,
    RequiresApprovalError,
)
from src.services.tools_integration.decorator import ToolSpecMetadata
from src.types import ToolKind, ToolSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return ToolRegistry()


class _Tool:
    """A minimal `BaseTool` stand-in that records what it was called with.

    Real `BaseTool` subclassing needs a pydantic class per schema, which would
    obscure what these tests are about: that dispatch validates the payload and
    then routes the call. The parts dispatch actually reads — ``args_schema``
    and ``kind`` — are plain attributes here.

    Declared fields are all optional, mirroring what `@tool` produces for a
    function with defaults: a required-argument test belongs in
    ``test_validation.py``, where the schemas are explicit. ``extra="forbid"``
    matches what ``validation.normalize_schema`` builds, so an undeclared key
    is refused here the same way it would be for an MCP tool.
    """

    def __init__(self, name, description, fields=None, kind="local", result=None):
        self.name = name
        self.description = description
        self.args_schema = (
            create_model(
                f"{name}_args",
                __config__=ConfigDict(extra="forbid"),
                **{
                    f: (Optional[t], None)  # noqa: UP007 - runtime pydantic needs it
                    for f, t in (fields or {}).items()
                },
            )
            if fields
            else None
        )
        self.kind = kind
        self._result = result
        self.calls = []

    def invoke(self, args):
        self.calls.append((self.kind, args))
        if self._result is not None:
            return self._result
        return f"called:{self.kind}:{list(args.values())[0] if args else ''}"


# ---------------------------------------------------------------------------
# register / get_spec / get_callable
# ---------------------------------------------------------------------------

class TestRegisterAndGet:

    def test_register_and_get_spec(self, registry):
        spec = ToolSpecMetadata(name="my_tool", description="does thing")
        registry.register(spec, callable_=lambda: "ok")
        got = registry.get_spec("my_tool")
        assert got.name == "my_tool"
        assert got.description == "does thing"

    def test_get_spec_missing_returns_none(self, registry):
        assert registry.get_spec("nonexistent") is None

    def test_register_and_get_callable(self, registry):
        registry.register(
            ToolSpecMetadata(name="my_tool", description="d"),
            callable_=lambda x: x + 1,
        )
        assert registry.get_callable("my_tool") is not None
        assert registry.get_callable("my_tool")(5) == 6

    def test_get_callable_missing(self, registry):
        assert registry.get_callable("missing") is None

    def test_register_ToolSpec_dataclass(self, registry):
        spec = ToolSpec(name="ts_tool", description="from dataclass")
        registry.register(spec, callable_=lambda: "done")
        assert registry.get_spec("ts_tool").description == "from dataclass"

    def test_register_tool_spec_from_dataclass(self, registry):
        """ToolSpec dataclass is converted to ToolSpecMetadata correctly."""
        spec = ToolSpec(
            name="role_tool",
            description="restricted",
            risk_level="high",
            requires_approval=True,
            allowed_roles=["hr"],
        )
        registry.register(spec)
        got = registry.get_spec("role_tool")
        assert got.risk_level == "high"
        assert got.requires_approval is True
        assert got.allowed_roles == ("hr",)


# ---------------------------------------------------------------------------
# register_builtin
# ---------------------------------------------------------------------------

class TestRegisterBuiltin:

    def test_builtin_uses_docstring_as_description(self, registry):
        def my_func():
            """Does the thing."""
            return 42

        registry.register_builtin("my_func", my_func)
        assert registry.get_spec("my_func").description == "Does the thing."

    def test_builtin_executable(self, registry):
        registry.register_builtin("double", lambda x: x * 2)
        result = registry.execute("double", {"x": 5}, "general-purpose")
        assert result == 10

    def test_builtin_with_risk_and_approval(self, registry):
        registry.register_builtin(
            "delete_all", lambda: "gone", risk_level="high",
            requires_approval=True,
        )
        assert registry.get_spec("delete_all").requires_approval is True
        assert registry.get_spec("delete_all").risk_level == "high"


# ---------------------------------------------------------------------------
# get_tools_for_role
# ---------------------------------------------------------------------------

class TestGetToolsForRole:

    def test_general_accessible_tools_appear_for_any_role(self, registry):
        registry.register_builtin("read", lambda: "ok")
        tools = registry.get_tools_for_role("hr")
        assert len(tools) == 1
        assert tools[0].name == "read"

    def test_role_restricted_tools_only_for_allowed_roles(self, registry):
        registry.register_builtin("payroll", lambda: "data",
                                   allowed_roles=("hr", "payroll"))
        hr_tools = registry.get_tools_for_role("hr")
        assert len(hr_tools) == 1
        assert hr_tools[0].name == "payroll"

        eng_tools = registry.get_tools_for_role("engineering")
        assert len(eng_tools) == 0

    def test_multiple_restricted_tools_filter_correctly(self, registry):
        registry.register_builtin("a", lambda: 1, allowed_roles=("hr",))
        registry.register_builtin("b", lambda: 2, allowed_roles=("eng",))
        registry.register_builtin("c", lambda: 3)  # visible to all

        hr_tools = {t.name for t in registry.get_tools_for_role("hr")}
        assert hr_tools == {"a", "c"}

        eng_tools = {t.name for t in registry.get_tools_for_role("eng")}
        assert eng_tools == {"b", "c"}

    def test_wildcard_role_allows_all(self, registry):
        registry.register_builtin("x", lambda: 1, allowed_roles=("*",))
        tools = registry.get_tools_for_role("anything_goes")
        assert len(tools) == 1

    def test_empty_allowed_roles_means_all_roles(self, registry):
        registry.register_builtin("open", lambda: 1)
        for role in ("hr", "eng", "admin", "payroll"):
            assert len(registry.get_tools_for_role(role)) == 1


# `get_visible_tools(subagent_name)` used to be tested here: a sub-agent's
# tools resolved by role, capped at 20, with `general-purpose` seeing
# everything. A sub-agent's tools are now decided by its SKILL.md
# `allowed-tools` allowlist in `subagents.select_department_tools`, including
# the 20-tool visibility cap — covered in `tests/test_subagents.py`. Keeping
# the registry's own role-based filter tested here would only assert a
# mechanism nothing calls.


# ---------------------------------------------------------------------------
# get_tool_definitions
# ---------------------------------------------------------------------------

class TestGetToolDefinitions:

    def test_returns_schemas_for_registered_tools(self, registry):
        registry.register_builtin("read", lambda path: "", risk_level="medium")
        registry.register_builtin("write", lambda path, c: "",
                                   requires_approval=True)
        defs = registry.get_tool_definitions(["read", "write"])
        assert len(defs) == 2
        read_def = next(d for d in defs if d["name"] == "read")
        assert read_def["risk_level"] == "medium"
        assert read_def["requires_approval"] is False

    def test_skips_unknown_names(self, registry):
        registry.register_builtin("known", lambda: 1)
        defs = registry.get_tool_definitions(["known", "unknown"])
        assert len(defs) == 1
        assert defs[0]["name"] == "known"

    def test_empty_list_returns_empty(self, registry):
        assert registry.get_tool_definitions([]) == []


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------

class TestExecute:

    def test_execute_registered_tool(self, registry):
        registry.register_builtin("echo", lambda msg: msg)
        result = registry.execute("echo", {"msg": "hello"}, "general-purpose")
        assert result == "hello"

    def test_execute_raises_for_unregistered(self, registry):
        with pytest.raises(ValueError, match="not registered"):
            registry.execute("missing", {}, "general-purpose")

    def test_execute_raises_RequiresApprovalError(self, registry):
        registry.register_builtin("destructive", lambda: "bad",
                                   requires_approval=True)
        with pytest.raises(RequiresApprovalError):
            registry.execute("destructive", {}, "general-purpose")

    def test_execute_raises_PermissionError_for_unauthorized_role(self, registry):
        registry.register_builtin("payroll", lambda: "data",
                                   allowed_roles=("hr",))
        with pytest.raises(PermissionError):
            registry.execute("payroll", {}, "engineer")

    def test_execute_passes_kwargs_to_callable(self, registry):
        registry.register_builtin("add", lambda a, b: a + b)
        result = registry.execute("add", {"a": 3, "b": 7}, "general-purpose")
        assert result == 10

    def test_execute_raises_for_missing_callable(self, registry):
        registry.register(ToolSpecMetadata(name="noop", description="no code"))
        with pytest.raises(ValueError, match="no registered callable"):
            registry.execute("noop", {}, "general-purpose")


# ---------------------------------------------------------------------------
# list_tools
# ---------------------------------------------------------------------------

class TestListTools:

    def test_returns_all_tool_names(self, registry):
        registry.register_builtin("a", lambda: 1)
        registry.register_builtin("b", lambda: 2)
        assert sorted(registry.list_tools()) == ["a", "b"]

    def test_empty_registry(self, registry):
        assert registry.list_tools() == []


# ---------------------------------------------------------------------------
# RequiresApprovalError
# ---------------------------------------------------------------------------

class TestRequiresApprovalError:

    def test_is_exception(self, registry):
        assert issubclass(RequiresApprovalError, Exception)

    def test_message_includes_tool_name(self, registry):
        registry.register_builtin("x", lambda: 1, requires_approval=True)
        with pytest.raises(RequiresApprovalError, match="x"):
            registry.execute("x", {}, "general-purpose")


# ---------------------------------------------------------------------------
# dispatch: schema validation and transport selection
# ---------------------------------------------------------------------------

class TestDispatch:
    """dispatch is the single execution surface.

    Two things have to hold for every call, whatever the transport: the
    payload matches the tool's declared schema, and the tool is reached over
    the transport its declared ``kind`` names.
    """

    def test_valid_payload_reaches_the_tool(self, registry):
        tool = _Tool("echo", "Echo a value.", {"value": str})
        registry.register_builtin("echo", tool)
        assert registry.dispatch("echo", {"value": "hi"}) == "called:local:hi"
        assert tool.calls == [("local", {"value": "hi"})]

    def test_wrong_typed_payload_is_refused_before_the_call(self, registry):
        tool = _Tool("echo", "Echo.", {"value": str})
        registry.register_builtin("echo", tool)
        with pytest.raises(ValueError, match="value"):
            registry.dispatch("echo", {"value": 42})
        assert tool.calls == []

    def test_unknown_payload_key_is_refused(self, registry):
        tool = _Tool("echo", "Echo.", {"value": str})
        registry.register_builtin("echo", tool)
        with pytest.raises(ValueError):
            registry.dispatch("echo", {"value": "hi", "extra": 1})
        assert tool.calls == []

    def test_unknown_tool_is_refused(self, registry):
        with pytest.raises(ValueError, match="not registered"):
            registry.dispatch("nope", {})

    def test_kind_defaults_to_local(self, registry):
        registry.register_builtin("t", _Tool("t", "d", {}))
        assert registry.kind_of("t") == ToolKind.LOCAL

    def test_kind_is_read_off_the_tool_object(self, registry):
        """A tool constructed with a kind is authoritative for its transport.

        The MCP loader tags tools this way, so dispatch does not have to infer
        the transport from a name prefix at the call site.
        """
        tool = _Tool("svc_do", "d", {}, kind="mcp")
        registry.register_builtin("svc_do", tool)
        assert registry.kind_of("svc_do") == ToolKind.MCP

    def test_unknown_kind_falls_back_to_local(self, registry):
        registry.register_builtin("t", _Tool("t", "d", {}, kind="carrier-pigeon"))
        assert registry.kind_of("t") == ToolKind.LOCAL

    def test_mcp_tool_is_validated_and_invoked_locally(self, registry):
        """MCP and LOCAL differ in transport, not in the payload contract.

        An MCP tool is already a sync-wrapped tool object by the time it is
        registered, so dispatch validates it identically and then invokes it.
        """
        tool = _Tool("svc_do", "d", {"x": int}, kind="mcp")
        registry.register_builtin("svc_do", tool)
        assert registry.dispatch("svc_do", {"x": 1}) == "called:mcp:1"

        with pytest.raises(ValueError, match="x"):
            registry.dispatch("svc_do", {"x": "not an int"})
        assert tool.calls == [("mcp", {"x": 1})]

    def test_a2a_payload_is_serialized_as_json(self, registry, monkeypatch):
        """A2A carries text, so validated args are serialized, not splatted.

        The remote agent parses the same JSON a local call would have
        received, which keeps the payload contract transport-independent.
        """
        sent = {}

        def fake_call(url, description, timeout=None):
            sent["url"] = url
            sent["body"] = description
            return "remote reply"

        monkeypatch.setattr(
            "src.services.tools_integration.a2a_client.call_a2a_agent", fake_call
        )
        registry.register_a2a("remote_researcher", "http://localhost:9999")
        registry.register_builtin("echo", _Tool("echo", "d", {"value": str}))

        # Give the A2A entry a schema so validation is exercised on its path.
        registry._specs["remote_researcher"] = ToolSpecMetadata(
            name="remote_researcher", description="d", kind=ToolKind.A2A
        )

        result = registry.dispatch("remote_researcher", {"q": "quarterly revenue"})

        assert result == "remote reply"
        assert json.loads(sent["body"]) == {"q": "quarterly revenue"}
        assert sent["url"] == "http://localhost:9999"

    def test_a2a_without_a_url_is_refused(self, registry):
        registry._specs["orphan"] = ToolSpecMetadata(
            name="orphan", description="d", kind=ToolKind.A2A
        )
        with pytest.raises(ValueError, match="URL"):
            registry.dispatch("orphan", {})

    def test_approval_and_role_gates_still_apply(self, registry):
        registry.register_builtin(
            "danger", _Tool("danger", "d", {}), requires_approval=True
        )
        with pytest.raises(RequiresApprovalError):
            registry.dispatch("danger", {})

        registry.register_builtin(
            "payroll", _Tool("payroll", "d", {}), allowed_roles=("hr",)
        )
        with pytest.raises(PermissionError):
            registry.dispatch("payroll", {}, subagent_name="writer")

    def test_omitted_optionals_are_not_sent(self, registry):
        """An absent optional must not arrive as an explicit null.

        LangChain injects a default for every field that has one, so a
        schema with optionals would otherwise deliver keys the caller never
        passed — a payload that does not match the spec.
        """
        tool = _Tool("t", "d", {"a": str, "b": int})
        registry.register_builtin("t", tool)
        registry.dispatch("t", {"a": "x"})
        assert tool.calls == [("local", {"a": "x"})]


# ---------------------------------------------------------------------------
# select_for_query
# ---------------------------------------------------------------------------

class TestSelectForQuery:
    """Department allowlist resolution, by query relevance."""

    def test_resolves_names_to_real_tools(self, registry):
        for name in ("alpha", "beta"):
            registry.register_builtin(name, _Tool(name, f"{name} tool", {}))
        chosen = registry.select_for_query(["alpha", "beta"], "q", max_tools=5)
        assert {t.name for t in chosen} == {"alpha", "beta"}

    def test_unknown_names_are_dropped_with_a_warning(self, registry, caplog):
        registry.register_builtin("alpha", _Tool("alpha", "a", {}))
        with caplog.at_level(logging.WARNING):
            chosen = registry.select_for_query(["alpha", "phantom"], "q")
        assert [t.name for t in chosen] == ["alpha"]
        assert "phantom" in caplog.text

    def test_deduplicates(self, registry):
        registry.register_builtin("alpha", _Tool("alpha", "a", {}))
        chosen = registry.select_for_query(["alpha", "alpha"], "q")
        assert len(chosen) == 1

    def test_under_the_cap_returns_everything_without_a_model(self, registry):
        """No model call when the allowlist already fits.

        Relevance sorting costs a round trip; a department with three allowed
        tools for a five-tool budget needs no ranking.
        """
        for i in range(3):
            registry.register_builtin(f"t{i}", _Tool(f"t{i}", "d", {}))
        chosen = registry.select_for_query(["t0", "t1", "t2"], "q", max_tools=5)
        assert len(chosen) == 3

    def test_over_the_cap_shortlists_by_relevance(self, registry, monkeypatch):
        for i in range(8):
            registry.register_builtin(f"t{i}", _Tool(f"t{i}", f"tool {i}", {}))

        def fake_sort(model, query, defs, max_tools):
            # The sorter's ranking is preserved, so the order it returns is
            # the order the department gets.
            return [d for d in defs if d["name"] in {"t5", "t2"}]

        monkeypatch.setattr(
            "src.services.tools_integration.relevance.sort_tools", fake_sort
        )
        chosen = registry.select_for_query(
            [f"t{i}" for i in range(8)], "q", max_tools=2
        )
        assert {t.name for t in chosen} == {"t5", "t2"}

    def test_shorter_than_the_cap_is_returned_unranked(self, registry, monkeypatch):
        """A sorter that returns more than requested must not widen the cap.

        The cap is the bound on what a department sees; trusting the sorter's
        length would let a bad reply hand it the whole allowlist.
        """
        for i in range(8):
            registry.register_builtin(f"t{i}", _Tool(f"t{i}", "d", {}))

        def greedy_sort(model, query, defs, max_tools):
            return defs  # ignores max_tools entirely

        monkeypatch.setattr(
            "src.services.tools_integration.relevance.sort_tools", greedy_sort
        )
        chosen = registry.select_for_query(
            [f"t{i}" for i in range(8)], "q", max_tools=3
        )
        assert len(chosen) == 3

    def test_sorter_failure_returns_the_full_allowlist(self, registry, monkeypatch):
        """An unreachable sorter must not shrink the department's tools.

        Silently narrowing the allowlist would look like a relevance decision
        that never happened, and the department would quietly lose the
        capability it needed. The error is logged instead.
        """
        for i in range(8):
            registry.register_builtin(f"t{i}", _Tool(f"t{i}", "d", {}))

        def boom(*a, **k):
            raise RuntimeError("sorter exploded")

        monkeypatch.setattr(
            "src.services.tools_integration.relevance.sort_tools", boom
        )
        chosen = registry.select_for_query(
            [f"t{i}" for i in range(8)], "q", max_tools=3
        )
        assert [t.name for t in chosen] == [f"t{i}" for i in range(8)]

    def test_empty_sorter_reply_falls_back_to_a_bounded_slice(self, registry, monkeypatch):
        """A sorter that answered but named nothing gets a bounded fallback.

        The opposite of the unreachable case: it did return a judgement, and
        that judgement was "none of these", so the cap still applies.
        """
        for i in range(8):
            registry.register_builtin(f"t{i}", _Tool(f"t{i}", "d", {}))

        monkeypatch.setattr(
            "src.services.tools_integration.relevance.sort_tools",
            lambda *a, **k: [],
        )
        chosen = registry.select_for_query(
            [f"t{i}" for i in range(8)], "q", max_tools=3
        )
        assert [t.name for t in chosen] == ["t0", "t1", "t2"]
