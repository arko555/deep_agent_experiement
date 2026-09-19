"""Tests for ToolRegistry (Phase 2.1)."""

import pytest

from src.services.tools_integration.registry import (
    ToolRegistry,
    RequiresApprovalError,
)
from src.services.tools_integration.decorator import ToolSpecMetadata
from src.types import ToolSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return ToolRegistry()


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


# ---------------------------------------------------------------------------
# get_visible_tools (20 cap)
# ---------------------------------------------------------------------------

class TestGetVisibleTools:

    def test_general_purpose_sees_all_tools(self, registry):
        for i in range(5):
            registry.register_builtin(f"tool_{i}", lambda i=i: i)
        visible = registry.get_visible_tools("general-purpose")
        assert len(visible) == 5

    def test_visible_tools_capped_at_20(self, registry):
        for i in range(25):
            registry.register_builtin(f"t_{i}", lambda i=i: i)
        visible = registry.get_visible_tools("general-purpose")
        assert len(visible) == 20

    def test_role_scoped_visible_tools(self, registry):
        registry.register_builtin("a", lambda: 1, allowed_roles=("hr",))
        registry.register_builtin("b", lambda: 2, allowed_roles=("hr",))
        registry.register_builtin("c", lambda: 3)  # visible to all
        visible = registry.get_visible_tools("hr_user")
        names = {t.name for t in visible}
        assert names == {"a", "b", "c"}

    def test_role_prefix_resolution(self, registry):
        """Sub-agent name 'hr_specialist' resolves role to 'hr'."""
        registry.register_builtin("salary", lambda: 1, allowed_roles=("hr",))
        registry.register_builtin("report", lambda: 2)
        visible = registry.get_visible_tools("hr_specialist")
        names = {t.name for t in visible}
        assert "salary" in names


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
