"""Tests for ToolExecutor (Phase 2.5)."""

import asyncio
import pytest

from src.services.tools_integration.executor import ToolExecutor
from src.services.tools_integration.registry import (
    ToolRegistry,
    RequiresApprovalError,
)
from src.services.tools_integration.decorator import ToolSpecMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def executor(registry):
    return ToolExecutor(registry)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestRegistryProperty:

    def test_executor_wraps_given_registry(self, executor, registry):
        assert executor.registry is registry

    def test_executor_creates_default_registry(self, executor):
        assert isinstance(executor.registry, ToolRegistry)


# ---------------------------------------------------------------------------
# execute (async)
# ---------------------------------------------------------------------------

class TestExecute:

    def test_executes_low_risk_tool(self, executor):
        spec = ToolSpecMetadata("low", "low-risk", risk_level="low")
        executor.register(spec, callable_=lambda x: x * 2)
        result = asyncio.run(executor.execute("low", {"x": 5}, "worker"))
        assert result == 10

    def test_executes_medium_risk_tool(self, executor):
        spec = ToolSpecMetadata("med", "medium-risk", risk_level="medium")
        executor.register(spec, callable_=lambda: "ok")
        result = asyncio.run(executor.execute("med", {}, "worker"))
        assert result == "ok"

    def test_raises_for_high_risk_tool(self, executor):
        spec = ToolSpecMetadata("high", "high-risk",
                                  risk_level="high",
                                  requires_approval=True)
        executor.register(spec, callable_=lambda: "no")
        with pytest.raises(RequiresApprovalError):
            asyncio.run(executor.execute("high", {}, "worker"))

    def test_raises_for_unregistered_tool(self, executor):
        with pytest.raises(ValueError):
            asyncio.run(executor.execute("ghost", {}, "worker"))

    def test_raises_PermissionError_for_role(self, executor):
        spec = ToolSpecMetadata("restricted", "restricted",
                                  allowed_roles=("hr",))
        executor.register(spec, callable_=lambda: "no")
        with pytest.raises(PermissionError):
            asyncio.run(executor.execute("restricted", {}, "engineer"))

    def test_allows_registered_role(self, executor):
        spec = ToolSpecMetadata("restricted", "restricted",
                                  allowed_roles=("hr",))
        executor.register(spec, callable_=lambda: "yes")
        result = asyncio.run(executor.execute("restricted", {}, "hr_admin"))
        assert result == "yes"

    def test_trace_id_passed_to_registry(self, executor, monkeypatch):
        """Trace ID is forwarded to the registry's execute method."""
        captured = {}

        class TracedRegistry(ToolRegistry):
            def execute(self, *args, trace_id=None, **kwargs):
                captured["trace_id"] = trace_id
                return super().execute(*args, **kwargs)

        traced = TracedRegistry()
        traced.register(ToolSpecMetadata("t", "d"), callable_=lambda: "ok")
        ex = ToolExecutor(traced)
        asyncio.run(ex.execute("t", {}, "worker", trace_id="abc123"))
        assert captured["trace_id"] == "abc123"

    def test_default_trace_id_is_none(self, executor, monkeypatch):
        captured = {}

        class TracedRegistry(ToolRegistry):
            def execute(self, *args, trace_id=None, **kwargs):
                captured["trace_id"] = trace_id
                return super().execute(*args, **kwargs)

        traced = TracedRegistry()
        traced.register(ToolSpecMetadata("t", "d"), callable_=lambda: "ok")
        ex = ToolExecutor(traced)
        asyncio.run(ex.execute("t", {}, "worker"))
        assert captured["trace_id"] is None


# ---------------------------------------------------------------------------
# execute_sync
# ---------------------------------------------------------------------------

class TestExecuteSync:

    def test_sync_wrapper_runs_async_execute(self, executor):
        spec = ToolSpecMetadata("sync", "s", risk_level="low")
        executor.register(spec, callable_=lambda: "done")
        result = executor.execute_sync("sync", {}, "worker")
        assert result == "done"

    def test_sync_wrapper_propagates_approval_error(self, executor):
        spec = ToolSpecMetadata("no", "n", risk_level="high",
                                  requires_approval=True)
        executor.register(spec, callable_=lambda: "nope")
        with pytest.raises(RequiresApprovalError):
            executor.execute_sync("no", {}, "worker")


# ---------------------------------------------------------------------------
# register (delegate)
# ---------------------------------------------------------------------------

class TestRegister:

    def test_register_delegates_to_registry(self, executor, registry):
        spec = ToolSpecMetadata("delegated", "d", risk_level="low")
        executor.register(spec, callable_=lambda: "ok")
        assert registry.get_spec("delegated") is not None
        assert registry.get_spec("delegated").name == "delegated"

    def test_register_without_callable(self, executor, registry):
        spec = ToolSpecMetadata("no_code", "d")
        executor.register(spec)
        assert registry.get_spec("no_code") is not None
        assert registry.get_callable("no_code") is None


# ---------------------------------------------------------------------------
# Risk tier summary
# ---------------------------------------------------------------------------

class TestRiskTiers:

    def test_low_risk_executes_without_issues(self, executor):
        spec = ToolSpecMetadata("t", "d", risk_level="low")
        executor.register(spec, callable_=lambda: "ok")
        result = asyncio.run(executor.execute("t", {}, "worker"))
        assert result == "ok"

    def test_medium_risk_executes_after_validation(self, executor):
        spec = ToolSpecMetadata("t", "d", risk_level="medium")
        executor.register(spec, callable_=lambda: "ok")
        result = asyncio.run(executor.execute("t", {}, "worker"))
        assert result == "ok"

    def test_high_risk_requires_approval(self, executor):
        spec = ToolSpecMetadata("t", "d", risk_level="high",
                                  requires_approval=True)
        executor.register(spec, callable_=lambda: "ok")
        with pytest.raises(RequiresApprovalError):
            asyncio.run(executor.execute("t", {}, "worker"))
