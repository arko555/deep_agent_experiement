"""Tests for SubAgentEngine parallel dispatch."""

import asyncio
import pytest

from src.services.agent_orchestrator.subagent_engine import SubAgentEngine
from src.services.tools_integration.registry import ToolRegistry


class TestableSubAgentEngine(SubAgentEngine):
    """SubAgentEngine with overrideable _run_subagent_loop for testing."""

    def __init__(self, results=None, registry=None):
        super().__init__(registry=registry)
        self._results = results or {}
        self._fail = False

    def set_results(self, results):
        """Set {name: text} mapping to return from _run_subagent_loop."""
        self._results = results

    def set_fail(self, fail=True):
        self._fail = fail

    async def _run_subagent_loop(self, name, *args, **kwargs):
        if self._fail:
            raise RuntimeError(f"subagent {name} crashed")
        return self._results.get(name, f"result-for-{name}")


class TestSubAgentEngine:

    def test_invoke_parallel_returns_results_dict(self):
        engine = TestableSubAgentEngine(
            results={"research": "research result", "writer": "writer result"},
        )
        subagents = [
            {"name": "research", "description": "do research", "system_prompt": "sys", "tool_registry": []},
            {"name": "writer", "description": "write report", "system_prompt": "sys", "tool_registry": []},
        ]
        result = asyncio.run(engine.invoke_parallel(subagents, "find info"))
        assert result == {"research": "research result", "writer": "writer result"}

    def test_invoke_parallel_single_agent(self):
        engine = TestableSubAgentEngine(results={"research": "single result"})
        subagents = [
            {"name": "research", "description": "do research", "system_prompt": "sys", "tool_registry": []},
        ]
        result = asyncio.run(engine.invoke_parallel(subagents, "find info"))
        assert result == {"research": "single result"}

    def test_invoke_parallel_empty_list(self):
        engine = TestableSubAgentEngine()
        result = asyncio.run(engine.invoke_parallel([], "find info"))
        assert result == {}

    def test_invoke_parallel_handles_subagent_failure(self):
        engine = TestableSubAgentEngine()
        engine.set_fail()
        subagents = [
            {"name": "research", "description": "do research", "system_prompt": "sys", "tool_registry": []},
            {"name": "writer", "description": "write report", "system_prompt": "sys", "tool_registry": []},
        ]
        result = asyncio.run(engine.invoke_parallel(subagents, "find info"))
        assert set(result.keys()) == {"research", "writer"}
        assert "Error" in result["research"]
        assert "Error" in result["writer"]

    def test_default_registry(self):
        engine = TestableSubAgentEngine()
        assert engine.registry is not None

    def test_explicit_registry(self):
        reg = ToolRegistry()
        engine = TestableSubAgentEngine(registry=reg)
        assert engine.registry is reg


class TestSubAgentEngineLoop:

    def test_run_subagent_loop_returns_text(self):
        engine = TestableSubAgentEngine(results={"research": "subagent answer"})
        result = asyncio.run(engine._run_subagent_loop(
            name="research",
            system_prompt="You are research.",
            description="find papers",
            tool_defs=[],
            enhanced_query="find papers",
        ))
        assert "subagent answer" in result

    def test_run_subagent_loop_returns_answer(self):
        engine = TestableSubAgentEngine(results={"research": "answer"})
        result = asyncio.run(engine._run_subagent_loop(
            name="research",
            system_prompt="You are research.",
            description="find papers",
            tool_defs=[{"name": "search", "description": "search", "args_schema": {}}],
            enhanced_query="find papers",
        ))
        assert "answer" in result


class TestSubAgentEngineBatching:
    """Sub-agents past MAX_PARALLEL_TASKS run in later batches, not dropped."""

    def test_beyond_max_parallel_are_not_dropped(self):
        engine = TestableSubAgentEngine()
        subagents = [
            {"name": f"dept{i}", "description": "q", "system_prompt": "sys", "tool_registry": []}
            for i in range(7)
        ]
        result = asyncio.run(engine.invoke_parallel(subagents, "find info"))
        assert len(result) == 7
        assert set(result) == {f"dept{i}" for i in range(7)}

    def test_timeout_yields_error_result(self, monkeypatch):
        monkeypatch.setenv("SUBAGENT_TIMEOUT_SECONDS", "0.05")

        class SlowEngine(SubAgentEngine):
            async def _run_subagent_loop(self, **kwargs):
                await asyncio.sleep(10)
                return "never"

        engine = SlowEngine()
        result = asyncio.run(engine.invoke_parallel(
            [{"name": "slow", "description": "q", "system_prompt": "s", "tool_registry": []}],
            "q",
        ))
        assert "timed out" in result["slow"]

    def test_registry_supplies_tools_when_spec_has_none(self):
        reg = ToolRegistry()
        reg.register_builtin("search", lambda q: q, risk_level="low")
        engine = SubAgentEngine(registry=reg)
        defs = engine._tool_defs_for("research", reg)
        assert [d["name"] for d in defs] == ["search"]

    def test_default_registry_is_populated(self):
        """The no-arg registry must carry real tools.

        A bare ``ToolRegistry()`` has zero specs, so sub-agents dispatched by
        the graph — which constructs ``SubAgentEngine()`` with no argument —
        would silently receive no tools at all.
        """
        defs = SubAgentEngine()._tool_defs_for("research", SubAgentEngine().registry)
        assert defs, "default registry resolved no tools for a sub-agent"
        assert all("name" in d and "description" in d for d in defs)
