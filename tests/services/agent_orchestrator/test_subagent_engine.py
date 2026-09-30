"""Tests for SubAgentEngine parallel dispatch."""

import asyncio

from src.services.agent_orchestrator.subagent_engine import SubAgentEngine, SubagentRun


class TestableSubAgentEngine(SubAgentEngine):
    """SubAgentEngine with overrideable _run_subagent_loop for testing."""

    def __init__(self, results=None):
        super().__init__()
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
        return SubagentRun(text=self._results.get(name, f"result-for-{name}"))


def _texts(runs):
    """{name: SubagentRun} → {name: text}, for comparing final answers."""
    return {name: run.text for name, run in runs.items()}


class TestSubAgentEngine:

    def test_invoke_parallel_returns_results_dict(self):
        engine = TestableSubAgentEngine(
            results={"research": "research result", "writer": "writer result"},
        )
        subagents = [
            {"name": "research", "description": "do research", "system_prompt": "sys"},
            {"name": "writer", "description": "write report", "system_prompt": "sys"},
        ]
        result = asyncio.run(engine.invoke_parallel(subagents, "find info"))
        assert _texts(result) == {"research": "research result", "writer": "writer result"}

    def test_invoke_parallel_single_agent(self):
        engine = TestableSubAgentEngine(results={"research": "single result"})
        subagents = [
            {"name": "research", "description": "do research", "system_prompt": "sys"},
        ]
        result = asyncio.run(engine.invoke_parallel(subagents, "find info"))
        assert _texts(result) == {"research": "single result"}

    def test_invoke_parallel_empty_list(self):
        engine = TestableSubAgentEngine()
        result = asyncio.run(engine.invoke_parallel([], "find info"))
        assert result == {}

    def test_invoke_parallel_handles_subagent_failure(self):
        engine = TestableSubAgentEngine()
        engine.set_fail()
        subagents = [
            {"name": "research", "description": "do research", "system_prompt": "sys"},
            {"name": "writer", "description": "write report", "system_prompt": "sys"},
        ]
        result = asyncio.run(engine.invoke_parallel(subagents, "find info"))
        assert set(result.keys()) == {"research", "writer"}
        assert "Error" in result["research"].text
        assert "Error" in result["writer"].text


class TestSubAgentEngineLoop:

    def test_run_subagent_loop_returns_text(self):
        engine = TestableSubAgentEngine(results={"research": "subagent answer"})
        result = asyncio.run(engine._run_subagent_loop(
            name="research",
            system_prompt="You are research.",
            description="find papers",
            enhanced_query="find papers",
        ))
        assert "subagent answer" in result.text

    def test_run_subagent_loop_returns_answer(self):
        engine = TestableSubAgentEngine(results={"research": "answer"})
        result = asyncio.run(engine._run_subagent_loop(
            name="research",
            system_prompt="You are research.",
            description="find papers",
            enhanced_query="find papers",
        ))
        assert "answer" in result.text


class TestSubAgentEngineBatching:
    """Sub-agents past MAX_PARALLEL_TASKS run in later batches, not dropped."""

    def test_beyond_max_parallel_are_not_dropped(self):
        engine = TestableSubAgentEngine()
        subagents = [
            {"name": f"dept{i}", "description": "q", "system_prompt": "sys"}
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
                return SubagentRun(text="never")

        engine = SlowEngine()
        result = asyncio.run(engine.invoke_parallel(
            [{"name": "slow", "description": "q", "system_prompt": "s"}],
            "q",
        ))
        assert "timed out" in result["slow"].text


class TestEngineDoesNotSelectTools:
    """Tool choice belongs to the SKILL.md allowlist, not to the engine.

    The engine used to hold a ``ToolRegistry`` and call
    ``get_visible_tools(name)``. That result was never bound to anything, so it
    was a second filter that could only disagree with the real one.
    """

    def test_engine_has_no_registry(self):
        engine = SubAgentEngine()
        assert not hasattr(engine, "registry")
        assert not hasattr(engine, "_registry")

    def test_engine_passes_no_tool_defs_to_the_department(self):
        """The loop forwards the query and nothing tool-shaped.

        A ``tool_defs`` argument here would reintroduce a second source of
        tools alongside ``select_department_tools``.
        """
        seen = {}

        class SpyEngine(SubAgentEngine):
            async def _run_subagent_loop(self, name, *args, **kwargs):
                seen[name] = kwargs
                return SubagentRun(text="ok")

        asyncio.run(SpyEngine().invoke_parallel(
            [{"name": "research", "description": "q"}], "find info"
        ))
        assert set(seen["research"]) == {"system_prompt", "description", "enhanced_query"}

    def test_department_tools_come_from_the_allowlist(self):
        """What the engine delegates to resolves real tools from SKILL.md."""
        from src.services.agent_orchestrator.subagents import (
            SUBAGENTS,
            list_departments,
            select_department_tools,
        )
        from src.services.tools_integration.tools import get_all_tools

        toolset = get_all_tools()
        for dept in list_departments():
            spec = SUBAGENTS[dept["name"]]
            resolved = select_department_tools(spec, "test query", tools_dict=toolset)
            assert resolved, f"{dept['name']} resolved no tools"
            for t in resolved:
                assert t.name in toolset
