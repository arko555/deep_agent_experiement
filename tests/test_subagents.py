"""Tests for subagent tool-loop behavior and the parent's audit aggregation."""

import asyncio
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.services.agent_orchestrator.graph import _subagent_fanout_node
from src.services.agent_orchestrator.subagent_engine import SubagentRun
from src.services.agent_orchestrator.subagents import run_tool_loop


class _FakeModel:
    """Scripted model: returns queued responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        resp = self.responses.pop(0)
        resp.usage_metadata = {"input_tokens": 7, "output_tokens": 3}
        return resp


def _ai(text="", tool_calls=None):
    return AIMessage(content=text, tool_calls=tool_calls or [])


class _StubTool:
    """A named object standing in for a BaseTool in shortlisting tests."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"_StubTool({self.name!r})"


@tool
def write_file(path: str, content: str) -> str:
    """Fake write — records nothing, just confirms."""
    return f"wrote {path}"


@tool
def read_file(path: str) -> str:
    """Fake read."""
    return "contents"


class TestRunToolLoop:

    def test_returns_final_text_and_empty_write_ops(self):
        model = _FakeModel([_ai("done")])
        with patch("src.services.agent_orchestrator.agent_factory.get_model", return_value=model):
            text, usage, write_ops = run_tool_loop("sys", "do the thing", [])
        assert text == "done"
        assert usage == {"input": 7, "output": 3}
        assert write_ops == []

    def test_write_calls_are_tracked(self):
        model = _FakeModel([
            _ai(tool_calls=[{"name": "write_file",
                             "args": {"path": "workspace/a.md", "content": "x"},
                             "id": "1"}]),
            _ai("saved"),
        ])
        with patch("src.services.agent_orchestrator.agent_factory.get_model", return_value=model):
            text, _usage, write_ops = run_tool_loop("sys", "write a file", [write_file])
        assert text == "saved"
        assert len(write_ops) == 1
        assert write_ops[0]["tool"] == "write_file"
        assert write_ops[0]["args"]["path"] == "workspace/a.md"
        assert write_ops[0]["status"] == "executed"

    def test_read_calls_are_not_tracked_as_writes(self):
        model = _FakeModel([
            _ai(tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "1"}]),
            _ai("done"),
        ])
        with patch("src.services.agent_orchestrator.agent_factory.get_model", return_value=model):
            _, _, write_ops = run_tool_loop("sys", "read a file", [read_file])
        assert write_ops == []

    def test_usage_accumulates_across_turns(self):
        model = _FakeModel([
            _ai(tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "1"}]),
            _ai("done"),
        ])
        with patch("src.services.agent_orchestrator.agent_factory.get_model", return_value=model):
            _, usage, _ = run_tool_loop("sys", "read", [read_file])
        assert usage == {"input": 14, "output": 6}


class TestFanoutFoldsSubagentAudit:
    """The fanout node must fold sub-agent write ops and spend into the state.

    This used to be the top-level tools node's job. The tools node is gone —
    the orchestrator no longer binds tools — so the audit trail is now built
    from the ``SubagentRun`` each department returns.
    """

    def _state(self, **overrides):
        state = {
            "messages": [HumanMessage(content="go")],
            "subagent_results": {},
            "pending_writes": [],
            "audit_log": [],
            "token_usage": {"input": 0, "output": 0, "total": 0},
        }
        state.update(overrides)
        return state

    @staticmethod
    def _engine(runs):
        class StubEngine:
            def __init__(self, *a, **k):
                pass

            async def invoke_parallel(self, subs, query, *a, **k):
                return {s["name"]: runs[s["name"]] for s in subs}

        return StubEngine

    def _run(self, state, runs, monkeypatch):
        monkeypatch.setattr(
            "src.services.agent_orchestrator.graph.SubAgentEngine", self._engine(runs)
        )
        return _subagent_fanout_node(state)

    def test_child_write_ops_merged_into_pending_writes(self, monkeypatch):
        run = SubagentRun(
            text="done",
            usage={"input": 5, "output": 2},
            writes=[{"tool": "write_file", "tool_id": None,
                     "args": {"path": "workspace/child.md"}, "status": "executed"}],
        )
        result = self._run(
            self._state(department_targets=["writer"]), {"writer": run}, monkeypatch
        )

        # Child write surfaced in the parent's audit trail.
        pending = result["pending_writes"]
        assert len(pending) == 1
        assert pending[0]["tool"] == "write_file"
        assert pending[0]["args"]["path"] == "workspace/child.md"
        # Child token spend folded into the parent budget.
        assert result["token_usage"]["total"] == 7
        assert result["subagent_results"] == {"writer": "done"}
        # And there is an audit entry, so the Streamlit trail isn't empty.
        assert result["audit_log"]

    def test_child_usage_is_summed_across_departments(self, monkeypatch):
        runs = {
            "research": SubagentRun(text="r", usage={"input": 3, "output": 1}),
            "writer": SubagentRun(text="w", usage={"input": 4, "output": 2}),
        }
        result = self._run(
            self._state(department_targets=["research", "writer"]), runs, monkeypatch
        )
        assert result["token_usage"] == {"input": 7, "output": 3, "total": 10}
        assert result["subagent_results"] == {"research": "r", "writer": "w"}

    def test_departments_that_wrote_nothing_leave_state_untouched(self, monkeypatch):
        """A read-only department must not create empty audit entries."""
        run = SubagentRun(text="read only", usage={"input": 0, "output": 0})
        result = self._run(
            self._state(department_targets=["research"]), {"research": run}, monkeypatch
        )
        assert "pending_writes" not in result
        assert "audit_log" not in result
        assert "token_usage" not in result


class TestNoDelegationPath:
    """The `task` tool and sub-agent-to-sub-agent delegation are gone.

    The orchestrator is a pure router, so it never binds `task`; and no
    SKILL.md lists `task` in `allowed-tools`, so no sub-agent is ever offered
    it either. There is exactly one route from the router to a department:
    the fanout node.
    """

    def test_the_task_tool_module_is_gone(self):
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("src.services.agent_orchestrator.task_tool")

    def test_no_skill_declares_the_task_tool(self):
        from src.services.agent_orchestrator.subagents import SUBAGENTS

        for name, spec in SUBAGENTS.items():
            assert "task" not in spec.tools, f"{name} still allows the task tool"

    def test_no_department_tool_is_named_task(self):
        """Not even as a fallback: the tool layer has no such tool."""
        from src.services.tools_integration.tools import get_all_tools

        assert "task" not in get_all_tools()


class TestRegistryDrivesDispatch:
    """6.1: department dispatch is driven by the SUBAGENTS registry."""

    def test_tool_loop_dispatch_uses_registry_spec(self):
        import src.services.agent_orchestrator.subagents as subagents_mod
        from src.services.agent_orchestrator.subagents import SUBAGENTS
        from src.services.tools_integration.tools import get_all_tools

        calls = {}

        def fake_run(spec, query, tools_dict=None, max_iterations=10):
            calls["spec"] = spec
            calls["query"] = query
            calls["tools"] = [t.name for t in
                              subagents_mod.select_department_tools(spec, query, tools_dict)]
            return "done", {"input": 1, "output": 1}, []

        # Production resolves against the live tool layer.
        toolset = get_all_tools()
        with patch.object(subagents_mod, "run_tool_loop",
                          side_effect=lambda prompt, desc, tools, max_iterations=10:
                              fake_run(SUBAGENTS["research"], desc, toolset)):
            from src.services.agent_orchestrator.subagents import run_department

            text, usage, write_ops = run_department(SUBAGENTS["research"], "find x")

        assert text == "done"
        assert usage == {"input": 1, "output": 1}
        assert write_ops == []
        assert calls["query"] == "find x"
        # The department's tools are resolved from its SKILL.md allowlist
        # against the live tool set, so every name must be a real tool.
        assert calls["tools"]
        for name in calls["tools"]:
            assert name in toolset
        assert set(calls["tools"]) <= set(SUBAGENTS["research"].tools)

    def test_unknown_department_lists_registry_types(self):
        from src.services.agent_orchestrator.subagent_engine import SubAgentEngine
        from src.services.agent_orchestrator.subagents import SUBAGENTS

        result = asyncio.run(SubAgentEngine()._run_subagent_loop(
            name="nope", system_prompt="", description="d", enhanced_query="d",
        ))
        for name in SUBAGENTS:
            assert name in result.text


class TestRegistryDrivesParallelGrouping:
    """6.1: the parallel-vs-sequential split follows the registry flag."""

    def test_is_parallelizable_matches_registry(self):
        from src.services.agent_orchestrator.subagents import SUBAGENTS, is_parallelizable

        for name, spec in SUBAGENTS.items():
            assert is_parallelizable(name) is spec.parallelizable
            for alias in spec.aliases:
                assert is_parallelizable(alias) is spec.parallelizable
        assert is_parallelizable("does-not-exist") is False


# ---------------------------------------------------------------------------
# skills/ is the source of truth: departments load from SKILL.md, not Python
# ---------------------------------------------------------------------------

_SKILL = """---
name: {name}
description: >
  {desc_line_one}
  {desc_line_two}
allowed-tools: read_file write_file
parallelizable: {parallel}
---

# {title}

Do the {name} work.
"""


def _write_skill(root, name, desc="Handles widgets.", parallel=True):
    dept = root / name
    dept.mkdir(parents=True, exist_ok=True)
    (dept / "SKILL.md").write_text(
        _SKILL.format(
            name=name,
            desc_line_one=desc,
            desc_line_two="Use it when the user asks about widgets.",
            parallel=str(parallel).lower(),
            title=f"{name.title()} Skill",
        )
    )
    return dept


class TestSkillsAreTheRegistry:
    """A SKILL.md on disk becomes a routable department with no Python edit."""

    def test_new_directory_becomes_a_department(self, tmp_path):
        from src.services.agent_orchestrator.subagents import _skill_specs

        _write_skill(tmp_path, "legal")
        specs = _skill_specs(str(tmp_path))

        assert "legal" in specs
        spec = specs["legal"]
        assert spec.kind == "tool_loop"
        assert spec.skill == "legal"
        assert spec.parallelizable is True
        # Space-separated allowed-tools, so two real names rather than one
        # string with spaces in it.
        assert spec.tools == ("read_file", "write_file")
        # The folded-scalar description reaches the model intact.
        assert spec.description.startswith("Handles widgets.")
        assert "Use it when the user asks about widgets." in spec.description

    def test_parallelizable_false_is_honoured(self, tmp_path):
        from src.services.agent_orchestrator.subagents import _skill_specs

        _write_skill(tmp_path, "serial", parallel=False)
        assert _skill_specs(str(tmp_path))["serial"].parallelizable is False

    def test_no_spec_can_reenter_the_graph(self, tmp_path):
        """Nothing in the registry runs the whole compiled graph.

        A ``kind="graph"`` spec — the old ``general-purpose`` — would make the
        orchestrator its own sub-agent, so a delegated description got routed
        again from the top instead of the hierarchy staying one-way. The
        catch-all is now ``skills/general``, a department like any other.
        """
        from src.services.agent_orchestrator.subagents import build_subagent_registry

        _write_skill(tmp_path, "legal")
        registry = build_subagent_registry()
        assert registry, "the registry must not be empty"
        assert {s.kind for s in registry.values()} <= {"tool_loop", "a2a"}
        # Every routable department is skill-backed, so the router can name it.
        assert any(s.kind == "tool_loop" and s.skill for s in registry.values())

    def test_duplicate_names_keep_the_first(self, tmp_path):
        from src.services.agent_orchestrator.subagents import _skill_specs

        # Two directories both declaring the name "dup"; the first in sorted
        # order wins, and the shadowed one must not silently replace it.
        _write_skill(tmp_path, "a-first", desc="First definition.")
        (tmp_path / "a-second").mkdir()
        (tmp_path / "a-second" / "SKILL.md").write_text(
            _SKILL.format(name="a-first", desc_line_one="Shadowing definition.",
                          desc_line_two="", parallel="true", title="Shadow")
        )
        specs = _skill_specs(str(tmp_path))
        assert "a-first" in specs
        assert specs["a-first"].description.startswith("First definition.")

    def test_shipped_departments_are_discoverable(self):
        """The registry the app actually runs with, not a fixture."""
        from src.services.agent_orchestrator.subagents import list_departments

        names = {d["name"] for d in list_departments()}
        assert {"research", "writer"} <= names
        for dept in list_departments():
            assert dept["description"], f"{dept['name']} has no description"


class TestDepartmentToolShortlisting:
    """allowed-tools → visible cap → relevance sort → real bindable tools."""

    def _spec(self, tools):
        from src.services.agent_orchestrator.subagents import SubagentSpec

        return SubagentSpec(description="d", kind="tool_loop", skill="x", tools=tuple(tools))

    @staticmethod
    def _toolset(*names):
        """Minimal stand-ins for BaseTool: the code reads ``.name`` off them."""
        return {n: _StubTool(n) for n in names}

    def test_names_resolve_to_real_callables(self):
        from src.services.agent_orchestrator.subagents import select_department_tools

        live = self._toolset("read_file", "write_file")
        got = select_department_tools(self._spec(["read_file"]), "q", live)
        assert got == [live["read_file"]]

    def test_unknown_names_are_dropped_not_propagated(self):
        from src.services.agent_orchestrator.subagents import select_department_tools

        live = self._toolset("read_file")
        # A stale name in a SKILL.md must not become a tool the model can call
        # but never execute.
        got = select_department_tools(self._spec(["read_file", "ghost_tool"]), "q", live)
        assert got == [live["read_file"]]

    def test_empty_allowlist_yields_no_tools(self):
        from src.services.agent_orchestrator.subagents import select_department_tools

        assert select_department_tools(self._spec([]), "q", self._toolset("read_file")) == []

    def test_short_allowlist_skips_the_sort_call(self):
        from src.services.agent_orchestrator.subagents import select_department_tools

        with patch(
            "src.services.tools_integration.relevance.sort_tools"
        ) as sorter:
            got = select_department_tools(
                self._spec(["a", "b"]), "q", self._toolset("a", "b", "c")
            )
        assert len(got) == 2
        sorter.assert_not_called()

    def test_long_allowlist_is_narrowed_to_the_shortlist_cap(self):
        from src.services.agent_orchestrator.subagents import (
            MAX_SHORTLISTED_TOOLS, select_department_tools,
        )

        names = [f"t{i}" for i in range(12)]
        live = self._toolset(*names)

        def fake_sort(_model, _query, tool_defs, limit):
            assert len(tool_defs) == len(names)
            assert limit == MAX_SHORTLISTED_TOOLS
            return [{"name": n} for n in names[:3]]

        with patch("src.services.tools_integration.relevance.sort_tools",
                   side_effect=fake_sort):
            got = select_department_tools(self._spec(names), "q", live)

        assert got == [live[n] for n in names[:3]]

    def test_visible_cap_applies_before_sorting(self):
        from src.services.agent_orchestrator.subagents import (
            MAX_SHORTLISTED_TOOLS, MAX_VISIBLE_TOOLS, select_department_tools,
        )

        names = [f"t{i}" for i in range(40)]
        live = self._toolset(*names)
        seen = {}

        def fake_sort(_model, _query, tool_defs, limit):
            seen["count"] = len(tool_defs)
            return []

        with patch("src.services.tools_integration.relevance.sort_tools",
                   side_effect=fake_sort):
            got = select_department_tools(self._spec(names), "q", live)

        assert seen["count"] == MAX_VISIBLE_TOOLS
        # A sorter that returns nothing must still hand back a usable set.
        assert len(got) == MAX_SHORTLISTED_TOOLS

    def test_sort_failure_falls_back_to_the_capped_allowlist(self):
        from src.services.agent_orchestrator.subagents import select_department_tools

        names = [f"t{i}" for i in range(9)]
        live = self._toolset(*names)
        with patch("src.services.tools_integration.relevance.sort_tools",
                   side_effect=RuntimeError("model down")):
            got = select_department_tools(self._spec(names), "q", live)
        assert len(got) == 9

    def test_sorter_choices_are_matched_against_the_allowlist(self):
        """A hallucinated name from the sorter must not be bound."""
        from src.services.agent_orchestrator.subagents import select_department_tools

        names = [f"t{i}" for i in range(6)]
        live = self._toolset(*names)
        with patch("src.services.tools_integration.relevance.sort_tools",
                   side_effect=lambda *a: [{"name": "t0"}, {"name": "made_up"}]):
            got = select_department_tools(self._spec(names), "q", live)
        assert got == [live["t0"]]


class TestRunDepartmentUnifiesBothEntryPoints:
    """Fanout goes through the same executor the registry names."""

    def test_prompt_comes_from_the_skill_file(self):
        import src.services.agent_orchestrator.subagents as subagents_mod
        from src.services.agent_orchestrator.subagents import (
            COMPLETION_CONTRACT, SubagentSpec, run_department,
        )

        captured = {}

        def fake_loop(prompt, description, tools, max_iterations=10):
            captured["prompt"] = prompt
            captured["query"] = description
            captured["tools"] = tools
            return "answer", {"input": 0, "output": 0}, []

        with patch.object(subagents_mod, "run_tool_loop", side_effect=fake_loop):
            text, _usage, _writes = run_department(
                SubagentSpec(description="d", kind="tool_loop", skill="writer"),
                "write a post",
            )

        assert text == "answer"
        assert captured["query"] == "write a post"
        # The stable marker, then the SKILL.md body — not a synthesized
        # "You are the {dept} specialist" line.
        assert captured["prompt"].startswith(COMPLETION_CONTRACT)
        assert "Creative Writer Skill" in captured["prompt"]
        assert "specialist" not in captured["prompt"].lower()


class TestBudgetExhaustionIsReported:
    """A sub-agent that runs out of turns must not return an empty string."""

    def test_tool_call_only_final_turn_reports_truncation(self, monkeypatch):
        # Every scripted turn is a tool call, so the loop ends on a turn whose
        # content is "" — the case that used to reach the parent as "".
        model = _FakeModel([
            _ai(tool_calls=[{"name": "internet_search", "args": {}, "id": f"c{i}"}])
            for i in range(4)
        ])
        monkeypatch.setattr(
            "src.services.agent_orchestrator.agent_factory.get_model",
            lambda: model,
        )

        text, _usage, _writes = run_tool_loop("sys", "go", [], max_iterations=4)

        assert text.strip()
        assert "4 model turns" in text
        assert "internet_search" in text

    def test_files_written_before_truncation_are_named(self, monkeypatch):
        @tool
        def write_file(path: str, content: str) -> str:
            """Write a file."""
            return f"wrote {path}"

        model = _FakeModel([
            _ai(tool_calls=[{"name": "write_file",
                             "args": {"path": "workspace/notes.md", "content": "x"},
                             "id": "c0"}]),
            _ai(tool_calls=[{"name": "write_file",
                             "args": {"path": "workspace/notes.md", "content": "y"},
                             "id": "c1"}]),
        ])
        monkeypatch.setattr(
            "src.services.agent_orchestrator.agent_factory.get_model",
            lambda: model,
        )

        text, _usage, writes = run_tool_loop("sys", "go", [write_file], max_iterations=2)

        assert "workspace/notes.md" in text
        assert len(writes) == 2
