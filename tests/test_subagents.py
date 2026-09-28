"""Tests for subagent tool-loop behavior and the parent's audit aggregation."""

from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

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


class TestToolsNodeSubagentAudit:
    """local_tools_node must fold subagent write ops into pending_writes."""

    def test_child_write_ops_merged_into_pending_writes(self):
        from src.services.agent_orchestrator import agent_factory

        def fake_execute(subagent_type, description, depth, tools_dict=None):
            return (
                "done",
                {"input": 5, "output": 2},
                [{"tool": "write_file", "tool_id": None,
                  "args": {"path": "workspace/child.md"}, "status": "executed"}],
            )

        state = {
            "messages": [AIMessage(
                content="",
                tool_calls=[{"name": "task",
                             "args": {"subagent_type": "writer", "description": "d"},
                             "id": "t1"}],
            )],
            "recursion_depth": 0,
            "pending_writes": [],
            "token_usage": {},
        }

        with patch.object(agent_factory, "_execute_task", side_effect=fake_execute):
            result = agent_factory.local_tools_node(state)

        # Child write surfaced in the parent's audit trail.
        pending = result["pending_writes"]
        assert len(pending) == 1
        assert pending[0]["tool"] == "write_file"
        assert pending[0]["args"]["path"] == "workspace/child.md"
        # Child token spend folded into the parent budget.
        assert result["token_usage"]["total"] == 7
        # The task's ToolMessage is present for the orchestrator to consume.
        assert result["messages"][0].tool_call_id == "t1"

    def test_failed_task_yields_error_not_crash(self):
        """One failing task must not drop its siblings' results."""
        from src.services.agent_orchestrator import agent_factory

        def fake_execute(subagent_type, description, depth, tools_dict=None):
            # Deterministic per-task outcome (tasks run in a thread pool,
            # so call order is not guaranteed).
            if "query 1" in description:
                raise RuntimeError("subagent blew up")
            return ("first done", {"input": 1, "output": 1}, [])

        task_calls = [
            {"name": "task",
             "args": {"subagent_type": "research", "description": f"query {i}"},
             "id": f"t{i}"}
            for i in range(2)
        ]
        state = {
            "messages": [AIMessage(content="", tool_calls=task_calls)],
            "recursion_depth": 0,
            "pending_writes": [],
            "token_usage": {},
        }

        with patch.object(agent_factory, "_execute_task", side_effect=fake_execute):
            result = agent_factory.local_tools_node(state)

        contents = [m.content for m in result["messages"]]
        assert "first done" in contents[0]
        assert "Error executing subagent task" in contents[1]
        # Sibling's usage still counted.
        assert result["token_usage"]["total"] == 2

    def test_depth_rejection_uses_configured_limit(self):
        """6.3: the depth limit comes from config, not a magic number."""
        from src.services.agent_orchestrator import agent_factory

        state = {
            "messages": [AIMessage(
                content="",
                tool_calls=[{"name": "task",
                             "args": {"subagent_type": "research", "description": "d"},
                             "id": "t1"}],
            )],
            "recursion_depth": 2,
            "pending_writes": [],
            "token_usage": {},
        }

        def _no_execute(*args, **kwargs):
            raise AssertionError("task must be rejected at the depth limit")

        with patch.object(agent_factory, "get_max_subagent_depth", return_value=2), \
                patch.object(agent_factory, "_execute_task", side_effect=_no_execute):
            result = agent_factory.local_tools_node(state)

        assert "Maximum subagent depth (2) reached" in result["messages"][0].content


class TestRegistryDrivesTaskTool:
    """6.1/6.4: the SUBAGENTS registry drives the task tool schema and the
    direct-invoke bypass is closed."""

    def test_task_tool_schema_lists_exactly_the_registry_types(self):
        from src.services.agent_orchestrator.subagents import SUBAGENTS
        from src.services.tools_integration.tools import get_all_tools

        task_tool = get_all_tools()["task"]
        schema = task_tool.args_schema.model_json_schema()
        assert set(schema["properties"]["subagent_type"]["enum"]) == set(SUBAGENTS)

    def test_task_tool_description_names_every_registry_type(self):
        from src.services.agent_orchestrator.subagents import SUBAGENTS
        from src.services.tools_integration.tools import get_all_tools

        task_tool = get_all_tools()["task"]
        for name in SUBAGENTS:
            assert name in task_tool.description

    def test_direct_task_invoke_is_refused(self):
        from src.services.tools_integration.tools import get_all_tools

        result = get_all_tools()["task"].invoke(
            {"subagent_type": "research", "description": "x"})
        assert "must be executed by the tools node" in result


class TestRegistryDrivesDispatch:
    """6.1: _execute_task dispatch is driven by the SUBAGENTS registry."""

    def test_tool_loop_dispatch_uses_registry_spec(self):
        import src.services.agent_orchestrator.subagents as subagents_mod
        import src.services.tools_integration.tools as tools_mod
        from src.services.agent_orchestrator.subagents import SUBAGENTS

        calls = {}

        def fake_run(spec, query, tools_dict=None, max_iterations=10):
            calls["spec"] = spec
            calls["query"] = query
            calls["tools"] = [t.name for t in
                              subagents_mod.select_department_tools(spec, query, tools_dict)]
            return "done", {"input": 1, "output": 1}, []

        # Production passes the parent's visible tools (all built-ins here).
        toolset = tools_mod.get_all_tools()
        with patch.object(subagents_mod, "run_department", side_effect=fake_run):
            text, usage, write_ops = tools_mod._execute_task(
                "research", "find x", 0, toolset)

        assert text == "done"
        assert usage == {"input": 1, "output": 1}
        assert write_ops == []
        assert calls["spec"] is SUBAGENTS["research"]
        assert calls["query"] == "find x"
        # The department's tools are resolved from its SKILL.md allowlist
        # against the live tool set, so every name must be a real tool.
        assert calls["tools"]
        for name in calls["tools"]:
            assert name in toolset
        assert set(calls["tools"]) <= set(SUBAGENTS["research"].tools)

    def test_unknown_type_lists_registry_types(self):
        import src.services.tools_integration.tools as tools_mod
        from src.services.agent_orchestrator.subagents import SUBAGENTS

        error, usage, ops = tools_mod._execute_task("nope", "d", 0)
        assert usage == {} and ops == []
        for name in SUBAGENTS:
            assert name in error


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

    def test_runtime_and_skill_specs_are_both_present(self, tmp_path):
        from src.services.agent_orchestrator.subagents import (
            _RUNTIME_SUBAGENTS, build_subagent_registry,
        )

        _write_skill(tmp_path, "legal")
        registry = build_subagent_registry()
        for name in _RUNTIME_SUBAGENTS:
            assert name in registry
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
    """Fanout and the `task` tool share one executor."""

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
