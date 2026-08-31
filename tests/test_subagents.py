"""Tests for subagent tool-loop behavior and the parent's audit aggregation."""

from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from src.core.subagents import run_tool_loop


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
        with patch("src.core.agent_factory.get_model", return_value=model):
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
        with patch("src.core.agent_factory.get_model", return_value=model):
            text, usage, write_ops = run_tool_loop("sys", "write a file", [write_file])
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
        with patch("src.core.agent_factory.get_model", return_value=model):
            _, _, write_ops = run_tool_loop("sys", "read a file", [read_file])
        assert write_ops == []

    def test_usage_accumulates_across_turns(self):
        model = _FakeModel([
            _ai(tool_calls=[{"name": "read_file", "args": {"path": "x"}, "id": "1"}]),
            _ai("done"),
        ])
        with patch("src.core.agent_factory.get_model", return_value=model):
            _, usage, _ = run_tool_loop("sys", "read", [read_file])
        assert usage == {"input": 14, "output": 6}


class TestToolsNodeSubagentAudit:
    """local_tools_node must fold subagent write ops into pending_writes."""

    def test_child_write_ops_merged_into_pending_writes(self):
        from src.core import agent_factory

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
        from src.core import agent_factory

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
        from src.core import agent_factory

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
        from src.core.subagents import SUBAGENTS
        from src.core.tools import task as task_tool

        schema = task_tool.args_schema.model_json_schema()
        assert set(schema["properties"]["subagent_type"]["enum"]) == set(SUBAGENTS)

    def test_task_tool_description_names_every_registry_type(self):
        from src.core.subagents import SUBAGENTS
        from src.core.tools import task as task_tool

        for name in SUBAGENTS:
            assert name in task_tool.description

    def test_direct_task_invoke_is_refused(self):
        from src.core.tools import task as task_tool

        result = task_tool.invoke({"subagent_type": "research", "description": "x"})
        assert "must be executed by the tools node" in result


class TestRegistryDrivesDispatch:
    """6.1: _execute_task dispatch is driven by the SUBAGENTS registry."""

    def test_tool_loop_dispatch_uses_registry_spec(self):
        import src.core.tools as tools_mod
        from src.core.subagents import SUBAGENTS

        calls = {}

        def fake_run(system_prompt, description, loop_tools, max_iterations=10):
            calls["prompt"] = system_prompt
            calls["tools"] = [t.name for t in loop_tools]
            return "done", {"input": 1, "output": 1}, []

        toolset = {t.name: t for t in (
            tools_mod.internet_search, tools_mod.read_file, tools_mod.write_file
        )}
        with patch.object(tools_mod, "run_tool_loop", side_effect=fake_run):
            text, usage, write_ops = tools_mod._execute_task(
                "researcher", "find x", 0, toolset)

        assert text == "done"
        assert usage == {"input": 1, "output": 1}
        assert write_ops == []
        # Alias "researcher" resolves to the research spec; its restricted
        # toolset names come straight from the registry.
        assert calls["tools"] == list(SUBAGENTS["research"].tools)
        # Prompt = shared completion contract + research SKILL.md body.
        assert calls["prompt"].startswith("You are a specialized subagent")
        assert "Web Research Skill" in calls["prompt"]

    def test_unknown_type_lists_registry_types(self):
        import src.core.tools as tools_mod
        from src.core.subagents import SUBAGENTS

        error, usage, ops = tools_mod._execute_task("nope", "d", 0)
        assert usage == {} and ops == []
        for name in SUBAGENTS:
            assert name in error


class TestRegistryDrivesParallelGrouping:
    """6.1: the parallel-vs-sequential split follows the registry flag."""

    def test_is_parallelizable_matches_registry(self):
        from src.core.subagents import SUBAGENTS, is_parallelizable

        for name, spec in SUBAGENTS.items():
            assert is_parallelizable(name) is spec.parallelizable
            for alias in spec.aliases:
                assert is_parallelizable(alias) is spec.parallelizable
        assert is_parallelizable("does-not-exist") is False
