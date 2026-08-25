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

        def fake_execute(subagent_type, description, depth):
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

        def fake_execute(subagent_type, description, depth):
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
