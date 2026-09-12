"""Graph-level delegation tests (Phase 9).

Drives the compiled graph end-to-end with a scripted fake model so the
delegation contract itself is verified, not just the unit pieces:
task call → tools node → subagent loop → ToolMessage result, token-usage
aggregation into the parent, pending_writes merge, depth rejection, and
parallel-batch ordering/deadline behavior.
"""

import threading
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.core import agent_factory
from tests.fake_models import ScriptedChatModel, ai


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _initial_state(user_message: str) -> dict:
    return {
        "messages": [HumanMessage(content=user_message)],
        "current_plan": [],
        "workspace_files": [],
        "next_message": None,
        "review_verdict": None,
        "recursion_depth": 0,
        "pending_writes": [],
        "audit_log": [],
        "token_usage": {},
        "iteration_count": 0,
        "max_iterations": 25,
    }


def _run_graph(state: dict, thread_id: str) -> dict:
    agent = agent_factory.get_deep_agent()
    return agent.invoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
    )


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Run the graph from a temp cwd so workspace writes stay sandboxed."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def patched_model(monkeypatch):
    """Install a scripted fake as the model for orchestrator/reviewers/subagents."""
    installed = {}

    def _install(fake: ScriptedChatModel):
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)
        installed["fake"] = fake

    return _install


# ---------------------------------------------------------------------------
# 9.1: Delegation contract through the compiled graph
# ---------------------------------------------------------------------------

class TestDelegationContract:

    def test_task_result_tokens_and_writes_flow_back(self, sandbox, patched_model):
        fake = ScriptedChatModel(
            scripts={
                "orchestrator": [
                    ai("Delegating research.",
                       tool_calls=[{"name": "task",
                                    "args": {"subagent_type": "research",
                                             "description": "Research X"},
                                    "id": "call_1"}],
                       usage=(10, 2)),
                    ai("Report: research complete.", usage=(20, 4)),
                ],
                "critic": [ai("APPROVED — the answer is solid.")],
                "subagent": [
                    ai("", tool_calls=[{"name": "write_file",
                                        "args": {"path": "workspace/research_notes.md",
                                                 "content": "findings"},
                                        "id": "w1"}],
                       usage=(100, 10)),
                    ai("Saved workspace/research_notes.md. Research done.", usage=(50, 5)),
                ],
            },
            default=ai("done."),
        )
        patched_model(fake)

        result = _run_graph(_initial_state("Research X and report back."), "delegation-1")

        # ToolMessage carrying the subagent's final summary is returned.
        task_results = [m for m in result["messages"]
                        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "task"]
        assert len(task_results) == 1
        assert "Saved workspace/research_notes.md" in task_results[0].content

        # Child token usage is aggregated into the parent's budget:
        # orchestrator turns (10+20 in, 2+4 out) + subagent loop (100+50 in, 10+5 out).
        assert result["token_usage"]["input"] == 180
        assert result["token_usage"]["output"] == 21

        # Child file writes merge into the parent's audit trail.
        assert len(result["pending_writes"]) == 1
        entry = result["pending_writes"][0]
        assert entry["tool"] == "write_file"
        assert entry["args"]["path"] == "workspace/research_notes.md"

        # The file was actually written under the sandbox workspace.
        assert (sandbox / "workspace" / "research_notes.md").read_text() == "findings"

        # And the final answer still reaches the responder.
        assert isinstance(result["messages"][-1], AIMessage)
        assert result["messages"][-1].content == "Report: research complete."

    def test_depth_rejection_at_limit(self, sandbox, patched_model, monkeypatch):
        # Limit 0 ⇒ even a top-level task is refused; no subagent may run.
        monkeypatch.setattr(agent_factory, "get_max_subagent_depth", lambda: 0)
        fake = ScriptedChatModel(
            scripts={
                "orchestrator": [
                    ai("Trying to delegate.",
                       tool_calls=[{"name": "task",
                                    "args": {"subagent_type": "research",
                                             "description": "Research Y"},
                                    "id": "call_2"}],
                       usage=(1, 1)),
                    ai("Handled directly instead.", usage=(1, 1)),
                ],
                "critic": [ai("APPROVED.")],
            },
            default=ai("done."),
        )
        patched_model(fake)

        result = _run_graph(_initial_state("Research Y."), "delegation-depth")

        task_results = [m for m in result["messages"]
                        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "task"]
        assert len(task_results) == 1
        assert "Maximum subagent depth" in task_results[0].content
        # No subagent ran: nothing written, no audit entries.
        assert result["pending_writes"] == []
        assert not fake.counts.get("subagent")


# ---------------------------------------------------------------------------
# 9.2: Parallel batch behavior
# ---------------------------------------------------------------------------

class _ParallelFake:
    """Thread-safe scripted model for parallel-batch tests.

    Each subagent loop: first call writes a uniquely numbered note, second
    call finishes — regardless of how the two loops interleave.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._n = 0
        self._orch_calls = 0

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        role = ScriptedChatModel._detect_role(messages)
        if role == "orchestrator":
            with self._lock:
                self._orch_calls += 1
                first = self._orch_calls == 1
            if first:
                return ai("Delegating in parallel.", tool_calls=[
                    {"name": "task",
                     "args": {"subagent_type": "research", "description": "Research A"},
                     "id": "p1"},
                    {"name": "task",
                     "args": {"subagent_type": "research", "description": "Research B"},
                     "id": "p2"},
                ], usage=(5, 1))
            return ai("Parallel report.", usage=(5, 1))
        if role == "critic":
            return ai("APPROVED.")
        with self._lock:
            self._n += 1
            n = self._n
        if n % 2 == 1:  # odd: this loop's write turn; even: its completion
            return AIMessage(
                content="",
                tool_calls=[{"name": "write_file",
                             "args": {"path": f"workspace/notes_{n}.md",
                                      "content": f"note {n}"},
                             "id": f"w{n}"}],
            )
        return AIMessage(content=f"Saved workspace/notes_{n - 1}.md.")


class TestParallelBatch:

    def test_deterministic_ordering_and_distinct_files(self, sandbox, monkeypatch):
        fake = _ParallelFake()
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)
        result = _run_graph(_initial_state("Research A and B in parallel."), "parallel-1")

        task_results = [m for m in result["messages"]
                        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "task"]
        # ToolMessages come back in submission order (p1 then p2), even though
        # the subagents ran concurrently.
        assert [m.tool_call_id for m in task_results] == ["p1", "p2"]

        files = sorted((sandbox / "workspace").glob("notes_*.md"))
        assert len(files) == 2  # distinct output files, no clobbering
        for tm in task_results:
            match = [f for f in files if f"Saved workspace/{f.name}" in tm.content]
            assert len(match) == 1  # each result names exactly one of the files

    def test_shared_batch_deadline_bounds_the_batch(self, sandbox, monkeypatch):
        # A hung subagent costs ONE timeout for the whole batch, not N x.
        monkeypatch.setattr(agent_factory, "get_subagent_timeout_seconds", lambda: 0.2)

        class _HungFake(_ParallelFake):
            def invoke(self, messages, **kwargs):
                role = ScriptedChatModel._detect_role(messages)
                if role == "subagent":
                    time.sleep(0.5)  # longer than the batch deadline
                return super().invoke(messages, **kwargs)

        fake = _HungFake()
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)

        start = time.monotonic()
        result = _run_graph(_initial_state("Research A and B in parallel."), "parallel-hung")
        elapsed = time.monotonic() - start

        task_results = [m for m in result["messages"]
                        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "task"]
        assert len(task_results) == 2
        assert all(str(m.content).startswith("Error executing subagent task")
                   for m in task_results)
        # Shared deadline: total cost ~ one timeout, not two.
        assert elapsed < 0.2 + 0.15
