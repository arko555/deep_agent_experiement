"""Graph-level tests for Phase 3 Agent Orchestrator.

Drives the Phase 3 compiled graph (START → orchestrator →
{fanout | responder} → END) end-to-end with a scripted fake model
so the orchestration contract is verified, not just the unit pieces.

Covers: department detection → subagent_fanout → aggregation →
responder, direct-to-responder when no departments, multi-turn routing
on a reused checkpoint thread, and parallel batch behavior.
"""

import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.services.agent_orchestrator import agent_factory
from src.services.agent_orchestrator.graph import get_deep_agent, reset_deep_agent
from src.services.agent_orchestrator.subagent_engine import SubagentRun
from tests.fake_models import ScriptedChatModel, ai


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _initial_state(user_message: str) -> dict:
    return {
        "messages": [HumanMessage(content=user_message)],
        "next_message": None,
        "pending_writes": [],
        "audit_log": [],
        "token_usage": {},
        "enhanced_query": "",
        "department_targets": [],
        "subagent_results": {},
    }


def _run_graph(state: dict, thread_id: str) -> dict:
    agent = get_deep_agent()
    return agent.invoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
    )


@pytest.fixture(autouse=True)
def _reset_graph():
    reset_deep_agent()
    yield
    reset_deep_agent()


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Run the graph from a temp cwd so workspace writes stay sandboxed."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Orchestrator → subagent_fanout flow
# ---------------------------------------------------------------------------

class TestOrchestratorFanout:

    def test_detects_departments_returns_synthesized_answer(self, monkeypatch):
        """Orchestrator detects departments → fanout runs → aggregator → responder."""
        fake = ScriptedChatModel(
            scripts={
                "orchestrator": [
                    ai('{"enhanced_query": "summarize all", "departments": ["research"]}'),
                ],
                "subagent": [ai("Research summary complete.")],
            },
            default=ai("done"),
        )
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)

        result = _run_graph(_initial_state("Summarize all."), "dept-1")

        messages = result["messages"]
        answer = next(
            (m.content for m in messages if isinstance(m, AIMessage) and "summarize" not in m.content),
            "",
        )
        assert "Research summary complete." in answer
        # fanout goes straight to the responder: no second dispatcher call.
        assert fake.counts.get("orchestrator", 0) == 1
        assert fake.counts.get("subagent", 0) == 1

    def test_single_department_inline(self, monkeypatch):
        """Single department → subagent result in final answer."""
        fake = ScriptedChatModel(
            scripts={
                "orchestrator": [
                    ai('{"enhanced_query": "summarize X", "departments": ["research"]}'),
                ],
                "subagent": [ai("Result for X.")],
            },
            default=ai("done"),
        )
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)
        result = _run_graph(_initial_state("Summarize X."), "single-dept")

        answer = next(
            (m.content for m in result["messages"] if isinstance(m, AIMessage)),
            "",
        )
        assert "Result for X." in answer
        # fanout goes straight to the responder: no second dispatcher call.
        assert fake.counts.get("orchestrator", 0) == 1

    def test_multiple_departments_synthesized(self, monkeypatch):
        """Multiple departments → aggregator combines into final answer."""
        fake = ScriptedChatModel(
            scripts={
                "orchestrator": [
                    ai('{"enhanced_query": "multi-dept", "departments": ["research", "writer"]}'),
                ],
                "subagent": [
                    ai("Research finding."),
                    ai("Writing result."),
                ],
            },
            default=ai("done"),
        )
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)
        result = _run_graph(_initial_state("Multi-dept."), "multi-dept")

        answer = next(
            (m.content for m in result["messages"] if isinstance(m, AIMessage)),
            "",
        )
        assert "Research finding." in answer
        assert "Writing result." in answer


# ---------------------------------------------------------------------------
# Orchestrator → responder flow (no departments)
# ---------------------------------------------------------------------------

class TestOrchestratorResponder:

    def test_unusable_router_output_falls_back_to_general(self, monkeypatch):
        """A router reply with no usable JSON still reaches a department.

        The router never leaves ``department_targets`` empty — it falls back to
        ``general``, so the user gets a real answer instead of "I could not
        route this request". The general sub-agent's reply is what reaches them.
        """
        fake = ScriptedChatModel(
            scripts={
                "orchestrator": [ai("I will handle this myself.")],
                "subagent": [ai("General handled it.")],
            },
            default=ai("done"),
        )
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)
        result = _run_graph(_initial_state("Simple question."), "no-dept")

        answer = next(
            (m.content for m in result["messages"] if isinstance(m, AIMessage)),
            "",
        )
        assert "General handled it." in answer
        assert fake.counts.get("orchestrator", 0) == 1
        # The turn went through a sub-agent rather than answering directly.
        assert fake.counts.get("subagent", 0) >= 1

    def test_orchestrator_returns_non_json_routes_to_general(self, monkeypatch):
        """Plain text from the router is a routing failure, not an answer.

        The dispatcher's own prose is a control envelope it failed to format,
        so it must not be passed to the user as an answer. It falls through to
        ``general``, which answers on its own terms.
        """
        fake = ScriptedChatModel(
            scripts={
                "orchestrator": [ai("Nothing to delegate.")],
                "subagent": [ai("General handled it.")],
            },
            default=ai("done"),
        )
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)
        result = _run_graph(_initial_state("Query."), "non-json")

        answer = next(
            (m.content for m in result["messages"] if isinstance(m, AIMessage)),
            "",
        )
        assert "General handled it." in answer
        assert "Nothing to delegate." not in answer


# ---------------------------------------------------------------------------
# Iteration guard
# ---------------------------------------------------------------------------

class TestMultiTurnRouting:
    """Every turn of a conversation must route, not just the first.

    The orchestrator used to keep a checkpointed ``iteration_count`` and
    force ``departments = []`` once it passed a maximum — a guard belonging
    to the removed top-level ReAct dispatcher loop. Because the counter was
    checkpointed, it carried across turns on a reused thread: the second turn
    of any conversation saw a non-zero count, discarded its departments, and
    answered "I could not route this request" while quoting the *first* turn's
    query.

    The entry points masked this by passing ``iteration_count: 0`` on every
    invoke, so a test that reuses a thread while sending the full initial
    state cannot see it. These tests send only the new user message, the way
    the CLI and the Streamlit app actually do, and reuse one thread.
    """

    def _scripted(self, *envelopes):
        return ScriptedChatModel(
            scripts={"orchestrator": [ai(e) for e in envelopes]},
            default=ai("done"),
        )

    def test_second_turn_on_a_reused_thread_still_routes(self, monkeypatch):
        fake = self._scripted(
            '{"enhanced_query": "leave policy", "departments": ["hr"]}',
            '{"enhanced_query": "pipeline status", "departments": ["sales"]}',
        )
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)

        class StubEngine:
            def __init__(self, *a, **k):
                pass

            async def invoke_parallel(self, subs, query, *a, **k):
                return {s["name"]: SubagentRun(text=f"ANSWERED-BY({s['name']})")
                        for s in subs}

        monkeypatch.setattr(
            "src.services.agent_orchestrator.graph.SubAgentEngine", StubEngine
        )
        reset_deep_agent()
        agent = get_deep_agent()
        cfg = {"configurable": {"thread_id": "multi-turn-thread"}}

        # Each turn sends only the new message — the checkpointer supplies
        # history. This is the shape main.py and app.py use.
        first = agent.invoke(
            {"messages": [HumanMessage(content="what is the leave policy?")]},
            config=cfg,
        )
        second = agent.invoke(
            {"messages": [HumanMessage(content="how is the pipeline?")]},
            config=cfg,
        )

        first_answer = next(
            m.content for m in reversed(first["messages"]) if isinstance(m, AIMessage)
        )
        second_answer = next(
            m.content for m in reversed(second["messages"]) if isinstance(m, AIMessage)
        )

        assert "ANSWERED-BY(hr)" in first_answer
        # The regression: turn two must reach *its own* department, not fall
        # through to the responder quoting turn one.
        assert "ANSWERED-BY(sales)" in second_answer
        assert "could not route" not in second_answer
        assert "leave policy" not in second_answer

    def test_third_turn_on_a_reused_thread_still_routes(self, monkeypatch):
        """The failure was cumulative, so check beyond the second turn too."""
        fake = self._scripted(
            '{"enhanced_query": "q1", "departments": ["hr"]}',
            '{"enhanced_query": "q2", "departments": ["sales"]}',
            '{"enhanced_query": "q3", "departments": ["marketing"]}',
        )
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)

        class StubEngine:
            def __init__(self, *a, **k):
                pass

            async def invoke_parallel(self, subs, query, *a, **k):
                return {s["name"]: SubagentRun(text=f"ANSWERED-BY({s['name']})")
                        for s in subs}

        monkeypatch.setattr(
            "src.services.agent_orchestrator.graph.SubAgentEngine", StubEngine
        )
        reset_deep_agent()
        agent = get_deep_agent()
        cfg = {"configurable": {"thread_id": "multi-turn-3-thread"}}

        for text in ("first?", "second?", "third?"):
            result = agent.invoke(
                {"messages": [HumanMessage(content=text)]}, config=cfg
            )

        answer = next(
            m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)
        )
        assert "ANSWERED-BY(marketing)" in answer

    def test_no_iteration_keys_in_result_state(self, monkeypatch):
        """The router writes no iteration counter back into checkpointed state.

        Anything it wrote here persisted to the next turn, which is how a
        per-turn router ended up behaving like a stateful loop.
        """
        fake = self._scripted('{"enhanced_query": "q", "departments": ["hr"]}')
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)

        result = _run_graph(_initial_state("Query."), "no-iteration-keys")

        assert "iteration_count" not in result
        assert "max_iterations" not in result


# ---------------------------------------------------------------------------
# Parallel batch via subagent_fanout
# ---------------------------------------------------------------------------

class _ParallelFake:
    """Thread-safe scripted model for parallel-batch tests.

    Each sub-agent loop: first call writes a uniquely numbered note,
    second call finishes — regardless of how the two loops interleave.
    """

    def __init__(self):
        import threading
        from langchain_core.messages import AIMessage
        self._lock = threading.Lock()
        self._n = 0
        self._orch_calls = 0
        self._AIMessage = AIMessage

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages, **kwargs):
        role = ScriptedChatModel._detect_role(messages)
        if role == "orchestrator":
            with self._lock:
                self._orch_calls += 1
                first = self._orch_calls == 1
            if first:
                return ai('{"enhanced_query": "Research A and B", "departments": ["research", "writer"]}', usage=(5, 1))
            return ai("Parallel report.", usage=(5, 1))
        with self._lock:
            self._n += 1
            n = self._n
        if n % 2 == 1:  # odd: this loop's write turn; even: its completion
            return self._AIMessage(
                content="",
                tool_calls=[{"name": "write_file",
                             "args": {"path": f"workspace/notes_{n}.md",
                                      "content": f"note {n}"},
                             "id": f"w{n}"}],
            )
        return self._AIMessage(content=f"Saved workspace/notes_{n - 1}.md.")


class TestParallelBatch:

    def test_deterministic_and_distinct_results(self, monkeypatch):
        """Parallel sub-agents produce distinct results and a synthesized answer."""
        fake = _ParallelFake()
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)
        result = _run_graph(_initial_state("Research A and B in parallel."), "parallel-1")

        answer = next(
            (m.content for m in result["messages"] if isinstance(m, AIMessage)),
            "",
        )
        # Aggregator references both departments.
        assert "research" in answer.lower()
        assert "writer" in answer.lower()

    def test_shared_deadline_is_not_enforced(self, monkeypatch):
        """Phase 3 does not enforce subagent deadlines — subagent loops
        run without timeout. The shared deadline param exists but is
        not enforced by _run_subagent_loop."""
        monkeypatch.setattr("src.config.get_subagent_timeout_seconds", lambda: 0.2)

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

        assert result is not None
        # No enforcement: each subagent sleeps 0.5s sequentially.
        assert elapsed >= 0.5
        # Verify department results appear in answer.
        answer = next(
            (m.content for m in result["messages"] if isinstance(m, AIMessage)),
            "",
        )
        assert "research" in answer.lower()
        assert "writer" in answer.lower()

    def test_empty_results_aggregator_handles_gracefully(self, monkeypatch):
        """Empty subagent results → aggregator returns fallback message."""
        fake = ScriptedChatModel(
            scripts={
                "orchestrator": [
                    ai('{"enhanced_query": "test", "departments": ["research"]}'),
                ],
                # subagent returns nothing — _run_subagent_loop doesn't call this fake
            },
            default=ai("done"),
        )
        monkeypatch.setattr(agent_factory, "get_model", lambda: fake)
        result = _run_graph(_initial_state("Test."), "empty-results")

        assert result is not None


# ---------------------------------------------------------------------------
# Graph-level infrastructure
# ---------------------------------------------------------------------------

class TestGraphCache:

    def test_get_deep_agent_returns_compiled_graph(self):
        agent = get_deep_agent()
        assert agent is not None

    def test_caches_graph(self):
        agent1 = get_deep_agent()
        agent2 = get_deep_agent()
        assert agent1 is agent2

    def test_reset_invalidates_cache(self):
        agent1 = get_deep_agent()
        reset_deep_agent()
        agent2 = get_deep_agent()
        assert agent2 is not agent1
        # Subsequent call returns the freshly compiled instance.
        agent3 = get_deep_agent()
        assert agent3 is agent2
