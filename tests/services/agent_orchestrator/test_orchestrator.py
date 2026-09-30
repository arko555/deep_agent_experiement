"""Tests for the dispatcher orchestrator (Phase 3)."""

import pytest

from src.services.agent_orchestrator.orchestrator import call_orchestrator, _parse_dispatcher_output
from src.services.agent_orchestrator.state import AgentState
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage


def _state(**kwargs):
    return {
        "messages": [],
        "current_plan": [],
        "next_message": None,
        "review_verdict": None,
        "pending_writes": [],
        "audit_log": [],
        "token_usage": {},
        "thread_id": "test-thread",
        "enhanced_query": "",
        "department_targets": [],
        "subagent_results": {},
        **kwargs,
    }


class FakeModel:
    """A fake model that returns canned responses."""

    def __init__(self, responses):
        from collections import deque
        self._responses = deque(responses)

    def invoke(self, messages, **kwargs):
        resp = self._responses.popleft() if self._responses else self._responses[0]
        return resp

    def bind_tools(self, tools, **kwargs):
        return self


class TestParseDispatcherOutput:

    def test_valid_json_extracted(self):
        content = 'Here is the result: {"enhanced_query": "find research papers", "departments": ["research", "writer"]} done.'
        result = _parse_dispatcher_output(content)
        assert result is not None
        assert result["enhanced_query"] == "find research papers"
        assert result["departments"] == ["research", "writer"]

    def test_no_json_returns_none(self):
        content = "I will handle this myself."
        assert _parse_dispatcher_output(content) is None

    def test_empty_string_returns_none(self):
        assert _parse_dispatcher_output("") is None

    def test_mismatched_braces_returns_none(self):
        content = '{"enhanced_query": "test", "departments": ["research"'
        assert _parse_dispatcher_output(content) is None

    def test_departments_not_list_returns_none(self):
        content = '{"enhanced_query": "test", "departments": "research"}'
        assert _parse_dispatcher_output(content) is None


class TestCallOrchestrator:

    def test_detection_mode_returns_departments(self):
        fake = FakeModel([AIMessage(content='{"enhanced_query": "find papers", "departments": ["research", "writer"]}')])
        state = _state()
        result = call_orchestrator(state, fake)
        assert result["enhanced_query"] == "find papers"
        assert result["department_targets"] == ["research", "writer"]

    def test_unusable_router_output_falls_back_to_general(self):
        """An unparseable router reply still has to land somewhere.

        The responder has no tools of its own, so an empty department list
        would leave the user with "I could not route this request". The
        router therefore falls back to the ``general`` department, which
        handles greetings, refuses off-topic queries, and does the work.
        """
        fake = FakeModel([AIMessage(content="I will handle this myself.")]
                        )
        state = _state()
        result = call_orchestrator(state, fake)
        assert result["department_targets"] == ["general"]

    def test_empty_departments_list_falls_back_to_general(self):
        """A well-formed envelope carrying no departments is not a dead end."""
        fake = FakeModel([AIMessage(
            content='{"enhanced_query": "q", "departments": []}'
        )])
        state = _state()
        result = call_orchestrator(state, fake)
        assert result["department_targets"] == ["general"]

    def test_invented_department_names_fall_back_to_general(self):
        """A hallucinated department is dropped, then the fallback applies."""
        fake = FakeModel([AIMessage(
            content='{"enhanced_query": "q", "departments": ["legal", "nope"]}'
        )])
        state = _state()
        result = call_orchestrator(state, fake)
        assert result["department_targets"] == ["general"]

    def test_router_has_no_iteration_budget(self):
        """The router is single-shot: no counter, no ceiling, no error path.

        It used to keep a checkpointed ``iteration_count`` and bail with
        "Maximum iterations reached" once it hit a maximum. The graph is
        acyclic now, so the router runs once per turn and there is nothing to
        bound. It also must not write an iteration counter back into state —
        a counter here leaked across turns on a reused checkpoint thread and
        made every conversation after the first fail to route.
        """
        fake = FakeModel([AIMessage(
            content='{"enhanced_query": "q", "departments": ["research"]}'
        )])
        result = call_orchestrator(_state(), fake)

        assert "iteration_count" not in result
        assert "max_iterations" not in result
        # The one job it does have: still routes, and still answers.
        assert result["department_targets"] == ["research"]
        assert result["enhanced_query"] == "q"

    def test_enhanced_query_is_not_derived_from_prior_state(self):
        """Each turn's enhanced query is the router's own, not a carried-over one.

        The old ``iteration_count == 0`` branch read ``state["enhanced_query"]``
        back out of the checkpoint on later turns, so a second turn could
        answer with the first turn's question.
        """
        fake = FakeModel([AIMessage(
            content='{"enhanced_query": "this turn\'s question", "departments": ["sales"]}'
        )])
        state = _state(enhanced_query="a stale question from a previous turn")

        result = call_orchestrator(state, fake)

        assert result["enhanced_query"] == "this turn's question"

    def test_departments_from_llm_output(self):
        fake = FakeModel([AIMessage(content='Analyze: {"enhanced_query": "comprehensive review", "departments": ["research"]}')])
        state = _state()
        result = call_orchestrator(state, fake)
        assert result["enhanced_query"] == "comprehensive review"
        assert result["department_targets"] == ["research"]


class TestCallOrchestratorSessionMemory:

    def test_uses_session_memory_window(self, monkeypatch):
        """Orchestrator should call session_memory.get_window."""
        captured = {}

        def fake_get_window(thread_id, max_messages=20):
            captured["thread_id"] = thread_id
            captured["max_messages"] = max_messages
            return [HumanMessage(content="user message")]

        monkeypatch.setattr(
            "src.services.agent_orchestrator.orchestrator._get_window",
            fake_get_window,
        )

        fake = FakeModel([AIMessage(content='{"enhanced_query": "q", "departments": []}')])
        state = _state(thread_id="thread-1")
        call_orchestrator(state, fake)
        assert captured["thread_id"] == "thread-1"
        assert captured["max_messages"] == 20
