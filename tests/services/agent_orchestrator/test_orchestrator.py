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
        "recursion_depth": 0,
        "pending_writes": [],
        "audit_log": [],
        "token_usage": {},
        "iteration_count": 0,
        "max_iterations": 25,
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
        assert result["iteration_count"] == 1

    def test_detection_mode_no_departments_routes_to_responder(self):
        fake = FakeModel([AIMessage(content="I will handle this myself.")]
                        )
        state = _state()
        result = call_orchestrator(state, fake)
        assert result["department_targets"] == []
        assert result["iteration_count"] == 1

    def test_increments_iteration_count(self):
        fake = FakeModel([AIMessage(content='{"enhanced_query": "q", "departments": []}')])
        state = _state(iteration_count=5)
        result = call_orchestrator(state, fake)
        assert result["iteration_count"] == 6

    def test_max_iterations_returns_error(self):
        fake = FakeModel([AIMessage(content="should not reach")])
        state = _state(iteration_count=25, max_iterations=25)
        result = call_orchestrator(state, fake)
        assert "Maximum iterations reached" in result["next_message"].content

    def test_max_iterations_config_via_state(self):
        fake = FakeModel([AIMessage(content="should not reach")])
        state = _state(iteration_count=3, max_iterations=3)
        result = call_orchestrator(state, fake)
        assert "Maximum iterations" in result["next_message"].content

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
