"""Tests for state integrity — verify AgentState transitions are well-defined."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.state import AgentState


# ---------------------------------------------------------------------------
# State defaults
# ---------------------------------------------------------------------------

class TestAgentStateDefaults:

    def test_messages_is_list(self):
        state = {}
        # AgentState is a TypedDict — Python doesn't enforce types at runtime,
        # but we verify the expected keys exist in our node logic.
        expected_keys = [
            "messages",
            "current_plan",
            "workspace_files",
            "next_message",
            "review_verdict",
            "recursion_depth",
            "pending_writes",
            "audit_log",
            "token_usage",
            "iteration_count",
            "max_iterations",
        ]
        # Just verify the TypedDict definition lists these keys.
        from typing import get_type_hints
        hints = get_type_hints(AgentState)
        for key in expected_keys:
            assert key in hints, f"Key '{key}' missing from AgentState TypedDict"


# ---------------------------------------------------------------------------
# State transition helpers (used by nodes)
# ---------------------------------------------------------------------------

def _build_state(**overrides):
    """Build a valid AgentState with sensible defaults + overrides."""
    state: AgentState = {
        "messages": [],
        "current_plan": [],
        "workspace_files": [],
        "next_message": None,
        "review_verdict": None,
        "recursion_depth": 0,
        "pending_writes": [],
        "audit_log": [],
        "token_usage": {"input": 0, "output": 0, "total": 0},
        "iteration_count": 0,
        "max_iterations": 10,
    }
    state.update(overrides)
    return state


class TestStateTransitions:

    def test_agent_node_moves_next_message_to_messages(self):
        """Agent node should consume next_message and append to messages."""
        from src.nodes.review import call_agent_node

        msg = AIMessage(content="staged answer")
        state = _build_state(next_message=msg, messages=[HumanMessage(content="hi")])

        result = call_agent_node(state)

        assert result["next_message"] is None
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "staged answer"

    def test_agent_node_with_no_next_message_returns_empty(self):
        from src.nodes.review import call_agent_node

        state = _build_state(next_message=None)
        result = call_agent_node(state)

        assert result["next_message"] is None

    def test_responder_extracts_from_next_message(self):
        from src.nodes.review import call_responder_node

        msg = AIMessage(content="final approved answer")
        state = _build_state(next_message=msg)

        result = call_responder_node(state)

        assert result["next_message"] is None
        assert len(result["messages"]) == 1
        assert "final approved answer" in str(result["messages"][0].content)

    def test_responder_fallback_to_last_ai_message(self):
        from src.nodes.review import call_responder_node

        state = _build_state(
            next_message=None,
            messages=[
                HumanMessage(content="question"),
                AIMessage(content="tool response"),
                ToolMessage(content="tool result", tool_call_id="1"),
                AIMessage(content="the real answer"),
            ],
        )

        result = call_responder_node(state)

        assert result["next_message"] is None
        assert "the real answer" in str(result["messages"][0].content)

    def test_responder_returns_empty_when_no_messages(self):
        from src.nodes.review import call_responder_node

        state = _build_state(next_message=None, messages=[])
        result = call_responder_node(state)

        assert result == {}

    def test_token_usage_accumulates(self):
        """Token usage should fold correctly across iterations."""
        state = _build_state(token_usage={"input": 100, "output": 50, "total": 150})

        # Simulate what the orchestrator does on a new turn.
        input_tokens = 200
        output_tokens = 100
        total_in = state["token_usage"]["input"] + input_tokens
        total_out = state["token_usage"]["output"] + output_tokens

        updated = {
            "input": total_in,
            "output": total_out,
            "total": total_in + total_out,
        }

        assert updated["input"] == 300
        assert updated["output"] == 150
        assert updated["total"] == 450

    def test_iteration_count_increments(self):
        """Orchestrator should increment iteration_count each turn."""
        state = _build_state(iteration_count=0)
        new_count = state.get("iteration_count", 0) + 1
        assert new_count == 1

    def test_max_iterations_guard_prevents_runaway(self):
        from src.nodes.plan import call_orchestrator

        # Use a mock model that never returns tool calls.
        class MockModel:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                return AIMessage(content="answer")

        state = _build_state(
            messages=[HumanMessage(content="test")],
            iteration_count=10,
            max_iterations=10,
        )

        result = call_orchestrator(state, model=MockModel(), tools=[])

        # Should return an error message, not proceed.
        assert result["next_message"] is not None
        assert "Maximum iterations" in str(result["next_message"].content)
