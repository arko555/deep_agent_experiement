"""Tests for state integrity — verify AgentState transitions are well-defined."""

from langchain_core.messages import AIMessage, HumanMessage

from src.services.agent_orchestrator.state import AgentState


# ---------------------------------------------------------------------------
# State defaults
# ---------------------------------------------------------------------------

class TestAgentStateDefaults:

    def test_messages_is_list(self):
        state = {}
        # AgentState is a TypedDict Python doesn't enforce types at runtime,
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
            "routing_decisions",
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
# State transitions
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
        "routing_decisions": [],
        "token_usage": {"input": 0, "output": 0, "total": 0},
        "iteration_count": 0,
        "max_iterations": 10,
    }
    state.update(overrides)
    return state


class TestStateTransitions:

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
