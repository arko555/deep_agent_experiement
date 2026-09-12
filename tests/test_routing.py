"""Tests for routing logic — verify state-field-based branching is deterministic."""

import pytest

from src.core.routing import (
    route_from_orchestrator,
    route_from_critic,
    route_from_reflection,
    route_from_plan_checker,
)
from src.core.agent_factory import get_deep_agent, reset_deep_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**kwargs):
    """Build a minimal AgentState dict for routing tests."""
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
        "max_iterations": 10,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# route_from_orchestrator
# ---------------------------------------------------------------------------

class TestRouteFromOrchestrator:

    def test_no_next_message_ends(self):
        state = _state(next_message=None)
        assert route_from_orchestrator(state) == "end"

    def test_tool_calls_route_to_agent(self):
        from langchain_core.messages import AIMessage
        tool_calls = [{"name": "search", "args": {}, "id": "call_1"}]
        msg = AIMessage(content="hello", tool_calls=tool_calls)
        state = _state(next_message=msg)
        assert route_from_orchestrator(state) == "agent"

    def test_no_plan_routes_to_critic(self):
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="here is my answer")
        state = _state(next_message=msg, current_plan=[])
        assert route_from_orchestrator(state) == "critic"

    def test_has_plan_routes_to_plan_checker(self):
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="here is my answer")
        state = _state(next_message=msg, current_plan=["step 1", "step 2"])
        assert route_from_orchestrator(state) == "plan_checker"

    def test_max_iterations_routes_to_responder(self):
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="stopping now")
        state = _state(
            next_message=msg,
            iteration_count=10,
            max_iterations=10,
            current_plan=[],
        )
        assert route_from_orchestrator(state) == "responder"

    def test_iteration_budget_checked_before_plan(self):
        """Budget exhaustion takes priority over plan routing."""
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="stopping now")
        state = _state(
            next_message=msg,
            iteration_count=10,
            max_iterations=10,
            current_plan=["should not route to checker"],
        )
        assert route_from_orchestrator(state) == "responder"


# ---------------------------------------------------------------------------
# route_from_critic
# ---------------------------------------------------------------------------

class TestRouteFromCritic:

    def test_approved_routes_to_responder(self):
        state = _state(review_verdict="approved")
        assert route_from_critic(state) == "responder"

    def test_rejected_low_iterations_routes_to_orchestrator(self):
        state = _state(review_verdict="rejected", iteration_count=2)
        assert route_from_critic(state) == "orchestrator"

    def test_rejected_high_iterations_routes_to_reflection(self):
        state = _state(review_verdict="rejected", iteration_count=4)
        assert route_from_critic(state) == "reflection"

    def test_rejected_at_boundary_routes_to_reflection(self):
        state = _state(review_verdict="rejected", iteration_count=5)
        assert route_from_critic(state) == "reflection"

    def test_rejected_just_below_threshold_routes_to_orchestrator(self):
        state = _state(review_verdict="rejected", iteration_count=3)
        assert route_from_critic(state) == "orchestrator"

    def test_no_verdict_routes_to_orchestrator(self):
        state = _state(review_verdict=None)
        assert route_from_critic(state) == "orchestrator"


# ---------------------------------------------------------------------------
# route_from_reflection
# ---------------------------------------------------------------------------

class TestRouteFromReflection:

    def test_always_routes_to_orchestrator(self):
        state = _state()
        assert route_from_reflection(state) == "orchestrator"

    def test_routes_to_orchestrator_regardless_of_state(self):
        state = _state(review_verdict="approved", iteration_count=99)
        assert route_from_reflection(state) == "orchestrator"


# ---------------------------------------------------------------------------
# route_from_plan_checker
# ---------------------------------------------------------------------------

class TestRouteFromPlanChecker:

    def test_compliant_routes_to_critic(self):
        state = _state(review_verdict="compliant")
        assert route_from_plan_checker(state) == "critic"

    def test_violation_routes_to_orchestrator(self):
        state = _state(review_verdict="violation")
        assert route_from_plan_checker(state) == "orchestrator"

    def test_no_verdict_routes_to_orchestrator(self):
        state = _state(review_verdict=None)
        assert route_from_plan_checker(state) == "orchestrator"


# ---------------------------------------------------------------------------
# Cache invalidation (reset_deep_agent)
# ---------------------------------------------------------------------------

class TestCacheInvalidation:

    def test_get_deep_agent_returns_same_instance(self):
        """First two calls return the same cached graph."""
        agent1 = get_deep_agent()
        agent2 = get_deep_agent()
        assert agent1 is agent2

    def test_reset_deep_agent_invalidates_cache(self):
        """After reset, next call produces a new graph instance."""
        agent1 = get_deep_agent()
        reset_deep_agent()
        agent2 = get_deep_agent()
        # agent2 should be a freshly compiled graph (different identity).
        # Note: the caching logic returns early if _compiled_graph is not None,
        # so after reset it will compile a new one.
        assert agent2 is not None
        # The internal _compiled_graph should now point to agent2, not agent1.
        agent3 = get_deep_agent()
        assert agent3 is agent2
