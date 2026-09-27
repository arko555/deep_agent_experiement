"""Tests for Phase 3 routing logic."""

import pytest

from src.services.agent_orchestrator.routing import route_from_orchestrator
from src.services.agent_orchestrator.state import AgentState


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
        "department_targets": [],
        **kwargs,
    }


class TestRouteFromOrchestratorPhase3:

    def test_no_departments_routes_to_responder(self):
        state = _state(department_targets=[])
        assert route_from_orchestrator(state) == "responder"

    def test_departments_found_routes_to_subagent_fanout(self):
        state = _state(department_targets=["research", "writer"])
        assert route_from_orchestrator(state) == "subagent_fanout"

    def test_single_department_routes_to_subagent_fanout(self):
        state = _state(department_targets=["research"])
        assert route_from_orchestrator(state) == "subagent_fanout"

    def test_max_iterations_routes_to_responder(self):
        state = _state(
            department_targets=[],
            iteration_count=25,
            max_iterations=25,
        )
        assert route_from_orchestrator(state) == "responder"

    def test_iteration_below_max_with_departments_still_fanout(self):
        state = _state(
            department_targets=["research"],
            iteration_count=24,
            max_iterations=25,
        )
        assert route_from_orchestrator(state) == "subagent_fanout"

    def test_empty_department_targets_string(self):
        state = _state(department_targets="")
        assert route_from_orchestrator(state) == "responder"

    def test_missing_department_targets_key_routes_to_responder(self):
        state = _state()
        state.pop("department_targets")
        assert route_from_orchestrator(state) == "responder"

    def test_iteration_over_budget_with_departments_still_fanout(self):
        # Budget exhaustion is handled inside call_orchestrator (it returns an
        # empty department_targets), so the router itself only branches on
        # whether any departments were found.
        state = _state(
            department_targets=["research"],
            iteration_count=99,
            max_iterations=25,
        )
        assert route_from_orchestrator(state) == "subagent_fanout"
