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
        "pending_writes": [],
        "audit_log": [],
        "token_usage": {},
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

    def test_routing_ignores_any_iteration_state(self):
        """The router branches on departments alone.

        There is no iteration budget to consult: the graph is acyclic, so the
        orchestrator runs once per turn. Any iteration counter left in state
        by an older checkpoint is inert rather than a reason to stop routing.
        """
        state = _state(
            department_targets=["research"],
            iteration_count=99,
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
