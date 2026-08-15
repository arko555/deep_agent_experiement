"""Routing functions for the deep agent graph.

Routers branch on explicit state fields (``review_verdict``, tool calls,
plan presence) rather than inspecting message content, so routing can never
be fooled by a verdict keyword appearing inside a response.
"""

from src.state import AgentState


def route_from_orchestrator(state: AgentState):
    next_msg = state.get("next_message")
    if not next_msg:
        return "end"

    if hasattr(next_msg, "tool_calls") and next_msg.tool_calls:
        return "agent"

    # Budget exhausted: deliver the orchestrator's stop message instead of
    # burning more reviewer cycles on it.
    if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
        return "responder"

    plan = state.get("current_plan", [])
    if plan:
        return "plan_checker"
    return "critic"


def route_from_critic(state: AgentState):
    # The critic node sets ``review_verdict`` explicitly.
    if state.get("review_verdict") == "approved":
        return "responder"
    return "orchestrator"


def route_from_plan_checker(state: AgentState):
    # The plan checker sets ``review_verdict`` explicitly.
    if state.get("review_verdict") == "compliant":
        return "critic"
    return "orchestrator"
