"""Routing functions for the deep agent graph.

Routers branch on explicit state fields (``review_verdict``, tool calls,
plan presence) rather than inspecting message content, so routing can never
be fooled by a verdict keyword appearing inside a response.
"""

from src.core.config import get_max_iterations
from src.state import AgentState


def route_from_orchestrator(state: AgentState):
    next_msg = state.get("next_message")
    if not next_msg:
        return "end"

    if hasattr(next_msg, "tool_calls") and next_msg.tool_calls:
        return "agent"

    # Budget exhausted: deliver the orchestrator's stop message instead of
    # burning more reviewer cycles on it.
    max_iterations = state.get("max_iterations") or get_max_iterations()
    if state.get("iteration_count", 0) >= max_iterations:
        return "responder"

    plan = state.get("current_plan", [])
    if plan:
        return "plan_checker"
    return "critic"


def route_after_tools(state: AgentState):
    """Always return to orchestrator after tools execute.

    Write operations (write_file, edit_file) execute immediately in the
    tools node and are tracked in ``pending_writes`` for audit purposes only.
    No HITL pause — the graph continues unblocked.
    """
    return "orchestrator"



def route_from_critic(state: AgentState):
    # The critic node sets ``review_verdict`` explicitly.
    if state.get("review_verdict") == "approved":
        return "responder"

    # After repeated rejections, route through reflection to break the loop
    # and generate a revised strategy before going back to orchestrator.
    iteration_count = state.get("iteration_count", 0)
    if iteration_count >= 4:
        return "reflection"

    return "orchestrator"


def route_from_reflection(state: AgentState):
    # Reflection produces strategic guidance; go back to orchestrator to act on it.
    return "orchestrator"


def route_from_plan_checker(state: AgentState):
    # The plan checker sets ``review_verdict`` explicitly.
    if state.get("review_verdict") == "compliant":
        return "critic"
    return "orchestrator"
