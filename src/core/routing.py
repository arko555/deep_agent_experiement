"""Routing functions for the deep agent graph."""

from src.state import AgentState


def _get_content_text(content):
    """Extract text from multimodal content (string or list of dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
    return str(content)


def route_from_orchestrator(state: AgentState):
    next_msg = state.get("next_message")
    if not next_msg:
        return "END"

    if hasattr(next_msg, "tool_calls") and next_msg.tool_calls:
        return "agent"

    plan = state.get("current_plan", [])
    if plan:
        return "plan_checker"
    return "critic"


def route_from_critic(state: AgentState):
    next_msg = state.get("next_message")
    if not next_msg:
        return "responder"

    content = _get_content_text(next_msg.content if hasattr(next_msg, "content") else "")
    if "APPROVED" in content.upper():
        return "responder"
    else:
        return "orchestrator"


def route_from_plan_checker(state: AgentState):
    next_msg = state.get("next_message")
    if not next_msg:
        return "critic"

    content = _get_content_text(next_msg.content if hasattr(next_msg, "content") else "")
    if "COMPLIANT" in content.upper():
        return "critic"
    else:
        return "orchestrator"
