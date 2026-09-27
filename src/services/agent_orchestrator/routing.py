"""Routing functions for the deep agent graph.

The graph is a ReAct loop around the dispatcher:

    START → orchestrator → {tools | subagent_fanout | responder}
                 ↑              │
                 └──────────────┘   (tools node loops back until the model
                                       stops asking for tools)

``route_after_orchestrator`` is the single conditional edge out of the
dispatcher. It checks tool calls first, because a model that wants a tool emits
one *instead of* the JSON envelope.
"""

from src.services.agent_orchestrator.state import AgentState


def _has_tool_calls(state: AgentState) -> bool:
    """True when the dispatcher's last message requests tool execution."""
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", None) == "ai":
            return bool(getattr(message, "tool_calls", None))
        break  # only the most recent AI message decides
    return False


def route_after_orchestrator(state: AgentState):
    """Route out of the dispatcher: tools → fanout → responder, in that order.

    Tool calls win over departments. A model that wants to run a tool does not
    also return a department envelope, so there is no case where both are set
    and the ordering matters — but checking tool calls first is what makes an
    empty ``department_targets`` non-fatal.
    """
    if _has_tool_calls(state):
        return "tools"
    if state.get("department_targets"):
        return "subagent_fanout"
    return "responder"


def route_from_orchestrator(state: AgentState):
    """Backwards-compatible alias for :func:`route_after_orchestrator`.

    Kept because ``agent_orchestrator.__init__`` exports this name and external
    callers may import it. The Phase 3 graph uses the new name.
    """
    return route_after_orchestrator(state)
