"""Routing out of the orchestrator.

    START → orchestrator → subagent_fanout → responder → END
                    ↘                  → responder ↗

The orchestrator is a router: it binds no tools, so it never emits a
``tool_calls`` message and the turn cannot re-enter a top-level tools node.
Every tool call happens inside a sub-agent, which keeps communication one-way
— orchestrator → sub-agent → tool.
"""

from src.services.agent_orchestrator.state import AgentState


def _has_tool_calls(state: AgentState) -> bool:
    """True if the last AI message somehow requests tool execution.

    The router is invoked without bound tools, so this should be unreachable.
    It is kept as a guard rather than removed: if a tool ever gets bound to
    the router again, the turn would otherwise fall into ``subagent_fanout``
    with an unconsumed ``tool_calls`` message, and the model would be told it
    cannot act on its own request.
    """
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", None) == "ai":
            return bool(getattr(message, "tool_calls", None))
        break  # only the most recent AI message decides
    return False


def route_after_orchestrator(state: AgentState):
    """Route out of the router: fanout, or the responder when nothing was picked.

    A department list means sub-agents run and their results are synthesized
    by the responder. An empty list should be unreachable — ``call_orchestrator``
    applies the ``general`` fallback — but if it happens the responder answers
    rather than the turn ending with no output.
    """
    if _has_tool_calls(state):
        return "responder"
    if state.get("department_targets"):
        return "subagent_fanout"
    return "responder"


def route_from_orchestrator(state: AgentState):
    """Backwards-compatible alias for :func:`route_after_orchestrator`.

    Kept because ``agent_orchestrator.__init__`` exports this name and external
    callers may import it.
    """
    return route_after_orchestrator(state)
