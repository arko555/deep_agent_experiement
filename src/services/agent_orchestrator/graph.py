"""LangGraph compilation for the deep agent.

    START → orchestrator → subagent_fanout → responder → END
                    ↘                  → responder ↗

One direction, three nodes. The orchestrator routes and binds no tools; tool
calls happen inside a sub-agent's own ReAct loop.
"""

import asyncio
import logging
from datetime import datetime

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from src.services.agent_orchestrator.state import AgentState
from src.services.agent_orchestrator.orchestrator import call_orchestrator
from src.services.agent_orchestrator.routing import route_after_orchestrator
from src.services.agent_orchestrator.subagent_engine import SubAgentEngine
from src.services.agent_orchestrator.aggregator import aggregate
from src.services.agent_orchestrator.verification import verify
from src.services.session_memory.checkpoint import get_saver
from src.utils import get_message_text

logger = logging.getLogger(__name__)

_compiled_graph = None


def _local_orchestrator_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Orchestrator node: enhance the query, name the departments.

    The thread id lives in the runtime config, not in state, so it is read
    from there — otherwise the dispatcher sees an empty message window.

    No tool set is built here. This node used to call ``get_all_tools()`` and
    hand the result to the dispatcher, which ignored it — so every turn paid
    for loading the dynamic and MCP tool layers to produce a value the router
    never looked at. The router routes; the sub-agents bind tools.
    """
    from src.services.agent_orchestrator.agent_factory import get_model

    thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
    return call_orchestrator(state, model=get_model(), thread_id=thread_id)


def _subagent_fanout_node(state: AgentState) -> dict:
    """Fan out to department sub-agents in parallel.

    Only the department *name* crosses this boundary. The name resolves to a
    ``SubagentSpec`` in ``subagents.SUBAGENTS``, and the spec's ``skill`` points
    at a ``SKILL.md`` whose body becomes the sub-agent's system prompt. This
    node used to synthesize ``f"You are the {dept} specialist..."`` here, which
    bypassed ``skills/`` entirely and threw away the department prompt.

    Names that no longer resolve (a skill deleted mid-turn, a name the
    validator let through) are dropped with a warning. If nothing resolves,
    the empty result lets the responder answer directly instead of the user
    seeing an "unknown department" error.
    """
    from src.services.agent_orchestrator.subagents import SUBAGENTS

    departments = state.get("department_targets", [])
    enhanced_query = state.get("enhanced_query", "")

    subagents = []
    for dept in departments:
        if dept in SUBAGENTS:
            subagents.append({"name": dept, "description": enhanced_query})
        else:
            logger.warning(
                "Department %r no longer exists; dropping it from the fanout", dept
            )

    if not subagents:
        logger.warning("No requested department resolved; answering directly")
        return {"subagent_results": {}}

    engine = SubAgentEngine()
    runs = asyncio.run(engine.invoke_parallel(subagents, enhanced_query))

    # Token spend and file writes happen inside the sub-agents now, so the
    # parent folds them in here. The top-level tools node that used to do this
    # is gone with the `task` tool; without this the audit trail the Streamlit
    # app renders would be permanently empty.
    child_in = sum(r.usage.get("input", 0) for r in runs.values())
    child_out = sum(r.usage.get("output", 0) for r in runs.values())
    updates: dict = {"subagent_results": {n: r.text for n, r in runs.items()}}

    if child_in or child_out:
        current = state.get("token_usage", {}) or {}
        total_in = current.get("input", 0) + child_in
        total_out = current.get("output", 0) + child_out
        updates["token_usage"] = {
            "input": total_in,
            "output": total_out,
            "total": total_in + total_out,
        }

    child_writes = [op for r in runs.values() for op in r.writes]
    if child_writes:
        updates["pending_writes"] = list(state.get("pending_writes", [])) + [
            {
                "tool": op.get("tool", "write_file"),
                "tool_id": op.get("tool_id"),
                "args": op.get("args", {}),
                "status": op.get("status", "executed"),
            }
            for op in child_writes
        ]

    if child_writes:
        updates["audit_log"] = [{
            "timestamp": datetime.now().isoformat(),
            "action": "tool_call",
            "details": f"{len(child_writes)} file write(s) by sub-agents: "
                       + ", ".join(
                           op.get("args", {}).get("path", "?") for op in child_writes
                       ),
            "tool_calls": [
                {
                    "name": op.get("tool", "write_file"),
                    "args": op.get("args", {}),
                }
                for op in child_writes
            ],
        }]

    return updates


def _responder_node(state: AgentState) -> dict:
    """Respond with synthesized answer or department results.

    Always clears ``subagent_results`` on the way out: the channel is part of
    the checkpointed state, so a leftover value from a previous turn would be
    replayed as this turn's answer.
    """
    subagent_results = state.get("subagent_results") or {}
    enhanced_query = state.get("enhanced_query", "")
    next_msg = state.get("next_message")

    if subagent_results:
        verdict = verify(subagent_results, {"original_query": enhanced_query})
        if not verdict["approved"]:
            logger.warning("Sub-agent results failed verification: %s", verdict["reasons"])
        content = aggregate(subagent_results, enhanced_query)
    elif next_msg is not None:
        # The dispatcher's normal reply is a JSON control envelope
        # ({enhanced_query, departments}), not an answer. Anything else — e.g.
        # the max-iterations notice — is a real message and is passed through.
        text = get_message_text(getattr(next_msg, "content", ""))
        if _is_envelope(text) or not text.strip():
            content = _fallback_answer(state, enhanced_query)
        else:
            content = text
    else:
        content = _fallback_answer(state, enhanced_query)

    return {
        "messages": [AIMessage(content=content)],
        "subagent_results": {},
        "next_message": None,
    }


def _is_envelope(text: str) -> bool:
    """True if *text* is the dispatcher's JSON control envelope."""
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}") and "departments" in stripped


def _fallback_answer(state: AgentState, enhanced_query: str) -> str:
    """Best-effort user-facing answer when no department handled the query."""
    if enhanced_query:
        return (
            f"I could not route this request to a department. "
            f"Query: {enhanced_query}"
        )
    last_human = next(
        (m for m in reversed(state.get("messages", []))
         if getattr(m, "type", None) == "human"),
        None,
    )
    if last_human is not None:
        return (
            "I could not route this request to a department. "
            f"Query: {get_message_text(last_human.content)}"
        )
    return "I could not produce an answer for this request."


def get_deep_agent():
    """Compile and return the deep agent LangGraph.

    Shape::

        START → orchestrator → subagent_fanout → responder → END
                                    ↘              → responder ↗

    Three nodes, one direction. There is no ``tools`` node: the orchestrator is
    a router and binds no tools, and the ``task`` tool that would have made a
    top-level tools node meaningful has been removed along with sub-agent
    delegation. Every tool call happens inside a sub-agent's own ReAct loop
    (``subagents.run_department``), which is bounded by that loop's iteration
    budget rather than by a graph-level retry counter.
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", _local_orchestrator_node)
    workflow.add_node("subagent_fanout", _subagent_fanout_node)
    workflow.add_node("responder", _responder_node)

    workflow.add_edge(START, "orchestrator")

    # Single conditional edge out of the router: a department list goes to
    # fanout, an empty one to the responder. There is no "tools" branch —
    # that was the orchestrator calling tools itself, which the hierarchy
    # does not allow.
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "subagent_fanout": "subagent_fanout",
            "responder": "responder",
        },
    )

    # Fanout delivers the synthesized answer directly. Looping back through the
    # orchestrator would spend a second router LLM call re-deciding a query that
    # has already been routed; the graph is acyclic by design.
    workflow.add_edge("subagent_fanout", "responder")

    workflow.add_edge("responder", END)

    # Share session_memory's saver so the checkpointer's history is the same
    # history the orchestrator reads via get_window().
    _compiled_graph = workflow.compile(checkpointer=get_saver())
    return _compiled_graph


def reset_deep_agent():
    """Invalidate the cached compiled graph, model clients, and MCP tools.

    Also rebuilds the sub-agent registry from ``skills/``. ``SUBAGENTS`` is a
    module-level dict built at import time, so a SKILL.md added after startup
    is invisible until this is called — this is the hook that makes the
    documented "edit skills, then reset" workflow real.
    """
    global _compiled_graph
    _compiled_graph = None
    from src.services.agent_orchestrator import agent_factory
    from src.services.agent_orchestrator.subagents import refresh_subagents

    agent_factory.clear_caches()
    refresh_subagents()
