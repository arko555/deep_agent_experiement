"""LangGraph compilation for the deep agent.

    START → orchestrator → tools → orchestrator   (ReAct loop)
                        → subagent_fanout → responder → END
                        → responder → END
"""

import asyncio
import logging

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

# Consecutive unknown-tool calls tolerated before the ReAct loop gives up and
# lets the responder answer. A model that has been told "no tool named X" and
# asks again is not going to recover on its own; the ToolMessages already name
# the valid tools, so the responder can summarise without one.
MAX_INVALID_TOOL_RETRIES = 3


def _local_orchestrator_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Orchestrator node: enhance query, identify departments, call tools.

    The thread id lives in the runtime config, not in state, so it is read
    from there — otherwise the dispatcher sees an empty message window.

    The tool set comes from ``tools_integration.get_all_tools()``: the built-ins
    plus every ``@tool`` function ``load_dynamic_tools`` discovers under
    ``./tools`` (plus MCP and A2A tools when configured). Those imports are
    function-local because ``agent_factory`` imports this module at load time.
    """
    from src.services.agent_orchestrator.agent_factory import get_model
    from src.services.tools_integration.tools import get_all_tools

    thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
    return call_orchestrator(
        state,
        model=get_model(),
        thread_id=thread_id,
        tools=list(get_all_tools().values()),
    )


def _subagent_fanout_node(state: AgentState) -> dict:
    """Fan out to department sub-agents in parallel."""
    departments = state.get("department_targets", [])
    enhanced_query = state.get("enhanced_query", "")
    engine = SubAgentEngine()
    subagents = [
        {
            "name": dept,
            "system_prompt": f"You are the {dept} specialist. Address the query concisely.",
            "description": enhanced_query,
        }
        for dept in departments
    ]
    results = asyncio.run(engine.invoke_parallel(subagents, enhanced_query))
    return {"subagent_results": results}


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


def _tools_node(state: AgentState) -> dict:
    """Execute the tool calls the dispatcher requested.

    A thin adapter over ``agent_factory.local_tools_node``. The import is
    function-local because ``agent_factory`` imports this module at load time.
    """
    from src.services.agent_orchestrator.agent_factory import local_tools_node

    return local_tools_node(state)


def _route_after_tools(state: AgentState):
    """After tools run: back to the dispatcher for another turn, or finish.

    The dispatcher is re-entered so it can either call another tool or settle on
    an answer. Two guards stop a tool-calling model from looping forever:

    - ``consecutive_invalid_tools`` — the model asked for tools that don't
      exist. Retrying the same impossible call is futile, so after a few
      misses we let the responder answer from what we do have.
    - The iteration budget checked in ``call_orchestrator``.
    """
    from src.services.agent_orchestrator.orchestrator import _is_budget_exhausted

    if _is_budget_exhausted(state):
        return "responder"
    if state.get("consecutive_invalid_tools", 0) >= MAX_INVALID_TOOL_RETRIES:
        logger.warning(
            "Model requested unknown tools %d times in a row; ending the tool loop",
            state.get("consecutive_invalid_tools", 0),
        )
        return "responder"
    return "orchestrator"



def get_deep_agent():
    """Compile and return the deep agent LangGraph.

    Shape::

        START → orchestrator → tools → orchestrator   (ReAct loop)
                            → subagent_fanout → responder → END
                            → responder → END
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator", _local_orchestrator_node)
    workflow.add_node("tools", _tools_node)
    workflow.add_node("subagent_fanout", _subagent_fanout_node)
    workflow.add_node("responder", _responder_node)

    workflow.add_edge(START, "orchestrator")

    # Single conditional edge out of the dispatcher. The tools branch is what
    # makes write_file / internet_search / task / MCP / A2A reachable from the
    # application: the model emits a tool call, this edge sends the turn to the
    # tools node, and the results come back as ToolMessages.
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "tools": "tools",
            "subagent_fanout": "subagent_fanout",
            "responder": "responder",
        },
    )

    # Fanout delivers the synthesized answer directly. Looping back through the
    # orchestrator would spend a second dispatcher LLM call whose departments
    # are discarded by call_orchestrator (it forces [] once iteration_count > 0).
    workflow.add_edge("subagent_fanout", "responder")

    # ReAct loop: tool results go back to the dispatcher so it can chain calls
    # or finish. Bounded by the iteration budget, not by a hand-set hop cap.
    workflow.add_conditional_edges(
        "tools",
        _route_after_tools,
        {
            "orchestrator": "orchestrator",
            "responder": "responder",
        },
    )

    workflow.add_edge("responder", END)

    # Share session_memory's saver so the checkpointer's history is the same
    # history the orchestrator reads via get_window().
    _compiled_graph = workflow.compile(checkpointer=get_saver())
    return _compiled_graph


def reset_deep_agent():
    """Invalidate the cached compiled graph, model clients, and MCP tools."""
    global _compiled_graph
    _compiled_graph = None
    from src.services.agent_orchestrator import agent_factory

    agent_factory.clear_caches()
