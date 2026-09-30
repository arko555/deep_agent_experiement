"""Orchestrator node: enhance query, identify departments, route, verify, synthesize."""

import json
import logging

from langchain_core.messages import SystemMessage

from src.services.agent_orchestrator.state import AgentState
from src.utils import invoke_with_retry, get_message_text
from src.services.session_memory.window import get_window as _get_window

logger = logging.getLogger(__name__)

# The router's default destination. `skills/general/SKILL.md` is the catch-all:
# it handles greetings, politely refuses off-topic queries, and does the work
# itself when no specialist department claims the request. Named here because
# `call_orchestrator` falls back to it whenever the model returns nothing
# usable — without a floor, an unparseable reply left the turn with no
# department and the user got "I could not route this request".
_FALLBACK_DEPARTMENT = "general"

def _build_dispatcher_prompt() -> str:
    """Build the router prompt from the real department roster.

    The dispatcher is a *router*. It has no tools and executes nothing: it
    enhances the query and names the departments that should handle it, and
    the departments do the tool work. That is what keeps communication
    one-way — orchestrator → sub-agent → tool.

    The roster is rendered from ``skills/`` via ``list_departments()`` on every
    call, so a new SKILL.md appears here with no code change. It matters more
    than it used to: the description under each name is the only signal the
    router has for choosing, and a wrong pick sends the query to a department
    that will refuse it.
    """
    from src.services.agent_orchestrator.subagents import list_departments

    departments = list_departments()
    if departments:
        roster = "\n".join(
            f"- **{d['name']}**: {d['description']}" for d in departments
        )
        department_block = (
            "The departments available to handle the query:\n"
            f"{roster}\n\n"
            "Choose by matching the query against each department's description. "
            "Route to the department whose remit clearly covers the query.\n"
            "- If exactly one department fits, name just that one.\n"
            "- If a query has genuinely separate parts that two departments own "
            "(for example, research a topic and then write a post about it), "
            "name both — they will run in parallel.\n"
            f"- If no department's description fits, name '{_FALLBACK_DEPARTMENT}'. "
            "Do not invent a department name; only the names listed above are valid."
        )
    else:
        department_block = (
            "No departments are registered, so return an empty departments list."
        )

    return (
        "You are the router. You do not answer the user and you do not use any "
        "tools — the department you pick does that, and the result comes back to "
        "you to pass along.\n\n"
        "Given the conversation history and the user's latest message, return "
        "JSON of the form {enhanced_query, departments}, where:\n"
        "- enhanced_query: the user's request restated with enough context from "
        "the conversation to be actionable on its own.\n"
        "- departments: the list of department names that should handle it.\n\n"
        f"{department_block}\n\n"
        "Return only the JSON object, with no surrounding prose and no markdown "
        "fence."
    )


def call_orchestrator(
    state: AgentState,
    model,
    thread_id: str | None = None,
    max_history_messages: int = 20,
) -> dict:
    """Router: build the enhanced query and name the departments for it.

    Exactly two jobs, both done in one model call:

    1. Restate the user's request with enough of the conversation history
       folded in to be actionable on its own — that is ``enhanced_query``.
    2. Name the departments that should handle it, from the roster in the
       dispatcher system prompt.

    There is no iteration budget here, and there is no loop to bound. The
    router runs once per turn; the graph's own topology is acyclic
    (orchestrator → fanout → responder), so a second router call would be a
    second decision about a query already decided. Iteration counting lives
    where iteration actually happens: inside a sub-agent's ReAct tool loop
    (``subagents.run_tool_loop``), which is bounded by its own turn budget.

    This node previously kept ``iteration_count`` in the checkpointed state
    and forced ``departments = []`` once it exceeded a maximum. That guard
    belonged to the removed top-level ReAct dispatcher loop, and because the
    counter was checkpointed it leaked across turns on a reused thread: the
    second turn of any conversation saw a non-zero count, dropped its
    departments, and answered "I could not route this request" — quoting the
    *first* turn's query. Both entry points were masking it by passing
    ``iteration_count: 0`` on every invoke. The counter is gone from the
    state schema entirely rather than reset, so nothing has to compensate.

    Args:
        state: The current AgentState.
        model: The LLM instance to invoke.
        thread_id: Conversation thread id, taken from the graph's runtime
            config by the caller. Falls back to the state, then ``"default"``.
        max_history_messages: Maximum number of conversation messages to keep.

    The ``tools`` parameter this used to accept is gone. The dispatcher never
    bound tools, so every caller that passed a tool set was paying to load the
    dynamic and MCP tool layers for a value that was then discarded.

    Returns:
        A dict updating state:
        - ``next_message``: An AIMessage with the dispatcher's JSON output
          ({enhanced_query, departments}).
        - ``enhanced_query``: The LLM-enhanced version of the query.
        - ``department_targets``: Department names. Never empty — an
          unusable router response falls back to ``general``.
    """
    # Get message window from session_memory (Phase 3 requirement). The graph
    # checkpointer and session_memory share one saver, so this returns the real
    # conversation — including the current turn's user message. This history
    # is what the enhanced query is built from.
    resolved_thread = thread_id or state.get("thread_id") or "default"
    messages = _get_window(resolved_thread, max_messages=max_history_messages)

    # Built per call so a department added to skills/ mid-process is listed.
    formatted_messages = [
        SystemMessage(content=_build_dispatcher_prompt())
    ] + list(messages)

    # No bind_tools. The dispatcher routes; it does not execute. Tool calls
    # happen inside sub-agents, which is what keeps the hierarchy one-way:
    # orchestrator -> sub-agent -> tool.
    response = invoke_with_retry(model, formatted_messages)

    # Parse the LLM response to extract enhanced_query and departments.
    enhanced_query = ""
    departments: list[str] = []

    content = get_message_text(response.content) if response.content else ""
    parsed = _parse_dispatcher_output(content)
    if parsed:
        enhanced_query = parsed.get("enhanced_query", content)
        departments = _validate_departments(parsed.get("departments", []))

    # The router must always land somewhere. An empty or wholly-invented
    # department list would fall through to the responder, which has no tools
    # of its own and can only tell the user it could not route the request.
    if not departments:
        departments = [_FALLBACK_DEPARTMENT]
        logger.info(
            "Router returned no usable department; defaulting to %r", _FALLBACK_DEPARTMENT
        )

    logger.info(
        "Orchestrator: enhanced_query=%r, departments=%r",
        enhanced_query[:100] if enhanced_query else content[:100],
        departments,
    )

    updates = {
        "next_message": response,
        "enhanced_query": enhanced_query or content,
        "department_targets": departments,
    }

    try:
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0) or usage.get("input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0) or usage.get("output_tokens", 0)
            current_usage = state.get("token_usage", {})
            total_in = current_usage.get("input", 0) + input_tokens
            total_out = current_usage.get("output", 0) + output_tokens
            updates["token_usage"] = {"input": total_in, "output": total_out, "total": total_in + total_out}
    except Exception:
        pass

    return updates


def _validate_departments(departments: list) -> list[str]:
    """Keep only department names that resolve to a registered department.

    The model can invent a name. Without this check an unknown department
    reaches ``subagent_fanout``, where it resolves to nothing and the turn
    produces a confusing "unknown department" error instead of an answer.

    An empty result is legitimate here — ``call_orchestrator`` applies the
    ``general`` fallback afterwards.
    """
    from src.services.agent_orchestrator.subagents import list_departments

    valid = {d["name"] for d in list_departments()}
    kept: list[str] = []
    # A non-list (the model returned a bare string, or a nested structure)
    # is not worth guessing at — the caller falls back to `general`.
    if not isinstance(departments, list):
        if departments:
            logger.warning(
                "Dispatcher returned departments of type %s, expected list; ignoring.",
                type(departments).__name__,
            )
        return kept
    for name in departments:
        if not isinstance(name, str):
            continue
        cleaned = name.strip()
        if cleaned in valid:
            if cleaned not in kept:
                kept.append(cleaned)
        else:
            logger.warning(
                "Dispatcher named unknown department %r (known: %s); dropping it.",
                cleaned, ", ".join(sorted(valid)) or "none",
            )
    return kept


def _parse_dispatcher_output(content: str) -> dict | None:
    """Extract JSON {enhanced_query, departments} from dispatcher output."""
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(content[start:end + 1])
        if isinstance(parsed, dict):
            if "departments" in parsed and isinstance(parsed["departments"], list):
                return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None
