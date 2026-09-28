"""Orchestrator node: enhance query, identify departments, route, verify, synthesize."""

import json
import logging

from langchain_core.messages import AIMessage, SystemMessage

from src.services.agent_orchestrator.state import AgentState
from src.config import get_max_iterations
from src.utils import invoke_with_retry, get_message_text
from src.services.session_memory.window import get_window as _get_window

logger = logging.getLogger(__name__)

def _build_dispatcher_prompt() -> str:
    """Build the dispatcher prompt with the real department roster.

    The prompt used to claim departments were "described in the available
    tool registry" — they are not, the registry holds tools only — so the
    model had no way to know a department existed and the fanout branch never
    fired. The roster now comes from ``skills/`` via ``list_departments()``,
    which means a new SKILL.md appears here with no code change.

    Departments are reachable two ways and the prompt has to separate them, or
    one shadows the other: the ``task`` tool delegates to a *single* sub-agent
    and returns its answer inline, while the JSON envelope fans out to
    *several* departments in parallel and comes back as context. Without that
    split, "prefer a tool" made the model call ``task`` for every departmental
    query and the parallel fanout branch went unused.
    """
    from src.services.agent_orchestrator.subagents import list_departments

    departments = list_departments()
    if departments:
        roster = "\n".join(
            f"- **{d['name']}**: {d['description']}" for d in departments
        )
        department_block = (
            "These departments each have their own tools and run independently:\n"
            f"{roster}\n\n"
            "To hand work to one or more of them, return JSON: "
            "{enhanced_query, departments: [...]}, listing only names from the "
            "list above. Listing two or more runs them in parallel — use that "
            "when a query has genuinely separate parts (e.g. research a topic "
            "and draft a post about it). Use an empty list when no department "
            "is a better fit than answering directly."
        )
    else:
        department_block = (
            "No departments are currently registered, so always return an empty "
            "departments list and answer the user yourself."
        )

    return (
        "You are the dispatcher. Given the conversation history and user query, "
        "enhance the query, then either route it to departments or answer it "
        "yourself. You are the only actor who talks to the user, so department "
        "output arrives as context for you to synthesize into one final answer.\n\n"
        f"{department_block}\n\n"
        "You also have tools. When the query needs one concrete action — reading "
        "or writing a file, searching the web, getting the current time, text "
        "statistics — CALL the tool instead of describing what you would do. The "
        "`task` tool delegates a *single* sub-task to one subagent and returns "
        "its answer to you; reach for it when one subagent does the whole job. "
        "Tool results come back to you, and you then answer the user in plain "
        "language. Emit the JSON envelope instead when no single tool call "
        "covers the request."
    )


# Built per call rather than at import so a SKILL.md added at runtime is
# reflected. The module-level name is kept for callers that import it.
DISPATCHER_SYSTEM_PROMPT = _build_dispatcher_prompt()


def _is_budget_exhausted(state: AgentState) -> bool:
    """True when this turn has spent its orchestrator iteration budget."""
    max_iterations = state.get("max_iterations") or get_max_iterations()
    return state.get("iteration_count", 0) >= max_iterations


def call_orchestrator(
    state: AgentState,
    model,
    thread_id: str | None = None,
    max_history_messages: int = 20,
    tools: list | None = None,
) -> dict:
    """Dispatcher orchestrator: enhance query, identify departments.

    This is the Phase 3 orchestrator. It differs from the legacy
    orchestrator in ``plan.py`` in that it:

    - Gets the message window from session_memory (not from state messages).
    - Uses a dispatcher system prompt (no AGENTS.md).
    - Returns structured output: {enhanced_query, departments: [...]}.
    - Routes to subagent_fanout if departments are found, or responder
      if none are detected.

    Args:
        state: The current AgentState.
        model: The LLM instance to invoke.
        thread_id: Conversation thread id, taken from the graph's runtime
            config by the caller. Falls back to the state, then ``"default"``.
        max_history_messages: Maximum number of conversation messages to keep.
        tools: LangChain tool objects bound to the model for this turn. These
            come from ``tools_integration`` — built-ins plus everything
            ``load_dynamic_tools`` discovers under ``./tools``. When the model
            calls one, the turn routes to the tools node instead of the
            fanout/responder.

    Returns:
        A dict updating state:
        - ``next_message``: An AIMessage with the dispatcher's JSON output
          ({enhanced_query, departments}).
        - ``iteration_count``: Incremented by 1.
        - ``enhanced_query``: The LLM-enhanced version of the query.
        - ``department_targets``: List of department names (empty → responder).
    """
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations") or get_max_iterations()
    if iteration_count >= max_iterations:
        error_msg = AIMessage(content="Maximum iterations reached. Stopping to prevent runaway execution.")
        return {"next_message": error_msg, "iteration_count": iteration_count,
                "enhanced_query": "", "department_targets": []}

    # Get message window from session_memory (Phase 3 requirement). The graph
    # checkpointer and session_memory share one saver, so this returns the real
    # conversation — including the current turn's user message.
    resolved_thread = thread_id or state.get("thread_id") or "default"
    messages = _get_window(resolved_thread, max_messages=max_history_messages)

    # Built per call so a department added to skills/ mid-process is listed.
    formatted_messages = [
        SystemMessage(content=_build_dispatcher_prompt())
    ] + list(messages)

    # Bind the tool set so the dispatcher can actually call a tool. The tools
    # node reads ``state["messages"][-1]``, so a tool-calling response must also
    # be appended to the message log below — otherwise the tools node sees the
    # user's HumanMessage and finds no ``tool_calls`` on it.
    tool_calling_model = model.bind_tools(tools) if tools else model
    response = invoke_with_retry(tool_calling_model, formatted_messages)

    # Parse the LLM response to extract enhanced_query and departments.
    enhanced_query = ""
    departments: list[str] = []

    content = get_message_text(response.content) if response.content else ""
    parsed = _parse_dispatcher_output(content)
    if parsed:
        enhanced_query = parsed.get("enhanced_query", content)
        departments = _validate_departments(parsed.get("departments", []))

    logger.info(
        "Orchestrator: enhanced_query=%r, departments=%r",
        enhanced_query[:100] if enhanced_query else content[:100],
        departments,
    )

    updates = {
        "next_message": response,
        "iteration_count": iteration_count + 1,
        "enhanced_query": (enhanced_query or content) if iteration_count == 0 else state.get("enhanced_query", enhanced_query or content),
        "department_targets": departments if iteration_count == 0 else [],
    }

    # A tool-calling response must land in the message log: the tools node
    # reads messages[-1] to find the calls. A pure-envelope response must NOT,
    # because the responder appends the user-facing answer and the raw JSON
    # envelope would pollute history.
    if getattr(response, "tool_calls", None):
        updates["messages"] = [response]

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
    Dropping it lets the responder answer directly.
    """
    from src.services.agent_orchestrator.subagents import list_departments

    valid = {d["name"] for d in list_departments()}
    kept: list[str] = []
    for name in departments or []:
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
