"""Subagent executors for the `task` tool.

Research and writer subagents run a minimal tool loop (model + bound tools)
so they can actually search, read, and write files instead of answering in a
single completion. The general-purpose subagent reuses the full compiled
graph and is handled in ``tools._execute_task``.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.utils import get_message_text, invoke_with_retry


def _extract_usage(response):
    """Return (input_tokens, output_tokens) from a chat response's usage_metadata."""
    meta = getattr(response, "usage_metadata", None)
    if not isinstance(meta, dict):
        return 0, 0
    return (
        meta.get("input_tokens", 0) or 0,
        meta.get("output_tokens", 0) or 0,
    )


def run_tool_loop(system_prompt: str, description: str, tools: list, max_iterations: int = 10):
    """Run a minimal ReAct-style loop: model with bound tools until it answers
    without tool calls (or the iteration budget is exhausted).

    Args:
        system_prompt: Role-specific system prompt for the subagent.
        description: The task description from the parent orchestrator.
        tools: Restricted list of LangChain tools this subagent may use.
        max_iterations: Maximum model turns before giving up.

    Returns:
        (final_text, usage, write_ops) — the subagent's final answer, its
        total token usage across all loop turns, and the file-write
        operations it performed (for the parent's audit trail).
    """
    # Lazy import to avoid a circular dependency at module load time.
    from src.core.agent_factory import get_model

    model = get_model()
    model_with_tools = model.bind_tools(tools) if tools else model
    tool_map = {t.name: t for t in tools}

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=description)]
    usage = {"input": 0, "output": 0}
    write_ops = []

    for _ in range(max_iterations):
        response = invoke_with_retry(model_with_tools, messages)
        in_tokens, out_tokens = _extract_usage(response)
        usage["input"] += in_tokens
        usage["output"] += out_tokens
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return get_message_text(response.content), usage, write_ops

        for tool_call in response.tool_calls:
            # Track file writes so the parent's pending_writes audit trail
            # covers subagent activity, not just top-level tools.
            if tool_call["name"] in ("write_file", "edit_file"):
                write_ops.append({
                    "tool": tool_call["name"],
                    "tool_id": None,
                    "args": tool_call["args"],
                    "status": "executed",
                })
            tool = tool_map.get(tool_call["name"])
            if tool is None:
                result = f"Tool '{tool_call['name']}' not found."
            else:
                try:
                    result = invoke_with_retry(tool, tool_call["args"])
                except Exception as e:
                    result = f"Error executing tool {tool_call['name']}: {e}"
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            ))

    # Budget exhausted — return the last model turn if it has any text.
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    fallback = (
        get_message_text(last_ai.content)
        if last_ai is not None
        else "Error: subagent reached max iterations without a final answer."
    )
    return fallback, usage, write_ops
