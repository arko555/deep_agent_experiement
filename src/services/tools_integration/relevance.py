"""Tool relevance sorting: LLM picks 3-5 relevant tools from 20 visible."""

import json
import logging
from typing import Any

from src.utils import invoke_with_retry

logger = logging.getLogger(__name__)


def sort_tools(
    model,
    enhanced_query: str,
    tool_defs: list[dict[str, Any]],
    max_tools: int = 5,
) -> list[dict[str, Any]]:
    """Rank tools by relevance to an enhanced query.

    The LLM receives the enhanced query plus all visible tool
    schemas (up to 20) and returns at most *max_tools* names,
    ranked by relevance. Each returned entry is the original
    tool definition dict from *tool_defs*.

    Args:
        model: An LLM client with a synchronous ``invoke(messages)``
            method (e.g. a ChatAnthropic/ChatOpenAI instance or a
            test double).
        enhanced_query: The context-enriched query to score against.
        tool_defs: Tool schema dicts from
            ``ToolRegistry.get_tool_definitions()``.
        max_tools: Maximum tools to return (default 5).

    Returns:
        Up to *max_tools* tool definition dicts, ordered most
        relevant first.  Returns all tools if there are fewer than
        *max_tools*.
    """
    if not tool_defs:
        return []

    tools_json = json.dumps(
        [{k: v for k, v in d.items() if k != "args_schema"} for d in tool_defs],
        indent=2,
    )

    system_prompt = (
        "You are a tool relevance filter. Given the user's query below, "
        "return a JSON list of the names of the most relevant tools "
        f"(at most {max_tools}), ordered by relevance. "
        "Return ONLY a JSON array of tool name strings."
    )

    user_prompt = (
        f"Query: {enhanced_query}\n\n"
        f"Available tools ({len(tool_defs)}):\n{tools_json}\n\n"
        "Return a JSON array of the most relevant tool names."
    )

    try:
        response = invoke_with_retry(model, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        logger.warning("Tool relevance sort failed: %s", e)
        return tool_defs[:max_tools]

    selected_names = _extract_tool_names(response, max_tools)
    if not selected_names:
        return tool_defs[:max_tools]

    # Preserve the LLM's ranking order, but only include names we have.
    by_name = {d["name"]: d for d in tool_defs}
    ranked = [by_name[name] for name in selected_names if name in by_name]
    return ranked


def _extract_tool_names(response: Any, max_tools: int) -> list[str]:
    """Parse tool names from an LLM response."""
    from langchain_core.messages import AIMessage

    if isinstance(response, AIMessage):
        content = response.content
    elif isinstance(response, dict):
        content = response.get("content", "")
    elif isinstance(response, str):
        content = response
    else:
        content = str(response)

    text = content if isinstance(content, str) else ""

    # Try to extract a JSON array from the response.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, list):
                return [str(item) for item in parsed][:max_tools]
        except (json.JSONDecodeError, TypeError):
            pass

    return []
