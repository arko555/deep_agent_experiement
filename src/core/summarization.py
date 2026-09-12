"""Conversation summarization for long message histories.

When the message list exceeds ``max_history_messages``, old messages are
compressed into a concise summary and prepended as a system/context message
instead of being silently truncated. This preserves context that would
otherwise be lost.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from src.core.utils import get_message_text

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Default threshold: summarize when history grows beyond this many messages.
DEFAULT_MAX_MESSAGES = 20

# Keep at least this many recent messages intact (the summary replaces older ones).
MIN_KEEP_RECENT = 8


def _extract_role_label(msg: BaseMessage) -> str:
    """Return a short role label for a message type."""
    if isinstance(msg, HumanMessage):
        return "User"
    if isinstance(msg, AIMessage):
        return "Agent"
    if isinstance(msg, ToolMessage):
        return f"Tool({msg.name})"
    if isinstance(msg, SystemMessage):
        return "System"
    return type(msg).__name__


def compress_messages(
    messages: Sequence[BaseMessage],
    max_history: int = DEFAULT_MAX_MESSAGES,
    min_keep: int = MIN_KEEP_RECENT,
) -> list[BaseMessage]:
    """Return the message list, compressing old messages when it exceeds ``max_history``.

    Strategy:
    1. Keep the last ``min_keep`` messages intact (most recent context).
    2. Compress everything before those into a single summary HumanMessage.
    3. Prepend the summary so the full list is at most ``max_history + 1`` items.

    This avoids silent truncation — the agent sees a digest of what happened
    earlier in the conversation.
    """
    if len(messages) <= max_history:
        return list(messages)

    # Messages to keep intact (most recent). Clamped so the returned list
    # stays within max_history + 1 items (summary included) and there is
    # always at least one old message to summarize.
    keep_count = min(max(min_keep, max_history // 2), max(0, max_history - 1))
    recent = list(messages[-keep_count:])

    # Messages to summarize (oldest).
    old = list(messages[:-keep_count])

    summary = _summarize_old_messages(old)

    logger.info(
        "Compressed %d old messages into summary (%.1f chars); keeping %d recent.",
        len(old),
        len(summary.content) if isinstance(summary.content, str) else 0,
        keep_count,
    )

    return [summary] + recent


def _summarize_old_messages(messages: list[BaseMessage]) -> HumanMessage:
    """Create a digest of older messages for context preservation."""
    parts: list[str] = []

    ai_count = 0
    human_count = 0
    tool_count = 0
    tool_errors: list[str] = []

    for msg in messages:
        role = _extract_role_label(msg)
        text = get_message_text(getattr(msg, "content", ""))[:150].strip()

        if isinstance(msg, AIMessage):
            ai_count += 1
            if text:
                parts.append(f"Agent: {text}")
        elif isinstance(msg, HumanMessage):
            human_count += 1
            if text:
                parts.append(f"User: {text}")
        elif isinstance(msg, ToolMessage):
            tool_count += 1
            if "Error" in text or "error" in text.lower():
                tool_errors.append(f"{msg.name}: {text[:80]}")
            elif text:
                parts.append(f"[{msg.name}]: {text[:100]}")

    summary_lines = [
        "=== Earlier Conversation Summary ===",
        f"Exchanged {len(messages)} messages ({ai_count} agent, {human_count} user, {tool_count} tool results).",
    ]

    if tool_errors:
        summary_lines.append("Errors encountered:")
        for err in tool_errors[:5]:
            summary_lines.append(f"  - {err}")

    # Include key content snippets.
    if parts:
        summary_lines.append("Key exchanges:")
        for part in parts[:10]:
            summary_lines.append(f"  {part}")
        if len(parts) > 10:
            summary_lines.append(f"  ... ({len(parts) - 10} more exchanges omitted)")

    return HumanMessage(
        content="\n".join(summary_lines)
        + "\n=== End Summary ===\n(The conversation continues below.)"
    )
