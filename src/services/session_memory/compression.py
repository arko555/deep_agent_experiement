"""Message compression: summarize old turns while preserving tool-call boundaries."""

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

from src.utils import get_message_text

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_MAX_MESSAGES = 20
MIN_KEEP_RECENT = 8


def _extract_role_label(msg: BaseMessage) -> str:
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
    if max_history <= 0:
        raise ValueError("max_history must be positive")
    if min_keep < 0:
        raise ValueError("min_keep must be nonnegative")
    if len(messages) <= max_history:
        return list(messages)

    keep_count = min(max(min_keep, max_history // 2), max_history - 1)
    boundary = len(messages) - keep_count
    if keep_count:
        safe_boundary = 0
        pending: set[str] = set()
        for index, msg in enumerate(messages[:boundary]):
            if isinstance(msg, AIMessage):
                pending.update(
                    call["id"] for call in msg.tool_calls if call["id"] is not None
                )
            elif isinstance(msg, ToolMessage):
                pending.discard(msg.tool_call_id)
            if not pending:
                safe_boundary = index + 1
        boundary = safe_boundary

    if boundary == 0:
        return list(messages)

    recent = list(messages[boundary:])
    old = list(messages[:boundary])

    summary = _summarize_old_messages(old)

    logger.info(
        "Compressed %d old messages into summary (%.1f chars); keeping %d recent.",
        len(old),
        len(summary.content) if isinstance(summary.content, str) else 0,
        len(recent),
    )

    return [summary, *recent]


def _summarize_old_messages(messages: list[BaseMessage]) -> HumanMessage:
    parts: list[str] = []

    ai_count = 0
    human_count = 0
    tool_count = 0
    tool_errors: list[str] = []

    for msg in messages:
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
        f"Exchanged {len(messages)} messages ({ai_count} agent, "
        f"{human_count} user, {tool_count} tool results).",
    ]

    if tool_errors:
        summary_lines.append("Errors encountered:")
        for err in tool_errors[:5]:
            summary_lines.append(f"  - {err}")

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
