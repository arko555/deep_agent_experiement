"""Message window delivery: fetch compressed window, append messages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.services.session_memory.checkpoint import get_session
from src.services.session_memory.compression import compress_messages

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


DEFAULT_MAX_MESSAGES = 20


def get_window(thread_id: str, max_messages: int = DEFAULT_MAX_MESSAGES) -> list[BaseMessage]:
    """Get a compressed message window for a thread."""
    messages = get_session(thread_id)
    if len(messages) <= max_messages:
        return list(messages)
    return compress_messages(messages, max_history=max_messages)


def append_message(thread_id: str, message: BaseMessage) -> None:
    """Append a message to a thread's history."""
    from src.services.session_memory.checkpoint import append_message as _append

    _append(thread_id, message)
