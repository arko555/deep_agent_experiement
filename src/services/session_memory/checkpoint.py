"""Checkpointing: MemorySaver wrapper, thread management, session retrieval."""

import uuid

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver


_saver = MemorySaver()
# In-memory index of the latest checkpoint ID per thread, used by get_session
# and get_turn_token_usage to avoid relying on list() ordering.
_latest: dict[str, str] = {}


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def get_session(thread_id: str) -> list[BaseMessage]:
    """Retrieve the conversation history for a thread."""
    checkpoint_id = _latest.get(thread_id)
    if checkpoint_id is None:
        # Fall back to list() for threads created before this index existed.
        results = list(_saver.list(_config(thread_id)))
        if not results:
            return []
        checkpoint_id = results[0].checkpoint.get("id")

    config = _config(thread_id)
    results = list(_saver.list(config))
    for r in results:
        if r.checkpoint.get("id") == checkpoint_id:
            channel_values = r.checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])
            if isinstance(messages, list):
                return list(messages)
            return []
    return []


def append_message(thread_id: str, message: BaseMessage) -> None:
    """Append a message to a thread's history."""
    config = _config(thread_id)
    existing = get_session(thread_id)
    existing.append(message)
    checkpoint_id = uuid.uuid4().hex[:8]
    checkpoint = {
        "channel_values": {"messages": list(existing)},
        "channel_versions": {"messages": checkpoint_id},
        "id": checkpoint_id,
    }
    new_versions = {"messages": checkpoint_id}
    metadata = {
        "session_id": thread_id,
        "ts": "2024-01-01T00:00:00",
        "thread_id": thread_id,
        "parent_config": None,
    }
    _saver.put(config, checkpoint, metadata, new_versions)
    _latest[thread_id] = checkpoint_id


def get_turn_token_usage(thread_id: str):
    """Get token usage for the latest turn in a thread."""
    checkpoint_id = _latest.get(thread_id)
    if checkpoint_id is None:
        results = list(_saver.list(_config(thread_id)))
        if not results:
            return {}
        checkpoint_id = results[0].checkpoint.get("id")

    config = _config(thread_id)
    results = list(_saver.list(config))
    for r in results:
        if r.checkpoint.get("id") == checkpoint_id:
            metadata = r.metadata
            if isinstance(metadata, dict):
                return metadata.get("token_usage", {})
            return {}
    return {}
