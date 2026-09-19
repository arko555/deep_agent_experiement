"""Async A2A bridge: call remote sub-agents via Agent-to-Agent protocol."""

import asyncio
import logging
from typing import cast

import httpx

from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import GetTaskRequest, Role, SendMessageRequest, TaskState

from src.config import get_subagent_timeout_seconds

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}

INTERRUPTED_STATES = {
    TaskState.TASK_STATE_INPUT_REQUIRED,
    TaskState.TASK_STATE_AUTH_REQUIRED,
}

POLL_INTERVAL_SECONDS = 1.0
MAX_POLLS = 300


async def _call(url: str, description: str, timeout: float) -> str:
    """Internal async A2A call: resolve card, send, poll, return text."""
    async with httpx.AsyncClient(timeout=timeout) as http:
        card = await A2ACardResolver(httpx_client=http, base_url=url).get_agent_card()
        client = await create_client(
            agent=card,
            client_config=ClientConfig(streaming=True, httpx_client=http),
        )
        try:
            request = SendMessageRequest(
                message=new_text_message(description, role=Role.ROLE_USER)
            )

            answer_parts: list[str] = []
            direct_message: str | None = None
            task_id: str | None = None
            state: int | None = None

            async for chunk in client.send_message(request):
                if chunk.HasField("message"):
                    direct_message = get_message_text(chunk.message)
                elif chunk.HasField("artifact_update"):
                    text = get_artifact_text(chunk.artifact_update.artifact)
                    if text:
                        answer_parts.append(text)
                elif chunk.HasField("task"):
                    task_id = chunk.task.id
                    state = chunk.task.status.state
                    for artifact in chunk.task.artifacts:
                        text = get_artifact_text(artifact)
                        if text:
                            answer_parts.append(text)
                elif chunk.HasField("status_update"):
                    state = chunk.status_update.status.state
                    task_id = task_id or chunk.status_update.task_id

            if direct_message is not None:
                return direct_message

            polls = 0
            done = TERMINAL_STATES | INTERRUPTED_STATES
            while state not in done and task_id and polls < MAX_POLLS:
                polls += 1
                await asyncio.sleep(POLL_INTERVAL_SECONDS)
                task = await client.get_task(GetTaskRequest(id=task_id, history_length=0))
                state = task.status.state
                if state in TERMINAL_STATES:
                    answer_parts = [t for a in task.artifacts if (t := get_artifact_text(a))]

            if state == TaskState.TASK_STATE_FAILED:
                raise RuntimeError("remote agent reported TASK_STATE_FAILED")
            if state in INTERRUPTED_STATES:
                raise RuntimeError(
                    f"remote agent stopped for more input (state={state})"
                )
            if state in (TaskState.TASK_STATE_CANCELED, TaskState.TASK_STATE_REJECTED):
                raise RuntimeError(f"remote agent did not complete (state={state})")

            text = "\n".join(part for part in answer_parts if part).strip()
            if not text:
                raise RuntimeError("remote agent returned no text result")
            return text
        finally:
            await client.close()


async def call_a2a_agent_async(
    url: str, description: str, trace_id: str | None = None, timeout: float | None = None
) -> str:
    """Invoke a remote A2A agent asynchronously.

    Blocks until the remote agent produces a result or *timeout*
    elapses. Raises on transport failure or a non-successful
    terminal state.
    """
    if timeout is None:
        timeout = get_subagent_timeout_seconds()
    if trace_id is not None:
        logger.info("A2A call trace_id=%s to %s", trace_id, url)
    try:
        return cast("str", await _call(url, description, timeout))
    except Exception as e:
        raise RuntimeError(f"A2A agent at {url} failed: {e}") from e


def call_a2a_agent(
    url: str, description: str, trace_id: str | None = None, timeout: float | None = None
) -> str:
    """Sync entry point: call an A2A agent via the async bridge.

    Mirrors the interface used by ``tools._execute_task``.
    """
    import asyncio

    return asyncio.run(call_a2a_agent_async(url, description, trace_id, timeout))
