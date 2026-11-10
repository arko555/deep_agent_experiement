"""Call remote A2A agents and return their answer as text (Phase 10).

The `a2a-sdk` client is async-only, so calls are driven through the sync
bridge. This module is transport-agnostic: it resolves the remote agent's card
from its base URL, sends one text message, and reduces the response to a plain
string for the parent orchestrator.

A2A 1.x uses protobuf types (``Role.ROLE_USER``, ``Part(text=...)``), not the
Pydantic/``TextPart`` shapes from 0.3-era tutorials.
"""

import asyncio
import logging
from typing import cast

import httpx

from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import GetTaskRequest, Role, SendMessageRequest, TaskState

from src.async_bridge import run_sync
from src.config import get_subagent_timeout_seconds

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_REJECTED,
}

# Non-terminal, but the agent will not make progress until a new message
# arrives — polling these would spin forever.
INTERRUPTED_STATES = {
    TaskState.TASK_STATE_INPUT_REQUIRED,
    TaskState.TASK_STATE_AUTH_REQUIRED,
}

POLL_INTERVAL_SECONDS = 1.0
MAX_POLLS = 300  # ~5 minutes at the default interval


def _build_message(description: str, trace_id: str | None = None):
    """Build the outgoing user message, tagging it with the trace id.

    A2A carries a free-form metadata struct; an agent that reads it can
    correlate its logs with this turn. It is advisory: an agent that ignores
    it still works.
    """
    message = new_text_message(description, role=Role.ROLE_USER)
    if trace_id:
        message.metadata.update({"trace_id": str(trace_id)})
    return message


async def _call(
    url: str, description: str, timeout: float, trace_id: str | None = None
) -> str:
    # One HTTP client per call, created and closed inside this single
    # coroutine: httpx binds its pool to the loop that created it, so a client
    # must never outlive or cross loops.
    async with httpx.AsyncClient(timeout=timeout) as http:
        card = await A2ACardResolver(httpx_client=http, base_url=url).get_agent_card()
        client = await create_client(
            agent=card,
            client_config=ClientConfig(streaming=True, httpx_client=http),
        )
        try:
            request = SendMessageRequest(
                message=_build_message(description, trace_id)
            )

            answer_parts: list[str] = []
            direct_message: str | None = None
            task_id: str | None = None
            state: int | None = None

            # A stream is either a single Message (immediate answer) or a task
            # lifecycle (Task, then status/artifact updates).
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
                    # Progress chatter; artifacts carry the actual result.
                    state = chunk.status_update.status.state
                    task_id = task_id or chunk.status_update.task_id

            if direct_message is not None:
                return direct_message

            # Only needed when the remote agent returned before reaching a
            # terminal state.
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


def call_a2a_agent(
    url: str,
    description: str,
    timeout: float | None = None,
    trace_id: str | None = None,
) -> str:
    """Send *description* to the A2A agent at *url*; return its answer as text.

    Blocks until the remote agent produces a result or *timeout* elapses.
    *trace_id* defaults to the active observability request, so the remote
    agent receives the same id this turn is tagged with. Raises on transport
    failure or a non-successful terminal state, so the caller surfaces it as a
    task error.
    """
    if timeout is None:
        timeout = get_subagent_timeout_seconds()
    if trace_id is None:
        from src import observability

        trace_id = observability.current_request_id()
    try:
        return cast("str", run_sync(_call(url, description, timeout, trace_id)))
    except Exception as e:
        raise RuntimeError(f"A2A agent at {url} failed: {e}") from e
