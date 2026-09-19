"""Tests for A2A bridge (Phase 2.7).

The A2A bridge's internal ``_call`` runs an async loop: it resolves a
card via ``A2ACardResolver``, creates a client via ``create_client``
(async), polls via ``client.send_message`` (async generator) and
``client.get_task``, and always ``client.close()`` in a finally block.

These tests patch all of those and use real ``a2a.types`` protobuf
objects for the response stream so the extraction path is real.
No network, no bridge thread, no httpx connection.
"""

import asyncio
import logging
import pytest
from a2a.types import (
    Artifact,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.client import ClientConfig

from src.services.tools_integration.a2a_bridge import _call, call_a2a_agent_async


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

def _text_message(text, role=Role.ROLE_USER):
    return Message(message_id="m1", role=role, parts=[Part(text=text)])


def _task(state, artifacts=(), task_id="t1"):
    return Task(
        id=task_id,
        context_id="c1",
        status=TaskStatus(state=state),
        artifacts=list(artifacts),
    )


def _artifact(text, artifact_id="a1"):
    return Artifact(artifact_id=artifact_id, name="out", parts=[Part(text=text)])


class _FakeClient:
    """A client that yields chunks and answers get_task."""

    def __init__(self, chunks=(), polls=()):
        self._chunks = list(chunks)
        self._polls = list(polls)
        self.closed = False
        self.sent = []

    async def send_message(self, request, **kwargs):
        self.sent.append(request)
        for chunk in self._chunks:
            yield chunk

    async def get_task(self, request, **kwargs):
        return self._polls.pop(0)

    async def close(self):
        self.closed = True


class _FakeResolver:
    def __init__(self, httpx_client=None, base_url=None):
        self.base_url = base_url

    async def get_agent_card(self):
        return object()  # opaque; the fake client ignores it


class _FakeHTTPClient:
    """Async context manager stand-in for httpx.AsyncClient."""

    def __init__(self, *args, **kwargs):
        pass

    async def get(self, url, **kwargs):
        return type("R", (), {"status_code": 200, "text": "{}"})()

    async def post(self, url, **kwargs):
        return type("R", (), {"status_code": 200, "text": "{}"})()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


# ---------------------------------------------------------------------------
# Helper: patch module and run _call
# ---------------------------------------------------------------------------

def _run_call(monkeypatch, chunks=(), polls=(), url="http://agent.test",
              description="do the thing", timeout=10.0):
    monkeypatch.setattr(
        "src.services.tools_integration.a2a_bridge.A2ACardResolver",
        _FakeResolver)
    monkeypatch.setattr("httpx.AsyncClient", _FakeHTTPClient)

    fake = _FakeClient(chunks, polls)

    async def _create_client(**kw):
        return fake

    monkeypatch.setattr(
        "src.services.tools_integration.a2a_bridge.create_client",
        _create_client)

    return asyncio.run(_call(url, description, timeout)), fake


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

class TestResponseExtraction:

    def test_direct_message_is_answer(self, monkeypatch):
        result, client = _run_call(monkeypatch, chunks=[
            StreamResponse(message=_text_message("direct answer")),
        ])
        assert result == "direct answer"
        assert client.closed is True

    def test_task_artifacts_are_answer(self, monkeypatch):
        result, client = _run_call(monkeypatch, chunks=[
            StreamResponse(task=_task(
                TaskState.TASK_STATE_COMPLETED,
                [_artifact("from artifact")]),
            ),
        ])
        assert result == "from artifact"

    def test_artifact_updates_concatenated(self, monkeypatch):
        result, client = _run_call(monkeypatch, chunks=[
            StreamResponse(artifact_update=TaskArtifactUpdateEvent(
                task_id="t1", context_id="c1",
                artifact=_artifact("part one "))),
            StreamResponse(artifact_update=TaskArtifactUpdateEvent(
                task_id="t1", context_id="c1",
                artifact=_artifact("part two"))),
            StreamResponse(task=_task(TaskState.TASK_STATE_COMPLETED)),
        ])
        assert result == "part one \npart two"

    def test_status_updates_not_treated_as_results(self, monkeypatch):
        """Progress chatter must not leak into the answer."""
        result, client = _run_call(monkeypatch, chunks=[
            StreamResponse(status_update=TaskStatusUpdateEvent(
                task_id="t1", context_id="c1",
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING,
                                  message=_text_message("thinking..."))),
            ),
            StreamResponse(task=_task(
                TaskState.TASK_STATE_COMPLETED,
                [_artifact("real answer")]),
            ),
        ])
        assert result == "real answer"


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

class TestPolling:

    def test_non_terminal_task_is_polled(self, monkeypatch):
        result, client = _run_call(monkeypatch,
            chunks=[StreamResponse(task=_task(TaskState.TASK_STATE_WORKING))],
            polls=[_task(TaskState.TASK_STATE_COMPLETED,
                           [_artifact("polled answer")])],
        )
        assert result == "polled answer"
        assert client.closed is True

    def test_failed_task_raises(self, monkeypatch):
        with pytest.raises(RuntimeError, match="TASK_STATE_FAILED"):
            _run_call(monkeypatch, chunks=[
                StreamResponse(task=_task(TaskState.TASK_STATE_FAILED)),
            ])

    @pytest.mark.parametrize("state", [
        TaskState.TASK_STATE_INPUT_REQUIRED,
        TaskState.TASK_STATE_AUTH_REQUIRED,
    ])
    def test_interrupted_state_raises(self, monkeypatch, state):
        with pytest.raises(RuntimeError, match="stopped for more input"):
            _run_call(monkeypatch, chunks=[
                StreamResponse(task=_task(state)),
            ])

    def test_terminal_canceled_raises(self, monkeypatch):
        with pytest.raises(RuntimeError, match="did not complete"):
            _run_call(monkeypatch, chunks=[
                StreamResponse(task=_task(TaskState.TASK_STATE_CANCELED)),
            ])

    def test_terminal_rejected_raises(self, monkeypatch):
        with pytest.raises(RuntimeError, match="did not complete"):
            _run_call(monkeypatch, chunks=[
                StreamResponse(task=_task(TaskState.TASK_STATE_REJECTED)),
            ])

    def test_empty_result_raises(self, monkeypatch):
        with pytest.raises(RuntimeError, match="no text result"):
            _run_call(monkeypatch, chunks=[
                StreamResponse(task=_task(TaskState.TASK_STATE_COMPLETED)),
            ])


# ---------------------------------------------------------------------------
# request structure
# ---------------------------------------------------------------------------

class TestRequest:

    def test_sends_user_message(self, monkeypatch):
        _, client = _run_call(monkeypatch, chunks=[
            StreamResponse(message=_text_message("x")),
        ])
        assert len(client.sent) == 1
        req = client.sent[0]
        assert isinstance(req, SendMessageRequest)
        assert req.message.role == Role.ROLE_USER
        assert "do the thing" in req.message.parts[0].text


# ---------------------------------------------------------------------------
# call_a2a_agent_async wrapper
# ---------------------------------------------------------------------------

class TestCallA2AAgentAsync:

    def test_success(self, monkeypatch):
        monkeypatch.setattr(
            "src.services.tools_integration.a2a_bridge.A2ACardResolver",
            _FakeResolver)
        monkeypatch.setattr("httpx.AsyncClient", _FakeHTTPClient)

        fake = _FakeClient([StreamResponse(
            message=_text_message("hello async"))])

        async def _create_client(**kw):
            return fake

        monkeypatch.setattr(
            "src.services.tools_integration.a2a_bridge.create_client",
            _create_client)

        result = asyncio.run(_call(
            "http://agent.test", "do the thing", 30.0))
        assert result == "hello async"
        assert fake.closed is True

    def test_default_timeout_from_config(self, monkeypatch):
        import src.config
        monkeypatch.setattr(src.config, "get_subagent_timeout_seconds",
                            lambda: 42.0)
        monkeypatch.setattr(
            "src.services.tools_integration.a2a_bridge.A2ACardResolver",
            _FakeResolver)
        monkeypatch.setattr("httpx.AsyncClient", _FakeHTTPClient)

        async def _create_client(**kw):
            return _FakeClient([StreamResponse(
                message=_text_message("timeout test"))])

        monkeypatch.setattr(
            "src.services.tools_integration.a2a_bridge.create_client",
            _create_client)
        result = asyncio.run(call_a2a_agent_async(
            "http://agent.test", "timeout test"))
        assert result == "timeout test"

    def test_trace_id_logged_when_provided(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "src.services.tools_integration.a2a_bridge.A2ACardResolver",
            _FakeResolver)
        monkeypatch.setattr("httpx.AsyncClient", _FakeHTTPClient)

        async def _create_client(**kw):
            return _FakeClient([StreamResponse(
                message=_text_message("trace"))])

        monkeypatch.setattr(
            "src.services.tools_integration.a2a_bridge.create_client",
            _create_client)

        with caplog.at_level(logging.INFO,
                             logger="src.services.tools_integration.a2a_bridge"):
            asyncio.run(call_a2a_agent_async(
                "http://agent.test", "trace test",
                trace_id="abc", timeout=5.0))
        assert any("trace_id=abc" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:

    def test_transport_failure_names_the_url(self, monkeypatch):
        class _BoomResolver:
            def __init__(self, httpx_client=None, base_url=None):
                pass

            async def get_agent_card(self):
                raise ConnectionError("refused")

        monkeypatch.setattr(
            "src.services.tools_integration.a2a_bridge.A2ACardResolver",
            _BoomResolver)
        monkeypatch.setattr("httpx.AsyncClient", _FakeHTTPClient)
        monkeypatch.setattr(
            "src.services.tools_integration.a2a_bridge.create_client",
            lambda **kw: _FakeClient())

        with pytest.raises(RuntimeError, match="http://agent.test"):
            asyncio.run(call_a2a_agent_async(
                "http://agent.test", "desc", timeout=5.0))
