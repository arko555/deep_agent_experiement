"""Tests for A2A subagent connectivity (Phase 10).

The A2A client is faked and ``run_sync`` is replaced with a plain event loop,
so these tests talk to no remote agent and spawn no bridge thread. Response
chunks are real ``a2a.types`` protobuf objects, so the extraction path is
exercised for real.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

from src.services.tools_integration import a2a_client
from src.services.agent_orchestrator.subagents import SubagentSpec

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

def _text_message(text, role=Role.ROLE_AGENT):
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


class _FakeA2AClient:
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


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """No real network, no bridge thread, no polling sleep."""
    monkeypatch.setattr(a2a_client, "run_sync", lambda coro: asyncio.run(coro))
    monkeypatch.setattr(a2a_client, "POLL_INTERVAL_SECONDS", 0)
    yield
    monkeypatch.undo()


def _install(monkeypatch, fake_client, card_error=None):
    """Point the A2A client module at *fake_client*."""

    class _FakeResolver:
        def __init__(self, httpx_client=None, base_url=None):
            self.base_url = base_url

        async def get_agent_card(self):
            if card_error is not None:
                raise card_error
            return object()  # opaque card; the fake client ignores it

    async def _create_client(agent=None, client_config=None, **kwargs):
        return fake_client

    monkeypatch.setattr(a2a_client, "A2ACardResolver", _FakeResolver)
    monkeypatch.setattr(a2a_client, "create_client", _create_client)


def _call(monkeypatch, fake_client, **kwargs):
    _install(monkeypatch, fake_client, **kwargs)
    return a2a_client.call_a2a_agent("http://agent.test", "do the thing")


# ---------------------------------------------------------------------------
# 10: response extraction
# ---------------------------------------------------------------------------

class TestResponseExtraction:

    def test_direct_message_is_the_answer(self, monkeypatch):
        client = _FakeA2AClient([StreamResponse(message=_text_message("direct answer"))])
        assert _call(monkeypatch, client) == "direct answer"

    def test_task_artifacts_are_the_answer(self, monkeypatch):
        client = _FakeA2AClient([
            StreamResponse(task=_task(
                TaskState.TASK_STATE_COMPLETED, [_artifact("from artifact")]
            )),
        ])
        assert _call(monkeypatch, client) == "from artifact"

    def test_artifact_update_chunks_are_concatenated(self, monkeypatch):
        client = _FakeA2AClient([
            StreamResponse(artifact_update=TaskArtifactUpdateEvent(
                task_id="t1", context_id="c1", artifact=_artifact("part one "))),
            StreamResponse(artifact_update=TaskArtifactUpdateEvent(
                task_id="t1", context_id="c1", artifact=_artifact("part two"))),
            StreamResponse(task=_task(TaskState.TASK_STATE_COMPLETED)),
        ])
        assert _call(monkeypatch, client) == "part one \npart two"

    def test_status_updates_are_not_treated_as_results(self, monkeypatch):
        """Progress chatter must not leak into the answer."""
        client = _FakeA2AClient([
            StreamResponse(status_update=TaskStatusUpdateEvent(
                task_id="t1", context_id="c1",
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING,
                                  message=_text_message("thinking..."))),
            ),
            StreamResponse(task=_task(
                TaskState.TASK_STATE_COMPLETED, [_artifact("real answer")]
            )),
        ])
        assert _call(monkeypatch, client) == "real answer"

    def test_client_is_always_closed(self, monkeypatch):
        client = _FakeA2AClient([StreamResponse(message=_text_message("x"))])
        _call(monkeypatch, client)
        assert client.closed is True


# ---------------------------------------------------------------------------
# 10: task lifecycle
# ---------------------------------------------------------------------------

class TestTaskLifecycle:

    def test_non_terminal_task_is_polled_to_completion(self, monkeypatch):
        client = _FakeA2AClient(
            chunks=[StreamResponse(task=_task(TaskState.TASK_STATE_WORKING))],
            polls=[_task(TaskState.TASK_STATE_COMPLETED, [_artifact("polled answer")])],
        )
        assert _call(monkeypatch, client) == "polled answer"

    def test_failed_task_raises(self, monkeypatch):
        client = _FakeA2AClient([StreamResponse(task=_task(TaskState.TASK_STATE_FAILED))])
        with pytest.raises(RuntimeError, match="TASK_STATE_FAILED"):
            _call(monkeypatch, client)

    @pytest.mark.parametrize("state", [
        TaskState.TASK_STATE_INPUT_REQUIRED,
        TaskState.TASK_STATE_AUTH_REQUIRED,
    ])
    def test_interrupted_task_raises_instead_of_spinning(self, monkeypatch, state):
        """These states never self-resolve; polling them would hang the caller."""
        client = _FakeA2AClient([StreamResponse(task=_task(state))])
        with pytest.raises(RuntimeError, match="stopped for more input"):
            _call(monkeypatch, client)

    def test_empty_result_raises(self, monkeypatch):
        client = _FakeA2AClient([StreamResponse(task=_task(TaskState.TASK_STATE_COMPLETED))])
        with pytest.raises(RuntimeError, match="no text result"):
            _call(monkeypatch, client)


# ---------------------------------------------------------------------------
# 10: failure reporting
# ---------------------------------------------------------------------------

class TestFailureReporting:

    def test_transport_failure_names_the_url(self, monkeypatch):
        client = _FakeA2AClient()
        with pytest.raises(RuntimeError, match=r"http://agent\.test"):
            _call(monkeypatch, client, card_error=ConnectionError("refused"))


# ---------------------------------------------------------------------------
# 10: registry + dispatch integration
# ---------------------------------------------------------------------------

class TestExecutionDispatch:

    def test_engine_dispatches_a2a_kind_to_the_remote_agent(self, monkeypatch):
        """An ``a2a`` spec calls the remote agent rather than a local loop.

        It has no SKILL.md, so a tool loop would run with no prompt and no
        tools — a silent empty turn. The engine picks the executor by kind.
        """
        from src.services.agent_orchestrator.subagent_engine import SubAgentEngine
        from src.services.agent_orchestrator.subagents import (
            SUBAGENTS,
        )
        import src.services.tools_integration.a2a_client as a2a_mod

        spec = SubagentSpec(description="remote", kind="a2a", url="http://agent.test")
        monkeypatch.setitem(SUBAGENTS, "remote_x", spec)
        monkeypatch.setattr(a2a_mod, "call_a2a_agent",
                            lambda url, description: f"remote({url}):{description}")
        # If the local executor were reached, this would raise — the a2a spec
        # has no `skill`, so there is no prompt to build.
        monkeypatch.setattr(
            "src.services.agent_orchestrator.subagents.run_department",
            _boom,
        )

        run = asyncio.run(SubAgentEngine()._run_subagent_loop(
            name="remote_x", system_prompt="", description="task desc",
            enhanced_query="task desc",
        ))

        assert run.text == "remote(http://agent.test):task desc"
        # A remote agent reports no local token usage and writes no local files.
        assert run.usage == {"input": 0, "output": 0}
        assert run.writes == []

    def test_tool_loop_kind_still_runs_locally(self, monkeypatch):
        from src.services.agent_orchestrator.subagent_engine import SubAgentEngine
        import src.services.tools_integration.a2a_client as a2a_mod

        monkeypatch.setattr(
            a2a_mod, "call_a2a_agent", _boom,
        )
        monkeypatch.setattr(
            "src.services.agent_orchestrator.subagents.run_department",
            lambda spec, query: ("local answer", {"input": 2, "output": 1}, []),
        )

        run = asyncio.run(SubAgentEngine()._run_subagent_loop(
            name="research", system_prompt="", description="q", enhanced_query="q",
        ))
        assert run.text == "local answer"
        assert run.usage == {"input": 2, "output": 1}

    def test_a2a_specs_come_from_config(self, monkeypatch):
        import src.services.agent_orchestrator.subagents as subagents_mod

        monkeypatch.setattr(
            subagents_mod, "get_a2a_agents",
            lambda: {"remote_researcher": {"url": "http://r.test",
                                           "description": "Remote research"}},
        )
        specs = subagents_mod._a2a_specs()
        spec = specs["remote_researcher"]
        assert spec.kind == "a2a"
        assert spec.url == "http://r.test"
        assert spec.description == "Remote research"
        assert spec.parallelizable is True


def _boom(*args, **kwargs):
    raise AssertionError("the wrong executor was reached")


# ---------------------------------------------------------------------------
# 10: import-time registry wiring (the reason subagents.py loads .env itself)
# ---------------------------------------------------------------------------

def test_configured_a2a_agents_reach_the_router_roster():
    """Run in a fresh interpreter: the registry is built at import time.

    The roster is what the router names departments from, so an A2A agent
    missing here is an agent nothing can dispatch to.
    """
    env = {
        **_base_env(),
        "A2A_AGENTS": json.dumps({
            "remote_researcher": {"url": "http://r.test",
                                  "description": "Remote research agent"}
        }),
    }
    code = (
        "from src.services.agent_orchestrator.subagents import "
        "SUBAGENTS, is_parallelizable, list_departments\n"
        "roster = {d['name']: d['description'] for d in list_departments()}\n"
        "print('REG', sorted(SUBAGENTS))\n"
        "print('ROSTER', sorted(roster))\n"
        "print('PAR', is_parallelizable('remote_researcher'))\n"
        "print('DOC', roster.get('remote_researcher'))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=90,
    )
    assert out.returncode == 0, out.stderr
    assert "remote_researcher" in out.stdout
    assert "PAR True" in out.stdout
    # Routable, and its description reaches the dispatcher's prompt.
    assert "ROSTER" in out.stdout and "remote_researcher" in out.stdout.split("ROSTER")[1]
    assert "DOC Remote research agent" in out.stdout


def _base_env():
    import os

    env = {**os.environ}
    env.pop("A2A_AGENTS", None)
    env.pop("MCP_SERVERS", None)
    return env
