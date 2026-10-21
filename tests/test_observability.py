"""Observability tests.

Two layers:

- The original tracer tests (8.3): ``DeepAgentTracer`` is a real callback
  handler whose events exist independently of the request layer.
- The request/session tests: every metric is tagged at the granularity of
  one user message (``request_id``), and a conversation folds up into
  session metrics (``session_id`` = thread id). Spans are the structured
  record; external tool calls contribute latency only.
"""

import uuid

import pytest

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src import observability
from src.services.agent_orchestrator import agent_factory
from tests.fake_models import ScriptedChatModel, ai


@pytest.fixture
def tracer(monkeypatch):
    """A fresh tracer wired into the module global for the test's duration."""
    t = agent_factory.DeepAgentTracer()
    monkeypatch.setattr(agent_factory, "_tracer", t)
    return t


@pytest.fixture
def obs(monkeypatch):
    """Observability on, with a clean request log, for the test's duration."""
    observability.set_tracing_enabled(True)
    observability.clear_request_log()
    observability.end_request()  # drop any request context a prior test left
    yield observability
    observability.end_request()
    observability.set_tracing_enabled(False)
    observability.clear_request_log()


def test_with_config_wrap_records_llm_events(tracer):
    # The exact wiring get_model() uses: with_config(callbacks=[tracer]).
    model = FakeListChatModel(responses=["hello"])
    wrapped = agent_factory._maybe_attach_callbacks(model)
    wrapped.invoke([HumanMessage(content="hi")])

    types = [e["type"] for e in tracer.get_events()]
    assert "chat_model_start" in types
    # Chat model completion is reported as chat_model_end (or llm_end,
    # depending on langchain version) — the tracer records either.
    assert "chat_model_end" in types or "llm_end" in types


def test_without_wrap_nothing_is_recorded(tracer):
    # A bare model (observability disabled) must not record events.
    model = FakeListChatModel(responses=["hello"])
    model.invoke([HumanMessage(content="hi")])
    assert tracer.get_events() == []


def test_no_wrap_when_disabled(monkeypatch):
    monkeypatch.setattr(agent_factory, "_tracer", None)
    model = FakeListChatModel(responses=["x"])
    assert agent_factory._maybe_attach_callbacks(model) is model


# ---------------------------------------------------------------------------
# Request lifecycle — one request id per user message
# ---------------------------------------------------------------------------

class TestRequestLifecycle:

    def test_start_and_end_produce_a_logged_request(self, obs):
        req = obs.start_request("thread-1")
        assert req is not None
        obs.end_request("completed")

        log = obs.get_request_log()
        assert len(log) == 1
        assert log[0]["request_id"] == req.request_id
        assert log[0]["session_id"] == "thread-1"
        assert log[0]["status"] == "completed"
        assert log[0]["duration_ms"] is not None

    def test_each_turn_gets_a_distinct_request_id(self, obs):
        first = obs.start_request("thread-1")
        obs.end_request()
        second = obs.start_request("thread-1")
        obs.end_request()

        assert first.request_id != second.request_id
        log = obs.get_request_log()
        assert len(log) == 2
        assert {r["request_id"] for r in log} == {
            first.request_id, second.request_id,
        }

    def test_no_request_when_disabled(self, obs):
        obs.set_tracing_enabled(False)
        assert obs.start_request("t") is None
        assert obs.get_request_log() == []

    def test_off_disabled_by_default(self, monkeypatch):
        # Fresh module state: nothing set the flag, so requests are None.
        monkeypatch.setattr(observability, "_TRACING_ENABLED", False)
        assert observability.start_request("t") is None

    def test_end_without_start_is_a_noop(self, obs):
        assert obs.end_request() is None

    def test_unended_request_is_marked_abandoned_by_the_next_turn(self, obs):
        """A crashed turn must not leak its id into the next one."""
        stale = obs.start_request("thread-1")
        # No end_request — as if the node raised past the responder.
        fresh = obs.start_request("thread-1")

        assert fresh.request_id != stale.request_id
        log = obs.get_request_log()
        assert len(log) == 1  # only the stale one is logged so far
        assert log[0]["status"] == "abandoned"
        assert log[0]["request_id"] == stale.request_id

    def test_end_request_clears_the_context(self, obs):
        obs.start_request("t")
        obs.end_request()
        assert obs.current_request() is None


# ---------------------------------------------------------------------------
# Spans — parented, request-tagged
# ---------------------------------------------------------------------------

class TestSpans:

    def test_span_carries_request_and_session_ids(self, obs):
        obs.start_request("sess-1")
        span = obs.start_span("research", "department")
        obs.end_span(span)

        assert span.request_id == obs.current_request().request_id
        assert span.session_id == "sess-1"

    def test_span_hierarchy_parent_child(self, obs):
        obs.start_request("t")
        dept = obs.start_span("research", "department")
        tool = obs.start_span("external_tool:fetch_url", "external_tool")
        obs.end_span(tool)
        obs.end_span(dept)
        obs.end_request()

        record = obs.get_last_request()
        tool_dict, dept_dict = record["spans"][1], record["spans"][0]
        assert tool_dict["parent_span_id"] == dept_dict["span_id"]
        assert dept_dict["parent_span_id"] is None

    def test_unknown_kind_recorded_as_node(self, obs):
        obs.start_request("t")
        span = obs.start_span("thing", "not-a-kind")
        assert span.kind == "node"

    def test_no_spans_without_a_request(self, obs):
        assert obs.start_span("x", "node") is None

    def test_error_status_and_attribute(self, obs):
        obs.start_request("t")
        span = obs.start_span("dept", "department")
        obs.end_span(span, "error", error="ValueError: boom")
        obs.end_request()

        record = obs.get_last_request()
        assert record["spans"][0]["status"] == "error"
        assert record["spans"][0]["attributes"]["error"] == "ValueError: boom"


# ---------------------------------------------------------------------------
# External tool spans — latency only, at the boundary
# ---------------------------------------------------------------------------

class TestExternalToolSpans:

    def test_latency_only_success(self, obs):
        obs.start_request("t")
        with obs.external_tool_span("mcp_calc", "mcp"):
            pass  # the "call"
        obs.end_request()

        span = obs.get_last_request()["spans"][0]
        assert span["kind"] == "external_tool"
        assert span["status"] == "completed"
        assert span["duration_ms"] >= 0
        assert span["attributes"] == {"tool": "mcp_calc", "transport": "mcp"}

    def test_latency_only_records_error_and_reraises(self, obs):
        obs.start_request("t")
        with pytest.raises(RuntimeError):
            with obs.external_tool_span("remote", "a2a"):
                raise RuntimeError("connection refused")
        obs.end_request()

        span = obs.get_last_request()["spans"][0]
        assert span["status"] == "error"
        assert "connection refused" in span["attributes"]["error"]
        assert span["duration_ms"] is not None

    def test_no_span_without_a_request(self, obs):
        with obs.external_tool_span("t", "mcp") as span:
            assert span is None

    def test_mcp_dispatch_is_measured_at_the_boundary(self, obs):
        """The registry records one latency span per external call."""
        from src.services.tools_integration.registry import ToolRegistry

        class McpTool:
            name = "srv_calc"
            description = "calc"
            kind = "mcp"

            def invoke(self, args):
                return 42

        registry = ToolRegistry()
        registry.register_builtin("srv_calc", McpTool(), kind="mcp")

        obs.start_request("t")
        assert registry.dispatch("srv_calc", {}) == 42
        obs.end_request()

        span = obs.get_last_request()["spans"][0]
        assert span["kind"] == "external_tool"
        assert span["attributes"] == {"tool": "srv_calc", "transport": "mcp"}

    def test_local_dispatch_records_no_external_span(self, obs):
        from src.services.tools_integration.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register_builtin("local_thing", lambda: "ok")

        obs.start_request("t")
        assert registry.dispatch("local_thing", {}) == "ok"
        obs.end_request()

        assert obs.get_last_request()["spans"] == []


# ---------------------------------------------------------------------------
# Session summary — every request id of one conversation
# ---------------------------------------------------------------------------

class TestSessionSummary:

    def test_session_collects_its_request_ids(self, obs):
        ids = []
        for _ in range(3):
            req = obs.start_request("sess-A")
            ids.append(req.request_id)
            obs.end_request()

        obs.start_request("sess-B")  # another conversation; must not mix in
        obs.end_request()

        summary = obs.get_session_summary("sess-A")
        assert summary["request_count"] == 3
        assert summary["request_ids"] == ids
        assert summary["session_id"] == "sess-A"

    def test_session_tokens_come_from_llm_spans(self, obs):
        obs.start_request("sess-A")
        span = obs.start_llm_span("gpt", run_id="r1")
        obs.end_llm_span(span, input_tokens=100, output_tokens=50)
        span2 = obs.start_llm_span("gpt", run_id="r2")
        obs.end_llm_span(span2, input_tokens=10, output_tokens=5)
        obs.end_request()

        summary = obs.get_session_summary("sess-A")
        assert summary["llm_calls"] == 2
        assert summary["input_tokens"] == 110
        assert summary["output_tokens"] == 55

    def test_session_external_latency_sums(self, obs):
        obs.start_request("sess-A")
        with obs.external_tool_span("a", "mcp"):
            pass
        with obs.external_tool_span("b", "a2a"):
            pass
        obs.end_request()

        summary = obs.get_session_summary("sess-A")
        assert summary["external_tool_calls"] == 2
        assert summary["external_tool_time_ms"] >= 0

    def test_unknown_session_is_none(self, obs):
        assert obs.get_session_summary("nope") is None

    def test_departments_used_is_collected(self, obs):
        obs.start_request("sess-A")
        s1 = obs.start_span("research", "department")
        obs.end_span(s1)
        s2 = obs.start_span("writer", "department")
        obs.end_span(s2)
        obs.end_request()

        assert obs.get_session_summary("sess-A")["departments_used"] == [
            "research", "writer",
        ]


# ---------------------------------------------------------------------------
# LLM spans from the callback handler
# ---------------------------------------------------------------------------

class TestLLMSpans:

    def test_handler_records_llm_span_with_tokens(self, tracer, obs):
        obs.start_request("t")

        class UsageModel(FakeListChatModel):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                result.generations[0].message.usage_metadata = {
                    "input_tokens": 12, "output_tokens": 34, "total_tokens": 46,
                }
                result.generations[0].message.model = "fake-model"
                return result

        wrapped = agent_factory._maybe_attach_callbacks(UsageModel(responses=["hi"]))
        wrapped.invoke([HumanMessage(content="hi")])
        obs.end_request()

        llm_spans = [s for s in obs.get_last_request()["spans"] if s["kind"] == "llm"]
        assert len(llm_spans) == 1
        assert llm_spans[0]["attributes"]["input_tokens"] == 12
        assert llm_spans[0]["attributes"]["output_tokens"] == 34
        assert llm_spans[0]["status"] == "completed"

    def test_handler_without_a_request_records_no_span(self, tracer, obs):
        # Tracer events fire, but no request is open — nothing to tag.
        model = FakeListChatModel(responses=["hello"])
        wrapped = agent_factory._maybe_attach_callbacks(model)
        wrapped.invoke([HumanMessage(content="hi")])

        assert obs.get_request_log() == []
        types = [e["type"] for e in tracer.get_events()]
        assert "chat_model_start" in types

    def test_two_calls_pair_via_run_id(self, tracer, obs):
        obs.start_request("t")
        wrapped = agent_factory._maybe_attach_callbacks(
            FakeListChatModel(responses=["one", "two"])
        )
        wrapped.invoke([HumanMessage(content="a")])
        wrapped.invoke([HumanMessage(content="b")])
        obs.end_request()

        llm_spans = [s for s in obs.get_last_request()["spans"] if s["kind"] == "llm"]
        assert len(llm_spans) == 2
        assert all(s["status"] == "completed" for s in llm_spans)


# ---------------------------------------------------------------------------
# Full graph — one request per invoke, tagged end to end
# ---------------------------------------------------------------------------

class _TracedScriptedModel:
    """ScriptedChatModel wrapped in the real tracer contract.

    ``ScriptedChatModel`` is a bare fake: ``get_model`` is monkeypatched to
    return it directly, so ``with_config`` never wraps it and a real run's
    callback events never fire. This wrapper emits them — start and end with
    the same ``run_id``, the pairing ``DeepAgentTracer`` depends on — using
    the canned responses' own ``usage_metadata``.
    """

    def __init__(self, inner, tracer):
        self._inner = inner
        self._tracer = tracer

    def bind_tools(self, tools, **kwargs):
        self._inner.bind_tools(tools, **kwargs)
        return self

    def invoke(self, messages, **kwargs):
        run_id = str(uuid.uuid4())
        self._tracer.on_chat_model_start({}, messages, run_id=run_id)
        response = self._inner.invoke(messages, **kwargs)
        self._tracer.on_llm_end(response, run_id=run_id)
        return response

    @property
    def counts(self):
        return self._inner.counts


def _scripted(monkeypatch, tracer, subagent_reply="department answer"):
    fake = ScriptedChatModel(
        scripts={
            "orchestrator": [AIMessage(
                content='{"enhanced_query": "do a thing", "departments": ["general"]}'
            )],
            "subagent": [ai(subagent_reply, usage=(10, 5))],
        },
        default=AIMessage(content="done"),
    )
    traced = _TracedScriptedModel(fake, tracer)
    monkeypatch.setattr(
        "src.services.agent_orchestrator.agent_factory.get_model", lambda: traced
    )
    return traced


class TestGraphTurnTracing:

    def test_one_invoke_is_one_request(self, tracer, obs, monkeypatch):
        from src.services.agent_orchestrator.graph import (
            get_deep_agent, reset_deep_agent,
        )

        _scripted(monkeypatch, tracer)
        reset_deep_agent()
        try:
            cfg = {"configurable": {"thread_id": "trace-thread"}}
            get_deep_agent().invoke(
                {"messages": [HumanMessage(content="hi")]}, config=cfg
            )
            first = obs.get_last_request()

            get_deep_agent().invoke(
                {"messages": [HumanMessage(content="again")]}, config=cfg
            )
            second = obs.get_last_request()
        finally:
            reset_deep_agent()

        assert first["request_id"] != second["request_id"]
        assert first["session_id"] == second["session_id"] == "trace-thread"
        assert first["status"] == "completed"
        # The turn's LLM calls are tagged: orchestrator + sub-agent at least.
        assert any(s["kind"] == "llm" for s in first["spans"])

    def test_session_summary_across_two_turns(self, tracer, obs, monkeypatch):
        from src.services.agent_orchestrator.graph import (
            get_deep_agent, reset_deep_agent,
        )

        _scripted(monkeypatch, tracer)
        reset_deep_agent()
        try:
            cfg = {"configurable": {"thread_id": "summary-thread"}}
            get_deep_agent().invoke(
                {"messages": [HumanMessage(content="turn one")]}, config=cfg
            )
            get_deep_agent().invoke(
                {"messages": [HumanMessage(content="turn two")]}, config=cfg
            )
        finally:
            reset_deep_agent()

        summary = obs.get_session_summary("summary-thread")
        assert summary["request_count"] == 2
        assert len(summary["request_ids"]) == 2
        assert summary["llm_calls"] >= 2  # one router call per turn at least

    def test_department_span_captures_tool_shortlist(self, tracer, obs, monkeypatch):
        from src.services.agent_orchestrator.graph import (
            get_deep_agent, reset_deep_agent,
        )

        _scripted(monkeypatch, tracer)
        reset_deep_agent()
        try:
            cfg = {"configurable": {"thread_id": "dept-thread"}}
            get_deep_agent().invoke(
                {"messages": [HumanMessage(content="hi")]}, config=cfg
            )
        finally:
            reset_deep_agent()

        req = obs.get_last_request()
        dept_spans = [s for s in req["spans"] if s["kind"] == "department"]
        assert len(dept_spans) == 1
        assert dept_spans[0]["attributes"]["department"] == "general"
        assert "shortlisted_tools" in dept_spans[0]["attributes"]

    def test_no_request_leaks_between_invokes(self, tracer, obs, monkeypatch):
        from src.services.agent_orchestrator.graph import (
            get_deep_agent, reset_deep_agent,
        )

        _scripted(monkeypatch, tracer)
        reset_deep_agent()
        try:
            cfg = {"configurable": {"thread_id": "leak-thread"}}
            get_deep_agent().invoke(
                {"messages": [HumanMessage(content="a")]}, config=cfg
            )
        finally:
            reset_deep_agent()

        # After the invoke returns, no request is active in the caller's
        # context — the responder closed it.
        assert obs.current_request() is None
