"""Observability tests (8.3): DeepAgentTracer is a real callback handler, not a stub."""

import pytest

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from src.core import agent_factory


@pytest.fixture
def tracer(monkeypatch):
    """A fresh tracer wired into the module global for the test's duration."""
    t = agent_factory.DeepAgentTracer()
    monkeypatch.setattr(agent_factory, "_tracer", t)
    return t


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
