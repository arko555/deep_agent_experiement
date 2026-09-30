"""Tests for graph.py — Phase 3 graph assembly."""


from langchain_core.messages import AIMessage, HumanMessage

from src.services.agent_orchestrator.graph import get_deep_agent, reset_deep_agent
from src.services.agent_orchestrator.subagent_engine import SubagentRun
from tests.fake_models import ScriptedChatModel


def _initial_state(user_message: str) -> dict:
    return {
        "messages": [HumanMessage(content=user_message)] if user_message else [],
        "next_message": None,
        "pending_writes": [],
        "audit_log": [],
        "token_usage": {},
        "enhanced_query": "",
        "department_targets": [],
        "subagent_results": {},
    }


class FakeModel:
    """A fake model that returns canned responses."""

    def __init__(self, responses):
        from collections import deque
        self._responses = deque(responses)

    def invoke(self, messages, **kwargs):
        return self._responses.popleft() if self._responses else self._responses[0]

    def bind_tools(self, tools, **kwargs):
        return self


class TestGetDeepAgent:

    def test_returns_compiled_graph(self):
        agent = get_deep_agent()
        assert agent is not None

    def test_caches_graph(self):
        agent1 = get_deep_agent()
        agent2 = get_deep_agent()
        assert agent1 is agent2

    def test_reset_invalidates_cache(self):
        agent1 = get_deep_agent()
        reset_deep_agent()
        agent2 = get_deep_agent()
        assert agent2 is not None
        assert agent2 is not agent1

    def test_graph_has_nodes(self):
        agent = get_deep_agent()
        assert agent is not None


class TestGraphExecution:

    def test_graph_accepts_state_with_config(self, monkeypatch):
        """Graph invocation with thread_id config should not error."""
        fake = FakeModel([AIMessage(content="I will handle this myself.")])
        monkeypatch.setattr(
            "src.services.agent_orchestrator.agent_factory.get_model",
            lambda: fake,
        )
        agent = get_deep_agent()
        result = agent.invoke(
            _initial_state("test"),
            config={"configurable": {"thread_id": "test-thread-2"}},
        )
        assert result is not None


class TestConversationMemory:
    """The orchestrator must see the real conversation, not an empty window."""

    def test_user_message_reaches_the_model(self, monkeypatch):
        seen = []

        class RecordingModel:
            def bind_tools(self, tools, **kwargs):
                return self

            def invoke(self, messages, **kwargs):
                seen.append(messages)
                return AIMessage(content='{"enhanced_query": "q", "departments": []}')

        monkeypatch.setattr(
            "src.services.agent_orchestrator.agent_factory.get_model",
            lambda: RecordingModel(),
        )
        state = _initial_state("what is the weather in Tokyo?")
        get_deep_agent().invoke(
            state, config={"configurable": {"thread_id": "memory-thread"}}
        )
        sent = " ".join(str(getattr(m, "content", m)) for m in seen[0])
        assert "Tokyo" in sent, "the user's question never reached the dispatcher"

    def test_second_turn_does_not_replay_first_turn_results(self, monkeypatch):
        """subagent_results is checkpointed — it must be cleared each turn."""
        calls = {"n": 0}

        def dispatcher(messages, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return AIMessage(
                    content='{"enhanced_query": "turn one", "departments": ["research"]}'
                )
            return AIMessage(
                content='{"enhanced_query": "turn two", "departments": []}'
            )

        class Model:
            def bind_tools(self, tools, **kwargs):
                return self

            def invoke(self, messages, **kwargs):
                return dispatcher(messages)

        class StubEngine:
            def __init__(self, *a, **k):
                pass

            async def invoke_parallel(self, subs, query, *a, **k):
                # Echo the query so a stale result from turn one is
                # distinguishable from a fresh one for turn two.
                return {s["name"]: SubagentRun(text=f"RESULT-FOR({query})") for s in subs}

        monkeypatch.setattr(
            "src.services.agent_orchestrator.agent_factory.get_model",
            lambda: Model(),
        )
        monkeypatch.setattr(
            "src.services.agent_orchestrator.graph.SubAgentEngine", StubEngine
        )
        reset_deep_agent()
        agent = get_deep_agent()
        cfg = {"configurable": {"thread_id": "staleness-thread"}}

        agent.invoke(_initial_state("turn one"), config=cfg)
        result = agent.invoke(_initial_state("turn two"), config=cfg)

        assert result["subagent_results"] == {}
        answer = next(
            m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)
        )
        assert "RESULT-FOR(turn one)" not in answer
        assert "RESULT-FOR(turn two)" in answer

    def test_dispatcher_json_never_reaches_the_user(self, monkeypatch):
        """The router's control envelope is never shown to the user.

        ``{"enhanced_query": ..., "departments": ...}`` is a routing decision,
        not an answer. An empty department list is not a dead end — it falls
        through to ``general``, and what the user sees is that sub-agent's
        reply.
        """
        fake = ScriptedChatModel(
            scripts={
                "orchestrator": [AIMessage(
                    content='{"enhanced_query": "tell me about X", "departments": []}'
                )],
                "subagent": [AIMessage(content="Here is a real answer.")],
            },
            default=AIMessage(content="done"),
        )
        monkeypatch.setattr(
            "src.services.agent_orchestrator.agent_factory.get_model",
            lambda: fake,
        )
        reset_deep_agent()
        result = get_deep_agent().invoke(
            _initial_state("hi"), config={"configurable": {"thread_id": "json-thread"}}
        )
        answer = next(
            m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage)
        )
        assert not answer.strip().startswith("{")
        assert "departments" not in answer
        assert "Here is a real answer." in answer
