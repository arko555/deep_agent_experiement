"""Orchestrator node tests (src/nodes/plan.py)."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.nodes.plan import call_orchestrator


class _RecordingModel:
    """Minimal model double that records the messages it was invoked with."""

    def __init__(self):
        self.last_messages = None

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, **kwargs):
        self.last_messages = messages
        return AIMessage(content="ok")


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return ""


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="hi")],
        "iteration_count": 0,
        "max_iterations": 10,
        "token_usage": {},
    }


def test_system_prompt_lists_actual_tools():
    # 8.1: the prompt must reflect the tools actually bound this turn, not
    # the stale "No tools available." text.
    model = _RecordingModel()
    call_orchestrator(_state(), model=model, tools=[search_web])

    system_prompt = next(
        m.content for m in model.last_messages
        if isinstance(getattr(m, "content", None), str) and "Available Tools" in m.content
    )
    assert "No tools available." not in system_prompt
    assert "search_web" in system_prompt
