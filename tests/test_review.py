"""Reviewer-node tests (src/nodes/review.py) — shared retry behavior (8.2)."""

from langchain_core.messages import AIMessage, HumanMessage


class _FlakyModel:
    """Fails the first call with a transient error, then succeeds."""

    def __init__(self, final_content):
        self.final_content = final_content
        self.calls = 0

    def invoke(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise Exception("Connection timed out (429 too many requests)")
        return AIMessage(content=self.final_content)


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="hi"), AIMessage(content="an answer")],
        "next_message": None,
        "current_plan": [],
        "iteration_count": 0,
        "audit_log": [],
    }


def test_critic_recovers_from_transient_failure():
    from src.nodes.review import call_critic_node
    model = _FlakyModel("APPROVED — the answer is solid.")
    result = call_critic_node(_state(), model)
    assert model.calls == 2  # retried after the transient error
    assert result["review_verdict"] == "approved"


def test_plan_checker_recovers_from_transient_failure():
    from src.nodes.review import call_plan_checker_node
    state = _state()
    state["current_plan"] = ["step one"]
    model = _FlakyModel("COMPLIANT — follows the plan.")
    result = call_plan_checker_node(state, model)
    assert model.calls == 2
    assert result["review_verdict"] == "compliant"


def test_reflection_recovers_from_transient_failure():
    from src.nodes.review import call_reflection_node
    model = _FlakyModel("Revised strategy: split the task first.")
    result = call_reflection_node(_state(), model)
    assert model.calls == 2
    assert result["review_verdict"] is None  # cleared for clean routing
