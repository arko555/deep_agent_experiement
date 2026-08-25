from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import AgentState
from src.core.memory import get_memory_content
from src.core.utils import get_message_text

def call_agent_node(state: AgentState) -> dict:
    """
    Thought/planning node. Named 'agent' to match app.py expectations for thoughts.
    Moves next_message (staged content) into messages (conversation history).
    """
    next_msg = state.get("next_message")
    if next_msg is not None:
        return {"messages": [next_msg], "next_message": None}
    return {"next_message": None}

def call_responder_node(state: AgentState) -> dict:
    """
    Final answer node. Extracts the approved response and formats it for delivery.

    Priority:
    1. next_message — explicitly staged final answer (survives critic approval)
    2. messages[-1] — defensive fallback: last AI message in history
    """
    next_msg = state.get("next_message")
    if next_msg is None:
        # Defensive fallback: use the last AI message in conversation history.
        messages = state.get("messages", [])
        # Find the last AI message (skip tool/system messages)
        next_msg = next((m for m in reversed(messages)
                         if isinstance(m, AIMessage)), None)

    if next_msg is None:
        return {}

    # Format: normalize to a plain-text AIMessage so downstream consumers
    # (UI, chat history) always receive string content, even when the staged
    # message carries multimodal content.
    return {
        "messages": [AIMessage(content=get_message_text(next_msg.content))],
        "next_message": None,
    }

def call_critic_node(state: AgentState, model) -> dict:
    """
    Adversarial review node. Evaluates the agent's proposed response.

    Returns:
    - {"review_verdict": "approved"} → routed to responder, which delivers
      the staged ``next_message`` (the actual answer, not this verdict).
    - {"next_message": <critique>, "review_verdict": "rejected"} → routed
      back to orchestrator, which receives the critique as revision feedback.
    """
    system_prompt = """You are a Critical Reviewer. Your job is to find flaws, inaccuracies, or omissions in the agent's proposed response.

    Compare the agent's proposed response (the last message below) to the original user intent and the current plan.
    If the response is solid and addresses the user's request, respond with "APPROVED" as your first word.
    If there are flaws, describe them clearly and concisely.
    """

    staged = state.get("next_message")
    messages = state["messages"]
    if staged is None and not messages:
        # Nothing to review; let the responder handle delivery.
        return {"review_verdict": "approved"}

    # Evaluate the staged message (if present) in context of the full conversation.
    review_messages = [SystemMessage(content=system_prompt)] + list(messages)
    if staged is not None:
        review_messages.append(staged)

    response = model.invoke(review_messages)
    content_str = get_message_text(response.content)

    if content_str.strip().upper().startswith("APPROVED"):
        # Approved — the staged answer stays in next_message so the
        # responder delivers the actual answer, not the approval note.
        return {"review_verdict": "approved"}
    # Rejected — stage the critique; the orchestrator consumes it as feedback.
    return {"next_message": response, "review_verdict": "rejected"}

def call_reflection_node(state: AgentState, model) -> dict:
    """
    Self-correction / reflection node.

    When the agent is stuck in a loop (many iterations) or has produced
    repeated failures, this node prompts it to step back, analyze what
    went wrong, and formulate a revised strategy before retrying.

    Returns:
    - {"next_message": <reflection>} → routed to orchestrator as strategic guidance
    """
    iteration_count = state.get("iteration_count", 0)
    messages = state.get("messages", [])
    audit_log = state.get("audit_log", [])

    # Gather context about what has gone wrong.
    tool_errors = []
    for entry in audit_log:
        if entry.get("action") == "tool_call":
            for tc in entry.get("tool_calls", []):
                tool_errors.append(f"{tc['name']}({tc.get('args', {})})")

    # Find rejection reasons from the staged message.
    staged = state.get("next_message")
    rejection_text = ""
    if staged is not None:
        from src.core.utils import get_message_text
        rejection_text = get_message_text(getattr(staged, "content", ""))

    system_prompt = f"""You are a Reflection Agent. Your job is to analyze why the current approach is failing and devise a new strategy.

Context:
- Iteration count: {iteration_count}
- Tools attempted: {tool_errors[:10] if tool_errors else 'None yet'}
- Review feedback: {rejection_text[:500] if rejection_text else 'No specific rejection'}

The current plan is: {state.get('current_plan', [])}

Instructions:
1. Identify what is going wrong (e.g., wrong tools, incomplete info, poor structure).
2. Propose a concrete revised strategy with specific steps.
3. Be concise — focus on actionable changes, not rehashing the problem.
"""

    review_messages = [SystemMessage(content=system_prompt)] + list(messages)
    response = model.invoke(review_messages)

    return {
        "next_message": response,
        "review_verdict": None,  # Clear verdict so routing resets cleanly.
    }


def call_plan_checker_node(state: AgentState, model) -> dict:
    """
    Verifies if the agent is following the current plan.

    Returns:
    - {"review_verdict": "compliant"} (no plan, or aligned) → routed to critic
    - {"next_message": <violation>, "review_verdict": "violation"} → routed
      back to orchestrator as revision feedback
    """
    plan = state.get("current_plan", [])
    if not plan:
        return {"review_verdict": "compliant"}  # No plan to check; routing goes to critic

    system_prompt = f"""You are a Plan Compliance Checker.
    The current plan is: {plan}

    Check if the agent's proposed response (the last message below) aligns with the plan.
    If it does, respond with "COMPLIANT" as your first word.
    If it does not, explain why and what it should do next.
    """

    staged = state.get("next_message")
    review_messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    if staged is not None:
        review_messages.append(staged)

    response = model.invoke(review_messages)
    content_str = get_message_text(response.content)

    if content_str.strip().upper().startswith("COMPLIANT"):
        # Compliant — the staged answer stays in next_message for the critic/responder.
        return {"review_verdict": "compliant"}
    # Non-compliant — stage the violation; the orchestrator consumes it as feedback.
    return {"next_message": response, "review_verdict": "violation"}
