from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.state import AgentState
from src.core.memory import get_memory_content

def _get_text_content(content) -> str:
    """Extract text from multimodal content (string or list of dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
    return str(content)

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
    Final answer node. Extracts the approved response and delivers it.

    Priority:
    1. next_message — explicitly staged final answer (from critic approval)
    2. messages[-1] — fallback: pick up the last message in history
       (after plan_checker cleared next_message on compliance)
    """
    next_msg = state.get("next_message")
    if next_msg is not None:
        return {"messages": [next_msg], "next_message": None}

    # Fallback: use the last message in conversation history.
    # This handles the plan_checker compliant path where next_message is None
    # but the orchestrator's response sits in messages.
    messages = state.get("messages", [])
    if messages:
        # Find the last AI message (skip tool/system messages)
        last_ai = next((m for m in reversed(messages)
                        if isinstance(m, AIMessage)), None)
        if last_ai:
            return {"messages": [last_ai], "next_message": None}

    return {}

def call_critic_node(state: AgentState, model) -> dict:
    """
    Adversarial review node. Evaluates the agent's proposed response.

    Returns:
    - {"next_message": <original_msg>} on APPROVAL → responder delivers it
    - {"next_message": <critique>} on REJECTION → routed back to orchestrator
    """
    system_prompt = """You are a Critical Reviewer. Your job is to find flaws, inaccuracies, or omissions in the agent's proposed response.

    Compare the agent's response to the original user intent and the current plan.
    If the response is solid and addresses the user's request, respond with "APPROVED".
    If there are flaws, describe them clearly and concisely.
    """

    # Evaluate the staged message in context of the full conversation
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    content_str = _get_text_content(response.content)

    if "APPROVED" in content_str.upper():
        # Approved — pass through the original message being critiqued
        # so the responder delivers the actual answer, not the approval note.
        original_msg = state["messages"][-1]
        return {"next_message": original_msg}
    else:
        # Rejected — return critique; routing sends it back to orchestrator
        return {"next_message": response}

def call_plan_checker_node(state: AgentState, model) -> dict:
    """
    Verifies if the agent is following the current plan.

    Returns:
    - {"next_message": None} when no plan or compliant → routed to critic/responder
    - {"next_message": <violation>} when non-compliant → routed to orchestrator
    """
    plan = state.get("current_plan", [])
    if not plan:
        return {"next_message": None}  # No plan to check; routing goes to critic

    system_prompt = f"""You are a Plan Compliance Checker.
    The current plan is: {plan}

    Check if the agent's last action/message aligns with the plan.
    If it does, respond with "COMPLIANT".
    If it does not, explain why and what it should do next.
    """

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    content_str = _get_text_content(response.content)

    if "COMPLIANT" in content_str.upper():
        # Compliant — routing goes to critic for quality review
        return {"next_message": None}
    else:
        # Non-compliant — routing goes back to orchestrator
        return {"next_message": response}
