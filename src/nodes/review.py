from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from src.state import AgentState
from src.core.memory import get_memory_content

def call_agent_node(state: AgentState) -> dict:
    """
    Thought/planning node. Named 'agent' to match app.py expectations for thoughts.
    """
    return {"messages": [state["next_message"]], "next_message": None}

def call_responder_node(state: AgentState) -> dict:
    """
    Final answer node. Named 'responder' to bypass app.py thought formatting and set full_response.
    """
    return {"messages": [state["next_message"]], "next_message": None}

def call_critic_node(state: AgentState, model) -> dict:
    """
    Adversarial review node. Evaluates the agent's proposed response.
    """
    last_message = state["messages"][-1]

    # If the last message is a tool call, the critic might not be needed yet,
    # but we want to critique the *result* of the tool call or the agent's reasoning.
    # For simplicity, we critique the agent's last message if it's an AIMessage.

    system_prompt = """You are a Critical Reviewer. Your job is to find flaws, inaccuracies, or omissions in the agent's proposed response.

    Compare the agent's response to the original user intent and the current plan.
    If the response is perfect, respond with "APPROVED".
    If there are flaws, describe them clearly.
    """

    # We want to critique the last message in the conversation
    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    # We use the model to decide if it's approved or needs work
    response = model.invoke(messages)

    # If the critic says "APPROVED", we move on.
    # Otherwise, we treat the critique as a new message for the agent to address.
    if "APPROVED" in response.content.upper():
        return {"next_message": response, "messages": [response]}
    else:
        return {"next_message": response, "messages": [response]}

def call_plan_checker_node(state: AgentState, model) -> dict:
    """
    Verifies if the agent is following the current plan.
    """
    plan = state.get("current_plan", [])
    if not plan:
        return {"next_message": None} # No plan to check

    messages = state["messages"]
    system_prompt = f"""You are a Plan Compliance Checker.
    The current plan is: {plan}

    Check if the agent's last action/message aligns with the plan.
    If it does, respond with "COMPLIANT".
    If it does not, explain why and what it should do next.
    """

    messages = [SystemMessage(content=system_prompt)] + messages
    response = model.invoke(messages)

    if "COMPLIANT" in response.content.upper():
        return {"next_message": None}
    else:
        return {"next_message": response, "messages": [response]}
