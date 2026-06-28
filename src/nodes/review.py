from src.state import AgentState

def call_agent_node(state: AgentState) -> dict:
    """
    Thought/planning node. Named 'agent' to match app.py expectations for thoughts.
    
    Args:
        state: The current AgentState.
        
    Returns:
        A dict updating `messages` and clearing `next_message`.
    """
    return {"messages": [state["next_message"]], "next_message": None}

def call_responder_node(state: AgentState) -> dict:
    """
    Final answer node. Named 'responder' to bypass app.py thought formatting and set full_response.
    
    Args:
        state: The current AgentState.
        
    Returns:
        A dict updating `messages` and clearing `next_message`.
    """
    return {"messages": [state["next_message"]], "next_message": None}
