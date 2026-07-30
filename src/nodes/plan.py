from typing import List
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from src.state import AgentState
from src.core.memory import get_system_prompt

@tool
def write_todos(todos: List[str]) -> str:
    """Update or set the list of planned tasks/todos. Use this to keep track of your progress.
    
    Args:
        todos: The list of tasks/todos to complete.
    """
    return f"Updated todo list with {len(todos)} items."

def call_orchestrator(state: AgentState, model, tools: list, max_history_messages: int = 20) -> dict:
    """
    Executes the main orchestrator agent step.

    Args:
        state: The current AgentState.
        model: The LLM instance to invoke.
        tools: The list of tools bound to the LLM.
        max_history_messages: Maximum number of conversation messages to keep.

    Returns:
        A dict updating `next_message`.
    """
    system_prompt = get_system_prompt()
    messages = state["messages"]
    # Truncate old messages to prevent exceeding context limits
    if len(messages) > max_history_messages:
        messages = messages[-max_history_messages:]
    formatted_messages = [SystemMessage(content=system_prompt)] + messages
    model_with_tools = model.bind_tools(tools)
    for attempt in range(3):
        try:
            response = model_with_tools.invoke(formatted_messages)
            break
        except Exception as e:
            if attempt == 2:
                raise
            import time
            time.sleep(2 ** attempt)
    return {"next_message": response, "messages": [response]}
