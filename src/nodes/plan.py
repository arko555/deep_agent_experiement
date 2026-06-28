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

def call_orchestrator(state: AgentState, model, tools: list) -> dict:
    """
    Executes the main orchestrator agent step.
    
    Args:
        state: The current AgentState.
        model: The LLM instance to invoke.
        tools: The list of tools bound to the LLM.
        
    Returns:
        A dict updating `next_message`.
    """
    system_prompt = get_system_prompt()
    formatted_messages = [SystemMessage(content=system_prompt)] + state["messages"]
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(formatted_messages)
    return {"next_message": response}
