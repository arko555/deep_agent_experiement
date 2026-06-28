import os
from typing import Annotated, List, Literal, TypedDict, Optional
from operator import add
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

# pyrefly: ignore [missing-import]
from src.state import AgentState
# pyrefly: ignore [missing-import]
from src.core.memory import get_workspace_files, get_system_prompt
from src.core.guardrails import validate_and_normalize_path
from src.core.rag import internet_search as raw_internet_search
from src.nodes.plan import call_orchestrator, write_todos
from src.nodes.review import call_agent_node, call_responder_node

load_dotenv()

# --- Model Selection ---
def get_model():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            temperature=0,
        )
    elif openai_key:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
        )
    else:
        return ChatOllama(
            model="gemma4:12b-mlx",
            temperature=0.0,
        )

# --- Guardrailed Tool Definitions ---

@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
) -> str:
    """Search the web for current information. Use when you need data from the internet.
    
    Args:
        query: The search query.
        max_results: Maximum results to return.
        topic: The search topic.
    """
    return raw_internet_search(query, max_results=max_results, topic=topic)

@tool
def read_file(path: str) -> str:
    """Read the content of a file. Use this to load SKILL.md dynamically or inspect workspace files."""
    try:
        clean_path = validate_and_normalize_path(path, must_be_in_workspace=False)
        if not os.path.exists(clean_path):
            return f"Error: File {clean_path} does not exist."
        with open(clean_path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file in the `./workspace` directory. Use this to save reports, drafts, or notes."""
    try:
        clean_path = validate_and_normalize_path(path, must_be_in_workspace=True)
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
        with open(clean_path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {clean_path}"
    except Exception as e:
        return f"Error writing to file {path}: {str(e)}"

@tool
def edit_file(path: str, search_text: str, replace_text: str) -> str:
    """Edit an existing file in the `./workspace` directory by replacing search_text with replace_text."""
    try:
        clean_path = validate_and_normalize_path(path, must_be_in_workspace=True)
        if not os.path.exists(clean_path):
            return f"Error: File {clean_path} does not exist."
        with open(clean_path, "r") as f:
            content = f.read()
        if search_text not in content:
            return f"Error: '{search_text}' not found in {clean_path}"
        new_content = content.replace(search_text, replace_text)
        with open(clean_path, "w") as f:
            f.write(new_content)
        return f"Successfully updated {clean_path}"
    except Exception as e:
        return f"Error editing file {path}: {str(e)}"

@tool
def task(subagent_type: str, description: str) -> str:
    """Delegate a complex sub-task to a specialized or general-purpose subagent.
    
    Args:
        subagent_type: The type/role of subagent (e.g. 'general-purpose', 'research', 'writer').
        description: The task description for the subagent.
    """
    sub_agent = get_deep_agent()
    res = sub_agent.invoke({
        "messages": [HumanMessage(content=description)],
        "current_plan": [],
        "workspace_files": [],
        "subagent_role": subagent_type
    })
    return res["messages"][-1].content

# Map tool names to tool definitions
tools_map = {
    "write_todos": write_todos,
    "internet_search": internet_search,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "task": task,
}

# --- Graph Wrapper Nodes (To avoid circular dependencies) ---

def local_orchestrator_node(state: AgentState):
    return call_orchestrator(state, model=get_model(), tools=list(tools_map.values()))

def local_agent_node(state: AgentState):
    return call_agent_node(state)

def local_responder_node(state: AgentState):
    return call_responder_node(state)

def local_tools_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []
    updates = {}
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        if tool_name in tools_map:
            tool_func = tools_map[tool_name]
            try:
                result = tool_func.invoke(tool_args)
                if tool_name == "write_todos":
                    updates["current_plan"] = tool_args.get("todos", [])
            except Exception as e:
                result = f"Error executing tool {tool_name}: {str(e)}"
        else:
            result = f"Tool {tool_name} not found."
            
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name))
        
    updates["messages"] = tool_messages
    updates["workspace_files"] = get_workspace_files()
    return updates

# --- Routing logic ---
def route_from_orchestrator(state: AgentState):
    next_msg = state.get("next_message")
    if next_msg and hasattr(next_msg, "tool_calls") and next_msg.tool_calls:
        return "agent"
    return "responder"

# --- Graph Construction ---
def get_deep_agent():
    workspace_root = os.getenv("WORKSPACE_ROOT", "./workspace")
    if not os.path.exists(workspace_root):
        os.makedirs(workspace_root)
        
    skills_root = "./skills"
    if not os.path.exists(skills_root):
        os.makedirs(skills_root)

    workflow = StateGraph(AgentState)
    
    workflow.add_node("orchestrator", local_orchestrator_node)
    workflow.add_node("agent", local_agent_node)
    workflow.add_node("responder", local_responder_node)
    workflow.add_node("tools", local_tools_node)
    
    workflow.set_entry_point("orchestrator")
    
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "agent": "agent",
            "responder": "responder"
        }
    )
    
    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "orchestrator")
    workflow.add_edge("responder", END)
    
    return workflow.compile()
