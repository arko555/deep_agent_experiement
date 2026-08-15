"""Deep agent graph factory.

Constructs and caches the LangGraph state graph for the deep agent workflow.
Routing, tools, and retry logic are imported from their dedicated modules.
"""

import os
from datetime import datetime

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# pyrefly: ignore [missing-import]
from src.state import AgentState
# pyrefly: ignore [missing-import]
from src.core.memory import get_workspace_files, get_system_prompt
from src.core.routing import route_from_orchestrator, route_from_critic, route_from_plan_checker
from src.core.utils import invoke_with_retry
from src.core.tools import get_all_tools, _execute_task
from src.nodes.plan import call_orchestrator
from src.nodes.review import call_agent_node, call_responder_node, call_critic_node, call_plan_checker_node

load_dotenv()


# --- Cached Compiled Graph ---
# Compile once, reuse across all invocations (including recursive subagents).
# Safe because node lambdas call get_model() at runtime, not compile time.
_compiled_graph = None


# --- Model Selection ---
def get_model():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if anthropic_key:
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620"),
            temperature=0,
        )
    elif openrouter_key:
        return ChatOpenRouter(
            model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
            api_key=openrouter_key,
            temperature=0,
        )
    elif openai_key:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
        )
    elif google_key:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
            temperature=0,
        )
    else:
        return ChatOllama(
            model="gemma4:12b-mlx",
            temperature=0.0,
        )


# --- Graph Wrapper Nodes ---

def local_orchestrator_node(state: AgentState):
    current_tools = get_all_tools()
    return call_orchestrator(state, model=get_model(), tools=list(current_tools.values()))


def local_agent_node(state: AgentState):
    return call_agent_node(state)


def local_responder_node(state: AgentState):
    return call_responder_node(state)


def local_tools_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []
    updates = {}

    current_tools = get_all_tools()

    audit_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": "tool_call",
        "details": "",
        "tool_calls": [],
    }

    # Depth only increases when a subagent is actually spawned, so ordinary
    # tool loops don't exhaust the delegation budget.
    recursion_depth = state.get("recursion_depth", 0)

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        audit_entry["tool_calls"].append({"name": tool_name, "args": tool_args})

        if tool_name == "task":
            # Handle task specially: enforce recursion depth from state.
            # The task tool is still registered (for LLM discovery) but we
            # execute it here so we can pass recursion_depth from state.
            if recursion_depth >= 3:
                result = (
                    f"Error: Maximum recursion depth (3) reached. "
                    f"Cannot delegate further subagents. "
                    f"Handle this task directly or consolidate remaining work."
                )
            else:
                subagent_type = tool_args.get("subagent_type", "general-purpose")
                description = tool_args.get("description", "")
                result = _execute_task(subagent_type, description, recursion_depth)
                recursion_depth += 1
        elif tool_name in current_tools:
            tool_func = current_tools[tool_name]
            # Inline retry for tool execution: returns an error string on
            # failure rather than raising, so the graph can continue.
            result = None
            for attempt in range(3):
                try:
                    result = tool_func.invoke(tool_args)
                    break
                except Exception as e:
                    if attempt == 2:
                        result = f"Error executing tool {tool_name}: {str(e)}"
                    else:
                        import time

                        time.sleep(2**attempt)

            if result is not None:
                if tool_name == "write_todos":
                    updates["current_plan"] = tool_args.get("todos", [])

                if tool_name in ["write_file", "edit_file"]:
                    result = f"PENDING_APPROVAL: {result}"
        else:
            result = f"Tool '{tool_name}' not found."

        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name))

    updates["messages"] = tool_messages
    updates["workspace_files"] = get_workspace_files()
    updates["audit_log"] = [audit_entry]
    updates["recursion_depth"] = recursion_depth
    return updates


# --- Graph Construction ---

def get_deep_agent():
    """Return the compiled deep agent graph (cached after first call)."""
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

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
    workflow.add_node("critic", lambda state: call_critic_node(state, get_model()))
    workflow.add_node("plan_checker", lambda state: call_plan_checker_node(state, get_model()))

    workflow.set_entry_point("orchestrator")

    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "agent": "agent",
            "critic": "critic",
            "plan_checker": "plan_checker",
            "responder": "responder",
            "end": END,
        },
    )

    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "orchestrator")
    workflow.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "responder": "responder",
            "orchestrator": "orchestrator",
        },
    )
    workflow.add_conditional_edges(
        "plan_checker",
        route_from_plan_checker,
        {
            "critic": "critic",
            "orchestrator": "orchestrator",
        },
    )
    workflow.add_edge("responder", END)

    _compiled_graph = workflow.compile(checkpointer=MemorySaver())
    return _compiled_graph
