import os
import importlib
import inspect
import pkgutil
import sys
from typing import Annotated, List, Literal, TypedDict, Optional, Dict
from datetime import datetime
import time

from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

# pyrefly: ignore [missing-import]
from src.state import AgentState
# pyrefly: ignore [missing-import]
from src.core.memory import get_workspace_files, get_system_prompt, get_tools_summary
from src.core.guardrails import validate_and_normalize_path
from src.core.rag import internet_search as raw_internet_search
from src.nodes.plan import call_orchestrator, write_todos
from src.nodes.review import call_agent_node, call_responder_node, call_critic_node, call_plan_checker_node
from src.nodes.research import ResearchResult
from src.nodes.write import WriteResult

load_dotenv()

# --- Cached Compiled Graph ---
# Compile once, reuse across all invocations (including recursive subagents).
# Safe because node lambdas call get_model() at runtime, not compile time.
_compiled_graph = None

def _execute_task(subagent_type: str, description: str, recursion_depth: int):
    """
    Execute a subagent task with recursion depth enforcement.

    Called from local_tools_node (not as a LangChain tool invoke) so that
    we have access to the graph state for depth tracking.
    """
    MAX_RECURSION = 3

    if subagent_type in ["research", "researcher"]:
        model = get_model().with_structured_output(ResearchResult)
        res = model.invoke([HumanMessage(content=description)])
        return str(res)
    elif subagent_type in ["write", "writer"]:
        model = get_model().with_structured_output(WriteResult)
        res = model.invoke([HumanMessage(content=description)])
        return str(res)
    else:
        # General-purpose subagent — use the cached compiled graph
        sub_agent = get_deep_agent()
        res = sub_agent.invoke({
            "messages": [HumanMessage(content=description)],
            "current_plan": [],
            "workspace_files": [],
            "recursion_depth": recursion_depth + 1,
            "audit_log": [],
            "token_usage": {},
            "iteration_count": 0,
            "max_iterations": 50,
        })
        return res["messages"][-1].content


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
    # Note: This tool is handled specially in local_tools_node for recursion
    # depth enforcement. The invoke path is kept for discovery but execution
    # goes through _execute_task in the tools node.
    return _execute_task(subagent_type, description, 0)

@tool
def list_tools() -> str:
    """List all available tools and their descriptions. Use this to discover new capabilities."""
    tools_dict = get_all_tools()
    summary = []
    for name, tool_obj in tools_dict.items():
        description = getattr(tool_obj, 'description', 'No description provided.')
        summary.append(f"- **{name}**: {description}")
    return "\n".join(summary)

# --- Dynamic Tool Loading ---

def load_dynamic_tools(tools_dir: str) -> Dict[str, tool]:
    dynamic_tools = {}
    if not os.path.exists(tools_dir):
        return dynamic_tools

    if os.path.abspath(tools_dir) not in sys.path:
        sys.path.append(os.path.abspath(tools_dir))

    for loader, module_name, is_pkg in pkgutil.iter_modules([tools_dir]):
        try:
            module = importlib.import_module(module_name)
            importlib.reload(module)

            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, "_is_langchain_tool"):
                    dynamic_tools[name] = obj
        except Exception as e:
            print(f"Error loading dynamic tool {module_name}: {e}")

    return dynamic_tools

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
        "tool_calls": []
    }

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        audit_entry["tool_calls"].append({"name": tool_name, "args": tool_args})

        if tool_name == "task":
            # Handle task specially: enforce recursion depth from state.
            # The task tool is still registered (for LLM discovery) but we
            # execute it here so we can pass recursion_depth from state.
            current_depth = state.get("recursion_depth", 0)
            if current_depth >= 3:
                result = (
                    f"Error: Maximum recursion depth (3) reached. "
                    f"Cannot delegate further subagents. "
                    f"Handle this task directly or consolidate remaining work."
                )
            else:
                subagent_type = tool_args.get("subagent_type", "general-purpose")
                description = tool_args.get("description", "")
                result = _execute_task(subagent_type, description, current_depth)
        elif tool_name in current_tools:
            tool_func = current_tools[tool_name]
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
                        time.sleep(2 ** attempt)

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
    updates["iteration_count"] = state.get("iteration_count", 0) + 1
    updates["recursion_depth"] = state.get("recursion_depth", 0) + 1
    return updates

# --- Retry wrapper for API calls ---
def _invoke_with_retry(model_with_tools, messages, max_retries=3, base_delay=2.0):
    """Invoke an LLM with exponential backoff retry on transient failures."""
    for attempt in range(max_retries):
        try:
            return model_with_tools.invoke(messages)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    raise RuntimeError("Exhausted all retries")


# --- Routing logic ---
def route_from_orchestrator(state: AgentState):
    next_msg = state.get("next_message")
    if not next_msg:
        return "END"

    if hasattr(next_msg, "tool_calls") and next_msg.tool_calls:
        return "agent"

    # No tool calls → check plan first, then critic
    plan = state.get("current_plan", [])
    if plan:
        return "plan_checker"
    return "critic"


def _get_content_text(content):
    """Extract text from multimodal content (string or list of dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
    return str(content)


def route_from_critic(state: AgentState):
    next_msg = state.get("next_message")
    if not next_msg:
        return "responder"  # approved — pass original message through

    content = _get_content_text(next_msg.content if hasattr(next_msg, "content") else "")
    if "APPROVED" in content.upper():
        return "responder"  # approved → deliver final answer
    else:
        return "orchestrator"  # rejected → orchestrator reformulates


def route_from_plan_checker(state: AgentState):
    next_msg = state.get("next_message")
    if not next_msg:
        return "critic"  # no plan or compliant → continue to critic for quality review

    content = _get_content_text(next_msg.content if hasattr(next_msg, "content") else "")
    if "COMPLIANT" in content.upper():
        return "critic"  # compliant → critic checks quality
    else:
        return "orchestrator"  # non-compliant → orchestrator reformulates

# --- Graph Construction ---
def get_all_tools() -> Dict[str, tool]:
    """Returns a dictionary of all available tools (built-in + dynamic)."""
    built_in_tools = {
        "write_todos": write_todos,
        "internet_search": internet_search,
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "task": task,
        "list_tools": list_tools,
    }
    dynamic_tools = load_dynamic_tools("./tools")
    return {**built_in_tools, **dynamic_tools}

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
            "responder": "responder"
        }
    )

    workflow.add_edge("agent", "tools")
    workflow.add_edge("tools", "orchestrator")
    workflow.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "responder": "responder",
            "orchestrator": "orchestrator"
        }
    )
    workflow.add_conditional_edges(
        "plan_checker",
        route_from_plan_checker,
        {
            "critic": "critic",
            "orchestrator": "orchestrator"
        }
    )
    workflow.add_edge("responder", END)

    _compiled_graph = workflow.compile()
    return _compiled_graph
