"""Tool definitions for the deep agent graph.

This module holds all LangChain tool definitions and the dynamic tool loader.
Import from here rather than duplicating inline in agent_factory.
"""

import os
import importlib
import inspect
from typing import List, Literal, Dict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.core.guardrails import validate_and_normalize_path
from src.core.rag import internet_search as raw_internet_search

# ---------------------------------------------------------------------------
# Built-in Tools
# ---------------------------------------------------------------------------


@tool
def write_todos(todos: List[str]) -> str:
    """Update or set the list of planned tasks/todos. Use this to keep track of your progress.

    Args:
        todos: The list of tasks/todos to complete.
    """
    return f"Updated todo list with {len(todos)} items."


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
        description = getattr(tool_obj, "description", "No description provided.")
        summary.append(f"- **{name}**: {description}")
    return "\n".join(summary)


# ---------------------------------------------------------------------------
# Subagent Executor
# ---------------------------------------------------------------------------

def _execute_task(subagent_type: str, description: str, recursion_depth: int):
    """
    Execute a subagent task with recursion depth enforcement.

    Called from local_tools_node (not as a LangChain tool invoke) so that
    we have access to the graph state for depth tracking.
    """
    MAX_RECURSION = 3

    if subagent_type in ["research", "researcher"]:
        # Lazy import to avoid circular dependency at module load time
        from src.nodes.research import ResearchResult
        from src.core.agent_factory import get_model

        model = get_model().with_structured_output(ResearchResult)
        res = model.invoke([HumanMessage(content=description)])
        return str(res)
    elif subagent_type in ["write", "writer"]:
        from src.nodes.write import WriteResult
        from src.core.agent_factory import get_model

        model = get_model().with_structured_output(WriteResult)
        res = model.invoke([HumanMessage(content=description)])
        return str(res)
    else:
        # General-purpose subagent — use the cached compiled graph
        from src.core.agent_factory import get_deep_agent

        sub_agent = get_deep_agent()
        res = sub_agent.invoke(
            {
                "messages": [HumanMessage(content=description)],
                "current_plan": [],
                "workspace_files": [],
                "recursion_depth": recursion_depth + 1,
                "audit_log": [],
                "token_usage": {},
                "iteration_count": 0,
                "max_iterations": 50,
            }
        )
        return res["messages"][-1].content


# ---------------------------------------------------------------------------
# Dynamic Tool Loading (3.4: importlib instead of sys.path)
# ---------------------------------------------------------------------------

def load_dynamic_tools(tools_dir: str) -> Dict[str, tool]:
    """Load tools from a directory using ``importlib.util.spec_from_file_location``.

    This avoids polluting ``sys.path`` and prevents import conflicts between
    dynamically loaded modules and the rest of the application.
    """
    dynamic_tools = {}
    if not os.path.exists(tools_dir):
        return dynamic_tools

    for entry in os.listdir(tools_dir):
        if not entry.endswith(".py"):
            continue
        filepath = os.path.join(tools_dir, entry)
        module_name = entry[:-3]

        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and hasattr(obj, "_is_langchain_tool"):
                    dynamic_tools[name] = obj
        except Exception as e:
            print(f"Error loading dynamic tool {entry}: {e}")

    return dynamic_tools


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

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
