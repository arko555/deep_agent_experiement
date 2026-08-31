"""Tool definitions for the deep agent graph.

This module holds all LangChain tool definitions and the dynamic tool loader.
Import from here rather than duplicating inline in agent_factory.
"""

import os
import importlib
import inspect
import uuid
from typing import List, Literal, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.core.guardrails import validate_and_normalize_path
from src.core.rag import internet_search as raw_internet_search
from src.core.subagents import SUBAGENTS, build_role_prompt, resolve_subagent, run_tool_loop

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


# subagent_type values and the tool description are driven by the SUBAGENTS
# registry (6.1) — adding a type there updates the schema automatically.
_SubagentType = Literal[(*SUBAGENTS.keys(),)]  # type: ignore[valid-type]


def _task_impl(subagent_type: _SubagentType,  # type: ignore[valid-type]
               description: str) -> str:
    # The `task` tool is executed by the tools node (local_tools_node), which
    # enforces recursion depth and aggregates child token usage/writes. A
    # direct invoke has no graph state and would skip the depth guard, so it
    # is refused outright (6.4) rather than executed at depth 0.
    return ("Error: the task tool must be executed by the tools node so subagent "
            "recursion depth can be enforced; it cannot be invoked directly.")


def _build_task_docstring() -> str:
    types = "\n".join(
        f"- **{name}**: {spec.description}" for name, spec in SUBAGENTS.items()
    )
    return f"""Delegate a complex sub-task to a specialized or general-purpose subagent.

Available subagent types:
{types}

Args:
    subagent_type: The type/role of the subagent (one of: {', '.join(SUBAGENTS)}).
    description: The task description for the subagent. Name a unique output path
        under ./workspace (e.g., workspace/<topic>.md) for each file to produce.
"""


_task_impl.__doc__ = _build_task_docstring()
task = tool(_task_impl)


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

def _execute_task(subagent_type: str, description: str, recursion_depth: int,
                  tools_dict: dict | None = None):
    """
    Execute a subagent task.

    Called from local_tools_node (not as a LangChain tool invoke) so that
    the caller has access to the graph state for depth tracking and can
    aggregate the child's token usage back into the parent.

    Dispatch is driven by the SUBAGENTS registry (6.1): kind="tool_loop"
    types run a restricted tool loop with their SKILL.md-derived prompt;
    kind="graph" types run the full cached compiled graph.

    Args:
        subagent_type: Type/alias as issued in the task tool call.
        description: The task description for the subagent.
        recursion_depth: Parent's nesting level; the child runs at +1.
        tools_dict: Tools visible to the parent this turn, used to resolve
            the spec's restricted tool names. Falls back to get_all_tools().

    Returns:
        (result_text, child_token_usage, child_write_ops) where
        child_token_usage is {"input": n, "output": n} and child_write_ops
        is the list of file-write operations the subagent performed (for the
        parent's pending_writes audit trail).
    """
    spec = resolve_subagent(subagent_type)
    if spec is None:
        return (
            f"Error: Unknown subagent_type '{subagent_type}'. "
            f"Use one of: {', '.join(SUBAGENTS)}.",
            {},
            [],
        )

    if spec.kind == "tool_loop":
        # Restricted tool loop with the spec's toolset (resolved by name) and
        # a prompt derived from its SKILL.md + shared completion contract.
        toolset = tools_dict if tools_dict is not None else get_all_tools()
        loop_tools = [toolset[name] for name in spec.tools if name in toolset]
        text, usage, write_ops = run_tool_loop(
            build_role_prompt(spec),
            description,
            loop_tools,
        )
        return text, usage, write_ops

    # kind == "graph": general-purpose subagent — use the cached compiled graph
    from src.core.agent_factory import get_deep_agent
    from src.core.config import get_max_iterations
    from src.core.utils import get_message_text

    sub_agent = get_deep_agent()
    # Fresh checkpoint thread per subagent invocation: the checkpointer
    # requires a thread_id, and reusing the parent's thread would merge
    # child state into the parent's checkpoint (and vice versa).
    res = sub_agent.invoke(
        {
            "messages": [HumanMessage(content=description)],
            "current_plan": [],
            "workspace_files": [],
            "recursion_depth": recursion_depth + 1,
            "audit_log": [],
            "token_usage": {},
            "iteration_count": 0,
            "max_iterations": get_max_iterations(),
        },
        config={"configurable": {"thread_id": f"subagent-{uuid.uuid4()}"}}
    )
    # Extract the last non-empty AI message (the responder's final
    # answer) rather than messages[-1], which could be a ToolMessage.
    final = next(
        (m for m in reversed(res.get("messages", []))
         if isinstance(m, AIMessage) and get_message_text(m.content)),
        None,
    )
    text = (
        get_message_text(final.content)
        if final is not None
        else "Error: subagent produced no final answer."
    )
    # The child's full graph already tracked its own writes; surface them
    # to the parent's audit trail.
    return text, res.get("token_usage", {}) or {}, res.get("pending_writes", []) or []


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
