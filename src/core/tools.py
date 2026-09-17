"""Tool definitions for the deep agent graph.

This module holds all LangChain tool definitions and the dynamic tool loader.
Import from here rather than duplicating inline in agent_factory.
"""

import os
import logging
import importlib.util
import inspect
import uuid
from typing import List, Literal

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, tool

from src.core.async_bridge import run_sync
from src.core.research_fetch import fetch_public_url
from src.core.guardrails import (
    get_workspace_root,
    validate_and_normalize_path,
    validate_read_path,
)
from src.core.mcp_client import load_mcp_tools
from src.core.memory import _dir_tree_hash, get_workspace_files
from src.core.rag import internet_search as raw_internet_search
from src.core.subagents import SUBAGENTS, build_role_prompt, resolve_subagent, run_tool_loop

logger = logging.getLogger(__name__)

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
    """Read a file. Only AGENTS.md and files under ./workspace or ./skills are readable."""
    try:
        clean_path = validate_read_path(path)
        if not os.path.exists(clean_path):
            return f"Error: File {clean_path} does not exist."
        with open(clean_path, "r") as f:
            return f.read()
    except ValueError as e:
        return str(e)
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
def list_files() -> str:
    """List all files in the `./workspace` directory (relative paths, one per line).
    Use this to see what research notes or drafts already exist before reading or writing."""
    files = get_workspace_files()
    if not files:
        return "The workspace is empty."
    return "\n".join(files)


@tool
def search_files(pattern: str, max_matches: int = 30) -> str:
    """Search for a substring across all text files in `./workspace` and return matching lines.

    Args:
        pattern: The substring to search for (case-sensitive).
        max_matches: Maximum number of matches to return (default 30).
    """
    root = get_workspace_root()
    files = get_workspace_files()
    matches = []
    for rel_path in files:
        try:
            full_path = validate_read_path(str(root / rel_path))
            with open(full_path, "r", errors="ignore") as f:
                for line_no, line in enumerate(f, 1):
                    if pattern in line:
                        matches.append(f"{rel_path}:{line_no}: {line.rstrip()}")
                        if len(matches) >= max_matches:
                            break
        except (OSError, ValueError):
            continue
        if len(matches) >= max_matches:
            break
    if not matches:
        return f"No matches for '{pattern}' in workspace."
    note = "" if len(matches) < max_matches else f"\n(truncated at {max_matches} matches)"
    return "\n".join(matches) + note


FETCH_URL_TIMEOUT_SECONDS = 15
FETCH_URL_MAX_BYTES = 1_000_000  # ~1 MB; larger responses are truncated


@tool
def fetch_url(url: str) -> str:
    """Fetch the full page content of a URL as text, for research beyond snippets.

    Args:
        url: The http(s) URL to fetch.
    """
    try:
        return str(run_sync(fetch_public_url(
            url, FETCH_URL_TIMEOUT_SECONDS, FETCH_URL_MAX_BYTES
        )))
    except TimeoutError:
        return f"Error fetching {url}: total fetch deadline exceeded"
    except (httpx.HTTPError, httpx.InvalidURL, ValueError, OSError) as e:
        return f"Error fetching {url}: {e}"


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
task = tool("task")(_task_impl)


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

    if spec.kind == "a2a":
        # Remote agent over the A2A protocol: send the task description and
        # return its answer. A remote agent reports no token usage, and its
        # file writes are not visible here, so both come back empty. Failures
        # raise, and the caller turns them into a task error message.
        # Imported lazily to keep the a2a SDK off the import path until used.
        from src.core.a2a_client import call_a2a_agent

        if not spec.url:
            return (
                f"Error: A2A subagent '{subagent_type}' has no url configured.",
                {},
                [],
            )
        return call_a2a_agent(spec.url, description), {}, []

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

# 7.1: cache loaded dynamic tools with the same mtime-hash invalidation as the
# skills cache, so get_all_tools() does not re-exec ./tools/*.py when unchanged.
_dynamic_tools_cache: dict[str, BaseTool] | None = None
_dynamic_tools_cache_key: str = ""


def load_dynamic_tools(tools_dir: str) -> dict[str, BaseTool]:
    """Load ``@tool`` functions from all ``.py`` files under *tools_dir*.

    Uses ``importlib.util.spec_from_file_location`` so nothing is added to
    ``sys.path``. Modules are exec'd once and cached; the cache is invalidated
    by a (mtime, size) tree hash of the directory (7.1). Each module gets a
    unique name derived from its path within *tools_dir*, so files in
    subdirectories can't collide (7.2). Built-ins win tool-name collisions,
    reported as a logged warning below.
    """
    global _dynamic_tools_cache, _dynamic_tools_cache_key

    # Include the absolute dir in the key: the tree hash covers relative paths
    # only, so two checkouts with identical ./tools contents must not share cache.
    key = os.path.abspath(tools_dir) + "|" + _dir_tree_hash(tools_dir)
    if _dynamic_tools_cache is not None and key == _dynamic_tools_cache_key:
        return _dynamic_tools_cache

    dynamic_tools: dict[str, BaseTool] = {}
    if os.path.exists(tools_dir):
        for root, _dirs, files in os.walk(tools_dir):
            for entry in sorted(files):
                if not entry.endswith(".py") or entry == "__init__.py":
                    continue
                filepath = os.path.join(root, entry)
                rel_name = os.path.relpath(filepath, tools_dir)[:-3]
                module_name = "dynamic_tool__" + rel_name.replace(os.sep, "__").replace("/", "__")

                try:
                    spec = importlib.util.spec_from_file_location(module_name, filepath)
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                except Exception as e:
                    logger.error("Error loading dynamic tool module %s: %s", entry, e)
                    continue

                found_tools = {
                    name: obj for name, obj in inspect.getmembers(module)
                    if isinstance(obj, BaseTool)
                }
                if not found_tools:
                    logger.warning(
                        "Dynamic tool module %s defines no @tool functions; ignored.", entry
                    )
                    continue
                for name, obj in found_tools.items():
                    if name in dynamic_tools:
                        logger.warning(
                            "Duplicate dynamic tool name '%s' in %s; keeping first definition.",
                            name, entry,
                        )
                        continue
                    dynamic_tools[name] = obj

    _dynamic_tools_cache = dynamic_tools
    _dynamic_tools_cache_key = key
    return dynamic_tools


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

def get_all_tools() -> dict[str, BaseTool]:
    """Returns a dictionary of all available tools (built-in + dynamic)."""
    built_in_tools = {
        "write_todos": write_todos,
        "internet_search": internet_search,
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "list_files": list_files,
        "search_files": search_files,
        "fetch_url": fetch_url,
        "task": task,
        "list_tools": list_tools,
    }
    dynamic_tools = load_dynamic_tools("./tools")
    mcp_tools = load_mcp_tools()
    # Built-ins win name collisions (7.2), with a logged warning per collision.
    for name in sorted(set(built_in_tools) & set(dynamic_tools)):
        logger.warning(
            "Dynamic tool '%s' shadows a built-in tool; the built-in wins.", name
        )
    for name in sorted(set(built_in_tools) & set(mcp_tools)):
        logger.warning(
            "MCP tool '%s' shadows a built-in tool; the built-in wins.", name
        )
    # Precedence: built-ins > MCP > dynamic file tools.
    return {**dynamic_tools, **mcp_tools, **built_in_tools}
