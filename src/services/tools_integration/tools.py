"""Tool definitions for the deep agent graph.

This module holds all LangChain tool definitions and the dynamic tool loader.
Import from here rather than duplicating inline in agent_factory.

This is the bottom of the dependency stack: nothing here imports
``agent_orchestrator``. The ``task`` tool used to live here, which forced
three lazy upward imports and made the one-way hierarchy unenforceable. It was
moved up to ``agent_orchestrator.task_tool`` and has since been removed along
with the sub-agent-to-sub-agent delegation it served.
"""

import os
import logging
import importlib.util
import inspect
from typing import Any, Literal

import httpx
from langchain_core.tools import BaseTool, tool

from src.async_bridge import run_sync
from src.services.tools_integration.research_fetch import fetch_public_url
from src.services.tools_integration.guardrails import (
    get_workspace_files,
    get_workspace_root,
    validate_and_normalize_path,
    validate_read_path,
)
from src.services.tools_integration.mcp_client import load_mcp_tools
from src.services.tools_integration.discovery import _dir_tree_hash
from src.services.tools_integration.registry import ToolRegistry
from src.types import ToolKind

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in Tools
# ---------------------------------------------------------------------------


@tool
def write_todos(todos: list[str]) -> str:
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
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Tavily API key not found. Please set TAVILY_API_KEY in .env."
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=tavily_api_key)
        res = tavily.search(query, max_results=max_results, topic=topic)
        results = res.get("results", []) if isinstance(res, dict) else []
        if not results:
            return "No results found."
        lines = ["## Search results for: " + query, ""]
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "")
            lines.append(f"{i}. **{title}**")
            if url:
                lines.append(f"   URL: {url}")
            if content:
                snippet = content[:300].replace("\n", " ")
                lines.append(f"   {snippet}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching the web: {e!s}"


@tool
def read_file(path: str) -> str:
    """Read a file. Only AGENTS.md and files under ./workspace or ./skills are readable."""
    try:
        clean_path = validate_read_path(path)
        if not os.path.exists(clean_path):
            return f"Error: File {clean_path} does not exist."
        with open(clean_path) as f:
            return f.read()
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error reading file {path}: {e!s}"


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
        return f"Error writing to file {path}: {e!s}"


@tool
def edit_file(path: str, search_text: str, replace_text: str) -> str:
    """Edit an existing file in the `./workspace` directory by replacing search_text with replace_text."""
    try:
        clean_path = validate_and_normalize_path(path, must_be_in_workspace=True)
        if not os.path.exists(clean_path):
            return f"Error: File {clean_path} does not exist."
        with open(clean_path) as f:
            content = f.read()
        if search_text not in content:
            return f"Error: '{search_text}' not found in {clean_path}"
        new_content = content.replace(search_text, replace_text)
        with open(clean_path, "w") as f:
            f.write(new_content)
        return f"Successfully updated {clean_path}"
    except Exception as e:
        return f"Error editing file {path}: {e!s}"


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
            with open(full_path, errors="ignore") as f:
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


@tool
def list_tools() -> str:
    """List all available tools and their descriptions. Use this to discover new capabilities.

    Reports the tool layer's own view. ``task`` is deliberately absent: it
    belongs to the sub-agent layer, and a sub-agent is offered tools by its
    allowlist, not by asking this.
    """
    tools_dict = get_all_tools()
    summary = []
    for name, tool_obj in tools_dict.items():
        description = getattr(tool_obj, "description", "No description provided.")
        summary.append(f"- **{name}**: {description}")
    return "\n".join(summary)


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
    """Return every tool this service owns: built-in, dynamic file, and MCP.

    This is the complete tool set: a department's ``allowed-tools`` allowlist
    in ``subagents.select_department_tools`` resolves against exactly this, and
    the orchestrator binds none of it. Tool calls happen only inside a
    sub-agent's ReAct loop.
    """
    built_in_tools = {
        "write_todos": write_todos,
        "internet_search": internet_search,
        "read_file": read_file,
        "write_file": write_file,
        "edit_file": edit_file,
        "list_files": list_files,
        "search_files": search_files,
        "fetch_url": fetch_url,
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


def _kind_of_tool(tool: Any) -> ToolKind:
    """Read a tool's transport off the tool object, defaulting to local.

    MCP tools are constructed with ``kind="mcp"`` at load time; everything
    else is a local call unless it says otherwise.
    """
    try:
        return ToolKind(getattr(tool, "kind", ToolKind.LOCAL))
    except ValueError:
        return ToolKind.LOCAL


def create_tool_registry() -> ToolRegistry:
    """Build a ToolRegistry populated with all available tools.

    Registration respects each tool's ``@tool_spec`` metadata so the
    ``risk_level`` and ``allowed_roles`` a tool declares in ``tools/*.py``
    actually reach the registry. Previously every tool went through
    ``register_builtin``, which discarded that metadata and left
    ``allowed_roles=()`` — meaning every role could see every tool and the
    role scoping declared by ``tools/*.py`` was inert.

    Visibility rules:

    - **Built-in and MCP tools are always visible.** They are the platform's
      own capabilities, so they are registered with ``allowed_roles=("*",)``
      and no role filter can hide them.
    - **Dynamic tools (everything under ``./tools``) are visible as added.**
      A tool declaring ``allowed_roles=("*",)`` is visible to everyone; a
      tool declaring specific roles is visible only to those. A dynamic tool
      with no ``@tool_spec`` at all falls back to always-visible, so adding a
      bare ``@tool`` file to ``tools/`` works without extra ceremony.

    The ``BaseTool`` object is registered, not the bare function behind it:
    dispatch needs the object's ``args_schema`` to validate a payload and its
    ``kind`` to choose a transport, and neither survives ``tool.func``.

    Returns:
        A fully populated ToolRegistry.
    """
    registry = ToolRegistry()
    dynamic_names = set(load_dynamic_tools("./tools"))
    mcp_names = set(load_mcp_tools())

    for name, tool in get_all_tools().items():
        # Register the tool object itself; `tool_spec` metadata hangs off the
        # underlying function.
        callable_ = tool
        spec = getattr(getattr(tool, "func", tool), "__tool_spec__", None)

        if spec is not None:
            # Honour the tool's own declaration (risk level + role scoping).
            registry.register(spec, callable_)
        else:
            # Built-ins, MCP tools, and undeclared dynamic tools are always
            # visible. A dynamic tool that *did* declare a narrower scope is
            # handled by the branch above, so this cannot widen it by accident.
            registry.register_builtin(
                name,
                callable_,
                risk_level="low",
                requires_approval=False,
                allowed_roles=("*",),
                kind=_kind_of_tool(tool),
            )

        # Log what a dynamic tool actually resolved to, so a tool that is
        # discovered but invisible to a role is visible in the logs.
        if name in dynamic_names and spec is not None:
            logger.debug(
                "Registered dynamic tool '%s' (risk=%s, roles=%s)",
                name, spec.risk_level, spec.allowed_roles or "all",
            )

    if mcp_names:
        logger.debug("MCP tools registered as always-visible: %s", sorted(mcp_names))

    # Configured A2A agents, so dispatch can reach one by name. They are
    # registered as their own kind rather than as tools with a callable: A2A
    # has no introspection, so there is no schema to bind and the payload is
    # serialized into the message at dispatch time.
    from src.config import get_a2a_agents

    for agent_name, agent_config in get_a2a_agents().items():
        registry.register_a2a(
            agent_name,
            agent_config["url"],
            description=agent_config.get("description", ""),
        )

    return registry
