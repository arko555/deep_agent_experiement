"""Load tools from configured MCP servers (Phase 10.1).

MCP tools are async-only: `langchain-mcp-adapters` builds them with
``coroutine=`` and no sync ``func``, so ``tool.invoke(args)`` raises
``NotImplementedError``. Since every call site in this codebase executes tools
synchronously (``invoke_with_retry``, ``run_tool_loop``, the tools node), each
returned tool is re-wrapped as a sync ``StructuredTool`` that drives the
original coroutine through the async bridge. Call sites need no changes.

Servers come from ``config.get_mcp_servers()`` (the ``MCP_SERVERS`` env var);
nothing here runs until at least one server is configured.
"""

import json
import logging

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.core.async_bridge import run_sync
from src.core.config import get_mcp_servers

logger = logging.getLogger(__name__)

# Cached by the serialized server config: reloading is only correct when the
# configuration changes (the config-derived analogue of the filesystem tree
# hash used for dynamic tools in tools.load_dynamic_tools).
_mcp_tools_cache: dict[str, BaseTool] | None = None
_mcp_tools_cache_key: str = ""


def _to_sync_tool(mcp_tool: BaseTool) -> StructuredTool:
    """Wrap an async MCP tool so it supports synchronous ``.invoke()``.

    ``args_schema`` is carried over verbatim (MCP supplies a raw JSON Schema
    dict). ``response_format`` is deliberately not set: the wrapper's ``func``
    returns already-unwrapped content, so declaring
    ``content_and_artifact`` would unpack it a second time.
    """

    def _run(**kwargs):
        return run_sync(mcp_tool.ainvoke(kwargs))

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        # MCP supplies a raw JSON Schema; a server may omit it for a no-arg tool.
        args_schema=mcp_tool.args_schema or {"type": "object", "properties": {}},
        func=_run,
    )


def load_mcp_tools() -> dict[str, BaseTool]:
    """Return tools from all configured MCP servers, keyed by tool name.

    Returns ``{}`` when no server is configured. Tool names are prefixed with
    the server name (``<server>_<tool>``) so two servers exposing the same tool
    name cannot silently collide. Built-in tools win name collisions — see
    ``tools.get_all_tools``, which logs those.
    """
    global _mcp_tools_cache, _mcp_tools_cache_key

    servers = get_mcp_servers()
    if not servers:
        return {}

    key = json.dumps(servers, sort_keys=True)
    if _mcp_tools_cache is not None and key == _mcp_tools_cache_key:
        return _mcp_tools_cache

    # tool_name_prefix=True: MCP tool names come from the server unsanitized,
    # so namespacing by server is what keeps multiple servers from colliding.
    client = MultiServerMCPClient(servers, tool_name_prefix=True)
    try:
        mcp_tools = run_sync(client.get_tools())
    except Exception as e:
        logger.error("Failed to load MCP tools from %s: %s", list(servers), e)
        mcp_tools = []

    loaded: dict[str, BaseTool] = {}
    for mcp_tool in mcp_tools:
        if mcp_tool.name in loaded:
            logger.warning(
                "Duplicate MCP tool name '%s'; keeping first definition.", mcp_tool.name
            )
            continue
        loaded[mcp_tool.name] = _to_sync_tool(mcp_tool)

    if loaded:
        logger.info("Loaded %d MCP tool(s): %s", len(loaded), ", ".join(sorted(loaded)))

    _mcp_tools_cache = loaded
    _mcp_tools_cache_key = key
    return loaded


def clear_mcp_tools_cache() -> None:
    """Drop the cached MCP tools so the next load re-reads the configuration."""
    global _mcp_tools_cache, _mcp_tools_cache_key
    _mcp_tools_cache = None
    _mcp_tools_cache_key = ""
