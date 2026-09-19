"""Async MCP bridge: load and call MCP server tools."""

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.async_bridge import run_sync
from src.config import get_mcp_servers

logger = logging.getLogger(__name__)

# Cached by the serialized server config; reloading is only correct when
# the configuration changes.
_mcp_tools_cache: dict[str, BaseTool] | None = None
_mcp_tools_cache_key: str = ""


async def _load_async(servers: dict) -> dict[str, BaseTool]:
    """Load MCP tools from configured servers, returning sync-wrapped tools."""
    client = MultiServerMCPClient(servers, tool_name_prefix=True)
    try:
        mcp_tools = await client.get_tools()
    except Exception as e:
        logger.error("Failed to load MCP tools from %s: %s", list(servers), e)
        return {}

    loaded: dict[str, BaseTool] = {}
    for mcp_tool in mcp_tools:
        if mcp_tool.name in loaded:
            logger.warning(
                "Duplicate MCP tool name '%s'; keeping first definition.",
                mcp_tool.name,
            )
            continue
        loaded[mcp_tool.name] = _to_sync_tool(mcp_tool)

    if loaded:
        logger.info("Loaded %d MCP tool(s): %s", len(loaded), ", ".join(sorted(loaded)))
    return loaded


def _to_sync_tool(mcp_tool: BaseTool) -> StructuredTool:
    """Wrap an async MCP tool so it supports synchronous ``.invoke()``."""

    def _run(**kwargs):
        return run_sync(mcp_tool.ainvoke(kwargs))

    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema or {"type": "object", "properties": {}},
        func=_run,
    )


async def load_mcp_tools_async() -> dict[str, BaseTool]:
    """Return tools from all configured MCP servers, keyed by tool name.

    Returns ``{}`` when no server is configured. Results are cached
    by server config; call ``clear_mcp_tools_cache()`` to invalidate.
    """
    global _mcp_tools_cache, _mcp_tools_cache_key

    servers = get_mcp_servers()
    if not servers:
        return {}

    key = json.dumps(servers, sort_keys=True)
    if _mcp_tools_cache is not None and key == _mcp_tools_cache_key:
        return _mcp_tools_cache

    loaded = await _load_async(servers)
    _mcp_tools_cache = loaded
    _mcp_tools_cache_key = key
    return loaded


def clear_mcp_tools_cache() -> None:
    """Drop the cached MCP tools so the next load re-reads the configuration."""
    global _mcp_tools_cache, _mcp_tools_cache_key
    _mcp_tools_cache = None
    _mcp_tools_cache_key = ""


def load_mcp_tools() -> dict[str, BaseTool]:
    """Sync entry point: load MCP tools via the async bridge.

    Mirrors the interface used by ``tools.get_all_tools``.
    """
    import asyncio

    return asyncio.run(load_mcp_tools_async())
