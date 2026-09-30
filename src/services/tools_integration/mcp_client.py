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
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.async_bridge import run_sync
from src.config import get_mcp_servers
from src.services.tools_integration.validation import (
    ValidatedTool,
    normalize_schema,
    prune_unset,
)

logger = logging.getLogger(__name__)

# Cached by the serialized server config: reloading is only correct when the
# configuration changes (the config-derived analogue of the filesystem tree
# hash used for dynamic tools in tools.load_dynamic_tools).
_mcp_tools_cache: dict[str, BaseTool] | None = None
_mcp_tools_cache_key: str = ""


class MCPTool(ValidatedTool, StructuredTool):
    """A sync-callable MCP tool that validates its payload and knows its kind.

    ``kind`` is what lets dispatch choose the MCP transport from the tool
    itself rather than from a name convention at the call site.
    """

    kind: str = "mcp"

    def run(self, *args, **kwargs):
        # Validate the raw input, not the parsed one: langchain-core skips
        # parsing entirely for a no-arg tool, so `{"unexpected": 1}` would
        # otherwise be discarded silently instead of refused. `invoke` passes
        # the input positionally; `run` callers may name it.
        if "tool_input" in kwargs:
            tool_input = kwargs["tool_input"]
        elif args:
            tool_input = args[0]
        else:
            tool_input = None
        self._validate_raw_input(tool_input)
        return super().run(*args, **kwargs)


def _unwrap_mcp_result(result: Any) -> Any:
    """Reduce an MCP tool result to the text a sub-agent should read.

    A real MCP server returns a list of content blocks
    (``[{"type": "text", "text": ...}]``), which is what a sub-agent's
    ``ToolMessage`` would otherwise carry — raw protocol structure presented to
    a model as if it were the answer. Text blocks are joined; non-text blocks
    (images, resources) have no sensible prose form, so their type is named
    rather than dropped silently.

    Anything that is not a content-block list is returned as-is, since a server
    or adapter may already have unwrapped it.
    """
    if not isinstance(result, list) or not result:
        return result
    if not all(isinstance(block, dict) and "type" in block for block in result):
        return result

    texts = [b.get("text", "") for b in result if b.get("type") == "text"]
    others = [b.get("type", "unknown") for b in result if b.get("type") != "text"]
    if others:
        texts.append(f"[non-text content: {', '.join(others)}]")
    return "\n".join(t for t in texts if t)


def _to_sync_tool(mcp_tool: BaseTool) -> "MCPTool":
    """Wrap an async MCP tool so it supports synchronous ``.invoke()``.

    MCP describes its tools with a raw JSON Schema **dict**, and langchain-core
    skips validation entirely for a dict ``args_schema`` — the call goes
    straight to the server with whatever the model sent. Normalizing the dict
    into a Pydantic model, and validating the raw input against it, is what
    makes an MCP tool hold the same payload contract as a built-in: a bad call
    is refused here instead of corrupting the request at the server.

    ``response_format`` is deliberately not set: the wrapper's ``func``
    already returns unwrapped text, so declaring
    ``content_and_artifact`` would unpack it a second time.
    """

    def _run(**kwargs):
        # `_parse_input` injects a default for every field that has one, so the
        # omitted optionals arrive here as explicit None. Trim them: the server
        # should receive the payload that was actually asked for.
        return _unwrap_mcp_result(run_sync(mcp_tool.ainvoke(prune_unset(kwargs))))

    # A server may omit its schema entirely for a no-arg tool.
    raw_schema = mcp_tool.args_schema or {"type": "object", "properties": {}}
    schema = normalize_schema(raw_schema)

    return MCPTool(
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=schema,
        func=_run,
        # Recorded so dispatch can tell an MCP call from a local one.
        kind="mcp",
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
