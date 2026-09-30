"""Tests for MCP bridge (Phase 2.6)."""

import asyncio
import json
import logging
from typing import ClassVar

import pytest

from langchain_core.tools import BaseTool, StructuredTool

from src.services.tools_integration import mcp_bridge, mcp_client
from src.services.tools_integration.mcp_bridge import (
    load_mcp_tools_async,
    _to_sync_tool,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class _FakeMCPTool(BaseTool):
    """Mimics a langchain-mcp-adapters tool: async-only, no sync func."""

    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}

    def __init__(self, name, result="ok", schema=None, description="remote tool"):
        super().__init__(name=name, description=description)
        self.args_schema = schema or {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        self._result = result
        self.received = []

    async def ainvoke(self, args, **kwargs):
        self.received.append(args)
        return self._result

    def _run(self, **kwargs):
        """Required by BaseTool abstract method; not used for async tools."""
        return self._result


class _FakeMCPClient:
    """Stands in for MultiServerMCPClient."""

    def __init__(self, connections, *, tool_name_prefix=False):
        self.connections = connections
        self.tool_name_prefix = tool_name_prefix
        self.get_tools_calls = 0
        self.tools = []
        _FakeMCPClient.instances.append(self)

    async def get_tools(self, **kwargs):
        self.get_tools_calls += 1
        return list(self.tools)

    instances: ClassVar[list] = []


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Fresh cache, no bridge thread, no real client."""
    mcp_bridge.clear_mcp_tools_cache()
    _FakeMCPClient.instances = []
    monkeypatch.setattr(mcp_bridge, "MultiServerMCPClient", _FakeMCPClient)
    # The bridge delegates tool wrapping to mcp_client, so `run_sync` — the
    # seam that drives an async MCP coroutine synchronously — is patched
    # there. Patching it on the bridge would no longer intercept anything.
    monkeypatch.setattr(mcp_client, "run_sync", lambda coro: asyncio.run(coro))
    yield
    monkeypatch.undo()
    mcp_bridge.clear_mcp_tools_cache()


def _configure(monkeypatch, servers=None):
    servers = servers or {"svc": {"transport": "http", "url": "http://localhost:8000/mcp"}}
    monkeypatch.setenv("MCP_SERVERS", json.dumps(servers))


# ---------------------------------------------------------------------------
# load_mcp_tools_async
# ---------------------------------------------------------------------------

class TestLoadMcpToolsAsync:

    def test_returns_dict_of_tools(self, monkeypatch):
        _configure(monkeypatch)
        fake = _FakeMCPTool("svc_echo", result="hello")

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [fake]
            return client

        monkeypatch.setattr(mcp_bridge, "MultiServerMCPClient", _ctor)

        result = asyncio.run(load_mcp_tools_async())
        assert "svc_echo" in result
        assert result["svc_echo"].invoke({"x": 3}) == "hello"
        assert fake.received == [{"x": 3}]

    def test_empty_config_returns_empty(self, monkeypatch):
        monkeypatch.delenv("MCP_SERVERS", raising=False)
        result = asyncio.run(load_mcp_tools_async())
        assert result == {}
        assert _FakeMCPClient.instances == []

    def test_tool_names_from_prefixed_client(self, monkeypatch):
        _configure(monkeypatch)

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [
                _FakeMCPTool("server_alpha"),
                _FakeMCPTool("server_beta"),
            ]
            return client

        monkeypatch.setattr(mcp_bridge, "MultiServerMCPClient", _ctor)
        result = asyncio.run(load_mcp_tools_async())
        assert set(result.keys()) == {"server_alpha", "server_beta"}

    def test_skips_duplicate_names(self, monkeypatch, caplog):
        _configure(monkeypatch)

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [
                _FakeMCPTool("dup", result="first"),
                _FakeMCPTool("dup", result="second"),
            ]
            return client

        monkeypatch.setattr(mcp_bridge, "MultiServerMCPClient", _ctor)
        with caplog.at_level(logging.WARNING, logger="src.services.tools_integration.mcp_bridge"):
            result = asyncio.run(load_mcp_tools_async())
        assert result["dup"].invoke({"x": 1}) == "first"
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _to_sync_tool
# ---------------------------------------------------------------------------

class TestToSyncTool:

    def test_wraps_async_tool_as_structured(self):
        async_tool = _FakeMCPTool("t", result="sync ok")
        sync = _to_sync_tool(async_tool)
        assert isinstance(sync, BaseTool)
        assert sync.name == "t"
        assert sync.invoke({"x": 9}) == "sync ok"
        assert async_tool.received == [{"x": 9}]

    def test_preserves_description(self):
        async_tool = _FakeMCPTool("t", description="my desc")
        sync = _to_sync_tool(async_tool)
        assert sync.description == "my desc"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:

    def test_caches_across_calls(self, monkeypatch):
        _configure(monkeypatch)
        _FakeMCPClient.instances = []

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [_FakeMCPTool("cached")]
            return client

        monkeypatch.setattr(mcp_bridge, "MultiServerMCPClient", _ctor)

        asyncio.run(load_mcp_tools_async())
        asyncio.run(load_mcp_tools_async())
        assert len(_FakeMCPClient.instances) == 1

    def test_reset_deep_agent_clears_cache(self, monkeypatch):
        _configure(monkeypatch)
        _FakeMCPClient.instances = []

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [_FakeMCPTool("t")]
            return client

        monkeypatch.setattr(mcp_bridge, "MultiServerMCPClient", _ctor)

        asyncio.run(load_mcp_tools_async())
        from src.services.agent_orchestrator import agent_factory
        agent_factory.reset_deep_agent()
        asyncio.run(load_mcp_tools_async())
        assert len(_FakeMCPClient.instances) == 2

    def test_cache_resets_on_mcp_reset(self, monkeypatch):
        _configure(monkeypatch)
        _FakeMCPClient.instances = []

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [_FakeMCPTool("t")]
            return client

        monkeypatch.setattr(mcp_bridge, "MultiServerMCPClient", _ctor)

        asyncio.run(load_mcp_tools_async())
        mcp_bridge.clear_mcp_tools_cache()
        asyncio.run(load_mcp_tools_async())
        assert len(_FakeMCPClient.instances) == 2
