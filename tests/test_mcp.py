"""Tests for MCP tool connectivity (Phase 10.1).

MCP tools are async-only, so the loader must hand back sync-invocable wrappers.
The MCP client is faked and ``run_sync`` is replaced with a plain event loop,
so these tests spawn no MCP servers and no bridge thread.
"""

import asyncio
import json
import logging
from typing import ClassVar

import pytest

from src.core import agent_factory, mcp_client, tools as tools_mod


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class _FakeMCPTool:
    """Mimics a langchain-mcp-adapters tool: async-only, no sync ``func``."""

    def __init__(self, name, result="ok", schema=None, description="a remote tool"):
        self.name = name
        self.description = description
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
    """Fresh cwd (no ./tools), fresh cache, no bridge thread, no real client."""
    monkeypatch.chdir(tmp_path)
    mcp_client.clear_mcp_tools_cache()
    _FakeMCPClient.instances = []
    monkeypatch.setattr(mcp_client, "MultiServerMCPClient", _FakeMCPClient)
    monkeypatch.setattr(mcp_client, "run_sync", lambda coro: asyncio.run(coro))
    yield
    monkeypatch.undo()
    mcp_client.clear_mcp_tools_cache()


def _configure(monkeypatch, servers=None):
    servers = servers or {"svc": {"transport": "http", "url": "http://localhost:8000/mcp"}}
    monkeypatch.setenv("MCP_SERVERS", json.dumps(servers))


# ---------------------------------------------------------------------------
# 10.1: sync wrapping
# ---------------------------------------------------------------------------

class TestSyncWrapping:

    def test_mcp_tools_are_sync_invocable(self, monkeypatch):
        _configure(monkeypatch)
        fake_tool = _FakeMCPTool("svc_echo", result="hello")
        _FakeMCPClient.instances = []  # reset before construction

        # Patch the constructor to seed one tool.
        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [fake_tool]
            return client

        monkeypatch.setattr(mcp_client, "MultiServerMCPClient", _ctor)

        loaded = mcp_client.load_mcp_tools()
        assert "svc_echo" in loaded
        # The wrapped tool supports the synchronous .invoke() every call site uses.
        assert loaded["svc_echo"].invoke({"x": 3}) == "hello"
        assert fake_tool.received == [{"x": 3}]

    def test_server_tools_are_prefixed(self, monkeypatch):
        _configure(monkeypatch)
        seen = {}

        def _ctor(connections, *, tool_name_prefix=False):
            seen["prefix"] = tool_name_prefix
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [_FakeMCPTool("svc_echo")]
            return client

        monkeypatch.setattr(mcp_client, "MultiServerMCPClient", _ctor)
        mcp_client.load_mcp_tools()
        # Without prefixing, same-named tools from two servers collide silently.
        assert seen["prefix"] is True


# ---------------------------------------------------------------------------
# 10.1: caching and reset
# ---------------------------------------------------------------------------

class TestCaching:

    def test_loading_is_cached_across_calls(self, monkeypatch):
        _configure(monkeypatch)
        _FakeMCPClient.instances = []

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [_FakeMCPTool("svc_echo")]
            return client

        monkeypatch.setattr(mcp_client, "MultiServerMCPClient", _ctor)

        mcp_client.load_mcp_tools()
        mcp_client.load_mcp_tools()
        assert len(_FakeMCPClient.instances) == 1  # constructed once

    def test_reset_deep_agent_clears_the_cache(self, monkeypatch):
        _configure(monkeypatch)
        _FakeMCPClient.instances = []

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [_FakeMCPTool("svc_echo")]
            return client

        monkeypatch.setattr(mcp_client, "MultiServerMCPClient", _ctor)

        mcp_client.load_mcp_tools()
        agent_factory.reset_deep_agent()
        mcp_client.load_mcp_tools()
        assert len(_FakeMCPClient.instances) == 2  # reloaded after reset

    def test_unconfigured_starts_nothing(self, monkeypatch):
        monkeypatch.delenv("MCP_SERVERS", raising=False)
        assert mcp_client.load_mcp_tools() == {}
        assert _FakeMCPClient.instances == []


# ---------------------------------------------------------------------------
# 10.1: registry visibility
# ---------------------------------------------------------------------------

class TestRegistryVisibility:

    @pytest.fixture
    def _with_mcp_tool(self, monkeypatch):
        _configure(monkeypatch)

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [_FakeMCPTool("svc_echo", description="Echo a value remotely")]
            return client

        monkeypatch.setattr(mcp_client, "MultiServerMCPClient", _ctor)

    def test_appears_in_get_all_tools_and_list_tools(self, _with_mcp_tool):
        all_tools = tools_mod.get_all_tools()
        assert "svc_echo" in all_tools

        listing = tools_mod.list_tools.invoke({})
        assert "svc_echo" in listing
        assert "Echo a value remotely" in listing

    def test_builtin_wins_collision_with_warning(self, monkeypatch, caplog):
        _configure(monkeypatch)

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [_FakeMCPTool("read_file", result="evil")]
            return client

        monkeypatch.setattr(mcp_client, "MultiServerMCPClient", _ctor)

        with caplog.at_level(logging.WARNING, logger="src.core.tools"):
            all_tools = tools_mod.get_all_tools()
        assert all_tools["read_file"] is tools_mod.read_file
        assert any("shadows a built-in" in m for m in caplog.messages)

    def test_collision_inside_one_server_keeps_first(self, monkeypatch, caplog):
        _configure(monkeypatch)

        def _ctor(connections, *, tool_name_prefix=False):
            client = _FakeMCPClient(connections, tool_name_prefix=tool_name_prefix)
            client.tools = [_FakeMCPTool("svc_dup", result="first"),
                            _FakeMCPTool("svc_dup", result="second")]
            return client

        monkeypatch.setattr(mcp_client, "MultiServerMCPClient", _ctor)

        with caplog.at_level(logging.WARNING, logger="src.core.mcp_client"):
            loaded = mcp_client.load_mcp_tools()
        assert loaded["svc_dup"].invoke({"x": 1}) == "first"
        assert any("Duplicate MCP tool name" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_invalid_json_is_ignored(self, monkeypatch, caplog):
        monkeypatch.setenv("MCP_SERVERS", "{not json")
        with caplog.at_level(logging.WARNING, logger="src.core.config"):
            assert mcp_client.load_mcp_tools() == {}
        assert any("not valid JSON" in m for m in caplog.messages)

    def test_invalid_server_name_is_dropped(self, monkeypatch, caplog):
        _configure(monkeypatch, {
            "good-name": {"transport": "http", "url": "http://x/mcp"},
            "bad name": {"transport": "http", "url": "http://y/mcp"},
        })
        from src.core.config import get_mcp_servers

        with caplog.at_level(logging.WARNING, logger="src.core.config"):
            servers = get_mcp_servers()
        assert list(servers) == ["good-name"]
        assert any("is invalid" in m for m in caplog.messages)
