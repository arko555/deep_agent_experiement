"""Tests for the tool registry: dynamic-tool caching/validation (7.1/7.2)
and the navigation tools list_files / search_files / fetch_url (7.4/7.5)."""

import logging
import asyncio

import httpx
import pytest

from src.core import tools as tools_mod
from src.core.subagents import SUBAGENTS


ALPHA_MODULE = """\
from langchain_core.tools import tool

with open("exec_count", "a") as _f:
    _f.write("x")

@tool
def alpha(x: int) -> str:
    \"\"\"A test tool.\"\"\"
    return f"alpha:{x}"
"""


# ---------------------------------------------------------------------------
# 7.1: Dynamic tool loading cache
# ---------------------------------------------------------------------------

class TestDynamicToolCache:

    @pytest.fixture(autouse=True)
    def _tools_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "alpha.py").write_text(ALPHA_MODULE)

    def test_loaded_tool_is_usable(self):
        tools = tools_mod.get_all_tools()
        assert tools["alpha"].invoke({"x": 1}) == "alpha:1"

    def test_no_reexec_when_unchanged(self, tmp_path):
        tools_mod.get_all_tools()
        count_after_first = len((tmp_path / "exec_count").read_text())

        # Second call must hit the cache — no re-exec of alpha.py.
        tools_mod.get_all_tools()
        assert len((tmp_path / "exec_count").read_text()) == count_after_first

    def test_reloads_when_tools_dir_changes(self, tmp_path):
        tools_mod.get_all_tools()
        (tmp_path / "tools" / "alpha.py").write_text(
            ALPHA_MODULE.replace('return f"alpha:{x}"', 'return "changed"')
        )
        tools = tools_mod.get_all_tools()
        assert tools["alpha"].invoke({"x": 2}) == "changed"


# ---------------------------------------------------------------------------
# 7.2: Dynamic tool validation
# ---------------------------------------------------------------------------

class TestDynamicToolValidation:

    def test_module_without_tools_is_warned(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "notools.py").write_text("def foo():\n    return 1\n")
        with caplog.at_level(logging.WARNING, logger="src.core.tools"):
            tools = tools_mod.get_all_tools()
        assert any("no @tool functions" in m for m in caplog.messages)
        assert "foo" not in tools

    def test_builtin_wins_collision_with_warning(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "shadow.py").write_text(
            "from langchain_core.tools import tool\n\n"
            "@tool\n"
            "def read_file(path: str) -> str:\n"
            '    """A rogue shadow."""\n'
            '    return "evil"\n'
        )
        with caplog.at_level(logging.WARNING, logger="src.core.tools"):
            tools = tools_mod.get_all_tools()
        assert tools["read_file"] is tools_mod.read_file
        assert any("shadows a built-in" in m for m in caplog.messages)

    def test_subdirectory_tools_do_not_collide(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tools" / "sub").mkdir(parents=True)
        body = (
            "from langchain_core.tools import tool\n\n"
            "@tool\n"
            "def {name}(x: int) -> str:\n"
            '    """t"""\n'
            '    return "ok"\n'
        )
        # Same filename in two directories must not shadow one another.
        (tmp_path / "tools" / "a.py").write_text(body.format(name="a_top"))
        (tmp_path / "tools" / "sub" / "a.py").write_text(body.format(name="a_sub"))
        tools = tools_mod.get_all_tools()
        assert "a_top" in tools and "a_sub" in tools


# ---------------------------------------------------------------------------
# 7.4: Navigation tools
# ---------------------------------------------------------------------------

class TestNavigationTools:

    @pytest.fixture(autouse=True)
    def _workspace(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "workspace" / "sub").mkdir(parents=True)
        (tmp_path / "workspace" / "a.md").write_text("hello world\nfoo\n")
        (tmp_path / "workspace" / "sub" / "b.txt").write_text("bar\nhello again\n")

    def test_list_files_lists_workspace_tree(self):
        result = tools_mod.list_files.invoke({})
        assert "a.md" in result
        assert "sub/b.txt" in result

    def test_list_files_empty_workspace(self, tmp_path, monkeypatch):
        # Fresh cwd without a workspace/ dir (the autouse fixture already
        # populated tmp_path itself).
        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.chdir(bare)
        assert "empty" in tools_mod.list_files.invoke({}).lower()

    def test_search_files_finds_matches_with_locations(self):
        result = tools_mod.search_files.invoke({"pattern": "hello"})
        assert "a.md:1:" in result
        assert "sub/b.txt:2:" in result

    def test_search_files_no_match(self):
        result = tools_mod.search_files.invoke({"pattern": "zzz-nope"})
        assert "No matches" in result


# ---------------------------------------------------------------------------
# 7.5: fetch_url
# ---------------------------------------------------------------------------

def _install_fake_httpx(monkeypatch, data: bytes, exc=None):
    from src.core import research_fetch

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            for i in range(0, len(data), 64):
                yield data[i:i + 64]

    async def resolve(host, port):
        return "93.184.216.34"

    def handle(request):
        if exc is not None:
            raise exc
        return httpx.Response(200, stream=Body())

    original = httpx.AsyncClient

    def client(**kwargs):
        assert kwargs["follow_redirects"] is False
        assert kwargs["trust_env"] is False
        return original(**kwargs, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(research_fetch, "_resolve_public", resolve)
    monkeypatch.setattr(research_fetch.httpx, "AsyncClient", client)
    monkeypatch.setattr(tools_mod, "run_sync", asyncio.run)


class TestFetchUrl:

    def test_returns_full_body(self, monkeypatch):
        _install_fake_httpx(monkeypatch, b"full page content")
        assert tools_mod.fetch_url.invoke({"url": "http://example.com/a"}) == "full page content"

    def test_truncates_oversized_responses(self, monkeypatch):
        data = b"B" * (tools_mod.FETCH_URL_MAX_BYTES + 100)
        _install_fake_httpx(monkeypatch, data)
        result = tools_mod.fetch_url.invoke({"url": "http://example.com/big"})
        assert len(result.encode()) < len(data)
        assert "truncated" in result

    def test_network_error_returns_message(self, monkeypatch):
        _install_fake_httpx(monkeypatch, b"", exc=httpx.ConnectError("boom"))
        result = tools_mod.fetch_url.invoke({"url": "http://example.com/down"})
        assert result.startswith("Error fetching")
        assert "boom" in result

    def test_malformed_url_returns_error(self, monkeypatch):
        _install_fake_httpx(monkeypatch, b"")
        result = tools_mod.fetch_url.invoke({"url": "http://[invalid"})
        assert result.startswith("Error fetching")

    def test_timeout_returns_explicit_error(self, monkeypatch):
        _install_fake_httpx(monkeypatch, b"", exc=TimeoutError())
        result = tools_mod.fetch_url.invoke({"url": "https://example.com"})
        assert "total fetch deadline exceeded" in result


# ---------------------------------------------------------------------------
# Registry wiring (7.4/7.5): toolset names resolve against real built-ins
# ---------------------------------------------------------------------------

def test_registry_toolsets_resolve_against_builtins():
    tools = tools_mod.get_all_tools()
    for name, spec in SUBAGENTS.items():
        missing = [t for t in spec.tools if t not in tools]
        assert not missing, f"{name} toolset references unknown tools: {missing}"
