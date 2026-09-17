"""Tests for guardrails — path validation and normalization."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.guardrails import (
    clear_workspace,
    get_workspace_root,
    validate_and_normalize_path,
    validate_read_path,
)
from src.core.memory import get_memory_content, get_skill_body, get_workspace_files
from src.core.tools import edit_file, list_files, read_file, search_files, write_file


@pytest.fixture(autouse=True)
def isolated_filesystem(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WORKSPACE_ROOT", raising=False)


# ---------------------------------------------------------------------------
# Normal relative paths (no workspace enforcement)
# ---------------------------------------------------------------------------

class TestValidatePathNormalRelative:

    def test_plain_relative_path_passes_through(self):
        assert validate_and_normalize_path("skills/hello.py") == "skills/hello.py"

    def test_dot_slash_prefix_normalized(self):
        result = validate_and_normalize_path("./workspace/draft.md")
        assert result == "workspace/draft.md" or result == "./workspace/draft.md"

    def test_nested_relative_path_passes(self):
        assert validate_and_normalize_path("a/b/c.txt") == "a/b/c.txt"


# ---------------------------------------------------------------------------
# Workspace enforcement
# ---------------------------------------------------------------------------

class TestValidatePathWorkspaceEnforcement:

    def test_workspace_prefix_kept(self):
        result = validate_and_normalize_path("workspace/report.md", must_be_in_workspace=True)
        assert result == "workspace/report.md"

    def test_non_workspace_path_forced_into_workspace(self):
        result = validate_and_normalize_path("docs/notes.txt", must_be_in_workspace=True)
        assert result == "workspace/docs/notes.txt"

    def test_absolute_path_forced_into_workspace_basename(self):
        result = validate_and_normalize_path("/etc/passwd", must_be_in_workspace=True)
        assert result == "workspace/passwd"

    def test_traversal_path_forced_into_workspace_basename(self):
        result = validate_and_normalize_path("../../secret", must_be_in_workspace=True)
        assert result == "workspace/secret"


# ---------------------------------------------------------------------------
# Strict mode — raises on suspicious paths
# ---------------------------------------------------------------------------

class TestValidatePathStrict:

    def test_traversal_raises_in_strict_mode(self):
        with pytest.raises(ValueError, match="Access denied"):
            validate_and_normalize_path("../config", strict=True)

    def test_absolute_path_raises_in_strict_mode(self):
        with pytest.raises(ValueError, match="Access denied"):
            validate_and_normalize_path("/etc/shadow", strict=True)

    def test_safe_relative_path_passes_in_strict_mode(self):
        result = validate_and_normalize_path("workspace/safe.md", strict=True)
        assert result == "workspace/safe.md"

    def test_dotdot_raises_in_strict_mode(self):
        with pytest.raises(ValueError, match="Access denied"):
            validate_and_normalize_path("a/../b", strict=True)


# ---------------------------------------------------------------------------
# Non-strict read mode — redirects silently
# ---------------------------------------------------------------------------

class TestValidatePathNonStrictRedirect:

    def test_absolute_path_redirected_to_basename(self):
        result = validate_and_normalize_path("/usr/local/bin/foo")
        assert result == "foo"

    def test_traversal_redirected_to_basename(self):
        result = validate_and_normalize_path("../../etc/hosts")
        # normpath resolves ".." so the basename is "hosts"
        assert result == "hosts"


# ---------------------------------------------------------------------------
# 7.3: Default-deny read allowlist
# ---------------------------------------------------------------------------

class TestValidateReadPath:

    def test_workspace_path_allowed(self):
        assert validate_read_path("workspace/a.md") == "workspace/a.md"

    def test_workspace_dot_slash_normalized(self):
        assert validate_read_path("./workspace/a.md") == "workspace/a.md"

    def test_skills_path_allowed(self):
        assert validate_read_path("skills/research/SKILL.md") == "skills/research/SKILL.md"

    def test_agents_md_allowed(self):
        assert validate_read_path("AGENTS.md") == "AGENTS.md"

    def test_dot_slash_agents_md_allowed(self):
        assert validate_read_path("./AGENTS.md") == "AGENTS.md"

    @pytest.mark.parametrize("path", [
        ".env",
        "./.env",
        "src/core/tools.py",
        "../../etc/passwd",
        "/etc/shadow",
        "workspace/../config.yaml",  # normalizes OUT of the allowlist
    ])
    def test_outside_allowlist_denied(self, path):
        with pytest.raises(ValueError, match="Access denied"):
            validate_read_path(path)


class TestReadFileSandbox:

    @pytest.fixture(autouse=True)
    def _workspace(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "workspace").mkdir()
        (tmp_path / "workspace" / "notes.md").write_text("hello workspace")
        (tmp_path / ".env").write_text("SECRET=1")

    def test_read_file_denies_env(self):
        result = read_file.invoke({"path": "./.env"})
        assert result.startswith("Access denied")
        assert "workspace" in result and "skills" in result

    def test_read_file_allows_workspace(self):
        result = read_file.invoke({"path": "workspace/notes.md"})
        assert result == "hello workspace"

    def test_read_file_missing_file_still_reports_missing(self):
        result = read_file.invoke({"path": "workspace/nope.md"})
        assert "does not exist" in result


@pytest.mark.parametrize("configured", [None, "custom/nested", "absolute"])
def test_workspace_root_shared_by_tools(tmp_path, monkeypatch, configured):
    expected = tmp_path / (configured or "workspace")
    if configured:
        monkeypatch.setenv(
            "WORKSPACE_ROOT", str(expected) if configured == "absolute" else configured
        )
    assert get_workspace_root() == expected
    assert "Successfully" in write_file.invoke({"path": "workspace/note.txt", "content": "find me"})
    assert "Successfully" in write_file.invoke({"path": "bare.txt", "content": "bare"})
    assert (expected / "note.txt").read_text() == "find me"
    assert read_file.invoke({"path": "workspace/note.txt"}) == "find me"
    assert read_file.invoke({"path": str(expected / "note.txt")}) == "find me"
    assert "Successfully" in edit_file.invoke({
        "path": "workspace/note.txt", "search_text": "find", "replace_text": "found"
    })
    assert get_workspace_files() == ["bare.txt", "note.txt"]
    assert list_files.invoke({}) == "bare.txt\nnote.txt"
    assert search_files.invoke({"pattern": "found"}) == "note.txt:1: found me"


def test_workspace_root_changes_at_call_time(tmp_path, monkeypatch):
    for name in ("first", "second"):
        monkeypatch.setenv("WORKSPACE_ROOT", name)
        write_file.invoke({"path": "workspace/note", "content": name})
        assert read_file.invoke({"path": "workspace/note"}) == name
        assert (tmp_path / name / "note").read_text() == name


def test_workspace_prefix_is_component_not_string(tmp_path):
    write_file.invoke({"path": "workspace-other/note", "content": "safe"})
    assert (tmp_path / "workspace/workspace-other/note").read_text() == "safe"
    assert not (tmp_path / "workspace-other").exists()
    with pytest.raises(ValueError, match="Access denied"):
        validate_read_path("workspace-other/note")


@pytest.mark.parametrize("target_kind", ["file", "directory", "skills"])
def test_escaping_links_denied_for_all_file_tools(tmp_path, target_kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target_dir = tmp_path / ("skills" if target_kind == "skills" else "outside")
    target_dir.mkdir()
    target = target_dir / "note"
    target.write_text("outside content")
    link = workspace / "link"
    if target_kind == "directory":
        link.symlink_to(target_dir, target_is_directory=True)
        path = "workspace/link/note"
    else:
        link.symlink_to(target)
        path = "workspace/link"
    assert "Access denied" in read_file.invoke({"path": path})
    assert "Access denied" in write_file.invoke({"path": path, "content": "changed"})
    assert "Access denied" in edit_file.invoke({
        "path": path, "search_text": "outside", "replace_text": "changed"
    })
    assert target.read_text() == "outside content"
    assert get_workspace_files() == []
    assert "No matches" in search_files.invoke({"pattern": "outside"})


def test_internal_symlink_regular_file_is_allowed(tmp_path):
    (tmp_path / "workspace").mkdir()
    (tmp_path / "workspace/real").write_text("safe")
    (tmp_path / "workspace/link").symlink_to("real")
    assert read_file.invoke({"path": "workspace/link"}) == "safe"


def test_search_revalidates_enumerated_paths(tmp_path, monkeypatch):
    from src.core import tools
    (tmp_path / "workspace").mkdir()
    (tmp_path / "outside").write_text("do not open")
    (tmp_path / "workspace/link").symlink_to(tmp_path / "outside")
    monkeypatch.setattr(tools, "get_workspace_files", lambda: ["link", "../outside"])
    assert "No matches" in search_files.invoke({"pattern": "do not open"})


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_non_regular_targets_rejected(tmp_path, kind):
    (tmp_path / "workspace").mkdir()
    target = tmp_path / "workspace/target"
    target.mkdir() if kind == "directory" else os.mkfifo(target)
    assert "Access denied" in read_file.invoke({"path": "workspace/target"})
    assert "Access denied" in write_file.invoke({"path": "workspace/target", "content": "no"})
    assert "Access denied" in edit_file.invoke({
        "path": "workspace/target", "search_text": "x", "replace_text": "y"
    })
    assert get_workspace_files() == []


def test_parent_link_with_dotdot_is_not_normalized_away(tmp_path):
    (tmp_path / "workspace").mkdir()
    (tmp_path / "outside/sub").mkdir(parents=True)
    (tmp_path / "workspace/link").symlink_to(tmp_path / "outside/sub")
    with pytest.raises(ValueError, match="Access denied"):
        validate_and_normalize_path("workspace/link/../note", must_be_in_workspace=True)


def test_memory_and_skills_obey_read_policy(tmp_path):
    (tmp_path / "outside").write_text("not shared")
    (tmp_path / "AGENTS.md").symlink_to(tmp_path / "outside")
    (tmp_path / "skills/example").mkdir(parents=True)
    (tmp_path / "skills/example/SKILL.md").symlink_to(tmp_path / "outside")
    assert get_memory_content() == ""
    assert get_skill_body("example") == ""
    (tmp_path / "AGENTS.md").unlink()
    (tmp_path / "AGENTS.md").write_text("shared")
    assert get_memory_content() == "shared"


def test_clear_keeps_nested_files_and_never_follows_links(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ROOT", "custom")
    (tmp_path / "custom/sub").mkdir(parents=True)
    (tmp_path / "custom/sub/nested").write_text("keep existing clear semantics")
    (tmp_path / "custom/top").write_text("remove")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside/note").write_text("untouched")
    (tmp_path / "custom/link").symlink_to(tmp_path / "outside", target_is_directory=True)
    clear_workspace()
    assert not (tmp_path / "custom/top").exists()
    assert not (tmp_path / "custom/link").is_symlink()
    assert (tmp_path / "outside/note").read_text() == "untouched"
    assert (tmp_path / "custom/sub/nested").exists()


def test_ui_helpers_share_policy_without_starting_streamlit(tmp_path, monkeypatch):
    # Extract only the helpers: importing app would run the UI and graph.
    source = Path(__file__).resolve().parents[1] / "app.py"
    module = ast.parse(source.read_text())
    helpers = ast.Module(body=[
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"update_workspace_files", "read_workspace_bytes"}
    ], type_ignores=[])
    state = SimpleNamespace(workspace_files=[])
    namespace = {
        "st": SimpleNamespace(session_state=state),
        "get_workspace_files": get_workspace_files,
        "get_workspace_root": get_workspace_root,
        "validate_read_path": validate_read_path,
    }
    exec(compile(helpers, str(source), "exec"), namespace)
    monkeypatch.setenv("WORKSPACE_ROOT", "custom")
    (tmp_path / "custom").mkdir()
    (tmp_path / "custom/note").write_bytes(b"preview and download")
    namespace["update_workspace_files"]()
    assert state.workspace_files == ["note"]
    assert namespace["read_workspace_bytes"]("note") == b"preview and download"
    (tmp_path / "outside").write_bytes(b"denied")
    (tmp_path / "custom/note").unlink()
    (tmp_path / "custom/note").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="Access denied"):
        namespace["read_workspace_bytes"]("note")


def test_graph_initialization_uses_custom_workspace(tmp_path, monkeypatch):
    from src.core import agent_factory
    monkeypatch.setenv("WORKSPACE_ROOT", "custom")
    monkeypatch.setattr(agent_factory, "_compiled_graph", None)
    agent_factory.get_deep_agent()
    assert (tmp_path / "custom").is_dir()
    assert not (tmp_path / "workspace").exists()
