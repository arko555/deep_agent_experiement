"""Tests for guardrails — path validation and normalization."""

import pytest

from src.core.guardrails import validate_and_normalize_path, validate_read_path
from src.core.tools import read_file


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
