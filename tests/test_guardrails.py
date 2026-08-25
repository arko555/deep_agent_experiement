"""Tests for guardrails — path validation and normalization."""

import pytest

from src.core.guardrails import validate_and_normalize_path


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
