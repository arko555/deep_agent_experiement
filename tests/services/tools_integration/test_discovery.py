"""Tests for discover_subagents and discover_tools (Phase 2.3)."""

import os
import pytest
from pathlib import Path

from src.services.tools_integration.discovery import (
    discover_subagents,
    discover_tools,
    _parse_skill_file,
    _dir_tree_hash,
    _directory_tree_key,
    pyfile_to_module_name,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# _parse_skill_file
# ---------------------------------------------------------------------------

class TestParseSkillFile:

    def test_valid_frontmatter(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            "---\n"
            "name: research\n"
            "description: Web research\n"
            "department: research\n"
            "allowed-tools: search_web, fetch_url\n"
            "protocol: react\n"
            "---\n"
            "You are the research sub-agent.\n",
        )
        result = _parse_skill_file(str(skill_file))
        assert result is not None
        assert result["name"] == "research"
        assert result["description"] == "Web research"
        assert result["department"] == "research"
        assert result["allowed-tools"] == ["search_web", "fetch_url"]
        assert result["protocol"] == "react"
        assert "You are the research sub-agent." in result["body"]
        assert result["skill_file"] == str(skill_file)

    def test_minimal_frontmatter(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("---\nname: writer\n---\nWrite things.\n")
        result = _parse_skill_file(str(skill_file))
        assert result["name"] == "writer"
        assert result["description"] == ""
        assert result["department"] == ""
        assert result["allowed-tools"] == []

    def test_missing_file_returns_none(self):
        assert _parse_skill_file("/nonexistent/SKILL.md") is None

    def test_empty_file_returns_none(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("")
        assert _parse_skill_file(str(skill_file)) is None

    def test_no_frontmatter_returns_none(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("Just text, no YAML.\n")
        assert _parse_skill_file(str(skill_file)) is None

    def test_allowed_tools_parsing(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            "---\nname: test\nallowed-tools: tool_a, tool_b, tool_c\n---\n",
        )
        result = _parse_skill_file(str(skill_file))
        assert result["allowed-tools"] == ["tool_a", "tool_b", "tool_c"]

    def test_whitespace_in_allowed_tools_is_stripped(self, tmp_path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(
            "---\nname: test\nallowed-tools: tool_a ,  tool_b \n---\n",
        )
        result = _parse_skill_file(str(skill_file))
        assert result["allowed-tools"] == ["tool_a", "tool_b"]


# ---------------------------------------------------------------------------
# discover_subagents
# ---------------------------------------------------------------------------

class TestDiscoverSubagents:

    def test_finds_skills_in_directory(self, tmp_path):
        skills_dir = tmp_path / "skills"
        research = skills_dir / "research"
        research.mkdir(parents=True)
        (research / "SKILL.md").write_text(
            "---\nname: research\ndepartment: research\n---\n"
            "Research skill body.\n",
        )
        writer = skills_dir / "writer"
        writer.mkdir()
        (writer / "SKILL.md").write_text(
            "---\nname: writer\n---\nWrite skill body.\n",
        )
        subagents = discover_subagents(str(skills_dir))
        names = {s["name"] for s in subagents}
        assert names == {"research", "writer"}

    def test_each_subagent_has_required_fields(self, tmp_path):
        skills_dir = tmp_path / "skills"
        dept = skills_dir / "hr"
        dept.mkdir(parents=True)
        (dept / "SKILL.md").write_text(
            "---\nname: hr_agent\ndepartment: hr\nallowed-tools: read_file\n"
            "---\nHR body.\n",
        )
        subagents = discover_subagents(str(skills_dir))
        assert len(subagents) == 1
        s = subagents[0]
        assert "name" in s
        assert "department" in s
        assert "system_prompt" in s
        assert "allowed_tools" in s
        assert "skill_file" in s
        assert "protocol" in s

    def test_system_prompt_contains_name_and_body(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        dept = skills_dir / "research"
        dept.mkdir(parents=True)
        (dept / "SKILL.md").write_text(
            "---\nname: research\n---\nFind facts.\n",
        )
        subagents = discover_subagents(str(skills_dir))
        prompt = subagents[0]["system_prompt"]
        assert "research" in prompt
        assert "Find facts." in prompt

    def test_empty_directory_returns_empty(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        assert discover_subagents(str(skills_dir)) == []

    def test_missing_directory_returns_empty(self, tmp_path):
        assert discover_subagents(str(tmp_path / "no_such")) == []

    def test_skips_non_directory_entries(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "README.md").write_text("not a skill")
        assert discover_subagents(str(skills_dir)) == []

    def test_skips_missing_SKILL_md(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        dept = skills_dir / "unknown"
        dept.mkdir(parents=True)
        assert discover_subagents(str(skills_dir)) == []

    def test_caching_returns_same_result(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        dept = skills_dir / "research"
        dept.mkdir(parents=True)
        (dept / "SKILL.md").write_text(
            "---\nname: research\n---\nBody.\n",
        )
        result1 = discover_subagents(str(skills_dir))
        result2 = discover_subagents(str(skills_dir))
        assert result1 is result2  # same cached object

    def test_cache_invalidates_on_change(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        dept = skills_dir / "research"
        dept.mkdir(parents=True)
        skill = dept / "SKILL.md"
        skill.write_text("---\nname: research\n---\nBody v1.\n")

        result1 = discover_subagents(str(skills_dir))
        assert result1[0]["name"] == "research"

        # Modify file (changes mtime+size → new tree hash)
        import time
        time.sleep(0.01)
        skill.write_text("---\nname: research\n---\nBody v2.\n")

        result2 = discover_subagents(str(skills_dir))
        assert result2 is not result1  # fresh result
        assert "Body v2" in result2[0]["system_prompt"]

    def test_department_falls_back_to_directory_name(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        dept = skills_dir / "marketing"
        dept.mkdir(parents=True)
        (dept / "SKILL.md").write_text("---\nname: mkt\n---\nBody.\n")
        subagents = discover_subagents(str(skills_dir))
        assert subagents[0]["department"] == "marketing"

    def test_default_protocol_is_react(self, tmp_path):
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir(parents=True)
        dept = skills_dir / "x"
        dept.mkdir(parents=True)
        (dept / "SKILL.md").write_text("---\nname: x\n---\nBody.\n")
        subagents = discover_subagents(str(skills_dir))
        assert subagents[0]["protocol"] == "react"

    def test_uses_repo_skills_by_default(self):
        """Default ./skills path points at repo root."""
        default = str(REPO_ROOT / "skills")
        if os.path.isdir(default):
            result = discover_subagents("./skills")
            assert isinstance(result, list)


# ---------------------------------------------------------------------------
# discover_tools
# ---------------------------------------------------------------------------

class TestDiscoverTools:

    def test_finds_tool_spec_functions(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "my_tool.py").write_text(
            "from src.services.tools_integration.decorator import tool_spec\n\n"
            "@tool_spec(name='fetch_data', description='Fetch data')\n"
            "def fetch_data():\n    return 'data'\n",
        )
        tools = discover_tools(str(tools_dir))
        assert len(tools) == 1
        assert tools[0]["name"] == "fetch_data"
        assert tools[0]["description"] == "Fetch data"

    def test_skips_modules_without_tool_spec(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "plain.py").write_text(
            "def plain(): pass\n",
        )
        assert discover_tools(str(tools_dir)) == []

    def test_skips_init_files(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "__init__.py").write_text(
            "from src.services.tools_integration.decorator import tool_spec\n"
            "@tool_spec(name='init_tool')\ndef init_tool(): pass\n",
        )
        assert discover_tools(str(tools_dir)) == []

    def test_skips_private_modules(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "_private.py").write_text(
            "from src.services.tools_integration.decorator import tool_spec\n"
            "@tool_spec(name='priv')\ndef priv(): pass\n",
        )
        # _private.py starts with underscore — should still be found since
        # the implementation uses rglob("*.py") with no name filter.
        # (This documents the current behavior; change if desired.)
        tools = discover_tools(str(tools_dir))
        assert len(tools) == 1

    def test_missing_directory_returns_empty(self, tmp_path):
        assert discover_tools(str(tmp_path / "no_tools")) == []

    def test_multiple_tools_from_same_file(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "multi.py").write_text(
            "from src.services.tools_integration.decorator import tool_spec\n\n"
            "@tool_spec(name='alpha', description='A')\n"
            "def alpha(): pass\n\n"
            "@tool_spec(name='beta', description='B')\n"
            "def beta(): pass\n",
        )
        tools = discover_tools(str(tools_dir))
        names = {t["name"] for t in tools}
        assert names == {"alpha", "beta"}

    def test_each_tool_has_source_file(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "src.py").write_text(
            "from src.services.tools_integration.decorator import tool_spec\n"
            "@tool_spec(name='x')\ndef x(): pass\n",
        )
        tools = discover_tools(str(tools_dir))
        assert tools[0]["source_file"].endswith("src.py")

    def test_risk_and_approval_metadata(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "secure.py").write_text(
            "from src.services.tools_integration.decorator import tool_spec\n"
            "@tool_spec(name='delete', description='Del', "
            "risk_level='high', requires_approval=True, "
            "allowed_roles=('admin',))\n"
            "def delete(): pass\n",
        )
        tools = discover_tools(str(tools_dir))
        t = tools[0]
        assert t["risk_level"] == "high"
        assert t["requires_approval"] is True
        assert t["allowed_roles"] == ["admin"]


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

class TestDirTreeHash:

    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")
        f2 = tmp_path / "b.txt"
        f2.write_text("world")
        h1 = _dir_tree_hash(str(tmp_path))
        h2 = _dir_tree_hash(str(tmp_path))
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello")
        h1 = _dir_tree_hash(str(tmp_path))
        import time
        time.sleep(0.01)
        f1.write_text("world")
        h2 = _dir_tree_hash(str(tmp_path))
        assert h1 != h2

    def test_missing_directory_returns_consistent_hash(self, tmp_path):
        h1 = _dir_tree_hash(str(tmp_path / "nope"))
        h2 = _dir_tree_hash(str(tmp_path / "nope"))
        assert h1 == h2

    def test_combined_key_is_deterministic(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "x.txt").write_text("1")
        (b / "y.txt").write_text("2")
        key1 = _directory_tree_key([str(a), str(b)])
        key2 = _directory_tree_key([str(b), str(a)])
        assert key1 == key2


class TestPyfileToModuleName:

    def test_simple_file(self, tmp_path):
        f = tmp_path / "tools" / "my_tool.py"
        f.parent.mkdir(parents=True)
        assert pyfile_to_module_name(f, str(tmp_path / "tools")) == "my_tool"

    def test_nested_file(self, tmp_path):
        f = tmp_path / "tools" / "subdir" / "deep.py"
        f.parent.mkdir(parents=True)
        result = pyfile_to_module_name(f, str(tmp_path / "tools"))
        assert "subdir" in result and "deep" in result
