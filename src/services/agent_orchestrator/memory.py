import os
from typing import Optional

from src.services.tools_integration.discovery import _dir_tree_hash
# `get_workspace_files` and `get_workspace_root` moved to guardrails, but three
# modules still import them from here. They stay in `__all__` until those
# importers are repointed, so the move is not a silent breaking change.
from src.services.tools_integration.guardrails import (
    get_workspace_files,
    get_workspace_root,
    validate_read_path,
)

__all__ = [
    "get_skill_info",
    "get_skill_body",
    "get_skills_summary",
    "get_memory_content",
    "get_workspace_files",
    "get_workspace_root",
]


# The skills summary is memoized on a hash of `skills/`, so a SKILL.md added
# mid-session is picked up without a restart. `_dir_tree_hash` is the one copy
# of that digest, in `tools_integration.discovery`; this module used to carry a
# second, subtly different version that did not sort `files`, so the same
# directory could hash two ways and the cache could miss an invalidation.
_skills_cache: Optional[str] = None
_skills_cache_hash: str = ""


def get_skill_info(skill_path: str) -> Optional[dict]:
    skill_md = os.path.join(skill_path, "SKILL.md")
    if os.path.exists(skill_md):
        try:
            skill_md = validate_read_path(skill_md)
            with open(skill_md, "r") as f:
                content = f.read()
                if content.startswith("---"):
                    parts = content.split("---")
                    if len(parts) >= 3:
                        header = parts[1]
                        info = {}
                        for line in header.split("\n"):
                            if ":" in line:
                                k, v = line.split(":", 1)
                                info[k.strip()] = v.strip()
                        return info
        except Exception:
            pass
    return None


def get_skill_body(skill_name: str) -> str:
    """Return the markdown body of ``skills/<skill_name>/SKILL.md`` with the
    frontmatter stripped. Returns "" if the file is missing or unreadable."""
    try:
        skill_md = validate_read_path(
            os.path.join("skills", skill_name, "SKILL.md")
        )
        with open(skill_md) as f:
            content = f.read()
    except (OSError, ValueError):
        return ""
    if content.startswith("---"):
        # Frontmatter is the first "---" block; rejoin the remainder so any
        # "---" lines in the body itself survive intact.
        parts = content.split("---")
        if len(parts) >= 3:
            content = "---".join(parts[2:])
    return content.strip()


def get_skills_summary() -> str:
    """Return a markdown summary of available skills (cached via mtime hash)."""
    global _skills_cache, _skills_cache_hash

    current_hash = _dir_tree_hash("./skills")
    if _skills_cache is not None and current_hash == _skills_cache_hash:
        return _skills_cache

    summary = []
    skills_dir = "./skills"
    if os.path.exists(skills_dir):
        for skill_name in os.listdir(skills_dir):
            skill_path = os.path.join(skills_dir, skill_name)
            if os.path.isdir(skill_path):
                info = get_skill_info(skill_path)
                if info:
                    summary.append(
                        f"- **{info.get('name', skill_name)}**: {info.get('description', '').strip()}"
                    )
    result = "\n".join(summary) if summary else "No specialized skills available."

    _skills_cache = result
    _skills_cache_hash = current_hash
    return result


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def get_memory_content() -> str:
    """Return ``AGENTS.md`` if it exists, else "".

    Appended to a sub-agent's system prompt alongside its SKILL.md body, so
    project conventions reach departments as well as the router.
    """
    path = "./AGENTS.md"
    if os.path.exists(path):
        try:
            path = validate_read_path(path)
            with open(path, "r") as f:
                return f.read()
        except Exception:
            pass
    return ""
