import os
from typing import Optional

from src.services.tools_integration.guardrails import validate_read_path

__all__ = [
    "get_skill_info",
    "get_skill_body",
    "get_memory_content",
]


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
