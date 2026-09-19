"""Auto-discovery of sub-agents (skills/SKILL.md) and tools (tools/*.py @tool_spec)."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

_skills_cache: list[dict[str, Any]] | None = None
_skills_cache_key: str = ""


def _dir_tree_hash(directory: str) -> str:
    """Hex digest of (mtime, size) for every file under *directory*."""
    if not os.path.exists(directory):
        return hashlib.md5(b"missing").hexdigest()
    parts = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in sorted(files):
            fp = os.path.join(root, fname)
            try:
                stat = os.stat(fp)
                parts.append(f"{fp}:{stat.st_mtime}:{stat.st_size}")
            except OSError:
                parts.append(f"{fp}:missing")
    return hashlib.md5("\n".join(parts).encode()).hexdigest() if parts else hashlib.md5(b"empty").hexdigest()


def _directory_tree_key(paths: list[str]) -> str:
    """Combine tree hashes for multiple directories."""
    return "|".join(f"{p}:{_dir_tree_hash(p)}" for p in sorted(paths))


# ---------------------------------------------------------------------------
# Sub-agent discovery from skills/SKILL.md
# ---------------------------------------------------------------------------

def _parse_skill_file(filepath: str) -> dict[str, Any] | None:
    """Parse a SKILL.md YAML frontmatter into a sub-agent definition."""
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return None

    if not content.strip():
        return None

    # Extract YAML front matter (between --- markers).
    if not content.startswith("---"):
        return None

    parts = content.split("---", 2)
    if len(parts) < 3:
        return None

    yaml_text = parts[1]
    body = parts[2].strip()

    metadata: dict[str, Any] = {
        "name": "",
        "description": "",
        "department": "",
        "allowed-tools": [],
        "protocol": "react",
    }
    for line in yaml_text.splitlines():
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key in ("name", "description", "department"):
            metadata[key] = value
        elif key == "allowed-tools":
            metadata[key] = [t.strip() for t in value.split(",") if t.strip()]
        elif key == "protocol":
            metadata[key] = value

    metadata["body"] = body
    metadata["skill_file"] = filepath
    return metadata


def discover_subagents(skills_dir: str = "./skills") -> list[dict[str, Any]]:
    """Scan ``skills/*/SKILL.md`` for sub-agent definitions.

    Returns a list of dicts with ``name``, ``department``,
    ``system_prompt``, ``allowed_tools``, and ``skill_file``.

    Results are cached keyed by the directory tree hash of
    ``skills_dir``; calls re-scan only when the directory
    contents change (mtime+size).
    """
    global _skills_cache, _skills_cache_key

    key = _directory_tree_key([skills_dir]) if os.path.exists(skills_dir) else ""
    if _skills_cache is not None and key == _skills_cache_key:
        return _skills_cache

    subagents: list[dict[str, Any]] = []
    skills_path = Path(skills_dir)
    if not skills_path.exists():
        logger.warning("Skills directory %s does not exist", skills_dir)
        _skills_cache = subagents
        _skills_cache_key = key
        return subagents

    for skill_dir in sorted(skills_path.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            continue

        metadata = _parse_skill_file(str(skill_file))
        if metadata is None:
            logger.warning("Skipping invalid skill file: %s", skill_file)
            continue

        # Build a simple system prompt from the skill body.
        system_prompt = (
            f"You are the {metadata.get('name', skill_dir.name)} sub-agent. "
            f"{metadata.get('body', '')}"
        )

        subagents.append({
            "name": metadata.get("name", skill_dir.name),
            "department": metadata.get("department") or skill_dir.name,
            "system_prompt": system_prompt,
            "allowed_tools": metadata.get("allowed-tools", []),
            "skill_file": str(skill_file),
            "protocol": metadata.get("protocol", "react"),
        })

    logger.info("Discovered %d subagent(s) from %s", len(subagents), skills_dir)
    _skills_cache = subagents
    _skills_cache_key = key
    return subagents


# ---------------------------------------------------------------------------
# Tool discovery from tools/*.py @tool_spec tags
# ---------------------------------------------------------------------------

def discover_tools(tools_dir: str = "./tools") -> list[dict[str, Any]]:
    """Scan ``tools/*.py`` for `@tool_spec` metadata tags.

    Each module's functions are inspected for a ``__tool_spec__``
    attribute set by the :func:`tool_spec` decorator.

    Returns a list of dicts with ``name``, ``description``,
    ``risk_level``, ``requires_approval``, and ``allowed_roles``.
    """
    tools: list[dict[str, Any]] = []
    tools_path = Path(tools_dir)
    if not tools_path.exists():
        return tools

    import importlib.util
    import inspect

    for py_file in sorted(tools_path.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue

        module_name = f"discovered_tool__{pyfile_to_module_name(py_file, tools_dir)}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(py_file))
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.warning("Failed to load tool module %s: %s", py_file.name, e)
            continue

        for name, obj in inspect.getmembers(module, inspect.isfunction):
            spec_meta = getattr(obj, "__tool_spec__", None)
            if spec_meta is None:
                continue
            tools.append({
                "name": spec_meta.name,
                "description": spec_meta.description,
                "risk_level": spec_meta.risk_level,
                "requires_approval": spec_meta.requires_approval,
                "allowed_roles": list(spec_meta.allowed_roles),
                "source_file": str(py_file),
            })

    return tools


def pyfile_to_module_name(filepath: Path, tools_dir: str) -> str:
    """Convert a file path under *tools_dir* to a module name."""
    rel = os.path.relpath(str(filepath), tools_dir)
    return rel[:-3].replace(os.sep, "__").replace("/", "__")
