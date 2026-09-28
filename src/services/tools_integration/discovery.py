"""Auto-discovery of sub-agents (skills/SKILL.md) and tools (tools/*.py @tool_spec)."""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Repository root, derived from this file's location rather than the process
# cwd. Default scan targets are anchored here so that a chdir — a test's
# tmp_path, or an embedding app that starts elsewhere — cannot silently make
# discovery return nothing. Relative to ``discovery.py`` this is
# src/services/tools_integration → src/services → src → <root>.
REPO_ROOT = Path(__file__).resolve().parents[3]


def default_skills_dir() -> str:
    """The shipped ``skills/`` directory, independent of the process cwd."""
    return str(REPO_ROOT / "skills")


def default_tools_dir() -> str:
    """The shipped ``tools/`` directory, independent of the process cwd."""
    return str(REPO_ROOT / "tools")

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

_skills_cache: list[dict[str, Any]] | None = None
_skills_cache_key: str = ""


def invalidate_discovery_cache() -> None:
    """Drop the cached sub-agent scan.

    The cache key is a tree hash of mtime+size, so it self-invalidates when a
    file changes. This exists for the case where a file is replaced with the
    same size and mtime granularity hides the edit — and so ``reset_deep_agent``
    has an explicit hook to call.
    """
    global _skills_cache, _skills_cache_key
    _skills_cache = None
    _skills_cache_key = ""


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
    """Parse a SKILL.md YAML frontmatter into a sub-agent definition.

    Frontmatter is parsed with ``yaml.safe_load`` so YAML features actually
    used by these files work: folded scalars (``description: >`` spanning
    several indented lines) and block scalars (``|``). A line-by-line parser
    silently stored the literal ``">"`` as the description and dropped every
    continuation line.
    """
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

    try:
        loaded = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as e:
        logger.warning("Invalid YAML frontmatter in %s: %s", filepath, e)
        return None
    if not isinstance(loaded, dict):
        logger.warning("Frontmatter in %s is not a mapping; ignoring", filepath)
        return None

    metadata: dict[str, Any] = {
        "name": str(loaded.get("name") or "").strip(),
        "description": " ".join(str(loaded.get("description") or "").split()),
        "department": str(loaded.get("department") or "").strip(),
        "allowed-tools": _parse_tool_list(loaded.get("allowed-tools")),
        "parallelizable": bool(loaded.get("parallelizable", True)),
        "protocol": str(loaded.get("protocol") or "react").strip(),
    }

    metadata["body"] = body
    metadata["skill_file"] = filepath
    return metadata


def _parse_tool_list(value: Any) -> list[str]:
    """Normalize an ``allowed-tools`` value to a list of tool names.

    Accepts a YAML list, or a string in either the comma-separated
    (``a, b, c``) or space-separated (``a b c``) form — both appear in the
    wild, and the previous comma-only split turned a space-separated list
    into a single bogus tool name.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [t for t in re.split(r"[,\s]+", str(value).strip()) if t]


def discover_subagents(skills_dir: str | None = None) -> list[dict[str, Any]]:
    """Scan ``skills/*/SKILL.md`` for sub-agent definitions.

    Returns a list of dicts with ``name``, ``department``,
    ``system_prompt``, ``allowed_tools``, and ``skill_file``.

    ``skills_dir`` defaults to the repository's ``skills/`` (see
    :func:`default_skills_dir`), not the process cwd.

    Results are cached keyed by the directory tree hash of
    ``skills_dir``; calls re-scan only when the directory
    contents change (mtime+size).
    """
    skills_dir = skills_dir or default_skills_dir()
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
            "description": metadata.get("description", ""),
            "department": metadata.get("department") or skill_dir.name,
            "system_prompt": system_prompt,
            "allowed_tools": metadata.get("allowed-tools", []),
            "parallelizable": metadata.get("parallelizable", True),
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

def discover_tools(tools_dir: str | None = None) -> list[dict[str, Any]]:
    """Scan ``tools/*.py`` for `@tool_spec` metadata tags.

    Each module's functions are inspected for a ``__tool_spec__``
    attribute set by the :func:`tool_spec` decorator.

    ``tools_dir`` defaults to the repository's ``tools/`` (see
    :func:`default_tools_dir`), not the process cwd.

    Returns a list of dicts with ``name``, ``description``,
    ``risk_level``, ``requires_approval``, and ``allowed_roles``.
    """
    tools_dir = tools_dir or default_tools_dir()
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
