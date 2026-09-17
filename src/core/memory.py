import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from src.core.guardrails import get_workspace_root, validate_read_path


# ---------------------------------------------------------------------------
# Caching helpers (3.5: memoize with filesystem-change invalidation)
# ---------------------------------------------------------------------------

def _dir_tree_hash(directory: str) -> str:
    """Return a hex digest of (mtime, size) for every file under *directory*.

    ``__pycache__`` is ignored so bytecode written by dynamic module loading
    doesn't churn the hash.
    """
    parts = []
    if not os.path.exists(directory):
        return hashlib.md5(b"missing").hexdigest()
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            fp = os.path.join(root, fname)
            try:
                stat = os.stat(fp)
                parts.append(f"{fp}:{stat.st_mtime}:{stat.st_size}")
            except OSError:
                parts.append(f"{fp}:missing")
    return hashlib.md5("\n".join(parts).encode()).hexdigest() if parts else hashlib.md5(b"empty").hexdigest()


# Module-level cache stores
_skills_cache: Optional[str] = None
_skills_cache_hash: str = ""

_tools_cache: Optional[str] = None
_tools_cache_key: str = ""


# ---------------------------------------------------------------------------
# Workspace & Skills
# ---------------------------------------------------------------------------

def get_workspace_files() -> List[str]:
    """List workspace files relative to the configured root.

    Enumeration uses the shared call-time resolver and does not follow
    directory symlinks, so links out of the workspace are never listed.
    """
    workspace_dir = get_workspace_root()
    if not workspace_dir.is_dir():
        return []
    files = []
    for root, dirs, _filenames in os.walk(workspace_dir, followlinks=False):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        for f in _filenames:
            full = Path(root) / f
            # Files reached through a link chain must still be real files;
            # links themselves are skipped rather than followed.
            if full.is_symlink() or not full.is_file():
                continue
            try:
                validate_read_path(str(full))
            except (OSError, ValueError):
                continue
            files.append(full.relative_to(workspace_dir).as_posix())
    return sorted(files)


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
# Tools Summary
# ---------------------------------------------------------------------------

def get_tools_summary(tools_dict: Dict) -> str:
    """Return a markdown summary of available tools (cached via tool-name set + filesystem hash)."""
    global _tools_cache, _tools_cache_key

    # Build a cache key from the set of tool names plus a hash of the dynamic
    # tools directory, so edits to tool files (e.g. description changes)
    # invalidate the cache the same way the skills cache does.
    names = ",".join(sorted(tools_dict.keys())) if tools_dict else ""
    key = f"{names}|{_dir_tree_hash('./tools')}"
    if _tools_cache is not None and key == _tools_cache_key:
        return _tools_cache

    if not tools_dict:
        result = "No tools available."
    else:
        summary = []
        for name, tool_obj in tools_dict.items():
            description = getattr(tool_obj, "description", "No description provided.")
            summary.append(f"- **{name}**: {description}")
        result = "\n".join(summary)

    _tools_cache = result
    _tools_cache_key = key
    return result


# ---------------------------------------------------------------------------
# Memory & System Prompt
# ---------------------------------------------------------------------------

def get_memory_content() -> str:
    path = "./AGENTS.md"
    if os.path.exists(path):
        try:
            path = validate_read_path(path)
            with open(path, "r") as f:
                return f.read()
        except Exception:
            pass
    return ""


def get_system_prompt(tools_dict: Optional[Dict] = None) -> str:
    skills_summary = get_skills_summary()
    tools_summary = get_tools_summary(tools_dict) if tools_dict else "No tools available."

    prompt = f"""You are a generic Deep Agent, an expert orchestrator designed to perform any task.

**Conversational Guidance:**
- If the user sends a simple greeting (e.g., "hello", "hi", "good morning"), respond warmly and briefly, then ask how you can help.
- If the user asks a straightforward question that doesn't require tools, answer directly.
- For complex, multi-step tasks, follow the planning workflow below.

1. **Strategic Planning**: Use `write_todos` to map out your approach for complex requests.
2. **On-Demand Skills**: You have access to a library of skills in the `skills/` directory.
   - You ONLY see names and descriptions of skills in your system prompt initially.
   - For any specialized task (e.g. research, writing, coding), you MUST look for matching skills and use `read_file` to load the `SKILL.md` before executing.

Available Skills:
{skills_summary}

Available Tools:
{tools_summary}

3. **Subagents**: Use the `task` tool to delegate to specialized subagents
   (the task tool description lists the available types).
   - **Delegation file contract**: every task description MUST name a unique output
     path under `./workspace` for each file the subagent should produce
     (e.g., `workspace/<topic>.md`). Subagents may run in parallel, so never
     assign two tasks the same path.
   - The 'general-purpose' subagent is also generic and can load the same skills.
4. **NO /tmp/ FOLDER**: NEVER save files to the `/tmp/` directory. This is a critical requirement.
5. **STRICT Workspace Usage**: ALL file outputs, intermediate notes, and final reports MUST be written to the `./workspace/` directory exclusively. Use the `write_file` and `edit_file` tools to manage files within this directory.
6. **Isolated Context**: Use subagents to keep the main conversation thread clean and focused on high-level orchestration.
7. **Shared Data**: Refer to `AGENTS.md` for project conventions and mission statements.

Follow the instructions in the loaded SKILL.md exactly once they are retrieved."""

    agents_md = get_memory_content()
    if agents_md:
        prompt += f"\n\n=== Shared Data / AGENTS.md ===\n{agents_md}"

    return prompt
