"""Resolved-path file containment (not race-proof against concurrent mutation).

This lives in ``tools_integration`` because it is the layer the file tools
execute in: ``read_file``/``write_file``/``list_files`` are what need
containment, so a lower service must not have to import a higher one to get
it. ``agent_orchestrator`` imports down from here, never the reverse.
"""

import os
from pathlib import Path


def get_workspace_root() -> Path:
    """Return the canonical workspace root, reading WORKSPACE_ROOT at call time."""
    return Path(os.getenv("WORKSPACE_ROOT", "./workspace")).resolve()


def _display_path(path: Path) -> str:
    """Keep cwd-relative return paths where possible for compatibility."""
    return os.path.relpath(path) if path.is_relative_to(Path.cwd()) else str(path)


def _checked_file(path: Path, root: Path) -> str:
    """Check target and parents against their own authorized root.

    Every path component is resolved and must remain under *root*, so a
    symlink anywhere in the chain (including parent links) cannot move the
    target outside it; an existing target must be a regular file. Returned
    display paths stay cwd-relative for compatibility. This is resolved-path
    containment, not protection against adversarial filesystem changes between
    check and I/O (TOCTOU): the caller accepted that limitation.
    """
    try:
        path.relative_to(root)
        for component in (path, *path.parents):
            if not component.resolve().is_relative_to(root):
                raise ValueError("resolved path escapes its authorized root")
            if component == root:
                break
        resolved = path.resolve()
        if resolved.exists() and not resolved.is_file():
            raise ValueError("target is not a regular file")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Access denied to path '{path}': {exc}") from exc
    return _display_path(resolved)


def validate_and_normalize_path(
    path: str,
    must_be_in_workspace: bool = False,
    strict: bool = False,
) -> str:
    """Normalize paths, containing writes in the call-time workspace.

    Bare writes are workspace-relative; workspace/ is a logical alias even with
    WORKSPACE_ROOT configured. Non-strict out-of-root absolute/traversal writes
    retain the safe basename fallback. Strict mode rejects absolute/traversal
    inputs. Reads must use validate_read_path, not this normalization helper.
    """
    raw = Path(path)
    if strict and (raw.is_absolute() or ".." in path.replace("\\", "/").split("/")):
        raise ValueError(
            f"Access denied to path '{path}'. Paths must be relative without traversal."
        )
    clean = Path(os.path.normpath(path))
    if not must_be_in_workspace:
        if clean.is_absolute() or (clean.parts and clean.parts[0] == ".."):
            clean = Path(clean.name)
        if clean.parts and clean.parts[0] == "workspace":
            root = get_workspace_root()
            return _checked_file(root.joinpath(*clean.parts[1:]), root)
        return _checked_file(Path.cwd() / clean, Path.cwd())

    root = get_workspace_root()
    # Select the workspace alias before normpath can erase traversal through a
    # symlink. Every existing parent is checked by _checked_file.
    if raw.parts and raw.parts[0] == "workspace":
        target = root.joinpath(*raw.parts[1:])
    elif (Path.cwd() / raw).is_relative_to(root):
        target = Path.cwd() / raw
    elif clean.is_absolute() or (clean.parts and clean.parts[0] == ".."):
        target = root / clean.name
    else:
        target = root / raw
    return _checked_file(target, root)


def validate_read_path(path: str) -> str:
    """Default-deny reads: AGENTS.md, skills/, and the configured workspace.

    Root selection precedes resolution: a workspace link cannot grant access to
    skills or repository files. Absolute paths are accepted only in the workspace.
    """
    raw = Path(path)
    workspace = get_workspace_root()
    if raw.parts and raw.parts[0] == "workspace":
        return _checked_file(workspace.joinpath(*raw.parts[1:]), workspace)
    candidate = raw if raw.is_absolute() else Path.cwd() / raw
    if candidate.is_relative_to(workspace):
        return _checked_file(candidate, workspace)
    if raw.parts and raw.parts[0] == "skills":
        return _checked_file(Path.cwd() / raw, Path.cwd() / "skills")
    if raw == Path("AGENTS.md"):
        # A symlink must not turn this single-file exception into a new grant.
        target = Path.cwd() / raw
        if target.resolve() == target:
            return _checked_file(target, Path.cwd())
    raise ValueError(
        f"Access denied: reading '{path}' is not allowed. "
        "You can only read AGENTS.md, and files under ./workspace or ./skills."
    )


def clear_workspace() -> None:
    """Clear top-level files, preserving the UI's non-recursive behavior.

    Links are unlinked, never followed; directories and special files remain. Like the
    validators, this does not defend against concurrent filesystem mutation.
    """
    root = get_workspace_root()
    if not root.is_dir():
        return
    for entry in root.iterdir():
        if entry.is_symlink():
            entry.unlink()
        elif entry.is_file():
            _checked_file(entry, root)
            entry.unlink()


def get_workspace_files() -> list[str]:
    """List workspace files relative to the configured root.

    Enumeration uses the shared call-time resolver and does not follow
    directory symlinks, so links out of the workspace are never listed.

    Lives here rather than in ``agent_orchestrator.memory`` because it is
    workspace containment: the same concern as the validators above, and the
    ``list_files`` tool that calls it is in this service. Keeping it here lets
    ``tools_integration`` depend on nothing above it.
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
