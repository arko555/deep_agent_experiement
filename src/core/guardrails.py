import os
import logging

logger = logging.getLogger(__name__)


def validate_and_normalize_path(
    path: str,
    must_be_in_workspace: bool = False,
    strict: bool = False,
) -> str:
    """
    Validates and normalizes paths to prevent path traversal outside the repository.

    When ``must_be_in_workspace=True``, the path is forced into ``workspace/``.
    When ``strict=True``, traversal or absolute paths raise ``ValueError``.
    When ``strict=False`` (default), suspicious paths are silently redirected:
    - absolute paths → repo-relative basename
    - ``..`` traversal → stripped to a safe component

    Args:
        path: The file path to validate.
        must_be_in_workspace: If True, forces path to stay within workspace/.
        strict: If True, raises ValueError on traversal or absolute paths instead
            of silently redirecting.

    Returns:
        The normalized, safe path.

    Raises:
        ValueError: If ``strict=True`` and the path contains traversal or is absolute.
    """
    clean_path = os.path.normpath(path)

    # Strict mode rejects any traversal attempt up front, even when normpath
    # would collapse it (e.g. "a/../b" normalizes to "b").
    if strict and ".." in path.replace("\\", "/").split("/"):
        raise ValueError(
            f"Access denied to path '{path}'. "
            "Paths must be relative and stay within the repository."
        )

    # --- Handle traversal / absolute paths consistently -----------------------
    if clean_path.startswith("..") or os.path.isabs(clean_path):
        if must_be_in_workspace:
            base = os.path.basename(clean_path)
            return os.path.join("workspace", base)
        if strict:
            raise ValueError(
                f"Access denied to path '{path}'. "
                "Paths must be relative and stay within the repository."
            )
        # Non-strict read: redirect to a safe repo-relative basename,
        # consistent with the write path's fallback.
        logger.warning(
            "Suspicious path '%s' redirected to '%s' (read mode).",
            path,
            os.path.basename(clean_path),
        )
        return os.path.basename(clean_path)

    # --- Enforce workspace boundary when required -----------------------------
    if must_be_in_workspace:
        if not (clean_path.startswith("workspace") or clean_path.startswith("./workspace")):
            clean_path = os.path.join("workspace", clean_path)

    return clean_path


# Default-deny read allowlist (7.3): paths (repo-relative, normalized) that
# read_file may touch. Symmetric with the write sandbox, which is workspace-only.
READ_ALLOWLIST_PREFIXES = ("workspace/", "skills/")
READ_ALLOWLIST_FILES = {"AGENTS.md"}


def validate_read_path(path: str) -> str:
    """
    Validates a path for reading against the read allowlist (default-deny).

    Only paths that normalize to ``AGENTS.md``, or inside ``workspace/`` or
    ``skills/``, are allowed. Everything else raises — including absolute
    paths and ``..`` traversal (normpath collapses both before the check).

    Args:
        path: The file path to validate.

    Returns:
        The normalized path, safe to open.

    Raises:
        ValueError: If the path is outside the read allowlist.
    """
    clean_path = os.path.normpath(path)
    if clean_path in READ_ALLOWLIST_FILES or any(
        clean_path.startswith(prefix) for prefix in READ_ALLOWLIST_PREFIXES
    ):
        return clean_path
    raise ValueError(
        f"Access denied: reading '{path}' is not allowed. "
        "You can only read AGENTS.md, and files under ./workspace or ./skills."
    )
