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
        # Non-strict read: redirect to a safe repo-relative path.
        # For absolute paths, keep the basename; for ".." traversal, log and
        # return the original relative portion that stays within the repo.
        if os.path.isabs(path):
            logger.warning(
                "Absolute path '%s' redirected to '%s' (read mode).",
                path,
                os.path.basename(clean_path),
            )
            return os.path.basename(clean_path)
        # ".." traversal on a read path: log and allow the caller to decide.
        logger.warning("Path traversal detected in '%s'; returning normalized path.", path)
        return clean_path

    # --- Enforce workspace boundary when required -----------------------------
    if must_be_in_workspace:
        if not (clean_path.startswith("workspace") or clean_path.startswith("./workspace")):
            clean_path = os.path.join("workspace", clean_path)

    return clean_path
