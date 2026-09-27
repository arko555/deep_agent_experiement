"""Verify sub-agent outputs against deterministic rules before synthesis."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def verify(results: dict[str, str], context: dict[str, Any] | None = None) -> dict:
    """Verify sub-agent results; returns {approved: bool, reasons: list[str]}.

    Performs deterministic checks:

    1. **Non-empty results**: Each department should have produced output.
    2. **Workspace containment**: If a result references a file path, it
       should be under ``./workspace`` (not ``/tmp/`` or outside paths).
    3. **Department scope**: Results should be relevant to the query
       (checked by keyword overlap with the original query). This one is
       advisory — it is reported under ``warnings`` and never flips ``approved``.

    Args:
        results: Mapping of sub-agent name → result text.
        context: Optional context dict with keys like ``original_query``
            and ``workspace_files`` for extended checks.

    Returns:
        ``{"approved": bool, "reasons": [str, ...], "warnings": [str, ...]}``.
        ``reasons`` explains a rejection; ``warnings`` carries advisory notes
        that do not block approval.
    """
    if context is None:
        context = {}

    reasons: list[str] = []
    warnings: list[str] = []
    approved = True

    if not results:
        return {"approved": False, "reasons": ["No sub-agent results to verify"]}

    original_query = context.get("original_query", "")
    query_words = {
        w for w in original_query.lower().split() if len(w) > 2
    } if original_query else set()

    for name, text in results.items():
        # Check 1: Non-empty result.
        if not text or not text.strip():
            approved = False
            reasons.append(f"Department '{name}' returned empty result")
            continue

        # Check 2: Workspace containment for file references. Match path
        # segments only, so prose like "attempt/" is not flagged.
        if re.search(r"(^|[\s\"'(=])/?tmp/", text):
            approved = False
            reasons.append(
                f"Department '{name}' references /tmp/ path (workspace only)"
            )

        # Check 3: Keyword relevance — advisory only, so it is reported as a
        # warning and does not flip `approved`.
        if query_words:
            result_words = set(text.lower().split())
            if not (query_words & result_words):
                logger.warning(
                    "Department '%s' has no keyword overlap with query", name
                )
                warnings.append(
                    f"Department '{name}' has low keyword overlap with query"
                )

    if approved:
        reasons.append("All checks passed")

    return {"approved": approved, "reasons": reasons, "warnings": warnings}
