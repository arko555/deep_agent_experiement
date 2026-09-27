"""Aggregate results from multiple department sub-agents into a final answer."""

import logging

logger = logging.getLogger(__name__)


def aggregate(results: dict[str, str], original_query: str) -> str:
    """Combine multi-department results into a coherent final answer.

    For a single department, returns the result directly. For multiple
    departments, produces a synthesized summary that references each
    department's contribution.

    Args:
        results: Mapping of sub-agent name → result text from
            ``SubAgentEngine.invoke_parallel``.
        original_query: The user's original query for context.

    Returns:
        A synthesized answer string.
    """
    if not results:
        return f"No department results available for: {original_query}"

    if len(results) == 1:
        name, text = next(iter(results.items()))
        logger.info("Single department '%s' — returning result directly", name)
        return text

    # Multi-department synthesis.
    parts = []
    for name, text in results.items():
        parts.append(f"## {name}\n\n{text}")

    return (
        f"Based on the analysis from {len(results)} departments "
        f"regarding: {original_query}\n\n"
        + "\n\n".join(parts)
    )
