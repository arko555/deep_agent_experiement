"""Text statistics tool — a dependency-free dynamic tool.

Discovered automatically by ``tools.load_dynamic_tools`` (which execs every
``.py`` under ``./tools`` and picks up ``@tool`` functions) and registered by
``tools.get_all_tools`` with precedence *built-ins > MCP > dynamic*, so the
name here must not collide with a built-in.

Use this as the reference for writing your own: decorate with ``@tool`` for the
LangChain schema, and stack ``@tool_spec`` underneath to declare risk and role
visibility. The docstring *is* the tool schema the model sees, so write it for
the model, not for yourself.
"""

import re
from collections import Counter

from langchain_core.tools import tool

from src.services.tools_integration.decorator import tool_spec


@tool
@tool_spec(
    name="text_stats",
    description=(
        "Compute word, character, sentence and line counts for a block of text, "
        "plus its most frequent words. Use for summarising, sizing, or checking "
        "how long something is."
    ),
    risk_level="low",
    allowed_roles=("*",),
)
def text_stats(text: str, top_n: int = 5) -> str:
    """Compute word, character, sentence and line counts for a block of text.

    Args:
        text: The text to measure.
        top_n: How many of the most frequent words to report.
    """
    if not text:
        return "No text supplied."

    words = re.findall(r"\b\w+\b", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    lines = [
        f"words: {len(words)}",
        f"characters (incl. spaces): {len(text)}",
        f"characters (excl. spaces): {len(text.replace(' ', ''))}",
        f"sentences: {len(sentences)}",
        f"lines: {len(text.splitlines()) or 1}",
    ]

    if words:
        # Drop a short English stop-word list so the frequency table carries
        # signal rather than being "the, and, of".
        stop = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
            "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
            "its", "new", "now", "old", "see", "two", "way", "who", "did", "yes",
        }
        counts = Counter(w.lower() for w in words if w.lower() not in stop and len(w) > 2)
        if counts:
            top = ", ".join(f"{w} ({c})" for w, c in counts.most_common(max(1, top_n)))
            lines.append(f"most frequent words: {top}")

    return "\n".join(lines)
