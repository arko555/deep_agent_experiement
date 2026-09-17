"""Utility helpers for deep agent."""

import time


# Substrings (lowercased) that mark a failure as transient — worth retrying
# with backoff. Anything else (bad arguments, validation errors, missing
# files) is deterministic and should fail fast instead of burning ~14s of
# sleeps on retries that can never succeed.
_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "rate limit",
    "ratelimit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "server error",
    "temporarily",
)


def is_transient_error(exc: Exception) -> bool:
    """Heuristic classification: network/transient failures retry, the rest don't."""
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(marker in haystack for marker in _TRANSIENT_MARKERS)


def get_message_text(content) -> str:
    """Extract text from multimodal content (string or list of dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
    return str(content)


def invoke_with_retry(model_with_tools, messages, max_retries=3, base_delay=2.0):
    """Invoke an LLM or tool with exponential backoff on transient failures.

    Non-transient errors (bad args, validation, missing files) are re-raised
    immediately so deterministic failures don't pay the retry cost.
    """
    for attempt in range(max_retries):
        try:
            return model_with_tools.invoke(messages)
        except Exception as e:
            if attempt == max_retries - 1 or not is_transient_error(e):
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError("Exhausted all retries")
