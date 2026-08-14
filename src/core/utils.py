"""Utility helpers for deep agent."""

import time


def invoke_with_retry(model_with_tools, messages, max_retries=3, base_delay=2.0):
    """Invoke an LLM with exponential backoff retry on transient failures."""
    for attempt in range(max_retries):
        try:
            return model_with_tools.invoke(messages)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
    raise RuntimeError("Exhausted all retries")
