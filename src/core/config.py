"""Runtime configuration for the deep agent.

Single source of truth for budgets and limits, so the top-level orchestrator
and delegated subagents share the same values. Every setting can be overridden
via environment variable without code changes; values are read at call time
so a process can pick up `.env` edits on the next invocation.
"""

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_max_iterations() -> int:
    """Per-turn orchestrator iteration budget (shared by parent and subagents)."""
    return _env_int("AGENT_MAX_ITERATIONS", 25)


def get_max_parallel_tasks() -> int:
    """Concurrency cap for independent research/writer subagent tasks."""
    return max(1, _env_int("MAX_PARALLEL_TASKS", 4))


def get_subagent_timeout_seconds() -> float:
    """Wall-clock guard for a single subagent task before the parent moves on."""
    try:
        return float(os.getenv("SUBAGENT_TIMEOUT_SECONDS", "600"))
    except (TypeError, ValueError):
        return 600.0
