"""Runtime configuration for the deep agent.

Single source of truth for budgets and limits, so the top-level orchestrator
and delegated subagents share the same values. Every setting can be overridden
via environment variable without code changes; values are read at call time
so a process can pick up `.env` edits on the next invocation.
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# Server/agent names are interpolated into tool names (`<server>_<tool>`) and
# into the `task` tool's subagent_type enum. LLM providers reject names outside
# this alphabet, so invalid keys are dropped rather than passed through.
_VALID_NAME = re.compile(r"^[a-z0-9_-]+$")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_json(name: str, default: dict) -> dict:
    """Parse a JSON-object env var, falling back to *default* on unset/invalid."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as e:
        logger.warning("%s is not valid JSON (%s); ignoring it.", name, e)
        return default
    if not isinstance(value, dict):
        logger.warning("%s must be a JSON object; ignoring it.", name)
        return default
    return value


def get_max_iterations() -> int:
    """Per-turn orchestrator iteration budget (shared by parent and subagents)."""
    return _env_int("AGENT_MAX_ITERATIONS", 25)


def get_max_subagent_depth() -> int:
    """Maximum subagent nesting level for `task` delegation.

    Semantics: a parent at depth D spawns children at D+1, and delegation is
    rejected when the parent's depth >= this limit. So the deepest subagent
    sits at depth == limit and cannot delegate further — a top-level agent
    (depth 0) can nest `limit` levels of subagents deep.
    """
    return _env_int("AGENT_MAX_SUBAGENT_DEPTH", 3)


def get_max_parallel_tasks() -> int:
    """Concurrency cap for independent research/writer subagent tasks."""
    return max(1, _env_int("MAX_PARALLEL_TASKS", 4))


def get_subagent_timeout_seconds() -> float:
    """Wall-clock guard for a single subagent task before the parent moves on."""
    try:
        return float(os.getenv("SUBAGENT_TIMEOUT_SECONDS", "600"))
    except (TypeError, ValueError):
        return 600.0


def get_mcp_servers() -> dict:
    """MCP server connections, as `langchain-mcp-adapters` expects them.

    Read from the `MCP_SERVERS` env var (JSON object). Each value is one
    connection, e.g.::

        MCP_SERVERS='{"math": {"transport": "stdio", "command": "python",
                               "args": ["/abs/path/server.py"]},
                      "weather": {"transport": "http",
                                  "url": "http://localhost:8000/mcp"}}'

    Keys become a prefix on every tool the server exposes
    (`<server>_<tool>`, via ``tool_name_prefix=True``), so they must match
    ``[a-z0-9_-]+``; invalid keys are dropped with a warning.

    Read at call time, so `.env` edits apply on the next invocation.
    """
    servers = _env_json("MCP_SERVERS", {})
    valid = {}
    for name, connection in servers.items():
        if not _VALID_NAME.match(name):
            logger.warning(
                "MCP server name '%s' is invalid (use [a-z0-9_-]+); ignoring it.", name
            )
            continue
        if not isinstance(connection, dict):
            logger.warning("MCP server '%s' config must be an object; ignoring it.", name)
            continue
        valid[name] = connection
    return valid


def get_a2a_agents() -> dict:
    """Remote A2A agents to expose as subagents, keyed by subagent type name.

    Read from the `A2A_AGENTS` env var (JSON object), e.g.::

        A2A_AGENTS='{"remote_researcher": {"url": "http://localhost:9999",
                                           "description": "Remote research agent"}}'

    Unlike the other getters this is consumed once at import time (see
    ``subagents.py``), because it feeds the `task` tool's static type enum.

    Keys must match ``[a-z0-9_-]+``; entries without a usable ``url`` are
    dropped with a warning.
    """
    agents = _env_json("A2A_AGENTS", {})
    valid = {}
    for name, config in agents.items():
        if not _VALID_NAME.match(name):
            logger.warning(
                "A2A agent name '%s' is invalid (use [a-z0-9_-]+); ignoring it.", name
            )
            continue
        if not isinstance(config, dict) or not isinstance(config.get("url"), str):
            logger.warning("A2A agent '%s' needs a string 'url'; ignoring it.", name)
            continue
        valid[name] = config
    return valid
