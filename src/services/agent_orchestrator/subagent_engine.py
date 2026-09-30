"""Parallel sub-agent dispatch.

Batching only: it splits the requested departments into batches of
``MAX_PARALLEL_TASKS``, runs each batch under ``asyncio.gather`` with a shared
deadline, and merges the results. The per-department work — SKILL.md prompt,
tool shortlisting, and the ReAct loop — lives in
``subagents.run_department``, shared with the ``task`` tool so both entry
points behave identically.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from src.config import get_max_parallel_tasks, get_subagent_timeout_seconds

logger = logging.getLogger(__name__)


@dataclass
class SubagentRun:
    """One department's outcome, including what it cost and wrote.

    ``usage`` and ``writes`` used to be aggregated by the top-level tools node,
    which the pure-router hierarchy removed. Carrying them here keeps the
    parent's ``token_usage`` and ``pending_writes`` accounting — and so the
    Streamlit audit trail — fed by sub-agents rather than going empty when the
    tools node went away.
    """

    text: str
    usage: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    writes: list[dict[str, Any]] = field(default_factory=list)


class SubAgentEngine:
    """Batches department sub-agents and runs them concurrently.

    It does not decide what tools a department may use. That is
    ``subagents.select_department_tools``: the SKILL.md ``allowed-tools``
    allowlist, capped, then shortlisted by the LLM for the query at hand.

    This class used to hold a ``ToolRegistry`` and call
    ``get_visible_tools(name)`` to resolve each department's tools by role.
    Nothing consumed the result — ``_run_subagent_loop`` derives tools from
    the registered spec — so it was a second, weaker filter running beside the
    real one, and building the registry in the constructor loaded every MCP
    tool on every dispatch for a value that was only ever read back out.
    Removing it leaves ``select_department_tools`` the one place a
    sub-agent's tools are decided.

    All sub-agents in a batch share a common deadline.
    """

    async def invoke_parallel(
        self,
        subagents: list[dict[str, Any]],
        enhanced_query: str,
    ) -> dict[str, SubagentRun]:
        """Invoke multiple sub-agents in parallel.

        Sub-agents beyond ``MAX_PARALLEL_TASKS`` are run in sequential
        batches rather than dropped, so every requested department is
        represented in the returned dict.

        Args:
            subagents: List of sub-agent specs, each a dict with keys
                ``name`` (str) and optionally ``system_prompt`` (str) and
                ``description``. ``system_prompt`` is ignored — the
                department's SKILL.md is the source of its prompt.
            enhanced_query: The context-enriched query to pass to each.

        Returns:
            Dict mapping sub-agent name → its :class:`SubagentRun`.
        """
        deadline = get_subagent_timeout_seconds()
        max_parallel = get_max_parallel_tasks()

        async def _run_subagent(spec: dict[str, Any]) -> tuple[str, SubagentRun]:
            name = spec.get("name", "unknown")
            system_prompt = spec.get("system_prompt", "")
            description = spec.get("description", enhanced_query)

            try:
                run = await asyncio.wait_for(
                    self._run_subagent_loop(
                        name=name,
                        system_prompt=system_prompt,
                        description=description,
                        enhanced_query=enhanced_query,
                    ),
                    timeout=deadline,
                )
                return name, run
            except TimeoutError:
                logger.error("Sub-agent '%s' timed out after %ss", name, deadline)
                return name, SubagentRun(
                    text=f"Error: sub-agent '{name}' timed out after {deadline}s"
                )
            except Exception as e:
                logger.error("Sub-agent '%s' failed: %s", name, e)
                return name, SubagentRun(text=f"Error: {e}")

        results: dict[str, SubagentRun] = {}
        specs = list(subagents)

        # A department whose SKILL.md sets `parallelizable: false` runs alone.
        # The top-level tools node used to make this split; with the router
        # having no tools that node is gone, so honouring the flag here is what
        # keeps the frontmatter meaningful rather than silently ignored.
        from src.services.agent_orchestrator.subagents import is_parallelizable

        parallel: list[dict[str, Any]] = []
        serial: list[dict[str, Any]] = []
        for spec in specs:
            name = spec.get("name", "unknown")
            (parallel if is_parallelizable(name) else serial).append(spec)

        if serial and parallel:
            logger.info(
                "Running %d parallel department(s) first; %d marked "
                "parallelizable: false will run after",
                len(parallel), len(serial),
            )

        for group in (parallel, serial):
            # Batches of max_parallel, gathered concurrently, batches in order.
            for start in range(0, len(group), max_parallel):
                batch = group[start:start + max_parallel]
                if len(group) > max_parallel:
                    logger.info(
                        "Dispatching sub-agents %d-%d of %d",
                        start, start + len(batch) - 1, len(group),
                    )
                results.update(
                    dict(await asyncio.gather(*[_run_subagent(s) for s in batch]))
                )
        return results

    async def _run_subagent_loop(
        self,
        name: str,
        system_prompt: str,
        description: str,
        enhanced_query: str,
    ) -> SubagentRun:
        """Run one department through the executor its kind calls for.

        This used to be a second, divergent implementation: a single model
        call with tool descriptions pasted into the prompt as JSON text, so
        the "ReAct loop" could never produce a tool call and any tool output
        it appeared to have was hallucinated. It now delegates to
        ``subagents.run_department``, which loads the department's SKILL.md,
        shortlists its tool allowlist for this query, and runs a real ReAct
        loop with those tools bound.

        ``kind`` picks the executor. A ``tool_loop`` department runs locally; an
        ``a2a`` one is a remote agent, which the caller reaches by URL. The
        ``task`` tool used to be the sole dispatcher for both — and its
        deletion left remote agents with no route at all, since a spec with no
        ``skill`` has no prompt to run a tool loop with. Remote agents are
        sub-agents, so they route like any other department.

        ``system_prompt`` is accepted and ignored: the department's registered
        spec is the source of its prompt, so a caller-supplied one could only
        disagree with the SKILL.md it claims to come from.

        ``run_department`` and ``call_a2a_agent`` are blocking (sync LLM, tool
        calls, HTTP), so they are offloaded to a worker thread — otherwise they
        block the event loop and ``asyncio.gather`` degenerates into serial
        execution.
        """
        from src.services.agent_orchestrator.subagents import (
            run_department,
            SUBAGENTS,
        )

        spec = SUBAGENTS.get(name)
        if spec is None:
            return SubagentRun(
                text=(
                    f"Error: unknown department '{name}'. "
                    f"Available: {', '.join(sorted(SUBAGENTS)) or 'none'}."
                )
            )

        try:
            if spec.kind == "a2a":
                from src.services.tools_integration.a2a_client import call_a2a_agent

                if not spec.url:
                    # `_a2a_specs` always sets one, but a hand-built spec need
                    # not, and a missing URL must not become a call to None.
                    raise ValueError(f"A2A department '{name}' has no url configured")
                text = await asyncio.to_thread(
                    call_a2a_agent, spec.url, enhanced_query or description
                )
                # A remote agent reports no local token usage and writes no
                # local files; its own spend is its business, not ours.
                return SubagentRun(text=str(text))
            text, usage, writes = await asyncio.to_thread(
                run_department, spec, enhanced_query or description
            )
            return SubagentRun(text=str(text), usage=usage, writes=writes)
        except Exception as e:
            logger.error("Department '%s' failed: %s", name, e)
            return SubagentRun(text=f"Error: {e}")


async def invoke_parallel(
    subagents: list[dict[str, Any]],
    enhanced_query: str,
) -> dict[str, SubagentRun]:
    """Module-level convenience function for parallel sub-agent dispatch."""
    return await SubAgentEngine().invoke_parallel(subagents, enhanced_query)
