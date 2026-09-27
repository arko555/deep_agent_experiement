"""Parallel sub-agent dispatch and ReAct loop execution."""

import asyncio
import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_max_parallel_tasks, get_subagent_timeout_seconds
from src.services.tools_integration.registry import ToolRegistry
from src.services.tools_integration.relevance import sort_tools
from src.utils import get_message_text, invoke_with_retry

logger = logging.getLogger(__name__)


class SubAgentEngine:
    """Manages parallel invocation of department sub-agents.

    For each sub-agent:
    1. Get visible tools (≤20) from the registry.
    2. Sort to 3-5 relevant tools via LLM relevance sorter.
    3. Build input: system prompt + tool defs + enhanced query.
    4. Run ReAct loops in parallel via asyncio.gather.

    All sub-agents in a batch share a common deadline.
    """

    def __init__(self, registry: ToolRegistry | None = None):
        # Must be the *populated* registry: a bare ToolRegistry() has no specs,
        # which would leave every sub-agent with zero tools.
        if registry is None:
            from src.services.tools_integration.tools import create_tool_registry

            registry = create_tool_registry()
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def _tool_defs_for(self, name: str, reg: ToolRegistry) -> list[dict[str, Any]]:
        """Resolve the tools a sub-agent may see (≤20, per SKILL.md limit)."""
        try:
            visible = reg.get_visible_tools(name)
        except Exception as e:  # unknown role / malformed spec — not fatal
            logger.warning("Tool visibility lookup failed for '%s': %s", name, e)
            return []
        return reg.get_tool_definitions([spec.name for spec in visible])

    async def invoke_parallel(
        self,
        subagents: list[dict[str, Any]],
        enhanced_query: str,
        tools_registry: ToolRegistry | None = None,
    ) -> dict[str, str]:
        """Invoke multiple sub-agents in parallel.

        Sub-agents beyond ``MAX_PARALLEL_TASKS`` are run in sequential
        batches rather than dropped, so every requested department is
        represented in the returned dict.

        Args:
            subagents: List of sub-agent specs, each a dict with keys:
                ``name`` (str), ``system_prompt`` (str), and optionally
                ``tool_registry`` (list of tool defs) and ``description``.
            enhanced_query: The context-enriched query to pass to each.
            tools_registry: Optional ToolRegistry for visibility checks.
                Falls back to self._registry.

        Returns:
            Dict mapping sub-agent name → result text.
        """
        reg = tools_registry or self._registry
        deadline = get_subagent_timeout_seconds()
        max_parallel = get_max_parallel_tasks()

        async def _run_subagent(spec: dict[str, Any]) -> tuple[str, str]:
            name = spec.get("name", "unknown")
            system_prompt = spec.get("system_prompt", "")
            # An explicit tool list on the spec wins; otherwise ask the registry.
            tool_defs = spec.get("tool_registry") or self._tool_defs_for(name, reg)
            description = spec.get("description", enhanced_query)

            try:
                result = await asyncio.wait_for(
                    self._run_subagent_loop(
                        name=name,
                        system_prompt=system_prompt,
                        description=description,
                        tool_defs=tool_defs,
                        enhanced_query=enhanced_query,
                    ),
                    timeout=deadline,
                )
                return name, result
            except TimeoutError:
                logger.error("Sub-agent '%s' timed out after %ss", name, deadline)
                return name, f"Error: sub-agent '{name}' timed out after {deadline}s"
            except Exception as e:
                logger.error("Sub-agent '%s' failed: %s", name, e)
                return name, f"Error: {e}"

        results: dict[str, str] = {}
        specs = list(subagents)
        # Batches of max_parallel, gathered concurrently, batches run in order.
        for start in range(0, len(specs), max_parallel):
            batch = specs[start:start + max_parallel]
            if start:
                logger.info(
                    "Dispatching sub-agents %d-%d of %d",
                    start, start + len(batch) - 1, len(specs),
                )
            results.update(dict(await asyncio.gather(*[_run_subagent(s) for s in batch])))
        return results

    async def _run_subagent_loop(
        self,
        name: str,
        system_prompt: str,
        description: str,
        tool_defs: list[dict[str, Any]],
        enhanced_query: str,
    ) -> str:
        """Run a single sub-agent ReAct loop with tool relevance sorting.

        ``invoke_with_retry`` is a blocking sync call, so it is offloaded to
        a worker thread — otherwise it blocks the event loop and
        ``asyncio.gather`` degenerates into serial execution.
        """
        model = self._get_model()
        if not tool_defs:
            sorted_tools: list[dict[str, Any]] = []
        else:
            sorted_tools = await asyncio.to_thread(
                sort_tools, model, enhanced_query, tool_defs, 5
            )

        tool_section = (
            f"\nAvailable tools: {json.dumps(sorted_tools, indent=2)}"
            if sorted_tools
            else ""
        )
        full_prompt = (
            f"{system_prompt}{tool_section}\n\nQuery: {description}\n\n"
            f"Enhanced query: {enhanced_query}\n\nRespond comprehensively."
        )

        # Step 4: single-call sub-agent turn (no history, per the sub-agent
        # context contract: enhanced query + system prompt + tool defs only).
        response = await asyncio.to_thread(
            invoke_with_retry,
            model,
            [SystemMessage(content=full_prompt), HumanMessage(content=description)],
        )
        return get_message_text(getattr(response, "content", response))

    def _get_model(self):
        """Lazy import to avoid circular dependencies."""
        from src.services.agent_orchestrator.agent_factory import get_model
        return get_model()


async def invoke_parallel(
    subagents: list[dict[str, Any]],
    enhanced_query: str,
    tools_registry: ToolRegistry | None = None,
) -> dict[str, str]:
    """Module-level convenience function for parallel sub-agent dispatch."""
    engine = SubAgentEngine(tools_registry)
    return await engine.invoke_parallel(subagents, enhanced_query, tools_registry)
