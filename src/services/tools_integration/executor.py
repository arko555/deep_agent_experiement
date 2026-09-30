"""Async tool execution with risk-tiered controls and auth hooks."""

import logging
from typing import Any

from src.services.tools_integration.registry import (
    ToolRegistry,
    RequiresApprovalError,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tools with risk-tiered controls, schema validation, and auth.

    The executor wraps a :class:`ToolRegistry` and provides the
    execution surface used by the graph and sub-agents. Risk tiers:

    - **low**: direct execution via the registry.
    - **medium**: validate arguments, then execute.
    - **high**: raise :class:`RequiresApprovalError`.

    Auth checks (role-based) are enforced by the registry; this class
    surfaces the results and logs every call for observability.
    """

    def __init__(self, registry: ToolRegistry | None = None):
        self._registry = registry or ToolRegistry()

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    async def execute(
        self,
        tool_name: str,
        args: dict,
        subagent_name: str,
        trace_id: str | None = None,
    ) -> Any:
        """Execute a tool asynchronously with risk-tier controls.

        Args:
            tool_name: Name of the tool to execute.
            args: Keyword arguments for the tool call.
            subagent_name: Name of the calling sub-agent.
            trace_id: Optional trace identifier for observability.

        Returns:
            The tool's return value.

        Raises:
            RequiresApprovalError: If the tool requires approval.
            ValueError: If the tool is not registered or has no callable.
            PermissionError: If the sub-agent is not authorized.
        """
        logger.info(
            "ToolExecutor.execute('%s', subagent=%s, trace=%s)",
            tool_name, subagent_name, trace_id,
        )

        result = self._registry.execute(
            tool_name=tool_name,
            args=args,
            subagent_name=subagent_name,
            trace_id=trace_id,
        )
        return result

    def execute_sync(
        self,
        tool_name: str,
        args: dict,
        subagent_name: str,
        trace_id: str | None = None,
    ) -> Any:
        """Synchronous convenience wrapper around :meth:`execute`."""
        import asyncio

        return asyncio.run(self.execute(tool_name, args, subagent_name, trace_id))

    def register(self, spec, callable_=None) -> None:
        """Delegate to the underlying registry."""
        self._registry.register(spec, callable_)
