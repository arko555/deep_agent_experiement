"""Tool registry with risk metadata, role-based visibility, and schema validation."""

import logging
from typing import Any, Callable

from src.services.tools_integration.decorator import ToolSpecMetadata
from src.types import ToolSpec

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry of all tools with risk, role, and approval metadata.

    Maps tool names to their :class:`ToolSpecMetadata` and optional
    callable. Provides role-based filtering, visibility caps, schema
    validation, and risk-tiered execution hooks.
    """

    def __init__(self):
        self._specs: dict[str, ToolSpecMetadata] = {}
        self._callables: dict[str, Callable] = {}

    def register(self, spec: ToolSpec | ToolSpecMetadata, callable_: Callable | None = None) -> None:
        """Register a tool spec and optionally its callable."""
        if isinstance(spec, ToolSpec):
            metadata = ToolSpecMetadata(
                name=spec.name,
                description=spec.description,
                risk_level=spec.risk_level,
                requires_approval=spec.requires_approval,
                allowed_roles=tuple(spec.allowed_roles),
            )
        else:
            metadata = spec
        self._specs[metadata.name] = metadata
        if callable_ is not None:
            self._callables[metadata.name] = callable_

    def get_spec(self, tool_name: str) -> ToolSpecMetadata | None:
        """Return the spec for a tool, or None if unregistered."""
        return self._specs.get(tool_name)

    def get_callable(self, tool_name: str) -> Callable | None:
        """Return the callable for a tool, or None."""
        return self._callables.get(tool_name)

    def register_builtin(self, name: str, callable_: Callable, risk_level: str = "low",
                         requires_approval: bool = False, allowed_roles: tuple = ()) -> None:
        """Register a built-in tool function with metadata."""
        self.register(
            ToolSpecMetadata(
                name=name,
                description=getattr(callable_, "__doc__", "") or "",
                risk_level=risk_level,
                requires_approval=requires_approval,
                allowed_roles=allowed_roles,
            ),
            callable_,
        )

    def get_tools_for_role(self, role: str) -> list[ToolSpecMetadata]:
        """Return tools visible to a given role."""
        return [
            spec for spec in self._specs.values()
            if not spec.allowed_roles or role in spec.allowed_roles or "*" in spec.allowed_roles
        ]

    def get_visible_tools(self, subagent_name: str) -> list[ToolSpecMetadata]:
        """Return tools visible to a sub-agent, capped at 20 (SKILL.md limit)."""
        if subagent_name == "general-purpose":
            return list(self._specs.values())[:20]
        role = subagent_name.split("_")[0] if "_" in subagent_name else subagent_name
        return self.get_tools_for_role(role)[:20]

    def get_tool_definitions(self, tool_names: list[str]) -> list[dict[str, Any]]:
        """Return tool schemas for the given tool names."""
        definitions = []
        for name in tool_names:
            spec = self._specs.get(name)
            if spec is None:
                continue
            definitions.append({
                "name": spec.name,
                "description": spec.description,
                "risk_level": spec.risk_level,
                "requires_approval": spec.requires_approval,
            })
        return definitions

    def _validate_args(self, tool_name: str, args: dict) -> None:
        """Validate args against the tool's schema if available."""
        # Schema validation is a no-op here unless we have the
        # underlying tool object with a JSON schema.  The executor
        # layer supplements this with type-aware validation.
        logger.debug("Schema validation for '%s' (args: %s)", tool_name, args)

    def execute(
        self,
        tool_name: str,
        args: dict,
        subagent_name: str,
        trace_id: str | None = None,
    ) -> Any:
        """Execute a tool with risk-tier controls and schema validation.

        Risk tiers handled by the caller (ToolExecutor): this method
        validates, authorizes, and dispatches to the tool function.

        Raises:
            RequiresApprovalError: If the tool requires approval.
            ValueError: If the tool is not registered.
            PermissionError: If the sub-agent lacks authorization.
        """
        spec = self._specs.get(tool_name)
        if spec is None:
            raise ValueError(f"Tool '{tool_name}' is not registered")

        if spec.requires_approval:
            raise RequiresApprovalError(
                f"Tool '{tool_name}' requires approval (risk={spec.risk_level})"
            )

        if spec.allowed_roles:
            role = subagent_name.split("_")[0] if "_" in subagent_name else subagent_name
            if role not in spec.allowed_roles and "*" not in spec.allowed_roles:
                raise PermissionError(
                    f"Sub-agent '{subagent_name}' not authorized for '{tool_name}'"
                )

        self._validate_args(tool_name, args)

        logger.info(
            "Executing tool '%s' (risk=%s, trace=%s)", tool_name, spec.risk_level, trace_id
        )

        if tool_name in self._callables:
            return self._callables[tool_name](**args)

        raise ValueError(f"Tool '{tool_name}' has no registered callable")

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._specs.keys())


class RequiresApprovalError(Exception):
    """Raised when a high-risk tool requires approval before execution."""
    pass
