"""@tool_spec decorator: attach risk, role, and approval metadata to tools."""

from dataclasses import dataclass
from typing import Optional

from src.types import ToolKind


@dataclass(frozen=True)
class ToolSpecMetadata:
    """Metadata attached to a tool via @tool_spec."""

    name: str
    description: str
    risk_level: str = "low"  # "low" | "medium" | "high"
    requires_approval: bool = False
    allowed_roles: tuple = ()  # e.g., ("hr", "payroll") or () for all
    # Which transport reaches this tool, and therefore what a correct payload
    # is. Defaults to a local call, so existing @tool_spec uses are unaffected.
    kind: ToolKind = ToolKind.LOCAL


def tool_spec(
    name: Optional[str] = None,
    description: Optional[str] = None,
    risk_level: str = "low",
    requires_approval: bool = False,
    allowed_roles: Optional[tuple] = None,
    kind: ToolKind = ToolKind.LOCAL,
):
    """Decorator that attaches ToolSpec metadata to a tool function.

    Applied to @tool-decorated functions to register them in
    ToolRegistry with risk-tier and role-scoping information.
    Stores metadata in ``__tool_spec__`` without altering
    the function's behavior.
    """
    if allowed_roles is None:
        allowed_roles = ()

    def decorator(func):
        meta = ToolSpecMetadata(
            name=name or func.__name__,
            description=description or (func.__doc__ or ""),
            risk_level=risk_level,
            requires_approval=requires_approval,
            allowed_roles=allowed_roles,
            kind=kind,
        )
        func.__tool_spec__ = meta  # type: ignore[attr-defined]
        return func

    return decorator
