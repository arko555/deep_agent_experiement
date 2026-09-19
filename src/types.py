"""Shared type definitions for the three-service architecture."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SubAgent:
    """A department sub-agent identified by skills/<dept>/SKILL.md."""

    name: str
    description: str
    department: str
    system_prompt: str
    tool_registry: list = field(default_factory=list)
    protocol: str = "react"


@dataclass
class ToolSpec:
    """Metadata attached to a tool via @tool_spec."""

    name: str
    description: str
    risk_level: str = "low"          # "low" | "medium" | "high"
    requires_approval: bool = False
    allowed_roles: list = field(default_factory=list)
    handler: Optional[Any] = None
    timeout_seconds: int = 600
