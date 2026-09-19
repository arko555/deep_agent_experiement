"""Tools Integration service — risk-tiered tool execution ecosystem.

Sub-agents discover and invoke tools through this service. It
provides the canonical registry that every sub-agent queries
for its tool schema before any tool is called.

Key components:
- ``ToolRegistry`` — central registry with risk metadata,
  role-based visibility (≤20 tools per sub-agent), and
  schema validation.
- ``ToolExecutor`` — risk-tiered execution (low/direct,
  medium/validate+execute, high/approval gate).
- ``@tool_spec`` — decorator for attaching risk, role, and
  approval metadata to tools.
- ``discover_subagents`` — scan ``skills/*/SKILL.md`` for
  sub-agent definitions with tree-hash caching.
- ``sort_tools`` — LLM-based relevance ranking of 20 → 5.

Bridges for external tool sources live alongside:
- ``mcp_bridge`` — async MCP server tools, sync-wrapped at
  the boundary.
- ``a2a_bridge`` — async A2A remote agent calls.
"""

from src.services.tools_integration.registry import ToolRegistry, RequiresApprovalError
from src.services.tools_integration.executor import ToolExecutor
from src.services.tools_integration.relevance import sort_tools
from src.services.tools_integration.discovery import discover_subagents
from src.services.tools_integration.decorator import tool_spec, ToolSpecMetadata
from src.services.tools_integration.mcp_bridge import load_mcp_tools
from src.services.tools_integration.tools import create_tool_registry
from src.services.tools_integration.a2a_bridge import call_a2a_agent

__all__ = [
    "ToolRegistry",
    "ToolExecutor",
    "RequiresApprovalError",
    "sort_tools",
    "discover_subagents",
    "tool_spec",
    "ToolSpecMetadata",
    "load_mcp_tools",
    "call_a2a_agent",
    "create_tool_registry",
]
