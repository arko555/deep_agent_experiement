"""Tools Integration service — risk-tiered tool execution ecosystem.

Sub-agents discover and invoke tools through this service. It
provides the canonical registry that every sub-agent queries
for its tool schema before any tool is called.

Key components:
- ``ToolRegistry`` — central registry with risk metadata,
  role-based visibility (≤20 tools per sub-agent), and
  schema validation.
- ``@tool_spec`` — decorator for attaching risk, role, and
  approval metadata to tools.
- ``discover_subagents`` — scan ``skills/*/SKILL.md`` for
  sub-agent definitions with tree-hash caching.
- ``sort_tools`` — LLM-based relevance ranking of 20 → 5.
- ``mcp_client`` / ``a2a_client`` — MCP server tools and A2A
  remote agents, one implementation per protocol.
"""

from src.services.tools_integration.registry import ToolRegistry, RequiresApprovalError
from src.services.tools_integration.relevance import sort_tools
from src.services.tools_integration.discovery import discover_subagents
from src.services.tools_integration.decorator import tool_spec, ToolSpecMetadata
from src.services.tools_integration.mcp_client import load_mcp_tools
from src.services.tools_integration.tools import create_tool_registry
from src.services.tools_integration.a2a_client import call_a2a_agent

__all__ = [
    "RequiresApprovalError",
    "ToolRegistry",
    "ToolSpecMetadata",
    "call_a2a_agent",
    "create_tool_registry",
    "discover_subagents",
    "load_mcp_tools",
    "sort_tools",
    "tool_spec",
]
