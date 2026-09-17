"""Subagent registry and executors for the `task` tool.

SUBAGENTS is the single source of truth for subagent types: it drives the
`task` tool schema (the allowed subagent_type values), `_execute_task`
dispatch, the parallel-vs-sequential split in the tools node, and role
prompt construction. Adding a subagent type = adding one SUBAGENTS entry.

Research and writer subagents run a minimal tool loop (model + bound tools)
so they can actually search, read, and write files instead of answering in a
single completion. The general-purpose subagent reuses the full compiled
graph and is handled in ``tools._execute_task``.
"""

from dataclasses import dataclass

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.config import get_a2a_agents
from src.core.memory import get_memory_content, get_skill_body
from src.core.utils import get_message_text, invoke_with_retry

# A2A subagent types come from configuration, but the `task` tool's type enum
# is built statically from this registry — so `.env` must be loaded before the
# registry is assembled. Import order is tools -> subagents, which runs before
# the load_dotenv() in agent_factory, hence the explicit call here. Consequence:
# changing the configured A2A agents requires a process restart.
load_dotenv()


# ---------------------------------------------------------------------------
# Subagent Registry (6.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubagentSpec:
    """One subagent type, as registered in SUBAGENTS."""

    description: str   # surfaced in the `task` tool schema
    kind: str          # "tool_loop" (restricted tools), "graph" (full graph), or "a2a" (remote agent)
    parallelizable: bool = False  # may run concurrently with sibling tasks
    aliases: tuple = ()            # accepted alternate names for subagent_type
    skill: str | None = None       # skills/<skill>/SKILL.md — prompt source (tool_loop only)
    tools: tuple = ()              # tool names for the restricted toolset (tool_loop only)
    url: str | None = None         # remote agent base URL (a2a only)


_BUILTIN_SUBAGENTS: dict[str, SubagentSpec] = {
    "general-purpose": SubagentSpec(
        description="Full deep agent with all tools; use for complex or context-heavy sub-tasks.",
        kind="graph",
        parallelizable=False,
        aliases=("general",),
    ),
    "research": SubagentSpec(
        description="Web research: searches the internet and writes findings to a workspace file.",
        kind="tool_loop",
        parallelizable=True,
        aliases=("researcher",),
        skill="research",
        tools=("internet_search", "fetch_url", "read_file", "write_file",
               "list_files", "search_files"),
    ),
    "writer": SubagentSpec(
        description="Content writer: reads workspace notes and saves the draft to a file.",
        kind="tool_loop",
        parallelizable=True,
        skill="writer",
        tools=("read_file", "write_file", "edit_file", "list_files", "search_files"),
    ),
}


def _a2a_specs() -> dict[str, SubagentSpec]:
    """Build a spec per configured remote A2A agent (the `A2A_AGENTS` env var).

    Remote calls are independent of each other, so they are parallelizable and
    ride the same batch deadline as the other parallelizable types.
    """
    return {
        name: SubagentSpec(
            description=config.get("description") or f"Remote A2A agent at {config['url']}.",
            kind="a2a",
            parallelizable=True,
            url=config["url"],
        )
        for name, config in get_a2a_agents().items()
    }


SUBAGENTS: dict[str, SubagentSpec] = {**_BUILTIN_SUBAGENTS, **_a2a_specs()}


def resolve_subagent(subagent_type: str) -> SubagentSpec | None:
    """Resolve a subagent type or alias to its spec; None if unknown."""
    for name, spec in SUBAGENTS.items():
        if subagent_type == name or subagent_type in spec.aliases:
            return spec
    return None


def is_parallelizable(subagent_type: str) -> bool:
    """Whether tasks of this type may run concurrently with sibling tasks.

    Unknown types are not parallelizable (they run sequentially, where the
    unknown-type error is produced).
    """
    spec = resolve_subagent(subagent_type)
    return bool(spec and spec.parallelizable)


# ---------------------------------------------------------------------------
# Role Prompts (6.2: SKILL.md + shared completion contract)
# ---------------------------------------------------------------------------

COMPLETION_CONTRACT = """You are a specialized subagent delegated a task by the parent orchestrator.
Complete the task using only the tools provided to you.

Ground rules:
- Save any file output to the exact path named in your task description, under
  `./workspace`. If the task does not name a path, choose a unique filename based on
  your topic (e.g., `workspace/<topic>.md`) — never reuse a shared default
  filename, as other subagents may run in parallel.
- Strictly do NOT write files outside `./workspace` or to `/tmp/`.

When you are done, stop calling tools and end with a short completion summary containing:
- The path(s) of the file(s) you saved under `./workspace` (if any)
- A concise summary of what you did and the key results"""


def build_role_prompt(spec: SubagentSpec) -> str:
    """System prompt for a tool-loop subagent: shared completion contract,
    then the subagent's role instructions from its SKILL.md, then AGENTS.md."""
    parts = [COMPLETION_CONTRACT]
    if spec.skill:
        body = get_skill_body(spec.skill)
        if body:
            parts.append(f"=== Your role skill (skills/{spec.skill}/SKILL.md) ===\n{body}")
    agents_md = get_memory_content()
    if agents_md:
        parts.append(f"=== Shared Data / AGENTS.md ===\n{agents_md}")
    return "\n\n".join(parts)


def _extract_usage(response):
    """Return (input_tokens, output_tokens) from a chat response's usage_metadata."""
    meta = getattr(response, "usage_metadata", None)
    if not isinstance(meta, dict):
        return 0, 0
    return (
        meta.get("input_tokens", 0) or 0,
        meta.get("output_tokens", 0) or 0,
    )


def run_tool_loop(system_prompt: str, description: str, tools: list, max_iterations: int = 10):
    """Run a minimal ReAct-style loop: model with bound tools until it answers
    without tool calls (or the iteration budget is exhausted).

    Args:
        system_prompt: Role-specific system prompt for the subagent.
        description: The task description from the parent orchestrator.
        tools: Restricted list of LangChain tools this subagent may use.
        max_iterations: Maximum model turns before giving up.

    Returns:
        (final_text, usage, write_ops) — the subagent's final answer, its
        total token usage across all loop turns, and the file-write
        operations it performed (for the parent's audit trail).
    """
    # Lazy import to avoid a circular dependency at module load time.
    from src.core.agent_factory import get_model

    model = get_model()
    model_with_tools = model.bind_tools(tools) if tools else model
    tool_map = {t.name: t for t in tools}

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=description)]
    usage = {"input": 0, "output": 0}
    write_ops = []

    for _ in range(max_iterations):
        response = invoke_with_retry(model_with_tools, messages)
        in_tokens, out_tokens = _extract_usage(response)
        usage["input"] += in_tokens
        usage["output"] += out_tokens
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            return get_message_text(response.content), usage, write_ops

        for tool_call in response.tool_calls:
            # Track file writes so the parent's pending_writes audit trail
            # covers subagent activity, not just top-level tools.
            if tool_call["name"] in ("write_file", "edit_file"):
                write_ops.append({
                    "tool": tool_call["name"],
                    "tool_id": None,
                    "args": tool_call["args"],
                    "status": "executed",
                })
            tool = tool_map.get(tool_call["name"])
            if tool is None:
                result = f"Tool '{tool_call['name']}' not found."
            else:
                try:
                    result = invoke_with_retry(tool, tool_call["args"])
                except Exception as e:
                    result = f"Error executing tool {tool_call['name']}: {e}"
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            ))

    # Budget exhausted — return the last model turn if it has any text.
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    fallback = (
        get_message_text(last_ai.content)
        if last_ai is not None
        else "Error: subagent reached max iterations without a final answer."
    )
    return fallback, usage, write_ops
