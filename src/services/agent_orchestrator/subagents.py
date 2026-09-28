"""Subagent registry and executors for the `task` tool and department fanout.

SUBAGENTS is the single source of truth for subagent types: it drives the
`task` tool schema (the allowed subagent_type values), `_execute_task`
dispatch, the parallel-vs-sequential split in the tools node, and role
prompt construction.

**Departments come from ``skills/``.** Each ``skills/<name>/SKILL.md`` yields
a ``tool_loop`` sub-agent whose system prompt is the markdown body and whose
``allowed-tools`` frontmatter is the hard tool allowlist. Adding a department
is adding a directory — no Python change. ``general-purpose`` (full graph) and
A2A agents are runtime capabilities, not skills, so they stay hardcoded.

Every tool_loop sub-agent runs the same executor, ``run_department``: it
shortlists its allowlist down to the tools relevant to the current query, then
runs a real ReAct loop with those tools bound, so it can actually search, read
and write files instead of answering in a single completion.
"""

import logging
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.config import get_a2a_agents
from src.services.agent_orchestrator.memory import get_memory_content, get_skill_body
from src.utils import get_message_text, invoke_with_retry

logger = logging.getLogger(__name__)

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


def _skill_specs(skills_dir: str | None = None) -> dict[str, SubagentSpec]:
    """Build a spec per ``skills/*/SKILL.md``.

    This is what makes ``skills/`` data-driven: dropping a new department
    directory in produces a routable sub-agent with no Python change. The
    SKILL.md frontmatter supplies the description (surfaced to the model),
    the ``allowed-tools`` allowlist, and whether the department is
    parallelizable; the markdown body is the system prompt.

    ``skills_dir`` defaults to the repository's ``skills/`` rather than the
    process cwd, so a rebuild after a chdir still finds every department.
    """
    from src.services.tools_integration.discovery import discover_subagents

    specs: dict[str, SubagentSpec] = {}
    for entry in discover_subagents(skills_dir):
        name = entry.get("name") or entry.get("department")
        if not name:
            logger.warning("Skipping skill with no name: %s", entry.get("skill_file"))
            continue
        # A duplicate name would silently shadow an earlier department, so
        # keep the first and say so rather than picking a winner quietly.
        if name in specs:
            logger.warning(
                "Duplicate subagent name '%s' in %s; keeping the first definition.",
                name, entry.get("skill_file"),
            )
            continue
        specs[name] = SubagentSpec(
            description=entry.get("description")
            or f"Department sub-agent defined by {entry.get('skill_file')}.",
            kind="tool_loop",
            parallelizable=bool(entry.get("parallelizable", True)),
            skill=entry.get("department") or name,
            tools=tuple(entry.get("allowed_tools") or ()),
        )
    return specs


# `general-purpose` is a capability of the runtime, not a skill: it runs the
# full graph and has no SKILL.md, so it stays hardcoded. Everything else comes
# from skills/ or from the A2A_AGENTS config.
_RUNTIME_SUBAGENTS: dict[str, SubagentSpec] = {
    "general-purpose": SubagentSpec(
        description="Full deep agent with all tools; use for complex or context-heavy sub-tasks.",
        kind="graph",
        parallelizable=False,
        aliases=("general",),
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


def build_subagent_registry() -> dict[str, SubagentSpec]:
    """Assemble the full registry: skills + runtime capabilities + A2A agents.

    Called at import for the module-level ``SUBAGENTS`` (which the ``task``
    tool's schema is built from) and again by ``refresh_subagents`` after a
    skills change.
    """
    return {**_skill_specs(), **_RUNTIME_SUBAGENTS, **_a2a_specs()}


SUBAGENTS: dict[str, SubagentSpec] = build_subagent_registry()


def refresh_subagents() -> dict[str, SubagentSpec]:
    """Rebuild ``SUBAGENTS`` from disk after skills/ changes.

    ``SUBAGENTS`` is a module-level dict that several modules import by
    value, so this rebinds it in place on the defining module and every
    importer sees the update. ``reset_deep_agent()`` calls this.
    """
    global SUBAGENTS
    from src.services.tools_integration.discovery import invalidate_discovery_cache

    invalidate_discovery_cache()
    SUBAGENTS = build_subagent_registry()
    logger.info(
        "Sub-agent registry refreshed from skills/: %s", sorted(SUBAGENTS)
    )
    return SUBAGENTS


def list_departments() -> list[dict[str, str]]:
    """The routable departments, for the dispatcher's prompt.

    Only ``tool_loop`` specs backed by a SKILL.md are routable departments:
    ``general-purpose`` is a delegation target, and A2A agents are remote
    services, so neither is something the dispatcher should route a query to.
    """
    return [
        {"name": name, "description": spec.description}
        for name, spec in sorted(SUBAGENTS.items())
        if spec.kind == "tool_loop" and spec.skill
    ]


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
    from src.services.agent_orchestrator.agent_factory import get_model

    model = get_model()
    model_with_tools = model.bind_tools(tools) if tools else model
    tool_map = {t.name: t for t in tools}

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=description)]
    usage = {"input": 0, "output": 0}
    write_ops: list[dict[str, Any]] = []

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

    # Budget exhausted. The last model turn is usually another tool call, whose
    # content is "" — returning that hands the parent an empty string, which
    # reads as "the subagent found nothing" and leaves it to redo the whole
    # task. Report the truncation and what was actually done instead, so the
    # parent can either use the partial work or say the subagent ran out of
    # budget.
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    final_text = get_message_text(last_ai.content) if last_ai is not None else ""
    if not final_text.strip():
        called = [
            c["name"]
            for m in messages
            if isinstance(m, AIMessage)
            for c in (getattr(m, "tool_calls", None) or [])
        ]
        detail = f" It called: {', '.join(called)}." if called else ""
        wrote = ", ".join(op["args"].get("path", "?") for op in write_ops) or "none"
        final_text = (
            f"Stopped after {max_iterations} model turns without a final answer."
            f"{detail} Files written: {wrote}."
        )
        logger.warning(
            "Subagent hit its %d-turn budget with no final answer (%d tool calls)",
            max_iterations, len(called),
        )
    return final_text, usage, write_ops


# ---------------------------------------------------------------------------
# Unified department executor
# ---------------------------------------------------------------------------

# Tools per sub-agent, per CLAUDE.md: the allowlist is capped at 20 visible and
# the LLM narrows it to 3-5 for the specific query.
MAX_VISIBLE_TOOLS = 20
MAX_SHORTLISTED_TOOLS = 5


def select_department_tools(
    spec: SubagentSpec,
    query: str,
    tools_dict: dict | None = None,
    model=None,
) -> list:
    """Resolve a department's tool allowlist into real, bindable tools.

    Three stages, per the documented contract:

    1. ``spec.tools`` (the SKILL.md ``allowed-tools`` frontmatter) is the hard
       allowlist. Names that do not resolve are dropped with a warning — a
       stale name in a SKILL.md must not become a tool the model can call but
       never execute.
    2. The allowlist is capped at ``MAX_VISIBLE_TOOLS``.
    3. ``relevance.sort_tools`` narrows to ``MAX_SHORTLISTED_TOOLS`` for *this*
       query, so a department is handed the tools its current task needs
       rather than its whole declared set.

    The return value is real ``BaseTool`` objects. ``get_tool_definitions``
    returns metadata only, so resolving names back to callables here is what
    makes the sub-agent able to actually *call* what it was offered.

    Falls back to the capped allowlist if relevance sorting fails.
    """
    from src.services.tools_integration.relevance import sort_tools

    toolset = tools_dict if tools_dict is not None else _all_tools()

    allowed: list = []
    for name in spec.tools:
        tool_obj = toolset.get(name)
        if tool_obj is None:
            logger.warning(
                "Department '%s' allows unknown tool '%s' (skills/%s/SKILL.md); skipping.",
                getattr(spec, "skill", "?"), name, getattr(spec, "skill", "?"),
            )
            continue
        allowed.append(tool_obj)

    if not allowed:
        return []
    if len(allowed) <= MAX_SHORTLISTED_TOOLS:
        return allowed

    visible = allowed[:MAX_VISIBLE_TOOLS]
    try:
        from src.services.agent_orchestrator.agent_factory import get_model

        sorter = model if model is not None else get_model()
        tool_defs = [
            {
                "name": t.name,
                "description": getattr(t, "description", "") or "",
            }
            for t in visible
        ]
        chosen = sort_tools(sorter, query, tool_defs, MAX_SHORTLISTED_TOOLS)
        by_name = {t.name: t for t in visible}
        shortlist = [by_name[c["name"]] for c in chosen if c.get("name") in by_name]
    except Exception as e:
        logger.warning(
            "Tool shortlisting failed for '%s' (%s); using capped allowlist.",
            getattr(spec, "skill", "?"), e,
        )
        return visible

    if not shortlist:
        return visible[:MAX_SHORTLISTED_TOOLS]
    logger.info(
        "Department '%s': shortlisted %d of %d allowed tools for this query.",
        getattr(spec, "skill", "?"), len(shortlist), len(visible),
    )
    return shortlist


def _all_tools() -> dict:
    """Lazily fetch the full tool set (avoids an import cycle at module load)."""
    from src.services.tools_integration.tools import get_all_tools

    return get_all_tools()


def run_department(
    spec: SubagentSpec,
    query: str,
    tools_dict: dict | None = None,
    max_iterations: int = 10,
):
    """Run one department end to end: prompt, tools, ReAct loop.

    This is the single executor behind both entry points — the ``task`` tool
    (``tools._execute_task``) and department fanout
    (``subagent_engine.SubAgentEngine``). Both previously had their own
    implementation and the fanout one could not execute tools at all.

    Args:
        spec: The department's registered spec (supplies the SKILL.md prompt
            and the tool allowlist).
        query: The enhanced query to fulfil.
        tools_dict: Live tool mapping to resolve the allowlist against.
        max_iterations: Cap on model turns before giving up.

    Returns:
        ``(final_text, usage, write_ops)`` — same shape as ``run_tool_loop``.
    """
    tools = select_department_tools(spec, query, tools_dict)
    return run_tool_loop(
        build_role_prompt(spec),
        query,
        tools,
        max_iterations=max_iterations,
    )
