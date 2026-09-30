"""Subagent registry and the department executor.

SUBAGENTS is the single source of truth for departments: it drives the
router's roster (``list_departments``), the parallel-vs-sequential split in
``subagent_engine``, and each sub-agent's prompt construction.

**Departments come from ``skills/``.** Each ``skills/<name>/SKILL.md`` yields
a ``tool_loop`` sub-agent whose system prompt is the markdown body and whose
``allowed-tools`` frontmatter is the hard tool allowlist. Adding a department
is adding a directory — no Python change. ``skills/general`` is the catch-all
department, and configured A2A agents are remote services, not skills.

Every tool_loop sub-agent runs the same executor, ``run_department``: it
shortlists its allowlist down to the tools relevant to the current query, then
runs a real ReAct loop with those tools bound, so it can actually search, read
and write files instead of answering in a single completion.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.config import get_a2a_agents, get_max_iterations
from src.services.agent_orchestrator.memory import get_memory_content, get_skill_body
from src.utils import get_message_text, invoke_with_retry

if TYPE_CHECKING:
    from src.services.tools_integration.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Built on first use, not at import: it reads the toolset from disk and the
# network, which must not happen merely by importing this module.
_TOOL_REGISTRY: "ToolRegistry | None" = None

# A2A subagent types come from configuration, but the department roster the
# router sees is built statically from this registry at import — so `.env` must
# be loaded before the registry is assembled. Import order is tools ->
# subagents, which runs before the load_dotenv() in agent_factory, hence the
# explicit call here. Consequence: changing the configured A2A agents requires
# a process restart.
load_dotenv()


# ---------------------------------------------------------------------------
# Subagent Registry (6.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubagentSpec:
    """One subagent type, as registered in SUBAGENTS."""

    description: str   # the only signal the router has for choosing this one
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


# There is no hardcoded runtime subagent. A `general-purpose` spec running the
# whole compiled graph as a "sub-agent" would make the orchestrator its own
# sub-agent — a delegated description gets routed again from the top and the
# hierarchy stops being one-way. The catch-all is `skills/general/SKILL.md`: a
# real department with a real prompt and a real tool allowlist, chosen by the
# router rather than by a self-referential spec.


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
    """Assemble the full registry: skills from disk + configured A2A agents.

    Called at import for the module-level ``SUBAGENTS`` (which the router's
    roster and every department lookup resolve against) and again by
    ``refresh_subagents`` after a skills change.
    """
    return {**_skill_specs(), **_a2a_specs()}


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

    Both kinds of sub-agent are routable. A ``tool_loop`` spec is backed by a
    SKILL.md; an ``a2a`` spec is a remote agent reached by URL. A remote agent
    is a sub-agent that happens to live behind a network hop, so the router
    treats it exactly like a local department — which is what it has to do
    now that the ``task`` tool, previously the only thing that dispatched an
    ``a2a`` spec, is gone.

    A ``tool_loop`` spec with no skill is still excluded: it has no prompt to
    run a tool loop with, so routing to it would produce an empty turn.
    """
    return [
        {"name": name, "description": spec.description}
        for name, spec in sorted(SUBAGENTS.items())
        if (spec.kind == "tool_loop" and spec.skill) or spec.kind == "a2a"
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


def _registry_for(tool_map: dict) -> "ToolRegistry":
    """The registry a sub-agent's tool calls dispatch through.

    Prefers the process-wide registry from ``create_tool_registry``, which
    already holds the full tool set with its declared kinds and schemas. Tools
    it does not know — a test double, a tool registered after it was built —
    are added to it, so a bound tool is never silently uncallable.

    The registry is built once and cached: it loads dynamic and MCP tools from
    disk and the network, which must not happen per sub-agent per turn.
    """
    from src.services.tools_integration.registry import ToolRegistry

    global _TOOL_REGISTRY
    if _TOOL_REGISTRY is None:
        try:
            from src.services.tools_integration.tools import create_tool_registry

            _TOOL_REGISTRY = create_tool_registry()
        except Exception as e:
            # A registry is a convenience here, not a precondition: the loop
            # still runs, with the bound tools as the only authority.
            logger.warning("Could not build the tool registry (%s); using bound tools only.", e)
            _TOOL_REGISTRY = ToolRegistry()

    for name, tool in tool_map.items():
        if _TOOL_REGISTRY.get_tool(name) is None:
            _TOOL_REGISTRY.register_builtin(name, tool, allowed_roles=("*",))
    return _TOOL_REGISTRY


class _ToolDispatcher:
    """Adapts ``ToolRegistry.dispatch`` to the ``invoke_with_retry`` contract.

    ``invoke_with_retry`` calls ``.invoke(payload)`` and reads its second
    positional as ``max_retries``, so it needs an object rather than a bare
    callable. Binding the tool name here keeps the retry on the transport —
    where a transient network failure actually is — and leaves validation
    errors to re-raise immediately, which is what lets the sub-agent see
    them and correct the payload.
    """

    def __init__(self, registry: "ToolRegistry", tool_name: str):
        self._registry = registry
        self._tool_name = tool_name

    def invoke(self, payload: dict) -> Any:
        return self._registry.dispatch(self._tool_name, payload)


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

    # The one execution surface. Every call goes through the registry so the
    # payload is checked against the tool's declared schema and the transport
    # is chosen from the tool's declared kind. A tool the registry has never
    # seen still runs — the bound `tool_map` is the authority on what a
    # sub-agent may call, and the registry is populated independently.
    registry = _registry_for(tool_map)

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
                    result = invoke_with_retry(
                        _ToolDispatcher(registry, tool_call["name"]), tool_call["args"]
                    )
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
       allowlist. A name may be exact or a ``prefix*`` namespace claim, which
       is how a department claims an MCP server's tools. Names that do not
       resolve are dropped with a warning — a stale name in a SKILL.md must
       not become a tool the model can call but never execute.
    2. The allowlist is capped at ``MAX_VISIBLE_TOOLS``.
    3. ``relevance.sort_tools`` narrows to ``MAX_SHORTLISTED_TOOLS`` for *this*
       query, so a department is handed the tools its current task needs
       rather than its whole declared set.

    The return value is real ``BaseTool`` objects, not metadata dicts —
    resolving names back to callables is what makes the sub-agent able to
    actually *call* what it was offered.

    Stage 3 delegates to ``ToolRegistry.select_for_query``, which owns the
    relevance shortlist; this function owns the allowlist, namespace matching,
    and the visibility cap — the two halves of one rule, split where the
    knowledge sits.

    Falls back to the capped allowlist if relevance sorting fails.
    """
    toolset = tools_dict if tools_dict is not None else _all_tools()

    allowed: list = []
    for name in spec.tools:
        if name.endswith("*"):
            # A trailing '*' claims a whole namespace. MCP tools arrive
            # server-prefixed (``<server>_<tool>``), so a department cannot
            # name them one by one in static frontmatter and would otherwise
            # be unable to claim its own server at all. Prefix matches still
            # pass through the visibility cap and the relevance shortlist
            # below, so a large namespace stays bounded.
            prefix = name[:-1]
            matched = [t for tname, t in toolset.items() if tname.startswith(prefix)]
            if not matched:
                logger.warning(
                    "Department '%s' allows namespace '%s' (skills/%s/SKILL.md); "
                    "no tools matched.",
                    getattr(spec, "skill", "?"), name, getattr(spec, "skill", "?"),
                )
            allowed.extend(matched)
            continue
        tool_obj = toolset.get(name)
        if tool_obj is None:
            logger.warning(
                "Department '%s' allows unknown tool '%s' (skills/%s/SKILL.md); skipping.",
                getattr(spec, "skill", "?"), name, getattr(spec, "skill", "?"),
            )
            continue
        allowed.append(tool_obj)

    # A namespace claim and an explicit name can resolve to the same tool;
    # keep the first occurrence so the bound list has no duplicates.
    deduped: list = []
    seen: set[str] = set()
    for tool_obj in allowed:
        if tool_obj.name in seen:
            continue
        seen.add(tool_obj.name)
        deduped.append(tool_obj)
    allowed = deduped

    if not allowed:
        return []
    if len(allowed) <= MAX_SHORTLISTED_TOOLS:
        return allowed

    # The visibility cap and the relevance shortlist are the registry's job, so
    # the rule lives in one place: this function decides which names a
    # department *declares*, the registry decides which of those it *sees*.
    visible = allowed[:MAX_VISIBLE_TOOLS]
    from src.services.tools_integration.registry import ToolRegistry

    registry = ToolRegistry()
    for tool_obj in visible:
        registry.register_builtin(tool_obj.name, tool_obj, allowed_roles=("*",))

    shortlist = registry.select_for_query(
        [t.name for t in visible], query, max_tools=MAX_SHORTLISTED_TOOLS, model=model
    )

    logger.info(
        "Department '%s': shortlisted %d of %d allowed tools for this query.",
        getattr(spec, "skill", "?"), len(shortlist), len(visible),
    )
    return shortlist


def _all_tools() -> dict:
    """The full tool set a department's allowlist resolves against.

    This is ``tools_integration.get_all_tools()`` — every built-in, dynamic
    file, and MCP tool. It previously also carried the ``task`` delegation
    tool; that lived in ``agent_orchestrator.task_tool`` because it reached up
    into this registry, and both it and the sub-agent-to-sub-agent delegation
    it served have been removed. A department's allowlist resolves against the
    tool layer alone, so the dependency arrow points one way.
    """
    from src.services.tools_integration.tools import get_all_tools

    return get_all_tools()


def run_department(
    spec: SubagentSpec,
    query: str,
    tools_dict: dict | None = None,
    max_iterations: int | None = None,
):
    """Run one department end to end: prompt, tools, ReAct loop.

    The single executor behind department fanout
    (``subagent_engine.SubAgentEngine``). The fanout used to have a second,
    weaker implementation that could not execute tools at all; this is the one
    both paths go through.

    Args:
        spec: The department's registered spec (supplies the SKILL.md prompt
            and the tool allowlist).
        query: The enhanced query to fulfil.
        tools_dict: Live tool mapping to resolve the allowlist against.
        max_iterations: Cap on model turns before giving up. ``None`` takes
            ``AGENT_MAX_ITERATIONS``. This is the only iteration budget in
            the system: the ReAct loop below is the only thing that
            iterates, so this is where that setting belongs. The orchestrator
            has no budget because it is a single-shot router.

    Returns:
        ``(final_text, usage, write_ops)`` — same shape as ``run_tool_loop``.
    """
    if max_iterations is None:
        max_iterations = get_max_iterations()
    tools = select_department_tools(spec, query, tools_dict)
    return run_tool_loop(
        build_role_prompt(spec),
        query,
        tools,
        max_iterations=max_iterations,
    )
