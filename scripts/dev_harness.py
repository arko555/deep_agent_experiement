"""Developer harness: end-to-end exercise of the deep agent as a user drives it.

Simulates a real multi-turn session and asserts the things that are easy to
break silently in this codebase:

  1. session_memory is written by the graph, and the dispatcher sees the real
     history on turn 2+ (not an empty window).
  2. The dispatcher's JSON envelope produces a real ``enhanced_query`` and
     ``department_targets``.
  3. The tool registry discovers ``@tool`` functions from ``./tools``.
  4. A tool is *suggested from the registry* based on the query text, then
     actually executed and its result returned to the dispatcher.
  5. Every tool call and intermediate message is logged.

The dispatcher is a deterministic stand-in for the LLM (see ``ScriptedDispatcher``)
so the run needs no API key and produces the same result every time. Point
AGENT_HARNESS_LIVE=1 at a real model to exercise the same path for real.

Run it:
    uv run python scripts/dev_harness.py

Watch it live:
    uv run python scripts/dev_harness.py 2>&1 | tee /tmp/harness.log
    tail -f /tmp/harness.log
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402

from src.services.agent_orchestrator import agent_factory, graph  # noqa: E402
from src.services.agent_orchestrator.subagents import (  # noqa: E402
    SUBAGENTS,
    list_departments,
    select_department_tools,
)
from src.services.session_memory.checkpoint import get_session  # noqa: E402
from src.services.session_memory.window import get_window  # noqa: E402
from src.services.tools_integration.discovery import discover_tools  # noqa: E402
from src.services.tools_integration.registry import ToolRegistry  # noqa: E402
from src.services.tools_integration.tools import (  # noqa: E402
    create_tool_registry,
    get_all_tools,
    load_dynamic_tools,
)

if TYPE_CHECKING:
    from src.services.agent_orchestrator.state import AgentState

# ---------------------------------------------------------------------------
# Logging: every tool call and intermediate message lands here.
# ---------------------------------------------------------------------------

LOG = logging.getLogger("harness")


def setup_logging(verbose: bool = True) -> None:
    """Send app + harness logs to stdout and a file, so `tail -f` works."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler("/tmp/harness.log", mode="w"))
    except OSError:
        pass
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    # These are chatty and not useful for this harness.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def step(text: str) -> None:
    print(f"\n--- {text}")


# ---------------------------------------------------------------------------
# A deterministic stand-in for the LLM dispatcher.
# ---------------------------------------------------------------------------

# Keyword -> tool name. Stands in for "the model picks the right tool".
TOOL_HINTS = {
    "text_stats": ("word", "words", "count", "character", "characters", "text stats"),
    "get_current_time": ("time", "current time", "clock", "date", "today"),
}


class ScriptedDispatcher:
    """Fake chat model covering both layers of the hierarchy, no API key.

    The orchestrator is a pure router: it binds no tools and answers only with
    the ``{"enhanced_query", "departments"}`` control envelope. A sub-agent
    binds the tools ``select_department_tools`` shortlisted for it and runs the
    ReAct loop itself — call a tool, read the ToolMessage, then answer.

    An earlier version of this fake gave the *orchestrator* the tool call,
    which is the one capability Option B removed. It could not fail, because
    the router has no tools to call and silently fell through to a department
    with an empty enhanced query.
    """

    def __init__(self, tool_names: list[str], bound: list[str] | None = None):
        self._tool_names = tool_names
        self._bound = list(bound or [])
        self.seen_prompts: list[list[BaseMessage]] = []
        # (department, tool, result) for every tool a sub-agent executed, so the
        # harness can assert the ReAct loop really ran.
        self.subagent_tool_calls: list[tuple[str, str, str]] = []
        LOG.info("ScriptedDispatcher initialised with %d registry tools", len(tool_names))

    def bind_tools(self, tools, **kwargs):
        bound = [getattr(t, "name", str(t)) for t in tools]
        LOG.info("bind_tools: sub-agent received %d tools: %s", len(bound), bound)
        clone = ScriptedDispatcher(self._tool_names, bound=bound)
        clone.seen_prompts = self.seen_prompts
        clone.subagent_tool_calls = self.subagent_tool_calls
        return clone

    def invoke(self, messages, **kwargs):
        self.seen_prompts.append(list(messages))

        if _is_subagent_prompt(messages):
            return self._subagent_turn(messages)

        # The router binds no tools, so this turn can only be the envelope.
        user_text = _last_user_text(messages)
        dept = self._route_to(user_text)
        LOG.info("Router returning JSON envelope for %r -> %s", user_text, dept)
        return AIMessage(content=json.dumps({
            "enhanced_query": f"[enhanced] {user_text}",
            "departments": [dept],
        }))

    def _route_to(self, user_text: str) -> str:
        """Pick a department the way the real router would.

        A query needing a ``./tools`` tool goes to ``general``, the only
        department whose allowlist lists them; anything else goes to
        ``research``. Routing a tool-needing query to a department that cannot
        call the tool would make the ReAct loop unreachable.
        """
        if self._suggest_tool(user_text, self._tool_names):
            return "general"
        return "research"

    def _subagent_turn(self, messages) -> AIMessage:
        """One turn of a sub-agent's ReAct loop.

        Calls a hinted tool on the first turn if the department was actually
        offered one, then summarises the ToolMessage it gets back.
        """
        query = _last_user_text(messages)
        clean = query.replace("[enhanced] ", "").strip()

        if _tools_since_last_human(messages):
            result = _last_tool_result(messages)
            tool = _last_tool_name(messages)
            self.subagent_tool_calls.append((tool, query, result))
            LOG.info("Sub-agent read result of '%s'; answering", tool)
            return AIMessage(content=f"Research findings for: {clean} — {result}")

        tool_name = self._suggest_tool(query, self._bound)
        if tool_name:
            LOG.info(
                "Sub-agent chose tool '%s' for %r (offered: %s)",
                tool_name, clean, self._bound,
            )
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": _args_for(tool_name, query),
                    "id": f"call_{tool_name}",
                    "type": "tool_call",
                }],
            )

        LOG.info("Sub-agent answering without tools: %r", clean)
        return AIMessage(content=f"Research findings for: {clean}")

    def _suggest_tool(self, user_text: str, offered: list[str]) -> str | None:
        """Pick a tool the department was actually offered, if the query hints one.

        Only ``offered`` counts. A hint matched against the whole registry
        would have the sub-agent call a tool its SKILL.md allowlist excluded,
        which the model could not do against a real LLM.
        """
        lowered = user_text.lower()
        for name, hints in TOOL_HINTS.items():
            if name not in offered:
                LOG.debug(
                    "Tool '%s' hinted at but not offered to this department; skipping", name
                )
                continue
            if any(h in lowered for h in hints):
                return name
        return None


def _is_subagent_prompt(messages) -> bool:
    """True if this is a sub-agent turn rather than a dispatcher turn.

    A department turn is ``SystemMessage(COMPLETION_CONTRACT + SKILL.md body)``
    followed by ``HumanMessage(enhanced_query)``; the dispatcher gets a
    SystemMessage beginning with "You are the dispatcher". The marker is the
    contract's opening line, not the word "specialist" — department prompts now
    come from ``skills/<name>/SKILL.md`` and say whatever that file says, so
    matching on a synthesized phrase silently stops matching and the fake
    model answers a sub-agent turn with a dispatcher's JSON envelope.
    """
    for m in messages:
        if getattr(m, "type", "") == "system":
            content = str(m.content).lower()
            if "you are the router" in content or "you are the dispatcher" in content:
                return False
            if "specialized subagent delegated a task" in content:
                return True
    return False


def _tools_since_last_human(messages) -> bool:
    """True if a ToolMessage appears *after* the most recent HumanMessage.

    The dispatcher's window spans the whole conversation, so turn 2 sees turn
    1's tool output. Only messages after this turn's user message indicate
    that a tool ran as part of the current turn.
    """
    last_human = -1
    for i, m in enumerate(messages):
        if getattr(m, "type", "") == "human":
            last_human = i
    return any(
        getattr(m, "type", "") == "tool" for m in messages[last_human + 1:]
    )


def _last_user_text(messages) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "human":
            return str(m.content)
    return ""


def _last_tool_result(messages) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "tool":
            return str(m.content).replace("\n", " ")[:120]
    return "(no tool result)"


def _last_tool_name(messages) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "tool":
            return getattr(m, "name", "") or ""
    return ""


def _args_for(tool_name: str, user_text: str) -> dict:
    if tool_name == "text_stats":
        return {"text": user_text}
    if tool_name == "get_current_time":
        return {"timezone": "UTC"}
    return {}


# ---------------------------------------------------------------------------
# A tool registry restricted to ./tools — the visibility the harness asserts.
# ---------------------------------------------------------------------------

def tools_dir_only_registry(tools_dir: str = "./tools") -> ToolRegistry:
    """Build a registry containing only tools discovered under *tools_dir*.

    This is the visibility rule the harness checks: the dispatcher and
    sub-agents should see the project's own tools, not the built-ins.
    """
    dynamic = load_dynamic_tools(tools_dir)
    registry = ToolRegistry()
    for name, tool_obj in dynamic.items():
        func = getattr(tool_obj, "func", tool_obj)
        spec = getattr(func, "__tool_spec__", None)
        if spec is not None:
            registry.register(spec, func)
        else:
            registry.register_builtin(name, func)
    return registry


# ---------------------------------------------------------------------------
# The session driver.
# ---------------------------------------------------------------------------

def initial_state(user_message: str) -> AgentState:
    """Build the same state dict main.py sends for a user turn."""
    return {
        "messages": [HumanMessage(content=user_message)] if user_message else [],
        "enhanced_query": "",
        "department_targets": [],
        "subagent_results": {},
        "next_message": None,
        "token_usage": {},
    }


def run_turn(agent, thread_id: str, user_message: str) -> dict:
    """Send one user turn, logging every intermediate message."""
    LOG.info("USER TURN >>> %r", user_message)
    result = agent.invoke(
        initial_state(user_message),
        config={"configurable": {"thread_id": thread_id}},
    )
    for m in result.get("messages", []):
        kind = getattr(m, "type", type(m).__name__)
        preview = str(getattr(m, "content", m)).replace("\n", " ")[:100]
        if kind == "tool":
            calls = getattr(m, "tool_call_id", "?")
            LOG.info("  [TOOL MESSAGE id=%s] %s", calls, preview)
        else:
            LOG.info("  [%s] %s", kind.upper(), preview)
    return result


def main() -> int:
    setup_logging()
    banner("DEEP AGENT — DEVELOPER HARNESS")

    # -- 1. Tool discovery from ./tools -----------------------------------
    step("1. Tool discovery — what does ./tools yield?")
    discovered = discover_tools("./tools")
    dynamic = load_dynamic_tools("./tools")
    print(f"  discover_tools()  -> {[t['name'] for t in discovered]}")
    print(f"  load_dynamic_tools-> {sorted(dynamic)}")
    for t in discovered:
        print(f"    - {t['name']}: risk={t['risk_level']} roles={t['allowed_roles']} "
              f"src={Path(t['source_file']).name}")

    # -- 2. Department tool allowlists -------------------------------------
    step("2. Tool registry and per-department allowlists")
    full_registry = create_tool_registry()
    scoped = tools_dir_only_registry("./tools")
    print(f"  create_tool_registry() (built-ins + dynamic) -> {len(full_registry._specs)} tools")
    print(f"    {sorted(full_registry._specs)}")
    print(f"  tools/ scoped registry                          -> {len(scoped._specs)} tools")
    print(f"    {sorted(scoped._specs)}")
    # A department's tools come from its SKILL.md `allowed-tools` allowlist,
    # not from a role lookup: `subagents.select_department_tools` resolves the
    # names (including `server*` namespace claims) against the live tool set.
    toolset = get_all_tools()
    for dept in list_departments():
        spec = SUBAGENTS[dept["name"]]
        resolved = select_department_tools(spec, "test query", tools_dict=toolset)
        print(f"  dept {dept['name']!r:18} allows {list(spec.tools)}")
        print(f"  {'':25}resolves to {[t.name for t in resolved]}")

    # -- 3. Multi-turn session --------------------------------------------
    step("3. Multi-turn session — memory, enhanced query, departments")
    thread_id = "dev-harness-thread"
    tool_names = sorted(scoped._specs)

    # `select_department_tools` shortlists with a live LLM call
    # (`relevance.sort_tools`) whenever a department allows more than 5 tools.
    # `general` allows 9, so the shortlist decides whether `text_stats` reaches
    # the sub-agent at all — and that is a live API call whose ranking is not
    # this harness's to assert on. Stub it to keyword-match the same hints the
    # fake sub-agent uses, so the run is deterministic and offline. The
    # shortlist's *wiring* is still exercised: allowlist -> cap -> shortlist ->
    # real BaseTool objects bound.
    def _deterministic_sort(_model, query, tool_defs, max_tools=5):
        lowered = str(query).lower()
        hinted = [t for t in tool_defs if any(h in lowered for h in TOOL_HINTS.get(t["name"], ()))]
        chosen = hinted or list(tool_defs)
        return [{"name": t["name"], "reason": "harness keyword match"} for t in chosen[:max_tools]]

    import src.services.tools_integration.relevance as relevance_mod
    relevance_mod.sort_tools = _deterministic_sort  # type: ignore[assignment]

    dispatcher = ScriptedDispatcher(tool_names)
    agent_factory.get_model = lambda: dispatcher  # type: ignore[assignment]
    graph.reset_deep_agent()
    agent = graph.get_deep_agent()
    LOG.info("Compiled graph for thread %s", thread_id)

    # Turn 1: a tool-needing query.
    r1 = run_turn(agent, thread_id, "count the words in this sentence please")
    answer1 = _final_answer(r1)
    print(f"\n  ANSWER 1: {answer1}")
    print(f"  enhanced_query     : {r1.get('enhanced_query')!r}")
    print(f"  department_targets : {r1.get('department_targets')!r}")

    # session_memory must now hold turn 1.
    hist1 = get_session(thread_id)
    print(f"\n  session_memory after turn 1: {len(hist1)} messages")
    for m in hist1:
        print(f"    [{getattr(m, 'type', '?')}] {str(m.content)[:70]}")

    # Turn 2: a conversational follow-up — proves history reaches the dispatcher.
    print()
    r2 = run_turn(agent, thread_id, "thanks, now summarise what you just did")
    answer2 = _final_answer(r2)
    print(f"\n  ANSWER 2: {answer2}")
    print(f"  enhanced_query     : {r2.get('enhanced_query')!r}")
    print(f"  department_targets : {r2.get('department_targets')!r}")

    hist2 = get_window(thread_id, max_messages=20)
    print(f"\n  window handed to dispatcher on turn 2: {len(hist2)} messages")
    for m in hist2:
        print(f"    [{getattr(m, 'type', '?')}] {str(m.content)[:70]}")

    # -- 4. Assertions ----------------------------------------------------
    banner("RESULTS")
    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "tools/ discovery finds its tools",
        {"text_stats", "get_current_time"} <= set(dynamic),
        f"found {sorted(dynamic)}",
    ))
    checks.append((
        "scoped registry exposes only tools/ tools",
        set(scoped._specs) == set(dynamic),
        f"registry={sorted(scoped._specs)}",
    ))
    checks.append((
        "full registry is a superset (built-ins included)",
        set(full_registry._specs) > set(scoped._specs),
        f"{len(full_registry._specs)} vs {len(scoped._specs)}",
    ))
    checks.append((
        "a sub-agent dispatched a real tool call",
        bool(dispatcher.subagent_tool_calls),
        f"calls={[(t, q[:20]) for t, q, _ in dispatcher.subagent_tool_calls]}",
    ))
    checks.append((
        "tool result is summarised to the user",
        bool(answer1) and any(
            result[:40] in answer1 for _, _, result in dispatcher.subagent_tool_calls
        ),
        f"answer1={answer1!r}",
    ))
    checks.append((
        "orchestrator bound no tools (pure router)",
        not any(
            getattr(m, "type", "") == "tool" for m in r1.get("messages", [])
        ),
        "the router's own message state holds no ToolMessage",
    ))
    checks.append((
        "session_memory recorded turn 1",
        len(hist1) > 0,
        f"{len(hist1)} messages persisted",
    ))
    checks.append((
        "turn 2 saw turn 1's history",
        len(hist2) > len(hist1) and any(
            "count the words" in str(m.content) for m in hist2
        ),
        f"window grew {len(hist1)} -> {len(hist2)} and contains turn 1 text",
    ))
    checks.append((
        "department envelope produced enhanced_query",
        bool(r2.get("department_targets")) and bool(r2.get("enhanced_query")),
        f"depts={r2.get('department_targets')} enhanced={r2.get('enhanced_query')!r}",
    ))
    checks.append((
        "no raw JSON envelope leaked to the user",
        '"enhanced_query"' not in (answer2 or ""),
        f"answer2={answer2!r}",
    ))
    checks.append((
        "subagent_results cleared between turns",
        r2.get("subagent_results") == {},
        f"subagent_results={r2.get('subagent_results')}",
    ))

    failures = 0
    for label, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"  [{mark}] {label}")
        print(f"         {detail}")

    print(f"\n  {len(checks) - failures}/{len(checks)} checks passed.")
    print("  full log: /tmp/harness.log\n")
    return 1 if failures else 0


def _final_answer(result: dict) -> str:
    for m in reversed(result.get("messages", [])):
        if isinstance(m, AIMessage):
            return str(m.content)
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
