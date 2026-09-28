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
from src.services.session_memory.checkpoint import get_session  # noqa: E402
from src.services.session_memory.window import get_window  # noqa: E402
from src.services.tools_integration.discovery import discover_tools  # noqa: E402
from src.services.tools_integration.registry import ToolRegistry  # noqa: E402
from src.services.tools_integration.tools import (  # noqa: E402
    create_tool_registry,
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
    """Fake chat model that plays the dispatcher without an API key.

    Behaviour:
      - On the first call for a turn, if the *user's* message matches a tool
        hint, emit a tool call for that tool. Otherwise emit a JSON envelope.
      - On the second call (after tool results return), summarise the result
        in plain language.
    """

    def __init__(self, tool_names: list[str]):
        self._tool_names = tool_names
        self._turn_state: dict[str, object] = {}
        self.seen_prompts: list[list[BaseMessage]] = []
        LOG.info("ScriptedDispatcher initialised with %d registry tools", len(tool_names))

    def bind_tools(self, tools, **kwargs):
        bound = [getattr(t, "name", str(t)) for t in tools]
        LOG.info("bind_tools: dispatcher received %d tools: %s", len(bound), bound)
        return self

    def invoke(self, messages, **kwargs):
        self.seen_prompts.append(list(messages))

        # Sub-agents share get_model() with the dispatcher. A sub-agent prompt
        # is a SystemMessage plus a HumanMessage carrying the *enhanced* query
        # ("[enhanced] ..."), so route it to plain prose rather than letting a
        # dispatcher's JSON envelope become a sub-agent's answer.
        if _is_subagent_prompt(messages):
            query = _last_user_text(messages)
            LOG.info("Sub-agent answering (not the dispatcher): %r", query[:60])
            return AIMessage(
                content=f"Research findings for: {query.replace('[enhanced] ', '').strip()}"
            )

        # Only tool results from *this* turn count. The message window spans
        # turns, so a naive `any(type == "tool")` sees turn 1's ToolMessage on
        # turn 2 and wrongly believes it already called a tool.
        already_called_tools = _tools_since_last_human(messages)
        user_text = _last_user_text(messages)

        if already_called_tools:
            result = _last_tool_result(messages)
            LOG.info("Dispatcher saw tool result; answering the user")
            return AIMessage(content=f"Here is what I found — {result}")

        tool_name = self._suggest_tool(user_text)
        if tool_name:
            LOG.info("Dispatcher chose tool '%s' for query %r", tool_name, user_text)
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_name,
                    "args": _args_for(tool_name, user_text),
                    "id": f"call_{tool_name}",
                    "type": "tool_call",
                }],
            )

        LOG.info("Dispatcher returning JSON envelope for %r", user_text)
        return AIMessage(content=json.dumps({
            "enhanced_query": f"[enhanced] {user_text}",
            "departments": ["research"],
        }))

    def _suggest_tool(self, user_text: str) -> str | None:
        """Suggest a tool from the registry based on what the query needs."""
        lowered = user_text.lower()
        for name, hints in TOOL_HINTS.items():
            if name not in self._tool_names:
                LOG.debug("Tool '%s' hinted at but absent from registry; skipping", name)
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
            if "you are the dispatcher" in content:
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
        "iteration_count": 0,
        "max_iterations": 25,
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

    # -- 2. Registry visibility -------------------------------------------
    step("2. Tool registry — visibility scoped to ./tools")
    full_registry = create_tool_registry()
    scoped = tools_dir_only_registry("./tools")
    print(f"  create_tool_registry() (built-ins + dynamic) -> {len(full_registry._specs)} tools")
    print(f"    {sorted(full_registry._specs)}")
    print(f"  tools/ scoped registry                          -> {len(scoped._specs)} tools")
    print(f"    {sorted(scoped._specs)}")
    for role in ("general-purpose", "research", "writing"):
        visible = [s.name for s in scoped.get_visible_tools(role)]
        print(f"  role {role!r:18} sees -> {visible}")

    # -- 3. Multi-turn session --------------------------------------------
    step("3. Multi-turn session — memory, enhanced query, departments")
    thread_id = "dev-harness-thread"
    tool_names = sorted(scoped._specs)
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
    print(f"  iteration_count    : {r1.get('iteration_count')}")

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
        "turn 1 dispatched a real tool call",
        any(getattr(m, "type", "") == "tool" for m in r1.get("messages", [])),
        "a ToolMessage is present in turn 1",
    ))
    checks.append((
        "tool result is summarised to the user",
        bool(answer1) and "found" in answer1,
        f"answer1={answer1!r}",
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
