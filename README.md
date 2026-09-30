# Deep Agent — Modular LangGraph Architecture

A production-grade **Deep Agent** built with LangChain and LangGraph. One
orchestrator LLM routes each request to specialist department sub-agents, which
run their own tool loops in parallel and return grounded answers.

---

## 🌟 Why This Architecture?

Standard LLM implementations blur tool use, planning, and answering into one
context window. The Deep Agent architecture separates them:

*   **One-way hierarchy**: `orchestrator → sub-agent → tool`. The orchestrator
    is a pure **router** — it enhances the query and names the departments that
    should handle it, and binds no tools. It cannot drift into doing specialist
    work itself, and it runs exactly once per turn.
*   **Context isolation**: each department sees only the enhanced query, its
    own `SKILL.md` role prompt, and the 3-5 tools shortlisted for *this* query.
    Parallel branches cannot see or corrupt each other's context.
*   **Governance**:
    *   **Auditability**: sub-agent token spend and every file write fold back
        into the parent's `token_usage` / `pending_writes` / `audit_log`.
    *   **Sandboxed file access**: writes are forced into `./workspace`;
        reads use a default-deny allowlist (`AGENTS.md`, `workspace/`,
        `skills/`) with symlink-resolved path checks — the agent cannot read
        `.env` or source files.
    *   **Resource management**: the only iteration budget in the system bounds
        a sub-agent's ReAct tool loop (`AGENT_MAX_ITERATIONS`); each department
        runs under a shared wall-clock deadline.
*   **Extensibility**:
    *   **Dynamic departments**: drop a `SKILL.md` into `skills/` — no code
        changes required.
    *   **Dynamic tools**: drop a Python file with an `@tool` function into
        `tools/`.
    *   **External integrations**: attach tools from MCP servers and remote
        agents via A2A, both config-driven (see [External integrations](#-external-integrations)).

---

## 🏗️ Architecture Deep Dive

The system is a **stateful LangGraph state machine** with three nodes and one
direction — there is no top-level ReAct loop:

```text
START → orchestrator → subagent_fanout → responder → END
                ↘                       → responder ↗
```

- **orchestrator** — one LLM call. Reads the message window from
  `session_memory`, restates the request with conversation context folded in
  (`enhanced_query`), and names the departments that should handle it
  (`department_targets`), returned as a JSON envelope
  `{enhanced_query, departments}`. The department roster in its prompt is
  built from the live `skills/` directory on every call; invented names are
  dropped by validation, and an unusable reply falls back to the `general`
  catch-all department, so routing always lands somewhere.
- **subagent_fanout** — runs each named department concurrently
  (`asyncio.gather` in batches of `MAX_PARALLEL_TASKS`), each under one shared
  wall-clock deadline (`SUBAGENT_TIMEOUT_SECONDS`). Departments marked
  `parallelizable: false` in their SKILL.md run after the parallel group, one
  batch at a time. One department failing or timing out becomes an error
  string in its slot — it never drops its siblings' results.
- **responder** — verifies fanout output with cheap deterministic checks
  (non-empty, no `/tmp/` references; keyword overlap is advisory only),
  aggregates multi-department results under `## <name>` headers, delivers the
  answer, and clears the staging channels (`subagent_results`,
  `next_message`) so nothing from this turn leaks into the next.

The graph is **acyclic** — fanout feeds the responder directly. A second
router call would re-decide a query that has already been routed.

### The one iteration budget lives where iteration happens

Tool calls happen **only** inside a sub-agent's ReAct loop
(`subagents.run_tool_loop`): bind the shortlisted tools, invoke, execute tool
calls, append results, repeat until the model answers without tool calls or
`AGENT_MAX_ITERATIONS` (default 25) turns are spent. There is deliberately no
counter in the parent graph state — the router runs once per turn and has
nothing to count, and a checkpointed counter would carry across turns and
break routing on turn 2 of every conversation.

### How a department runs

1. The department *name* resolves to a `SubagentSpec` in `SUBAGENTS`, built
   from `skills/*/SKILL.md` plus configured A2A agents. There is no hardcoded
   general-purpose spec — `skills/general` is the catch-all, a department like
   any other.
2. `select_department_tools` resolves the SKILL.md `allowed-tools` allowlist
   into real, bindable tools: exact names or `prefix*` namespace claims (how a
   department claims an MCP server's tools), capped at 20, then narrowed to
   3-5 by an LLM relevance sort for this query. Unknown names are dropped with
   a warning — a stale SKILL.md can never yield an uncallable tool.
3. `run_tool_loop` runs the ReAct loop with the SKILL.md body (plus the shared
   completion contract and `AGENTS.md`) as the system prompt. Every tool call
   dispatches through one execution surface — `ToolRegistry.dispatch` — which
   validates the payload against the tool's declared schema and routes by the
   tool's declared kind (`local` / `mcp` / `a2a`).
4. The loop returns `(final_text, token_usage, file_writes)`; the fanout node
   folds the usage and writes into the parent's audit trail.

A payload that does not match a tool's schema is **rejected and named back to
the model**, never coerced — the sub-agent gets the error as a `ToolMessage`
and another turn to correct it. MCP tools are normalized from raw JSON Schema
dicts into Pydantic models so they enforce the same contract as built-ins.

### Built-in tools

`write_todos`, `internet_search` (Tavily), `fetch_url` (SSRF-guarded), `read_file`,
`write_file`, `edit_file`, `list_files`, `search_files`, `list_tools` — plus
dynamic tools from `tools/*.py` and MCP server tools. Name-collision
precedence: **built-ins > MCP > dynamic file tools**.

---

## 🔌 External integrations

**MCP tools** — set `MCP_SERVERS` to a JSON object describing MCP servers;
their tools load into `get_all_tools()` alongside built-ins with no per-server
code changes. Read at call time; changes apply on the next invocation. The
async-only MCP SDK is bridged to the synchronous graph via
`src/async_bridge.py` (a persistent event loop on a daemon thread — don't
replace it with per-call `asyncio.run`).

**A2A departments** — set `A2A_AGENTS` to a JSON object of remote agents; each
becomes a `kind="a2a"` department the router can select like any other. Remote
calls report no local token usage or file writes, by design. Read at **import**
time (the registry is built statically), so changing A2A agents needs a
process restart — unlike `MCP_SERVERS`.

```dotenv
MCP_SERVERS='{"math": {"transport": "stdio", "command": "python", "args": ["/abs/server.py"]},
              "weather": {"transport": "http", "url": "http://localhost:8000/mcp"}}'
A2A_AGENTS='{"remote_researcher": {"url": "http://localhost:9999", "description": "..."}}'
```

---

## 🗺️ Module Reference

| File | Role |
|------|------|
| `agent.py` | Thin facade — re-exports `get_deep_agent()` |
| `main.py` | CLI — interactive REPL, one-shot query, `--tools` listing |
| `app.py` | Streamlit UI — chat, audit log, skills, workspace files (the `ui` extra) |
| `src/services/agent_orchestrator/graph.py` | Graph compilation — 3 nodes, 1 direction; the compiled-graph cache |
| `src/services/agent_orchestrator/orchestrator.py` | The router — dispatcher prompt, JSON parsing, department validation |
| `src/services/agent_orchestrator/routing.py` | The conditional edge out of the router |
| `src/services/agent_orchestrator/state.py` | `AgentState` TypedDict (no iteration counter — by design) |
| `src/services/agent_orchestrator/subagents.py` | `SUBAGENTS` registry, tool shortlisting, the ReAct tool loop |
| `src/services/agent_orchestrator/subagent_engine.py` | Parallel department batching and dispatch |
| `src/services/agent_orchestrator/agent_factory.py` | Model selection + caching, observability tracer, reset hooks |
| `src/services/agent_orchestrator/aggregator.py` | Merge department results |
| `src/services/agent_orchestrator/verification.py` | Deterministic checks on fanout output |
| `src/services/agent_orchestrator/memory.py` | SKILL.md parsing, skills summary, `AGENTS.md` |
| `src/services/session_memory/` | Checkpointer (shared `MemorySaver`), message window, compression |
| `src/services/tools_integration/tools.py` | Built-in tools, dynamic-tool loader, tool-set precedence |
| `src/services/tools_integration/registry.py` | `ToolRegistry.dispatch` — the single tool execution surface |
| `src/services/tools_integration/validation.py` | Payload contract: schema normalization, rejection not coercion |
| `src/services/tools_integration/guardrails.py` | Path validation — write sandbox + default-deny read allowlist |
| `src/services/tools_integration/mcp_client.py` | MCP server tool loading (cached by config hash) |
| `src/services/tools_integration/a2a_client.py` | A2A remote-agent calls |
| `src/services/tools_integration/research_fetch.py` | SSRF-safe URL fetch (IP pinning, redirect cap, size cap) |
| `src/services/tools_integration/relevance.py` | The 20 → 3-5 tool shortlist for a query |
| `src/services/tools_integration/discovery.py` | Scan `skills/` and `tools/`, cached by directory tree hash |
| `src/config.py` | All budgets/limits, read from env at call time |
| `src/async_bridge.py` | Sync-over-async bridge for the async-only SDKs |
| `skills/` | `SKILL.md` per department — prompt + tool allowlist |
| `tools/` | Custom dynamically-loaded LangChain tools |
| `AGENTS.md` | Shared conventions injected into sub-agent prompts |

---

## 🚀 Getting Started

### Prerequisites
- [uv](https://github.com/astral-sh/uv) package manager
- At least one LLM provider key: Anthropic, OpenRouter, OpenAI, or Google (Ollama also supported for local models)
- Tavily API key (for web search — the `research` and `marketing` departments use it)

### Setup
```bash
# 1. Install dependencies (add --extra ui for the Streamlit app)
uv sync --extra dev
uv sync --extra ui

# 2. Create .env in the repo root with your keys
```

There is no `.env.example`; create `.env` directly:

```dotenv
ANTHROPIC_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### Configuration (all optional — sensible defaults)

| Variable | Default | Purpose |
|----------|---------|---------|
| `*_MODEL` | provider default | Override the chosen model (`ANTHROPIC_MODEL`, `OPENAI_MODEL`, …) |
| `TAVILY_API_KEY` | — | Web search tool |
| `WORKSPACE_ROOT` | `./workspace` | Relocate the workspace directory |
| `AGENT_MAX_ITERATIONS` | `25` | Per-sub-agent ReAct turn budget (the only loop budget) |
| `MAX_PARALLEL_TASKS` | `4` | Concurrent sub-agent departments |
| `SUBAGENT_TIMEOUT_SECONDS` | `600` | Shared deadline per parallel batch |
| `OBSERVABILITY=1` | off | Attach the `DeepAgentTracer` callback handler to all models |
| `MCP_SERVERS` | — | JSON object of MCP servers (read at call time) |
| `A2A_AGENTS` | — | JSON object of remote agents (read at import — restart to change) |

Model provider priority: **Anthropic → OpenRouter → OpenAI → Google → Ollama**
(first key present wins). A `.env` change applies on the next
`reset_deep_agent()` — provider clients and the compiled graph are cached for
the process lifetime.

### Running the Application

**CLI REPL:**
```bash
uv run python main.py
```

**One-shot query:**
```bash
uv run python main.py "What are the leave policies?"
```

**List the tool registry:**
```bash
uv run python main.py --tools
```

**Streamlit UI:**
```bash
uv run streamlit run app.py
```

**Programmatically** (the checkpointer owns history — pass only the new
message plus a thread id):

```python
from langchain_core.messages import HumanMessage
from agent import get_deep_agent

agent = get_deep_agent()
config = {"configurable": {"thread_id": "my-thread"}}

result = agent.invoke(
    {"messages": [HumanMessage(content="Research AI agents and draft a blog post.")]},
    config=config,
)
```

> Don't pass full `messages` from a new caller — the history comes from the
> checkpoint. There is no per-turn state to reset; the router runs once per
> turn and carries no counters.

---

## ➕ Adding a Department

Create a new directory under `skills/`:

```text
skills/
└── my_skill/
    └── SKILL.md      ← YAML frontmatter + instructions
```

**`SKILL.md` template:**
```markdown
---
name: my_skill
description: >
  One-sentence description the ROUTER uses to decide when to route here.
  This is the only signal it has — write it for routing, not for humans.
allowed-tools: internet_search, fetch_url, read_file, write_file
parallelizable: true
---

# My Skill

## Instructions
1. Step one...
2. Step two...
```

That's the whole extension story: the router's roster is rebuilt from
`skills/` on every call, the body becomes the sub-agent's system prompt, and
`allowed-tools` is the hard tool allowlist (a trailing `my_server*` claims a
whole MCP server's tools). Call `reset_deep_agent()` after adding one in a
running process. Names are deduplicated — first in sorted directory order
wins, with a warning.

## ➕ Adding a Tool

Drop a Python file into `tools/`:

```python
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """What the tool does — the docstring IS the schema the model sees."""
    ...
```

It is discovered on the next `get_all_tools()`. Stack
`@tool_spec(name=..., risk_level=...)` underneath `@tool` to declare risk and
role metadata (see `tools/sample_tool.py` and `tools/text_stats.py`).

---

## 🧪 Testing & Development

```bash
uv run pytest                # 384 tests — no API keys or network needed
uv run ruff check .          # lint
uv run mypy src              # type check
uv run python main.py        # CLI REPL
```

Tests use `tests/fake_models.py::ScriptedChatModel`, a deterministic fake LLM
installed via `monkeypatch`, so the whole graph runs with no real API calls.
External-integration tests patch the async bridge with plain event loops, so
no MCP/A2A servers are spawned; `tests/test_mcp_live.py` is the one suite that
runs a real MCP server (`tests/fixtures/mcp_server.py`).

For the full architectural map — including why the code is shaped the way it
is, and the debugging gotchas — read [CODEBASE_GUIDE.md](CODEBASE_GUIDE.md).
