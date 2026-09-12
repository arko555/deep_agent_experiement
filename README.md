# Deep Agent — Modular LangGraph Architecture

A production-grade **Deep Agent** built with LangChain and LangGraph. One orchestrator LLM plans, delegates to subagents, and routes its own work through adversarial reviewers before delivering a final answer.

---

## 🌟 Why This Architecture?

Standard LLM implementations suffer from "context drift" (losing the goal) and unverified answers. The Deep Agent architecture solves these through:

*   **High Accuracy & Reliability**:
    *   **Adversarial Review**: A dedicated **Critic** node challenges the agent's reasoning before a final answer is delivered; a **Plan Checker** verifies the agent actually completes the tasks it set for itself.
    *   **Self-Correction**: After 4+ iterations of critic rejections, a **Reflection** node breaks the loop and generates a revised strategy instead of spinning forever.
*   **Enterprise-Grade Governance**:
    *   **Auditability**: Every tool call, subagent delegation, and review decision is recorded in a structured `audit_log`.
    *   **Sandboxed file access**: Writes are forced into `./workspace`; reads use a default-deny allowlist (`AGENTS.md`, `workspace/`, `skills/`) — the agent cannot read `.env` or source files.
    *   **Resource Management**: Iteration and token budgets prevent runaway loops; subagent delegation is depth-capped.
*   **Extensibility**:
    *   **Dynamic skills**: Drop a `SKILL.md` into `skills/` — no code changes required.
    *   **Registry-driven subagents**: Add a new agent type with one `SubagentSpec` entry (`src/core/subagents.py`).
    *   **External integrations**: Attach tools from MCP servers and remote agents via A2A, both config-driven (see [External integrations](#-external-integrations)).

---

## 🏗️ Architecture Deep Dive

The system is a **stateful LangGraph state machine**. The orchestrator never appends to `messages` itself — each LLM turn is staged in `next_message`; the `agent` node moves staging into `messages` before tools run. Reviewer rejections also stage their critique as explicit revision feedback for the next orchestrator turn.

Conversation history is owned by a **checkpointer** (`MemorySaver`). Callers pass only the new `HumanMessage`, `iteration_count: 0`, and a `configurable.thread_id`; the rest of the history comes from the checkpoint. Each UI session, CLI process, and subagent graph invocation uses its own thread id.

### The Orchestration Loop
The `orchestrator` is the "brain":
1.  **Plans** — creates a roadmap with the `write_todos` tool.
2.  **Delegates** — spawns subagents via the `task` tool (research, writer, general-purpose, or remote A2A agents).
3.  **Verifies** — results pass through the **Critic** and **Plan Checker** before delivery.

### AgentState
Shared state flows through every node: `messages` (append-only), `current_plan`, `next_message` (staging), `review_verdict` (explicit routing signal), `recursion_depth`, `pending_writes` / `audit_log` (audit), `workspace_files`, `token_usage`, `iteration_count` / `max_iterations`.

### Subagents are data, not code paths
`SUBAGENTS` in `src/core/subagents.py` is the single source of truth: it drives the `task` tool's schema and docstring, dispatch, parallel-vs-sequential split, and prompt construction. Three kinds:

| Type | Kind | What it does | Parallelizable |
|------|------|--------------|----------------|
| `research` | `tool_loop` | Web research with a restricted toolset (`internet_search`, `fetch_url`, file tools) + `skills/research/SKILL.md` prompt | ✅ |
| `writer` | `tool_loop` | Draft writing with file tools + `skills/writer/SKILL.md` prompt | ✅ |
| `general-purpose` | `graph` | Re-invokes the full compiled graph at `depth + 1` | ❌ (shared state) |
| `<name>` (configured) | `a2a` | Calls a remote agent over the A2A protocol (`A2A_AGENTS` env var) | ✅ |

Parallelizable tasks run in a `ThreadPoolExecutor` under **one shared deadline for the whole batch** (a hung batch costs one timeout, not N×); results are collected in submission order. Child token usage and file writes fold back into the parent's `token_usage` / `pending_writes`.

### Built-in tools
`write_todos`, `internet_search` (Tavily), `fetch_url`, `read_file`, `write_file`, `edit_file`, `list_files`, `search_files`, `task`, `list_tools` — plus dynamic tools from `tools/*.py` and MCP tools. Name-collision precedence: **built-ins > MCP > dynamic file tools**.

---

## 🔄 Graph Flow

Routing branches on **explicit state fields** (`next_message.tool_calls`, `review_verdict`, `current_plan`, `iteration_count`) — never on message text (see `src/core/routing.py`).

```text
START → orchestrator
  │
  ├─ has tool_calls                    → agent → tools → orchestrator
  ├─ no tool_calls, budget exhausted   → responder → END
  ├─ no tool_calls, current_plan set   → plan_checker
  │                                      ├─ compliant     → critic
  │                                      └─ non-compliant → orchestrator (reformulate)
  └─ no tool_calls, no plan            → critic
                                           ├─ approved                 → responder → END
                                           ├─ rejected, iter < 4      → orchestrator (reformulate)
                                           └─ rejected, iter >= 4     → reflection → orchestrator
```

Notes:
- `tools` execute file writes immediately; `pending_writes` tracks them for audit only. There is **no** human-in-the-loop approval gate (LangGraph `interrupt()` was removed — it is incompatible with Streamlit's request-response model).
- The **responder** delivers the final answer and clears staging.

---

## 🔌 External integrations

**MCP tools** — set `MCP_SERVERS` to a JSON object describing MCP servers; their tools load into `get_all_tools()` alongside built-ins with no per-server code changes. Read at call time; changes apply on the next invocation. The async-only MCP SDK is bridged to the synchronous graph via `src/core/async_bridge.py` (a persistent event loop on a daemon thread — don't replace it with per-call `asyncio.run`).

**A2A subagents** — set `A2A_AGENTS` to a JSON object of remote agents; each becomes a `kind="a2a"` entry in the registry, invocable through `task` and parallelized under the shared batch deadline. Read at **import** time (it feeds the `task` tool's static type enum), so changing A2A agents needs a process restart — unlike `MCP_SERVERS`.

---

## 🗺️ Module Reference

| File | Role |
|------|------|
| `agent.py` | Thin facade — re-exports `get_deep_agent()` |
| `main.py` | CLI REPL over the cached compiled graph |
| `app.py` | Streamlit UI — chat, action log, plan, skills, workspace files (the `ui` extra) |
| `src/state.py` | `AgentState` TypedDict |
| `src/core/agent_factory.py` | Graph construction, `tools` node (task batch execution), model selection + caching |
| `src/core/subagents.py` | `SUBAGENTS` registry + `SubagentSpec`; role prompt builder |
| `src/core/routing.py` | Conditional edges (explicit-state routing) |
| `src/core/tools.py` | Built-in tools, `task` tool, dynamic-tool loader, MCP tiering |
| `src/core/config.py` | All budgets/limits, read from env at call time |
| `src/core/memory.py` | `AGENTS.md` + `skills/` scanning, system prompts |
| `src/core/guardrails.py` | Path validation — write sandbox + default-deny read allowlist |
| `src/core/mcp_client.py` | MCP server tool loading (cached by config hash) |
| `src/core/a2a_client.py` | A2A remote-agent calls |
| `src/core/async_bridge.py` | Sync-over-async bridge for the async-only SDKs |
| `src/core/rag.py` | `internet_search()` — wraps Tavily API |
| `src/core/summarization.py` | Conversation summarization when history grows large |
| `src/core/utils.py` | Shared LLM retry wrapper |
| `src/nodes/plan.py` | `orchestrator` node; `write_todos` tool |
| `src/nodes/review.py` | `agent`, `critic`, `plan_checker`, `reflection`, `responder` nodes |
| `skills/` | `SKILL.md` files for subagent role prompts |
| `tools/` | Custom dynamically-loaded LangChain tools |
| `AGENTS.md` | Shared conventions injected into prompts |

---

## 🚀 Getting Started

### Prerequisites
- [uv](https://github.com/astral-sh/uv) package manager
- At least one LLM provider key: Anthropic, OpenRouter, OpenAI, or Google (Ollama also supported for local models)
- Tavily API key (for web search — the `research` subagent uses it)

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
| `AGENT_MAX_ITERATIONS` | `25` | Orchestrator loop budget per turn |
| `AGENT_MAX_SUBAGENT_DEPTH` | `3` | Max delegation nesting (child runs at parent+1) |
| `MAX_PARALLEL_TASKS` | `4` | Concurrent subagent tasks |
| `SUBAGENT_TIMEOUT_SECONDS` | `600` | Shared deadline per task batch |
| `OBSERVABILITY=1` | off | Attach the `DeepAgentTracer` callback handler to all models |
| `MCP_SERVERS` | — | JSON object of MCP servers (read at call time) |
| `A2A_AGENTS` | — | JSON object of remote agents (read at import — restart to change) |

Model provider priority: **Anthropic → OpenRouter → OpenAI → Google → Ollama** (first key present wins).

### Running the Application

**CLI REPL:**
```bash
uv run python main.py
```

**Streamlit UI:**
```bash
uv run streamlit run app.py
```

**Programmatically** (the checkpointer owns history — pass only the new message plus a fresh iteration budget and a thread id):

```python
from langchain_core.messages import HumanMessage
from agent import get_deep_agent

agent = get_deep_agent()
config = {"configurable": {"thread_id": "my-thread"}}

result = agent.invoke(
    {"messages": [HumanMessage(content="Research AI agents and draft a blog post.")],
     "iteration_count": 0},
    config=config,
)
```

> Don't pass full `messages` from a new caller — the history comes from the checkpoint. Subagent invocations use their own thread ids to avoid cross-contaminating checkpoints.

---

## ➕ Adding Skills

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
  One-sentence description the agent uses to decide when to load this skill.
---

# My Skill

## Instructions
1. Step one...
2. Step two...
```

The agent discovers new skills at runtime — no code changes needed (summaries are cached and invalidated on filesystem changes).

## ➕ Adding Subagents

Add one `SubagentSpec` entry to `_BUILTIN_SUBAGENTS` in `src/core/subagents.py`:

```python
"translator": SubagentSpec(
    description="Translates workspace documents; reads the source file and writes the translation.",
    kind="tool_loop",
    parallelizable=True,
    skill="translator",
    tools=("read_file", "write_file"),
),
```

The `task` tool schema, docstring, dispatch, and parallel grouping all pick it up automatically.

---

## 🧪 Testing & Development

```bash
uv run pytest                # 148 tests — no API keys or network needed
uv run ruff check .          # lint
uv run mypy src              # type check
```

Tests use `tests/fake_models.py::ScriptedChatModel`, a deterministic fake LLM installed via `monkeypatch`, so the whole graph runs with no real API calls. External-integration tests patch the async bridge with plain event loops, so no MCP/A2A servers are spawned.
