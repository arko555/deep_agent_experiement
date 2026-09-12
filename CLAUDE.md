# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --extra dev          # runtime deps + pytest/ruff/mypy (streamlit is the separate `ui` extra)
uv run pytest                # full suite (no API keys or network needed)
uv run pytest tests/test_routing.py::test_name   # single test
uv run ruff check .          # lint
uv run mypy src              # type check
uv run python main.py        # interactive CLI REPL over the compiled graph
uv sync --extra ui && uv run streamlit run app.py   # Streamlit UI
```

pytest config lives in `pyproject.toml` (`testpaths=["tests"]`, `pythonpath=["."]`, `addopts="--strict-markers -v"`), so bare `pytest` works from the repo root.

**Environment:** `.env` needs one provider key plus `TAVILY_API_KEY` for web search (the `research` subagent uses it). Provider keys are read in priority order — see *Model selection* below. `WORKSPACE_ROOT` optionally relocates `./workspace`; `MCP_SERVERS` and `A2A_AGENTS` enable external integrations. There is no `.env.example` in the repo despite what README/RUNBOOK say.

## Architecture

A LangGraph state machine implementing a "deep agent": one orchestrator LLM plans, delegates to subagents, and routes its own work through adversarial reviewers before delivering a final answer. `app.py` (Streamlit) and `main.py` (CLI) are two entry points over the **same cached compiled graph**; `agent.py` is just a `get_deep_agent()` re-export.

### Graph flow

```
START → orchestrator
  next_message has tool_calls        → agent → tools → orchestrator
  no tool_calls, budget exhausted    → responder → END
  no tool_calls, current_plan set    → plan_checker → compliant ? critic : orchestrator
  no tool_calls, no plan             → critic → approved ? responder
                                              → rejected & iteration_count >= 4 → reflection → orchestrator
                                              → rejected otherwise               → orchestrator
```

Routing lives in `src/core/routing.py` and branches on **explicit state fields** (`next_message.tool_calls`, `review_verdict`, `current_plan`, `iteration_count`) — never on message text. Keep it that way: inspecting response content for keywords is what Phase 1 fixed.

### The `next_message` staging pattern (central mechanism)

The orchestrator never appends to `messages` itself — each LLM turn is written to `next_message`. The `agent` node moves staging into `messages` before tools run. Reviewer rejections *also* stage their critique into `next_message`; the next orchestrator turn consumes it as explicit revision feedback (appended as a `HumanMessage`) and then **overwrites** it. `responder` delivers the final answer and clears staging. Any node you add that writes `next_message` must account for the orchestrator overwriting it every turn.

### Conversation history is owned by the checkpointer

The graph is compiled with `MemorySaver`. Callers pass **only** the new `HumanMessage` plus `iteration_count: 0` and a `configurable.thread_id`; the rest of the history comes from the checkpoint. Every UI session, CLI process, and subagent graph invocation uses its own `thread_id` (a subagent reusing the parent's thread would cross-contaminate checkpoints). Don't "helpfully" pass full `messages` from a new caller.

### Subagents are data, not code paths (`src/core/subagents.py`)

`SUBAGENTS` is the single source of truth: it drives the `task` tool's `Literal` type, its docstring, `_execute_task` dispatch, the parallel-vs-sequential split, and prompt construction. **Adding a subagent type = adding one `SubagentSpec` entry.** Three kinds:

- `kind="tool_loop"` (research, writer) → `run_tool_loop` runs the spec's restricted toolset with a prompt of `COMPLETION_CONTRACT` + `skills/<skill>/SKILL.md` body + AGENTS.md.
- `kind="graph"` (general-purpose) → re-invokes the full compiled graph at `recursion_depth + 1`.
- `kind="a2a"` (config-driven) → calls a remote agent at `spec.url`; see *External integrations* below.

The `task` tool body deliberately **refuses direct invocation** (6.4): it is always executed by `local_tools_node`, where recursion depth and child-token aggregation have the graph state. Never call `_execute_task` from a tool body or bypass the tools node.

Parallelizable types run in a `ThreadPoolExecutor` under **one shared deadline for the whole batch** (a hung batch costs one timeout, not N×); results are collected in submission order so `ToolMessage` ordering is deterministic. Child token usage and file writes fold back into the parent's `token_usage` / `pending_writes`.

### Caches and invalidation

- Compiled graph (`_compiled_graph`) and provider clients (`_model_cache`) — cleared by `reset_deep_agent()`. Call it when tools, skills, or model config change at runtime.
- Skills summary and tools summary (`src/core/memory.py`) — invalidated by `_dir_tree_hash` (mtime+size of the tree).
- Dynamic tools (`tools.load_dynamic_tools`) — same tree-hash invalidation; modules are `exec`'d via `importlib` (nothing is added to `sys.path`), built-ins win name collisions.

### Guardrails are asymmetric by design

Writes: `write_file`/`edit_file` are forced into `./workspace` via `validate_and_normalize_path(..., must_be_in_workspace=True)`. Reads: **default-deny allowlist** — only `AGENTS.md`, `workspace/`, `skills/` via `validate_read_path`. Don't loosen one without the other; reading `.env` is intentionally blocked.

### Model selection and configuration

`src/core/config.py` is the single home for budgets, read from env **at call time** (so `.env` edits apply on the next invocation): `AGENT_MAX_ITERATIONS` (25), `AGENT_MAX_SUBAGENT_DEPTH` (3; child runs at parent+1, delegation rejected when parent depth ≥ limit), `MAX_PARALLEL_TASKS` (4), `SUBAGENT_TIMEOUT_SECONDS` (600). `get_model()` picks a provider by key priority — Anthropic → OpenRouter → OpenAI → Google → Ollama — with `*_MODEL` env vars overriding the defaults. `OBSERVABILITY=1` attaches the `DeepAgentTracer` `BaseCallbackHandler` to every model through `get_model()`.

### Tests

`tests/fake_models.py::ScriptedChatModel` is a `.invoke`-able fake installed via `monkeypatch.setattr(agent_factory, "get_model", ...)`, so the whole graph runs deterministically with no API keys. It identifies the acting role from a substring of the first (system) message: `"generic Deep Agent"` (orchestrator), `"Critical Reviewer"` (critic), `"Plan Compliance"` (plan_checker), `"Reflection Agent"`, `"specialized subagent"` (tool-loop subagents). **Rewording those prompt openings silently breaks role detection** (the fake falls through to `default`). Tests that touch files chdir into `tmp_path`.

### External integrations: MCP tools and A2A subagents

`MCP_SERVERS` (JSON) loads tools from MCP servers into `get_all_tools()`, tiered built-ins > MCP > dynamic file tools. `A2A_AGENTS` (JSON) registers remote agents as `kind="a2a"` subagent types, invocable through `task` and parallelized under the shared batch deadline.

**Both SDKs are async-only and the graph is 100% synchronous.** MCP tools are built with `coroutine=` and no sync `func`, so `tool.invoke()` raises `NotImplementedError` — this is settled upstream, not a version quirk. `src/core/async_bridge.py` (`run_sync`) drives a persistent event loop on a daemon thread, and MCP tools are re-wrapped as sync `StructuredTool`s so no call site changes. Don't try to "simplify" the wrappers away or call `asyncio.run` directly: per-call `asyncio.run` has documented hangs with MCP and fails outright on a thread that already has a loop.

**Config read-time asymmetry — deliberate:** `MCP_SERVERS` is read at call time (matching the rest of `config.py`), but `A2A_AGENTS` is read at **import** time because it feeds the `task` tool's static `Literal` enum. `subagents.py` therefore calls `load_dotenv()` itself, since it is imported before the `load_dotenv()` in `agent_factory.py`. Changing A2A agents needs a restart; changing MCP servers does not.

Test fakes for both live in `tests/test_mcp.py` / `tests/test_a2a.py` and patch `run_sync` with a plain event loop, so no servers are spawned and no bridge thread starts. `tests/test_a2a.py` builds real `a2a.types` protobuf chunks.

### Docs: PHASES.md is authoritative

`PHASES.md` is the current implementation record and gap audit; Phases 1–10 are complete. `README.md`, `RUNBOOK.md`, and `ARCHITECTURE_EXPLANATION.md` are **stale** — they still describe removed HITL/`interrupt()` approval, deleted `src/nodes/research.py` and `src/nodes/write.py`, and pre-Phase-1 routing that always returned `"orchestrator"`. Trust the code and `PHASES.md`.

---

## Working Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
