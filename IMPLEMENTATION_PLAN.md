# Implementation Plan — Three-Service Agent Architecture

## Architecture Overview

The system splits into **three independent services** that communicate through explicit interfaces:

```
┌──────────────────────────────────────────────────────────────────┐
│  SERVICE 1: Session Memory                                       │
│  Owns: conversation history, checkpointing, compression,       │
│        message window delivery                                    │
│  Imports: src.config (budgets), src.async_bridge (if needed)   │
│  Does NOT import: agent_orchestrator, tools_integration          │
├──────────────────────────────────────────────────────────────────┤
│  SERVICE 2: Agent Orchestrator                                   │
│  Owns: LangGraph, orchestrator node, routing, subagent engine, │
│        result aggregation, verification                           │
│  Imports: session_memory (get_window, append_message),           │
│           tools_integration (ToolRegistry, ToolExecutor)         │
├──────────────────────────────────────────────────────────────────┤
│  SERVICE 3: Tools Integration                                    │
│  Owns: tool registry (with risk metadata), auto-discovery,     │
│        relevance sorting, async execution, MCP/A2A bridges      │
│  Imports: src.config (budgets), src.async_bridge               │
│  Does NOT import: agent_orchestrator, session_memory             │
└──────────────────────────────────────────────────────────────────┘

Shared at `src/` root: state.py, config.py, async_bridge.py, utils.py
Entry points: agent.py, app.py, main.py (root level)
```

**Dependency direction:** `session_memory ← agent_orchestrator → tools_integration`. No circular dependencies.

---

## Directory Map

### Shared at `src/` root (used by all services)

| File | Purpose |
|---------|---------|
| `src/services/agent_orchestrator/state.py` | `AgentState` TypedDict — LangGraph state schema |
| `src/config.py` | Budget and limit getters (env-var driven) |
| `src/async_bridge.py` | Event loop on daemon thread for async-only integrations |
| `src/utils.py` | `invoke_with_retry`, `get_message_text`, `is_transient_error` |
| `src/types.py` | `SubAgent`, `ToolSpec` dataclasses |

### Session Memory — `src/services/session_memory/`

| File | Purpose |
|---------|---------|
| `checkpoint.py` | MemorySaver wrapper, `get_session`, `append_message`, token tracking |
| `compression.py` | `compress_messages` — tool-call-safe history reduction |
| `window.py` | `get_window(thread_id, max_messages)`, `append_message` |

### Agent Orchestrator — `src/services/agent_orchestrator/`

| File | Purpose |
|---------|---------|
| `graph.py` | LangGraph compilation: START → orchestrator → {fanout \| responder} → END |
| `orchestrator.py` | `call_orchestrator`: message window, query enhancement, routing, verification, synthesis |
| `routing.py` | `route_from_orchestrator`: fanout vs responder based on department detection |
| `subagent_engine.py` | Parallel dispatch via `asyncio.gather`, per-subagent ReAct loops |
| `aggregator.py` | Combine multi-department results into final answer |
| `verification.py` | Deterministic checks on sub-agent outputs |
| `subagents.py` | Subagent registry and registry-driven task dispatch |
| `agent_factory.py` | Model selection, graph compilation, `get_deep_agent`, `reset_deep_agent` |
| `plan.py` | Orchestrator node — planning step |
| `review.py` | Review nodes — reviewer, responder, critic, reflection, plan checker |
| `guardrails.py` | Workspace path validation and guardrails |
| `memory.py` | Workspace file management, skills/memory prompt assembly |

### Tools Integration — `src/services/tools_integration/`

| File | Purpose |
|---------|---------|
| `registry.py` | `ToolRegistry` — risk-tiered execution, role-scoped visibility |
| `discovery.py` | Auto-discover skills/*/SKILL.md + tools/*.py `@tool_spec` tags |
| `executor.py` | `ToolExecutor` — async, risk-tiered, auth hooks |
| `relevance.py` | `sort_tools` — 20 visible → 3-5 relevant via LLM |
| `mcp_client.py` | MCP tool connectivity (sync-wrapped) |
| `mcp_bridge.py` | Async MCP bridge |
| `a2a_client.py` | A2A remote agent connectivity |
| `a2a_bridge.py` | Async A2A bridge |
| `tools.py` | Built-in tool definitions with `@tool_spec` metadata |
| `research_fetch.py` | Web research/fetch tool |
| `rag.py` | RAG retrieval tool |
| `decorator.py` | `@tool_spec` decorator for tool metadata |

### Unchanged (stay at repo root)

```
tools/                # Tool definition files (now use @tool_spec decorator)
skills/               # Department sub-agents (SKILL.md + tools)
app.py                # Streamlit entry point (updated imports)
main.py               # CLI entry point (updated imports)
agent.py              # get_deep_agent facade (updated imports)
tests/                # Reorganized to mirror service structure
pyproject.toml        # Pytest config (unchanged: pythonpath=["."])
```

**Removed:** `src/core/`, `src/nodes/` — all contents distributed to services above or kept at `src/` root as shared infrastructure.

---

## Phase 0: Directory Restructure ✅ COMPLETE

**Goal:** Move all service-specific code to its service directory. Shared infrastructure stays at `src/` root. Empty `src/core/` and `src/nodes/` directories removed.

### Completed

| # | Action | File(s) | Status |
|---|--------|---------|--------|
| 0.1 | Move config-related files to shared root | `src/core/config.py` → `src/config.py` | ✅ |
| 0.2 | Move utils to shared root | `src/core/utils.py` → `src/utils.py` | ✅ |
| 0.3 | Move async_bridge to shared root | `src/core/async_bridge.py` → `src/async_bridge.py` | ✅ |
| 0.4 | Move guardrails to agent_orchestrator | `src/core/guardrails.py` → `src/services/agent_orchestrator/guardrails.py` | ✅ |
| 0.5 | Move memory to agent_orchestrator | `src/core/memory.py` → `src/services/agent_orchestrator/memory.py` | ✅ |
| 0.6 | Move agent_factory to agent_orchestrator | `src/core/agent_factory.py` → `src/services/agent_orchestrator/agent_factory.py` | ✅ |
| 0.7 | Move subagents to agent_orchestrator | `src/core/subagents.py` → `src/services/agent_orchestrator/subagents.py` | ✅ |
| 0.8 | Move routing to agent_orchestrator | `src/core/routing.py` → `src/services/agent_orchestrator/routing.py` | ✅ |
| 0.9 | Move plan node to agent_orchestrator | `src/nodes/plan.py` → `src/services/agent_orchestrator/plan.py` | ✅ |
| 0.10 | Move review node to agent_orchestrator | `src/nodes/review.py` → `src/services/agent_orchestrator/review.py` | ✅ |
| 0.11 | Move mcp_client to tools_integration | `src/core/mcp_client.py` → `src/services/tools_integration/mcp_client.py` | ✅ |
| 0.12 | Move a2a_client to tools_integration | `src/core/a2a_client.py` → `src/services/tools_integration/a2a_client.py` | ✅ |
| 0.13 | Move tools to tools_integration | `src/core/tools.py` → `src/services/tools_integration/tools.py` | ✅ |
| 0.14 | Move research_fetch to tools_integration | `src/core/research_fetch.py` → `src/services/tools_integration/research_fetch.py` | ✅ |
| 0.15 | Move rag to tools_integration | `src/core/rag.py` → `src/services/tools_integration/rag.py` | ✅ |
| 0.16 | Move summarization (unused — kept) | `src/core/summarization.py` → kept at `src/core/` | ✅ |
| 0.17 | Update all import paths across 26+ files | src/, tests/, agent.py, app.py | ✅ |
| 0.18 | Remove empty `src/core/` and `src/nodes/` | Directory deletion | ✅ |

### Verification (Phase 0)

```bash
uv run pytest -q 2>&1 | tail -3
# All 245 tests passing
```

---

## Phase 1: Session Memory Service — PARTIALLY COMPLETE

**Goal:** Implement the Session Memory service fully. It has no service dependencies (only shared `src.config` and `src.utils`). Update agent_orchestrator to use real Session Memory.

### Completed

| # | Action | File(s) | Status |
|---|--------|---------|--------|
| 1.1 | Move/checkpoint logic | `src/services/session_memory/checkpoint.py` | ✅ `MemorySaver` wrapper per `thread_id`, `get_session(thread_id) → List[BaseMessage]`, `append_message(thread_id, message)`, `get_turn_token_usage(thread_id)`. |
| 1.2 | Move compression logic | `src/services/session_memory/compression.py` | ✅ `compress_messages(messages, max_history=20)` — from old `src/core/summarization.py`. Tool-call-safe boundary logic preserved. |
| 1.3 | Create window module | `src/services/session_memory/window.py` | ✅ `get_window(thread_id, max_messages=20)`, `append_message(thread_id, message)`. |
| 1.4 | Create `__init__.py` exports | `src/services/session_memory/__init__.py` | ✅ Export `get_window`, `append_message`, `compress_messages`. |
| 1.6 | Add `SubAgent` and `ToolSpec` dataclasses | `src/types.py` | ✅ `SubAgent(name, description, department, system_prompt, tool_registry, protocol)`, `ToolSpec(name, description, risk_level, requires_approval, allowed_roles, handler, timeout_seconds)`. |
| 1.7 | Update orchestrator to use real session_memory | `src/services/agent_orchestrator/plan.py` | ✅ Uses `compress_messages` from session_memory. |

### Not Yet Done (blocked by tests)

| # | Action | File(s) | Note |
|---|--------|---------|------|
| 1.5 | Simplify `src/services/agent_orchestrator/state.py` — remove `review_verdict`, `recursion_depth`, `pending_writes`, `audit_log` | `src/services/agent_orchestrator/state.py` | Blocked: `test_routing.py`, `test_delegation.py`, `test_subagents.py` still reference these fields. |
| 1.8 | Reorganize tests to `tests/services/session_memory/` | Tests | ✅ Removed duplicate `tests/test_summarization.py` (identical to `tests/services/session_memory/test_compression.py`). Compression and window tests now live in `tests/services/session_memory/`.

### Verification (Phase 1)

```bash
uv run pytest tests/services/session_memory/ -v
# 36 tests passing
uv run pytest -q 2>&1 | tail -3
# 334 total, all pass — no regressions
```

---

## Phase 2: Tools Integration Service ✅ COMPLETE

**Goal:** Implement the Tools Integration service fully. It has no service dependencies (only shared `src.config`, `src.async_bridge`, `src.utils`). Update agent_orchestrator to use real Tools Integration.

### Completed

| # | Action | File(s) | Status |
|---|--------|---------|--------|
| 2.1 | Create registry with risk metadata | `src/services/tools_integration/registry.py` | ✅ `ToolRegistry`: `register`, `get_spec`, `get_callable`, `register_builtin`, `get_tools_for_role`, `get_visible_tools` (≤20 cap), `get_tool_definitions`, `execute`, `list_tools`. `RequiresApprovalError` exception. |
| 2.2 | Create tool spec decorator | `src/services/tools_integration/decorator.py` | ✅ `@tool_spec(name, description, risk_level, requires_approval, allowed_roles)`. `ToolSpecMetadata` dataclass. |
| 2.3 | Create discovery service | `src/services/tools_integration/discovery.py` | ✅ `discover_subagents()` and `discover_tools()` with tree-hash caching. |
| 2.4 | Create relevance sorter | `src/services/tools_integration/relevance.py` | ✅ `sort_tools(enhanced_query, tool_defs, max_tools=5)`. Strips `args_schema` from LLM input. |
| 2.5 | Create async executor | `src/services/tools_integration/executor.py` | ✅ `ToolExecutor(registry)`: `execute` (async), `execute_sync` (sync wrapper), `register`. Risk-tiered via registry. |
| 2.6 | Create MCP bridge (async-native) | `src/services/tools_integration/mcp_bridge.py` | ✅ `load_mcp_tools()`, `clear_mcp_tools_cache()`. |
| 2.7 | Create A2A bridge (async-native) | `src/services/tools_integration/a2a_bridge.py` | ✅ `call_a2a_agent(url, description, trace_id)`. |
| 2.8 | Update tool files with `@tool_spec` | `tools/*.py` | ✅ All tools decorated. `_task_impl` has dynamic docstring from SUBAGENTS registry. |
| 2.9 | Update agent_orchestrator to use real tools_integration | `src/services/agent_orchestrator/agent_factory.py` | ✅ `local_orchestrator_node` uses `get_all_tools()` for tool objects; `local_tools_node` uses `create_tool_registry()` + `ToolExecutor.execute_sync()` for non-task tools; `_execute_task` still receives `get_all_tools()` for `run_tool_loop`. |
| 2.10 | Tests | All 6 test files | ✅ All 119 tools integration tests passing. |

### Verification (Phase 2)

```bash
uv run pytest tests/services/tools_integration/ -v
uv run pytest -q 2>&1 | tail -3
# No regressions
```

---

## Phase 3: Agent Orchestrator Service (Full Implementation) ✅ COMPLETE

All Phase 3 items done:
- ✅ `graph.py` — START → orchestrator → {subagent_fanout \| responder} → END; `get_deep_agent()` / `reset_deep_agent()`
- ✅ `orchestrator.py` — enhance query, identify departments, return `{enhanced_query, department_targets}`; uses session_memory for message window
- ✅ `routing.py` — Phase 3 `route_from_orchestrator`: departments → subagent_fanout, else responder; legacy functions preserved
- ✅ `subagent_engine.py` — `SubAgentEngine.invoke_parallel()` with asyncio.gather, shared deadline, ReAct loops
- ✅ `aggregator.py` — single dept returns directly, multi-dept synthesizes
- ✅ `verification.py` — deterministic checks (non-empty, workspace containment, keyword overlap)
- ✅ `__init__.py` — exports all Phase 3 functions
- ✅ `agent.py` wired to import from `src.services.agent_orchestrator.graph`

### Verification (Phase 3)

```bash
uv run pytest tests/services/agent_orchestrator/ -v
uv run pytest -q 2>&1 | tail -3
# 375 tests passing, no regressions
```

---

## Phase 4: Entry Points & Cross-Cutting Concerns

**Goal:** Update entry points, implement auth stubs, add observability hooks, update UI.

### Changes

| # | Action | File(s) | Effort | Details |
|---|--------|---------|--------|---------|
| 4.1 | Update app.py imports | `app.py` | Medium | Replace `src.core.guardrails`, `src.core.memory` imports with service equivalents. Update stream handler: show "Querying {department}..." for fanout, "Synthesizing..." for aggregation. New session state keys: `routing_decisions`, `active_subagents`, `enhanced_query`. Remove "Shared Memory" sidebar (no AGENTS.md). |
| 4.2 | Update main.py | `main.py` | Low | Import `get_deep_agent` from new location (same `agent.py` facade). |
| 4.3 | Auth interface (deferred) | `src/services/tools_integration/auth_interface.py` | Low | Define `AuthInterface` class with `authenticate(subagent_name, tool_name)` method. Executor calls `auth_interface.authenticate()` but implementation is a pass-through until Phase 5. |
| 4.4 | Observability hooks | `src/services/agent_orchestrator/subagent_engine.py`, `src/services/tools_integration/executor.py` | Medium | Add trace logging: sub-agent start/end, tool call name/latency/result, error tracking. Reuse `DeepAgentTracer` pattern from current `agent_factory.py`. |
| 4.5 | Risk-tiered fast path | `src/services/tools_integration/executor.py` | Low | `execute()`: if `spec.risk_level == "low"` → direct call. If `"medium"` → validate + execute. If `"high"` → raise `RequiresApprovalError`. |
| 4.6 | Input sanitization at boundary | `src/services/agent_orchestrator/orchestrator.py` | Low | Strip obvious injection patterns (`ignore previous`, `system prompt`, etc.) from tool results before entering orchestrator context. |
| 4.7 | Update CLAUDE.md | `CLAUDE.md` | Medium | Reflect new paths, remove old paths, update test count. |
| 4.8 | Update ARCHITECTURE_EXPLANATION.md | `ARCHITECTURE_EXPLANATION.md` | Medium | Rewrite for new architecture. |
| 4.9 | Update PHASES.md | `PHASES.md` | Low | Add Phase 0–4 entries. |

### Verification (Phase 4)

```bash
uv run pytest -q 2>&1 | tail -3
uv run python main.py  # smoke test CLI
```

---

## Phase 5: Cleanup & Finalization

**Goal:** Remove all transitional code, finalize docs, ensure clean architecture.

### Changes

| # | Action | File(s) | Effort |
|---|--------|---------|--------|
| 5.1 | Simplify `src/services/agent_orchestrator/state.py` — remove `review_verdict`, `recursion_depth`, `pending_writes`, `audit_log` | `src/services/agent_orchestrator/state.py` | Low |
| 5.2 | Remove transitional stubs | All Phase 0 stubs | Low |
| 5.3 | Remove old `src/core/__pycache__` | — | Trivial |
| 5.4 | Fix any broken imports | Various | Medium |
| 5.5 | Final test run | — | Verify 209+ tests pass |
| 5.6 | Update test count in CLAUDE.md | `CLAUDE.md` | Trivial |

### Verification (Phase 5)

```bash
uv run pytest -q 2>&1 | tail -3
# All tests pass, no references to src.core or src.nodes remain:
grep -rn "src.core\|src.nodes" src/ tests/ app.py main.py agent.py 2>/dev/null
# Should return empty (except maybe comments)
```

---

## Progress

**Phase 0: Directory Restructure — COMPLETE** ✅

All Phase 0 items done:
- ✅ All service-specific code moved to `src/services/{agent_orchestrator,tools_integration,session_memory}/`
- ✅ Shared infrastructure (`state.py`, `config.py`, `async_bridge.py`, `utils.py`) remains at `src/` root
- ✅ `src/core/` and `src/nodes/` directories removed (no stale code)
- ✅ All imports updated across 26+ files
- ✅ Test patch targets and logger names updated
- ✅ All 245 tests passing after restructure

**Phase 1: Session Memory — PARTIALLY COMPLETE** ✅

Completed:
- ✅ `checkpoint.py` — MemorySaver wrapper, get/append session
- ✅ `compression.py` — compress_messages from old summarization
- ✅ `window.py` — get_window, append_message
- ✅ `src/types.py` — SubAgent and ToolSpec dataclasses
- ✅ Orchestrator uses real session_memory (`plan.py`)

Not Yet Done:
- ⬜ Update `src/services/agent_orchestrator/state.py` — simplified schema (blocked by tests referencing old fields)
- ⬜ Reorganize tests to `tests/services/session_memory/`

**Phase 2: Tools Integration — COMPLETE** ✅

All Phase 2 items done:
- ✅ `ToolRegistry` with risk metadata, role visibility, schema validation, tiered execution
- ✅ `@tool_spec` decorator for tool metadata
- ✅ Auto-discovery (skills/*/SKILL.md + tools/*.py @tool_spec) with tree-hash caching
- ✅ Relevance sorter (20 → 3-5 via LLM, strips args_schema)
- ✅ `ToolExecutor` (async, risk-tiered, via registry)
- ✅ MCP bridge, A2A bridge
- ✅ All tools decorated with `@tool_spec`
- ✅ Orchestrator updated to use real registry/executor
- ✅ All 119 tools integration tests passing; 364 total

---

## Implementation Checklist

### Phase 0: Directory Restructure
- [x] All service-specific code moved to service directories
- [x] Shared infrastructure (`state.py`, `config.py`, `async_bridge.py`, `utils.py`) at `src/` root
- [x] `src/core/` and `src/nodes/` removed
- [x] All imports updated across all files
- [x] Verify all existing tests still collected and passing (245 tests)

### Phase 1: Session Memory
- [x] Implement `checkpoint.py` — MemorySaver wrapper, get/append session
- [x] Implement `compression.py` — compress_messages from old summarization
- [x] Implement `window.py` — get_window, append_message
- [x] Create `src/types.py` — SubAgent and ToolSpec dataclasses
- [ ] Update `src/services/agent_orchestrator/state.py` — simplified schema (blocked by tests referencing old fields)
- [x] Update orchestrator to use real session_memory (`plan.py`)
- [x] Reorganize tests to `tests/services/session_memory/` (removed duplicate root-level `tests/test_summarization.py`)

### Phase 2: Tools Integration
- [x] Implement `registry.py` — ToolRegistry with risk, roles, approval, validation
- [x] Implement `decorator.py` — @tool_spec for all tools/
- [x] Implement `discovery.py` — auto-discover skills/ + tagged tools/
- [x] Implement `relevance.py` — sort 20 → 3-5 via LLM
- [x] Implement `executor.py` — async, risk-tiered, auth hooks
- [x] Implement `mcp_bridge.py` — async MCP from old mcp_client
- [x] Implement `a2a_bridge.py` — async A2A from old a2a_client
- [x] Decorate all tools in tools/ with @tool_spec
- [x] Update orchestrator stubs to use real tools_integration
- [x] Write tests for all modules (119 tests passing)

**Phase 2 verification:**
```bash
uv run pytest tests/services/tools_integration/ -v  # 119 passed
uv run pytest -q 2>&1 | tail -3                      # 364 total, all pass
```

### Phase 3: Agent Orchestrator
- [ ] Implement `graph.py` — new graph: START → orchestrator → {fanout\|responder} → END
- [ ] Implement `orchestrator.py` — enhance, route, verify, synthesize
- [ ] Implement `routing.py` — fanout vs responder
- [ ] Implement `subagent_engine.py` — parallel ReAct with asyncio.gather
- [ ] Implement `aggregator.py` — combine multi-dept results
- [ ] Implement `verification.py` — deterministic checks on sub-agent outputs
- [ ] Wire into agent.py
- [ ] Write tests for all modules

### Phase 3: Agent Orchestrator — COMPLETE ✅

All Phase 3 items done (see ✅ COMPLETE section above).

### Phase 4: Entry Points & Cross-Cutting

### Phase 5: Cleanup
- [ ] Update app.py — new imports, fanout UI, remove AGENTS.md sidebar
- [ ] Update main.py — import from new agent.py
- [ ] Auth interface stub
- [ ] Observability hooks
- [ ] Input sanitization
- [ ] Update CLAUDE.md, ARCHITECTURE_EXPLANATION.md, PHASES.md

### Phase 5: Cleanup
- [ ] Simplify `src/services/agent_orchestrator/state.py` (remove old fields)
- [ ] Remove stubs
- [ ] Fix any broken imports
- [ ] Final full test run

---

## Key Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| AGENTS.md | Removed entirely | No uncontrolled data stores; SKILL.md is the only memory file |
| Sub-agent discovery | Auto-discover via `skills/*/SKILL.md` + `tools/*.py` `@tool_spec` tags | Zero-config onboarding: add folder + tagged tools = live |
| Tool visibility | Each sub-agent sees ≤20 tools (from SKILL.md), LLM sorts 3-5 relevant | Controlled context, focused ReAct loops |
| Auth | Per-department, interface defined now, implemented later | Risk-tiered execution gates honor auth when ready |
| Multi-dept dispatch | `asyncio.gather` for true parallelism | All sub-agents run concurrently, results combined |
| Graph reviewers | Removed (critic/plan_checker/reflection) | Simpler flow; verification step replaces adversarial review |
| Orchestrator synthesis | Always synthesize unless single department | Quality > speed for multi-dept; single-dept returns directly |
| Risk tiers | low (fast path) / medium (validate+execute) / high (gate) | Matches article's risk-tiered controls |
| Test count baseline | 209 passing (pre-restructure); simplified state in Phase 5 when old code is removed |
