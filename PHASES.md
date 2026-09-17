# Agentic Implementation — Phase Tracker

## Status Legend
- ⬜ Not started
- 🔄 In progress
- ✅ Complete
- ⛔ Blocked

---

## Phase 1: Critical Fixes — Safety & Correctness ✅

- ✅ **1.1** Enforce recursion depth in `task` tool — `local_tools_node` checks `state["recursion_depth"]` before executing task; rejects at depth ≥ 3 with descriptive error
- ✅ **1.2** Graph caching — `_compiled_graph` module variable; `get_deep_agent()` returns cached instance; subagents share one compiled graph
- ✅ **1.3** Critic routing fixed — on rejection, routes to `orchestrator` (not `agent`→`tools`); on approval, passes original message through to `responder`
- ✅ **1.4** Plan checker routing fixed — on non-compliant, routes to `orchestrator`; on compliant/no-plan, routes to `critic` for quality review
- ✅ **1.5** Responder fallback — when `next_message` is None (plan_checker compliant path), responder picks up `messages[-1]` as the final answer

**Files changed:** `src/core/agent_factory.py`, `src/nodes/review.py`

### Corrected Graph Flow (After Phase 1)

```
START → orchestrator → {agent | critic | plan_checker | responder}
  orchestrator has tool_calls? → agent → tools → orchestrator
  orchestrator no tool_calls + plan exists? → plan_checker
    plan_checker compliant? → critic
    plan_checker non-compliant? → orchestrator (reformulate)
  orchestrator no tool_calls, no plan? → critic
    critic APPROVED? → responder → END
    critic rejected? → orchestrator (reformulate)
```

---

## Phase 2: High Priority — State Integrity ✅
- ✅ **2.1** Stop polluting message history — orchestrator returns only `next_message`, not appending to `messages`
- ✅ **2.2** Wire up `max_iterations` enforcement in orchestrator (reject with error message)
- ✅ **2.3** Wire up `token_usage` tracking (populate from LLM response metadata)
- ✅ **2.4** Merge identical `agent` and `responder` nodes into distinct behaviors:
  - `agent` = staging pass-through (moves `next_message` → `messages`)
  - `responder` = final answer formatter (extracts and formats the response)
- ✅ **2.5** Fix `internet_search` return format — return structured markdown, not Python `repr()`

**Files affected:** `src/nodes/plan.py`, `src/nodes/review.py`, `src/core/agent_factory.py`, `src/core/rag.py`

---

## Phase 3: Medium — Maintainability & Architecture ✅
- ✅ **3.1** Extract routing functions → `src/core/routing.py`
- ✅ **3.2** Extract tool definitions → `src/core/tools.py`
- ✅ **3.3** Extract retry wrapper → `src/core/utils.py`
- ✅ **3.4** Fix dynamic tool loading — use `importlib.util.spec_from_file_location` instead of modifying `sys.path`
- ✅ **3.5** Add caching for skills/tools summaries (memoize with filesystem change invalidation)
- ✅ **3.6** Fix guardrails inconsistency — consistent fallback behavior between read and write paths
- ✅ **3.7** Add LangGraph checkpointing (`MemorySaver`) for state persistence

**Files changed:** `src/core/agent_factory.py`, `src/core/routing.py`, new `src/core/tools.py`, `src/core/utils.py`, `src/core/memory.py`, `src/core/guardrails.py`, `src/nodes/plan.py`

---

## Phase 4: Missing Patterns — Capability Upgrade
- ✅ **4.1** Add reflection/self-correction node — agent reconsiders approach before retrying
- ✅ **4.2** Add parallel execution for independent sub-tasks (multiple research queries)
- ❌ **4.3** ~~Replace `PENDING_APPROVAL` string hack with LangGraph native `interrupt()` / HITL~~ — removed (incompatible with Streamlit; see Critical Fixes)
- ✅ **4.4** Add observability hooks — LangGraph tracing callbacks

**Files affected:** `src/nodes/review.py`, `src/core/agent_factory.py`, `src/core/routing.py`

> ⚠️ **Audit correction (2026-08-25):** the 4.4 implementation is a stub — `DeepAgentTracer`'s methods are never registered as callbacks and OBSERVABILITY=1 emits zero events. See Gap Audit item C3 and Phase 8.3. ✅ Resolved in 8.3 (2026-09-12): `DeepAgentTracer` is a real `BaseCallbackHandler` attached via `get_model()`, recording actual events (fake-model test in `tests/test_observability.py`).

---

## Phase 5: Production Readiness
- ✅ **5.1** Add pytest test framework + unit tests for routing logic, guardrails, state transitions
- ✅ **5.2** Add linting/formatting (ruff, mypy) to `pyproject.toml`
- ✅ **5.3** Make `streamlit` an optional dependency (`[project.optional-dependencies]`)
- ✅ **5.4** Add conversation summarization when `max_history_messages` is reached

**Files affected:** `pyproject.toml`, new `tests/` directory, `src/core/summarization.py`, `src/nodes/plan.py`

---

## Critical Fixes — Post-Phase Audit

Three critical issues were identified after Phase 4 implementation:

### 🔴 HITL (interrupt) incompatible with Streamlit

`_file_approval_node` called `interrupt()` which raises `GraphInterrupt`. Streamlit is a request-response web framework — it renders once and exits. There's no mechanism to "pause" a Streamlit session waiting for a button click, then resume the same graph execution. **Resolution**: Removed HITL entirely. Writes execute immediately; `pending_writes` tracks operations for audit/logging only.

### 🔴 Writes executed before approval (security bypass)

Even if HITL worked, writes were invoked via `invoke_with_retry(tool_func, tool_args)` *before* being queued into `pending_writes`. The file already existed on disk by the time the approval prompt appeared — the "approval" could not prevent the write. **Resolution**: Writes now execute immediately; no approval gate exists. If HITL is needed, it requires a fundamentally different architecture (checkpoint/save state → render approval UI → resume with `agent.invoke(config, input=...)`).

### 🔴 Global cache never invalidates

`_compiled_graph` was set once and reused forever. Any change to tools, skills, system prompts, or model configuration was invisible until the process restarted. **Resolution**: Added `reset_deep_agent()` function that clears the cache; call when configuration changes.

### Additional: Reflection didn't clear review_verdict

When critic rejected and routed to reflection, `review_verdict` remained `"rejected"`. When routing back to orchestrator, the stale verdict caused ambiguous routing semantics. **Resolution**: Reflection node now clears `review_verdict` to `None`.

---

## Gap Audit — 2026-08-25

Goal: make this a *proper* multi-agent system — first-class tools and subagent delegation. Findings below were verified against the code on 2026-08-25; each maps to Phases 6–9.

### A. Subagent system is not first-class
1. **Subagent definitions scattered in 4 places** — type list hardcoded in the `task` tool signature (`src/core/tools.py:96`), parallel-group tuple in `local_tools_node` (`src/core/agent_factory.py:276`), if/elif dispatch + aliases in `_execute_task` (`src/core/tools.py:124`), and a dead `role=` branch in `get_system_prompt` (`src/core/memory.py:143`). Adding one agent type means editing 3–4 files; there is no registry.
2. **`task` tool body is a live bypass** — the tool function itself calls `_execute_task(subagent_type, description, 0)` at depth 0 (`src/core/tools.py:106`). It's only safe because `local_tools_node` intercepts task calls; any direct invoke path would skip the recursion guard.
3. **Depth limit is a magic number** — `3` hardcoded in `local_tools_node` while every other budget lives in `config.py`. Actual semantics: child starts at parent+1, so top-level + 3 delegations succeed (depths 0–3); docs say "3 levels", which reads differently.
4. **Output-file collision risk** — both researcher and writer prompts suggest the *same* filenames (`workspace/research_summary.md`, `workspace/blog_post.md`). Parallel research tasks can silently clobber each other; the orchestrator is never told to assign unique output paths.
5. **Parallel batch timeout compounds** — each `future.result(timeout=600)` in the collection loop gets its own full timeout, so a hung batch of N tasks can block ~N×timeout instead of one shared deadline (`src/core/agent_factory.py:288`).

### B. Tool system
6. **Dynamic tools re-executed every turn** — `get_all_tools()` → `load_dynamic_tools()` runs `os.listdir` + `exec_module` on every `./tools/*.py`, and it's called per orchestrator turn, per tools-node pass, and per `list_tools`. The skills *summary* is mtime-cached (3.5) but the tool *loading* is not.
7. **No validation of dynamic tools** — a module defining no `@tool` functions loads silently; same-named tools in two files collide with last-wins and no warning; module names are file basenames, so subdirectories collide.
8. **Reads are unsandboxed (security)** — `read_file` uses `must_be_in_workspace=False`, so the agent can read any repo path: `.env` (API keys), source, git metadata. Writes are sandboxed to `./workspace`; reads are not. Asymmetric guardrails.
9. **No workspace navigation** — no list/search tools; subagents discover files only by guessing paths and eating `read_file` errors.
10. **Research depth capped at snippets** — `internet_search` returns 300-char snippets and there is no page-fetch tool, so the research subagent can't read beyond search results. (Optional fix; needs a dependency.)

### C. Correctness & consistency
11. **Orchestrator prompt lies about tools** — `plan.py:31` calls `get_system_prompt()` with no tools dict, so every turn injects `Available Tools: No tools available.` while the same turn binds the real tools via `bind_tools`. The `role=` branch of `get_system_prompt` has no callers.
12. **Reviewers bypass the shared retry wrapper** — critic/plan_checker/reflection call raw `model.invoke` (`src/nodes/review.py:72,127,161`); a transient 429 during review crashes the run while orchestrator and tools paths back off and retry.
13. **Observability is a stub** — `DeepAgentTracer`'s methods are never registered as LangChain callbacks; `_apply_observability_hooks` only sets `graph._tracer`; no `invoke`/`stream` call passes callbacks. OBSERVABILITY=1 produces zero events. Phase 4.4's ✅ overstates reality.
14. **Dead state fields / dead code** — `AgentState.max_tokens` and `requires_approval` are never read by production code (only test fixtures); `route_after_tools` is imported and tested but the graph hardcodes `tools → orchestrator`.
15. **Model client rebuilt on every call** — `get_model()` constructs a fresh provider client for each orchestrator turn, each reviewer visit, and each subagent loop, instead of one cached instance per configuration.

### D. Test coverage
16. **No graph-level delegation tests** — existing subagent tests are unit-level (`run_tool_loop` with fake models). Nothing exercises task → tools-node → result aggregation (token usage into parent, `pending_writes` merge, depth rejection, parallel ordering) through the compiled graph.

### Accepted / not gaps
- Subagent tool-loop threads can't be killed on timeout — `future.result(timeout)` discards the result but the worker thread keeps running until its own iteration budget ends. Bounded by `max_iterations` in the loop; acceptable at current scale.
- Writes execute immediately with audit-only tracking — deliberate decision (Critical Fixes above), not a gap.

---

## Phase 6: First-Class Subagent System ✅

Make subagents data, not code paths: one registry drives the `task` tool schema, dispatch, parallelism, and prompts.

- ✅ **6.1** Subagent registry — `SUBAGENTS` dict + `SubagentSpec` dataclass in `src/core/subagents.py` (key: type; fields: aliases, description for the task tool, `kind`, tool list, `parallelizable`). Drives from it: the `task` tool enum + docstring (`_SubagentType = Literal[(*SUBAGENTS.keys(),)]`, docstring built per registry entry), `_execute_task` dispatch (`resolve_subagent`), and the parallel-vs-sequential split in `local_tools_node` (`is_parallelizable`). All four duplicated string lists removed (Audit A1).
- ✅ **6.2** Single source for role prompts — research/writer prompts are `COMPLETION_CONTRACT` + their `skills/<role>/SKILL.md` body (+ AGENTS.md), built by `build_role_prompt(spec)`; role-specific completion details moved into the SKILL.md files. `nodes/research.py` / `nodes/write.py` deleted; dead `role=` branch of `get_system_prompt` removed (Audit A1, B-branch).
- ✅ **6.3** Configurable depth limit — `AGENT_MAX_SUBAGENT_DEPTH` (default 3) via `config.get_max_subagent_depth()`, replacing the magic number in `local_tools_node`; semantics documented in the config docstring (child runs at parent+1; delegation rejected when parent depth ≥ limit ⇒ top-level nests `limit` deep) (Audit A3).
- ✅ **6.4** Close the `task` bypass — tool body now returns an explicit error ("must be executed by the tools node") instead of `_execute_task(..., 0)`; no invoke path can skip the recursion guard (Audit A2).
- ✅ **6.5** Delegation file contract — orchestrator prompt: every `task` description must name a unique output path under `./workspace`; the shared completion contract tells subagents to use the named path or pick a unique filename, removing the shared-default-filename clobber risk (Audit A4).
- ✅ **6.6** Shared batch deadline — one `time.monotonic()` deadline per task batch in `local_tools_node`; each `future.result()` gets only the remaining time, so a hung batch costs one timeout, not N× (Audit A5).

**Files changed:** `src/core/subagents.py` (registry + prompt builder), `src/core/tools.py`, `src/core/agent_factory.py`, `src/core/config.py`, `src/core/memory.py`; deleted `src/nodes/research.py`, `src/nodes/write.py`; role specifics added to `skills/research/SKILL.md`, `skills/writer/SKILL.md`

**Success criteria:** adding a fourth subagent type = editing one registry entry (verified by inspection); `pytest` green; new unit test asserting the registry drives tool schema, dispatch, and parallel grouping.

---

## Phase 7: Tool System Hardening & Workspace Navigation ✅

- ✅ **7.1** Cache dynamic tool loading — same mtime-hash invalidation pattern as the skills cache; `get_all_tools()` must not re-execute `./tools/*.py` when nothing changed (Audit B6).
- ✅ **7.2** Validate dynamic tools — warn on modules that define no `@tool` functions; built-ins win name collisions with a logged warning; unique module names so subdirectories can't collide (Audit B7).
- ✅ **7.3** Sandbox reads (default-deny) — restrict `read_file` to an allowlist: `./workspace`, `./skills`, `./AGENTS.md`; deny everything else with a message stating what's allowed; symmetric with the write sandbox. Tests in `test_guardrails.py` (Audit B8).
- ✅ **7.4** Navigation tools — add `list_files` (workspace tree) and `search_files` (substring match across workspace files); wire both into research/writer toolsets via the registry so subagents stop guessing paths (Audit B9).
- ✅ **7.5** *(optional)* `fetch_url` — full page fetch for the research subagent so research goes beyond 300-char snippets (httpx; response size cap + timeout; add dependency) (Audit B10).

**Files affected:** `src/core/tools.py`, `src/core/guardrails.py`, `src/core/subagents.py` (registry toolsets), `tests/test_guardrails.py`, `pyproject.toml` (7.5 only)

**Success criteria:** tools reload only when `./tools` changes (cache test); `read_file("./.env")` denied with a clear error; research/writer agents can list and locate workspace files via tools, not guessing.

---

## Phase 8: Correctness & Consistency ✅

- ✅ **8.1** Truthful orchestrator prompt — pass the real tools dict into `get_system_prompt(tools_dict=...)` so "Available Tools" reflects what's actually bound; remove or wire the dead `role=` branch. Test: with tools present, the prompt contains no `No tools available.` (Audit C11). (`role=` branch was already removed in 6.2.)
- ✅ **8.2** Shared retry on all LLM calls — route critic / plan_checker / reflection through `invoke_with_retry` like the orchestrator and tools paths do (Audit C12).
- ✅ **8.3** Observability: make it real or remove it — `DeepAgentTracer` is now a real `BaseCallbackHandler`, attached via `_maybe_attach_callbacks` (`model.with_config(callbacks=[...])`) inside `get_model()`, so OBSERVABILITY=1 records real chat/tool/chain events on every model path (orchestrator, reviewers, subagent loops). Verified by fake-model test; stub `_apply_observability_hooks` deleted. (Audit C13)
- ✅ **8.4** Remove dead state/code — drop `AgentState.max_tokens` and `requires_approval` (never read in production) and the unused `route_after_tools` function with its tests (Audit C14).
- ✅ **8.5** Cache model instances — `_get_cached_model()` builds one provider client per (provider, model) configuration, reused by orchestrator/reviewers/subagent loops, cleared by `reset_deep_agent()` (Audit C15).

**Files affected:** `src/nodes/plan.py`, `src/core/memory.py` (no change needed — 8.1 only required the caller), `src/nodes/review.py`, `src/core/agent_factory.py`, `src/state.py`, `src/core/routing.py`, `tests/test_routing.py`, new `tests/test_plan.py`, `tests/test_review.py`, `tests/test_observability.py`; `tests/test_state.py` fixtures updated (8.4)

**Success criteria:** prompt assertion test passes; fake-model test — one transient failure then success in each reviewer path completes instead of crashing; OBSERVABILITY=1 run either records events (asserted by test) or the feature is removed and this doc updated to match.

---

## Phase 9: Multi-Agent Integration Tests ✅

Current tests cover nodes and routing in isolation; the delegation contract itself is untested end-to-end.

- ✅ **9.1** Graph-level delegation test — fake orchestrator emitting `task` tool calls through the compiled graph: assert ToolMessage results returned, child token usage aggregated into parent `token_usage`, child file writes merged into `pending_writes`, and depth rejection at the configured limit (Audit D16).
- ✅ **9.2** Parallel batch behavior — two research tasks in one turn: deterministic ToolMessage ordering, distinct output files on disk, and the shared deadline from 6.6 honored (Audit A4/A5).

**Files affected:** new `tests/test_delegation.py`, fake-model test helpers under `tests/`

**Success criteria:** the delegation contract (result delivery, token aggregation, audit merge, depth cap, parallel ordering) is verified through the graph, not just unit-level.

---

## Phase 10: External Tools via MCP + Remote Subagents via A2A ✅

- ✅ **10.1** MCP tool connectivity — `src/core/mcp_client.py` loads tools from configured MCP servers (`MCP_SERVERS` env var, JSON) into the same registry as built-ins, with config-hash caching and `clear_mcp_tools_cache()` wired into `reset_deep_agent()`. Tools appear in `list_tools` and are callable by the orchestrator and subagent toolsets with no per-server code changes. Precedence: built-ins > MCP > dynamic file tools, with logged warnings on collision.
- ✅ **10.2** A2A remote subagents — `src/core/a2a_client.py` calls remote agents over the A2A protocol; configured via the `A2A_AGENTS` env var (JSON) and registered as `kind="a2a"` entries in `SUBAGENTS`, so they are invocable through the `task` tool and ride the existing parallel batch + shared deadline path.
- ✅ **10.3** Sync bridge — `src/core/async_bridge.py` drives the async-only SDKs from the entirely synchronous graph (persistent event loop on a daemon thread, started lazily).

**Files changed:** new `src/core/async_bridge.py`, `src/core/mcp_client.py`, `src/core/a2a_client.py`; `src/core/config.py` (`_env_json`, `get_mcp_servers`, `get_a2a_agents`), `src/core/subagents.py` (`SubagentSpec.url`, `kind="a2a"`, config-driven registry), `src/core/tools.py` (MCP tier, A2A dispatch), `src/core/agent_factory.py` (reset clears the MCP cache), `pyproject.toml`; new `tests/test_async_bridge.py`, `tests/test_mcp.py`, `tests/test_a2a.py`

**Success criteria:** an MCP server's tools appear in `list_tools` and are callable with no per-server code changes (verified with a fake client); a configured A2A agent appears in the `task` tool's enum and dispatches to the remote agent (verified with a fake A2A client, plus an import-time subprocess test).

### The async problem (why 10.3 exists)

Both SDKs are **async-only**, and the graph is **100% synchronous** (`invoke_with_retry` calls `.invoke()`, `local_tools_node` uses `ThreadPoolExecutor`). MCP tools are built with `coroutine=` and no sync `func`, so `tool.invoke()` raises `NotImplementedError` — settled upstream (issue #29; the sync-bridge PR #601 was closed unmerged). So MCP tools are re-wrapped as sync `StructuredTool`s that drive the original coroutine through the bridge, leaving every existing call site unchanged.

### Known limitations

- MCP `get_tools()` is connection-based, so a **stdio MCP server spawns a subprocess per tool call**. Correct and loop-agnostic, but slow.
- MCP tool names are config-derived, so they are not auto-inherited by subagent toolsets — a `SubagentSpec.tools` tuple must name them explicitly to reach a subagent.
- `A2A_AGENTS` is read at **import** time (it feeds the `task` tool's static type enum; `subagents.py` calls `load_dotenv()` itself, since it is imported before `agent_factory`'s). Changing A2A agents therefore needs a process restart, unlike `MCP_SERVERS`, which is read at call time.
- A remote A2A agent reports no token usage and no local file writes, so it contributes nothing to `token_usage` or `pending_writes`.
- `langchain-mcp-adapters` is upstream-deprecated in favour of `langchain[mcp]` (beta, equally async-only); the `<0.4` pin isolates that future migration to `mcp_client.py`.

---

## Implementation Notes

### Routing Flow (After Critical Fixes)

```
START → orchestrator → {agent | critic | plan_checker | responder}
  orchestrator has tool_calls? → agent → tools → orchestrator
    tools execute immediately; pending_writes tracked for audit only
  orchestrator no tool_calls + plan exists? → plan_checker
    plan_checker compliant? → critic
    plan_checker non-compliant? → orchestrator (reformulate)
  orchestrator no tool_calls, no plan? → critic
    critic APPROVED? → responder → END
    critic rejected? → {orchestrator | reflection}
      critic rejected + iteration >= 4? → reflection → orchestrator
      critic rejected + iteration < 4? → orchestrator (reformulate)
```

### Key Design Decisions

1. **Recursion depth**: Configurable via `AGENT_MAX_SUBAGENT_DEPTH` (default 3). Child runs at parent+1; delegation rejected when parent depth ≥ limit (top-level nests `limit` deep).
2. **Message history**: Orchestrator writes to `next_message` only. `agent` node moves to `messages`. `responder` extracts final answer.
3. **Subagent types**: Research/writer run real tool loops with restricted toolsets (`internet_search`/files for research, file tools for writer) and their role system prompts. General-purpose uses the full cached graph (with recursion limit). All subagent token usage is aggregated into the parent's `token_usage`.
4. **Critic/plan_checker**: Route back to orchestrator on rejection, not to agent→tools (which can't handle critique text).
5. **Reflection**: After 4+ iterations of rejections, route through reflection node to break loops and generate revised strategy.
6. **Write execution**: `write_file`/`edit_file` execute immediately; `pending_writes` tracks operations for audit/logging only. HITL via `interrupt()` was removed — incompatible with Streamlit (web framework can't block/resume graph execution).
7. **Parallel tasks**: Research/writer subagents execute in parallel via ThreadPoolExecutor when multiple independent tasks are issued; general-purpose tasks run sequentially (shared state).
8. **Observability**: `OBSERVABILITY=1` env var enables `DeepAgentTracer` for monitoring LLM calls, tool executions, and state transitions.
9. **Cache invalidation**: `reset_deep_agent()` clears the cached compiled graph; call when tools, skills, or model configuration change. The next `get_deep_agent()` compiles a fresh instance.
10. **Reflection cleanup**: Reflection node clears `review_verdict` to prevent routing state corruption when returning to orchestrator.

> ⚠️ Items 3 and 8 are superseded in detail by Phases 6 and 8 (registry-driven subagents; real or removed observability). Kept here as the historical design record.

---

## Progress Log

| Date | Phase | What |
|------|-------|------|
| 2026-07-29 | Phase 1 | Recursion depth enforcement, graph caching, critic/plan_checker routing fix, responder fallback |
| 2026-08-14 | Phase 3 | Extract routing/tools/utils modules, importlib dynamic loading, skills caching, guardrails consistency, LangGraph checkpointing |
| 2026-08-16 | Phase 2 | next_message staging (no message-history pollution), max_iterations + token_usage enforcement, distinct agent/responder nodes, markdown internet_search, review_verdict-based routing |
| 2026-08-17 | Delegation | Research/writer subagents now run real tool loops (restricted toolsets + role system prompts wired in); recursion depth tracks nesting level instead of delegation count; subagent token usage aggregated into parent; `task` subagent_type constrained to Literal with explicit unknown-type error; robust final-answer extraction from general-purpose subagents |
| 2026-08-18 | Phase 4 | Reflection/self-correction node (breaks loops after 4+ iterations), parallel execution for independent research/writer tasks via ThreadPoolExecutor, replaced PENDING_APPROVAL string hack with LangGraph `interrupt()` HITL + file_approval node, observability hooks via `DeepAgentTracer` (OBSERVABILITY=1 env var) |
| 2026-08-18 | Phase 5 | pytest test framework (routing/guardrails/state/summarization tests), ruff+mypy linting config in pyproject.toml, streamlit moved to optional `[ui]` dependency, conversation summarization replaces silent message truncation when max_history_messages reached |
| 2026-08-19 | Critical Fixes | Removed HITL/interrupt (incompatible with Streamlit — web framework can't block/resume), writes execute immediately with audit-only tracking, added `reset_deep_agent()` for cache invalidation, reflection clears `review_verdict` to prevent routing corruption |
| 2026-08-25 | Gap Audit | Verified gaps vs "proper multi-agent system": scattered subagent definitions (A1), live `task` bypass (A2), unsandboxed `read_file` incl. `.env` (B8), orchestrator prompt claiming "No tools available." (C11), reviewers bypassing retry (C12), stub observability (C13), dead state fields/routing fn (C14); added Phases 6–9 and optional Phase 10 |
| 2026-08-31 | Phase 6 | First-class subagent system: `SUBAGENTS` registry drives task schema/dispatch/parallel grouping; role prompts = completion contract + SKILL.md (duplicated node prompts deleted); `AGENT_MAX_SUBAGENT_DEPTH` config; direct `task` invoke refused; delegation file contract in orchestrator prompt; shared batch deadline for parallel tasks |
| 2026-09-12 | Phases 7–9 | Phase 7 verified complete (dynamic-tool cache + validation, read-sandbox allowlist, list_files/search_files/fetch_url). Phase 8: truthful "Available Tools" in orchestrator prompt (8.1), reviewers routed through `invoke_with_retry` (8.2), `DeepAgentTracer` made a real `BaseCallbackHandler` attached via `get_model()` with the stub hooking deleted (8.3), dead state fields + `route_after_tools` removed (8.4), per-configuration model client cache cleared by `reset_deep_agent()` (8.5). Phase 9: graph-level delegation tests — ToolMessage results, child token-usage aggregation, `pending_writes` merge, depth cap, parallel ordering + shared batch deadline (`tests/test_delegation.py`, fake model in `tests/fake_models.py`). Also fixed 3 pre-existing test failures (strict-mode `..` check, tool-call fixture shape, summarization keep-count clamp) |
| 2026-09-12 | Phase 10 | MCP tool connectivity + A2A remote subagents. Sync bridge (`async_bridge.py`) drives the async-only SDKs from the synchronous graph; MCP tools wrapped as sync `StructuredTool`s and cached by config hash, merged into `get_all_tools()` as a third tier; A2A agents config-driven into the `SUBAGENTS` registry as `kind="a2a"` so they are invocable via `task` and parallelize under the shared batch deadline. 29 new tests (fake MCP/A2A clients, real protobuf chunks, import-time subprocess check). Found and fixed a real bug in the process: `INPUT_REQUIRED`/`AUTH_REQUIRED` are non-terminal but never self-resolve, so polling them would hang the caller |
