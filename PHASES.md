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

## Phase 2: High Priority — State Integrity
- ⬜ **2.1** Stop polluting message history — orchestrator returns only `next_message`, not appending to `messages`
- ⬜ **2.2** Wire up `max_iterations` enforcement in orchestrator (reject with error message)
- ⬜ **2.3** Wire up `token_usage` tracking (populate from LLM response metadata)
- ⬜ **2.4** Merge identical `agent` and `responder` nodes into distinct behaviors:
  - `agent` = staging pass-through (moves `next_message` → `messages`)
  - `responder` = final answer formatter (extracts and formats the response)
- ⬜ **2.5** Fix `internet_search` return format — return structured markdown, not Python `repr()`

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
- ⬜ **4.1** Add reflection/self-correction node — agent reconsiders approach before retrying
- ⬜ **4.2** Add parallel execution for independent sub-tasks (multiple research queries)
- ⬜ **4.3** Replace `PENDING_APPROVAL` string hack with LangGraph native `interrupt()` / HITL
- ⬜ **4.4** Add observability hooks — LangGraph tracing callbacks

**Files affected:** `src/nodes/review.py`, `src/core/agent_factory.py`, `app.py`

---

## Phase 5: Production Readiness
- ⬜ **5.1** Add pytest test framework + unit tests for routing logic, guardrails, state transitions
- ⬜ **5.2** Add linting/formatting (ruff, mypy) to `pyproject.toml`
- ⬜ **5.3** Make `streamlit` an optional dependency (`[project.optional-dependencies]`)
- ⬜ **5.4** Add conversation summarization when `max_history_messages` is reached

**Files affected:** `pyproject.toml`, new `tests/` directory

---

## Implementation Notes

### Routing Flow (After Fixes)

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

### Key Design Decisions

1. **Recursion depth**: Hard limit of 3 levels. `recursion_depth` incremented on each `task` call.
2. **Message history**: Orchestrator writes to `next_message` only. `agent` node moves to `messages`. `responder` extracts final answer.
3. **Subagent types**: Research/writer use structured output (no tools). General-purpose uses full graph (with recursion limit).
4. **Critic/plan_checker**: Route back to orchestrator on rejection, not to agent→tools (which can't handle critique text).

---

## Progress Log

| Date | Phase | What |
|------|-------|------|
| 2026-07-29 | Phase 1 | Recursion depth enforcement, graph caching, critic/plan_checker routing fix, responder fallback |
| 2026-08-14 | Phase 3 | Extract routing/tools/utils modules, importlib dynamic loading, skills caching, guardrails consistency, LangGraph checkpointing |
