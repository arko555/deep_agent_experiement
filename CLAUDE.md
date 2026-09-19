# CLAUDE.md

## Commands

```bash
uv sync --extra dev          # runtime deps + pytest/ruff/mypy
uv run pytest                # full suite
uv run pytest tests/services/agent_orchestrator/test_routing.py::test_name   # single test
uv run ruff check .          # lint
uv run mypy src              # type check
uv run python main.py        # CLI REPL
uv sync --extra ui && uv run streamlit run app.py   # Streamlit UI
```

pytest config in `pyproject.toml` (`testpaths=["tests"]`, `pythonpath=["."]`).

## Architecture

Three services with clear dependency direction (`session_memory ← agent_orchestrator → tools_integration`):

| Service | Path | Responsibility |
|---------|------|----------------|
| Session Memory | `src/services/session_memory/` | History, checkpointing, message window, compression |
| Agent Orchestrator | `src/services/agent_orchestrator/` | LangGraph, routing, subagent dispatch, verification, aggregation |
| Tools Integration | `src/services/tools_integration/` | Registry (risk/roles), discovery, relevance, async execution, MCP/A2A bridges |

Shared at `src/` root: `state.py`, `config.py`, `async_bridge.py`, `utils.py`. Entry: `agent.py`, `app.py`, `main.py`.

**Flow:** START → orchestrator (enhance query, identify departments) → `{fanout (parallel ReAct) | responder}` → orchestrator (verify, synthesize) → responder → END

### Sub-agents

- **Discovery**: `skills/*/SKILL.md` + `tools/*.py` `@tool_spec` tags — auto-discovered at startup
- **Visibility**: each sub-agent sees ≤20 tools; LLM sorts 3-5 relevant via `tools_integration.relevance.sort_tools`
- **No history** passed to sub-agents — only enhanced query, system prompt, and tool defs
- **Parallel dispatch** via `asyncio.gather` under shared deadline
- **No AGENTS.md** — SKILL.md is the only memory file

### Tools

Risk-tiered execution in `ToolRegistry`/`ToolExecutor`: low → fast path, medium → validate+execute, high → approval gate. Role-scoped visibility per sub-agent. Schema validation before every call. Auth per department — interface defined, implementation deferred.

### Caches

- Compiled graph — `reset_deep_agent()` clears it
- Discovery — invalidated by directory tree hash (mtime+size of `skills/` + `tools/`)

### Model selection

`src/config.py` reads budgets at call time. `get_model()` picks provider: Anthropic → OpenRouter → OpenAI → Google → Ollama. `OBSERVABILITY=1` attaches tracer.

### Tests

`tests/` mirrors services: `tests/services/agent_orchestrator/`, `tests/services/session_memory/`, `tests/services/tools_integration/`, plus shared. `tests/fake_models.py::ScriptedChatModel` provides deterministic fake.

---

## Working Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
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
- If you notice unrelated dead code, mention it — don't delete it.

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
