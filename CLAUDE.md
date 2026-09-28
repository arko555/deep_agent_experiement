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

## Architecture & Constraints
- **Dependencies:** `session_memory ← agent_orchestrator → tools_integration`.
- **Flow:** START → orchestrator (enhance query, identify departments) → `{tools | fanout | responder}`. The `tools` branch is a ReAct loop back to the orchestrator, bounded by `consecutive_invalid_tools` (3) and the iteration budget.
- **Sub-agents:** `SUBAGENTS` is built from `skills/*/SKILL.md` (`subagents._skill_specs`) plus hardcoded `general-purpose` and configured A2A agents. Frontmatter is parsed with `yaml.safe_load` — folded (`>`) and block (`|`) scalars work, and `allowed-tools` accepts comma- *or* space-separated names. Adding a department is adding a directory.
- **One executor:** both the `task` tool (`tools._execute_task`) and department fanout (`subagent_engine.SubAgentEngine`) call `subagents.run_department`. The engine only batches.
- **Department tools:** SKILL.md `allowed-tools` is the hard allowlist → capped at 20 visible → `relevance.sort_tools` narrows to 3-5 for *this* query → real `BaseTool` objects are bound. Unknown names are dropped with a warning, so a stale SKILL.md cannot yield an uncallable tool.
- **Sub-agent Context:** No history passed — only the enhanced query, the `COMPLETION_CONTRACT` + SKILL.md prompt, and the bound tools. Parallel dispatch via `asyncio.gather`; the executor is blocking so it runs via `asyncio.to_thread`.
- **Departments reach the dispatcher** through `_build_dispatcher_prompt()` (roster from `list_departments()`, built per call) and are validated by `_validate_departments` before fanout, so an invented name cannot reach the executor.
- **Memory Files:** `SKILL.md` only. No `AGENTS.md`.
- **Tools:** Risk-tiered execution (`ToolRegistry`/`ToolExecutor`). Schema validation before every call. Role-scoped visibility.
- **Caches:** Directory tree hash (mtime+size of `skills/` + `tools/`) invalidates discovery cache. `reset_deep_agent()` clears the compiled graph, model clients, MCP tools, and rebuilds `SUBAGENTS`. Discovery defaults are anchored to the repo root, not the process cwd.
- **Import cycle:** `subagents` → `discovery` → `tools_integration/__init__` → `tools` → `subagents`. `tools.py` therefore imports `SUBAGENTS`/`resolve_subagent`/`run_department` lazily; tests patch them on `subagents`, not on `tools`.
- **Model Routing:** `src/config.py` checks budgets at call time. Order: Anthropic → OpenRouter → OpenAI → Google → Ollama. `OBSERVABILITY=1` attaches tracer.
