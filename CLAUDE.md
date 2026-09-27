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
- **Flow:** START → orchestrator (enhance query, identify departments) → `{fanout (parallel ReAct) | responder}` → orchestrator (verify, synthesize) → responder → END.
- **Sub-agents:** Auto-discovered via `skills/*/SKILL.md` + `tools/*.py` `@tool_spec`.
- **Sub-agent Limits:** Sees ≤20 tools; LLM sorts 3-5 via `tools_integration.relevance.sort_tools`.
- **Sub-agent Context:** No history passed—only enhanced query, system prompt, and tool defs. Parallel dispatch via `asyncio.gather`.
- **Memory Files:** `SKILL.md` only. No `AGENTS.md`.
- **Tools:** Risk-tiered execution (`ToolRegistry`/`ToolExecutor`). Schema validation before every call. Role-scoped visibility.
- **Caches:** Directory tree hash (mtime+size of `skills/` + `tools/`) invalidates discovery cache. `reset_deep_agent()` clears compiled graph.
- **Model Routing:** `src/config.py` checks budgets at call time. Order: Anthropic → OpenRouter → OpenAI → Google → Ollama. `OBSERVABILITY=1` attaches tracer.
