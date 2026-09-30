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
- **Dependencies:** `session_memory ← agent_orchestrator → tools_integration`. One direction: the orchestrator knows the tool layer, never the reverse.
- **Flow:** START → orchestrator (enhance query, name departments) → `{fanout | responder}` → END. Three nodes, one direction. The orchestrator is a **pure router**: it returns a JSON envelope `{enhanced_query, departments}` and binds no tools. There is no `tools` node and no top-level ReAct loop.
- **No iteration count in the orchestrator.** The graph is acyclic, so the router runs once per turn and has nothing to count. `iteration_count`/`max_iterations` were removed from `AgentState` — they belonged to the deleted ReAct dispatcher loop, and as *checkpointed* keys they carried across turns, so turn 2 of every conversation dropped its departments and answered "I could not route this request". `AGENT_MAX_ITERATIONS` now bounds only the sub-agent ReAct loop, which is the one thing that actually iterates. Do not reintroduce a per-turn counter in parent state; entry points must not need to reset one.
- **Hierarchy:** `orchestrator → sub-agent → tool`. Tool calls happen **only** inside a sub-agent's ReAct loop (`subagents.run_tool_loop`). A sub-agent never calls another sub-agent — the `task` delegation tool and sub-agent nesting are gone, so there is no depth counter and no `recursion_depth` in state.
- **Sub-agents:** `SUBAGENTS` is built from `skills/*/SKILL.md` (`subagents._skill_specs`) plus configured A2A agents. `skills/general` is the catch-all — there is no hardcoded `general-purpose` spec; one would make the orchestrator its own sub-agent and break the one-way flow. Frontmatter is parsed with `yaml.safe_load` — folded (`>`) and block (`|`) scalars work, and `allowed-tools` accepts comma- *or* space-separated names. Adding a department is adding a directory.
- **One executor:** department fanout (`subagent_engine.SubAgentEngine`) calls `subagents.run_department`. The engine only batches — it does not select tools, hold a registry, or re-enter the graph. `kind` picks the executor: `tool_loop` runs locally, `a2a` calls the remote agent by URL.
- **Department tools:** SKILL.md `allowed-tools` is the hard allowlist → capped at 20 visible → `relevance.sort_tools` narrows to 3-5 for *this* query → real `BaseTool` objects are bound. Unknown names are dropped with a warning, so a stale SKILL.md cannot yield an uncallable tool. The sub-agent does **not** choose its own tools; `select_department_tools` is the single filter.
- **Sub-agent Context:** No history passed — only the enhanced query, the `COMPLETION_CONTRACT` + SKILL.md prompt, and the bound tools. Parallel dispatch via `asyncio.gather`; the executor is blocking so it runs via `asyncio.to_thread`.
- **Audit trail:** the fanout node folds each department's `SubagentRun.usage` into `token_usage` and its `writes` into `pending_writes` + `audit_log`. Token spend and file writes now happen in sub-agents, so this is the only place the parent's accounting is kept.
- **Departments reach the dispatcher** through `_build_dispatcher_prompt()` (roster from `list_departments()`, built per call) and are validated by `_validate_departments` before fanout, so an invented name cannot reach the executor. The router always lands somewhere: an unusable reply falls back to `general`.
- **Memory Files:** `SKILL.md` only. No `AGENTS.md`.
- **Tools:** One execution surface — `ToolRegistry.dispatch(name, args)`, called from `subagents.run_tool_loop` (the only tool-call site in the repo). It validates the payload against the tool's declared `args_schema`, then routes the call by the tool's declared `ToolKind`: `local` → direct invoke, `mcp` → the sync-wrapped MCP tool, `a2a` → args serialized to JSON in the message body. Plain HTTP is deliberately not a kind (it would be an unguarded SSRF surface).
- **Payload contract:** a payload that does not match the spec is **rejected and named back to the model**, never coerced — `run_tool_loop` turns the `ValueError` into a `ToolMessage` and gives the sub-agent another turn. MCP declares its tools as raw JSON Schema *dicts*, which langchain-core does not validate at all, so `validation.normalize_schema` converts them to Pydantic models and `MCPTool` validates the raw input (langchain skips parsing entirely for a no-arg tool). Omitted optionals are pruned — langchain injects a `None` default for every field that has one, which would otherwise deliver keys the caller never passed.
- **Risk tiers:** `ToolRegistry`/`ToolExecutor` still carry `risk_level`/`requires_approval`/`allowed_roles`, and `dispatch` enforces approval + role. **No tool is actually gated** — `create_tool_registry` registers everything with `allowed_roles=("*",)` and nothing declares `requires_approval`. Treat these as available hooks, not active policy.
- **Caches:** Directory tree hash (mtime+size of `skills/` + `tools/`) invalidates discovery cache. `reset_deep_agent()` clears the compiled graph, model clients, MCP tools, and rebuilds `SUBAGENTS`. Discovery defaults are anchored to the repo root, not the process cwd.
- **Import cycle:** `subagents` → `discovery` → `tools_integration/__init__` → `tools`. The tool layer no longer reaches back up, so `tools.py` imports nothing from `agent_orchestrator`.
- **Model Routing:** `src/config.py` checks budgets at call time. Order: Anthropic → OpenRouter → OpenAI → Google → Ollama. `OBSERVABILITY=1` attaches tracer.
