# Codebase Guide

A map of this repository for a new senior ML engineer. It covers every service,
directory and file — including code that is not on the live request path.

Language note: the prose is deliberately plain, but it does not explain what a
graph checkpoint or a tool schema is. If you know LangGraph and LLM agents,
you can go straight to the "how a request actually flows" section.

---

## 1. What this project is

A **deep agent**: a single orchestrating LLM that plans a task, picks which
specialist sub-agents to run, runs them, checks their output, and answers.

It is a dispatcher-orchestrator shape, not a peer-to-peer multi-agent mesh:

```
user → orchestrator (enhance + pick departments)
          ├── fanout  (all selected sub-agents, in parallel)
          └── responder (single, no departments found)

fanout → responder → answer
```

Three services, with a strict one-way dependency rule:

```
session_memory  ←  agent_orchestrator  →  tools_integration
```

- `session_memory` knows nothing about the others. It stores and returns
  message history. Everyone depends on it, it depends on no one.
- `agent_orchestrator` owns the LangGraph, the model client, and dispatch.
  It reads history from `session_memory` and tools from `tools_integration`.
- `tools_integration` owns every callable: built-ins, file-loaded tools, MCP
  server tools, and the risk/role registry around them. It imports nothing from
  the other two services, so it can be tested and reasoned about alone.

`CLAUDE.md` states this rule. Breaking it is the fastest way to get circular
imports, which is why several modules use function-local imports on purpose
(noted inline where they appear).

---

## 2. How a request actually flows

This is the part to keep in your head. Everything else is detail hanging off it.

### Step 0 — Entry point hands the graph one message

`main.py` (CLI) and `app.py` (Streamlit) both do the same thing:

```python
agent.invoke({"messages": [HumanMessage(content=prompt)], "iteration_count": 0},
             config={"configurable": {"thread_id": ...}})
```

Two things matter:

- Only the **new** message is sent. History lives in the checkpointer, keyed by
  `thread_id`.
- `iteration_count` resets to 0 every turn, so the per-turn budget is not
  consumed by earlier turns.

`thread_id` is the entire memory identity. The CLI mints `cli-<uuid>` per
process; Streamlit mints one `uuid` into `st.session_state` and reuses it.

### Step 1 — orchestrator node

`src/services/agent_orchestrator/graph.py::\_local_orchestrator_node` calls
`orchestrator.py::call_orchestrator`.

The node reads `thread_id` from the **runtime config**, not from state. That is
not cosmetic: the dispatcher builds its prompt from
`session_memory.get_window(thread_id)`, and if the thread id is missing it reads
an empty window on every single turn. This was a real bug — see §12.

`call_orchestrator` then:
1. Checks the iteration budget. Over budget → returns a "Maximum iterations
   reached" `AIMessage` and no departments, which routes to the responder.
2. Fetches the message window from `session_memory` (default 20 messages,
   compressed if longer).
3. Prepends `DISPATCHER_SYSTEM_PROMPT`, which asks for JSON:
   `{"enhanced_query": ..., "departments": [...]}`.
4. Parses that JSON. The parser is deliberately lenient — it takes the text
   between the first `{` and the last `}` and requires `departments` to be a
   list. Prose around the JSON is fine.
5. Accumulates `usage_metadata` into `token_usage`.

It returns `next_message` (the raw response), `enhanced_query`,
`department_targets`, and the bumped counter.

Only on `iteration_count == 0` are `enhanced_query` and `department_targets`
actually adopted. Later iterations keep the first values and force departments
to `[]`, which prevents the loop from re-dispatching forever.

### Step 2 — conditional edge

`routing.py::route_after_orchestrator` checks tool calls first, then
departments: tool call → `tools`, departments present → `subagent_fanout`,
otherwise → `responder`. Tool calls win because a model that wants a tool emits
one *instead of* the JSON envelope.

Departments reach this edge only after `_validate_departments` drops names that
resolve to no registered department. The model can invent a name, and without
that check an invented department reaches the executor and surfaces as a
confusing "unknown department" error instead of an answer.

### Step 3a — fanout node (departments found)

`graph.py::\_subagent_fanout_node` passes **department names only** to
`SubAgentEngine().invoke_parallel(...)`. It deliberately does not build a
prompt: the name resolves to a `SubagentSpec`, and the spec's `skill` points at
a `SKILL.md` whose body becomes the sub-agent's system prompt. The node used to
synthesize `f"You are the {dept} specialist..."`, which bypassed `skills/`
entirely and discarded the department prompt. A name that no longer resolves is
dropped with a warning; if nothing resolves, the node returns empty results and
the responder answers directly rather than showing the user an error.

Note what is *not* here: no conversation history, no tools passed in. This is the
sub-agent context contract — a sub-agent sees the enhanced query, the
`COMPLETION_CONTRACT` + `SKILL.md` prompt, and whatever tools
`select_department_tools` shortlisted, nothing else. That is the point of the
architecture: parallel branches cannot see or corrupt each other's context.

### Step 3b — responder node (no departments)

`_responder_node` handles three cases:
- `subagent_results` non-empty → verify, then aggregate.
- `next_message` present → use its text, unless the text looks like the
  dispatcher's JSON envelope.
- neither → "No answer available."

`_is_envelope()` exists because the dispatcher's raw JSON reply is stored in
`next_message`. Without it, an unroutable query would be answered with
`{"enhanced_query": ..., "departments": []}`. With it, the text passes through
and the user sees a real (or error) message.

The node **always clears** `subagent_results` and `next_message` on the way out.
This is not tidiness. Those channels are part of the checkpointed state, so a
value left over from a previous turn gets replayed as this turn's answer.

### Step 4 — END

`fanout → responder → END`. Fanout does **not** loop back through the
orchestrator: a second dispatcher call would spend a token round-trip and then
have its departments thrown away by the `iteration_count > 0` rule.

---

## 3. Repository layout

```
.
├── CODEBASE_GUIDE.md          ← this file
├── CLAUDE.md                  ← project rules for the agent working on the repo
├── AGENTS.md                  ← project context injected into the *agent's* prompt
├── IMPLEMENTATION_PLAN.md     ← phase-by-phase build plan
├── README.md                  ← user-facing setup and usage
├── pyproject.toml             ← deps, ruff, mypy, pytest config
├── agent.py                   ← 3-line re-export shim
├── main.py                    ← CLI REPL entry point
├── app.py                     ← Streamlit UI entry point
├── tf_idf.py                  ← standalone demo, fully commented out
├── src/
│   ├── config.py              ← env-var getters, read at call time
│   ├── types.py               ← ToolSpec / SubAgent dataclasses
│   ├── utils.py               ← retry, transient classification, text extraction
│   ├── async_bridge.py        ← one event loop on a daemon thread
│   └── services/
│       ├── session_memory/
│       ├── agent_orchestrator/
│       └── tools_integration/
├── skills/
│   ├── research/SKILL.md
│   └── writer/SKILL.md
├── tools/
│   └── sample_tool.py
├── tests/                     ← 392 tests
└── workspace/                 ← agent file output (git-tracked sample notes)
```

Only two directories are data-driven at runtime: `skills/` and `tools/`. Add a
skill or a tool file and the system picks it up. Everything else is imported
normally.

For a new `skills/<name>/SKILL.md` to be routable, the frontmatter needs `name`
and a `description`; `allowed-tools` and `parallelizable` are optional. The
description is what the dispatcher matches a query against, so it is the part
worth writing well. Names are deduplicated (first in sorted directory order
wins, with a warning) and a duplicate does not shadow the earlier department.

Discovery defaults are anchored to the repo root rather than the process cwd, so
a `chdir` — a test's `tmp_path`, an app started elsewhere — cannot silently
reduce the registry to nothing. `reset_deep_agent()` calls `refresh_subagents()`
so the "edit a skill, then reset" workflow is real.

---

## 4. Service: `session_memory`

Four small files. This service is the memory of a conversation, nothing more.

**Directory:** `src/services/session_memory/`

### `checkpoint.py` — the shared saver (CORE)

Owns one module-level `MemorySaver` instance and hands it out via
`get_saver()`. This function is the reason the agent remembers anything.

The failure mode it prevents is worth understanding once. If the graph compiles
with `MemorySaver()` of its own while `get_session()` reads a different
instance, the graph happily writes checkpoints nobody can read. No error, no
warning — the dispatcher just sees an empty message window on every turn. The
fix is to have exactly one saver and route everyone through `get_saver()`.

- `get_session(thread_id)` — pull the message list for a thread. A small
  `_latest` dict maps thread → newest checkpoint id so we never depend on
  `MemorySaver.list()` ordering; threads created before that index existed fall
  back to `list()[0]`.
- `append_message(thread_id, message)` — read, append, write a new checkpoint
  with a fresh id, update the index.
- `get_turn_token_usage(thread_id)` — pull `token_usage` out of the newest
  checkpoint's metadata.

### `window.py` — what the orchestrator actually sees (CORE)

`get_window(thread_id, max_messages=20)` returns history as-is if it fits, and
`compress_messages(...)` if it does not. `append_message` is a thin re-export.

### `compression.py` — shrink long history

Two constants: `DEFAULT_MAX_MESSAGES = 20`, `MIN_KEEP_RECENT = 8`.

`compress_messages` keeps the most recent 8 messages verbatim, replaces the rest
with one summary `HumanMessage`, and formats the summary as a numbered
role-labeled list. The important detail is that it never splits a tool call from
its `ToolMessage` — an orphaned tool result is a validation error for most
providers, so the function cuts on boundaries only.

`_extract_role_label` handles the several shapes a message can arrive in
(`AIMessage`, `HumanMessage`, `SystemMessage`, dicts).

### `__init__.py`

Exports `get_window`, `append_message`, `compress_messages`. Note that
`get_saver` is **not** exported — `graph.py` imports it from `checkpoint`
directly, so the dependency on the concrete module is visible at the call site.

---

## 5. Service: `agent_orchestrator`

The largest service and the one that owns the architecture. Files in rough
import-order of importance.

### `graph.py` — graph compilation (CORE)

The single owner of the compiled graph. Everything that needs a graph calls
`get_deep_agent()` here.

```
START → orchestrator → (conditional) → tools ─────────→ orchestrator   (ReAct loop)
                             ↘            → subagent_fanout → responder → END
                             ↘            → responder ↗
```

The `tools → orchestrator` edge is conditional: it goes back for another turn
unless the iteration budget is spent or `consecutive_invalid_tools` has hit
`MAX_INVALID_TOOL_RETRIES` (3). A model that has been told "no tool named X"
and asks again is not going to recover on its own, so after a few misses the
responder answers from what is available.

Nodes are module-level functions, not closures:

- `_local_orchestrator_node(state, config)` — pulls `thread_id` from
  `config["configurable"]`, calls `call_orchestrator`. Declared with
  `config: RunnableConfig | None` rather than a bare `dict` because LangGraph
  type-checks that parameter and warns otherwise.
- `_subagent_fanout_node(state)` — builds department specs, runs
  `asyncio.run(engine.invoke_parallel(...))`. Safe here because LangGraph runs
  sync nodes on a worker thread with no event loop of its own. Calling this
  function from inside a running loop would fail.
- `_responder_node(state)` — described in §2.
- `_is_envelope(text)` and `_fallback_answer(state, query)` — the two helpers
  that make an unroutable query produce a sentence rather than raw JSON.

`get_deep_agent()` compiles once and caches in `_compiled_graph`. It compiles
with `checkpointer=get_saver()` — the shared saver, per §4.

`reset_deep_agent()` drops the cached graph and calls `agent_factory.clear_caches()`.
Call it after editing skills, tools, or model config; otherwise the running
process keeps the old graph. Tests call it between cases.

### `orchestrator.py` — the dispatcher (CORE)

`DISPATCHER_SYSTEM_PROMPT` is short on purpose. The legacy prompt (in the
deleted `plan.py`) was a long planner prompt; this one asks for a JSON contract
and nothing more, because the fanout node supplies the actual specialists.

`call_orchestrator(state, model, thread_id, max_history_messages=20)` — see §2
step 1. Note the module does *not* import `get_model`; the model is passed in.
The graph node does the import, function-locally, to avoid a cycle
(`graph` → `orchestrator` → `agent_factory` → `graph`).

`_parse_dispatcher_output` is the only parsing logic: find first `{`, find last
`}`, `json.loads`, require a dict with a list `departments`. Everything else
falls through to "treat the raw text as the answer".

### `routing.py` — one function (CORE)

`route_from_orchestrator`. This file used to hold three more routers and a
`max_iterations` branch that was unreachable. It is now 16 lines, and that is
the correct size for a conditional edge over one field.

### `state.py` — the graph state (CORE)

An `Annotated` `TypedDict`. Two fields use `Annotated[..., add]` so LangGraph
appends instead of replacing: `messages` and `audit_log`. Everything else is
last-write-wins.

Fields, grouped by who writes them:

| Field | Written by | Notes |
|---|---|---|
| `messages` | tools node, responder | append-only |
| `current_plan` | `write_todos` tool | |
| `workspace_files` | tools node | re-synced after every tool batch |
| `next_message` | orchestrator | staging slot, cleared by responder |
| `enhanced_query` | orchestrator | fixed on iteration 0 |
| `department_targets` | orchestrator | fixed on iteration 0 |
| `subagent_results` | fanout | cleared by responder |
| `token_usage` | orchestrator, tools node | child spend folded in |
| `iteration_count` / `max_iterations` | orchestrator | budget |
| `recursion_depth` | tools node | delegation nesting level |
| `pending_writes` | tools node | audit trail, informational |
| `audit_log` | tools node | append-only |
| `routing_decisions` | currently unused | kept for future routing history |
| `review_verdict` | currently unused | legacy from the deleted reviewer pass |
| `thread_id` | runtime config | mirrors `config["configurable"]["thread_id"]` |

The last three are legacy surface. They are cheap to keep and would be needed if
critic or plan-checker passes come back.

### `subagent_engine.py` — parallel department dispatch (CORE)

`SubAgentEngine.__init__` takes an optional registry and **defaults to
`create_tool_registry()`**, not a bare `ToolRegistry()`. A bare registry has zero
specs, which would leave every sub-agent with no tools and no error.

`invoke_parallel(subagents, enhanced_query, tools_registry=None)`:
- Batches by `get_max_parallel_tasks()` (default 4). Batches run in order;
  within a batch, `asyncio.gather` runs them concurrently.
- Each sub-agent gets the same wall-clock deadline.
- `_run_subagent` wraps the body in `asyncio.wait_for`, turning `TimeoutError`
  and any other exception into an error *string* for that sub-agent, so one
  failure never drops its siblings' results.
- Tool visibility: `registry.get_visible_tools(name)` → `get_tool_definitions`.
  A visibility lookup that raises is logged and yields an empty list, not a crash.

`_run_subagent_loop` is where a subtle performance bug lived.
`invoke_with_retry` is a **blocking** sync call. Calling it directly inside
`async def` blocks the event loop, and `asyncio.gather` quietly degenerates into
serial execution — four sub-agents took 1.22s instead of 0.31s. The executor
now runs via `asyncio.to_thread`.

The engine is **batching only**. It used to be a second, divergent sub-agent
implementation — a single model call with the tool descriptions pasted into the
prompt as JSON text, so the "ReAct loop" could never produce a tool call and any
tool output it appeared to have was hallucinated. It now delegates to
`subagents.run_department`, the same executor the `task` tool uses. The
`system_prompt` and `tool_defs` parameters are still accepted and ignored so
callers keep working; the registered spec is the source of truth. It also used
to embed the enhanced query in the system prompt *and* pass it as the
`HumanMessage`, so every sub-agent saw the query twice.

The loop body is a **single** model call, not a ReAct loop. The sub-agent gets
the enhanced query, a role prompt, and the top-5 relevant tool definitions, and
answers. Module-level `invoke_parallel(...)` is a module-level coroutine
convenience wrapper — note it is `async def` now, which is a public API change
for anything importing it from `agent_orchestrator.__init__`.

### `agent_factory.py` — model clients, tracer, tools node (CORE)

The biggest file, three unrelated concerns:

**1. `DeepAgentTracer`** — a real `BaseCallbackHandler` recording
`on_chat_model_start/end`, `on_llm_start/end`, `on_tool_start/end`,
`on_chain_start/end` into a list. Only constructed when `OBSERVABILITY=1`.
`get_tracer()` returns it or `None`. Events are inspectable via `get_events()`
and clearable via `clear()`.

`_maybe_attach_callbacks` wraps the model with `with_config(callbacks=[tracer])`.
Because `with_config` merges callbacks into every downstream invoke, the
orchestrator, sub-agent loops, and tool loops all record events without each
call site passing callbacks. This is why the tracer is attached in one place.

**2. Model selection** — `get_model()` → `_get_cached_model()`. The fallback
order is Anthropic → OpenRouter → OpenAI → Google → Ollama, chosen by which API
key is present, all at `temperature=0`. One client is cached per
`(provider, model_name)` in `_model_cache`.

The cache lives for the process and is cleared by `clear_caches()`. So a
`.env` change requires `reset_deep_agent()`. That is the same contract as the
graph cache and it is deliberate.

**3. `local_tools_node`** — the ReAct tools node. This is where the classic
tool-calling loop lives:
- Non-`task` tool calls run first, sequentially, through `ToolExecutor`.
- `task` calls are collected, checked against `get_max_subagent_depth()`, then
  split by `is_parallelizable(subagent_type)`.
- Parallelizable ones go into a `ThreadPoolExecutor(max_workers=min(
  get_max_parallel_tasks(), len(independent)))`; the rest run one at a time.
- One shared `batch_deadline` for the whole batch, so a hung batch costs one
  timeout rather than N. `executor.shutdown(wait=False, cancel_futures=True)` —
  a hung worker must not block the parent past the deadline.
- Results are collected in submission order, so `ToolMessage` ordering is
  deterministic across runs.
- Child token usage and child file writes are folded into the parent's
  `token_usage` and `pending_writes`. A parent's number is the whole tree's
  spend, not just its own level.

`get_deep_agent()` here is a wrapper that ensures `./workspace` and `./skills`
exist, then delegates to `graph.get_deep_agent()`. The bootstrap is kept because
the `task` tool and the Streamlit app depend on those directories existing.

### `aggregator.py` — merge department results (CORE, small)

Single department → return its text unchanged. Multiple → a header naming the
count, then `## <name>` sections. It is string composition, not an LLM call.
Worth knowing: it is the reason a fanout answer reads as a concatenation.

### `verification.py` — the check on fanout output (CORE, small)

`verify(results, context) -> {"approved", "reasons", "warnings"}`. Three cheap
deterministic checks, no LLM:
- empty results,
- any error string in the results,
- a `/tmp/` path (regex `(^|[\s"'(=])/?tmp/`, so `Not/tmp/x` does not match),
- low keyword overlap with the query — words longer than 2 characters only.

The keyword check writes to `warnings`, not `reasons`. It used to fail the
verdict, which contradicted itself: an answer can use different words than the
query and still be right.

The responder logs a warning on `not approved` and aggregates anyway. Verification
is advisory here, not a gate.

### `memory.py` — skills, tools, workspace, and the long system prompt

`_dir_tree_hash(directory)` walks a directory and hashes every file's relative
path plus its mtime and size. Used to invalidate the skills cache, the dynamic
tool cache, and the discovery cache. Same function, three call sites.

- `get_workspace_files()` — relative paths of everything under `./workspace`.
- `get_skill_info(path)` / `get_skill_body(name)` — parse a `SKILL.md`
  front-matter block and return metadata or body.
- `get_skills_summary()` / `get_tools_summary(tools)` — one-line-per-item
  summaries for the system prompt.
- `get_memory_content()` — read `./AGENTS.md` if present.
- `get_system_prompt(tools_dict)` — the **legacy** long planner prompt. It
  describes the write-todos → load skill → delegate workflow, forbids `/tmp/`,
  and requires output under `./workspace`.

**This prompt is not used on the live path.** The Phase 3 dispatcher prompt in
`orchestrator.py` replaced it. It is kept because `build_role_prompt` and the
tests still reach into this module, and because a fuller prompt is a reasonable
starting point if the planner is ever brought back. Read it as reference.

### `subagents.py` — the sub-agent registry (CORE)

`SubagentSpec` fields: `description`, `kind`, `parallelizable`, `aliases`,
`skill`, `tools`, `url`. The `kind` value decides everything downstream:

| `kind` | Behavior | Entry point |
|---|---|---|
| `tool_loop` | restricted tools + `SKILL.md` prompt + a real ReAct loop | `run_tool_loop` |
| `graph` | the full compiled graph, fresh thread | `graph.get_deep_agent()` |
| `a2a` | HTTP call to a remote agent | `a2a_client.call_a2a_agent` |

`SUBAGENTS` is assembled by `build_subagent_registry()` from three sources:

1. **`skills/*/SKILL.md`** via `discovery.discover_subagents` — each becomes a
   `tool_loop` spec whose `skill` names the SKILL.md (the prompt source) and
   whose `tools` come from the `allowed-tools` frontmatter. This is what makes
   `skills/` genuinely data-driven; adding a department is adding a directory.
2. **`_RUNTIME_SUBAGENTS`** — `general-purpose` (graph, not parallelizable, alias
   `general`). Hardcoded: it is a capability of the runtime, not a skill, and has
   no `SKILL.md`.
3. **`_a2a_specs()`** — remote agents from the `A2A_AGENTS` env var.

`refresh_subagents()` rebuilds it; `reset_deep_agent()` calls it.

`list_departments()` is the subset that is *routable* — `tool_loop` specs backed
by a SKILL.md. `general-purpose` is a delegation target and A2A agents are
remote services, so neither belongs in the dispatcher's roster.

**Tool selection** (`select_department_tools`) is three stages, matching the
documented contract: the SKILL.md `allowed-tools` allowlist is the hard limit,
capped at 20 visible, then `relevance.sort_tools` narrows to 3-5 for the
current query. Names that do not resolve are dropped with a warning — a stale
name in a SKILL.md must not become a tool the model can call but never execute.
The result is **real `BaseTool` objects**, not the metadata dicts
`get_tool_definitions` returns; resolving names back to callables is what makes
the shortlist bindable at all. A failing sorter falls back to the capped
allowlist.

**Budget exhaustion is reported, not swallowed.** When `run_tool_loop` runs out
of turns, the last model message is usually another tool call whose content is
`""`. Returning that hands the parent an empty string, which reads as "the
subagent found nothing" and leaves it to redo the whole task. The loop instead
reports the truncation, the tools it called, and the files it wrote.

`SUBAGENTS` is the single source of truth. It drives the `task` tool's
`subagent_type` enum, the dispatch in `tools._execute_task`, the
parallel-vs-sequential split, and role prompts. Adding a sub-agent is adding one
dict entry.

The module calls `load_dotenv()` itself. This is load-order surgery: the
`task` tool's type enum is built statically at import time from `SUBAGENTS`, and
the import chain (`tools` → `subagents`) runs before `agent_factory`'s
`load_dotenv()`. The cost is that changing `A2A_AGENTS` needs a process restart.

`COMPLETION_CONTRACT` is the shared closing instruction for every tool-loop
sub-agent: save to the exact path named in the task, never `/tmp/`, never a
shared default filename (parallel sub-agents would overwrite each other), and
end with a short summary of what was written.

`build_role_prompt(spec)` = contract + the spec's `SKILL.md` body + `AGENTS.md`.

`run_tool_loop(prompt, description, tools, max_iterations=10)` is the real ReAct
loop: bind tools, invoke, execute any tool calls, append results, repeat until
the model answers without tool calls or the budget runs out. Returns
`(final_text, usage, write_ops)`. On budget exhaustion it returns the last AI
message's text rather than an error, so a chatty sub-agent still contributes
something.

### `guardrails.py` — filesystem containment (CORE for safety)

Every file tool routes through here.

- `get_workspace_root()` — the `./workspace` `Path`.
- `validate_and_normalize_path(path, must_be_in_workspace=False)` — resolve,
  then confirm the resolved path is inside an allowed root. Symlinks are
  resolved before the check, so a symlink pointing out of the workspace is
  rejected. Known caveat: there is a TOCTOU window between the check and the
  `open()`. Fine for a local agent, not fine for a multi-tenant service.
- `validate_read_path(path)` — read is wider than write: `AGENTS.md`, plus
  anything under `./workspace` or `./skills`.
- `clear_workspace()` — used by the Streamlit sidebar reset button.

The invariant: **reads** may touch `AGENTS.md`, `workspace/`, `skills/`;
**writes** may only touch `workspace/`.

### Deleted files, still referenced in git history

- `plan.py` — the old planner. Superseded by `orchestrator.py`; the docstrings
  in `orchestrator.py` and `memory.py` compare against it.
- `review.py` — critic / plan_checker / reflection passes. Removed; the
  `review_verdict` state field and the tracer's role names are their residue.
- `tools_integration/rag.py` — the old TF-IDF retrieval layer. Replaced by
  `fetch_url` + `internet_search`; see `tf_idf.py` in §9.

If you read an old blog post or commit message referencing these, the code is
gone.

---

## 6. Service: `tools_integration`

Everything callable, plus the policy layer around it.

**Directory:** `src/services/tools_integration/`

### `tools.py` — the built-in tool set (CORE, largest file, 466 lines)

Ten built-ins, all defined with LangChain's `@tool` decorator so the docstring
*is* the schema:

| Tool | What it does |
|---|---|
| `write_todos` | records a plan; result folded into `current_plan` |
| `internet_search` | Tavily; needs `TAVILY_API_KEY`; returns 300-char snippets |
| `read_file` | via `validate_read_path` |
| `write_file` | via `validate_and_normalize_path(must_be_in_workspace=True)` |
| `edit_file` | string replace in a workspace file |
| `list_files` | workspace listing |
| `search_files` | substring search across workspace, capped at 30 matches |
| `fetch_url` | full page text via `fetch_public_url` |
| `task` | delegate to a sub-agent |
| `list_tools` | discovery — the model can ask what it can do |

Note the pattern in every file tool: validate, then return a **string** on
error. Tools return strings here, not raise, so one failed call does not kill
the loop. (`_execute_task` is the deliberate exception — it must not run
outside the tools node, so it refuses loudly.)

`task` is built by hand rather than with a plain decorator, because its
`subagent_type` parameter is a `Literal` generated from `SUBAGENTS.keys()` and
its docstring is generated from the same registry. Adding a sub-agent updates
the tool schema automatically. Direct invocation returns an error string —
the depth guard lives in `local_tools_node`, which has the graph state, so
bypassing it would break the recursion limit.

`FETCH_URL_TIMEOUT_SECONDS = 15` and `FETCH_URL_MAX_BYTES = 1_000_000` sit next
to `fetch_url` on purpose: the size cap is what keeps a 200 MB response from
landing in the model's context.

**`_execute_task`** is the dispatch:

- resolve the spec (unknown type → error string listing valid types),
- `tool_loop` → restricted toolset from `spec.tools`, `run_tool_loop` with
  `build_role_prompt(spec)`,
- `a2a` → `call_a2a_agent(spec.url, description)`, importing the SDK lazily so
  it stays off the import path until used. A remote agent reports no token usage
  and its file writes are invisible, so both come back empty — by design,
  not an oversight,
- `graph` → the full compiled graph on a **fresh** thread
  (`subagent-<uuid>`). Reusing the parent's thread would merge child state into
  the parent checkpoint and back.

The graph branch returns the last **non-empty AIMessage**, not `messages[-1]`,
which could be a `ToolMessage`.

**`load_dynamic_tools(tools_dir)`** loads `@tool` functions from every `.py`
under `./tools` using `importlib.util.spec_from_file_location` — nothing is
added to `sys.path`. Each module gets a unique name from its path, so files in
subdirectories cannot collide. Results are cached against
`abspath(tools_dir) + "|" + _dir_tree_hash(tools_dir)`; the absolute path is in
the key because the tree hash only covers relative paths, and two checkouts with
identical `./tools` contents must not share a cache.

**`get_all_tools()`** merges three sources with a fixed precedence:

```
built-ins  >  MCP tools  >  dynamic file tools
```

Collisions are logged; the higher-precedence tool wins. This ordering is the
reason a file in `./tools/` cannot shadow `read_file`.

**`create_tool_registry()`** turns `get_all_tools()` into a populated
`ToolRegistry`, unwrapping `tool.func` for `StructuredTool` instances.

### `registry.py` — specs, visibility, risk (CORE)

`ToolRegistry` holds two dicts: specs and callables.

- `register_builtin(name, callable_, risk_level="low")` — the path used for
  built-ins; description comes from the function's docstring.
- `get_tools_for_role(role)` — everything with no `allowed_roles`, or with this
  role, or with `*`.
- `get_visible_tools(subagent_name)` — **capped at 20.** `general-purpose` gets
  the first 20 specs; anything else is filtered by role first. The 20 is the
  sub-agent context budget from `CLAUDE.md`.
- `get_tool_definitions(names)` — the JSON-schema-shaped dicts the LLM ranks.
- `_validate_args` — currently a `logger.debug` no-op. The real gate is
  LangChain's own schema check before the call. Do not assume this validates
  anything.
- `execute(...)` — resolves the callable, checks the risk tier and approval flag,
  calls through.
- `RequiresApprovalError` — raised for high-risk tools.

### `executor.py` — risk-tiered execution (CORE)

A thin wrapper over the registry, present mostly so the policy is in one place
and the call site names a tier rather than re-implementing the checks. Tiers:
`low` executes directly, `medium` validates then executes, `high` raises
`RequiresApprovalError`. `execute` is `async`; `execute_sync` wraps it with
`asyncio.run`, which is what `local_tools_node` calls.

### `decorator.py` — `@tool_spec` (CORE)

Attaches `ToolSpecMetadata` (name, description, `risk_level`, `requires_approval`,
`allowed_roles`) as a `__tool_spec__` attribute. It does not change the
function. Stack it under `@tool`, as `tools/sample_tool.py` does:

```python
@tool
@tool_spec(name="get_current_time", description="...", risk_level="low")
def get_current_time(timezone: str = "UTC") -> str: ...
```

### `discovery.py` — scan `skills/` and `tools/` (CORE for extensibility)

`discover_subagents("./skills")` walks `skills/*/SKILL.md`, parses the YAML-ish
front matter, and returns `name`, `department`, `system_prompt`,
`allowed_tools`, `skill_file`, `protocol`. Cached by directory tree hash.

`discover_tools("./tools")` does the same for `@tool_spec` tags.

`pyfile_to_module_name` maps a path to a stable module name.

This module and `tools.load_dynamic_tools` overlap on purpose: `discovery`
reads metadata without executing, `load_dynamic_tools` actually imports. Use
discovery to list what exists; use the loader to get callables.

### `relevance.py` — the 20 → 5 filter (CORE)

`sort_tools(model, enhanced_query, tool_defs, max_tools=5)` sends the query plus
all visible tool schemas to the LLM and asks for a JSON array of names, ordered
by relevance. `_extract_tool_names` scans for the first `[` and last `]`.

Every failure path degrades to `tool_defs[:max_tools]` — first-N, unranked —
rather than raising. A sort failure must not stop a sub-agent from working.

`args_schema` is stripped before sending: it is large and the model only needs
names and descriptions to rank.

### `research_fetch.py` — SSRF-safe URL fetch (CORE for safety)

`fetch_public_url(url, timeout, max_bytes)`:
- resolve the hostname, and **pin** the resolved IP so the connection cannot be
  redirected to an internal address between check and connect,
- reject non-public addresses (`_public_address`),
- allow at most `MAX_REDIRECTS = 5`,
- no proxies,
- cap the response at `max_bytes` and the whole call at `timeout`.

This is async. The `fetch_url` tool calls it through `async_bridge.run_sync`.

### `mcp_client.py` and `mcp_bridge.py` — MCP server tools

Two implementations of the same thing. Read both; they are near-duplicates and
knowing which is live saves confusion.

`mcp_client.py` — **the one on the live path.** `load_mcp_tools()` is sync,
driven through `async_bridge.run_sync`. Servers come from
`config.get_mcp_servers()`. Nothing runs until `MCP_SERVERS` is set.

`mcp_bridge.py` — async-native. `load_mcp_tools_async()` plus a sync
`load_mcp_tools()` that calls `asyncio.run(...)` on it. The `__init__.py` exports
*this* one, so `from tools_integration import load_mcp_tools` and
`from tools_integration.mcp_client import load_mcp_tools` are **different
functions** that return the same thing. The registry cache is per-module, so
each one keeps its own copy. If you touch MCP loading, be deliberate about
which import you use.

Both wrap each MCP tool as a sync `StructuredTool`, because
`langchain-mcp-adapters` builds tools with `coroutine=` and no `func` — a plain
`.invoke()` raises `NotImplementedError`. Server names prefix every tool
(`<server>_<tool>`) so two servers cannot silently collide. Cached by the
serialized server config; `clear_mcp_tools_cache()` invalidates.
`agent_factory.clear_caches()` clears both copies.

One subtlety in the wrapper: `args_schema` is carried over verbatim (MCP
supplies a raw JSON Schema), but `response_format` is deliberately *not* set.
The wrapper's `func` already returns unwrapped content, and declaring
`content_and_artifact` would unpack it a second time.

### `a2a_client.py` and `a2a_bridge.py` — remote agents over A2A

Same duplication, same live/dead split: `a2a_client.py` is live (it uses
`async_bridge`), `a2a_bridge.py` is the async-native version with a
`trace_id` parameter. `tools_integration.__init__` exports the bridge's.

Both: fetch the agent card from the base URL, send one text message, then poll.
`TERMINAL_STATES` are done/failed/canceled/rejected. `INTERRUPTED_STATES`
(`input-required`, `auth-required`) are non-terminal but will never advance on
their own, so polling them would spin forever — treat them as a stop condition.
`POLL_INTERVAL_SECONDS = 1.0`, `MAX_POLLS = 300` (about five minutes).

A2A 1.x uses protobuf types (`Role.ROLE_USER`, `Part(text=...)`), not the
Pydantic `TextPart` shapes in older tutorials. Copying a 0.3-era example will
not work.

The httpx client is created and closed inside the single coroutine on purpose —
httpx binds its connection pool to the loop that created it, so a client must
not outlive or cross loops.

### `registry.py` / `executor.py` — see above.

### `__init__.py`

The docstring is the best short description of this service in the repo. Note it
exports `load_mcp_tools` and `call_a2a_agent` from the **bridge** modules, not
the client modules. See the warning above.

---

## 7. Shared top-level modules

### `src/config.py` (CORE)

Every getter reads the environment **at call time**, so a `.env` edit applies to
the next invocation without a restart. Two private helpers: `_env_int` (falls
back on unparseable input) and `_env_json` (logs and falls back on invalid JSON
or a non-dict).

| Env var | Getter | Default |
|---|---|---|
| `AGENT_MAX_ITERATIONS` | `get_max_iterations()` | 25 |
| `AGENT_MAX_SUBAGENT_DEPTH` | `get_max_subagent_depth()` | 3 |
| `MAX_PARALLEL_TASKS` | `get_max_parallel_tasks()` | 4 (floor 1) |
| `SUBAGENT_TIMEOUT_SECONDS` | `get_subagent_timeout_seconds()` | 600 |
| `MCP_SERVERS` | `get_mcp_servers()` | `{}` |
| `A2A_AGENTS` | `get_a2a_agents()` | `{}` |

Depth semantics are worth reading twice: a parent at depth D spawns at D+1, and
delegation is refused when the parent's depth is already at the limit. So the
top-level agent (depth 0) nests three levels.

`get_mcp_servers` and `get_a2a_agents` validate keys against `^[a-z0-9_-]+$` and
drop invalid ones with a warning, because those names become part of a tool name
(`<server>_<tool>`) and part of the `task` tool's static type enum — and LLM
providers reject type enums outside that alphabet.

Example:

```
MCP_SERVERS='{"math": {"transport": "stdio", "command": "python", "args": ["/abs/server.py"]},
               "weather": {"transport": "http", "url": "http://localhost:8000/mcp"}}'
A2A_AGENTS='{"remote_researcher": {"url": "http://localhost:9999", "description": "..."}}'
```

### `src/utils.py` (CORE)

- `is_transient_error(exc)` — substring match over a `_TRANSIENT_MARKERS` tuple
  (timeout, connection, rate limit, 429, 5xx, overloaded, temporarily...). This
  is a heuristic. It is what keeps deterministic failures from burning ~14
  seconds of pointless sleeps.
- `invoke_with_retry(model, messages, max_retries=3, base_delay=2.0)` — calls
  `model.invoke`, retries only transient errors with exponential backoff
  (2s, 4s, 8s), re-raises immediately otherwise. **Blocking** — see the
  `asyncio.to_thread` note in §5.
- `get_message_text(content)` — pulls text out of a string or a multimodal
  content list. Needed because providers return either.

### `src/async_bridge.py` (CORE for MCP/A2A)

One event loop, on one daemon thread, for the whole process. `run_sync(coro)`
schedules onto it with `run_coroutine_threadsafe` and blocks for the result.

Why a persistent loop instead of `asyncio.run` per call:
- `asyncio.run` raises if the calling thread already has a running loop,
- MCP issue #29 documented hangs when MCP calls are wrapped in `asyncio.run`,
- one loop is callable from the main thread *and* from the `ThreadPoolExecutor`
  workers that run parallel sub-agent tasks.

The loop starts lazily, so a process that never touches MCP or A2A never spawns
a thread. `close_async_bridge()` stops it and joins the thread; safe to call when
it was never started, and used by tests.

Windows caveat in the docstring: the stdio MCP transport wants a
`ProactorEventLoop`, and this module inherits the platform default. Only
exercised on macOS today.

### `src/types.py` (mostly unused)

`ToolSpec` and `SubAgent` dataclasses. `ToolSpec` is consumed by
`ToolRegistry.register`. `SubAgent` has no live caller — `SubagentSpec` in
`subagents.py` is the real one. Kept for type clarity and tests.

---

## 8. Entry points

### `agent.py` — 3 lines

```python
from src.services.agent_orchestrator.graph import get_deep_agent
__all__ = ["get_deep_agent"]
```

A shim so `main.py` and `app.py` can do `from agent import get_deep_agent`
without importing a deep service path. Also the single place to swap in a
different graph implementation for a test.

### `main.py` — CLI REPL

Mints `cli-<uuid>` once, then loops: read input, send only the new message with
`iteration_count: 0`, print the last `AIMessage`, print total tokens and this
turn's iteration count. Handles `EOFError`/`KeyboardInterrupt` and `exit`/`quit`.
About 48 lines; read it as the reference for correct invocation.

### `app.py` — Streamlit UI (364 lines)

Calls `agent.stream(..., stream_mode="updates")` and renders each node's state
delta live, which is why you can watch the graph work.

Sidebar panels: governance (iteration count, token usage, current node), active
skills (from `get_skill_info`), audit log, system status, shared memory
(`AGENTS.md`), and a workspace file browser. `clear_workspace()` is wired to a
reset button.

State lives in `st.session_state`: `messages`, `current_plan`,
`workspace_files`, `audit_log`, `token_usage`, `last_action`, `current_node`,
`iteration_count`, and one `thread_id` minted at startup.

The one rule to copy if you build another UI: send only the new message and
reset `iteration_count` every turn. The checkpointer holds the rest.

---

## 9. Content files and the not-in-use code

### `skills/research/SKILL.md`

YAML front matter (`name`, `description`, `license`, `compatibility`,
`allowed-tools`) then markdown body: the research procedure (search → `fetch_url`
the promising pages → alternate queries if thin), the workspace discipline
(`list_files` before writing, never guess a path another sub-agent may have
taken), and a completion contract (summary, key facts, sources, confidence
score). Worked example at the end.

### `skills/writer/SKILL.md`

Same shape. Reads notes from the workspace, outlines, matches tone to audience,
writes GitHub-flavored markdown to a workspace file. Completion contract names
the saved path, the outline, and the tone.

Both are on the live path — `build_role_prompt` reads them for `tool_loop`
sub-agents. Adding a `skills/<name>/SKILL.md` plus one `SUBAGENTS` entry is the
whole extension story.

### `tools/sample_tool.py` — the extension template (19 lines)

`get_current_time`, decorated `@tool` over `@tool_spec` with `risk_level="low"`.
Copy this file, change the function, and it is discovered on the next
`get_all_tools()`. Note it imports `pytz` inside the function body, so a missing
optional dependency does not break module load for every other tool.

### `AGENTS.md` — dual purpose

Read by `get_memory_content()` and injected into both the legacy system prompt
and every `build_role_prompt`. Short: mission, conventions (tools are
functional, skills are directories with a `SKILL.md`, sub-agents isolate context,
all intermediate work goes to `./workspace`), and the three known roles. It is
project context for the *agent*, not documentation for you — the editing
guidance for humans is `CLAUDE.md`.

### `CLAUDE.md`

Project rules for whoever (or whatever) is editing this repo: commands, the
dependency order, the flow, the sub-agent limits, the cache-invalidation
contract, and the model routing order. Read it before you change architecture.

### `IMPLEMENTATION_PLAN.md`, `README.md`

The phase plan this codebase was built against, and the user-facing setup guide.

### `tf_idf.py` — fully commented out (NOT IN USE)

Every line is prefixed with `#`. It contains a from-scratch TF-IDF vectorizer
(no sklearn) and a sliding-window attention demo in PyTorch. It was the demo for
the old `rag.py` retrieval layer, which has since been deleted. Keep it as
reading material if you want a compact TF-IDF or a small attention
implementation; it does not run and nothing imports it.

### `workspace/` — git-tracked sample output

Six markdown files: `research_notes.md`, `notes_1.md`, `notes_3.md`,
`notes_23.md`, `multi_agent_research.md`, `research_multi_agent_systems.md`.
These are real agent output, committed as examples. `CLAUDE.md` says memory
files are `SKILL.md` only and there is no per-session `AGENTS.md`; `workspace/`
is scratch output, not memory. A test or the Streamlit reset button can clear
it — the reset button deletes the contents, not the directory.

### `app_new.log`

Streamlit's `FileHandler` target. Git-tracked by accident. Safe to ignore or
untrack.

---

## 10. Configuration reference

Everything is env-var driven. There is no config file.

| Variable | Used by | Effect |
|---|---|---|
| `ANTHROPIC_API_KEY` | `agent_factory` | first in the model fallback chain |
| `OPENROUTER_API_KEY` | `agent_factory` | second |
| `OPENAI_API_KEY` | `agent_factory` | third |
| `GOOGLE_API_KEY` | `agent_factory` | fourth |
| *(none of the above)* | `agent_factory` | falls back to Ollama `gemma4:12b-mlx` |
| `*_MODEL` | `agent_factory` | per-provider model id override |
| `TAVILY_API_KEY` | `tools.internet_search` | missing → tool returns a setup message, not a crash |
| `OBSERVABILITY` | `agent_factory` | `=1` attaches `DeepAgentTracer` |
| `AGENT_MAX_ITERATIONS` | `orchestrator` | per-turn dispatcher budget |
| `AGENT_MAX_SUBAGENT_DEPTH` | `tools` node | delegation nesting cap |
| `MAX_PARALLEL_TASKS` | `subagent_engine`, tools node | concurrency cap |
| `SUBAGENT_TIMEOUT_SECONDS` | both | wall-clock guard, shared per batch |
| `MCP_SERVERS` | `mcp_client` | JSON object of server connections |
| `A2A_AGENTS` | `subagents` | JSON object; **needs a process restart** |

`load_dotenv()` is called in `agent_factory` and again in `subagents`. Copy that
second call if you add a module that must see `.env` at import time.

---

## 11. Tests

392 tests, `testpaths=["tests"]`, `pythonpath=["."]`.

Layout mirrors `src/`:

```
tests/
├── fake_models.py                       ScriptedChatModel
├── test_a2a.py  test_async_bridge.py  test_config.py
├── test_delegation.py  test_guardrails.py  test_mcp.py
├── test_observability.py  test_research_fetch.py
├── test_state.py  test_subagents.py  test_tools.py  test_utils.py
└── services/
    ├── agent_orchestrator/   test_aggregator, test_graph, test_orchestrator,
    │                         test_routing, test_subagent_engine, test_verification
    ├── session_memory/       test_compression, test_window
    └── tools_integration/    test_a2a_bridge, test_discovery, test_executor,
                              test_mcp_bridge, test_registry, test_relevance
```

`tests/fake_models.py::ScriptedChatModel` is the key piece of test
infrastructure. It is one instance shared by every actor, and it tells which
actor is calling by **scanning the messages for a role substring** — "generic
Deep
 Agent" → orchestrator, "specialized subagent" → a tool-loop sub-agent. That
works because each actor's system prompt is distinct. Its `counts` dict is how
tests assert call counts, which is how the wasted-second-dispatcher-call
regression was caught. It must scan all messages, not just the first, because
Phase 3 nodes pass dict-form messages.

Current baseline: **370 pass, 0 fail.** ruff reports 87 findings and mypy 22
errors, all pre-existing and none in the files touched most recently. Run
`uv run ruff check .` and `uv run mypy src` to see the current numbers rather
than trusting these.

---

## 12. Gotchas worth knowing before you debug

**Memory appears broken but nothing errors.** Two `MemorySaver` instances.
Everything routes through `session_memory.checkpoint.get_saver()`; do not
construct one anywhere else.

**The dispatcher sees an empty conversation.** `thread_id` is not in state. It
comes from `config["configurable"]["thread_id"]`, and `_local_orchestrator_node`
is the only place that reads it.

**`asyncio.gather` is not parallel.** Any blocking call inside an `async def`
serializes the batch. `invoke_with_retry` is blocking; wrap it in
`asyncio.to_thread`. You can confirm with a timing probe — four sub-agents should
take roughly a quarter of the serial time.

**`_subagent_fanout_node` uses `asyncio.run`.** That is correct there because
LangGraph runs sync nodes on a worker thread. Do not call the function from
inside a coroutine.

**`subagent_results` must be cleared by the responder.** It is checkpointed
state. A leftover value is replayed as the next turn's answer.

**A test that never sends a user message proves nothing.** `tests/.../test_graph.py::_initial_state`
used to drop its argument and build `messages: []`, which is why a real memory
bug survived a fully green suite. When you touch graph tests, check that the
state builder actually uses the message you pass it.

**`A2A_AGENTS` changes need a restart.** The `task` tool's type enum is built at
import time.

**There are two `load_mcp_tools` and two `call_a2a_agent`.** The package
`__init__` exports the *bridge* versions; `tools.py` uses the *client* versions.
They cache separately. Pick one deliberately and write it in the import.

**`mypy` and `ruff` are not clean.** They are a floor, not a gate. Neither is
enforced in CI.

---

## 13. Where things were, and why

The Phase 3 rewrite moved from a long-plan orchestrator with explicit
plan/critic/reflection passes to a JSON dispatcher with a fanout. The net
effect: three services, one compiled graph, no separate planning step, and
verification reduced to a cheap deterministic check.

Reading the deleted files (`plan.py`, `review.py`, `rag.py`) via git history is
worth ten minutes if you want to understand why the current shape is as small as
it is. The current code is the residue of removing them, and most of what looks
redundant — the `review_verdict` field, the tracer's unused role names, the
legacy `get_system_prompt` — is a scar from that removal rather than a mistake.
