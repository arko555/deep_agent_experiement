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
agent.invoke({"messages": [HumanMessage(content=prompt)]},
             config={"configurable": {"thread_id": ...}})
```

Only the **new** message is sent. History lives in the checkpointer, keyed by
`thread_id`. That is the whole contract — there is no per-turn state to reset.

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
1. Fetches the message window from `session_memory` (default 20 messages,
   compressed if longer). This history is what the enhanced query is built
   from.
2. Prepends the dispatcher system prompt, which lists the live department
   roster and asks for JSON:
   `{"enhanced_query": ..., "departments": [...]}`.
3. Parses that JSON. The parser is deliberately lenient — it takes the text
   between the first `{` and the last `}` and requires `departments` to be a
   list. Prose around the JSON is fine.
4. Validates each department name against the live roster
   (`_validate_departments`), dropping invented ones with a warning.
5. Applies the `general` fallback if the validated list is empty — the router
   always lands somewhere, and `skills/general/SKILL.md` is the catch-all that
   greets, answers, or refuses.
6. Accumulates `usage_metadata` into `token_usage`.

`DISPATCHER_SYSTEM_PROMPT` is built per call (`_build_dispatcher_prompt()`)
from `list_departments()`, so a SKILL.md added at runtime appears in the
roster without a restart. The module-level name is kept only for importers.

It returns `next_message` (the raw response), `enhanced_query`, and
`department_targets`. Those are its only two jobs: restate the request with
enough history folded in to be actionable, and name who should handle it.

**There is no iteration budget here.** The graph is acyclic, so the router runs
once per turn and has nothing to count. Iteration counting lives where
iteration actually happens — inside a sub-agent's ReAct tool loop, bounded by
`AGENT_MAX_ITERATIONS` (`subagents.run_tool_loop`). A per-turn router that
carries a counter in checkpointed state is a bug, not a safety feature: see
§12 for the one that shipped.

### Step 2 — conditional edge

`routing.py::route_after_orchestrator` is two checks over one field:
departments present → `subagent_fanout`, otherwise → `responder`. There is no
`tools` branch — the router binds no tools, so it cannot emit a tool call. A
`_has_tool_calls` guard is kept anyway: it routes a turn whose last AI message
carries tool calls to the *responder*, because such a message falling into
`subagent_fanout` would leave an unconsumed tool request and a model told it
cannot act on its own ask. The guard should be unreachable; it is cheap
insurance against a tool ever being bound to the router again.

Departments reach this edge only after `_validate_departments` drops names that
resolve to no registered department. The model can invent a name, and without
that check an invented department reaches the executor and surfaces as a
confusing "unknown department" error instead of an answer. And the list is
never empty going *in*: an unusable router reply falls back to `general`
(see §5, `orchestrator.py`), so the empty-departments route is itself a
should-never-happen path.

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
orchestrator: a second dispatcher call would spend a token round-trip
re-deciding a query that has already been routed.

---

## 3. Repository layout

```
.
├── CODEBASE_GUIDE.md          ← this file
├── CLAUDE.md                  ← project rules for the agent working on the repo
├── AGENTS.md                  ← project context injected into the *agent's* prompt
├── README.md                  ← user-facing setup and usage
├── pyproject.toml             ← deps, ruff, mypy, pytest config
├── agent.py                   ← 3-line re-export shim
├── main.py                    ← CLI (REPL, one-shot query, --tools listing)
├── app.py                     ← Streamlit UI entry point
├── src/
│   ├── config.py              ← env-var getters, read at call time
│   ├── types.py               ← ToolKind enum (local / mcp / a2a)
│   ├── utils.py               ← retry, transient classification, text extraction
│   ├── async_bridge.py        ← one event loop on a daemon thread
│   └── services/
│       ├── session_memory/
│       ├── agent_orchestrator/
│       └── tools_integration/
├── skills/
│   ├── general/SKILL.md       ← the catch-all department
│   ├── hr/SKILL.md            ← employee records, leave, payroll (workday_* MCP tools)
│   ├── marketing/SKILL.md     ← campaigns, copy, market research
│   ├── operations/SKILL.md    ← SOPs, runbooks, approvals
│   ├── research/SKILL.md      ← web research
│   ├── sales/SKILL.md         ← pipeline, accounts, deals
│   └── writer/SKILL.md        ← drafting and content
├── tools/
│   ├── mock_workday_server.py ← mock Workday MCP server for end-to-end runs
│   ├── sample_tool.py         ← the extension template
│   └── text_stats.py          ← dependency-free dynamic-tool reference
├── tests/                     ← 384 tests, plus tests/fixtures/mcp_server.py
└── workspace/                 ← agent file output (two git-tracked samples)
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
START → orchestrator → (conditional) → subagent_fanout → responder → END
                             ↘                            ↗
                             ↘         → responder ─────↗
```

Three nodes, one direction. There is no `tools` node: the orchestrator is a
pure router that returns `{enhanced_query, departments}` and binds no tools, so
there is no top-level ReAct loop and no graph-level retry counter. Tool calls
happen inside a sub-agent's own loop, bounded by that loop's iteration budget.

Nodes are module-level functions, not closures:

- `_local_orchestrator_node(state, config)` — pulls `thread_id` from
  `config["configurable"]`, calls `call_orchestrator`. Declared with
  `config: RunnableConfig | None` rather than a bare `dict` because LangGraph
  type-checks that parameter and warns otherwise. Builds no tool set: it used
  to call `get_all_tools()` and pass the result to a dispatcher that ignored
  it, paying to load the dynamic and MCP tool layers every turn for nothing.
- `_subagent_fanout_node(state)` — builds department specs, runs
  `asyncio.run(engine.invoke_parallel(...))`, then folds each department's
  token spend into `token_usage` and its file writes into `pending_writes` +
  `audit_log`. That accounting used to live in the tools node; with spend and
  writes now happening inside sub-agents, this is the only place the parent's
  totals are kept. Safe to call `asyncio.run` here because LangGraph runs sync
  nodes on a worker thread with no event loop of its own; calling this
  function from inside a running loop would fail.
- `_responder_node(state)` — described in §2.
- `_is_envelope(text)` and `_fallback_answer(state, query)` — the two helpers
  that make an unroutable query produce a sentence rather than raw JSON.

`get_deep_agent()` compiles once and caches in `_compiled_graph`. It compiles
with `checkpointer=get_saver()` — the shared saver, per §4.

`reset_deep_agent()` drops the cached graph, calls `agent_factory.clear_caches()`
(model clients + both MCP caches), rebuilds `SUBAGENTS` from `skills/`, and
nulls `subagents._TOOL_REGISTRY` so the next sub-agent dispatch rebuilds the
shared tool registry. Call it after editing skills, tools, or model config;
otherwise the running process keeps the old graph. Tests call it between cases.

### `orchestrator.py` — the dispatcher (CORE)

`_build_dispatcher_prompt()` is short on purpose. The legacy prompt (in the
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

`route_after_orchestrator`. This file used to hold three more routers and a
`max_iterations` branch that was unreachable. It is now ~40 lines: the
department check, a `_has_tool_calls` guard routing a stray tool-call message
to the responder (see §2 step 2).

### `state.py` — the graph state (CORE)

An `Annotated` `TypedDict`. Two fields use `Annotated[..., add]` so LangGraph
appends instead of replacing: `messages` and `audit_log`. Everything else is
last-write-wins.

Fields, grouped by who writes them:

| Field | Written by | Notes |
|---|---|---|
| `messages` | responder | append-only |
| `current_plan` | nobody | `write_todos` result stays inside the sub-agent; never synced back |
| `workspace_files` | currently unused | was re-synced after every tools-node batch |
| `next_message` | orchestrator | staging slot, cleared by responder |
| `enhanced_query` | orchestrator | rewritten every turn; never carried over |
| `department_targets` | orchestrator | rewritten every turn; never carried over |
| `subagent_results` | fanout | cleared by responder |
| `token_usage` | orchestrator, fanout | sub-agent spend folded in |
| `pending_writes` | fanout | sub-agent file writes, informational |
| `audit_log` | fanout | append-only |
| `routing_decisions` | currently unused | kept for future routing history |
| `review_verdict` | currently unused | legacy from the deleted reviewer pass |
| `thread_id` | runtime config | mirrors `config["configurable"]["thread_id"]` |

`iteration_count` and `max_iterations` are **not** in this table because they
are no longer in the schema. They belonged to the top-level ReAct dispatcher
loop that the pure-router hierarchy removed; see §12.

`recursion_depth` and `consecutive_invalid_tools` were removed with the tools
node: there is no sub-agent nesting left to bound, and no graph-level loop for
an unknown-tool counter to cut short.

The last three are legacy surface. They are cheap to keep and would be needed if
critic or plan-checker passes come back.

### `subagent_engine.py` — parallel department dispatch (CORE)

`SubAgentEngine` has no constructor and no registry. It batches and nothing
else.

`invoke_parallel(subagents, enhanced_query)` → `dict[str, SubagentRun]`:
- Batches by `get_max_parallel_tasks()` (default 4). Batches run in order;
  within a batch, `asyncio.gather` runs them concurrently.
- Each sub-agent gets the same wall-clock deadline.
- `_run_subagent` wraps the body in `asyncio.wait_for`, turning `TimeoutError`
  and any other exception into an error *string* for that sub-agent, so one
  failure never drops its siblings' results.
- Departments marked `parallelizable: false` in their SKILL.md run after the
  parallel group, one batch at a time. The deleted tools node used to make this
  split; honouring it here is what keeps the frontmatter flag from being
  silently ignored.
- The engine selects no tools. It holds no registry and takes no `tool_defs`;
  `subagents.select_department_tools` resolves each department's SKILL.md
  allowlist. Building a `ToolRegistry` in the constructor to do so loaded every
  MCP tool on every dispatch for a value that was only ever read back out.

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
answers.

### `agent_factory.py` — model clients, tracer, cache hooks (CORE)

Three concerns, all of them infrastructure:

**1. `DeepAgentTracer`** — a real `BaseCallbackHandler` recording
`on_chat_model_start/end`, `on_llm_start/end`, `on_tool_start/end`,
`on_chain_start/end` into a list. Only constructed when `OBSERVABILITY=1`.
Events are inspectable via `get_events()` and clearable via `clear()`.

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

**3. Cache hooks** — `get_deep_agent()` here is a wrapper that ensures
`./workspace` and `./skills` exist, then delegates to `graph.get_deep_agent()`.
The bootstrap is kept because the Streamlit app and the write tools depend on
those directories existing. `clear_caches()` drops the model clients and MCP
registries; `reset_deep_agent()` clears the compiled graph too and rebuilds
`SUBAGENTS` from `skills/`.

This file used to own `local_tools_node`, the top-level ReAct node where the
orchestrator called tools itself and delegated via the `task` tool. Both are
gone with the pure-router hierarchy, which left only the concerns above. It
also still declares a vestigial `_compiled_graph = None` global that nothing
ever populates — `graph.py` owns the real cache — plus its own
`reset_deep_agent` that re-exports graph's and additionally nulls the
sub-agent tool registry. `main.py` imports the *factory's* `get_deep_agent`
(the workspace/skills bootstrap wrapper), while `app.py` and `agent.py` import
`graph.get_deep_agent` via the shim.

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

### `memory.py` — skills, tools, workspace, and AGENTS.md

`_dir_tree_hash` is imported from `tools_integration.discovery` — it is the
one copy of that digest, used to invalidate the skills summary cache. (This
module used to carry a second, subtly different version that did not sort
`files`, so the same directory could hash two ways.)

- `get_skill_info(path)` / `get_skill_body(name)` — parse a `SKILL.md`
  front-matter block and return metadata or the body with the frontmatter
  stripped (rejoining on `---`, so a `---` line inside the body survives).
  Both route through `validate_read_path`.
- `get_memory_content()` — read `./AGENTS.md` if present. Appended to every
  sub-agent's role prompt as well.

The legacy long planner prompt (`get_system_prompt`) is **gone** — it died
with the old plan.py orchestrator. What remains is read-path helpers only.


### `subagents.py` — the sub-agent registry (CORE)

`SubagentSpec` fields: `description`, `kind`, `parallelizable`, `aliases`,
`skill`, `tools`, `url`. The `kind` value decides everything downstream:

| `kind` | Behavior | Entry point |
|---|---|---|
| `tool_loop` | restricted tools + `SKILL.md` prompt + a real ReAct loop | `run_tool_loop` |
| `a2a` | HTTP call to a remote agent | `a2a_client.call_a2a_agent` |

There is no `graph` kind. A `general-purpose` spec used to run the whole
compiled graph as a "sub-agent", which made the orchestrator its own
sub-agent — a delegated description got routed again from the top, and the
hierarchy stopped being one-way. The catch-all is `skills/general/SKILL.md`, a
department like any other, chosen by the router rather than by a
self-referential spec.

`SUBAGENTS` is assembled by `build_subagent_registry()` from two sources:

1. **`skills/*/SKILL.md`** via `discovery.discover_subagents` — each becomes a
   `tool_loop` spec whose `skill` names the SKILL.md (the prompt source) and
   whose `tools` come from the `allowed-tools` frontmatter. This is what makes
   `skills/` genuinely data-driven; adding a department is adding a directory.
2. **`_a2a_specs()`** — remote agents from the `A2A_AGENTS` env var.

`refresh_subagents()` rebuilds it; `reset_deep_agent()` calls it.

`list_departments()` is the subset that is *routable* — `tool_loop` specs backed
by a SKILL.md, plus `a2a` specs. A remote agent is a sub-agent that happens to
live behind a network hop, so the router treats it like any other department;
it has to, since the `task` tool that used to be the only dispatcher for them
is gone. A `tool_loop` spec with no skill is still excluded: it has no prompt
to run a loop with.

**Tool selection** (`select_department_tools`) is three stages, matching the
documented contract: the SKILL.md `allowed-tools` allowlist is the hard limit,
capped at 20 visible, then `relevance.sort_tools` narrows to 3-5 for the
current query. Names that do not resolve are dropped with a warning — a stale
name in a SKILL.md must not become a tool the model can call but never execute.
A trailing `*` claims a whole `server*` namespace, which is how a department
claims its MCP tools without naming each one in static frontmatter. The result
is **real `BaseTool` objects**, not metadata dicts; resolving names back to
callables is what makes the shortlist bindable at all. A failing sorter falls back to the capped
allowlist. The sub-agent does not pick its own tools — this function is the
single filter. (If the allowlist is 5 tools or fewer the shortlist step is
skipped entirely — there is nothing to rank.)

**Budget exhaustion is reported, not swallowed.** When `run_tool_loop` runs out
of turns, the last model message is usually another tool call whose content is
`""`. Returning that hands the parent an empty string, which reads as "the
subagent found nothing" and leaves it to redo the whole task. The loop instead
reports the truncation, the tools it called, and the files it wrote.

`SUBAGENTS` is the single source of truth. It drives the dispatcher's roster,
the executor choice in `subagent_engine`, the parallel-vs-sequential split, and
role prompts. Adding a sub-agent is adding a directory or a config entry.

The module calls `load_dotenv()` itself. This is load-order surgery: the roster
the router sees is built statically at import time from `SUBAGENTS`, and the
import chain runs before `agent_factory`'s `load_dotenv()`. The cost is that
changing `A2A_AGENTS` needs a process restart.

`COMPLETION_CONTRACT` is the shared closing instruction for every tool-loop
sub-agent: save to the exact path named in the task, never `/tmp/`, never a
shared default filename (parallel sub-agents would overwrite each other), and
end with a short summary of what was written.

`build_role_prompt(spec)` = contract + the spec's `SKILL.md` body + `AGENTS.md`.

`run_tool_loop(prompt, description, tools, max_iterations)` is the real ReAct
loop: bind tools, invoke, execute any tool calls, append results, repeat until
the model answers without tool calls or the budget runs out. Returns
`(final_text, usage, write_ops)`. Two details inside it:

- **Every tool call dispatches through `ToolRegistry.dispatch`** via a small
  `_ToolDispatcher` adapter (it exists because `invoke_with_retry` expects an
  object with `.invoke(payload)`). The bound `tool_map` decides what the
  sub-agent *may* call; the registry validates the payload and picks the
  transport. The registry itself is built once per process
  (`_registry_for`, module-level `_TOOL_REGISTRY`) so dynamic and MCP tools
  are not reloaded per sub-agent per turn; a tool the registry has never seen
  is registered on the spot, so a bound tool is never silently uncallable.
  This is the only tool-call site in the repo.
- **Budget exhaustion is reported, not swallowed.** When the loop runs out of
  turns, the last model message is usually another tool call whose content is
  `""`. Returning that hands the parent an empty string, which reads as "the
  subagent found nothing" and leaves it to redo the whole task. The loop
  instead reports the truncation, the tools it called, and the files it wrote.

**This loop is the only place in the system with an iteration budget.**
`run_department` passes `AGENT_MAX_ITERATIONS` (default 25) when the caller does
not override it, so that setting bounds a sub-agent's tool turns and nothing
else. The orchestrator has no budget because it does not iterate.

### `tools_integration/guardrails.py` — filesystem containment (CORE for safety)

Every file tool routes through here. It lives in `tools_integration` because
that is the layer the file tools execute in; `agent_orchestrator` imports
down from it, never the reverse. (`agent_orchestrator.memory` re-exports
`get_workspace_files`/`get_workspace_root` for its remaining importers.)

- `get_workspace_root()` — the `WORKSPACE_ROOT` (default `./workspace`) path,
  resolved, read at call time.
- `validate_and_normalize_path(path, must_be_in_workspace=False, strict=False)`
  — normalize, then confirm the resolved target *and every parent component*
  stays under its authorized root, so a symlink anywhere in the chain cannot
  move the target out. An existing target must be a regular file.
  `must_be_in_workspace=True` (the write path) allows `workspace/` as a
  logical alias even when `WORKSPACE_ROOT` is set elsewhere; out-of-root
  absolute/traversal writes fall back to the safe basename rather than
  erroring, unless `strict=True` rejects them outright. Reads must use
  `validate_read_path`, not this helper.
- `validate_read_path(path)` — read is wider than write: `AGENTS.md` (only if
  not itself a symlink), plus anything under `./workspace` or `./skills`.
  Root selection precedes resolution, so a workspace symlink cannot grant
  access to skills or repo files.
- `get_workspace_files()` — workspace listing relative to the configured root;
  walks without following directory symlinks and re-validates each hit.
- `clear_workspace()` — used by the Streamlit sidebar reset button. Unlinks
  top-level files and links only; directories remain.

The invariant: **reads** may touch `AGENTS.md`, `workspace/`, `skills/`;
**writes** may only touch `workspace/`. Known caveat, stated in the code: this
is resolved-path containment, not protection against adversarial filesystem
mutation between check and I/O (TOCTOU). Fine for a local agent, not fine for
a multi-tenant service.

### Deleted files, still referenced in git history

- `plan.py` — the old planner. Superseded by `orchestrator.py`; the docstrings
  in `orchestrator.py` and `memory.py` compare against it.
- `review.py` — critic / plan_checker / reflection passes. Removed; the
  `review_verdict` state field and the tracer's role names are their residue.
- `tools_integration/rag.py` — the old TF-IDF retrieval layer. Replaced by
  `fetch_url` + `internet_search`.

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
| `write_todos` | records a plan; result stays inside the sub-agent's loop |
| `internet_search` | Tavily; needs `TAVILY_API_KEY`; returns 300-char snippets |
| `read_file` | via `validate_read_path` |
| `write_file` | via `validate_and_normalize_path(must_be_in_workspace=True)` |
| `edit_file` | string replace in a workspace file |
| `list_files` | workspace listing (via `guardrails.get_workspace_files`) |
| `search_files` | substring search across workspace, capped at 30 matches |
| `fetch_url` | full page text via `fetch_public_url` |
| `list_tools` | discovery — the model can ask what it can do |

There is no `task` tool. It used to live here, which forced three lazy upward
imports from the tool layer into `agent_orchestrator` and made the one-way
hierarchy unenforceable. It moved to `agent_orchestrator.task_tool` and has
since been deleted along with the sub-agent-to-sub-agent delegation it served.

Note the pattern in every file tool: validate, then return a **string** on
error. Tools return strings here, not raise, so one failed call does not kill
the loop.

Dispatching to a department is not this layer's job. `subagent_engine` picks an
executor from the spec's `kind`: `tool_loop` runs `run_department` locally,
`a2a` calls `call_a2a_agent` with the SDK imported lazily so it stays off the
import path until used. A remote agent reports no local token usage and writes
no local files, so its `SubagentRun` carries neither — by design, not an
oversight.

`FETCH_URL_TIMEOUT_SECONDS = 15` and `FETCH_URL_MAX_BYTES = 1_000_000` sit next
to `fetch_url` on purpose: the size cap is what keeps a 200 MB response from
landing in the model's context.

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
`ToolRegistry`, registering the `BaseTool` object itself — not `tool.func`.
The object carries the `args_schema` that validates a payload and the `kind`
that selects a transport, and both are lost if you keep only the function.
Configured `A2A_AGENTS` are registered too, as `kind="a2a"` tools.

### `validation.py` — the payload contract (CORE)

Every tool call carries a payload the model produced, and it has to match the
tool's parameter spec or the call is wrong. This module is where that is
enforced.

- `normalize_schema(schema)` — turns a raw JSON Schema **dict** into a Pydantic
  model. This exists because **langchain-core does not validate a `dict`
  `args_schema` at all** (`_parse_input` returns the input unchanged), and MCP
  describes every one of its tools with exactly that. A `@tool` built-in
  already has a model, so it passes through untouched. Handles `required`,
  nested objects, arrays, and enums; degrades to returning the input unchanged
  for a schema it cannot model, so a broken schema stays callable.
- `validate_args(schema, args)` — raises `ValueError` naming the offending
  field. **Rejection, not coercion:** a payload the model did not intend is
  never quietly repaired.
- `prune_unset(validated)` — drops `None`s that exist only because a field has
  a default. langchain passes along every field that has one, so a normalized
  schema would otherwise turn an omitted optional into an explicit `null` at the
  server.
- `ValidatedTool` — mixin validating the *raw* input. Needed because
  `_to_args_and_kwargs` short-circuits ("StructuredTool with no args") for a
  fieldless model and never parses, so a no-arg tool would otherwise discard
  whatever it was handed instead of refusing it.

### `registry.py` — selection, dispatch, risk (CORE)

`ToolRegistry` holds three dicts: specs, callables, and the registered tool
objects (`_tools`, which is what dispatch needs).

- `register_builtin(name, callable_, risk_level="low", ...)` — the path used
  for built-ins; description comes from the tool or its docstring.
- `select_for_query(allowed, query, max_tools)` — the registry's half of the
  department filter: resolves names to real tools, drops unknown names with a
  warning, and narrows to `max_tools` by relevance. `subagents.select_department_tools`
  calls it and remains the single place a department's tools are decided; it
  owns the allowlist, `prefix*` namespace matching, and the 20-tool cap.
  Two fallbacks are deliberate: a sorter that *reached* but named nothing
  usable yields a bounded slice, while a sorter that *could not be reached*
  returns the full allowlist — shrinking it would look like a relevance
  decision that never happened.
- `kind_of(name)` — the transport, read off the tool object first (an MCP tool
  is constructed with `kind="mcp"`), falling back to the spec. An unrecognized
  kind is logged and treated as local.
- `dispatch(name, args, subagent_name="")` — **the single execution entry
  point.** Validates, checks approval and role, then routes by kind: `local`
  and `mcp` both end at the tool's own `invoke` (an MCP tool is already a
  sync-wrapped object by then), `a2a` serializes the validated args to JSON in
  the message body, since A2A carries text rather than a structured payload.
  Validation happens *before* the transport is chosen: a malformed payload is
  the same mistake whichever protocol would have carried it.
- `RequiresApprovalError` — raised when a tool declares `requires_approval`
  and no approval was given.

There is no `get_visible_tools(subagent_name)`. It resolved a department's
tools by role behind a 20-tool cap, and nothing ever bound the result — a
second, weaker filter running beside
`subagents.select_department_tools`, which resolves the SKILL.md allowlist
instead. Deleting it leaves the allowlist the one place a sub-agent's tools are
decided.

`ToolKind` (in `src/types.py`) is the discriminator — `local` / `mcp` / `a2a`.
There is deliberately **no `http` member**: no such tool type exists, and
adding one is a genuine SSRF surface needing scheme checks, private-IP
rejection, and reuse of `research_fetch`'s guardrails.

### The risk-tier wrapper that was here

`executor.py` (`ToolExecutor`, risk tiers low/medium/high) was deleted in the
dead-code cleanup: it had no live-path caller, and its `execute` was an alias
of `registry.dispatch`. The *data* it served stays — `requires_approval` and
`allowed_roles` remain on `ToolSpecMetadata`, and `dispatch` still enforces
them — but no tool declares either, so no call is gated today. Which tools
*should* gate is a decision, not a refactor.

### `decorator.py` — `@tool_spec` (CORE)

Attaches `ToolSpecMetadata` (name, description, `risk_level`, `requires_approval`,
`allowed_roles`, `kind`) as a `__tool_spec__` attribute. It does not change the
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

Each candidate is sent with its **parameter spec** as well as its name and
description, under a `"parameters"` key. It is choosing from up to 20 tools, and
the sorter cannot tell two same-named tools apart without knowing what each one
accepts. `_render_schema` converts a Pydantic `args_schema` back to JSON Schema
and truncates at `MAX_SCHEMA_CHARS = 400` so one verbose spec cannot crowd the
rest of the roster out; the rendered form is prompt-only, and the returned
entries are still the caller's own dicts.

### `research_fetch.py` — SSRF-safe URL fetch (CORE for safety)

`fetch_public_url(url, timeout, max_bytes)`:
- resolve the hostname, and **pin** the resolved IP so the connection cannot be
  redirected to an internal address between check and connect,
- reject non-public addresses (`_public_address`),
- allow at most `MAX_REDIRECTS = 5`,
- no proxies,
- cap the response at `max_bytes` and the whole call at `timeout`.

This is async. The `fetch_url` tool calls it through `async_bridge.run_sync`.

### `mcp_client.py` — MCP server tools

`load_mcp_tools()` is sync, driven through `async_bridge.run_sync`. Servers come
from `config.get_mcp_servers()`. Nothing runs until `MCP_SERVERS` is set.
(A second, async-native `mcp_bridge.py` implementation was deleted in the
dead-code cleanup: it survived only via a package re-export, and its separate
cache was a guaranteed no-op in production.)

`mcp_client` wraps each MCP tool as a sync `StructuredTool`, because
`langchain-mcp-adapters` builds tools with `coroutine=` and no `func` — a plain
`.invoke()` raises `NotImplementedError`. Server names prefix every tool
(`<server>_<tool>`) so two servers cannot silently collide. Cached by the
serialized server config; `clear_mcp_tools_cache()` invalidates.
`agent_factory.clear_caches()` clears both copies.

Three things happen in the wrapper that are easy to miss:

- **The schema is normalized, not carried over.** MCP supplies a raw JSON Schema
  dict, and langchain-core skips validation entirely for a dict `args_schema` —
  `{"employee_id": 12345}` would go straight to the server. `normalize_schema`
  turns the dict into a Pydantic model so an MCP tool holds the same payload
  contract as a built-in. `MCPTool` also validates the *raw* input in `run`,
  because langchain-core short-circuits parsing for a fieldless model and would
  otherwise discard a no-arg tool's arguments without looking at them.
- **Omitted optionals are pruned.** langchain-core injects a default for every
  field that has one, so absent optionals arrive as explicit `null`;
  `prune_unset` strips them before the call so the server receives the payload
  that was actually asked for.
- **Results are unwrapped.** A real server returns content blocks
  (`[{"type": "text", ...}]`); `_unwrap_mcp_result` joins the text and names any
  non-text block rather than dropping it. `response_format` is deliberately
  *not* set — the `func` already returns plain text, and declaring
  `content_and_artifact` would unpack it a second time.


### `a2a_client.py` — remote agents over A2A

Fetch the agent card from the base URL, send one text message, then poll.
(An async-native `a2a_bridge.py` duplicate was deleted with the cleanup.)
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

### `src/types.py`

`ToolKind` — the StrEnum discriminating tool transports (`local` / `mcp` /
`a2a`, deliberately no `http`). Consumed by `ToolRegistry.register` and the
dispatch routing.

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

### `main.py` — CLI (151 lines)

More than a REPL now. `uv run python main.py` mints `cli-<uuid>` once and
loops: read input, send only the new message, log every intermediate message
(tool calls and results included), print the last `AIMessage` and total
tokens. Flags:

- a bare argument list runs **one-shot** — `uv run python main.py "count the
  words"` runs a single query and exits;
- `--tools` prints the tool registry (name, risk, roles) and exits;
- `--debug` turns on verbose per-message logging.

Logging goes to stdout *and* `deep_agent.log` (best-effort — an OSError falls
back to stdout only). `httpx`/`httpcore`/`urllib3` are silenced to WARNING so
third-party HTTP chatter does not drown the turn log.

### `app.py` — Streamlit UI (356 lines)

Calls `agent.stream(..., stream_mode="updates")` and renders each node's state
delta live, which is why you can watch the graph work.

Sidebar panels: governance (token usage, current node), active skills (from
`get_skill_info`), audit log, system status, shared memory (`AGENTS.md`), and a
workspace file browser. `clear_workspace()` is wired to a reset button.

State lives in `st.session_state`: `messages`, `current_plan`,
`workspace_files`, `audit_log`, `token_usage`, `last_action`, `current_node`,
and one `thread_id` minted at startup.

The one rule to copy if you build another UI: send only the new message. The
checkpointer holds the rest, under `thread_id`.

---

## 9. Content files and the not-in-use code

### `skills/` — seven departments, all data-driven

`general`, `hr`, `marketing`, `operations`, `research`, `sales`, `writer` —
each a directory with one `SKILL.md`. Same shape in every file: YAML
frontmatter (`name`, `description`, `allowed-tools`, `parallelizable`) then a
markdown body that becomes the sub-agent's system prompt.

The notable ones:

- **`general`** is the router's fallback destination and is marked
  `parallelizable: false` — it runs alone, after the parallel group. Its body
  branches on turn kind: greeting/small talk (no tools), off-topic questions
  (polite refusal), then real work.
- **`hr`** shows the MCP namespace claim: its allowlist is
  `workday_*, read_file, write_file, ...` — the `workday_*` prefix claims
  every tool of a configured `workday` MCP server without naming them in
  static frontmatter.
- **`research` / `writer`** are the original pair: search → `fetch_url` →
  workspace notes, and outline → tone-matched draft with a named completion
  contract.

The `description` frontmatter is the only signal the router has for choosing
a department, so it is the part worth writing well. Adding a department is
adding a directory.

### `tools/sample_tool.py` and `tools/text_stats.py`

`sample_tool.py` is 19 lines: `get_current_time`, decorated `@tool` over
`@tool_spec` with `risk_level="low"`. `text_stats.py` is the fuller reference
— dependency-free, documented for the model (the docstring *is* the tool
schema), and noting that the name must not collide with a built-in since
dynamic tools lose the precedence fight. Copy either, change the function,
and it is discovered on the next `get_all_tools()`. Note `sample_tool` imports
`pytz` inside the function body, so a missing optional dependency does not
break module load for every other tool.

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

### `README.md`

The user-facing setup guide.

### `workspace/` — git-tracked sample output

Two markdown files (`multi_agent_research.md`,
`research_multi_agent_systems.md`), kept as examples of real agent output.
`CLAUDE.md` says memory files are `SKILL.md` only and there is no per-session
`AGENTS.md`; `workspace/` is scratch output, not memory. A test or the
Streamlit reset button can clear it — the reset button deletes the contents,
not the directory. Scratch files and runtime logs are gitignored; the Streamlit
app logs to stdout only, and the CLI writes `deep_agent.log` (gitignored).

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
| `AGENT_MAX_ITERATIONS` | `subagents.run_department` | per-sub-agent ReAct turn budget |
| `MAX_PARALLEL_TASKS` | `subagent_engine` | concurrency cap |
| `SUBAGENT_TIMEOUT_SECONDS` | `subagent_engine` | wall-clock guard, shared per batch |
| `MCP_SERVERS` | `mcp_client` | JSON object of server connections |
| `A2A_AGENTS` | `subagents` | JSON object; **needs a process restart** |

`load_dotenv()` is called in `agent_factory` and again in `subagents`. Copy that
second call if you add a module that must see `.env` at import time.

---

## 11. Tests

384 tests, `testpaths=["tests"]`, `pythonpath=["."]`.

Layout mirrors `src/`:

```
tests/
├── fake_models.py                       ScriptedChatModel
├── fixtures/mcp_server.py               a real MCP server for the live test
├── test_a2a.py  test_async_bridge.py  test_config.py
├── test_delegation.py  test_guardrails.py  test_mcp.py  test_mcp_live.py
├── test_observability.py  test_research_fetch.py  test_state.py
├── test_subagents.py  test_tools.py  test_utils.py  test_validation.py
└── services/
    ├── agent_orchestrator/   test_aggregator, test_graph, test_orchestrator,
    │                         test_routing, test_subagent_engine, test_verification
    ├── session_memory/       test_compression, test_window
    └── tools_integration/    test_discovery, test_registry, test_relevance
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

Two suites sit slightly apart: `test_validation.py` (the payload-contract
module and the MCP wrapper's validation behavior) and `test_mcp_live.py`
(spins up `tests/fixtures/mcp_server.py` and exercises a real MCP round trip
— everything else fakes the transport).

Current baseline: **384 pass, 0 fail.** ruff is clean. mypy reports 9 errors
in 4 files (checkpoint.py's MemorySaver dict-vs-RunnableConfig signatures and
untyped third-party stubs), all pre-existing. Run
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

**Do not put a counter in parent state.** This is the bug that shipped. The
orchestrator used to keep a checkpointed `iteration_count` and force
`departments = []` once it passed a maximum — a guard belonging to the
top-level ReAct dispatcher loop that the pure-router hierarchy had already
removed. Because the counter was checkpointed, it carried across turns on a
reused thread: **turn 2 of every conversation** saw a non-zero count, dropped
its departments, and answered "I could not route this request" while quoting
turn 1's query.

It survived a green suite for two reasons, both worth remembering. Both entry
points passed `iteration_count: 0` on every invoke, compensating at the call
site. And the one multi-turn graph test sent a full initial state each turn,
which reset the counter the same way. A regression test has to reuse the
thread and send *only* the new message, the way the CLI and Streamlit do —
that is `test_delegation.py::TestMultiTurnRouting`.

The general rule: a value that persists in checkpointed state persists across
turns. Per-turn values do not belong there, and a "safety" counter is worse
than none, because it fires on turn two of every conversation and looks like a
routing failure.

**A test that never sends a user message proves nothing.** `tests/.../test_graph.py::_initial_state`
used to drop its argument and build `messages: []`, which is why a real memory
bug survived a fully green suite. When you touch graph tests, check that the
state builder actually uses the message you pass it.

**`A2A_AGENTS` changes need a restart.** `subagents.SUBAGENTS` is a module-level
dict assembled at import time from the configured agents plus `skills/`.

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
