# LLM Observability Implementation Plan

**Target:** deep-agents-langchain-blog repo
**Status:** Proposed (not yet implemented)
**Builds on:** existing `DeepAgentTracer` in `src/services/agent_orchestrator/agent_factory.py:50`

---

## 1. Goals

Add LLM observability to the deep-agent framework so that every turn produces a traceable, attributable, costed record of what happened across the `orchestrator → sub-agent → tool` boundary.

Specifically:

1. **Trace ID per turn** — every user turn gets one trace ID that tags every LLM call, tool call, and chain step from orchestrator through sub-agents to tools.
2. **Span hierarchy** — parent/child relationships so you can see which sub-agent made which tool call, and which tool call produced which model response.
3. **Structured event export** — JSON serialization of all events, writable to a file or streamable, so the trace can be inspected or shipped to an external backend.
4. **Cost tracking** — per-turn, per-department, and per-tool cost estimates using token usage already collected (`token_usage`, `SubagentRun.usage`).
5. **Per-department context** — which department ran, which tools were shortlisted (and relevance scores), SKILL.md source, success/failure, latency.
6. **Zero architectural violations** — no iteration counter in parent state, no reverse dependency (`tools_integration` never reaches into `agent_orchestrator`), no top-level ReAct loop, no `recursion_depth` in state.

Non-goals (for this phase):
- A hosted dashboard / UI (the Streamlit app already renders `audit_log`; observability export can feed it later).
- Automatic OTLP export to a specific vendor (the export surface is JSON; OTLP can be added later as an optional exporter).
- Real-time streaming of traces.

---

## 2. Current state (what exists)

| Component | Location | What it does |
|---|---|---|
| `DeepAgentTracer` | `agent_factory.py:50` | `BaseCallbackHandler` recording `chat_model/llm/tool/chain` start/end into an in-memory list. |
| `_tracer` global | `agent_factory.py:107` | Built only when `OBSERVABILITY=1`. |
| `_maybe_attach_callbacks` | `agent_factory.py:110` | Wraps model with `with_config(callbacks=[_tracer])`. |
| `token_usage` state | `state.py:10` | `{"input", "output", "total"}` aggregated in orchestrator (`orchestrator.py:175`) and fanout (`graph.py:88`). |
| `SubagentRun` | `subagent_engine.py:22` | `text`, `usage`, `writes` per department. |
| `ToolRegistry.dispatch` | `registry.py:217` | Single tool execution surface; validates payload, routes by `ToolKind`. |

**Gaps:** no trace ID, no span parent/child linkage, no per-turn context, no cost, no department/tool attribution, no structured export, no external-backend hook.

---

## 3. Design overview

```
Trace (one per turn, trace_id)
├── Span: orchestrator (LLM call)
│   └── enhanced_query, department_targets, input/output tokens, cost
├── Span: fanout (parallel batch)
│   ├── Span: department "hr" (sub-agent ReAct loop)
│   │   ├── tool call: workday_get_employee (input → output, latency)
│   │   ├── tool call: read_file
│   │   └── final answer, token usage, cost
│   ├── Span: department "writer"
│   │   └── ...
│   └── Span: department "sales" (A2A, remote — reports URL + latency, no local tokens)
└── Span: responder (verification + aggregation)
```

**Mechanism:** LangChain's callback handler already receives `run_id` and `parent_run_id` on every event. The tracer will capture these to build parent-child span relationships. A context variable (`contextvars.ContextVar`) holds the current `trace_id` so every event in the turn is tagged with it, propagating across `asyncio.to_thread` boundaries (sub-agent dispatch) without threading state through graph state.

New module: `src/services/agent_orchestrator/tracing.py`

---

## 4. Component design

### 4.1 `tracing.py` — core trace infrastructure

**`TraceContext` (dataclass)**
- `trace_id: str` — uuid4, one per graph turn
- `turn_id: str` — optional, for correlating multiple turns in a thread
- `started_at: str` (ISO)
- `thread_id: str | None` — from `RunnableConfig`

**`Span` (dataclass)**
- `span_id: str` — uuid4
- `trace_id: str`
- `parent_span_id: str | None`
- `name: str` — e.g. `"orchestrator"`, `"hr"`, `"workday_get_employee"`
- `kind: str` — `"llm"`, `"tool"`, `"chain"`, `"fanout"`, `"responder"`
- `started_at`, `ended_at: str | None`
- `input_tokens`, `output_tokens: int`
- `cost_usd: float`
- `attributes: dict` — department, tool name, kind, query, error, etc.
- `status: str` — `"running"`, `"completed"`, `"error"`

**Context management**
- `CURRENT_TRACE: ContextVar[TraceContext | None]` — the active trace for the current turn
- `CURRENT_SPAN: ContextVar[Span | None]` — the active span stack (for parent linkage)
- `start_trace(thread_id) -> TraceContext` — generates trace ID, pushes context
- `end_trace()` — pops context
- `start_span(name, kind, attributes) -> Span` — creates span, links to current span as parent, pushes
- `end_span(span, status, ...) -> None` — pops, records end time

**Why contextvars, not state:** The constraint forbids per-turn counters in checkpointed state. A trace ID is not a counter, but putting it in state would require every node to read/write it and every entry point to initialize it. Contextvars flow implicitly through the call stack and across `asyncio.to_thread` (which copies context by default in Python 3.11+), so nodes don't need to pass it explicitly.

**A2A/MCP boundary:** For remote calls, the trace ID can be included in the A2A message body or MCP metadata as an optional header so the remote side can correlate. Local side still records the span.

**`TraceStore` (dataclass / class)**
- Holds all spans and events for the current trace
- `add_event(event_dict)`
- `add_span(span)`
- `get_trace() -> dict` — full serializable trace
- `export_json(path) -> None`
- `reset()`

### 4.2 Enhanced `DeepAgentTracer` (`agent_factory.py`)

Keep the existing class (backward compatible) but extend it to:

- Capture `run_id`, `parent_run_id` from callback kwargs (they're passed in `**kwargs`)
- Tag every event with `trace_id` from `CURRENT_TRACE`
- Map langchain event types to spans:
  - `on_chat_model_start` / `on_llm_start` → start `"llm"` span, record `run_id`
  - `on_chat_model_end` / `on_llm_end` → end `"llm"` span, extract `usage_metadata` for tokens/cost
  - `on_tool_start` → start `"tool"` span, parent = current LLM span's run_id
  - `on_tool_end` → end `"tool"` span, record output length
  - `on_chain_start` / `on_chain_end` → `"chain"` spans (graph node steps)
- Store events + spans in `TraceStore`

### 4.3 Cost estimation (`tracing.py` or `config.py`)

Pricing table (per 1K tokens) keyed by provider/model. Read from env where available, defaults for common models:

| Provider | Model | input /1K | output /1K |
|---|---|---|---|
| openrouter | `OPENROUTER_MODEL` (env-configurable) | `$OPENROUTER_INPUT_PRICE` | `$OPENROUTER_OUTPUT_PRICE` |
| anthropic | claude-3-5-sonnet | $3.00 | $15.00 |
| openai | gpt-4o | $2.50 | $10.00 |
| google | gemini-2.0-flash | $0.15 | $0.60 |
| ollama | local | $0 | $0 |

Default prices fall back to $0 if unknown (never fail observability because of a missing price).

Functions:
- `estimate_cost(provider, model, input_tokens, output_tokens) -> float`
- Env overrides: `OBSERVABILITY_PRICE_<PROVIDER>_<MODEL>_INPUT`, etc.

### 4.4 Integration points

#### (a) Graph entry — `graph.py` `_local_orchestrator_node`

Generate the trace ID at the start of the turn. Since this node runs once per turn (acyclic graph), it's the natural trace boundary:

```python
def _local_orchestrator_node(state, config=None):
    thread_id = ...
    trace_ctx = start_trace(thread_id)   # sets CURRENT_TRACE
    try:
        return call_orchestrator(state, model=get_model(), thread_id=thread_id)
    finally:
        # trace stays open through fanout/responder; ended in responder node
        pass
```

Wait — the trace must span orchestrator → fanout → responder, all in one `invoke`. The graph runs synchronously within `agent.invoke()`, so the trace context set in the orchestrator node is still current when fanout and responder run (same thread, same call stack via `asyncio.run` inside the node). The trace should be ended in the responder node (the last node before END).

Alternative: wrap the whole `get_deep_agent().invoke(...)` call. But the tracer is attached at the model level, not the graph level, and we want the trace to start at the orchestrator's first LLM call. Starting it in the orchestrator node and ending it in the responder node keeps the boundary inside the graph where it belongs.

Actually, cleaner: start the trace in `get_deep_agent()` wrapper? No — `get_deep_agent()` just compiles the graph. The trace belongs per-invoke, not per-compile.

Best approach: start trace in orchestrator node, end trace in responder node, export in responder node (or after invoke returns). This keeps everything inside the graph.

#### (b) Fanout node — `graph.py` `_subagent_fanout_node`

Already folds `token_usage` and `writes` into state. Add:
- Start a `"fanout"` span
- For each department run, the sub-agent creates its own span (see below)
- Record per-department: name, kind (tool_loop/a2a), parallelizable, tools shortlisted, success/failure, latency
- Write span events to `TraceStore`
- If export-to-file is enabled, write the trace to disk here or in responder

#### (c) Sub-agent executor — `subagents.py` `run_department` / `run_tool_loop`

`run_department(spec, query, ...)`:
- Start a `"department"` span with attributes: department name, skill path, tools allowed (count), tools shortlisted (count + names), kind
- Call `run_tool_loop(...)`
- End span with: token usage, cost, write ops, success/error, final text length

`run_tool_loop(...)`:
- The ReAct loop. Each model call is already traced by `DeepAgentTracer` (via `get_model()` → `_maybe_attach_callbacks`). 
- Add: department name to the span attributes so tool spans know which department they belong to
- The existing `DeepAgentTracer` captures tool start/end via `on_tool_start/end` — these become child spans of the current LLM span

#### (d) Tool dispatch — `tools_integration/registry.py` `dispatch`

Optionally add a span around each `dispatch` call. The tracer already captures `on_tool_start/end` at the langchain level. The registry level is where we know `subagent_name` (for role gating) and can record `risk_level`, `requires_approval`, `kind`. This is optional enrichment; the langchain-level tool events may suffice for phase 1.

#### (e) Responder node — `graph.py` `_responder_node`

- End the trace
- Assemble the full trace dict from `TraceStore`
- If `OBSERVABILITY_EXPORT_PATH` env is set, write JSON there
- The trace dict can also be folded into state (e.g., `state["trace"]`) — but that adds a state field. Alternative: return it via a side channel. Since the responder returns `{"messages": [...], "subagent_results": {}, "next_message": None}`, we could add `"trace"` to the updates. Let me check — `AgentState` doesn't have a `trace` field. Adding one is fine (it's not a counter, it's a per-turn result). But to keep state minimal, we can export to file and/or expose via `TraceStore.get_trace()` for the caller to retrieve.

Actually, the cleanest: export to file in responder if `OBSERVABILITY_EXPORT_PATH` is set, and keep `TraceStore` as a module-global the caller can query. The caller (CLI/Streamlit) can call `get_trace()` after `invoke()`.

Wait, but `TraceStore` needs to be per-trace. If it's module-global, concurrent turns could overwrite each other. Since the graph runs synchronously per invoke (one thread at a time per thread_id, but multiple threads could invoke concurrently), we need per-trace isolation. Contextvars handle this: `CURRENT_TRACE` is a contextvar, and `TraceStore` can be stored on the `TraceContext` object itself. So `trace_ctx.store` holds the spans/events for that trace.

Let me revise: `TraceContext` holds a `TraceStore`. `CURRENT_TRACE: ContextVar[TraceContext | None]`. The tracer accesses `CURRENT_TRACE.get().store` to record events.

#### (f) Streamlit UI (`app.py`)

Read the exported trace file or query `TraceStore` and display it in the existing audit-log / skills view. This is optional for phase 1 but worth noting.

### 4.5 Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `OBSERVABILITY` | off | `=1` attaches `DeepAgentTracer` (existing) |
| `OBSERVABILITY_EXPORT_PATH` | — | If set, write trace JSON to this path after each turn |
| `OBSERVABILITY_EXPORT_FORMAT` | `json` | `json` or `jsonl` |
| `OBSERVABILITY_PRICE_<PROVIDER>_<MODEL>_INPUT` | — | Override input price per 1K tokens |
| `OBSERVABILITY_PRICE_<PROVIDER>_<MODEL>_OUTPUT` | — | Override output price per 1K tokens |

All read at call time from env (consistent with `src/config.py` pattern). No config file.

### 4.6 What NOT to do (constraints from CLAUDE.md)

- ❌ No `iteration_count` / `max_iterations` in `AgentState`
- ❌ No top-level ReAct loop or `tools` node
- ❌ No reverse dependency: `tools_integration` must not import from `agent_orchestrator`
- ❌ No per-turn counter that entry points must reset
- ❌ No sub-agent-to-sub-agent delegation (`task` tool is gone)
- ✅ Trace ID and spans are not counters — they're identifiers and can use contextvars / state

---

## 5. File changes

### New file: `src/services/agent_orchestrator/tracing.py`

```python
"""LLM observability: trace context, spans, cost, export."""

import json
import os
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_span_id: str | None
    name: str
    kind: str  # "llm" | "tool" | "chain" | "fanout" | "department" | "responder"
    started_at: str
    ended_at: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    attributes: dict = field(default_factory=dict)
    status: str = "running"  # "running" | "completed" | "error"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TraceContext:
    trace_id: str
    thread_id: str | None
    started_at: str
    store: "TraceStore" = field(default_factory=TraceStore)


class TraceStore:
    def __init__(self):
        self.spans: list[dict] = []
        self.events: list[dict] = []

    def add_span(self, span: Span):
        self.spans.append(span.to_dict())

    def add_event(self, event: dict):
        self.events.append(event)

    def to_dict(self) -> dict:
        return {"spans": self.spans, "events": self.events}

    def clear(self):
        self.spans.clear()
        self.events.clear()


CURRENT_TRACE: ContextVar[TraceContext | None] = ContextVar("current_trace", default=None)


def start_trace(thread_id: str | None = None) -> TraceContext:
    ctx = TraceContext(
        trace_id=str(uuid.uuid4()),
        thread_id=thread_id,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    token = CURRENT_TRACE.set(ctx)
    ctx._token = token  # for reset
    return ctx


def end_trace():
    ctx = CURRENT_TRACE.get()
    if ctx is not None and hasattr(ctx, "_token"):
        CURRENT_TRACE.reset(ctx._token)


def current_trace() -> TraceContext | None:
    return CURRENT_TRACE.get()


# --- cost ---

_PRICE_TABLE: dict[tuple[str, str], tuple[float, float]] = {}  # (provider, model) -> (input, output) per 1K


def _env_price(provider, model, kind):
    """Read OBSERVABILITY_PRICE_<PROVIDER>_<MODEL>_<INPUT|OUTPUT> from env."""
    key = f"OBSERVABILITY_PRICE_{provider.upper()}_{model.upper().replace('-', '_').replace('.', '_')}_{kind}"
    raw = os.getenv(key)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return None


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    key = (provider, model)
    if key in _PRICE_TABLE:
        in_price, out_price = _PRICE_TABLE[key]
    else:
        in_price = _env_price(provider, model, "INPUT") or 0.0
        out_price = _env_price(provider, model, "OUTPUT") or 0.0
        _PRICE_TABLE[key] = (in_price, out_price)
    return (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price


def export_trace(trace_ctx: TraceContext, path: str, fmt: str = "json"):
    data = trace_ctx.store.to_dict()
    data["trace_id"] = trace_ctx.trace_id
    data["thread_id"] = trace_ctx.thread_id
    data["started_at"] = trace_ctx.started_at
    data["ended_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w") as f:
        if fmt == "jsonl":
            for ev in data["events"]:
                f.write(json.dumps(ev) + "\n")
            for sp in data["spans"]:
                f.write(json.dumps(sp) + "\n")
        else:
            json.dump(data, f, indent=2)
```

### Modified: `src/services/agent_orchestrator/agent_factory.py`

Extend `DeepAgentTracer` to:
- Capture `run_id` / `parent_run_id` from callback kwargs
- Tag events with `trace_id` from `current_trace()`
- Map events to spans via `TraceStore`
- Add `get_trace() -> dict` and `export_trace(path)` methods

### Modified: `src/services/agent_orchestrator/graph.py`

- `_local_orchestrator_node`: `start_trace(thread_id)` at entry
- `_subagent_fanout_node`: start `"fanout"` span; per-department span attributes
- `_responder_node`: end trace; export to `OBSERVABILITY_EXPORT_PATH` if set

### Modified: `src/services/agent_orchestrator/subagents.py`

- `run_department`: wrap in department span (name, skill, tools shortlisted, usage, cost)
- `run_tool_loop`: no structural change (tracer handles model/tool events), but ensure department name is available for span attributes

### Modified: `src/services/tools_integration/registry.py`

- `dispatch`: optional span enrichment (risk_level, kind, subagent_name) — can be phase 2

### New tests: `tests/test_observability.py` (extend existing)

- Trace ID generated and stable across a turn
- Events tagged with trace_id
- Span hierarchy (parent/child) captured
- Cost estimation known values
- Export writes valid JSON
- Disabled when `OBSERVABILITY != 1`
- No state fields added (iteration_count etc. absent)

---

## 6. Test plan

Using `tests/fake_models.py::ScriptedChatModel` (deterministic, no API keys) plus monkeypatching:

1. **`test_trace_id_per_turn`** — invoke graph twice, assert two different trace IDs, each tagging all its events.
2. **`test_span_hierarchy`** — orchestrator span is parent of fanout span; fanout span is parent of department spans; LLM span is parent of tool spans. Assert via `parent_span_id` / `run_id` linkage.
3. **`test_cost_estimation`** — `estimate_cost` returns known values for known models; defaults to 0 for unknown; env override works.
4. **`test_export_json`** — `OBSERVABILITY_EXPORT_PATH` set → file written, valid JSON, contains trace_id, spans, events.
5. **`test_disabled_no_events`** — `OBSERVABILITY` unset → no events recorded.
6. **`test_no_state_pollution`** — `AgentState` has no `iteration_count`/`max_iterations`/trace counter fields.

Run with `uv run pytest tests/test_observability.py` — no API keys or network needed.

---

## 7. Rollout phases

**Phase 1 — Core trace & spans** (`tracing.py` + enhanced tracer + graph integration)
- Trace ID per turn via contextvar
- Span capture from langchain callback events (run_id/parent_run_id)
- Department, fanout, responder spans
- Cost estimation stub
- No export yet (in-memory only)

**Phase 2 — Export & cost**
- `export_trace` to JSON/JSONL file via `OBSERVABILITY_EXPORT_PATH`
- Cost per turn/department/tool using token usage already tracked
- Env-based pricing overrides

**Phase 3 — Tool dispatch enrichment + UI**
- `ToolRegistry.dispatch` span attributes (risk, kind, role)
- Streamlit view of trace file
- A2A trace ID propagation (optional header in remote call)

Each phase is independently testable and can be merged without enabling observability by default (`OBSERVABILITY` still gates everything).

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Contextvar not propagating across `asyncio.to_thread` | Python 3.11+ `to_thread` copies context by default; test explicitly |
| Async bridge (MCP) runs on daemon thread — context may not propagate | Record spans on the local side before the remote call; remote correlation via trace ID in payload (future) |
| Observability overhead slows the hot path | Tracer is a no-op when `OBSERVABILITY != 1`; when on, it's just dict appends |
| Multiple concurrent turns share module-global tracer | `CURRENT_TRACE` is a contextvar — each invocation gets its own trace context |
| Adding `trace` to `AgentState` breaks checkpoint schema | Keep trace out of state; export via file or `TraceStore` query after `invoke()` |

---

## 9. Immediate next step

Implement Phase 1: create `tracing.py`, extend `DeepAgentTracer`, wire trace start/end into `graph.py` orchestrator and responder nodes, add tests.
