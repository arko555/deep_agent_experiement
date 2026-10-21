"""Request-scoped observability for the deep agent.

Every metric this module records carries two ids:

- ``request_id`` — one per user message (per graph turn). This is the
  granularity all observability metrics are tagged at: every LLM call,
  department execution, and external tool call that happens while the turn
  is being served records its span against the request's id.
- ``session_id`` — one per conversation (the LangGraph ``thread_id``). A
  session's summary (:func:`get_session_summary`) lists every request id
  that belongs to it, so a whole conversation can be replayed, costed, or
  compared as a unit.

Shared infrastructure, like ``src.config`` and ``src.utils``: the
orchestrator and the tool layer both import it. That is what lets the single
tool execution surface (``ToolRegistry.dispatch``) record external-call
latency without the tool layer importing back up into ``agent_orchestrator``
— the service dependency arrow stays one-way.

External tool calls (the MCP and A2A transports) record **latency only**:
one span per call with its duration and status, and nothing from inside the
tool. Local tool calls are left to the sub-agent loop's own accounting.

The ids travel in :class:`contextvars.ContextVar` s rather than in
checkpointed graph state. A request id is an identifier, not a counter, but
the same reasoning applies either way: per-turn values do not belong in
checkpointed state, and putting them there would make every entry point
responsible for resetting them. Contextvars flow down the call stack and
across ``asyncio.to_thread`` boundaries (which copy the current context), so
the parallel departments of one turn all inherit the same request id without
any node passing it around.

Everything here is a no-op unless observability is on — ``agent_factory``
builds its tracer only when ``OBSERVABILITY=1``, and every function returns
``None`` when there is no active request, so call sites never branch on
whether tracing is enabled.
"""

import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

# What a span can be about. ``node`` is the catch-all for graph-node
# executions; anything unrecognized is recorded as one rather than rejected —
# an observability layer that raises is worse than one that mislabels.
SPAN_KINDS = ("llm", "department", "fanout", "external_tool", "node")

# Completed requests kept for inspection. Old ones fall off the end; this is
# a debug surface, not a persistent store.
_MAX_LOGGED_REQUESTS = 200


@dataclass
class Span:
    """One timed unit of work inside a request, parented into a tree."""

    span_id: str
    request_id: str
    session_id: str
    parent_span_id: str | None
    name: str
    kind: str
    started_at: str
    duration_ms: float | None = None
    status: str = "running"  # "running" | "completed" | "error" | "abandoned"
    attributes: dict = field(default_factory=dict)
    # perf_counter anchor for the duration and whether this span owns the
    # contextvar stack position it pushed. Internal; excluded from to_dict.
    _perf: float = field(default=0.0, repr=False, compare=False)
    _owns_stack: bool = field(default=False, repr=False, compare=False)
    _stack_token: object = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": dict(self.attributes),
        }


@dataclass
class RequestTrace:
    """Everything recorded for one user message (one graph turn).

    ``spans`` are the structured, parented measurements; ``events`` mirror
    the raw LangChain callback stream (tagged, in arrival order) for
    debugging. ``open_llm_spans`` maps a callback ``run_id`` to the LLM span
    awaiting its end event, so the tracer can pair starts with ends even
    when several model calls are in flight.
    """

    request_id: str
    session_id: str
    started_at: str
    spans: list[Span] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    open_llm_spans: dict[str, Span] = field(default_factory=dict)
    ended_at: str | None = None
    duration_ms: float | None = None
    status: str = "running"
    _perf: float = field(default=0.0, repr=False, compare=False)
    _logged: bool = field(default=False, repr=False, compare=False)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "spans": [s.to_dict() for s in self.spans],
            "events": list(self.events),
        }


# The request being served. This is a module global, not a contextvar:
# LangGraph executes each node in its own copied context, so a contextvar
# set in the orchestrator node is invisible to the fanout and responder
# nodes, and the request has to survive the whole turn. It shares the
# single-flight contract of the other process-wide objects (``_tracer``,
# ``_compiled_graph``): one turn at a time per process, which is what the
# CLI and Streamlit entry points do. Concurrent invokes would cross-tag —
# the same trade-off those caches already make.
# The span *stack* stays a contextvar: a span is opened and closed within
# one node's execution, and worker threads opened via ``asyncio.to_thread``
# copy the current context, so a department's spans stack under the span
# that dispatched it without leaking into sibling threads.
_active_request: RequestTrace | None = None

_current_span: ContextVar[Span | None] = ContextVar(
    "observability_span", default=None
)

# Completed or abandoned requests, oldest first. Bounded: the log exists so a
# caller (CLI, Streamlit, tests) can read the last turns back, not to grow
# with the process.
_REQUEST_LOG: deque[RequestTrace] = deque(maxlen=_MAX_LOGGED_REQUESTS)

# Serializes the request lifecycle against concurrent invokes. Contention is
# not expected (the single-flight contract), but a lost or cross-tagged
# request is a silent failure, so the mutations of the global are locked.
_request_lock = threading.Lock()


def _now() -> str:
    return datetime.now(UTC).isoformat()


# The one switch. Read from the environment at import, like the tracer global
# in agent_factory, and flipped by tests through :func:`set_tracing_enabled`.
# This module owns the flag rather than reading agent_factory's ``_tracer``:
# the tool layer imports *this* module, and a look upward into
# ``agent_orchestrator`` from here would put the dependency arrow both ways.
_TRACING_ENABLED = os.getenv("OBSERVABILITY") == "1"


def tracing_enabled() -> bool:
    """True when observability is on (``OBSERVABILITY=1`` at import)."""
    return _TRACING_ENABLED


def set_tracing_enabled(enabled: bool) -> None:
    """Turn observability on or off for this process (tests, and callers that
    read config later than import)."""
    global _TRACING_ENABLED
    _TRACING_ENABLED = bool(enabled)


# ---------------------------------------------------------------------------
# Request lifecycle — one request per user message
# ---------------------------------------------------------------------------

def start_request(session_id: str | None) -> RequestTrace | None:
    """Begin the trace for one user message; ``None`` when observability is off.

    The session id is the conversation's thread id, so a session's requests
    collect under :func:`get_session_summary` naturally. A still-running
    previous request is finalized as ``abandoned`` first — an exception past
    the responder can skip its ``end_request``, and without this the next
    turn's spans would attribute to a turn that is over.
    """
    global _active_request
    if not tracing_enabled():
        return None
    with _request_lock:
        stale = _active_request
        if stale is not None and stale.status == "running":
            stale.status = "abandoned"
            stale.ended_at = _now()
            stale.duration_ms = (time.perf_counter() - stale._perf) * 1000.0
            _log_request(stale)
            logger.warning(
                "Request %s never ended; marked abandoned.", stale.request_id
            )

        req = RequestTrace(
            request_id=str(uuid.uuid4()),
            session_id=session_id or "default",
            started_at=_now(),
        )
        req._perf = time.perf_counter()
        _active_request = req
    return req


def end_request(status: str = "completed") -> RequestTrace | None:
    """Finalize the active request and file it into the request log.

    Idempotent on a request already finalized (the log entry happens once);
    clearing the active request is what matters on the second call.
    """
    global _active_request
    with _request_lock:
        req = _active_request
        _active_request = None
        if req is None or req.status != "running":
            return req
        req.status = status
        req.ended_at = _now()
        req.duration_ms = (time.perf_counter() - req._perf) * 1000.0
        _log_request(req)
    return req


def current_request() -> RequestTrace | None:
    """The request being served in this process, or ``None``."""
    return _active_request


def _log_request(req: RequestTrace) -> None:
    if req._logged:
        return
    req._logged = True
    _REQUEST_LOG.append(req)


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------

def start_span(name: str, kind: str, attributes: dict | None = None) -> Span | None:
    """Open a span parented to the currently open one; ``None`` with no request.

    The span is appended to the request immediately, so a crash mid-span
    still leaves the record with a ``running`` status rather than vanishing.
    The span becomes the parent context for anything opened while it is on
    the stack — a department span is what LLM and external-tool spans of
    that department hang off.
    """
    req = _active_request
    if req is None:
        return None
    if kind not in SPAN_KINDS:
        kind = "node"
    span = _new_span(name, kind, req, _current_span.get())
    if attributes:
        span.attributes.update(attributes)
    span._owns_stack = True
    span._stack_token = _current_span.set(span)
    return span


def end_span(span: Span | None, status: str = "completed", **attributes) -> None:
    """Close a span: record its duration and fold in any final attributes.

    Only a span that pushed the span stack pops it — LLM spans from the
    callback handler are leaves and never touch the stack.
    """
    if span is None:
        return
    span.duration_ms = (time.perf_counter() - span._perf) * 1000.0
    span.status = status
    if attributes:
        span.attributes.update(attributes)
    if span._owns_stack:
        span._owns_stack = False
        try:
            _current_span.reset(span._stack_token)
        except ValueError:
            # reset() only works in the context that set the token; a span
            # ended somewhere unexpected keeps the stack as-is rather than
            # raising into the caller's error path.
            logger.debug("Span %s ended outside its starting context.", span.name)


def start_llm_span(name: str, run_id: object = None) -> Span | None:
    """Open an LLM span from the callback handler, registered by run_id.

    Parented to the active span (the department, or the graph node) but
    never pushed onto the stack — the handler records leaves, and the
    run_id registration is how ``on_chat_model_end`` finds its span when
    several model calls are in flight.
    """
    req = _active_request
    if req is None:
        return None
    span = _new_span(name, "llm", req, _current_span.get())
    if run_id is not None:
        req.open_llm_spans[str(run_id)] = span
    return span


def end_llm_span(
    span: Span | None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str | None = None,
) -> None:
    """Close an LLM span with its token counts (the raw material for cost)."""
    if span is None:
        return
    attributes: dict = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if model:
        attributes["model"] = model
    end_span(span, "completed", **attributes)


def _new_span(name: str, kind: str, req: RequestTrace, parent: Span | None) -> Span:
    span = Span(
        span_id=str(uuid.uuid4()),
        request_id=req.request_id,
        session_id=req.session_id,
        parent_span_id=parent.span_id if parent else None,
        name=name,
        kind=kind,
        started_at=_now(),
    )
    span._perf = time.perf_counter()
    req.spans.append(span)
    return span


@contextmanager
def external_tool_span(tool_name: str, transport: str):
    """Latency-only span around one external tool call (MCP or A2A).

    Deliberately measures the boundary and nothing else: duration and
    status, with the tool name and transport as the only attributes. Nothing
    from inside the tool — not its payload, not its result — is recorded, so
    the span stays safe to export and cheap to keep.
    """
    span = start_span(
        f"external_tool:{tool_name}",
        "external_tool",
        attributes={"tool": tool_name, "transport": transport},
    )
    try:
        yield span
    except Exception as e:
        end_span(span, "error", error=f"{type(e).__name__}: {e}")
        raise
    end_span(span, "completed")


# ---------------------------------------------------------------------------
# Readers — the request/session accessors a caller or UI consumes
# ---------------------------------------------------------------------------

def get_request_log() -> list[dict]:
    """Every recorded request, oldest first, as serializable dicts."""
    return [req.to_dict() for req in _REQUEST_LOG]


def get_last_request() -> dict | None:
    """The most recently finalized request, or ``None``."""
    if not _REQUEST_LOG:
        return None
    return _REQUEST_LOG[-1].to_dict()


def get_session_summary(session_id: str) -> dict | None:
    """Fold every recorded request of one conversation into session metrics.

    Token totals come from the ``llm`` spans — the per-call usage the
    callback handler read off the responses — so department-level totals
    (which sum the same calls) are never counted twice.
    """
    requests = [r for r in _REQUEST_LOG if r.session_id == session_id]
    if not requests:
        return None

    llm_calls = 0
    input_tokens = 0
    output_tokens = 0
    external_calls = 0
    external_ms = 0.0
    departments: set[str] = set()

    for req in requests:
        for span in req.spans:
            if span.kind == "llm":
                llm_calls += 1
                input_tokens += span.attributes.get("input_tokens", 0)
                output_tokens += span.attributes.get("output_tokens", 0)
            elif span.kind == "external_tool":
                external_calls += 1
                if span.duration_ms is not None:
                    external_ms += span.duration_ms
            elif span.kind == "department":
                departments.add(span.attributes.get("department", span.name))

    return {
        "session_id": session_id,
        "request_count": len(requests),
        "request_ids": [r.request_id for r in requests],
        "requests": [
            {
                "request_id": r.request_id,
                "started_at": r.started_at,
                "ended_at": r.ended_at,
                "duration_ms": r.duration_ms,
                "status": r.status,
            }
            for r in requests
        ],
        "llm_calls": llm_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "external_tool_calls": external_calls,
        "external_tool_time_ms": round(external_ms, 3),
        "departments_used": sorted(departments),
    }


def clear_request_log() -> None:
    """Drop the request log (tests, and the Streamlit reset button)."""
    _REQUEST_LOG.clear()
