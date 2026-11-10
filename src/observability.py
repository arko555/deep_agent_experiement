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

import json
import logging
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

# What a span can be about. ``node`` is the catch-all for graph-node
# executions; anything unrecognized is recorded as one rather than rejected —
# an observability layer that raises is worse than one that mislabels.
SPAN_KINDS = (
    "llm",
    "department",
    "fanout",
    "external_tool",
    "chain",
    "responder",
    "node",
)

# Built-in per-1K-token prices (USD) for common models. Unknown models fall back
# to 0.0 so observability never fails because of a missing price.
_DEFAULT_PRICES: dict[tuple[str, str], tuple[float, float]] = {
    ("anthropic", "claude-3-5-sonnet-20240620"): (3.00, 15.00),
    ("openai", "gpt-4o"): (2.50, 10.00),
    ("google", "gemini-2.0-flash"): (0.15, 0.60),
    ("ollama", "local"): (0.0, 0.0),
}

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
    duration_s: float | None = None
    status: str = "running"  # "running" | "completed" | "error" | "abandoned"
    cost_usd: float = 0.0
    attributes: dict = field(default_factory=dict)
    # perf_counter anchor for the duration and whether this span owns the
    # contextvar stack position it pushed. Internal; excluded from to_dict.
    _perf: float = field(default=0.0, repr=False, compare=False)
    _owns_stack: bool = field(default=False, repr=False, compare=False)
    _stack_token: "Token[Span | None] | None" = field(
        default=None, repr=False, compare=False
    )

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
            "duration_s": self.duration_s,
            "status": self.status,
            "cost_usd": self.cost_usd,
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
    open_chain_spans: dict[str, Span] = field(default_factory=dict)
    ended_at: str | None = None
    duration_ms: float | None = None
    duration_s: float | None = None
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
            "duration_s": self.duration_s,
            "status": self.status,
            "spans": [s.to_dict() for s in self.spans],
            "events": list(self.events),
            "services": _service_metrics(self.spans),
        }

    @property
    def root_spans(self) -> list["Span"]:
        """Spans with no parent (normally one; fan-out may produce several)."""
        return [s for s in self.spans if s.parent_span_id is None]

    def children(self, span_id: str) -> list["Span"]:
        """Direct children of a span, in creation order."""
        return [s for s in self.spans if s.parent_span_id == span_id]


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
            stale.duration_s = stale.duration_ms / 1000.0
            _log_request(stale)
            logger.warning(
                "Request %s never ended; marked abandoned.", stale.request_id
            )

        # A new turn starts at the root: drop any span a previous turn left
        # open (a node that raised past end_span) so its spans do not parent
        # under a dead turn.
        _current_span.set(None)

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
        req.duration_s = req.duration_ms / 1000.0
        _log_request(req)
    # Best-effort export: file (JSON/JSONL) and optional OTLP/HTTP.
    export_path = os.getenv("OBSERVABILITY_EXPORT_PATH")
    if req is not None:
        if export_path:
            fmt = os.getenv("OBSERVABILITY_EXPORT_FORMAT", "json") or "json"
            export_request(req, _export_path_for(req, export_path, fmt), fmt)
        otlp_endpoint = os.getenv("OBSERVABILITY_OTLP_ENDPOINT")
        if otlp_endpoint:
            export_otlp(req, otlp_endpoint)
    return req


def current_request() -> RequestTrace | None:
    """The request being served in this process, or ``None``."""
    return _active_request


def current_request_id() -> str | None:
    """The active request id, or ``None`` — for propagating into remote calls."""
    req = _active_request
    return req.request_id if req is not None else None


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
    span.duration_s = span.duration_ms / 1000.0
    span.status = status
    if attributes:
        span.attributes.update(attributes)
    if span._owns_stack:
        span._owns_stack = False
        token = span._stack_token
        if token is not None:
            try:
                _current_span.reset(token)
            except ValueError:
                # reset() only works in the context that set the token; a span
                # ended somewhere unexpected keeps the stack as-is rather than
                # raising into the caller's error path.
                logger.debug(
                    "Span %s ended outside its starting context.", span.name
                )


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
    provider: str | None = None,
) -> None:
    """Close an LLM span with its token counts (the raw material for cost)."""
    if span is None:
        # Still record nothing; but keep cost computation out of the no-op path.
        return
    if provider is None:
        provider = os.getenv("OBSERVABILITY_DEFAULT_PROVIDER", "openrouter")
    cost = estimate_cost(provider, model or "", input_tokens, output_tokens)
    span.cost_usd = cost
    total_tokens = input_tokens + output_tokens
    attributes: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost,
    }
    if model:
        attributes["model"] = model
    if provider:
        attributes["provider"] = provider
    end_span(span, "completed", **attributes)


def start_chain_span(name: str, run_id: object = None) -> Span | None:
    """Open a ``chain`` span from a LangChain chain callback.

    Chain callbacks fire for framework-internal runnables as well as real
    graph nodes, so the tracer gates them behind ``OBSERVABILITY_CHAIN_SPANS``.
    Like LLM spans they are leaves: the span is registered by ``run_id`` and
    never pushed onto the contextvar stack.
    """
    req = _active_request
    if req is None:
        return None
    span = _new_span(name, "chain", req, _current_span.get())
    if run_id is not None:
        req.open_chain_spans[str(run_id)] = span
    return span


def end_chain_span(run_id: object = None) -> Span | None:
    """Close the chain span opened for *run_id*, if it is still open."""
    req = _active_request
    if req is None or run_id is None:
        return None
    span = req.open_chain_spans.pop(str(run_id), None)
    if span is not None:
        end_span(span, "completed")
    return span


def _env_key(provider: str, model: str, kind: str) -> str:
    return (
        "OBSERVABILITY_PRICE_"
        + provider.upper()
        + "_"
        + model.upper().replace("-", "_").replace(".", "_")
        + "_"
        + kind
    )


def _env_price(provider: str, model: str, kind: str) -> float | None:
    """Read an env-based price override, if present."""
    raw = os.getenv(_env_key(provider, model, kind))
    if raw:
        try:
            return float(raw)
        except ValueError:
            return None
    # OpenRouter convenience env vars.
    if provider.lower() == "openrouter":
        alt = os.getenv(f"OPENROUTER_{kind}_PRICE")
        if alt:
            try:
                return float(alt)
            except ValueError:
                return None
    return None


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for a call. Unknown models default to $0."""
    key = (provider.lower(), model)
    default_in, default_out = _DEFAULT_PRICES.get(key, (0.0, 0.0))
    in_price = _env_price(provider, model, "INPUT")
    if in_price is None:
        in_price = default_in
    out_price = _env_price(provider, model, "OUTPUT")
    if out_price is None:
        out_price = default_out
    return (input_tokens / 1000.0) * in_price + (output_tokens / 1000.0) * out_price


def export_request(request, path: str, fmt: str = "json") -> str | None:
    """Write a request to disk as JSON or JSONL. Returns path on success."""
    if request is None or not path:
        return None
    data = request.to_dict() if hasattr(request, "to_dict") else dict(request)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if fmt == "jsonl":
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        return path
    except Exception as e:  # pragma: no cover - best effort
        logger.warning("Failed to export observability request to %s: %s", path, e)
        return None


def export_last_request(path: str | None = None, fmt: str | None = None) -> str | None:
    path = path or os.getenv("OBSERVABILITY_EXPORT_PATH")
    if not path:
        return None
    fmt = fmt or os.getenv("OBSERVABILITY_EXPORT_FORMAT", "json") or "json"
    if not _REQUEST_LOG:
        return None
    return export_request(_REQUEST_LOG[-1], path, fmt)


def _export_path_for(req: RequestTrace, path: str, fmt: str) -> str:
    """Resolve the export path, adding a request-unique name when configured.

    An existing directory as *path* always produces one file per request.
    ``OBSERVABILITY_EXPORT_UNIQUE=1`` does the same for a file path by
    inserting the request id before the extension, so concurrent invokes
    cannot interleave or overwrite each other.
    """
    if os.path.isdir(path):
        return os.path.join(path, f"{req.request_id}.{fmt}")
    if os.getenv("OBSERVABILITY_EXPORT_UNIQUE") == "1":
        base, ext = os.path.splitext(path)
        return f"{base}-{req.request_id}{ext or '.' + fmt}"
    return path


def _otlp_value(value) -> dict:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, (dict, list, tuple)):
        return {"stringValue": json.dumps(value, default=str, sort_keys=True)}
    return {"stringValue": "" if value is None else str(value)}


def _otlp_attributes(attrs: dict) -> list[dict]:
    return [{"key": str(k), "value": _otlp_value(v)} for k, v in attrs.items()]


def _iso_to_ns(value: str | None) -> int:
    if not value:
        return time.time_ns()
    try:
        return int(datetime.fromisoformat(value).timestamp() * 1_000_000_000)
    except ValueError:
        return time.time_ns()


def _otlp_span_dict(span: dict, trace_id_hex: str) -> dict:
    start_ns = _iso_to_ns(span.get("started_at"))
    end_ns = start_ns + int((span.get("duration_ms") or 0.0) * 1_000_000)
    attrs = span.get("attributes") or {}
    status: dict = {"code": 2 if span.get("status") == "error" else 1}
    if attrs.get("error"):
        status["message"] = str(attrs["error"])
    return {
        "traceId": trace_id_hex,
        "spanId": (span.get("span_id") or "").replace("-", "")[:16],
        "parentSpanId": (span.get("parent_span_id") or "").replace("-", "")[:16],
        "name": span.get("name") or "unknown",
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": _otlp_attributes(attrs),
        "status": status,
    }


def export_otlp(
    request, endpoint: str | None = None, timeout: float = 5.0
) -> bool:
    """POST a request to an OTLP/HTTP collector; best effort, never raises.

    OTLP/HTTP accepts the protobuf-JSON mapping, so this uses ``httpx`` (already
    a dependency) instead of pulling in the OpenTelemetry SDK. *endpoint* is the
    collector base URL; ``/v1/traces`` is appended when it is not already
    present. Returns ``True`` when the collector accepted the payload.
    """
    endpoint = endpoint or os.getenv("OBSERVABILITY_OTLP_ENDPOINT")
    if request is None or not endpoint:
        return False
    try:
        import httpx
    except Exception:  # pragma: no cover - httpx is a hard dependency
        logger.warning("OTLP export skipped: httpx is not installed")
        return False

    data = request.to_dict() if hasattr(request, "to_dict") else dict(request)
    trace_hex = (data.get("request_id") or "").replace("-", "") or uuid.uuid4().hex
    spans = data.get("spans") or []
    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": _otlp_attributes(
                        {"service.name": "deep-agents-langchain"}
                    )
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "src.observability", "version": "1"},
                        "spans": [_otlp_span_dict(s, trace_hex) for s in spans],
                    }
                ],
            }
        ]
    }
    url = endpoint.rstrip("/")
    if not url.endswith("/v1/traces"):
        url += "/v1/traces"
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Failed to export observability request to %s: %s", url, e)
        return False


def _service_metrics(spans: list) -> dict:
    """Aggregate per-service metrics across a request's spans."""
    services: dict[str, dict] = {}
    for span in spans:
        data = span.to_dict() if hasattr(span, "to_dict") else span
        attrs = data.get("attributes", {}) or {}
        service = attrs.get("service") or data.get("name") or "unknown"
        entry = services.setdefault(
            service,
            {
                "service": service,
                "span_count": 0,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "duration_s": 0.0,
            },
        )
        entry["span_count"] += 1
        entry["duration_s"] += data.get("duration_s") or 0.0
        entry["cost_usd"] += data.get("cost_usd") or 0.0
        if data.get("kind") == "llm":
            entry["llm_calls"] += 1
            entry["input_tokens"] += attrs.get("input_tokens", 0) or 0
            entry["output_tokens"] += attrs.get("output_tokens", 0) or 0
            entry["total_tokens"] += attrs.get("total_tokens", 0) or 0
    return services


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
    # Determine the owning service from the span kind / parent so downstream
    # metrics can be grouped per service across the request.
    if kind == "department":
        service = name
    elif kind == "fanout":
        service = "fanout"
    else:
        service = None
        if parent is not None:
            service = parent.attributes.get("service")
        if not service:
            service = "orchestrator"
    span.attributes["service"] = service
    span.attributes.setdefault("caller", service)
    span._perf = time.perf_counter()
    req.spans.append(span)
    return span


@contextmanager
def external_tool_span(tool_name: str, transport: str, attributes: dict | None = None):
    """Latency-only span around one external tool call (MCP or A2A)."""
    span = start_span(
        f"external_tool:{tool_name}",
        "external_tool",
        attributes={"tool": tool_name, "transport": transport},
    )
    if span is not None and attributes:
        span.attributes.update(attributes)
    if span is not None:
        # The tool is its own service; the caller stays whatever dispatched
        # the span (a department or the orchestrator), which is what makes the
        # ``caller`` attribute useful for grouping.
        span.attributes["service"] = tool_name
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
    total_cost = 0.0
    departments: set[str] = set()

    for req in requests:
        for span in req.spans:
            total_cost += getattr(span, "cost_usd", 0.0) or 0.0
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
        "total_cost_usd": total_cost,
        "departments_used": sorted(departments),
    }


def clear_request_log() -> None:
    """Drop the request log (tests, and the Streamlit reset button)."""
    _REQUEST_LOG.clear()


# ---------------------------------------------------------------------------
# Analysis — trajectory-level rollups over a finished request
# ---------------------------------------------------------------------------
#
# The collection layer above records spans; this layer answers the questions a
# reviewer asks of a finished turn: where did the time go, which tools ran and
# how often did they fail, and which spans are slow enough to be bottlenecks.
# Like everything here it is pure and side-effect free, and it accepts either a
# ``RequestTrace`` or its ``to_dict()`` form so the UI and exporters can share it.

_DEFAULT_BOTTLENECK_MS = 5000.0


def _bottleneck_threshold() -> float:
    """Slow-span cutoff in ms (``OBSERVABILITY_BOTTLENECK_MS``)."""
    raw = os.getenv("OBSERVABILITY_BOTTLENECK_MS")
    if raw:
        try:
            return float(raw)
        except ValueError:
            return _DEFAULT_BOTTLENECK_MS
    return _DEFAULT_BOTTLENECK_MS


def _attr(span: dict, key: str, default=None):
    """Read one span attribute, tolerating dict spans from ``to_dict()``."""
    return (span.get("attributes") or {}).get(key, default)


def analyze_request(request, bottleneck_ms: float | None = None) -> dict | None:
    """Roll a finished request into summary/performance/tool/error/bottleneck views.

    Accepts a ``RequestTrace`` or the dict produced by ``to_dict()``. Returns
    ``None`` for a missing request so callers can render an empty state rather
    than branch on its type.
    """
    if request is None:
        return None
    data = request.to_dict() if hasattr(request, "to_dict") else dict(request)
    spans = data.get("spans") or []
    if bottleneck_ms is None:
        bottleneck_ms = _bottleneck_threshold()
    return {
        "request_id": data.get("request_id"),
        "session_id": data.get("session_id"),
        "status": data.get("status"),
        "duration_ms": data.get("duration_ms"),
        "summary": _analyze_summary(spans),
        "performance": _analyze_performance(spans),
        "tool_usage": _analyze_tool_usage(spans),
        "error_analysis": _analyze_errors(spans),
        "departments": _analyze_departments(spans),
        "bottlenecks": _analyze_bottlenecks(spans, bottleneck_ms),
    }


def analyze_last_request(bottleneck_ms: float | None = None) -> dict | None:
    """Analyze the most recently finalized request, or ``None``."""
    if not _REQUEST_LOG:
        return None
    return analyze_request(_REQUEST_LOG[-1], bottleneck_ms)


def _analyze_summary(spans: list[dict]) -> dict:
    kind_counts: dict[str, int] = {}
    durations: list[float] = []
    for span in spans:
        kind = span.get("kind", "node")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        duration = span.get("duration_ms")
        if duration is not None:
            durations.append(duration)
    return {
        "total_spans": len(spans),
        "span_types": kind_counts,
        "total_duration_ms": round(sum(durations), 3),
        "error_count": sum(1 for s in spans if s.get("status") == "error"),
    }


def _analyze_performance(spans: list[dict]) -> dict:
    llm = [s for s in spans if s.get("kind") == "llm"]
    tools = [s for s in spans if s.get("kind") == "external_tool"]
    llm_ms = [s["duration_ms"] for s in llm if s.get("duration_ms") is not None]
    tool_ms = [s["duration_ms"] for s in tools if s.get("duration_ms") is not None]
    return {
        "llm_call_count": len(llm),
        "llm_total_ms": round(sum(llm_ms), 3),
        "llm_avg_ms": round(sum(llm_ms) / len(llm_ms), 3) if llm_ms else 0.0,
        "tool_call_count": len(tools),
        "tool_total_ms": round(sum(tool_ms), 3),
        "total_tokens": sum(_attr(s, "total_tokens", 0) for s in llm),
        "total_cost_usd": round(
            sum((s.get("cost_usd") or 0.0) for s in spans), 6
        ),
    }


def _analyze_tool_usage(spans: list[dict]) -> dict:
    tools = [s for s in spans if s.get("kind") == "external_tool"]
    names = [_attr(s, "tool") or s.get("name", "unknown") for s in tools]
    frequency: dict[str, int] = {}
    for name in names:
        frequency[name] = frequency.get(name, 0) + 1
    succeeded = sum(1 for s in tools if s.get("status") == "completed")
    return {
        "tools_used": sorted(frequency),
        "total_calls": len(tools),
        "success_rate": (succeeded / len(tools)) if tools else 1.0,
        "tool_frequency": frequency,
    }


def _analyze_errors(spans: list[dict]) -> dict:
    errored = [s for s in spans if s.get("status") == "error"]
    error_types: dict[str, int] = {}
    for span in errored:
        message = _attr(span, "error") or "unknown"
        # Errors are recorded as "Type: message"; the type is what groups.
        error_type = message.split(":", 1)[0].strip() or "unknown"
        error_types[error_type] = error_types.get(error_type, 0) + 1
    return {
        "total_errors": len(errored),
        "error_types": error_types,
        "error_rate": (len(errored) / len(spans)) if spans else 0.0,
    }


def _descendants(by_parent: dict, span_id: str | None) -> list[dict]:
    found: list[dict] = []
    for child in by_parent.get(span_id, []):
        found.append(child)
        found.extend(_descendants(by_parent, child.get("span_id")))
    return found


def _analyze_departments(spans: list[dict]) -> list[dict]:
    """Per-department rollup: latency, LLM cost/tokens, and tool calls.

    Department spans are the roots of a sub-agent's work; the LLM and external
    tool spans beneath them are what actually cost time and money.
    """
    by_parent: dict = {}
    for span in spans:
        by_parent.setdefault(span.get("parent_span_id"), []).append(span)

    departments = []
    for dept in (s for s in spans if s.get("kind") == "department"):
        descendants = _descendants(by_parent, dept.get("span_id"))
        llm = [d for d in descendants if d.get("kind") == "llm"]
        departments.append(
            {
                "department": _attr(dept, "department") or dept.get("name"),
                "duration_ms": dept.get("duration_ms"),
                "status": dept.get("status"),
                "llm_calls": len(llm),
                "input_tokens": sum(_attr(s, "input_tokens", 0) for s in llm),
                "output_tokens": sum(_attr(s, "output_tokens", 0) for s in llm),
                "cost_usd": round(
                    sum((s.get("cost_usd") or 0.0) for s in llm), 6
                ),
                "tool_calls": sum(
                    1 for d in descendants if d.get("kind") == "external_tool"
                ),
            }
        )
    return departments


def _analyze_bottlenecks(spans: list[dict], threshold_ms: float) -> list[dict]:
    bottlenecks = []
    for span in spans:
        duration = span.get("duration_ms")
        if duration is not None and duration >= threshold_ms:
            bottlenecks.append(
                {
                    "span_id": span.get("span_id"),
                    "kind": span.get("kind"),
                    "name": span.get("name"),
                    "duration_ms": duration,
                }
            )
    return sorted(bottlenecks, key=lambda x: x["duration_ms"], reverse=True)
