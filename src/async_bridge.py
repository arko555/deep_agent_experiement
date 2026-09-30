"""Sync bridge for async-only integrations (MCP tools, A2A agents).

Both `langchain-mcp-adapters` and `a2a-sdk` are async-only: MCP tools are
built with a coroutine and no sync ``func`` (so ``tool.invoke()`` raises
``NotImplementedError``), and the A2A client has no sync variant in any
published version. This module lets the (entirely synchronous) graph drive
them anyway.

One event loop runs on a daemon thread for the process lifetime; ``run_sync``
schedules a coroutine onto it and blocks for the result. A persistent loop is
used rather than a per-call ``asyncio.run`` because:

- ``asyncio.run`` raises if the calling thread already has a running loop.
- MCP issue #29 documented hangs when wrapping MCP calls in ``asyncio.run``.
- A single loop is safe to call from both the main thread and from the
  ``ThreadPoolExecutor`` workers used for parallel subagent tasks.

The loop is created lazily on first use, so a process that never touches an
MCP server or A2A agent never spawns a thread.

Windows note: the stdio MCP transport needs a ``ProactorEventLoop``; this
module inherits the platform default and is only exercised on macOS today.
"""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return the bridge loop, starting its thread on first use."""
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="deep-agent-async-bridge", daemon=True
            )
            thread.start()
            _loop, _thread = loop, thread
        return _loop


def run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run *coro* on the bridge loop and return its result, blocking the caller.

    Exceptions propagate to the caller unchanged.
    """
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result()


def close_async_bridge() -> None:
    """Stop the bridge loop and join its thread. Intended for tests/teardown.

    Safe to call when the bridge was never started. A later ``run_sync``
    transparently starts a fresh loop.
    """
    global _loop, _thread
    with _lock:
        loop, thread = _loop, _thread
        _loop, _thread = None, None
    if loop is not None and not loop.is_closed():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None:
        thread.join(timeout=5)
    if loop is not None:
        loop.close()
