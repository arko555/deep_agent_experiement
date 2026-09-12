"""Tests for the sync bridge that drives async-only integrations."""

import asyncio

import pytest

from src.core import async_bridge


@pytest.fixture(autouse=True)
def _teardown_bridge():
    yield
    async_bridge.close_async_bridge()


def test_run_sync_returns_coroutine_result():
    async def _work():
        await asyncio.sleep(0)
        return 42

    assert async_bridge.run_sync(_work()) == 42


def test_run_sync_propagates_exceptions():
    async def _boom():
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        async_bridge.run_sync(_boom())


def test_run_sync_is_reentrant_and_survives_close():
    async def _echo(value):
        return value

    assert async_bridge.run_sync(_echo("first")) == "first"
    # Closing tears the loop down; the next call must start a fresh one.
    async_bridge.close_async_bridge()
    assert async_bridge.run_sync(_echo("second")) == "second"


def test_run_sync_from_a_worker_thread():
    """Mirrors how local_tools_node drives tools from ThreadPoolExecutor workers."""
    import threading

    async def _work():
        return "from-thread"

    box = {}
    thread = threading.Thread(target=lambda: box.update(v=async_bridge.run_sync(_work())))
    thread.start()
    thread.join(timeout=5)
    assert box["v"] == "from-thread"


def test_close_without_start_is_a_noop():
    async_bridge.close_async_bridge()  # never started — must not raise
