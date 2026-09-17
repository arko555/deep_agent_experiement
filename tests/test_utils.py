"""Tests for retry classification — transient errors retry, deterministic ones fail fast."""

import pytest

from src.core.utils import invoke_with_retry, is_transient_error


class _FakeInvoker:
    """Returns/raises queued outcomes in order, one per .invoke() call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, *args):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestIsTransientError:

    def test_timeout_is_transient(self):
        assert is_transient_error(TimeoutError("read timed out"))

    def test_rate_limit_is_transient(self):
        assert is_transient_error(Exception("429 Too Many Requests"))

    def test_server_error_is_transient(self):
        assert is_transient_error(Exception("503 service unavailable"))

    def test_connection_error_is_transient(self):
        assert is_transient_error(ConnectionError("connection reset by peer"))

    def test_bad_args_not_transient(self):
        assert not is_transient_error(ValueError("invalid arguments for tool"))

    def test_missing_file_not_transient(self):
        assert not is_transient_error(FileNotFoundError("workspace/notes.md"))


class TestInvokeWithRetry:

    def test_non_transient_raises_immediately(self):
        fake = _FakeInvoker([ValueError("bad args"), "never reached"])
        with pytest.raises(ValueError):
            invoke_with_retry(fake, [], base_delay=0)
        assert fake.calls == 1  # no retry on deterministic failure

    def test_transient_retries_then_succeeds(self):
        fake = _FakeInvoker([ConnectionError("connection reset"), "ok"])
        result = invoke_with_retry(fake, [], base_delay=0)
        assert result == "ok"
        assert fake.calls == 2

    def test_transient_exhausts_retries(self):
        fake = _FakeInvoker(
            [ConnectionError("reset"), ConnectionError("reset"), ConnectionError("reset")]
        )
        with pytest.raises(ConnectionError):
            invoke_with_retry(fake, [], max_retries=3, base_delay=0)
        assert fake.calls == 3
