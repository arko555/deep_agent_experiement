"""Tests for shared runtime config (budgets and limits)."""

from src.config import (
    get_max_iterations,
    get_max_subagent_depth,
    get_max_parallel_tasks,
    get_subagent_timeout_seconds,
)


class TestConfigDefaults:

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("AGENT_MAX_ITERATIONS", raising=False)
        monkeypatch.delenv("AGENT_MAX_SUBAGENT_DEPTH", raising=False)
        monkeypatch.delenv("MAX_PARALLEL_TASKS", raising=False)
        monkeypatch.delenv("SUBAGENT_TIMEOUT_SECONDS", raising=False)
        assert get_max_iterations() == 25
        assert get_max_subagent_depth() == 3
        assert get_max_parallel_tasks() == 4
        assert get_subagent_timeout_seconds() == 600.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AGENT_MAX_ITERATIONS", "40")
        monkeypatch.setenv("AGENT_MAX_SUBAGENT_DEPTH", "5")
        monkeypatch.setenv("MAX_PARALLEL_TASKS", "2")
        monkeypatch.setenv("SUBAGENT_TIMEOUT_SECONDS", "120")
        assert get_max_iterations() == 40
        assert get_max_subagent_depth() == 5
        assert get_max_parallel_tasks() == 2
        assert get_subagent_timeout_seconds() == 120.0

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("AGENT_MAX_ITERATIONS", "not-a-number")
        monkeypatch.setenv("AGENT_MAX_SUBAGENT_DEPTH", "also-bad")
        monkeypatch.setenv("SUBAGENT_TIMEOUT_SECONDS", "also-bad")
        assert get_max_iterations() == 25
        assert get_max_subagent_depth() == 3
        assert get_subagent_timeout_seconds() == 600.0

    def test_max_parallel_tasks_floor_of_one(self, monkeypatch):
        monkeypatch.setenv("MAX_PARALLEL_TASKS", "0")
        assert get_max_parallel_tasks() == 1
