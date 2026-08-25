"""Tests for shared runtime config (budgets and limits)."""

from src.core import config


class TestConfigDefaults:

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("AGENT_MAX_ITERATIONS", raising=False)
        monkeypatch.delenv("MAX_PARALLEL_TASKS", raising=False)
        monkeypatch.delenv("SUBAGENT_TIMEOUT_SECONDS", raising=False)
        assert config.get_max_iterations() == 25
        assert config.get_max_parallel_tasks() == 4
        assert config.get_subagent_timeout_seconds() == 600.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("AGENT_MAX_ITERATIONS", "40")
        monkeypatch.setenv("MAX_PARALLEL_TASKS", "2")
        monkeypatch.setenv("SUBAGENT_TIMEOUT_SECONDS", "120")
        assert config.get_max_iterations() == 40
        assert config.get_max_parallel_tasks() == 2
        assert config.get_subagent_timeout_seconds() == 120.0

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("AGENT_MAX_ITERATIONS", "not-a-number")
        monkeypatch.setenv("SUBAGENT_TIMEOUT_SECONDS", "also-bad")
        assert config.get_max_iterations() == 25
        assert config.get_subagent_timeout_seconds() == 600.0

    def test_max_parallel_tasks_floor_of_one(self, monkeypatch):
        monkeypatch.setenv("MAX_PARALLEL_TASKS", "0")
        assert config.get_max_parallel_tasks() == 1
