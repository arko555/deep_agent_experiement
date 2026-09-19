"""Tests for sort_tools (Phase 2.4)."""

import json
import pytest
from langchain_core.messages import AIMessage

from src.services.tools_integration.relevance import sort_tools


# ---------------------------------------------------------------------------
# Fake model
# ---------------------------------------------------------------------------

class _FakeModel:
    """Returns canned responses. Tracks invocations."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.responses:
            return AIMessage(content="[]")
        return self.responses.pop(0)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:

    def test_empty_tool_defs_returns_empty(self):
        model = _FakeModel([])
        result = sort_tools(model, "find something", [])
        assert result == []


# ---------------------------------------------------------------------------
# Basic ranking
# ---------------------------------------------------------------------------

class TestSorting:

    def test_returns_ranked_tools(self):
        tool_defs = [
            {"name": "search_web", "description": "Search the web"},
            {"name": "read_file", "description": "Read a file"},
            {"name": "write_file", "description": "Write a file"},
        ]
        model = _FakeModel([AIMessage(content='["search_web", "read_file"]')])
        result = sort_tools(model, "find info", tool_defs, max_tools=2)
        assert len(result) == 2
        assert result[0]["name"] == "search_web"
        assert result[1]["name"] == "read_file"

    def test_max_tools_capped(self):
        tool_defs = [
            {"name": f"tool_{i}", "description": f"T{i}"} for i in range(10)
        ]
        names = [f"tool_{i}" for i in range(7)]
        model = _FakeModel([AIMessage(content=json.dumps(names))])
        result = sort_tools(model, "query", tool_defs, max_tools=5)
        assert len(result) == 5
        # All returned tools must be from the input defs.
        for t in result:
            assert t["name"] in {d["name"] for d in tool_defs}

    def test_returns_fewer_than_max_when_fewer_available(self):
        tool_defs = [
            {"name": "only_one", "description": "Solo tool"},
        ]
        model = _FakeModel([AIMessage(content='["only_one"]')])
        result = sort_tools(model, "query", tool_defs, max_tools=5)
        assert len(result) == 1
        assert result[0]["name"] == "only_one"


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

class TestResponseParsing:

    def test_handles_ai_message(self):
        tool_defs = [
            {"name": "a", "description": "A"},
            {"name": "b", "description": "B"},
        ]
        model = _FakeModel([AIMessage(content='["a"]')])
        result = sort_tools(model, "q", tool_defs)
        assert len(result) == 1
        assert result[0]["name"] == "a"

    def test_handles_text_in_brackets_alone(self):
        """Model returns extra text around the JSON array."""
        tool_defs = [
            {"name": "x", "description": "X"},
            {"name": "y", "description": "Y"},
        ]
        model = _FakeModel([
            AIMessage(content='I think you want ["x"] here.')
        ])
        result = sort_tools(model, "q", tool_defs)
        assert len(result) == 1
        assert result[0]["name"] == "x"

    def test_fallback_on_invalid_json(self):
        tool_defs = [
            {"name": "a", "description": "A"},
            {"name": "b", "description": "B"},
        ]
        model = _FakeModel([AIMessage(content="not json at all")])
        result = sort_tools(model, "q", tool_defs, max_tools=2)
        # Falls back to first N tools when JSON parse fails
        assert len(result) == 2

    def test_fallback_on_exception_from_model(self):
        tool_defs = [
            {"name": "a", "description": "A"},
        ]

        class BoomModel:
            def invoke(self, messages):
                raise RuntimeError("model exploded")

        result = sort_tools(BoomModel(), "q", tool_defs, max_tools=1)
        assert len(result) == 1  # fallback


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------

class TestBehavior:

    def test_does_not_contain_args_schema(self):
        """args_schema is filtered from the prompt sent to the model."""
        tool_defs = [
            {"name": "a", "description": "A", "args_schema": {"type": "object"}},
        ]
        model = _FakeModel([AIMessage(content='["a"]')])
        sort_tools(model, "q", tool_defs)
        system_msg = model.calls[0][0]["content"]
        assert "args_schema" not in system_msg

    def test_skips_names_not_in_tool_defs(self):
        tool_defs = [
            {"name": "real", "description": "R"},
        ]
        model = _FakeModel([AIMessage(content='["real", "phantom"]')])
        result = sort_tools(model, "q", tool_defs)
        assert len(result) == 1
        assert result[0]["name"] == "real"

    def test_model_receives_system_and_user_messages(self):
        tool_defs = [{"name": "a", "description": "A"}]
        model = _FakeModel([AIMessage(content='["a"]')])
        sort_tools(model, "my query", tool_defs)
        assert len(model.calls) == 1
        messages = model.calls[0]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "my query" in messages[1]["content"]

    def test_all_tools_returned_when_under_max(self):
        tool_defs = [
            {"name": f"t{i}", "description": f"T{i}"} for i in range(3)
        ]
        model = _FakeModel([AIMessage(content='["t0", "t1", "t2"]')])
        result = sort_tools(model, "q", tool_defs, max_tools=5)
        assert len(result) == 3

    def test_empty_response_returns_fallback(self):
        tool_defs = [
            {"name": "a", "description": "A"},
        ]
        model = _FakeModel([AIMessage(content="")])
        result = sort_tools(model, "q", tool_defs)
        assert len(result) == 1  # fallback to first
