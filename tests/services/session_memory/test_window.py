"""Tests for session window — verify message window delivery and compression."""

from langchain_core.messages import AIMessage, HumanMessage

from src.services.session_memory.window import DEFAULT_MAX_MESSAGES, get_window
from src.services.session_memory.checkpoint import append_message, get_session


class TestGetWindowUnderThreshold:

    def test_empty_thread_returns_empty(self):
        assert get_window("empty") == []

    def test_at_or_below_max_returns_all(self):
        append_message("win1", HumanMessage(content="hello"))
        append_message("win1", AIMessage(content="world"))
        result = get_window("win1")
        assert len(result) == 2
        assert result[0].content == "hello"
        assert result[1].content == "world"

    def test_returns_exact_max_when_equal(self):
        for i in range(DEFAULT_MAX_MESSAGES):
            append_message("win2", HumanMessage(content=f"msg{i}"))
        result = get_window("win2")
        assert len(result) == DEFAULT_MAX_MESSAGES


class TestGetWindowOverThreshold:

    def test_compresses_when_over_limit(self):
        for i in range(DEFAULT_MAX_MESSAGES + 5):
            append_message("win3", HumanMessage(content=f"msg{i}"))
        result = get_window("win3")
        assert len(result) <= DEFAULT_MAX_MESSAGES

    def test_returns_list_of_base_messages(self):
        append_message("win4", HumanMessage(content="a"))
        append_message("win4", AIMessage(content="b"))
        for i in range(DEFAULT_MAX_MESSAGES + 2):
            append_message("win4", HumanMessage(content=f"extra{i}"))
        result = get_window("win4")
        assert isinstance(result, list)
        assert all(hasattr(msg, "content") for msg in result)


class TestWindowIsolation:

    def test_different_threads_have_different_windows(self):
        append_message("x", HumanMessage(content="x msg"))
        append_message("x", AIMessage(content="x response"))
        for i in range(DEFAULT_MAX_MESSAGES + 3):
            append_message("y", HumanMessage(content=f"y msg{i}"))
        assert len(get_window("x")) <= DEFAULT_MAX_MESSAGES
        assert len(get_window("y")) <= DEFAULT_MAX_MESSAGES
