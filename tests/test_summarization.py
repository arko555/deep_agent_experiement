"""Tests for conversation summarization — verify compression preserves context."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

from src.core.summarization import (
    compress_messages,
    _extract_role_label,
    _summarize_old_messages,
    DEFAULT_MAX_MESSAGES,
    MIN_KEEP_RECENT,
)


# ---------------------------------------------------------------------------
# Role label extraction
# ---------------------------------------------------------------------------

class TestExtractRoleLabel:

    def test_human_message(self):
        assert _extract_role_label(HumanMessage(content="hi")) == "User"

    def test_ai_message(self):
        assert _extract_role_label(AIMessage(content="answer")) == "Agent"

    def test_tool_message(self):
        msg = ToolMessage(content="result", tool_call_id="1", name="search")
        assert _extract_role_label(msg) == "Tool(search)"

    def test_system_message(self):
        assert _extract_role_label(SystemMessage(content="prompt")) == "System"


# ---------------------------------------------------------------------------
# compress_messages — under threshold
# ---------------------------------------------------------------------------

class TestCompressMessagesUnderThreshold:

    def test_short_list_returned_as_is(self):
        msgs = [HumanMessage(content="hello")]
        result = compress_messages(msgs, max_history=20)
        assert len(result) == 1
        assert result[0] is msgs[0]

    def test_at_threshold_no_compression(self):
        msgs = [HumanMessage(content=f"msg{i}") for i in range(20)]
        result = compress_messages(msgs, max_history=20)
        assert len(result) == 20

    def test_over_threshold_does_compress(self):
        msgs = [HumanMessage(content=f"msg{i}") for i in range(25)]
        result = compress_messages(msgs, max_history=20)
        # Should have: 1 summary + min_keep recent
        assert len(result) <= 21
        # First item should be a summary (HumanMessage with summary content).
        assert isinstance(result[0], HumanMessage)
        assert "Summary" in result[0].content


# ---------------------------------------------------------------------------
# compress_messages — compression behavior
# ---------------------------------------------------------------------------

class TestCompressMessagesBehavior:

    def test_summary_preserves_key_content(self):
        msgs = [
            HumanMessage(content="Research quantum computing"),
            AIMessage(content="I'll search for recent papers"),
            ToolMessage(content="Found 5 papers", tool_call_id="1", name="search"),
            AIMessage(content="Here's a summary of findings..."),
            HumanMessage(content="Now write the report"),
            AIMessage(content="Writing report..."),
            HumanMessage(content="Add citations"),
            AIMessage(content="Added 10 citations"),
            HumanMessage(content="Make it longer"),
            AIMessage(content="Expanded to 2000 words"),
            HumanMessage(content="Translate to French"),
            AIMessage(content="Translating..."),
            HumanMessage(content="Fix grammar"),
            AIMessage(content="Fixed."),
            HumanMessage(content="Add conclusion"),
            AIMessage(content="Done with conclusion."),
            HumanMessage(content="Review tone"),
            AIMessage(content="Tone is formal."),
            HumanMessage(content="Check facts"),
            AIMessage(content="All facts verified."),
            HumanMessage(content="Final review"),
            AIMessage(content="Everything looks good."),
            HumanMessage(content="Export as PDF"),
            AIMessage(content="Exporting..."),
            HumanMessage(content="Send to client"),
            AIMessage(content="Sent successfully."),
        ]
        result = compress_messages(msgs, max_history=8)

        # Should have a summary + recent messages.
        assert len(result) > 1
        summary_content = str(result[0].content)
        assert "Summary" in summary_content
        assert "Agent" in summary_content or "25" in summary_content

    def test_tool_errors_appear_in_summary(self):
        msgs = [
            HumanMessage(content="Do something"),
            ToolMessage(
                content="Error: file not found",
                tool_call_id="1",
                name="read_file",
            ),
            AIMessage(content="Let me try another approach"),
            ToolMessage(
                content="Error: permission denied",
                tool_call_id="2",
                name="write_file",
            ),
            AIMessage(content="I'll handle this differently"),
        ]
        # Force compression by setting a very low max_history.
        result = compress_messages(msgs, max_history=2)

        summary_content = str(result[0].content)
        assert "Error" in summary_content or "error" in summary_content.lower()

    def test_min_keep_recent_preserved(self):
        msgs = [AIMessage(content=f"msg{i}") for i in range(30)]
        result = compress_messages(msgs, max_history=10)

        # The last MIN_KEEP_RECENT messages should be intact.
        recent_count = len(result) - 1  # minus the summary
        assert recent_count >= MIN_KEEP_RECENT

    def test_summary_comes_first(self):
        msgs = [HumanMessage(content=f"msg{i}") for i in range(25)]
        result = compress_messages(msgs, max_history=10)

        assert isinstance(result[0], HumanMessage)
        assert "Summary" in result[0].content


# ---------------------------------------------------------------------------
# _summarize_old_messages edge cases
# ---------------------------------------------------------------------------

class TestSummarizeOldMessages:

    def test_empty_list_produces_minimal_summary(self):
        result = _summarize_old_messages([])
        assert isinstance(result, HumanMessage)
        assert "Summary" in result.content

    def test_single_message_summarized(self):
        msgs = [HumanMessage(content="hello world")]
        result = _summarize_old_messages(msgs)
        assert "hello world" in result.content
