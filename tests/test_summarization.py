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


class TestCompressMessagesToolGroups:

    @pytest.mark.parametrize("max_history", [2, 3, 4])
    def test_cut_inside_multi_tool_group_keeps_all_results(self, max_history):
        calls = AIMessage(content="", tool_calls=[
            {"name": "search", "args": {}, "id": call_id}
            for call_id in ["a", "b", "c"]
        ])
        # Results may arrive in a different order than the assistant's calls.
        group = [calls] + [
            ToolMessage(content=f"result {call_id}", tool_call_id=call_id)
            for call_id in ["c", "a", "b"]
        ]
        msgs = [HumanMessage(content="older context"), *group]
        original = list(msgs)

        result = compress_messages(msgs, max_history=max_history)

        assert result[1:] == group
        assert all(actual is expected for actual, expected in zip(result[1:], group))
        assert {call["id"] for call in result[1].tool_calls} == {
            msg.tool_call_id for msg in result[2:]
        }
        assert len(result) > max_history  # protocol validity wins over the limit
        assert msgs == original

    def test_single_result_at_limit_retains_its_call(self):
        calls = AIMessage(content="", tool_calls=[
            {"name": "search", "args": {}, "id": "single"},
        ])
        msgs = [
            HumanMessage(content="old"), calls,
            ToolMessage(content="result", tool_call_id="single"),
            AIMessage(content="answer"),
        ]

        result = compress_messages(msgs, max_history=3)

        assert result[1:] == msgs[1:]

    def test_oversized_group_without_older_messages_is_unchanged(self):
        calls = AIMessage(content="", tool_calls=[
            {"name": "search", "args": {}, "id": call_id}
            for call_id in ["a", "b"]
        ])
        msgs = [calls] + [
            ToolMessage(content="result", tool_call_id=call_id)
            for call_id in ["a", "b"]
        ]

        result = compress_messages(msgs, max_history=2)

        assert result == msgs
        assert all(actual is expected for actual, expected in zip(result, msgs))

    def test_boundary_between_exchanges_does_not_retain_old_calls(self):
        groups = []
        for prefix in ["old", "recent"]:
            call_ids = [f"{prefix}-a", f"{prefix}-b"]
            groups.append([
                AIMessage(content="", tool_calls=[
                    {"name": "search", "args": {}, "id": call_id}
                    for call_id in call_ids
                ]),
                *[ToolMessage(content=call_id, tool_call_id=call_id)
                  for call_id in reversed(call_ids)],
            ])
        msgs = groups[0] + groups[1]

        result = compress_messages(msgs, max_history=3)

        assert result[1:] == groups[1]
        assert "Exchanged 3 messages" in result[0].content
        assert "2 tool results" in result[0].content

    def test_limit_one_summarizes_entire_tool_group(self):
        msgs = [
            AIMessage(content="", tool_calls=[
                {"name": "search", "args": {}, "id": call_id}
                for call_id in ["a", "b"]
            ]),
            ToolMessage(content="first result", tool_call_id="a"),
            ToolMessage(content="last result", tool_call_id="b"),
        ]

        result = compress_messages(msgs, max_history=1)

        assert len(result) == 1
        assert "Exchanged 3 messages" in result[0].content
        assert "2 tool results" in result[0].content

    def test_safe_boundary_after_group_summarizes_whole_group(self):
        calls = AIMessage(content="", tool_calls=[
            {"name": "search", "args": {}, "id": "old-call"},
        ])
        msgs = [
            HumanMessage(content="old"), calls,
            ToolMessage(content="old result", tool_call_id="old-call"),
            AIMessage(content="final answer"),
        ]

        result = compress_messages(msgs, max_history=2)

        assert result[1:] == msgs[-1:]
        assert "1 tool results" in result[0].content


class TestCompressMessagesLimits:

    @pytest.mark.parametrize("max_history,min_keep", [(0, 8), (-1, 8), (20, -1)])
    @pytest.mark.parametrize("count", [0, 25])
    def test_invalid_limits_raise(self, max_history, min_keep, count):
        msgs = [HumanMessage(content="message") for _ in range(count)]
        with pytest.raises(ValueError):
            compress_messages(msgs, max_history=max_history, min_keep=min_keep)

    @pytest.mark.parametrize("min_keep", [0, 8])
    def test_limit_one_returns_only_summary(self, min_keep):
        msgs = [
            HumanMessage(content="first message"),
            AIMessage(content="last message"),
        ]
        result = compress_messages(msgs, max_history=1, min_keep=min_keep)
        assert len(result) == 1
        assert "Exchanged 2 messages" in result[0].content
        assert "first message" in result[0].content
        assert "last message" in result[0].content

    def test_zero_min_keep_uses_normal_retention(self):
        msgs = [HumanMessage(content=f"message {i}") for i in range(5)]
        result = compress_messages(msgs, max_history=2, min_keep=0)
        assert len(result) == 2
        assert result[1] is msgs[-1]


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
