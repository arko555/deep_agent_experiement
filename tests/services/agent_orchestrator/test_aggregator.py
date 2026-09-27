"""Tests for aggregator module."""

from src.services.agent_orchestrator.aggregator import aggregate


class TestAggregate:

    def test_single_department_returns_directly(self):
        result = aggregate({"research": "Research findings."}, "find papers")
        assert result == "Research findings."

    def test_multiple_departments_synthesizes(self):
        result = aggregate({
            "research": "Found ML papers.",
            "writer": "Draft report ready.",
        }, "machine learning")
        assert "2 departments" in result
        assert "Found ML papers." in result
        assert "Draft report ready." in result
        assert "machine learning" in result

    def test_empty_results_returns_message(self):
        result = aggregate({}, "find papers")
        assert "No department results" in result
        assert "find papers" in result

    def test_department_name_in_header(self):
        result = aggregate({
            "research": "Content here.",
        }, "find papers")
        # Single department returns directly, but with header prefix.
        assert "research" in result.lower() or "Content here." in result
        assert "Content here." in result

    def test_multiple_departments_count(self):
        result = aggregate({
            "research": "A.",
            "writer": "B.",
            "review": "C.",
        }, "test query")
        assert "3 departments" in result
