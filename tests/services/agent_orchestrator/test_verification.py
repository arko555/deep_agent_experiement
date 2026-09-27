"""Tests for verification module."""

from src.services.agent_orchestrator.verification import verify


class TestVerify:

    def test_empty_results_fails(self):
        result = verify({})
        assert result["approved"] is False
        assert "No sub-agent results" in result["reasons"][0]

    def test_single_department_approved(self):
        result = verify({"research": "Found 5 papers on the topic."})
        assert result["approved"] is True
        assert any("checks passed" in r for r in result["reasons"])

    def test_multiple_departments_approved(self):
        result = verify({
            "research": "Found papers on machine learning.",
            "writer": "Draft report is ready.",
        }, context={"original_query": "machine learning"})
        assert result["approved"] is True

    def test_empty_result_fails(self):
        result = verify({"research": ""})
        assert result["approved"] is False
        assert "returned empty result" in result["reasons"][0]

    def test_tmp_reference_fails(self):
        result = verify({
            "research": "Saved findings to /tmp/research.txt",
        })
        assert result["approved"] is False
        assert "/tmp/" in result["reasons"][0]

    def test_workspace_path_passes(self):
        result = verify({
            "research": "Saved findings to workspace/research.md",
        })
        assert result["approved"] is True

    def test_low_keyword_overlap_warns_but_passes(self):
        result = verify({
            "research": "Completely unrelated content with no overlap.",
        }, context={"original_query": "machine learning"})
        # Advisory only: it must not appear in `reasons`, which explain
        # rejections, or the verdict contradicts itself.
        assert result["approved"] is True
        assert result["reasons"] == ["All checks passed"]
        assert any("keyword overlap" in w for w in result["warnings"])

    def test_attempt_word_is_not_a_tmp_path(self):
        # "attempt/" must not trip the /tmp/ containment check.
        result = verify({
            "research": "I will attempt the fetch for machine learning results",
        }, context={"original_query": "machine learning"})
        assert result["approved"] is True
        assert result["reasons"] == ["All checks passed"]

    def test_none_result_treated_as_empty(self):
        result = verify({"research": None})
        assert result["approved"] is False

    def test_context_without_query_passes(self):
        result = verify({"research": "Some content"})
        assert result["approved"] is True


class TestVerifyWithMultipleFailures:

    def test_two_empty_results(self):
        result = verify({
            "research": "",
            "writer": "",
        })
        assert result["approved"] is False
        assert len([r for r in result["reasons"] if "returned empty" in r]) == 2

    def test_empty_and_tmp_fails(self):
        result = verify({
            "research": "",
            "writer": "Saved to /tmp/test.txt",
        })
        assert result["approved"] is False
