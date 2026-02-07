"""Tests for cat_agent.tools.web_search."""

from unittest.mock import patch

import pytest

from cat_agent.tools.web_search import WebSearch


class TestWebSearch:

    def test_format_results(self):
        results = [
            {"title": "T1", "snippet": "S1", "date": "2024"},
            {"title": "T2", "snippet": "S2"},
        ]
        out = WebSearch._format_results(results)
        assert "T1" in out
        assert "S1" in out
        assert "T2" in out
        assert "```" in out

    def test_call_requires_query(self):
        w = WebSearch()
        with pytest.raises(Exception):
            w.call("{}")

    def test_call_with_mocked_search(self):
        w = WebSearch()
        with patch.object(w, "search", return_value=[{"title": "A", "snippet": "B"}]) as mock_search:
            out = w.call({"query": "test"})
        mock_search.assert_called_once_with("test")
        assert "A" in out and "B" in out
