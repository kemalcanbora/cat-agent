"""Tests for cat_agent.tools.retrieval."""

from unittest.mock import patch

from cat_agent.tools.retrieval import Retrieval


class TestRetrieval:

    def test_init_requires_query_and_files(self):
        with patch("cat_agent.tools.retrieval.DocParser"):
            r = Retrieval({"rag_searchers": ["front_page_search"]})
        assert r.name == "retrieval"
        assert "query" in r.parameters["required"]
        assert "files" in r.parameters["required"]

    def test_call_empty_files_returns_empty(self):
        with patch("cat_agent.tools.retrieval.DocParser"):
            r = Retrieval({"rag_searchers": ["front_page_search"]})
        with patch("cat_agent.tools.retrieval._check_deps_for_rag"):
            out = r.call({"query": "q", "files": []})
        assert out == []

    def test_call_with_mocked_doc_parse_and_search(self):
        with patch("cat_agent.tools.retrieval.DocParser"):
            r = Retrieval({"rag_searchers": ["front_page_search"]})
        record_dict = {"url": "u", "raw": [{"content": "c", "metadata": {}, "token": 1}], "title": "T"}
        with patch("cat_agent.tools.retrieval._check_deps_for_rag"):
            with patch.object(r, "doc_parse") as mock_parse:
                mock_parse.call.return_value = record_dict
                with patch.object(r, "search") as mock_search:
                    mock_search.call.return_value = []
                    out = r.call({"query": "q", "files": ["/path/to/file.txt"]})
        assert out == []
        assert mock_parse.call.call_count == 1
        mock_search.call.assert_called_once()
