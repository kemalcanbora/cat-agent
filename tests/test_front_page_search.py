"""Tests for cat_agent.tools.search_tools.front_page_search."""

from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.front_page_search import FrontPageSearch


class TestFrontPageSearch:

    def test_sort_by_scores_single_doc(self):
        chunk1 = Chunk(content="x", metadata={"source": "u", "chunk_id": 0}, token=10)
        chunk2 = Chunk(content="y", metadata={"source": "u", "chunk_id": 1}, token=10)
        rec = Record(url="u", raw=[chunk1, chunk2], title="T")
        search = FrontPageSearch()
        result = search.sort_by_scores("query", [rec], max_ref_token=1000)
        assert len(result) >= 1
        # (url, chunk_id, score)
        assert all(len(t) == 3 for t in result)
        assert result[0][0] == "u"

    def test_sort_by_scores_multiple_docs_returns_empty(self):
        c = Chunk(content="x", metadata={"source": "u", "chunk_id": 0}, token=1)
        rec = Record(url="u", raw=[c], title="T")
        search = FrontPageSearch()
        result = search.sort_by_scores("query", [rec, rec], max_ref_token=1000)
        assert result == []
