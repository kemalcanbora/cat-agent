"""Tests for cat_agent.tools.search_tools.base_search."""

from unittest.mock import patch

from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.base_search import BaseSearch
from cat_agent.tools.search_tools.front_page_search import FrontPageSearch


class TestBaseSearch:

    def test_get_the_front_part_single_doc_small_token(self):
        chunk = Chunk(content="hi", metadata={"source": "u", "chunk_id": 0}, token=2)
        rec = Record(url="u", raw=[chunk], title="T")
        out = BaseSearch._get_the_front_part([rec], max_ref_token=100)
        assert len(out) == 1
        assert out[0]["text"] == ["hi"]

    def test_get_the_front_part_single_doc(self):
        chunk = Chunk(content="hello", metadata={"source": "u", "chunk_id": 0}, token=5)
        rec = Record(url="u", raw=[chunk], title="T")
        out = BaseSearch._get_the_front_part([rec], max_ref_token=100)
        assert len(out) == 1
        assert out[0]["url"] == "u"
        assert "hello" in out[0]["text"][0]

    def test_format_docs_list_of_strings(self):
        with patch("cat_agent.tools.search_tools.base_search.DocParser") as MockDP:
            MockDP.return_value.split_doc_to_chunk.return_value = [
                Chunk(content="c1", metadata={"source": "doc_0", "chunk_id": 0}, token=1),
            ]
            base = FrontPageSearch()  # concrete subclass of BaseSearch
            docs = [["page1 text"], ["page2 text"]]
            new_docs, all_tokens = base.format_docs(docs)
        assert len(new_docs) == 2
        assert all_tokens >= 0
