"""Tests for cat_agent.tools.doc_parser."""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.tools.doc_parser import Chunk, Record, DocParser


class TestDocParserModels:

    def test_chunk_to_dict(self):
        c = Chunk(content="x", metadata={"a": 1}, token=10)
        d = c.to_dict()
        assert d["content"] == "x"
        assert d["metadata"] == {"a": 1}
        assert d["token"] == 10

    def test_record_to_dict(self):
        c = Chunk(content="y", metadata={}, token=1)
        r = Record(url="http://u", raw=[c], title="T")
        d = r.to_dict()
        assert d["url"] == "http://u"
        assert d["title"] == "T"
        assert len(d["raw"]) == 1
        assert d["raw"][0]["content"] == "y"


class TestDocParser:

    @pytest.fixture
    def doc_parser_path(self):
        return tempfile.mkdtemp()

    def test_init(self, doc_parser_path):
        with patch("cat_agent.tools.doc_parser.Storage"):
            p = DocParser({"path": doc_parser_path})
        assert p.name == "doc_parser"
        assert "url" in p.parameters["properties"]

    def test_init_uses_cfg_max_ref_token_and_parser_page_size(self, doc_parser_path):
        with patch("cat_agent.tools.doc_parser.Storage"):
            p = DocParser({"path": doc_parser_path, "max_ref_token": 2000, "parser_page_size": 500})
        assert p.max_ref_token == 2000
        assert p.parser_page_size == 500

    def test_call_cache_hit_returns_cached_record(self, doc_parser_path):
        cached = {"url": "http://x.com/d.pdf", "title": "Doc", "raw": [{"content": "c", "metadata": {}, "token": 1}]}
        with patch("cat_agent.tools.doc_parser.Storage") as MockStorage:
            mock_db = MagicMock()
            mock_db.get.return_value = json.dumps(cached)
            MockStorage.return_value = mock_db
            p = DocParser({"path": doc_parser_path})
            out = p.call({"url": "http://x.com/d.pdf"})
        assert out["url"] == cached["url"]
        assert out["title"] == "Doc"
        assert len(out["raw"]) == 1

    def test_split_doc_to_chunk_single_page_under_size(self, doc_parser_path):
        doc = [
            {"page_num": 1, "content": [{"text": "Short paragraph.", "token": 5}]},
        ]
        with patch("cat_agent.tools.doc_parser.Storage"):
            p = DocParser({"path": doc_parser_path})
        chunks = p.split_doc_to_chunk(doc, path="file:///x.pdf", title="T", parser_page_size=1000)
        assert len(chunks) == 1
        assert "Short paragraph" in chunks[0].content
        assert chunks[0].metadata["chunk_id"] == 0
        assert chunks[0].metadata["source"] == "file:///x.pdf"

    def test_split_doc_to_chunk_multiple_pages(self, doc_parser_path):
        doc = [
            {"page_num": 1, "content": [{"text": "Page one.", "token": 3}]},
            {"page_num": 2, "content": [{"text": "Page two.", "token": 3}]},
        ]
        with patch("cat_agent.tools.doc_parser.Storage"):
            p = DocParser({"path": doc_parser_path})
        chunks = p.split_doc_to_chunk(doc, path="x", title="T", parser_page_size=100)
        assert len(chunks) >= 1
        assert "[page:" in chunks[0].content or "Page one" in chunks[0].content

    def test_get_last_part_returns_overlap_from_same_page(self, doc_parser_path):
        with patch("cat_agent.tools.doc_parser.Storage"):
            p = DocParser({"path": doc_parser_path})
        chunk = ["[page: 1]", ["First sentence.", 1], ["Last part.", 1]]
        overlap = p._get_last_part(chunk)
        assert "Last part" in overlap or "First" in overlap
