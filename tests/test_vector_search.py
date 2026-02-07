"""Tests for cat_agent.tools.search_tools.vector_search."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.vector_search import VectorSearch


def _make_fake_langchain_modules(mock_faiss, mock_doc):
    """Build fake langchain.* modules so 'from X import Y' in sort_by_scores gets mocks."""
    mock_doc.metadata = {"source": "u", "chunk_id": 0}
    db = MagicMock()
    db.similarity_search_with_score.return_value = [(mock_doc, 0.8)]
    mock_faiss.from_documents.return_value = db

    mock_schema = MagicMock()
    mock_schema.Document = MagicMock  # callable that returns a MagicMock instance
    mock_vectorstores = MagicMock()
    mock_vectorstores.FAISS = mock_faiss
    mock_embeddings_mod = MagicMock()
    mock_embeddings_mod.OpenAIEmbeddings = MagicMock(return_value=MagicMock())
    return {"langchain.schema": mock_schema, "langchain_community.vectorstores": mock_vectorstores,
            "langchain_community.embeddings": mock_embeddings_mod}


class TestVectorSearch:

    def test_sort_by_scores_extracts_text_from_query_json(self):
        chunk = Chunk(content="content", metadata={"source": "u", "chunk_id": 0}, token=1)
        rec = Record(url="u", raw=[chunk], title="T")
        search = VectorSearch()
        mock_faiss = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        fake_modules = _make_fake_langchain_modules(mock_faiss, mock_doc)
        with patch.dict(sys.modules, fake_modules):
            with patch("os.getenv", return_value="fake_key"):
                out = search.sort_by_scores('{"text": "real query"}', [rec])
        assert len(out) == 1
        assert out[0][0] == "u"
        assert out[0][1] == 0

    def test_sort_by_scores_langchain_missing_raises(self):
        search = VectorSearch()
        chunk = Chunk(content="c", metadata={"source": "u", "chunk_id": 0}, token=1)
        rec = Record(url="u", raw=[chunk], title="T")
        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "langchain.schema" or name == "langchain":
                raise ModuleNotFoundError("No module named 'langchain'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ModuleNotFoundError, match="langchain"):
                search.sort_by_scores("query", [rec])

    def test_sort_by_scores_plain_query_no_json(self):
        chunk = Chunk(content="c", metadata={"source": "u", "chunk_id": 0}, token=1)
        rec = Record(url="u", raw=[chunk], title="T")
        search = VectorSearch()
        mock_faiss = MagicMock()
        mock_faiss.from_documents.return_value.similarity_search_with_score.return_value = []
        mock_schema = MagicMock()
        mock_schema.Document = MagicMock
        mock_vectorstores = MagicMock()
        mock_vectorstores.FAISS = mock_faiss
        mock_embeddings_mod = MagicMock()
        mock_embeddings_mod.OpenAIEmbeddings = MagicMock(return_value=MagicMock())
        fake_modules = {"langchain.schema": mock_schema, "langchain_community.vectorstores": mock_vectorstores,
                        "langchain_community.embeddings": mock_embeddings_mod}
        with patch.dict(sys.modules, fake_modules):
            with patch("os.getenv", return_value="fake_key"):
                out = search.sort_by_scores("plain query", [rec])
        assert out == []
