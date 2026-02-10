"""Tests for cat_agent.tools.search_tools.leann_search."""

import builtins
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.leann_search import LeannSearch


class TestLeannSearchInit:

    def test_init_default_rebuild_rag_none(self):
        s = LeannSearch()
        assert s.rebuild_rag is None

    def test_init_cfg_rebuild_rag_true(self):
        s = LeannSearch({"rebuild_rag": True})
        assert s.rebuild_rag is True

    def test_init_cfg_rebuild_rag_false(self):
        s = LeannSearch({"rebuild_rag": False})
        assert s.rebuild_rag is False


class TestLeannSearchSortByScores:

    def test_sort_by_scores_leann_not_installed_raises(self):
        search = LeannSearch()
        chunk = Chunk(content="text", metadata={"source": "u", "chunk_id": 0}, token=1)
        rec = Record(url="u", raw=[chunk], title="T")
        real_import = getattr(builtins, "__import__")

        def fake_import(name, *args, **kwargs):
            if name == "leann":
                raise ModuleNotFoundError("No module named 'leann'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(ModuleNotFoundError, match="LEANN"):
                search.sort_by_scores("query", [rec])

    def test_sort_by_scores_with_mocked_leann(self):
        chunk = Chunk(content="some content", metadata={"source": "http://u", "chunk_id": 0}, token=5)
        rec = Record(url="http://u", raw=[chunk], title="T")
        search = LeannSearch({"rebuild_rag": True})
        mock_builder = MagicMock()
        mock_searcher = MagicMock()
        mock_result = MagicMock()
        mock_result.metadata = {"url": "http://u", "chunk_id": 0}
        mock_result.score = 0.9
        mock_searcher.search.return_value = [mock_result]
        # Inject fake leann module so "from leann import LeannBuilder, LeannSearcher" gets our mocks
        fake_leann = MagicMock()
        fake_leann.LeannBuilder = MagicMock(return_value=mock_builder)
        fake_leann.LeannSearcher = MagicMock(return_value=mock_searcher)
        real_import = getattr(builtins, "__import__")

        def fake_import(name, *args, **kwargs):
            if name == "leann":
                return fake_leann
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with patch("os.path.exists", return_value=False):
                out = search.sort_by_scores("query", [rec])
        assert len(out) == 1
        assert out[0][0] == "http://u"
        assert out[0][1] == 0
        assert out[0][2] == 0.9

    def test_rebuild_rag_none_reuses_existing_index(self):
        """When rebuild_rag is None (default) and index exists, the index must be reused, not rebuilt."""
        chunk = Chunk(content="some content", metadata={"source": "http://u", "chunk_id": 0}, token=5)
        rec = Record(url="http://u", raw=[chunk], title="T")
        search = LeannSearch()  # rebuild_rag defaults to None
        assert search.rebuild_rag is None

        mock_builder = MagicMock()
        mock_searcher = MagicMock()
        mock_result = MagicMock()
        mock_result.metadata = {"url": "http://u", "chunk_id": 0}
        mock_result.score = 0.8
        mock_searcher.search.return_value = [mock_result]

        fake_leann = MagicMock()
        fake_leann.LeannBuilder = MagicMock(return_value=mock_builder)
        fake_leann.LeannSearcher = MagicMock(return_value=mock_searcher)
        real_import = getattr(builtins, "__import__")

        def fake_import(name, *args, **kwargs):
            if name == "leann":
                return fake_leann
            return real_import(name, *args, **kwargs)

        # Simulate index already exists on disk
        with patch("builtins.__import__", side_effect=fake_import):
            with patch("os.path.exists", return_value=True):
                with patch.object(LeannSearch, "_load_metadata", return_value=[("http://u", 0)]):
                    out = search.sort_by_scores("query", [rec])

        # Builder should NOT have been called — index must be reused
        fake_leann.LeannBuilder.assert_not_called()
        mock_builder.add_text.assert_not_called()
        mock_builder.build_index.assert_not_called()
        # Searcher should still have been used
        assert len(out) == 1
        assert out[0][0] == "http://u"

    def test_rebuild_rag_true_always_rebuilds_even_when_index_exists(self):
        """When rebuild_rag is True the index must be rebuilt even if it already exists."""
        chunk = Chunk(content="some content", metadata={"source": "http://u", "chunk_id": 0}, token=5)
        rec = Record(url="http://u", raw=[chunk], title="T")
        search = LeannSearch({"rebuild_rag": True})

        mock_builder = MagicMock()
        mock_searcher = MagicMock()
        mock_result = MagicMock()
        mock_result.metadata = {"url": "http://u", "chunk_id": 0}
        mock_result.score = 0.7
        mock_searcher.search.return_value = [mock_result]

        fake_leann = MagicMock()
        fake_leann.LeannBuilder = MagicMock(return_value=mock_builder)
        fake_leann.LeannSearcher = MagicMock(return_value=mock_searcher)
        real_import = getattr(builtins, "__import__")

        def fake_import(name, *args, **kwargs):
            if name == "leann":
                return fake_leann
            return real_import(name, *args, **kwargs)

        # Even though index exists, rebuild_rag=True forces a rebuild
        with patch("builtins.__import__", side_effect=fake_import):
            with patch("os.path.exists", return_value=True):
                with patch.object(LeannSearch, "_remove_existing_index"):
                    with patch.object(LeannSearch, "_save_metadata"):
                        out = search.sort_by_scores("query", [rec])

        # Builder MUST have been called — forced rebuild
        fake_leann.LeannBuilder.assert_called_once()
        mock_builder.add_text.assert_called_once()
        mock_builder.build_index.assert_called_once()
        assert len(out) == 1
