"""Tests for cat_agent.tools.search_tools.vector_search."""

import json
import tempfile
from unittest.mock import patch

import pytest

from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.vector_search import VectorSearch, _corpus_fingerprint


def _skip_without_native():
    pytest.importorskip('cat_agent._native')


class TestVectorSearch:

    def test_sort_by_scores_returns_ranked_chunks(self):
        _skip_without_native()
        chunk_a = Chunk(content='alpha beta gamma', metadata={'source': 'u', 'chunk_id': 0}, token=3)
        chunk_b = Chunk(content='delta epsilon zeta', metadata={'source': 'u', 'chunk_id': 1}, token=3)
        rec = Record(url='u', raw=[chunk_a, chunk_b], title='T')
        with tempfile.TemporaryDirectory() as tmpdir:
            search = VectorSearch({
                'vector_index_path': f'{tmpdir}/vector.usearch',
                'vector_meta_path': f'{tmpdir}/vector.usearch.meta.json',
                'embedding_backend': 'hash',
            })
            out = search.sort_by_scores('alpha', [rec])
        assert len(out) == 2
        scores = {chunk_id: score for _, chunk_id, score in out}
        assert scores[0] >= scores[1]

    def test_sort_by_scores_extracts_text_from_query_json(self):
        _skip_without_native()
        chunk = Chunk(content='keyword content', metadata={'source': 'u', 'chunk_id': 0}, token=2)
        rec = Record(url='u', raw=[chunk], title='T')
        with tempfile.TemporaryDirectory() as tmpdir:
            search = VectorSearch({
                'vector_index_path': f'{tmpdir}/vector.usearch',
                'vector_meta_path': f'{tmpdir}/vector.usearch.meta.json',
            })
            out = search.sort_by_scores('{"text": "keyword"}', [rec])
        assert len(out) == 1
        assert out[0][0] == 'u'

    def test_reuses_index_for_unchanged_corpus(self):
        _skip_without_native()
        chunk = Chunk(content='reuse me', metadata={'source': 'u', 'chunk_id': 0}, token=1)
        rec = Record(url='u', raw=[chunk], title='T')
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = {
                'vector_index_path': f'{tmpdir}/vector.usearch',
                'vector_meta_path': f'{tmpdir}/vector.usearch.meta.json',
            }
            search = VectorSearch(cfg)
            search.sort_by_scores('first query', [rec])
            with patch.object(search, '_load_or_build_index', wraps=search._load_or_build_index) as mocked:
                search.sort_by_scores('second query', [rec])
                mocked.assert_not_called()

    def test_onnx_backend_requires_model_path(self):
        with pytest.raises(ValueError, match='embedding_model_path'):
            VectorSearch({'embedding_backend': 'onnx'})


class TestCorpusFingerprint:

    def test_fingerprint_changes_with_dimensions(self):
        chunk = Chunk(content='c', metadata={'source': 'u', 'chunk_id': 0}, token=1)
        rec = Record(url='u', raw=[chunk], title='T')
        assert _corpus_fingerprint([rec], 384, 'cos') != _corpus_fingerprint([rec], 256, 'cos')
