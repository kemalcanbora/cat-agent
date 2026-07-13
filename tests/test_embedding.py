"""Tests for cat_agent.tools.search_tools.embedding."""

import math

import pytest

from cat_agent.tools.search_tools.embedding import HashEmbedder, build_embedder


class TestHashEmbedder:

    def test_embed_returns_normalized_vectors(self):
        pytest.importorskip('cat_agent._native')
        embedder = HashEmbedder(dimensions=64)
        vectors = embedder.embed(['alpha beta', 'gamma delta'])
        assert len(vectors) == 2
        assert len(vectors[0]) == 64
        norm = math.sqrt(sum(value * value for value in vectors[0]))
        assert norm == pytest.approx(1.0, rel=1e-5)

    def test_similar_texts_have_higher_cosine_than_unrelated(self):
        pytest.importorskip('cat_agent._native')
        embedder = HashEmbedder(dimensions=256)
        left, right, other = embedder.embed([
            'machine learning retrieval index',
            'machine learning search index',
            'cooking pasta tomato sauce',
        ])

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb)

        assert cosine(left, right) > cosine(left, other)


class TestBuildEmbedder:

    def test_default_backend_is_hash(self):
        pytest.importorskip('cat_agent._native')
        embedder = build_embedder({})
        assert isinstance(embedder, HashEmbedder)
        assert embedder.dimensions == 384

    def test_onnx_backend_requires_model_path(self):
        with pytest.raises(ValueError, match='embedding_model_path'):
            build_embedder({'embedding_backend': 'onnx'})
