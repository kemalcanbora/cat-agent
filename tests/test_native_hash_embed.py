"""Tests for native hash embeddings."""

from __future__ import annotations

import hashlib
import math

import pytest

from cat_agent.tools.search_tools.embedding import HashEmbedder
from cat_agent.tools.search_tools.keyword_search import split_text_into_keywords


def _legacy_hash_embed(texts: list[str], dimensions: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        vector = [0.0] * dimensions
        for token in split_text_into_keywords(text[:2000]):
            index = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16) % dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        vectors.append(vector)
    return vectors


class TestNativeHashEmbed:
    def test_matches_legacy_python_reference(self):
        pytest.importorskip('cat_agent._native')
        texts = [
            'machine learning retrieval index',
            'Rust HNSW vector search embeddings',
            '中文检索和英文 stemming 一起测试',
        ]
        dimensions = 384
        native = HashEmbedder(dimensions=dimensions).embed(texts)
        legacy = _legacy_hash_embed(texts, dimensions)
        assert len(native) == len(legacy)
        for left, right in zip(native, legacy):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                assert a == pytest.approx(b, rel=1e-5, abs=1e-5)

    def test_exposed_on_native_module(self):
        native = pytest.importorskip('cat_agent._native')
        vectors = native.hash_embed(['alpha beta'], 64)
        assert len(vectors) == 1
        assert len(vectors[0]) == 64
