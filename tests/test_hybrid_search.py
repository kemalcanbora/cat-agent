# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for cat_agent.tools.search_tools.hybrid_search."""

from unittest.mock import MagicMock

import pytest

from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.front_page_search import POSITIVE_INFINITY
from cat_agent.tools.search_tools.hybrid_search import HybridSearch


def _doc(url: str = 'doc-a', n: int = 2) -> Record:
    raw = [Chunk(content=f'c{i}', metadata={}, token=1) for i in range(n)]
    return Record(url=url, raw=raw, title='t')


def test_hybrid_rejects_self_in_rag_searchers():
    with pytest.raises(ValueError, match='can not be in'):
        HybridSearch({'rag_searchers': ['hybrid_search']})


def test_hybrid_sort_by_scores_rrf_merge(monkeypatch):
    doc = _doc()

    s1 = MagicMock()
    s1.sort_by_scores.return_value = [
        ('doc-a', 0, 10.0),
        ('doc-a', 1, 1.0),
    ]
    s2 = MagicMock()
    s2.sort_by_scores.return_value = [
        ('doc-a', 1, 9.0),
        ('doc-a', 0, 2.0),
    ]

    monkeypatch.setattr(
        'cat_agent.tools.search_tools.hybrid_search.TOOL_REGISTRY',
        {'kw': lambda cfg: s1, 'vec': lambda cfg: s2},
    )
    hs = HybridSearch({'rag_searchers': ['kw', 'vec']})
    ranked = hs.sort_by_scores('q', [doc])
    assert ranked[0][0] == 'doc-a'
    assert all(score > 0 for _, _, score in ranked)
    assert {chunk for _, chunk, _ in ranked} == {0, 1}


def test_hybrid_preserves_positive_infinity(monkeypatch):
    doc = _doc(n=1)
    s1 = MagicMock()
    s1.sort_by_scores.return_value = [('doc-a', 0, POSITIVE_INFINITY)]
    monkeypatch.setattr(
        'cat_agent.tools.search_tools.hybrid_search.TOOL_REGISTRY',
        {'kw': lambda cfg: s1},
    )
    hs = HybridSearch({'rag_searchers': ['kw']})
    ranked = hs.sort_by_scores('q', [doc])
    assert ranked[0][2] == POSITIVE_INFINITY
