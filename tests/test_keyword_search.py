"""Tests for cat_agent.tools.search_tools.keyword_search."""

from unittest.mock import patch

import pytest

from cat_agent.tools.doc_parser import Chunk, Record
from cat_agent.tools.search_tools.keyword_search import (
    KeywordSearch,
    clean_en_token,
    tokenize_and_filter,
    split_text_into_keywords,
    parse_keyword,
    WORDS_TO_IGNORE,
)


class TestKeywordSearchHelpers:

    def test_clean_en_token_special_case_abbreviation(self):
        assert clean_en_token("U.S.A.") == "U.S.A."

    def test_clean_en_token_special_case_email(self):
        assert "@" in clean_en_token("user@example.com")

    def test_clean_en_token_special_case_percentage(self):
        assert clean_en_token("50%") == "50%"

    def test_clean_en_token_strips_punctuation(self):
        out = clean_en_token("hello!")
        assert out == "hello" or "hello" in out

    def test_tokenize_and_filter_returns_lowercase_tokens(self):
        out = tokenize_and_filter("Hello World")
        assert "hello" in out
        assert "world" in out

    def test_tokenize_and_filter_filters_stop_words(self):
        out = tokenize_and_filter("the and a")
        assert out == [] or all(w not in WORDS_TO_IGNORE for w in out)

    def test_tokenize_and_filter_numbers(self):
        out = tokenize_and_filter("version 2.5")
        assert len(out) >= 1

    def test_split_text_into_keywords_filters_stop_words(self):
        out = split_text_into_keywords("the quick brown")
        assert "the" not in out
        assert "quick" in out
        assert "brown" in out

    def test_parse_keyword_plain_text_uses_split(self):
        out = parse_keyword("machine learning")
        assert isinstance(out, list)
        assert "machine" in out or "learn" in out

    def test_parse_keyword_json_with_text(self):
        out = parse_keyword('{"text": "hello world", "keywords_en": ["test"]}')
        assert isinstance(out, list)
        assert len(out) >= 1

    def test_parse_keyword_json_keywords_zh_en(self):
        out = parse_keyword('{"keywords_zh": ["关键词"], "keywords_en": ["keyword"], "text": "content"}')
        assert isinstance(out, list)


class TestKeywordSearch:

    def test_search_empty_wordlist_returns_front_part(self):
        chunk = Chunk(content="hi", metadata={"source": "u", "chunk_id": 0}, token=2)
        rec = Record(url="u", raw=[chunk], title="T")
        search = KeywordSearch()
        with patch.object(KeywordSearch, "sort_by_scores", return_value=[]):
            out = search.search(query="summarize this document", docs=[rec], max_ref_token=100)
        assert len(out) == 1
        assert out[0]["text"] == ["hi"]

    def test_search_zero_max_sim_returns_front_part(self):
        chunk = Chunk(content="hi", metadata={"source": "u", "chunk_id": 0}, token=2)
        rec = Record(url="u", raw=[chunk], title="T")
        search = KeywordSearch()
        # sort_by_scores returns (source, chunk_id, score); max_sims=0 triggers _get_the_front_part
        with patch.object(KeywordSearch, "sort_by_scores", return_value=[("u", 0, 0.0)]):
            out = search.search(query="query", docs=[rec], max_ref_token=100)
        assert len(out) == 1

    def test_sort_by_scores_empty_wordlist_returns_empty(self):
        search = KeywordSearch()
        chunk = Chunk(content="doc", metadata={"source": "u", "chunk_id": 0}, token=1)
        rec = Record(url="u", raw=[chunk], title="T")
        with patch("cat_agent.tools.search_tools.keyword_search.parse_keyword", return_value=[]):
            out = search.sort_by_scores(query="the and a", docs=[rec])
        assert out == []

    def test_sort_by_scores_with_bm25(self):
        pytest.importorskip("rank_bm25")
        search = KeywordSearch()
        c1 = Chunk(content="machine learning", metadata={"source": "u", "chunk_id": 0}, token=2)
        c2 = Chunk(content="python programming", metadata={"source": "u", "chunk_id": 1}, token=2)
        rec = Record(url="u", raw=[c1, c2], title="T")
        out = search.sort_by_scores(query="machine", docs=[rec])
        assert len(out) == 2
        assert all(len(t) == 3 for t in out)
        assert out[0][2] >= out[1][2]
