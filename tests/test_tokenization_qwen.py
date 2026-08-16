"""Tests for cat_agent.utils.tokenization_qwen (o200k_base heuristic)."""

from cat_agent.utils.tokenization_qwen import (
    DEFAULT_ENCODING,
    ENDOFTEXT,
    IMEND,
    IMSTART,
    SPECIAL_TOKENS_SET,
    count_tokens,
    tokenizer,
)


class TestTokenizationHeuristic:

    def test_default_encoding_is_o200k(self):
        assert DEFAULT_ENCODING == 'o200k_base'
        assert tokenizer.encoding_name == 'o200k_base'

    def test_count_tokens_returns_positive_for_non_empty(self):
        n = count_tokens('hello world')
        assert isinstance(n, int)
        assert n >= 1

    def test_count_tokens_empty(self):
        assert count_tokens('') == 0

    def test_count_tokens_unicode(self):
        n = count_tokens('你好世界')
        assert isinstance(n, int)
        assert n >= 1

    def test_count_tokens_longer_text_more_tokens(self):
        short = count_tokens('hi')
        long = count_tokens('hello world, this is a longer piece of text with more words.')
        assert long >= short

    def test_special_tokens_constants(self):
        assert ENDOFTEXT == '<|endoftext|>'
        assert IMSTART == '<|im_start|>'
        assert IMEND == '<|im_end|>'

    def test_special_tokens_set_contains_im_start_end(self):
        assert IMSTART in SPECIAL_TOKENS_SET
        assert IMEND in SPECIAL_TOKENS_SET
        assert ENDOFTEXT in SPECIAL_TOKENS_SET

    def test_tokenize_and_convert_roundtrip_prefix(self):
        text = 'Observation:'
        tokens = tokenizer.tokenize(text)
        assert tokens
        assert tokenizer.convert_tokens_to_string(tokens) == text

    def test_tokenizer_truncate_short_text_unchanged(self):
        text = 'short'
        out = tokenizer.truncate(text, max_token=100)
        assert out == text

    def test_tokenizer_truncate_respects_max_token(self):
        text = 'hello world ' * 50
        out = tokenizer.truncate(text, max_token=5)
        assert isinstance(out, str)
        assert count_tokens(out) <= 5
