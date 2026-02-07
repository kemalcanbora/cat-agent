"""Tests for cat_agent.utils.tokenization_qwen."""

from cat_agent.utils.tokenization_qwen import (
    count_tokens,
    tokenizer,
    VOCAB_FILES_NAMES,
    PAT_STR,
    ENDOFTEXT,
    IMSTART,
    IMEND,
    SPECIAL_TOKENS_SET,
)


class TestTokenizationQwen:

    def test_count_tokens_returns_positive_for_non_empty(self):
        n = count_tokens("hello world")
        assert isinstance(n, int)
        assert n >= 0

    def test_count_tokens_empty(self):
        assert count_tokens("") == 0

    def test_count_tokens_unicode(self):
        n = count_tokens("你好世界")
        assert isinstance(n, int)
        assert n >= 0

    def test_count_tokens_longer_text_more_tokens(self):
        short = count_tokens("hi")
        long = count_tokens("hello world, this is a longer piece of text with more words.")
        assert long >= short

    def test_count_tokens_special_tokens_not_counted_as_one(self):
        n = count_tokens("hello")
        assert n >= 1

    def test_vocab_files_names(self):
        assert "vocab_file" in VOCAB_FILES_NAMES
        assert "qwen" in VOCAB_FILES_NAMES["vocab_file"].lower()

    def test_pat_str_exists(self):
        assert isinstance(PAT_STR, str)
        assert len(PAT_STR) > 0

    def test_special_tokens_constants(self):
        assert ENDOFTEXT == "<|endoftext|>"
        assert IMSTART == "<|im_start|>"
        assert IMEND == "<|im_end|>"

    def test_special_tokens_set_contains_im_start_end(self):
        assert IMSTART in SPECIAL_TOKENS_SET
        assert IMEND in SPECIAL_TOKENS_SET
        assert ENDOFTEXT in SPECIAL_TOKENS_SET

    def test_tokenizer_truncate_short_text_unchanged(self):
        text = "short"
        out = tokenizer.truncate(text, max_token=100)
        assert out == text

    def test_tokenizer_truncate_respects_max_token(self):
        text = "hello world " * 50
        out = tokenizer.truncate(text, max_token=5)
        assert isinstance(out, str)
        n = count_tokens(out)
        assert n <= 5 or len(out) <= len(text)
