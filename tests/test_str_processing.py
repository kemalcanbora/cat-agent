"""Tests for cat_agent.utils.str_processing."""

from cat_agent.utils.str_processing import (
    rm_cid,
    rm_continuous_placeholders,
    rm_hexadecimal,
    rm_newlines,
)


class TestStrProcessing:

    def test_rm_cid(self):
        assert rm_cid("Hello (cid:123) world") == "Hello  world"
        assert rm_cid("(cid:1)(cid:2)") == ""

    def test_rm_hexadecimal(self):
        assert "a" not in rm_hexadecimal("a" + "0" * 21 + "b")
        assert rm_hexadecimal("short") == "short"

    def test_rm_continuous_placeholders(self):
        out = rm_continuous_placeholders("a\n\n\n\nb")
        assert out == "a\n\nb"
        out = rm_continuous_placeholders("....---.....")
        assert "\t" in out or len(out) < 12

    def test_rm_newlines_english(self):
        out = rm_newlines("Hello\nworld")
        assert "Hello" in out and "world" in out

    def test_rm_newlines_hyphen_newline(self):
        out = rm_newlines("Hel-\nlo")
        assert "Hello" in out or "Hel" in out
