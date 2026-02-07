"""Tests for cat_agent.tools.web_extractor."""

from unittest.mock import patch

import pytest

from cat_agent.tools.web_extractor import WebExtractor


class TestWebExtractor:

    def test_call_requires_url(self):
        e = WebExtractor()
        with pytest.raises(Exception):
            e.call("{}")

    def test_call_with_mocked_parser(self):
        e = WebExtractor()
        with patch("cat_agent.tools.web_extractor.SimpleDocParser") as MockParser:
            MockParser.return_value.call.return_value = "<html>parsed</html>"
            out = e.call({"url": "https://example.com"})
        assert out == "<html>parsed</html>"
        MockParser.return_value.call.assert_called_once()
        call_kw = MockParser.return_value.call.call_args[0][0]
        assert call_kw["url"] == "https://example.com"
