"""Tests for cat_agent.tools.image_gen."""

import json

import pytest

from cat_agent.tools.image_gen import ImageGen


class TestImageGen:

    def test_call_returns_url_with_quoted_prompt(self):
        g = ImageGen()
        out = g.call({"prompt": "a cat"})
        data = json.loads(out)
        assert "image_url" in data
        assert "prompt" in data["image_url"]
        assert "a%20cat" in data["image_url"] or "a+cat" in data["image_url"]

    def test_call_requires_prompt(self):
        g = ImageGen()
        with pytest.raises(Exception):
            g.call("{}")
