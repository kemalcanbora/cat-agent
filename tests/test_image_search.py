"""Tests for cat_agent.tools.image_search."""

import pytest

from cat_agent.tools.image_search import ImageSearch


class TestImageSearch:

    def test_call_no_images_returns_error(self):
        s = ImageSearch()
        out = s.call({"img_idx": 0}, messages=[])
        assert "no images" in out.lower() or "Error" in out

    def test_call_requires_img_idx(self):
        s = ImageSearch()
        with pytest.raises(Exception):
            s.call("{}", messages=[])
