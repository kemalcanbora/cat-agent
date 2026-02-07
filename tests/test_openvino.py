"""Tests for cat_agent.llm.openvino."""

import pytest

from cat_agent.llm.openvino import OpenVINO


class TestOpenVinoLlm:

    def test_init_requires_ov_model_dir(self):
        with pytest.raises(ValueError, match="ov_model_dir"):
            OpenVINO({})
