"""Tests for cat_agent.llm.mlx_lm_llm (MLX-LM backend)."""

from unittest.mock import patch

import pytest

from cat_agent.llm.schema import USER, ContentItem, Message


class TestMLXLm:

    def test_convert_messages_str_and_list_content(self):
        pytest.importorskip("mlx_lm")
        from cat_agent.llm.mlx_lm_llm import MLXLm

        # Avoid running real __init__ (which would try to load a model).
        with patch.object(MLXLm, "__init__", lambda self, cfg=None: None):
            model = MLXLm.__new__(MLXLm)
            model._convert_messages = MLXLm._convert_messages.__get__(model)

            # String content
            msgs = [Message(USER, "Hello")]
            out = model._convert_messages(msgs)
            assert out == [{"role": "user", "content": "Hello"}]

            # List-of-items content
            msgs2 = [Message(USER, [ContentItem(text="A"), ContentItem(text="B")])]
            out2 = model._convert_messages(msgs2)
            assert out2[0]["role"] == "user"
            assert isinstance(out2[0]["content"], str)
            assert len(out2[0]["content"]) > 0

    def test_prepare_generate_kwargs_basic_and_sampler(self):
        pytest.importorskip("mlx_lm")
        from cat_agent.llm.mlx_lm_llm import MLXLm

        with patch.object(MLXLm, "__init__", lambda self, cfg=None: None):
            model = MLXLm.__new__(MLXLm)
            model._prepare_generate_kwargs = MLXLm._prepare_generate_kwargs.__get__(model)

            # Basic max_tokens conversion
            out = model._prepare_generate_kwargs({"max_new_tokens": 64})
            assert out["max_tokens"] == 64

            # Temperature/top_p create a sampler
            out2 = model._prepare_generate_kwargs(
                {"temperature": 0.5, "top_p": 0.8, "max_new_tokens": 32}
            )
            assert out2["max_tokens"] == 32
            # Sampler is a callable created by mlx_lm.sample_utils.make_sampler
            assert "sampler" in out2
            assert callable(out2["sampler"])

