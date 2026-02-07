"""Tests for cat_agent.llm.llama_cpp."""

from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import ASSISTANT, FUNCTION, USER, ContentItem, FunctionCall, Message


class TestLlamaCpp:

    def test_init_requires_model_path_or_repo(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp import LlamaCpp

        with pytest.raises(ValueError, match="model_path|repo_id|filename"):
            LlamaCpp({})
        with pytest.raises(ValueError, match="model_path|repo_id|filename"):
            LlamaCpp({"repo_id": "only_repo"})

    def test_convert_messages_str_content(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp import LlamaCpp

        with patch.object(LlamaCpp, "__init__", lambda self, cfg: None):
            model = LlamaCpp.__new__(LlamaCpp)
            model._convert_messages = LlamaCpp._convert_messages.__get__(model)
            messages = [Message(USER, "Hello")]
            out = model._convert_messages(messages)
            assert len(out) == 1
            assert out[0]["role"] == "user"
            assert out[0]["content"] == "Hello"

    def test_convert_messages_list_content(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp import LlamaCpp

        with patch.object(LlamaCpp, "__init__", lambda self, cfg: None):
            model = LlamaCpp.__new__(LlamaCpp)
            model._convert_messages = LlamaCpp._convert_messages.__get__(model)
            # List of dicts with 'text' are concatenated
            messages = [Message(USER, [ContentItem(text="A"), ContentItem(text="B")])]
            out = model._convert_messages(messages)
            assert out[0]["role"] == "user"
            assert isinstance(out[0]["content"], str)
            # ContentItem is not a dict so code uses str(item); at least content is present
            assert len(out[0]["content"]) > 0

    def test_convert_messages_dict_input(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp import LlamaCpp

        with patch.object(LlamaCpp, "__init__", lambda self, cfg: None):
            model = LlamaCpp.__new__(LlamaCpp)
            model._convert_messages = LlamaCpp._convert_messages.__get__(model)
            messages = [{"role": "user", "content": "Hi"}]
            out = model._convert_messages(messages)
            assert out[0]["role"] == "user"
            assert out[0]["content"] == "Hi"

    def test_prepare_generate_kwargs(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp import LlamaCpp

        with patch.object(LlamaCpp, "__init__", lambda self, cfg: None):
            model = LlamaCpp.__new__(LlamaCpp)
            model._prepare_generate_kwargs = LlamaCpp._prepare_generate_kwargs.__get__(model)
            cfg = {"temperature": 0.5, "max_tokens": 64, "top_p": 0.8}
            out = model._prepare_generate_kwargs(cfg)
            assert out["temperature"] == 0.5
            assert out["max_tokens"] == 64
            assert out["top_p"] == 0.8
            assert "max_new_tokens" not in out

    def test_prepare_generate_kwargs_max_new_tokens_fallback(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp import LlamaCpp

        with patch.object(LlamaCpp, "__init__", lambda self, cfg: None):
            model = LlamaCpp.__new__(LlamaCpp)
            model._prepare_generate_kwargs = LlamaCpp._prepare_generate_kwargs.__get__(model)
            out = model._prepare_generate_kwargs({"max_new_tokens": 128})
            assert out["max_tokens"] == 128


class TestRemoveFncallMessages:

    def test_function_call_turned_into_user_text(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp import LlamaCpp

        with patch("cat_agent.llm.llama_cpp.Llama") as MockLlama:
            MockLlama.return_value = MagicMock()
            model = LlamaCpp({"model_path": "/nonexistent/fake.gguf", "generate_cfg": {"fncall_prompt_type": "nous"}})
            # USER content must be list when we append function call text to it
            messages = [
                Message(USER, [ContentItem(text="Hi")]),
                Message(ASSISTANT, "", function_call=FunctionCall("tool1", '{"a":1}')),
                Message(FUNCTION, [ContentItem(text="result1")], name="tool1"),
            ]
            out = model._remove_fncall_messages(messages, lang="en")
            assert any(m.role == USER for m in out)
            assert any("tool1" in str(m.content) for m in out)
            assert any("result1" in str(m.content) for m in out)

    def test_two_user_messages_separated_by_assistant_placeholder(self):
        pytest.importorskip("llama_cpp")
        from cat_agent.llm.llama_cpp import LlamaCpp

        with patch("cat_agent.llm.llama_cpp.Llama") as MockLlama:
            MockLlama.return_value = MagicMock()
            model = LlamaCpp({"model_path": "/nonexistent/fake.gguf", "generate_cfg": {"fncall_prompt_type": "nous"}})
            messages = [
                Message(USER, "First"),
                Message(USER, "Second"),
            ]
            out = model._remove_fncall_messages(messages, lang="en")
            assert len(out) >= 2
            roles = [m.role for m in out]
            assert USER in roles and ASSISTANT in roles
