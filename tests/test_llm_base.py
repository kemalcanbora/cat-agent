"""Tests for cat_agent.llm.base (LLM_REGISTRY, register_llm, ModelServiceError, BaseChatModel, truncate_input_messages_roughly)."""

from unittest.mock import patch

import pytest

from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.llm.base import (
    LLM_REGISTRY,
    ModelServiceError,
    BaseChatModel,
    register_llm,
    truncate_input_messages_roughly,
)


# ---------------------------------------------------------------------------
# register_llm / LLM_REGISTRY
# ---------------------------------------------------------------------------

class TestRegisterLlm:

    def test_register_llm_adds_to_registry(self):
        try:
            @register_llm("_test_fake_model")
            class FakeModel(BaseChatModel):
                def _chat_with_functions(self, *args, **kwargs): raise NotImplementedError
                def _chat_stream(self, *args, **kwargs): raise NotImplementedError
                def _chat_no_stream(self, *args, **kwargs): return [Message(ASSISTANT, "ok")]

            assert "_test_fake_model" in LLM_REGISTRY
            assert LLM_REGISTRY["_test_fake_model"] is FakeModel
        finally:
            for k in list(LLM_REGISTRY.keys()):
                if k.startswith("_test_"):
                    del LLM_REGISTRY[k]


# ---------------------------------------------------------------------------
# ModelServiceError
# ---------------------------------------------------------------------------

class TestModelServiceError:

    def test_init_with_exception(self):
        exc = ValueError("bad")
        e = ModelServiceError(exception=exc)
        assert str(e) == "bad"
        assert e.exception is exc
        assert e.code is None
        assert e.message is None
        assert e.extra is None

    def test_init_with_code_and_message(self):
        e = ModelServiceError(code="500", message="Internal error")
        assert "500" in str(e)
        assert "Internal error" in str(e)
        assert e.exception is None
        assert e.code == "500"
        assert e.message == "Internal error"

    def test_init_with_extra(self):
        e = ModelServiceError(code="400", message="Bad request", extra={"key": "value"})
        assert e.extra == {"key": "value"}


# ---------------------------------------------------------------------------
# BaseChatModel
# ---------------------------------------------------------------------------

class _ConcreteChatModel(BaseChatModel):
    """Minimal concrete implementation for testing."""

    def _chat_with_functions(self, messages, functions, stream, delta_stream, generate_cfg, lang):
        raise NotImplementedError

    def _chat_stream(self, messages, delta_stream, generate_cfg):
        yield [Message(role=ASSISTANT, content="Hi")]

    def _chat_no_stream(self, messages, generate_cfg):
        return [Message(role=ASSISTANT, content="Hi")]


class TestBaseChatModel:

    def test_support_multimodal_default_false(self):
        assert _ConcreteChatModel(cfg={}).support_multimodal_input is False
        assert _ConcreteChatModel(cfg={}).support_multimodal_output is False
        assert _ConcreteChatModel(cfg={}).support_audio_input is False

    def test_init_stores_model_and_generate_cfg(self):
        m = _ConcreteChatModel(cfg={"model": "test-model", "generate_cfg": {"temperature": 0.7}})
        assert m.model == "test-model"
        assert m.generate_cfg.get("temperature") == 0.7

    def test_init_model_type_empty_by_default(self):
        m = _ConcreteChatModel(cfg={})
        assert m.model_type == ""

    def test_init_max_retries_from_generate_cfg(self):
        m = _ConcreteChatModel(cfg={"generate_cfg": {"max_retries": 3}})
        assert m.max_retries == 3

    def test_init_use_raw_api_from_env(self):
        with patch.dict("os.environ", {"CAT_AGENT_USE_RAW_API": "true"}):
            m = _ConcreteChatModel(cfg={})
        assert m.use_raw_api is True

    def test_init_use_raw_api_from_cfg_overrides_env(self):
        with patch.dict("os.environ", {"CAT_AGENT_USE_RAW_API": "true"}):
            m = _ConcreteChatModel(cfg={"generate_cfg": {"use_raw_api": False}})
        assert m.use_raw_api is False

    def test_init_use_raw_api_defaults_false_without_native_capability(self):
        import os
        env = {k: v for k, v in os.environ.items() if k != "CAT_AGENT_USE_RAW_API"}
        with patch.dict(os.environ, env, clear=True):
            m = _ConcreteChatModel(cfg={})
        assert m.supports_native_tools is False
        assert m.use_raw_api is False

    def test_quick_chat_returns_text(self):
        m = _ConcreteChatModel(cfg={})
        out = m.quick_chat("Hello")
        assert out == "Hi"

    def test_chat_empty_messages_raises(self):
        m = _ConcreteChatModel(cfg={})
        with pytest.raises(ValueError, match="can not be empty"):
            list(m.chat(messages=[], stream=True))


# ---------------------------------------------------------------------------
# truncate_input_messages_roughly
# ---------------------------------------------------------------------------

class TestTruncateInputMessagesRoughly:

    def test_empty_messages_returns_empty(self):
        result = truncate_input_messages_roughly([], max_tokens=1000)
        assert result == []

    def test_two_system_messages_raises(self):
        messages = [
            Message(role=SYSTEM, content="First"),
            Message(role=SYSTEM, content="Second"),
        ]
        with pytest.raises(ModelServiceError, match="no more than one system"):
            truncate_input_messages_roughly(messages, max_tokens=10000)

    def test_first_message_assistant_raises(self):
        messages = [
            Message(role=ASSISTANT, content="Reply"),
        ]
        with pytest.raises(ModelServiceError, match="start with a user message"):
            truncate_input_messages_roughly(messages, max_tokens=10000)

    def test_system_plus_user_under_limit_returns_unchanged(self):
        messages = [
            Message(role=SYSTEM, content="You are helpful."),
            Message(role=USER, content="Hi"),
            Message(role=ASSISTANT, content="Hello"),
        ]
        # Use a very large max_tokens so we don't actually truncate (avoids complex token counting in test)
        result = truncate_input_messages_roughly(messages, max_tokens=1_000_000)
        assert len(result) == 3
        assert result[0].role == SYSTEM
        assert result[1].content == "Hi"
        assert result[2].content == "Hello"

    def test_over_budget_truncates_via_native_path(self):
        pytest.importorskip("cat_agent._native")
        from cat_agent.utils.tokenization_qwen import count_tokens, ensure_qwen_tokenizer

        ensure_qwen_tokenizer()
        messages = [
            Message(role=SYSTEM, content="You are helpful."),
            Message(role=USER, content="Brief question."),
            Message(role=ASSISTANT, content="Brief answer."),
            Message(role=USER, content="word " * 2500),
        ]
        result = truncate_input_messages_roughly(messages, max_tokens=128)
        total = 0
        for msg in result:
            text = msg.content if isinstance(msg.content, str) else " ".join(
                item.text for item in msg.content if getattr(item, "text", None)
            )
            total += count_tokens(text)
        assert total <= 128


class TestConvCatAgentMessagesToOai:

    def test_assistant_without_content_gets_empty_string(self):
        from cat_agent.llm.base.model import BaseChatModel

        out = BaseChatModel._conv_cat_agent_messages_to_oai([
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant'},  # missing content — previously caused Ollama 400
        ])
        assert out[-1]['role'] == 'assistant'
        assert out[-1]['content'] == ''

    def test_system_user_none_content_normalised(self):
        from cat_agent.llm.base.model import BaseChatModel

        out = BaseChatModel._conv_cat_agent_messages_to_oai([
            {'role': 'system', 'content': None},
            {'role': 'user'},
        ])
        assert out[0]['content'] == ''
        assert out[1]['content'] == ''
