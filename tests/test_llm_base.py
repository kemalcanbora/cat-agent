"""Tests for cat_agent.llm.base (LLM_REGISTRY, register_llm, ModelServiceError, BaseChatModel, _truncate_input_messages_roughly)."""

import copy
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.llm.base import (
    LLM_REGISTRY,
    ModelServiceError,
    BaseChatModel,
    register_llm,
    _truncate_input_messages_roughly,
)


# ---------------------------------------------------------------------------
# register_llm / LLM_REGISTRY
# ---------------------------------------------------------------------------

class TestRegisterLlm:

    def test_register_llm_adds_to_registry(self):
        before = set(LLM_REGISTRY.keys())
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

    def test_quick_chat_returns_text(self):
        m = _ConcreteChatModel(cfg={})
        out = m.quick_chat("Hello")
        assert out == "Hi"

    def test_chat_empty_messages_raises(self):
        m = _ConcreteChatModel(cfg={})
        with pytest.raises(ValueError, match="can not be empty"):
            list(m.chat(messages=[], stream=True))


# ---------------------------------------------------------------------------
# _truncate_input_messages_roughly
# ---------------------------------------------------------------------------

class TestTruncateInputMessagesRoughly:

    def test_empty_messages_returns_empty(self):
        result = _truncate_input_messages_roughly([], max_tokens=1000)
        assert result == []

    def test_two_system_messages_raises(self):
        messages = [
            Message(role=SYSTEM, content="First"),
            Message(role=SYSTEM, content="Second"),
        ]
        with pytest.raises(ModelServiceError, match="no more than one system"):
            _truncate_input_messages_roughly(messages, max_tokens=10000)

    def test_first_message_assistant_raises(self):
        messages = [
            Message(role=ASSISTANT, content="Reply"),
        ]
        with pytest.raises(ModelServiceError, match="start with a user message"):
            _truncate_input_messages_roughly(messages, max_tokens=10000)

    def test_system_plus_user_under_limit_returns_unchanged(self):
        messages = [
            Message(role=SYSTEM, content="You are helpful."),
            Message(role=USER, content="Hi"),
            Message(role=ASSISTANT, content="Hello"),
        ]
        # Use a very large max_tokens so we don't actually truncate (avoids complex token counting in test)
        result = _truncate_input_messages_roughly(messages, max_tokens=1_000_000)
        assert len(result) == 3
        assert result[0].role == SYSTEM
        assert result[1].content == "Hi"
        assert result[2].content == "Hello"
