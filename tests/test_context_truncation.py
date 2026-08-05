"""Tests for context truncation visibility and silent num_ctx warnings."""

from unittest.mock import MagicMock, patch

import pytest

from cat_agent.agent import BasicAgent
from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.observability import clear_handlers
from cat_agent.observability.context import run_context
from cat_agent.observability.events import EventEnvelope


class CollectingHandler:
    def __init__(self):
        self.events: list[EventEnvelope] = []

    def on_event(self, event: EventEnvelope) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def _clear_global_handlers():
    clear_handlers()
    yield
    clear_handlers()


class TestClientSideTruncation:

    def test_truncation_events_increment(self):
        pytest.importorskip('cat_agent._native')
        from cat_agent.llm.base.model import BaseChatModel
        from cat_agent.utils.tokenization_qwen import ensure_qwen_tokenizer

        ensure_qwen_tokenizer()
        handler = CollectingHandler()

        class StubModel(BaseChatModel):
            def _chat_stream(self, messages, delta_stream, generate_cfg):
                yield [Message(role=ASSISTANT, content='ok')]

            def _chat_no_stream(self, messages, generate_cfg):
                return [Message(role=ASSISTANT, content='ok')]

            def _chat_with_functions(self, messages, functions, stream, delta_stream, generate_cfg, lang):
                return self._chat_stream(messages, delta_stream, generate_cfg)

        model = StubModel(cfg={'generate_cfg': {'max_input_tokens': 64}})
        with run_context(agent_name='bot', agent_class='BasicAgent', handlers=[handler]) as ctx:
            list(model.chat(
                messages=[
                    Message(role=SYSTEM, content='You are helpful.'),
                    Message(role=USER, content='word ' * 2500),
                ],
                stream=True,
            ))
            assert ctx.metrics.truncation_events >= 1
            assert ctx.metrics.max_context_ratio > 1.0

        trunc = [e for e in handler.events if e.event_type == 'context.truncated']
        assert trunc
        assert trunc[0].payload['before_tokens'] > trunc[0].payload['after_tokens']
        assert trunc[0].payload['max_input_tokens'] == 64


class TestSilentServerTruncationWarning:

    def test_warning_when_reported_below_local_estimate(self):
        from cat_agent.utils.tokenization_qwen import count_tokens, ensure_qwen_tokenizer
        from cat_agent.utils.message_utils import extract_text_from_message

        ensure_qwen_tokenizer()
        messages = [Message(role=USER, content='hello world ' * 50)]
        local = sum(
            count_tokens(extract_text_from_message(m, add_upload_info=False)) for m in messages
        )
        reported = max(1, int(local * 0.5))

        llm = MagicMock()
        llm.model = 'test-model'
        msg = Message(
            role=ASSISTANT,
            content='ok',
            extra={
                'usage': {
                    'prompt_tokens': reported,
                    'completion_tokens': 1,
                    'total_tokens': reported + 1,
                }
            },
        )
        llm.chat = MagicMock(return_value=iter([[msg]]))
        agent = BasicAgent(llm=llm, handlers=[])

        with patch('cat_agent.agent.logger.warning') as warn:
            with run_context(agent_name='bot', agent_class='BasicAgent', handlers=[]) as ctx:
                list(agent._call_llm(messages))
                assert ctx.metrics.silent_truncation_warned is True
                list(agent._call_llm(messages))
            assert warn.call_count == 1
            assert 'truncated the prompt' in str(warn.call_args)

    def test_no_warning_at_tolerance_boundary(self, monkeypatch):
        import cat_agent.agent as agent_mod
        import cat_agent.settings as settings

        monkeypatch.setattr(settings, 'PROMPT_TRUNCATION_TOLERANCE', 0.95)
        monkeypatch.setattr(agent_mod, 'PROMPT_TRUNCATION_TOLERANCE', 0.95)

        from cat_agent.utils.tokenization_qwen import count_tokens, ensure_qwen_tokenizer
        from cat_agent.utils.message_utils import extract_text_from_message

        ensure_qwen_tokenizer()
        messages = [Message(role=USER, content='hello world ' * 40)]
        local = sum(
            count_tokens(extract_text_from_message(m, add_upload_info=False)) for m in messages
        )
        # 3% below local — within the default 5% tolerance
        reported = max(1, int(local * 0.97))

        llm = MagicMock()
        llm.model = 'test-model'
        msg = Message(
            role=ASSISTANT,
            content='ok',
            extra={
                'usage': {
                    'prompt_tokens': reported,
                    'completion_tokens': 1,
                    'total_tokens': reported + 1,
                }
            },
        )
        llm.chat = MagicMock(return_value=iter([[msg]]))
        agent = BasicAgent(llm=llm, handlers=[])

        with patch('cat_agent.agent.logger.warning') as warn:
            with run_context(agent_name='bot', agent_class='BasicAgent', handlers=[]):
                list(agent._call_llm(messages))
            assert warn.call_count == 0

    def test_no_warning_when_usage_unavailable(self):
        llm = MagicMock()
        llm.model = 'test-model'
        llm.chat = MagicMock(return_value=iter([[Message(role=ASSISTANT, content='ok')]]))
        agent = BasicAgent(llm=llm, handlers=[])

        with patch('cat_agent.agent.logger.warning') as warn:
            with run_context(agent_name='bot', agent_class='BasicAgent', handlers=[]) as ctx:
                list(agent._call_llm([Message(role=USER, content='hi')]))
                assert ctx.metrics.usage_available is False
                assert ctx.metrics.total_tokens == 0
            assert warn.call_count == 0
