"""Tests for RunMetrics accumulation and nesting."""

from unittest.mock import MagicMock

import pytest

from cat_agent.agent import BasicAgent
from cat_agent.llm.schema import ASSISTANT, USER, Message
from cat_agent.observability import clear_handlers
from cat_agent.observability.context import RunMetrics, run_context
from cat_agent.observability.events import EventEnvelope


class CollectingHandler:
    def __init__(self):
        self.events: list[EventEnvelope] = []

    def on_event(self, event: EventEnvelope) -> None:
        self.events.append(event)


def _make_llm_with_usage(prompt=10, completion=5):
    llm = MagicMock()
    llm.model = 'test-model'
    msg = Message(
        role=ASSISTANT,
        content='Hi',
        extra={
            'usage': {
                'prompt_tokens': prompt,
                'completion_tokens': completion,
                'total_tokens': prompt + completion,
            }
        },
    )
    llm.chat = MagicMock(return_value=iter([[msg]]))
    return llm


@pytest.fixture(autouse=True)
def _clear_global_handlers():
    clear_handlers()
    yield
    clear_handlers()


class TestRunMetricsBasics:

    def test_tokens_per_sec_and_total(self):
        m = RunMetrics(prompt_tokens=100, completion_tokens=50, llm_ms=2000.0)
        assert m.total_tokens == 150
        assert m.tokens_per_sec == pytest.approx(25.0)

    def test_tokens_per_sec_none_without_data(self):
        assert RunMetrics().tokens_per_sec is None
        assert RunMetrics(completion_tokens=10, llm_ms=0).tokens_per_sec is None

    def test_iadd_folds_fields(self):
        a = RunMetrics(llm_calls=1, prompt_tokens=10, max_context_ratio=0.5, usage_available=True)
        b = RunMetrics(llm_calls=2, tool_calls=1, prompt_tokens=20, max_context_ratio=0.8)
        a += b
        assert a.llm_calls == 3
        assert a.tool_calls == 1
        assert a.prompt_tokens == 30
        assert a.max_context_ratio == 0.8
        assert a.usage_available is True


class TestRunMetricsAccumulation:

    def test_metrics_on_run_end_with_usage(self):
        handler = CollectingHandler()
        llm = _make_llm_with_usage(prompt=100, completion=20)
        agent = BasicAgent(llm=llm, name='bot', handlers=[handler])
        list(agent.run([Message(role=USER, content='Hello')]))
        run_end = next(e for e in handler.events if e.event_type == 'run.end')
        metrics = run_end.payload['metrics']
        assert metrics['llm_calls'] == 1
        assert metrics['prompt_tokens'] == 100
        assert metrics['completion_tokens'] == 20
        assert metrics['usage_available'] is True
        assert metrics['llm_ms'] > 0

    def test_counters_without_handlers(self):
        llm = _make_llm_with_usage(prompt=7, completion=3)
        agent = BasicAgent(llm=llm, name='bot', handlers=[])
        with run_context(agent_name='bot', agent_class='BasicAgent', handlers=[]) as ctx:
            list(agent._call_llm([Message(role=USER, content='Hi')]))
            assert ctx.metrics.llm_calls == 1
            assert ctx.metrics.prompt_tokens == 7
            assert ctx.metrics.completion_tokens == 3
            assert ctx.metrics.usage_available is True

    def test_no_usage_keeps_usage_available_false(self):
        handler = CollectingHandler()
        llm = MagicMock()
        llm.model = 'test-model'
        llm.chat = MagicMock(return_value=iter([[Message(role=ASSISTANT, content='Hi')]]))
        agent = BasicAgent(llm=llm, handlers=[handler])
        list(agent.run([Message(role=USER, content='Hello')]))
        run_end = next(e for e in handler.events if e.event_type == 'run.end')
        metrics = run_end.payload['metrics']
        assert metrics['usage_available'] is False
        assert metrics['prompt_tokens'] == 0
        assert metrics['completion_tokens'] == 0
        assert metrics['prompt_tokens'] + metrics['completion_tokens'] == 0

    def test_nested_run_folds_into_parent(self):
        parent_handler = CollectingHandler()
        child_handler = CollectingHandler()
        parent_llm = _make_llm_with_usage(prompt=10, completion=2)
        child_llm = _make_llm_with_usage(prompt=50, completion=8)

        parent = BasicAgent(llm=parent_llm, name='router', handlers=[parent_handler])
        child = BasicAgent(llm=child_llm, name='assistant', handlers=[child_handler])

        class NestedAgent(BasicAgent):
            def _run(self, messages, lang='en', **kwargs):
                list(child.run([Message(role=USER, content='sub')], handlers=[child_handler]))
                yield from super()._run(messages, lang=lang, **kwargs)

        nested = NestedAgent(llm=parent_llm, name='router', handlers=[parent_handler])
        list(nested.run([Message(role=USER, content='Hello')]))

        parent_end = next(e for e in parent_handler.events if e.event_type == 'run.end')
        child_end = next(e for e in child_handler.events if e.event_type == 'run.end')

        assert child_end.payload['metrics']['llm_calls'] == 1
        assert child_end.payload['metrics']['prompt_tokens'] == 50
        # Parent own call (10) + folded child (50)
        assert parent_end.payload['metrics']['llm_calls'] == 2
        assert parent_end.payload['metrics']['prompt_tokens'] == 60
        assert parent_end.payload['metrics']['completion_tokens'] == 10


class TestStreamOptionsFallback:

    def test_stream_options_rejected_retries_and_caches(self):
        from openai import BadRequestError
        import httpx

        from cat_agent.llm.oai import TextChatAtOAI

        model = TextChatAtOAI(cfg={'model': 'test', 'api_key': 'x', 'api_base': 'http://localhost'})
        calls = []

        class FakeUsage:
            prompt_tokens = 1
            completion_tokens = 1
            total_tokens = 2

        class FakeDelta:
            content = 'ok'
            reasoning_content = None
            tool_calls = None

        class FakeChoice:
            delta = FakeDelta()

        class FakeChunk:
            def __init__(self, choices, usage=None):
                self.choices = choices
                self.usage = usage

        def fake_create(**kwargs):
            calls.append(kwargs)
            if 'stream_options' in kwargs:
                raise BadRequestError(
                    'stream_options not supported',
                    response=httpx.Response(400, request=httpx.Request('POST', 'http://x')),
                    body=None,
                )
            return iter([
                FakeChunk([FakeChoice()]),
                FakeChunk([], usage=FakeUsage()),
            ])

        model._chat_complete_create = lambda **kwargs: fake_create(**kwargs)
        msgs = [Message(role=USER, content='hi')]
        out = list(model._chat_stream(msgs, delta_stream=False, generate_cfg={}))
        assert len(calls) == 2
        assert 'stream_options' in calls[0]
        assert 'stream_options' not in calls[1]
        assert model._supports_stream_options is False
        assert out[-1][-1].extra['usage']['prompt_tokens'] == 1

        # Second call should not retry
        calls.clear()
        list(model._chat_stream(msgs, delta_stream=False, generate_cfg={}))
        assert len(calls) == 1
        assert 'stream_options' not in calls[0]

    def test_include_usage_false_skips_stream_options(self):
        from cat_agent.llm.oai import TextChatAtOAI

        model = TextChatAtOAI(cfg={'model': 'test', 'api_key': 'x', 'api_base': 'http://localhost'})
        calls = []

        class FakeDelta:
            content = 'ok'
            reasoning_content = None
            tool_calls = None

        class FakeChoice:
            delta = FakeDelta()

        class FakeChunk:
            choices = [FakeChoice()]
            usage = None

        def fake_create(**kwargs):
            calls.append(kwargs)
            return iter([FakeChunk()])

        model._chat_complete_create = lambda **kwargs: fake_create(**kwargs)
        list(model._chat_stream(
            [Message(role=USER, content='hi')],
            delta_stream=False,
            generate_cfg={'include_usage': False},
        ))
        assert len(calls) == 1
        assert 'stream_options' not in calls[0]
