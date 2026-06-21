"""Tests for cat_agent.observability."""

from unittest.mock import MagicMock, patch

import pytest

from cat_agent.agent import BasicAgent
from cat_agent.agents.fncall_agent import FnCallAgent
from cat_agent.llm.schema import ASSISTANT, USER, FunctionCall, Message
from cat_agent.observability import CallbackHandler, clear_handlers
from cat_agent.observability.events import EventEnvelope
from cat_agent.tools import BaseTool


class CollectingHandler:
    def __init__(self):
        self.events: list[EventEnvelope] = []

    def on_event(self, event: EventEnvelope) -> None:
        self.events.append(event)


def _make_mock_llm():
    llm = MagicMock()
    llm.model = 'test-model'
    llm.chat = MagicMock(return_value=iter([]))
    return llm


@pytest.fixture(autouse=True)
def _clear_global_handlers():
    clear_handlers()
    yield
    clear_handlers()


class TestObservabilityRunEvents:

    def test_run_start_and_end_emitted(self):
        handler = CollectingHandler()
        llm = _make_mock_llm()
        llm.chat.return_value = iter([[Message(role=ASSISTANT, content='Hi')]])
        agent = BasicAgent(llm=llm, name='bot', handlers=[handler])
        list(agent.run([Message(role=USER, content='Hello')]))
        types = [e.event_type for e in handler.events]
        assert types[0] == 'run.start'
        assert types[-1] == 'run.end'
        assert handler.events[0].trace_id == handler.events[-1].trace_id
        assert handler.events[0].agent_name == 'bot'
        assert handler.events[0].payload['message_count'] >= 1

    def test_no_events_without_handlers(self):
        handler = CollectingHandler()
        llm = _make_mock_llm()
        llm.chat.return_value = iter([[Message(role=ASSISTANT, content='Hi')]])
        agent = BasicAgent(llm=llm, handlers=[])
        list(agent.run([Message(role=USER, content='Hello')]))
        assert handler.events == []

    def test_per_run_handlers_override_agent_handlers(self):
        agent_handler = CollectingHandler()
        run_handler = CollectingHandler()
        llm = _make_mock_llm()
        llm.chat.return_value = iter([[Message(role=ASSISTANT, content='Hi')]])
        agent = BasicAgent(llm=llm, handlers=[agent_handler])
        list(agent.run([Message(role=USER, content='Hello')], handlers=[run_handler]))
        assert agent_handler.events == []
        assert run_handler.events


class TestObservabilityLlmEvents:

    def test_llm_start_and_end_emitted(self):
        handler = CollectingHandler()
        llm = _make_mock_llm()
        llm.chat.return_value = iter([
            [Message(role=ASSISTANT, content='partial')],
            [Message(role=ASSISTANT, content='final')],
        ])
        agent = BasicAgent(llm=llm, handlers=[handler])
        list(agent.run([Message(role=USER, content='Hello')]))
        types = [e.event_type for e in handler.events]
        assert 'llm.start' in types
        assert 'llm.end' in types
        llm_end = next(e for e in handler.events if e.event_type == 'llm.end')
        assert llm_end.payload['model'] == 'test-model'
        assert llm_end.payload['chunk_count'] == 2

    def test_llm_chunk_only_when_enabled(self):
        handler = CollectingHandler()
        llm = _make_mock_llm()
        llm.chat.return_value = iter([
            [Message(role=ASSISTANT, content='a')],
            [Message(role=ASSISTANT, content='ab')],
        ])
        agent = BasicAgent(llm=llm, handlers=[handler])
        list(agent.run([Message(role=USER, content='Hello')], emit_stream_chunks=True))
        assert any(e.event_type == 'llm.chunk' for e in handler.events)

        handler.events.clear()
        list(agent.run([Message(role=USER, content='Hello')]))
        assert not any(e.event_type == 'llm.chunk' for e in handler.events)


class TestObservabilityToolEvents:

    def test_tool_start_and_end_emitted(self):
        handler = CollectingHandler()
        llm = _make_mock_llm()
        tool_call = Message(
            role=ASSISTANT,
            content='',
            function_call=FunctionCall(name='my_tool', arguments='{"x": 1}'),
            extra={'function_id': '1'},
        )
        final = Message(role=ASSISTANT, content='done')
        llm.chat.side_effect = [
            iter([[tool_call]]),
            iter([[final]]),
        ]

        tool = MagicMock(spec=BaseTool)
        tool.name = 'my_tool'
        tool.function = {'name': 'my_tool', 'description': 'test', 'parameters': {}}
        tool.file_access = False
        tool.call.return_value = 'tool-output'

        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            agent = FnCallAgent(llm=llm, function_list=[tool], handlers=[handler])
            list(agent.run([Message(role=USER, content='compute')]))

        types = [e.event_type for e in handler.events]
        assert types.count('tool.start') == 1
        assert types.count('tool.end') == 1
        tool_end = next(e for e in handler.events if e.event_type == 'tool.end')
        assert tool_end.payload['tool_name'] == 'my_tool'
        assert tool_end.payload['success'] is True
        assert tool_end.payload['result_chars'] == len('tool-output')

    def test_missing_tool_marks_failure(self):
        handler = CollectingHandler()
        llm = _make_mock_llm()

        class ToolProbeAgent(BasicAgent):
            def _run(self, messages, lang='en', **kwargs):
                self._call_tool('missing_tool', '{}')
                yield [Message(role=ASSISTANT, content='ok')]

        agent = ToolProbeAgent(llm=llm, handlers=[handler])
        list(agent.run([Message(role=USER, content='Hello')]))
        tool_end = next(e for e in handler.events if e.event_type == 'tool.end')
        assert tool_end.payload['success'] is False


class TestCallbackHandler:

    def test_callback_handler_invoked(self):
        seen = []
        handler = CallbackHandler(lambda event: seen.append(event.event_type))
        llm = _make_mock_llm()
        llm.chat.return_value = iter([[Message(role=ASSISTANT, content='Hi')]])
        agent = BasicAgent(llm=llm, handlers=[handler])
        list(agent.run([Message(role=USER, content='Hello')]))
        assert 'run.start' in seen
        assert 'llm.start' in seen
        assert 'run.end' in seen


class TestEventSummary:

    def test_summary_includes_type_and_payload(self):
        from cat_agent.observability.events import AgentEvent

        event = AgentEvent.llm_end(
            trace_id='trace-1',
            run_id='run-1',
            span_id='span-1',
            parent_span_id=None,
            agent_name='bot',
            agent_class='BasicAgent',
            duration_ms=12.5,
            model='test-model',
            has_tool_call=False,
            usage=None,
            chunk_count=1,
        )
        text = event.summary()
        assert 'llm.end' in text
        assert 'trace=trace-1' in text
        assert 'model=test-model' in text
        assert str(event) == text
