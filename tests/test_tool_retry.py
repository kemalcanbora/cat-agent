"""Tests for opt-in per-tool retry."""

from __future__ import annotations

import asyncio
import json
from typing import List, Union
from unittest.mock import patch

import pytest

from cat_agent.agents.fncall_agent import FnCallAgent
from cat_agent.llm.schema import ASSISTANT, FUNCTION, FunctionCall, Message
from cat_agent.observability.events import EventEnvelope
from cat_agent.tools.base import BaseTool, ToolServiceError
from cat_agent.tools.retry import ToolRetryConfig, retry_config_for_tool
from cat_agent.utils.backoff import compute_backoff_delay


class _FlakyTool(BaseTool):
    name = 'flaky'
    description = 'Fails then succeeds'
    parameters = {'type': 'object', 'properties': {}, 'required': []}

    def __init__(self, cfg=None, fail_times: int = 1):
        super().__init__(cfg)
        self.fail_times = fail_times
        self.calls = 0

    def call(self, params: Union[str, dict], **kwargs) -> str:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError(f'fail-{self.calls}')
        return f'ok-{self.calls}'


class _HardTool(BaseTool):
    name = 'hard'
    description = 'Hard fail'
    parameters = {'type': 'object', 'properties': {}, 'required': []}

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.calls = 0

    def call(self, params: Union[str, dict], **kwargs) -> str:
        self.calls += 1
        raise ToolServiceError(message='hard-fail')


class _AlwaysFailTool(BaseTool):
    name = 'always_fail'
    description = 'Always fails'
    parameters = {'type': 'object', 'properties': {}, 'required': []}

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.calls = 0

    def call(self, params: Union[str, dict], **kwargs) -> str:
        self.calls += 1
        raise RuntimeError(f'fail-{self.calls}')


class _FakeLLM:
    def __init__(self, tool_name: str):
        self.model = 'fake'
        self._tool_name = tool_name
        self._turn = 0

    def _next(self) -> List[Message]:
        self._turn += 1
        if self._turn == 1:
            return [
                Message(
                    role=ASSISTANT,
                    content='',
                    function_call=FunctionCall(name=self._tool_name, arguments='{}'),
                    extra={'function_id': '1'},
                )
            ]
        return [Message(role=ASSISTANT, content='done')]

    def chat(self, messages, functions=None, stream=True, delta_stream=False, extra_generate_cfg=None):
        out = self._next()
        return iter([out]) if stream else out

    async def achat(self, messages, functions=None, extra_generate_cfg=None, **kwargs):
        return self._next()


class _CollectHandler:
    def __init__(self):
        self.events: List[EventEnvelope] = []

    def on_event(self, event: EventEnvelope) -> None:
        self.events.append(event)


class TestBackoff:
    def test_compute_backoff_grows_and_caps(self):
        d = compute_backoff_delay(1.0, exponential_base=2.0, max_delay=5.0, jitter=False)
        assert d == 2.0
        d2 = compute_backoff_delay(4.0, exponential_base=2.0, max_delay=5.0, jitter=False)
        assert d2 == 5.0


class TestToolRetryConfig:
    def test_from_cfg_none_when_absent(self):
        tool = _FlakyTool({})
        assert retry_config_for_tool(tool) is None

    def test_from_cfg_none_when_max_attempts_one(self):
        tool = _FlakyTool({'retry': {'max_attempts': 1}})
        assert retry_config_for_tool(tool) is None

    def test_from_cfg_parses(self):
        tool = _FlakyTool({
            'retry': {
                'max_attempts': 3,
                'retryable_exceptions': ['ConnectionError'],
                'initial_delay': 0.01,
            }
        })
        cfg = retry_config_for_tool(tool)
        assert cfg is not None
        assert cfg.max_attempts == 3
        assert cfg.retryable_exceptions == (ConnectionError,)


class TestToolRetryBehavior:
    def test_retry_off_by_default_single_call(self):
        tool = _FlakyTool({}, fail_times=1)
        agent = FnCallAgent(llm=_FakeLLM('flaky'), function_list=[tool])
        result = agent._call_tool('flaky', '{}')
        assert tool.calls == 1
        assert 'ConnectionError' in result or 'fail-1' in result

    def test_succeeds_on_second_attempt(self):
        tool = _FlakyTool({
            'retry': {'max_attempts': 3, 'initial_delay': 0.001, 'exponential_base': 1.0},
        }, fail_times=1)
        with patch('time.sleep'):
            agent = FnCallAgent(llm=_FakeLLM('flaky'), function_list=[tool])
            result = agent._call_tool('flaky', '{}')
        assert tool.calls == 2
        assert result == 'ok-2'

    def test_exhausted_returns_error_string(self):
        tool = _AlwaysFailTool({
            'retry': {'max_attempts': 3, 'initial_delay': 0.001},
        })
        with patch('time.sleep'):
            agent = FnCallAgent(llm=_FakeLLM('always_fail'), function_list=[tool])
            result = agent._call_tool('always_fail', '{}')
        assert tool.calls == 3
        assert isinstance(result, str)
        assert 'RuntimeError' in result
        assert 'fail-3' in result

    def test_hard_tool_service_error_not_retried(self):
        tool = _HardTool({'retry': {'max_attempts': 5, 'initial_delay': 0.001}})
        agent = FnCallAgent(llm=_FakeLLM('hard'), function_list=[tool])
        with pytest.raises(ToolServiceError, match='hard-fail'):
            agent._call_tool('hard', '{}')
        assert tool.calls == 1

    def test_non_retryable_exception_type_not_retried(self):
        tool = _AlwaysFailTool({
            'retry': {
                'max_attempts': 5,
                'retryable_exceptions': ['ConnectionError'],
                'initial_delay': 0.001,
            },
        })
        agent = FnCallAgent(llm=_FakeLLM('always_fail'), function_list=[tool])
        result = agent._call_tool('always_fail', '{}')
        assert tool.calls == 1
        assert 'RuntimeError' in result

    def test_retry_events_not_in_message_history(self):
        tool = _FlakyTool({
            'retry': {'max_attempts': 3, 'initial_delay': 0.001},
        }, fail_times=1)
        handler = _CollectHandler()
        with patch('time.sleep'):
            agent = FnCallAgent(
                llm=_FakeLLM('flaky'),
                function_list=[tool],
                handlers=[handler],
            )
            responses = list(agent.run([Message(role='user', content='go')]))
        retry_events = [e for e in handler.events if e.event_type == 'tool.retry']
        assert len(retry_events) == 1
        assert retry_events[0].payload['attempt'] == 1
        # Final response has exactly one function observation (not one per attempt)
        final = responses[-1]
        function_msgs = [
            m for m in final
            if (m.get('role') if isinstance(m, dict) else m.role) == FUNCTION
        ]
        assert len(function_msgs) == 1
        content = function_msgs[0].get('content') if isinstance(function_msgs[0], dict) else function_msgs[0].content
        assert content == 'ok-2'

    @pytest.mark.asyncio
    async def test_async_retry_and_cancellation_during_backoff(self):
        tool = _AlwaysFailTool({
            'retry': {'max_attempts': 5, 'initial_delay': 60.0},
        })
        sleep_started = asyncio.Event()

        async def _slow_sleep(_delay):
            sleep_started.set()
            await asyncio.Future()  # park until cancelled

        agent = FnCallAgent(llm=_FakeLLM('always_fail'), function_list=[tool])
        with patch('asyncio.sleep', side_effect=_slow_sleep):
            task = asyncio.create_task(agent._acall_tool('always_fail', '{}'))
            await sleep_started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        # First attempt failed; cancelled while sleeping before retry 2
        assert tool.calls == 1

    def test_unconfigured_matches_direct_call_count(self):
        """Zero-change: no retry cfg ⇒ exactly one underlying call (same as pre-feature)."""
        tool = _FlakyTool({}, fail_times=0)
        agent = FnCallAgent(llm=_FakeLLM('flaky'), function_list=[tool])
        assert agent._call_tool('flaky', '{}') == 'ok-1'
        assert tool.calls == 1
