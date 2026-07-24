"""Tests for tool attempt_timeout, run_timeout, and code_interpreter cfg timeout."""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import List, Optional, Union
from unittest.mock import patch

import pytest

from cat_agent.agents.fncall_agent import FnCallAgent
from cat_agent.llm.schema import ASSISTANT, FUNCTION, FunctionCall, Message
from cat_agent.tools.base import BaseTool
from cat_agent.tools.timeout import attempt_timeout_for_tool, prepare_tool_call_kwargs


class _SlowAsyncTool(BaseTool):
    name = 'slow'
    description = 'Sleeps'
    parameters = {'type': 'object', 'properties': {'n': {'type': 'number'}}, 'required': ['n']}

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.calls = 0
        self.finished = 0

    def call(self, params: Union[str, dict], **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else params
        import time
        self.calls += 1
        time.sleep(float(data['n']))
        self.finished += 1
        return f'done:{data["n"]}'

    async def acall(self, params: Union[str, dict], **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else params
        self.calls += 1
        await asyncio.sleep(float(data['n']))
        self.finished += 1
        return f'done:{data["n"]}'


class _TimeoutAwareTool(BaseTool):
    name = 'timeout_aware'
    description = 'Records timeout kwarg'
    parameters = {'type': 'object', 'properties': {}, 'required': []}

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.seen_timeout: Optional[int] = None

    def call(self, params: Union[str, dict], timeout: Optional[int] = None, **kwargs) -> str:
        self.seen_timeout = timeout
        return f'timeout={timeout}'


class _FastTool(BaseTool):
    name = 'fast'
    description = 'Fast'
    parameters = {'type': 'object', 'properties': {}, 'required': []}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        return 'fast-ok'

    async def acall(self, params: Union[str, dict], **kwargs) -> str:
        return 'fast-ok'


class _FakeLLM:
    def __init__(self, tool_calls: List[tuple[str, dict]], final_text: str = 'done'):
        self.model = 'fake'
        self._tool_calls = tool_calls
        self._final_text = final_text
        self._turn = 0

    def _next(self) -> List[Message]:
        self._turn += 1
        if self._turn == 1:
            return [
                Message(
                    role=ASSISTANT,
                    content='',
                    function_call=FunctionCall(name=name, arguments=json.dumps(args)),
                    extra={'function_id': str(i + 1)},
                )
                for i, (name, args) in enumerate(self._tool_calls)
            ]
        return [Message(role=ASSISTANT, content=self._final_text)]

    def chat(self, messages, functions=None, stream=True, delta_stream=False, extra_generate_cfg=None):
        out = self._next()
        return iter([out]) if stream else out

    async def achat(self, messages, functions=None, extra_generate_cfg=None, **kwargs):
        return self._next()


class TestAttemptTimeout:
    @pytest.mark.asyncio
    async def test_timeout_returns_error_string_not_exception(self):
        tool = _SlowAsyncTool({'attempt_timeout': 0.05})
        agent = FnCallAgent(llm=_FakeLLM([('slow', {'n': 5})]), function_list=[tool])
        result = await agent._acall_tool('slow', json.dumps({'n': 5}))
        assert isinstance(result, str)
        assert 'TimeoutError' in result
        assert 'attempt_timeout' in result
        assert tool.finished == 0

    @pytest.mark.asyncio
    async def test_one_tool_timeout_leaves_siblings_ordered(self):
        slow = _SlowAsyncTool({'attempt_timeout': 0.05})
        slow.name = 'slow'
        fast = _FastTool()
        llm = _FakeLLM([
            ('slow', {'n': 5}),
            ('fast', {}),
        ])
        agent = FnCallAgent(llm=llm, function_list=[slow, fast])
        responses = []
        async for rsp in agent.arun([Message(role='user', content='go')]):
            responses.append(rsp)
        final = responses[-1]
        # Collect function results in order
        fn_msgs = [m for m in final if (m.role if not isinstance(m, dict) else m['role']) == FUNCTION]
        assert len(fn_msgs) == 2
        names = [m.name if not isinstance(m, dict) else m['name'] for m in fn_msgs]
        assert names == ['slow', 'fast']
        contents = [m.content if not isinstance(m, dict) else m['content'] for m in fn_msgs]
        assert 'TimeoutError' in contents[0]
        assert contents[1] == 'fast-ok'

    def test_sync_attempt_timeout_warns_and_forwards_tool_timeout(self):
        tool = _TimeoutAwareTool({'attempt_timeout': 7})
        agent = FnCallAgent(llm=_FakeLLM([('timeout_aware', {})]), function_list=[tool])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = agent._call_tool('timeout_aware', '{}')
        assert any('not enforceable on the sync tool path' in str(w.message) for w in caught)
        assert tool.seen_timeout == 7
        assert result == 'timeout=7'

    @pytest.mark.asyncio
    async def test_run_timeout_tightens_tool_wait(self):
        tool = _SlowAsyncTool()  # no per-tool attempt_timeout
        agent = FnCallAgent(llm=_FakeLLM([('slow', {'n': 5})]), function_list=[tool])
        result = None
        async for rsp in agent.arun([Message(role='user', content='go')], run_timeout=0.05):
            result = rsp
        assert result is not None
        fn_msgs = [m for m in result if (m.role if not isinstance(m, dict) else m['role']) == FUNCTION]
        assert len(fn_msgs) == 1
        content = fn_msgs[0].content if not isinstance(fn_msgs[0], dict) else fn_msgs[0]['content']
        assert 'TimeoutError' in content

    def test_sync_run_timeout_warns(self):
        tool = _FastTool()
        agent = FnCallAgent(llm=_FakeLLM([('fast', {})]), function_list=[tool])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            list(agent.run([Message(role='user', content='go')], run_timeout=1.0))
        assert any('run_timeout is not enforceable' in str(w.message) for w in caught)

    def test_unconfigured_timeout_no_wait_for(self):
        tool = _FastTool()
        agent = FnCallAgent(llm=_FakeLLM([('fast', {})]), function_list=[tool])
        with patch('asyncio.wait_for', side_effect=AssertionError('wait_for should not run')):
            # sync path never uses wait_for
            assert agent._call_tool('fast', '{}') == 'fast-ok'

    @pytest.mark.asyncio
    async def test_unconfigured_async_no_wait_for(self):
        tool = _FastTool()
        agent = FnCallAgent(llm=_FakeLLM([('fast', {})]), function_list=[tool])

        real_wait_for = asyncio.wait_for

        async def _guarded(coro, timeout=None):
            raise AssertionError(f'wait_for should not run when unconfigured, timeout={timeout}')

        with patch('asyncio.wait_for', side_effect=_guarded):
            assert await agent._acall_tool('fast', '{}') == 'fast-ok'


class TestCodeInterpreterCfgTimeout:
    def test_cfg_timeout_used_when_kwarg_absent(self):
        from cat_agent.tools import code_interpreter as ci_mod
        from cat_agent.tools.code_interpreter import CodeInterpreter
        from cat_agent.tools.base import BaseToolWithFileAccess

        with patch('cat_agent.tools.code_interpreter._check_docker_availability'):
            with patch('cat_agent.tools.code_interpreter._check_host_deps'):
                ci = CodeInterpreter({'timeout': 10})
        codes = []

        def fake_execute(kc, code):
            codes.append(code)
            return 'ok'

        import os
        with patch.object(BaseToolWithFileAccess, 'call', lambda self, params=None, files=None, **kw: None):
            with patch.object(ci, '_execute_code', side_effect=fake_execute):
                with patch.object(ci, 'instance_id', 'cfgto'):
                    with patch.dict(ci_mod._KERNEL_CLIENTS, {f'cfgto_{os.getpid()}': object()}, clear=False):
                        result = ci.call('{"code": "print(1)"}')
        assert any('_M6CountdownTimer.start(10)' in c for c in codes)
        assert result == 'ok'

    def test_explicit_timeout_kwarg_overrides_cfg(self):
        from cat_agent.tools import code_interpreter as ci_mod
        from cat_agent.tools.code_interpreter import CodeInterpreter
        from cat_agent.tools.base import BaseToolWithFileAccess

        with patch('cat_agent.tools.code_interpreter._check_docker_availability'):
            with patch('cat_agent.tools.code_interpreter._check_host_deps'):
                ci = CodeInterpreter({'timeout': 10})
        codes = []

        def fake_execute(kc, code):
            codes.append(code)
            return 'ok'

        import os
        with patch.object(BaseToolWithFileAccess, 'call', lambda self, params=None, files=None, **kw: None):
            with patch.object(ci, '_execute_code', side_effect=fake_execute):
                with patch.object(ci, 'instance_id', 'cfgto2'):
                    with patch.dict(ci_mod._KERNEL_CLIENTS, {f'cfgto2_{os.getpid()}': object()}, clear=False):
                        ci.call('{"code": "print(1)"}', timeout=3)
        assert any('_M6CountdownTimer.start(3)' in c for c in codes)

    def test_attempt_timeout_forwards_to_timeout_param(self):
        tool = _TimeoutAwareTool({'attempt_timeout': 12})
        assert attempt_timeout_for_tool(tool) == 12.0
        kwargs = prepare_tool_call_kwargs(tool, {}, 12.0)
        assert kwargs['timeout'] == 12
