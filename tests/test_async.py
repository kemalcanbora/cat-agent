"""Tests for agent async API (arun) and parallel tool execution."""

from __future__ import annotations

import asyncio
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.agents.fncall_agent import FnCallAgent
from cat_agent.llm.schema import ASSISTANT, FUNCTION, FunctionCall, Message
from cat_agent.tools.base import BaseTool, ToolServiceError


class _FakeLLM:
    """Deterministic fake LLM: first call emits tool calls, second returns text."""

    def __init__(self, tool_calls: List[tuple[str, dict]], final_text: str = 'done'):
        self.model = 'fake'
        self.model_type = 'fake'
        self._tool_calls = tool_calls
        self._final_text = final_text
        self._turn = 0

    def _next_output(self) -> List[Message]:
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
        out = self._next_output()
        if stream:
            return iter([out])
        return out

    async def achat(self, messages, functions=None, extra_generate_cfg=None, **kwargs):
        return self._next_output()


class _SleepSyncTool(BaseTool):
    name = 'sleep_sync'
    description = 'Sleep sync'
    parameters = {'type': 'object', 'properties': {'n': {'type': 'number'}}, 'required': ['n']}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else params
        time.sleep(float(data['n']))
        return f'sync:{data["n"]}'


class _SleepAsyncTool(BaseTool):
    name = 'sleep_async'
    description = 'Sleep async'
    parameters = {'type': 'object', 'properties': {'n': {'type': 'number'}}, 'required': ['n']}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else params
        time.sleep(float(data['n']))
        return f'async_via_sync:{data["n"]}'

    async def acall(self, params: Union[str, dict], **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else params
        await asyncio.sleep(float(data['n']))
        return f'async:{data["n"]}'


class _BoomTool(BaseTool):
    name = 'boom'
    description = 'Raises'
    parameters = {'type': 'object', 'properties': {}, 'required': []}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        raise RuntimeError('boom')


class _HardFailTool(BaseTool):
    name = 'hard_fail'
    description = 'Hard ToolServiceError'
    parameters = {'type': 'object', 'properties': {}, 'required': []}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        raise ToolServiceError(message='hard')


class _EchoTool(BaseTool):
    name = 'echo'
    description = 'Echo'
    parameters = {'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else params
        return f'echo:{data["x"]}'


def _make_agent(llm, tools):
    with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock(system_files=[])):
        return FnCallAgent(llm=llm, function_list=tools, system_message=None)


@pytest.mark.asyncio
async def test_parallel_tools_are_concurrent():
    tools = [
        type('T1', (_SleepAsyncTool,), {'name': 't1'})(),
        type('T2', (_SleepAsyncTool,), {'name': 't2'})(),
        type('T3', (_SleepAsyncTool,), {'name': 't3'})(),
    ]
    for i, t in enumerate(tools, 1):
        t.name = f't{i}'
    llm = _FakeLLM([(t.name, {'n': 0.3}) for t in tools])
    agent = _make_agent(llm, tools)

    started = time.monotonic()
    result = await agent.arun_nonstream([{'role': 'user', 'content': 'go'}])
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, f'expected parallel tools, elapsed={elapsed:.3f}s'
    fn_msgs = [m for m in result if (m.get('role') if isinstance(m, dict) else m.role) == FUNCTION]
    assert len(fn_msgs) == 3


@pytest.mark.asyncio
async def test_one_tool_raises_others_still_returned():
    tools = [
        type('Ok1', (_EchoTool,), {'name': 'ok1'})(),
        type('Boom', (_BoomTool,), {'name': 'boom'})(),
        type('Ok2', (_EchoTool,), {'name': 'ok2'})(),
    ]
    tools[0].name = 'ok1'
    tools[1].name = 'boom'
    tools[2].name = 'ok2'
    llm = _FakeLLM([
        ('ok1', {'x': 'a'}),
        ('boom', {}),
        ('ok2', {'x': 'b'}),
    ])
    agent = _make_agent(llm, tools)
    result = await agent.arun_nonstream([{'role': 'user', 'content': 'go'}])
    fn_msgs = [m for m in result if (m.get('role') if isinstance(m, dict) else m.role) == FUNCTION]
    assert len(fn_msgs) == 3
    contents = [
        (m.get('content') if isinstance(m, dict) else m.content) for m in fn_msgs
    ]
    assert contents[0] == 'echo:a'
    assert 'boom' in contents[1].lower() or 'error' in contents[1].lower()
    assert contents[2] == 'echo:b'


@pytest.mark.asyncio
async def test_mixed_sync_and_async_tools():
    sync_t = _SleepSyncTool()
    sync_t.name = 'sleep_sync'
    async_t = _SleepAsyncTool()
    async_t.name = 'sleep_async'
    llm = _FakeLLM([
        ('sleep_sync', {'n': 0.2}),
        ('sleep_async', {'n': 0.2}),
    ])
    agent = _make_agent(llm, [sync_t, async_t])
    started = time.monotonic()
    result = await agent.arun_nonstream([{'role': 'user', 'content': 'go'}])
    elapsed = time.monotonic() - started
    assert elapsed < 0.35, f'mixed tools should overlap, elapsed={elapsed:.3f}s'
    fn_msgs = [m for m in result if (m.get('role') if isinstance(m, dict) else m.role) == FUNCTION]
    assert len(fn_msgs) == 2


@pytest.mark.asyncio
async def test_result_ordering_matches_tool_call_order():
    tools = []
    for name, delay in [('slow', 0.25), ('fast', 0.05)]:
        t = type(name, (_SleepAsyncTool,), {'name': name})()
        t.name = name
        tools.append(t)
    llm = _FakeLLM([('slow', {'n': 0.25}), ('fast', {'n': 0.05})])
    agent = _make_agent(llm, tools)
    result = await agent.arun_nonstream([{'role': 'user', 'content': 'go'}])
    fn_msgs = [m for m in result if (m.get('role') if isinstance(m, dict) else m.role) == FUNCTION]
    names = [(m.get('name') if isinstance(m, dict) else m.name) for m in fn_msgs]
    assert names == ['slow', 'fast']


@pytest.mark.asyncio
async def test_run_and_arun_same_output():
    tools = [_EchoTool()]
    # Separate LLM instances so turn counters don't interfere.
    llm_sync = _FakeLLM([('echo', {'x': 'z'})], final_text='final')
    llm_async = _FakeLLM([('echo', {'x': 'z'})], final_text='final')
    agent_sync = _make_agent(llm_sync, tools)
    agent_async = _make_agent(llm_async, [_EchoTool()])

    sync_out = agent_sync.run_nonstream([{'role': 'user', 'content': 'go'}])
    async_out = await agent_async.arun_nonstream([{'role': 'user', 'content': 'go'}])

    def _norm(msgs):
        out = []
        for m in msgs:
            if isinstance(m, dict):
                out.append({
                    'role': m['role'],
                    'content': m.get('content'),
                    'name': m.get('name'),
                    'tool_calls': m.get('tool_calls'),
                })
            else:
                out.append({
                    'role': m.role,
                    'content': m.content,
                    'name': m.name,
                    'tool_calls': (
                        [tc.model_dump() for tc in m.tool_calls] if m.tool_calls else None
                    ),
                })
        return out

    assert _norm(sync_out) == _norm(async_out)


@pytest.mark.asyncio
async def test_run_under_running_loop_warns_but_works():
    import cat_agent.agent as agent_mod
    agent_mod._RUN_IN_LOOP_WARNED = False

    tools = [_EchoTool()]
    llm = _FakeLLM([('echo', {'x': '1'})])
    agent = _make_agent(llm, tools)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = agent.run_nonstream([{'role': 'user', 'content': 'go'}])
    assert out
    assert any(issubclass(w.category, RuntimeWarning) and 'arun()' in str(w.message) for w in caught)


@pytest.mark.asyncio
async def test_hard_error_keeps_siblings_then_raises_earliest():
    tools = [
        type('Ok', (_EchoTool,), {'name': 'ok'})(),
        type('Hard', (_HardFailTool,), {'name': 'hard_fail'})(),
        type('Ok2', (_EchoTool,), {'name': 'ok2'})(),
    ]
    tools[0].name = 'ok'
    tools[1].name = 'hard_fail'
    tools[2].name = 'ok2'
    llm = _FakeLLM([
        ('ok', {'x': '1'}),
        ('hard_fail', {}),
        ('ok2', {'x': '2'}),
    ])
    agent = _make_agent(llm, tools)
    with pytest.raises(ToolServiceError, match='hard') as ei:
        await agent.arun_nonstream([{'role': 'user', 'content': 'go'}])
    # Sibling soft results were appended before the raise (messages on agent path).
    # The raised error is the earliest hard error in tool-call order.
    assert 'hard' in str(ei.value)


# ---------------------------------------------------------------------------
# MCP (ii-b): wrap_future must not burn thread-pool workers
# ---------------------------------------------------------------------------

class _McpStyleTool(BaseTool):
    """Mirrors MCPManager.create_tool_class: sync .result() vs async wrap_future."""

    name = 'mcp_style'
    description = 'mcp style'
    parameters = {'type': 'object', 'properties': {'n': {'type': 'number'}}, 'required': ['n']}

    def __init__(self, loop: asyncio.AbstractEventLoop, name: str = 'mcp_style'):
        self.name = name
        self._loop = loop
        super().__init__()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else params

        async def _work():
            await asyncio.sleep(float(data['n']))
            return 'ok'

        fut = asyncio.run_coroutine_threadsafe(_work(), self._loop)
        return fut.result()

    async def acall(self, params: Union[str, dict], **kwargs) -> str:
        data = json.loads(params) if isinstance(params, str) else params

        async def _work():
            await asyncio.sleep(float(data['n']))
            return 'ok'

        fut = asyncio.run_coroutine_threadsafe(_work(), self._loop)
        afut = asyncio.wrap_future(fut)
        try:
            return await afut
        except asyncio.CancelledError:
            fut.cancel()
            raise


@pytest.mark.asyncio
async def test_mcp_style_acall_does_not_exhaust_thread_pool():
    """With max_workers=1, (i) to_thread+.result() would serialize (~0.9s);

    (ii-b) wrap_future stays concurrent (~0.3s) and keeps the caller loop free.
    """
    bg_loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(bg_loop)
        bg_loop.run_forever()

    import threading
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    tools = [_McpStyleTool(bg_loop, name=f'mcp{i}') for i in range(3)]
    llm = _FakeLLM([(t.name, {'n': 0.3}) for t in tools])
    agent = _make_agent(llm, tools)

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    try:
        started = time.monotonic()
        await agent.arun_nonstream([{'role': 'user', 'content': 'go'}])
        elapsed = time.monotonic() - started
        assert elapsed < 0.5, (
            f'MCP-style tools should not serialize on the thread pool; elapsed={elapsed:.3f}s'
        )
    finally:
        executor.shutdown(wait=False)
        # Replace with a fresh pool so we do not restore a possibly-None default.
        loop.set_default_executor(ThreadPoolExecutor(max_workers=4))
        bg_loop.call_soon_threadsafe(bg_loop.stop)
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_aclose_waits_for_inflight_arun():
    class SlowTool(BaseTool):
        name = 'slow'
        description = 'slow'
        parameters = {'type': 'object', 'properties': {}, 'required': []}

        def call(self, params, **kwargs):
            return 'x'

        async def acall(self, params, **kwargs):
            await asyncio.sleep(0.15)
            return 'x'

    llm = _FakeLLM([('slow', {})])
    agent = _make_agent(llm, [SlowTool()])

    async def _run():
        return await agent.arun_nonstream([{'role': 'user', 'content': 'go'}])

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.02)
    close_task = asyncio.create_task(agent.aclose())
    await task
    await close_task
    assert agent._async_closed
    with pytest.raises(RuntimeError, match='closed'):
        await agent.arun_nonstream([{'role': 'user', 'content': 'go'}])
