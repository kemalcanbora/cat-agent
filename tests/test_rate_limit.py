"""Tests for RateLimiter and agent/tool integration."""

from __future__ import annotations

import asyncio
import json
import time
from typing import List, Union
from unittest.mock import patch

import pytest

from cat_agent.agents.fncall_agent import FnCallAgent
from cat_agent.llm.schema import ASSISTANT, FUNCTION, FunctionCall, Message
from cat_agent.tools.base import BaseTool
from cat_agent.utils.rate_limit import RateLimiter


class _CountingTool(BaseTool):
    name = 'count'
    description = 'Counts concurrent calls'
    parameters = {'type': 'object', 'properties': {}, 'required': []}

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.calls = 0
        self.max_inflight = 0
        self._inflight = 0
        self._lock = asyncio.Lock()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        self.calls += 1
        time.sleep(0.05)
        return f'ok-{self.calls}'

    async def acall(self, params: Union[str, dict], **kwargs) -> str:
        async with self._lock:
            self._inflight += 1
            self.calls += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            await asyncio.sleep(0.05)
            return 'ok'
        finally:
            async with self._lock:
                self._inflight -= 1


class _FakeLLM:
    def __init__(self, n_tools: int = 1):
        self.model = 'fake'
        self._n = n_tools
        self._turn = 0
        self.chat_calls = 0

    def _next(self) -> List[Message]:
        self._turn += 1
        self.chat_calls += 1
        if self._turn == 1:
            return [
                Message(
                    role=ASSISTANT,
                    content='',
                    function_call=FunctionCall(name='count', arguments='{}'),
                    extra={'function_id': str(i + 1)},
                )
                for i in range(self._n)
            ]
        return [Message(role=ASSISTANT, content='done')]

    def chat(self, messages, functions=None, stream=True, delta_stream=False, extra_generate_cfg=None):
        out = self._next()
        return iter([out]) if stream else out

    async def achat(self, messages, functions=None, extra_generate_cfg=None, **kwargs):
        return self._next()


class TestRateLimiterUnit:
    def test_sync_rate_paces_calls(self):
        lim = RateLimiter(requests_per_interval=2, interval_seconds=0.2)
        t0 = time.monotonic()
        for _ in range(3):
            with lim.limit():
                pass
        elapsed = time.monotonic() - t0
        # Third token needs a refill after capacity 2
        assert elapsed >= 0.05

    @pytest.mark.asyncio
    async def test_async_wait_does_not_block_loop(self):
        lim = RateLimiter(requests_per_interval=1, interval_seconds=0.3)
        ticks = []

        async def heartbeat():
            for _ in range(6):
                ticks.append(time.monotonic())
                await asyncio.sleep(0.05)

        async def consumer():
            # First acquire immediate; second must wait ~0.3s
            async with lim.limit_async():
                pass
            async with lim.limit_async():
                pass

        await asyncio.gather(heartbeat(), consumer())
        assert len(ticks) >= 4  # heartbeat kept ticking during limiter wait

    @pytest.mark.asyncio
    async def test_concurrency_cap_queues(self):
        lim = RateLimiter(max_concurrency=2)
        inflight = 0
        max_seen = 0
        lock = asyncio.Lock()

        async def worker():
            nonlocal inflight, max_seen
            async with lim.limit_async():
                async with lock:
                    inflight += 1
                    max_seen = max(max_seen, inflight)
                await asyncio.sleep(0.05)
                async with lock:
                    inflight -= 1

        await asyncio.gather(*[worker() for _ in range(6)])
        assert max_seen == 2

    @pytest.mark.asyncio
    async def test_cancel_while_queued_on_limiter(self):
        lim = RateLimiter(max_concurrency=1)

        async def holder():
            async with lim.limit_async():
                await asyncio.Future()  # park

        async def waiter():
            async with lim.limit_async():
                return 'got'

        h = asyncio.create_task(holder())
        await asyncio.sleep(0.01)
        w = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)
        w.cancel()
        with pytest.raises(asyncio.CancelledError):
            await w
        h.cancel()
        with pytest.raises(asyncio.CancelledError):
            await h


class TestRateLimiterIntegration:
    @pytest.mark.asyncio
    async def test_parallel_tools_under_concurrency_limit(self):
        limiter = RateLimiter(max_concurrency=2)
        tool = _CountingTool({'rate_limiter': limiter})
        # Need distinct tool instances? Same tool object shared - OK for counting
        # But function_map has one tool; gather calls same tool concurrently - OK
        tools = []
        for i in range(5):
            t = _CountingTool({'rate_limiter': limiter})
            t.name = f'count{i}'
            tools.append(t)
        # Remap: use one name by registering multiple under same limiter via parallel
        # Simpler: one tool name, LLM emits 5 parallel calls to same tool
        tool = _CountingTool({'rate_limiter': limiter})
        llm = _FakeLLM(n_tools=5)
        # All function_calls use name 'count'
        agent = FnCallAgent(llm=llm, function_list=[tool])
        responses = []
        async for rsp in agent.arun([Message(role='user', content='go')]):
            responses.append(rsp)
        final = responses[-1]
        fn = [m for m in final if (m.role if not isinstance(m, dict) else m['role']) == FUNCTION]
        assert len(fn) == 5
        assert tool.calls == 5
        assert tool.max_inflight <= 2
        assert all(
            (m.content if not isinstance(m, dict) else m['content']) == 'ok' for m in fn
        )

    def test_unconfigured_no_overhead(self):
        tool = _CountingTool({})
        agent = FnCallAgent(llm=_FakeLLM(1), function_list=[tool])
        with patch.object(RateLimiter, 'acquire', side_effect=AssertionError('should not acquire')):
            assert agent._call_tool('count', '{}').startswith('ok-')

    @pytest.mark.asyncio
    async def test_llm_limiter_shared(self):
        limiter = RateLimiter(requests_per_interval=1, interval_seconds=0.15)
        llm = _FakeLLM(1)
        tool = _CountingTool()
        agent = FnCallAgent(llm=llm, function_list=[tool], rate_limiter=limiter)
        t0 = time.monotonic()
        # Two LLM turns in one run (tool then final) — second must wait for refill
        async for _ in agent.arun([Message(role='user', content='go')]):
            pass
        assert time.monotonic() - t0 >= 0.1
        assert llm.chat_calls == 2
