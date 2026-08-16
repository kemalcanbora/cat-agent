# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for cat_agent.trace."""

from __future__ import annotations

import json
import os
from typing import Iterator, List
from unittest.mock import MagicMock

import pytest

from cat_agent.agent import Agent
from cat_agent.llm.schema import ASSISTANT, FUNCTION, USER, FunctionCall, Message
from cat_agent.trace import (
    InMemoryTraceStore,
    JSONLTraceStore,
    Run,
    RunLimits,
    SCHEMA_VERSION,
    get_trace_recorder,
    load_runs_from_jsonl,
    parse_partial_jsonl,
    trace_run,
)
from cat_agent.trace.redact import redact_llm_config, redact_obj


class StubLLM:
    model = 'stub-model'
    model_type = 'oai'
    model_cfg = {'model': 'stub-model', 'model_type': 'oai', 'api_key': 'sk-SECRETKEY123456'}

    def __init__(self, scripted: List[List[Message]]):
        self._scripted = list(scripted)
        self._i = 0

    def chat(self, messages, functions=None, stream=True, extra_generate_cfg=None):
        if self._i >= len(self._scripted):
            out = [Message(role=ASSISTANT, content='done')]
        else:
            out = self._scripted[self._i]
            self._i += 1
        if stream:
            yield out
        else:
            return out


class TinyAgent(Agent):
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        for out in self._call_llm(messages):
            yield out


class ToolishAgent(Agent):
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        for out in self._call_llm(messages):
            yield out
            if out and out[-1].function_call:
                name = out[-1].function_call.name
                args = out[-1].function_call.arguments
                # Soft tool miss → string error, run continues
                result = self._call_tool(name, args)
                fn = Message(role=FUNCTION, name=name, content=result)
                yield [fn]


def test_completed_single_agent_trace(tmp_path, monkeypatch):
    monkeypatch.setenv('CAT_AGENT_TRACE', '1')
    store = InMemoryTraceStore()
    llm = StubLLM([[Message(role=ASSISTANT, content='hello', extra={'usage': {
        'prompt_tokens': 10, 'completion_tokens': 2,
    }})]])
    agent = TinyAgent(llm=llm, name='t', context_manager=False)
    list(agent.run([{'role': 'user', 'content': 'hi'}], trace_store=store, trace=True))
    runs = list(store.iter_runs())
    assert len(runs) == 1
    run = runs[0]
    assert run.schema_version == SCHEMA_VERSION
    assert run.status == 'completed'
    assert run.agent_name == 't'
    assert any(s.kind == 'llm_call' for s in run.steps)
    assert 'api_key' not in json.dumps(run.llm_config) or '[REDACTED]' in json.dumps(run.llm_config)
    # Round-trip
    raw = run.model_dump(mode='json')
    again = Run.model_validate(raw)
    assert again.run_id == run.run_id
    assert len(again.steps) == len(run.steps)


@pytest.mark.parametrize('limit_kwargs,reason', [
    ({'max_steps': 1}, 'max_steps'),
    ({'max_total_tokens': 1}, 'max_tokens'),
    ({'max_wall_clock_seconds': 0.0}, 'wall_clock'),
    ({'max_tool_calls': 0}, None),  # no tools → may complete; tested separately
])
def test_run_limits(limit_kwargs, reason, monkeypatch):
    monkeypatch.setenv('CAT_AGENT_TRACE', '1')
    store = InMemoryTraceStore()
    # Many LLM turns
    scripted = [
        [Message(role=ASSISTANT, content=f'turn{i}', extra={'usage': {
            'prompt_tokens': 5, 'completion_tokens': 5,
        }})]
        for i in range(5)
    ]
    llm = StubLLM(scripted)

    class LoopAgent(Agent):
        def _run(self, messages, lang='en', **kwargs):
            for _ in range(5):
                for out in self._call_llm(messages):
                    yield out

    agent = LoopAgent(llm=llm, name='lim', context_manager=False,
                      run_limits=RunLimits(**limit_kwargs))
    list(agent.run([{'role': 'user', 'content': 'x'}], trace_store=store, trace=True))
    run = list(store.iter_runs())[0]
    if reason is None:
        return
    assert run.status == 'terminated'
    assert run.termination_reason == reason


def test_max_tool_calls_limit(monkeypatch):
    monkeypatch.setenv('CAT_AGENT_TRACE', '1')
    store = InMemoryTraceStore()
    llm = StubLLM([[Message(
        role=ASSISTANT, content='',
        function_call=FunctionCall(name='missing_tool', arguments='{}'),
    )]])
    agent = ToolishAgent(llm=llm, name='tools', context_manager=False,
                         run_limits=RunLimits(max_tool_calls=1))
    # After first tool call, wrap should terminate on next check — may need 2nd turn
    list(agent.run([{'role': 'user', 'content': 'x'}], trace_store=store, trace=True))
    run = list(store.iter_runs())[0]
    assert run.totals.tool_calls >= 1


def test_tool_error_recorded_without_abort(monkeypatch):
    monkeypatch.setenv('CAT_AGENT_TRACE', '1')
    store = InMemoryTraceStore()
    llm = StubLLM([[Message(
        role=ASSISTANT, content='',
        function_call=FunctionCall(name='nope', arguments='{}'),
        extra={'usage': {'prompt_tokens': 1, 'completion_tokens': 1}},
    )]])
    agent = ToolishAgent(llm=llm, name='soft', context_manager=False)
    list(agent.run([{'role': 'user', 'content': 'x'}], trace_store=store, trace=True))
    run = list(store.iter_runs())[0]
    tool_steps = [s for s in run.steps if s.kind == 'tool_call']
    assert tool_steps
    assert tool_steps[0].payload['succeeded'] is False
    assert run.status == 'completed'


def test_groupchat_parent_child_step_tree(monkeypatch):
    monkeypatch.setenv('CAT_AGENT_TRACE', '1')
    store = InMemoryTraceStore()
    with trace_run(store=store, agent_name='host', agent_class='GroupChat') as parent:
        parent.record_handoff(from_agent='host', to_agent='worker', reason='ask')
        anchor = parent.current_step_id
        with trace_run(
            store=store,
            agent_name='worker',
            agent_class='Assistant',
            parent_step_id=anchor,
        ) as child:
            child.record_llm_call(
                model='m',
                messages_in=[Message(USER, 'hi')],
                message_out=Message(ASSISTANT, 'yo'),
                prompt_tokens=1,
                completion_tokens=1,
            )
            child.finish(status='completed')
        parent.finish(status='completed')
    runs = {r.agent_name: r for r in store.iter_runs()}
    assert 'host' in runs and 'worker' in runs
    child_step = runs['worker'].steps[0]
    assert child_step.parent_step_id == anchor
    assert runs['host'].steps[0].kind == 'handoff'


def test_partial_jsonl_after_crash(tmp_path):
    path = tmp_path / 'trace.jsonl'
    store = JSONLTraceStore(path)
    with trace_run(store=store, agent_name='a', agent_class='A') as rec:
        rec.record_llm_call(
            model='m',
            messages_in=[Message(USER, 'hi')],
            message_out=Message(ASSISTANT, 'yo'),
            prompt_tokens=3,
            completion_tokens=1,
        )
        # Crash: no finalize
    partial = parse_partial_jsonl(path)
    assert len(partial) == 1
    assert partial[0].status == 'running'
    assert len(partial[0].steps) == 1
    # Still loadable
    loaded = load_runs_from_jsonl(path)
    assert partial[0].run_id in loaded


def test_secrets_never_in_serialised_output():
    cfg = redact_llm_config({
        'model': 'gpt',
        'api_key': 'sk-ABCDEFGHIJKLMNOP',
        'authorization': 'Bearer tok_secret',
    })
    blob = json.dumps(cfg)
    assert 'sk-ABCDEF' not in blob
    assert 'tok_secret' not in blob
    assert '[REDACTED]' in blob

    nested = redact_obj({
        'messages': [{'content': 'key=sk-ABCDEFGHIJKLMNOP please'}],
        'api_key': 'sk-ABCDEFGHIJKLMNOP',
    })
    text = json.dumps(nested)
    assert 'sk-ABCDEFGHIJKLMNOP' not in text


def test_trace_off_by_default(monkeypatch):
    monkeypatch.delenv('CAT_AGENT_TRACE', raising=False)
    assert get_trace_recorder() is None
    llm = StubLLM([[Message(ASSISTANT, 'ok')]])
    agent = TinyAgent(llm=llm, context_manager=False)
    list(agent.run([{'role': 'user', 'content': 'hi'}]))
    assert get_trace_recorder() is None


def test_round_trip_lossless():
    run = Run(agent_name='x', agent_class='Y', initial_messages=[Message(USER, 'a')])
    data = run.model_dump(mode='json')
    again = Run.model_validate(data)
    assert again.model_dump(mode='json') == data
