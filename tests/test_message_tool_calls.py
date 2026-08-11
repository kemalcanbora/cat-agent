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

"""Phase 3 schema + ToolCallStreamMerger tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cat_agent.llm.schema import (
    ASSISTANT,
    FUNCTION,
    TOOL,
    USER,
    FunctionCall,
    Message,
    ToolCall,
)
from cat_agent.llm.tool_call_stream import ToolCallStreamMerger


class TestMessageToolCallsCompat:
    def test_construct_function_call_yields_tool_calls(self):
        msg = Message(
            role=ASSISTANT,
            content='',
            function_call=FunctionCall(name='echo', arguments='{"a":1}'),
            extra={'function_id': 'id_1'},
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == 'id_1'
        assert msg.function_call is not None
        assert msg.function_call.name == 'echo'
        assert msg.function_call.arguments == '{"a":1}'

    def test_role_tool_accepted(self):
        msg = Message(role=TOOL, content='ok', tool_call_id='c1', name='echo')
        assert msg.role == TOOL
        assert msg.tool_call_id == 'c1'

    def test_role_nonsense_rejected(self):
        with pytest.raises(ValueError, match='nonsense'):
            Message(role='nonsense', content='x')

    def test_get_empty_content_vs_missing(self):
        msg = Message(role=USER, content='')
        assert msg.get('content', 'fallback') == ''
        assert msg.get('no_such_field', 'fallback') == 'fallback'
        assert msg.get('tool_calls', 'fallback') is None or msg.get('tool_calls') is None

    def test_dump_emits_tool_calls_only(self):
        msg = Message(
            role=ASSISTANT,
            content='',
            tool_calls=[
                ToolCall(id='a', function=FunctionCall(name='t', arguments='{}')),
                ToolCall(id='b', function=FunctionCall(name='u', arguments='{"x":1}')),
            ],
        )
        dumped = msg.model_dump()
        assert 'tool_calls' in dumped
        assert len(dumped['tool_calls']) == 2
        assert 'function_call' not in dumped

    def test_legacy_function_call_dict_loads_via_message_starstar(self):
        """Pre-existing audit JSONL / cache rows with function_call still rehydrate."""
        legacy = {
            'role': ASSISTANT,
            'content': '',
            'function_call': {'name': 'echo', 'arguments': '{"v":1}'},
            'extra': {'function_id': 'call_legacy'},
        }
        msg = Message(**legacy)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == 'call_legacy'
        assert msg.tool_calls[0].function.name == 'echo'
        assert msg.tool_calls[0].function.arguments == '{"v":1}'
        assert msg.function_call.name == 'echo'
        assert 'function_call' not in msg.model_dump()


class TestToolCallStreamMerger:
    def test_argument_json_split_mid_token(self):
        merger = ToolCallStreamMerger()
        merger.push(SimpleNamespace(
            index=0, id='call_1',
            function=SimpleNamespace(name='echo', arguments='{"a":'),
        ))
        merger.push(SimpleNamespace(
            index=0, id=None,
            function=SimpleNamespace(name=None, arguments='1}'),
        ))
        calls = merger.tool_calls()
        assert len(calls) == 1
        assert calls[0].id == 'call_1'
        assert calls[0].function.name == 'echo'
        assert calls[0].function.arguments == '{"a":1}'

    def test_two_concurrent_calls_interleaved(self):
        merger = ToolCallStreamMerger()
        # index 0 name+id, index 1 name+id, then interleaved args
        merger.push({'index': 0, 'id': 'c0', 'function': {'name': 'alpha', 'arguments': ''}})
        merger.push({'index': 1, 'id': 'c1', 'function': {'name': 'beta', 'arguments': ''}})
        merger.push({'index': 0, 'function': {'arguments': '{"x":'}})
        merger.push({'index': 1, 'function': {'arguments': '{"y":'}})
        merger.push({'index': 0, 'function': {'arguments': '1}'}})
        merger.push({'index': 1, 'function': {'arguments': '2}'}})
        calls = merger.tool_calls()
        assert [c.id for c in calls] == ['c0', 'c1']
        assert calls[0].function.name == 'alpha'
        assert calls[0].function.arguments == '{"x":1}'
        assert calls[1].function.name == 'beta'
        assert calls[1].function.arguments == '{"y":2}'

    def test_chunk_with_index_but_no_name(self):
        merger = ToolCallStreamMerger()
        merger.push(SimpleNamespace(
            index=0, id='only_id',
            function=SimpleNamespace(name='weather', arguments=''),
        ))
        # Continuation: index only, no name, no id
        merger.push(SimpleNamespace(
            index=0, id=None,
            function=SimpleNamespace(name=None, arguments='{"city":"Paris"}'),
        ))
        calls = merger.tool_calls()
        assert len(calls) == 1
        assert calls[0].function.name == 'weather'
        assert calls[0].function.arguments == '{"city":"Paris"}'
        assert calls[0].id == 'only_id'


class TestParallelRoundTripJobs:
    def test_single_message_two_tool_calls_drives_two_jobs(self):
        from cat_agent.agent import BasicAgent

        class _Stub(BasicAgent):
            def _run(self, *a, **k):
                yield []

        agent = _Stub(llm={'model': 'x', 'api_key': 'EMPTY', 'model_type': 'oai'})
        msg = Message(
            role=ASSISTANT,
            content='',
            tool_calls=[
                ToolCall(id='p', function=FunctionCall(name='get_weather', arguments='{"city":"Paris"}')),
                ToolCall(id='b', function=FunctionCall(name='get_weather', arguments='{"city":"Berlin"}')),
            ],
        )
        jobs = list(agent._iter_tool_call_jobs([msg]))
        assert len(jobs) == 2
        assert [j[1] for j in jobs] == ['p', 'b']
        assert [j[2] for j in jobs] == ['get_weather', 'get_weather']
