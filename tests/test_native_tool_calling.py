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

"""Native tool-calling path: non-stream parsing, wire format, prompt fallback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.base.model import BaseChatModel
from cat_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt
from cat_agent.llm.function_calling import BaseFnCallModel
from cat_agent.llm.oai import TextChatAtOAI, _messages_from_completion_message
from cat_agent.llm.schema import ASSISTANT, FUNCTION, USER, ContentItem, FunctionCall, Message, ToolCall


def _tc(name: str, arguments: str, tc_id: str):
    return SimpleNamespace(
        id=tc_id,
        type='function',
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class TestMessagesFromCompletion:
    def test_three_tool_calls_preserve_ids(self):
        msg = SimpleNamespace(
            content=None,
            reasoning_content=None,
            tool_calls=[
                _tc('alpha', '{"x":1}', 'call_a'),
                _tc('beta', '{"y":2}', 'call_b'),
                _tc('alpha', '{"x":3}', 'call_c'),
            ],
        )
        out = _messages_from_completion_message(msg)
        tool_msgs = [m for m in out if m.tool_calls]
        assert len(tool_msgs) == 1
        tcs = tool_msgs[0].tool_calls
        assert [tc.id for tc in tcs] == ['call_a', 'call_b', 'call_c']
        assert [tc.function.name for tc in tcs] == ['alpha', 'beta', 'alpha']
        assert tcs[0].function.arguments == '{"x":1}'
        assert tcs[2].function.arguments == '{"x":3}'
        # Compat property reads the first call.
        assert tool_msgs[0].function_call.name == 'alpha'


class TestWireFormat:
    def test_tool_result_uses_tool_call_id_not_id(self):
        out = BaseChatModel._conv_cat_agent_messages_to_oai([
            {'role': 'assistant', 'content': '', 'function_call': {'name': 't', 'arguments': '{}'},
             'extra': {'function_id': 'call_9'}},
            {'role': 'function', 'name': 't', 'content': 'ok', 'extra': {'function_id': 'call_9'}},
        ])
        tool = [m for m in out if m['role'] == 'tool'][0]
        assert tool['tool_call_id'] == 'call_9'
        assert 'id' not in tool or tool.get('id') != 'call_9'
        assert 'extra' not in tool
        assert tool['content'] == 'ok'

    def test_outbound_has_tools_shape_via_raw_chat_cfg(self):
        captured = {}

        class _Stub(TextChatAtOAI):
            def __init__(self):
                # Bypass network client setup.
                from cat_agent.llm.function_calling import BaseFnCallModel
                BaseFnCallModel.__init__(self, {'model': 'stub', 'generate_cfg': {}})
                self.model = 'stub'

            def _chat_no_stream(self, messages, generate_cfg):
                captured['generate_cfg'] = dict(generate_cfg)
                captured['messages'] = messages
                return [Message(role=ASSISTANT, content='done')]

            def _chat_stream(self, messages, delta_stream, generate_cfg):
                raise AssertionError('should use non-stream')

        m = _Stub()
        assert m.supports_native_tools is True
        assert m.use_raw_api is True
        result = m.chat(
            messages=[Message(role=USER, content='hi')],
            functions=[{'name': 'echo', 'parameters': {'type': 'object'}}],
            stream=False,
        )
        assert result[0].content == 'done'
        assert 'tools' in captured['generate_cfg']
        assert captured['generate_cfg']['tools'][0]['type'] == 'function'
        assert 'functions' not in captured['generate_cfg']

    def test_distinct_ids_for_same_tool_on_wire(self):
        out = BaseChatModel._conv_cat_agent_messages_to_oai([
            {'role': 'assistant', 'content': '', 'function_call': {'name': 'echo', 'arguments': '{"a":1}'},
             'extra': {'function_id': 'id_1'}},
            {'role': 'assistant', 'content': '', 'function_call': {'name': 'echo', 'arguments': '{"a":2}'},
             'extra': {'function_id': 'id_2'}},
            {'role': 'function', 'name': 'echo', 'content': '1', 'extra': {'function_id': 'id_1'}},
            {'role': 'function', 'name': 'echo', 'content': '2', 'extra': {'function_id': 'id_2'}},
        ])
        assistant = [m for m in out if m['role'] == 'assistant'][0]
        ids = [tc['id'] for tc in assistant['tool_calls']]
        assert ids == ['id_1', 'id_2']
        tool_ids = [m['tool_call_id'] for m in out if m['role'] == 'tool']
        assert tool_ids == ['id_1', 'id_2']
        assert all('extra' not in m for m in out)


class TestStreamEqualsNoStream:
    def test_same_logical_response(self):
        tool_calls = [
            _tc('a', '{}', 'c1'),
            _tc('b', '{"k":1}', 'c2'),
        ]
        completion_msg = SimpleNamespace(content='thinking', reasoning_content=None, tool_calls=tool_calls)
        no_stream = _messages_from_completion_message(completion_msg)

        from cat_agent.llm.schema import ToolCall
        stream_out = [
            Message(
                role=ASSISTANT,
                content='thinking',
                tool_calls=[
                    ToolCall(id='c1', function=FunctionCall(name='a', arguments='{}')),
                    ToolCall(id='c2', function=FunctionCall(name='b', arguments='{"k":1}')),
                ],
            ),
        ]
        assert [m.model_dump() for m in no_stream] == [m.model_dump() for m in stream_out]


class TestOaiDefaultNative:
    def test_oai_defaults_use_raw_api_true(self):
        import os
        env = {k: v for k, v in os.environ.items() if k != 'CAT_AGENT_USE_RAW_API'}
        with patch.dict(os.environ, env, clear=True):
            m = TextChatAtOAI({'model': 'gpt-test', 'api_key': 'EMPTY'})
        assert m.supports_native_tools is True
        assert m.use_raw_api is True

    def test_env_false_forces_prompt_path(self):
        with patch.dict('os.environ', {'CAT_AGENT_USE_RAW_API': 'false'}):
            m = TextChatAtOAI({'model': 'gpt-test', 'api_key': 'EMPTY'})
        assert m.use_raw_api is False


class TestAsyncNativeEndToEnd:
    @pytest.mark.asyncio
    async def test_achat_nonstream_native_tool_calls(self):
        tool_calls = [
            _tc('echo', '{"v":1}', 'call_1'),
            _tc('echo', '{"v":2}', 'call_2'),
        ]
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None, reasoning_content=None, tool_calls=tool_calls))],
            usage=None,
        )

        m = TextChatAtOAI({'model': 'gpt-test', 'api_key': 'EMPTY'})
        assert m.use_raw_api is True

        def fake_create(*args, **kwargs):
            assert kwargs.get('stream') is False
            assert 'tools' in kwargs
            return fake_response

        m._chat_complete_create = fake_create  # type: ignore[method-assign]
        m._sync_chat_complete_create = fake_create  # type: ignore[method-assign]

        # Bypass async client bridge: call _chat_no_stream via chat(stream=False).
        out = m.chat(
            messages=[Message(role=USER, content='go')],
            functions=[{'name': 'echo', 'parameters': {'type': 'object'}}],
            stream=False,
        )
        tool_msgs = [x for x in out if x.tool_calls]
        assert len(tool_msgs) == 1
        assert {tc.id for tc in tool_msgs[0].tool_calls} == {'call_1', 'call_2'}


class TestPromptPathStillWorks:
    def test_nous_forced(self):
        text = '<tool_call>\n{"name": "my_tool", "arguments": {"x": 1}}\n</tool_call>'
        out = NousFnCallPrompt().postprocess_fncall_messages(
            [Message(role=ASSISTANT, content=[ContentItem(text=text)])])
        fns = [m for m in out if m.function_call]
        assert len(fns) == 1
        assert fns[0].function_call.name == 'my_tool'
        assert fns[0].extra.get('function_id')


def _count_tool_calls(messages: List[Message]) -> int:
    return sum(len(m.tool_calls) for m in messages if m.tool_calls)


_MULTI_TOOL_MARKUP = (
    '<tool_call>\n'
    '{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
    '</tool_call>\n'
    '<tool_call>\n'
    '{"name": "get_weather", "arguments": {"city": "Berlin"}}\n'
    '</tool_call>'
)

_WEATHER_FUNCTIONS = [{
    'name': 'get_weather',
    'description': 'Weather for a city',
    'parameters': {
        'type': 'object',
        'properties': {'city': {'type': 'string'}},
        'required': ['city'],
    },
}]


class TestLocalAndOaiMultiCallParity:
    """Prompt-path (local) and native oai path must keep the same tool-call count."""

    def test_same_tool_call_count_for_equivalent_multi_call_response(self):
        class _LocalPromptBackend(BaseFnCallModel):
            @property
            def supports_native_tools(self) -> bool:
                return False

            def __init__(self):
                super().__init__({'model': 'local-stub', 'generate_cfg': {'use_raw_api': False}})

            def _chat_stream(self, messages, delta_stream, generate_cfg):
                raise AssertionError('use non-stream')

            def _chat_no_stream(self, messages, generate_cfg):
                return [Message(role=ASSISTANT, content=_MULTI_TOOL_MARKUP)]

        class _OaiNativeBackend(TextChatAtOAI):
            def __init__(self):
                BaseFnCallModel.__init__(self, {'model': 'oai-stub', 'generate_cfg': {}})
                self.model = 'oai-stub'

            def _chat_stream(self, messages, delta_stream, generate_cfg):
                raise AssertionError('use non-stream')

            def _chat_no_stream(self, messages, generate_cfg):
                msg = SimpleNamespace(
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        _tc('get_weather', '{"city": "Paris"}', 'call_1'),
                        _tc('get_weather', '{"city": "Berlin"}', 'call_2'),
                    ],
                )
                return _messages_from_completion_message(msg)

        local = _LocalPromptBackend()
        oai = _OaiNativeBackend()
        assert local.use_raw_api is False
        assert oai.use_raw_api is True

        user = [Message(role=USER, content='Weather in Paris and Berlin')]
        local_out = local.chat(messages=user, functions=_WEATHER_FUNCTIONS, stream=False)
        oai_out = oai.chat(messages=user, functions=_WEATHER_FUNCTIONS, stream=False)

        assert _count_tool_calls(local_out) == 2
        assert _count_tool_calls(oai_out) == 2
        assert _count_tool_calls(local_out) == _count_tool_calls(oai_out)


class TestQuickChatOaiIds:
    def test_preserves_function_id(self):
        class _Stub(BaseChatModel):
            def __init__(self):
                super().__init__({'model': 'x', 'generate_cfg': {'use_raw_api': True}})

            def _chat_stream(self, messages, delta_stream, generate_cfg):
                yield [
                    Message(
                        role=ASSISTANT,
                        content='',
                        function_call=FunctionCall(name='t', arguments='{}'),
                        extra={'function_id': 'real_id'},
                    )
                ]

            def _chat_no_stream(self, messages, generate_cfg):
                return []

            def _chat_with_functions(self, *a, **k):
                raise NotImplementedError

        stub = _Stub()
        chunks = list(stub.quick_chat_oai(
            messages=[{'role': 'user', 'content': 'hi'}],
            tools=[{'type': 'function', 'function': {'name': 't'}}],
        ))
        assert chunks[-1]['choices'][0]['message']['tool_calls'][0]['id'] == 'real_id'


class TestFormatPreservesParallelToolCalls:
    """format_as_text_message must not collapse parallel tool_calls or regenerate ids."""

    def test_assistant_keeps_all_tool_call_ids(self):
        from cat_agent.utils.message_utils import format_as_text_message

        asst = Message(
            role=ASSISTANT,
            content='calling tools',
            tool_calls=[
                ToolCall(id='call_a', function=FunctionCall(name='alpha', arguments='{}')),
                ToolCall(id='call_b', function=FunctionCall(name='beta', arguments='{"x":1}')),
            ],
        )
        out = format_as_text_message(asst, add_upload_info=False)
        assert out.tool_calls is not None
        assert [tc.id for tc in out.tool_calls] == ['call_a', 'call_b']
        assert [tc.function.name for tc in out.tool_calls] == ['alpha', 'beta']

    def test_function_keeps_tool_call_id_and_content(self):
        from cat_agent.utils.message_utils import format_as_text_message

        body = '{"total": 2, "calls": [{"identifier": "HORIZON-X"}]}'
        fn = Message(
            role=FUNCTION,
            content=body,
            name='sedia_search',
            tool_call_id='call_a',
            extra={'function_id': 'call_a'},
        )
        out = format_as_text_message(fn, add_upload_info=False)
        assert out.tool_call_id == 'call_a'
        assert out.content == body
        assert out.name == 'sedia_search'

    def test_oai_wire_ids_match_after_format(self):
        from cat_agent.llm.oai import TextChatAtOAI
        from cat_agent.utils.message_utils import format_as_text_message

        asst = Message(
            role=ASSISTANT,
            content='',
            tool_calls=[
                ToolCall(id='call_a', function=FunctionCall(name='sedia_search', arguments='{"q":"a"}')),
                ToolCall(id='call_b', function=FunctionCall(name='sedia_search', arguments='{"q":"b"}')),
            ],
        )
        fn_a = Message(
            role=FUNCTION,
            content='{"total":1}',
            name='sedia_search',
            tool_call_id='call_a',
            extra={'function_id': 'call_a'},
        )
        fn_b = Message(
            role=FUNCTION,
            content='{"total":2}',
            name='sedia_search',
            tool_call_id='call_b',
            extra={'function_id': 'call_b'},
        )
        msgs = [
            Message(role=USER, content='search'),
            asst,
            fn_a,
            fn_b,
        ]
        # chat() formats before convert_messages_to_dicts
        msgs = [format_as_text_message(m, add_upload_info=False) for m in msgs]
        llm = TextChatAtOAI({'model': 'x', 'api_key': 'x', 'base_url': 'http://127.0.0.1:9/v1'})
        wire = llm.convert_messages_to_dicts(msgs)
        asst_wire = next(w for w in wire if w['role'] == 'assistant')
        tool_wires = [w for w in wire if w['role'] == 'tool']
        assert [tc['id'] for tc in asst_wire['tool_calls']] == ['call_a', 'call_b']
        assert [w['tool_call_id'] for w in tool_wires] == ['call_a', 'call_b']
        assert [w['content'] for w in tool_wires] == ['{"total":1}', '{"total":2}']
