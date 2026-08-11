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

"""Structure-preserving truncation: tool-call fields must survive the native round-trip."""

from __future__ import annotations

import copy

import pytest

from cat_agent.llm.base.truncation import (
    _message_to_native,
    _native_to_message,
    truncate_input_messages_roughly,
)
from cat_agent.llm.schema import ASSISTANT, FUNCTION, USER, ContentItem, FunctionCall, Message

pytest.importorskip('cat_agent._native')


def _dump(msg: Message) -> dict:
    return msg.model_dump()


def _tool_history() -> list[Message]:
    return [
        Message(role=USER, content='What is the weather in Paris and Berlin?'),
        Message(
            role=ASSISTANT,
            content='',
            function_call=FunctionCall(name='get_weather', arguments='{"city":"Paris"}'),
            extra={'function_id': 'call_paris'},
        ),
        Message(
            role=FUNCTION,
            name='get_weather',
            content=[ContentItem(text='sunny, 22C')],
            extra={'function_id': 'call_paris'},
        ),
        Message(
            role=ASSISTANT,
            content='',
            function_call=FunctionCall(name='get_weather', arguments='{"city":"Berlin"}'),
            extra={'function_id': 'call_berlin'},
        ),
        Message(
            role=FUNCTION,
            name='get_weather',
            content=[ContentItem(text='cloudy, 18C')],
            extra={'function_id': 'call_berlin'},
        ),
    ]


class TestUnderBudgetRoundTrip:
    """The regression that matters: bug reproduces with no token pressure."""

    def test_tool_call_and_result_round_trip_byte_identical(self):
        messages = _tool_history()
        out = truncate_input_messages_roughly(messages, max_tokens=1_000_000)
        assert len(out) == len(messages)
        for original, restored in zip(messages, out):
            assert _dump(restored) == _dump(original)

    def test_function_call_and_function_id_intact(self):
        messages = _tool_history()
        out = truncate_input_messages_roughly(messages, max_tokens=1_000_000)
        assert out[1].function_call is not None
        assert out[1].function_call.name == 'get_weather'
        assert out[1].function_call.arguments == '{"city":"Paris"}'
        assert out[1].extra == {'function_id': 'call_paris'}
        assert out[2].role == FUNCTION
        assert out[2].extra == {'function_id': 'call_paris'}
        assert out[2].name == 'get_weather'
        assert out[3].extra == {'function_id': 'call_berlin'}
        assert out[4].extra == {'function_id': 'call_berlin'}

    def test_counting_text_is_not_python_repr(self):
        msg = Message(
            role=ASSISTANT,
            content='',
            function_call=FunctionCall(name='foo', arguments='{"a":1}'),
            extra={'function_id': 'x'},
        )
        native = _message_to_native(msg)
        assert 'FunctionCall' not in native['text']
        assert "{'name'" not in native['text']
        assert 'foo' in native['text']
        assert '{"a":1}' in native['text']
        assert 'tool_calls' in native
        assert 'function_call' not in native
        # Recovered content must not be the counting text.
        restored = _native_to_message(native)
        assert restored.content == ''
        assert restored.function_call is not None
        assert restored.extra == {'function_id': 'x'}


class TestNativeConverters:
    def test_message_to_native_to_message_identity(self):
        histories = [
            _tool_history(),
            [
                Message(role=USER, content='hi'),
                Message(role=ASSISTANT, content='hello'),
            ],
            [
                Message(role=USER, content='q'),
                Message(
                    role=ASSISTANT,
                    content=[ContentItem(text='thinking')],
                    function_call=FunctionCall(name='t', arguments='{}'),
                    extra={'function_id': '1'},
                    reasoning_content='reason',
                ),
                Message(
                    role=FUNCTION,
                    name='t',
                    content=[ContentItem(text='ok')],
                    extra={'function_id': '1'},
                ),
            ],
        ]
        for messages in histories:
            for msg in messages:
                assert _dump(_native_to_message(_message_to_native(msg))) == _dump(msg)


class TestOverBudget:
    def test_dropped_step_drops_call_and_results_together(self):
        # Several tool-call steps; tight budget forces middle-step drops.
        messages = [
            Message(role=USER, content='start ' + ('word ' * 20)),
        ]
        for i in range(6):
            fid = f'call_{i}'
            messages.append(
                Message(
                    role=ASSISTANT,
                    content='',
                    function_call=FunctionCall(
                        name='tool',
                        arguments=f'{{"i":{i},"pad":"{"x" * 80}"}}',
                    ),
                    extra={'function_id': fid},
                )
            )
            messages.append(
                Message(
                    role=FUNCTION,
                    name='tool',
                    content=[ContentItem(text=('result ' * 40) + str(i))],
                    extra={'function_id': fid},
                )
            )
        messages.append(Message(role=USER, content='follow up'))
        messages.append(Message(role=ASSISTANT, content='done'))

        out = truncate_input_messages_roughly(messages, max_tokens=256)

        # No FUNCTION result without a matching preceding call id among kept msgs.
        kept_ids = set()
        for msg in out:
            if msg.function_call and msg.extra:
                kept_ids.add(msg.extra.get('function_id'))
        for msg in out:
            if msg.role == FUNCTION:
                fid = (msg.extra or {}).get('function_id')
                assert fid in kept_ids, f'orphaned function result id={fid!r}'

    def test_omit_preserves_role_name_and_id(self):
        # Large early function body is a candidate for "omit" while the call remains.
        messages = [
            Message(role=USER, content='q1'),
            Message(
                role=ASSISTANT,
                content='',
                function_call=FunctionCall(name='big_tool', arguments='{}'),
                extra={'function_id': 'call_omit'},
            ),
            Message(
                role=FUNCTION,
                name='big_tool',
                content=[ContentItem(text='PAYLOAD ' * 500)],
                extra={'function_id': 'call_omit'},
            ),
            Message(role=USER, content='q2 ' + ('more ' * 100)),
            Message(role=ASSISTANT, content='final answer ' + ('x' * 50)),
        ]
        out = truncate_input_messages_roughly(messages, max_tokens=200)

        fn_msgs = [m for m in out if m.role == FUNCTION]
        if not fn_msgs:
            pytest.skip('budget dropped the function message entirely (still valid)')
        for fn in fn_msgs:
            assert fn.name == 'big_tool'
            assert fn.extra == {'function_id': 'call_omit'}
            # Body may be omitted or truncated, but identity remains.
            if isinstance(fn.content, list):
                body = fn.content[0].text if fn.content else ''
            else:
                body = fn.content
            assert body is not None


class TestPropertyKeptMessages:
    def test_kept_messages_round_trip_via_converters(self):
        messages = _tool_history()
        # Simulate "kept" as under-budget truncator output, then converter identity.
        kept = truncate_input_messages_roughly(copy.deepcopy(messages), max_tokens=1_000_000)
        for msg in kept:
            assert _dump(_native_to_message(_message_to_native(msg))) == _dump(msg)
