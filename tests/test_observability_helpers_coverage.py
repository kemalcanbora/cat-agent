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

"""Coverage tests for cat_agent.observability.helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from cat_agent.llm.schema import ASSISTANT, FUNCTION, USER, ContentItem, FunctionCall, Message, ToolCall
from cat_agent.observability.context import RedactConfig, RunContext
from cat_agent.observability.helpers import (
    agent_model_name,
    extract_usage,
    format_llm_obs_output,
    format_obs_io,
    format_run_obs_output,
    format_tool_args,
    messages_have_tool_call,
    messages_to_payload,
    result_char_count,
    truncate_result_preview,
)


def _ctx(**redact_kwargs) -> RunContext:
    return RunContext(
        trace_id='t',
        run_id='r',
        span_id='s',
        parent_span_id=None,
        agent_name='a',
        agent_class='A',
        redact=RedactConfig(**redact_kwargs),
    )


def test_agent_model_name_variants():
    assert agent_model_name(None) is None
    assert agent_model_name(SimpleNamespace(model=' gpt-4 ')) == 'gpt-4'
    assert agent_model_name(SimpleNamespace(model='', model_id='m-id')) == 'm-id'
    assert agent_model_name(SimpleNamespace(model_path='/models/foo.gguf')) == 'foo.gguf'
    assert agent_model_name(SimpleNamespace(repo_id='org/repo')) == 'org/repo'
    assert agent_model_name(SimpleNamespace(filename='weights.bin')) == 'weights.bin'
    assert agent_model_name(SimpleNamespace()) is None


def test_format_tool_args_and_redact():
    dumped = format_tool_args({'a': 1}, None)
    assert 'a' in dumped and '1' in dumped
    assert format_tool_args('raw', None) == 'raw'
    assert format_tool_args({'a': 1}, _ctx(redact_tool_args=True)) == '<redacted>'


def test_result_char_count_and_truncate():
    assert result_char_count('abc', None) == 3
    assert result_char_count(['x', 'y'], None) == len(str(['x', 'y']))
    assert result_char_count({'k': 1}, None) > 0
    assert result_char_count('abcdef', _ctx(max_result_chars=3)) == 3

    assert truncate_result_preview('short', None) == 'short'
    assert truncate_result_preview(['a'], None) == str(['a'])
    assert truncate_result_preview({'n': 1}, None).startswith('{')
    long = 'x' * 50
    assert truncate_result_preview(long, _ctx(max_result_chars=10)) == 'x' * 10 + '...'


def test_format_obs_io_messages_dicts_and_redact():
    assert format_obs_io(None, None) is None
    assert format_obs_io('hi', _ctx(redact_messages=True)) == '<redacted>'

    msgs = [
        Message(USER, 'q', name=None),
        Message(
            ASSISTANT,
            'a',
            reasoning_content='think',
            tool_calls=[ToolCall(function=FunctionCall(name='t', arguments='{}'))],
        ),
        {'role': 'user', 'content': 'd', 'name': 'u'},
        {
            'role': 'assistant',
            'content': '',
            'function_call': {'name': 'legacy', 'arguments': '{}'},
            'extra': {'function_id': 'fid'},
            'reasoning_content': 'r',
        },
        42,
    ]
    text = format_obs_io(msgs, None)
    assert 'q' in text
    assert 'tool_calls' in text
    assert 'legacy' in text
    assert '42' in text

    truncated = format_obs_io('y' * 100, _ctx(max_result_chars=5))
    assert truncated == 'y' * 5 + '...'

    assert format_obs_io({'only': True}, None).startswith('{')


def test_format_llm_and_run_obs_output():
    assert format_llm_obs_output([]) is None
    assert format_llm_obs_output([Message(USER, 'x')], _ctx(redact_messages=True)) == '<redacted>'

    assistant = Message(
        ASSISTANT,
        [ContentItem(text='answer')],
        reasoning_content='reason',
        tool_calls=[ToolCall(function=FunctionCall(name='search', arguments='{"q":1}'))],
    )
    out = format_llm_obs_output([assistant])
    assert 'reason' in out
    assert 'answer' in out
    assert '[tool_call search' in out

    fc_only = Message(
        ASSISTANT,
        '',
        function_call=FunctionCall(name='only', arguments='{}'),
    )
    assert '[tool_call only' in format_llm_obs_output(fc_only)

    dict_tc = {
        'role': 'assistant',
        'content': '',
        'tool_calls': [{'function': {'name': 'd', 'arguments': '{}'}}],
    }
    assert '[tool_call d' in format_llm_obs_output([dict_tc])

    dict_fc = {'role': 'assistant', 'content': '', 'function_call': {'name': 'e', 'arguments': '1'}}
    assert '[tool_call e' in format_llm_obs_output([dict_fc])

    obj_fc = SimpleNamespace(name='obj', arguments='2')
    assert '[tool_call obj' in format_llm_obs_output([
        {'role': 'assistant', 'content': '', 'function_call': obj_fc},
    ])

    long_parts = Message(ASSISTANT, 'z' * 100)
    assert format_llm_obs_output([long_parts], _ctx(max_result_chars=8)).endswith('...')

    # no assistant content -> fall back to format_obs_io
    fallback = format_llm_obs_output([Message(USER, 'only-user')])
    assert 'only-user' in fallback

    assert format_run_obs_output([]) is None
    assert format_run_obs_output([Message(USER, 'u')], _ctx(redact_messages=True)) == '<redacted>'

    run_msgs = [
        Message(ASSISTANT, 'done', name='bot'),
        Message(FUNCTION, 'tool-out', name='calc'),
        Message(USER, 'ignored'),
    ]
    run_out = format_run_obs_output(run_msgs)
    assert 'bot:' in run_out
    assert 'calc: tool-out' in run_out

    long_run = format_run_obs_output([Message(ASSISTANT, 'w' * 80)], _ctx(max_result_chars=12))
    assert long_run.endswith('...')

    assert 'ignored' in format_run_obs_output([Message(USER, 'ignored')])


def test_messages_have_tool_call_extract_usage_payload():
    plain = [Message(USER, 'hi')]
    assert messages_have_tool_call(plain) is False
    assert messages_have_tool_call([]) is False

    with_tc = [Message(ASSISTANT, '', tool_calls=[ToolCall(function=FunctionCall(name='t', arguments='{}'))])]
    assert messages_have_tool_call(with_tc) is True

    with_fc = [Message(ASSISTANT, '', function_call=FunctionCall(name='f', arguments='{}'))]
    assert messages_have_tool_call(with_fc) is True

    assert extract_usage([]) is None
    assert extract_usage(plain) is None
    usage = {'prompt_tokens': 1, 'completion_tokens': 2}
    msgs = [
        Message(USER, 'a'),
        Message(ASSISTANT, 'b', extra={'usage': usage}),
        Message(ASSISTANT, 'c', extra={}),
    ]
    assert extract_usage(msgs) == usage

    # attribute-style message without extra usage
    fake = MagicMock()
    fake.extra = None
    assert extract_usage([fake]) is None

    payload = messages_to_payload([Message(USER, 'p')])
    assert payload[0]['role'] == 'user'
    assert payload[0]['content'] == 'p'
    assert messages_to_payload([]) == []
