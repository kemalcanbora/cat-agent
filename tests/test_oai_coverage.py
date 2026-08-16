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

"""Coverage tests for cat_agent.llm.oai (mocked OpenAI client)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cat_agent.llm.base import ModelServiceError
from cat_agent.llm.oai import (
    TextChatAtOAI,
    _merge_usage,
    _messages_from_completion_message,
)
from cat_agent.llm.schema import ASSISTANT, USER, Message
from openai import OpenAIError


def _llm(**cfg):
    base = {'model': 'gpt-test', 'api_key': 'sk-test', 'api_base': 'http://127.0.0.1:9/v1'}
    base.update(cfg)
    return TextChatAtOAI(base)


def test_merge_usage_dict_and_object():
    msg = Message(role=ASSISTANT, content='hi')
    _merge_usage(msg, None)
    assert msg.extra is None or 'usage' not in (msg.extra or {})

    _merge_usage(msg, {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3})
    assert msg.extra['usage']['total_tokens'] == 3

    msg2 = Message(role=ASSISTANT, content='x')
    _merge_usage(msg2, SimpleNamespace(prompt_tokens=4, completion_tokens=5, total_tokens=9))
    assert msg2.extra['usage']['prompt_tokens'] == 4


def test_messages_from_completion_message_variants():
    # reasoning only
    out = _messages_from_completion_message(SimpleNamespace(
        reasoning_content='think', content='', tool_calls=None,
    ))
    assert out[0].reasoning_content == 'think'

    # content + tool calls (object style)
    tc = SimpleNamespace(
        id='c1',
        function=SimpleNamespace(name='echo', arguments='{}'),
    )
    out = _messages_from_completion_message(SimpleNamespace(
        reasoning_content=None, content='hi', tool_calls=[tc],
    ))
    assert out[0].content == 'hi'
    assert out[0].tool_calls[0].function.name == 'echo'

    # tool calls only (dict style)
    out = _messages_from_completion_message(SimpleNamespace(
        reasoning_content=None,
        content=None,
        tool_calls=[{'id': 'c2', 'function': {'name': 'f', 'arguments': '{"a":1}'}}],
    ))
    assert out[0].tool_calls[0].id == 'c2'

    # content only
    out = _messages_from_completion_message(SimpleNamespace(
        reasoning_content=None, content='plain', tool_calls=[],
    ))
    assert out[0].content == 'plain'

    # empty → placeholder
    out = _messages_from_completion_message(SimpleNamespace(
        reasoning_content=None, content='', tool_calls=[],
    ))
    assert out[0].role == ASSISTANT


def test_init_api_base_aliases_and_extra_body():
    m = _llm(base_url='http://b/v1', api_key='k')
    assert m.model == 'gpt-test'
    assert m.supports_native_tools is True

    # Exercise extra_body bridging in _chat_complete_create
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='ok', reasoning_content=None, tool_calls=None,
        ))],
        usage=None,
    )
    with patch('openai.OpenAI', return_value=fake_client):
        m._chat_complete_create(
            model='gpt-test',
            messages=[{'role': 'user', 'content': 'hi'}],
            stream=False,
            top_k=5,
            repetition_penalty=1.1,
            request_timeout=3,
        )
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert 'extra_body' in kwargs
    assert kwargs['timeout'] == 3
    assert 'top_k' not in kwargs


def test_complete_create_extra_body():
    m = _llm()
    fake_client = MagicMock()
    fake_client.completions.create.return_value = 'done'
    with patch('openai.OpenAI', return_value=fake_client):
        out = m._complete_create(model='gpt-test', prompt='hi', top_k=2, request_timeout=1)
    assert out == 'done'
    assert fake_client.completions.create.call_args.kwargs['timeout'] == 1


def test_chat_no_stream_and_error():
    m = _llm()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='answer', reasoning_content=None, tool_calls=None,
        ))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    m._chat_complete_create = MagicMock(return_value=response)
    out = m._chat_no_stream([Message(role=USER, content='q')], {})
    assert out[0].content == 'answer'
    assert out[0].extra['usage']['total_tokens'] == 3

    m._chat_complete_create = MagicMock(side_effect=OpenAIError('boom'))
    with pytest.raises(ModelServiceError):
        m._chat_no_stream([Message(role=USER, content='q')], {})


def test_create_chat_stream_fallback():
    m = _llm()
    m._supports_stream_options = None
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if 'stream_options' in kwargs:
            # Servers that reject stream_options raise BadRequestError or TypeError.
            raise TypeError('unexpected keyword argument stream_options')
        return iter([])

    m._chat_complete_create = fake_create
    list(m._create_chat_stream([{'role': 'user', 'content': 'x'}], {'temperature': 0}))
    assert m._supports_stream_options is False
    assert any('stream_options' not in c for c in calls)

    # Already known unsupported
    m._supports_stream_options = False
    calls.clear()
    m._create_chat_stream([{'role': 'user', 'content': 'x'}], {'include_usage': True})
    assert all('stream_options' not in c for c in calls)

    # include_usage false
    m._supports_stream_options = None
    m._chat_complete_create = MagicMock(return_value=iter([]))
    m._create_chat_stream([], {'include_usage': False})
    assert 'stream_options' not in m._chat_complete_create.call_args.kwargs


def _chunk(delta_kw, usage=None):
    delta = SimpleNamespace(**delta_kw)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_chat_stream_delta_and_full():
    m = _llm()

    def stream_resp():
        yield _chunk({'content': 'Hel', 'reasoning_content': None, 'tool_calls': None})
        yield _chunk({'content': 'lo', 'reasoning_content': 'r', 'tool_calls': None})
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, total_tokens=2,
        ))

    m._create_chat_stream = MagicMock(return_value=stream_resp())
    deltas = list(m._chat_stream([Message(USER, 'q')], delta_stream=True, generate_cfg={}))
    assert any(d[0].content for d in deltas)

    def stream_full():
        tc = SimpleNamespace(
            index=0,
            id='call_1',
            function=SimpleNamespace(name='echo', arguments='{"a":'),
        )
        tc2 = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name=None, arguments='1}'),
        )
        yield _chunk({
            'content': '',
            'reasoning_content': 'think',
            'tool_calls': [tc],
        })
        yield _chunk({
            'content': 'text',
            'reasoning_content': None,
            'tool_calls': [tc2],
        })
        yield SimpleNamespace(
            choices=[],
            usage={'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5},
        )

    m._create_chat_stream = MagicMock(return_value=stream_full())
    full = list(m._chat_stream([Message(USER, 'q')], delta_stream=False, generate_cfg={}))
    assert full
    last = full[-1]
    assert any(getattr(x, 'tool_calls', None) for x in last) or any(
        x.content for x in last
    )


def test_chat_stream_openai_error():
    m = _llm()
    m._create_chat_stream = MagicMock(side_effect=OpenAIError('fail'))
    with pytest.raises(ModelServiceError):
        list(m._chat_stream([Message(USER, 'q')], delta_stream=False, generate_cfg={}))


def test_convert_messages_to_dicts():
    m = _llm()
    msgs = m.convert_messages_to_dicts([Message(USER, 'hello')])
    assert isinstance(msgs, list)
    assert msgs[0]['role'] == 'user'


def test_ensure_async_client_and_aclose():
    m = _llm()
    fake = MagicMock()
    fake.close = AsyncMock()
    with patch('openai.AsyncOpenAI', return_value=fake):
        c1 = m._ensure_async_client()
        c2 = m._ensure_async_client()
        assert c1 is c2 is fake

    import asyncio
    asyncio.run(m.aclose())
    fake.close.assert_awaited()
    asyncio.run(m.aclose())  # no-op when None


@pytest.mark.asyncio
async def test_achat_collects_via_bridge():
    m = _llm()
    with patch.object(
        m, 'chat',
        return_value=[Message(ASSISTANT, 'async-hi')],
    ):
        m._async_client = MagicMock()
        out = await m.achat([Message(USER, 'q')])
    assert out[0].content == 'async-hi'


def test_bridged_create_on_thread_local():
    m = _llm()
    called = {}

    def bridged(*a, **k):
        called['yes'] = True
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='b', reasoning_content=None, tool_calls=None,
            ))],
            usage=None,
        )

    m._thread_local.bridged_create = bridged
    out = m._chat_complete_create(model='m', messages=[], stream=False, top_k=1)
    assert called['yes']
    assert out.choices[0].message.content == 'b'
    m._thread_local.bridged_create = None


def test_chat_public_nonstream_and_stream():
    m = _llm()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content='pub', reasoning_content=None, tool_calls=None,
        ))],
        usage=None,
    )
    m._chat_complete_create = MagicMock(return_value=response)
    out = m.chat([Message(USER, 'q')], stream=False)
    assert out[0].content == 'pub'

    def chunks():
        yield _chunk({'content': 's', 'reasoning_content': None, 'tool_calls': None})

    m._create_chat_stream = MagicMock(return_value=chunks())
    streamed = list(m.chat([Message(USER, 'q')], stream=True, delta_stream=False))
    assert streamed
