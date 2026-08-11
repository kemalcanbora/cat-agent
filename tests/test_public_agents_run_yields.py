"""Guard: every public Agent in agents.__all__ must produce a non-empty _run result.

Catches the ParallelDocQA failure mode — a generator ``_run`` that ``return``s an
iterator (result swallowed) or any other path that stops yielding — regardless of
which mechanism broke it.
"""

from __future__ import annotations

import json
from typing import List, Type
from unittest.mock import MagicMock, patch

import pytest

import cat_agent.agents as agents_mod
from cat_agent.agent import Agent, BasicAgent
from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, ContentItem, Message
from cat_agent.utils.parallel_executor import serial_exec


class _FixedLLM:
    def __init__(self, reply: str = 'ok'):
        self.model = 'fake'
        self.model_type = 'fake'
        self.reply = reply
        self.calls = 0

    def chat(self, messages, functions=None, stream=True, delta_stream=False, extra_generate_cfg=None):
        self.calls += 1
        text = self.reply
        blob = ''
        for m in messages or []:
            c = getattr(m, 'content', '') or ''
            if isinstance(c, str):
                blob += c
        # GroupChatAutoRouter: pick a known member name.
        if 'choose the next suitable role' in blob.lower() or 'role play game' in blob.lower():
            text = 'Bot'
        # Router system prompt contains Call:/Reply: — answer directly (no Call).
        elif 'Call:' in blob and 'Reply:' in blob:
            text = 'Hello from router.'
        out = [Message(role=ASSISTANT, content=text)]
        if stream:
            return iter([out])
        return out


def _public_agent_classes() -> List[tuple]:
    out = []
    for name in agents_mod.__all__:
        obj = getattr(agents_mod, name)
        if not isinstance(obj, type):
            continue
        if not issubclass(obj, Agent) or obj is Agent:
            continue
        out.append((name, obj))
    return out


PUBLIC_AGENTS = _public_agent_classes()


def _drain(_run_result) -> list:
    """Consume _run whether it is a generator or a plain returned iterator."""
    if _run_result is None:
        return []
    return list(_run_result)


def _build_and_run(name: str, cls: Type[Agent]):
    """Construct a minimal instance and invoke ``_run``; return drained outputs."""
    llm = _FixedLLM(reply=f'reply-from-{name}')
    user = [Message(role=USER, content='Hello')]

    if name == 'BasicAgent':
        agent = cls(llm=llm, system_message='')
        return _drain(agent._run(user, lang='en'))

    if name in ('FnCallAgent', 'Assistant', 'DocQAAgent', 'ReActChat'):
        mock_mem = MagicMock()
        retrieved = json.dumps([{'url': 't.txt', 'text': ['knowledge snippet']}])
        mock_mem.run.return_value = iter([[Message(role=ASSISTANT, content=retrieved)]])
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=mock_mem):
            kwargs = {'llm': llm, 'system_message': ''}
            if name == 'ReActChat':
                kwargs['function_list'] = ['storage']
                llm.reply = 'Final Answer: done'
            agent = cls(**kwargs)
            if name == 'ReActChat':
                with patch.object(agent, '_call_tool', return_value='obs'):
                    return _drain(agent._run(user, lang='en'))
            if name in ('Assistant', 'DocQAAgent'):
                # Assistant honors knowledge=; DocQAAgent always hits mem (mocked above).
                return _drain(agent._run(user, lang='en', knowledge=retrieved))
            return _drain(agent._run(user, lang='en'))

    if name == 'ParallelDocQA':
        records = {
            '/tmp/g.txt': {
                'url': '/tmp/g.txt',
                'title': 'g',
                'raw': [{'content': 'chunk about hello', 'token': 4, 'metadata': {}}],
            },
        }

        def parser_call(params, parser_page_size=None, max_ref_token=None):
            return records[params['url']]

        class _PipeLLM(_FixedLLM):
            def chat(self, messages, functions=None, stream=True, delta_stream=False, extra_generate_cfg=None):
                self.calls += 1
                blob = ''
                for m in messages or []:
                    c = getattr(m, 'content', '') or ''
                    if isinstance(c, str):
                        blob += c
                if '# Document:' in blob:
                    content = json.dumps({'res': 'ans', 'content': 'hello found'})
                elif 'extract keywords' in blob.lower() or blob.rstrip().endswith('Keywords:'):
                    content = json.dumps({'keywords_en': ['hello'], 'keywords_zh': []})
                else:
                    content = 'summary hello'
                out = [Message(role=ASSISTANT, content=content)]
                return iter([out]) if stream else out

        with patch('cat_agent.agents.doc_qa.parallel_doc_qa.DocParser') as DocParserCls, \
                patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()), \
                patch('cat_agent.agents.doc_qa.parallel_doc_qa.get_file_type', return_value='txt'), \
                patch(
                    'cat_agent.agents.doc_qa.parallel_doc_qa.parallel_exec',
                    side_effect=lambda fn, data, jitter=0.0, **kw: serial_exec(fn, data),
                ):
            DocParserCls.return_value.call.side_effect = parser_call
            pipe = _PipeLLM()
            agent = cls(llm=pipe, use_polars=False, max_chunks=8, system_message='')
            agent.function_map['retrieval'].call = MagicMock(return_value=json.dumps([
                {'url': '/tmp/g.txt', 'text': ['hello found']},
            ]))
            msgs = [Message(USER, [
                ContentItem(text='hi'),
                ContentItem(file='/tmp/g.txt'),
            ])]
            return _drain(agent._run(msgs, lang='en'))

    if name == 'UserAgent':
        agent = cls(name='human')
        return _drain(agent._run(user))

    if name == 'GroupChatAutoRouter':
        sub = BasicAgent(llm=llm, name='Bot', description='A bot', system_message='')
        agent = cls(llm=llm, agents=[sub], name='host')
        # _run requires messages[0] to be SYSTEM (normally prepended by Agent.run).
        msgs = [Message(role=SYSTEM, content=agent.system_message), Message(role=USER, content='Hello')]
        return _drain(agent._run(msgs, lang='en'))

    if name == 'GroupChat':
        sub = BasicAgent(llm=llm, name='Bot', description='A bot', system_message='')
        chat = cls(
            agents=[sub],
            agent_selection_method='round_robin',
            inject_hub_tools=False,
            llm=llm,
        )
        # Plain-function _run that returns an iterator (the footgun under test).
        return _drain(chat._run(user, lang='en', need_batch_response=False, max_round=1))

    if name == 'Router':
        sub = BasicAgent(llm=llm, name='Bot', description='helper', system_message='')
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            router = cls(llm=llm, agents=[sub], inject_hub_tools=False, system_message='')
            # Router rebuilds system_message in __init__; ignore empty override above —
            # actually Router always sets ROUTER_PROMPT. Call _run with user only
            # (Assistant path retrieves empty knowledge via mocked Memory).
            mock = MagicMock()
            mock.run.return_value = iter([[Message(role=ASSISTANT, content='')]])
            with patch.object(router, 'mem', mock):
                return _drain(router._run(user, lang='en'))

    raise AssertionError(f'No factory for public agent {name!r}; add one to the guard test')


@pytest.mark.parametrize('name,cls', PUBLIC_AGENTS, ids=[n for n, _ in PUBLIC_AGENTS])
def test_public_agent_run_produces_nonempty_result(name, cls):
    out = _build_and_run(name, cls)
    assert out, (
        f'{name}._run produced an empty result — likely a swallowed generator return '
        f'(return <iterator> inside a generator) or a broken early exit'
    )
    assert isinstance(out[0], list) and out[0], f'{name} yielded an empty message list'
