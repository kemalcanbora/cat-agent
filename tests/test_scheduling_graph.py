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

"""Tests for cat_agent.scheduling.graph."""

from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cat_agent.graph import END
from cat_agent.graph.state import GraphState
from cat_agent.llm.schema import ASSISTANT, USER, Message
from cat_agent.scheduling.channels.base import DeliveryResult
from cat_agent.scheduling.graph import (
    _near_dup_key,
    build_collector,
    build_report_graph,
    build_report_markdown,
    build_reporter,
    dedupe_sources,
    deliver_report,
    run_collector,
    sources_to_prompt,
)
from cat_agent.scheduling.models import Job, Source


def _job(**overrides) -> Job:
    base = dict(
        id='report:alice:ai',
        user_id='alice',
        kind='report',
        topic='AI news',
        channel='smtp',
        target='alice@example.com',
        interval_seconds=3600,
    )
    base.update(overrides)
    return Job(**base)


def _source(**overrides) -> Source:
    base = dict(
        id='src-1',
        user_id='alice',
        url='https://example.com/a',
        title='Alpha',
        summary='first summary',
        collected_at=1.0,
    )
    base.update(overrides)
    return Source(**base)


def _compiled_nodes(reporter=None):
    """Compile graph and return (agent, nodes dict, fetch router)."""
    agent = build_report_graph(reporter=reporter or MagicMock(name='reporter'))
    return agent, agent.graph.nodes, agent.graph.branches['fetch']


# ---------------------------------------------------------------------------
# _near_dup_key / dedupe_sources
# ---------------------------------------------------------------------------


class TestNearDupAndDedupe:

    def test_near_dup_key_prefers_content_hash(self):
        src = _source(content_hash='abc123')
        assert _near_dup_key(src) == 'abc123'

    def test_near_dup_key_hashes_title_and_summary(self):
        src = _source(title='Hello', summary='World', content_hash=None)
        blob = 'hello\nworld'
        expected = hashlib.sha256(blob.encode('utf-8')).hexdigest()[:32]
        assert _near_dup_key(src) == expected

    def test_near_dup_key_normalizes_case_and_whitespace(self):
        a = _source(title='  Hello  ', summary='World', content_hash=None)
        b = _source(title='hello', summary='world', content_hash=None, id='src-2')
        # strip only on the combined blob, not per field — verify stable hashing
        assert _near_dup_key(b) == hashlib.sha256(
            'hello\nworld'.encode('utf-8'),
        ).hexdigest()[:32]
        assert _near_dup_key(a) == hashlib.sha256(
            '  hello  \nworld'.strip().lower().encode('utf-8'),
        ).hexdigest()[:32]

    def test_dedupe_by_content_hash(self):
        a = _source(id='1', content_hash='same', url='https://a')
        b = _source(id='2', content_hash='same', url='https://b', title='Other')
        c = _source(id='3', content_hash='other', url='https://c')
        out = dedupe_sources([a, b, c])
        assert [s.id for s in out] == ['1', '3']

    def test_dedupe_by_title_summary_when_no_hash(self):
        a = _source(id='1', title='Same', summary='Body', content_hash=None)
        b = _source(
            id='2',
            title='same',
            summary='body',
            content_hash=None,
            url='https://other',
        )
        c = _source(id='3', title='Diff', summary='Body', content_hash=None)
        out = dedupe_sources([a, b, c])
        assert [s.id for s in out] == ['1', '3']

    def test_dedupe_empty(self):
        assert dedupe_sources([]) == []


# ---------------------------------------------------------------------------
# sources_to_prompt
# ---------------------------------------------------------------------------


class TestSourcesToPrompt:

    def test_basic_formatting(self):
        job = _job()
        sources = [
            _source(title='One', url='https://one', summary='s1'),
            _source(id='2', title='Two', url='https://two', summary='s2'),
        ]
        text = sources_to_prompt(job, sources)
        assert 'Topic: AI news' in text
        assert 'User: alice' in text
        assert 'Sources (2):' in text
        assert '1. One' in text
        assert '   URL: https://one' in text
        assert '   Summary: s1' in text
        assert '2. Two' in text

    def test_includes_tags_when_present(self):
        job = _job()
        src = _source(tags='science,ai')
        text = sources_to_prompt(job, [src])
        assert '   Tags: science,ai' in text

    def test_omits_tags_when_empty(self):
        job = _job()
        src = _source(tags='')
        text = sources_to_prompt(job, [src])
        assert 'Tags:' not in text


# ---------------------------------------------------------------------------
# build_collector / build_reporter
# ---------------------------------------------------------------------------


class TestBuilders:

    def test_build_collector_wires_name_and_tools(self):
        mock_assistant = MagicMock(name='collector_instance')
        with patch('cat_agent.scheduling.graph.Assistant', return_value=mock_assistant) as ctor:
            out = build_collector(llm={'model': 'x'})
        assert out is mock_assistant
        kwargs = ctor.call_args.kwargs
        assert kwargs['name'] == 'source_collector'
        assert kwargs['function_list'] == ['web_search', 'web_extractor', 'save_source']
        assert kwargs['llm'] == {'model': 'x'}
        assert 'You are a source collector' in kwargs['system_message']

    def test_build_reporter_wires_name_and_no_tools(self):
        mock_assistant = MagicMock(name='reporter_instance')
        with patch('cat_agent.scheduling.graph.Assistant', return_value=mock_assistant) as ctor:
            out = build_reporter()
        assert out is mock_assistant
        kwargs = ctor.call_args.kwargs
        assert kwargs['name'] == 'source_reporter'
        assert kwargs['function_list'] == []
        assert 'You are a report writer' in kwargs['system_message']

    def test_build_reporter_real_instance_accepts_empty_tools(self):
        mock_llm = MagicMock()
        mock_llm.model = 'm'
        mock_llm.model_type = 'openai'
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            reporter = build_reporter(llm=mock_llm)
        assert reporter.name == 'source_reporter'
        assert reporter.function_map == {}


# ---------------------------------------------------------------------------
# build_report_graph FunctionNodes
# ---------------------------------------------------------------------------


class TestReportGraphNodes:

    def test_compile_returns_named_graph_agent(self):
        agent = build_report_graph(reporter=MagicMock())
        assert agent.name == 'ScheduledReportGraph'
        assert set(agent.graph.nodes) >= {
            'fetch', 'dedupe', 'prepare_summarize', 'summarize',
            'after_summarize', 'redact', 'deliver',
        }
        assert agent.graph.entry == 'fetch'
        assert 'fetch' in agent.graph.branches

    def test_fetch_loads_undelivered_when_sources_missing(self):
        _, nodes, _ = _compiled_nodes()
        store = MagicMock()
        sources = [_source()]
        store.list_undelivered.return_value = sources
        job = _job()
        state = GraphState(scratch={
            'store': store,
            'job': job,
            'max_items': 7,
        })
        out = nodes['fetch'].fn(state)
        store.list_undelivered.assert_called_once_with('alice', max_items=7)
        assert out.scratch['sources'] is sources

    def test_fetch_keeps_preloaded_sources(self):
        _, nodes, _ = _compiled_nodes()
        store = MagicMock()
        preloaded = [_source(id='pre')]
        state = GraphState(scratch={
            'store': store,
            'job': _job(),
            'sources': preloaded,
        })
        nodes['fetch'].fn(state)
        store.list_undelivered.assert_not_called()
        assert state.scratch['sources'] is preloaded

    def test_fetch_defaults_max_items_to_50(self):
        _, nodes, _ = _compiled_nodes()
        store = MagicMock()
        store.list_undelivered.return_value = []
        state = GraphState(scratch={'store': store, 'job': _job()})
        nodes['fetch'].fn(state)
        store.list_undelivered.assert_called_once_with('alice', max_items=50)

    def test_route_after_fetch_empty_sets_flag_and_ends(self):
        _, _, route = _compiled_nodes()
        state = GraphState(scratch={'sources': []})
        assert route(state) == END
        assert state.scratch['empty'] is True

    def test_route_after_fetch_none_sources_ends(self):
        _, _, route = _compiled_nodes()
        state = GraphState(scratch={})
        assert route(state) == END
        assert state.scratch['empty'] is True

    def test_route_after_fetch_with_sources_goes_dedupe(self):
        _, _, route = _compiled_nodes()
        state = GraphState(scratch={'sources': [_source()]})
        assert route(state) == 'dedupe'
        assert 'empty' not in state.scratch

    def test_dedupe_node_filters_near_dups(self):
        _, nodes, _ = _compiled_nodes()
        a = _source(id='1', content_hash='h')
        b = _source(id='2', content_hash='h', title='dup')
        state = GraphState(scratch={'sources': [a, b]})
        nodes['dedupe'].fn(state)
        assert [s.id for s in state.scratch['sources']] == ['1']

    def test_prepare_summarize_sets_user_message(self):
        _, nodes, _ = _compiled_nodes()
        job = _job(topic='Widgets')
        sources = [_source(title='T', url='https://t', summary='sum')]
        state = GraphState(scratch={'job': job, 'sources': sources})
        nodes['prepare_summarize'].fn(state)
        assert len(state.messages) == 1
        assert state.messages[0].role == USER
        assert 'Topic: Widgets' in state.messages[0].content
        assert '1. T' in state.messages[0].content

    def test_after_summarize_reads_last_assistant_message(self):
        _, nodes, _ = _compiled_nodes()
        state = GraphState(messages=[
            Message(role=USER, content='prompt'),
            Message(role=ASSISTANT, content='# Report body'),
        ])
        nodes['after_summarize'].fn(state)
        assert state.scratch['report_markdown'] == '# Report body'

    def test_after_summarize_supports_dict_messages(self):
        _, nodes, _ = _compiled_nodes()
        state = GraphState(messages=[
            {'role': 'user', 'content': 'x'},
            {'role': 'assistant', 'content': 'from-dict'},
        ])
        nodes['after_summarize'].fn(state)
        assert state.scratch['report_markdown'] == 'from-dict'

    def test_after_summarize_stringifies_non_str_content(self):
        _, nodes, _ = _compiled_nodes()
        state = GraphState(messages=[
            {'role': 'assistant', 'content': ['chunk']},
        ])
        nodes['after_summarize'].fn(state)
        assert state.scratch['report_markdown'] == "['chunk']"

    def test_after_summarize_empty_when_no_assistant(self):
        _, nodes, _ = _compiled_nodes()
        state = GraphState(messages=[Message(role=USER, content='only user')])
        nodes['after_summarize'].fn(state)
        assert state.scratch['report_markdown'] == ''

    def test_redact_node_redacts_pii(self):
        _, nodes, _ = _compiled_nodes()
        state = GraphState(scratch={
            'report_markdown': 'Contact me at alice@example.com please',
        })
        nodes['redact'].fn(state)
        assert 'alice@example.com' not in state.scratch['report_markdown']
        assert '[PII]' in state.scratch['report_markdown']

    def test_deliver_dry_run_skips_send(self):
        _, nodes, _ = _compiled_nodes()
        runner = MagicMock()
        state = GraphState(scratch={
            'dry_run': True,
            'job': _job(),
            'report_markdown': '# hi',
            'async_runner': runner,
        })
        out = nodes['deliver'].fn(state)
        runner.assert_not_called()
        assert 'delivered' not in out.scratch

    def test_deliver_with_async_runner_and_channel(self):
        _, nodes, _ = _compiled_nodes()
        channel = MagicMock()
        channel.send = AsyncMock(return_value=DeliveryResult(ok=True))
        ran = []

        def runner(coro):
            ran.append(True)
            asyncio.run(coro)

        state = GraphState(scratch={
            'job': _job(topic='Rockets'),
            'report_markdown': '## body',
            'channel': channel,
            'async_runner': runner,
        })
        with patch(
            'cat_agent.scheduling.graph.markdown_to_html',
            return_value='<p>body</p>',
        ):
            nodes['deliver'].fn(state)

        assert ran == [True]
        assert state.scratch['delivered'] is True
        channel.send.assert_awaited_once()
        kwargs = channel.send.await_args.kwargs
        assert kwargs['target'] == 'alice@example.com'
        assert kwargs['subject'] == 'Scheduled report: Rockets'
        assert kwargs['body_markdown'] == '## body'
        assert kwargs['body_html'] == '<p>body</p>'

    def test_deliver_without_runner_uses_event_loop(self):
        _, nodes, _ = _compiled_nodes()
        channel = MagicMock()
        channel.send = AsyncMock(return_value=DeliveryResult(ok=True))
        loop = MagicMock()

        def _consume(coro):
            coro.close()

        loop.run_until_complete.side_effect = _consume
        state = GraphState(scratch={
            'job': _job(),
            'report_markdown': 'x',
            'channel': channel,
        })
        with patch('asyncio.get_event_loop', return_value=loop):
            nodes['deliver'].fn(state)
        loop.run_until_complete.assert_called_once()
        assert state.scratch['delivered'] is True


# ---------------------------------------------------------------------------
# build_report_markdown / deliver_report / run_collector
# ---------------------------------------------------------------------------


class TestAsyncHelpers:

    @pytest.mark.asyncio
    async def test_build_report_markdown_uses_reporter_and_redacts(self):
        job = _job()
        sources = [
            _source(id='1', content_hash='h'),
            _source(id='2', content_hash='h', title='dup'),
        ]
        reporter = MagicMock()
        reporter.arun_nonstream = AsyncMock(return_value=[
            {'role': 'assistant', 'content': 'Email bob@corp.com in report'},
        ])
        with patch(
            'cat_agent.scheduling.graph.build_reporter',
            return_value=reporter,
        ) as build:
            body = await build_report_markdown(job, sources, llm='fake')

        build.assert_called_once()
        assert build.call_args.kwargs['llm'] == 'fake'
        prompt = reporter.arun_nonstream.await_args.args[0][0]['content']
        assert 'Sources (1):' in prompt  # deduped
        assert 'bob@corp.com' not in body
        assert '[PII]' in body

    @pytest.mark.asyncio
    async def test_build_report_markdown_empty_when_no_assistant(self):
        reporter = MagicMock()
        reporter.arun_nonstream = AsyncMock(return_value=[
            {'role': 'user', 'content': 'echo'},
        ])
        with patch('cat_agent.scheduling.graph.build_reporter', return_value=reporter):
            body = await build_report_markdown(_job(), [_source()])
        assert body == ''

    @pytest.mark.asyncio
    async def test_deliver_report_sends_via_channel(self):
        job = _job(topic='Space')
        channel = MagicMock()
        channel.send = AsyncMock(return_value=DeliveryResult(ok=True, provider_id='smtp'))
        store = MagicMock()
        await deliver_report(job, '# md', [_source()], store, channel=channel)
        channel.send.assert_awaited_once()
        kwargs = channel.send.await_args.kwargs
        assert kwargs['target'] == job.target
        assert kwargs['subject'] == 'Scheduled report: Space'
        assert kwargs['body_markdown'] == '# md'

    @pytest.mark.asyncio
    async def test_deliver_report_raises_when_not_ok(self):
        job = _job()
        channel = MagicMock()
        channel.send = AsyncMock(
            return_value=DeliveryResult(ok=False, error='bounce'),
        )
        with patch(
            'cat_agent.scheduling.graph.send_with_retry',
            new=AsyncMock(return_value=DeliveryResult(ok=False, error='bounce')),
        ):
            with pytest.raises(RuntimeError, match='bounce'):
                await deliver_report(job, 'x', [], MagicMock(), channel=channel)

    @pytest.mark.asyncio
    async def test_deliver_report_raises_generic_message_without_error(self):
        job = _job()
        channel = MagicMock()
        with patch(
            'cat_agent.scheduling.graph.send_with_retry',
            new=AsyncMock(return_value=DeliveryResult(ok=False, error=None)),
        ):
            with pytest.raises(RuntimeError, match='delivery failed'):
                await deliver_report(job, 'x', [], MagicMock(), channel=channel)

    @pytest.mark.asyncio
    async def test_deliver_report_resolves_channel_from_job(self):
        job = _job(channel='webhook')
        channel = MagicMock()
        channel.send = AsyncMock(return_value=DeliveryResult(ok=True))
        with patch(
            'cat_agent.scheduling.graph.get_channel',
            return_value=channel,
        ) as get_ch:
            await deliver_report(job, 'body', [], MagicMock())
        get_ch.assert_called_once_with('webhook')
        channel.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_collector_returns_delta_and_uses_context(self):
        job = _job(kind='collect', topic='Cats')
        store = MagicMock()
        store.list_undelivered.side_effect = [
            [_source()],  # before
            [_source(), _source(id='2')],  # after
        ]
        collector = MagicMock()
        collector.arun_nonstream = AsyncMock(return_value=[])

        with patch('cat_agent.scheduling.graph.enable_optional_tools') as enable, \
                patch(
                    'cat_agent.scheduling.graph.build_collector',
                    return_value=collector,
                ) as build, \
                patch('cat_agent.scheduling.graph.scheduling_context') as ctx:
            ctx.return_value.__enter__ = MagicMock(return_value=None)
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            delta = await run_collector(job, store, llm='llm-x', handlers=['h'])

        enable.assert_called_once_with('web_search', 'web_extractor')
        build.assert_called_once_with(llm='llm-x', handlers=['h'])
        ctx.assert_called_once_with(store, job_id=job.id)
        collector.arun_nonstream.assert_awaited_once()
        prompt = collector.arun_nonstream.await_args.args[0][0]['content']
        assert 'Cats' in prompt
        assert 'user_id=alice' in prompt
        assert delta == 1
        assert store.list_undelivered.call_count == 2

    @pytest.mark.asyncio
    async def test_run_collector_never_returns_negative(self):
        job = _job()
        store = MagicMock()
        store.list_undelivered.side_effect = [
            [_source(), _source(id='2')],
            [_source()],
        ]
        collector = MagicMock()
        collector.arun_nonstream = AsyncMock(return_value=[])
        with patch('cat_agent.scheduling.graph.enable_optional_tools'), \
                patch(
                    'cat_agent.scheduling.graph.build_collector',
                    return_value=collector,
                ), \
                patch('cat_agent.scheduling.graph.scheduling_context') as ctx:
            ctx.return_value.__enter__ = MagicMock(return_value=None)
            ctx.return_value.__exit__ = MagicMock(return_value=False)
            delta = await run_collector(job, store)
        assert delta == 0
