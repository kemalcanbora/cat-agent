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

"""Coverage tests for ParallelDocQA _run and helper branches."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.agents.doc_qa.parallel_doc_qa import (
    MAX_RAG_TOKEN_SIZE,
    ParallelDocQA,
)
from cat_agent.agents.doc_qa.parallel_doc_qa_member import NO_RESPONSE
from cat_agent.llm.schema import ASSISTANT, USER, ContentItem, Message


def _make_agent(**kwargs):
    mock_llm = MagicMock()
    mock_llm.model = 'gpt-4'
    mock_llm.model_type = 'openai'
    defaults = dict(llm=mock_llm, use_polars=False, max_chunks=32)
    defaults.update(kwargs)
    with patch('cat_agent.agents.doc_qa.parallel_doc_qa.DocParser'), \
            patch('cat_agent.agents.doc_qa.parallel_doc_qa.ParallelDocQASummary'), \
            patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
        return ParallelDocQA(**defaults)


def test_max_chunks_validation():
    with pytest.raises(ValueError, match='max_chunks'):
        _make_agent(max_chunks=0)


def test_parse_and_chunk_files():
    agent = _make_agent()
    agent.doc_parse.call.return_value = {
        'url': '/tmp/a.txt', 'title': 'a', 'raw': [{'content': 'c', 'token': 1}],
    }
    with patch.object(agent, '_get_files', return_value=['/tmp/a.txt']):
        records = agent._parse_and_chunk_files([Message(USER, 'q')])
    assert len(records) == 1
    agent.doc_parse.call.assert_called_once()


def test_prepare_parallel_data_standard_skips_empty():
    agent = _make_agent(use_polars=False)
    records = [
        {'url': 'u', 'raw': []},
        {'url': 'u2', 'raw': [
            {'content': 'chunk-a', 'token': 2},
            {'content': 'chunk-b', 'token': 3},
        ]},
    ]
    data = agent._prepare_parallel_data_standard(records, [Message(USER, 'q')], 'en', 'q')
    assert len(data) == 2
    assert data[0]['knowledge'] == 'chunk-a'


def test_prepare_parallel_data_polars_fallback_and_empty():
    agent = _make_agent(use_polars=True)
    # Force polars path unavailable → standard
    agent.use_polars = False
    records = [{'url': 'u', 'raw': [{'content': 'long enough text here', 'token': 5}]}]
    data = agent._prepare_parallel_data_polars(records, [Message(USER, 'q')], 'en', 'q')
    assert len(data) == 1

    agent.use_polars = True
    with patch('cat_agent.agents.doc_qa.parallel_doc_qa.POLARS_AVAILABLE', False):
        data = agent._prepare_parallel_data_polars(records, [Message(USER, 'q')], 'en', 'q')
        assert len(data) == 1

    # Empty raw → warning / empty
    agent.use_polars = False
    assert agent._prepare_parallel_data_polars(
        [{'url': 'u', 'raw': []}], [Message(USER, 'q')], 'en', 'q',
    ) == []


def test_prepare_parallel_data_polars_real_path():
    pl = pytest.importorskip('polars')
    agent = _make_agent(use_polars=True)
    records = [{
        'url': '/d.txt',
        'title': 'd',
        'raw': [
            {'content': 'short', 'token': 0, 'metadata': {}},  # filtered
            {'content': 'this is a long enough chunk about refunds', 'token': 12, 'metadata': {}},
            {'content': 'another long enough chunk about shipping', 'token': 11, 'metadata': {}},
        ],
    }]
    data = agent._prepare_parallel_data_polars(
        records, [Message(USER, 'what about refunds?')], 'en', 'what about refunds?',
    )
    assert len(data) >= 1
    assert 'knowledge' in data[0]
    assert data[0]['lang'] == 'en'


def test_smart_chunk_selection_polars():
    pl = pytest.importorskip('polars')
    agent = _make_agent(use_polars=True)
    agent.max_chunks_per_question = 2
    df = pl.DataFrame({
        'index': [0, 1, 2, 3],
        'chunk_content': [
            'refund policy details here',
            'office hours are nine to five',
            'shipping takes three days',
            'refund window is thirty days',
        ],
        'chunk_token': [120, 50, 200, 400],
    })
    selected = agent._smart_chunk_selection_polars(df, 'refund policy details', max_chunks=2)
    assert len(selected) == 2

    # No keywords → head
    selected2 = agent._smart_chunk_selection_polars(df, 'a b c', max_chunks=1)
    assert len(selected2) == 1


def test_aggregate_results_standard_and_polars():
    agent = _make_agent(use_polars=False)
    results = [
        (1, json.dumps({'res': 'ans', 'content': 'Answer one'})),
        (0, json.dumps({'res': 'none', 'content': NO_RESPONSE})),
        (2, 'I am sorry, cannot help'),
        (3, 'plain text answer without json'),
        (4, 'not useful'),
    ]
    member_res, filtered = agent._aggregate_results_standard(results)
    assert 'Answer one' in member_res
    assert any('plain text' in t for _, t in filtered)

    # empty
    assert agent._aggregate_results_standard([]) == ('', [])

    # polars path
    agent.use_polars = True
    pl = pytest.importorskip('polars')
    member_res2, filtered2 = agent._aggregate_results_polars([
        (0, json.dumps({'res': 'ans', 'content': 'P-ans'})),
        (1, json.dumps({'res': 'none', 'content': 'n'})),
    ])
    assert 'P-ans' in member_res2

    # polars empty after filter
    assert agent._aggregate_results_polars([
        (0, json.dumps({'res': 'none', 'content': 'n'})),
    ]) == ('', [])

    # fallback when polars unavailable
    with patch('cat_agent.agents.doc_qa.parallel_doc_qa.POLARS_AVAILABLE', False):
        mr, fr = agent._aggregate_results_polars([
            (0, json.dumps({'res': 'ans', 'content': 'x'})),
        ])
        assert 'x' in mr

    # exception fallback
    with patch('cat_agent.agents.doc_qa.parallel_doc_qa.pl') as fake_pl:
        fake_pl.DataFrame.side_effect = RuntimeError('boom')
        mr, fr = agent._aggregate_results_polars([
            (0, json.dumps({'res': 'ans', 'content': 'y'})),
        ])
        assert 'y' in mr


def test_retrieve_according_to_member_responses_paths():
    agent = _make_agent()
    agent.function_map['retrieval'].call = MagicMock(return_value=json.dumps([{
        'url': '/f.txt', 'text': ['snippet body'],
    }]))

    with patch.object(agent, '_get_files', return_value=['/f.txt']), \
            patch('cat_agent.agents.doc_qa.parallel_doc_qa.GenKeyword') as GK, \
            patch('cat_agent.agents.doc_qa.parallel_doc_qa.count_tokens', return_value=10), \
            patch(
                'cat_agent.agents.doc_qa.parallel_doc_qa.format_knowledge_to_source_and_content',
                return_value=[{'source': '/f.txt', 'content': 'snippet body'}],
            ):
        gk = GK.return_value
        gk.run.return_value = iter([
            [Message(ASSISTANT, '```json\n{"keywords_en": ["a"]}\n```')],
        ])
        out = agent._retrieve_according_to_member_responses(
            messages=[Message(USER, [ContentItem(text='q'), ContentItem(file='/f.txt')])],
            lang='en',
            user_question='q',
            member_res='member text',
        )
        assert 'snippet body' in out

    # Oversized member_res + invalid json keyword → query fallback path
    with patch.object(agent, '_get_files', return_value=['/f.txt']), \
            patch('cat_agent.agents.doc_qa.parallel_doc_qa.GenKeyword') as GK, \
            patch(
                'cat_agent.agents.doc_qa.parallel_doc_qa.count_tokens',
                return_value=MAX_RAG_TOKEN_SIZE + 1,
            ), \
            patch(
                'cat_agent.agents.doc_qa.parallel_doc_qa.format_knowledge_to_source_and_content',
                return_value=[{'source': 's', 'content': 'c'}],
            ):
        gk = GK.return_value
        gk.run.return_value = iter([[Message(ASSISTANT, 'not-json')]])
        agent.function_map['retrieval'].call = MagicMock(return_value=[
            {'url': 'u', 'text': ['t']},
        ])
        out = agent._retrieve_according_to_member_responses(
            messages=[Message(USER, 'q')],
            lang='en',
            user_question='q',
            member_res='huge',
        )
        assert out


def test_run_no_records_and_no_chunks():
    agent = _make_agent()
    with patch.object(agent, '_parse_and_chunk_files', return_value=[]):
        out = list(agent._run([Message(USER, 'q')]))
        assert 'No valid documents' in out[0][0].content

    with patch.object(agent, '_parse_and_chunk_files', return_value=[{'raw': []}]), \
            patch.object(agent, '_prepare_parallel_data_polars', return_value=[]):
        out = list(agent._run([Message(USER, 'q')]))
        assert 'No valid content' in out[0][0].content


def test_run_retry_then_summary():
    agent = _make_agent(use_polars=False)
    records = [{'url': '/t.txt', 'raw': [
        {'content': 'chunk text long enough', 'token': 5},
    ]}]
    summary_msgs = [[Message(ASSISTANT, 'SUMMARY')]]

    call_count = {'n': 0}

    def fake_parallel(fn, data, jitter=0.0, **kw):
        call_count['n'] += 1
        if call_count['n'] == 1:
            return [(0, json.dumps({'res': 'none', 'content': NO_RESPONSE}))]
        return [(0, json.dumps({'res': 'ans', 'content': 'good answer'}))]

    agent.summary_agent.run = MagicMock(return_value=iter(summary_msgs))

    with patch.object(agent, '_parse_and_chunk_files', return_value=records), \
            patch(
                'cat_agent.agents.doc_qa.parallel_doc_qa.parallel_exec',
                side_effect=fake_parallel,
            ), \
            patch.object(
                agent,
                '_retrieve_according_to_member_responses',
                return_value='KB',
            ):
        out = list(agent._run([Message(USER, 'question?')], lang='en'))

    assert call_count['n'] == 2
    assert out[-1][0].content == 'SUMMARY'
    agent.summary_agent.run.assert_called()


def test_ask_member_agent():
    agent = _make_agent()
    member = MagicMock()
    member.run.return_value = iter([[Message(ASSISTANT, 'member-out')]])
    with patch('cat_agent.agents.doc_qa.parallel_doc_qa.ParallelDocQAMember', return_value=member):
        idx, content = agent._ask_member_agent(
            index=3,
            messages=[Message(USER, 'q')],
            lang='en',
            knowledge='kb',
            instruction='q',
        )
    assert idx == 3
    assert content == 'member-out'


def test_prepare_polars_exception_fallback():
    agent = _make_agent(use_polars=True)
    records = [{'url': 'u', 'raw': [{'content': 'long enough content xx', 'token': 5}]}]
    with patch('cat_agent.agents.doc_qa.parallel_doc_qa.pl') as fake_pl:
        fake_pl.DataFrame.side_effect = RuntimeError('polars down')
        data = agent._prepare_parallel_data_polars(
            records, [Message(USER, 'q')], 'en', 'q',
        )
    assert len(data) == 1


def test_prepare_polars_smart_selection_trigger():
    pl = pytest.importorskip('polars')
    agent = _make_agent(use_polars=True)
    agent.max_chunks_per_question = 1
    records = [{
        'url': 'u',
        'title': 't',
        'raw': [
            {'content': 'alpha refund topic text here', 'token': 10, 'metadata': {}},
            {'content': 'beta shipping topic text here', 'token': 10, 'metadata': {}},
        ],
    }]
    data = agent._prepare_parallel_data_polars(
        records, [Message(USER, 'refund topic')], 'en', 'refund topic please',
    )
    assert len(data) == 1


def test_retrieve_non_string_content():
    agent = _make_agent()
    agent.function_map['retrieval'].call = MagicMock(return_value={'docs': [1]})
    with patch.object(agent, '_get_files', return_value=['/f']), \
            patch('cat_agent.agents.doc_qa.parallel_doc_qa.GenKeyword') as GK, \
            patch('cat_agent.agents.doc_qa.parallel_doc_qa.count_tokens', return_value=1), \
            patch(
                'cat_agent.agents.doc_qa.parallel_doc_qa.format_knowledge_to_source_and_content',
                return_value=[{'source': 's', 'content': 'body'}],
            ):
        GK.return_value.run.return_value = iter([
            [Message(ASSISTANT, '{"keywords_en":["x"]}')],
        ])
        out = agent._retrieve_according_to_member_responses(
            messages=[Message(USER, 'q')],
            lang='zh',
            user_question='q',
            member_res='m',
        )
        assert 'body' in out
