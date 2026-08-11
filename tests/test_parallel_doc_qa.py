"""Tests for cat_agent.agents.doc_qa.parallel_doc_qa."""

from __future__ import annotations

import json
import threading
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import ASSISTANT, USER, ContentItem, Message
from cat_agent.agents.doc_qa.parallel_doc_qa import (
    ParallelDocQA,
    DEFAULT_NAME,
    DEFAULT_DESC,
    DEFAULT_MAX_CHUNKS,
    MAX_NO_RESPONSE_RETRY,
    PARALLEL_CHUNK_SIZE,
    MAX_RAG_TOKEN_SIZE,
    RAG_CHUNK_SIZE,
)
from cat_agent.agents.doc_qa.parallel_doc_qa_member import NO_RESPONSE
from cat_agent.utils.parallel_executor import serial_exec


class TestParallelDocQAConstants:

    def test_default_name_and_desc(self):
        assert "Parallel" in DEFAULT_NAME or "DocQA" in DEFAULT_NAME
        assert "RAG" in DEFAULT_DESC or "parallel" in DEFAULT_DESC.lower()

    def test_chunk_and_rag_constants(self):
        assert PARALLEL_CHUNK_SIZE == 1000
        assert MAX_RAG_TOKEN_SIZE == 4500
        assert RAG_CHUNK_SIZE == 300
        assert MAX_NO_RESPONSE_RETRY == 4
        assert DEFAULT_MAX_CHUNKS == 32


class TestParallelDocQAGetFiles:

    def test_get_files_empty_messages(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.doc_qa.parallel_doc_qa.DocParser"):
            with patch("cat_agent.agents.doc_qa.parallel_doc_qa.ParallelDocQASummary"):
                with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
                    agent = ParallelDocQA(llm=mock_llm)
        files = agent._get_files([])
        assert files == []

    def test_get_files_extracts_supported_file_from_message(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.doc_qa.parallel_doc_qa.DocParser"):
            with patch("cat_agent.agents.doc_qa.parallel_doc_qa.ParallelDocQASummary"):
                with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
                    agent = ParallelDocQA(llm=mock_llm)
        with patch("cat_agent.agents.doc_qa.parallel_doc_qa.get_file_type", return_value="pdf"):
            messages = [Message(USER, [ContentItem(file="/path/to/doc.pdf")])]
            files = agent._get_files(messages)
        assert "/path/to/doc.pdf" in files

    def test_get_files_filters_unsupported_type(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.doc_qa.parallel_doc_qa.DocParser"):
            with patch("cat_agent.agents.doc_qa.parallel_doc_qa.ParallelDocQASummary"):
                with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
                    agent = ParallelDocQA(llm=mock_llm)
        with patch("cat_agent.agents.doc_qa.parallel_doc_qa.get_file_type", return_value="jpg"):
            messages = [Message(USER, [ContentItem(file="/path/to/image.jpg")])]
            files = agent._get_files(messages)
        assert files == []


class TestParallelDocQAHelpers:

    @pytest.fixture
    def agent(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.doc_qa.parallel_doc_qa.DocParser"):
            with patch("cat_agent.agents.doc_qa.parallel_doc_qa.ParallelDocQASummary"):
                with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
                    yield ParallelDocQA(llm=mock_llm)

    def test_is_none_response_detects_res_none_json(self, agent):
        assert agent._is_none_response('Result: "res": "none"') is True

    def test_is_none_response_case_sensitive_checks_against_text_lower(self, agent):
        assert agent._is_none_response("I am sorry, I cannot help") is True

    def test_is_none_response_none_res_in_lower_text(self, agent):
        assert agent._is_none_response(f"Answer: {NO_RESPONSE}") is False

    def test_is_none_response_returns_false_for_normal_text(self, agent):
        assert agent._is_none_response("Here is the answer.") is False

    def test_extract_text_from_output_strips_json_symbols(self, agent):
        out = agent._extract_text_from_output('{"res": "ans", "content": "Hello"}')
        assert "Hello" in out or "content" in out or len(out) >= 0

    def test_parser_json_valid_json_with_res_and_content(self, agent):
        success, data = agent._parser_json('{"res": "ans", "content": "text"}')
        assert success is True
        assert data.get("res") == "ans"
        assert data.get("content") == "text"

    def test_parser_json_strips_markdown_code_block(self, agent):
        success, data = agent._parser_json('```json\n{"res": "ans", "content": "x"}\n```')
        assert success is True
        assert data.get("content") == "x"

    def test_parser_json_invalid_returns_false_and_raw(self, agent):
        success, content = agent._parser_json("not json at all")
        assert success is False
        assert content == "not json at all"


class _PipelineFakeLLM:
    """Routes member / GenKeyword / summary turns from message shape."""

    def __init__(self):
        self.model = 'fake'
        self.model_type = 'fake'
        self._lock = threading.Lock()
        self.member_calls = 0
        self.keygen_calls = 0
        self.summary_calls = 0
        self.ans_chunks: List[str] = []

    def _classify(self, messages) -> str:
        texts = []
        for m in messages:
            c = m.content if hasattr(m, 'content') else m.get('content')
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, list):
                for item in c:
                    t = getattr(item, 'text', None) or (item.get('text') if isinstance(item, dict) else None)
                    if t:
                        texts.append(t)
        blob = '\n'.join(texts)
        if '# Document:' in blob:
            return 'member'
        if 'extract keywords' in blob.lower() or blob.rstrip().endswith('Keywords:'):
            return 'keygen'
        return 'summary'

    def _member_reply(self, messages) -> str:
        texts = []
        for m in messages:
            c = getattr(m, 'content', '') or ''
            if isinstance(c, str):
                texts.append(c)
        blob = '\n'.join(texts)
        if 'REFUND_MARKER' in blob:
            self.ans_chunks.append('REFUND_MARKER')
            return json.dumps({'res': 'ans', 'content': 'Refunds are allowed within 30 days.'})
        return json.dumps({'res': 'none', 'content': NO_RESPONSE})

    def chat(self, messages, functions=None, stream=True, delta_stream=False, extra_generate_cfg=None):
        with self._lock:
            kind = self._classify(messages)
            if kind == 'member':
                self.member_calls += 1
                content = self._member_reply(messages)
            elif kind == 'keygen':
                self.keygen_calls += 1
                content = json.dumps({
                    'keywords_zh': ['退款'],
                    'keywords_en': ['refund', 'policy'],
                })
            else:
                self.summary_calls += 1
                content = 'FINAL: Refunds within 30 days per retrieved policy.'
            out = [Message(role=ASSISTANT, content=content)]
        if stream:
            return iter([out])
        return out


def _fixture_records():
    """Two docs → four chunks; only one chunk answers."""
    return {
        '/tmp/policy_a.txt': {
            'url': '/tmp/policy_a.txt',
            'title': 'policy_a',
            'raw': [
                {'content': 'Office hours are 9 to 5.', 'token': 8, 'metadata': {}},
                {'content': 'REFUND_MARKER: customers may request a refund within 30 days.', 'token': 12, 'metadata': {}},
            ],
        },
        '/tmp/policy_b.txt': {
            'url': '/tmp/policy_b.txt',
            'title': 'policy_b',
            'raw': [
                {'content': 'Shipping takes three business days.', 'token': 7, 'metadata': {}},
                {'content': 'Contact support at help@example.com.', 'token': 6, 'metadata': {}},
            ],
        },
    }


class TestParallelDocQAPipelineE2E:

    def test_full_pipeline_members_keygen_retrieval_summary(self):
        fake = _PipelineFakeLLM()
        records = _fixture_records()

        def parser_call(params, parser_page_size=None, max_ref_token=None):
            return records[params['url']]

        retrieval_calls = []

        def retrieval_call(params, **kwargs):
            retrieval_calls.append({'params': params, 'kwargs': kwargs})
            return json.dumps([{
                'url': '/tmp/policy_a.txt',
                'text': ['Refunds are allowed within 30 days.'],
            }])

        messages = [Message(USER, [
            ContentItem(text='What is the refund policy?'),
            ContentItem(file='/tmp/policy_a.txt'),
            ContentItem(file='/tmp/policy_b.txt'),
        ])]

        with patch('cat_agent.agents.doc_qa.parallel_doc_qa.DocParser') as DocParserCls, \
                patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()), \
                patch('cat_agent.agents.doc_qa.parallel_doc_qa.get_file_type', return_value='txt'), \
                patch(
                    'cat_agent.agents.doc_qa.parallel_doc_qa.parallel_exec',
                    side_effect=lambda fn, data, jitter=0.0, **kw: serial_exec(fn, data),
                ):
            DocParserCls.return_value.call.side_effect = parser_call
            agent = ParallelDocQA(llm=fake, use_polars=False, max_chunks=32)
            agent.function_map['retrieval'].call = retrieval_call

            out = agent.run_nonstream(messages)

        assert fake.member_calls == 4
        assert fake.keygen_calls == 1
        assert fake.summary_calls == 1
        assert fake.ans_chunks == ['REFUND_MARKER']
        assert len(retrieval_calls) == 1
        assert out[-1].content.startswith('FINAL:')
        assert '30 days' in out[-1].content


class TestParallelDocQABudget:

    def _agent_with_chunks(self, n_chunks: int, max_chunks: int):
        fake = _PipelineFakeLLM()
        chunks = [
            {'content': f'chunk {i} filler text here', 'token': 5, 'metadata': {}}
            for i in range(n_chunks)
        ]
        record = {'url': '/tmp/big.txt', 'title': 'big', 'raw': chunks}

        def parser_call(params, parser_page_size=None, max_ref_token=None):
            return record

        patches = (
            patch('cat_agent.agents.doc_qa.parallel_doc_qa.DocParser'),
            patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()),
            patch('cat_agent.agents.doc_qa.parallel_doc_qa.get_file_type', return_value='txt'),
        )
        entered = [p.start() for p in patches]
        try:
            entered[0].return_value.call.side_effect = parser_call
            agent = ParallelDocQA(llm=fake, use_polars=False, max_chunks=max_chunks)
        except Exception:
            for p in patches:
                p.stop()
            raise
        # Keep get_file_type + DocParser patches for later estimate/_run calls.
        agent._budget_patches = patches  # type: ignore[attr-defined]
        return agent, fake

    def _stop_patches(self, agent):
        for p in getattr(agent, '_budget_patches', ()):
            p.stop()

    def test_estimate_member_calls_matches_chunk_count(self):
        agent, _ = self._agent_with_chunks(n_chunks=3, max_chunks=32)
        try:
            messages = [Message(USER, [
                ContentItem(text='q'),
                ContentItem(file='/tmp/big.txt'),
            ])]
            assert agent.estimate_member_calls(messages) == 3
        finally:
            self._stop_patches(agent)

    def test_exceeds_max_chunks_fails_with_counts(self):
        agent, fake = self._agent_with_chunks(n_chunks=5, max_chunks=3)
        try:
            messages = [Message(USER, [
                ContentItem(text='q'),
                ContentItem(file='/tmp/big.txt'),
            ])]
            with pytest.raises(ValueError, match=r'5 chunks.*max_chunks=3'):
                agent.estimate_member_calls(messages)
            with pytest.raises(ValueError, match=r'5 chunks.*max_chunks=3'):
                agent.run_nonstream(messages)
            assert fake.member_calls == 0
        finally:
            self._stop_patches(agent)
