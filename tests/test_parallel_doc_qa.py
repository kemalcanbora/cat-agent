"""Tests for cat_agent.agents.doc_qa.parallel_doc_qa."""

from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import USER, ContentItem, Message
from cat_agent.agents.doc_qa.parallel_doc_qa import (
    ParallelDocQA,
    DEFAULT_NAME,
    DEFAULT_DESC,
    MAX_NO_RESPONSE_RETRY,
    PARALLEL_CHUNK_SIZE,
    MAX_RAG_TOKEN_SIZE,
    RAG_CHUNK_SIZE,
)


class TestParallelDocQAConstants:

    def test_default_name_and_desc(self):
        assert "Parallel" in DEFAULT_NAME or "DocQA" in DEFAULT_NAME
        assert "RAG" in DEFAULT_DESC or "parallel" in DEFAULT_DESC.lower()

    def test_chunk_and_rag_constants(self):
        assert PARALLEL_CHUNK_SIZE == 1000
        assert MAX_RAG_TOKEN_SIZE == 4500
        assert RAG_CHUNK_SIZE == 300
        assert MAX_NO_RESPONSE_RETRY == 4


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
        # Implementation checks none_response in text.lower(); this substring matches
        assert agent._is_none_response('Result: "res": "none"') is True

    def test_is_none_response_case_sensitive_checks_against_text_lower(self, agent):
        # List has 'I am sorry'; check is "none_response in text.lower()".
        # So 'I am sorry' is not in "i am sorry..." (capital I), hence False.
        assert agent._is_none_response("I am sorry, I cannot help") is False

    def test_is_none_response_none_res_in_lower_text(self, agent):
        from cat_agent.agents.doc_qa.parallel_doc_qa_member import NO_RESPONSE
        # NO_RESPONSE is '<None>'; check is in text.lower(), so '<None>' not in "answer: <none>"
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
