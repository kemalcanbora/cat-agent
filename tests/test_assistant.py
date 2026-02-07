"""Tests for cat_agent.agents.assistant."""

import json
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import SYSTEM, USER, Message
from cat_agent.agents.assistant import (
    KNOWLEDGE_SNIPPET,
    KNOWLEDGE_TEMPLATE,
    format_knowledge_to_source_and_content,
    get_current_date_str,
)


class TestFormatKnowledgeToSourceAndContent:

    def test_string_non_json_appends_single_doc(self):
        out = format_knowledge_to_source_and_content("plain text")
        assert len(out) == 1
        assert out[0]["source"] == "Uploaded document"
        assert out[0]["content"] == "plain text"

    def test_string_valid_json_list(self):
        data = [{"url": "http://x.com/doc.pdf", "text": ["snippet1", "snippet2"]}]
        out = format_knowledge_to_source_and_content(json.dumps(data))
        assert len(out) == 1
        assert "doc.pdf" in out[0]["source"]
        assert "snippet1" in out[0]["content"] and "snippet2" in out[0]["content"]

    def test_list_of_docs(self):
        data = [
            {"url": "http://a.com/1.pdf", "text": ["a1"]},
            {"url": "http://b.com/2.pdf", "text": ["b1", "b2"]},
        ]
        out = format_knowledge_to_source_and_content(data)
        assert len(out) == 2
        assert "1.pdf" in out[0]["source"]
        assert out[0]["content"] == "a1"
        assert "2.pdf" in out[1]["source"]
        assert "b1" in out[1]["content"] and "b2" in out[1]["content"]


class TestGetCurrentDateStr:

    def test_en_format(self):
        s = get_current_date_str(lang="en")
        assert s.startswith("Current date: ")
        assert "," in s

    def test_zh_format(self):
        s = get_current_date_str(lang="zh")
        assert s.startswith("Current time: ")
        assert any(day in s for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    def test_hours_from_utc(self):
        s = get_current_date_str(lang="en", hours_from_utc=0)
        assert "Current date:" in s

    def test_invalid_lang_raises(self):
        with pytest.raises(NotImplementedError):
            get_current_date_str(lang="de")


class TestKnowledgeConstants:

    def test_templates_have_placeholders(self):
        assert "{knowledge}" in KNOWLEDGE_TEMPLATE["en"]
        assert "{source}" in KNOWLEDGE_SNIPPET["en"]
        assert "{content}" in KNOWLEDGE_SNIPPET["zh"]


class TestAssistant:

    def test_inherits_from_fncall_agent(self):
        from cat_agent.agent import Agent
        from cat_agent.agents.assistant import Assistant

        assert issubclass(Assistant, Agent)

    def test_prepend_knowledge_with_explicit_knowledge(self):
        from cat_agent.agents.assistant import Assistant

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            asst = Assistant(llm=mock_llm, files=[])
            messages = [Message(role=USER, content="Hi")]
            out = asst._prepend_knowledge_prompt(messages, lang="en", knowledge='[{"url":"u","text":["c"]}]')
            assert len(out) >= 1
            assert out[0].role == SYSTEM
            assert "c" in str(out[0].content)
            assert "u" in str(out[0].content) or "[file]" in str(out[0].content)
