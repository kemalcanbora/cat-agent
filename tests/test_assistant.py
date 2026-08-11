"""Tests for cat_agent.agents.assistant."""

import json
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
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

    def test_assistant_accepts_handlers(self):
        from cat_agent.agents.assistant import Assistant
        from cat_agent.observability.events import EventEnvelope

        class CollectingHandler:
            def on_event(self, event: EventEnvelope) -> None:
                pass

        mock_llm = MagicMock()
        mock_llm.model = 'gpt-4'
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            asst = Assistant(llm=mock_llm, handlers=[CollectingHandler()])
        assert asst._handlers

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


class _CaptureLLM:
    """Records the prompt messages seen by chat(); returns a fixed assistant reply."""

    def __init__(self, reply: str = 'ASSISTANT_REPLY'):
        self.model = 'fake'
        self.model_type = 'fake'
        self.reply = reply
        self.last_messages = None
        self.calls = 0

    def chat(self, messages, functions=None, stream=True, delta_stream=False, extra_generate_cfg=None):
        self.calls += 1
        self.last_messages = messages
        out = [Message(role=ASSISTANT, content=self.reply)]
        if stream:
            return iter([out])
        return out


def _flatten_text(messages) -> str:
    parts = []
    for m in messages or []:
        c = m.content if hasattr(m, 'content') else m.get('content')
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for item in c:
                t = getattr(item, 'text', None) or (item.get('text') if isinstance(item, dict) else None)
                if t:
                    parts.append(str(t))
    return '\n'.join(parts)


class TestAssistantRunE2E:
    """Assistant._run: RAG knowledge injection then FnCallAgent loop must yield."""

    def test_explicit_knowledge_reaches_prompt_and_response_is_yielded(self):
        from cat_agent.agents.assistant import Assistant

        fake = _CaptureLLM(reply='ANSWER_FROM_KNOWLEDGE')
        knowledge = json.dumps([{
            'url': 'fixture_policy.txt',
            'text': ['SECRET_KNOWLEDGE_MARKER refunds within 30 days'],
        }])
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            asst = Assistant(llm=fake, files=[], system_message='')

        out = list(asst.run(
            [Message(role=USER, content='What is the refund policy?')],
            knowledge=knowledge,
        ))
        assert out, 'Assistant._run must yield at least one response'
        assert out[-1][-1].content == 'ANSWER_FROM_KNOWLEDGE'
        prompt = _flatten_text(fake.last_messages)
        assert 'SECRET_KNOWLEDGE_MARKER' in prompt
        assert 'Knowledge Base' in prompt or 'content from' in prompt.lower()
        assert fake.calls >= 1

    def test_mem_retrieval_path_injects_knowledge_into_prompt(self):
        from cat_agent.agents.assistant import Assistant

        fake = _CaptureLLM(reply='MEM_PATH_ANSWER')
        retrieved = json.dumps([{
            'url': 'rag_hit.txt',
            'text': ['RETRIEVED_CHUNK_MARKER office hours 9-5'],
        }])
        mock_mem = MagicMock()
        mock_mem.run.return_value = iter([[Message(role=ASSISTANT, content=retrieved)]])

        with patch('cat_agent.agents.fncall_agent.Memory', return_value=mock_mem):
            asst = Assistant(llm=fake, files=['/tmp/doc.txt'], system_message='')

        out = list(asst.run([Message(role=USER, content='When are you open?')]))
        assert out
        assert out[-1][-1].content == 'MEM_PATH_ANSWER'
        mock_mem.run.assert_called()
        prompt = _flatten_text(fake.last_messages)
        assert 'RETRIEVED_CHUNK_MARKER' in prompt
