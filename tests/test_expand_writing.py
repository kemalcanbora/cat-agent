"""Tests for cat_agent.agents.writing.expand_writing."""

from unittest.mock import MagicMock

import pytest

from cat_agent.llm.schema import CONTENT, USER, Message
from cat_agent.agents.writing.expand_writing import (
    ExpandWriting,
    PROMPT_TEMPLATE as EXPAND_PROMPT,
)


class TestExpandWritingTemplates:

    def test_placeholders(self):
        assert "{ref_doc}" in EXPAND_PROMPT["en"]
        assert "{user_request}" in EXPAND_PROMPT["en"]
        assert "{outline}" in EXPAND_PROMPT["en"]
        assert "{index}" in EXPAND_PROMPT["en"]
        assert "{capture}" in EXPAND_PROMPT["en"]
        assert "{ref_doc}" in EXPAND_PROMPT["zh"]


class TestExpandWritingRun:

    def test_run_formats_with_outline_index_capture(self):
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(return_value=iter([[Message("assistant", "Section text.")]]))
        agent = ExpandWriting(llm=mock_llm)
        messages = [Message(role=USER, content="My Title")]
        list(agent._run(
            messages,
            knowledge="Refs.",
            outline="I. Intro\nII. Body",
            index="1",
            capture="Intro",
            lang="en",
        ))
        call_msgs = mock_llm.chat.call_args[1]["messages"]
        last_content = call_msgs[-1][CONTENT]
        assert "Refs." in last_content
        assert "Intro" in last_content
        assert "I. Intro" in last_content or "outline" in last_content.lower()

    def test_run_adds_capture_later_en(self):
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(return_value=iter([[Message("assistant", "Ok.")]]))
        agent = ExpandWriting(llm=mock_llm)
        messages = [Message(role=USER, content="Title")]
        list(agent._run(
            messages,
            knowledge="",
            outline="I. A\nII. B",
            index="1",
            capture="A",
            capture_later="B",
            lang="en",
        ))
        call_msgs = mock_llm.chat.call_args[1]["messages"]
        content = call_msgs[-1][CONTENT]
        assert "B" in content
        assert "stop" in content.lower() or "Stop" in content

    def test_run_adds_capture_later_zh(self):
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(return_value=iter([[Message("assistant", "Ok.")]]))
        agent = ExpandWriting(llm=mock_llm)
        messages = [Message(role=USER, content="Title")]
        list(agent._run(
            messages,
            knowledge="",
            outline="I. A\nII. B",
            index="1",
            capture="A",
            capture_later="Next",
            lang="zh",
        ))
        call_msgs = mock_llm.chat.call_args[1]["messages"]
        content = call_msgs[-1][CONTENT]
        assert "Next" in content

    def test_run_invalid_lang_raises(self):
        mock_llm = MagicMock()
        agent = ExpandWriting(llm=mock_llm)
        messages = [Message(role=USER, content="Title")]
        with pytest.raises(KeyError):
            list(agent._run(
                messages,
                knowledge="",
                outline="I. A",
                index="1",
                capture="A",
                lang="de",
            ))
