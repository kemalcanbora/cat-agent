"""Tests for cat_agent.agents.writing.outline_writing."""

from unittest.mock import MagicMock

from cat_agent.llm.schema import CONTENT, USER, Message
from cat_agent.agents.writing.outline_writing import (
    OutlineWriting,
    PROMPT_TEMPLATE as OUTLINE_PROMPT,
)


class TestOutlineWritingTemplates:

    def test_placeholders(self):
        assert "{ref_doc}" in OUTLINE_PROMPT["en"]
        assert "{user_request}" in OUTLINE_PROMPT["en"]
        assert "outline" in OUTLINE_PROMPT["en"].lower()
        assert "{ref_doc}" in OUTLINE_PROMPT["zh"]


class TestOutlineWritingRun:

    def test_run_formats_last_message_and_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(return_value=iter([[Message("assistant", "I. Intro\nII. Body")]]))
        agent = OutlineWriting(llm=mock_llm)
        messages = [Message(role=USER, content="Write about cats.")]
        list(agent._run(messages, knowledge="Cat facts.", lang="en"))
        call_msgs = mock_llm.chat.call_args[1]["messages"]
        content = call_msgs[-1][CONTENT]
        assert "Cat facts." in content
        assert "Write about cats." in content
