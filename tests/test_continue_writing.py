"""Tests for cat_agent.agents.writing.continue_writing."""

from unittest.mock import MagicMock

from cat_agent.llm.schema import CONTENT, USER, Message
from cat_agent.agents.writing.continue_writing import (
    ContinueWriting,
    PROMPT_TEMPLATE as CONTINUE_PROMPT,
)


class TestContinueWritingTemplates:

    def test_placeholders(self):
        assert "{ref_doc}" in CONTINUE_PROMPT["en"]
        assert "{user_request}" in CONTINUE_PROMPT["en"]
        assert "{ref_doc}" in CONTINUE_PROMPT["zh"]
        assert "{user_request}" in CONTINUE_PROMPT["zh"]


class TestContinueWritingRun:

    def test_run_formats_last_message_and_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(return_value=iter([[Message("assistant", "Done.")]]))
        agent = ContinueWriting(llm=mock_llm)
        messages = [Message(role=USER, content="Previous paragraph here.")]
        out = list(agent._run(messages, knowledge="Ref doc.", lang="en"))
        assert len(out) == 1
        assert len(out[0]) == 1
        call_msgs = mock_llm.chat.call_args[1]["messages"]
        last_content = call_msgs[-1][CONTENT]
        assert "Ref doc." in last_content
        assert "Previous paragraph here." in last_content
