"""Tests for cat_agent.agents.virtual_memory_agent."""

from unittest.mock import MagicMock, patch

from cat_agent.llm.schema import ASSISTANT, USER, ContentItem, Message
from cat_agent.agents.virtual_memory_agent import (
    VirtualMemoryAgent,
    DEFAULT_NAME,
    DEFAULT_DESC,
)


class TestVirtualMemoryAgent:

    def test_default_name_and_description(self):
        assert "Memory" in DEFAULT_NAME
        assert "retrieve" in DEFAULT_DESC or "information" in DEFAULT_DESC

    def test_adds_retrieval_tool(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = VirtualMemoryAgent(llm=mock_llm, files=[])
        assert "retrieval" in agent.function_map

    def test_format_file_en(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = VirtualMemoryAgent(llm=mock_llm, files=[])
        messages = [Message(role=USER, content=[ContentItem(file="/path/to/doc.pdf")])]
        out = agent._format_file(messages, lang="en")
        assert len(out) == 1
        assert "[file]" in out[0].content[0].text
        assert "doc.pdf" in out[0].content[0].text

    def test_format_file_zh(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = VirtualMemoryAgent(llm=mock_llm, files=[])
        messages = [Message(role=USER, content=[ContentItem(file="/path/to/文件.pdf")])]
        out = agent._format_file(messages, lang="zh")
        assert len(out) == 1
        assert "[file]" in out[0].content[0].text
        assert "文件.pdf" in out[0].content[0].text

    def test_format_file_preserves_non_file_content_items(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = VirtualMemoryAgent(llm=mock_llm, files=[])
        messages = [Message(role=USER, content=[
            ContentItem(text="Question about the doc"),
            ContentItem(file="/path/to/doc.pdf"),
        ])]
        out = agent._format_file(messages, lang="en")
        assert len(out) == 1
        assert len(out[0].content) == 2
        assert out[0].content[0].text == "Question about the doc"
        assert "[file]" in out[0].content[1].text

    def test_format_file_non_list_content_passthrough(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = VirtualMemoryAgent(llm=mock_llm, files=[])
        messages = [Message(role=USER, content="Just text")]
        out = agent._format_file(messages, lang="en")
        assert len(out) == 1
        assert out[0].content == "Just text"

    def test_format_file_assistant_message_unchanged(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = VirtualMemoryAgent(llm=mock_llm, files=[])
        messages = [Message(role=ASSISTANT, content="Reply")]
        out = agent._format_file(messages, lang="en")
        assert len(out) == 1
        assert out[0].role == ASSISTANT
        assert out[0].content == "Reply"
