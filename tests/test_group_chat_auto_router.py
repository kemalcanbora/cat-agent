"""Tests for cat_agent.agents.group_chat_auto_router."""

from unittest.mock import MagicMock

from cat_agent.agent import BasicAgent
from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.agents.group_chat_auto_router import GroupChatAutoRouter


class TestGroupChatAutoRouter:

    def test_prompt_templates(self):
        assert "{agent_descs}" in GroupChatAutoRouter.PROMPT_TEMPLATE_EN
        assert "{agent_names}" in GroupChatAutoRouter.PROMPT_TEMPLATE_EN
        assert "{agent_descs}" in GroupChatAutoRouter.PROMPT_TEMPLATE_ZH
        assert "{agent_names}" in GroupChatAutoRouter.PROMPT_TEMPLATE_ZH

    def test_init_builds_system_prompt(self):
        mock_llm = MagicMock()
        sub = BasicAgent(llm=mock_llm)
        sub.name = "Alice"
        sub.description = "Helper"
        host = GroupChatAutoRouter(llm=mock_llm, agents=[sub])
        assert "Alice" in host.system_message
        assert "Helper" in host.system_message

    def test_init_uses_zh_template_when_chinese_in_descs(self):
        mock_llm = MagicMock()
        sub = BasicAgent(llm=mock_llm)
        sub.name = "助手"
        sub.description = "中文助手"
        host = GroupChatAutoRouter(llm=mock_llm, agents=[sub])
        assert "助手" in host.system_message
        assert "[STOP]" in host.system_message

    def test_run_builds_dialogue_from_messages(self):
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(return_value=iter([[Message(ASSISTANT, "Alice", name="Host")]]))
        sub = BasicAgent(llm=mock_llm)
        sub.name = "Alice"
        sub.description = "A"
        host = GroupChatAutoRouter(llm=mock_llm, agents=[sub])
        messages = [
            Message(SYSTEM, "You are the host."),
            Message(USER, "Hello", name="User"),
            Message(ASSISTANT, "Hi there", name="Alice"),
        ]
        out = list(host._run(messages, lang="en"))
        assert len(out) >= 1
        call_messages = mock_llm.chat.call_args[1]["messages"]
        assert len(call_messages) == 2  # system + user (dialogue string)
        user_content = call_messages[1].content if hasattr(call_messages[1], "content") else call_messages[1]["content"]
        assert "User:" in user_content
        assert "Hello" in user_content
        assert "Alice:" in user_content
        assert "Hi there" in user_content

    def test_run_skips_system_and_empty_content(self):
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(return_value=iter([[Message(ASSISTANT, "Bob", name="Host")]]))
        sub = BasicAgent(llm=mock_llm)
        sub.name = "Bob"
        sub.description = "B"
        host = GroupChatAutoRouter(llm=mock_llm, agents=[sub])
        messages = [
            Message(SYSTEM, "System prompt"),
            Message(USER, "Say something", name="User"),
        ]
        list(host._run(messages, lang="en"))
        user_content = mock_llm.chat.call_args[1]["messages"][1].content
        assert "User:" in user_content
        assert "Say something" in user_content
        assert "System prompt" not in user_content

    def test_run_same_speaker_appends_to_previous_line(self):
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(return_value=iter([[Message(ASSISTANT, "Alice", name="Host")]]))
        sub = BasicAgent(llm=mock_llm)
        sub.name = "Alice"
        sub.description = "A"
        host = GroupChatAutoRouter(llm=mock_llm, agents=[sub])
        messages = [
            Message(SYSTEM, "Sys"),
            Message(ASSISTANT, "First line", name="Alice"),
            Message(ASSISTANT, "Second line", name="Alice"),
        ]
        list(host._run(messages, lang="en"))
        user_content = mock_llm.chat.call_args[1]["messages"][1].content
        assert "Alice:" in user_content
        assert "First line" in user_content
        assert "Second line" in user_content
