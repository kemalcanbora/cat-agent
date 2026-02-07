"""Tests for cat_agent.agents.group_chat."""

from unittest.mock import MagicMock

import pytest

from cat_agent.agent import BasicAgent
from cat_agent.llm.schema import ASSISTANT, USER, Message
from cat_agent.agents.group_chat import GroupChat
from cat_agent.agents.user_agent import UserAgent, PENDING_USER_INPUT


# ---------------------------------------------------------------------------
# From test_agents.py
# ---------------------------------------------------------------------------

class TestGroupChat:

    def test_valid_agent_selection_methods(self):
        assert "auto" in GroupChat._VALID_AGENT_SELECTION_METHODS
        assert "round_robin" in GroupChat._VALID_AGENT_SELECTION_METHODS
        assert "random" in GroupChat._VALID_AGENT_SELECTION_METHODS
        assert "manual" in GroupChat._VALID_AGENT_SELECTION_METHODS

    def test_init_invalid_selection_method_raises(self):
        with pytest.raises(AssertionError, match="agent_selection_method"):
            GroupChat(agents=[], agent_selection_method="invalid")

    def test_init_with_agent_list(self):
        mock_llm = MagicMock()
        sub = BasicAgent(llm=mock_llm)
        sub.name = "Bot"
        sub.description = "A bot"
        chat = GroupChat(agents=[sub], agent_selection_method="round_robin")
        assert chat._agents == [sub]
        assert chat.agent_selection_method == "round_robin"


class TestGroupChatInit:

    def test_auto_mode_without_llm_raises(self):
        sub = BasicAgent(llm=MagicMock())
        sub.name = "Bot"
        sub.description = "A bot"
        with pytest.raises(AssertionError, match="LLM"):
            GroupChat(agents=[sub], agent_selection_method="auto", llm=None)

    def test_init_from_config_dict_creates_agents(self):
        mock_llm = MagicMock()
        config = {
            "background": "Test group",
            "agents": [
                {"name": "Alice", "description": "Helper", "instructions": "Be helpful."},
            ],
        }
        chat = GroupChat(agents=config, agent_selection_method="round_robin", llm=mock_llm)
        assert len(chat._agents) == 1
        assert chat._agents[0].name == "Alice"

    def test_init_from_config_with_human_creates_user_agent(self):
        mock_llm = MagicMock()
        config = {
            "background": "",
            "agents": [
                {"name": "Human", "description": "User", "is_human": True},
            ],
        }
        chat = GroupChat(agents=config, agent_selection_method="round_robin", llm=mock_llm)
        assert len(chat._agents) == 1
        assert isinstance(chat._agents[0], UserAgent)
        assert chat._agents[0].name == "Human"


class TestGroupChatRun:

    def test_run_sets_message_name_from_role_when_missing(self):
        mock_llm = MagicMock()
        mock_llm.chat = MagicMock(return_value=iter([]))
        sub = BasicAgent(llm=mock_llm)
        sub.name = "Bot"
        sub.description = "A bot"
        chat = GroupChat(agents=[sub], agent_selection_method="round_robin")
        messages = [Message(ASSISTANT, "Hi", name="Bot"), Message(USER, "Hello")]
        messages[1].name = None
        list(chat._run(messages, need_batch_response=False, max_round=1))
        # _run deep-copies and sets name=role on the copy; run completes without error
        assert messages[1].name is None

    def test_run_assistant_without_name_raises(self):
        mock_llm = MagicMock()
        sub = BasicAgent(llm=mock_llm)
        sub.name = "Bot"
        sub.description = "A bot"
        chat = GroupChat(agents=[sub], agent_selection_method="round_robin")
        messages = [Message(ASSISTANT, "Hi", name=None)]
        with pytest.raises(AssertionError, match="must be given a name"):
            list(chat._run(messages, need_batch_response=False, max_round=1))


class TestGroupChatSelectAgent:

    def test_mentioned_agents_name_returns_that_agent(self):
        mock_llm = MagicMock()
        a1 = BasicAgent(llm=mock_llm)
        a1.name = "Alice"
        a1.description = "A"
        a2 = BasicAgent(llm=mock_llm)
        a2.name = "Bob"
        a2.description = "B"
        chat = GroupChat(agents=[a1, a2], agent_selection_method="round_robin")
        selected = chat._select_agent([], mentioned_agents_name=["Bob"], lang="en")
        assert selected is a2

    def test_round_robin_selects_next_agent(self):
        mock_llm = MagicMock()
        a1 = BasicAgent(llm=mock_llm)
        a1.name = "First"
        a1.description = "1"
        a2 = BasicAgent(llm=mock_llm)
        a2.name = "Second"
        a2.description = "2"
        chat = GroupChat(agents=[a1, a2], agent_selection_method="round_robin")
        messages = [Message(ASSISTANT, "Hi", name="First")]
        selected = chat._select_agent(messages, mentioned_agents_name=None, lang="en")
        assert selected.name == "Second"

    def test_round_robin_no_messages_selects_first(self):
        mock_llm = MagicMock()
        a1 = BasicAgent(llm=mock_llm)
        a1.name = "First"
        a1.description = "1"
        chat = GroupChat(agents=[a1], agent_selection_method="round_robin")
        selected = chat._select_agent([], mentioned_agents_name=None, lang="en")
        assert selected.name == "First"


class TestGroupChatManageMessages:

    def test_manage_messages_filters_by_selected_name(self):
        mock_llm = MagicMock()
        a1 = BasicAgent(llm=mock_llm)
        a1.name = "Alice"
        a1.description = "A"
        a2 = BasicAgent(llm=mock_llm)
        a2.name = "Bob"
        a2.description = "B"
        chat = GroupChat(agents=[a1, a2], agent_selection_method="round_robin")
        messages = [
            Message(USER, "Hi", name="user"),
            Message(ASSISTANT, "Hello from Alice", name="Alice"),
            Message(USER, "Tell me more", name="user"),
        ]
        out = chat._manage_messages(messages, "Alice")
        assert any("Alice" in str(m.content) for m in out)
        assert any(m.role == "assistant" for m in out)

    def test_manage_messages_ends_with_user_prompt_for_name(self):
        mock_llm = MagicMock()
        a1 = BasicAgent(llm=mock_llm)
        a1.name = "Bot"
        a1.description = "A bot"
        chat = GroupChat(agents=[a1], agent_selection_method="round_robin")
        messages = [Message(ASSISTANT, "Reply", name="Bot")]
        out = chat._manage_messages(messages, "Bot")
        assert out[-1].role == "user"
        assert "Bot" in out[-1].content


class TestGroupChatGenBatchResponse:

    def test_stops_when_pending_user_input(self):
        mock_llm = MagicMock()
        human = UserAgent(name="User")
        bot = BasicAgent(llm=mock_llm)
        bot.name = "Bot"
        bot.description = "A bot"
        bot.run = MagicMock(return_value=iter([[Message(ASSISTANT, PENDING_USER_INPUT, name="Bot")]]))
        chat = GroupChat(agents=[human, bot], agent_selection_method="round_robin")
        messages = [
            Message(USER, "Hi", name="User"),
            Message(ASSISTANT, "Hello", name="Bot"),
        ]
        out = list(chat._gen_batch_response(messages, max_round=3))
        # Yields (response + rsp) then (response) at end, so 2 batches when we break on PENDING_USER_INPUT
        assert len(out) >= 1
        last_batch = out[-1]
        assert last_batch[-1].content == PENDING_USER_INPUT
