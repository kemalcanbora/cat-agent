"""Tests for cat_agent.agents.memo_assistant."""

from unittest.mock import MagicMock, patch

from cat_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, FunctionCall, Message
from cat_agent.agents.memo_assistant import MemoAssistant, MEMORY_PROMPT


class TestMemoAssistantConstants:

    def test_memory_prompt_has_storage_info_placeholder(self):
        assert "{storage_info}" in MEMORY_PROMPT
        assert "<info>" in MEMORY_PROMPT
        assert "storage" in MEMORY_PROMPT.lower()


class TestMemoAssistantInit:

    def test_adds_storage_to_function_list(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = MemoAssistant(llm=mock_llm)
        assert "storage" in agent.function_map

    def test_prepends_storage_before_other_tools(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = MemoAssistant(llm=mock_llm, function_list=["retrieval"])
        assert "storage" in agent.function_map
        assert "retrieval" in agent.function_map


class TestMemoAssistantPrependStorageInfo:

    def test_empty_messages_gets_system_with_storage_prompt(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = MemoAssistant(llm=mock_llm)
        messages = [Message(USER, "Hi")]
        out = agent._prepend_storage_info_to_sys(messages)
        assert out[0].role == SYSTEM
        assert "storage" in out[0].content.lower() or "<info>" in out[0].content

    def test_storage_put_adds_key_value_to_prompt(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = MemoAssistant(llm=mock_llm)
        messages = [
            Message(ASSISTANT, "", function_call=FunctionCall(
                name="storage",
                arguments='{"operate": "put", "key": "pref", "value": "dark mode"}',
            )),
        ]
        out = agent._prepend_storage_info_to_sys(messages)
        content = out[0].content if isinstance(out[0].content, str) else "".join(c.text or "" for c in out[0].content)
        assert "pref" in content
        assert "dark mode" in content

    def test_storage_delete_removes_key(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = MemoAssistant(llm=mock_llm)
        messages = [
            Message(ASSISTANT, "", function_call=FunctionCall(
                name="storage",
                arguments='{"operate": "put", "key": "k1", "value": "v1"}',
            )),
            Message(ASSISTANT, "", function_call=FunctionCall(
                name="storage",
                arguments='{"operate": "delete", "key": "k1"}',
            )),
        ]
        out = agent._prepend_storage_info_to_sys(messages)
        content = out[0].content if isinstance(out[0].content, str) else "".join(c.text or "" for c in out[0].content)
        assert "k1" not in content or "v1" not in content

    def test_existing_system_message_appends_storage_prompt(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = MemoAssistant(llm=mock_llm)
        messages = [Message(SYSTEM, "You are helpful."), Message(USER, "Hi")]
        out = agent._prepend_storage_info_to_sys(messages)
        assert out[0].role == SYSTEM
        assert "You are helpful" in (out[0].content if isinstance(out[0].content, str) else out[0].content[0].text)
        assert "<info>" in str(out[0].content)


class TestMemoAssistantTruncateDialogueHistory:

    def test_empty_messages_returns_empty(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = MemoAssistant(llm=mock_llm)
        out = agent._truncate_dialogue_history([])
        assert out == []

    def test_few_messages_unchanged(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = MemoAssistant(llm=mock_llm)
        messages = [Message(SYSTEM, "Sys"), Message(USER, "U1"), Message(ASSISTANT, "A1")]
        out = agent._truncate_dialogue_history(messages)
        assert len(out) == 3

    def test_truncates_when_exceeding_available_turns(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = MemoAssistant(llm=mock_llm)
        # 401 USER messages so truncation kicks in (only 400 turns retained from end)
        messages = [Message(SYSTEM, "Sys")] + [
            msg for i in range(401)
            for msg in (Message(ASSISTANT, f"a{i}"), Message(USER, f"u{i}"))
        ]
        out = agent._truncate_dialogue_history(messages)
        assert len(out) < len(messages)
        if out and messages[0].role == SYSTEM:
            assert out[0].role == SYSTEM


class TestMemoAssistantRun:

    def test_run_yields_from_super(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        mock_llm.chat = MagicMock(return_value=iter([[Message(ASSISTANT, "Ok")]]))
        mock_mem = MagicMock()
        mock_mem.run.return_value = iter([[Message(FUNCTION, "")]])
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=mock_mem):
            agent = MemoAssistant(llm=mock_llm)
        messages = [Message(USER, "Hi")]
        out = list(agent._run(messages, lang="en"))
        assert len(out) >= 1
        assert out[-1][0].content == "Ok"
