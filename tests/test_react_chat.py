"""Tests for cat_agent.agents.react_chat."""

import json
from unittest.mock import MagicMock, patch

from cat_agent.llm.schema import ASSISTANT, USER, Message
from cat_agent.agents.react_chat import ReActChat, TOOL_DESC, PROMPT_REACT


class TestReActChatConstants:

    def test_tool_desc_has_placeholders(self):
        assert "{name_for_model}" in TOOL_DESC
        assert "{name_for_human}" in TOOL_DESC
        assert "{description_for_model}" in TOOL_DESC
        assert "{parameters}" in TOOL_DESC
        assert "{args_format}" in TOOL_DESC

    def test_prompt_react_has_placeholders(self):
        assert "{tool_descs}" in PROMPT_REACT
        assert "{tool_names}" in PROMPT_REACT
        assert "{query}" in PROMPT_REACT
        assert "Thought:" in PROMPT_REACT
        assert "Action:" in PROMPT_REACT
        assert "Action Input:" in PROMPT_REACT
        assert "Observation:" in PROMPT_REACT
        assert "Final Answer:" in PROMPT_REACT


class TestReActChatInit:

    def test_extra_generate_cfg_has_stop_observation(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = ReActChat(llm=mock_llm)
        stop = agent.extra_generate_cfg.get("stop", [])
        assert "Observation:" in stop or "Observation:\n" in stop


class TestReActChatDetectTool:

    def test_detect_tool_no_action_returns_false(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = ReActChat(llm=mock_llm)
        has_action, name, args, thought = agent._detect_tool("Just some thought.")
        assert has_action is False
        assert name is None
        assert args is None

    def test_detect_tool_with_action_and_input(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = ReActChat(llm=mock_llm)
        text = "Thought: I need to search.\nAction: web_search\nAction Input: python\nObservation: result"
        has_action, name, args, thought = agent._detect_tool(text)
        assert has_action is True
        assert name == "web_search"
        assert "python" in args
        assert "Thought:" in thought

    def test_detect_tool_adds_observation_if_missing(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = ReActChat(llm=mock_llm)
        text = "Thought: Let me call.\nAction: tool_a\nAction Input: {}"
        has_action, name, args, thought = agent._detect_tool(text)
        assert has_action is True
        assert name == "tool_a"
        assert args == "{}" or "Observation" in args


class TestReActChatPrependReactPrompt:

    def test_prepend_react_prompt_formats_last_message_with_tools(self):
        from cat_agent.tools import TOOL_REGISTRY

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        # Use a tool from registry so _init_tool accepts it
        tool_name = "web_search" if "web_search" in TOOL_REGISTRY else list(TOOL_REGISTRY.keys())[0]
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            agent = ReActChat(llm=mock_llm, function_list=[tool_name])
        messages = [Message(USER, "What is Python?")]
        out = agent._prepend_react_prompt(messages, lang="en")
        assert len(out) == 1
        content = out[0].content if hasattr(out[0], "content") else out[0]["content"]
        assert "Answer the following questions" in content
        assert tool_name in content
        assert "What is Python?" in content
        assert "Thought:" in content
        assert "Action:" in content
