"""Tests for cat_agent.agents.react_chat."""

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
        tool_name = 'storage'
        assert tool_name in TOOL_REGISTRY
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


class _ReActFakeLLM:
    """First turn: Action; second turn: Final Answer."""

    def __init__(self):
        self.model = 'fake'
        self.model_type = 'fake'
        self.calls = 0

    def chat(self, messages, functions=None, stream=True, delta_stream=False, extra_generate_cfg=None):
        self.calls += 1
        if self.calls == 1:
            content = (
                'I need storage.\n'
                'Action: storage\n'
                'Action Input: {"operate": "get", "key": "k"}'
            )
        else:
            content = 'Final Answer: VALUE_FROM_TOOL'
        out = [Message(role=ASSISTANT, content=content)]
        if stream:
            return iter([out])
        return out


class TestReActChatRunE2E:

    def test_thought_action_loop_calls_tool_and_yields_final(self):
        fake = _ReActFakeLLM()
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            agent = ReActChat(llm=fake, function_list=['storage'], system_message='')

        with patch.object(agent, '_call_tool', return_value='TOOL_OBS_MARKER') as call_tool:
            out = list(agent.run([Message(USER, 'Get key k')]))

        assert out, 'ReActChat._run must yield responses'
        call_tool.assert_called()
        assert call_tool.call_args[0][0] == 'storage'
        final = out[-1][-1].content
        assert 'TOOL_OBS_MARKER' in final or 'Final Answer' in final or 'VALUE_FROM_TOOL' in final
        assert fake.calls >= 2
