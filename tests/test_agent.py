"""Tests for cat_agent.agent (Agent, BasicAgent)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.agent import Agent, BasicAgent
from cat_agent.llm.schema import ASSISTANT, CONTENT, ROLE, SYSTEM, USER, ContentItem, FunctionCall, Message
from cat_agent.tools import TOOL_REGISTRY, BaseTool
from cat_agent.tools.base import ToolServiceError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_llm():
    llm = MagicMock()
    llm.chat = MagicMock(return_value=iter([]))
    return llm


def _msg(role: str, content, **kwargs):
    return Message(role=role, content=content, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    return _make_mock_llm()


# ---------------------------------------------------------------------------
# Tests: Agent.__init__
# ---------------------------------------------------------------------------

class TestAgentInit:

    @patch("cat_agent.agent.get_chat_model")
    def test_llm_dict_calls_get_chat_model(self, get_chat_model, mock_llm):
        get_chat_model.return_value = mock_llm
        with patch.object(Agent, "_run", return_value=iter([])):
            # Use BasicAgent as concrete subclass
            agent = BasicAgent(llm={"model": "test"})
        get_chat_model.assert_called_once_with({"model": "test"})
        assert agent.llm is mock_llm

    def test_llm_object_stored_directly(self, mock_llm):
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm)
        assert agent.llm is mock_llm

    def test_system_message_name_description_stored(self, mock_llm):
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(
                llm=mock_llm,
                system_message="You are helpful.",
                name="my_agent",
                description="Does things.",
            )
        assert agent.system_message == "You are helpful."
        assert agent.name == "my_agent"
        assert agent.description == "Does things."

    def test_empty_function_list(self, mock_llm):
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm, function_list=[])
        assert agent.function_map == {}

    def test_init_tool_by_string_name(self, mock_llm):
        # Use a tool that is in TOOL_REGISTRY (e.g. storage)
        if "storage" not in TOOL_REGISTRY:
            pytest.skip("storage tool not registered")
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm, function_list=["storage"])
        assert "storage" in agent.function_map
        assert isinstance(agent.function_map["storage"], TOOL_REGISTRY["storage"])

    def test_init_tool_by_base_tool_instance(self, mock_llm):
        tool = MagicMock(spec=BaseTool)
        tool.name = "my_tool"
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm, function_list=[tool])
        assert agent.function_map["my_tool"] is tool

    def test_init_tool_unknown_raises(self, mock_llm):
        with patch.object(Agent, "_run", return_value=iter([])):
            with pytest.raises(ValueError, match="is not registered"):
                BasicAgent(llm=mock_llm, function_list=["nonexistent_tool_xyz"])


# ---------------------------------------------------------------------------
# Tests: Agent.run
# ---------------------------------------------------------------------------

class TestAgentRun:

    def test_run_converts_dict_messages_to_message_objects(self, mock_llm):
        mock_llm.chat.return_value = iter([[Message(role=ASSISTANT, content="Hi")]])
        agent = BasicAgent(llm=mock_llm)
        messages = [{"role": "user", "content": "Hello"}]
        responses = list(agent.run(messages))
        assert len(responses) == 1
        assert responses[0][0]["role"] == ASSISTANT
        assert responses[0][0]["content"] == "Hi"

    def test_run_returns_dict_when_input_all_dict(self, mock_llm):
        mock_llm.chat.return_value = iter([[Message(role=ASSISTANT, content="Hi")]])
        agent = BasicAgent(llm=mock_llm)
        messages = [{"role": "user", "content": "Hello"}]
        responses = list(agent.run(messages))
        assert isinstance(responses[0][0], dict)

    def test_run_returns_message_objects_when_input_has_message(self, mock_llm):
        mock_llm.chat.return_value = iter([[Message(role=ASSISTANT, content="Hi")]])
        agent = BasicAgent(llm=mock_llm)
        messages = [_msg(USER, "Hello")]
        responses = list(agent.run(messages))
        assert isinstance(responses[0][0], Message)
        assert responses[0][0].role == ASSISTANT
        assert responses[0][0].content == "Hi"

    def test_run_prepends_system_message_when_absent(self, mock_llm):
        def chat(messages, **kwargs):
            assert messages[0][ROLE] == SYSTEM
            assert messages[0][CONTENT] == "You are helpful."
            return iter([[Message(role=ASSISTANT, content="OK")]])

        mock_llm.chat.side_effect = chat
        agent = BasicAgent(llm=mock_llm, system_message="You are helpful.")
        messages = [_msg(USER, "Hi")]
        list(agent.run(messages))
        mock_llm.chat.assert_called_once()

    def test_run_sets_lang_en_when_no_chinese(self, mock_llm):
        def chat(messages, **kwargs):
            assert kwargs.get("extra_generate_cfg", {}).get("lang") == "en"
            return iter([[Message(role=ASSISTANT, content="OK")]])

        mock_llm.chat.side_effect = chat
        agent = BasicAgent(llm=mock_llm)
        list(agent.run([_msg(USER, "Hello")]))
        mock_llm.chat.assert_called_once()

    def test_run_sets_lang_zh_when_chinese(self, mock_llm):
        def chat(messages, **kwargs):
            assert kwargs.get("extra_generate_cfg", {}).get("lang") == "zh"
            return iter([[Message(role=ASSISTANT, content="好")]])

        mock_llm.chat.side_effect = chat
        agent = BasicAgent(llm=mock_llm)
        list(agent.run([_msg(USER, "你好")]))
        mock_llm.chat.assert_called_once()

    def test_run_assigns_agent_name_to_response_without_name(self, mock_llm):
        mock_llm.chat.return_value = iter([[Message(role=ASSISTANT, content="Hi", name=None)]])
        agent = BasicAgent(llm=mock_llm, name="my_bot")
        responses = list(agent.run([_msg(USER, "Hello")]))
        assert responses[0][0].name == "my_bot"

    def test_run_does_not_overwrite_existing_response_name(self, mock_llm):
        mock_llm.chat.return_value = iter([[Message(role=ASSISTANT, content="Hi", name="other")]])
        agent = BasicAgent(llm=mock_llm, name="my_bot")
        responses = list(agent.run([_msg(USER, "Hello")]))
        assert responses[0][0].name == "other"

    def test_run_empty_messages_return_type_message(self, mock_llm):
        mock_llm.chat.return_value = iter([[Message(role=ASSISTANT, content="")]])
        agent = BasicAgent(llm=mock_llm)
        responses = list(agent.run([]))
        assert isinstance(responses[0][0], Message)


# ---------------------------------------------------------------------------
# Tests: Agent.run_nonstream
# ---------------------------------------------------------------------------

class TestAgentRunNonstream:

    def test_run_nonstream_returns_last_response_only(self, mock_llm):
        r1 = [Message(role=ASSISTANT, content="First")]
        r2 = [Message(role=ASSISTANT, content="Second")]
        mock_llm.chat.return_value = iter([r1, r2])
        agent = BasicAgent(llm=mock_llm)
        result = agent.run_nonstream([_msg(USER, "Hi")])
        assert result == r2
        assert len(result) == 1
        assert result[0].content == "Second"


# ---------------------------------------------------------------------------
# Tests: Agent._call_tool
# ---------------------------------------------------------------------------

class TestAgentCallTool:

    def test_tool_not_in_map_returns_error_string(self, mock_llm):
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm)
        out = agent._call_tool("nonexistent")
        assert out == "Tool nonexistent does not exists."

    def test_tool_returns_str(self, mock_llm):
        tool = MagicMock()
        tool.call.return_value = "result text"
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm)
            agent.function_map["t1"] = tool
        out = agent._call_tool("t1", "{}")
        assert out == "result text"

    def test_tool_returns_list_of_content_items(self, mock_llm):
        items = [ContentItem(text="a"), ContentItem(text="b")]
        tool = MagicMock()
        tool.call.return_value = items
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm)
            agent.function_map["t1"] = tool
        out = agent._call_tool("t1", "{}")
        assert out == items

    def test_tool_returns_dict_serialized_to_json(self, mock_llm):
        tool = MagicMock()
        tool.call.return_value = {"key": "value"}
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm)
            agent.function_map["t1"] = tool
        out = agent._call_tool("t1", "{}")
        assert json.loads(out) == {"key": "value"}

    def test_tool_raises_tool_service_error_reraised(self, mock_llm):
        tool = MagicMock()
        tool.call.side_effect = ToolServiceError(message="bad")
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm)
            agent.function_map["t1"] = tool
        with pytest.raises(ToolServiceError):
            agent._call_tool("t1", "{}")

    def test_tool_raises_generic_exception_returns_error_message(self, mock_llm):
        tool = MagicMock()
        tool.call.side_effect = ValueError("something broke")
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm)
            agent.function_map["t1"] = tool
        out = agent._call_tool("t1", "{}")
        assert "ValueError" in out
        assert "something broke" in out


# ---------------------------------------------------------------------------
# Tests: Agent._detect_tool
# ---------------------------------------------------------------------------

class TestAgentDetectTool:

    def test_no_function_call(self, mock_llm):
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm)
        msg = Message(role=ASSISTANT, content="Just text")
        need, name, args, text = agent._detect_tool(msg)
        assert need is False
        assert name is None
        assert args is None
        assert text == "Just text"

    def test_with_function_call(self, mock_llm):
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm)
        fc = FunctionCall(name="get_weather", arguments='{"location": "NYC"}')
        msg = Message(role=ASSISTANT, content="", function_call=fc)
        need, name, args, text = agent._detect_tool(msg)
        assert need is True
        assert name == "get_weather"
        assert args == '{"location": "NYC"}'
        assert text == ""

    def test_empty_content_becomes_empty_string(self, mock_llm):
        with patch.object(Agent, "_run", return_value=iter([])):
            agent = BasicAgent(llm=mock_llm)
        msg = Message(role=ASSISTANT, content=None)
        _, _, _, text = agent._detect_tool(msg)
        assert text == ""


# ---------------------------------------------------------------------------
# Tests: BasicAgent._run
# ---------------------------------------------------------------------------

class TestBasicAgentRun:

    def test_calls_call_llm_with_lang(self, mock_llm):
        mock_llm.chat.return_value = iter([[Message(role=ASSISTANT, content="Hi")]])
        agent = BasicAgent(llm=mock_llm)
        messages = [_msg(USER, "Hello")]
        list(agent.run(messages, lang="zh"))
        call_kwargs = mock_llm.chat.call_args[1]
        assert call_kwargs.get("extra_generate_cfg", {}).get("lang") == "zh"

    def test_passes_seed_when_provided(self, mock_llm):
        mock_llm.chat.return_value = iter([[Message(role=ASSISTANT, content="Hi")]])
        agent = BasicAgent(llm=mock_llm)
        list(agent.run([_msg(USER, "Hello")], seed=42))
        call_kwargs = mock_llm.chat.call_args[1]
        assert call_kwargs.get("extra_generate_cfg", {}).get("seed") == 42
