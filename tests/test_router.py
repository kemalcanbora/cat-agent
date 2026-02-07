"""Tests for cat_agent.agents.router."""

from unittest.mock import MagicMock, patch

from cat_agent.agent import BasicAgent
from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, ContentItem, Message
from cat_agent.agents.router import Router, ROUTER_PROMPT


class TestRouter:

    def test_supplement_name_special_token_str_content(self):
        msg = Message(ASSISTANT, "Hello", name="Bot1")
        out = Router.supplement_name_special_token(msg)
        assert "Call: Bot1" in out.content
        assert "Reply:" in out.content
        assert "Hello" in out.content

    def test_supplement_name_special_token_list_content(self):
        msg = Message(ASSISTANT, [ContentItem(text="Hello")], name="Bot1")
        out = Router.supplement_name_special_token(msg)
        assert out.content is not None
        if isinstance(out.content, list):
            text = out.content[0].text if hasattr(out.content[0], "text") else out.content[0].get("text", "")
            assert "Call: Bot1" in text
            assert "Reply:" in text
        else:
            assert "Call: Bot1" in str(out.content)
            assert "Reply:" in str(out.content)

    def test_supplement_name_special_token_no_name_unchanged(self):
        msg = Message(ASSISTANT, "Hi", name=None)
        out = Router.supplement_name_special_token(msg)
        assert out.content == "Hi"

    def test_router_prompt_contains_placeholders(self):
        assert "{agent_descs}" in ROUTER_PROMPT
        assert "{agent_names}" in ROUTER_PROMPT

    def test_router_init_requires_agents(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        sub_agent = BasicAgent(llm=mock_llm)
        sub_agent.name = "Sub"
        sub_agent.description = "A sub agent"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            router = Router(llm=mock_llm, agents=[sub_agent])
        assert router._agents == [sub_agent]
        assert router.system_message
        assert "Sub" in router.system_message

    def test_run_when_call_in_response_delegates_to_selected_agent(self):
        from cat_agent.agents.assistant import Assistant

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        sub_a = BasicAgent(llm=mock_llm)
        sub_a.name = "AgentA"
        sub_a.description = "A"
        sub_a.run = MagicMock(return_value=iter([[Message(ASSISTANT, "From A", name="AgentA")]]))
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            router = Router(llm=mock_llm, agents=[sub_a])
        # Make Assistant._run (super) yield one response with "Call: AgentA"
        def fake_super_run(self, messages, lang=None, **kwargs):
            yield [Message(ASSISTANT, "Call: AgentA\nReply: (to be filled)", name=None)]
        with patch.object(Assistant, "_run", fake_super_run):
            out = list(router._run([Message(SYSTEM, "Sys"), Message(USER, "Hi")], lang="en"))
        # Router should have called selected_agent.run and yielded that response
        sub_a.run.assert_called_once()
        assert len(out) >= 1
        assert out[-1][0].content == "From A" or "From A" in str(out)

    def test_run_uses_first_agent_when_selected_name_invalid(self):
        from cat_agent.agents.assistant import Assistant

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        sub_a = BasicAgent(llm=mock_llm)
        sub_a.name = "OnlyAgent"
        sub_a.description = "Only"
        sub_a.run = MagicMock(return_value=iter([[Message(ASSISTANT, "Ok", name="OnlyAgent")]]))
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            router = Router(llm=mock_llm, agents=[sub_a])
        def fake_super_run(self, messages, lang=None, **kwargs):
            yield [Message(ASSISTANT, "Call: NonExistent\nReply: ...", name=None)]
        with patch.object(Assistant, "_run", fake_super_run):
            list(router._run([Message(SYSTEM, "Sys"), Message(USER, "Hi")], lang="en"))
        # Fallback to agent_names[0] = "OnlyAgent"
        sub_a.run.assert_called_once()
