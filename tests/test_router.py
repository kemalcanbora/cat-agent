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
        assert "After an assistant replies" in ROUTER_PROMPT

    def test_router_init_requires_agents(self):
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        sub_agent = BasicAgent(llm=mock_llm)
        sub_agent.name = "Sub"
        sub_agent.description = "A sub agent"
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            router = Router(llm=mock_llm, agents=[sub_agent], inject_hub_tools=False)
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
            router = Router(llm=mock_llm, agents=[sub_a], inject_hub_tools=False)

        calls = iter([
            [Message(ASSISTANT, "Call: AgentA\nReply: (to be filled)", name=None)],
            [Message(ASSISTANT, "Thanks, here is the answer.", name=None)],
        ])

        def fake_super_run(self, messages, lang=None, **kwargs):
            yield next(calls)

        with patch.object(Assistant, "_run", fake_super_run):
            out = list(router._run([Message(SYSTEM, "Sys"), Message(USER, "Hi")], lang="en"))
        sub_a.run.assert_called_once()
        assert len(out) >= 1
        assert out[-1][-1].content == "Thanks, here is the answer."

    def test_run_unknown_agent_feeds_error_back(self):
        from cat_agent.agents.assistant import Assistant

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        sub_a = BasicAgent(llm=mock_llm)
        sub_a.name = "OnlyAgent"
        sub_a.description = "Only"
        sub_a.run = MagicMock(return_value=iter([[Message(ASSISTANT, "Ok", name="OnlyAgent")]]))
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            router = Router(llm=mock_llm, agents=[sub_a], inject_hub_tools=False)

        calls = iter([
            [Message(ASSISTANT, "Call: NonExistent\nReply: ...", name=None)],
            [Message(ASSISTANT, "Answering directly.", name=None)],
        ])

        def fake_super_run(self, messages, lang=None, **kwargs):
            yield next(calls)

        with patch.object(Assistant, "_run", fake_super_run):
            out = list(router._run([Message(SYSTEM, "Sys"), Message(USER, "Hi")], lang="en"))
        # No longer falls back to first agent — error is fed back, then direct answer
        sub_a.run.assert_not_called()
        assert out[-1][-1].content == "Answering directly."

    def test_call_with_leaked_text_is_truncated(self):
        from cat_agent.agents.assistant import Assistant
        from cat_agent.agents.router import _truncate_to_call

        leaked = "Call: AgentA\nNow I will add 1+2=3 myself..."
        assert _truncate_to_call(leaked) == "Call: AgentA"

        mock_llm = MagicMock()
        mock_llm.model = "gpt-4"
        mock_llm.model_type = "openai"
        sub_a = BasicAgent(llm=mock_llm)
        sub_a.name = "AgentA"
        sub_a.description = "A"
        sub_a.run = MagicMock(return_value=iter([[Message(ASSISTANT, "From A", name="AgentA")]]))
        with patch("cat_agent.agents.fncall_agent.Memory", return_value=MagicMock()):
            router = Router(llm=mock_llm, agents=[sub_a], inject_hub_tools=False)

        calls = iter([
            [Message(ASSISTANT, leaked)],
            [Message(ASSISTANT, "Done.")],
        ])

        def fake_super_run(self, messages, lang=None, **kwargs):
            yield next(calls)

        with patch.object(Assistant, "_run", fake_super_run):
            out = list(router._run([Message(USER, "Hi")], lang="en"))

        # Streamed Call turn should not include the leaked calculation
        call_batches = [b for b in out if b and "Call:" in str(b[-1].content)]
        assert call_batches
        assert "1+2" not in call_batches[0][-1].content
        sub_a.run.assert_called_once()
        # Specialist sees a clean user message, not Call: scaffolding
        sent = sub_a.run.call_args.kwargs.get("messages") or sub_a.run.call_args[0][0]
        assert all("Call:" not in str(m.content) for m in sent)
        assert sent[0].role == USER
        assert sent[0].content == "Hi"

    def test_adapt_for_specialist_keeps_prior_answers(self):
        from cat_agent.agents.router import _adapt_for_specialist

        working = [
            Message(USER, "Sum 1 and 2"),
            Message(ASSISTANT, "Call: MathGuy\nReply:", name="Router"),
            Message(USER, "The sum is 3.", name="MathGuy"),
        ]
        out = _adapt_for_specialist(working, ["MathGuy", "Explainer"])
        assert len(out) == 2
        assert out[0].role == USER and out[0].content == "Sum 1 and 2"
        assert out[1].role == ASSISTANT and out[1].name == "MathGuy"
        assert "Call:" not in out[1].content
