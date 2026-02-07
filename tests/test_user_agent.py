"""Tests for cat_agent.agents.user_agent."""

from cat_agent.llm.schema import USER, Message
from cat_agent.agents.user_agent import PENDING_USER_INPUT, UserAgent


class TestUserAgent:

    def test_run_yields_pending_input(self):
        agent = UserAgent(name="human")
        out = list(agent._run([Message(USER, "anything")]))
        assert len(out) == 1
        assert out[0][0].content == PENDING_USER_INPUT
        assert out[0][0].role == "user"

    def test_constant_value(self):
        assert "PENDING_USER_INPUT" in PENDING_USER_INPUT or "INTERRUPT" in PENDING_USER_INPUT
