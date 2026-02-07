"""Tests for cat_agent.multi_agent_hub."""

from unittest.mock import MagicMock, patch

import pytest

from cat_agent.agent import Agent
from cat_agent.multi_agent_hub import MultiAgentHub


class TestMultiAgentHub:

    def test_agent_names(self):
        with patch.object(MultiAgentHub, "__init__", lambda self: None):
            hub = MultiAgentHub.__new__(MultiAgentHub)
            a1 = MagicMock(spec=Agent)
            a1.name = "A"
            a2 = MagicMock(spec=Agent)
            a2.name = "B"
            hub._agents = [a1, a2]
            assert hub.agent_names == ["A", "B"]

    def test_agents_validation_empty_raises(self):
        with patch.object(MultiAgentHub, "__init__", lambda self: None):
            hub = MultiAgentHub.__new__(MultiAgentHub)
            hub._agents = []
            with pytest.raises(AssertionError):
                _ = hub.agents

    def test_agents_validation_duplicate_names_raises(self):
        with patch.object(MultiAgentHub, "__init__", lambda self: None):
            hub = MultiAgentHub.__new__(MultiAgentHub)
            a1 = MagicMock(spec=Agent)
            a1.name = "Same"
            a2 = MagicMock(spec=Agent)
            a2.name = "Same"
            hub._agents = [a1, a2]
            with pytest.raises(AssertionError):
                _ = hub.agents
