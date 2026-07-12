"""Tests for cat_agent.agents.fncall_agent."""

from unittest.mock import MagicMock, patch

from cat_agent.agents.fncall_agent import FnCallAgent
from cat_agent.tools import BaseTool


class _DummyTool(BaseTool):
    name = 'dummy_tool'
    description = 'A dummy tool'
    parameters = [{'name': 'query', 'type': 'string', 'required': False}]

    def call(self, params: str, **kwargs):
        return 'ok'


class TestFnCallAgentFunctionSchemas:

    def test_function_schemas_cached_until_tool_changes(self):
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            agent = FnCallAgent(llm={'model': 'qwen2.5-7b'}, function_list=[_DummyTool()])

        first = agent.function_schemas
        second = agent.function_schemas
        assert first is second
        assert first[0]['name'] == 'dummy_tool'

        agent._init_tool(_DummyTool())
        third = agent.function_schemas
        assert third is not first
