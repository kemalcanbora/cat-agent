"""Tests for on-prem / air-gap security controls."""

from unittest.mock import MagicMock, patch

import pytest

from cat_agent.security.offline import (
    OfflineViolationError,
    guard_outbound_request,
    is_offline_mode,
)
from cat_agent.security.readiness import run_offline_readiness_check
from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, TOOL_REGISTRY, enable_optional_tools


class TestOfflineMode:

    def test_is_offline_mode_truthy_values(self, monkeypatch):
        for value in ('1', 'true', 'yes', 'on', 'TRUE'):
            monkeypatch.setenv('CAT_AGENT_OFFLINE', value)
            assert is_offline_mode() is True

    def test_is_offline_mode_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv('CAT_AGENT_OFFLINE', raising=False)
        assert is_offline_mode() is False

    def test_guard_outbound_request_raises_when_offline(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_OFFLINE', '1')
        with pytest.raises(OfflineViolationError, match='Blocked outbound request'):
            guard_outbound_request(purpose='test request')

    def test_guard_outbound_request_allows_when_online(self, monkeypatch):
        monkeypatch.delenv('CAT_AGENT_OFFLINE', raising=False)
        guard_outbound_request(purpose='test request')


@pytest.fixture(autouse=True)
def _restore_tool_registries():
    initial_registry = dict(TOOL_REGISTRY)
    initial_optional = dict(OPTIONAL_TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(initial_registry)
    OPTIONAL_TOOL_REGISTRY.clear()
    OPTIONAL_TOOL_REGISTRY.update(initial_optional)


class TestToolPolicy:

    def test_network_tools_not_in_default_registry(self):
        assert 'web_search' in OPTIONAL_TOOL_REGISTRY
        assert 'image_search' in OPTIONAL_TOOL_REGISTRY
        assert 'web_search' not in TOOL_REGISTRY
        assert 'image_search' not in TOOL_REGISTRY

    def test_enable_optional_tools_moves_to_registry(self):
        enable_optional_tools('web_search')
        assert 'web_search' in TOOL_REGISTRY
        assert 'web_search' not in OPTIONAL_TOOL_REGISTRY

    def test_agent_skips_network_tool_in_offline_mode(self, monkeypatch):
        from cat_agent.agents.react_chat import ReActChat

        enable_optional_tools('web_search')
        monkeypatch.setenv('CAT_AGENT_OFFLINE', '1')
        mock_llm = MagicMock()
        mock_llm.model = 'gpt-4'
        mock_llm.model_type = 'openai'
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            agent = ReActChat(llm=mock_llm, function_list=['web_search', 'storage'])
        assert 'web_search' not in agent.function_map
        assert 'storage' in agent.function_map

    def test_agent_rejects_opt_in_tool_without_enable(self):
        from cat_agent.agents.react_chat import ReActChat

        mock_llm = MagicMock()
        mock_llm.model = 'gpt-4'
        mock_llm.model_type = 'openai'
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            with pytest.raises(ValueError, match='opt-in'):
                ReActChat(llm=mock_llm, function_list=['web_search'])

    def test_agent_rejects_mcp_in_offline_mode(self, monkeypatch):
        from cat_agent.agents.react_chat import ReActChat

        monkeypatch.setenv('CAT_AGENT_OFFLINE', '1')
        mock_llm = MagicMock()
        mock_llm.model = 'gpt-4'
        mock_llm.model_type = 'openai'
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            with pytest.raises(ValueError, match='MCP servers'):
                ReActChat(
                    llm=mock_llm,
                    function_list=[{'mcpServers': {'demo': {'url': 'http://localhost'}}}],
                )

    def test_offline_readiness_lists_disabled_tools(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_OFFLINE', '1')
        report = run_offline_readiness_check()
        assert report.offline_mode is True
        assert 'web_search' in report.disabled_tools
        assert 'image_search' in report.disabled_tools
        assert report.format_report()


class TestWebSearchBackends:

    def test_searxng_requires_url(self, monkeypatch):
        from cat_agent.tools.web_search import WebSearch

        monkeypatch.delenv('CAT_AGENT_SEARXNG_URL', raising=False)
        monkeypatch.delenv('SEARXNG_URL', raising=False)
        with pytest.raises(ValueError, match='CAT_AGENT_SEARXNG_URL'):
            WebSearch._search_searxng('query')

    def test_searxng_search_parses_results(self, monkeypatch):
        from cat_agent.tools.web_search import WebSearch

        monkeypatch.setenv('CAT_AGENT_SEARXNG_URL', 'http://searxng.internal:8080')

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    'results': [
                        {'title': 'T1', 'content': 'Snippet 1', 'publishedDate': '2024'},
                    ]
                }

        with patch('cat_agent.tools.web_search.requests.get', return_value=FakeResponse()) as mock_get:
            results = WebSearch._search_searxng('cats')
        mock_get.assert_called_once()
        assert results[0]['title'] == 'T1'
        assert results[0]['snippet'] == 'Snippet 1'

    def test_serper_blocked_in_offline_mode(self, monkeypatch):
        from cat_agent.tools.web_search import WebSearch

        monkeypatch.setenv('CAT_AGENT_OFFLINE', '1')
        monkeypatch.setenv('SERPER_API_KEY', 'test-key')
        with pytest.raises(OfflineViolationError):
            WebSearch._search_serper('query')
