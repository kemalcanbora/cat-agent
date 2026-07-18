"""Tests for the long-term memory store and session memory manager."""

import base64
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.memory.manager import MemoryManager
from cat_agent.memory.store import MemoryStore
from cat_agent.security.crypto import is_encrypted_value


@pytest.fixture
def encryption_key_env(monkeypatch):
    key = base64.urlsafe_b64encode(b'0' * 32).decode('ascii')
    monkeypatch.setenv('CAT_AGENT_ENCRYPTION_KEY', key)
    monkeypatch.setenv('CAT_AGENT_ENCRYPT_AT_REST', '1')
    monkeypatch.delenv('CAT_AGENT_REQUIRE_ENCRYPTED_STORAGE', raising=False)
    return key


@pytest.fixture
def plaintext_env(monkeypatch):
    monkeypatch.setenv('CAT_AGENT_ENCRYPT_AT_REST', '0')
    monkeypatch.delenv('CAT_AGENT_ENCRYPTION_KEY', raising=False)


class TestMemoryStore:

    def test_add_get_roundtrip(self, tmp_path, plaintext_env):
        store = MemoryStore(path=str(tmp_path))
        record = store.add('User prefers Turkish replies', kind='fact')
        loaded = store.get(record.memory_id)
        assert loaded.text == 'User prefers Turkish replies'
        assert loaded.kind == 'fact'
        assert loaded.scope == 'default'

    def test_records_encrypted_at_rest(self, tmp_path, encryption_key_env):
        store = MemoryStore(path=str(tmp_path))
        store.add('IBAN TR12 0001 secret detail')
        with sqlite3.connect(store.db_path) as conn:
            payload = conn.execute('SELECT payload FROM memories').fetchone()[0]
        assert is_encrypted_value(payload)
        assert 'secret' not in payload

    def test_scope_isolation(self, tmp_path, plaintext_env):
        store = MemoryStore(path=str(tmp_path))
        store.add('alice fact', scope='user:alice')
        store.add('bob fact', scope='user:bob')
        assert [r.text for r in store.list('user:alice')] == ['alice fact']
        assert [r.text for r in store.list('user:bob')] == ['bob fact']

    def test_delete_and_clear(self, tmp_path, plaintext_env):
        store = MemoryStore(path=str(tmp_path))
        record = store.add('to be deleted')
        assert store.delete(record.memory_id) is True
        assert store.get(record.memory_id) is None
        store.add('one')
        store.add('two')
        assert store.clear() == 2
        assert store.list() == []

    def test_search_returns_relevant_first(self, tmp_path, plaintext_env):
        store = MemoryStore(path=str(tmp_path))
        store.add('The database migration finished on Friday')
        store.add('User favorite color is blue')
        store.add('Deployment target is an air-gapped kubernetes cluster')
        results = store.search('what is the favorite color of the user?', top_k=2)
        assert results
        assert 'favorite color' in results[0].text

    def test_search_empty_store(self, tmp_path, plaintext_env):
        store = MemoryStore(path=str(tmp_path))
        assert store.search('anything') == []

    def test_invalid_kind_rejected(self, tmp_path, plaintext_env):
        store = MemoryStore(path=str(tmp_path))
        with pytest.raises(ValueError, match='kind'):
            store.add('x', kind='banana')


class TestMemoryManager:

    def _manager(self, tmp_path, **cfg):
        cfg.setdefault('path', str(tmp_path))
        return MemoryManager(cfg=cfg)

    def test_remember_and_recall(self, tmp_path, plaintext_env):
        manager = self._manager(tmp_path, scope='user:kemal')
        manager.remember('User works at a bank with strict compliance rules')
        results = manager.recall('where does the user work?')
        assert results
        assert 'bank' in results[0].text

    def test_inject_memories_appends_to_system(self, tmp_path, plaintext_env):
        manager = self._manager(tmp_path)
        manager.remember('User prefers answers in Turkish')
        messages = [
            Message(role=SYSTEM, content='You are helpful.'),
            Message(role=USER, content='Which language does the user prefer for answers?'),
        ]
        injected = manager.inject_memories(messages)
        assert 'Long-term memory' in injected[0].content
        assert 'Turkish' in injected[0].content
        # Original list untouched
        assert 'Long-term memory' not in messages[0].content

    def test_inject_memories_without_matches_is_noop(self, tmp_path, plaintext_env):
        manager = self._manager(tmp_path)
        messages = [Message(role=USER, content='hello')]
        assert manager.inject_memories(messages) == messages

    def test_record_exchange_stores_episode(self, tmp_path, plaintext_env):
        manager = self._manager(tmp_path)
        messages = [Message(role=USER, content='What is the capital of France?')]
        response = [Message(role=ASSISTANT, content='The capital of France is Paris.')]
        record = manager.record_exchange(messages, response)
        assert record is not None
        assert record.kind == 'episode'
        assert 'Paris' in record.text

    def test_record_exchange_disabled(self, tmp_path, plaintext_env):
        manager = self._manager(tmp_path, auto_record=False)
        messages = [Message(role=USER, content='q')]
        response = [Message(role=ASSISTANT, content='a')]
        assert manager.record_exchange(messages, response) is None

    def test_compact_messages_under_budget_is_noop(self, tmp_path, plaintext_env):
        manager = self._manager(tmp_path, session_window_tokens=100000)
        manager.llm = MagicMock()
        messages = [Message(role=USER, content='short question')]
        assert manager.compact_messages(messages) == messages
        manager.llm.quick_chat.assert_not_called()

    def test_compact_messages_summarizes_old_turns(self, tmp_path, plaintext_env):
        manager = self._manager(tmp_path, session_window_tokens=50, keep_recent_turns=1)
        mock_llm = MagicMock()
        mock_llm.quick_chat.return_value = 'User discussed project alpha budget.'
        manager.llm = mock_llm

        long_text = 'word ' * 200
        messages = [
            Message(role=SYSTEM, content='You are helpful.'),
            Message(role=USER, content=long_text),
            Message(role=ASSISTANT, content=long_text),
            Message(role=USER, content='latest question'),
        ]
        compacted = manager.compact_messages(messages)
        assert len(compacted) == 3
        assert compacted[0].role == SYSTEM
        assert 'Summary of earlier conversation' in compacted[1].content
        assert compacted[2].content == 'latest question'
        # Summary persisted for future sessions
        summaries = [r for r in manager.store.list() if r.kind == 'summary']
        assert len(summaries) == 1

    def test_compact_messages_survives_llm_failure(self, tmp_path, plaintext_env):
        manager = self._manager(tmp_path, session_window_tokens=50, keep_recent_turns=1)
        mock_llm = MagicMock()
        mock_llm.quick_chat.side_effect = RuntimeError('model down')
        manager.llm = mock_llm

        long_text = 'word ' * 200
        messages = [
            Message(role=USER, content=long_text),
            Message(role=USER, content='latest'),
        ]
        assert manager.compact_messages(messages) == messages

    def test_cross_session_recall(self, tmp_path, plaintext_env):
        first = self._manager(tmp_path, scope='user:kemal')
        first.remember('Kemal deploys cat-agent on an air-gapped server')

        second = self._manager(tmp_path, scope='user:kemal')
        results = second.recall('where is cat-agent deployed?')
        assert results
        assert 'air-gapped' in results[0].text


class TestAssistantMemoryWiring:

    def test_assistant_accepts_memory_cfg(self, tmp_path, plaintext_env):
        from cat_agent.agents.assistant import Assistant

        mock_llm = MagicMock()
        mock_llm.model = 'gpt-4'
        mock_llm.model_type = 'openai'
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            agent = Assistant(llm=mock_llm, memory_cfg={'path': str(tmp_path)})
        assert agent.memory_manager is not None
        assert agent.memory_manager.llm is mock_llm

    def test_assistant_without_memory_cfg(self, plaintext_env):
        from cat_agent.agents.assistant import Assistant

        mock_llm = MagicMock()
        mock_llm.model = 'gpt-4'
        mock_llm.model_type = 'openai'
        with patch('cat_agent.agents.fncall_agent.Memory', return_value=MagicMock()):
            agent = Assistant(llm=mock_llm)
        assert agent.memory_manager is None
