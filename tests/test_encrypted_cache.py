"""Tests for encrypted doc-parser cache."""

import base64
import json
import os
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

from cat_agent.security.crypto import (
    ENCRYPTION_MARKER,
    EncryptionKeyError,
    decrypt_value,
    encrypt_value,
    is_encrypted_value,
    resolve_encryption_key,
)
from cat_agent.security.encrypted_cache import (
    PlaintextCacheError,
    count_plaintext_values,
    ensure_encrypted_cache_ready,
    migrate_plaintext_cache,
)
from cat_agent.tools.doc_parser import DocParser
from cat_agent.tools.storage import Storage, _sqlite_path


@pytest.fixture
def encryption_key_env(monkeypatch):
    key = base64.urlsafe_b64encode(b'0' * 32).decode('ascii')
    monkeypatch.setenv('CAT_AGENT_ENCRYPTION_KEY', key)
    monkeypatch.setenv('CAT_AGENT_ENCRYPT_CACHE', '1')
    monkeypatch.delenv('CAT_AGENT_REQUIRE_ENCRYPTED_CACHE', raising=False)
    return key


class TestCrypto:

    def test_encrypt_decrypt_roundtrip(self, encryption_key_env):
        key = resolve_encryption_key()
        token = encrypt_value('{"secret":"patient data"}', key)
        assert token.startswith(ENCRYPTION_MARKER)
        assert decrypt_value(token, key) == '{"secret":"patient data"}'

    def test_resolve_key_from_env(self, encryption_key_env):
        assert len(resolve_encryption_key()) == 32

    def test_resolve_key_missing_raises(self, monkeypatch):
        monkeypatch.delenv('CAT_AGENT_ENCRYPTION_KEY', raising=False)
        monkeypatch.setattr(
            'cat_agent.security.crypto._keyring_get_password',
            lambda: None,
        )
        with pytest.raises(EncryptionKeyError):
            resolve_encryption_key(create_if_missing=False)


class TestEncryptedStorage:

    def test_storage_encrypts_values_at_rest(self, encryption_key_env):
        root = tempfile.mkdtemp()
        storage = Storage({'storage_root_path': root, 'encrypt_at_rest': True})
        storage.put('cache-key', 'sensitive document text')
        assert storage.get('cache-key') == 'sensitive document text'

        with sqlite3.connect(_sqlite_path(root)) as conn:
            stored = conn.execute('SELECT value FROM kv WHERE key = ?', ('cache-key',)).fetchone()[0]
        assert is_encrypted_value(stored)
        assert 'sensitive document text' not in stored

    def test_storage_reads_legacy_plaintext_when_not_required(self, encryption_key_env, monkeypatch):
        root = tempfile.mkdtemp()
        db_path = _sqlite_path(root)
        os.makedirs(root, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                'CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)'
            )
            conn.execute(
                'INSERT INTO kv(key, value) VALUES (?, ?)',
                ('legacy', '{"title":"plain"}'),
            )
            conn.commit()

        storage = Storage({'storage_root_path': root, 'encrypt_at_rest': True})
        assert storage.get('legacy') == '{"title":"plain"}'

    def test_require_encrypted_cache_rejects_plaintext(self, encryption_key_env, monkeypatch):
        root = tempfile.mkdtemp()
        db_path = _sqlite_path(root)
        os.makedirs(root, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                'CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)'
            )
            conn.execute(
                'INSERT INTO kv(key, value) VALUES (?, ?)',
                ('legacy', '{"title":"plain"}'),
            )
            conn.commit()

        monkeypatch.setenv('CAT_AGENT_REQUIRE_ENCRYPTED_CACHE', '1')
        with pytest.raises(PlaintextCacheError, match='plaintext cache record'):
            Storage({'storage_root_path': root, 'encrypt_at_rest': True})


class TestCacheMigration:

    def test_migrate_plaintext_cache_encrypts_records(self, encryption_key_env):
        root = tempfile.mkdtemp()
        db_path = _sqlite_path(root)
        os.makedirs(root, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                'CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)'
            )
            conn.execute(
                'INSERT INTO kv(key, value) VALUES (?, ?)',
                ('doc1', '{"url":"file:///secret.pdf","raw":[]}'),
            )
            conn.commit()

        migrated = migrate_plaintext_cache(root)
        assert migrated == 1
        assert count_plaintext_values(db_path) == 0

        storage = Storage({'storage_root_path': root, 'encrypt_at_rest': True})
        assert json.loads(storage.get('doc1'))['url'] == 'file:///secret.pdf'

    def test_ensure_encrypted_cache_ready_after_migration(self, encryption_key_env, monkeypatch):
        root = tempfile.mkdtemp()
        migrate_plaintext_cache(root)
        monkeypatch.setenv('CAT_AGENT_REQUIRE_ENCRYPTED_CACHE', '1')
        ensure_encrypted_cache_ready(root)


class TestDocParserEncryptedCache:

    def test_doc_parser_uses_encrypted_storage_when_key_available(self, encryption_key_env):
        root = tempfile.mkdtemp()
        doc = [{'title': 'T', 'content': [{'text': 'Body', 'token': 2}]}]
        with patch('cat_agent.tools.doc_parser.SimpleDocParser') as mock_extractor_cls:
            mock_extractor_cls.return_value.call.return_value = doc
            parser = DocParser({'path': root, 'max_ref_token': 10_000})
            parser.call({'url': 'file:///tmp/report.pdf'})

        with sqlite3.connect(_sqlite_path(root)) as conn:
            value = conn.execute('SELECT value FROM kv').fetchone()[0]
        assert is_encrypted_value(value)

    def test_doc_parser_works_without_encryption_key(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_ENCRYPT_AT_REST', '1')
        monkeypatch.delenv('CAT_AGENT_ENCRYPTION_KEY', raising=False)
        monkeypatch.setattr('cat_agent.security.crypto._keyring_get_password', lambda: None)
        monkeypatch.setattr('cat_agent.security.crypto._keyring_set_password', lambda value: False)

        root = tempfile.mkdtemp()
        doc = [{'title': 'T', 'content': [{'text': 'Body', 'token': 2}]}]
        with patch('cat_agent.tools.doc_parser.SimpleDocParser') as mock_extractor_cls:
            mock_extractor_cls.return_value.call.return_value = doc
            parser = DocParser({'path': root, 'max_ref_token': 10_000})
            parser.call({'url': 'file:///tmp/report.pdf'})

        with sqlite3.connect(_sqlite_path(root)) as conn:
            value = conn.execute('SELECT value FROM kv').fetchone()[0]
        assert not is_encrypted_value(value)
