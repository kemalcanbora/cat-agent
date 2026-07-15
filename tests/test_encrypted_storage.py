"""Tests for encrypted file storage and workspace migration."""

import base64
import json
import os
import tempfile

import pytest

from cat_agent.security.encrypted_files import (
    ENCRYPTED_SUFFIX,
    encrypted_path,
    file_exists,
    migrate_plaintext_file,
    read_bytes,
    read_json,
    write_bytes,
    write_json,
)
from cat_agent.security.encrypted_migrate import migrate_workspace_storage


@pytest.fixture
def encryption_key_env(monkeypatch):
    key = base64.urlsafe_b64encode(b'1' * 32).decode('ascii')
    monkeypatch.setenv('CAT_AGENT_ENCRYPTION_KEY', key)
    monkeypatch.setenv('CAT_AGENT_ENCRYPT_AT_REST', '1')
    return key


class TestEncryptedFiles:

    def test_write_read_bytes_encrypted(self, encryption_key_env):
        path = os.path.join(tempfile.mkdtemp(), 'index.usearch')
        write_bytes(path, b'binary-index-data')
        assert file_exists(path)
        assert os.path.isfile(encrypted_path(path))
        assert read_bytes(path) == b'binary-index-data'

    def test_write_read_json_encrypted(self, encryption_key_env):
        path = os.path.join(tempfile.mkdtemp(), 'meta.json')
        write_json(path, {'fingerprint': 'abc', 'chunks': []})
        loaded = read_json(path)
        assert loaded['fingerprint'] == 'abc'

    def test_migrate_plaintext_file(self, encryption_key_env):
        path = os.path.join(tempfile.mkdtemp(), 'rag_index.json')
        with open(path, 'wb') as handle:
            handle.write(b'{"token":"data"}')
        assert migrate_plaintext_file(path) is True
        assert os.path.isfile(encrypted_path(path))
        assert read_bytes(path) == b'{"token":"data"}'


class TestWorkspaceMigration:

    def test_migrate_workspace_storage_encrypts_sqlite_and_indexes(self, encryption_key_env):
        workspace = tempfile.mkdtemp()
        storage_dir = os.path.join(workspace, 'tools', 'storage')
        os.makedirs(storage_dir, exist_ok=True)
        db_path = os.path.join(storage_dir, 'storage.sqlite')
        import sqlite3

        with sqlite3.connect(db_path) as conn:
            conn.execute('CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            conn.execute('INSERT INTO kv(key, value) VALUES (?, ?)', ('mem', 'secret note'))
            conn.commit()

        index_dir = os.path.join(workspace, 'storage', 'keyword_indexes')
        os.makedirs(index_dir, exist_ok=True)
        index_path = os.path.join(index_dir, 'rag_index.json')
        with open(index_path, 'wb') as handle:
            handle.write(b'index-bytes')

        report = migrate_workspace_storage(workspace)
        assert report['sqlite_records'] == 1
        assert report['index_files'] == 1
        assert os.path.isfile(encrypted_path(index_path))
