"""Tests for encrypted file fallback when no key is available."""

import os
import tempfile

import pytest

from cat_agent.security.encrypted_files import file_exists, read_bytes, write_bytes


@pytest.fixture(autouse=True)
def _disable_keyring(monkeypatch):
    monkeypatch.setenv('CAT_AGENT_ENCRYPT_AT_REST', '1')
    monkeypatch.delenv('CAT_AGENT_ENCRYPTION_KEY', raising=False)
    monkeypatch.delenv('CAT_AGENT_REQUIRE_ENCRYPTED_STORAGE', raising=False)
    monkeypatch.setattr(
        'cat_agent.security.crypto._keyring_get_password',
        lambda: None,
    )
    monkeypatch.setattr(
        'cat_agent.security.crypto._keyring_set_password',
        lambda value: False,
    )


class TestEncryptedFileFallback:

    def test_write_bytes_falls_back_to_plaintext_without_key(self):
        path = os.path.join(tempfile.mkdtemp(), 'rag_index.json')
        write_bytes(path, b'{"index": true}')
        assert os.path.isfile(path)
        assert read_bytes(path) == b'{"index": true}'
        assert file_exists(path)
