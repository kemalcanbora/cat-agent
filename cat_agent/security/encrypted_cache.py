"""Encrypted doc-parser cache policy and migration helpers."""

from __future__ import annotations

import os
import sqlite3
from typing import Iterable, Optional, Tuple

from cat_agent.log import logger
from cat_agent.security.at_rest import is_encrypt_at_rest_enabled, is_encrypted_at_rest_required
from cat_agent.security.crypto import (
    EncryptionKeyError,
    decrypt_value,
    encrypt_value,
    is_encrypted_value,
    resolve_encryption_key,
)
from cat_agent.tools.storage import _sqlite_path


class PlaintextCacheError(RuntimeError):
    """Raised when encrypted cache is required but plaintext data exists."""


def is_encrypt_cache_enabled() -> bool:
    return is_encrypt_at_rest_enabled()


def is_encrypted_cache_required() -> bool:
    return is_encrypted_at_rest_required()


def should_encrypt_doc_parser_cache() -> bool:
    return is_encrypt_at_rest_enabled()


def _connect(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def _iter_values(db_path: str) -> Iterable[Tuple[str, str]]:
    if not os.path.isfile(db_path):
        return
    with _connect(db_path) as conn:
        for key, value in conn.execute('SELECT key, value FROM kv ORDER BY key'):
            yield key, value


def count_plaintext_values(db_path: str) -> int:
    return sum(1 for _, value in _iter_values(db_path) if not is_encrypted_value(value))


def ensure_encrypted_cache_ready(storage_root: str) -> Optional[bytes]:
    """Validate cache policy and return the encryption key when available."""
    if not should_encrypt_doc_parser_cache():
        return None

    db_path = _sqlite_path(storage_root)
    plaintext_count = count_plaintext_values(db_path)
    if plaintext_count and is_encrypted_at_rest_required():
        raise PlaintextCacheError(
            f'Found {plaintext_count} plaintext cache record(s) under {storage_root}. '
            'Run `cat-agent encrypt-storage` before enabling '
            'CAT_AGENT_REQUIRE_ENCRYPTED_STORAGE=1.'
        )

    try:
        return resolve_encryption_key(create_if_missing=True)
    except EncryptionKeyError:
        if is_encrypted_at_rest_required():
            raise
        logger.warning(
            'Encryption is enabled but no key is available; SQLite cache at {} will use plaintext.',
            storage_root,
        )
        return None


def maybe_decrypt_value(value: str, key: Optional[bytes], *, encrypt_at_rest: bool) -> str:
    if is_encrypted_value(value):
        if key is None:
            raise EncryptionKeyError(
                'Encrypted cache value found but no decryption key is available.'
            )
        return decrypt_value(value, key)
    if encrypt_at_rest and is_encrypted_at_rest_required():
        raise PlaintextCacheError('Plaintext cache value found while encrypted storage is required.')
    return value


def maybe_encrypt_value(value: str, key: Optional[bytes], *, encrypt_at_rest: bool) -> str:
    if not encrypt_at_rest or key is None:
        return value
    if is_encrypted_value(value):
        return value
    return encrypt_value(value, key)


def migrate_plaintext_cache(storage_root: str) -> int:
    """Encrypt all plaintext values in a cache database. Returns migrated count."""
    os.makedirs(storage_root, exist_ok=True)
    db_path = _sqlite_path(storage_root)
    if not os.path.isfile(db_path):
        return 0

    key = resolve_encryption_key(create_if_missing=True)
    migrated = 0
    with _connect(db_path) as conn:
        rows = conn.execute('SELECT key, value FROM kv ORDER BY key').fetchall()
        for stored_key, value in rows:
            if is_encrypted_value(value):
                continue
            conn.execute(
                'UPDATE kv SET value = ? WHERE key = ?',
                (encrypt_value(value, key), stored_key),
            )
            migrated += 1
        conn.commit()
    return migrated
