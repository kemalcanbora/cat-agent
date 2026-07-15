"""Encrypted binary and JSON file helpers for RAG indexes and local data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

from cat_agent.security.at_rest import is_encrypt_at_rest_enabled, is_encrypted_at_rest_required
from cat_agent.security.crypto import (
    EncryptionKeyError,
    decrypt_bytes,
    encrypt_bytes,
    is_encrypted_bytes,
    resolve_encryption_key,
)

ENCRYPTED_SUFFIX = '.enc'
BINARY_MARKER = b'ENC1'


class PlaintextStorageError(RuntimeError):
    """Raised when encrypted storage is required but plaintext files exist."""


def encrypted_path(path: str | Path) -> str:
    return f'{path}{ENCRYPTED_SUFFIX}'


def _resolve_key(*, create_if_missing: bool) -> bytes:
    return resolve_encryption_key(create_if_missing=create_if_missing)


def write_bytes(path: str | Path, data: bytes, *, encrypt: Optional[bool] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    use_encryption = is_encrypt_at_rest_enabled() if encrypt is None else encrypt
    if use_encryption:
        key = _resolve_key(create_if_missing=True)
        enc_path = Path(encrypted_path(path))
        enc_path.write_bytes(encrypt_bytes(data, key))
        if path.is_file() and not is_encrypted_bytes(path.read_bytes()[:4]):
            path.unlink()
        return
    path.write_bytes(data)
    enc_path = Path(encrypted_path(path))
    if enc_path.is_file():
        enc_path.unlink()


def read_bytes(path: str | Path, *, encrypt: Optional[bool] = None) -> Optional[bytes]:
    path = Path(path)
    enc_path = Path(encrypted_path(path))
    use_encryption = is_encrypt_at_rest_enabled() if encrypt is None else encrypt

    if enc_path.is_file():
        key = _resolve_key(create_if_missing=False)
        return decrypt_bytes(enc_path.read_bytes(), key)

    if not path.is_file():
        return None

    raw = path.read_bytes()
    if is_encrypted_bytes(raw):
        key = _resolve_key(create_if_missing=False)
        return decrypt_bytes(raw, key)

    if use_encryption and is_encrypted_at_rest_required():
        raise PlaintextStorageError(f'Plaintext file found at {path} while encrypted storage is required.')
    return raw


def write_json(path: str | Path, payload: Any, *, encrypt: Optional[bool] = None) -> None:
    write_bytes(path, json.dumps(payload, ensure_ascii=False).encode('utf-8'), encrypt=encrypt)


def read_json(path: str | Path, *, encrypt: Optional[bool] = None) -> dict:
    raw = read_bytes(path, encrypt=encrypt)
    if raw is None:
        return {}
    try:
        data = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def file_exists(path: str | Path) -> bool:
    path = Path(path)
    return path.is_file() or Path(encrypted_path(path)).is_file()


def migrate_plaintext_file(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    if Path(encrypted_path(path)).is_file():
        return False
    data = path.read_bytes()
    if is_encrypted_bytes(data):
        return False
    write_bytes(path, data, encrypt=True)
    return True
