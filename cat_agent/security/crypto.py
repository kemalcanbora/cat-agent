"""AES-GCM helpers and key resolution for encrypted local storage."""

from __future__ import annotations

import base64
import os
import secrets
from typing import Optional

from cat_agent.log import logger

ENCRYPTION_MARKER = 'ENC1:'
BINARY_MARKER = b'ENC1'
NONCE_SIZE = 12
KEY_SIZE = 32
KEYRING_SERVICE = 'cat-agent'
KEYRING_USERNAME = 'encryption-key'


class EncryptionKeyError(RuntimeError):
    """Raised when an encryption key cannot be resolved."""


def _decode_key_material(raw: str) -> bytes:
    text = raw.strip()
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            key = decoder(text.encode('ascii'))
        except Exception:
            continue
        if len(key) == KEY_SIZE:
            return key
    raise EncryptionKeyError(
        'CAT_AGENT_ENCRYPTION_KEY must be a base64-encoded 32-byte AES key.'
    )


def _keyring_get_password() -> Optional[str]:
    try:
        import keyring
    except ImportError:
        return None
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    except Exception as error:
        logger.warning('OS keyring read failed: {}', error)
        return None


def _keyring_set_password(value: str) -> bool:
    try:
        import keyring
    except ImportError:
        return False
    try:
        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, value)
        return True
    except Exception as error:
        logger.warning('OS keyring write failed: {}', error)
        return False


def resolve_encryption_key(*, create_if_missing: bool = False) -> bytes:
    """Return the AES-256 key used for encrypted caches."""
    env_key = os.getenv('CAT_AGENT_ENCRYPTION_KEY', '').strip()
    if env_key:
        return _decode_key_material(env_key)

    stored = _keyring_get_password()
    if stored:
        return _decode_key_material(stored)

    if not create_if_missing:
        raise EncryptionKeyError(
            'No encryption key found. Set CAT_AGENT_ENCRYPTION_KEY or store a key in the OS keyring '
            f'({KEYRING_SERVICE}/{KEYRING_USERNAME}).'
        )

    key = secrets.token_bytes(KEY_SIZE)
    encoded = base64.urlsafe_b64encode(key).decode('ascii')
    if _keyring_set_password(encoded):
        logger.info('Generated a new encryption key and stored it in the OS keyring.')
        return key

    raise EncryptionKeyError(
        'No encryption key found and the OS keyring is unavailable. '
        'Set CAT_AGENT_ENCRYPTION_KEY to a base64-encoded 32-byte key.'
    )


def is_encrypted_value(value: str) -> bool:
    return value.startswith(ENCRYPTION_MARKER)


def is_encrypted_bytes(data: bytes) -> bool:
    return data.startswith(BINARY_MARKER)


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(NONCE_SIZE)
    ciphertext = AESGCM(key).encrypt(nonce, data, None)
    return BINARY_MARKER + nonce + ciphertext


def decrypt_bytes(token: bytes, key: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not is_encrypted_bytes(token):
        raise ValueError('Value is not encrypted binary data.')
    raw = token[len(BINARY_MARKER):]
    nonce, ciphertext = raw[:NONCE_SIZE], raw[NONCE_SIZE:]
    return AESGCM(key).decrypt(nonce, ciphertext, None)


def encrypt_value(plaintext: str, key: bytes) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(NONCE_SIZE)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext.encode('utf-8'), None)
    payload = base64.urlsafe_b64encode(nonce + ciphertext).decode('ascii')
    return f'{ENCRYPTION_MARKER}{payload}'


def decrypt_value(token: str, key: bytes) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not is_encrypted_value(token):
        raise ValueError('Value is not encrypted.')
    raw = base64.urlsafe_b64decode(token[len(ENCRYPTION_MARKER):].encode('ascii'))
    nonce, ciphertext = raw[:NONCE_SIZE], raw[NONCE_SIZE:]
    return AESGCM(key).decrypt(nonce, ciphertext, None).decode('utf-8')
