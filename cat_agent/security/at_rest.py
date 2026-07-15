"""Policy flags for encrypted local storage."""

from __future__ import annotations

import os


def is_encrypt_at_rest_enabled() -> bool:
    for name in ('CAT_AGENT_ENCRYPT_AT_REST', 'CAT_AGENT_ENCRYPT_CACHE'):
        value = os.getenv(name, '').strip().lower()
        if value:
            return value not in {'0', 'false', 'no', 'off'}
    return True


def is_encrypted_at_rest_required() -> bool:
    for name in ('CAT_AGENT_REQUIRE_ENCRYPTED_STORAGE', 'CAT_AGENT_REQUIRE_ENCRYPTED_CACHE'):
        value = os.getenv(name, '').strip().lower()
        if value in {'1', 'true', 'yes', 'on'}:
            return True
    return False
