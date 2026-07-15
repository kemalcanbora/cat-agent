"""Air-gap mode: block outbound network access when enabled."""

from __future__ import annotations

import os
import socket
from typing import Callable, Optional

from cat_agent.log import logger

_ORIGINAL_SOCKET_CONNECT: Optional[Callable] = None
_GUARDS_INSTALLED = False


class OfflineViolationError(RuntimeError):
    """Raised when offline mode blocks an outbound network operation."""


def is_offline_mode() -> bool:
    value = os.getenv('CAT_AGENT_OFFLINE', '').strip().lower()
    return value in {'1', 'true', 'yes', 'on'}


def guard_outbound_request(*, purpose: str) -> None:
    if is_offline_mode():
        raise OfflineViolationError(
            f'Offline mode is enabled (CAT_AGENT_OFFLINE=1). '
            f'Blocked outbound request: {purpose}'
        )


def _guarded_socket_connect(self, address):  # noqa: ANN001
    guard_outbound_request(purpose=f'socket connect to {address}')
    return _ORIGINAL_SOCKET_CONNECT(self, address)


def install_offline_guards() -> None:
    global _ORIGINAL_SOCKET_CONNECT, _GUARDS_INSTALLED
    if _GUARDS_INSTALLED or not is_offline_mode():
        return

    import requests

    _original_request = requests.sessions.Session.request

    def guarded_request(session, method, url, *args, **kwargs):  # noqa: ANN001
        guard_outbound_request(purpose=f'HTTP {method} {url}')
        return _original_request(session, method, url, *args, **kwargs)

    requests.sessions.Session.request = guarded_request

    _ORIGINAL_SOCKET_CONNECT = socket.socket.connect
    socket.socket.connect = _guarded_socket_connect  # type: ignore[method-assign]
    _GUARDS_INSTALLED = True
    logger.info('Installed offline network guards (CAT_AGENT_OFFLINE=1)')
