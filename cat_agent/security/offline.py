"""Air-gap mode: block outbound network access when enabled."""

from __future__ import annotations

import ipaddress
import os
import socket
from typing import Callable, List, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

from cat_agent.log import logger

_ORIGINAL_SOCKET_CONNECT: Optional[Callable] = None
_GUARDS_INSTALLED = False


class OfflineViolationError(RuntimeError):
    """Raised when offline mode blocks an outbound network operation."""


def is_offline_mode() -> bool:
    value = os.getenv('CAT_AGENT_OFFLINE', '').strip().lower()
    return value in {'1', 'true', 'yes', 'on'}


def _host_from_url(url: str) -> str:
    candidate = url.strip()
    if not candidate:
        return ''
    if '://' not in candidate:
        candidate = f'http://{candidate}'
    return (urlparse(candidate).hostname or '').strip()


def _collect_allowlist_entries() -> List[str]:
    entries: List[str] = []
    raw = os.getenv('CAT_AGENT_OFFLINE_ALLOW_HOSTS', '')
    for part in raw.split(','):
        part = part.strip()
        if part:
            entries.append(part)

    for env_name in ('OPENAI_BASE_URL', 'CAT_AGENT_LLM_BASE_URL'):
        host = _host_from_url(os.getenv(env_name, ''))
        if host and host not in entries:
            entries.append(host)

    return entries


def get_offline_allow_hosts() -> List[str]:
    """Return the configured offline allowlist entries (including auto-exempt LLM hosts)."""
    return _collect_allowlist_entries()


def _parse_allowlist(entries: Sequence[str]) -> Tuple[set[str], List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]]:
    hosts: set[str] = set()
    networks: List[ipaddress._BaseNetwork] = []
    for entry in entries:
        candidate = entry.strip()
        if not candidate:
            continue
        if '/' in candidate:
            try:
                networks.append(ipaddress.ip_network(candidate, strict=False))
                continue
            except ValueError:
                pass
        try:
            hosts.add(str(ipaddress.ip_address(candidate)))
        except ValueError:
            hosts.add(candidate.lower())
    return hosts, networks


def is_host_allowed(host: str) -> bool:
    if not host:
        return False
    hosts, networks = _parse_allowlist(_collect_allowlist_entries())
    normalized = host.strip().lower()
    if normalized in hosts:
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    if str(ip) in hosts:
        return True
    return any(ip in network for network in networks)


def guard_outbound_request(
    *,
    purpose: str,
    host: Optional[str] = None,
    url: Optional[str] = None,
) -> None:
    if not is_offline_mode():
        return

    target_host = (host or '').strip() or _host_from_url(url or '')
    if target_host and is_host_allowed(target_host):
        return

    raise OfflineViolationError(
        f'Offline mode is enabled (CAT_AGENT_OFFLINE=1). '
        f'Blocked outbound request: {purpose}'
    )


def _guarded_socket_connect(self, address):  # noqa: ANN001
    if isinstance(address, str):
        return _ORIGINAL_SOCKET_CONNECT(self, address)

    if isinstance(address, tuple) and address:
        host = address[0]
        if isinstance(host, str):
            guard_outbound_request(purpose=f'socket connect to {address}', host=host)
            return _ORIGINAL_SOCKET_CONNECT(self, address)

    guard_outbound_request(purpose=f'socket connect to {address}')
    return _ORIGINAL_SOCKET_CONNECT(self, address)


def install_offline_guards() -> None:
    global _ORIGINAL_SOCKET_CONNECT, _GUARDS_INSTALLED
    if _GUARDS_INSTALLED or not is_offline_mode():
        return

    import requests

    _original_request = requests.sessions.Session.request

    def guarded_request(session, method, url, *args, **kwargs):  # noqa: ANN001
        guard_outbound_request(purpose=f'HTTP {method} {url}', url=str(url))
        return _original_request(session, method, url, *args, **kwargs)

    requests.sessions.Session.request = guarded_request

    _ORIGINAL_SOCKET_CONNECT = socket.socket.connect
    socket.socket.connect = _guarded_socket_connect  # type: ignore[method-assign]
    _GUARDS_INSTALLED = True
    allowed = get_offline_allow_hosts()
    if allowed:
        logger.info(
            'Installed offline network guards (CAT_AGENT_OFFLINE=1) with allowlist: {}',
            ', '.join(allowed),
        )
    else:
        logger.info('Installed offline network guards (CAT_AGENT_OFFLINE=1)')
