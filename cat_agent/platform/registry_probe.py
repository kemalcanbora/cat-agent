# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry reachability / TLS / Vault credential probes for ``cat-agent doctor``."""

from __future__ import annotations

import base64
import ssl
import urllib.error
import urllib.request
from typing import Optional

from cat_agent.platform.builder import registry_host
from cat_agent.platform.config import PlatformConfig
from cat_agent.platform.gateway import GatewayError, read_vault_kv_data


class RegistryError(Exception):
    """One-sentence registry / credential failure for doctor."""


def registry_base_url(config: PlatformConfig) -> str:
    """HTTP(S) base URL for the registry (OCI /v2/)."""
    host = registry_host(config)
    # Local stacks use plain HTTP (insecure-registries / loopback). Non-loopback
    # hosts default to HTTPS; TLS failures surface as a readable sentence.
    if host.startswith('127.') or host.startswith('localhost') or host.startswith('['):
        scheme = 'http'
    else:
        scheme = 'https'
    return f'{scheme}://{host}'


def probe_registry_reachability(config: PlatformConfig, *, timeout: float = 5.0) -> str:
    """GET /v2/; 200 or 401 means reachable. Returns a short ok sentence."""
    base = registry_base_url(config)
    url = f'{base}/v2/'
    req = urllib.request.Request(url, method='GET')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, 'status', None) or resp.getcode()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return f'registry: reachable at {base} (HTTP {exc.code} — auth required)'
        raise RegistryError(
            f'registry HTTP {exc.code} at {url}'
        ) from exc
    except ssl.SSLError as exc:
        raise RegistryError(
            f'registry TLS failure talking to {base}: {exc}. '
            'Use HTTP + insecure-registries for this stack, or install a trusted CA.'
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, 'reason', exc)
        if isinstance(reason, ssl.SSLError):
            raise RegistryError(
                f'registry TLS failure talking to {base}: {reason}. '
                'Use HTTP + insecure-registries for this stack, or install a trusted CA.'
            ) from exc
        raise RegistryError(
            f'registry unreachable at {base}: {reason}'
        ) from exc
    except TimeoutError as exc:
        raise RegistryError(f'registry timed out at {base}') from exc
    return f'registry: reachable at {base} (HTTP {code})'


def probe_registry_auth(
    config: PlatformConfig,
    *,
    username: str,
    password: str,
    timeout: float = 5.0,
) -> str:
    """Authenticated GET /v2/ with pull (or push) credentials."""
    base = registry_base_url(config)
    url = f'{base}/v2/'
    token = base64.b64encode(f'{username}:{password}'.encode()).decode()
    req = urllib.request.Request(
        url,
        headers={'Authorization': f'Basic {token}'},
        method='GET',
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = getattr(resp, 'status', None) or resp.getcode()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise RegistryError(
                f'registry auth failed at {base} (HTTP {exc.code}); '
                'check the registry credentials in Vault at '
                f'{config.registry_pull_vault_path()} / '
                f'{config.registry_push_vault_path()}'
            ) from exc
        raise RegistryError(f'registry HTTP {exc.code} at {url} during auth probe') from exc
    except ssl.SSLError as exc:
        raise RegistryError(
            f'registry TLS failure talking to {base}: {exc}. '
            'Use HTTP + insecure-registries for this stack, or install a trusted CA.'
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, 'reason', exc)
        raise RegistryError(f'registry unreachable at {base}: {reason}') from exc
    return f'registry auth: ok (HTTP {code})'


def vault_registry_creds_exist(config: PlatformConfig, which: str) -> None:
    """Assert push or pull Vault secret has non-empty username/password."""
    if which == 'push':
        path = config.registry_push_vault_path()
    elif which == 'pull':
        path = config.registry_pull_vault_path()
    else:
        raise ValueError(which)
    try:
        data = read_vault_kv_data(config.vault_addr, path)
    except GatewayError as exc:
        raise RegistryError(str(exc)) from exc
    user = str(data.get('username') or '').strip()
    password = str(data.get('password') or '').strip()
    if not user or not password:
        raise RegistryError(
            f'Vault {path} missing username/password; run: cat-agent stack seed --registry'
        )
