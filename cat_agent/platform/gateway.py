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

"""LLM gateway helpers: alias listing, Vault key existence, Consul DNS checks."""

from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import urllib.error
import urllib.request
from typing import List, Optional, Sequence
from urllib.parse import urlparse

from cat_agent.platform.config import PlatformConfig

GATEWAY_HOST = 'llm-gateway.service.consul'


class GatewayError(Exception):
    """One-sentence failure talking to the LLM gateway or its deps."""


def gateway_models_url(llm_gateway: str) -> str:
    """Map config llm_gateway (.../v1) to OpenAI-compatible /v1/models."""
    base = (llm_gateway or '').rstrip('/')
    if base.endswith('/v1'):
        return f'{base}/models'
    return f'{base}/v1/models'


def gateway_health_url(llm_gateway: str) -> str:
    base = (llm_gateway or '').rstrip('/')
    if base.endswith('/v1'):
        root = base[: -len('/v1')]
    else:
        root = base
    return f'{root}/health/liveliness'


def _http_json(
    url: str,
    *,
    headers: Optional[dict] = None,
    timeout: float = 10.0,
) -> dict | list:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        detail = ''
        try:
            detail = exc.read().decode('utf-8', errors='replace')[:200]
        except Exception:  # noqa: BLE001
            pass
        if exc.code in (401, 403):
            raise GatewayError(
                f'LLM gateway auth failed at {url} (HTTP {exc.code}); '
                'check LITELLM_MASTER_KEY in Vault'
            ) from exc
        raise GatewayError(
            f'LLM gateway HTTP {exc.code} at {url}'
            + (f': {detail}' if detail else '')
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, 'reason', exc)
        raise GatewayError(
            f'LLM gateway unreachable at {url}: {reason}'
        ) from exc
    except TimeoutError as exc:
        raise GatewayError(f'LLM gateway timed out at {url}') from exc
    try:
        return json.loads(raw or '{}')
    except json.JSONDecodeError as exc:
        raise GatewayError(f'LLM gateway returned non-JSON from {url}') from exc


def ensure_dev_vault_token() -> None:
    """Local compose Vault uses ``-dev`` root token when ``VAULT_TOKEN`` is unset."""
    os.environ.setdefault('VAULT_TOKEN', 'root')


def _vault_token(token: Optional[str] = None) -> str:
    tok = (token if token is not None else os.environ.get('VAULT_TOKEN', '')).strip()
    if not tok:
        raise GatewayError(
            'VAULT_TOKEN is not set; cannot talk to Vault '
            '(local stack: export VAULT_TOKEN=root, or run cat-agent stack seed first)'
        )
    return tok


def read_vault_kv_data(
    vault_addr: str,
    api_path: str,
    *,
    token: Optional[str] = None,
    timeout: float = 5.0,
) -> dict:
    """Read KV v2 secret data (API path like secret/data/...). Never logs values."""
    tok = _vault_token(token)
    addr = vault_addr.rstrip('/')
    path = api_path.lstrip('/')
    url = f'{addr}/v1/{path}'
    req = urllib.request.Request(url, headers={'X-Vault-Token': tok})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise GatewayError(
                f'Vault secret not found at {api_path}; '
                'run: cat-agent stack seed  (from cat-agent-stack, with VAULT_TOKEN=root)'
            ) from exc
        if exc.code in (401, 403):
            raise GatewayError(
                f'Vault denied read of {api_path} (HTTP {exc.code}); check VAULT_TOKEN'
            ) from exc
        raise GatewayError(f'Vault HTTP {exc.code} reading {api_path}') from exc
    except urllib.error.URLError as exc:
        raise GatewayError(f'Vault unreachable at {vault_addr}: {exc.reason}') from exc
    data = (body.get('data') or {}).get('data') or {}
    if not isinstance(data, dict):
        raise GatewayError(f'Vault secret {api_path} has unexpected shape')
    return data


def write_vault_kv_data(
    vault_addr: str,
    api_path: str,
    data: dict,
    *,
    token: Optional[str] = None,
    timeout: float = 10.0,
) -> None:
    """Write KV v2 secret data (API path like secret/data/...). Never logs values."""
    tok = _vault_token(token)
    addr = vault_addr.rstrip('/')
    path = api_path.lstrip('/')
    url = f'{addr}/v1/{path}'
    body = json.dumps({'data': data}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={'X-Vault-Token': tok, 'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise GatewayError(
                f'Vault denied write of {api_path} (HTTP {exc.code}); check VAULT_TOKEN'
            ) from exc
        raise GatewayError(f'Vault HTTP {exc.code} writing {api_path}') from exc
    except urllib.error.URLError as exc:
        raise GatewayError(f'Vault unreachable at {vault_addr}: {exc.reason}') from exc


def write_vault_policy(
    vault_addr: str,
    name: str,
    policy_hcl: str,
    *,
    token: Optional[str] = None,
    timeout: float = 10.0,
) -> None:
    """Create/update a Vault ACL policy by name."""
    tok = _vault_token(token)
    addr = vault_addr.rstrip('/')
    url = f'{addr}/v1/sys/policies/acl/{name}'
    body = json.dumps({'policy': policy_hcl}).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={'X-Vault-Token': tok, 'Content-Type': 'application/json'},
        method='PUT',
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        raise GatewayError(
            f'Vault HTTP {exc.code} writing policy {name}'
        ) from exc
    except urllib.error.URLError as exc:
        raise GatewayError(f'Vault unreachable at {vault_addr}: {exc.reason}') from exc


def vault_team_key_exists(cfg: PlatformConfig, team: str) -> None:
    """Assert team virtual-key secret exists and api_key is non-empty (value unused)."""
    path = cfg.team_llm_vault_path(team)
    data = read_vault_kv_data(cfg.vault_addr, path)
    key = data.get('api_key')
    if not key or not str(key).strip():
        raise GatewayError(
            f'Vault {path} exists but api_key is empty; run: cat-agent stack seed'
        )


def master_key_from_vault(cfg: PlatformConfig) -> str:
    data = read_vault_kv_data(cfg.vault_addr, cfg.llm_credentials_path)
    key = data.get('LITELLM_MASTER_KEY')
    if not key or not str(key).strip():
        raise GatewayError(
            f'Vault {cfg.llm_credentials_path} missing LITELLM_MASTER_KEY; '
            'run: cat-agent stack seed'
        )
    return str(key).strip()


def _model_id_from_item(item: dict) -> Optional[str]:
    mid = (
        item.get('id')
        or item.get('model_name')
        or item.get('model')
        or item.get('name')
    )
    return str(mid) if mid else None


def parse_model_ids(payload: dict | list) -> List[str]:
    """Extract model/alias ids from /v1/models or /api/tags style JSON."""
    ids: List[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                mid = _model_id_from_item(item)
                if mid:
                    ids.append(mid)
        return sorted(set(ids))
    if not isinstance(payload, dict):
        return []
    data = payload.get('data')
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                mid = _model_id_from_item(item)
                if mid:
                    ids.append(mid)
    # Ollama /api/tags and some /model/info shapes use {models: [...]}
    models = payload.get('models')
    if isinstance(models, list):
        for item in models:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict):
                mid = _model_id_from_item(item)
                if mid:
                    ids.append(mid)
    return sorted(set(ids))


def is_wildcard_model_id(model_id: str) -> bool:
    """True for LiteLLM pass-through patterns that are not concrete model names."""
    mid = (model_id or '').strip()
    if not mid:
        return True
    if mid in ('*', '*/*'):
        return True
    return mid.endswith('/*')


def concrete_model_ids(models: Sequence[str]) -> List[str]:
    return [m for m in models if not is_wildcard_model_id(m)]


def model_alias_matches(alias: str, available: Sequence[str]) -> bool:
    """Exact match, Ollama ``:latest`` variants, and stripped ``provider/`` prefixes."""
    wanted = (alias or '').strip()
    if not wanted:
        return False
    names = set()
    for raw in available:
        mid = (raw or '').strip()
        if not mid or is_wildcard_model_id(mid):
            continue
        names.add(mid)
        if '/' in mid:
            names.add(mid.split('/', 1)[1])
    candidates = {wanted}
    if ':' not in wanted:
        candidates.add(f'{wanted}:latest')
    elif wanted.endswith(':latest'):
        candidates.add(wanted[: -len(':latest')])
    return bool(candidates & names)


def fetch_gateway_aliases(
    llm_gateway: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
) -> List[str]:
    """List advertised aliases/models from the gateway. Fail closed on errors."""
    url = gateway_models_url(llm_gateway)
    headers = {'Accept': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    payload = _http_json(url, headers=headers, timeout=timeout)
    aliases = parse_model_ids(payload)
    if not aliases:
        raise GatewayError(
            f'LLM gateway at {url} returned no models/aliases; check LiteLLM config'
        )
    return aliases


def fetch_ollama_model_ids(
    api_base: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 10.0,
) -> List[str]:
    """List concrete models from an Ollama / OpenAI-compatible base URL."""
    base = (api_base or '').rstrip('/')
    if not base:
        raise GatewayError('OLLAMA_API_BASE is empty; cannot list models')
    headers = {'Accept': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    if base.endswith('/v1'):
        urls = [f'{base}/models']
    else:
        urls = [f'{base}/v1/models', f'{base}/api/tags']
    last_err: Optional[GatewayError] = None
    for url in urls:
        try:
            payload = _http_json(url, headers=headers, timeout=timeout)
        except GatewayError as exc:
            last_err = exc
            continue
        ids = concrete_model_ids(parse_model_ids(payload))
        if ids:
            return ids
    if last_err is not None:
        raise last_err
    raise GatewayError(
        f'Ollama at {base} returned no models; pull a model or check OLLAMA_API_BASE'
    )


def fetch_ollama_models_for_config(cfg: PlatformConfig) -> List[str]:
    """Read OLLAMA_* from Vault platform LLM secret and list backend models."""
    data = read_vault_kv_data(cfg.vault_addr, cfg.llm_credentials_path)
    api_base = str(
        data.get('OLLAMA_API_BASE') or data.get('OLLAMA_BASE_URL') or ''
    ).strip()
    api_key = str(data.get('OLLAMA_API_KEY') or '').strip() or None
    if not api_base:
        raise GatewayError(
            f'Vault {cfg.llm_credentials_path} missing OLLAMA_API_BASE; '
            'run: cat-agent stack seed'
        )
    return fetch_ollama_model_ids(api_base, api_key=api_key)


def validate_manifest_alias(alias: str, aliases: Sequence[str]) -> None:
    concrete = concrete_model_ids(aliases)
    if model_alias_matches(alias, concrete):
        return
    if not concrete:
        raise GatewayError(
            f'model.alias {alias!r} could not be verified: backend returned no '
            'concrete models (gateway may only advertise wildcards). '
            'Check LiteLLM check_provider_endpoint / OLLAMA_API_BASE'
        )
    preview = concrete[:20]
    valid = ', '.join(preview)
    if len(concrete) > len(preview):
        valid += f', … (+{len(concrete) - len(preview)} more)'
    raise GatewayError(
        f'model.alias {alias!r} is not available from the LLM backend; valid: {valid}'
    )


def _dns_query_a(name: str, server: str, timeout: float = 3.0) -> List[str]:
    """Minimal UDP DNS A-record query (no dnspython dependency)."""
    labels = name.rstrip('.').split('.')
    q = b''.join(bytes([len(l)]) + l.encode('ascii') for l in labels) + b'\x00'
    # header: id=0xCAFE, flags=RD, qdcount=1
    header = struct.pack('!HHHHHH', 0xCAFE, 0x0100, 1, 0, 0, 0)
    # QTYPE=A (1), QCLASS=IN (1)
    packet = header + q + struct.pack('!HH', 1, 1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(packet, (server, 53))
        data, _ = sock.recvfrom(512)
    finally:
        sock.close()
    if len(data) < 12:
        return []
    ancount = struct.unpack('!H', data[6:8])[0]
    # skip question
    i = 12
    while i < len(data) and data[i] != 0:
        i += 1 + data[i]
    i += 5  # null + qtype + qclass
    addrs: List[str] = []
    for _ in range(ancount):
        if i >= len(data):
            break
        if data[i] & 0xC0 == 0xC0:
            i += 2
        else:
            while i < len(data) and data[i] != 0:
                i += 1 + data[i]
            i += 1
        if i + 10 > len(data):
            break
        rtype, _, _, rdlength = struct.unpack('!HHIH', data[i : i + 10])
        i += 10
        rdata = data[i : i + rdlength]
        i += rdlength
        if rtype == 1 and rdlength == 4:
            addrs.append(socket.inet_ntoa(rdata))
    return addrs


def resolve_gateway_via_consul_dns(
    consul_dns: str,
    *,
    hostname: str = GATEWAY_HOST,
    docker_network: str = '',
    skip_docker_network: bool = False,
    timeout: float = 10.0,
) -> str:
    """Resolve hostname using consul_dns; prefer docker network (matches allocs)."""
    dns = (consul_dns or '').strip()
    if not dns:
        raise GatewayError('consul_dns is not set in platform config')

    network = (docker_network or '').strip()
    if network and not skip_docker_network and shutil.which('docker'):
        try:
            proc = subprocess.run(
                [
                    'docker',
                    'run',
                    '--rm',
                    f'--network={network}',
                    f'--dns={dns}',
                    'alpine:3.20',
                    'getent',
                    'hosts',
                    hostname,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GatewayError(
                f'consul_dns check failed (docker resolve {hostname} via {dns}): {exc}'
            ) from exc
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or '').strip()[:200]
            raise GatewayError(
                f'consul_dns {dns} did not resolve {hostname} on network {network}'
                + (f': {err}' if err else '')
            )
        line = (proc.stdout or '').strip().splitlines()[0] if proc.stdout else ''
        ip = line.split()[0] if line else ''
        if not ip:
            raise GatewayError(
                f'consul_dns {dns} returned empty address for {hostname}'
            )
        return ip

    # Host-side UDP query (works when Consul DNS is reachable from the operator).
    try:
        addrs = _dns_query_a(hostname, dns)
    except OSError as exc:
        raise GatewayError(
            f'consul_dns {dns} not reachable for {hostname}: {exc}'
        ) from exc
    if not addrs:
        raise GatewayError(
            f'consul_dns {dns} returned no A record for {hostname} '
            '(stale IP after Docker network recreate?)'
        )
    return addrs[0]


def fetch_aliases_for_config(cfg: PlatformConfig) -> List[str]:
    """Fetch aliases using Vault master key; used by doctor and deploy."""
    api_key = master_key_from_vault(cfg)
    # Prefer in-network URL when docker_network + consul_dns (host can't resolve
    # *.service.consul). Fall back to rewriting host for published :4000.
    return fetch_gateway_aliases_reachable(cfg, api_key=api_key)


def fetch_gateway_aliases_reachable(
    cfg: PlatformConfig,
    *,
    api_key: str,
    timeout: float = 15.0,
) -> List[str]:
    """Reach the gateway the way allocs do when possible; else host fallbacks."""
    if cfg.platform_host_is_remote():
        from cat_agent.platform.stack import operator_llm_gateway

        gateway = operator_llm_gateway(cfg)
        root = gateway.rstrip('/')
        if not root.endswith('/v1'):
            gateway = f'{root}/v1'
        return fetch_gateway_aliases(gateway, api_key=api_key, timeout=timeout)

    gateway = cfg.llm_gateway
    network = (cfg.docker_network or '').strip()
    dns = (cfg.consul_dns or '').strip()

    if network and dns and shutil.which('docker'):
        url = gateway_models_url(gateway)
        # curl from a container on the compose network with Consul DNS.
        try:
            proc = subprocess.run(
                [
                    'docker',
                    'run',
                    '--rm',
                    f'--network={network}',
                    f'--dns={dns}',
                    'curlimages/curl:8.5.0',
                    '-sS',
                    '-f',
                    '-H',
                    f'Authorization: Bearer {api_key}',
                    '-H',
                    'Accept: application/json',
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GatewayError(
                f'LLM gateway unreachable via Consul DNS ({url}): {exc}'
            ) from exc
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or '').strip()[:240]
            raise GatewayError(
                f'LLM gateway unreachable via Consul DNS ({url})'
                + (f': {err}' if err else '')
            )
        try:
            payload = json.loads(proc.stdout or '{}')
        except json.JSONDecodeError as exc:
            raise GatewayError(
                f'LLM gateway returned non-JSON from {url}'
            ) from exc
        aliases = parse_model_ids(payload)
        if not aliases:
            raise GatewayError(
                f'LLM gateway at {url} returned no models/aliases; check LiteLLM config'
            )
        return aliases

    # Host fallback: try as configured, then localhost published port.
    try:
        return fetch_gateway_aliases(gateway, api_key=api_key, timeout=timeout)
    except GatewayError as first:
        parsed = urlparse(gateway)
        if parsed.hostname and parsed.hostname.endswith('.service.consul'):
            port = parsed.port or 4000
            alt = gateway.replace(parsed.netloc, f'127.0.0.1:{port}', 1)
            try:
                return fetch_gateway_aliases(alt, api_key=api_key, timeout=timeout)
            except GatewayError:
                raise first from None
        raise


def ensure_alias_or_raise(cfg: PlatformConfig, alias: str) -> List[str]:
    """Fail-closed: model.alias must exist on the LLM backend (not a fixed allowlist)."""
    models = fetch_aliases_for_config(cfg)
    concrete = concrete_model_ids(models)
    if not concrete:
        concrete = fetch_ollama_models_for_config(cfg)
    validate_manifest_alias(alias, concrete)
    return concrete
