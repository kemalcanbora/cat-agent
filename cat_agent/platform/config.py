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

"""Operator config: ~/.cat-agent/config.toml + env + flags."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

KNOWN_PLATFORM_KEYS = frozenset(
    {
        'nomad_addr',
        'nomad_token',
        'namespace',
        'datacenters',
        'registry',
        'llm_gateway',
        # Vault KV v2 API path for provider/master secrets (CLI: secret/platform/llm).
        # Team virtual keys live at {llm_credentials_path}/teams/{team}.
        'llm_credentials_path',
        # Vault KV v2 API base for registry accounts (CLI: secret/platform/registry).
        # Push/pull are separate secrets: {registry_credentials_path}/push|pull.
        'registry_credentials_path',
        'otel_endpoint',
        'vault_addr',
        'max_timeout_seconds',
        'base_image',
        # Optional Docker network mode. Docker Desktop / macOS only in the
        # cat-agent-stack on Docker Desktop / macOS (bridge CNI broken there). On Linux Nomad:
        # leave empty and use bridge.
        'docker_network',
        # Consul DNS server IP for docker_network tasks (*.service.consul).
        # Docker Desktop stacks pin compose Consul (e.g. 10.32.0.2); that IP
        # is not portable — on Linux use the real Consul DNS address.
        'consul_dns',
        # Host-facing URL template after deploy, e.g.
        # http://{team}-{name}.localhost:8088 (Traefik on the local stack).
        'public_url_template',
        # Traefik Consul Catalog Host() rule, e.g. "{team}-{name}.localhost"
        # or "{team}-{name}.agents.example.internal" for LAN/corp DNS.
        'ingress_host_template',
    }
)


class ConfigError(ValueError):
    """Invalid operator configuration."""


@dataclass
class PlatformConfig:
    nomad_addr: str = 'http://127.0.0.1:4646'
    nomad_token: str = ''
    namespace: str = 'default'
    datacenters: List[str] = field(default_factory=lambda: ['dc1'])
    registry: str = 'local'
    llm_gateway: str = 'http://llm-gateway.service.consul:4000/v1'
    # KV v2 API path (Nomad template / Vault HTTP). CLI write uses secret/platform/llm.
    llm_credentials_path: str = 'secret/data/platform/llm'
    # Base path; pull = …/pull (Nomad jobs), push = …/push (developer deploy).
    registry_credentials_path: str = 'secret/data/platform/registry'
    otel_endpoint: str = ''
    vault_addr: str = 'http://127.0.0.1:8200'
    max_timeout_seconds: int = 3600
    base_image: str = 'cat-agent-runtime:latest'
    docker_network: str = ''
    consul_dns: str = ''
    # e.g. "http://{team}-{name}.localhost:8088" — printed by deploy; empty skips.
    public_url_template: str = ''
    # Traefik Host(`…`) value. Default keeps local-stack *.localhost behavior.
    ingress_host_template: str = '{team}-{name}.localhost'

    def is_local_registry(self) -> bool:
        return not self.registry or self.registry.strip().lower() in ('local', '')

    def registry_display(self) -> str:
        if self.is_local_registry():
            return 'local (images are not pushed; only this node can run them)'
        return self.registry.strip()

    @staticmethod
    def local_image_tag(team: str, name: str, content_tag: str) -> str:
        """Bare local image name: ``{team}/{name}:{content_tag}``.

        Nomad job IDs stay ``agent-{team}-{name}`` (Consul service name, Traefik
        router id, ``ls``/``rm`` resolution). Only the Docker image reference
        uses this slash form so remote registries get per-team repositories.
        """
        return f'{team}/{name}:{content_tag.strip()}'

    def image_ref(self, local_tag: str) -> str:
        """Job ``image`` value from a local tag (``{team}/{name}:{content}``).

        Local registry: bare tag. Remote: ``{registry}/{team}/{name}:{content}``.
        """
        local_tag = local_tag.strip()
        if self.is_local_registry():
            return local_tag
        return f'{self.registry.rstrip("/")}/{local_tag}'

    def registry_pull_vault_path(self) -> str:
        """KV v2 API path for the pull-only registry account."""
        base = (
            self.registry_credentials_path or 'secret/data/platform/registry'
        ).rstrip('/')
        return f'{base}/pull'

    def registry_push_vault_path(self) -> str:
        """KV v2 API path for the push (developer) registry account."""
        base = (
            self.registry_credentials_path or 'secret/data/platform/registry'
        ).rstrip('/')
        return f'{base}/push'

    def team_llm_vault_path(self, team: str) -> str:
        """Vault KV v2 API path for a team's LiteLLM virtual key.

        Example: ``secret/data/platform/llm/teams/demo`` (CLI:
        ``vault kv put secret/platform/llm/teams/demo api_key=...``).
        """
        base = (self.llm_credentials_path or 'secret/data/platform/llm').rstrip('/')
        return f'{base}/teams/{team}'

    def public_url(self, team: str, name: str) -> str:
        """Host-reachable agent URL from ``public_url_template``, or empty."""
        tmpl = (self.public_url_template or '').strip()
        if not tmpl:
            return ''
        return tmpl.format(team=team, name=name, agent=name)

    def ingress_host(self, team: str, name: str) -> str:
        """Hostname for Traefik ``Host(`…`)`` (Consul Catalog tags)."""
        tmpl = (self.ingress_host_template or '').strip() or '{team}-{name}.localhost'
        return tmpl.format(team=team, name=name, agent=name)

    def platform_host_is_remote(self) -> bool:
        """Nomad/Vault run on another host (operator laptop is not the stack)."""
        from urllib.parse import urlparse

        loopback = {'127.0.0.1', 'localhost', '::1'}
        for raw in (self.nomad_addr, self.vault_addr):
            host = (urlparse(raw or '').hostname or '').lower()
            if host and host not in loopback:
                return True
        return False


def _repo_root() -> Path:
    """``cat-agent`` package root (parent of ``cat_agent/``)."""
    return Path(__file__).resolve().parents[2]


def _sibling_stack_config() -> Path | None:
    """``../cat-agent-stack/cat-agent.config.toml`` next to this repo, if present."""
    candidate = _repo_root().parent / 'cat-agent-stack' / 'cat-agent.config.toml'
    return candidate if candidate.is_file() else None


def default_config_path() -> Path:
    """Resolve platform config — owned by cat-agent-stack, not the library repo.

    Order: ``CAT_AGENT_CONFIG`` → ``$CAT_AGENT_STACK_DIR/cat-agent.config.toml`` →
    cwd only if it *is* a stack root (``docker-compose.yml`` present) → sibling
    ``../cat-agent-stack/cat-agent.config.toml`` → ``~/.cat-agent/config.toml``.

    Agent example dirs under cat-agent are not searched for this file.
    """
    override = os.environ.get('CAT_AGENT_CONFIG', '').strip()
    if override:
        return Path(override).expanduser()
    stack_dir = os.environ.get('CAT_AGENT_STACK_DIR', '').strip()
    if stack_dir:
        stack_cfg = Path(stack_dir).expanduser().resolve() / 'cat-agent.config.toml'
        if stack_cfg.is_file():
            return stack_cfg
    cwd = Path.cwd()
    if (cwd / 'docker-compose.yml').is_file():
        cwd_cfg = cwd / 'cat-agent.config.toml'
        if cwd_cfg.is_file():
            return cwd_cfg
    sibling = _sibling_stack_config()
    if sibling is not None:
        return sibling
    return Path.home() / '.cat-agent' / 'config.toml'


def _load_toml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover — py3.10
        import tomli as tomllib  # type: ignore

    data = tomllib.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ConfigError(f'{path}: expected a TOML table')
    platform = data.get('platform', data)
    if not isinstance(platform, dict):
        raise ConfigError(f'{path}: [platform] must be a table')
    unknown = sorted(set(platform) - KNOWN_PLATFORM_KEYS)
    if unknown:
        raise ConfigError(
            f'{path}: unknown key(s) under [platform]: {", ".join(unknown)}'
        )
    # Reject unknown top-level tables other than platform
    top_unknown = sorted(k for k in data if k != 'platform')
    if top_unknown and 'platform' in data:
        raise ConfigError(
            f'{path}: unknown top-level table(s): {", ".join(top_unknown)} '
            '(only [platform] is allowed)'
        )
    return dict(platform)


def load_platform_config(
    *,
    path: str | Path | None = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> PlatformConfig:
    """Load config with precedence: overrides > env > file > defaults."""
    cfg_path = Path(path) if path else default_config_path()
    file_vals = _load_toml(cfg_path) if cfg_path.is_file() else {}

    env_map = {
        'nomad_addr': os.environ.get('NOMAD_ADDR'),
        'nomad_token': os.environ.get('NOMAD_TOKEN'),
        'namespace': os.environ.get('NOMAD_NAMESPACE'),
        'registry': os.environ.get('CAT_AGENT_REGISTRY'),
        'llm_gateway': os.environ.get('CAT_AGENT_LLM_GATEWAY'),
        'llm_credentials_path': os.environ.get('CAT_AGENT_LLM_CREDENTIALS_PATH'),
        'registry_credentials_path': os.environ.get(
            'CAT_AGENT_REGISTRY_CREDENTIALS_PATH'
        ),
        'otel_endpoint': os.environ.get('CAT_AGENT_OTEL_ENDPOINT'),
        'vault_addr': os.environ.get('VAULT_ADDR'),
        'base_image': os.environ.get('CAT_AGENT_BASE_IMAGE'),
        'docker_network': os.environ.get('CAT_AGENT_DOCKER_NETWORK'),
        'consul_dns': os.environ.get('CAT_AGENT_CONSUL_DNS'),
        'public_url_template': os.environ.get('CAT_AGENT_PUBLIC_URL_TEMPLATE'),
        'ingress_host_template': os.environ.get('CAT_AGENT_INGRESS_HOST_TEMPLATE'),
    }
    env_vals = {k: v for k, v in env_map.items() if v not in (None, '')}
    dcs = os.environ.get('NOMAD_DATACENTER') or os.environ.get('NOMAD_DATACENTERS')
    if dcs:
        env_vals['datacenters'] = [x.strip() for x in dcs.split(',') if x.strip()]

    merged: Dict[str, Any] = {}
    merged.update(file_vals)
    merged.update(env_vals)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v

    if 'datacenters' in merged and isinstance(merged['datacenters'], str):
        merged['datacenters'] = [
            x.strip() for x in merged['datacenters'].split(',') if x.strip()
        ]

    known = {f.name for f in fields(PlatformConfig)}
    unknown = sorted(set(merged) - known)
    if unknown:
        raise ConfigError(f'unknown config key(s): {", ".join(unknown)}')

    return PlatformConfig(**{k: v for k, v in merged.items() if k in known})
