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

"""Local Nomad stack lifecycle: compose + Vault seed from ``.env``.

Stack files live in the sibling **cat-agent-stack** repo (or any dir with
``docker-compose.yml``). Flow:

    export CAT_AGENT_STACK_DIR=/path/to/cat-agent-stack
    export CAT_AGENT_CONFIG=$CAT_AGENT_STACK_DIR/cat-agent.config.toml
    cp $CAT_AGENT_STACK_DIR/.env.example $CAT_AGENT_STACK_DIR/.env
    cat-agent stack bootstrap
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TextIO

from cat_agent.platform.config import PlatformConfig
from cat_agent.platform.gateway import (
    GatewayError,
    ensure_dev_vault_token,
    gateway_health_url,
    read_vault_kv_data,
    write_vault_kv_data,
    write_vault_policy,
)

TEAM_RE = re.compile(r'^[a-z][a-z0-9-]{0,31}$')
DEFAULT_HOST_GATEWAY = 'http://127.0.0.1:4000'
DEFAULT_MASTER_KEY = 'sk-local-litellm-master'


class StackError(Exception):
    """User-facing one-sentence stack failure."""


def _out(msg: str, file: TextIO = sys.stdout) -> None:
    print(msg, file=file)


def _sibling_stack_dir() -> Path | None:
    """``…/PycharmProjects/cat-agent-stack`` next to this cat-agent checkout."""
    # cat_agent/platform/stack.py → parents[2] = repo root
    repo_root = Path(__file__).resolve().parents[2]
    sibling = repo_root.parent / 'cat-agent-stack'
    if (sibling / 'docker-compose.yml').is_file():
        return sibling
    return None


def resolve_stack_dir(explicit: Optional[str] = None) -> Path:
    """Stack root containing ``docker-compose.yml``.

    Order: ``--dir`` → ``CAT_AGENT_STACK_DIR`` → cwd → sibling
    ``../cat-agent-stack``.
    """
    if explicit:
        root = Path(explicit).expanduser().resolve()
    elif os.environ.get('CAT_AGENT_STACK_DIR', '').strip():
        root = Path(os.environ['CAT_AGENT_STACK_DIR']).expanduser().resolve()
    else:
        cwd = Path.cwd()
        if (cwd / 'docker-compose.yml').is_file():
            root = cwd.resolve()
        else:
            sibling = _sibling_stack_dir()
            if sibling is None:
                raise StackError(
                    'no stack directory found; set CAT_AGENT_STACK_DIR to the '
                    'cat-agent-stack checkout (or pass --dir / run from that repo)'
                )
            root = sibling
    compose = root / 'docker-compose.yml'
    if not compose.is_file():
        raise StackError(
            f'no docker-compose.yml under {root}; set CAT_AGENT_STACK_DIR to the '
            'cat-agent-stack checkout (or pass --dir)'
        )
    return root


def load_stack_env(stack_dir: Path) -> Path | None:
    """Load ``{stack_dir}/.env`` without overriding existing process env."""
    path = stack_dir / '.env'
    if not path.is_file():
        return None
    from dotenv import load_dotenv

    load_dotenv(path, override=False)
    return path


def ensure_host_data_dirs(stack_dir: Path) -> Dict[str, str]:
    """Export absolute HOST_* bind paths (Docker Desktop sharing)."""
    nomad = os.environ.get('HOST_NOMAD_DATA', '').strip() or str(
        stack_dir / '.nomad-data'
    )
    zot = os.environ.get('HOST_ZOT_DATA', '').strip() or str(stack_dir / '.zot-data')
    Path(nomad).mkdir(parents=True, exist_ok=True)
    Path(zot).mkdir(parents=True, exist_ok=True)
    os.environ['HOST_NOMAD_DATA'] = nomad
    os.environ['HOST_ZOT_DATA'] = zot
    return {'HOST_NOMAD_DATA': nomad, 'HOST_ZOT_DATA': zot}


def compose_argv(stack_dir: Path, profiles: Sequence[str] = ()) -> List[str]:
    cmd = ['docker', 'compose', '-f', str(stack_dir / 'docker-compose.yml')]
    for profile in profiles:
        p = (profile or '').strip()
        if p:
            cmd.extend(['--profile', p])
    return cmd


def run_compose(
    stack_dir: Path,
    args: Sequence[str],
    *,
    profiles: Sequence[str] = (),
    check: bool = True,
) -> int:
    ensure_host_data_dirs(stack_dir)
    cmd = compose_argv(stack_dir, profiles) + list(args)
    _out(f"==> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(stack_dir))
    if check and proc.returncode != 0:
        raise StackError(f'docker compose failed (exit {proc.returncode})')
    return int(proc.returncode)


def _ensure_openai_compat_base(base: str) -> str:
    """Append ``/v1`` only when missing (Ollama Cloud already uses ``…/v1``)."""
    b = (base or '').rstrip('/')
    if not b:
        return b
    if b.endswith('/v1'):
        return b
    return f'{b}/v1'


def resolve_llm_seed(env: Mapping[str, str] | None = None) -> Dict[str, str]:
    """Map process/.env provider vars to Vault ``secret/platform/llm`` fields.

    Defaults to Mac-host Ollama (``host.docker.internal:11434``). There are no
    stub LLM containers — set ``OPENAI_API_KEY`` for a separate OpenAI upstream,
    or leave it unset to reuse Ollama's OpenAI-compatible endpoint.
    """
    e = env if env is not None else os.environ
    master = (e.get('LITELLM_MASTER_KEY') or DEFAULT_MASTER_KEY).strip()

    host_ollama = 'http://host.docker.internal:11434'

    ollama_key = (e.get('OLLAMA_API_KEY') or 'local-ollama').strip()
    ollama_base = (
        e.get('OLLAMA_API_BASE') or e.get('OLLAMA_BASE_URL') or host_ollama
    ).strip()

    openai_key = (e.get('OPENAI_API_KEY') or '').strip()
    if openai_key:
        openai_base = (
            e.get('OPENAI_API_BASE')
            or e.get('OPENAI_BASE_URL')
            or 'https://api.openai.com/v1'
        ).strip()
    else:
        # Reuse Ollama OpenAI-compatible API when no separate OpenAI key.
        openai_key = ollama_key
        openai_base = (
            e.get('OPENAI_API_BASE')
            or e.get('OPENAI_BASE_URL')
            or _ensure_openai_compat_base(ollama_base)
        ).strip()

    return {
        'LITELLM_MASTER_KEY': master,
        'OPENAI_API_KEY': openai_key,
        'OPENAI_API_BASE': openai_base,
        'OLLAMA_API_KEY': ollama_key,
        'OLLAMA_API_BASE': ollama_base,
    }


def _ensure_dev_vault_token() -> None:
    ensure_dev_vault_token()


def wait_vault(vault_addr: str, *, timeout_s: float = 60.0) -> None:
    addr = vault_addr.rstrip('/')
    deadline = time.monotonic() + timeout_s
    last = ''
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f'{addr}/v1/sys/health')
            with urllib.request.urlopen(req, timeout=3) as resp:
                if 200 <= resp.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(1)
    raise StackError(f'Vault not ready at {vault_addr}' + (f': {last}' if last else ''))


def wait_gateway(gateway_root: str, *, timeout_s: float = 180.0) -> None:
    root = gateway_root.rstrip('/')
    if root.endswith('/v1'):
        url = gateway_health_url(root)
    else:
        url = f'{root}/health/liveliness'
    deadline = time.monotonic() + timeout_s
    last = ''
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(2)
    raise StackError(
        f'LiteLLM not healthy at {url}'
        + (f': {last}' if last else '')
        + '; run: cat-agent stack seed'
    )


def seed_llm_vault(cfg: PlatformConfig, *, env: Mapping[str, str] | None = None) -> Dict[str, str]:
    """Write provider + master key into Vault. Never prints secret values."""
    _ensure_dev_vault_token()
    fields = resolve_llm_seed(env)
    wait_vault(cfg.vault_addr)
    write_vault_kv_data(
        cfg.vault_addr,
        cfg.llm_credentials_path,
        fields,
    )
    _out(
        '==> seeded Vault LLM credentials '
        f'(OPENAI_API_BASE={fields["OPENAI_API_BASE"]} '
        f'OLLAMA_API_BASE={fields["OLLAMA_API_BASE"]})'
    )
    return fields


def _tpm_rpm(
    *,
    max_tokens_per_day: int,
    tpm_limit: Optional[int],
    rpm_limit: Optional[int],
) -> tuple[int, int]:
    tpm = tpm_limit if tpm_limit is not None else max(1, max_tokens_per_day // (24 * 60))
    rpm = rpm_limit if rpm_limit is not None else max(1, tpm // 500)
    return int(tpm), int(rpm)


def _http_json_post(
    url: str,
    payload: dict,
    *,
    headers: Optional[dict] = None,
    timeout: float = 30.0,
) -> dict:
    data = json.dumps(payload).encode()
    hdrs = {'Content-Type': 'application/json', **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        detail = ''
        try:
            detail = exc.read().decode('utf-8', errors='replace')[:300]
        except Exception:  # noqa: BLE001
            pass
        raise StackError(
            f'HTTP {exc.code} POST {url}' + (f': {detail}' if detail else '')
        ) from exc
    except urllib.error.URLError as exc:
        raise StackError(f'unreachable {url}: {exc.reason}') from exc
    try:
        return json.loads(raw or '{}')
    except json.JSONDecodeError as exc:
        raise StackError(f'non-JSON from {url} (len={len(raw)})') from exc


def seed_team_key(
    cfg: PlatformConfig,
    team: str = 'demo',
    *,
    gateway_url: Optional[str] = None,
    max_tokens_per_day: int = 2_000_000,
    tpm_limit: Optional[int] = None,
    rpm_limit: Optional[int] = None,
    stack_dir: Optional[Path] = None,
) -> None:
    """Mint a LiteLLM virtual key and store only ``api_key`` in Vault."""
    _ensure_dev_vault_token()
    if not TEAM_RE.match(team):
        raise StackError(
            f"invalid team name '{team}': use lowercase [a-z0-9-] (max 32)"
        )
    tpm, rpm = _tpm_rpm(
        max_tokens_per_day=max_tokens_per_day,
        tpm_limit=tpm_limit,
        rpm_limit=rpm_limit,
    )
    wait_vault(cfg.vault_addr)
    try:
        read_vault_kv_data(cfg.vault_addr, cfg.llm_credentials_path)
    except GatewayError:
        seed_llm_vault(cfg)

    master = read_vault_kv_data(
        cfg.vault_addr, cfg.llm_credentials_path
    ).get('LITELLM_MASTER_KEY')
    if not master:
        raise StackError('Vault LLM secret missing LITELLM_MASTER_KEY')

    root = (gateway_url or os.environ.get('CAT_AGENT_STACK_GATEWAY') or DEFAULT_HOST_GATEWAY).rstrip(
        '/'
    )
    try:
        wait_gateway(root, timeout_s=30.0)
    except StackError:
        if stack_dir is not None:
            _out('==> LiteLLM not ready; restarting litellm to reload Vault')
            run_compose(stack_dir, ['restart', 'litellm'])
        wait_gateway(root, timeout_s=180.0)

    alias = f'cat-agent-{team}'
    auth = {'Authorization': f'Bearer {master}'}
    # Drop stale alias (Postgres persists across Vault -dev recreates).
    try:
        _http_json_post(
            f'{root}/key/delete',
            {'key_aliases': [alias]},
            headers=auth,
        )
    except StackError:
        pass

    _out(f'==> mint virtual key team={team} tpm_limit={tpm} rpm_limit={rpm}')
    body = _http_json_post(
        f'{root}/key/generate',
        {
            'key_alias': alias,
            'models': ['*'],
            'tpm_limit': tpm,
            'rpm_limit': rpm,
            'metadata': {'team': team, 'source': 'cat-agent-stack-seed'},
        },
        headers=auth,
    )
    api_key = str(body.get('key') or '').strip()
    if not api_key:
        err = body.get('error') or body.get('detail') or body.get('message') or ''
        raise StackError(
            f'key/generate missing key (keys={sorted(body.keys())} error={err!r})'
        )

    team_path = cfg.team_llm_vault_path(team)
    write_vault_kv_data(
        cfg.vault_addr,
        team_path,
        {'api_key': api_key},
    )
    policy_name = f'cat-agent-llm-{team}'
    # CLI path omits /data/; policy uses KV v2 API path.
    policy = (
        f'path "secret/data/platform/llm/teams/{team}" {{\n'
        f'  capabilities = ["read"]\n'
        f'}}\n'
    )
    write_vault_policy(cfg.vault_addr, policy_name, policy)
    stored = read_vault_kv_data(cfg.vault_addr, team_path)
    key_len = len(str(stored.get('api_key') or ''))
    if key_len < 8:
        raise StackError(f'Vault api_key for {team} looks empty (len={key_len})')
    _out(
        f'==> done team={team} vault={team_path} policy={policy_name} '
        f'api_key_len={key_len} tpm={tpm} rpm={rpm}'
    )


def seed_registry_vault(cfg: PlatformConfig, *, env: Mapping[str, str] | None = None) -> None:
    """Write Zot push/pull creds + pull-only policy. Never prints values."""
    _ensure_dev_vault_token()
    e = env if env is not None else os.environ
    push_user = (e.get('ZOT_PUSH_USER') or 'zot-push').strip()
    push_pass = (e.get('ZOT_PUSH_PASSWORD') or 'cat-agent-zot-push-dev').strip()
    pull_user = (e.get('ZOT_PULL_USER') or 'zot-pull').strip()
    pull_pass = (e.get('ZOT_PULL_PASSWORD') or 'cat-agent-zot-pull-dev').strip()

    wait_vault(cfg.vault_addr)
    push_path = cfg.registry_push_vault_path()
    pull_path = cfg.registry_pull_vault_path()
    write_vault_kv_data(
        cfg.vault_addr,
        push_path,
        {'username': push_user, 'password': push_pass},
    )
    write_vault_kv_data(
        cfg.vault_addr,
        pull_path,
        {'username': pull_user, 'password': pull_pass},
    )
    policy = (
        'path "secret/data/platform/registry/pull" {\n'
        '  capabilities = ["read"]\n'
        '}\n'
    )
    write_vault_policy(
        cfg.vault_addr, 'cat-agent-registry-pull', policy
    )
    push = read_vault_kv_data(cfg.vault_addr, push_path)
    pull = read_vault_kv_data(cfg.vault_addr, pull_path)
    if len(str(push.get('username') or '')) < 2 or len(str(push.get('password') or '')) < 8:
        raise StackError('push registry secret looks empty')
    if len(str(pull.get('username') or '')) < 2 or len(str(pull.get('password') or '')) < 8:
        raise StackError('pull registry secret looks empty')
    _out(f'==> seeded registry Vault secrets push={push_path} pull={pull_path}')


def cmd_stack_up(args: Any) -> int:
    stack_dir = resolve_stack_dir(getattr(args, 'dir', None))
    load_stack_env(stack_dir)
    profiles = list(getattr(args, 'profile', None) or [])
    compose_args: List[str] = ['up']
    if getattr(args, 'build', False):
        compose_args.append('--build')
    if getattr(args, 'detach', False):
        compose_args.append('-d')
    extra = list(getattr(args, 'compose_args', None) or [])
    run_compose(stack_dir, compose_args + extra, profiles=profiles)
    if getattr(args, 'seed', False):
        return cmd_stack_seed(args)
    return 0


def cmd_stack_down(args: Any) -> int:
    stack_dir = resolve_stack_dir(getattr(args, 'dir', None))
    load_stack_env(stack_dir)
    profiles = list(getattr(args, 'profile', None) or [])
    return run_compose(stack_dir, ['down'] + list(getattr(args, 'compose_args', None) or []), profiles=profiles)


def cmd_stack_compose(args: Any) -> int:
    stack_dir = resolve_stack_dir(getattr(args, 'dir', None))
    load_stack_env(stack_dir)
    profiles = list(getattr(args, 'profile', None) or [])
    passthrough = list(getattr(args, 'compose_args', None) or [])
    if not passthrough:
        raise StackError('usage: cat-agent stack compose -- <docker compose args>')
    return run_compose(stack_dir, passthrough, profiles=profiles, check=False)


def cmd_stack_seed(args: Any) -> int:
    stack_dir = resolve_stack_dir(getattr(args, 'dir', None))
    env_path = load_stack_env(stack_dir)
    if env_path:
        _out(f'==> loaded {env_path}')
    else:
        _out(f'==> no {stack_dir / ".env"} (using process env / stub defaults)')

    from cat_agent.platform.config import load_platform_config

    overrides: Dict[str, Any] = {}
    if getattr(args, 'nomad_addr', None):
        overrides['nomad_addr'] = args.nomad_addr
    cfg_path = getattr(args, 'config', None)
    if not cfg_path:
        candidate = stack_dir / 'cat-agent.config.toml'
        if candidate.is_file():
            cfg_path = str(candidate)
    cfg = load_platform_config(path=cfg_path, overrides=overrides or None)

    seed_llm_vault(cfg)
    team = getattr(args, 'team', None) or os.environ.get('TEAM') or 'demo'
    max_tokens = int(
        getattr(args, 'max_tokens_per_day', None)
        or os.environ.get('MAX_TOKENS_PER_DAY')
        or 2_000_000
    )
    tpm = getattr(args, 'tpm_limit', None)
    if tpm is None and os.environ.get('TPM_LIMIT'):
        tpm = int(os.environ['TPM_LIMIT'])
    rpm = getattr(args, 'rpm_limit', None)
    if rpm is None and os.environ.get('RPM_LIMIT'):
        rpm = int(os.environ['RPM_LIMIT'])
    seed_team_key(
        cfg,
        team,
        max_tokens_per_day=max_tokens,
        tpm_limit=tpm,
        rpm_limit=rpm,
        stack_dir=stack_dir,
    )
    if getattr(args, 'registry', False) or 'registry' in (
        getattr(args, 'profile', None) or []
    ):
        seed_registry_vault(cfg)
    return 0


def cmd_stack_bootstrap(args: Any) -> int:
    """``up --build -d`` + seed from ``.env`` (happy path)."""
    args.build = True
    args.detach = True
    args.seed = True
    return cmd_stack_up(args)
