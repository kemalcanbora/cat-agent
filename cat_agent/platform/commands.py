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

"""CLI implementations for ``cat-agent deploy`` and related commands."""

from __future__ import annotations

import getpass
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO

from cat_agent.platform.builder import BuildError, build_agent_image, build_base_image
from cat_agent.platform.config import (
    ConfigError,
    PlatformConfig,
    default_config_path,
    load_platform_config,
)
from cat_agent.platform.gateway import (
    GATEWAY_HOST,
    GatewayError,
    ensure_alias_or_raise,
    fetch_aliases_for_config,
    read_vault_kv_data,
    resolve_gateway_via_consul_dns,
    vault_team_key_exists,
)
from cat_agent.platform.manifest import ManifestError, load_manifest
from cat_agent.platform.nomad import (
    NomadClient,
    NomadError,
    NomadNotFound,
    NomadRejected,
    NomadUnreachable,
)
from cat_agent.platform.registry_check import (
    RegistryNameError,
    validate_manifest_registry_names,
)
from cat_agent.platform.registry_probe import (
    RegistryError,
    probe_registry_auth,
    probe_registry_reachability,
    vault_registry_creds_exist,
)
from cat_agent.platform.render import manifest_sha, render_all


class CommandError(Exception):
    """User-facing one-sentence failure."""


def _out(msg: str, file: TextIO = sys.stdout) -> None:
    print(msg, file=file)


def _known_tools() -> set[str]:
    from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, TOOL_REGISTRY

    return set(TOOL_REGISTRY) | set(OPTIONAL_TOOL_REGISTRY)


def _import_entrypoint_module(source: Path, entrypoint: str) -> None:
    """Import the agent module so ``@tool`` registrations are visible to validate_tools."""
    import importlib
    import sys

    module_name = entrypoint.split(':', 1)[0]
    src = str(source.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    importlib.import_module(module_name)


def _resolve_entrypoint_registry(source: Path, entrypoint: str):
    """Import ``module:attr`` and return the :class:`AgentRegistry` it builds."""
    import sys

    from cat_agent.serve.factory import load_registry

    src = str(source.resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    return load_registry(entrypoint)


def _load_cfg(args: Any) -> PlatformConfig:
    from cat_agent.platform.config import default_config_path
    from cat_agent.platform.gateway import ensure_dev_vault_token

    overrides: Dict[str, Any] = {}
    if getattr(args, 'nomad_addr', None):
        overrides['nomad_addr'] = args.nomad_addr
    if getattr(args, 'registry', None):
        overrides['registry'] = args.registry
    path = getattr(args, 'config', None)
    cfg_path = Path(path) if path else default_config_path()
    # Load sibling .env next to config.toml (e.g. cat-agent-stack/.env).
    env_file = cfg_path.parent / '.env'
    if env_file.is_file():
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
    # Local Vault -dev; .env may set VAULT_TOKEN=root, else default.
    ensure_dev_vault_token()
    return load_platform_config(path=path, overrides=overrides or None)


def _client(cfg: PlatformConfig) -> NomadClient:
    return NomadClient(cfg)


def _require_mac_docker_network(cfg: PlatformConfig) -> None:
    """Fail closed on macOS when bridge CNI would be used (Docker Desktop netns)."""
    if sys.platform != 'darwin':
        return
    if cfg.platform_host_is_remote():
        _out('remote platform: docker_network not required on operator Mac')
        return
    network = (cfg.docker_network or '').strip()
    if network:
        _out(f'docker_network: {network}')
        return
    raise CommandError(
        'platform.docker_network is unset — on macOS/Docker Desktop Nomad bridge '
        'CNI fails with: unknown FS magic on "/var/run/docker/netns/…". '
        'Use cat-agent-stack config: export CAT_AGENT_CONFIG=…/cat-agent-stack/'
        'cat-agent.config.toml  (or keep the stack as a sibling of cat-agent so '
        'it is auto-discovered), then: cat-agent doctor'
    )


def cmd_doctor(args: Any) -> int:
    cfg = _load_cfg(args)
    failed = False
    cfg_path = Path(getattr(args, 'config', None) or '') if getattr(args, 'config', None) else default_config_path()
    _out(f'config: {cfg_path}')
    _out(f'registry: {cfg.registry_display()}')
    _out(f'nomad_addr: {cfg.nomad_addr}')
    _out(f'namespace: {cfg.namespace}')
    _out(f'llm_gateway: {cfg.llm_gateway}')
    _out(f'llm_credentials_path: {cfg.llm_credentials_path}')
    _out(f'vault_addr: {cfg.vault_addr}')
    if cfg.docker_network:
        _out(f'docker_network: {cfg.docker_network}')
    elif sys.platform == 'darwin' and not cfg.platform_host_is_remote():
        _out(
            'docker_network: NOT SET — macOS deploy will fail (unknown FS magic / '
            'bridge CNI); set CAT_AGENT_CONFIG to cat-agent-stack/cat-agent.config.toml'
        )
        failed = True
    elif cfg.platform_host_is_remote():
        _out('platform: remote (operator checks use vault/nomad host, not local docker_network)')
    if cfg.consul_dns:
        _out(f'consul_dns: {cfg.consul_dns}')
    _out(f'ingress_host_template: {cfg.ingress_host_template}')
    if cfg.public_url_template:
        _out(f'public_url_template: {cfg.public_url_template}')

    if shutil.which('docker') is None:
        _out('docker: NOT FOUND on PATH')
        failed = True
    else:
        _out('docker: ok')

    try:
        client = _client(cfg)
        leader = client.status_leader()
        _out(f'nomad: reachable (leader={leader})')
        nodes = client.nodes()
        _out(f'nomad nodes: {len(nodes)}')
        if not nodes:
            _out(
                'WARNING: no client nodes registered — jobs will be accepted but never run'
            )
            failed = True
        else:
            healthy_docker = 0
            for stub in nodes:
                nid = stub.get('ID')
                if not nid:
                    continue
                detail = client.node(nid)
                drivers = detail.get('Drivers') or {}
                docker = drivers.get('docker') or {}
                if docker.get('Healthy'):
                    healthy_docker += 1
            _out(f'docker driver healthy on {healthy_docker}/{len(nodes)} node(s)')
            if healthy_docker == 0:
                failed = True
    except NomadUnreachable as exc:
        _out(str(exc))
        failed = True
    except NomadError as exc:
        _out(str(exc))
        failed = True

    # Consul DNS must actually resolve the gateway hostname (stale IPs after
    # Docker network recreate are a common silent failure).
    if cfg.docker_network and cfg.consul_dns and not cfg.platform_host_is_remote():
        try:
            ip = resolve_gateway_via_consul_dns(
                cfg.consul_dns,
                hostname=GATEWAY_HOST,
                docker_network=cfg.docker_network,
            )
            _out(f'consul_dns: {GATEWAY_HOST} → {ip}')
        except GatewayError as exc:
            _out(str(exc))
            failed = True
    elif cfg.docker_network and not cfg.consul_dns and not cfg.platform_host_is_remote():
        _out(
            'consul_dns: not set (docker_network allocs will not resolve '
            f'{GATEWAY_HOST})'
        )
        failed = True
    elif cfg.consul_dns and cfg.platform_host_is_remote():
        _out(
            f'consul_dns: skipped on remote operator '
            f'(allocs on stack host use {cfg.consul_dns})'
        )

    # Gateway reachability + advertised aliases.
    try:
        aliases = fetch_aliases_for_config(cfg)
        _out(f'llm gateway: ok aliases={",".join(aliases)}')
    except GatewayError as exc:
        _out(str(exc))
        failed = True

    # Team key path: existence only (demo team by default for local stack).
    team = getattr(args, 'team', None) or 'demo'
    try:
        vault_team_key_exists(cfg, team)
        _out(f'vault team key: ok path={cfg.team_llm_vault_path(team)}')
    except GatewayError as exc:
        _out(str(exc))
        failed = True

    if cfg.is_local_registry():
        _out(
            'registry mode: local (images are not pushed; only this node can run them)'
        )
    else:
        _out(f'registry mode: remote ({cfg.registry.strip()})')
        _out(f'registry_credentials_path: {cfg.registry_credentials_path}')
        try:
            _out(probe_registry_reachability(cfg))
        except RegistryError as exc:
            _out(str(exc))
            failed = True
        try:
            vault_registry_creds_exist(cfg, 'pull')
            _out(f'vault registry pull: ok path={cfg.registry_pull_vault_path()}')
            vault_registry_creds_exist(cfg, 'push')
            _out(f'vault registry push: ok path={cfg.registry_push_vault_path()}')
        except RegistryError as exc:
            _out(str(exc))
            failed = True
        else:
            try:
                data = read_vault_kv_data(cfg.vault_addr, cfg.registry_pull_vault_path())
                _out(
                    probe_registry_auth(
                        cfg,
                        username=str(data['username']),
                        password=str(data['password']),
                    )
                )
            except (RegistryError, GatewayError, KeyError) as exc:
                _out(str(exc))
                failed = True

    return 1 if failed else 0


def cmd_ls(args: Any) -> int:
    cfg = _load_cfg(args)
    client = _client(cfg)
    jobs = client.list_agents(team=getattr(args, 'team', None))
    rows = []
    for job in jobs:
        meta = job.get('Meta') or {}
        rows.append(
            {
                'id': job.get('ID'),
                'team': meta.get('team'),
                'agent': meta.get('agent'),
                'trigger': meta.get('trigger'),
                'jobs_mode': meta.get('jobs_mode'),
                'status': job.get('Status'),
                'image_tag': meta.get('image_tag'),
                'deployed_by': meta.get('deployed_by'),
            }
        )
    if getattr(args, 'json', False):
        _out(json.dumps(rows, indent=2))
    else:
        if not rows:
            _out('(no cat-agent jobs)')
        for r in rows:
            _out(
                f"{r['team']}/{r['agent']}\t{r['trigger']}\t{r['jobs_mode']}\t"
                f"{r['status']}\t{r['image_tag']}\t{r['deployed_by']}"
            )
    return 0


def _parse_agent_ref(name: str, team: Optional[str]) -> tuple[str, Optional[str]]:
    """Accept ``calculator``, ``calculator --team demo``, or ``demo/calculator``."""
    raw = (name or '').strip()
    if '/' in raw:
        left, right = raw.split('/', 1)
        left, right = left.strip(), right.strip()
        if not left or not right or '/' in right:
            raise CommandError(
                f'invalid agent ref {name!r}; use name or team/name'
            )
        if team and team != left:
            raise CommandError(
                f'team mismatch: ref has {left!r} but --team is {team!r}'
            )
        return right, left
    return raw, team


def _resolve_jobs_by_name(
    client: NomadClient, name: str, team: Optional[str]
) -> List[Dict[str, Any]]:
    agent, team = _parse_agent_ref(name, team)
    matches = []
    for job in client.list_agents(team=team):
        meta = job.get('Meta') or {}
        if meta.get('agent') == agent:
            matches.append(job)
    if not matches:
        raise CommandError(f'no cat-agent job named {name!r}')
    if len(matches) > 1 and not team:
        teams = sorted({(j.get('Meta') or {}).get('team') for j in matches})
        raise CommandError(
            f'name {agent!r} is ambiguous across teams {teams}; pass --team'
        )
    return matches


def cmd_status(args: Any) -> int:
    cfg = _load_cfg(args)
    client = _client(cfg)
    jobs = _resolve_jobs_by_name(client, args.name, getattr(args, 'team', None))
    for job in jobs:
        jid = job['ID']
        meta = job.get('Meta') or {}
        _out(f"job {jid} status={job.get('Status')}")
        _out(f"  meta: team={meta.get('team')} agent={meta.get('agent')} "
             f"sha={meta.get('manifest_sha')} tag={meta.get('image_tag')}")
        for alloc in client.allocations(jid):
            _out(
                f"  alloc {alloc.get('ID', '')[:8]} "
                f"client={alloc.get('ClientStatus')} desired={alloc.get('DesiredStatus')}"
            )
        local_dir = getattr(args, 'dir', None)
        if local_dir:
            path = Path(local_dir) / 'agent.yaml'
            if path.is_file():
                raw = path.read_text(encoding='utf-8')
                local_sha = manifest_sha(raw)
                if local_sha != meta.get('manifest_sha'):
                    _out(
                        f'WARNING: local agent.yaml sha {local_sha} differs from '
                        f"deployed {meta.get('manifest_sha')}"
                    )
    return 0


def cmd_logs(args: Any) -> int:
    cfg = _load_cfg(args)
    client = _client(cfg)
    jobs = _resolve_jobs_by_name(client, args.name, getattr(args, 'team', None))
    job = jobs[0]
    allocs = client.allocations(job['ID'])
    if not allocs:
        raise CommandError(f"no allocations for {job['ID']}")
    alloc = sorted(allocs, key=lambda a: a.get('CreateIndex') or 0)[-1]
    task = 'agent'
    tg = (alloc.get('TaskStates') or {})
    if tg:
        task = next(iter(tg))
    text = client.logs(alloc['ID'], task, stderr=bool(getattr(args, 'stderr', False)))
    _out(text)
    return 0


def _ensure_registry_vault(cfg: PlatformConfig) -> None:
    """Seed Zot push/pull Vault secrets when missing (remote deploy from Mac)."""
    if cfg.is_local_registry():
        return
    try:
        vault_registry_creds_exist(cfg, 'pull')
        vault_registry_creds_exist(cfg, 'push')
        return
    except RegistryError:
        pass
    from cat_agent.platform.stack import StackError, seed_registry_vault

    _out('vault registry creds missing; seeding Zot push/pull secrets')
    try:
        seed_registry_vault(cfg)
    except StackError as exc:
        raise CommandError(
            f'registry Vault secrets missing and auto-seed failed: {exc}. '
            'Set VAULT_TOKEN=root on the stack host and run: cat-agent stack seed --registry'
        ) from exc


def _ensure_team_llm_key(cfg: PlatformConfig, team: str) -> None:
    """Mint ``secret/.../llm/teams/{team}`` if missing (no stack checkout required)."""
    try:
        vault_team_key_exists(cfg, team)
        return
    except GatewayError:
        pass
    from cat_agent.platform.stack import StackError, operator_llm_gateway, seed_team_key

    gateway = operator_llm_gateway(cfg)
    _out(f'vault team key missing for {team!r}; minting via {gateway}')
    try:
        seed_team_key(cfg, team, gateway_url=gateway)
    except StackError as exc:
        raise CommandError(
            f'team {team!r} has no Vault LLM key and auto-seed failed: {exc}. '
            'Set VAULT_TOKEN=root; LiteLLM must be reachable on the stack host :4000.'
        ) from exc


def cmd_deploy(args: Any) -> int:
    cfg = _load_cfg(args)
    _require_mac_docker_network(cfg)
    no_push = bool(getattr(args, 'no_push', False)) or cfg.is_local_registry()
    if getattr(args, 'no_push', False):
        # force local rendering even if registry is remote
        cfg.registry = 'local'
        no_push = True

    source = Path(getattr(args, 'dir', '.') or '.').resolve()
    manifest_path = source / 'agent.yaml'
    try:
        manifest = load_manifest(manifest_path)
        manifest.validate_timeout_ceiling(cfg.max_timeout_seconds)
        # Import factory module first so example ``@tool`` names are in TOOL_REGISTRY.
        try:
            _import_entrypoint_module(source, manifest.runtime.entrypoint)
        except Exception as exc:  # noqa: BLE001 — surface as deploy error
            raise CommandError(
                f'failed to import entrypoint {manifest.runtime.entrypoint!r}: {exc}'
            ) from exc
        manifest.validate_tools(_known_tools())
        try:
            registry = _resolve_entrypoint_registry(
                source, manifest.runtime.entrypoint
            )
            validate_manifest_registry_names(manifest.name, registry.names())
        except RegistryNameError as exc:
            raise CommandError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — factory may need LLM env
            raise CommandError(
                f'failed to resolve entrypoint registry '
                f'{manifest.runtime.entrypoint!r} for name check: {exc}'
            ) from exc
    except ManifestError as exc:
        raise CommandError(str(exc)) from exc

    _ensure_team_llm_key(cfg, manifest.team)
    if not no_push and not cfg.is_local_registry():
        _ensure_registry_vault(cfg)

    # Fail closed: model.alias must exist on the live LLM backend (Ollama list).
    # Escape hatch: --skip-alias-check (offline / CI without gateway).
    if not getattr(args, 'skip_alias_check', False):
        try:
            models = ensure_alias_or_raise(cfg, manifest.model.alias)
            _out(
                f'model check: {manifest.model.alias!r} ok '
                f'({len(models)} backend model(s))'
            )
        except GatewayError as exc:
            raise CommandError(str(exc)) from exc
    else:
        _out('model check: skipped (--skip-alias-check)')

    raw_yaml = manifest_path.read_text(encoding='utf-8')
    sha = manifest_sha(raw_yaml)
    image_tag = getattr(args, 'image_tag', None)

    if image_tag is None:
        try:
            image_tag = build_agent_image(
                manifest,
                cfg,
                source,
                push=not no_push,
            )
        except BuildError as exc:
            raise CommandError(str(exc)) from exc
    elif not no_push and not cfg.is_local_registry():
        # tag provided by CI; still push if remote registry
        from cat_agent.platform.builder import docker_login_and_push

        try:
            docker_login_and_push(cfg, image_tag)
        except BuildError as exc:
            raise CommandError(str(exc)) from exc

    jobs = render_all(
        manifest,
        cfg,
        image_tag=image_tag,
        manifest_sha_value=sha,
        deployed_by=os.environ.get('USER') or getpass.getuser(),
    )

    if getattr(args, 'dry_run', False):
        for jid, hcl in jobs:
            _out(f'# --- job {jid} ---')
            _out(hcl)
        return 0

    client = _client(cfg)
    for jid, hcl in jobs:
        try:
            client.submit_hcl(hcl)
            _out(f'submitted {jid}')
            if manifest.trigger.type == 'http' and not jid.endswith('-task'):
                for line in client.watch_deployment(jid):
                    _out(line)
        except NomadError as exc:
            raise CommandError(str(exc)) from exc

    _out(f'deployed {manifest.team}/{manifest.name} tag={image_tag}')
    if manifest.trigger.type == 'http':
        public = cfg.public_url(manifest.team, manifest.name)
        if public:
            _out(f'url: {public}')
        else:
            _out(
                f'url: (set platform.public_url_template to print a host URL; '
                f'Consul: {manifest.job_id()}.service.consul)'
            )
    if cfg.is_local_registry():
        _out('registry: local (images are not pushed; only this node can run them)')
    return 0


def cmd_rm(args: Any) -> int:
    cfg = _load_cfg(args)
    client = _client(cfg)
    jobs = _resolve_jobs_by_name(client, args.name, getattr(args, 'team', None))
    # Prefer service job(s); detect dispatch mode from meta
    service_jobs = []
    task_jobs = []
    for job in jobs:
        meta = job.get('Meta') or {}
        jid = job['ID']
        if jid.endswith('-task') or meta.get('jobs_mode') == 'dispatch' and 'task' in jid:
            task_jobs.append(job)
        else:
            service_jobs.append(job)

    # Also find sibling task job by id convention
    for job in list(service_jobs):
        meta = job.get('Meta') or {}
        if meta.get('jobs_mode') == 'dispatch':
            task_id = f"{job['ID']}-task"
            try:
                task_jobs.append(client.get_job(task_id))
            except NomadNotFound:
                pass

    force = bool(getattr(args, 'force', False))
    for task in task_jobs:
        running = [
            a
            for a in client.allocations(task['ID'])
            if a.get('ClientStatus') == 'running'
        ]
        if running and not force:
            raise CommandError(
                f"{len(running)} dispatched task(s) still running for {task['ID']}; "
                're-run with --force or wait for them to finish'
            )

    if not getattr(args, 'yes', False):
        raise CommandError('refusing to delete without --yes')

    # Order: service first, then task job
    for job in service_jobs:
        try:
            client.stop(job['ID'], purge=True)
            _out(f"stopped {job['ID']}")
        except NomadRejected as exc:
            raise CommandError(str(exc)) from exc
    for job in task_jobs:
        try:
            client.stop(job['ID'], purge=True)
            _out(f"stopped {job['ID']}")
        except NomadRejected as exc:
            raise CommandError(str(exc)) from exc
    return 0


def cmd_rollback(args: Any) -> int:
    cfg = _load_cfg(args)
    client = _client(cfg)
    jobs = _resolve_jobs_by_name(client, args.name, getattr(args, 'team', None))
    job = jobs[0]
    versions = client.job_versions(job['ID'])
    if not versions:
        raise CommandError(f"no versions stored for {job['ID']}")
    target = getattr(args, 'to', None)
    if target is None:
        # previous version
        if len(versions) < 2:
            raise CommandError('no previous version to roll back to')
        chosen = versions[1]
    else:
        chosen = next((v for v in versions if v.get('Version') == int(target)), None)
        if chosen is None:
            raise CommandError(f'version {target} not found')
    client.submit(chosen)
    _out(f"rolled back {job['ID']} to version {chosen.get('Version')}")
    return 0


def cmd_build_base(args: Any) -> int:
    cfg = _load_cfg(args)
    root = Path(__file__).resolve().parents[2]
    dockerfile = root / 'runtime' / 'Dockerfile.base'
    if not dockerfile.is_file():
        raise CommandError(f'base Dockerfile not found: {dockerfile}')
    no_push = bool(getattr(args, 'no_push', False)) or cfg.is_local_registry()
    try:
        tag = build_base_image(cfg, dockerfile, push=not no_push)
    except BuildError as exc:
        raise CommandError(str(exc)) from exc
    _out(f'built base image {tag}')
    return 0


def run_command(name: str, args: Any) -> int:
    from cat_agent.platform.stack import (
        StackError,
        cmd_stack_bootstrap,
        cmd_stack_compose,
        cmd_stack_down,
        cmd_stack_seed,
        cmd_stack_up,
    )

    handlers = {
        'doctor': cmd_doctor,
        'ls': cmd_ls,
        'status': cmd_status,
        'logs': cmd_logs,
        'deploy': cmd_deploy,
        'rm': cmd_rm,
        'rollback': cmd_rollback,
        'build-base': cmd_build_base,
        'stack-up': cmd_stack_up,
        'stack-down': cmd_stack_down,
        'stack-compose': cmd_stack_compose,
        'stack-seed': cmd_stack_seed,
        'stack-bootstrap': cmd_stack_bootstrap,
    }
    try:
        return handlers[name](args)
    except (
        CommandError,
        ManifestError,
        ConfigError,
        NomadError,
        BuildError,
        GatewayError,
        StackError,
    ) as exc:
        _out(str(exc), file=sys.stderr)
        return 1
