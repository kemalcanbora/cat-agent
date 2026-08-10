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

"""Render agent manifests to Nomad HCL job specs."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from cat_agent.platform.config import PlatformConfig
from cat_agent.platform.manifest import PROTECTED_ENV_KEYS, AgentManifest, ManifestError

TEMPLATES_DIR = Path(__file__).resolve().parent / 'templates'


def _jinja_env():
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def manifest_sha(raw_yaml: str) -> str:
    return hashlib.sha256(raw_yaml.encode('utf-8')).hexdigest()[:16]


def build_render_context(
    manifest: AgentManifest,
    config: PlatformConfig,
    *,
    image_tag: str,
    image_ref: str,
    manifest_sha_value: str,
    deployed_by: str = 'unknown',
    deployed_at: str | None = None,
) -> Dict[str, Any]:
    """Derive everything templates need; templates never compute."""
    for key in manifest.env:
        if key.upper() in PROTECTED_ENV_KEYS or key.upper().startswith('CAT_AGENT_SERVE_'):
            raise ManifestError(
                f'env.{key}: protected key cannot override platform injection'
            )

    kill_timeout = max(manifest.serve.shutdown_timeout, manifest.resources.timeout) + 15
    # Nomad rejects jobs when kill_timeout exceeds update.progress_deadline.
    progress_deadline_secs = max(600, kill_timeout + 60)
    tools_csv = ','.join(sorted(manifest.tools.allow))
    team_env = dict(sorted(manifest.env.items()))
    platform_env = {
        'CAT_AGENT_MANAGED': '1',
        'CAT_AGENT_ENTRYPOINT': manifest.runtime.entrypoint,
        'CAT_AGENT_MODE': 'service',
        'OPENAI_BASE_URL': config.llm_gateway,
        'CAT_AGENT_LLM_BASE_URL': config.llm_gateway,
        'CAT_AGENT_TOOLS_ALLOW': tools_csv,
        'CAT_AGENT_SERVE_HOST': '0.0.0.0',
        'CAT_AGENT_SERVE_PORT': '8080',
        'PORT': '8080',
    }
    if config.otel_endpoint:
        platform_env['OTEL_EXPORTER_OTLP_ENDPOINT'] = config.otel_endpoint

    # Team env merged last but cannot override platform keys
    merged_env = dict(platform_env)
    for k, v in team_env.items():
        if k in platform_env or k.upper() in PROTECTED_ENV_KEYS:
            raise ManifestError(f'env.{k}: cannot override platform key')
        merged_env[k] = v

    return {
        'job_id': manifest.job_id(),
        'dispatch_job_id': manifest.dispatch_job_id(),
        'namespace': config.namespace,
        'datacenters': list(config.datacenters),
        'team': manifest.team,
        'agent': manifest.name,
        'trigger': manifest.trigger.type,
        'jobs_mode': manifest.trigger.jobs,
        'image_tag': image_tag,
        'image_ref': image_ref,
        'manifest_sha': manifest_sha_value,
        'deployed_by': deployed_by,
        'deployed_at': deployed_at
        or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'cpu': manifest.resources.cpu,
        'memory': manifest.resources.memory,
        'count': manifest.scaling.min,
        'scaling_max': manifest.scaling.max,
        'kill_timeout': kill_timeout,
        'shutdown_timeout': manifest.serve.shutdown_timeout,
        'run_timeout': manifest.resources.timeout,
        'deploy_deadline': f'{progress_deadline_secs}s',
        'progress_deadline': f'{progress_deadline_secs}s',
        'env': merged_env,
        'secrets': list(manifest.secrets),
        'vault_addr': config.vault_addr,
        'has_secrets': bool(manifest.secrets),
        'entrypoint': manifest.runtime.entrypoint,
        'path': manifest.trigger.path,
        'cron': manifest.trigger.cron or '',
        'timezone': manifest.trigger.timezone,
        'job_cpu': manifest.trigger.job_cpu,
        'job_memory': manifest.trigger.job_memory,
        'job_timeout': manifest.trigger.job_timeout,
        'max_concurrency': manifest.serve.max_concurrency,
        'max_queue': manifest.serve.max_queue,
        'model_alias': manifest.model.alias,
        'llm_gateway': config.llm_gateway,
        'llm_credentials_path': config.llm_credentials_path,
        'llm_team_vault_path': config.team_llm_vault_path(manifest.team),
        'is_local_registry': config.is_local_registry(),
        'registry_credentials_path': config.registry_credentials_path,
        'registry_pull_vault_path': config.registry_pull_vault_path(),
        'docker_network': (config.docker_network or '').strip(),
        'consul_dns': (config.consul_dns or '').strip(),
        'ingress_host': config.ingress_host(manifest.team, manifest.name),
    }


def render_template(name: str, ctx: Dict[str, Any]) -> str:
    env = _jinja_env()
    return env.get_template(name).render(**ctx)


def render_all(
    manifest: AgentManifest,
    config: PlatformConfig,
    *,
    image_tag: str,
    manifest_sha_value: str,
    deployed_by: str = 'unknown',
    deployed_at: str | None = None,
) -> List[Tuple[str, str]]:
    """Return ``[(job_id, hcl), ...]``. Dispatch agents yield two jobs."""
    image_ref = config.image_ref(image_tag)
    ctx = build_render_context(
        manifest,
        config,
        image_tag=image_tag,
        image_ref=image_ref,
        manifest_sha_value=manifest_sha_value,
        deployed_by=deployed_by,
        deployed_at=deployed_at,
    )
    out: List[Tuple[str, str]] = []
    if manifest.trigger.type == 'http':
        out.append((manifest.job_id(), render_template('service.hcl.tmpl', ctx)))
        if manifest.trigger.jobs == 'dispatch':
            task_ctx = dict(ctx)
            task_ctx['env'] = dict(ctx['env'])
            task_ctx['env']['CAT_AGENT_MODE'] = 'task'
            out.append(
                (
                    manifest.dispatch_job_id(),
                    render_template('dispatch.hcl.tmpl', task_ctx),
                )
            )
    elif manifest.trigger.type == 'schedule':
        out.append((manifest.job_id(), render_template('periodic.hcl.tmpl', ctx)))
    elif manifest.trigger.type == 'worker':
        out.append((manifest.job_id(), render_template('worker.hcl.tmpl', ctx)))
    else:  # pragma: no cover
        raise ManifestError(f'unknown trigger type: {manifest.trigger.type}')
    return out
