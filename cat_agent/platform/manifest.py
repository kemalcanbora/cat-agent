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

"""agent.yaml schema and validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NAME_RE = re.compile(r'^[a-z][a-z0-9-]{1,38}$')

PLATFORM_OWNED_KEYS = frozenset(
    {'image', 'dockerfile', 'replicas', 'datacenter', 'namespace'}
)
PROTECTED_ENV_KEYS = frozenset(
    {
        'OPENAI_BASE_URL',
        'OPENAI_API_KEY',
        'CAT_AGENT_LLM_BASE_URL',
        'CAT_AGENT_LLM_API_KEY',
        'CAT_AGENT_SERVE_HOST',
        'CAT_AGENT_SERVE_PORT',
        'CAT_AGENT_SERVE_TOKEN',
        'CAT_AGENT_MANAGED',
        'CAT_AGENT_ENTRYPOINT',
        'CAT_AGENT_MODE',
        'CAT_AGENT_TOOLS_ALLOW',
    }
)


class ManifestError(ValueError):
    """Raised when an agent.yaml is invalid."""


def _parse_duration_seconds(value: Any, *, field: str) -> int:
    if isinstance(value, (int, float)):
        secs = int(value)
    elif isinstance(value, str):
        raw = value.strip().lower()
        if raw.endswith('s') and raw[:-1].isdigit():
            secs = int(raw[:-1])
        elif raw.endswith('m') and raw[:-1].isdigit():
            secs = int(raw[:-1]) * 60
        elif raw.endswith('h') and raw[:-1].isdigit():
            secs = int(raw[:-1]) * 3600
        elif raw.isdigit():
            secs = int(raw)
        else:
            raise ManifestError(
                f'{field}: expected duration like "300s" or an integer number of seconds, '
                f'got {value!r}'
            )
    else:
        raise ManifestError(f'{field}: expected duration, got {type(value).__name__}')
    if secs <= 0:
        raise ManifestError(f'{field}: must be positive, got {secs}')
    return secs


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    entrypoint: str
    python: str = '3.11'


class TriggerSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    type: Literal['http', 'schedule', 'worker']
    path: str = '/invoke'
    cron: Optional[str] = None
    timezone: str = 'UTC'
    jobs: Literal['inline', 'dispatch'] = 'inline'
    job_cpu: int = 500
    job_memory: int = 512
    job_timeout: int = 1800

    @field_validator('job_timeout', mode='before')
    @classmethod
    def _job_timeout(cls, v: Any) -> int:
        return _parse_duration_seconds(v, field='trigger.job_timeout')


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    alias: str = 'default'
    type: Optional[str] = None
    max_tokens_per_day: int = 2_000_000

    @model_validator(mode='after')
    def _no_local(self) -> ModelSpec:
        if self.type is not None and self.type.lower() == 'local':
            raise ManifestError(
                'model.type: local is not deployable to Nomad. Local models '
                '(llama_cpp, transformers, mlx_lm, …) still work locally via '
                '`cat-agent serve` and example `__main__` runners — they are not deleted.'
            )
        return self


class ResourcesSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    cpu: int = 500
    memory: int = 512
    timeout: int = 300
    gpu: Optional[Any] = None

    @field_validator('timeout', mode='before')
    @classmethod
    def _timeout(cls, v: Any) -> int:
        return _parse_duration_seconds(v, field='resources.timeout')

    @model_validator(mode='after')
    def _no_gpu(self) -> ResourcesSpec:
        if self.gpu is not None:
            raise ManifestError(
                'resources.gpu: GPU / local-model placements are not deployable. '
                'Local models still work via `cat-agent serve` on a machine that has '
                'the hardware — use that path instead of Nomad deploy.'
            )
        return self


class ScalingSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    min: int = 1
    max: int = 1


class ServeSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    max_concurrency: int = 8
    max_queue: int = 8
    shutdown_timeout: int = 45

    @field_validator('shutdown_timeout', mode='before')
    @classmethod
    def _shutdown(cls, v: Any) -> int:
        return _parse_duration_seconds(v, field='serve.shutdown_timeout')


class ToolsSpec(BaseModel):
    model_config = ConfigDict(extra='forbid')

    allow: List[str] = Field(default_factory=list)


class AgentManifest(BaseModel):
    model_config = ConfigDict(extra='forbid')

    apiVersion: str = 'v1'
    name: str
    team: str
    runtime: RuntimeSpec
    trigger: TriggerSpec
    model: ModelSpec = Field(default_factory=ModelSpec)
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)
    scaling: ScalingSpec = Field(default_factory=ScalingSpec)
    serve: ServeSpec = Field(default_factory=ServeSpec)
    tools: ToolsSpec = Field(default_factory=ToolsSpec)
    secrets: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)

    @field_validator('name', 'team')
    @classmethod
    def _name_ok(cls, v: str, info: Any) -> str:
        if not NAME_RE.match(v):
            raise ManifestError(
                f'{info.field_name}: must match ^[a-z][a-z0-9-]{{1,38}}$ '
                f'(Nomad job / Consul service name), got {v!r}'
            )
        return v

    @field_validator('secrets')
    @classmethod
    def _secret_names(cls, values: List[str]) -> List[str]:
        for s in values:
            if '=' in s:
                raise ManifestError(
                    f'secrets: {s!r} looks like a value (contains "="). '
                    'List secret *names* only; values come from Vault.'
                )
            if len(s) > 64:
                raise ManifestError(
                    f'secrets: {s!r} is longer than 64 characters — paste a name, not a value.'
                )
            if not s:
                raise ManifestError('secrets: empty name is not allowed')
        return values

    @model_validator(mode='before')
    @classmethod
    def _reject_platform_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for key in PLATFORM_OWNED_KEYS:
            if key in data:
                raise ManifestError(
                    f'{key}: this is a platform decision and cannot appear in agent.yaml. '
                    'The platform chooses base images, replica policy, datacenter, and namespace.'
                )
        return data

    @model_validator(mode='after')
    def _cross_field(self) -> AgentManifest:
        if self.scaling.min < 1:
            raise ManifestError(
                'scaling.min: 0 is not supported. Nomad does not start a job from zero '
                'on an inbound request; the deployment would never receive traffic. '
                'Use min >= 1 (scale-to-zero needs a fronting proxy + Nomad Autoscaler).'
            )
        if self.scaling.max < self.scaling.min:
            raise ManifestError('scaling.max: must be >= scaling.min')

        if self.trigger.type == 'schedule' and not self.trigger.cron:
            raise ManifestError('trigger.cron: required when trigger.type is schedule')

        if self.trigger.jobs == 'dispatch' and self.trigger.type != 'http':
            raise ManifestError(
                'trigger.jobs: dispatch is only valid with trigger.type: http '
                '(nothing to dispatch from for schedule/worker).'
            )

        if self.trigger.jobs == 'inline' and self.scaling.max > 1:
            raise ManifestError(
                'trigger.jobs: inline cannot scale past 1 replica (scaling.max > 1). '
                'Inline jobs live in one process memory; use trigger.jobs: dispatch to scale out.'
            )

        for key in self.env:
            upper = key.upper()
            if upper in PROTECTED_ENV_KEYS or upper.startswith('CAT_AGENT_SERVE_'):
                raise ManifestError(
                    f'env.{key}: reserved. OPENAI_* / CAT_AGENT_SERVE_* / gateway keys are '
                    'injected by the platform so traffic goes through the LLM gateway '
                    '(quota and cost attribution). Remove it from agent.yaml.'
                )
            if upper in ('OPENAI_BASE_URL', 'OPENAI_API_KEY'):
                raise ManifestError(
                    f'env.{key}: bypasses the LLM gateway. Remove it from agent.yaml.'
                )

        if ':' not in self.runtime.entrypoint:
            raise ManifestError(
                'runtime.entrypoint: expected "module:attr" returning an AgentRegistry'
            )
        return self

    def validate_timeout_ceiling(self, max_timeout_seconds: int) -> None:
        if self.resources.timeout > max_timeout_seconds:
            raise ManifestError(
                f'resources.timeout: {self.resources.timeout}s exceeds the platform ceiling '
                f'({max_timeout_seconds}s). For long work use trigger.jobs: dispatch '
                'instead of holding an HTTP connection open.'
            )

    def validate_tools(self, known: Set[str]) -> None:
        unknown = sorted(set(self.tools.allow) - known)
        if unknown:
            raise ManifestError(
                'tools.allow: unknown tool name(s): '
                + ', '.join(unknown)
                + '. Check TOOL_REGISTRY / OPTIONAL_TOOL_REGISTRY (enable optional tools first).'
            )

    def job_id(self) -> str:
        return f'agent-{self.team}-{self.name}'

    def dispatch_job_id(self) -> str:
        return f'agent-{self.team}-{self.name}-task'


def validate_manifest(data: Dict[str, Any]) -> AgentManifest:
    """Validate a raw mapping, re-raising :class:`ManifestError` without Pydantic wrapping."""
    try:
        return AgentManifest.model_validate(data)
    except ManifestError:
        raise
    except Exception as exc:
        # Unwrap ManifestError nested in pydantic ValidationError
        if hasattr(exc, 'errors'):
            for err in exc.errors():  # type: ignore[attr-defined]
                ctx = err.get('ctx') or {}
                inner = ctx.get('error')
                if isinstance(inner, ManifestError):
                    raise inner from exc
                msg = err.get('msg', '')
                if isinstance(msg, str) and msg.startswith('Value error, '):
                    raise ManifestError(msg[len('Value error, ') :]) from exc
            parts = []
            for err in exc.errors():  # type: ignore[attr-defined]
                loc = '.'.join(str(x) for x in err.get('loc', ()))
                parts.append(f"{loc}: {err.get('msg', str(exc))}")
            raise ManifestError('; '.join(parts) if parts else str(exc)) from exc
        raise ManifestError(str(exc)) from exc


def load_manifest(path: str | Path) -> AgentManifest:
    """Load and validate ``agent.yaml`` from *path*."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ManifestError(
            "PyYAML is required for agent manifests; pip install 'cat-agent[platform]'"
        ) from exc

    p = Path(path)
    if not p.is_file():
        raise ManifestError(f'agent.yaml not found: {p}')
    try:
        raw = yaml.safe_load(p.read_text(encoding='utf-8'))
    except Exception as exc:
        raise ManifestError(f'failed to parse {p}: {exc}') from exc
    if not isinstance(raw, dict):
        raise ManifestError(f'{p}: expected a mapping at the top level')
    return validate_manifest(raw)
