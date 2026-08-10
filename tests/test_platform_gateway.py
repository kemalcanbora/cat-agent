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

"""Gateway / Vault template contract for rendered Nomad HCL (no network)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from cat_agent.platform.config import PlatformConfig, load_platform_config
from cat_agent.platform.manifest import ManifestError, validate_manifest
from cat_agent.platform.render import render_all

GOLDEN = Path(__file__).parent / 'golden' / 'platform'
FIXED_AT = '2026-01-01T00:00:00Z'


def _cfg(**kwargs) -> PlatformConfig:
    base = dict(
        nomad_addr='http://127.0.0.1:4646',
        namespace='agents',
        datacenters=['dc1'],
        registry='local',
        llm_gateway='http://llm-gateway.service.consul:4000/v1',
        llm_credentials_path='secret/data/platform/llm',
        vault_addr='http://vault.internal:8200',
    )
    base.update(kwargs)
    return PlatformConfig(**base)


def _env_block(hcl: str) -> str:
    """Return the contents of the task env { ... } block (not nested braces)."""
    m = re.search(r'\n\s*env\s*\{', hcl)
    if not m:
        raise AssertionError('no env { block in HCL')
    start = m.end()
    depth = 1
    i = start
    while i < len(hcl) and depth:
        if hcl[i] == '{':
            depth += 1
        elif hcl[i] == '}':
            depth -= 1
        i += 1
    return hcl[start : i - 1]


def _render_variants():
    """All four template kinds (service, dispatch×2, periodic, worker)."""
    fixed = dict(
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )
    out: list[tuple[str, str]] = []

    http = validate_manifest(
        {
            'name': 'funding-scout',
            'team': 'growth',
            'runtime': {'entrypoint': 'scout:registry'},
            'trigger': {'type': 'http'},
            'resources': {'cpu': 500, 'memory': 512, 'timeout': '300s'},
            'serve': {'shutdown_timeout': '45s'},
            'tools': {'allow': ['storage', 'web_search']},
            'env': {'LOG_LEVEL': 'INFO'},
            'model': {'alias': 'smart'},
        }
    )
    out.extend(
        render_all(
            http,
            _cfg(),
            image_tag='growth/funding-scout:abc',
            manifest_sha_value='deadbeef',
            **fixed,
        )
    )

    dispatch = validate_manifest(
        {
            'name': 'heavy-scout',
            'team': 'growth',
            'runtime': {'entrypoint': 'scout:registry'},
            'trigger': {'type': 'http', 'jobs': 'dispatch'},
            'scaling': {'min': 1, 'max': 3},
            'model': {'alias': 'fast'},
        }
    )
    out.extend(
        render_all(
            dispatch,
            _cfg(),
            image_tag='growth/heavy-scout:abc',
            manifest_sha_value='face',
            **fixed,
        )
    )

    periodic = validate_manifest(
        {
            'name': 'nightly-report',
            'team': 'ops',
            'runtime': {'entrypoint': 'report:registry'},
            'trigger': {
                'type': 'schedule',
                'cron': '0 */5 * * *',
                'timezone': 'Europe/Istanbul',
            },
            'model': {'alias': 'default'},
        }
    )
    out.extend(
        render_all(
            periodic,
            _cfg(),
            image_tag='ops/nightly-report:abc',
            manifest_sha_value='cafe',
            **fixed,
        )
    )

    worker = validate_manifest(
        {
            'name': 'queue-worker',
            'team': 'ops',
            'runtime': {'entrypoint': 'worker:registry'},
            'trigger': {'type': 'worker'},
            'model': {'alias': 'default'},
        }
    )
    out.extend(
        render_all(
            worker,
            _cfg(),
            image_tag='ops/queue-worker:abc',
            manifest_sha_value='babe',
            **fixed,
        )
    )
    return out


def test_openai_api_key_only_in_vault_template_not_env():
    """Vault template and env{} are mutually exclusive for OPENAI_API_KEY."""
    jobs = _render_variants()
    assert len(jobs) == 5  # service + dispatch service/task + periodic + worker
    for job_id, hcl in jobs:
        assert 'secrets/llm.env' in hcl, job_id
        assert 'OPENAI_API_KEY=' in hcl, job_id
        assert 'secret/data/platform/llm/teams/' in hcl, job_id
        env_body = _env_block(hcl)
        assert 'OPENAI_API_KEY' not in env_body, (
            f'{job_id}: OPENAI_API_KEY must not appear inside env {{}}; '
            f'got snippet: {env_body!r}'
        )


def test_cat_agent_llm_model_carries_alias():
    m = validate_manifest(
        {
            'name': 'funding-scout',
            'team': 'growth',
            'runtime': {'entrypoint': 'scout:registry'},
            'trigger': {'type': 'http'},
            'model': {'alias': 'smart'},
        }
    )
    hcl = render_all(
        m,
        _cfg(),
        image_tag='growth/funding-scout:abc',
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )[0][1]
    assert 'CAT_AGENT_LLM_MODEL = "smart"' in hcl
    assert 'CAT_AGENT_MODEL_ALIAS = "smart"' in hcl


def test_team_cannot_override_openai_base_url():
    with pytest.raises(ManifestError, match='OPENAI_BASE_URL|protected|gateway'):
        validate_manifest(
            {
                'name': 'funding-scout',
                'team': 'growth',
                'runtime': {'entrypoint': 'scout:registry'},
                'trigger': {'type': 'http'},
                'env': {'OPENAI_BASE_URL': 'http://evil'},
            }
        )


def test_team_cannot_override_openai_api_key_via_env():
    with pytest.raises(ManifestError, match='OPENAI_API_KEY|protected'):
        validate_manifest(
            {
                'name': 'funding-scout',
                'team': 'growth',
                'runtime': {'entrypoint': 'scout:registry'},
                'trigger': {'type': 'http'},
                'env': {'OPENAI_API_KEY': 'sk-from-manifest'},
            }
        )


def test_team_llm_vault_path_derived():
    cfg = _cfg(llm_credentials_path='secret/data/platform/llm')
    assert cfg.team_llm_vault_path('demo') == 'secret/data/platform/llm/teams/demo'
    m = validate_manifest(
        {
            'name': 'calc',
            'team': 'demo',
            'runtime': {'entrypoint': 'app:registry'},
            'trigger': {'type': 'http'},
        }
    )
    hcl = render_all(
        m,
        cfg,
        image_tag='demo/calc:abc',
        manifest_sha_value='x',
        deployed_by='t',
        deployed_at=FIXED_AT,
    )[0][1]
    assert 'secret/data/platform/llm/teams/demo' in hcl
    assert 'cat-agent-llm-demo' in hcl


def test_llm_credentials_path_from_config_file(tmp_path):
    path = tmp_path / 'c.toml'
    path.write_text(
        '[platform]\n'
        'llm_credentials_path = "secret/data/platform/llm"\n'
        'consul_dns = "10.32.0.2"\n',
        encoding='utf-8',
    )
    cfg = load_platform_config(path=path)
    assert cfg.llm_credentials_path == 'secret/data/platform/llm'
    assert cfg.consul_dns == '10.32.0.2'
    assert cfg.team_llm_vault_path('ops') == 'secret/data/platform/llm/teams/ops'


def test_golden_service_docker_network_includes_consul_dns():
    m = validate_manifest(
        {
            'name': 'funding-scout',
            'team': 'growth',
            'runtime': {'entrypoint': 'scout:registry'},
            'trigger': {'type': 'http'},
            'resources': {'cpu': 500, 'memory': 512, 'timeout': '300s'},
            'serve': {'shutdown_timeout': '45s'},
            'tools': {'allow': ['storage', 'web_search']},
            'env': {'LOG_LEVEL': 'INFO'},
        }
    )
    hcl = render_all(
        m,
        _cfg(
            docker_network='nomad_deploy_hashicorp',
            consul_dns='10.32.0.2',
        ),
        image_tag='growth/funding-scout:abc',
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )[0][1]
    assert 'dns_servers = ["10.32.0.2"]' in hcl
    assert 'llm-gateway.service.consul' in hcl  # comment + OPENAI_BASE_URL
    assert 'DOCKER DESKTOP / macOS ONLY' in hcl
    assert 'pinned Consul DNS IP' in hcl
    path = GOLDEN / 'service_docker_network.hcl'
    assert path.read_text(encoding='utf-8') == hcl
