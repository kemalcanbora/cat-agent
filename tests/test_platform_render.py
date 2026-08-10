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

"""Golden HCL render tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from cat_agent.platform.config import PlatformConfig
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
        vault_addr='http://vault.internal:8200',
    )
    base.update(kwargs)
    return PlatformConfig(**base)


def _http_manifest(**extra):
    data = {
        'name': 'funding-scout',
        'team': 'growth',
        'runtime': {'entrypoint': 'scout:registry'},
        'trigger': {'type': 'http'},
        'resources': {'cpu': 500, 'memory': 512, 'timeout': '300s'},
        'serve': {'shutdown_timeout': '45s'},
        'tools': {'allow': ['storage', 'web_search']},
        'env': {'LOG_LEVEL': 'INFO'},
    }
    data.update(extra)
    return validate_manifest(data)


def test_kill_timeout_exceeds_both():
    m = _http_manifest(
        resources={'timeout': '300s'},
        serve={'shutdown_timeout': '45s'},
    )
    jobs = render_all(
        m,
        _cfg(),
        image_tag='growth/funding-scout:abc',
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )
    hcl = jobs[0][1]
    # max(45, 300)+15 = 315
    assert 'kill_timeout = "315s"' in hcl


def test_team_env_cannot_override_openai(monkeypatch):
    with pytest.raises(ManifestError, match='OPENAI_BASE_URL|reserved|gateway'):
        _http_manifest(env={'OPENAI_BASE_URL': 'http://evil'})


def test_secrets_not_in_env_block():
    m = _http_manifest(secrets=['SMTP_PASSWORD'])
    jobs = render_all(
        m,
        _cfg(),
        image_tag='growth/funding-scout:abc',
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )
    hcl = jobs[0][1]
    assert 'SMTP_PASSWORD =' not in hcl.split('env {')[1].split('}')[0]
    assert 'secrets/app.env' in hcl
    assert 'vault {' in hcl


def test_byte_identical_render():
    m = _http_manifest()
    kwargs = dict(
        image_tag='growth/funding-scout:abc',
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )
    a = render_all(m, _cfg(), **kwargs)
    b = render_all(m, _cfg(), **kwargs)
    assert a[0][1] == b[0][1]


def test_force_pull_absent_and_comment_present():
    m = _http_manifest()
    hcl = render_all(
        m,
        _cfg(),
        image_tag='growth/funding-scout:abc',
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )[0][1]
    assert 'force_pull =' not in hcl
    assert 'force_pull must stay false' in hcl
    assert 'image = "growth/funding-scout:abc"' in hcl


def test_remote_registry_prefix():
    m = _http_manifest()
    hcl = render_all(
        m,
        _cfg(registry='registry.example/agents'),
        image_tag='growth/funding-scout:abc',
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )[0][1]
    assert 'job "agent-growth-funding-scout"' in hcl
    assert 'image = "registry.example/agents/growth/funding-scout:abc"' in hcl
    assert 'auth {' in hcl
    assert 'cat-agent-registry-pull' in hcl


def _assert_golden(name: str, hcl: str, update: bool = False):
    GOLDEN.mkdir(parents=True, exist_ok=True)
    path = GOLDEN / name
    if update or not path.exists():
        path.write_text(hcl, encoding='utf-8')
    assert path.read_text(encoding='utf-8') == hcl


def test_golden_service():
    m = _http_manifest()
    hcl = render_all(
        m,
        _cfg(),
        image_tag='growth/funding-scout:abc',
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )[0][1]
    _assert_golden('service.hcl', hcl)


def test_golden_periodic():
    m = validate_manifest(
        {
            'name': 'nightly-report',
            'team': 'ops',
            'runtime': {'entrypoint': 'report:registry'},
            'trigger': {
                'type': 'schedule',
                'cron': '0 */5 * * *',
                'timezone': 'Europe/Istanbul',
            },
        }
    )
    hcl = render_all(
        m,
        _cfg(),
        image_tag='ops/nightly-report:abc',
        manifest_sha_value='cafe',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )[0][1]
    _assert_golden('periodic.hcl', hcl)


def test_golden_worker():
    m = validate_manifest(
        {
            'name': 'queue-worker',
            'team': 'ops',
            'runtime': {'entrypoint': 'worker:registry'},
            'trigger': {'type': 'worker'},
        }
    )
    hcl = render_all(
        m,
        _cfg(),
        image_tag='ops/queue-worker:abc',
        manifest_sha_value='babe',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )[0][1]
    _assert_golden('worker.hcl', hcl)


def test_golden_dispatch_pair():
    m = validate_manifest(
        {
            'name': 'heavy-scout',
            'team': 'growth',
            'runtime': {'entrypoint': 'scout:registry'},
            'trigger': {'type': 'http', 'jobs': 'dispatch'},
            'scaling': {'min': 1, 'max': 3},
        }
    )
    jobs = render_all(
        m,
        _cfg(),
        image_tag='growth/heavy-scout:abc',
        manifest_sha_value='face',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )
    assert len(jobs) == 2
    _assert_golden('dispatch_service.hcl', jobs[0][1])
    _assert_golden('dispatch_task.hcl', jobs[1][1])


def test_ingress_host_template_in_traefik_tags():
    m = _http_manifest()
    hcl = render_all(
        m,
        _cfg(ingress_host_template='{team}-{name}.agents.example.internal'),
        image_tag='growth/funding-scout:abc',
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )[0][1]
    assert 'Host(`growth-funding-scout.agents.example.internal`)' in hcl
    assert 'Host(`growth-funding-scout.localhost`)' not in hcl
