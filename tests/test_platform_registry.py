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

"""Local vs remote registry rendering and auth blocks (no network)."""

from __future__ import annotations

from cat_agent.platform.config import PlatformConfig
from cat_agent.platform.manifest import validate_manifest
from cat_agent.platform.render import render_all

FIXED_AT = '2026-01-01T00:00:00Z'
# Deliberate: image tags are {team}/{name}:{content}, NOT agent-{team}-{name}.
# Nomad job IDs stay agent-{team}-{name} for Consul / Traefik / ls|rm.
LOCAL_TAG = 'growth/funding-scout:abc'


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


def _manifest():
    return validate_manifest(
        {
            'name': 'funding-scout',
            'team': 'growth',
            'runtime': {'entrypoint': 'scout:registry'},
            'trigger': {'type': 'http'},
            'resources': {'cpu': 500, 'memory': 512, 'timeout': '300s'},
            'serve': {'shutdown_timeout': '45s'},
        }
    )


def test_local_image_tag_helper():
    assert (
        PlatformConfig.local_image_tag('growth', 'funding-scout', 'abc')
        == 'growth/funding-scout:abc'
    )


def test_local_ref_is_bare_team_name():
    cfg = _cfg(registry='local')
    assert cfg.is_local_registry()
    assert cfg.image_ref(LOCAL_TAG) == 'growth/funding-scout:abc'
    hcl = render_all(
        _manifest(),
        cfg,
        image_tag=LOCAL_TAG,
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )[0][1]
    assert 'job "agent-growth-funding-scout"' in hcl  # job id unchanged
    assert 'image = "growth/funding-scout:abc"' in hcl
    assert 'image = "agent-growth-funding-scout' not in hcl
    assert 'auth {' not in hcl
    assert 'cat-agent-registry-pull' not in hcl
    assert 'secrets/registry.env' not in hcl
    assert 'force_pull =' not in hcl
    assert 'REGISTRY_PASS' not in hcl
    assert 'zot-pull' not in hcl
    assert 'cat-agent-zot' not in hcl


def test_remote_ref_prefixes_registry_and_adds_auth():
    cfg = _cfg(registry='127.0.0.1:5001')
    assert not cfg.is_local_registry()
    assert cfg.image_ref(LOCAL_TAG) == '127.0.0.1:5001/growth/funding-scout:abc'
    assert (
        cfg.registry_pull_vault_path() == 'secret/data/platform/registry/pull'
    )
    hcl = render_all(
        _manifest(),
        cfg,
        image_tag=LOCAL_TAG,
        manifest_sha_value='deadbeef',
        deployed_by='tester',
        deployed_at=FIXED_AT,
    )[0][1]
    assert 'job "agent-growth-funding-scout"' in hcl
    assert 'image = "127.0.0.1:5001/growth/funding-scout:abc"' in hcl
    assert 'auth {' in hcl
    assert 'username = "${REGISTRY_USER}"' in hcl
    assert 'password = "${REGISTRY_PASS}"' in hcl
    assert 'cat-agent-registry-pull' in hcl
    assert 'secrets/registry.env' in hcl
    assert 'secret/data/platform/registry/pull' in hcl
    assert 'force_pull =' not in hcl
    # Credentials must never appear as plaintext in rendered HCL.
    assert 'cat-agent-zot-pull-dev' not in hcl
    assert 'cat-agent-zot-push-dev' not in hcl
    # Template source uses Vault interpolation, not a literal password= value.
    assert 'password = "' not in hcl.replace('password = "${REGISTRY_PASS}"', '')


def test_force_pull_false_both_modes():
    for registry in ('local', 'registry.example/agents'):
        hcl = render_all(
            _manifest(),
            _cfg(registry=registry),
            image_tag=LOCAL_TAG,
            manifest_sha_value='deadbeef',
            deployed_by='tester',
            deployed_at=FIXED_AT,
        )[0][1]
        assert 'force_pull =' not in hcl
        assert 'force_pull must stay false' in hcl


def test_remote_auth_on_all_trigger_templates():
    cases = [
        (
            {
                'name': 'funding-scout',
                'team': 'growth',
                'runtime': {'entrypoint': 'scout:registry'},
                'trigger': {'type': 'http'},
            },
            'growth/funding-scout:abc',
            1,
        ),
        (
            {
                'name': 'nightly-report',
                'team': 'ops',
                'runtime': {'entrypoint': 'report:registry'},
                'trigger': {
                    'type': 'schedule',
                    'cron': '0 */5 * * *',
                    'timezone': 'UTC',
                },
            },
            'ops/nightly-report:abc',
            1,
        ),
        (
            {
                'name': 'queue-worker',
                'team': 'ops',
                'runtime': {'entrypoint': 'worker:registry'},
                'trigger': {'type': 'worker'},
            },
            'ops/queue-worker:abc',
            1,
        ),
        (
            {
                'name': 'heavy-scout',
                'team': 'growth',
                'runtime': {'entrypoint': 'scout:registry'},
                'trigger': {'type': 'http', 'jobs': 'dispatch'},
            },
            'growth/heavy-scout:abc',
            2,
        ),
    ]
    cfg = _cfg(registry='127.0.0.1:5001')
    for data, tag, n_jobs in cases:
        jobs = render_all(
            validate_manifest(data),
            cfg,
            image_tag=tag,
            manifest_sha_value='x',
            deployed_by='t',
            deployed_at=FIXED_AT,
        )
        assert len(jobs) == n_jobs
        for _jid, hcl in jobs:
            assert 'auth {' in hcl
            assert 'cat-agent-registry-pull' in hcl
            assert 'secrets/registry.env' in hcl
            assert 'force_pull =' not in hcl
            assert 'cat-agent-zot' not in hcl
