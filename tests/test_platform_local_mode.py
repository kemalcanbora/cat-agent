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

"""Local-image mode and doctor registry messaging."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cat_agent.platform import commands as platform_commands
from cat_agent.platform.config import PlatformConfig
from cat_agent.platform.manifest import validate_manifest
from cat_agent.platform.render import render_all


def test_local_registry_bare_tag():
    m = validate_manifest(
        {
            'name': 'calc',
            'team': 'demo',
            'runtime': {'entrypoint': 'app:registry'},
            'trigger': {'type': 'http'},
        }
    )
    cfg = PlatformConfig(registry='local')
    hcl = render_all(
        m,
        cfg,
        image_tag='demo/calc:abc',
        manifest_sha_value='x',
        deployed_by='t',
        deployed_at='2026-01-01T00:00:00Z',
    )[0][1]
    # Job id / Traefik router stay flat; only Docker image uses team/name.
    assert 'job "agent-demo-calc"' in hcl
    assert 'Host(`demo-calc.localhost`)' in hcl
    assert 'image = "demo/calc:abc"' in hcl
    assert 'image = "agent-demo-calc' not in hcl
    assert 'auth {' not in hcl
    assert 'force_pull =' not in hcl


def test_remote_registry_full_ref():
    m = validate_manifest(
        {
            'name': 'calc',
            'team': 'demo',
            'runtime': {'entrypoint': 'app:registry'},
            'trigger': {'type': 'http'},
        }
    )
    cfg = PlatformConfig(registry='registry.example/agents')
    hcl = render_all(
        m,
        cfg,
        image_tag='demo/calc:abc',
        manifest_sha_value='x',
        deployed_by='t',
        deployed_at='2026-01-01T00:00:00Z',
    )[0][1]
    assert 'job "agent-demo-calc"' in hcl
    assert 'image = "registry.example/agents/demo/calc:abc"' in hcl
    assert 'auth {' in hcl
    assert 'cat-agent-registry-pull' in hcl
    assert 'force_pull =' not in hcl
    assert 'cat-agent-zot' not in hcl


def test_doctor_prints_local_mode(monkeypatch, tmp_path):
    monkeypatch.delenv('NOMAD_ADDR', raising=False)
    monkeypatch.delenv('CAT_AGENT_REGISTRY', raising=False)
    cfg_path = tmp_path / 'c.toml'
    cfg_path.write_text(
        '[platform]\nregistry = "local"\nnomad_addr = "http://127.0.0.1:9"\n',
        encoding='utf-8',
    )
    args = SimpleNamespace(
        config=str(cfg_path), nomad_addr=None, registry=None, team='demo'
    )
    lines: list[str] = []

    def capture(msg: str, file=None):
        lines.append(msg)

    from cat_agent.platform.gateway import GatewayError
    from cat_agent.platform.nomad import NomadUnreachable

    # Call via the module so patches apply after test_platform_isolation
    # deletes and reloads cat_agent.platform.* from sys.modules.
    with patch.object(platform_commands, '_out', side_effect=capture):
        with patch.object(platform_commands.shutil, 'which', return_value='/bin/docker'):
            with patch.object(platform_commands, 'NomadClient') as Cls:
                Cls.return_value.status_leader.side_effect = NomadUnreachable(
                    'Nomad unreachable'
                )
                with patch.object(
                    platform_commands,
                    'fetch_aliases_for_config',
                    side_effect=GatewayError('LLM gateway unreachable'),
                ):
                    with patch.object(
                        platform_commands,
                        'vault_team_key_exists',
                        side_effect=GatewayError('Vault secret not found'),
                    ):
                        rc = platform_commands.cmd_doctor(args)
    joined = '\n'.join(lines)
    assert 'registry: local (images are not pushed; only this node can run them)' in joined
    assert rc == 1


def test_deploy_skips_push_when_local(tmp_path, monkeypatch):
    (tmp_path / 'agent.yaml').write_text(
        'name: calc\nteam: demo\nruntime:\n  entrypoint: app:registry\n'
        'trigger:\n  type: http\n',
        encoding='utf-8',
    )
    (tmp_path / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')
    args = SimpleNamespace(
        config=None,
        nomad_addr=None,
        registry='local',
        dir=str(tmp_path),
        dry_run=True,
        image_tag='demo/calc:test',
        no_push=False,
        skip_alias_check=True,
    )

    def fake_build(*a, **k):
        raise AssertionError('build should be skipped when --image-tag is set')

    monkeypatch.setattr(platform_commands, 'build_agent_image', fake_build)
    monkeypatch.setattr(platform_commands, '_known_tools', lambda: set())
    monkeypatch.setattr(
        platform_commands,
        '_resolve_entrypoint_registry',
        lambda *_a, **_k: SimpleNamespace(names=lambda: ['calc']),
    )
    rc = platform_commands.cmd_deploy(args)
    assert rc == 0
