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

"""Coverage tests for cat_agent.platform.commands (mocked Nomad / FS)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.platform import commands as cmd
from cat_agent.platform.commands import CommandError
from cat_agent.platform.config import ConfigError, PlatformConfig
from cat_agent.platform.gateway import GatewayError
from cat_agent.platform.manifest import ManifestError
from cat_agent.platform.nomad import NomadError, NomadNotFound, NomadRejected, NomadUnreachable
from cat_agent.platform.registry_probe import RegistryError


def _cfg(**kwargs) -> PlatformConfig:
    base = dict(
        nomad_addr='http://127.0.0.1:4646',
        registry='local',
        docker_network='nomad_net',
        consul_dns='10.0.0.2',
        vault_addr='http://127.0.0.1:8200',
        llm_gateway='http://gw/v1',
        public_url_template='http://{team}-{name}.localhost',
        ingress_host_template='{team}-{name}.localhost',
    )
    base.update(kwargs)
    return PlatformConfig(**base)


@pytest.fixture(autouse=True)
def _skip_team_key_seed(monkeypatch):
    monkeypatch.setattr(cmd, '_ensure_team_llm_key', lambda *a, **k: None)


def _args(**kwargs):
    base = dict(
        config=None,
        nomad_addr=None,
        registry=None,
        team=None,
        json=False,
        name='calc',
        dir=None,
        stderr=False,
        yes=False,
        force=False,
        to=None,
        no_push=False,
        dry_run=False,
        skip_alias_check=True,
        image_tag='demo/calc:tag',
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _job(jid='agent-demo-calc', team='demo', agent='calc', **meta_extra):
    meta = {
        'managed_by': 'cat-agent',
        'team': team,
        'agent': agent,
        'trigger': 'http',
        'jobs_mode': 'inline',
        'image_tag': 't1',
        'deployed_by': 'me',
        'manifest_sha': 'abc',
    }
    meta.update(meta_extra)
    return {'ID': jid, 'Meta': meta, 'Status': 'running'}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def test_known_tools_includes_registries():
    tools = cmd._known_tools()
    assert isinstance(tools, set)
    assert tools


def test_parse_agent_ref_variants():
    assert cmd._parse_agent_ref('calc', None) == ('calc', None)
    assert cmd._parse_agent_ref('demo/calc', None) == ('calc', 'demo')
    assert cmd._parse_agent_ref('demo/calc', 'demo') == ('calc', 'demo')
    with pytest.raises(CommandError, match='invalid'):
        cmd._parse_agent_ref('a/b/c', None)
    with pytest.raises(CommandError, match='mismatch'):
        cmd._parse_agent_ref('demo/calc', 'other')
    with pytest.raises(CommandError, match='invalid'):
        cmd._parse_agent_ref('/calc', None)


def test_resolve_jobs_by_name_errors():
    client = MagicMock()
    client.list_agents.return_value = []
    with pytest.raises(CommandError, match='no cat-agent job'):
        cmd._resolve_jobs_by_name(client, 'missing', None)

    jobs = [
        _job('j1', 'a', 'calc'),
        _job('j2', 'b', 'calc'),
    ]

    def list_agents(team=None):
        if team:
            return [j for j in jobs if (j.get('Meta') or {}).get('team') == team]
        return list(jobs)

    client.list_agents.side_effect = list_agents
    with pytest.raises(CommandError, match='ambiguous'):
        cmd._resolve_jobs_by_name(client, 'calc', None)
    matches = cmd._resolve_jobs_by_name(client, 'calc', 'a')
    assert len(matches) == 1


def test_require_mac_docker_network():
    cfg = _cfg(docker_network='')
    with patch.object(cmd.sys, 'platform', 'linux'):
        cmd._require_mac_docker_network(cfg)
    with patch.object(cmd.sys, 'platform', 'darwin'):
        with pytest.raises(CommandError, match='docker_network'):
            cmd._require_mac_docker_network(cfg)
        cfg2 = _cfg(docker_network='net')
        with patch.object(cmd, '_out') as out:
            cmd._require_mac_docker_network(cfg2)
            out.assert_called()
        remote = _cfg(docker_network='', vault_addr='http://192.168.1.128:8200')
        with patch.object(cmd, '_out') as out:
            cmd._require_mac_docker_network(remote)
            out.assert_called()


def test_ensure_registry_vault_seeds_when_missing():
    cfg = _cfg(registry='192.168.1.128:5001')
    with patch.object(cmd, 'vault_registry_creds_exist', side_effect=RegistryError('no')), \
            patch('cat_agent.platform.stack.seed_registry_vault') as seed, \
            patch.object(cmd, '_out'):
        cmd._ensure_registry_vault(cfg)
        seed.assert_called_once()


def test_ensure_registry_vault_skips_local():
    cfg = _cfg(registry='local')
    with patch.object(cmd, 'vault_registry_creds_exist') as exist:
        cmd._ensure_registry_vault(cfg)
        exist.assert_not_called()


def test_load_cfg_with_env_and_overrides(tmp_path):
    cfg_path = tmp_path / 'c.toml'
    cfg_path.write_text('[platform]\nregistry = "local"\n', encoding='utf-8')
    (tmp_path / '.env').write_text('FOO=1\n', encoding='utf-8')
    args = _args(config=str(cfg_path), nomad_addr='http://n:1', registry='local')
    with patch('cat_agent.platform.gateway.ensure_dev_vault_token'), \
            patch.object(cmd, 'load_platform_config', return_value=_cfg()) as load, \
            patch('dotenv.load_dotenv') as ld:
        out = cmd._load_cfg(args)
    assert out.registry == 'local'
    load.assert_called_once()
    overrides = load.call_args.kwargs.get('overrides') or {}
    assert overrides.get('nomad_addr') == 'http://n:1'
    ld.assert_called_once()


def test_client_wraps_nomad():
    cfg = _cfg()
    with patch('cat_agent.platform.commands.NomadClient') as NC:
        cmd._client(cfg)
        NC.assert_called_once_with(cfg)


# ---------------------------------------------------------------------------
# ls / status / logs
# ---------------------------------------------------------------------------


def test_cmd_ls_text_and_json_and_empty():
    client = MagicMock()
    client.list_agents.return_value = [_job()]
    args = _args(json=False, team='demo')
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out') as out:
        assert cmd.cmd_ls(args) == 0
        assert any('demo/calc' in str(c) for c in out.call_args_list)

    args.json = True
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out') as out:
        assert cmd.cmd_ls(args) == 0
        assert '[' in str(out.call_args[0][0])

    client.list_agents.return_value = []
    args.json = False
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out') as out:
        assert cmd.cmd_ls(args) == 0
        out.assert_called_with('(no cat-agent jobs)')


def test_cmd_status_with_local_sha_warning(tmp_path):
    client = MagicMock()
    client.list_agents.return_value = [_job()]
    client.allocations.return_value = [
        {'ID': 'abcdef12xxxx', 'ClientStatus': 'running', 'DesiredStatus': 'run'},
    ]
    agent_yaml = tmp_path / 'agent.yaml'
    agent_yaml.write_text('name: calc\n', encoding='utf-8')
    args = _args(name='demo/calc', dir=str(tmp_path))
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, 'manifest_sha', return_value='different'), \
            patch.object(cmd, '_out') as out:
        assert cmd.cmd_status(args) == 0
        assert any('WARNING' in str(c) for c in out.call_args_list)


def test_cmd_status_shows_endpoint_info():
    client = MagicMock()
    client.list_agents.return_value = [_job()]
    client.allocations.return_value = [
        {'ID': 'abcdef12xxxx', 'ClientStatus': 'running', 'DesiredStatus': 'run'},
    ]
    args = _args(name='demo/calc')
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out') as out:
        assert cmd.cmd_status(args) == 0
        texts = [str(c) for c in out.call_args_list]
        joined = '\n'.join(texts)
        assert '/agents/calc/run' in joined
        assert 'curl' in joined
        assert 'Authorization' in joined
        assert 'http://demo-calc.localhost' in joined


def test_cmd_status_endpoint_info_dispatch_mode():
    client = MagicMock()
    client.list_agents.return_value = [_job(jobs_mode='dispatch')]
    client.allocations.return_value = []
    args = _args(name='demo/calc')
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out') as out:
        assert cmd.cmd_status(args) == 0
        texts = [str(c) for c in out.call_args_list]
        joined = '\n'.join(texts)
        assert '/agents/calc/jobs' in joined
        assert '202' in joined


def test_cmd_status_no_endpoint_info_for_schedule():
    client = MagicMock()
    client.list_agents.return_value = [_job(trigger='schedule')]
    client.allocations.return_value = []
    args = _args(name='demo/calc')
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out') as out:
        assert cmd.cmd_status(args) == 0
        texts = [str(c) for c in out.call_args_list]
        joined = '\n'.join(texts)
        assert 'curl' not in joined


def test_cmd_logs_uses_latest_alloc_and_task_state():
    client = MagicMock()
    client.list_agents.return_value = [_job()]
    client.allocations.return_value = [
        {'ID': 'old', 'CreateIndex': 1, 'TaskStates': {}},
        {'ID': 'new', 'CreateIndex': 9, 'TaskStates': {'worker': {}}},
    ]
    client.logs.return_value = 'hello-log'
    args = _args(name='calc', team='demo', stderr=True)
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out') as out:
        assert cmd.cmd_logs(args) == 0
        client.logs.assert_called_with('new', 'worker', stderr=True)
        out.assert_called_with('hello-log')


def test_cmd_logs_no_allocs():
    client = MagicMock()
    client.list_agents.return_value = [_job()]
    client.allocations.return_value = []
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client):
        with pytest.raises(CommandError, match='no allocations'):
            cmd.cmd_logs(_args())


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------


def test_cmd_deploy_dry_run(tmp_path):
    yaml = tmp_path / 'agent.yaml'
    yaml.write_text('x\n', encoding='utf-8')
    manifest = MagicMock()
    manifest.runtime.entrypoint = 'app:registry'
    manifest.model.alias = 'default'
    manifest.trigger.type = 'http'
    manifest.team = 'demo'
    manifest.name = 'calc'
    manifest.job_id.return_value = 'agent-demo-calc'
    registry = MagicMock()
    registry.names.return_value = ['calc']
    args = _args(dir=str(tmp_path), dry_run=True, skip_alias_check=True, image_tag='t:1')
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module'), \
            patch.object(cmd, '_known_tools', return_value=set()), \
            patch.object(cmd, '_resolve_entrypoint_registry', return_value=registry), \
            patch.object(cmd, 'validate_manifest_registry_names'), \
            patch.object(cmd, 'manifest_sha', return_value='sha'), \
            patch.object(cmd, 'render_all', return_value=[('jid', 'hcl-body')]), \
            patch.object(cmd, '_out') as out:
        assert cmd.cmd_deploy(args) == 0
        assert any('job jid' in str(c) for c in out.call_args_list)


def test_cmd_deploy_submit_and_watch(tmp_path):
    yaml = tmp_path / 'agent.yaml'
    yaml.write_text('x\n', encoding='utf-8')
    manifest = MagicMock()
    manifest.runtime.entrypoint = 'app:registry'
    manifest.model.alias = 'default'
    manifest.trigger.type = 'http'
    manifest.team = 'demo'
    manifest.name = 'calc'
    manifest.job_id.return_value = 'agent-demo-calc'
    registry = MagicMock()
    registry.names.return_value = ['calc']
    client = MagicMock()
    client.watch_deployment.return_value = ['deploying…']
    args = _args(dir=str(tmp_path), dry_run=False, skip_alias_check=False, image_tag='t:1')
    cfg = _cfg(public_url_template='http://{team}-{name}.local')
    with patch.object(cmd, '_load_cfg', return_value=cfg), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module'), \
            patch.object(cmd, '_known_tools', return_value=set()), \
            patch.object(cmd, '_resolve_entrypoint_registry', return_value=registry), \
            patch.object(cmd, 'validate_manifest_registry_names'), \
            patch.object(cmd, 'ensure_alias_or_raise', return_value=['default']), \
            patch.object(cmd, 'manifest_sha', return_value='sha'), \
            patch.object(cmd, 'render_all', return_value=[('agent-demo-calc', 'hcl')]), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_deploy(args) == 0
    client.submit_hcl.assert_called_once()
    client.watch_deployment.assert_called_once()


def test_cmd_deploy_build_image_and_errors(tmp_path):
    yaml = tmp_path / 'agent.yaml'
    yaml.write_text('x\n', encoding='utf-8')
    manifest = MagicMock()
    manifest.runtime.entrypoint = 'app:registry'
    manifest.model.alias = 'default'
    manifest.trigger.type = 'schedule'
    manifest.team = 'demo'
    manifest.name = 'calc'
    registry = MagicMock()
    registry.names.return_value = ['calc']
    args = _args(dir=str(tmp_path), image_tag=None, skip_alias_check=True, dry_run=True)
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module'), \
            patch.object(cmd, '_known_tools', return_value=set()), \
            patch.object(cmd, '_resolve_entrypoint_registry', return_value=registry), \
            patch.object(cmd, 'validate_manifest_registry_names'), \
            patch.object(cmd, 'manifest_sha', return_value='sha'), \
            patch.object(cmd, 'build_agent_image', return_value='built:1'), \
            patch.object(cmd, 'render_all', return_value=[('j', 'h')]), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_deploy(args) == 0

    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', side_effect=ManifestError('bad')):
        with pytest.raises(CommandError, match='bad'):
            cmd.cmd_deploy(args)


def test_cmd_deploy_import_and_alias_failures(tmp_path):
    yaml = tmp_path / 'agent.yaml'
    yaml.write_text('x\n', encoding='utf-8')
    manifest = MagicMock()
    manifest.runtime.entrypoint = 'app:registry'
    manifest.model.alias = 'x'
    args = _args(dir=str(tmp_path), skip_alias_check=False)
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module', side_effect=RuntimeError('boom')):
        with pytest.raises(CommandError, match='failed to import'):
            cmd.cmd_deploy(args)

    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module'), \
            patch.object(cmd, '_known_tools', return_value=set()), \
            patch.object(cmd, '_resolve_entrypoint_registry', side_effect=RuntimeError('env')), \
            patch.object(cmd, 'validate_manifest_registry_names'):
        with pytest.raises(CommandError, match='failed to resolve'):
            cmd.cmd_deploy(args)

    registry = MagicMock()
    registry.names.return_value = ['calc']
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module'), \
            patch.object(cmd, '_known_tools', return_value=set()), \
            patch.object(cmd, '_resolve_entrypoint_registry', return_value=registry), \
            patch.object(cmd, 'validate_manifest_registry_names'), \
            patch.object(cmd, 'ensure_alias_or_raise', side_effect=GatewayError('no alias')):
        with pytest.raises(CommandError, match='no alias'):
            cmd.cmd_deploy(args)


def test_cmd_deploy_no_push_forces_local(tmp_path):
    yaml = tmp_path / 'agent.yaml'
    yaml.write_text('x\n', encoding='utf-8')
    manifest = MagicMock()
    manifest.runtime.entrypoint = 'app:registry'
    manifest.model.alias = 'default'
    manifest.trigger.type = 'http'
    manifest.team = 'demo'
    manifest.name = 'calc'
    manifest.job_id.return_value = 'j'
    registry = MagicMock()
    registry.names.return_value = ['calc']
    cfg = _cfg(registry='remote.example', public_url_template='')
    args = _args(dir=str(tmp_path), no_push=True, dry_run=True, skip_alias_check=True, image_tag='t')
    with patch.object(cmd, '_load_cfg', return_value=cfg), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module'), \
            patch.object(cmd, '_known_tools', return_value=set()), \
            patch.object(cmd, '_resolve_entrypoint_registry', return_value=registry), \
            patch.object(cmd, 'validate_manifest_registry_names'), \
            patch.object(cmd, 'manifest_sha', return_value='sha'), \
            patch.object(cmd, 'render_all', return_value=[('j', 'h')]), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_deploy(args) == 0
    assert cfg.registry == 'local'


def test_cmd_deploy_remote_push_tag(tmp_path):
    yaml = tmp_path / 'agent.yaml'
    yaml.write_text('x\n', encoding='utf-8')
    manifest = MagicMock()
    manifest.runtime.entrypoint = 'app:registry'
    manifest.model.alias = 'default'
    manifest.trigger.type = 'http'
    manifest.team = 'demo'
    manifest.name = 'calc'
    manifest.job_id.return_value = 'j'
    registry = MagicMock()
    registry.names.return_value = ['calc']
    cfg = _cfg(registry='remote.example')
    args = _args(dir=str(tmp_path), dry_run=True, skip_alias_check=True, image_tag='img:1', no_push=False)
    with patch.object(cmd, '_load_cfg', return_value=cfg), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module'), \
            patch.object(cmd, '_known_tools', return_value=set()), \
            patch.object(cmd, '_resolve_entrypoint_registry', return_value=registry), \
            patch.object(cmd, 'validate_manifest_registry_names'), \
            patch.object(cmd, 'manifest_sha', return_value='sha'), \
            patch('cat_agent.platform.builder.docker_login_and_push') as push, \
            patch.object(cmd, 'render_all', return_value=[('j', 'h')]), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_deploy(args) == 0
        push.assert_called_once()


def test_cmd_deploy_registry_name_error(tmp_path):
    from cat_agent.platform.registry_check import RegistryNameError

    yaml = tmp_path / 'agent.yaml'
    yaml.write_text('x\n', encoding='utf-8')
    manifest = MagicMock()
    manifest.runtime.entrypoint = 'app:registry'
    manifest.name = 'calc'
    args = _args(dir=str(tmp_path), skip_alias_check=True)
    registry = MagicMock()
    registry.names.return_value = ['other']
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module'), \
            patch.object(cmd, '_known_tools', return_value=set()), \
            patch.object(cmd, '_resolve_entrypoint_registry', return_value=registry), \
            patch.object(
                cmd, 'validate_manifest_registry_names',
                side_effect=RegistryNameError('name mismatch'),
            ):
        with pytest.raises(CommandError, match='name mismatch'):
            cmd.cmd_deploy(args)


def test_cmd_deploy_build_error(tmp_path):
    from cat_agent.platform.builder import BuildError

    yaml = tmp_path / 'agent.yaml'
    yaml.write_text('x\n', encoding='utf-8')
    manifest = MagicMock()
    manifest.runtime.entrypoint = 'app:registry'
    manifest.model.alias = 'default'
    registry = MagicMock()
    registry.names.return_value = ['calc']
    args = _args(dir=str(tmp_path), image_tag=None, skip_alias_check=True)
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module'), \
            patch.object(cmd, '_known_tools', return_value=set()), \
            patch.object(cmd, '_resolve_entrypoint_registry', return_value=registry), \
            patch.object(cmd, 'validate_manifest_registry_names'), \
            patch.object(cmd, 'manifest_sha', return_value='sha'), \
            patch.object(cmd, 'build_agent_image', side_effect=BuildError('build fail')):
        with pytest.raises(CommandError, match='build fail'):
            cmd.cmd_deploy(args)


def test_cmd_deploy_nomad_error(tmp_path):
    yaml = tmp_path / 'agent.yaml'
    yaml.write_text('x\n', encoding='utf-8')
    manifest = MagicMock()
    manifest.runtime.entrypoint = 'app:registry'
    manifest.model.alias = 'default'
    manifest.trigger.type = 'http'
    manifest.team = 'demo'
    manifest.name = 'calc'
    registry = MagicMock()
    registry.names.return_value = ['calc']
    client = MagicMock()
    client.submit_hcl.side_effect = NomadError('reject')
    args = _args(dir=str(tmp_path), dry_run=False, skip_alias_check=True, image_tag='t')
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_require_mac_docker_network'), \
            patch.object(cmd, 'load_manifest', return_value=manifest), \
            patch.object(cmd, '_import_entrypoint_module'), \
            patch.object(cmd, '_known_tools', return_value=set()), \
            patch.object(cmd, '_resolve_entrypoint_registry', return_value=registry), \
            patch.object(cmd, 'validate_manifest_registry_names'), \
            patch.object(cmd, 'manifest_sha', return_value='sha'), \
            patch.object(cmd, 'render_all', return_value=[('j', 'h')]), \
            patch.object(cmd, '_client', return_value=client):
        with pytest.raises(CommandError, match='reject'):
            cmd.cmd_deploy(args)


# ---------------------------------------------------------------------------
# rm / rollback / build-base
# ---------------------------------------------------------------------------


def test_cmd_rm_requires_yes():
    client = MagicMock()
    client.list_agents.return_value = [_job()]
    client.allocations.return_value = []
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client):
        with pytest.raises(CommandError, match='--yes'):
            cmd.cmd_rm(_args(yes=False, team='demo'))


def test_cmd_rm_force_and_rejected():
    client = MagicMock()
    client.list_agents.return_value = [_job()]
    client.allocations.return_value = []
    client.stop.side_effect = NomadRejected('nope')
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client):
        with pytest.raises(CommandError, match='nope'):
            cmd.cmd_rm(_args(yes=True, team='demo'))


def test_cmd_rm_sibling_task_not_found():
    service = _job(jobs_mode='dispatch')
    client = MagicMock()
    client.list_agents.return_value = [service]
    client.get_job.side_effect = NomadNotFound('missing')
    client.allocations.return_value = []
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_rm(_args(yes=True, team='demo', name='calc')) == 0
        client.stop.assert_called()


def test_cmd_rollback_previous_and_to():
    client = MagicMock()
    client.list_agents.return_value = [_job()]
    client.job_versions.return_value = [
        {'Version': 2, 'ID': 'j'},
        {'Version': 1, 'ID': 'j'},
    ]
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out') as out:
        assert cmd.cmd_rollback(_args(to=None, team='demo')) == 0
        client.submit.assert_called_with({'Version': 1, 'ID': 'j'})
        assert any('version 1' in str(c) for c in out.call_args_list)

    client.submit.reset_mock()
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_rollback(_args(to=2, team='demo')) == 0
        client.submit.assert_called_with({'Version': 2, 'ID': 'j'})


def test_cmd_rollback_errors():
    client = MagicMock()
    client.list_agents.return_value = [_job()]
    client.job_versions.return_value = []
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client):
        with pytest.raises(CommandError, match='no versions'):
            cmd.cmd_rollback(_args(team='demo'))

    client.job_versions.return_value = [{'Version': 1}]
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch.object(cmd, '_client', return_value=client):
        with pytest.raises(CommandError, match='no previous'):
            cmd.cmd_rollback(_args(to=None, team='demo'))
        with pytest.raises(CommandError, match='not found'):
            cmd.cmd_rollback(_args(to=99, team='demo'))


def test_cmd_build_base_ok_and_missing():
    args = _args(no_push=True)

    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch('cat_agent.platform.commands.Path') as PathCls:
        parents = [MagicMock(), MagicMock(), MagicMock()]
        parents[2].__truediv__.return_value.__truediv__.return_value.is_file.return_value = False
        PathCls.return_value.resolve.return_value.parents = parents
        with pytest.raises(CommandError, match='not found'):
            cmd.cmd_build_base(args)

    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch('cat_agent.platform.commands.Path') as PathCls:
        df = MagicMock()
        df.is_file.return_value = True
        parents = [MagicMock(), MagicMock(), MagicMock()]
        parents[2].__truediv__.return_value.__truediv__.return_value = df
        PathCls.return_value.resolve.return_value.parents = parents
        with patch.object(cmd, 'build_base_image', return_value='base:9') as build, \
                patch.object(cmd, '_out') as out:
            assert cmd.cmd_build_base(args) == 0
            build.assert_called_once()
            out.assert_called_with('built base image base:9')

    from cat_agent.platform.builder import BuildError
    with patch.object(cmd, '_load_cfg', return_value=_cfg()), \
            patch('cat_agent.platform.commands.Path') as PathCls:
        df = MagicMock()
        df.is_file.return_value = True
        parents = [MagicMock(), MagicMock(), MagicMock()]
        parents[2].__truediv__.return_value.__truediv__.return_value = df
        PathCls.return_value.resolve.return_value.parents = parents
        with patch.object(cmd, 'build_base_image', side_effect=BuildError('fail')):
            with pytest.raises(CommandError, match='fail'):
                cmd.cmd_build_base(args)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def test_cmd_doctor_happy_local():
    cfg = _cfg(docker_network='net', consul_dns='10.0.0.1')
    client = MagicMock()
    client.status_leader.return_value = 'leader'
    client.nodes.return_value = [{'ID': 'n1'}]
    client.node.return_value = {'Drivers': {'docker': {'Healthy': True}}}
    args = _args(team='demo')
    with patch.object(cmd, '_load_cfg', return_value=cfg), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd.shutil, 'which', return_value='/usr/bin/docker'), \
            patch.object(cmd, 'resolve_gateway_via_consul_dns', return_value='1.2.3.4'), \
            patch.object(cmd, 'fetch_aliases_for_config', return_value=['default']), \
            patch.object(cmd, 'vault_team_key_exists'), \
            patch.object(cmd, 'default_config_path', return_value=Path('/tmp/c.toml')), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_doctor(args) == 0


def test_cmd_doctor_failures():
    cfg = _cfg(docker_network='', consul_dns='', registry='remote.example')
    args = _args(team='demo')
    client = MagicMock()
    client.status_leader.side_effect = NomadUnreachable('down')
    with patch.object(cmd, '_load_cfg', return_value=cfg), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd.shutil, 'which', return_value=None), \
            patch.object(cmd.sys, 'platform', 'darwin'), \
            patch.object(cmd, 'fetch_aliases_for_config', side_effect=GatewayError('gw')), \
            patch.object(cmd, 'vault_team_key_exists', side_effect=GatewayError('vault')), \
            patch.object(cmd, 'probe_registry_reachability', side_effect=RegistryError('reg')), \
            patch.object(cmd, 'vault_registry_creds_exist', side_effect=RegistryError('creds')), \
            patch.object(cmd, 'default_config_path', return_value=Path('/tmp/c.toml')), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_doctor(args) == 1


def test_cmd_doctor_remote_registry_ok_and_auth_fail():
    cfg = _cfg(registry='remote.example', docker_network='net', consul_dns='')
    client = MagicMock()
    client.status_leader.return_value = 'L'
    client.nodes.return_value = [{'ID': 'n1'}]
    client.node.return_value = {'Drivers': {'docker': {'Healthy': True}}}
    args = _args()
    with patch.object(cmd, '_load_cfg', return_value=cfg), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd.shutil, 'which', return_value='/bin/docker'), \
            patch.object(cmd, 'fetch_aliases_for_config', return_value=['a']), \
            patch.object(cmd, 'vault_team_key_exists'), \
            patch.object(cmd, 'probe_registry_reachability', return_value='ok'), \
            patch.object(cmd, 'vault_registry_creds_exist'), \
            patch.object(cmd, 'read_vault_kv_data', return_value={'username': 'u', 'password': 'p'}), \
            patch.object(cmd, 'probe_registry_auth', return_value='auth-ok'), \
            patch.object(cmd, 'default_config_path', return_value=Path('/tmp/c.toml')), \
            patch.object(cmd, '_out'):
        # docker_network set but consul_dns empty → failed
        assert cmd.cmd_doctor(args) == 1

    with patch.object(cmd, '_load_cfg', return_value=_cfg(registry='r', docker_network='n', consul_dns='1.1.1.1')), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd.shutil, 'which', return_value='/bin/docker'), \
            patch.object(cmd, 'resolve_gateway_via_consul_dns', return_value='9.9.9.9'), \
            patch.object(cmd, 'fetch_aliases_for_config', return_value=['a']), \
            patch.object(cmd, 'vault_team_key_exists'), \
            patch.object(cmd, 'probe_registry_reachability', return_value='ok'), \
            patch.object(cmd, 'vault_registry_creds_exist'), \
            patch.object(cmd, 'read_vault_kv_data', side_effect=KeyError('username')), \
            patch.object(cmd, 'default_config_path', return_value=Path('/tmp/c.toml')), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_doctor(args) == 1


def test_cmd_doctor_no_nodes_and_unhealthy_docker():
    cfg = _cfg(docker_network='n', consul_dns='1.1.1.1')
    client = MagicMock()
    client.status_leader.return_value = 'L'
    client.nodes.return_value = []
    args = _args()
    with patch.object(cmd, '_load_cfg', return_value=cfg), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd.shutil, 'which', return_value='/bin/docker'), \
            patch.object(cmd, 'resolve_gateway_via_consul_dns', return_value='1.1.1.1'), \
            patch.object(cmd, 'fetch_aliases_for_config', return_value=['a']), \
            patch.object(cmd, 'vault_team_key_exists'), \
            patch.object(cmd, 'default_config_path', return_value=Path('/tmp/c.toml')), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_doctor(args) == 1

    client.nodes.return_value = [{'ID': 'n1'}, {}]
    client.node.return_value = {'Drivers': {'docker': {'Healthy': False}}}
    with patch.object(cmd, '_load_cfg', return_value=cfg), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd.shutil, 'which', return_value='/bin/docker'), \
            patch.object(cmd, 'resolve_gateway_via_consul_dns', return_value='1.1.1.1'), \
            patch.object(cmd, 'fetch_aliases_for_config', return_value=['a']), \
            patch.object(cmd, 'vault_team_key_exists'), \
            patch.object(cmd, 'default_config_path', return_value=Path('/tmp/c.toml')), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_doctor(args) == 1


def test_cmd_doctor_nomad_error_and_consul_gateway_error():
    cfg = _cfg(docker_network='n', consul_dns='1.1.1.1')
    client = MagicMock()
    client.status_leader.side_effect = NomadError('bad')
    args = _args()
    with patch.object(cmd, '_load_cfg', return_value=cfg), \
            patch.object(cmd, '_client', return_value=client), \
            patch.object(cmd.shutil, 'which', return_value='/bin/docker'), \
            patch.object(cmd, 'resolve_gateway_via_consul_dns', side_effect=GatewayError('dns')), \
            patch.object(cmd, 'fetch_aliases_for_config', return_value=['a']), \
            patch.object(cmd, 'vault_team_key_exists'), \
            patch.object(cmd, 'default_config_path', return_value=Path('/tmp/c.toml')), \
            patch.object(cmd, '_out'):
        assert cmd.cmd_doctor(args) == 1


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------


def test_run_command_dispatch_and_errors():
    args = _args()
    with patch.object(cmd, 'cmd_ls', return_value=0) as ls:
        assert cmd.run_command('ls', args) == 0
        ls.assert_called_once_with(args)

    with patch.object(cmd, 'cmd_doctor', side_effect=CommandError('oops')), \
            patch.object(cmd, '_out') as out:
        assert cmd.run_command('doctor', args) == 1
        assert out.call_args.kwargs.get('file') is not None or True

    with patch.object(cmd, 'cmd_ls', side_effect=ConfigError('cfg')), \
            patch.object(cmd, '_out'):
        assert cmd.run_command('ls', args) == 1

    with patch('cat_agent.platform.stack.cmd_stack_up', return_value=0) as up:
        assert cmd.run_command('stack-up', args) == 0
        up.assert_called_once()


def test_import_entrypoint_and_resolve_registry(tmp_path):
    mod = tmp_path / 'ep_mod.py'
    mod.write_text('registry = object()\n', encoding='utf-8')
    with patch('importlib.import_module') as im:
        cmd._import_entrypoint_module(tmp_path, 'ep_mod:registry')
        im.assert_called_with('ep_mod')
    with patch('cat_agent.serve.factory.load_registry', return_value='REG') as lr:
        assert cmd._resolve_entrypoint_registry(tmp_path, 'ep_mod:registry') == 'REG'
        lr.assert_called_with('ep_mod:registry')
