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

"""Deploy-time model existence checks and doctor gateway checks (no live network)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cat_agent.platform.commands import CommandError, cmd_deploy, cmd_doctor
from cat_agent.platform.config import PlatformConfig
from cat_agent.platform.gateway import (
    GatewayError,
    concrete_model_ids,
    ensure_alias_or_raise,
    fetch_gateway_aliases,
    fetch_ollama_model_ids,
    model_alias_matches,
    parse_model_ids,
    resolve_gateway_via_consul_dns,
    validate_manifest_alias,
)


def _cfg(**kwargs) -> PlatformConfig:
    base = dict(
        llm_gateway='http://llm-gateway.service.consul:4000/v1',
        llm_credentials_path='secret/data/platform/llm',
        vault_addr='http://127.0.0.1:8200',
        docker_network='nomad_deploy_hashicorp',
        consul_dns='10.32.0.2',
        registry='local',
    )
    base.update(kwargs)
    return PlatformConfig(**base)


def test_parse_model_ids_openai_shape():
    assert parse_model_ids(
        {'data': [{'id': 'default'}, {'id': 'smart'}, {'id': 'fast'}]}
    ) == ['default', 'fast', 'smart']


def test_parse_model_ids_ollama_tags():
    assert parse_model_ids(
        {'models': [{'name': 'minimax-m3:cloud'}, {'name': 'qwen3:1.7b'}]}
    ) == ['minimax-m3:cloud', 'qwen3:1.7b']


def test_validate_unknown_alias_lists_valid():
    with pytest.raises(
        GatewayError,
        match="model.alias 'nope' is not available from the LLM backend; "
        'valid: default, smart',
    ):
        validate_manifest_alias('nope', ['default', 'smart'])


def test_validate_rejects_wildcard_only_list():
    with pytest.raises(GatewayError, match='no concrete models'):
        validate_manifest_alias('minimax-m3:cloud', ['*'])


def test_model_alias_matches_latest_variants():
    assert model_alias_matches('qwen3', ['qwen3:latest'])
    assert model_alias_matches('qwen3:latest', ['qwen3'])
    assert model_alias_matches('openai/minimax-m3:cloud', ['minimax-m3:cloud']) is False
    assert model_alias_matches('minimax-m3:cloud', ['openai/minimax-m3:cloud'])


def test_concrete_model_ids_filters_wildcards():
    assert concrete_model_ids(['*', 'openai/*', 'minimax-m3:cloud']) == [
        'minimax-m3:cloud'
    ]


def test_fetch_gateway_unreachable_message():
    import urllib.error

    with patch(
        'cat_agent.platform.gateway.urllib.request.urlopen',
        side_effect=urllib.error.URLError('connection refused'),
    ):
        with pytest.raises(GatewayError, match='unreachable'):
            fetch_gateway_aliases(
                'http://127.0.0.1:9/v1',
                api_key='sk-test',
                timeout=1,
            )


def test_fetch_ollama_model_ids_falls_back_to_api_tags(monkeypatch):
    calls: list[str] = []

    def fake_http(url, *, headers=None, timeout=10.0):
        calls.append(url)
        if url.endswith('/v1/models'):
            raise GatewayError(f'LLM gateway unreachable at {url}: refused')
        assert url.endswith('/api/tags')
        return {'models': [{'name': 'llama3:latest'}]}

    monkeypatch.setattr('cat_agent.platform.gateway._http_json', fake_http)
    assert fetch_ollama_model_ids('http://127.0.0.1:11434') == ['llama3:latest']
    assert calls == [
        'http://127.0.0.1:11434/v1/models',
        'http://127.0.0.1:11434/api/tags',
    ]


def test_ensure_alias_fail_closed_when_gateway_down():
    cfg = _cfg()
    with patch(
        'cat_agent.platform.gateway.fetch_aliases_for_config',
        side_effect=GatewayError(
            'LLM gateway unreachable at http://llm-gateway.service.consul:4000/v1/models: '
            'connection refused'
        ),
    ):
        with pytest.raises(GatewayError, match='unreachable'):
            ensure_alias_or_raise(cfg, 'default')


def test_ensure_alias_falls_back_to_ollama_when_gateway_is_wildcard_only():
    cfg = _cfg()
    with patch(
        'cat_agent.platform.gateway.fetch_aliases_for_config',
        return_value=['*'],
    ):
        with patch(
            'cat_agent.platform.gateway.fetch_ollama_models_for_config',
            return_value=['minimax-m3:cloud', 'qwen3:1.7b'],
        ) as ollama:
            got = ensure_alias_or_raise(cfg, 'minimax-m3:cloud')
    assert got == ['minimax-m3:cloud', 'qwen3:1.7b']
    ollama.assert_called_once()


def test_ensure_alias_rejects_unknown_after_ollama_fallback():
    cfg = _cfg()
    with patch(
        'cat_agent.platform.gateway.fetch_aliases_for_config',
        return_value=['*'],
    ):
        with patch(
            'cat_agent.platform.gateway.fetch_ollama_models_for_config',
            return_value=['qwen3:1.7b'],
        ):
            with pytest.raises(GatewayError, match="model.alias 'nope'"):
                ensure_alias_or_raise(cfg, 'nope')


def _fake_entrypoint_registry(*_a, **_k):
    """Deploy name-check needs a registry exposing the manifest agent name."""
    return SimpleNamespace(names=lambda: ['calc'])


def _deploy_cfg(tmp_path) -> str:
    """Hermetic platform config so macOS deploy does not depend on sibling stack."""
    cfg_path = tmp_path / 'platform.toml'
    cfg_path.write_text(
        '[platform]\n'
        'registry = "local"\n'
        'docker_network = "nomad_deploy_hashicorp"\n'
        'consul_dns = "10.32.0.2"\n'
        'llm_gateway = "http://llm-gateway.service.consul:4000/v1"\n'
        'vault_addr = "http://127.0.0.1:8200"\n',
        encoding='utf-8',
    )
    return str(cfg_path)


def test_deploy_fails_when_model_missing(tmp_path, monkeypatch):
    (tmp_path / 'agent.yaml').write_text(
        'name: calc\nteam: demo\nruntime:\n  entrypoint: app:registry\n'
        'trigger:\n  type: http\nmodel:\n  alias: nope-model\n',
        encoding='utf-8',
    )
    (tmp_path / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')
    args = SimpleNamespace(
        config=_deploy_cfg(tmp_path),
        nomad_addr=None,
        registry='local',
        dir=str(tmp_path),
        dry_run=True,
        image_tag='demo/calc:test',
        no_push=False,
        skip_alias_check=False,
    )
    monkeypatch.setattr(
        'cat_agent.platform.commands._known_tools', lambda: set()
    )
    monkeypatch.setattr(
        'cat_agent.platform.commands._resolve_entrypoint_registry',
        _fake_entrypoint_registry,
    )
    monkeypatch.setattr(
        'cat_agent.platform.commands.ensure_alias_or_raise',
        lambda cfg, alias: (_ for _ in ()).throw(
            GatewayError(
                "model.alias 'nope-model' is not available from the LLM backend; "
                'valid: minimax-m3:cloud'
            )
        ),
    )
    with pytest.raises(CommandError, match='nope-model'):
        cmd_deploy(args)


def test_deploy_fails_closed_on_unreachable_gateway(tmp_path, monkeypatch):
    (tmp_path / 'agent.yaml').write_text(
        'name: calc\nteam: demo\nruntime:\n  entrypoint: app:registry\n'
        'trigger:\n  type: http\nmodel:\n  alias: default\n',
        encoding='utf-8',
    )
    (tmp_path / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')
    args = SimpleNamespace(
        config=_deploy_cfg(tmp_path),
        nomad_addr=None,
        registry='local',
        dir=str(tmp_path),
        dry_run=True,
        image_tag='demo/calc:test',
        no_push=False,
        skip_alias_check=False,
    )
    monkeypatch.setattr(
        'cat_agent.platform.commands._known_tools', lambda: set()
    )
    monkeypatch.setattr(
        'cat_agent.platform.commands._resolve_entrypoint_registry',
        _fake_entrypoint_registry,
    )
    monkeypatch.setattr(
        'cat_agent.platform.commands.ensure_alias_or_raise',
        lambda cfg, alias: (_ for _ in ()).throw(
            GatewayError('LLM gateway unreachable at http://x:4000/v1/models: timed out')
        ),
    )
    with pytest.raises(CommandError, match='unreachable'):
        cmd_deploy(args)


def test_deploy_skip_alias_check_allows_offline(tmp_path, monkeypatch):
    (tmp_path / 'agent.yaml').write_text(
        'name: calc\nteam: demo\nruntime:\n  entrypoint: app:registry\n'
        'trigger:\n  type: http\nmodel:\n  alias: nonexistent\n',
        encoding='utf-8',
    )
    (tmp_path / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')
    args = SimpleNamespace(
        config=_deploy_cfg(tmp_path),
        nomad_addr=None,
        registry='local',
        dir=str(tmp_path),
        dry_run=True,
        image_tag='demo/calc:test',
        no_push=False,
        skip_alias_check=True,
    )
    monkeypatch.setattr(
        'cat_agent.platform.commands._known_tools', lambda: set()
    )
    monkeypatch.setattr(
        'cat_agent.platform.commands._resolve_entrypoint_registry',
        _fake_entrypoint_registry,
    )

    def boom(*_a, **_k):
        raise AssertionError('model check must not run when skipped')

    monkeypatch.setattr('cat_agent.platform.commands.ensure_alias_or_raise', boom)
    assert cmd_deploy(args) == 0


def test_doctor_reports_stale_consul_dns(monkeypatch, tmp_path):
    cfg_path = tmp_path / 'c.toml'
    cfg_path.write_text(
        '[platform]\n'
        'registry = "local"\n'
        'nomad_addr = "http://127.0.0.1:9"\n'
        'docker_network = "nomad_deploy_hashicorp"\n'
        'consul_dns = "10.32.0.2"\n'
        'llm_gateway = "http://llm-gateway.service.consul:4000/v1"\n'
        'vault_addr = "http://127.0.0.1:8200"\n',
        encoding='utf-8',
    )
    args = SimpleNamespace(
        config=str(cfg_path),
        nomad_addr=None,
        registry=None,
        team='demo',
    )
    lines: list[str] = []

    def capture(msg: str, file=None):
        lines.append(msg)

    monkeypatch.setattr('cat_agent.platform.commands._out', capture)
    monkeypatch.setattr(
        'cat_agent.platform.commands.shutil.which', lambda *_a, **_k: '/bin/docker'
    )

    from cat_agent.platform.nomad import NomadUnreachable

    with patch('cat_agent.platform.commands.NomadClient') as Cls:
        Cls.return_value.status_leader.side_effect = NomadUnreachable('Nomad unreachable')
        with patch(
            'cat_agent.platform.commands.resolve_gateway_via_consul_dns',
            side_effect=GatewayError(
                'consul_dns 10.32.0.2 did not resolve llm-gateway.service.consul '
                'on network nomad_deploy_hashicorp'
            ),
        ):
            with patch(
                'cat_agent.platform.commands.fetch_aliases_for_config',
                side_effect=GatewayError('LLM gateway unreachable'),
            ):
                with patch(
                    'cat_agent.platform.commands.vault_team_key_exists',
                    side_effect=GatewayError('Vault secret not found'),
                ):
                    rc = cmd_doctor(args)
    joined = '\n'.join(lines)
    assert rc == 1
    assert 'consul_dns 10.32.0.2 did not resolve llm-gateway.service.consul' in joined


def test_doctor_ok_when_consul_dns_resolves(monkeypatch, tmp_path):
    cfg_path = tmp_path / 'c.toml'
    cfg_path.write_text(
        '[platform]\n'
        'registry = "local"\n'
        'nomad_addr = "http://127.0.0.1:4646"\n'
        'docker_network = "nomad_deploy_hashicorp"\n'
        'consul_dns = "10.32.0.2"\n'
        'llm_gateway = "http://llm-gateway.service.consul:4000/v1"\n'
        'vault_addr = "http://127.0.0.1:8200"\n',
        encoding='utf-8',
    )
    args = SimpleNamespace(
        config=str(cfg_path),
        nomad_addr=None,
        registry=None,
        team='demo',
    )
    lines: list[str] = []
    monkeypatch.setattr(
        'cat_agent.platform.commands._out', lambda msg, file=None: lines.append(msg)
    )
    monkeypatch.setattr(
        'cat_agent.platform.commands.shutil.which', lambda *_a, **_k: '/bin/docker'
    )

    class FakeClient:
        def status_leader(self):
            return '127.0.0.1:4647'

        def nodes(self):
            return [{'ID': 'n1'}]

        def node(self, _nid):
            return {'Drivers': {'docker': {'Healthy': True}}}

    with patch('cat_agent.platform.commands.NomadClient', return_value=FakeClient()):
        with patch(
            'cat_agent.platform.commands.resolve_gateway_via_consul_dns',
            return_value='10.32.0.10',
        ):
            with patch(
                'cat_agent.platform.commands.fetch_aliases_for_config',
                return_value=['minimax-m3:cloud', 'qwen3:1.7b'],
            ):
                with patch(
                    'cat_agent.platform.commands.vault_team_key_exists',
                    return_value=None,
                ):
                    rc = cmd_doctor(args)
    joined = '\n'.join(lines)
    assert rc == 0
    assert 'llm-gateway.service.consul → 10.32.0.10' in joined
    assert 'aliases=minimax-m3:cloud,qwen3:1.7b' in joined


def test_resolve_consul_dns_requires_value():
    with pytest.raises(GatewayError, match='consul_dns is not set'):
        resolve_gateway_via_consul_dns('')
