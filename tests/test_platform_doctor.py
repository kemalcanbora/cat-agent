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

"""Doctor registry mode / reachability / TLS / auth messages (no network)."""

from __future__ import annotations

import ssl
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch

from cat_agent.platform import commands as platform_commands
from cat_agent.platform.config import PlatformConfig
from cat_agent.platform.gateway import GatewayError
from cat_agent.platform.nomad import NomadUnreachable
from cat_agent.platform.registry_probe import (
    RegistryError,
    probe_registry_auth,
    probe_registry_reachability,
)


def _args(cfg_path):
    return SimpleNamespace(
        config=str(cfg_path), nomad_addr=None, registry=None, team='demo'
    )


@contextmanager
def _patch_commands(**kwargs) -> Iterator[None]:
    with ExitStack() as stack:
        for name, value in kwargs.items():
            if isinstance(value, dict):
                stack.enter_context(patch.object(platform_commands, name, **value))
            else:
                stack.enter_context(patch.object(platform_commands, name, value))
        yield


def _capture_doctor(monkeypatch, cfg_text: str, tmp_path, **patches):
    cfg_path = tmp_path / 'c.toml'
    cfg_path.write_text(cfg_text, encoding='utf-8')
    lines: list[str] = []
    monkeypatch.setattr(
        platform_commands, '_out', lambda msg, file=None: lines.append(msg)
    )
    monkeypatch.setattr(platform_commands.shutil, 'which', lambda _: '/bin/docker')
    with _patch_commands(**patches):
        rc = platform_commands.cmd_doctor(_args(cfg_path))
    return rc, '\n'.join(lines)


def test_doctor_reports_local_mode(monkeypatch, tmp_path):
    monkeypatch.delenv('CAT_AGENT_REGISTRY', raising=False)
    rc, joined = _capture_doctor(
        monkeypatch,
        '[platform]\nregistry = "local"\nnomad_addr = "http://127.0.0.1:9"\n',
        tmp_path,
        NomadClient={'side_effect': NomadUnreachable('Nomad unreachable')},
        fetch_aliases_for_config={
            'side_effect': GatewayError('LLM gateway unreachable')
        },
        vault_team_key_exists={'side_effect': GatewayError('Vault secret not found')},
    )
    assert 'registry: local (images are not pushed; only this node can run them)' in joined
    assert 'registry mode: local' in joined
    assert rc == 1


_REMOTE_CFG = (
    '[platform]\n'
    'registry = "127.0.0.1:5001"\n'
    'nomad_addr = "http://127.0.0.1:4646"\n'
    'docker_network = "cat-agent-stack_hashicorp"\n'
    'consul_dns = "10.32.0.2"\n'
)


def test_doctor_reports_remote_mode_and_ok(monkeypatch, tmp_path):
    monkeypatch.delenv('CAT_AGENT_REGISTRY', raising=False)
    monkeypatch.delenv('CAT_AGENT_DOCKER_NETWORK', raising=False)
    monkeypatch.delenv('CAT_AGENT_CONSUL_DNS', raising=False)

    class FakeNomad:
        def status_leader(self):
            return '127.0.0.1:4647'

        def nodes(self):
            return [{'ID': 'n1'}]

        def node(self, nid):
            return {'Drivers': {'docker': {'Healthy': True}}}

    rc, joined = _capture_doctor(
        monkeypatch,
        _REMOTE_CFG,
        tmp_path,
        NomadClient={'return_value': FakeNomad()},
        fetch_aliases_for_config={'return_value': ['default']},
        vault_team_key_exists={'return_value': None},
        resolve_gateway_via_consul_dns={'return_value': '10.32.0.3'},
        probe_registry_reachability={
            'return_value': (
                'registry: reachable at http://127.0.0.1:5001 '
                '(HTTP 401 — auth required)'
            )
        },
        vault_registry_creds_exist={'return_value': None},
        read_vault_kv_data={
            'return_value': {'username': 'zot-pull', 'password': 'x'}
        },
        probe_registry_auth={'return_value': 'registry auth: ok (HTTP 200)'},
    )
    assert 'registry mode: remote (127.0.0.1:5001)' in joined
    assert 'registry: reachable' in joined
    assert 'vault registry pull: ok' in joined
    assert 'vault registry push: ok' in joined
    assert 'registry auth: ok' in joined
    assert rc == 0


def test_probe_unreachable_sentence():
    cfg = PlatformConfig(registry='127.0.0.1:5001')
    import urllib.error

    with patch(
        'cat_agent.platform.registry_probe.urllib.request.urlopen',
        side_effect=urllib.error.URLError('Connection refused'),
    ):
        try:
            probe_registry_reachability(cfg)
            assert False, 'expected RegistryError'
        except RegistryError as exc:
            msg = str(exc)
            assert 'registry unreachable' in msg
            assert '127.0.0.1:5001' in msg
            assert 'Traceback' not in msg


def test_probe_tls_failure_sentence():
    cfg = PlatformConfig(registry='registry.example:5000')
    import urllib.error

    with patch(
        'cat_agent.platform.registry_probe.urllib.request.urlopen',
        side_effect=urllib.error.URLError(ssl.SSLError('CERTIFICATE_VERIFY_FAILED')),
    ):
        try:
            probe_registry_reachability(cfg)
            assert False, 'expected RegistryError'
        except RegistryError as exc:
            msg = str(exc)
            assert 'TLS failure' in msg
            assert 'Traceback' not in msg
            assert 'insecure-registries' in msg or 'trusted CA' in msg


def test_probe_auth_failure_sentence():
    cfg = PlatformConfig(registry='127.0.0.1:5001')
    import urllib.error

    err = urllib.error.HTTPError(
        'http://127.0.0.1:5001/v2/', 401, 'Unauthorized', hdrs=None, fp=None
    )
    with patch(
        'cat_agent.platform.registry_probe.urllib.request.urlopen',
        side_effect=err,
    ):
        try:
            probe_registry_auth(cfg, username='u', password='p')
            assert False, 'expected RegistryError'
        except RegistryError as exc:
            msg = str(exc)
            assert 'registry auth failed' in msg
            assert 'HTTP 401' in msg
            assert 'secret/data/platform/registry' in msg
            assert 'Traceback' not in msg


def test_doctor_prints_distinct_registry_failures(monkeypatch, tmp_path):
    monkeypatch.delenv('CAT_AGENT_REGISTRY', raising=False)
    monkeypatch.delenv('CAT_AGENT_DOCKER_NETWORK', raising=False)
    monkeypatch.delenv('CAT_AGENT_CONSUL_DNS', raising=False)

    class FakeNomad:
        def status_leader(self):
            return 'x'

        def nodes(self):
            return [{'ID': 'n1'}]

        def node(self, nid):
            return {'Drivers': {'docker': {'Healthy': True}}}

    rc, joined = _capture_doctor(
        monkeypatch,
        '[platform]\n'
        'registry = "registry.example:5000"\n'
        'nomad_addr = "http://127.0.0.1:4646"\n'
        'docker_network = "cat-agent-stack_hashicorp"\n'
        'consul_dns = "10.32.0.2"\n',
        tmp_path,
        NomadClient={'return_value': FakeNomad()},
        fetch_aliases_for_config={'return_value': ['default']},
        vault_team_key_exists={'return_value': None},
        resolve_gateway_via_consul_dns={'return_value': '10.32.0.3'},
        probe_registry_reachability={
            'side_effect': RegistryError(
                'registry TLS failure talking to https://registry.example:5000: '
                'CERTIFICATE_VERIFY_FAILED. Use HTTP + insecure-registries for this '
                'stack, or install a trusted CA.'
            )
        },
        vault_registry_creds_exist={
            'side_effect': RegistryError('vault registry pull: missing'),
        },
    )
    assert 'registry mode: remote (registry.example:5000)' in joined
    assert 'TLS failure' in joined
    assert 'Traceback' not in joined
    assert rc == 1
