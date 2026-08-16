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

"""Coverage tests for cat_agent.platform.gateway helpers (no live Nomad)."""

from __future__ import annotations

import io
import json
import os
import urllib.error
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cat_agent.platform.config import PlatformConfig
from cat_agent.platform import gateway as gw


def _cfg(**kwargs) -> PlatformConfig:
    base = dict(
        llm_gateway='http://llm-gateway.service.consul:4000/v1',
        vault_addr='http://127.0.0.1:8200',
        llm_credentials_path='secret/data/platform/llm',
    )
    base.update(kwargs)
    return PlatformConfig(**base)


@contextmanager
def _urlopen_payload(payload, *, status=200):
    del status  # reserved for future status-code tests
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode()
    else:
        raw = json.dumps(payload).encode()

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    buf = _Resp(raw)
    with patch('cat_agent.platform.gateway.urllib.request.urlopen', return_value=buf):
        yield

def test_gateway_models_url_variants():
    assert gw.gateway_models_url('http://x/v1') == 'http://x/v1/models'
    assert gw.gateway_models_url('http://x/v1/') == 'http://x/v1/models'
    assert gw.gateway_models_url('http://x') == 'http://x/v1/models'


def test_gateway_health_url_variants():
    assert gw.gateway_health_url('http://x/v1') == 'http://x/health/liveliness'
    assert gw.gateway_health_url('http://x') == 'http://x/health/liveliness'


def test_ensure_dev_vault_token(monkeypatch):
    monkeypatch.delenv('VAULT_TOKEN', raising=False)
    gw.ensure_dev_vault_token()
    assert os.environ['VAULT_TOKEN'] == 'root'


def test_vault_token_missing(monkeypatch):
    monkeypatch.delenv('VAULT_TOKEN', raising=False)
    with pytest.raises(gw.GatewayError, match='VAULT_TOKEN'):
        gw._vault_token(None)


def test_vault_token_explicit():
    assert gw._vault_token('  abc  ') == 'abc'


def test_is_wildcard_model_id():
    assert gw.is_wildcard_model_id('') is True
    assert gw.is_wildcard_model_id('*') is True
    assert gw.is_wildcard_model_id('*/*') is True
    assert gw.is_wildcard_model_id('openai/*') is True
    assert gw.is_wildcard_model_id('gpt-4') is False


def test_concrete_and_parse_model_ids():
    assert gw.concrete_model_ids(['*', 'a']) == ['a']
    assert gw.parse_model_ids('bad') == []
    assert gw.parse_model_ids([{'id': 'm1'}, {'name': 'm2'}]) == ['m1', 'm2']
    assert gw.parse_model_ids({'data': [{'model_name': 'x'}]}) == ['x']
    assert gw.parse_model_ids({'models': ['plain', {'model': 'dicted'}]}) == [
        'dicted',
        'plain',
    ]


def test_model_alias_matches_empty_and_provider():
    assert gw.model_alias_matches('', ['a']) is False
    assert gw.model_alias_matches('a', ['*']) is False
    assert gw.model_alias_matches('m', ['provider/m']) is True


def test_validate_manifest_alias_ok_and_errors():
    gw.validate_manifest_alias('default', ['default'])
    with pytest.raises(gw.GatewayError, match='no concrete'):
        gw.validate_manifest_alias('x', ['*'])
    with pytest.raises(gw.GatewayError, match='not available'):
        gw.validate_manifest_alias('missing', ['a', 'b'])


def test_validate_manifest_alias_truncates_preview():
    many = [f'm{i}' for i in range(25)]
    with pytest.raises(gw.GatewayError, match=r'\+5 more'):
        gw.validate_manifest_alias('nope', many)


def test_http_json_success():
    payload = {'data': [{'id': 'm'}]}
    with _urlopen_payload(payload):
        assert gw._http_json('http://example/v1/models') == payload


def test_http_json_auth_error():
    err = urllib.error.HTTPError('http://x', 401, 'no', hdrs=None, fp=io.BytesIO(b''))
    with patch('cat_agent.platform.gateway.urllib.request.urlopen', side_effect=err):
        with pytest.raises(gw.GatewayError, match='auth failed'):
            gw._http_json('http://x')


def test_http_json_other_http_error():
    err = urllib.error.HTTPError(
        'http://x', 500, 'err', hdrs=None, fp=io.BytesIO(b'detail')
    )
    with patch('cat_agent.platform.gateway.urllib.request.urlopen', side_effect=err):
        with pytest.raises(gw.GatewayError, match='HTTP 500'):
            gw._http_json('http://x')


def test_http_json_url_error_and_timeout():
    with patch(
        'cat_agent.platform.gateway.urllib.request.urlopen',
        side_effect=urllib.error.URLError('refused'),
    ):
        with pytest.raises(gw.GatewayError, match='unreachable'):
            gw._http_json('http://x')
    with patch(
        'cat_agent.platform.gateway.urllib.request.urlopen',
        side_effect=TimeoutError(),
    ):
        with pytest.raises(gw.GatewayError, match='timed out'):
            gw._http_json('http://x')


def test_http_json_non_json():
    with _urlopen_payload(b'not-json'):
        with pytest.raises(gw.GatewayError, match='non-JSON'):
            gw._http_json('http://x')


def test_read_vault_kv_data_ok(monkeypatch):
    monkeypatch.setenv('VAULT_TOKEN', 'root')
    body = {'data': {'data': {'api_key': 'k'}}}
    with _urlopen_payload(body):
        assert gw.read_vault_kv_data('http://vault', 'secret/data/x') == {'api_key': 'k'}


def test_read_vault_kv_data_errors(monkeypatch):
    monkeypatch.setenv('VAULT_TOKEN', 'root')
    for code, match in [(404, 'not found'), (403, 'denied'), (500, 'HTTP 500')]:
        err = urllib.error.HTTPError(
            'http://v', code, 'x', hdrs=None, fp=io.BytesIO(b'')
        )
        with patch('cat_agent.platform.gateway.urllib.request.urlopen', side_effect=err):
            with pytest.raises(gw.GatewayError, match=match):
                gw.read_vault_kv_data('http://vault', 'secret/data/x')


def test_read_vault_kv_data_url_error_and_bad_shape(monkeypatch):
    monkeypatch.setenv('VAULT_TOKEN', 'root')
    with patch(
        'cat_agent.platform.gateway.urllib.request.urlopen',
        side_effect=urllib.error.URLError('down'),
    ):
        with pytest.raises(gw.GatewayError, match='unreachable'):
            gw.read_vault_kv_data('http://vault', 'secret/data/x')
    with _urlopen_payload({'data': {'data': ['not-dict']}}):
        with pytest.raises(gw.GatewayError, match='unexpected shape'):
            gw.read_vault_kv_data('http://vault', 'secret/data/x')


def test_write_vault_kv_data_and_policy(monkeypatch):
    monkeypatch.setenv('VAULT_TOKEN', 'root')
    with _urlopen_payload({}):
        gw.write_vault_kv_data('http://vault', 'secret/data/x', {'a': 1})
    with _urlopen_payload({}):
        gw.write_vault_policy('http://vault', 'pol', 'path "x" {}')


def test_write_vault_errors(monkeypatch):
    monkeypatch.setenv('VAULT_TOKEN', 'root')
    err = urllib.error.HTTPError('http://v', 403, 'x', hdrs=None, fp=io.BytesIO(b''))
    with patch('cat_agent.platform.gateway.urllib.request.urlopen', side_effect=err):
        with pytest.raises(gw.GatewayError, match='denied write'):
            gw.write_vault_kv_data('http://vault', 'secret/data/x', {})
    with patch('cat_agent.platform.gateway.urllib.request.urlopen', side_effect=err):
        with pytest.raises(gw.GatewayError, match='writing policy'):
            gw.write_vault_policy('http://vault', 'p', 'hcl')


def test_vault_team_key_and_master_key(monkeypatch):
    monkeypatch.setenv('VAULT_TOKEN', 'root')
    cfg = _cfg()
    with patch.object(gw, 'read_vault_kv_data', return_value={'api_key': 'sk'}):
        gw.vault_team_key_exists(cfg, 'demo')
    with patch.object(gw, 'read_vault_kv_data', return_value={'api_key': '  '}):
        with pytest.raises(gw.GatewayError, match='api_key is empty'):
            gw.vault_team_key_exists(cfg, 'demo')
    with patch.object(
        gw, 'read_vault_kv_data', return_value={'LITELLM_MASTER_KEY': 'mk'}
    ):
        assert gw.master_key_from_vault(cfg) == 'mk'
    with patch.object(gw, 'read_vault_kv_data', return_value={}):
        with pytest.raises(gw.GatewayError, match='LITELLM_MASTER_KEY'):
            gw.master_key_from_vault(cfg)


def test_fetch_gateway_aliases_ok_and_empty():
    with patch.object(gw, '_http_json', return_value={'data': [{'id': 'a'}]}):
        assert gw.fetch_gateway_aliases('http://x/v1', api_key='k') == ['a']
    with patch.object(gw, '_http_json', return_value={'data': []}):
        with pytest.raises(gw.GatewayError, match='no models'):
            gw.fetch_gateway_aliases('http://x/v1')


def test_fetch_ollama_model_ids():
    with pytest.raises(gw.GatewayError, match='empty'):
        gw.fetch_ollama_model_ids('')
    with patch.object(
        gw, '_http_json', return_value={'data': [{'id': 'llama'}]}
    ):
        assert gw.fetch_ollama_model_ids('http://ollama/v1') == ['llama']
    with patch.object(
        gw,
        '_http_json',
        side_effect=[gw.GatewayError('fail'), {'models': [{'name': 'q'}]}],
    ):
        assert gw.fetch_ollama_model_ids('http://ollama') == ['q']
    with patch.object(gw, '_http_json', side_effect=gw.GatewayError('all fail')):
        with pytest.raises(gw.GatewayError, match='all fail'):
            gw.fetch_ollama_model_ids('http://ollama')
    with patch.object(gw, '_http_json', return_value={'data': [{'id': '*'}]}):
        with pytest.raises(gw.GatewayError, match='no models'):
            gw.fetch_ollama_model_ids('http://ollama/v1')


def test_fetch_ollama_models_for_config():
    cfg = _cfg()
    with patch.object(
        gw,
        'read_vault_kv_data',
        return_value={'OLLAMA_API_BASE': 'http://o/v1', 'OLLAMA_API_KEY': 'k'},
    ):
        with patch.object(gw, 'fetch_ollama_model_ids', return_value=['m']) as fetch:
            assert gw.fetch_ollama_models_for_config(cfg) == ['m']
            fetch.assert_called_once_with('http://o/v1', api_key='k')
    with patch.object(gw, 'read_vault_kv_data', return_value={}):
        with pytest.raises(gw.GatewayError, match='OLLAMA_API_BASE'):
            gw.fetch_ollama_models_for_config(cfg)


def test_dns_query_a_minimal(monkeypatch):
    # Build a tiny DNS response with one A record for 1.2.3.4
    # question for a.b, answer with pointer + A
    qname = b'\x01a\x01b\x00'
    header = b'\xca\xfe\x81\x80\x00\x01\x00\x01\x00\x00\x00\x00'
    question = qname + b'\x00\x01\x00\x01'
    answer = b'\xc0\x0c\x00\x01\x00\x01\x00\x00\x00\x3c\x00\x04\x01\x02\x03\x04'
    packet = header + question + answer

    class FakeSock:
        def settimeout(self, t):
            pass

        def sendto(self, data, addr):
            pass

        def recvfrom(self, n):
            return packet, ('127.0.0.1', 53)

        def close(self):
            pass

    monkeypatch.setattr(gw.socket, 'socket', lambda *a, **k: FakeSock())
    assert gw._dns_query_a('a.b', '127.0.0.1') == ['1.2.3.4']


def test_resolve_gateway_via_consul_dns_requires_dns():
    with pytest.raises(gw.GatewayError, match='consul_dns'):
        gw.resolve_gateway_via_consul_dns('')


def test_resolve_gateway_via_consul_dns_host_udp(monkeypatch):
    monkeypatch.setattr(gw.shutil, 'which', lambda *_: None)
    monkeypatch.setattr(gw, '_dns_query_a', lambda *a, **k: ['10.0.0.1'])
    assert gw.resolve_gateway_via_consul_dns('10.32.0.2') == '10.0.0.1'


def test_resolve_gateway_via_consul_dns_empty_addrs(monkeypatch):
    monkeypatch.setattr(gw.shutil, 'which', lambda *_: None)
    monkeypatch.setattr(gw, '_dns_query_a', lambda *a, **k: [])
    with pytest.raises(gw.GatewayError, match='no A record'):
        gw.resolve_gateway_via_consul_dns('10.32.0.2')


def test_resolve_gateway_via_consul_dns_docker(monkeypatch):
    monkeypatch.setattr(gw.shutil, 'which', lambda cmd: '/bin/docker' if cmd == 'docker' else None)
    proc = SimpleNamespace(returncode=0, stdout='10.1.2.3 host\n', stderr='')
    monkeypatch.setattr(gw.subprocess, 'run', lambda *a, **k: proc)
    assert (
        gw.resolve_gateway_via_consul_dns('10.32.0.2', docker_network='net')
        == '10.1.2.3'
    )


def test_resolve_gateway_via_consul_dns_docker_fail(monkeypatch):
    monkeypatch.setattr(gw.shutil, 'which', lambda cmd: '/bin/docker' if cmd == 'docker' else None)
    proc = SimpleNamespace(returncode=1, stdout='', stderr='nxdomain')
    monkeypatch.setattr(gw.subprocess, 'run', lambda *a, **k: proc)
    with pytest.raises(gw.GatewayError, match='did not resolve'):
        gw.resolve_gateway_via_consul_dns('10.32.0.2', docker_network='net')


def test_fetch_gateway_aliases_reachable_host_fallback():
    cfg = _cfg(docker_network='', consul_dns='')
    with patch.object(gw, 'fetch_gateway_aliases', return_value=['a']) as fetch:
        assert gw.fetch_gateway_aliases_reachable(cfg, api_key='k') == ['a']
        fetch.assert_called_once()


def test_fetch_gateway_aliases_reachable_consul_rewrite():
    cfg = _cfg(docker_network='', consul_dns='')
    with patch.object(
        gw,
        'fetch_gateway_aliases',
        side_effect=[gw.GatewayError('first'), ['alt']],
    ) as fetch:
        out = gw.fetch_gateway_aliases_reachable(cfg, api_key='k')
    assert out == ['alt']
    assert '127.0.0.1' in fetch.call_args_list[1].args[0]


def test_fetch_aliases_for_config_and_ensure_alias():
    cfg = _cfg()
    with patch.object(gw, 'master_key_from_vault', return_value='mk'):
        with patch.object(
            gw, 'fetch_gateway_aliases_reachable', return_value=['default']
        ):
            assert gw.fetch_aliases_for_config(cfg) == ['default']
    with patch.object(gw, 'fetch_aliases_for_config', return_value=['default']):
        assert gw.ensure_alias_or_raise(cfg, 'default') == ['default']
    with patch.object(gw, 'fetch_aliases_for_config', return_value=['*']):
        with patch.object(gw, 'fetch_ollama_models_for_config', return_value=['real']):
            assert gw.ensure_alias_or_raise(cfg, 'real') == ['real']
