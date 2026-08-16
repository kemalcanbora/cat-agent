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

"""Coverage tests for cat_agent.platform.stack (mocked compose/subprocess)."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.platform.config import PlatformConfig
from cat_agent.platform import stack as st


def _cfg(**kwargs) -> PlatformConfig:
    base = dict(
        vault_addr='http://127.0.0.1:8200',
        llm_credentials_path='secret/data/platform/llm',
    )
    base.update(kwargs)
    return PlatformConfig(**base)


def test_ensure_openai_compat_base():
    assert st._ensure_openai_compat_base('') == ''
    assert st._ensure_openai_compat_base('http://o/v1') == 'http://o/v1'
    assert st._ensure_openai_compat_base('http://o') == 'http://o/v1'


def test_resolve_llm_seed_defaults():
    fields = st.resolve_llm_seed({})
    assert fields['LITELLM_MASTER_KEY'] == st.DEFAULT_MASTER_KEY
    assert 'host.docker.internal' in fields['OLLAMA_API_BASE']
    assert fields['OPENAI_API_KEY'] == fields['OLLAMA_API_KEY']


def test_resolve_llm_seed_with_openai_key():
    fields = st.resolve_llm_seed({
        'OPENAI_API_KEY': 'sk-openai',
        'LITELLM_MASTER_KEY': 'mk',
        'OLLAMA_API_KEY': 'ok',
        'OLLAMA_API_BASE': 'http://ollama',
    })
    assert fields['LITELLM_MASTER_KEY'] == 'mk'
    assert fields['OPENAI_API_KEY'] == 'sk-openai'
    assert fields['OPENAI_API_BASE'] == 'https://api.openai.com/v1'


def test_tpm_rpm_defaults_and_overrides():
    tpm, rpm = st._tpm_rpm(max_tokens_per_day=2_000_000, tpm_limit=None, rpm_limit=None)
    assert tpm == 2_000_000 // (24 * 60)
    assert rpm == max(1, tpm // 500)
    assert st._tpm_rpm(
        max_tokens_per_day=1, tpm_limit=100, rpm_limit=5
    ) == (100, 5)


def test_compose_argv_with_profiles(tmp_path):
    compose = tmp_path / 'docker-compose.yml'
    compose.write_text('services: {}\n', encoding='utf-8')
    argv = st.compose_argv(tmp_path, profiles=['registry', '', ' gpu '])
    assert argv[:3] == ['docker', 'compose', '-f']
    assert '--profile' in argv
    assert 'registry' in argv
    assert 'gpu' in argv


def test_resolve_stack_dir_explicit(tmp_path):
    (tmp_path / 'docker-compose.yml').write_text('x\n', encoding='utf-8')
    assert st.resolve_stack_dir(str(tmp_path)) == tmp_path.resolve()


def test_resolve_stack_dir_env(tmp_path, monkeypatch):
    (tmp_path / 'docker-compose.yml').write_text('x\n', encoding='utf-8')
    monkeypatch.setenv('CAT_AGENT_STACK_DIR', str(tmp_path))
    assert st.resolve_stack_dir(None) == tmp_path.resolve()


def test_resolve_stack_dir_cwd(tmp_path, monkeypatch):
    (tmp_path / 'docker-compose.yml').write_text('x\n', encoding='utf-8')
    monkeypatch.delenv('CAT_AGENT_STACK_DIR', raising=False)
    monkeypatch.chdir(tmp_path)
    assert st.resolve_stack_dir(None) == tmp_path.resolve()


def test_resolve_stack_dir_missing(tmp_path, monkeypatch):
    monkeypatch.delenv('CAT_AGENT_STACK_DIR', raising=False)
    monkeypatch.chdir(tmp_path)
    with patch.object(st, '_sibling_stack_dir', return_value=None):
        with pytest.raises(st.StackError, match='no stack directory'):
            st.resolve_stack_dir(None)


def test_resolve_stack_dir_no_compose(tmp_path):
    with pytest.raises(st.StackError, match='no docker-compose.yml'):
        st.resolve_stack_dir(str(tmp_path))


def test_ensure_host_data_dirs(tmp_path, monkeypatch):
    monkeypatch.delenv('HOST_NOMAD_DATA', raising=False)
    monkeypatch.delenv('HOST_ZOT_DATA', raising=False)
    out = st.ensure_host_data_dirs(tmp_path)
    assert Path(out['HOST_NOMAD_DATA']).is_dir()
    assert Path(out['HOST_ZOT_DATA']).is_dir()
    assert os.environ['HOST_NOMAD_DATA'] == out['HOST_NOMAD_DATA']


def test_load_stack_env_missing_and_present(tmp_path, monkeypatch):
    assert st.load_stack_env(tmp_path) is None
    env_file = tmp_path / '.env'
    env_file.write_text('STACK_TEST_VAR=1\n', encoding='utf-8')
    monkeypatch.delenv('STACK_TEST_VAR', raising=False)
    with patch('dotenv.load_dotenv') as load:
        path = st.load_stack_env(tmp_path)
    assert path == env_file
    load.assert_called_once()


def test_run_compose_success_and_failure(tmp_path, monkeypatch):
    (tmp_path / 'docker-compose.yml').write_text('x\n', encoding='utf-8')
    monkeypatch.setattr(st, 'ensure_host_data_dirs', lambda d: {})
    with patch('cat_agent.platform.stack.subprocess.run') as run:
        run.return_value = SimpleNamespace(returncode=0)
        assert st.run_compose(tmp_path, ['ps']) == 0
        run.return_value = SimpleNamespace(returncode=7)
        with pytest.raises(st.StackError, match='exit 7'):
            st.run_compose(tmp_path, ['ps'])
        assert st.run_compose(tmp_path, ['ps'], check=False) == 7


def test_wait_vault_ready(monkeypatch):
    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        'cat_agent.platform.stack.urllib.request.urlopen',
        lambda *a, **k: Resp(),
    )
    monkeypatch.setattr(st.time, 'sleep', lambda *_: None)
    st.wait_vault('http://vault', timeout_s=1)


def test_wait_vault_timeout(monkeypatch):
    def boom(*_a, **_k):
        raise OSError('down')

    monkeypatch.setattr(
        'cat_agent.platform.stack.urllib.request.urlopen',
        boom,
    )
    times = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(st.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(st.time, 'sleep', lambda *_: None)
    with pytest.raises(st.StackError, match='Vault not ready'):
        st.wait_vault('http://vault', timeout_s=1)


def test_wait_gateway_ready_and_timeout(monkeypatch):
    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        'cat_agent.platform.stack.urllib.request.urlopen',
        lambda *a, **k: Resp(),
    )
    monkeypatch.setattr(st.time, 'sleep', lambda *_: None)
    st.wait_gateway('http://gw/v1', timeout_s=1)

    def boom(*_a, **_k):
        raise OSError('down')

    monkeypatch.setattr(
        'cat_agent.platform.stack.urllib.request.urlopen',
        boom,
    )
    times = iter([0.0, 0.0, 1000.0])
    monkeypatch.setattr(st.time, 'monotonic', lambda: next(times))
    with pytest.raises(st.StackError, match='LiteLLM not healthy'):
        st.wait_gateway('http://gw', timeout_s=1)


def test_http_json_post_ok_and_errors(monkeypatch):
    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(
        'cat_agent.platform.stack.urllib.request.urlopen',
        lambda *a, **k: Resp(json.dumps({'ok': True}).encode()),
    )
    assert st._http_json_post('http://x', {'a': 1}) == {'ok': True}

    import urllib.error

    err = urllib.error.HTTPError(
        'http://x', 400, 'bad', hdrs=None, fp=io.BytesIO(b'nope')
    )

    def raise_http(*_a, **_k):
        raise err

    monkeypatch.setattr(
        'cat_agent.platform.stack.urllib.request.urlopen',
        raise_http,
    )
    with pytest.raises(st.StackError, match='HTTP 400'):
        st._http_json_post('http://x', {})

    def raise_url(*_a, **_k):
        raise urllib.error.URLError('refused')

    monkeypatch.setattr(
        'cat_agent.platform.stack.urllib.request.urlopen',
        raise_url,
    )
    with pytest.raises(st.StackError, match='unreachable'):
        st._http_json_post('http://x', {})


def test_seed_llm_vault(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(st, 'wait_vault', lambda *a, **k: None)
    monkeypatch.setattr(st, 'write_vault_kv_data', MagicMock())
    monkeypatch.setattr(st, '_out', lambda *a, **k: None)
    fields = st.seed_llm_vault(cfg, env={'LITELLM_MASTER_KEY': 'mk'})
    assert fields['LITELLM_MASTER_KEY'] == 'mk'


def test_seed_team_key_invalid_team():
    with pytest.raises(st.StackError, match='invalid team'):
        st.seed_team_key(_cfg(), 'BAD_TEAM')


def test_seed_team_key_happy(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(st, 'wait_vault', lambda *a, **k: None)
    monkeypatch.setattr(st, 'wait_gateway', lambda *a, **k: None)
    monkeypatch.setattr(st, '_out', lambda *a, **k: None)
    monkeypatch.setattr(
        st,
        'read_vault_kv_data',
        MagicMock(
            side_effect=[
                {'LITELLM_MASTER_KEY': 'mk'},
                {'LITELLM_MASTER_KEY': 'mk'},
                {'api_key': 'sk-long-enough'},
            ]
        ),
    )
    monkeypatch.setattr(
        st,
        '_http_json_post',
        MagicMock(
            side_effect=[
                {},
                {'key': 'sk-generated-key-value'},
            ]
        ),
    )
    write = MagicMock()
    policy = MagicMock()
    monkeypatch.setattr(st, 'write_vault_kv_data', write)
    monkeypatch.setattr(st, 'write_vault_policy', policy)
    st.seed_team_key(cfg, 'demo')
    write.assert_called()
    policy.assert_called()


def test_seed_registry_vault(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(st, 'wait_vault', lambda *a, **k: None)
    monkeypatch.setattr(st, '_out', lambda *a, **k: None)
    writes = MagicMock()
    monkeypatch.setattr(st, 'write_vault_kv_data', writes)
    monkeypatch.setattr(st, 'write_vault_policy', MagicMock())
    monkeypatch.setattr(
        st,
        'read_vault_kv_data',
        MagicMock(
            side_effect=[
                {'username': 'push', 'password': 'push-pass-long'},
                {'username': 'pull', 'password': 'pull-pass-long'},
            ]
        ),
    )
    st.seed_registry_vault(cfg, env={})
    assert writes.call_count == 2


def test_cmd_stack_up_down_compose(tmp_path, monkeypatch):
    (tmp_path / 'docker-compose.yml').write_text('x\n', encoding='utf-8')
    monkeypatch.setattr(st, 'load_stack_env', lambda d: None)
    monkeypatch.setattr(st, 'run_compose', MagicMock(return_value=0))
    args = SimpleNamespace(
        dir=str(tmp_path),
        profile=None,
        build=False,
        detach=True,
        seed=False,
        compose_args=None,
    )
    assert st.cmd_stack_up(args) == 0
    assert st.cmd_stack_down(args) == 0
    args.compose_args = ['ps']
    assert st.cmd_stack_compose(args) == 0
    args.compose_args = None
    with pytest.raises(st.StackError, match='usage'):
        st.cmd_stack_compose(args)


def test_cmd_stack_bootstrap_sets_flags(tmp_path, monkeypatch):
    (tmp_path / 'docker-compose.yml').write_text('x\n', encoding='utf-8')
    called = {}

    def fake_up(args):
        called['build'] = args.build
        called['detach'] = args.detach
        called['seed'] = args.seed
        return 0

    monkeypatch.setattr(st, 'cmd_stack_up', fake_up)
    args = SimpleNamespace(dir=str(tmp_path))
    assert st.cmd_stack_bootstrap(args) == 0
    assert called == {'build': True, 'detach': True, 'seed': True}


def test_cmd_stack_seed(tmp_path, monkeypatch):
    (tmp_path / 'docker-compose.yml').write_text('x\n', encoding='utf-8')
    monkeypatch.setattr(st, 'load_stack_env', lambda d: None)
    monkeypatch.setattr(st, '_out', lambda *a, **k: None)
    monkeypatch.setattr(
        'cat_agent.platform.config.load_platform_config',
        lambda **k: _cfg(),
    )
    monkeypatch.setattr(st, 'seed_llm_vault', MagicMock())
    monkeypatch.setattr(st, 'seed_team_key', MagicMock())
    monkeypatch.setattr(st, 'seed_registry_vault', MagicMock())
    args = SimpleNamespace(
        dir=str(tmp_path),
        nomad_addr=None,
        config=None,
        team='demo',
        max_tokens_per_day=None,
        tpm_limit=None,
        rpm_limit=None,
        registry=True,
        profile=[],
    )
    assert st.cmd_stack_seed(args) == 0
    st.seed_registry_vault.assert_called_once()
