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

"""Tests for agent.yaml validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cat_agent.platform.config import ConfigError, PlatformConfig, load_platform_config
from cat_agent.platform.manifest import AgentManifest, ManifestError, load_manifest, validate_manifest


def _minimal(**overrides):
    data = {
        'apiVersion': 'v1',
        'name': 'funding-scout',
        'team': 'growth',
        'runtime': {'entrypoint': 'scout:registry'},
        'trigger': {'type': 'http'},
    }
    data.update(overrides)
    return data


def test_valid_minimal():
    m = validate_manifest(_minimal())
    assert m.job_id() == 'agent-growth-funding-scout'
    assert m.trigger.jobs == 'inline'
    assert m.model.alias == 'default'


def test_load_manifest_file(tmp_path: Path):
    p = tmp_path / 'agent.yaml'
    p.write_text(yaml.dump(_minimal()), encoding='utf-8')
    m = load_manifest(p)
    assert m.name == 'funding-scout'


@pytest.mark.parametrize(
    'payload,needle',
    [
        ({'image': 'x'}, 'platform decision'),
        ({'dockerfile': 'x'}, 'platform decision'),
        ({'replicas': 2}, 'platform decision'),
        ({'datacenter': 'dc1'}, 'platform decision'),
        ({'namespace': 'agents'}, 'platform decision'),
        (
            {'model': {'alias': 'default', 'type': 'local'}},
            'still work locally via `cat-agent serve`',
        ),
        (
            {'resources': {'gpu': 1}},
            'still work via `cat-agent serve`',
        ),
        (
            {'env': {'OPENAI_BASE_URL': 'http://bypass'}},
            'LLM gateway',
        ),
        (
            {'env': {'OPENAI_API_KEY': 'sk-x'}},
            'LLM gateway',
        ),
        (
            {'env': {'CAT_AGENT_SERVE_PORT': '9'}},
            'reserved',
        ),
        (
            {'scaling': {'min': 0, 'max': 1}},
            'scaling.min',
        ),
        (
            {
                'trigger': {'type': 'http', 'jobs': 'inline'},
                'scaling': {'min': 1, 'max': 3},
            },
            'dispatch',
        ),
        (
            {'trigger': {'type': 'schedule', 'jobs': 'dispatch', 'cron': '0 * * * *'}},
            'dispatch is only valid',
        ),
        (
            {'secrets': ['FOO=bar']},
            'names',
        ),
        (
            {'secrets': ['x' * 65]},
            '64 characters',
        ),
        (
            {'name': 'Bad_Name'},
            'must match',
        ),
    ],
)
def test_rejections(payload, needle):
    data = _minimal()
    # deep-ish merge for nested keys
    for k, v in payload.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            merged = dict(data[k])
            merged.update(v)
            data[k] = merged
        else:
            data[k] = v
    with pytest.raises(ManifestError, match=needle):
        validate_manifest(data)


def test_timeout_ceiling():
    m = validate_manifest(
        _minimal(resources={'timeout': '7200s'})
    )
    with pytest.raises(ManifestError, match='dispatch'):
        m.validate_timeout_ceiling(3600)


def test_validate_tools():
    m = validate_manifest(
        _minimal(tools={'allow': ['web_search', 'nope']})
    )
    with pytest.raises(ManifestError, match='nope'):
        m.validate_tools({'web_search', 'storage'})


def test_config_local_registry(tmp_path: Path, monkeypatch):
    monkeypatch.delenv('NOMAD_ADDR', raising=False)
    monkeypatch.delenv('CAT_AGENT_REGISTRY', raising=False)
    p = tmp_path / 'config.toml'
    p.write_text(
        '[platform]\nregistry = "local"\nnomad_addr = "http://127.0.0.1:4646"\n',
        encoding='utf-8',
    )
    cfg = load_platform_config(path=p)
    assert cfg.is_local_registry()
    assert 'not pushed' in cfg.registry_display()
    assert cfg.image_ref('growth/funding-scout:abc') == 'growth/funding-scout:abc'


def test_config_remote_registry(tmp_path: Path, monkeypatch):
    monkeypatch.delenv('CAT_AGENT_REGISTRY', raising=False)
    p = tmp_path / 'config.toml'
    p.write_text(
        '[platform]\nregistry = "registry.example/agents"\n',
        encoding='utf-8',
    )
    cfg = load_platform_config(path=p)
    assert not cfg.is_local_registry()
    assert cfg.image_ref('my:tag') == 'registry.example/agents/my:tag'


def test_config_unknown_key(tmp_path: Path):
    p = tmp_path / 'config.toml'
    p.write_text('[platform]\nweird = 1\n', encoding='utf-8')
    with pytest.raises(ConfigError, match='weird'):
        load_platform_config(path=p)


def test_config_env_overrides_file(tmp_path: Path, monkeypatch):
    p = tmp_path / 'config.toml'
    p.write_text(
        '[platform]\nnomad_addr = "http://file:4646"\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('NOMAD_ADDR', 'http://env:4646')
    cfg = load_platform_config(path=p)
    assert cfg.nomad_addr == 'http://env:4646'


def test_default_config_path_uses_stack_dir_env(tmp_path: Path, monkeypatch):
    from cat_agent.platform.config import default_config_path

    stack = tmp_path / 'stack'
    stack.mkdir()
    cfg_file = stack / 'cat-agent.config.toml'
    cfg_file.write_text('[platform]\nregistry = "local"\n', encoding='utf-8')
    monkeypatch.delenv('CAT_AGENT_CONFIG', raising=False)
    monkeypatch.setenv('CAT_AGENT_STACK_DIR', str(stack))
    monkeypatch.chdir(tmp_path)
    assert default_config_path() == cfg_file


def test_config_cli_overrides_env(tmp_path: Path, monkeypatch):
    p = tmp_path / 'config.toml'
    p.write_text(
        '[platform]\nnomad_addr = "http://file:4646"\n',
        encoding='utf-8',
    )
    monkeypatch.setenv('NOMAD_ADDR', 'http://env:4646')
    cfg = load_platform_config(
        path=p, overrides={'nomad_addr': 'http://flag:4646'}
    )
    assert cfg.nomad_addr == 'http://flag:4646'


def test_ingress_host_template_default_and_custom():
    cfg = PlatformConfig()
    assert cfg.ingress_host('demo', 'calculator') == 'demo-calculator.localhost'
    corp = PlatformConfig(
        ingress_host_template='{team}-{agent}.agents.example.internal'
    )
    assert corp.ingress_host('demo', 'calculator') == (
        'demo-calculator.agents.example.internal'
    )
