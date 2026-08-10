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

"""Env contract for platform-managed agents (gateway must not be silently bypassed)."""

from __future__ import annotations

import os

from cat_agent.env import load_env_file, reset_env_loading
from cat_agent.llm.env_config import llm_config_from_env
from cat_agent.platform.manifest import PROTECTED_ENV_KEYS

GATEWAY = 'http://llm-gateway.service.consul:4000/v1'
EVIL = 'http://not-the-gateway'


def test_openai_base_url_remains_protected():
    assert 'OPENAI_BASE_URL' in PROTECTED_ENV_KEYS
    assert 'OPENAI_API_KEY' in PROTECTED_ENV_KEYS
    assert 'CAT_AGENT_LLM_BASE_URL' in PROTECTED_ENV_KEYS


def test_unmanaged_dotenv_would_inject_evil_base_url(tmp_path, monkeypatch):
    """Control: without CAT_AGENT_MANAGED, a stray .env can set OPENAI_BASE_URL.

    Establishes that the .env in the managed test is live ammunition — if
    managed suppression regresses, traffic can leave the gateway.
    """
    monkeypatch.chdir(tmp_path)
    reset_env_loading()
    monkeypatch.delenv('CAT_AGENT_MANAGED', raising=False)
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('CAT_AGENT_LLM_BASE_URL', raising=False)
    monkeypatch.delenv('OLLAMA_API_BASE', raising=False)
    (tmp_path / '.env').write_text(f'OPENAI_BASE_URL={EVIL}\n', encoding='utf-8')

    assert load_env_file() is True
    assert os.environ.get('OPENAI_BASE_URL') == EVIL
    cfg = llm_config_from_env()
    assert cfg.get('base_url') == EVIL.rstrip('/')


def test_managed_keeps_gateway_despite_hostile_dotenv(tmp_path, monkeypatch):
    """CAT_AGENT_MANAGED=1 must keep platform gateway URL when .env disagrees.

    This is the silent failure: the agent still answers, but requests stop
    going through the gateway and quota tracking dies. Assert the *resolved*
    config, not that load_dotenv was skipped — another override path must
    also lose.
    """
    monkeypatch.chdir(tmp_path)
    reset_env_loading()
    (tmp_path / '.env').write_text(
        f'OPENAI_BASE_URL={EVIL}\n'
        f'CAT_AGENT_LLM_BASE_URL={EVIL}\n'
        f'OLLAMA_API_BASE={EVIL}\n',
        encoding='utf-8',
    )

    # Platform injects these before the process starts (Nomad env / Docker).
    monkeypatch.setenv('CAT_AGENT_MANAGED', '1')
    monkeypatch.setenv('OPENAI_BASE_URL', GATEWAY)
    monkeypatch.setenv('CAT_AGENT_LLM_BASE_URL', GATEWAY)
    monkeypatch.delenv('OLLAMA_API_BASE', raising=False)

    assert load_env_file() is False  # skipped under managed
    # dotenv must not have rewritten the process env.
    assert os.environ.get('OPENAI_BASE_URL') == GATEWAY
    assert os.environ.get('CAT_AGENT_LLM_BASE_URL') == GATEWAY
    assert os.environ.get('OLLAMA_API_BASE') != EVIL

    cfg = llm_config_from_env()
    assert cfg.get('base_url') == GATEWAY.rstrip('/')
    assert EVIL not in (cfg.get('base_url') or '')


def test_managed_blocks_dotenv_when_gateway_not_yet_mirrored(tmp_path, monkeypatch):
    """Even if OPENAI_BASE_URL is unset, managed mode must not load evil from .env.

    Catches a regression where CAT_AGENT_MANAGED skip is removed but
    load_dotenv(override=False) still looks fine in tests that pre-set the
    gateway — here the var is absent, so a real dotenv load would inject evil.
    """
    monkeypatch.chdir(tmp_path)
    reset_env_loading()
    (tmp_path / '.env').write_text(f'OPENAI_BASE_URL={EVIL}\n', encoding='utf-8')

    monkeypatch.setenv('CAT_AGENT_MANAGED', '1')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('CAT_AGENT_LLM_BASE_URL', raising=False)
    monkeypatch.delenv('OLLAMA_API_BASE', raising=False)

    assert load_env_file() is False
    assert os.environ.get('OPENAI_BASE_URL') is None

    # Platform value arrives (or is the only intended source).
    monkeypatch.setenv('CAT_AGENT_LLM_BASE_URL', GATEWAY)
    cfg = llm_config_from_env()
    assert cfg.get('base_url') == GATEWAY.rstrip('/')
    assert os.environ.get('OPENAI_BASE_URL') != EVIL


def test_platform_llm_vars_beat_ollama_local(monkeypatch):
    monkeypatch.setenv(
        'CAT_AGENT_LLM_BASE_URL', 'http://llm-gateway.service.consul:4000/v1'
    )
    monkeypatch.setenv('CAT_AGENT_LLM_MODEL', 'smart')
    monkeypatch.setenv('CAT_AGENT_LLM_API_KEY', 'sk-team-virtual')
    monkeypatch.setenv('OLLAMA_API_BASE', 'http://127.0.0.1:11434')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://should-not-win')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-should-not-win')
    monkeypatch.setenv('OPENAI_MODEL', 'should-not-win')

    cfg = llm_config_from_env()
    assert cfg['base_url'] == 'http://llm-gateway.service.consul:4000/v1'
    assert cfg['model'] == 'smart'
    assert cfg['api_key'] == 'sk-team-virtual'


def test_local_ollama_path_still_works_without_platform_vars(monkeypatch):
    """Unmanaged local scripts may still use OLLAMA_API_BASE."""
    monkeypatch.delenv('CAT_AGENT_LLM_BASE_URL', raising=False)
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    monkeypatch.delenv('CAT_AGENT_LLM_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('CAT_AGENT_LLM_MODEL', raising=False)
    monkeypatch.setenv('OLLAMA_API_BASE', 'http://127.0.0.1:11434')
    monkeypatch.setenv('OLLAMA_API_KEY', 'ollama-secret')
    monkeypatch.setenv('LLM_MODEL', 'llama3.2')

    cfg = llm_config_from_env()
    assert cfg['base_url'] == 'http://127.0.0.1:11434'
    assert cfg['api_key'] == 'ollama-secret'
    assert cfg['model'] == 'llama3.2'


def test_apply_agent_yaml_env_sets_missing_keys(tmp_path, monkeypatch):
    from cat_agent.llm.env_config import apply_agent_yaml_env

    monkeypatch.delenv('CAT_AGENT_MANAGED', raising=False)
    monkeypatch.delenv('CAT_AGENT_LLM_MODEL_DATAGUY', raising=False)
    yaml_path = tmp_path / 'agent.yaml'
    yaml_path.write_text(
        'name: demo\nenv:\n  CAT_AGENT_LLM_MODEL_DATAGUY: gemma4:cloud\n',
        encoding='utf-8',
    )
    applied = apply_agent_yaml_env(yaml_path)
    assert applied['CAT_AGENT_LLM_MODEL_DATAGUY'] == 'gemma4:cloud'
    assert os.environ['CAT_AGENT_LLM_MODEL_DATAGUY'] == 'gemma4:cloud'


def test_model_falls_back_to_agent_yaml_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv('CAT_AGENT_LLM_MODEL', raising=False)
    monkeypatch.delenv('OPENAI_MODEL', raising=False)
    monkeypatch.delenv('LLM_MODEL', raising=False)
    monkeypatch.delenv('CAT_AGENT_AGENT_YAML', raising=False)
    yaml_path = tmp_path / 'agent.yaml'
    yaml_path.write_text(
        'name: demo\nmodel:\n  alias: minimax-m3:cloud\n',
        encoding='utf-8',
    )
    cfg = llm_config_from_env(agent_yaml=yaml_path)
    assert cfg['model'] == 'minimax-m3:cloud'


def test_env_model_beats_agent_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_MODEL', 'from-env')
    yaml_path = tmp_path / 'agent.yaml'
    yaml_path.write_text(
        'name: demo\nmodel:\n  alias: from-yaml\n',
        encoding='utf-8',
    )
    cfg = llm_config_from_env(agent_yaml=yaml_path)
    assert cfg['model'] == 'from-env'
