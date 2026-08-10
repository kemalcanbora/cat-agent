"""Unit tests for local stack seed/.env resolution (no Docker)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cat_agent.platform.stack import (
    StackError,
    ensure_host_data_dirs,
    resolve_llm_seed,
    resolve_stack_dir,
)


def test_resolve_llm_seed_defaults_to_host_ollama():
    fields = resolve_llm_seed({})
    assert fields['OLLAMA_API_KEY'] == 'local-ollama'
    assert fields['OLLAMA_API_BASE'] == 'http://host.docker.internal:11434'
    assert fields['OPENAI_API_KEY'] == 'local-ollama'
    assert fields['OPENAI_API_BASE'] == 'http://host.docker.internal:11434/v1'
    assert fields['LITELLM_MASTER_KEY'] == 'sk-local-litellm-master'


def test_resolve_llm_seed_real_openai():
    fields = resolve_llm_seed(
        {
            'OPENAI_API_KEY': 'sk-live',
            'OLLAMA_API_KEY': 'local-ollama',
            'OLLAMA_API_BASE': 'http://host.docker.internal:11434',
            'LITELLM_MASTER_KEY': 'sk-master',
        }
    )
    assert fields['OPENAI_API_KEY'] == 'sk-live'
    assert fields['OPENAI_API_BASE'] == 'https://api.openai.com/v1'
    assert fields['OLLAMA_API_BASE'] == 'http://host.docker.internal:11434'
    assert fields['LITELLM_MASTER_KEY'] == 'sk-master'


def test_resolve_llm_seed_openai_base_url_alias():
    fields = resolve_llm_seed(
        {'OPENAI_API_KEY': 'sk-x', 'OPENAI_BASE_URL': 'http://proxy/v1'}
    )
    assert fields['OPENAI_API_BASE'] == 'http://proxy/v1'


def test_resolve_llm_seed_does_not_double_v1_on_ollama_cloud():
    fields = resolve_llm_seed(
        {
            'OLLAMA_API_BASE': 'https://ollama.com/v1',
            'OLLAMA_API_KEY': 'ollama-key',
        }
    )
    assert fields['OLLAMA_API_BASE'] == 'https://ollama.com/v1'
    assert fields['OPENAI_API_BASE'] == 'https://ollama.com/v1'
    assert fields['OPENAI_API_KEY'] == 'ollama-key'


def test_resolve_stack_dir_explicit(tmp_path: Path):
    (tmp_path / 'docker-compose.yml').write_text('services: {}\n')
    assert resolve_stack_dir(str(tmp_path)) == tmp_path.resolve()


def test_resolve_stack_dir_missing(tmp_path: Path):
    with pytest.raises(StackError, match='docker-compose.yml'):
        resolve_stack_dir(str(tmp_path))


def test_ensure_host_data_dirs(tmp_path: Path, monkeypatch):
    monkeypatch.delenv('HOST_NOMAD_DATA', raising=False)
    monkeypatch.delenv('HOST_ZOT_DATA', raising=False)
    paths = ensure_host_data_dirs(tmp_path)
    assert Path(paths['HOST_NOMAD_DATA']).is_dir()
    assert Path(paths['HOST_ZOT_DATA']).is_dir()
