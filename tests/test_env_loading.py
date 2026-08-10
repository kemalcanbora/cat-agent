"""Tests for .env configuration loading."""

import os

from cat_agent.env import load_env_file, reset_env_loading


class TestEnvLoading:

    def test_load_env_file_reads_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        reset_env_loading()
        monkeypatch.delenv('CAT_AGENT_MANAGED', raising=False)
        (tmp_path / '.env').write_text(
            'CAT_AGENT_OFFLINE=1\n'
            'CAT_AGENT_OFFLINE_ALLOW_HOSTS=llm.internal\n',
            encoding='utf-8',
        )
        monkeypatch.delenv('CAT_AGENT_OFFLINE', raising=False)
        monkeypatch.delenv('CAT_AGENT_OFFLINE_ALLOW_HOSTS', raising=False)

        assert load_env_file() is True
        assert os.getenv('CAT_AGENT_OFFLINE') == '1'
        assert os.getenv('CAT_AGENT_OFFLINE_ALLOW_HOSTS') == 'llm.internal'

    def test_load_env_file_does_not_override_existing_env(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        reset_env_loading()
        monkeypatch.delenv('CAT_AGENT_MANAGED', raising=False)
        (tmp_path / '.env').write_text('CAT_AGENT_OFFLINE=1\n', encoding='utf-8')
        monkeypatch.setenv('CAT_AGENT_OFFLINE', '0')

        assert load_env_file() is True
        assert os.getenv('CAT_AGENT_OFFLINE') == '0'

    def test_load_env_file_honors_cat_agent_env_file(self, tmp_path, monkeypatch):
        custom = tmp_path / 'config' / 'prod.env'
        custom.parent.mkdir()
        custom.write_text('OPENAI_BASE_URL=http://llm.internal:8080/v1\n', encoding='utf-8')
        reset_env_loading()
        monkeypatch.delenv('CAT_AGENT_MANAGED', raising=False)
        monkeypatch.setenv('CAT_AGENT_ENV_FILE', str(custom))
        monkeypatch.delenv('OPENAI_BASE_URL', raising=False)

        assert load_env_file() is True
        assert os.getenv('OPENAI_BASE_URL') == 'http://llm.internal:8080/v1'

    def test_load_env_file_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        reset_env_loading()
        monkeypatch.delenv('CAT_AGENT_MANAGED', raising=False)
        (tmp_path / '.env').write_text('CAT_AGENT_OFFLINE=1\n', encoding='utf-8')
        monkeypatch.delenv('CAT_AGENT_OFFLINE', raising=False)

        assert load_env_file() is True
        assert load_env_file() is False

    def test_import_cat_agent_loads_dotenv(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        reset_env_loading()
        monkeypatch.delenv('CAT_AGENT_MANAGED', raising=False)
        (tmp_path / '.env').write_text('CAT_AGENT_OFFLINE=1\n', encoding='utf-8')
        monkeypatch.delenv('CAT_AGENT_OFFLINE', raising=False)

        import importlib

        import cat_agent

        importlib.reload(cat_agent)
        assert os.getenv('CAT_AGENT_OFFLINE') == '1'
        monkeypatch.delenv('CAT_AGENT_OFFLINE', raising=False)
