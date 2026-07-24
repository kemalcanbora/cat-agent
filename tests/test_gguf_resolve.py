"""Tests for cat_agent.llm.gguf path resolution."""

from pathlib import Path
from unittest.mock import patch

import pytest

from cat_agent.llm.gguf import resolve_gguf_path


class TestResolveGgufPath:

    def test_explicit_model_path_wins(self, tmp_path: Path):
        gguf = tmp_path / 'model.gguf'
        gguf.write_bytes(b'gguf')
        assert resolve_gguf_path(model_path=str(gguf)) == str(gguf)

    def test_requires_model_path_or_repo(self):
        with pytest.raises(ValueError, match='model_path|repo_id|filename'):
            resolve_gguf_path()
        with pytest.raises(ValueError, match='model_path|repo_id|filename'):
            resolve_gguf_path(repo_id='org/repo')

    def test_uses_huggingface_cache_before_download(self, tmp_path: Path):
        cached = tmp_path / 'cached.gguf'
        cached.write_bytes(b'gguf')

        with patch('cat_agent.llm.gguf._hf_cache_path', return_value=str(cached)) as mock_cache, \
                patch('huggingface_hub.hf_hub_download') as mock_dl:
            out = resolve_gguf_path(repo_id='org/repo', filename='model.gguf')
            assert out == str(cached)
            mock_cache.assert_called_once()
            mock_dl.assert_not_called()

    def test_falls_back_to_home_models(self, tmp_path: Path, monkeypatch):
        home = tmp_path / 'home'
        models = home / 'models'
        models.mkdir(parents=True)
        gguf = models / 'model.gguf'
        gguf.write_bytes(b'gguf')
        monkeypatch.setattr(Path, 'home', staticmethod(lambda: home))

        with patch('cat_agent.llm.gguf._hf_cache_path', return_value=None), \
                patch('huggingface_hub.hf_hub_download') as mock_dl:
            out = resolve_gguf_path(repo_id='org/repo', filename='model.gguf')
            assert out == str(gguf)
            mock_dl.assert_not_called()

    def test_downloads_when_not_cached(self):
        with patch('cat_agent.llm.gguf._hf_cache_path', return_value=None), \
                patch('cat_agent.llm.gguf.Path.home', return_value=Path('/no/such/home')), \
                patch('huggingface_hub.hf_hub_download', return_value='/dl/model.gguf') as mock_dl:
            out = resolve_gguf_path(repo_id='org/repo', filename='model.gguf', cache_dir='/c')
            assert out == '/dl/model.gguf'
            mock_dl.assert_called_once_with(
                repo_id='org/repo',
                filename='model.gguf',
                cache_dir='/c',
            )
