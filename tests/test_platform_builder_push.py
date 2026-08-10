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

"""Builder push path and cat_agent COPY behaviour (no network / no real docker)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cat_agent.platform.builder import (
    BuildError,
    build_agent_image,
    docker_login_and_push,
    stage_agent_build_context,
)
from cat_agent.platform.config import PlatformConfig
from cat_agent.platform.manifest import validate_manifest


def _manifest():
    return validate_manifest(
        {
            'name': 'calc',
            'team': 'demo',
            'runtime': {'entrypoint': 'app:registry'},
            'trigger': {'type': 'http'},
        }
    )


def _ok(cmd=None, cwd=None):
    return subprocess.CompletedProcess(cmd or [], 0, stdout='', stderr='')


def test_local_dockerfile_copies_cat_agent(tmp_path):
    example = tmp_path / 'example'
    example.mkdir()
    (example / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')
    # Fake monorepo root: pyproject + cat_agent next to example's parent.
    repo = tmp_path
    (repo / 'pyproject.toml').write_text('[project]\nname="x"\n', encoding='utf-8')
    (repo / 'cat_agent').mkdir()
    (repo / 'cat_agent' / '__init__.py').write_text('__version__="test"\n', encoding='utf-8')
    # Put example under repo
    src = repo / 'examples' / 'calc'
    src.mkdir(parents=True)
    (src / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')

    cfg = PlatformConfig(registry='local', base_image='python:3.11-slim')
    with tempfile.TemporaryDirectory() as tmp:
        ctx = Path(tmp)
        df = stage_agent_build_context(_manifest(), cfg, src, ctx)
        text = df.read_text(encoding='utf-8')
        assert 'COPY cat_agent' in text
        assert (ctx / 'cat_agent' / '__init__.py').is_file()
        assert 'PYTHONPATH=/opt/cat-agent' in text


def test_remote_dockerfile_has_no_cat_agent_copy(tmp_path):
    repo = tmp_path
    (repo / 'pyproject.toml').write_text('[project]\nname="x"\n', encoding='utf-8')
    (repo / 'cat_agent').mkdir()
    (repo / 'cat_agent' / '__init__.py').write_text('__version__="stale"\n', encoding='utf-8')
    src = repo / 'examples' / 'calc'
    src.mkdir(parents=True)
    (src / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')

    cfg = PlatformConfig(
        registry='127.0.0.1:5001',
        base_image='cat-agent-runtime:latest',
    )
    with tempfile.TemporaryDirectory() as tmp:
        ctx = Path(tmp)
        df = stage_agent_build_context(_manifest(), cfg, src, ctx)
        text = df.read_text(encoding='utf-8')
        assert 'COPY cat_agent' not in text
        assert '/opt/cat-agent' not in text
        assert not (ctx / 'cat_agent').exists()
        assert 'FROM cat-agent-runtime:latest' in text
        assert 'COPY examples/calc /app' in text or 'COPY' in text


def test_remote_build_fails_loudly_when_base_lacks_framework(tmp_path):
    """Behaviour: remote mode must not silently pick up a source-tree copy."""
    repo = tmp_path
    (repo / 'pyproject.toml').write_text('[project]\nname="x"\n', encoding='utf-8')
    (repo / 'cat_agent').mkdir()
    (repo / 'cat_agent' / '__init__.py').write_text('__version__="stale-tree"\n', encoding='utf-8')
    src = repo / 'examples' / 'calc'
    src.mkdir(parents=True)
    (src / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')

    cfg = PlatformConfig(
        registry='127.0.0.1:5001',
        base_image='python:3.11-slim',  # no cat-agent
        vault_addr='http://127.0.0.1:8200',
    )

    def run(cmd, cwd=None):
        if cmd[:2] == ['docker', 'build']:
            # Prove the staged Dockerfile never COPY'd cat_agent.
            df = Path(cwd) / 'Dockerfile'
            text = df.read_text(encoding='utf-8')
            assert 'COPY cat_agent' not in text
            assert not (Path(cwd) / 'cat_agent').exists()
            return _ok(cmd)
        if cmd[:2] == ['docker', 'run']:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout='',
                stderr="ModuleNotFoundError: No module named 'cat_agent'",
            )
        return _ok(cmd)

    with pytest.raises(BuildError, match='does not provide an importable cat_agent'):
        build_agent_image(
            _manifest(),
            cfg,
            src,
            image_tag='demo/calc:test',
            push=False,
            run=run,
        )


def test_remote_build_fails_when_import_resolves_under_opt_cat_agent(tmp_path):
    """Behaviour: a shadowed source COPY must fail the post-build check."""
    repo = tmp_path
    (repo / 'pyproject.toml').write_text('[project]\nname="x"\n', encoding='utf-8')
    (repo / 'cat_agent').mkdir()
    (repo / 'cat_agent' / '__init__.py').write_text('', encoding='utf-8')
    src = repo / 'examples' / 'calc'
    src.mkdir(parents=True)
    (src / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')

    cfg = PlatformConfig(
        registry='127.0.0.1:5001',
        base_image='cat-agent-runtime:latest',
    )

    def run(cmd, cwd=None):
        if cmd[:2] == ['docker', 'run']:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout='',
                stderr=(
                    'AssertionError: cat_agent resolved to '
                    '/opt/cat-agent/cat_agent/__init__.py — source COPY shadowed '
                    'the base image; remote builds must not COPY cat_agent'
                ),
            )
        return _ok(cmd)

    with pytest.raises(BuildError, match='does not provide an importable cat_agent'):
        build_agent_image(
            _manifest(),
            cfg,
            src,
            image_tag='demo/calc:test',
            push=False,
            run=run,
        )


def test_push_skipped_when_local(tmp_path):
    repo = tmp_path
    (repo / 'pyproject.toml').write_text('[project]\nname="x"\n', encoding='utf-8')
    (repo / 'cat_agent').mkdir()
    (repo / 'cat_agent' / '__init__.py').write_text('', encoding='utf-8')
    src = repo / 'examples' / 'calc'
    src.mkdir(parents=True)
    (src / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')
    cfg = PlatformConfig(registry='local', base_image='python:3.11-slim')

    calls = []

    def run(cmd, cwd=None):
        calls.append(list(cmd))
        return _ok(cmd)

    build_agent_image(
        _manifest(),
        cfg,
        src,
        image_tag='demo/calc:test',
        push=True,
        run=run,
    )
    assert not any(c[:2] == ['docker', 'push'] for c in calls)
    assert not any(c[:2] == ['docker', 'login'] for c in calls)


def test_remote_push_login_tag_push(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / 'pyproject.toml').write_text('[project]\nname="x"\n', encoding='utf-8')
    (repo / 'cat_agent').mkdir()
    (repo / 'cat_agent' / '__init__.py').write_text('', encoding='utf-8')
    src = repo / 'examples' / 'calc'
    src.mkdir(parents=True)
    (src / 'app.py').write_text('def registry():\n    pass\n', encoding='utf-8')
    cfg = PlatformConfig(
        registry='127.0.0.1:5001',
        base_image='cat-agent-runtime:latest',
        vault_addr='http://vault:8200',
    )
    monkeypatch.setattr(
        'cat_agent.platform.builder.read_push_credentials',
        lambda _cfg: ('zot-push', 'secret'),
    )
    calls = []

    def run(cmd, cwd=None):
        calls.append(list(cmd))
        if cmd[:2] == ['docker', 'run']:
            # Pretend base provides cat_agent outside /opt/cat-agent.
            return _ok(cmd)
        return _ok(cmd)

    def login_run():
        calls.append(['docker', 'login', '127.0.0.1:5001'])
        return _ok()

    build_agent_image(
        _manifest(),
        cfg,
        src,
        image_tag='demo/calc:test',
        push=True,
        run=run,
        login_run=login_run,
    )
    assert ['docker', 'login', '127.0.0.1:5001'] in calls
    assert ['docker', 'tag', 'demo/calc:test', '127.0.0.1:5001/demo/calc:test'] in calls
    assert ['docker', 'push', '127.0.0.1:5001/demo/calc:test'] in calls


def test_push_401_maps_to_actionable_message(monkeypatch):
    cfg = PlatformConfig(
        registry='127.0.0.1:5001',
        vault_addr='http://vault:8200',
    )
    monkeypatch.setattr(
        'cat_agent.platform.builder.read_push_credentials',
        lambda _cfg: ('zot-push', 'bad'),
    )

    def run(cmd, cwd=None):
        if cmd[:2] == ['docker', 'push']:
            return subprocess.CompletedProcess(
                cmd, 1, stdout='', stderr='unauthorized: authentication required'
            )
        return _ok(cmd)

    def login_run():
        return _ok()

    with pytest.raises(
        BuildError,
        match='push denied — check the registry credentials in Vault at '
        'secret/data/platform/registry/push',
    ):
        docker_login_and_push(
            cfg, 'demo/calc:test', run=run, login_run=login_run
        )
