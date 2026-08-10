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

"""Tests for ``python -m cat_agent.serve.task``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cat_agent.serve import AgentRegistry
from cat_agent.serve import task as task_mod
from tests.serve_fakes import FakeAgent


def _factory():
    reg = AgentRegistry()
    reg.register(FakeAgent('Bot', 'task-ok'), name='bot')
    return reg


class TestServeTask:

    def test_payload_run_success(self, tmp_path: Path, monkeypatch):
        payload = tmp_path / 'payload'
        payload.write_text(
            json.dumps({'messages': [{'role': 'user', 'content': 'hi'}]}),
            encoding='utf-8',
        )
        monkeypatch.setenv('CAT_AGENT_ENTRYPOINT', 'tests.test_serve_task:_factory')
        monkeypatch.setenv('CAT_AGENT_PAYLOAD', str(payload))
        monkeypatch.setenv('CAT_AGENT_JOB_ID', 'job-1')
        # Ensure import path resolves the factory module
        import sys
        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        code = task_mod.main([])
        assert code == 0
        result_path = payload.with_suffix('.result.json')
        assert result_path.is_file()
        data = json.loads(result_path.read_text(encoding='utf-8'))
        assert data['content'] == 'task-ok'
        assert data['job_id'] == 'job-1'

    def test_missing_payload_exits_nonzero(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_ENTRYPOINT', 'tests.test_serve_task:_factory')
        monkeypatch.delenv('CAT_AGENT_PAYLOAD', raising=False)
        assert task_mod.main([]) == 1

    def test_run_failure_exits_nonzero(self, tmp_path: Path, monkeypatch):
        def bad_factory():
            reg = AgentRegistry()
            reg.register(
                FakeAgent('Bot', 'x', raise_on_run=RuntimeError('boom')),
                name='bot',
            )
            return reg

        # Patch factory lookup by writing a module-level name
        import tests.test_serve_task as self_mod
        self_mod._bad = bad_factory  # type: ignore[attr-defined]

        payload = tmp_path / 'payload'
        payload.write_text(
            json.dumps({'messages': [{'role': 'user', 'content': 'hi'}]}),
            encoding='utf-8',
        )
        monkeypatch.setenv('CAT_AGENT_ENTRYPOINT', 'tests.test_serve_task:_bad')
        monkeypatch.setenv('CAT_AGENT_PAYLOAD', str(payload))
        assert task_mod.main([]) == 1
        err = json.loads(payload.with_suffix('.result.json').read_text(encoding='utf-8'))
        assert err['ok'] is False
        assert err['error_type'] == 'RuntimeError'
