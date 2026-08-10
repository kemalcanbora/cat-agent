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

"""Error hygiene tests for cat_agent.serve."""

from __future__ import annotations

import pytest

from cat_agent.log import logger
from cat_agent.security.pii import SECRET_PLACEHOLDER, redact_text
from cat_agent.serve import AgentRegistry, create_app
from cat_agent.serve.errors import GENERIC_ERROR
from tests.serve_fakes import FakeAgent

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402

PROVIDER_LEAK = (
    'Error code: 401 - Authorization: Bearer sk-live-SECRETKEY999 '
    'request to https://api.openai.com/v1/chat/completions?api_key=sk-live-SECRETKEY999 '
    'failed with sk-proj-LEAKEDTOKEN123456'
)


@pytest.fixture
def capture_serve_logs():
    """Capture loguru ERROR+ messages (stdlib caplog does not see loguru).

    ``cat_agent.log`` disables the ``cat_agent`` namespace by default; enable it
    for the duration of the test so serve error logs are observable.
    """
    lines: list[str] = []

    def _sink(message):
        lines.append(message.record['message'])

    logger.enable('cat_agent')
    sink_id = logger.add(_sink, level='ERROR')
    try:
        yield lines
    finally:
        logger.remove(sink_id)
        if not __import__('os').environ.get('CAT_AGENT_LOG_LEVEL'):
            logger.disable('cat_agent')


def _post(client: TestClient, name: str = 'bot', **extra):
    body = {'messages': [{'role': 'user', 'content': 'hi'}]}
    body.update(extra)
    return client.post(f'/agents/{name}/run', json=body)


class TestWireErrors:

    def test_non_verbose_hides_str_exc(self, monkeypatch):
        monkeypatch.setattr('cat_agent.serve.errors.verbose_errors_enabled', lambda: False)
        reg = AgentRegistry()
        reg.register(
            FakeAgent('Bot', raise_on_run=RuntimeError(PROVIDER_LEAK)),
            name='bot',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            r = _post(client)
        assert r.status_code == 500
        body = r.json()
        assert body['error_type'] == 'RuntimeError'
        assert body['error'] == GENERIC_ERROR
        assert 'sk-live-SECRETKEY999' not in r.text
        assert 'sk-proj-LEAKEDTOKEN123456' not in r.text
        assert 'Authorization: Bearer sk-' not in r.text

    def test_verbose_redacts_provider_secrets_from_body(self, monkeypatch):
        monkeypatch.setattr('cat_agent.serve.errors.verbose_errors_enabled', lambda: True)
        scrubbed = redact_text(PROVIDER_LEAK)
        assert 'sk-live-SECRETKEY999' not in scrubbed
        assert 'sk-proj-LEAKEDTOKEN123456' not in scrubbed
        assert 'Bearer sk-live-SECRETKEY999' not in scrubbed
        assert SECRET_PLACEHOLDER in scrubbed

        reg = AgentRegistry()
        reg.register(
            FakeAgent('Bot', raise_on_run=RuntimeError(PROVIDER_LEAK)),
            name='bot',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            r = _post(client)
        assert r.status_code == 500
        body = r.json()
        assert body['error_type'] == 'RuntimeError'
        assert body['error'] != GENERIC_ERROR
        assert 'sk-live-SECRETKEY999' not in body['error']
        assert 'sk-proj-LEAKEDTOKEN123456' not in body['error']
        assert 'Bearer sk-live-SECRETKEY999' not in body['error']
        assert 'api_key=sk-live-SECRETKEY999' not in body['error']
        assert SECRET_PLACEHOLDER in body['error']

    def test_server_log_always_has_full_detail_non_verbose(
        self, monkeypatch, capture_serve_logs,
    ):
        monkeypatch.setattr('cat_agent.serve.errors.verbose_errors_enabled', lambda: False)
        reg = AgentRegistry()
        reg.register(
            FakeAgent('Bot', raise_on_run=RuntimeError(PROVIDER_LEAK)),
            name='bot',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            r = _post(client)
        assert r.status_code == 500
        assert r.json()['error'] == GENERIC_ERROR
        assert any(PROVIDER_LEAK in line for line in capture_serve_logs), capture_serve_logs

    def test_server_log_always_has_full_detail_verbose(
        self, monkeypatch, capture_serve_logs,
    ):
        monkeypatch.setattr('cat_agent.serve.errors.verbose_errors_enabled', lambda: True)
        reg = AgentRegistry()
        reg.register(
            FakeAgent('Bot', raise_on_run=RuntimeError(PROVIDER_LEAK)),
            name='bot',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            r = _post(client)
        assert r.status_code == 500
        assert 'sk-live-SECRETKEY999' not in r.json()['error']
        assert any(PROVIDER_LEAK in line for line in capture_serve_logs), capture_serve_logs

    def test_pre_stream_failure_returns_http_500(self, monkeypatch):
        monkeypatch.setattr('cat_agent.serve.errors.verbose_errors_enabled', lambda: False)
        reg = AgentRegistry()
        reg.register(
            FakeAgent('Bot', raise_on_run=RuntimeError('boom-before-yield')),
            name='bot',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            r = _post(client, stream=True)
        assert r.status_code == 500
        assert r.json()['error_type'] == 'RuntimeError'
        assert r.json()['error'] == GENERIC_ERROR

    def test_mid_stream_failure_returns_sse_error_event(self, monkeypatch):
        monkeypatch.setattr('cat_agent.serve.errors.verbose_errors_enabled', lambda: False)
        reg = AgentRegistry()
        reg.register(
            FakeAgent(
                'Bot',
                reply='partial',
                raise_after_first_yield=RuntimeError('boom-mid-stream'),
            ),
            name='bot',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            with client.stream(
                'POST',
                '/agents/bot/run',
                json={
                    'messages': [{'role': 'user', 'content': 'hi'}],
                    'stream': True,
                },
            ) as resp:
                assert resp.status_code == 200
                text = ''.join(resp.iter_text())
        assert '"type": "turn"' in text or '"type":"turn"' in text
        assert 'partial' in text
        assert '"type": "error"' in text or '"type":"error"' in text
        assert 'RuntimeError' in text
        assert GENERIC_ERROR in text
        assert 'boom-mid-stream' not in text
