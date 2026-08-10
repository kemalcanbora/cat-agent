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

"""Request-id + access-log tests for cat_agent.serve."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cat_agent.log import logger
from cat_agent.serve import AgentRegistry, create_app, run_app
from cat_agent.serve.middleware import REQUEST_ID_HEADER
from tests.serve_fakes import FakeAgent

fastapi = pytest.importorskip('fastapi')
uvicorn = pytest.importorskip('uvicorn')
from fastapi.testclient import TestClient  # noqa: E402

SECRET_PROMPT = 'UNIQUE_PROMPT_TOKEN_ZX9Q_SHOULD_NOT_APPEAR_IN_ACCESS_LOG'


@pytest.fixture
def capture_serve_info_logs():
    lines: list[str] = []

    def _sink(message):
        lines.append(message.record['message'])

    logger.enable('cat_agent')
    sink_id = logger.add(_sink, level='INFO')
    try:
        yield lines
    finally:
        logger.remove(sink_id)
        if not os.environ.get('CAT_AGENT_LOG_LEVEL'):
            logger.disable('cat_agent')


class TestRequestIdAndAccessLog:

    def test_500_request_id_matches_header_body_and_error_log(
        self, monkeypatch, capture_serve_info_logs,
    ):
        monkeypatch.setattr('cat_agent.serve.errors.verbose_errors_enabled', lambda: False)
        reg = AgentRegistry()
        reg.register(
            FakeAgent('Bot', raise_on_run=RuntimeError('boom')),
            name='bot',
        )
        app = create_app(reg)
        client_id = 'client-req-500-abc'
        with TestClient(app) as client:
            r = client.post(
                '/agents/bot/run',
                headers={REQUEST_ID_HEADER: client_id},
                json={'messages': [{'role': 'user', 'content': 'hi'}]},
            )
        assert r.status_code == 500
        assert r.headers.get(REQUEST_ID_HEADER) == client_id
        body = r.json()
        assert body['request_id'] == client_id
        error_lines = [ln for ln in capture_serve_info_logs if 'request_id=' in ln]
        assert any(client_id in ln for ln in error_lines), capture_serve_info_logs
        access = [ln for ln in capture_serve_info_logs if ln.startswith('serve_access ')]
        assert len(access) == 1
        assert f'request_id={client_id}' in access[0]
        assert 'outcome=error' in access[0]
        assert 'status=500' in access[0]

    def test_429_request_id_matches_header_body_and_access_log(
        self, capture_serve_info_logs,
    ):
        started = threading.Event()
        release = threading.Event()
        reg = AgentRegistry(default_max_concurrency=1, default_max_queue=0)
        reg.register(
            FakeAgent('Bot', 'ok', started=started, release=release),
            name='bot',
            max_queue=0,
        )
        app = create_app(reg)
        client_id = 'client-req-429-xyz'
        with TestClient(app) as client:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f = pool.submit(
                    client.post,
                    '/agents/bot/run',
                    json={'messages': [{'role': 'user', 'content': 'hold'}]},
                )
                assert started.wait(timeout=5)
                r = client.post(
                    '/agents/bot/run',
                    headers={REQUEST_ID_HEADER: client_id},
                    json={'messages': [{'role': 'user', 'content': 'busy'}]},
                )
                release.set()
                f.result(timeout=5)

        assert r.status_code == 429
        assert r.headers.get(REQUEST_ID_HEADER) == client_id
        assert r.json()['request_id'] == client_id
        access_429 = [
            ln for ln in capture_serve_info_logs
            if ln.startswith('serve_access ') and 'outcome=capacity_full' in ln
        ]
        assert len(access_429) == 1
        assert f'request_id={client_id}' in access_429[0]
        assert 'status=429' in access_429[0]

    def test_access_log_omits_prompt_and_response_content(self, capture_serve_info_logs):
        reg = AgentRegistry()
        reg.register(FakeAgent('Bot', reply='VISIBLE_REPLY_QQ7'), name='bot')
        app = create_app(reg)
        with TestClient(app) as client:
            r = client.post(
                '/agents/bot/run',
                json={'messages': [{'role': 'user', 'content': SECRET_PROMPT}]},
            )
        assert r.status_code == 200
        assert r.json()['content'] == 'VISIBLE_REPLY_QQ7'
        access = [ln for ln in capture_serve_info_logs if ln.startswith('serve_access ')]
        assert len(access) == 1
        assert SECRET_PROMPT not in access[0]
        assert 'VISIBLE_REPLY_QQ7' not in access[0]
        assert 'outcome=ok' in access[0]
        assert 'stream=False' in access[0]

    def test_generates_request_id_when_missing(self):
        reg = AgentRegistry()
        reg.register(FakeAgent('Bot', 'ok'), name='bot')
        app = create_app(reg)
        with TestClient(app) as client:
            r = client.post(
                '/agents/bot/run',
                json={'messages': [{'role': 'user', 'content': 'hi'}]},
            )
        assert r.status_code == 200
        rid = r.headers.get(REQUEST_ID_HEADER)
        assert rid
        assert len(rid) >= 8

    def test_run_app_disables_uvicorn_access_log(self, monkeypatch):
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(uvicorn, 'run', fake_run)
        reg = AgentRegistry()
        reg.register(FakeAgent('E', 'x'), name='e')
        app = create_app(reg)
        run_app(app, host='127.0.0.1', port=8099)
        assert captured.get('access_log') is False
