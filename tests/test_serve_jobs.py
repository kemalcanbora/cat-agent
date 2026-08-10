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

"""Inline job table + HTTP routes."""

from __future__ import annotations

import threading
import time

import pytest

from cat_agent.serve import AgentRegistry, create_app
from cat_agent.serve.jobs import InlineJobTable
from tests.serve_fakes import FakeAgent

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402


def _payload():
    return {'messages': [{'role': 'user', 'content': 'hi'}]}


class TestInlineJobHTTP:

    def test_lifecycle_succeeds(self):
        reg = AgentRegistry()
        reg.register(FakeAgent('Bot', 'done'), name='bot')
        app = create_app(reg)
        with TestClient(app) as client:
            r = client.post('/agents/bot/jobs', json=_payload())
            assert r.status_code == 202
            job_id = r.json()['job_id']
            deadline = time.time() + 5
            while time.time() < deadline:
                st = client.get(f'/agents/bot/jobs/{job_id}')
                assert st.status_code == 200
                body = st.json()
                if body['state'] in ('succeeded', 'failed', 'cancelled'):
                    break
                time.sleep(0.05)
            assert body['state'] == 'succeeded'
            assert body['result']['content'] == 'done'

    def test_cancel_running(self):
        started = threading.Event()
        release = threading.Event()
        reg = AgentRegistry()
        reg.register(
            FakeAgent('Bot', 'done', started=started, release=release),
            name='bot',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            r = client.post('/agents/bot/jobs', json=_payload())
            assert r.status_code == 202
            job_id = r.json()['job_id']
            assert started.wait(5)
            c = client.delete(f'/agents/bot/jobs/{job_id}')
            assert c.status_code == 200
            assert c.json()['state'] == 'cancelled'
            release.set()

    def test_table_full_returns_429(self, monkeypatch):
        monkeypatch.setattr('cat_agent.settings.SERVE_JOB_MAX', 1)
        # create_app reads settings at construction — patch InlineJobTable via settings
        # already imported into create_app; recreate after patching the module attr
        # create_app imports SERVE_JOB_MAX locally — patch before create_app
        monkeypatch.setattr('cat_agent.serve.app.SERVE_JOB_MAX', 1, raising=False)

        from cat_agent.settings import SERVE_JOB_TTL_SECONDS

        reg = AgentRegistry()
        reg.register(FakeAgent('Bot', 'done', run_delay=0.01), name='bot')
        # Build app then replace job table with tiny max
        app = create_app(reg)
        app.state.jobs = InlineJobTable(max_jobs=1, finished_ttl_seconds=SERVE_JOB_TTL_SECONDS)
        with TestClient(app) as client:
            # Hold one terminal slot by submitting then immediately filling with another
            # while first still in table (TTL long)
            r1 = client.post('/agents/bot/jobs', json=_payload())
            assert r1.status_code == 202
            # Wait until finished so capacity isn't the issue — table still holds 1
            job_id = r1.json()['job_id']
            deadline = time.time() + 5
            while time.time() < deadline:
                if client.get(f'/agents/bot/jobs/{job_id}').json()['state'] == 'succeeded':
                    break
                time.sleep(0.02)
            r2 = client.post('/agents/bot/jobs', json=_payload())
            assert r2.status_code == 429
            assert r2.json()['error_type'] == 'JobTableFull'
            assert r2.headers.get('Retry-After')

    def test_ttl_eviction(self):
        table = InlineJobTable(max_jobs=2, finished_ttl_seconds=0.05)

        async def ok():
            return {'content': 'x'}

        import asyncio

        from cat_agent.serve.jobs import JobNotFound

        async def scenario():
            jid = await table.submit('bot', ok)
            deadline = time.time() + 2
            while time.time() < deadline:
                rec = await table.get('bot', jid)
                if rec.state == 'succeeded':
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.08)
            with pytest.raises(JobNotFound):
                await table.get('bot', jid)
            jid2 = await table.submit('bot', ok)
            assert jid2 != jid

        asyncio.run(scenario())

    def test_shutdown_marks_cancelled(self):
        started = threading.Event()
        release = threading.Event()
        reg = AgentRegistry()
        reg.register(
            FakeAgent('Bot', 'done', started=started, release=release),
            name='bot',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            r = client.post('/agents/bot/jobs', json=_payload())
            assert r.status_code == 202
            job_id = r.json()['job_id']
            assert started.wait(5)
            # Exit context → lifespan shutdown cancels jobs
        # After shutdown the table should have cancelled the job; poke via table
        # (HTTP client is closed). Read from app.state before GC — still on app.
        import asyncio

        async def check():
            rec = await app.state.jobs.get('bot', job_id)
            assert rec.state == 'cancelled'

        asyncio.run(check())
        release.set()
