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

"""Capacity / queue tests for cat_agent.serve."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from cat_agent.serve import AgentRegistry, create_app
from tests.serve_fakes import FakeAgent

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402


def _post(client: TestClient, name: str = 'bot'):
    return client.post(
        f'/agents/{name}/run',
        json={'messages': [{'role': 'user', 'content': 'hi'}]},
    )


def _assert_counters_zero(registry: AgentRegistry, name: str = 'bot') -> None:
    stats = registry.capacity_stats(name)
    assert stats['inflight'] == 0, stats
    assert stats['waiters'] == 0, stats


class TestCapacityQueue:

    def test_third_request_gets_429_with_retry_after(self):
        started = threading.Event()
        release = threading.Event()
        reg = AgentRegistry(default_max_concurrency=1, default_max_queue=1)
        reg.register(
            FakeAgent('Bot', 'ok', started=started, release=release),
            name='bot',
            max_concurrency=1,
            max_queue=1,
        )
        app = create_app(reg)

        with TestClient(app) as client:
            results = {}

            def run_held():
                results['held'] = _post(client)

            def run_queued():
                # Wait until first holds the slot, then queue
                assert started.wait(timeout=5)
                time.sleep(0.05)
                results['queued'] = _post(client)

            def run_rejected():
                assert started.wait(timeout=5)
                # Give queued request time to become a waiter
                deadline = time.time() + 2
                while time.time() < deadline:
                    if reg.capacity_stats('bot')['waiters'] >= 1:
                        break
                    time.sleep(0.01)
                results['rejected'] = _post(client)

            with ThreadPoolExecutor(max_workers=3) as pool:
                f_held = pool.submit(run_held)
                assert started.wait(timeout=5)
                f_queued = pool.submit(run_queued)
                f_rej = pool.submit(run_rejected)
                # Wait until rejected finishes (should be fast 429)
                f_rej.result(timeout=5)
                release.set()
                f_held.result(timeout=5)
                f_queued.result(timeout=5)

            rejected = results['rejected']
            assert rejected.status_code == 429
            assert rejected.headers.get('Retry-After') is not None
            body = rejected.json()
            assert body['agent'] == 'bot'
            assert body['error_type'] == 'CapacityFull'

            assert results['held'].status_code == 200
            assert results['queued'].status_code == 200
            _assert_counters_zero(reg)

    def test_max_queue_zero_rejects_when_busy_serves_when_idle(self):
        started = threading.Event()
        release = threading.Event()
        reg = AgentRegistry(default_max_concurrency=1, default_max_queue=0)
        reg.register(
            FakeAgent('Bot', 'ok', started=started, release=release),
            name='bot',
            max_concurrency=1,
            max_queue=0,
        )
        app = create_app(reg)

        with TestClient(app) as client:
            # Idle: must serve (no gate — release already unset would block; clear gates)
            # Use a one-shot idle agent path: release immediately for the idle call.
            release.set()
            idle = _post(client)
            assert idle.status_code == 200
            assert idle.json()['content'] == 'ok'
            _assert_counters_zero(reg)

            # Reset gates for the busy hold
            release.clear()
            started.clear()

            results = {}

            def hold():
                results['hold'] = _post(client)

            with ThreadPoolExecutor(max_workers=2) as pool:
                f_hold = pool.submit(hold)
                assert started.wait(timeout=5)
                busy = _post(client)
                assert busy.status_code == 429
                assert busy.json()['agent'] == 'bot'
                stats = reg.capacity_stats('bot')
                assert stats['waiters'] == 0
                assert stats['inflight'] == 1
                release.set()
                f_hold.result(timeout=5)

            assert results['hold'].status_code == 200
            _assert_counters_zero(reg)

    def test_counters_zero_after_success(self):
        reg = AgentRegistry(default_max_concurrency=1, default_max_queue=1)
        reg.register(FakeAgent('Bot', 'ok'), name='bot')
        app = create_app(reg)
        with TestClient(app) as client:
            assert _post(client).status_code == 200
            _assert_counters_zero(reg)

    def test_counters_zero_after_agent_exception(self):
        reg = AgentRegistry(default_max_concurrency=1, default_max_queue=1)
        reg.register(
            FakeAgent('Bot', raise_on_run=RuntimeError('boom')),
            name='bot',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            r = _post(client)
            assert r.status_code == 500
            assert r.json()['error_type'] == 'RuntimeError'
            _assert_counters_zero(reg)

    def test_counters_zero_after_429(self):
        started = threading.Event()
        release = threading.Event()
        reg = AgentRegistry(default_max_concurrency=1, default_max_queue=0)
        reg.register(
            FakeAgent('Bot', 'ok', started=started, release=release),
            name='bot',
            max_queue=0,
        )
        app = create_app(reg)
        with TestClient(app) as client:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f = pool.submit(_post, client)
                assert started.wait(timeout=5)
                assert _post(client).status_code == 429
                # After 429, waiters must be 0 (never reserved)
                assert reg.capacity_stats('bot')['waiters'] == 0
                release.set()
                f.result(timeout=5)
            _assert_counters_zero(reg)

    def test_counters_zero_after_client_disconnect(self):
        started = threading.Event()
        release = threading.Event()
        reg = AgentRegistry(default_max_concurrency=1, default_max_queue=2)
        reg.register(
            FakeAgent('Bot', 'ok', started=started, release=release),
            name='bot',
        )
        app = create_app(reg)

        with TestClient(app) as client:
            def stream_and_drop():
                with client.stream(
                    'POST',
                    '/agents/bot/run',
                    json={
                        'messages': [{'role': 'user', 'content': 'hi'}],
                        'stream': True,
                    },
                ) as resp:
                    assert resp.status_code == 200
                    assert started.wait(timeout=5)
                    # Disconnect without consuming the body / waiting for completion
                    resp.close()

            t = threading.Thread(target=stream_and_drop)
            t.start()
            assert started.wait(timeout=5)
            # Unblock the agent so the generator can finish and hit finally
            release.set()
            t.join(timeout=5)
            assert not t.is_alive()
            # Allow finally/release to run
            deadline = time.time() + 2
            while time.time() < deadline:
                if reg.capacity_stats('bot')['inflight'] == 0 and reg.capacity_stats('bot')['waiters'] == 0:
                    break
                time.sleep(0.01)
            _assert_counters_zero(reg)

    def test_readyz_reports_queue_fields(self):
        reg = AgentRegistry(default_max_concurrency=1, default_max_queue=3)
        reg.register(FakeAgent('Bot', 'ok'), name='bot', max_queue=3)
        app = create_app(reg)
        with TestClient(app) as client:
            body = client.get('/readyz').json()
            agent = body['agents']['bot']
            assert agent['max_queue'] == 3
            assert agent['queue_waiters'] == 0
            assert agent['capacity'] == 1
