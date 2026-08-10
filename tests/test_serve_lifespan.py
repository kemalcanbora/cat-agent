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

"""Lifespan / deferred-registry / readiness tests for cat_agent.serve."""

from __future__ import annotations

import pytest

from cat_agent.serve import AgentRegistry, create_app
from tests.serve_fakes import ConstructionTracker, FakeAgent, make_factory

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402


def _run(client: TestClient, name: str, text: str = 'hi'):
    return client.post(
        f'/agents/{name}/run',
        json={'messages': [{'role': 'user', 'content': text}]},
    )


class TestDeferredLifespan:

    def test_factory_builds_at_startup_and_serves(self):
        reg = AgentRegistry()
        reg.register_factory(make_factory('calc', reply='sum-ok'), name='calculator')
        app = create_app(reg)
        with TestClient(app) as client:
            assert client.get('/healthz').status_code == 200
            assert client.get('/healthz').json() == {'status': 'ok'}
            ready = client.get('/readyz')
            assert ready.status_code == 200
            body = ready.json()
            assert body['status'] == 'ready'
            assert body['agents']['calculator']['state'] == 'ready'
            assert body['agents']['calculator']['capacity'] == 1
            assert body['agents']['calculator']['inflight'] == 0
            r = _run(client, 'calculator')
            assert r.status_code == 200
            assert r.json()['content'] == 'sum-ok'

    def test_factory_failure_keeps_app_bound(self):
        reg = AgentRegistry()
        reg.register_factory(
            make_factory('broken', error=OSError('model file not found')),
            name='broken',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            assert client.get('/healthz').status_code == 200
            assert client.get('/health').status_code == 200
            ready = client.get('/readyz')
            assert ready.status_code == 503
            body = ready.json()
            assert body['status'] == 'not_ready'
            assert body['agents']['broken']['state'] == 'failed'
            assert body['agents']['broken']['error_type'] == 'OSError'
            assert 'model file not found' in body['agents']['broken']['error']
            r = _run(client, 'broken')
            assert r.status_code == 503

    def test_eager_registration_still_works(self):
        reg = AgentRegistry()
        reg.register(FakeAgent('Echo', 'hello-echo'), name='echo')
        app = create_app(reg)
        with TestClient(app) as client:
            assert client.get('/readyz').status_code == 200
            r = _run(client, 'echo')
            assert r.status_code == 200
            assert r.json()['content'] == 'hello-echo'

    def test_builds_are_sequential_not_concurrent(self):
        tracker = ConstructionTracker()
        reg = AgentRegistry()
        reg.register_factory(
            make_factory('a', delay=0.05, tracker=tracker),
            name='a',
        )
        reg.register_factory(
            make_factory('b', delay=0.05, tracker=tracker),
            name='b',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            assert client.get('/readyz').status_code == 200
        assert tracker.max_active == 1
        starts = [e for e in tracker.events if e[0] == 'start']
        ends = [e for e in tracker.events if e[0] == 'end']
        assert [s[1] for s in starts] == ['a', 'b']
        # First must finish before second starts
        end_a = next(e[2] for e in ends if e[1] == 'a')
        start_b = next(e[2] for e in starts if e[1] == 'b')
        assert end_a <= start_b

    def test_partial_failure_healthy_agent_still_serves(self):
        reg = AgentRegistry()
        reg.register_factory(make_factory('ok', reply='from-ok'), name='ok')
        reg.register_factory(
            make_factory('bad', error=OSError('model file not found')),
            name='bad',
        )
        app = create_app(reg)
        with TestClient(app) as client:
            assert client.get('/healthz').status_code == 200
            ready = client.get('/readyz')
            assert ready.status_code == 503
            body = ready.json()
            assert body['status'] == 'not_ready'
            assert body['agents']['ok']['state'] == 'ready'
            assert body['agents']['bad']['state'] == 'failed'
            assert body['agents']['bad']['error_type'] == 'OSError'

            ok = _run(client, 'ok')
            assert ok.status_code == 200
            assert ok.json()['content'] == 'from-ok'

            bad = _run(client, 'bad')
            assert bad.status_code == 503

    def test_health_unchanged_shape(self):
        reg = AgentRegistry()
        reg.register(FakeAgent('Echo', 'x'), name='echo')
        app = create_app(reg)
        with TestClient(app) as client:
            r = client.get('/health')
            assert r.status_code == 200
            assert r.json() == {'status': 'ok', 'agents': 1}

    def test_has_deferred_factories_flag(self):
        eager = AgentRegistry()
        eager.register(FakeAgent('E', 'x'), name='e')
        assert eager.has_deferred_factories is False

        deferred = AgentRegistry()
        deferred.register_factory(make_factory('d'), name='d')
        assert deferred.has_deferred_factories is True
        app = create_app(deferred)
        with TestClient(app):
            pass
        # Flag remains after successful build (for workers>1 guard later)
        assert deferred.has_deferred_factories is True
