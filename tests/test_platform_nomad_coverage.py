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

"""Coverage tests for cat_agent.platform.nomad (mocked HTTP session)."""

from __future__ import annotations

import pytest

from cat_agent.platform.config import PlatformConfig
from cat_agent.platform.nomad import (
    NomadClient,
    NomadNotFound,
    NomadRejected,
    NomadUnreachable,
)
from tests.platform_fakes import FakeResponse, FakeSession


def _cfg(**kwargs) -> PlatformConfig:
    base = dict(
        nomad_addr='http://127.0.0.1:4646',
        nomad_token='tok',
        namespace='ns1',
    )
    base.update(kwargs)
    return PlatformConfig(**base)


def _client(session: FakeSession, **cfg_kw) -> NomadClient:
    return NomadClient(_cfg(**cfg_kw), session=session)


def _ok(payload):
    def handler(_method, _path, _kwargs):
        return FakeResponse(200, payload)

    return handler


def test_headers_include_token():
    session = FakeSession()
    session.routes[('GET', '/status/leader')] = _ok('"127.0.0.1:4647"')
    client = _client(session)
    client.status_leader()
    assert session.calls[0]['headers']['X-Nomad-Token'] == 'tok'
    assert session.calls[0]['headers']['Content-Type'] == 'application/json'


def test_headers_omit_token_when_empty():
    session = FakeSession()
    session.routes[('GET', '/status/leader')] = _ok('"leader"')
    client = _client(session, nomad_token='')
    client.status_leader()
    assert 'X-Nomad-Token' not in session.calls[0]['headers']


def test_params_include_namespace():
    session = FakeSession()
    session.routes[('GET', '/nodes')] = _ok([])
    client = _client(session)
    client.nodes()
    assert session.calls[0]['params']['namespace'] == 'ns1'


def test_status_leader():
    session = FakeSession()
    session.routes[('GET', '/status/leader')] = _ok('leader-addr')
    assert _client(session).status_leader() == 'leader-addr'


def test_nodes_and_node():
    session = FakeSession()
    session.routes[('GET', '/nodes')] = _ok([{'ID': 'n1'}])
    session.routes[('GET', '/node/n1')] = _ok({'ID': 'n1', 'Name': 'node1'})
    client = _client(session)
    assert client.nodes() == [{'ID': 'n1'}]
    assert client.node('n1')['Name'] == 'node1'


def test_parse_job_and_submit():
    session = FakeSession()
    session.routes[('POST', '/jobs/parse')] = _ok({'ID': 'job-a'})
    session.routes[('POST', '/jobs')] = _ok({'EvalID': 'e1'})
    client = _client(session)
    job = client.parse_job('job "a" {}')
    assert job['ID'] == 'job-a'
    assert client.submit(job)['EvalID'] == 'e1'


def test_submit_hcl():
    session = FakeSession()
    session.routes[('POST', '/jobs/parse')] = _ok({'ID': 'hcl-job'})
    session.routes[('POST', '/jobs')] = _ok({'EvalID': 'e2'})
    out = _client(session).submit_hcl('job "hcl-job" {}')
    assert out['EvalID'] == 'e2'


def test_get_job_and_list_jobs():
    session = FakeSession()
    session.routes[('GET', '/job/j1')] = _ok({'ID': 'j1'})
    session.routes[('GET', '/jobs')] = _ok([{'ID': 'j1'}])
    client = _client(session)
    assert client.get_job('j1')['ID'] == 'j1'
    assert client.list_jobs() == [{'ID': 'j1'}]


def test_list_agents_filters_by_meta_and_team():
    session = FakeSession()

    def jobs(_m, _p, _k):
        return FakeResponse(200, [{'ID': 'a'}, {'ID': 'b'}, {'ID': 'c'}, {'Name': 'n'}])

    def get_job(_m, path, _k):
        jid = path.rsplit('/', 1)[-1]
        meta = {
            'a': {'managed_by': 'cat-agent', 'team': 'demo'},
            'b': {'managed_by': 'other'},
            'c': {'managed_by': 'cat-agent', 'team': 'ops'},
            'n': {},
        }.get(jid, {})
        return FakeResponse(200, {'ID': jid, 'Meta': meta})

    session.routes[('GET', '/jobs')] = jobs
    session.routes[('GET', '/job/')] = get_job  # prefix match
    client = _client(session)
    all_agents = client.list_agents()
    assert {j['ID'] for j in all_agents} == {'a', 'c'}
    demo = client.list_agents(team='demo')
    assert [j['ID'] for j in demo] == ['a']


def test_list_agents_skips_not_found():
    session = FakeSession()
    session.routes[('GET', '/jobs')] = _ok([{'ID': 'gone'}])

    def missing(_m, _p, _k):
        return FakeResponse(404, text='missing')

    session.routes[('GET', '/job/gone')] = missing
    assert _client(session).list_agents() == []


def test_allocations_deployment_versions():
    session = FakeSession()
    session.routes[('GET', '/job/j1/allocations')] = _ok([{'ID': 'alloc1'}])
    session.routes[('GET', '/job/j1/deployments')] = _ok([{'Status': 'successful'}])
    session.routes[('GET', '/job/j1/versions')] = _ok({'Versions': [{'Version': 1}]})
    client = _client(session)
    assert client.allocations('j1')[0]['ID'] == 'alloc1'
    assert client.deployment_status('j1')['Status'] == 'successful'
    assert client.job_versions('j1') == [{'Version': 1}]


def test_deployment_status_none_and_versions_list():
    session = FakeSession()
    session.routes[('GET', '/job/j1/deployments')] = _ok([])
    session.routes[('GET', '/job/j1/versions')] = _ok([{'Version': 2}])
    client = _client(session)
    assert client.deployment_status('j1') is None
    assert client.job_versions('j1') == [{'Version': 2}]


def test_stop_and_dispatch():
    session = FakeSession()
    session.routes[('DELETE', '/job/j1')] = _ok({'EvalID': 'stop'})
    session.routes[('POST', '/job/j1/dispatch')] = _ok({'DispatchedJobID': 'd1'})
    client = _client(session)
    assert client.stop('j1', purge=True)['EvalID'] == 'stop'
    # purge flag is passed on the DELETE call (merged with namespace in _params)
    delete_calls = [c for c in session.calls if c.get('method') == 'DELETE']
    assert delete_calls, f'expected DELETE call, got {session.calls!r}'
    purge = (delete_calls[-1].get('params') or {}).get('purge')
    assert purge in (True, 'true', 1, '1')
    assert client.dispatch('j1', payload='p', meta={'k': 'v'})['DispatchedJobID'] == 'd1'


def test_logs_returns_plain_string():
    session = FakeSession()

    def logs_handler(_m, _p, kwargs):
        assert kwargs['params']['type'] == 'stdout'
        return FakeResponse(200, 'line1\n')

    session.routes[('GET', '/client/fs/logs/alloc1')] = logs_handler
    assert _client(session).logs('alloc1', 'task') == 'line1\n'


def test_logs_non_string_payload():
    session = FakeSession()
    session.routes[('GET', '/client/fs/logs/alloc1')] = _ok({'raw': 1})
    out = _client(session).logs('alloc1', 'task', stderr=True)
    assert 'raw' in out


def test_request_unreachable():
    class BoomSession:
        def request(self, *a, **k):
            raise ConnectionError('down')

    client = NomadClient(_cfg(), session=BoomSession())
    with pytest.raises(NomadUnreachable, match='unreachable'):
        client.status_leader()


def test_request_404_403_and_400():
    session = FakeSession()

    def not_found(_m, _p, _k):
        return FakeResponse(404, text='gone')

    def forbidden(_m, _p, _k):
        return FakeResponse(403, text='nope')

    def bad(_m, _p, _k):
        return FakeResponse(500, text='boom')

    session.routes[('GET', '/job/missing')] = not_found
    session.routes[('GET', '/job/forbidden')] = forbidden
    session.routes[('GET', '/job/bad')] = bad
    client = _client(session)
    with pytest.raises(NomadNotFound):
        client.get_job('missing')
    with pytest.raises(NomadRejected, match='403'):
        client.get_job('forbidden')
    with pytest.raises(NomadRejected, match='500'):
        client.get_job('bad')


def test_request_empty_and_non_json():
    session = FakeSession()

    def empty(_m, _p, _k):
        return FakeResponse(200, payload=None, text='')

    def plain(_m, _p, _k):
        resp = FakeResponse(200, payload=None, text='not-json')
        resp.json = lambda: (_ for _ in ()).throw(ValueError('bad json'))
        return resp

    session.routes[('GET', '/status/leader')] = empty
    client = _client(session)
    assert client._request('GET', '/v1/status/leader') is None
    session.routes[('GET', '/status/leader')] = plain
    assert client._request('GET', '/v1/status/leader') == 'not-json'


def test_wait_healthy_success(monkeypatch):
    session = FakeSession()
    session.routes[('GET', '/job/j1/allocations')] = _ok([
        {'ClientStatus': 'running', 'DesiredStatus': 'run'},
    ])
    monkeypatch.setattr('cat_agent.platform.nomad.time.sleep', lambda *_: None)
    _client(session).wait_healthy('j1', timeout=5)


def test_wait_healthy_failed_alloc(monkeypatch):
    session = FakeSession()
    session.routes[('GET', '/job/j1/allocations')] = _ok([
        {'ClientStatus': 'failed'},
    ])
    monkeypatch.setattr('cat_agent.platform.nomad.time.sleep', lambda *_: None)
    with pytest.raises(NomadRejected, match='failed allocations'):
        _client(session).wait_healthy('j1', timeout=5)


def test_wait_healthy_timeout(monkeypatch):
    session = FakeSession()
    session.routes[('GET', '/job/j1/allocations')] = _ok([])
    times = iter([0.0, 0.0, 10.0])
    monkeypatch.setattr('cat_agent.platform.nomad.time.time', lambda: next(times))
    monkeypatch.setattr('cat_agent.platform.nomad.time.sleep', lambda *_: None)
    with pytest.raises(NomadRejected, match='timed out'):
        _client(session).wait_healthy('j1', timeout=5)


def test_watch_deployment_successful(monkeypatch):
    session = FakeSession()
    session.routes[('GET', '/job/j1/deployments')] = _ok([{'Status': 'successful'}])
    monkeypatch.setattr('cat_agent.platform.nomad.time.sleep', lambda *_: None)
    lines = list(_client(session).watch_deployment('j1', timeout=5))
    assert lines == ['deployment successful']


def test_watch_deployment_failed(monkeypatch):
    session = FakeSession()
    session.routes[('GET', '/job/j1/deployments')] = _ok([{'Status': 'failed'}])
    monkeypatch.setattr('cat_agent.platform.nomad.time.sleep', lambda *_: None)
    with pytest.raises(NomadRejected, match='failed'):
        list(_client(session).watch_deployment('j1', timeout=5))


def test_watch_deployment_fallback_running(monkeypatch):
    session = FakeSession()
    session.routes[('GET', '/job/j1/deployments')] = _ok([])
    session.routes[('GET', '/job/j1/allocations')] = _ok([
        {'ClientStatus': 'running'},
    ])
    monkeypatch.setattr('cat_agent.platform.nomad.time.sleep', lambda *_: None)
    lines = list(_client(session).watch_deployment('j1', timeout=5))
    assert 'allocation running' in lines


def test_watch_deployment_fallback_complete(monkeypatch):
    session = FakeSession()
    session.routes[('GET', '/job/j1/deployments')] = _ok([])
    session.routes[('GET', '/job/j1/allocations')] = _ok([
        {'ClientStatus': 'complete'},
    ])
    monkeypatch.setattr('cat_agent.platform.nomad.time.sleep', lambda *_: None)
    lines = list(_client(session).watch_deployment('j1', timeout=5))
    assert 'allocation complete' in lines


def test_watch_deployment_timeout(monkeypatch):
    session = FakeSession()
    session.routes[('GET', '/job/j1/deployments')] = _ok([{'Status': 'running'}])
    times = iter([0.0, 0.0, 1000.0])
    monkeypatch.setattr('cat_agent.platform.nomad.time.time', lambda: next(times))
    monkeypatch.setattr('cat_agent.platform.nomad.time.sleep', lambda *_: None)
    with pytest.raises(NomadRejected, match='timed out watching'):
        list(_client(session).watch_deployment('j1', timeout=5))
