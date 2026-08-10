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

"""run_app config resolution and workers>1 guard tests."""

from __future__ import annotations

import pytest

from cat_agent.serve import AgentRegistry, create_app, run_app
from cat_agent.serve.server import _UNSET, _resolve_host, _resolve_port, _resolve_shutdown_timeout
from tests.serve_fakes import FakeAgent, make_factory

uvicorn = pytest.importorskip('uvicorn')


class TestResolveHostPort:

    def test_omitted_host_uses_env(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_SERVE_HOST', '0.0.0.0')
        assert _resolve_host(_UNSET) == '0.0.0.0'

    def test_omitted_host_default(self, monkeypatch):
        monkeypatch.delenv('CAT_AGENT_SERVE_HOST', raising=False)
        assert _resolve_host(_UNSET) == '127.0.0.1'

    def test_explicit_host_wins(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_SERVE_HOST', '0.0.0.0')
        assert _resolve_host('192.168.1.1') == '192.168.1.1'

    def test_explicit_none_host_raises(self):
        with pytest.raises(ValueError, match='host=None'):
            _resolve_host(None)

    def test_omitted_port_uses_cat_agent_env(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_SERVE_PORT', '9090')
        monkeypatch.setenv('PORT', '3000')
        assert _resolve_port(_UNSET) == 9090

    def test_omitted_port_falls_back_to_port(self, monkeypatch):
        monkeypatch.delenv('CAT_AGENT_SERVE_PORT', raising=False)
        monkeypatch.setenv('PORT', '3000')
        assert _resolve_port(_UNSET) == 3000

    def test_explicit_port_wins(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_SERVE_PORT', '9090')
        monkeypatch.setenv('PORT', '3000')
        assert _resolve_port(8081) == 8081

    def test_explicit_none_port_raises(self):
        with pytest.raises(ValueError, match='port=None'):
            _resolve_port(None)

    def test_shutdown_timeout_from_settings(self, monkeypatch):
        monkeypatch.setattr('cat_agent.settings.SERVE_SHUTDOWN_TIMEOUT', 33)
        assert _resolve_shutdown_timeout(_UNSET) == 33

    def test_explicit_none_shutdown_raises(self):
        with pytest.raises(ValueError, match='timeout_graceful_shutdown=None'):
            _resolve_shutdown_timeout(None)


class TestRunAppKwargs:

    def test_run_app_passes_resolved_env(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_SERVE_HOST', '0.0.0.0')
        monkeypatch.setenv('CAT_AGENT_SERVE_PORT', '9555')
        monkeypatch.setattr('cat_agent.settings.SERVE_SHUTDOWN_TIMEOUT', 17)
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(uvicorn, 'run', fake_run)
        reg = AgentRegistry()
        reg.register(FakeAgent('E', 'x'), name='e')
        app = create_app(reg)
        run_app(app)
        assert captured['host'] == '0.0.0.0'
        assert captured['port'] == 9555
        assert captured['timeout_graceful_shutdown'] == 17
        assert captured['workers'] == 1

    def test_explicit_args_override_env(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_SERVE_HOST', '0.0.0.0')
        monkeypatch.setenv('CAT_AGENT_SERVE_PORT', '9555')
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(uvicorn, 'run', fake_run)
        reg = AgentRegistry()
        reg.register(FakeAgent('E', 'x'), name='e')
        app = create_app(reg)
        run_app(app, host='127.0.0.1', port=8080, timeout_graceful_shutdown=9)
        assert captured['host'] == '127.0.0.1'
        assert captured['port'] == 8080
        assert captured['timeout_graceful_shutdown'] == 9

    def test_explicit_none_does_not_call_uvicorn(self, monkeypatch):
        called = []

        def fake_run(*args, **kwargs):
            called.append(True)

        monkeypatch.setattr(uvicorn, 'run', fake_run)
        reg = AgentRegistry()
        reg.register(FakeAgent('E', 'x'), name='e')
        app = create_app(reg)
        with pytest.raises(ValueError, match='host=None'):
            run_app(app, host=None)
        assert called == []

    def test_workers_gt1_with_deferred_raises_before_uvicorn(self, monkeypatch):
        called = []

        def fake_run(*args, **kwargs):
            called.append(True)

        monkeypatch.setattr(uvicorn, 'run', fake_run)
        reg = AgentRegistry()
        reg.register_factory(make_factory('d'), name='d')
        # Build eagerly for create_app emptiness — factory still marks deferred
        app = create_app(reg)
        with pytest.raises(RuntimeError, match='workers>1'):
            run_app(app, workers=2)
        assert called == []

    def test_workers_gt1_with_eager_only_allowed(self, monkeypatch):
        captured = {}

        def fake_run(*args, **kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(uvicorn, 'run', fake_run)
        reg = AgentRegistry()
        reg.register(FakeAgent('E', 'x'), name='e')
        app = create_app(reg)
        run_app(app, workers=2)
        assert captured['workers'] == 2
