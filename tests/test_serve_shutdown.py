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

"""Graceful shutdown behaviour for cat_agent.serve."""

from __future__ import annotations

import threading

import pytest

from cat_agent.serve import AgentRegistry, create_app
from tests.serve_fakes import FakeAgent


class TrackingAgent(FakeAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aclose_calls = 0
        self.aclose_event = threading.Event()

    async def aclose(self) -> None:
        self.aclose_calls += 1
        self.aclose_event.set()
        await super().aclose()


class TestGracefulShutdown:

    def test_testclient_lifespan_calls_aclose(self):
        """Lifespan teardown path used by create_app (no real SIGTERM)."""
        fastapi = pytest.importorskip('fastapi')
        from fastapi.testclient import TestClient

        agent = TrackingAgent('Bot', 'ok')
        reg = AgentRegistry()
        reg.register(agent, name='bot')
        app = create_app(reg)
        with TestClient(app) as client:
            assert client.post(
                '/agents/bot/run',
                json={'messages': [{'role': 'user', 'content': 'hi'}]},
            ).status_code == 200
        assert agent.aclose_calls >= 1
