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

"""Tests for cat_agent.serve (FastAPI agent registry)."""

from __future__ import annotations

from typing import Iterator, List

import pytest

from cat_agent.agent import Agent
from cat_agent.llm.schema import ASSISTANT, Message
from cat_agent.serve import AgentRegistry, create_app, load_registry
from cat_agent.serve.factory import coerce_registry
from cat_agent.serve.registry import normalize_agent_name

fastapi = pytest.importorskip('fastapi')
from fastapi.testclient import TestClient  # noqa: E402


class EchoAgent(Agent):
    def __init__(self, name: str, reply: str, description: str = ''):
        super().__init__(name=name, description=description, system_message='')
        self._reply = reply

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        yield [Message(role=ASSISTANT, content=self._reply, name=self.name)]


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(EchoAgent('Echo', 'hello-echo', description='echo bot'), name='echo')
    reg.register(EchoAgent('Other', 'hello-other'), name='other')
    return reg


@pytest.fixture
def client(registry: AgentRegistry) -> TestClient:
    app = create_app(registry)
    with TestClient(app) as c:
        yield c


class TestRegistry:

    def test_normalize_rejects_spaces(self):
        with pytest.raises(ValueError):
            normalize_agent_name('Calculator Bot')

    def test_register_and_list(self, registry: AgentRegistry):
        assert len(registry) == 2
        assert 'echo' in registry
        names = [i.name for i in registry.list_info()]
        assert names == ['echo', 'other']

    def test_duplicate_rejected(self, registry: AgentRegistry):
        with pytest.raises(ValueError, match='already registered'):
            registry.register(EchoAgent('x', 'y'), name='echo')

    def test_coerce_dict(self):
        reg = coerce_registry({'a': EchoAgent('A', 'x')})
        assert list(reg.names()) == ['a']

    def test_coerce_single_agent(self):
        reg = coerce_registry(EchoAgent('solo', 'hi'))
        assert reg.names() == ['solo']

    def test_empty_registry_rejected_by_create_app(self):
        with pytest.raises(ValueError, match='empty'):
            create_app(AgentRegistry())


class TestHTTP:

    def test_health(self, client: TestClient):
        r = client.get('/health')
        assert r.status_code == 200
        assert r.json()['status'] == 'ok'
        assert r.json()['agents'] == 2

    def test_list_agents(self, client: TestClient):
        r = client.get('/agents')
        assert r.status_code == 200
        body = r.json()
        assert {a['name'] for a in body} == {'echo', 'other'}
        echo = next(a for a in body if a['name'] == 'echo')
        assert echo['description'] == 'echo bot'

    def test_run_json(self, client: TestClient):
        r = client.post(
            '/agents/echo/run',
            json={'messages': [{'role': 'user', 'content': 'hi'}]},
        )
        assert r.status_code == 200
        data = r.json()
        assert data['agent'] == 'echo'
        assert data['content'] == 'hello-echo'
        assert data['messages'][-1]['content'] == 'hello-echo'

    def test_run_unknown(self, client: TestClient):
        r = client.post(
            '/agents/missing/run',
            json={'messages': [{'role': 'user', 'content': 'hi'}]},
        )
        assert r.status_code == 404

    def test_run_stream_sse(self, client: TestClient):
        with client.stream(
            'POST',
            '/agents/echo/run',
            json={'messages': [{'role': 'user', 'content': 'hi'}], 'stream': True},
        ) as r:
            assert r.status_code == 200
            text = ''.join(r.iter_text())
        assert '"type": "turn"' in text or '"type":"turn"' in text
        assert '"type": "done"' in text or '"type":"done"' in text
        assert 'hello-echo' in text

    def test_bearer_required(self, registry: AgentRegistry):
        app = create_app(registry, bearer_token='secret')
        with TestClient(app) as c:
            denied = c.post(
                '/agents/echo/run',
                json={'messages': [{'role': 'user', 'content': 'hi'}]},
            )
            assert denied.status_code == 401
            ok = c.post(
                '/agents/echo/run',
                headers={'Authorization': 'Bearer secret'},
                json={'messages': [{'role': 'user', 'content': 'hi'}]},
            )
            assert ok.status_code == 200
            assert ok.json()['content'] == 'hello-echo'


class TestFactoryLoad:

    def test_load_registry_from_module(self, tmp_path, monkeypatch):
        mod_path = tmp_path / 'serve_factory_mod.py'
        mod_path.write_text(
            'from cat_agent.serve import AgentRegistry\n'
            'from cat_agent.agent import Agent\n'
            'from cat_agent.llm.schema import ASSISTANT, Message\n'
            'class E(Agent):\n'
            '    def __init__(self):\n'
            "        super().__init__(name='e', system_message='')\n"
            '    def _run(self, messages, lang="en", **kwargs):\n'
            "        yield [Message(role=ASSISTANT, content='ok', name=self.name)]\n"
            'def build():\n'
            '    r = AgentRegistry()\n'
            "    r.register(E(), name='e')\n"
            '    return r\n',
            encoding='utf-8',
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        reg = load_registry('serve_factory_mod:build')
        assert reg.names() == ['e']
