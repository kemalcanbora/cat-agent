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

"""Coverage tests for cat_agent.tools.mcp_manager (mocked / optional mcp)."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.tools import mcp_manager as mm


def _reset_manager():
    mm.MCPManager._instance = None


def _bare_manager():
    """Instance without __init__ (no event loop / mcp import)."""
    _reset_manager()
    return object.__new__(mm.MCPManager)


def test_import_error_when_mcp_missing():
    _reset_manager()
    saved = sys.modules.get('mcp')
    sys.modules['mcp'] = None
    try:
        with pytest.raises(ImportError, match='Could not import mcp'):
            mm.MCPManager()
    finally:
        if saved is not None:
            sys.modules['mcp'] = saved
        else:
            sys.modules.pop('mcp', None)
        _reset_manager()


def test_is_valid_mcp_servers_top_level():
    mgr = _bare_manager()
    assert mgr.is_valid_mcp_servers({}) is False
    assert mgr.is_valid_mcp_servers({'mcpServers': []}) is False
    assert mgr.is_valid_mcp_servers({'mcpServers': {}}) is True


def test_is_valid_mcp_servers_command_shape():
    mgr = _bare_manager()
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {
            'mem': {'command': 'npx', 'args': ['-y', 'pkg']},
        },
    }) is True
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {'mem': {'command': 1, 'args': []}},
    }) is False
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {'mem': {'command': 'npx'}},
    }) is False
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {'mem': {'command': 'npx', 'args': 'x'}},
    }) is False


def test_is_valid_mcp_servers_url_shape():
    mgr = _bare_manager()
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {'sse': {'url': 'http://localhost/sse'}},
    }) is True
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {'sse': {'url': 1}},
    }) is False
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {
            'sse': {'url': 'http://x', 'headers': {'A': 'b'}},
        },
    }) is True
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {
            'sse': {'url': 'http://x', 'headers': 'bad'},
        },
    }) is False


def test_is_valid_mcp_servers_env_and_non_dict_server():
    mgr = _bare_manager()
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {
            'x': {'command': 'c', 'args': [], 'env': {'K': 'V'}},
        },
    }) is True
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {
            'x': {'command': 'c', 'args': [], 'env': 'bad'},
        },
    }) is False
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {'x': 'not-a-dict'},
    }) is False


def test_init_config_rejects_invalid():
    mgr = _bare_manager()
    mgr.clients = {}
    mgr.loop = MagicMock()
    with pytest.raises(ValueError, match='not valid'):
        mgr.initConfig({'bad': True})


def test_create_tool_class_call():
    mgr = _bare_manager()
    mgr.clients = {}
    mgr.loop = MagicMock()

    client = MagicMock()
    tool = mgr.create_tool_class(
        register_name='srv-tool',
        register_client_id='cid1',
        tool_name='echo',
        tool_desc='d',
        tool_parameters={'type': 'object', 'properties': {}, 'required': []},
    )
    assert tool.name == 'srv-tool'
    assert tool.client_id == 'cid1'

    mgr.clients['cid1'] = client
    future = MagicMock()
    future.result.return_value = 'ok'

    with patch.object(mm, 'MCPManager', return_value=mgr):
        with patch('asyncio.run_coroutine_threadsafe', return_value=future) as rcts:
            assert tool.call('{"a": 1}') == 'ok'
            rcts.assert_called_once()


def test_monkey_patch_unavailable(monkeypatch):
    mgr = _bare_manager()
    mgr.processes = []
    monkeypatch.setitem(sys.modules, 'mcp', ModuleType('mcp'))
    sys.modules.pop('mcp.client', None)
    sys.modules.pop('mcp.client.stdio', None)
    mgr.monkey_patch_mcp_create_platform_compatible_process()


def test_monkey_patch_wraps_process():
    _reset_manager()
    mgr = _bare_manager()
    mgr.processes = []

    async def original(*_a, **_k):
        return SimpleNamespace(pid=1)

    stdio = ModuleType('mcp.client.stdio')
    stdio._create_platform_compatible_process = original
    client = ModuleType('mcp.client')
    client.stdio = stdio
    mcp_mod = ModuleType('mcp')
    mcp_mod.client = client

    with patch.dict(sys.modules, {
        'mcp': mcp_mod,
        'mcp.client': client,
        'mcp.client.stdio': stdio,
    }):
        mgr.monkey_patch_mcp_create_platform_compatible_process()
        process = asyncio.run(stdio._create_platform_compatible_process())
    assert process.pid == 1
    assert mgr.processes == [process]
    _reset_manager()


def test_cleanup_mcp_noop_without_instance():
    _reset_manager()
    mm._cleanup_mcp()


def _install_fake_mcp():
    class TextResourceContents:
        def __init__(self, text):
            self.text = text

    fake_types = ModuleType('mcp.types')
    fake_types.TextResourceContents = TextResourceContents
    fake_mcp = ModuleType('mcp')
    fake_mcp.ClientSession = object
    fake_mcp.types = fake_types
    return fake_mcp, fake_types, TextResourceContents


def test_mcp_client_reconnect_requires_client_id():
    _reset_manager()
    fake_mcp, fake_types, _ = _install_fake_mcp()
    with patch.dict(sys.modules, {'mcp': fake_mcp, 'mcp.types': fake_types}):
        client = mm.MCPClient()
        client.client_id = None
        with pytest.raises(RuntimeError, match='client_id is None'):
            asyncio.run(client.reconnect())


def test_mcp_client_execute_list_and_read_and_tool():
    _reset_manager()
    fake_mcp, fake_types, TextResourceContents = _install_fake_mcp()

    with patch.dict(sys.modules, {'mcp': fake_mcp, 'mcp.types': fake_types}):
        client = mm.MCPClient()
        session = MagicMock()

        async def send_ping():
            return None

        async def list_resources():
            return SimpleNamespace(resources=['r1', 'r2'])

        async def read_resource(uri):
            return SimpleNamespace(contents=[TextResourceContents('body')])

        async def call_tool(name, args):
            return SimpleNamespace(
                content=[SimpleNamespace(type='text', text='out')]
            )

        session.send_ping = send_ping
        session.list_resources = list_resources
        session.read_resource = read_resource
        session.call_tool = call_tool
        client.session = session

        async def run_all():
            assert 'r1' in await client.execute_function('list_resources', {})
            assert await client.execute_function(
                'read_resource', {'uri': 'file://x'}
            ) == 'body'
            assert await client.execute_function('echo', {'q': 1}) == 'out'
            assert 'Error' in await client.execute_function('read_resource', {})

        asyncio.run(run_all())


def test_mcp_client_execute_ping_fail_no_client_id():
    _reset_manager()
    fake_mcp, fake_types, _ = _install_fake_mcp()
    with patch.dict(sys.modules, {'mcp': fake_mcp, 'mcp.types': fake_types}):
        client = mm.MCPClient()
        client.client_id = None
        session = MagicMock()

        async def boom():
            raise RuntimeError('dead')

        session.send_ping = boom
        client.session = session
        out = asyncio.run(client.execute_function('echo', {}))
        assert 'client_id is None' in out


def test_init_with_mocked_mcp_skips_real_servers(monkeypatch):
    """Construct MCPManager with mcp present but without starting real servers."""
    _reset_manager()
    fake_mcp = ModuleType('mcp')
    monkeypatch.setitem(sys.modules, 'mcp', fake_mcp)
    monkeypatch.setattr(
        mm.MCPManager,
        'monkey_patch_mcp_create_platform_compatible_process',
        lambda self: None,
    )
    monkeypatch.setattr(mm.MCPManager, 'start_loop', lambda self: None)

    class FakeThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self):
            return None

        def join(self):
            return None

    monkeypatch.setattr(mm.threading, 'Thread', FakeThread)
    monkeypatch.setattr(mm, 'load_dotenv', lambda: None)
    mgr = mm.MCPManager()
    assert mgr.is_valid_mcp_servers({
        'mcpServers': {
            'fs': {
                'command': 'npx',
                'args': ['-y', '@modelcontextprotocol/server-filesystem', '/tmp'],
            },
        },
    })
    _reset_manager()
