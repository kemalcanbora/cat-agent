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

"""Extra coverage for cat_agent.tools.mcp_manager (mocked mcp, no real servers)."""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cat_agent.tools import mcp_manager as mm


def _reset():
    mm.MCPManager._instance = None


def _bare():
    _reset()
    return object.__new__(mm.MCPManager)


def _install_fake_mcp(monkeypatch):
    """Install a minimal mcp package graph into sys.modules."""

    class TextResourceContents:
        def __init__(self, text):
            self.text = text

    class ClientSession:
        def __init__(self, *a, **k):
            self._init = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            self._init = True

        async def list_tools(self):
            return SimpleNamespace(tools=[
                SimpleNamespace(
                    name='echo',
                    description='echo tool',
                    inputSchema={
                        'type': 'object',
                        'properties': {'q': {'type': 'string'}},
                    },
                ),
                SimpleNamespace(
                    name='bad_schema',
                    description='missing fields',
                    inputSchema={'type': 'object', 'properties': {}},
                ),
            ])

        async def list_resources(self):
            return SimpleNamespace(resources=['res-a'])

        async def list_resource_templates(self):
            return SimpleNamespace(resourceTemplates=['tmpl://{id}'])

        async def send_ping(self):
            return None

        async def call_tool(self, name, args):
            return SimpleNamespace(content=[SimpleNamespace(type='text', text=f'{name}:{args}')])

        async def read_resource(self, uri):
            return SimpleNamespace(contents=[TextResourceContents(f'body:{uri}')])

    class StdioServerParameters:
        def __init__(self, command, args, env=None):
            self.command = command
            self.args = args
            self.env = env

    @asynccontextmanager
    async def stdio_client(_params):
        yield (MagicMock(), MagicMock())

    @asynccontextmanager
    async def sse_client(url, headers=None, sse_read_timeout=300):
        yield (MagicMock(), MagicMock())

    @asynccontextmanager
    async def streamablehttp_client(url, headers=None, sse_read_timeout=None):
        yield (MagicMock(), MagicMock(), lambda: 'sid')

    types_mod = ModuleType('mcp.types')
    types_mod.TextResourceContents = TextResourceContents

    stdio_mod = ModuleType('mcp.client.stdio')
    stdio_mod.stdio_client = stdio_client
    stdio_mod._create_platform_compatible_process = AsyncMock(return_value=SimpleNamespace(pid=99))

    sse_mod = ModuleType('mcp.client.sse')
    sse_mod.sse_client = sse_client

    http_mod = ModuleType('mcp.client.streamable_http')
    http_mod.streamablehttp_client = streamablehttp_client

    client_mod = ModuleType('mcp.client')
    client_mod.stdio = stdio_mod
    client_mod.sse = sse_mod
    client_mod.streamable_http = http_mod

    mcp_mod = ModuleType('mcp')
    mcp_mod.ClientSession = ClientSession
    mcp_mod.StdioServerParameters = StdioServerParameters
    mcp_mod.types = types_mod
    mcp_mod.client = client_mod

    for name, mod in {
        'mcp': mcp_mod,
        'mcp.types': types_mod,
        'mcp.client': client_mod,
        'mcp.client.stdio': stdio_mod,
        'mcp.client.sse': sse_mod,
        'mcp.client.streamable_http': http_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return mcp_mod, ClientSession, TextResourceContents


@pytest.fixture(autouse=True)
def _cleanup_singleton():
    _reset()
    yield
    _reset()


def test_start_loop_exception_handler_filters_cancel_scope():
    mgr = _bare()
    mgr.loop = MagicMock()
    handlers = {}

    def set_handler(h):
        handlers['h'] = h

    mgr.loop.set_exception_handler = set_handler
    mgr.loop.run_forever = MagicMock()
    mgr.loop.default_exception_handler = MagicMock()

    with patch('asyncio.set_event_loop'):
        # Run only the setup part by making run_forever return immediately
        mgr.start_loop()

    h = handlers['h']
    h(mgr.loop, {'exception': RuntimeError('Attempted to exit cancel scope in a different task')})
    mgr.loop.default_exception_handler.assert_not_called()

    try:
        eg = ExceptionGroup(
            'g',
            [RuntimeError('Attempted to exit cancel scope in a different task')],
        )
        h(mgr.loop, {'exception': eg})
    except NameError:
        pass

    h(mgr.loop, {'exception': ValueError('other')})
    mgr.loop.default_exception_handler.assert_called()


def test_init_config_success_and_failure():
    mgr = _bare()
    mgr.clients = {}
    mgr.loop = MagicMock()
    future = MagicMock()
    future.result.return_value = ['tool']
    with patch('asyncio.run_coroutine_threadsafe', return_value=future):
        out = mgr.initConfig({'mcpServers': {'mem': {'command': 'npx', 'args': []}}})
    assert out == ['tool']

    future.result.side_effect = RuntimeError('fail')
    with patch('asyncio.run_coroutine_threadsafe', return_value=future):
        with pytest.raises(RuntimeError, match='fail'):
            mgr.initConfig({'mcpServers': {'mem': {'command': 'npx', 'args': []}}})


def test_init_config_async_stdio_with_resources(monkeypatch):
    _install_fake_mcp(monkeypatch)
    mgr = _bare()
    mgr.clients = {}

    async def run():
        tools = await mgr.init_config_async({
            'mcpServers': {
                'fs': {'command': 'npx', 'args': ['-y', 'pkg']},
            },
        })
        return tools

    # First tool has schema without required — filled in; second missing fields raises.
    # Override list_tools to only return one good tool + resources.
    from mcp import ClientSession

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(
                name='echo',
                description='d',
                inputSchema={'type': 'object', 'properties': {'q': {'type': 'string'}}},
            ),
        ])

    async def list_resources(self):
        return SimpleNamespace(resources=['r1'])

    monkeypatch.setattr(ClientSession, 'list_tools', list_tools)
    monkeypatch.setattr(ClientSession, 'list_resources', list_resources)

    tools = asyncio.run(run())
    names = [t.name for t in tools]
    assert any(n.endswith('-echo') for n in names)
    assert any('list_resources' in n for n in names)
    assert any('read_resource' in n for n in names)
    assert mgr.clients


def test_init_config_async_missing_schema_fields(monkeypatch):
    _install_fake_mcp(monkeypatch)
    mgr = _bare()
    mgr.clients = {}
    from mcp import ClientSession

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(
                name='bad',
                description='d',
                inputSchema={'type': 'object'},  # missing properties/required
            ),
        ])

    async def list_resources(self):
        return SimpleNamespace(resources=[])

    monkeypatch.setattr(ClientSession, 'list_tools', list_tools)
    monkeypatch.setattr(ClientSession, 'list_resources', list_resources)

    with pytest.raises(ValueError, match='Missing required fields'):
        asyncio.run(mgr.init_config_async({
            'mcpServers': {'s': {'command': 'c', 'args': []}},
        }))


def test_connection_server_sse_and_streamable(monkeypatch):
    _install_fake_mcp(monkeypatch)
    from mcp import ClientSession

    async def list_tools(self):
        return SimpleNamespace(tools=[])

    async def list_resources(self):
        raise RuntimeError('no resources')

    monkeypatch.setattr(ClientSession, 'list_tools', list_tools)
    monkeypatch.setattr(ClientSession, 'list_resources', list_resources)

    client = mm.MCPClient()

    async def go():
        await client.connection_server('sse', {
            'url': 'http://localhost/sse',
            'headers': {'Accept': 'text/event-stream'},
        })
        assert client.session is not None
        await client.connection_server('http', {
            'type': 'streamable-http',
            'url': 'http://localhost/mcp',
            'headers': {},
            'sse_read_timeout': 10,
        })

    asyncio.run(go())


def test_connection_server_failure(monkeypatch):
    _install_fake_mcp(monkeypatch)
    from mcp import ClientSession

    async def boom(self):
        raise RuntimeError('connect fail')

    monkeypatch.setattr(ClientSession, 'initialize', boom)
    client = mm.MCPClient()
    with pytest.raises(RuntimeError, match='connect fail'):
        asyncio.run(client.connection_server('x', {'command': 'c', 'args': []}))


def test_execute_function_list_resources_empty_and_error(monkeypatch):
    _install_fake_mcp(monkeypatch)
    client = mm.MCPClient()
    session = MagicMock()

    async def ping():
        return None

    async def list_resources():
        return SimpleNamespace(resources=[])

    session.send_ping = ping
    session.list_resources = list_resources
    client.session = session
    assert asyncio.run(client.execute_function('list_resources', {})) == 'No resources found'

    async def boom():
        raise RuntimeError('lr')

    session.list_resources = boom
    assert 'Error' in asyncio.run(client.execute_function('list_resources', {}))


def test_execute_function_read_empty_and_tool_empty(monkeypatch):
    _install_fake_mcp(monkeypatch)
    client = mm.MCPClient()
    session = MagicMock()

    async def ping():
        return None

    async def read_resource(uri):
        return SimpleNamespace(contents=[SimpleNamespace(blob=b'x')])  # not TextResourceContents

    async def call_tool(name, args):
        return SimpleNamespace(content=[SimpleNamespace(type='image', text='')])

    session.send_ping = ping
    session.read_resource = read_resource
    session.call_tool = call_tool
    client.session = session
    assert asyncio.run(client.execute_function('read_resource', {'uri': 'u'})) == 'Failed to read resource'
    assert asyncio.run(client.execute_function('echo', {})) == 'execute error'


def test_execute_function_reconnect_success(monkeypatch):
    _install_fake_mcp(monkeypatch)
    mgr = _bare()
    mgr.clients = {}
    mgr.loop = MagicMock()
    mm.MCPManager._instance = mgr

    client = mm.MCPClient()
    client.client_id = 'cid'
    client._last_mcp_server_name = 's'
    client._last_mcp_server = {'command': 'c', 'args': []}
    session = MagicMock()

    async def boom_ping():
        raise RuntimeError('dead')

    session.send_ping = boom_ping
    client.session = session

    new = mm.MCPClient()
    new.client_id = 'cid'
    new_session = MagicMock()

    async def ok_ping():
        return None

    async def call_tool(name, args):
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='reconnected')])

    new_session.send_ping = ok_ping
    new_session.call_tool = call_tool
    new.session = new_session

    async def fake_reconnect():
        return new

    with patch.object(client, 'reconnect', side_effect=fake_reconnect):
        out = asyncio.run(client.execute_function('echo', {'a': 1}))
    assert out == 'reconnected'


def test_execute_function_reconnect_exception(monkeypatch):
    _install_fake_mcp(monkeypatch)
    mgr = _bare()
    mgr.clients = {}
    mgr.loop = MagicMock()
    mm.MCPManager._instance = mgr

    client = mm.MCPClient()
    client.client_id = 'cid'
    session = MagicMock()

    async def boom_ping():
        raise RuntimeError('dead')

    session.send_ping = boom_ping
    client.session = session

    async def bad_reconnect():
        raise RuntimeError('reconnect failed')

    with patch.object(client, 'reconnect', side_effect=bad_reconnect):
        out = asyncio.run(client.execute_function('echo', {}))
    assert 'reconnect' in out.lower() or 'Session' in out


def test_tool_class_call_and_acall_paths():
    mgr = _bare()
    mgr.clients = {}
    mgr.loop = MagicMock()
    mm.MCPManager._instance = mgr

    client = MagicMock()

    async def exec_fn(name, args):
        return f'ok:{name}:{args}'

    client.execute_function = exec_fn
    mgr.clients['cid'] = client

    tool = mgr.create_tool_class(
        register_name='s-echo',
        register_client_id='cid',
        tool_name='echo',
        tool_desc='d',
        tool_parameters={'type': 'object', 'properties': {}, 'required': []},
    )

    future = MagicMock()
    future.result.return_value = 'sync-ok'
    with patch('asyncio.run_coroutine_threadsafe', return_value=future):
        assert tool.call('{"x": 1}') == 'sync-ok'

    future.result.side_effect = RuntimeError('call-fail')
    with patch('asyncio.run_coroutine_threadsafe', return_value=future):
        with pytest.raises(RuntimeError, match='call-fail'):
            tool.call('{}')

    async def run_acall():
        fut = asyncio.get_running_loop().create_future()
        fut.set_result('async-ok')
        with patch('asyncio.run_coroutine_threadsafe', return_value=fut):
            # wrap_future needs a concurrent.futures.Future — use a real one
            import concurrent.futures
            cf = concurrent.futures.Future()
            cf.set_result('async-ok')
            with patch('asyncio.run_coroutine_threadsafe', return_value=cf):
                return await tool.acall({'x': 2})

    assert asyncio.run(run_acall()) == 'async-ok'

    async def run_acall_cancel():
        import concurrent.futures
        cf = concurrent.futures.Future()

        async def waiter():
            with patch('asyncio.run_coroutine_threadsafe', return_value=cf):
                task = asyncio.create_task(tool.acall('{}'))
                await asyncio.sleep(0)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        await waiter()

    asyncio.run(run_acall_cancel())

    async def run_acall_error():
        import concurrent.futures
        cf = concurrent.futures.Future()
        cf.set_exception(RuntimeError('ae'))
        with patch('asyncio.run_coroutine_threadsafe', return_value=cf):
            with pytest.raises(RuntimeError, match='ae'):
                await tool.acall('{}')

    asyncio.run(run_acall_error())


def test_shutdown_terminates_processes():
    mgr = _bare()
    mgr.clients = {}
    mgr.loop = MagicMock()
    mgr.loop_thread = MagicMock()
    mgr.processes = [MagicMock(), MagicMock()]
    mgr.processes[1].terminate.side_effect = ProcessLookupError()

    client = MagicMock()

    async def cleanup():
        return None

    client.cleanup = cleanup
    mgr.clients['c1'] = client

    future = MagicMock()
    with patch('asyncio.run_coroutine_threadsafe', return_value=future), \
            patch('time.sleep'), \
            patch('asyncio.all_tasks', return_value=[MagicMock()]):
        mgr.shutdown()

    mgr.processes[0].terminate.assert_called()
    mgr.loop.call_soon_threadsafe.assert_called()
    mgr.loop_thread.join.assert_called()


def test_cleanup_mcp_with_instance():
    mgr = _bare()
    mgr.clients = {}
    mgr.loop = MagicMock()
    mgr.loop_thread = MagicMock()
    mgr.processes = []
    mm.MCPManager._instance = mgr
    with patch.object(mgr, 'shutdown') as sh:
        mm._cleanup_mcp()
        sh.assert_called_once()


def test_reconnect_builds_new_client(monkeypatch):
    _install_fake_mcp(monkeypatch)
    from mcp import ClientSession

    async def list_tools(self):
        return SimpleNamespace(tools=[])

    async def list_resources(self):
        return SimpleNamespace(resources=[])

    monkeypatch.setattr(ClientSession, 'list_tools', list_tools)
    monkeypatch.setattr(ClientSession, 'list_resources', list_resources)

    client = mm.MCPClient()
    client.client_id = 'cid'
    client._last_mcp_server_name = 's'
    client._last_mcp_server = {'command': 'c', 'args': []}

    new = asyncio.run(client.reconnect())
    assert new.client_id == 'cid'
    assert new.session is not None


def test_resource_template_list_failure_still_adds_read_tool(monkeypatch):
    _install_fake_mcp(monkeypatch)
    mgr = _bare()
    mgr.clients = {}
    from mcp import ClientSession

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(
                name='t',
                description='d',
                inputSchema={'type': 'object', 'properties': {}, 'required': []},
            ),
        ])

    async def list_resources(self):
        return SimpleNamespace(resources=['r'])

    async def list_resource_templates(self):
        raise RuntimeError('no templates')

    monkeypatch.setattr(ClientSession, 'list_tools', list_tools)
    monkeypatch.setattr(ClientSession, 'list_resources', list_resources)
    monkeypatch.setattr(ClientSession, 'list_resource_templates', list_resource_templates)

    tools = asyncio.run(mgr.init_config_async({
        'mcpServers': {'s': {'command': 'c', 'args': []}},
    }))
    assert any('read_resource' in t.name for t in tools)
