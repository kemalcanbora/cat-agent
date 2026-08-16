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

"""Coverage tests for cat_agent.tools.code_interpreter (mocked docker/subprocess)."""

from __future__ import annotations

import base64
import queue
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.tools import code_interpreter as ci


def _make_ci(tmp_path, **cfg):
    with patch.object(ci, '_check_docker_availability'):
        with patch.object(ci, '_check_host_deps'):
            tool = ci.CodeInterpreter({'work_dir': str(tmp_path), **cfg})
    return tool


def test_escape_ansi():
    raw = '\x1b[31mred\x1b[0m plain'
    assert 'red' in ci._escape_ansi(raw)
    assert '\x1b' not in ci._escape_ansi(raw)


def test_check_docker_availability_ok(monkeypatch):
    def run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout='Docker', stderr='')

    monkeypatch.setattr(ci.subprocess, 'run', run)
    ci._check_docker_availability()


def test_check_docker_availability_not_installed(monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError('docker')

    monkeypatch.setattr(ci.subprocess, 'run', boom)
    with pytest.raises(RuntimeError, match='not installed'):
        ci._check_docker_availability()


def test_check_docker_availability_daemon_down(monkeypatch):
    calls = {'n': 0}

    def run(cmd, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            return SimpleNamespace(returncode=0, stdout='v', stderr='')
        return SimpleNamespace(returncode=1, stdout='', stderr='dead')

    monkeypatch.setattr(ci.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='daemon'):
        ci._check_docker_availability()


def test_check_docker_availability_timeout(monkeypatch):
    def boom(*_a, **_k):
        raise ci.subprocess.TimeoutExpired(cmd='docker', timeout=5)

    monkeypatch.setattr(ci.subprocess, 'run', boom)
    with pytest.raises(RuntimeError, match='timed out'):
        ci._check_docker_availability()


def test_check_host_deps_import_error(monkeypatch):
    # Prefer setitem over patch.dict(sys.modules): patch.dict restores by
    # clear()+update(snapshot) and wipes modules imported inside the block.
    monkeypatch.setitem(__import__('sys').modules, 'jupyter_client', None)
    with pytest.raises(ImportError, match='code_interpreter'):
        ci._check_host_deps()


def test_args_format_custom_and_default(tmp_path):
    tool = _make_ci(tmp_path, args_format='X')
    assert tool.args_format == 'X'
    tool2 = _make_ci(tmp_path)
    assert 'backtick' in tool2.args_format.lower() or '`' in tool2.args_format


def test_build_docker_image_exists(tmp_path, monkeypatch):
    tool = _make_ci(tmp_path)

    def run(cmd, **kwargs):
        if cmd[:2] == ['docker', 'images']:
            return SimpleNamespace(returncode=0, stdout='abc123\n', stderr='')
        raise AssertionError(cmd)

    monkeypatch.setattr(ci.subprocess, 'run', run)
    tool._build_docker_image()


def test_build_docker_image_builds(tmp_path, monkeypatch):
    tool = _make_ci(tmp_path)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ['docker', 'images']:
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        return SimpleNamespace(returncode=0, stdout='ok', stderr='')

    monkeypatch.setattr(ci.subprocess, 'run', run)
    tool._build_docker_image()
    assert any(c[:2] == ['docker', 'build'] for c in calls)


def test_build_docker_image_fails(tmp_path, monkeypatch):
    tool = _make_ci(tmp_path)

    def run(cmd, **kwargs):
        if cmd[:2] == ['docker', 'images']:
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        return SimpleNamespace(returncode=1, stdout='', stderr='build fail')

    monkeypatch.setattr(ci.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='Failed to build'):
        tool._build_docker_image()


def test_get_free_ports(tmp_path):
    tool = _make_ci(tmp_path)
    ports = tool._get_free_ports(3)
    assert len(ports) == 3
    assert len(set(ports)) == 3


def test_serve_image_local_and_url(tmp_path, monkeypatch):
    tool = _make_ci(tmp_path)
    png = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
    )
    b64 = base64.b64encode(png).decode()

    class FakeImage:
        def save(self, path, fmt):
            from pathlib import Path
            Path(path).write_bytes(b'png')

    pil = ModuleType('PIL')
    pil_image = ModuleType('PIL.Image')
    pil_image.open = lambda *_a, **_k: FakeImage()
    pil.Image = pil_image
    monkeypatch.setitem(__import__('sys').modules, 'PIL', pil)
    monkeypatch.setitem(__import__('sys').modules, 'PIL.Image', pil_image)

    monkeypatch.delenv('M6_CODE_INTERPRETER_STATIC_URL', raising=False)
    local = tool._serve_image(b64)
    assert local.endswith('.png')
    monkeypatch.setenv('M6_CODE_INTERPRETER_STATIC_URL', 'http://static')
    url = tool._serve_image(b64)
    assert url.startswith('http://static/')


def test_execute_code_stream_and_idle(tmp_path):
    tool = _make_ci(tmp_path)
    msgs = [
        {
            'msg_type': 'stream',
            'content': {'name': 'stdout', 'text': 'hello'},
        },
        {
            'msg_type': 'status',
            'content': {'execution_state': 'idle'},
        },
    ]
    kc = MagicMock()
    kc.get_iopub_msg.side_effect = msgs
    out = tool._execute_code(kc, 'print(1)')
    assert 'hello' in out
    assert 'stdout' in out


def test_execute_code_result_and_error(tmp_path, monkeypatch):
    tool = _make_ci(tmp_path)
    png_b64 = base64.b64encode(b'fakepng').decode()
    msgs = [
        {
            'msg_type': 'execute_result',
            'content': {
                'data': {
                    'text/plain': '42',
                    'image/png': png_b64,
                }
            },
        },
        {
            'msg_type': 'error',
            'content': {'traceback': ['\x1b[31mTimeout\x1b[0m M6_CODE_INTERPRETER_TIMEOUT']},
        },
        {
            'msg_type': 'status',
            'content': {'execution_state': 'idle'},
        },
    ]
    kc = MagicMock()
    kc.get_iopub_msg.side_effect = msgs
    monkeypatch.setattr(tool, '_serve_image', lambda *_: '/tmp/fig.png')
    out = tool._execute_code(kc, 'x')
    assert '42' in out
    assert 'Timeout: Code execution exceeded' in out
    assert 'fig-001' in out


def test_execute_code_display_data_and_queue_empty(tmp_path, monkeypatch):
    tool = _make_ci(tmp_path)
    msgs = [
        {
            'msg_type': 'display_data',
            'content': {'data': {'text/plain': 'plot'}},
        },
        queue.Empty(),
    ]
    kc = MagicMock()
    kc.get_iopub_msg.side_effect = msgs
    out = tool._execute_code(kc, 'x')
    assert 'plot' in out
    assert 'Timeout' in out


def test_execute_code_unexpected_exception(tmp_path, monkeypatch):
    tool = _make_ci(tmp_path)
    kc = MagicMock()
    kc.get_iopub_msg.side_effect = RuntimeError('boom')
    monkeypatch.setattr(ci, 'print_traceback', lambda: None)
    out = tool._execute_code(kc, 'x')
    assert 'unexpected error' in out


def test_call_empty_and_json_and_cached_kernel(tmp_path):
    from cat_agent.tools.base import BaseToolWithFileAccess

    tool = _make_ci(tmp_path)
    with patch.object(BaseToolWithFileAccess, 'call', return_value=None):
        assert tool.call('{"code": "  "}') == ''

    kc = MagicMock()
    kernel_id = f'{tool.instance_id}_{__import__("os").getpid()}'
    ci._KERNEL_CLIENTS[kernel_id] = kc
    try:
        with patch.object(BaseToolWithFileAccess, 'call', return_value=None):
            with patch.object(tool, '_execute_code', return_value='ok\n') as ex:
                result = tool.call('{"code": "print(1)"}', timeout=1)
        assert result == 'ok\n'
        assert ex.call_count >= 2  # code + timer cancel
    finally:
        ci._KERNEL_CLIENTS.pop(kernel_id, None)


def test_call_extract_code_fallback_and_finished_message(tmp_path):
    from cat_agent.tools.base import BaseToolWithFileAccess

    tool = _make_ci(tmp_path)
    kc = MagicMock()
    kernel_id = f'{tool.instance_id}_{__import__("os").getpid()}'
    ci._KERNEL_CLIENTS[kernel_id] = kc
    try:
        with patch.object(BaseToolWithFileAccess, 'call', return_value=None):
            with patch.object(tool, '_execute_code', return_value='   '):
                out = tool.call('```python\nprint(1)\n```', timeout=None)
        assert out == 'Finished execution.'
    finally:
        ci._KERNEL_CLIENTS.pop(kernel_id, None)


def test_call_sns_theme_fix(tmp_path):
    from cat_agent.tools.base import BaseToolWithFileAccess

    tool = _make_ci(tmp_path)
    kc = MagicMock()
    kernel_id = f'{tool.instance_id}_{__import__("os").getpid()}'
    ci._KERNEL_CLIENTS[kernel_id] = kc
    captured = []
    try:
        def capture(_kc, code):
            captured.append(code)
            return 'done'

        with patch.object(BaseToolWithFileAccess, 'call', return_value=None):
            with patch.object(tool, '_execute_code', side_effect=capture):
                tool.call('{"code": "sns.set_theme()\\nx=1"}', timeout=0)
        body = captured[0]
        assert 'plt.rcParams["font.family"]' in body
    finally:
        ci._KERNEL_CLIENTS.pop(kernel_id, None)


def test_kill_kernels_and_containers(monkeypatch):
    client = MagicMock()
    ci._KERNEL_CLIENTS['k1'] = client
    ci._DOCKER_CONTAINERS['k1'] = 'cid'

    def run(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(ci.subprocess, 'run', run)
    ci._kill_kernels_and_containers()
    client.shutdown.assert_called_once()
    assert ci._KERNEL_CLIENTS == {}
    assert ci._DOCKER_CONTAINERS == {}


def test_del_cleans_resources(tmp_path, monkeypatch):
    tool = _make_ci(tmp_path)
    k = f'{tool.instance_id}_{__import__("os").getpid()}'
    client = MagicMock()
    ci._KERNEL_CLIENTS[k] = client
    ci._DOCKER_CONTAINERS[k] = 'cid123'
    monkeypatch.setattr(
        ci.subprocess,
        'run',
        lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''),
    )
    tool.__del__()
    assert k not in ci._KERNEL_CLIENTS
    assert k not in ci._DOCKER_CONTAINERS


def test_any_thread_event_loop_policy():
    policy = ci.AnyThreadEventLoopPolicy()
    loop = policy.get_event_loop()
    assert loop is not None
    # second call returns same or new without raising
    assert policy.get_event_loop() is not None


def test_start_kernel_mocked(tmp_path, monkeypatch):
    tool = _make_ci(tmp_path)
    monkeypatch.setattr(tool, '_build_docker_image', lambda: None)
    monkeypatch.setattr(tool, '_get_free_ports', lambda n=5: [10001, 10002, 10003, 10004, 10005])
    monkeypatch.setattr(ci.time, 'sleep', lambda *_: None)

    # Avoid missing font file copy failure by creating a stub source
    monkeypatch.setattr(ci, 'ALIB_FONT_FILE', str(tmp_path / 'font.ttf'))
    (tmp_path / 'font.ttf').write_bytes(b'font')

    def run(cmd, **kwargs):
        if cmd[:2] == ['docker', 'run']:
            return SimpleNamespace(returncode=0, stdout='container123\n', stderr='')
        if cmd[:3] == ['docker', 'ps', '-q']:
            return SimpleNamespace(returncode=0, stdout='container123\n', stderr='')
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(ci.subprocess, 'run', run)

    fake_kc = MagicMock()
    monkeypatch.setitem(
        __import__('sys').modules,
        'jupyter_client',
        SimpleNamespace(BlockingKernelClient=MagicMock(return_value=fake_kc)),
    )
    with patch('asyncio.set_event_loop_policy'):
        kc, cid = tool._start_kernel('kid')
    assert kc is fake_kc
    assert cid == 'container123'
    fake_kc.load_connection_file.assert_called_once()
    fake_kc.start_channels.assert_called_once()
    fake_kc.wait_for_ready.assert_called()
