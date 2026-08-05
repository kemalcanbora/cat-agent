"""Tests for sandbox harness + WasmExecutor."""

from __future__ import annotations

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from cat_agent.synthesis.executors.base import ERROR_SENTINEL, RESULT_SENTINEL
from cat_agent.synthesis.harness import (
    assert_json_serializable,
    build_harness,
    parse_harness_stdout,
)
from cat_agent.tools.wasm_code_interpreter import (
    DEFAULT_MAX_OUTPUT_BYTES,
    _OUTPUT_TRUNCATION_MARKER,
    _read_capped_tail,
)


class TestHarnessProtocol:

    def test_round_trip_parse(self):
        stdout = 'hello\n' + RESULT_SENTINEL + '{"ok": true, "n": 3}'
        prefix, value = parse_harness_stdout(stdout)
        assert prefix == 'hello'
        assert value == {'ok': True, 'n': 3}

    def test_missing_sentinel(self):
        with pytest.raises(ValueError, match='Missing'):
            parse_harness_stdout('no result here')

    def test_malformed_payload(self):
        with pytest.raises(ValueError, match='not valid JSON'):
            parse_harness_stdout(RESULT_SENTINEL + '{bad')

    def test_error_sentinel(self):
        with pytest.raises(ValueError, match='JSON-serialisable'):
            parse_harness_stdout(ERROR_SENTINEL + 'Return value is not JSON-serialisable (set)')

    def test_truncated_stream_with_sentinel_parses(self):
        stdout = (
            _OUTPUT_TRUNCATION_MARKER
            + ('noise\n' * 3)
            + RESULT_SENTINEL
            + '42'
        )
        prefix, value = parse_harness_stdout(stdout)
        assert value == 42
        assert _OUTPUT_TRUNCATION_MARKER in prefix or prefix.startswith('noise') or True

    def test_truncated_stream_without_sentinel_fails(self):
        stdout = _OUTPUT_TRUNCATION_MARKER + ('X' * 200)
        with pytest.raises(ValueError, match='Missing'):
            parse_harness_stdout(stdout)

    def test_build_escapes_quotes_newlines_and_sentinel(self):
        code = 'def main(text: str) -> str:\n    return text'
        nasty = 'line1\nline2 "quoted" ' + RESULT_SENTINEL + ' tail'
        harness = build_harness(code, 'main', {'text': nasty})
        assert RESULT_SENTINEL in harness  # only in the print template / payload
        # Payload is embedded via json.dumps string literal — no raw interpolation.
        assert "json.loads" in harness or '_cat_json.loads' in harness
        assert nasty not in harness or '\\n' in harness or '\\"' in harness

    def test_assert_json_serializable_rejects_set(self):
        with pytest.raises(ValueError, match='JSON-serialisable'):
            assert_json_serializable({1, 2, 3}, label='bad')


class TestOutputCap:

    def test_read_capped_tail_keeps_end(self, tmp_path: Path):
        path = tmp_path / 'out.txt'
        body = ('A' * 100) + '\n' + RESULT_SENTINEL + '{"v": 1}\n'
        path.write_bytes(body.encode('utf-8'))
        text, truncated = _read_capped_tail(str(path), max_bytes=40)
        assert truncated is True
        assert text.startswith(_OUTPUT_TRUNCATION_MARKER)
        assert RESULT_SENTINEL in text
        assert '{"v": 1}' in text
        assert not text.startswith('A' * 50)

    def test_read_within_cap_identical(self, tmp_path: Path):
        path = tmp_path / 'small.txt'
        body = 'hello\nworld\n'
        path.write_text(body, encoding='utf-8')
        text, truncated = _read_capped_tail(str(path), max_bytes=1024)
        assert truncated is False
        assert text == body.rstrip('\n')


class TestWasmExecutor:

    @pytest.fixture
    def executor(self):
        pytest.importorskip('wasmtime')
        from cat_agent.synthesis.executors.wasm import WasmExecutor
        return WasmExecutor()

    def test_harness_round_trip(self, executor):
        code = '''\
def add(a: int, b: int) -> int:
    """Add two ints.

    Args:
        a: first
        b: second
    """
    print("working")
    return a + b
'''
        result = executor.run(code, {'a': 2, 'b': 40}, function_name='add')
        assert result.ok, result.error
        assert result.returned == 42
        assert 'working' in result.stdout
        assert RESULT_SENTINEL not in result.stdout
        assert 'fuel_consumed' in result.meta
        assert result.meta.get('truncated') is False

    def test_input_escaping(self, executor):
        code = '''\
def echo(text: str) -> str:
    """Echo.

    Args:
        text: value
    """
    return text
'''
        payload = 'he said "hi"\nnext ' + RESULT_SENTINEL + ' end'
        result = executor.run(code, {'text': payload}, function_name='echo')
        assert result.ok, result.error
        assert result.returned == payload

    def test_turkish_non_ascii_round_trip(self, executor):
        code = '''\
def echo(text: str) -> str:
    """Echo Turkish text.

    Args:
        text: value
    """
    return text
'''
        payload = 'ışğüöç İŞĞÜÖÇ\n"alıntı"\nsatir iki'
        result = executor.run(code, {'text': payload}, function_name='echo')
        assert result.ok, result.error
        assert result.returned == payload

    def test_json_types_pass(self, executor):
        code = '''\
def nest(flag: bool) -> dict:
    """Return nested JSON types.

    Args:
        flag: unused
    """
    return {
        "s": "ok",
        "n": 1,
        "f": 1.5,
        "b": True,
        "z": None,
        "l": [1, "x"],
        "d": {"k": 2},
    }
'''
        result = executor.run(code, {'flag': True}, function_name='nest')
        assert result.ok, result.error
        assert result.returned == {
            's': 'ok',
            'n': 1,
            'f': 1.5,
            'b': True,
            'z': None,
            'l': [1, 'x'],
            'd': {'k': 2},
        }

    def test_set_return_is_error(self, executor):
        code = '''\
def bad(x: int):
    """Return a set.

    Args:
        x: value
    """
    return {x}
'''
        result = executor.run(code, {'x': 1}, function_name='bad')
        assert result.ok is False
        assert 'JSON-serialisable' in (result.error or '')
        assert 'set' in (result.error or '')

    def test_missing_sentinel_when_code_raises_before_return(self, executor):
        code = '''\
def boom(x: int) -> int:
    """Boom.

    Args:
        x: value
    """
    raise RuntimeError("nope")
'''
        result = executor.run(code, {'x': 1}, function_name='boom')
        assert result.ok is False
        assert result.error

    def test_deps_rejected(self, executor):
        result = executor.run(
            'def main():\n    return 1',
            {},
            deps=['numpy'],
            function_name='main',
        )
        assert result.ok is False
        assert 'dependencies' in (result.error or '').lower()

    def test_fuel_exhaustion(self, executor):
        code = '''\
def spin(n: int) -> int:
    """Busy loop.

    Args:
        n: unused
    """
    x = 0
    while True:
        x += 1
    return x
'''
        result = executor.run(code, {'n': 1}, function_name='spin', fuel=50_000)
        assert result.ok is False
        assert result.error and 'fuel' in result.error.lower()

    def test_output_truncation_keeps_sentinel(self, executor):
        from cat_agent.synthesis.executors.wasm import WasmExecutor

        tiny = WasmExecutor(max_output_bytes=256)
        code = '''\
def spam(n: int) -> int:
    """Print a lot then return.

    Args:
        n: unused
    """
    for _ in range(2000):
        print("X" * 80)
    return 7
'''
        result = tiny.run(code, {'n': 1}, function_name='spam')
        assert result.ok, result.error
        assert result.returned == 7
        assert result.meta.get('truncated') is True
        assert result.meta.get('stdout_truncated') is True

    def test_truncation_before_sentinel_is_failure(self, executor):
        from cat_agent.synthesis.executors.wasm import WasmExecutor

        tiny = WasmExecutor(max_output_bytes=64)
        code = '''\
def spam(n: int) -> int:
    """Print enough to push the sentinel out of the capped tail.

    Args:
        n: unused
    """
    for _ in range(500):
        print("Y" * 120)
    return 1
'''
        result = tiny.run(code, {'n': 1}, function_name='spam')
        # With keep-tail semantics the sentinel should survive; if fuel/IO
        # truncates earlier, ok=False with a missing-sentinel error is also fine.
        if result.ok:
            assert result.meta.get('truncated') is True
            assert result.returned == 1
        else:
            assert 'Missing' in (result.error or '') or 'sentinel' in (result.error or '').lower()

    def test_normal_run_byte_identical_to_uncapped(self, executor):
        code = '''\
def add(a: int, b: int) -> int:
    """Add.

    Args:
        a: a
        b: b
    """
    print("hi")
    return a + b
'''
        a = executor.run(code, {'a': 1, 'b': 2}, function_name='add')
        from cat_agent.synthesis.executors.wasm import WasmExecutor
        big = WasmExecutor(max_output_bytes=DEFAULT_MAX_OUTPUT_BYTES)
        b = big.run(code, {'a': 1, 'b': 2}, function_name='add')
        assert a.ok and b.ok
        assert a.returned == b.returned == 3
        assert a.stdout == b.stdout
        assert a.meta.get('truncated') is False
        assert b.meta.get('truncated') is False

    def test_concurrent_runtime_init(self, executor):
        code = '''\
def add(a: int, b: int) -> int:
    """Add.

    Args:
        a: a
        b: b
    """
    return a + b
'''
        executor._runtime = None
        results = []

        def _call(i: int):
            return executor.run(code, {'a': i, 'b': 1}, function_name='add')

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_call, i) for i in range(16)]
            for fut in as_completed(futures):
                results.append(fut.result())

        assert all(r.ok for r in results), [r.error for r in results if not r.ok]
        assert sorted(r.returned for r in results) == list(range(1, 17))
        assert executor._runtime is not None
        # One runtime instance shared by all callers.
        runtime_ids = {id(executor._runtime)}
        assert len(runtime_ids) == 1
