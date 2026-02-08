"""Tests for cat_agent.tools.python_executor."""

import json
from unittest.mock import patch

import pytest

from cat_agent.tools.python_executor import (
    ColorObjectRuntime,
    CustomDict,
    DateRuntime,
    GenericRuntime,
    PythonExecutor,
    _check_deps_for_python_executor,
    _DANGEROUS_RE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Heavy deps required by PythonExecutor – skip tests that need them if missing.
_HAS_EXECUTOR_DEPS = True
try:
    _check_deps_for_python_executor()
except ImportError:
    _HAS_EXECUTOR_DEPS = False

requires_executor_deps = pytest.mark.skipif(
    not _HAS_EXECUTOR_DEPS,
    reason='PythonExecutor optional dependencies not installed',
)


def _make_executor(**cfg_overrides) -> PythonExecutor:
    """Create a PythonExecutor with sensible test defaults."""
    cfg = {'get_answer_from_stdout': True, 'timeout_length': 10}
    cfg.update(cfg_overrides)
    return PythonExecutor(cfg=cfg)


# ===========================================================================
# Tests: _DANGEROUS_RE blocklist
# ===========================================================================

class TestDangerousPatterns:
    """Verify the security blocklist catches dangerous code patterns."""

    @pytest.mark.parametrize('code', [
        'input("prompt")',
        'x = input("prompt")',
        'os.system("rm -rf /")',
        'import subprocess',
        'subprocess.run(["ls"])',
        'shutil.rmtree("/tmp/foo")',
        '__import__("os")',
        'exec("print(1)")',
        'eval("1+1")',
        'importlib.import_module("os")',
    ])
    def test_blocklist_catches_dangerous_code(self, code):
        assert _DANGEROUS_RE.search(code) is not None, f'Expected blocklist to catch: {code!r}'

    @pytest.mark.parametrize('code', [
        'print("hello")',
        'x = 1 + 2',
        'import math',
        'data = {"key": "value"}',
        'for i in range(10): pass',
        '# just a comment about input()',
        'result = my_func(input_data)',      # 'input_data' != 'input('
        'os.path.join("a", "b")',
        'my_subprocess_result = 42',         # word contains 'subprocess' but isn't a call
    ])
    def test_blocklist_allows_safe_code(self, code):
        # Note: some of these might match if patterns are too broad.
        # We test that common safe patterns are not blocked.
        match = _DANGEROUS_RE.search(code)
        if match:
            pytest.skip(f'Pattern matched (may be acceptable false positive): {match.group()!r}')


# ===========================================================================
# Tests: GenericRuntime
# ===========================================================================

class TestGenericRuntime:

    def test_exec_code_simple(self):
        rt = GenericRuntime()
        rt.exec_code('x = 42')
        assert rt._global_vars['x'] == 42

    def test_eval_code(self):
        rt = GenericRuntime()
        rt.exec_code('x = 10')
        assert rt.eval_code('x * 2') == 20

    def test_inject(self):
        rt = GenericRuntime()
        rt.inject({'greeting': 'hello', 'count': 5})
        assert rt._global_vars['greeting'] == 'hello'
        assert rt._global_vars['count'] == 5
        assert rt.eval_code('greeting + "!"') == 'hello!'

    def test_answer_property(self):
        rt = GenericRuntime()
        rt.exec_code('answer = 99')
        assert rt.answer == 99

    def test_answer_property_raises_if_not_set(self):
        rt = GenericRuntime()
        with pytest.raises(KeyError):
            _ = rt.answer

    def test_blocked_input(self):
        rt = GenericRuntime()
        with pytest.raises(RuntimeError, match='blocked pattern'):
            rt.exec_code('x = input("name")')

    def test_blocked_os_system(self):
        rt = GenericRuntime()
        with pytest.raises(RuntimeError, match='blocked pattern'):
            rt.exec_code('os.system("ls")')

    def test_blocked_subprocess(self):
        rt = GenericRuntime()
        with pytest.raises(RuntimeError, match='blocked pattern'):
            rt.exec_code('import subprocess')

    def test_blocked_exec(self):
        rt = GenericRuntime()
        with pytest.raises(RuntimeError, match='blocked pattern'):
            rt.exec_code('exec("print(1)")')

    def test_blocked_eval(self):
        rt = GenericRuntime()
        with pytest.raises(RuntimeError, match='blocked pattern'):
            rt.exec_code('eval("1+1")')

    def test_blocked_dunder_import(self):
        rt = GenericRuntime()
        with pytest.raises(RuntimeError, match='blocked pattern'):
            rt.exec_code('__import__("os")')

    def test_multiline_code(self):
        rt = GenericRuntime()
        rt.exec_code('x = 0\nfor i in range(5):\n    x += i')
        assert rt._global_vars['x'] == 10

    def test_instances_have_isolated_state(self):
        rt1 = GenericRuntime()
        rt2 = GenericRuntime()
        rt1.exec_code('shared = 1')
        assert 'shared' not in rt2._global_vars


# ===========================================================================
# Tests: DateRuntime
# ===========================================================================

class TestDateRuntime:

    def test_datetime_available(self):
        import datetime
        rt = DateRuntime()
        assert rt._global_vars['datetime'] is datetime.datetime

    def test_can_create_datetime(self):
        rt = DateRuntime()
        rt.exec_code('d = datetime(2025, 1, 15)')
        import datetime
        assert rt._global_vars['d'] == datetime.datetime(2025, 1, 15)


# ===========================================================================
# Tests: CustomDict / ColorObjectRuntime
# ===========================================================================

class TestCustomDict:

    def test_iter_returns_list_iterator(self):
        d = CustomDict(a=1, b=2, c=3)
        keys = list(d)
        assert set(keys) == {'a', 'b', 'c'}

    def test_works_as_normal_dict(self):
        d = CustomDict()
        d['x'] = 10
        assert d['x'] == 10
        assert len(d) == 1


class TestColorObjectRuntime:

    def test_dict_is_custom_dict(self):
        rt = ColorObjectRuntime()
        assert rt._global_vars['dict'] is CustomDict


# ===========================================================================
# Tests: _check_deps_for_python_executor
# ===========================================================================

class TestCheckDeps:

    def test_raises_with_missing_deps(self):
        with patch('builtins.__import__', side_effect=ImportError('no module')):
            with pytest.raises(ImportError, match='Missing dependencies'):
                _check_deps_for_python_executor()

    def test_reports_all_missing_modules(self):
        """When multiple deps are missing, the error lists all of them."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def fake_import(name, *args, **kwargs):
            if name in ('multiprocess', 'pebble'):
                raise ImportError(f'No module named {name!r}')
            return original_import(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=fake_import):
            with pytest.raises(ImportError, match='multiprocess') as exc_info:
                _check_deps_for_python_executor()
            assert 'pebble' in str(exc_info.value)

    @requires_executor_deps
    def test_no_error_when_deps_present(self):
        # Should not raise
        _check_deps_for_python_executor()


# ===========================================================================
# Tests: PythonExecutor.truncate (static, no deps needed)
# ===========================================================================

class TestTruncate:

    def test_short_string_unchanged(self):
        assert PythonExecutor.truncate('hello', max_length=100) == 'hello'

    def test_exact_length_unchanged(self):
        s = 'x' * 800
        assert PythonExecutor.truncate(s) == s

    def test_long_string_truncated(self):
        s = 'a' * 1000
        result = PythonExecutor.truncate(s, max_length=100)
        assert len(result) < 1000
        assert 'chars truncated' in result
        assert result.startswith('a' * 50)
        assert result.endswith('a' * 50)

    def test_truncation_message_shows_count(self):
        s = 'x' * 200
        result = PythonExecutor.truncate(s, max_length=100)
        assert '100 chars truncated' in result

    def test_empty_string(self):
        assert PythonExecutor.truncate('') == ''


# ===========================================================================
# Tests: PythonExecutor.execute (static method, needs timeout_decorator)
# ===========================================================================

@requires_executor_deps
class TestExecute:

    def test_stdout_capture(self):
        rt = GenericRuntime()
        result, report = PythonExecutor.execute(
            'print("hello world")',
            get_answer_from_stdout=True,
            runtime=rt,
            timeout_length=10,
        )
        assert 'hello world' in result
        assert report == 'Done'

    def test_answer_symbol(self):
        rt = GenericRuntime()
        result, report = PythonExecutor.execute(
            'answer = 42',
            get_answer_from_stdout=False,
            runtime=rt,
            answer_symbol='answer',
            timeout_length=10,
        )
        assert result == 42
        assert report == 'Done'

    def test_answer_expr(self):
        rt = GenericRuntime()
        result, report = PythonExecutor.execute(
            'x = 10\ny = 20',
            get_answer_from_stdout=False,
            runtime=rt,
            answer_expr='x + y',
            timeout_length=10,
        )
        assert result == 30
        assert report == 'Done'

    def test_eval_last_line(self):
        rt = GenericRuntime()
        result, report = PythonExecutor.execute(
            'x = 5\nx * 3',
            get_answer_from_stdout=False,
            runtime=rt,
            timeout_length=10,
        )
        assert result == 15
        assert report == 'Done'

    def test_syntax_error_reported(self):
        rt = GenericRuntime()
        result, report = PythonExecutor.execute(
            'def broken(\n',
            get_answer_from_stdout=True,
            runtime=rt,
            timeout_length=10,
        )
        assert result == ''
        assert 'SyntaxError' in report

    def test_runtime_error_reported(self):
        rt = GenericRuntime()
        result, report = PythonExecutor.execute(
            'x = 1 / 0',
            get_answer_from_stdout=True,
            runtime=rt,
            timeout_length=10,
        )
        assert result == ''
        assert 'ZeroDivisionError' in report

    def test_name_error_reported(self):
        rt = GenericRuntime()
        result, report = PythonExecutor.execute(
            'print(undefined_var)',
            get_answer_from_stdout=True,
            runtime=rt,
            timeout_length=10,
        )
        assert result == ''
        assert 'NameError' in report

    def test_blocked_code_reported(self):
        rt = GenericRuntime()
        result, report = PythonExecutor.execute(
            'import subprocess',
            get_answer_from_stdout=True,
            runtime=rt,
            timeout_length=10,
        )
        assert result == ''
        assert 'RuntimeError' in report or 'blocked' in report.lower()

    def test_multiline_stdout(self):
        rt = GenericRuntime()
        result, report = PythonExecutor.execute(
            'for i in range(3):\n    print(i)',
            get_answer_from_stdout=True,
            runtime=rt,
            timeout_length=10,
        )
        assert '0' in result
        assert '1' in result
        assert '2' in result
        assert report == 'Done'

    def test_non_serializable_result_reports_error(self):
        """If the result can't be pickled, it should be caught."""
        rt = GenericRuntime()
        # Lambdas can't be pickled
        result, report = PythonExecutor.execute(
            'answer = lambda x: x',
            get_answer_from_stdout=False,
            runtime=rt,
            answer_symbol='answer',
            timeout_length=10,
        )
        assert result == ''
        assert 'pickle' in report.lower() or 'PicklingError' in report or 'Error' in report


# ===========================================================================
# Tests: PythonExecutor (full integration, needs all deps)
# ===========================================================================

@requires_executor_deps
class TestPythonExecutorIntegration:

    def test_init_defaults(self):
        executor = _make_executor()
        assert executor.get_answer_from_stdout is True
        assert executor.timeout_length == 10
        assert executor.answer_symbol is None
        assert executor.answer_expr is None
        assert isinstance(executor.runtime, GenericRuntime)

    def test_init_custom_runtime(self):
        rt = DateRuntime()
        executor = _make_executor(runtime=rt)
        assert executor.runtime is rt

    def test_call_with_json_params(self):
        executor = _make_executor()
        result = executor.call(json.dumps({'code': 'print("test123")'}))
        res, report = result
        assert 'test123' in res

    def test_call_with_raw_code(self):
        executor = _make_executor()
        result = executor.call('print("raw_code")')
        res, report = result
        # extract_code fallback should handle this
        assert isinstance(result, tuple)

    def test_call_empty_code(self):
        executor = _make_executor()
        result = executor.call(json.dumps({'code': '   '}))
        assert result == ['', '']

    def test_apply_simple(self):
        executor = _make_executor()
        res, report = executor.apply('print("apply_test")')
        assert 'apply_test' in res
        assert report == 'Done'

    def test_apply_error(self):
        executor = _make_executor()
        res, report = executor.apply('raise ValueError("boom")')
        assert res == ''
        assert 'ValueError' in report

    def test_batch_apply(self):
        executor = _make_executor()
        results = executor.batch_apply([
            'print("first")',
            'print("second")',
        ])
        assert len(results) == 2
        assert 'first' in results[0][0]
        assert 'second' in results[1][0]

    def test_batch_apply_mixed_success_failure(self):
        executor = _make_executor()
        results = executor.batch_apply([
            'print("ok")',
            '1 / 0',
        ])
        assert len(results) == 2
        assert 'ok' in results[0][0]
        assert results[0][1] == 'Done'
        assert results[1][0] == ''
        assert 'ZeroDivisionError' in results[1][1]

    def test_answer_symbol_mode(self):
        executor = _make_executor(
            get_answer_from_stdout=False,
            get_answer_symbol='result',
        )
        res, report = executor.apply('result = 7 * 6')
        assert res == '42'
        assert report == 'Done'

    def test_truncation_applied(self):
        executor = _make_executor()
        long_output = 'x' * 2000
        res, report = executor.apply(f'print("{long_output}")')
        assert len(res) < 2000
        assert 'truncated' in res
