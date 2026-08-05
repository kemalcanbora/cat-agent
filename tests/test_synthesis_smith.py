"""Tests for ToolSmith synthesis loop (mocked LLM + fake executor)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from cat_agent.llm.schema import ASSISTANT, Message
from cat_agent.synthesis.executors.base import ExecResult
from cat_agent.synthesis.smith import HOLDOUT_FAILED_MESSAGE, Status, ToolSmith
from cat_agent.synthesis.spec import Example, ToolSpec


class FakeExecutor:
    name = 'fake'
    supports_dependencies = False

    def __init__(self, behaviour='pass'):
        self.behaviour = behaviour
        self.calls: List[Dict[str, Any]] = []

    def run(
        self,
        code: str,
        inputs: Dict[str, Any],
        deps: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
        *,
        function_name: str = 'main',
    ) -> ExecResult:
        self.calls.append({'code': code, 'inputs': inputs, 'function_name': function_name})
        if self.behaviour == 'pass':
            return ExecResult(ok=True, stdout='', stderr='', error=None, returned=inputs['x'] + 1)
        return ExecResult(ok=False, stdout='', stderr='boom', error='boom')


def _spec(name: str = 'synth_add_one_smith') -> ToolSpec:
    return ToolSpec(
        name=name,
        description='Return x + 1',
        parameters={'x': 'integer'},
        returns='integer',
        examples=[
            Example(inputs={'x': 1}, expected=2),
            Example(inputs={'x': 2}, expected=3),
            Example(inputs={'x': 10}, expected=11),
            Example(inputs={'x': 0}, expected=1),
            Example(inputs={'x': -3}, expected=-2),
        ],
        holdout_ratio=0.4,
    )


def _good_code(name: str) -> str:
    return f'''\
```python
def {name}(x: int) -> int:
    """Return x + 1.

    Args:
        x: Integer value.
    """
    return x + 1
```
'''


def _overfit_literal_code(name: str, expected_values: list) -> str:
    lit = repr(expected_values[0])
    return f'''\
```python
def {name}(x: int) -> int:
    """Bad.

    Args:
        x: Integer value.
    """
    if x == 1:
        return {lit}
    return x + 1
```
'''


def _mock_llm(responses: List[str]) -> MagicMock:
    llm = MagicMock()
    llm.model = 'mock-model'
    queue = list(responses)

    def chat(**kwargs):
        text = queue.pop(0) if queue else _good_code('fallback')
        return iter([[Message(role=ASSISTANT, content=text)]])

    llm.chat = MagicMock(side_effect=chat)
    return llm


class TestToolSmith:

    def test_first_attempt_success(self, tmp_path):
        spec = _spec('smith_first_ok')
        smith = ToolSmith(
            llm=_mock_llm([_good_code(spec.function_name)]),
            executor=FakeExecutor('pass'),
            max_attempts=3,
            output_dir=str(tmp_path),
        )
        result = smith.synthesize(spec)
        assert result.ok
        assert result.status == Status.SUCCESS
        assert result.artifact_dir
        assert (tmp_path / 'generated_tools' / spec.function_name / 'impl.py').is_file()
        assert (tmp_path / 'generated_tools' / spec.function_name / 'spec.json').is_file()
        assert len(result.attempts) == 1

    def test_success_after_retries(self, tmp_path):
        spec = _spec('smith_retry_ok')
        bad = f'''\
```python
def {spec.function_name}(x: int) -> int:
    """Wrong.

    Args:
        x: Integer value.
    """
    return x + 99
```
'''

        class CodeAware(FakeExecutor):
            def run(self, code, inputs, deps=None, timeout_s=None, *, function_name='main'):
                self.calls.append({'code': code, 'inputs': inputs})
                if 'x + 99' in code:
                    return ExecResult(
                        ok=True, stdout='', stderr='', error=None, returned=inputs['x'] + 99)
                return ExecResult(
                    ok=True, stdout='', stderr='', error=None, returned=inputs['x'] + 1)

        smith = ToolSmith(
            llm=_mock_llm([bad, _good_code(spec.function_name)]),
            executor=CodeAware(),
            max_attempts=5,
            output_dir=str(tmp_path),
        )
        result = smith.synthesize(spec)
        assert result.ok
        assert result.status == Status.SUCCESS
        assert len(result.attempts) == 2
        assert result.attempts[0].stage == 'work'

    def test_exhaustion(self, tmp_path):
        spec = _spec('smith_exhaust')
        bad = f'''\
```python
def {spec.function_name}(x: int) -> int:
    """Wrong.

    Args:
        x: Integer value.
    """
    return 0
```
'''

        class AlwaysWrong(FakeExecutor):
            def run(self, code, inputs, deps=None, timeout_s=None, *, function_name='main'):
                return ExecResult(ok=True, stdout='', stderr='', error=None, returned=0)

        smith = ToolSmith(
            llm=_mock_llm([bad, bad, bad]),
            executor=AlwaysWrong(),
            max_attempts=3,
            output_dir=str(tmp_path),
        )
        result = smith.synthesize(spec)
        assert result.ok is False
        assert result.status == Status.EXHAUSTED
        assert len(result.attempts) == 3
        assert result.error

    def test_holdout_failure_terminates_without_retry(self, tmp_path):
        spec = _spec('smith_holdout')
        work, holdout = spec.split_examples()

        class HoldoutAware(FakeExecutor):
            def run(self, code, inputs, deps=None, timeout_s=None, *, function_name='main'):
                for ex in holdout:
                    if ex.inputs == inputs:
                        return ExecResult(
                            ok=True, stdout='', stderr='', error=None, returned=999)
                return ExecResult(
                    ok=True, stdout='', stderr='', error=None, returned=inputs['x'] + 1)

        llm = _mock_llm([_good_code(spec.function_name)] * 5)
        smith = ToolSmith(
            llm=llm,
            executor=HoldoutAware(),
            max_attempts=5,
            output_dir=str(tmp_path),
        )
        result = smith.synthesize(spec)
        assert result.ok is False
        assert result.status == Status.HOLDOUT_FAILED
        assert result.error == HOLDOUT_FAILED_MESSAGE
        assert result.holdout_failures
        assert llm.chat.call_count == 1
        work_inputs = {tuple(sorted(ex.inputs.items())) for ex in work}
        for call in llm.chat.call_args_list:
            messages = call.kwargs.get('messages') or call.args[0]
            blob = '\n'.join(str(getattr(m, 'content', m)) for m in messages)
            assert '[holdout#' not in blob
            assert 'Previous attempt feedback:' not in blob
            for failure in result.holdout_failures:
                key = tuple(sorted(failure['inputs'].items()))
                if key not in work_inputs:
                    assert repr(failure['inputs']) not in blob
                assert f"actual={failure.get('returned')!r}" not in blob

    def test_overfit_hardcoded_expected(self, tmp_path):
        # Use a non-trivial expected that is guaranteed to be in the work set.
        spec = ToolSpec(
            name='smith_overfit_lit',
            description='Return x + 1',
            parameters={'x': 'integer'},
            returns='integer',
            examples=[
                Example(inputs={'x': 1}, expected=1001),
                Example(inputs={'x': 2}, expected=1002),
                Example(inputs={'x': 3}, expected=1003),
                Example(inputs={'x': 4}, expected=1004),
                Example(inputs={'x': 5}, expected=1005),
            ],
            holdout_ratio=0.2,
        )
        code = _overfit_literal_code(spec.function_name, [1001])
        # Executor returns expected for work so only overfit should block first try.

        class MatchExpected(FakeExecutor):
            def run(self, code, inputs, deps=None, timeout_s=None, *, function_name='main'):
                for ex in spec.examples:
                    if ex.inputs == inputs:
                        return ExecResult(
                            ok=True, stdout='', stderr='', error=None, returned=ex.expected)
                return ExecResult(ok=True, stdout='', stderr='', error=None, returned=None)

        smith = ToolSmith(
            llm=_mock_llm([code, _good_code(spec.function_name)]),
            executor=MatchExpected(),
            max_attempts=3,
            output_dir=str(tmp_path),
        )
        result = smith.synthesize(spec)
        assert result.ok
        assert result.attempts[0].stage == 'overfit'
        assert 'Overfit guard' in (result.attempts[0].error or '')

    def test_non_serialisable_return_retries(self, tmp_path):
        pytest.importorskip('wasmtime')
        from cat_agent.synthesis.executors.wasm import WasmExecutor

        spec = ToolSpec(
            name='smith_set_return',
            description='Return a set (wrong) then fix',
            parameters={'x': 'integer'},
            returns='array',
            examples=[
                Example(inputs={'x': 1}, expected=[1]),
                Example(inputs={'x': 2}, expected=[2]),
                Example(inputs={'x': 3}, expected=[3]),
            ],
        )
        bad = f'''\
```python
def {spec.function_name}(x: int):
    """Bad.

    Args:
        x: Integer value.
    """
    return {{x}}
```
'''
        good = f'''\
```python
def {spec.function_name}(x: int):
    """Good.

    Args:
        x: Integer value.
    """
    return [x]
```
'''
        smith = ToolSmith(
            llm=_mock_llm([bad, good]),
            executor=WasmExecutor(),
            max_attempts=3,
            output_dir=str(tmp_path),
        )
        result = smith.synthesize(spec)
        assert result.ok
        assert result.attempts[0].stage == 'work'
        assert 'JSON-serialisable' in (result.attempts[0].error or '')
        assert result.status == Status.SUCCESS
