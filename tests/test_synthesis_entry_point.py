"""Tests for LLM stream text collection and entry-point extraction."""

from __future__ import annotations

from unittest.mock import MagicMock

from cat_agent.llm.schema import ASSISTANT, Message
from cat_agent.synthesis.entry_point import (
    ensure_entry_point,
    extract_impl_code,
    simplify_name_error,
)
from cat_agent.synthesis.executors.base import ExecResult
from cat_agent.synthesis.llm_text import collect_chat_text, strip_thinking_markup
from cat_agent.synthesis.smith import Status, ToolSmith
from cat_agent.synthesis.spec import Example, ToolSpec


class TestCollectChatText:

    def test_uses_last_full_content(self):
        stream = [
            [Message(role=ASSISTANT, content='def')],
            [Message(role=ASSISTANT, content='def vat_split():\n    pass')],
        ]
        assert 'def vat_split' in collect_chat_text(stream)

    def test_falls_back_to_reasoning_content(self):
        stream = [
            [
                Message(
                    role=ASSISTANT,
                    content='',
                    reasoning_content='```python\ndef vat_split(g, r):\n    return {}\n```',
                ),
            ],
        ]
        text = collect_chat_text(stream)
        assert 'def vat_split' in text

    def test_strips_think_blocks(self):
        raw = '<think>plan</think>\n```python\ndef f():\n    return 1\n```'
        assert 'plan' not in strip_thinking_markup(raw)
        assert 'def f' in strip_thinking_markup(raw)


class TestEntryPoint:

    def test_extract_prefers_named_fence(self):
        raw = '''\
Here is a sketch:
```python
def other(x):
    return x
```
Final:
```python
def vat_split(gross: float, rate: float) -> dict:
    return {"net": 0.0, "tax": 0.0}
```
'''
        code = extract_impl_code(raw, 'vat_split')
        assert 'def vat_split' in code
        assert 'def other' not in code

    def test_extract_from_reasoning_wrapped(self):
        raw = (
            '<think>I will write the function</think>\n'
            '```python\ndef vat_split(gross, rate):\n    return {"net": 0, "tax": 0}\n```'
        )
        code = extract_impl_code(raw, 'vat_split')
        assert 'def vat_split' in code

    def test_rename_single_mismatch(self):
        code = '''\
def split_vat(gross: float, rate: float) -> dict:
    """Split.

    Args:
        gross: Gross.
        rate: Rate.
    """
    return {"net": gross / (1 + rate), "tax": 0.0}
'''
        fixed, err = ensure_entry_point(code, 'vat_split')
        assert err is None
        assert 'def vat_split' in fixed
        assert 'def split_vat' not in fixed

    def test_missing_function_errors(self):
        code = 'x = 1\n'
        fixed, err = ensure_entry_point(code, 'vat_split')
        assert 'x = 1' in fixed
        assert err is not None
        assert 'vat_split' in err

    def test_already_correct(self):
        code = 'def vat_split(gross, rate):\n    return {}\n'
        fixed, err = ensure_entry_point(code, 'vat_split')
        assert err is None
        assert 'def vat_split' in fixed

    def test_simplify_name_error(self):
        msg = simplify_name_error(
            function_name='vat_split',
            error="Exited with i32 exit status 1",
            stderr="NameError: name 'vat_split' is not defined",
        )
        assert msg is not None
        assert 'def vat_split' in msg


def _mock_llm(responses):
    llm = MagicMock()
    llm.model = 'mock'
    queue = list(responses)

    def chat(**kwargs):
        text = queue.pop(0) if queue else '```python\ndef x():\n    return 1\n```'
        return iter([[Message(role=ASSISTANT, content=text)]])

    llm.chat = MagicMock(side_effect=chat)
    return llm


def test_smith_renames_wrong_function_name(tmp_path):
    """Live-run failure mode: model defines split_vat, harness calls vat_split."""
    spec = ToolSpec(
        name='vat_split',
        description='Split VAT',
        parameters={
            'gross': {'type': 'number', 'description': 'Gross'},
            'rate': {'type': 'number', 'description': 'Rate'},
        },
        returns='object',
        examples=[
            Example(inputs={'gross': 120.0, 'rate': 0.2}, expected={'net': 100.0, 'tax': 20.0}),
            Example(inputs={'gross': 100.0, 'rate': 0.0}, expected={'net': 100.0, 'tax': 0.0}),
            Example(inputs={'gross': 10.0, 'rate': 0.1}, expected={'net': 9.09, 'tax': 0.91}),
            Example(inputs={'gross': 1.0, 'rate': 0.2}, expected={'net': 0.83, 'tax': 0.17}),
            Example(inputs={'gross': 250.0, 'rate': 0.18}, expected={'net': 211.86, 'tax': 38.14}),
        ],
        holdout_ratio=0.3,
    )
    wrong = '''\
```python
def split_vat(gross: float, rate: float) -> dict:
    """Split VAT.

    Args:
        gross: Gross amount.
        rate: VAT rate.
    """
    net = round(gross / (1 + rate), 2)
    tax = round(gross - net, 2)
    return {"net": net, "tax": tax}
```
'''
    calls = []

    class Exec:
        name = 'fake'
        supports_dependencies = False

        def run(self, code, inputs, deps=None, timeout_s=None, *, function_name='main'):
            calls.append(code)
            assert f'def {function_name}' in code
            for ex in spec.examples:
                if ex.inputs == inputs:
                    return ExecResult(
                        ok=True, stdout='', stderr='', error=None, returned=ex.expected,
                    )
            return ExecResult(ok=False, stdout='', stderr='', error='miss', returned=None)

    smith = ToolSmith(
        llm=_mock_llm([wrong]),
        executor=Exec(),
        max_attempts=2,
        output_dir=str(tmp_path),
        mutation_enabled=False,
    )
    result = smith.synthesize(spec)
    assert result.ok, result.error
    assert result.status == Status.SUCCESS
    assert calls
    assert 'def vat_split' in calls[0]
    assert 'def split_vat' not in calls[0]


def test_smith_uses_reasoning_content_when_content_empty(tmp_path):
    spec = ToolSpec(
        name='add_one_r',
        description='Add one',
        parameters={'x': {'type': 'integer', 'description': 'v'}},
        returns='integer',
        examples=[
            Example(inputs={'x': 1}, expected=2),
            Example(inputs={'x': 2}, expected=3),
            Example(inputs={'x': 3}, expected=4),
            Example(inputs={'x': 10}, expected=11),
            Example(inputs={'x': 0}, expected=1),
        ],
        holdout_ratio=0.4,
    )
    code = (
        f'```python\ndef {spec.function_name}(x: int) -> int:\n'
        f'    """Add.\n\n    Args:\n        x: v.\n    """\n'
        f'    return x + 1\n```'
    )

    llm = MagicMock()
    llm.model = 'mock'

    def chat(**kwargs):
        return iter([[
            Message(role=ASSISTANT, content='', reasoning_content=code),
        ]])

    llm.chat = MagicMock(side_effect=chat)

    class Exec:
        name = 'fake'
        supports_dependencies = False

        def run(self, code, inputs, deps=None, timeout_s=None, *, function_name='main'):
            assert f'def {function_name}' in code
            return ExecResult(ok=True, stdout='', stderr='', error=None, returned=inputs['x'] + 1)

    smith = ToolSmith(
        llm=llm,
        executor=Exec(),
        max_attempts=2,
        output_dir=str(tmp_path),
        mutation_enabled=False,
    )
    result = smith.synthesize(spec)
    assert result.ok, result.error
