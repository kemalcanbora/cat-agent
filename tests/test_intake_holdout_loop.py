"""Tests for holdout_failed → re-interview loop."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

from cat_agent.llm.schema import ASSISTANT, Message
from cat_agent.synthesis.intake.interview import Question
from cat_agent.synthesis.intake.pipeline import synthesize_from_draft
from cat_agent.synthesis.smith import Status, SynthesisResult
from cat_agent.synthesis.spec import Example, ParameterSpec, ToolSpec


def _mock_llm(responses: List[str]) -> MagicMock:
    llm = MagicMock()
    llm.model = 'mock'
    queue = list(responses)

    def chat(**kwargs):
        text = queue.pop(0) if queue else (responses[-1] if responses else '{}')
        return iter([[Message(role=ASSISTANT, content=text)]])

    llm.chat = MagicMock(side_effect=chat)
    return llm


def _spec(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description='Return x + 1',
        parameters={'x': ParameterSpec(type='integer', description='v')},
        returns='integer',
        examples=[
            Example(inputs={'x': 1}, expected=2),
            Example(inputs={'x': 2}, expected=3),
            Example(inputs={'x': 3}, expected=4),
            Example(inputs={'x': 10}, expected=11),
            Example(inputs={'x': 0}, expected=1),
        ],
    )


def _holdout_result(spec: ToolSpec, *, ok: bool) -> SynthesisResult:
    if ok:
        return SynthesisResult(
            ok=True,
            status=Status.SUCCESS,
            spec=spec,
            code='def f(x): return x+1',
            artifact_dir='/tmp/x',
            attempts=[],
            registered_name=spec.registered_name,
        )
    return SynthesisResult(
        ok=False,
        status=Status.HOLDOUT_FAILED,
        spec=spec,
        code='def f(x): return x+1',
        artifact_dir=None,
        attempts=[],
        registered_name=spec.registered_name,
        error='holdout failed',
        holdout_failures=[{
            'inputs': {'x': 10},
            'expected': 11,
            'returned': 999,
            'ok': False,
        }],
    )


def test_holdout_failed_reopens_interview(tmp_path):
    name = 'holdout_loop_tool'
    draft_path = tmp_path / 'draft.md'
    draft_path.write_text(
        f"""\
# {name}
Return x + 1.
| x | result |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 10 | 11 |
| 0 | 1 |
""",
        encoding='utf-8',
    )
    compile_payload = f'''\
{{
  "name": "{name}",
  "description": "Return x + 1",
  "parameters": {{"x": {{"type": "integer", "description": "Integer value"}}}},
  "returns": "integer"
}}
'''
    llm = _mock_llm([
        'YES',
        'You add one. Right?',
        compile_payload,
    ])
    spec = _spec(name)
    calls = {'n': 0}

    def synthesize(self, s, *, provenance=None):
        calls['n'] += 1
        if calls['n'] == 1:
            return _holdout_result(s, ok=False)
        return _holdout_result(s, ok=True)

    kinds: List[str] = []

    def ask(q: Question) -> str:
        kinds.append(q.kind)
        if q.kind == 'confirm':
            return 'yes'
        assert '10' in q.text or '999' in q.text or '11' in q.text
        return '11'

    with patch('cat_agent.synthesis.intake.pipeline.ToolSmith.synthesize', synthesize):
        with patch(
            'cat_agent.synthesis.intake.pipeline.get_executor',
            return_value=MagicMock(name='wasm'),
        ):
            result = synthesize_from_draft(
                draft_path,
                llm=llm,
                ask=ask,
                max_holdout_rounds=2,
                output_dir=str(tmp_path),
            )

    assert 'holdout' in kinds
    assert result.holdout_rounds == 1
    assert result.ok
    assert calls['n'] == 2
    assert any(
        ex.inputs == {'x': 10} for ex in (result.spec.examples if result.spec else [])
    )


def test_holdout_round_cap(tmp_path):
    name = 'holdout_cap_tool'
    draft_path = tmp_path / 'draft.md'
    draft_path.write_text(
        f"""\
# {name}
Return x + 1.
| x | result |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 10 | 11 |
| 0 | 1 |
""",
        encoding='utf-8',
    )
    compile_payload = f'''\
{{
  "name": "{name}",
  "description": "Return x + 1",
  "parameters": {{"x": {{"type": "integer", "description": "v"}}}},
  "returns": "integer"
}}
'''
    llm = _mock_llm([
        'YES',
        'You add one. Right?',
        compile_payload,
    ])

    def synthesize(self, s, *, provenance=None):
        return _holdout_result(s, ok=False)

    kinds: List[str] = []

    def ask(q: Question) -> str:
        kinds.append(q.kind)
        return 'yes' if q.kind == 'confirm' else '11'

    with patch('cat_agent.synthesis.intake.pipeline.ToolSmith.synthesize', synthesize):
        with patch(
            'cat_agent.synthesis.intake.pipeline.get_executor',
            return_value=MagicMock(name='wasm'),
        ):
            result = synthesize_from_draft(
                draft_path,
                llm=llm,
                ask=ask,
                max_holdout_rounds=2,
                output_dir=str(tmp_path),
            )

    assert result.holdout_rounds == 2
    assert result.ok is False
    assert result.synthesis is not None
    assert result.synthesis.status == Status.HOLDOUT_FAILED
    assert kinds.count('holdout') == 2
