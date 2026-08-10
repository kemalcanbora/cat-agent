"""Tests for AST mutation generation and the ToolSmith mutation gate."""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock

from cat_agent.llm.schema import ASSISTANT, Message
from cat_agent.synthesis.executors.base import ExecResult
from cat_agent.synthesis.mutation import generate_mutants
from cat_agent.synthesis.smith import Status, ToolSmith
from cat_agent.synthesis.spec import Example, ToolSpec

# Shared with adversarial suite — keep values in sync.
WEAK_EXAMPLES = [
    Example(inputs={'iban': 'TR330006100519786457841326'}, expected=True),
    Example(inputs={'iban': 'TR250001000010000000000001'}, expected=True),
    Example(inputs={'iban': 'XX00'}, expected=False),
    Example(inputs={'iban': 'TR33000610051978645784132'}, expected=False),
    Example(inputs={'iban': ''}, expected=False),
]

STRONG_EXAMPLES = WEAK_EXAMPLES + [
    Example(inputs={'iban': 'TR330006100519786457841327'}, expected=False),
    Example(inputs={'iban': 'TR340006100519786457841326'}, expected=False),
    Example(inputs={'iban': 'DE89370400440532013000'}, expected=True),
    Example(inputs={'iban': 'NO9386011117947'}, expected=True),
    Example(inputs={'iban': 'TR33!006100519786457841326'}, expected=False),
]

REAL_IMPL = '''\
def validate_iban(iban):
    if not iban:
        return False
    if len(iban) < 15:
        return False
    s = iban[4:] + iban[:4]
    if not s.isalnum():
        return False
    n = "".join(str(int(c, 36)) for c in s)
    return int(n) % 97 == 1
'''

UNDERGENERALISED = '''\
def validate_iban(iban):
    return len(iban) == 26 and iban[:2] == "TR"
'''


class InProcessExecutor:
    """SandboxExecutor Protocol stand-in — no Wasm, no network."""

    name = 'inprocess'
    supports_dependencies = False

    def run(
        self,
        code: str,
        inputs: Dict[str, Any],
        deps: Optional[List[str]] = None,
        timeout_s: Optional[float] = None,
        *,
        function_name: str = 'main',
    ) -> ExecResult:
        try:
            ns: Dict[str, Any] = {}
            exec(code, ns)  # noqa: S102
            fn = ns.get(function_name)
            if fn is None:
                return ExecResult(
                    ok=False, stdout='', stderr='', error=f'missing {function_name}',
                )
            return ExecResult(
                ok=True, stdout='', stderr='', error=None, returned=fn(**inputs),
            )
        except Exception as exc:  # noqa: BLE001
            return ExecResult(ok=False, stdout='', stderr=str(exc), error=str(exc))


def _mutation_score(
    code: str,
    examples: Sequence[Example],
    *,
    limit: int = 12,
    function_name: str = 'validate_iban',
) -> tuple[float, int, int, List[str]]:
    """Return ``(score, total, killed, survivors)`` using in-process exec."""
    mutants = generate_mutants(code, limit=limit)
    executor = InProcessExecutor()
    killed = 0
    survivors: List[str] = []
    for mutant in mutants:
        dead = False
        for ex in examples:
            result = executor.run(
                mutant.code, ex.inputs, function_name=function_name,
            )
            if not result.ok or result.returned != ex.expected:
                dead = True
                break
        if dead:
            killed += 1
        else:
            survivors.append(mutant.description)
    total = len(mutants)
    score = killed / total if total else 1.0
    return score, total, killed, survivors


def _iban_spec(examples: list) -> ToolSpec:
    return ToolSpec(
        name='validate_iban',
        description='Validate an IBAN (ISO 13616 mod-97).',
        parameters={'iban': 'string'},
        returns='boolean',
        examples=list(examples),
        holdout_ratio=0.2,
    )


def _mock_llm(code: str) -> MagicMock:
    llm = MagicMock()
    llm.model = 'mock-model'
    fenced = f'```python\n{code}\n```'

    def chat(**kwargs):
        return iter([[Message(role=ASSISTANT, content=fenced)]])

    llm.chat = MagicMock(side_effect=chat)
    return llm


def test_generate_mutants_is_deterministic():
    a = generate_mutants(REAL_IMPL, limit=12)
    b = generate_mutants(REAL_IMPL, limit=12)
    assert [m.description for m in a] == [m.description for m in b]
    assert [m.code for m in a] == [m.code for m in b]


def test_invalid_source_returns_empty_list():
    assert generate_mutants('def broken(:\n', limit=12) == []


def test_every_mutant_is_parseable():
    mutants = generate_mutants(REAL_IMPL, limit=12)
    assert mutants
    for mutant in mutants:
        ast.parse(mutant.code)


def test_real_impl_strong_examples_above_threshold():
    """Most important: hardening must not reject a correct general solution."""
    score, total, killed, survivors = _mutation_score(REAL_IMPL, STRONG_EXAMPLES)
    assert total > 0
    assert score >= 0.8, (
        f'real mod-97 rejected under STRONG examples: '
        f'score={killed}/{total}={score:.2f} survivors={survivors}'
    )


def test_real_impl_weak_examples_below_threshold_via_smith(tmp_path):
    """Same real code with WEAK examples → WEAK_SPEC (under-specified)."""
    score, total, killed, survivors = _mutation_score(REAL_IMPL, WEAK_EXAMPLES)
    assert total > 0
    assert score < 0.8, (
        f'expected weak-spec signal under WEAK examples; '
        f'got score={killed}/{total}={score:.2f} survivors={survivors}'
    )

    smith = ToolSmith(
        llm=_mock_llm(REAL_IMPL),
        executor=InProcessExecutor(),
        max_attempts=1,
        output_dir=str(tmp_path),
        mutation_enabled=True,
        mutation_limit=12,
        mutation_threshold=0.8,
    )
    result = smith.synthesize(_iban_spec(WEAK_EXAMPLES))
    assert result.ok is False
    assert result.status == Status.WEAK_SPEC
    assert result.surviving_mutants


def test_a4_mutation_scores_under_both_example_sets():
    """A4 (len+prefix): report scores — mutation alone does not catch it.

    Every mutant of the under-generalised body still fails some WEAK example
    (e.g. ``==`` → ``!=``), so the kill ratio is 1.0 under both sets. The
    defence against A4 remains strong examples at the work stage, not mutation.
    """
    weak_score, weak_total, weak_killed, weak_surv = _mutation_score(
        UNDERGENERALISED, WEAK_EXAMPLES,
    )
    strong_score, strong_total, strong_killed, strong_surv = _mutation_score(
        UNDERGENERALISED, STRONG_EXAMPLES,
    )
    assert weak_total > 0 and strong_total > 0
    # Documented outcome for the exercise report:
    assert (weak_killed, weak_total, weak_score) == (6, 6, 1.0)
    assert (strong_killed, strong_total, strong_score) == (6, 6, 1.0)
    assert weak_surv == [] and strong_surv == []


def test_real_impl_strong_accepted_by_smith(tmp_path):
    smith = ToolSmith(
        llm=_mock_llm(REAL_IMPL),
        executor=InProcessExecutor(),
        max_attempts=1,
        output_dir=str(tmp_path),
        mutation_enabled=True,
        mutation_limit=12,
        mutation_threshold=0.8,
    )
    result = smith.synthesize(_iban_spec(STRONG_EXAMPLES))
    assert result.ok, result.error
    assert result.status == Status.SUCCESS
