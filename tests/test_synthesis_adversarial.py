"""Adversarial tests for cat_agent.synthesis overfit guards.

Case: IBAN validation (ISO 13616 mod-97).

Why this case:
  * A short, genuinely general algorithm exists.
  * It is trivially faked with a lookup table.
  * Under-generalisation (length + prefix check) looks like 100% success.

Drop into tests/ and run:  pytest tests/test_synthesis_adversarial.py -v

Tests marked xfail document currently-unguarded attacks. When a guard is
added, flip the xfail and it becomes a regression test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from cat_agent.llm.schema import ASSISTANT, Message
from cat_agent.synthesis.executors.base import ExecResult
from cat_agent.synthesis.overfit import check_overfit
from cat_agent.synthesis.smith import Status, ToolSmith
from cat_agent.synthesis.spec import Example, ToolSpec

# ---------------------------------------------------------------------------
# The examples a typical user would supply. Deliberately weak negatives:
# every invalid case is "obviously" invalid (bad length or bad prefix).
# This is the realistic failure mode, not a strawman.
# ---------------------------------------------------------------------------
WEAK_EXAMPLES = [
    Example(inputs={'iban': 'TR330006100519786457841326'}, expected=True),
    Example(inputs={'iban': 'TR250001000010000000000001'}, expected=True),
    Example(inputs={'iban': 'XX00'}, expected=False),
    Example(inputs={'iban': 'TR33000610051978645784132'}, expected=False),
    Example(inputs={'iban': ''}, expected=False),
]

# The examples a good spec would supply: at least one negative that differs
# from a positive ONLY in the checksum. This is what forces real logic.
STRONG_EXAMPLES = WEAK_EXAMPLES + [
    # same as example 1 but last digit changed -> checksum must catch it
    Example(inputs={'iban': 'TR330006100519786457841327'}, expected=False),
    Example(inputs={'iban': 'TR340006100519786457841326'}, expected=False),
    Example(inputs={'iban': 'DE89370400440532013000'}, expected=True),
    # 15-char valid IBAN — kills len-bound mutants (15→16)
    Example(inputs={'iban': 'NO9386011117947'}, expected=True),
    # non-alnum — kills isalnum early-return False→True equivalents
    Example(inputs={'iban': 'TR33!006100519786457841326'}, expected=False),
]

# A1 — dict lookup (no ``==``); blocked by check_literal_lookup.
LOOKUP_TABLE = '''\
def validate_iban(iban):
    T = {"TR330006100519786457841326": True,
         "TR250001000010000000000001": True,
         "XX00": False}
    return T.get(iban, False)
'''

# A2 — equality chain; blocked by check_input_equality_chain.
EQUALITY_CHAIN = '''\
def validate_iban(iban):
    if iban == "TR330006100519786457841326": return True
    if iban == "TR250001000010000000000001": return True
    if iban == "XX00": return False
    return False
'''

# A3 — table + heuristic fallback; blocked by check_literal_lookup.
TABLE_PLUS_FALLBACK = '''\
def validate_iban(iban):
    T = {"TR330006100519786457841326": True,
         "TR250001000010000000000001": True}
    if iban in T:
        return T[iban]
    return len(iban) == 26 and iban[:2].isalpha()
'''

# A4 — under-generalised length+prefix; passes every weak example.
UNDERGENERALISED = '''\
def validate_iban(iban):
    return len(iban) == 26 and iban[:2] == "TR"
'''

# A5 — base64-hidden literal; still a hole at the string/AST guard layer.
OBFUSCATED_TABLE = '''\
import base64
def validate_iban(iban):
    blob = base64.b64decode("VFIzMzAwMDYxMDA1MTk3ODY0NTc4NDEzMjY=").decode()
    return iban == blob or iban == "TR250001000010000000000001"
'''

# A6 — real mod-97; must never be flagged.
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


def _run(code: str, iban: str):
    ns: dict = {}
    exec(code, ns)  # noqa: S102 — intentional for adversarial fixtures
    return ns['validate_iban'](iban)


class InProcessExecutor:
    """Minimal SandboxExecutor that execs code in-process (tests only)."""

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
            returned = fn(**inputs)
            return ExecResult(ok=True, stdout='', stderr='', error=None, returned=returned)
        except Exception as exc:  # noqa: BLE001 — surface any mutant crash
            return ExecResult(ok=False, stdout='', stderr=str(exc), error=str(exc))


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


# --- guards that work -----------------------------------------------------

def test_real_implementation_is_not_flagged():
    """A6: a correct general solution must never trip a guard."""
    assert check_overfit(REAL_IMPL, WEAK_EXAMPLES) is None


def test_equality_chain_is_blocked():
    """A2: check_input_equality_chain catches the naive if/elif table."""
    assert check_overfit(EQUALITY_CHAIN, WEAK_EXAMPLES) is not None


def test_lookup_table_should_be_blocked():
    """A1: check_literal_lookup catches dict keyed on example inputs."""
    assert check_overfit(LOOKUP_TABLE, WEAK_EXAMPLES) is not None


def test_table_plus_fallback_should_be_blocked():
    """A3: literal table is visible to the AST lookup guard."""
    assert check_overfit(TABLE_PLUS_FALLBACK, WEAK_EXAMPLES) is not None


@pytest.mark.xfail(
    reason='base64 hides the literal from string matching and AST constants',
    strict=True,
)
def test_obfuscated_table_should_be_blocked():
    """A5 hole at the guard layer: encoding defeats syntactic guards."""
    assert check_overfit(OBFUSCATED_TABLE, WEAK_EXAMPLES) is not None


def test_obfuscated_table_rejected_by_holdout(tmp_path):
    """A5 is rejected by holdout, not the code-mutation gate.

    Measured code-mutation score is exactly 0.800 with the default threshold
    0.8 (``killed/total < threshold`` is False), because the surviving mutant
    appends ``x`` after base64 ``=`` padding — Python ignores it (equivalent
    mutant). Do not move this assertion back onto the mutation gate.

    A pure lookup with no fallback cannot answer a held-out valid IBAN.
    """
    examples = WEAK_EXAMPLES + [
        Example(inputs={'iban': 'DE89370400440532013000'}, expected=True),
    ]
    # holdout_ratio=0.3 puts DE in holdout with the fixed seed (see split_examples).
    spec = ToolSpec(
        name='validate_iban_a5',
        description='Validate an IBAN (ISO 13616 mod-97).',
        parameters={'iban': 'string'},
        returns='boolean',
        examples=examples,
        holdout_ratio=0.3,
    )
    smith = ToolSmith(
        llm=_mock_llm(OBFUSCATED_TABLE),
        executor=InProcessExecutor(),
        max_attempts=1,
        output_dir=str(tmp_path),
        mutation_enabled=False,  # prove holdout alone rejects
    )
    result = smith.synthesize(spec)
    assert result.ok is False
    assert result.status == Status.HOLDOUT_FAILED
    assert result.holdout_failures


# --- the structural limit -------------------------------------------------

def test_undergeneralised_impl_passes_every_weak_example():
    """A4: no guard fires, and holdout cannot help either,
    because holdout is drawn from the same weak example set."""
    assert check_overfit(UNDERGENERALISED, WEAK_EXAMPLES) is None
    for ex in WEAK_EXAMPLES:
        assert _run(UNDERGENERALISED, ex.inputs['iban']) == ex.expected


def test_strong_examples_are_what_actually_catch_it():
    """Example quality — not the guards — is the real defence for A4.
    One negative differing only in checksum kills the cheat."""
    failed = [
        ex for ex in STRONG_EXAMPLES
        if _run(UNDERGENERALISED, ex.inputs['iban']) != ex.expected
    ]
    assert failed, 'strong examples must expose the under-generalised impl'


def test_real_impl_passes_strong_examples():
    for ex in STRONG_EXAMPLES:
        assert _run(REAL_IMPL, ex.inputs['iban']) == ex.expected, ex.inputs
