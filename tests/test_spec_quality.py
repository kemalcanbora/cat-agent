"""Tests for intake-time ToolSpec quality linting."""

from __future__ import annotations

from cat_agent.synthesis.spec import Example, ToolSpec
from cat_agent.synthesis.spec_quality import lint_spec

# Keep in sync with tests/test_synthesis_adversarial.py
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


def _iban_spec(examples: list) -> ToolSpec:
    return ToolSpec(
        name='lint_iban',
        description='Validate an IBAN.',
        parameters={'iban': 'string'},
        returns='boolean',
        examples=list(examples),
        holdout_ratio=0.2,
    )


def test_weak_examples_warn_negatives_far_from_positives():
    warnings = lint_spec(_iban_spec(WEAK_EXAMPLES))
    codes = {w.code for w in warnings if w.severity == 'warn'}
    assert 'negatives_far_from_positives' in codes


def test_strong_examples_have_no_warn_severity():
    warnings = lint_spec(_iban_spec(STRONG_EXAMPLES))
    assert [w for w in warnings if w.severity == 'warn'] == []


def test_unused_parameter_when_constant_across_examples():
    spec = ToolSpec(
        name='lint_unused',
        description='Uses amount but currency never varies.',
        parameters={
            'amount': 'number',
            'currency': 'string',
        },
        returns='number',
        examples=[
            Example(inputs={'amount': 10, 'currency': 'EUR'}, expected=10),
            Example(inputs={'amount': 20, 'currency': 'EUR'}, expected=20),
            Example(inputs={'amount': 30, 'currency': 'EUR'}, expected=30),
            Example(inputs={'amount': 40, 'currency': 'EUR'}, expected=40),
        ],
        holdout_ratio=0.25,
    )
    warnings = lint_spec(spec)
    unused = [w for w in warnings if w.code == 'unused_parameter']
    assert unused
    assert unused[0].severity == 'info'
    assert 'currency' in unused[0].message


def test_unused_parameter_is_info_on_multiparam_fixture():
    """Varying one param at a time with the others held constant is legitimate.

    ``unused_parameter`` must stay ``info`` so ``allow_weak_spec=False`` does
    not abort on this common pattern.
    """
    spec = ToolSpec(
        name='lint_vat_split',
        description='Split gross into net and tax; currency is a label.',
        parameters={
            'gross': 'number',
            'rate': 'number',
            'currency': 'string',
        },
        returns='object',
        examples=[
            Example(
                inputs={'gross': 120, 'rate': 0.2, 'currency': 'EUR'},
                expected={'net': 100.0, 'tax': 20.0},
            ),
            Example(
                inputs={'gross': 100, 'rate': 0.0, 'currency': 'EUR'},
                expected={'net': 100.0, 'tax': 0.0},
            ),
            Example(
                inputs={'gross': 1, 'rate': 0.2, 'currency': 'EUR'},
                expected={'net': 0.83, 'tax': 0.17},
            ),
            Example(
                inputs={'gross': 240, 'rate': 0.2, 'currency': 'EUR'},
                expected={'net': 200.0, 'tax': 40.0},
            ),
            Example(
                inputs={'gross': 50, 'rate': 0.1, 'currency': 'EUR'},
                expected={'net': 45.45, 'tax': 4.55},
            ),
            Example(
                inputs={'gross': 0, 'rate': 0.2, 'currency': 'EUR'},
                expected={'net': 0.0, 'tax': 0.0},
            ),
        ],
        holdout_ratio=0.3,
    )
    warnings = lint_spec(spec)
    unused = [w for w in warnings if w.code == 'unused_parameter']
    assert len(unused) == 1
    assert unused[0].severity == 'info'
    assert 'currency' in unused[0].message
    assert 'all 6 examples' in unused[0].message
    # Must not abort allow_weak_spec=False
    assert not any(w.severity == 'warn' for w in unused)


def test_lint_spec_never_raises_on_degenerate_inputs():
    # Empty examples — ToolSpec validation requires >=2, so build via object.__new__
    # is unnecessary: call lint helpers through a minimal valid-then-mutate path.
    # Instead exercise lint_spec with the thinnest valid specs and odd payloads.
    thin = ToolSpec(
        name='lint_thin',
        description='Minimal.',
        parameters={'x': 'integer'},
        returns='integer',
        examples=[
            Example(inputs={'x': 1}, expected=1),
            Example(inputs={'x': 2}, expected=2),
            Example(inputs={'x': 3}, expected=3),
        ],
        holdout_ratio=0.3,
    )
    assert isinstance(lint_spec(thin), list)

    mixed = ToolSpec(
        name='lint_mixed',
        description='Non-string inputs.',
        parameters={'n': 'integer', 'flag': 'boolean'},
        returns='boolean',
        examples=[
            Example(inputs={'n': 1, 'flag': True}, expected=True),
            Example(inputs={'n': 2, 'flag': False}, expected=False),
            Example(inputs={'n': 3, 'flag': True}, expected=True),
            Example(inputs={'n': 4, 'flag': False}, expected=False),
        ],
        holdout_ratio=0.25,
    )
    assert isinstance(lint_spec(mixed), list)

    # Direct call with a broken stand-in must not raise.
    class _Broken:
        examples = None
        parameters = None

    assert isinstance(lint_spec(_Broken()), list)  # type: ignore[arg-type]
