"""Tests for input-space mutation (A4-class under-generalisation probe)."""

from __future__ import annotations

from typing import Any, Dict

from cat_agent.synthesis.mutation import (
    count_string_substitutions,
    perturb_inputs,
    probe_input_sensitivity,
)
from cat_agent.synthesis.spec import Example, ToolSpec

WEAK_EXAMPLES = [
    Example(inputs={'iban': 'TR330006100519786457841326'}, expected=True),
    Example(inputs={'iban': 'TR250001000010000000000001'}, expected=True),
    Example(inputs={'iban': 'XX00'}, expected=False),
    Example(inputs={'iban': 'TR33000610051978645784132'}, expected=False),
    Example(inputs={'iban': ''}, expected=False),
]

# Strong set: near-miss negatives + DE (22) + NO (15). The NO positive is why
# the substitution total is 657 (= 396 + 162 + 99), not 558 (TR×2 + DE only).
STRONG_EXAMPLES = WEAK_EXAMPLES + [
    Example(inputs={'iban': 'TR330006100519786457841327'}, expected=False),
    Example(inputs={'iban': 'TR340006100519786457841326'}, expected=False),
    Example(inputs={'iban': 'DE89370400440532013000'}, expected=True),
    Example(inputs={'iban': 'NO9386011117947'}, expected=True),
    Example(inputs={'iban': 'TR33!006100519786457841326'}, expected=False),
]

UNDERGENERALISED = '''\
def validate_iban(iban):
    return len(iban) == 26 and iban[:2] == "TR"
'''

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


def _run(code: str, inputs: Dict[str, Any]) -> Any:
    ns: Dict[str, Any] = {}
    exec(code, ns)  # noqa: S102
    return ns['validate_iban'](**inputs)


def _subst_variants_index_ge_4(base: str, variant: str) -> bool:
    if len(base) != len(variant):
        return False
    diffs = [i for i, (a, b) in enumerate(zip(base, variant)) if a != b]
    return len(diffs) == 1 and diffs[0] >= 4


def test_per_example_substitution_counts():
    """Denominators must be attributable — assert per length, not only totals.

    Rule: substitution only, indices >= 4 when len > 8, nine digit alts each.
    """
    assert count_string_substitutions('TR330006100519786457841326') == 198  # 22×9
    assert count_string_substitutions('DE89370400440532013000') == 162  # 18×9
    assert count_string_substitutions('NO9386011117947') == 99  # 11×9


def test_weak_substitution_total_is_396():
    positives = [ex for ex in WEAK_EXAMPLES if ex.expected is True]
    assert len(positives) == 2
    total = sum(count_string_substitutions(ex.inputs['iban']) for ex in positives)
    assert total == 396  # 2 × 198


def test_strong_substitution_total_is_657_not_558():
    """657 = 2×198 (TR26) + 162 (DE22) + 99 (NO15).

    Not negatives (never probed) and not probe feedback — the strong fixture
    includes the Norwegian positive that 558 omits.
    """
    positives = [ex for ex in STRONG_EXAMPLES if ex.expected is True]
    per = {
        ex.inputs['iban']: count_string_substitutions(ex.inputs['iban'])
        for ex in positives
    }
    assert per['TR330006100519786457841326'] == 198
    assert per['TR250001000010000000000001'] == 198
    assert per['DE89370400440532013000'] == 162
    assert per['NO9386011117947'] == 99
    assert sum(per.values()) == 657


def test_perturb_inputs_substitution_count_matches_helper():
    positives = [ex for ex in WEAK_EXAMPLES if ex.expected is True]
    total = 0
    for ex in positives:
        variants = perturb_inputs(ex, limit=10_000)
        base = ex.inputs['iban']
        subst = {
            v['iban'] for v in variants
            if _subst_variants_index_ge_4(base, v['iban'])
        }
        assert len(subst) == count_string_substitutions(base)
        total += len(subst)
    assert total == 396


def test_a4_insensitive_to_396_substitutions_mod97_sensitive():
    positives = [ex for ex in WEAK_EXAMPLES if ex.expected is True]
    a4_changed = 0
    real_changed = 0
    total = 0
    for ex in positives:
        base = ex.inputs['iban']
        base_a4 = _run(UNDERGENERALISED, ex.inputs)
        base_real = _run(REAL_IMPL, ex.inputs)
        for variant in perturb_inputs(ex, limit=10_000):
            iban = variant['iban']
            if not _subst_variants_index_ge_4(base, iban):
                continue
            total += 1
            if _run(UNDERGENERALISED, variant) != base_a4:
                a4_changed += 1
            if _run(REAL_IMPL, variant) != base_real:
                real_changed += 1
    assert total == 396
    assert a4_changed == 0
    assert real_changed == 396


def test_a4_strong_input_sensitivity_is_0_of_657():
    positives = [ex for ex in STRONG_EXAMPLES if ex.expected is True]
    changed = tried = 0
    for ex in positives:
        base = ex.inputs['iban']
        base_out = _run(UNDERGENERALISED, ex.inputs)
        for variant in perturb_inputs(ex, limit=10_000):
            if not _subst_variants_index_ge_4(base, variant['iban']):
                continue
            tried += 1
            if _run(UNDERGENERALISED, variant) != base_out:
                changed += 1
    assert tried == 657
    assert changed == 0


def test_probe_reports_insensitivity_for_a4_not_mod97():
    spec = ToolSpec(
        name='probe_iban',
        description='Validate IBAN.',
        parameters={'iban': 'string'},
        returns='boolean',
        examples=list(WEAK_EXAMPLES),
        holdout_ratio=0.2,
    )
    a4 = probe_input_sensitivity(
        UNDERGENERALISED, spec, WEAK_EXAMPLES, _run, limit=64,
    )
    real = probe_input_sensitivity(
        REAL_IMPL, spec, WEAK_EXAMPLES, _run, limit=64,
    )
    assert a4, 'A4 must be reported as input-insensitive'
    assert a4[0].variants_that_changed_output == 0
    assert a4[0].variants_per_example
    assert sum(a4[0].variants_per_example.values()) == 198
    assert real == []


def test_probe_skips_negatives():
    """Perturbing negatives carries no signal — they must not enter the probe."""
    only_neg = [ex for ex in WEAK_EXAMPLES if ex.expected is False]
    spec = ToolSpec(
        name='probe_neg',
        description='Validate IBAN.',
        parameters={'iban': 'string'},
        returns='boolean',
        examples=list(WEAK_EXAMPLES),
        holdout_ratio=0.2,
    )
    assert probe_input_sensitivity(UNDERGENERALISED, spec, only_neg, _run) == []


def test_perturb_inputs_is_deterministic():
    ex = WEAK_EXAMPLES[0]
    a = perturb_inputs(ex, limit=64)
    b = perturb_inputs(ex, limit=64)
    assert a == b
