"""Lint ToolSpec example sets for under-specification (intake-time, no LLM).

These checks answer: *do the examples pin down a unique behaviour?*
They complement code mutation, which only asks whether examples exercise
the implementation that happened to be produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set

from cat_agent.synthesis.spec import Example, ToolSpec


@dataclass
class SpecWarning:
    code: str  # stable id, e.g. "negatives_far_from_positives"
    message: str  # user-facing, actionable, names the fix
    severity: str  # "warn" | "info"


def lint_spec(spec: ToolSpec) -> List[SpecWarning]:
    """Return advisory warnings for a weak or incomplete example set.

    Never raises. Returns ``[]`` when the spec looks fine.
    """
    try:
        examples = list(spec.examples or [])
        parameters = dict(spec.parameters or {})
        warnings: List[SpecWarning] = []
        warnings.extend(_check_too_few_examples(examples))
        warnings.extend(_check_identical_expected(examples))
        warnings.extend(_check_unused_parameter(examples, parameters))
        warnings.extend(_check_negatives_far_from_positives(examples))
        return warnings
    except Exception:  # noqa: BLE001 — lint must never break intake
        return []


def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance (stdlib only)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def _check_too_few_examples(examples: Sequence[Example]) -> List[SpecWarning]:
    out: List[SpecWarning] = []
    n = len(examples)
    if n < 4:
        out.append(SpecWarning(
            code='too_few_examples',
            message=(
                f'Only {n} example(s) supplied; aim for at least 4 so work and '
                'holdout splits can discriminate behaviour. Add more concrete rows.'
            ),
            severity='warn',
        ))
    if examples and all(isinstance(ex.expected, bool) for ex in examples):
        positives = sum(1 for ex in examples if ex.expected is True)
        negatives = sum(1 for ex in examples if ex.expected is False)
        if positives < 2 or negatives < 2:
            out.append(SpecWarning(
                code='too_few_examples',
                message=(
                    f'Boolean specs need at least 2 positives and 2 negatives '
                    f'(found {positives} true / {negatives} false). '
                    'Add examples for the thin class.'
                ),
                severity='warn',
            ))
    return out


def _check_identical_expected(examples: Sequence[Example]) -> List[SpecWarning]:
    if len(examples) < 2:
        return []
    first = examples[0].expected
    if all(_same_value(ex.expected, first) for ex in examples):
        return [SpecWarning(
            code='identical_expected',
            message=(
                f'Every example expects {first!r}, so the opposite branch is '
                'unconstrained. Add at least one example with a different expected value.'
            ),
            severity='warn',
        )]
    return []


def _check_unused_parameter(
    examples: Sequence[Example],
    parameters: Dict[str, Any],
) -> List[SpecWarning]:
    if not examples or not parameters:
        return []
    out: List[SpecWarning] = []
    for name in parameters:
        values = []
        missing = False
        for ex in examples:
            if name not in ex.inputs:
                missing = True
                break
            values.append(ex.inputs[name])
        if missing or not values:
            continue
        if all(_same_value(v, values[0]) for v in values):
            out.append(SpecWarning(
                code='unused_parameter',
                message=(
                    f'{name!r} is the same in all {len(examples)} examples '
                    f'({values[0]!r}), so nothing in the spec requires the '
                    'implementation to read it.'
                ),
                severity='info',
            ))
    return out


def _check_negatives_far_from_positives(
    examples: Sequence[Example],
) -> List[SpecWarning]:
    if not examples:
        return []
    if not all(isinstance(ex.expected, bool) for ex in examples):
        return []
    positives = [ex for ex in examples if ex.expected is True]
    negatives = [ex for ex in examples if ex.expected is False]
    if not positives or not negatives:
        return []

    string_fields: Set[str] = set()
    for ex in examples:
        for key, value in ex.inputs.items():
            if isinstance(value, str):
                string_fields.add(key)
    if not string_fields:
        return []

    for field_name in sorted(string_fields):
        pos_vals = [
            ex.inputs[field_name] for ex in positives
            if isinstance(ex.inputs.get(field_name), str)
        ]
        neg_vals = [
            ex.inputs[field_name] for ex in negatives
            if isinstance(ex.inputs.get(field_name), str)
        ]
        if not pos_vals or not neg_vals:
            continue

        min_dist = min(levenshtein(p, n) for p in pos_vals for n in neg_vals)
        pos_lengths = {len(p) for p in pos_vals}
        neg_lengths = {len(n) for n in neg_vals}
        shares_length = bool(pos_lengths & neg_lengths)

        if min_dist > 2 or not shares_length:
            return [SpecWarning(
                code='negatives_far_from_positives',
                message=(
                    f'Negatives for {field_name!r} are far from positives '
                    f'(min edit distance {min_dist}'
                    f'{"" if shares_length else ", no shared length"}). '
                    'Add one negative that differs from a valid example by a '
                    'single character so checksum-style rules cannot be faked '
                    'with a shape check.'
                ),
                severity='warn',
            )]
    return []


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return a == b
