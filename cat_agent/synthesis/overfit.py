"""Heuristics that reject obviously overfit synthesised implementations."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence

from cat_agent.synthesis.spec import Example

_TRIVIAL_REPRS = {
    '0', '1', '-1', 'True', 'False', 'None', "''", '""', '[]', '{}',
    "'0'", "'1'", '"0"', '"1"',
}


def _is_trivial_literal(value: Any) -> bool:
    if value in (0, 1, -1, True, False, None, '', [], {}):
        return True
    return repr(value) in _TRIVIAL_REPRS


def check_hardcoded_expected(code: str, examples: Sequence[Example]) -> Optional[str]:
    """Return an error message if *code* embeds non-trivial expected literals."""
    import json

    hits: List[str] = []
    for example in examples:
        if _is_trivial_literal(example.expected):
            continue
        needle = repr(example.expected)
        if needle and needle in code:
            hits.append(needle)
            continue
        if isinstance(example.expected, str) and len(example.expected) >= 2:
            if json.dumps(example.expected) in code:
                hits.append(repr(example.expected))
    if hits:
        unique = sorted(set(hits))
        return (
            'Overfit guard: implementation embeds expected example values '
            f'{unique}. Derive the result from the inputs instead of hardcoding.'
        )
    return None


def check_input_equality_chain(code: str, examples: Sequence[Example]) -> Optional[str]:
    """Flag long chains of ``==`` comparisons against example input values."""
    import json

    input_values: List[Any] = []
    for example in examples:
        input_values.extend(example.inputs.values())

    match_count = 0
    for value in input_values:
        if _is_trivial_literal(value):
            continue
        patterns = [re.escape(repr(value))]
        if isinstance(value, str):
            patterns.append(re.escape(json.dumps(value)))
        for pattern in patterns:
            if re.search(rf'==\s*{pattern}', code) or re.search(rf'{pattern}\s*==', code):
                match_count += 1
                break

    # More than half of non-trivial inputs compared with == → likely a lookup table.
    non_trivial = sum(1 for v in input_values if not _is_trivial_literal(v))
    threshold = max(2, (non_trivial + 1) // 2)
    if match_count >= threshold and match_count >= 2:
        return (
            'Overfit guard: implementation compares inputs against example values '
            f'({match_count} matches). Write a general solution, not a case table.'
        )
    return None


def check_overfit(code: str, examples: Sequence[Example]) -> Optional[str]:
    """Run all overfit guards; return the first failure message or ``None``."""
    return (
        check_hardcoded_expected(code, examples)
        or check_input_equality_chain(code, examples)
    )
