"""Heuristics that reject obviously overfit synthesised implementations."""

from __future__ import annotations

import ast
import re
from typing import Any, List, Optional, Sequence, Set, Tuple

from cat_agent.synthesis.spec import Example

_TRIVIAL_STR_REPRS = {"''", '""', "'0'", "'1'", '"0"', '"1"'}


def _is_trivial_literal(value: Any) -> bool:
    """Return True when *value* is too common to use as an overfit signal.

    This guard is *deliberately* weak for boolean / small-enum returns: every
    legitimate validator contains the tokens ``True`` / ``False``, so treating
    those as non-trivial would flag correct implementations. Coverage for
    boolean-returning tools comes from :func:`check_literal_lookup` (AST
    tables) and the mutation gate in :mod:`cat_agent.synthesis.mutation`.

    Type-aware checks avoid ``==`` collisions such as ``1.0 == True``.
    """
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, int) and not isinstance(value, bool):
        return value in (0, 1, -1)
    if isinstance(value, float):
        return value in (0.0, 1.0, -1.0)
    if isinstance(value, str):
        return value in ('', '0', '1') or repr(value) in _TRIVIAL_STR_REPRS
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        return len(value) == 0
    return repr(value) in _TRIVIAL_STR_REPRS


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


def _const_key(node: ast.AST) -> Optional[Tuple[type, Any]]:
    """Return ``(type, value)`` for an ``ast.Constant``, else ``None``."""
    if isinstance(node, ast.Constant):
        return (type(node.value), node.value)
    return None


def _collect_literal_elements(node: ast.AST) -> List[Tuple[type, Any]]:
    """Extract constant keys/elements from a literal collection node."""
    out: List[Tuple[type, Any]] = []
    if isinstance(node, ast.Dict):
        for key in node.keys:
            if key is None:
                continue
            item = _const_key(key)
            if item is not None:
                out.append(item)
    elif isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        for elt in node.elts:
            item = _const_key(elt)
            if item is not None:
                out.append(item)
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {'dict'}:
            for kw in node.keywords:
                if kw.arg is None:
                    # dict(**mapping) — skip
                    continue
                # dict(a=1) keys are identifiers, not example values
                continue
            for arg in node.args:
                if isinstance(arg, (ast.List, ast.Tuple)):
                    for elt in arg.elts:
                        if isinstance(elt, (ast.Tuple, ast.List)) and len(elt.elts) >= 1:
                            item = _const_key(elt.elts[0])
                            if item is not None:
                                out.append(item)
        elif node.func.id in {'set', 'frozenset', 'list', 'tuple'}:
            for arg in node.args:
                if isinstance(arg, (ast.List, ast.Tuple, ast.Set)):
                    out.extend(_collect_literal_elements(arg))
                else:
                    item = _const_key(arg)
                    if item is not None:
                        out.append(item)
    return out


def check_literal_lookup(code: str, examples: Sequence[Example]) -> Optional[str]:
    """Flag dict/set literals (or ``in`` collections) keyed on example inputs.

    Catches ``T = {"IBAN...": True, ...}; return T.get(iban)`` style cheats
    that never use ``==`` and therefore slip past
    :func:`check_input_equality_chain`. Matching is by value **and** type so
    ``True`` does not collide with ``1``.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    example_inputs: Set[Tuple[type, Any]] = set()
    for example in examples:
        for value in example.inputs.values():
            if _is_trivial_literal(value):
                continue
            example_inputs.add((type(value), value))

    if len(example_inputs) < 2:
        return None

    best_hits = 0
    for node in ast.walk(tree):
        elements: List[Tuple[type, Any]] = []
        if isinstance(node, (ast.Dict, ast.Set, ast.List, ast.Tuple, ast.Call)):
            elements = _collect_literal_elements(node)
        elif isinstance(node, ast.Compare):
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    elements.extend(_collect_literal_elements(comparator))
        if not elements:
            continue
        hits = sum(1 for item in elements if item in example_inputs)
        if hits > best_hits:
            best_hits = hits

    if best_hits >= 2:
        return (
            'Overfit guard: implementation embeds a literal lookup table keyed '
            f'on example inputs ({best_hits} matches). Write a general solution, '
            'not a case table.'
        )
    return None


def check_overfit(code: str, examples: Sequence[Example]) -> Optional[str]:
    """Run all overfit guards; return the first failure message or ``None``."""
    return (
        check_hardcoded_expected(code, examples)
        or check_input_equality_chain(code, examples)
        or check_literal_lookup(code, examples)
    )
