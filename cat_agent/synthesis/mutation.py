"""AST mutation testing for under-specified ToolSpec example sets.

A surviving mutant means the examples do not exercise a branch — the *spec*
is weak, not necessarily the code. Used by :class:`~cat_agent.synthesis.smith.ToolSmith`
after holdout passes; never fed back into the LLM retry loop.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from cat_agent.synthesis.spec import Example, ToolSpec


@dataclass
class Mutant:
    code: str
    description: str  # e.g. "line 7: Eq -> NotEq"


_COMPARE_SWAPS = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.Gt: ast.LtE,
    ast.LtE: ast.Gt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}

_BINOP_SWAPS = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
}


def generate_mutants(code: str, *, limit: int = 12) -> List[Mutant]:
    """Return up to *limit* single-edit mutants of *code*, deterministically.

    Mutation sites are ordered by ``(lineno, col_offset)``. Invalid source
    yields an empty list (never raises).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    docstring_ids = _docstring_node_ids(tree)
    sites = _find_sites(tree, docstring_ids)
    mutants: List[Mutant] = []
    for site in sites:
        if len(mutants) >= limit:
            break
        mutant_tree = copy.deepcopy(tree)
        description = _apply_site(mutant_tree, site, docstring_ids)
        if not description:
            continue
        try:
            mutated = ast.unparse(mutant_tree)
            ast.parse(mutated)
        except Exception:
            continue
        lineno = site[0][0]
        mutants.append(Mutant(code=mutated, description=f'line {lineno}: {description}'))
    return mutants


# Site: ((lineno, col), kind, payload)
Site = Tuple[Tuple[int, int], str, Any]


def _docstring_node_ids(tree: ast.AST) -> set:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, 'body', None) or []
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _pos(node: ast.AST) -> Tuple[int, int]:
    return (getattr(node, 'lineno', 0) or 0, getattr(node, 'col_offset', 0) or 0)


def _find_sites(tree: ast.AST, docstring_ids: set) -> List[Site]:
    sites: List[Site] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if id(node) in docstring_ids:
            continue
        # Skip constants nested under import aliases (already skipped imports).
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                swap = _COMPARE_SWAPS.get(type(op))
                if swap is not None:
                    sites.append((_pos(node), 'compare', (i, type(op).__name__, swap)))
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                sites.append((_pos(node), 'boolop', ('And', ast.Or)))
            elif isinstance(node.op, ast.Or):
                sites.append((_pos(node), 'boolop', ('Or', ast.And)))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            sites.append((_pos(node), 'drop_not', None))
        elif isinstance(node, ast.BinOp):
            swap = _BINOP_SWAPS.get(type(node.op))
            if swap is not None:
                sites.append((_pos(node), 'binop', (type(node.op).__name__, swap)))
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                sites.append((_pos(node), 'const_bool', None))
            elif isinstance(node.value, int) and not isinstance(node.value, bool):
                sites.append((_pos(node), 'const_num', None))
            elif isinstance(node.value, float):
                sites.append((_pos(node), 'const_num', None))
            elif isinstance(node.value, str) and node.value != '':
                sites.append((_pos(node), 'const_str', None))

    sites.sort(key=lambda s: (s[0][0], s[0][1], s[1], repr(s[2])))
    return sites


class _Mutator(ast.NodeTransformer):
    """Apply exactly one mutation identified by position + kind."""

    def __init__(self, site: Site, docstring_ids: set):
        self.site = site
        self.docstring_ids = docstring_ids
        self.applied: Optional[str] = None
        self._done = False

    def _match(self, node: ast.AST) -> bool:
        return _pos(node) == self.site[0] and not self._done

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if self.site[1] == 'compare' and self._match(node):
            i, old_name, new_t = self.site[2]
            if 0 <= i < len(node.ops):
                node.ops[i] = new_t()
                self.applied = f'{old_name} -> {new_t.__name__}'
                self._done = True
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        if self.site[1] == 'boolop' and self._match(node):
            old_name, new_t = self.site[2]
            node.op = new_t()
            self.applied = f'{old_name} -> {new_t.__name__}'
            self._done = True
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if self.site[1] == 'drop_not' and self._match(node) and isinstance(node.op, ast.Not):
            self.applied = 'drop Not'
            self._done = True
            return node.operand
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.site[1] == 'binop' and self._match(node):
            old_name, new_t = self.site[2]
            node.op = new_t()
            self.applied = f'{old_name} -> {new_t.__name__}'
            self._done = True
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if id(node) in self.docstring_ids:
            return node
        if not self._match(node):
            return node
        kind = self.site[1]
        if kind == 'const_bool' and isinstance(node.value, bool):
            old = node.value
            node.value = not old
            self.applied = f'bool {old} -> {node.value}'
            self._done = True
        elif kind == 'const_num' and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            old = node.value
            node.value = old + 1
            self.applied = f'{old!r} -> {node.value!r}'
            self._done = True
        elif kind == 'const_str' and isinstance(node.value, str):
            node.value = node.value + 'x'
            self.applied = 'str + "x"'
            self._done = True
        return node


def _apply_site(tree: ast.AST, site: Site, docstring_ids: set) -> Optional[str]:
    mutator = _Mutator(site, docstring_ids)
    mutator.visit(tree)
    return mutator.applied


# ---------------------------------------------------------------------------
# Input-space mutation — different axis from code mutants above.
# Perturb *inputs*, keep code fixed, observe whether the output moves.
# ---------------------------------------------------------------------------

_DIGITS = '0123456789'
_UPPER = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
_INSERT_ALPHABET = _DIGITS + _UPPER


@dataclass
class Insensitivity:
    param: str
    base_inputs: Dict[str, Any]
    variants_tried: int
    variants_that_changed_output: int
    sample_unchanged: List[Any]  # at most 3, for the question text
    # Attributable denominator: example label → substitution-variant count.
    # Labels are ``repr`` of the probed parameter value (positives only).
    variants_per_example: Dict[str, int] = field(default_factory=dict)


def count_string_substitutions(value: str) -> int:
    """Number of single-char substitutions under the input-mutation rule.

    For strings longer than 8 characters the first 4 are left untouched
    (structural prefix). Each remaining character is replaced by every other
    glyph in its class (digits → 9 alternatives; uppercase → 25).

    Reference baselines for digit-bodied IBANs (indices ≥ 4, 9 alts each)::

        len 26 → 22 × 9 = 198
        len 22 → 18 × 9 = 162
        len 15 → 11 × 9 =  99

    Weak set (two 26-char positives): 2 × 198 = 396.
    Strong set (those plus DE22 + NO15): 396 + 162 + 99 = 657.
    (558 would be TR×2 + DE only — the NO positive is why the total is 657,
    not negative perturbation or probe feedback.)
    """
    return len(list(_string_substitutions(value)))


def _string_substitutions(value: str):
    """Yield single-character substitutions (deterministic order)."""
    start = 4 if len(value) > 8 else 0
    for i in range(start, len(value)):
        alphabet = _subst_alphabet(value[i])
        for repl in alphabet:
            if repl == value[i]:
                continue
            yield value[:i] + repl + value[i + 1:]


def perturb_inputs(example: Example, *, limit: int = 64) -> List[Dict[str, Any]]:
    """Return deterministic single-edit input variants of *example*, capped at *limit*.

    String operators (one at a time): substitution, deletion, insertion.
    For strings longer than 8 characters, substitution leaves the first 4
    characters untouched. Int/float: ``n±1`` and sign flip.

    Only the caller's chosen *example* is perturbed — callers that measure
    sensitivity must pass positives themselves (see :func:`probe_input_sensitivity`).
    """
    variants: List[Dict[str, Any]] = []
    seen = set()

    def _add(inputs: Dict[str, Any]) -> None:
        if len(variants) >= limit:
            return
        key = repr(sorted(inputs.items()))
        if key in seen:
            return
        seen.add(key)
        variants.append(dict(inputs))

    base = dict(example.inputs)
    for param in sorted(base.keys()):
        if len(variants) >= limit:
            break
        value = base[param]
        if isinstance(value, str):
            for new_val in _string_edits(value):
                if len(variants) >= limit:
                    break
                candidate = dict(base)
                candidate[param] = new_val
                _add(candidate)
        elif isinstance(value, bool):
            continue
        elif isinstance(value, int):
            for new_val in _int_edits(value):
                if len(variants) >= limit:
                    break
                candidate = dict(base)
                candidate[param] = new_val
                _add(candidate)
        elif isinstance(value, float):
            for new_val in _float_edits(value):
                if len(variants) >= limit:
                    break
                candidate = dict(base)
                candidate[param] = new_val
                _add(candidate)
    return variants


def probe_input_sensitivity(
    code: str,
    spec: ToolSpec,
    examples: Sequence[Example],
    runner: Callable[[str, Dict[str, Any]], Any],
    *,
    limit: int = 64,
) -> List[Insensitivity]:
    """Find parameters whose **positive** examples are insensitive to input edits.

    Negatives are never probed: perturbing an already-invalid input produces
    variants that both a shape check and a correct implementation reject, which
    dilutes the denominator without adding signal.

    *runner(code, inputs)* should return the tool output or raise. A finding is
    emitted when every variant of a positive example leaves the output unchanged.
    Advisory only — never fails synthesis.

    ``variants_per_example`` records the full substitution-only count for each
    probed value (see :func:`count_string_substitutions`), independent of *limit*.
    """
    findings: List[Insensitivity] = []
    for example in examples:
        if not _is_positive_expected(example.expected):
            continue
        base_inputs = dict(example.inputs)
        try:
            base_out = runner(code, base_inputs)
        except Exception:  # noqa: BLE001
            continue
        for param in sorted(base_inputs.keys()):
            value = base_inputs[param]
            if isinstance(value, bool):
                continue
            variants = _perturb_param(base_inputs, param, limit=limit)
            if not variants:
                continue
            changed = 0
            unchanged_samples: List[Any] = []
            for var_inputs in variants:
                try:
                    out = runner(code, var_inputs)
                except Exception:  # noqa: BLE001
                    changed += 1
                    continue
                if out != base_out:
                    changed += 1
                elif len(unchanged_samples) < 3:
                    unchanged_samples.append(var_inputs.get(param))
            label = repr(value)
            if isinstance(value, str):
                attrib = {label: count_string_substitutions(value)}
            else:
                attrib = {label: len(variants)}
            if changed == 0:
                findings.append(Insensitivity(
                    param=param,
                    base_inputs=dict(base_inputs),
                    variants_tried=len(variants),
                    variants_that_changed_output=0,
                    sample_unchanged=unchanged_samples,
                    variants_per_example=attrib,
                ))
    return findings


def measure_substitution_sensitivity(
    code: str,
    examples: Sequence[Example],
    runner: Callable[[str, Dict[str, Any]], Any],
    *,
    limit: int = 10_000,
) -> List[Dict[str, Any]]:
    """Substitution-only sensitivity stats for positives (manifest verification).

    Returns one dict per (positive example, string param)::

        {"param", "changed", "variants", "example_label"}
    """
    rows: List[Dict[str, Any]] = []
    for example in examples:
        if not _is_positive_expected(example.expected):
            continue
        base_inputs = dict(example.inputs)
        try:
            base_out = runner(code, base_inputs)
        except Exception:  # noqa: BLE001
            continue
        for param in sorted(base_inputs.keys()):
            value = base_inputs[param]
            if not isinstance(value, str):
                continue
            changed = 0
            tried = 0
            for new_val in _string_substitutions(value):
                if tried >= limit:
                    break
                tried += 1
                candidate = dict(base_inputs)
                candidate[param] = new_val
                try:
                    out = runner(code, candidate)
                except Exception:  # noqa: BLE001
                    changed += 1
                    continue
                if out != base_out:
                    changed += 1
            rows.append({
                'param': param,
                'changed': changed,
                'variants': tried,
                'example_label': repr(value),
            })
    return rows


def _is_positive_expected(expected: Any) -> bool:
    if isinstance(expected, bool):
        return expected is True
    return bool(expected)


def _subst_alphabet(ch: str) -> str:
    if ch in _DIGITS:
        return _DIGITS
    if ch in _UPPER or ch in _UPPER.lower():
        return _UPPER
    return _DIGITS + _UPPER


def _string_edits(value: str) -> List[str]:
    """Ordered: substitutions, then deletions, then insertions."""
    out: List[str] = []
    seen = set()

    def push(s: str) -> None:
        if s != value and s not in seen:
            seen.add(s)
            out.append(s)

    for s in _string_substitutions(value):
        push(s)

    for i in range(len(value)):
        push(value[:i] + value[i + 1:])

    for i in range(len(value) + 1):
        for ch in _INSERT_ALPHABET:
            push(value[:i] + ch + value[i:])

    return out


def _perturb_param(
    base_inputs: Dict[str, Any],
    param: str,
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    value = base_inputs[param]
    edits: List[Any]
    if isinstance(value, str):
        edits = _string_edits(value)
    elif isinstance(value, int) and not isinstance(value, bool):
        edits = _int_edits(value)
    elif isinstance(value, float):
        edits = _float_edits(value)
    else:
        return []
    variants: List[Dict[str, Any]] = []
    for edit in edits:
        if len(variants) >= limit:
            break
        candidate = dict(base_inputs)
        candidate[param] = edit
        variants.append(candidate)
    return variants


def _int_edits(value: int) -> List[int]:
    out: List[int] = []
    for candidate in (value + 1, value - 1, -value if value != 0 else None):
        if candidate is None or candidate == value:
            continue
        if candidate not in out:
            out.append(candidate)
    return out


def _float_edits(value: float) -> List[float]:
    out: List[float] = []
    for candidate in (value + 1.0, value - 1.0, -value if value != 0.0 else None):
        if candidate is None or candidate == value:
            continue
        if candidate not in out:
            out.append(candidate)
    return out
