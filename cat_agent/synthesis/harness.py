"""Harness that wraps user code so the sandbox can return a value via stdout."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from cat_agent.synthesis.executors.base import ERROR_SENTINEL, RESULT_SENTINEL


def assert_json_serializable(value: Any, *, label: str = 'value') -> None:
    """Raise ``ValueError`` if *value* cannot cross the process boundary as JSON."""
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'{label} is not JSON-serialisable and cannot cross the sandbox boundary: '
            f'{type(value).__name__}: {exc}'
        ) from exc


def build_harness(code: str, function_name: str, inputs: Dict[str, Any]) -> str:
    """Wrap *code* so ``function_name(**inputs)`` prints ``__CAT_RESULT__<json>``.

    Inputs are embedded via ``json.dumps`` into a string literal (never raw
    f-string interpolation) so quotes/newlines/sentinel text stay safe.

    Return values are dumped **without** ``default=str``; non-JSON-serialisable
    results emit ``__CAT_ERROR__`` instead.
    """
    assert_json_serializable(inputs, label='inputs')
    inputs_literal = json.dumps(json.dumps(inputs, default=str))
    result_sentinel = json.dumps(RESULT_SENTINEL)
    error_sentinel = json.dumps(ERROR_SENTINEL)
    return (
        '# -*- coding: utf-8 -*-\n'
        'import json as _cat_json\n'
        f'{code.rstrip()}\n\n'
        f'_cat_inputs = _cat_json.loads({inputs_literal})\n'
        f'_cat_result = {function_name}(**_cat_inputs)\n'
        'try:\n'
        '    _cat_payload = _cat_json.dumps(_cat_result)\n'
        'except TypeError as _cat_exc:\n'
        '    _cat_err = (\n'
        '        "Return value is not JSON-serialisable ("\n'
        '        + type(_cat_result).__name__\n'
        '        + "): "\n'
        '        + str(_cat_exc)\n'
        '        + ". Return only JSON-serialisable types '\
        '(dict/list/str/int/float/bool/None)."\n'
        '    )\n'
        f'    print({error_sentinel} + _cat_err)\n'
        'else:\n'
        f'    print({result_sentinel} + _cat_payload)\n'
    )


def parse_harness_stdout(stdout: str) -> Tuple[str, Any]:
    """Split stdout into (pre-sentinel text, returned value).

    Raises ``ValueError`` if the result sentinel is missing/malformed, or if an
    ``__CAT_ERROR__`` sentinel is present (non-serialisable return).
    """
    text = stdout or ''
    lines = text.split('\n')
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index]
        if line.startswith(ERROR_SENTINEL):
            message = line[len(ERROR_SENTINEL):] or 'non-JSON-serialisable return value'
            raise ValueError(message)
        if not line.startswith(RESULT_SENTINEL):
            continue
        payload = line[len(RESULT_SENTINEL):]
        try:
            returned = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f'Harness sentinel present but payload is not valid JSON: {exc}'
            ) from exc
        prefix = '\n'.join(lines[:index]).rstrip('\n')
        return prefix, returned
    raise ValueError(
        f'Missing {RESULT_SENTINEL} sentinel on stdout — the harness did not return a value.'
    )


def map_timeout_to_fuel(timeout_s: Optional[float], default_fuel: int) -> int:
    """Map a wall-clock timeout hint onto a WASM instruction fuel budget.

    Empirically ~80M fuel ≈ 1s of simple CPython-in-WASM work.
    """
    if timeout_s is None or timeout_s <= 0:
        return default_fuel
    return max(1_000_000, int(timeout_s * 80_000_000))
