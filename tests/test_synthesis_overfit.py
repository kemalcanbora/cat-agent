"""Isolated overfit-guard unit tests."""

from __future__ import annotations

from cat_agent.synthesis.overfit import (
    check_hardcoded_expected,
    check_input_equality_chain,
    check_overfit,
)
from cat_agent.synthesis.spec import Example


def _examples() -> list:
    return [
        Example(inputs={'code': 'TR001'}, expected={'ok': True}),
        Example(inputs={'code': 'TR002'}, expected={'ok': True}),
        Example(inputs={'code': 'TR003'}, expected={'ok': False}),
        Example(inputs={'code': 'TR004'}, expected={'ok': False}),
    ]


class TestOverfitGuards:

    def test_hardcoded_expected_fires(self):
        examples = [
            Example(inputs={'code': 'a'}, expected='RESULT_ALPHA'),
            Example(inputs={'code': 'b'}, expected='RESULT_BETA'),
            Example(inputs={'code': 'c'}, expected='RESULT_GAMMA'),
        ]
        bad = 'def f(code):\n    return "RESULT_ALPHA"\n'
        assert check_hardcoded_expected(bad, examples)

    def test_input_equality_chain_fires_without_expected_literals(self):
        """if/elif over example inputs, but no expected-value literals in the body."""
        examples = _examples()
        code = '''\
def classify(code: str) -> dict:
    if code == "TR001":
        return {"status": "one"}
    elif code == "TR002":
        return {"status": "two"}
    elif code == "TR003":
        return {"status": "three"}
    elif code == "TR004":
        return {"status": "four"}
    return {"status": "other"}
'''
        # Literal guard must not short-circuit — expected values are not in the code.
        assert check_hardcoded_expected(code, examples) is None
        msg = check_input_equality_chain(code, examples)
        assert msg is not None
        assert 'compares inputs against example values' in msg
        # Combined entry point should surface the same guard.
        assert check_overfit(code, examples) == msg
