"""Tests for compile_to_spec."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest

from cat_agent.llm.schema import ASSISTANT, USER, Message
from cat_agent.synthesis.intake.compile import compile_to_spec, sanitise_name
from cat_agent.synthesis.intake.draft import Draft
from cat_agent.synthesis.intake.interview import (
    DEFAULT_ROUNDING_RULE,
    ResolvedDecision,
)
from cat_agent.synthesis.spec import Example


def _mock_llm(responses: List[str]) -> MagicMock:
    llm = MagicMock()
    llm.model = 'mock'
    queue = list(responses)

    def chat(**kwargs):
        text = queue.pop(0) if queue else '{}'
        return iter([[Message(role=ASSISTANT, content=text)]])

    llm.chat = MagicMock(side_effect=chat)
    return llm


def _draft() -> Draft:
    return Draft.from_markdown(
        """\
# KDV Böl
Split VAT.
| gross | rate | result |
|---|---|---|
| 120 | 0.2 | {"net": 100.0, "tax": 20.0} |
| 100 | 0 | {"net": 100.0, "tax": 0.0} |
| 1 | 0.2 | {"net": 0.83, "tax": 0.17} |
""",
        locale='en-IE',
    )


_GOOD_SPEC = """\
{
  "name": "split_vat",
  "description": "Split a VAT-inclusive gross into net and tax.",
  "parameters": {
    "gross": {"type": "number", "description": "Gross amount including VAT"},
    "rate": {"type": "number", "description": "VAT rate as a fraction"}
  },
  "returns": "object"
}
"""


class TestCompile:

    def test_examples_never_mutated_by_model(self):
        payload = """\
{
  "name": "split_vat",
  "description": "Split VAT.",
  "parameters": {
    "gross": {"type": "number", "description": "Gross"},
    "rate": {"type": "number", "description": "Rate"}
  },
  "returns": "object",
  "examples": [
    {"inputs": {"gross": 999}, "expected": {"net": 0}}
  ]
}
"""
        draft = _draft()
        original = [(ex.inputs, ex.expected) for ex in draft.examples]
        result = compile_to_spec(
            draft,
            [],
            'You split VAT. Right?',
            llm=_mock_llm([payload]),
        )
        assert result.ok
        assert [(ex.inputs, ex.expected) for ex in result.spec.examples] == original

    def test_model_added_requires_trace(self):
        payload = """\
{
  "name": "split_vat",
  "description": "Split VAT.",
  "parameters": {
    "gross": {"type": "number", "description": "Gross"},
    "rate": {"type": "number", "description": "Rate"}
  },
  "returns": "object",
  "new_examples": [
    {
      "inputs": {"gross": 0, "rate": 0.2},
      "expected": {"net": 0, "tax": 0},
      "source_answer_excerpt": "zero gross should return zeros"
    }
  ]
}
"""
        history = [
            Message(role=ASSISTANT, content='What about zero?'),
            Message(role=USER, content='zero gross should return zeros'),
        ]
        draft = _draft()
        n = len(draft.examples)
        result = compile_to_spec(
            draft, history, 'ok?', llm=_mock_llm([payload]),
        )
        assert result.ok
        assert len(result.spec.examples) == n + 1
        assert result.model_added_examples

        bad = payload.replace('zero gross should return zeros', 'UNRELATED_PHRASE_XYZ')
        result2 = compile_to_spec(
            draft, history, 'ok?', llm=_mock_llm([bad]),
        )
        assert result2.ok
        assert len(result2.spec.examples) == n

    def test_param_descriptions_present(self):
        draft = _draft()
        result = compile_to_spec(
            draft, [], 'ok', llm=_mock_llm([_GOOD_SPEC]),
        )
        assert result.ok
        assert result.spec.parameters['gross'].description == 'Gross amount including VAT'

    def test_non_ascii_name_transliterates(self):
        name, changed = sanitise_name('ürün_tutarı')
        assert name == 'urun_tutari'
        assert changed is True

    def test_draft_heading_preferred_over_model_name(self):
        """Model invents VATSplitTool; draft H1 'KDV Böl' wins → kdv_bol."""
        payload = """\
{
  "name": "ürün_tutarı",
  "description": "Product amount.",
  "parameters": {
    "gross": {"type": "number", "description": "Gross amount"},
    "rate": {"type": "number", "description": "Rate"}
  },
  "returns": "object"
}
"""
        draft = _draft()
        result = compile_to_spec(
            draft, [], 'ok', llm=_mock_llm([payload]),
        )
        assert result.ok
        assert result.spec.name == 'kdv_bol'

    def test_empty_params_filled_from_draft(self):
        """Model returns empty parameters — draft columns recover (1.5 root cause)."""
        payload = """\
{
  "name": "broken_tool",
  "description": "x",
  "parameters": {},
  "returns": "number"
}
"""
        draft = _draft()
        result = compile_to_spec(
            draft, [], 'ok', llm=_mock_llm([payload]),
        )
        assert result.ok
        assert 'gross' in result.spec.parameters
        assert 'rate' in result.spec.parameters

    def test_fallback_when_model_returns_non_json(self):
        draft = _draft()
        result = compile_to_spec(
            draft,
            [],
            'Split a VAT-inclusive gross into net and tax.',
            llm=_mock_llm(['Sorry, I cannot help with that.']),
        )
        assert result.ok
        assert result.spec is not None
        assert 'gross' in result.spec.parameters
        assert 'rate' in result.spec.parameters
        assert len(result.spec.examples) == len(draft.examples)
        assert result.used_draft_fallback

    def test_corrupted_history_still_compiles_via_draft_only(self):
        """1.5 — draft that compiles cleanly still compiles when history is garbage."""
        draft = _draft()
        clean = compile_to_spec(
            draft, [], 'Split VAT into net and tax.', llm=_mock_llm([_GOOD_SPEC]),
        )
        assert clean.ok

        corrupt_history = [
            Message(role=ASSISTANT, content='???'),
            Message(role=USER, content='unspecified left open not decided'),
            Message(role=ASSISTANT, content='x'),
        ]
        result = compile_to_spec(
            draft,
            corrupt_history,
            'Split VAT into net and tax.',
            llm=_mock_llm(['NOT JSON AT ALL {{{']),
            draft_only=True,
            decisions=[
                ResolvedDecision(
                    topic='rounding',
                    rule=DEFAULT_ROUNDING_RULE,
                    source='assistant_default',
                    raw_answer='skip',
                )
            ],
        )
        assert result.ok
        assert result.spec is not None
        assert 'gross' in result.spec.parameters
        assert 'half' in result.spec.description.lower()
        assert 'unspecified' not in result.spec.description.lower()

    def test_decisions_fold_into_spec(self):
        draft = _draft()
        result = compile_to_spec(
            draft,
            [],
            'You split VAT.',
            llm=_mock_llm([_GOOD_SPEC]),
            decisions=[
                ResolvedDecision(
                    topic='rounding',
                    rule=DEFAULT_ROUNDING_RULE,
                    source='assistant_default',
                    raw_answer='doesnt matter',
                )
            ],
        )
        assert result.ok
        assert 'Resolved decisions' in result.spec.description
        assert 'half' in result.spec.description.lower()

    def test_no_generic_restate_prompt_in_codebase(self):
        import inspect
        import cat_agent.synthesis.intake.compile as mod
        src = inspect.getsource(mod)
        assert 'restate the inputs' not in src.lower()
        assert 'restate the inputs and the result' not in src.lower()

    def test_targeted_question_names_field(self):
        from cat_agent.synthesis.intake.compile import _targeted_question
        q = _targeted_question('parameters', _draft())
        assert 'restate' not in q.lower()
        assert 'input' in q.lower()
