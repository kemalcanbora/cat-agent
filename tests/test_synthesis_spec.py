"""Tests for ToolSpec validation and holdout splitting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cat_agent.synthesis.spec import (
    Example,
    ParameterSpec,
    ToolSpec,
    load_tool_spec,
    tool_spec_from_dict,
)
from cat_agent.tools.base import TOOL_REGISTRY


def _examples(n: int = 3) -> list:
    return [
        Example(inputs={'x': i}, expected=i + 1, note=f'case {i}')
        for i in range(n)
    ]


class TestToolSpecValidation:

    def test_valid_spec(self):
        spec = ToolSpec(
            name='add_one_synth_test',
            description='Add one',
            parameters={'x': 'integer'},
            returns='integer',
            examples=_examples(3),
        )
        assert spec.registered_name == 'generated_add_one_synth_test'
        assert spec.function_name == 'add_one_synth_test'
        assert isinstance(spec.parameters['x'], ParameterSpec)
        assert spec.parameters['x'].type == 'integer'

    def test_rejects_fewer_than_three_examples(self):
        with pytest.raises(ValueError, match='at least 3'):
            ToolSpec(
                name='too_few_examples_tool',
                description='x',
                parameters={'x': 'int'},
                returns='int',
                examples=_examples(2),
            )

    def test_rejects_invalid_name(self):
        with pytest.raises(ValueError, match='identifier'):
            ToolSpec(
                name='not-valid',
                description='x',
                parameters={'x': 'int'},
                returns='int',
                examples=_examples(),
            )

    def test_rejects_builtin_name(self):
        with pytest.raises(ValueError, match='builtin'):
            ToolSpec(
                name='list',
                description='x',
                parameters={'x': 'int'},
                returns='int',
                examples=_examples(),
            )

    def test_rejects_registry_collision(self):
        existing = next(iter(TOOL_REGISTRY))
        with pytest.raises(ValueError, match='already registered'):
            ToolSpec(
                name=existing,
                description='x',
                parameters={'x': 'int'},
                returns='int',
                examples=_examples(),
            )

    def test_holdout_split_deterministic(self):
        spec = ToolSpec(
            name='holdout_split_tool',
            description='x',
            parameters={'x': 'int'},
            returns='int',
            examples=_examples(5),
            holdout_ratio=0.3,
        )
        a_work, a_hold = spec.split_examples()
        b_work, b_hold = spec.split_examples()
        assert [e.inputs for e in a_work] == [e.inputs for e in b_work]
        assert [e.inputs for e in a_hold] == [e.inputs for e in b_hold]
        assert len(a_hold) >= 1
        assert len(a_work) >= 1
        assert len(a_hold) + len(a_work) == 5

    def test_load_json_spec(self, tmp_path: Path):
        payload = {
            'name': 'load_json_synth_tool',
            'description': 'Add one',
            'parameters': {'x': 'integer'},
            'returns': 'integer',
            'examples': [
                {'inputs': {'x': 1}, 'expected': 2},
                {'inputs': {'x': 2}, 'expected': 3},
                {'inputs': {'x': 3}, 'expected': 4},
            ],
        }
        path = tmp_path / 'spec.json'
        path.write_text(json.dumps(payload), encoding='utf-8')
        spec = load_tool_spec(path)
        assert spec.name == 'load_json_synth_tool'
        assert len(spec.examples) == 3

    def test_parameter_rich_form(self):
        spec = tool_spec_from_dict({
            'name': 'rich_param_synth_tool',
            'description': 'd',
            'parameters': {
                'amount': {
                    'type': 'number',
                    'description': 'Gross inclusive amount',
                },
            },
            'returns': 'object',
            'examples': [
                {'inputs': {'amount': 1.0}, 'expected': {'net': 1.0}},
                {'inputs': {'amount': 2.0}, 'expected': {'net': 2.0}},
                {'inputs': {'amount': 3.0}, 'expected': {'net': 3.0}},
            ],
        })
        assert spec.parameters['amount'].type == 'number'
        assert spec.parameters['amount'].description == 'Gross inclusive amount'

    def test_from_dict(self):
        spec = tool_spec_from_dict({
            'name': 'from_dict_synth_tool',
            'description': 'd',
            'parameters': {'a': 'str'},
            'returns': 'str',
            'examples': [
                {'inputs': {'a': 'x'}, 'expected': 'x'},
                {'inputs': {'a': 'y'}, 'expected': 'y'},
                {'inputs': {'a': 'z'}, 'expected': 'z'},
            ],
        })
        assert spec.parameters['a'].type == 'str'
        assert spec.parameters['a'].description == ''

    def test_missing_param_description_warns(self):
        from unittest.mock import patch

        with patch('cat_agent.synthesis.spec.logger.warning') as warn:
            ToolSpec(
                name='warn_desc_synth_tool',
                description='d',
                parameters={'x': 'str'},
                returns='str',
                examples=[
                    Example(inputs={'x': 'a'}, expected='a'),
                    Example(inputs={'x': 'b'}, expected='b'),
                    Example(inputs={'x': 'c'}, expected='c'),
                ],
            )
        assert warn.called
        joined = ' '.join(str(c) for c in warn.call_args_list)
        assert 'no description' in joined
