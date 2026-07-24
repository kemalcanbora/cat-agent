"""Tests for cat_agent.tools.decorator.@tool."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

import pytest
from pydantic import BaseModel, Field

from cat_agent.tools import TOOL_REGISTRY, BaseTool, ToolArgumentError, tool
from cat_agent.tools.base import is_tool_schema, register_tool


class Address(BaseModel):
    city: str = Field(description='City name')
    zip_code: str = Field(description='Postal code')


def test_basic_types_schema_and_call():
    @tool(allow_overwrite=True)
    def echo_basic(name: str, age: int, score: float, active: bool) -> dict:
        """Echo basic typed values.

        Args:
            name: Person name
            age: Age in years
            score: Numeric score
            active: Whether active
        """
        return {'name': name, 'age': age, 'score': score, 'active': active}

    assert isinstance(echo_basic, BaseTool)
    assert is_tool_schema(echo_basic.function)
    props = echo_basic.parameters['properties']
    assert props['name']['type'] == 'string'
    assert props['age']['type'] == 'integer'
    assert props['score']['type'] == 'number'
    assert props['active']['type'] == 'boolean'
    assert set(echo_basic.parameters['required']) == {'name', 'age', 'score', 'active'}

    out = echo_basic.call('{"name": "Ada", "age": 36, "score": 9.5, "active": true}')
    assert out == {'name': 'Ada', 'age': 36, 'score': 9.5, 'active': True}


def test_literal_becomes_string_enum():
    @tool(allow_overwrite=True)
    def set_unit(unit: Literal['celsius', 'fahrenheit']) -> str:
        """Set temperature unit.

        Args:
            unit: Temperature unit
        """
        return unit

    prop = set_unit.parameters['properties']['unit']
    assert prop['type'] == 'string'
    assert prop['enum'] == ['celsius', 'fahrenheit']
    assert set_unit.call({'unit': 'celsius'}) == 'celsius'


def test_optional_and_defaults():
    @tool(allow_overwrite=True)
    def greet(name: str, title: Optional[str] = None, excited: bool = False) -> str:
        """Greet someone.

        Args:
            name: Who to greet
            title: Optional title
            excited: Add emphasis
        """
        prefix = f'{title} ' if title else ''
        mark = '!' if excited else '.'
        return f'Hello {prefix}{name}{mark}'

    assert 'name' in greet.parameters['required']
    assert 'title' not in greet.parameters['required']
    assert 'excited' not in greet.parameters['required']
    assert greet.parameters['properties']['title']['type'] == 'string'

    assert greet.call({'name': 'Ada'}) == 'Hello Ada.'
    assert greet.call({'name': 'Ada', 'title': 'Dr', 'excited': True}) == 'Hello Dr Ada!'


def test_nested_pydantic_model_single_param():
    @tool(allow_overwrite=True)
    def save_address(address: Address) -> str:
        """Save a postal address.

        Args:
            address: Address payload
        """
        return f"{address.city}/{address.zip_code}"

    schema = save_address.parameters
    assert schema['type'] == 'object'
    assert 'city' in schema['properties']
    assert 'zip_code' in schema['properties']
    assert set(schema['required']) == {'city', 'zip_code'}

    result = save_address.call('{"city": "Berlin", "zip_code": "10115"}')
    assert result == 'Berlin/10115'


def test_list_and_dict_types():
    @tool(allow_overwrite=True)
    def summarize(numbers: List[float], meta: Dict[str, str]) -> dict:
        """Summarize numbers with metadata.

        Args:
            numbers: Values to sum
            meta: Extra metadata
        """
        return {'total': sum(numbers), 'meta': meta}

    props = summarize.parameters['properties']
    assert props['numbers']['type'] == 'array'
    assert props['numbers']['items']['type'] == 'number'
    assert props['meta']['type'] == 'object'

    out = summarize.call({'numbers': ['1', 2.5], 'meta': {'src': 'test'}})
    assert out == {'total': 3.5, 'meta': {'src': 'test'}}


def test_missing_required_arg_raises_tool_argument_error():
    @tool(allow_overwrite=True)
    def needs_both(a: int, b: int) -> int:
        """Add two ints.

        Args:
            a: First
            b: Second
        """
        return a + b

    with pytest.raises(ToolArgumentError) as exc_info:
        needs_both.call({'a': 1})
    err = exc_info.value
    assert err.tool_name == 'needs_both'
    assert err.param_name == 'b'
    assert 'required' in str(err).lower()


def test_type_coercion_from_string():
    @tool(allow_overwrite=True)
    def coerce_demo(count: int, ratio: float, flag: bool) -> dict:
        """Coerce stringy LLM args.

        Args:
            count: Integer count
            ratio: Float ratio
            flag: Boolean flag
        """
        return {'count': count, 'ratio': ratio, 'flag': flag}

    out = coerce_demo.call('{"count": "3", "ratio": "1.5", "flag": "true"}')
    assert out == {'count': 3, 'ratio': 1.5, 'flag': True}


def test_direct_python_invocation():
    @tool(allow_overwrite=True)
    def sum_two_number(a: float, b: float) -> float:
        """Adds two numbers.

        Args:
            a: First number
            b: Second number
        """
        return a + b

    assert sum_two_number(2, 3) == 5.0
    assert sum_two_number.call('{"a": 2, "b": 3}') == 5.0


def test_name_override():
    @tool(name='add_numbers', allow_overwrite=True)
    def _hidden(a: int, b: int) -> int:
        """Add integers.

        Args:
            a: First
            b: Second
        """
        return a + b

    assert _hidden.name == 'add_numbers'
    assert 'add_numbers' in TOOL_REGISTRY
    assert _hidden.call({'a': 1, 'b': 2}) == 3


def test_schema_equivalence_with_handwritten_class():
    @register_tool('sum_two_number_class', allow_overwrite=True)
    class SumTwoNumber(BaseTool):
        description = 'Adds two numbers.'
        parameters = {
            'type': 'object',
            'properties': {
                'a': {'description': 'First number', 'type': 'number'},
                'b': {'description': 'Second number', 'type': 'number'},
            },
            'required': ['a', 'b'],
        }

        def call(self, params, **kwargs):
            params = self._verify_json_format_args(params)
            return float(params['a']) + float(params['b'])

    @tool(name='sum_two_number_decorated', allow_overwrite=True)
    def sum_two_number(a: float, b: float) -> float:
        """Adds two numbers.

        Args:
            a: First number
            b: Second number
        """
        return a + b

    handwritten = SumTwoNumber()
    assert sum_two_number.parameters == handwritten.parameters
    assert sum_two_number.description == handwritten.description
    assert is_tool_schema(sum_two_number.function)


def test_async_function_preserved_on_direct_call():
    @tool(allow_overwrite=True)
    async def async_add(a: int, b: int) -> int:
        """Add asynchronously.

        Args:
            a: First
            b: Second
        """
        return a + b

    import asyncio

    assert asyncio.iscoroutinefunction(async_add.__wrapped__)
    assert asyncio.run(async_add(1, 2)) == 3
    assert async_add.call({'a': 1, 'b': 2}) == 3
    assert asyncio.run(async_add.acall({'a': 1, 'b': 2})) == 3
    assert asyncio.iscoroutinefunction(type(async_add).acall)


def test_bare_tool_and_tool_with_parens():
    @tool
    def bare(x: int) -> int:
        """Identity.

        Args:
            x: Value
        """
        return x

    @tool()
    def with_parens(x: int) -> int:
        """Identity too.

        Args:
            x: Value
        """
        return x

    assert bare(4) == 4
    assert with_parens(5) == 5
    assert bare.name == 'bare'
    assert with_parens.name == 'with_parens'
