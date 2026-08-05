"""Tests for observability I/O formatting helpers."""

from cat_agent.llm.schema import ASSISTANT, FUNCTION, FunctionCall, Message
from cat_agent.observability.helpers import format_llm_obs_output, format_run_obs_output


def test_format_llm_obs_output_tool_call():
    messages = [
        Message(
            role=ASSISTANT,
            content='',
            function_call=FunctionCall(name='earth_rotation_constants', arguments='{}'),
        ),
    ]
    out = format_llm_obs_output(messages)
    assert out is not None
    assert 'tool_call earth_rotation_constants' in out
    assert out != ''


def test_format_llm_obs_output_text():
    messages = [Message(role=ASSISTANT, content='Hello world')]
    assert format_llm_obs_output(messages) == 'Hello world'


def test_format_run_obs_output_includes_tool_result():
    messages = [
        Message(
            role=ASSISTANT,
            name='DataGuy',
            content='',
            function_call=FunctionCall(name='earth_rotation_constants', arguments='{}'),
        ),
        Message(role=FUNCTION, name='earth_rotation_constants', content='R = 6378137 m'),
    ]
    out = format_run_obs_output(messages)
    assert out is not None
    assert 'DataGuy' in out
    assert 'earth_rotation_constants' in out
    assert '6378137' in out
