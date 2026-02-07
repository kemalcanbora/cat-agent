"""Tests for cat_agent.utils.output_beautify."""

import pytest

from cat_agent.llm.schema import ASSISTANT, FUNCTION
from cat_agent.utils.output_beautify import (
    ANSWER_S,
    THOUGHT_S,
    TOOL_CALL_S,
    TOOL_CALL_E,
    TOOL_RESULT_S,
    typewriter_print,
    multimodal_typewriter_print,
)


class TestOutputBeautify:

    def test_constants(self):
        assert TOOL_CALL_S == "[TOOL_CALL]"
        assert THOUGHT_S == "[THINK]"
        assert ANSWER_S == "[ANSWER]"
        assert TOOL_RESULT_S == "[TOOL_RESPONSE]"
        assert TOOL_CALL_E == ""

    def test_typewriter_print_empty(self):
        out = typewriter_print([], "")
        assert out == ""

    def test_typewriter_print_assistant_content(self):
        msgs = [{"role": ASSISTANT, "content": "Hi there"}]
        out = typewriter_print(msgs, "")
        assert ANSWER_S in out
        assert "Hi there" in out

    def test_typewriter_print_reasoning_content(self):
        msgs = [{"role": ASSISTANT, "reasoning_content": "Let me think.", "content": "Done."}]
        out = typewriter_print(msgs, "")
        assert THOUGHT_S in out
        assert "Let me think." in out
        assert ANSWER_S in out
        assert "Done." in out

    def test_typewriter_print_function_call(self):
        msgs = [
            {"role": ASSISTANT, "function_call": {"name": "get_weather", "arguments": '{"city": "NYC"}'}},
        ]
        out = typewriter_print(msgs, "")
        assert TOOL_CALL_S in out
        assert "get_weather" in out
        assert "NYC" in out

    def test_typewriter_print_function_result(self):
        msgs = [{"role": FUNCTION, "name": "tool1", "content": "result"}]
        out = typewriter_print(msgs, "")
        assert TOOL_RESULT_S in out
        assert "tool1" in out
        assert "result" in out

    def test_typewriter_print_returns_full_text(self):
        msgs = [{"role": ASSISTANT, "content": "Answer"}]
        out = typewriter_print(msgs, "prev")
        assert out.startswith(ANSWER_S)
        assert "Answer" in out

    def test_typewriter_print_unsupported_role_raises(self):
        msgs = [{"role": "user", "content": "Hi"}]
        with pytest.raises(TypeError):
            typewriter_print(msgs, "")

    def test_multimodal_typewriter_print_returns_full_text(self):
        """Without Jupyter/PIL, display is skipped; we still get concatenated text."""
        msgs = [
            {"role": ASSISTANT, "content": "Here is the answer."},
        ]
        out = multimodal_typewriter_print(msgs, "")
        assert ANSWER_S in out
        assert "Here is the answer." in out

    def test_multimodal_typewriter_print_function_result_text(self):
        msgs = [
            {"role": FUNCTION, "name": "tool1", "content": "text result"},
        ]
        out = multimodal_typewriter_print(msgs, "")
        assert TOOL_RESULT_S in out
        assert "tool1" in out
        assert "text result" in out

    def test_multimodal_typewriter_print_function_result_list_content(self):
        msgs = [
            {"role": FUNCTION, "name": "img_tool", "content": [{"text": "Generated image"}, {"image": "http://example.com/img.png"}]},
        ]
        out = multimodal_typewriter_print(msgs, "")
        assert "img_tool" in out
        assert "Generated image" in out

    def test_multimodal_typewriter_print_unsupported_role_raises(self):
        msgs = [{"role": "system", "content": "Sys"}]
        with pytest.raises(TypeError, match="Unsupported message role"):
            multimodal_typewriter_print(msgs, "")
