"""Tests for cat_agent.llm.fncall_prompts.nous_fncall_prompt."""

import pytest

from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, ContentItem, FunctionCall, Message
from cat_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    FN_CALL_TEMPLATE,
    remove_incomplete_special_tokens as nous_remove_incomplete,
    extract_fn,
)


class TestNousFnCallPromptConstants:

    def test_fn_call_template_has_placeholders(self):
        assert "{tool_descs}" in FN_CALL_TEMPLATE
        assert "<tools>" in FN_CALL_TEMPLATE
        assert "<tool_call>" in FN_CALL_TEMPLATE


class TestNousRemoveIncompleteSpecialTokens:

    def test_exact_special_token_returns_empty(self):
        assert nous_remove_incomplete('<tool_call>\n{"name": "') == ''

    def test_other_text_unchanged(self):
        assert nous_remove_incomplete('hello') == 'hello'


class TestExtractFn:

    def test_extracts_name_and_args(self):
        text = '{"name": "my_tool", "arguments": {"x": 1}}'
        name, args = extract_fn(text)
        assert name == "my_tool"
        assert "x" in args

    def test_missing_name_returns_empty_strings(self):
        name, args = extract_fn("nothing here")
        assert name == ''
        assert args == ''


class TestNousFnCallPromptPreprocess:

    def test_system_and_user_passthrough(self):
        messages = [
            Message(SYSTEM, [ContentItem(text="System")]),
            Message(USER, [ContentItem(text="User")]),
        ]
        functions = [{"name": "f", "name_for_model": "f", "description": "d", "parameters": {}}]
        out = NousFnCallPrompt().preprocess_fncall_messages(messages, functions=functions, lang="en")
        assert len(out) >= 2
        assert out[0].role == SYSTEM
        assert any("System" in (c.text or "") for c in out[0].content)
        assert out[1].role == USER
        assert any("User" in (c.text or "") for c in out[1].content)
        assert "<tools>" in (out[0].content[-1].text if out[0].content else "")

    def test_assistant_with_function_call_converted_to_plaintext(self):
        messages = [
            Message(USER, [ContentItem(text="Call f")]),
            Message(ASSISTANT, [], function_call=FunctionCall(name="f", arguments='{"a":1}')),
        ]
        functions = [{"name": "f", "name_for_model": "f", "description": "d", "parameters": {}}]
        out = NousFnCallPrompt().preprocess_fncall_messages(messages, functions=functions, lang="en")
        text = "".join(c.text or "" for m in out for c in (m.content or []))
        assert "<tool_call>" in text
        assert "f" in text
        assert "a" in text or "1" in text

    def test_function_choice_not_auto_raises(self):
        with pytest.raises(NotImplementedError):
            NousFnCallPrompt().preprocess_fncall_messages(
                messages=[Message(USER, "Hi")],
                functions=[],
                lang="en",
                function_choice="my_tool",
            )


class TestNousFnCallPromptPostprocess:

    def test_system_user_passthrough(self):
        messages = [
            Message(SYSTEM, [ContentItem(text="Sys")]),
            Message(USER, [ContentItem(text="User")]),
        ]
        out = NousFnCallPrompt().postprocess_fncall_messages(messages)
        assert len(out) == 2
        assert out[0].role == SYSTEM
        assert out[1].role == USER

    def test_plaintext_tool_call_parsed_to_function_call(self):
        messages = [
            Message(USER, [ContentItem(text="Hi")]),
            Message(ASSISTANT, [ContentItem(text='<tool_call>\n{"name": "my_tool", "arguments": {"x": 1}}\n</tool_call>')]),
        ]
        out = NousFnCallPrompt().postprocess_fncall_messages(messages)
        fn_msgs = [m for m in out if m.function_call]
        assert len(fn_msgs) == 1
        assert fn_msgs[0].function_call.name == "my_tool"
        assert "x" in fn_msgs[0].function_call.arguments

    def test_function_choice_not_auto_raises(self):
        with pytest.raises(NotImplementedError):
            NousFnCallPrompt().postprocess_fncall_messages(
                messages=[Message(USER, "Hi")],
                function_choice="x",
            )
