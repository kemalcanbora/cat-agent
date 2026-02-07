"""Tests for cat_agent.llm.fncall_prompts.qwen_fncall_prompt."""

from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, ContentItem, FunctionCall, Message
from cat_agent.llm.fncall_prompts.qwen_fncall_prompt import (
    QwenFnCallPrompt,
    FN_NAME,
    FN_ARGS,
    FN_RESULT,
    FN_EXIT,
    get_function_description,
    remove_incomplete_special_tokens as qwen_remove_incomplete,
    remove_trailing_comment_of_fn_args,
)


class TestQwenFnCallPromptConstants:

    def test_special_tokens_defined(self):
        assert "✿" in FN_NAME
        assert "FUNCTION" in FN_NAME or "function" in FN_NAME.lower()
        assert "✿" in FN_ARGS
        assert "✿" in FN_RESULT
        assert "✿" in FN_EXIT


class TestGetFunctionDescription:

    def test_formats_with_name_and_description(self):
        func = {
            "name": "search",
            "name_for_human": "Search",
            "name_for_model": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        en = get_function_description(func, lang="en")
        assert "search" in en
        assert "Search the web" in en
        zh = get_function_description(func, lang="zh")
        assert "search" in zh

    def test_code_interpreter_has_code_instructions(self):
        func = {
            "name": "code_interpreter",
            "name_for_human": "Code",
            "name_for_model": "code_interpreter",
            "description": "Run code",
            "parameters": {},
        }
        en = get_function_description(func, lang="en")
        assert "code" in en.lower() or "backtick" in en.lower()


class TestQwenRemoveIncompleteSpecialTokens:

    def test_strips_trailing_special_token(self):
        assert qwen_remove_incomplete("hello" + FN_NAME) == "hello"
        assert qwen_remove_incomplete("hello" + FN_ARGS) == "hello"

    def test_unchanged_when_no_trailing_special(self):
        text = "hello world"
        assert qwen_remove_incomplete(text) == "hello world"


class TestRemoveTrailingCommentOfFnArgs:

    def test_keeps_valid_json(self):
        s = '{"a": 1}'
        assert remove_trailing_comment_of_fn_args(s) == s

    def test_strips_after_closing_brace(self):
        s = '{"a": 1} <!-- comment -->'
        assert remove_trailing_comment_of_fn_args(s) == '{"a": 1}'

    def test_code_block_strips_after_last_backticks(self):
        s = '```\ncode\n```'
        out = remove_trailing_comment_of_fn_args(s)
        assert "```" in out


class TestQwenFnCallPromptPreprocess:

    def test_system_user_passthrough_and_tool_system_added(self):
        messages = [
            Message(SYSTEM, [ContentItem(text="System")]),
            Message(USER, [ContentItem(text="User")]),
        ]
        functions = [{"name": "f", "name_for_model": "f", "description": "d", "parameters": {}}]
        out = QwenFnCallPrompt.preprocess_fncall_messages(messages, functions=functions, lang="en")
        assert out[0].role == SYSTEM
        assert out[1].role == USER
        full_sys = "".join(c.text or "" for c in out[0].content)
        assert "System" in full_sys
        assert FN_NAME in full_sys or "Tools" in full_sys

    def test_assistant_function_call_converted_to_plaintext(self):
        messages = [
            Message(USER, [ContentItem(text="Call f")]),
            Message(ASSISTANT, [], function_call=FunctionCall(name="f", arguments='{"a":1}')),
        ]
        functions = [{"name": "f", "name_for_model": "f", "description": "d", "parameters": {}}]
        out = QwenFnCallPrompt.preprocess_fncall_messages(messages, functions=functions, lang="en")
        text = "".join(c.text or "" for m in out for c in (m.content or []))
        assert FN_NAME in text
        assert FN_ARGS in text
        assert "f" in text


class TestQwenFnCallPromptPostprocess:

    def test_system_user_passthrough(self):
        messages = [
            Message(SYSTEM, [ContentItem(text="Sys")]),
            Message(USER, [ContentItem(text="User")]),
        ]
        out = QwenFnCallPrompt.postprocess_fncall_messages(messages)
        assert len(out) == 2
        assert out[0].role == SYSTEM
        assert out[1].role == USER

    def test_plaintext_function_format_parsed_to_function_call(self):
        # Format: ✿FUNCTION✿: name\n✿ARGS✿: args
        messages = [
            Message(USER, [ContentItem(text="Hi")]),
            Message(ASSISTANT, [
                ContentItem(text=f"{FN_NAME}: my_tool\n{FN_ARGS}: {{\"x\": 1}}\n"),
            ]),
        ]
        out = QwenFnCallPrompt.postprocess_fncall_messages(messages)
        fn_msgs = [m for m in out if m.function_call]
        assert len(fn_msgs) >= 1
        assert fn_msgs[0].function_call.name == "my_tool"
        assert "x" in fn_msgs[0].function_call.arguments
