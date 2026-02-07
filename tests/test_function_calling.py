"""Tests for cat_agent.llm.function_calling."""

import pytest

from cat_agent.llm.schema import ASSISTANT, FUNCTION, USER, ContentItem, FunctionCall, Message
from cat_agent.llm.function_calling import (
    simulate_response_completion_with_chat,
    validate_num_fncall_results,
)


class TestSimulateResponseCompletionWithChat:

    def test_empty_messages_unchanged(self):
        assert simulate_response_completion_with_chat([]) == []

    def test_no_trailing_assistant_unchanged(self):
        messages = [Message(USER, "Hi"), Message(ASSISTANT, "Hello")]
        # Last is assistant but we need one more to merge
        out = simulate_response_completion_with_chat(messages)
        assert len(out) == 1
        assert out[0].role == USER
        assert "Hi" in str(out[0].content) and "Hello" in str(out[0].content)

    def test_trailing_assistant_merged_into_user_str_content(self):
        messages = [
            Message(USER, "Question?"),
            Message(ASSISTANT, "Answer."),
        ]
        out = simulate_response_completion_with_chat(messages)
        assert len(out) == 1
        assert out[0].role == USER
        assert out[0].content == "Question?\n\nAnswer."

    def test_trailing_assistant_merged_into_user_list_content(self):
        messages = [
            Message(USER, [ContentItem(text="Q?")]),
            Message(ASSISTANT, [ContentItem(text="A.")]),
        ]
        out = simulate_response_completion_with_chat(messages)
        assert len(out) == 1
        assert out[0].role == USER
        assert isinstance(out[0].content, list)
        texts = [c.text for c in out[0].content if getattr(c, "text", None)]
        assert "Q?" in texts and "A." in texts


class TestValidateNumFncallResults:

    def test_matching_calls_and_results_ok(self):
        messages = [
            Message(USER, "hi"),
            Message(ASSISTANT, "", function_call=FunctionCall("f1", "{}")),
            Message(FUNCTION, [ContentItem(text="r1")], name="f1"),
        ]
        validate_num_fncall_results(messages, support_multimodal_input=False)

    def test_mismatch_count_raises(self):
        messages = [
            Message(USER, "hi"),
            Message(ASSISTANT, "", function_call=FunctionCall("f1", "{}")),
            # missing function result
        ]
        with pytest.raises(ValueError, match="number of function results"):
            validate_num_fncall_results(messages, support_multimodal_input=False)

    def test_file_in_result_raises(self):
        messages = [
            Message(USER, "hi"),
            Message(ASSISTANT, "", function_call=FunctionCall("f1", "{}")),
            Message(FUNCTION, [ContentItem(file="/path")], name="f1"),
        ]
        with pytest.raises(ValueError, match="file.*not supported"):
            validate_num_fncall_results(messages, support_multimodal_input=False)

    def test_image_when_not_multimodal_raises(self):
        messages = [
            Message(USER, "hi"),
            Message(ASSISTANT, "", function_call=FunctionCall("f1", "{}")),
            Message(FUNCTION, [ContentItem(image="http://img.png")], name="f1"),
        ]
        with pytest.raises(ValueError, match="does not accept images"):
            validate_num_fncall_results(messages, support_multimodal_input=False)

    def test_image_when_multimodal_ok(self):
        messages = [
            Message(USER, "hi"),
            Message(ASSISTANT, "", function_call=FunctionCall("f1", "{}")),
            Message(FUNCTION, [ContentItem(image="http://img.png")], name="f1"),
        ]
        validate_num_fncall_results(messages, support_multimodal_input=True)

    def test_order_mismatch_raises(self):
        messages = [
            Message(USER, "hi"),
            Message(ASSISTANT, "", function_call=FunctionCall("f1", "{}")),
            Message(ASSISTANT, "", function_call=FunctionCall("f2", "{}")),
            Message(FUNCTION, [ContentItem(text="r2")], name="f2"),
            Message(FUNCTION, [ContentItem(text="r1")], name="f1"),
        ]
        with pytest.raises(ValueError, match="same order|must match"):
            validate_num_fncall_results(messages, support_multimodal_input=False)
