"""Tests for cat_agent.llm.fncall_prompts.base_fncall_prompt."""

import pytest

from cat_agent.llm.schema import USER, Message
from cat_agent.llm.fncall_prompts.base_fncall_prompt import BaseFnCallPrompt


class TestBaseFnCallPrompt:

    def test_preprocess_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            BaseFnCallPrompt.preprocess_fncall_messages(
                messages=[Message(USER, "Hi")],
                functions=[],
                lang="en",
            )

    def test_postprocess_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            BaseFnCallPrompt.postprocess_fncall_messages(messages=[Message(USER, "Hi")])

    def test_preprocess_function_choice_none_asserts(self):
        with pytest.raises(AssertionError):
            BaseFnCallPrompt.preprocess_fncall_messages(
                messages=[Message(USER, "Hi")],
                functions=[],
                lang="en",
                function_choice="none",
            )

    def test_format_plaintext_train_samples_raises_via_preprocess(self):
        base = BaseFnCallPrompt()
        with pytest.raises(NotImplementedError):
            base.format_plaintext_train_samples(
                messages=[Message(USER, "Hi")],
                functions=[],
                lang="en",
            )
