"""Tests for cat_agent.llm.schema."""

import pytest

from cat_agent.llm.schema import (
    ASSISTANT,
    CONTENT,
    FUNCTION,
    ROLE,
    SYSTEM,
    USER,
    BaseModelCompatibleDict,
    ContentItem,
    FunctionCall,
    Message,
)


class TestSchemaConstants:

    def test_role_constants(self):
        assert SYSTEM == "system"
        assert USER == "user"
        assert ASSISTANT == "assistant"
        assert FUNCTION == "function"

    def test_content_role_keys(self):
        assert ROLE == "role"
        assert CONTENT == "content"


class TestFunctionCall:

    def test_init_and_repr(self):
        fc = FunctionCall(name="get_weather", arguments='{"location": "NYC"}')
        assert fc.name == "get_weather"
        assert fc.arguments == '{"location": "NYC"}'
        assert "get_weather" in repr(fc)

    def test_model_dump(self):
        fc = FunctionCall(name="tool", arguments="{}")
        d = fc.model_dump()
        assert d["name"] == "tool"
        assert d["arguments"] == "{}"


class TestContentItem:

    def test_text_only(self):
        item = ContentItem(text="hello")
        assert item.text == "hello"
        assert item.type == "text"
        assert item.value == "hello"
        assert item.get_type_and_value() == ("text", "hello")

    def test_image_only(self):
        item = ContentItem(image="https://x.com/img.png")
        assert item.image == "https://x.com/img.png"
        assert item.type == "image"
        assert item.value == "https://x.com/img.png"

    def test_file_only(self):
        item = ContentItem(file="/path/to/file.pdf")
        assert item.type == "file"
        assert item.value == "/path/to/file.pdf"

    def test_exactly_one_field_required(self):
        with pytest.raises(ValueError, match="Exactly one"):
            ContentItem()
        with pytest.raises(ValueError, match="Exactly one"):
            ContentItem(text="a", image="b")


class TestMessage:

    def test_minimal_user_message(self):
        msg = Message(role=USER, content="Hi")
        assert msg.role == USER
        assert msg.content == "Hi"
        assert msg.name is None
        assert msg.function_call is None

    def test_content_none_becomes_empty_string(self):
        msg = Message(role=USER, content=None)
        assert msg.content == ""

    def test_role_validator_rejects_invalid(self):
        with pytest.raises(ValueError, match="must be one of"):
            Message(role="invalid", content="x")

    def test_message_with_function_call(self):
        fc = FunctionCall(name="tool", arguments="{}")
        msg = Message(role=ASSISTANT, content="", function_call=fc)
        assert msg.function_call.name == "tool"

    def test_message_dict_access(self):
        msg = Message(role=USER, content="x")
        assert msg[ROLE] == USER
        assert msg[CONTENT] == "x"
        msg[CONTENT] = "y"
        assert msg.content == "y"

    def test_message_get(self):
        msg = Message(role=USER, content="x")
        assert msg.get(ROLE) == USER
        assert msg.get("missing", "default") == "default"


class TestBaseModelCompatibleDict:

    def test_getitem_setitem(self):
        class Sub(BaseModelCompatibleDict):
            x: str = ""

        s = Sub(x="a")
        assert s["x"] == "a"
        s["x"] = "b"
        assert s.x == "b"

    def test_get(self):
        class Sub(BaseModelCompatibleDict):
            a: str = ""

        s = Sub(a="v")
        assert s.get("a") == "v"
        assert s.get("b", "default") == "default"
