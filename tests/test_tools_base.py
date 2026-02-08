"""Tests for cat_agent.tools.base."""

import tempfile

import pytest

from cat_agent.tools.base import (
    TOOL_REGISTRY,
    ToolServiceError,
    is_tool_schema,
)
from cat_agent.tools.storage import Storage


class TestIsToolSchema:

    def test_valid_schema(self):
        schema = {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
        assert is_tool_schema(schema) is True

    def test_missing_keys_invalid(self):
        assert is_tool_schema({"name": "x", "description": "y"}) is False
        assert is_tool_schema({"name": "x", "parameters": {}}) is False

    def test_parameters_not_object_invalid(self):
        schema = {
            "name": "x",
            "description": "y",
            "parameters": {"type": "string", "properties": {}, "required": []},
        }
        assert is_tool_schema(schema) is False


class TestBaseTool:

    def test_storage_has_name_and_function(self):
        storage = Storage({"storage_root_path": tempfile.mkdtemp()})
        assert storage.name == "storage"
        fn = storage.function
        assert fn["name"] == "storage"
        assert "description" in fn
        assert "parameters" in fn

    def test_verify_json_format_args_required_missing_raises(self):
        storage = Storage({"storage_root_path": tempfile.mkdtemp()})
        with pytest.raises(Exception):  # jsonschema.ValidationError
            storage._verify_json_format_args("{}")

    def test_verify_json_format_args_dict_accepted(self):
        storage = Storage({"storage_root_path": tempfile.mkdtemp()})
        out = storage._verify_json_format_args({"operate": "get", "key": "/x"})
        assert out["operate"] == "get"
        assert out["key"] == "/x"

    def test_tool_service_error(self):
        e = ToolServiceError(message="bad")
        assert "bad" in str(e)
        e2 = ToolServiceError(exception=ValueError("x"))
        assert e2.exception is not None


class TestToolRegistry:

    def test_registry_contains_expected_tools(self):
        expected = {"storage", "simple_doc_parser", "doc_parser", "retrieval", "web_extractor", "web_search", "image_search", "front_page_search"}
        found = expected & set(TOOL_REGISTRY.keys())
        assert len(found) >= 5, f"Expected at least 5 of {expected}, got {found}"
