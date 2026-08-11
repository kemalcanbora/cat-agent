"""Tests for cat_agent.tools.base."""

import tempfile

import pytest

from cat_agent.tools.base import (
    TOOL_REGISTRY,
    ToolServiceError,
    is_tool_schema,
    list_params_to_json_schema,
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


class TestListParamsToJsonSchema:

    def test_converts_legacy_list_to_openai_object(self):
        legacy = [
            {"name": "q", "type": "string", "description": "query", "required": True},
            {"name": "n", "type": "number", "required": False},
        ]
        out = list_params_to_json_schema(legacy)
        assert out == {
            "type": "object",
            "properties": {
                "q": {"type": "string", "description": "query"},
                "n": {"type": "number"},
            },
            "required": ["q"],
        }
        assert is_tool_schema({"name": "t", "description": "d", "parameters": out})

    def test_empty_list_is_empty_object_schema(self):
        out = list_params_to_json_schema([])
        assert out == {"type": "object", "properties": {}, "required": []}


class TestBaseTool:

    def test_storage_has_name_and_function(self):
        storage = Storage({"storage_root_path": tempfile.mkdtemp()})
        assert storage.name == "storage"
        fn = storage.function
        assert fn["name"] == "storage"
        assert "description" in fn
        assert "parameters" in fn

    def test_function_exports_list_params_as_json_schema(self):
        """Hub tools still declare legacy list params; wire format must be object."""
        from cat_agent.multi_agent.tools import AskAgentTool

        class _Hub:
            pass

        tool = AskAgentTool(_Hub(), "DataGuy")
        assert isinstance(tool.parameters, list)
        fn = tool.function
        assert isinstance(fn["parameters"], dict)
        assert fn["parameters"]["type"] == "object"
        assert "name" in fn["parameters"]["properties"]
        assert "question" in fn["parameters"]["properties"]
        assert set(fn["parameters"]["required"]) == {"name", "question"}
        assert is_tool_schema(fn)

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
        expected = {
            "storage",
            "simple_doc_parser",
            "doc_parser",
            "retrieval",
            "front_page_search",
        }
        found = expected & set(TOOL_REGISTRY.keys())
        assert found == expected, f"Default registry mismatch: missing {expected - found}"
        assert "web_search" not in TOOL_REGISTRY
        assert "image_search" not in TOOL_REGISTRY
        assert "web_extractor" not in TOOL_REGISTRY
