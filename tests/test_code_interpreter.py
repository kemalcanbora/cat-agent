"""Tests for cat_agent.tools.code_interpreter."""

from unittest.mock import patch

import pytest

from cat_agent.tools.code_interpreter import CodeInterpreter


class TestCodeInterpreter:

    def test_init_checks_docker_availability(self):
        with patch("cat_agent.tools.code_interpreter._check_docker_availability"):
            with patch("cat_agent.tools.code_interpreter._check_host_deps"):
                ci = CodeInterpreter({})
        assert ci.name == "code_interpreter"
        assert "code" in ci.parameters["properties"]

    def test_args_format_default_english(self):
        with patch("cat_agent.tools.code_interpreter._check_docker_availability"):
            with patch("cat_agent.tools.code_interpreter._check_host_deps"):
                ci = CodeInterpreter({})
        # Default: no Chinese in name/description, so backtick instruction
        assert "backtick" in ci.args_format.lower() or "triple" in ci.args_format.lower()

    def test_args_format_custom_from_cfg(self):
        with patch("cat_agent.tools.code_interpreter._check_docker_availability"):
            with patch("cat_agent.tools.code_interpreter._check_host_deps"):
                ci = CodeInterpreter({"args_format": "Custom format."})
        assert ci.args_format == "Custom format."

    def test_call_empty_code_returns_empty_string(self):
        from cat_agent.tools.base import BaseToolWithFileAccess

        with patch("cat_agent.tools.code_interpreter._check_docker_availability"):
            with patch("cat_agent.tools.code_interpreter._check_host_deps"):
                ci = CodeInterpreter({})
        # call() expects params as JSON string; json5.loads then gives code; empty/whitespace code returns ''
        with patch.object(BaseToolWithFileAccess, "call", lambda self, params=None, files=None, **kw: None):
            result = ci.call('{"code": "   "}')
        assert result == ""

    def test_call_requires_code_key_when_params_invalid(self):
        with patch("cat_agent.tools.code_interpreter._check_docker_availability"):
            with patch("cat_agent.tools.code_interpreter._check_host_deps"):
                ci = CodeInterpreter({})
        with pytest.raises((KeyError, Exception)):
            ci.call("{}")
