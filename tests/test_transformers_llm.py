"""Tests for cat_agent.llm.transformers_llm."""

import pytest

from cat_agent.llm.transformers_llm import Transformers


class TestTransformersLlm:

    def test_import_error_message_includes_root_cause(self):
        from cat_agent.llm.transformers_llm import _format_transformers_import_error

        root = AttributeError("module 'lib' has no attribute 'GEN_EMAIL'")
        err = ImportError('wrapper')
        err.__cause__ = root
        message = _format_transformers_import_error(err)
        assert 'GEN_EMAIL' in message
        assert 'pyOpenSSL/cryptography' in message

    def test_init_requires_model(self):
        with pytest.raises(ValueError, match="model"):
            Transformers({})

    def test_init_requires_model_key_in_cfg(self):
        with pytest.raises(ValueError, match="model"):
            Transformers({"model_type": "transformers"})

    def test_registered_in_llm_registry(self):
        from cat_agent.llm.base import LLM_REGISTRY
        assert "transformers" in LLM_REGISTRY
        assert LLM_REGISTRY["transformers"] is Transformers
