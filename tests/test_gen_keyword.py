"""Tests for cat_agent.agents.keygen_strategies.gen_keyword."""

from unittest.mock import MagicMock

from cat_agent.agents.keygen_strategies.gen_keyword import GenKeyword


class TestGenKeyword:

    def test_prompt_templates_have_placeholder(self):
        assert "{user_request}" in GenKeyword.PROMPT_TEMPLATE_EN
        assert "{user_request}" in GenKeyword.PROMPT_TEMPLATE_ZH

    def test_init_merge_stop(self):
        mock_llm = MagicMock()
        gen = GenKeyword(llm=mock_llm)
        assert "Observation:" in gen.extra_generate_cfg.get("stop", [])
