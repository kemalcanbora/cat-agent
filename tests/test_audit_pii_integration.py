"""Integration tests for audit + PII in agent runs."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.agent import BasicAgent
from cat_agent.llm.schema import ASSISTANT, USER, Message


@pytest.fixture(autouse=True)
def _reset_audit(monkeypatch):
    import cat_agent.security.audit as audit_module

    monkeypatch.setattr(audit_module, '_AUDIT_LOG', None)
    yield
    monkeypatch.setattr(audit_module, '_AUDIT_LOG', None)


class TestAuditAgentIntegration:

    def test_agent_run_writes_audit_prompt_and_output(self, monkeypatch):
        audit_path = os.path.join(tempfile.mkdtemp(), 'audit.jsonl')
        monkeypatch.setenv('CAT_AGENT_AUDIT', '1')
        monkeypatch.setenv('CAT_AGENT_AUDIT_PATH', audit_path)
        monkeypatch.setenv('CAT_AGENT_PII_REDACT', '0')

        llm = MagicMock()
        llm.model = 'test-model'
        llm.chat.return_value = iter([
            [Message(role=ASSISTANT, content='Answer with no PII')],
        ])
        agent = BasicAgent(llm=llm, handlers=[])
        list(agent.run([Message(role=USER, content='Question')]))

        with open(audit_path, 'r', encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle if line.strip()]

        event_types = [record['event_type'] for record in records]
        assert 'audit.prompt' in event_types
        assert 'audit.model_output' in event_types

    def test_prompt_pii_redacted_before_llm_call(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_PII_REDACT', '1')
        monkeypatch.delenv('CAT_AGENT_AUDIT', raising=False)

        llm = MagicMock()
        llm.model = 'test-model'
        llm.chat.return_value = iter([[Message(role=ASSISTANT, content='ok')]])

        agent = BasicAgent(llm=llm, handlers=[])
        list(agent.run([Message(role=USER, content='Email secret@corp.com')]))

        sent_messages = llm.chat.call_args.kwargs['messages']
        assert 'secret@corp.com' not in sent_messages[0].content
        assert '[PII]' in sent_messages[0].content
