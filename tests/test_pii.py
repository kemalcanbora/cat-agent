"""Tests for offline PII redaction."""

import pytest

from cat_agent.llm.schema import USER, Message
from cat_agent.security.pii import (
    PII_PLACEHOLDER,
    is_pii_redact_prompts_enabled,
    maybe_redact_for_audit,
    maybe_redact_for_prompt,
    maybe_redact_messages_for_prompt,
    redact_structured_doc,
    redact_text,
)


class TestPiiRedaction:

    def test_redacts_email(self):
        text = 'Contact me at patient@hospital.example for results.'
        assert PII_PLACEHOLDER in redact_text(text)
        assert 'patient@hospital.example' not in redact_text(text)

    def test_preserves_iso_dates(self):
        # Phone regex used to treat YYYY-MM-DD as a phone number.
        text = 'deadline: 2026-09-30 status open; call HORIZON-JU-EUROHPC-2026-PQC-06-01'
        assert '2026-09-30' in redact_text(text)
        assert 'HORIZON-JU-EUROHPC-2026-PQC-06-01' in redact_text(text)

    def test_still_redacts_phones(self):
        text = 'Call +34 93 413 77 16 or +90 532 111 22 33'
        redacted = redact_text(text)
        assert '+34 93 413 77 16' not in redacted
        assert '+90 532 111 22 33' not in redacted
        assert PII_PLACEHOLDER in redacted

    def test_redacts_iban(self):
        text = 'Account TR330006100519786457841326'
        redacted = redact_text(text)
        assert PII_PLACEHOLDER in redacted

    def test_redacts_valid_turkish_tc_id(self):
        # Publicly documented valid-format example number for algorithm testing
        text = 'TC kimlik no 10000000146 bilgisi'
        redacted = redact_text(text)
        assert '10000000146' not in redacted
        assert PII_PLACEHOLDER in redacted

    def test_maybe_redact_messages_for_prompt(self, monkeypatch):
        monkeypatch.delenv('CAT_AGENT_PII_REDACT', raising=False)
        messages = [Message(USER, 'Email me at secret@corp.com')]
        redacted = maybe_redact_messages_for_prompt(messages)
        assert PII_PLACEHOLDER in redacted[0].content
        assert 'secret@corp.com' not in redacted[0].content

    def test_prompt_redaction_disabled(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_PII_REDACT_PROMPTS', '0')
        assert is_pii_redact_prompts_enabled() is False
        text = maybe_redact_for_prompt('secret@corp.com')
        assert text == 'secret@corp.com'

    def test_audit_payload_redaction(self, monkeypatch):
        monkeypatch.delenv('CAT_AGENT_PII_REDACT', raising=False)
        payload = {'email': 'secret@corp.com', 'nested': {'phone': '+90 532 000 00 00'}}
        redacted = maybe_redact_for_audit(payload)
        assert PII_PLACEHOLDER in redacted['email']
        assert PII_PLACEHOLDER in redacted['nested']['phone']

    def test_redact_structured_doc(self, monkeypatch):
        monkeypatch.delenv('CAT_AGENT_PII_REDACT', raising=False)
        doc = [{
            'title': 'Report for secret@corp.com',
            'content': [{'text': 'Patient phone +90 532 111 22 33', 'token': 5}],
        }]
        redacted = redact_structured_doc(doc)
        assert PII_PLACEHOLDER in redacted[0]['title']
        assert PII_PLACEHOLDER in redacted[0]['content'][0]['text']

    def test_master_switch_disables_all(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_PII_REDACT', '0')
        assert maybe_redact_for_prompt('secret@corp.com') == 'secret@corp.com'
        assert maybe_redact_for_audit({'email': 'secret@corp.com'}) == {'email': 'secret@corp.com'}
