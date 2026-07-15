"""Tests for tamper-evident audit trail."""

import json
import os
import tempfile

import pytest

from cat_agent.security.audit import (
    GENESIS_HASH,
    AuditLog,
    append_audit_record,
    export_audit_log,
    verify_audit_log,
)


@pytest.fixture(autouse=True)
def _reset_audit_singleton(monkeypatch):
    import cat_agent.security.audit as audit_module

    monkeypatch.setattr(audit_module, '_AUDIT_LOG', None)
    monkeypatch.delenv('CAT_AGENT_AUDIT', raising=False)
    yield
    monkeypatch.setattr(audit_module, '_AUDIT_LOG', None)


class TestAuditChain:

    def test_append_creates_hash_chain(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_AUDIT', '1')
        path = os.path.join(tempfile.mkdtemp(), 'audit.jsonl')
        monkeypatch.setenv('CAT_AGENT_AUDIT_PATH', path)

        append_audit_record('audit.prompt', {'messages': [{'role': 'user', 'content': 'hello'}]})
        append_audit_record('audit.model_output', {'messages': [{'role': 'assistant', 'content': 'hi'}]})

        report = verify_audit_log(path)
        assert report.ok() is True
        assert report.record_count == 2

        with open(path, 'r', encoding='utf-8') as handle:
            first = json.loads(handle.readline())
            second = json.loads(handle.readline())

        assert first['prev_hash'] == GENESIS_HASH
        assert second['prev_hash'] == first['record_hash']

    def test_tampered_record_fails_verification(self, monkeypatch):
        monkeypatch.setenv('CAT_AGENT_AUDIT', '1')
        path = os.path.join(tempfile.mkdtemp(), 'audit.jsonl')
        monkeypatch.setenv('CAT_AGENT_AUDIT_PATH', path)
        append_audit_record('audit.prompt', {'messages': []})

        with open(path, 'r+', encoding='utf-8') as handle:
            record = json.loads(handle.readline())
            record['payload']['tampered'] = True
            handle.seek(0)
            handle.write(json.dumps(record) + '\n')
            handle.truncate()

        report = verify_audit_log(path)
        assert report.ok() is False
        assert 'record_hash mismatch' in (report.first_error or '')

    def test_export_audit_log(self, monkeypatch):
        path = os.path.join(tempfile.mkdtemp(), 'audit.jsonl')
        log = AuditLog(path)
        log.append('audit.file_access', {'path_hash': 'abc', 'operation': 'read'})
        output = os.path.join(tempfile.mkdtemp(), 'export.jsonl')
        count = export_audit_log(path, output)
        assert count == 1
        with open(output, 'r', encoding='utf-8') as handle:
            exported = json.loads(handle.readline())
        assert exported['event_type'] == 'audit.file_access'

    def test_append_noop_when_audit_disabled(self, monkeypatch):
        path = os.path.join(tempfile.mkdtemp(), 'audit.jsonl')
        monkeypatch.setenv('CAT_AGENT_AUDIT_PATH', path)
        append_audit_record('audit.prompt', {'messages': []})
        assert not os.path.exists(path)
