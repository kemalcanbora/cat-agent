"""Tamper-evident hash-chained audit trail for regulated AI deployments."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from cat_agent.security.pii import maybe_redact_for_audit
from cat_agent.settings import DEFAULT_WORKSPACE

GENESIS_HASH = '0' * 64
_AUDIT_LOCK = threading.Lock()
_AUDIT_LOG: Optional['AuditLog'] = None


class AuditChainError(RuntimeError):
    """Raised when audit chain verification fails."""


@dataclass(frozen=True)
class AuditVerificationReport:
    path: str
    record_count: int
    valid: bool
    first_error: Optional[str] = None

    def ok(self) -> bool:
        return self.valid


def is_audit_enabled() -> bool:
    value = os.getenv('CAT_AGENT_AUDIT', '').strip().lower()
    return value in {'1', 'true', 'yes', 'on'}


def default_audit_path() -> str:
    configured = os.getenv('CAT_AGENT_AUDIT_PATH', '').strip()
    if configured:
        return configured
    return os.path.join(DEFAULT_WORKSPACE, 'storage', 'audit', 'audit.jsonl')


def _canonical_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _sign_record(record_hash: str) -> str:
    try:
        from cat_agent.security.crypto import resolve_encryption_key

        key = resolve_encryption_key(create_if_missing=False)
    except Exception:
        return ''
    return hmac.new(key, record_hash.encode('utf-8'), hashlib.sha256).hexdigest()


class AuditLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    def _last_record(self) -> Tuple[int, str]:
        if not os.path.isfile(self.path):
            return 0, GENESIS_HASH
        last_line = ''
        with open(self.path, 'r', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    last_line = line
        if not last_line:
            return 0, GENESIS_HASH
        record = json.loads(last_line)
        return int(record['sequence']), str(record['record_hash'])

    def append(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        trace_id: Optional[str] = None,
        run_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_class: Optional[str] = None,
    ) -> Dict[str, Any]:
        redacted_payload = maybe_redact_for_audit(payload)
        with _AUDIT_LOCK:
            sequence, prev_hash = self._last_record()
            body = {
                'sequence': sequence + 1,
                'timestamp': time.time(),
                'event_type': event_type,
                'trace_id': trace_id,
                'run_id': run_id,
                'agent_name': agent_name,
                'agent_class': agent_class,
                'payload': redacted_payload,
            }
            record_hash = _sha256_hex(_canonical_json(body))
            entry = {
                **body,
                'prev_hash': prev_hash,
                'record_hash': record_hash,
                'signature': _sign_record(record_hash),
            }
            with open(self.path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            return entry


def get_audit_log() -> AuditLog:
    global _AUDIT_LOG
    if _AUDIT_LOG is None:
        _AUDIT_LOG = AuditLog(default_audit_path())
    return _AUDIT_LOG


def append_audit_record(
    event_type: str,
    payload: Dict[str, Any],
    *,
    trace_id: Optional[str] = None,
    run_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    agent_class: Optional[str] = None,
) -> None:
    if not is_audit_enabled():
        return
    get_audit_log().append(
        event_type,
        payload,
        trace_id=trace_id,
        run_id=run_id,
        agent_name=agent_name,
        agent_class=agent_class,
    )


def verify_audit_log(path: str) -> AuditVerificationReport:
    if not os.path.isfile(path):
        return AuditVerificationReport(path=path, record_count=0, valid=True)

    prev_hash = GENESIS_HASH
    count = 0
    for line_number, line in enumerate(_iter_nonempty_lines(path), start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as error:
            return AuditVerificationReport(
                path=path,
                record_count=count,
                valid=False,
                first_error=f'Line {line_number}: invalid JSON ({error})',
            )

        count += 1
        if entry.get('prev_hash') != prev_hash:
            return AuditVerificationReport(
                path=path,
                record_count=count,
                valid=False,
                first_error=f'Line {line_number}: prev_hash mismatch (tampered chain)',
            )

        body = {
            'sequence': entry['sequence'],
            'timestamp': entry['timestamp'],
            'event_type': entry['event_type'],
            'trace_id': entry.get('trace_id'),
            'run_id': entry.get('run_id'),
            'agent_name': entry.get('agent_name'),
            'agent_class': entry.get('agent_class'),
            'payload': entry.get('payload', {}),
        }
        expected_hash = _sha256_hex(_canonical_json(body))
        if entry.get('record_hash') != expected_hash:
            return AuditVerificationReport(
                path=path,
                record_count=count,
                valid=False,
                first_error=f'Line {line_number}: record_hash mismatch (tampered payload)',
            )

        signature = entry.get('signature', '')
        if signature:
            expected_signature = _sign_record(expected_hash)
            if not hmac.compare_digest(signature, expected_signature):
                return AuditVerificationReport(
                    path=path,
                    record_count=count,
                    valid=False,
                    first_error=f'Line {line_number}: signature mismatch',
                )

        prev_hash = expected_hash

    return AuditVerificationReport(path=path, record_count=count, valid=True)


def export_audit_log(path: str, output_path: str) -> int:
    count = 0
    with open(output_path, 'w', encoding='utf-8') as out_handle:
        for line in _iter_nonempty_lines(path):
            out_handle.write(line + '\n')
            count += 1
    return count


def _iter_nonempty_lines(path: str) -> Iterable[str]:
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield stripped
