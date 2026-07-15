"""Audit trail handler that mirrors observability events into the hash chain."""

from __future__ import annotations

from cat_agent.observability.events import EventEnvelope
from cat_agent.security.audit import append_audit_record, is_audit_enabled


class AuditTrailHandler:
    """Append observability events to the tamper-evident audit log."""

    def on_event(self, event: EventEnvelope) -> None:
        if not is_audit_enabled():
            return
        append_audit_record(
            f'audit.{event.event_type}',
            {
                'span_id': event.span_id,
                'parent_span_id': event.parent_span_id,
                'payload': event.payload,
            },
            trace_id=event.trace_id,
            run_id=event.run_id,
            agent_name=event.agent_name,
            agent_class=event.agent_class,
        )
