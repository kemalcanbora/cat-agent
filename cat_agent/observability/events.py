"""Structured observability events for agent runs."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


def _utc_timestamp() -> float:
    return time.time()


@dataclass(frozen=True)
class EventEnvelope:
    event_type: str
    timestamp: float
    trace_id: str
    run_id: str
    span_id: str
    parent_span_id: Optional[str]
    agent_name: Optional[str]
    agent_class: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    def summary(self) -> str:
        """Human-readable one-line summary; no manual event parsing needed."""
        agent = self.agent_name or self.agent_class
        meta = f'trace={self.trace_id} run={self.run_id} agent={agent}'
        if not self.payload:
            return f'{self.event_type} {meta}'
        details = ' '.join(f'{k}={v}' for k, v in self.payload.items())
        return f'{self.event_type} {meta} {details}'

    def __str__(self) -> str:
        return self.summary()


class AgentEvent:
    """Factory helpers for typed observability events."""

    @staticmethod
    def run_start(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        message_count: int,
        lang: str,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_type='run.start',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={'message_count': message_count, 'lang': lang},
        )

    @staticmethod
    def run_end(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        duration_ms: float,
        yield_count: int,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_type='run.end',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={'duration_ms': duration_ms, 'yield_count': yield_count},
        )

    @staticmethod
    def run_error(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        duration_ms: float,
        error_type: str,
        error_message: str,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_type='run.error',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={
                'duration_ms': duration_ms,
                'error_type': error_type,
                'error_message': error_message,
            },
        )

    @staticmethod
    def llm_start(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        model: Optional[str],
        message_count: int,
        tool_count: int,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_type='llm.start',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={
                'model': model,
                'message_count': message_count,
                'tool_count': tool_count,
            },
        )

    @staticmethod
    def llm_chunk(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        chunk_index: int,
        message_count: int,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_type='llm.chunk',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={'chunk_index': chunk_index, 'message_count': message_count},
        )

    @staticmethod
    def llm_end(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        duration_ms: float,
        model: Optional[str],
        has_tool_call: bool,
        usage: Optional[Dict[str, int]],
        chunk_count: int,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_type='llm.end',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={
                'duration_ms': duration_ms,
                'model': model,
                'has_tool_call': has_tool_call,
                'usage': usage,
                'chunk_count': chunk_count,
            },
        )

    @staticmethod
    def tool_start(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        tool_name: str,
        tool_args: str,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_type='tool.start',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={'tool_name': tool_name, 'tool_args': tool_args},
        )

    @staticmethod
    def tool_end(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        tool_name: str,
        duration_ms: float,
        success: bool,
        result_chars: int,
        attempts: Optional[int] = None,
    ) -> EventEnvelope:
        payload: Dict[str, Any] = {
            'tool_name': tool_name,
            'duration_ms': duration_ms,
            'success': success,
            'result_chars': result_chars,
        }
        if attempts is not None:
            payload['attempts'] = attempts
        return EventEnvelope(
            event_type='tool.end',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload=payload,
        )

    @staticmethod
    def tool_error(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        tool_name: str,
        duration_ms: float,
        error_type: str,
        error_message: str,
        attempts: Optional[int] = None,
    ) -> EventEnvelope:
        payload: Dict[str, Any] = {
            'tool_name': tool_name,
            'duration_ms': duration_ms,
            'error_type': error_type,
            'error_message': error_message,
        }
        if attempts is not None:
            payload['attempts'] = attempts
        return EventEnvelope(
            event_type='tool.error',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload=payload,
        )

    @staticmethod
    def tool_retry(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        tool_name: str,
        attempt: int,
        max_attempts: int,
        error_type: str,
        error_message: str,
        delay_seconds: float,
    ) -> EventEnvelope:
        """Emitted between attempts; not a span boundary (OTel ignores non start/end/error)."""
        return EventEnvelope(
            event_type='tool.retry',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={
                'tool_name': tool_name,
                'attempt': attempt,
                'max_attempts': max_attempts,
                'error_type': error_type,
                'error_message': error_message,
                'delay_seconds': delay_seconds,
            },
        )

    @staticmethod
    def node_start(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        node: str,
        node_type: str,
        step: int,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_type='node.start',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={'node': node, 'node_type': node_type, 'step': step},
        )

    @staticmethod
    def node_end(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        node: str,
        node_type: str,
        step: int,
        duration_ms: float,
        next_node: str,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_type='node.end',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={
                'node': node,
                'node_type': node_type,
                'step': step,
                'duration_ms': duration_ms,
                'next': next_node,
            },
        )
