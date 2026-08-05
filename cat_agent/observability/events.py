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
        parts = []
        for k, v in self.payload.items():
            text = str(v)
            if len(text) > 80:
                text = text[:77] + '...'
            parts.append(f'{k}={text}')
        return f'{self.event_type} {meta} {" ".join(parts)}'

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
        input: Optional[str] = None,
    ) -> EventEnvelope:
        payload: Dict[str, Any] = {'message_count': message_count, 'lang': lang}
        if input is not None:
            payload['input'] = input
        return EventEnvelope(
            event_type='run.start',
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
        output: Optional[str] = None,
        metrics: Optional[Dict] = None,
    ) -> EventEnvelope:
        payload: Dict[str, Any] = {
            'duration_ms': duration_ms,
            'yield_count': yield_count,
            'metrics': metrics,
        }
        if output is not None:
            payload['output'] = output
        return EventEnvelope(
            event_type='run.end',
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
        input: Optional[str] = None,
    ) -> EventEnvelope:
        payload: Dict[str, Any] = {
            'model': model,
            'message_count': message_count,
            'tool_count': tool_count,
        }
        if input is not None:
            payload['input'] = input
        return EventEnvelope(
            event_type='llm.start',
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
        output: Optional[str] = None,
    ) -> EventEnvelope:
        payload: Dict[str, Any] = {
            'duration_ms': duration_ms,
            'model': model,
            'has_tool_call': has_tool_call,
            'usage': usage,
            'chunk_count': chunk_count,
        }
        if output is not None:
            payload['output'] = output
        return EventEnvelope(
            event_type='llm.end',
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
        output: Optional[str] = None,
    ) -> EventEnvelope:
        payload: Dict[str, Any] = {
            'tool_name': tool_name,
            'duration_ms': duration_ms,
            'success': success,
            'result_chars': result_chars,
        }
        if attempts is not None:
            payload['attempts'] = attempts
        if output is not None:
            payload['output'] = output
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
    def rate_limit_wait(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        scope: str,
        waited_seconds: float,
        tool_name: Optional[str] = None,
    ) -> EventEnvelope:
        payload: Dict[str, Any] = {
            'scope': scope,
            'waited_seconds': waited_seconds,
        }
        if tool_name is not None:
            payload['tool_name'] = tool_name
        return EventEnvelope(
            event_type='rate_limit.wait',
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
    def context_truncated(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        before_tokens: int,
        after_tokens: int,
        max_input_tokens: int,
        dropped_messages: int,
    ) -> EventEnvelope:
        return EventEnvelope(
            event_type='context.truncated',
            timestamp=_utc_timestamp(),
            trace_id=trace_id,
            run_id=run_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            agent_name=agent_name,
            agent_class=agent_class,
            payload={
                'before_tokens': before_tokens,
                'after_tokens': after_tokens,
                'max_input_tokens': max_input_tokens,
                'dropped_messages': dropped_messages,
            },
        )

    @staticmethod
    def synthesis_attempt(
        *,
        trace_id: str,
        run_id: str,
        span_id: str,
        parent_span_id: Optional[str],
        agent_name: Optional[str],
        agent_class: str,
        attempt: int,
        stage: str,
        work_passed: int,
        work_failed: int,
        holdout_passed: int,
        holdout_failed: int,
        duration_ms: float,
        ok: bool,
        error: Optional[str] = None,
    ) -> EventEnvelope:
        payload: Dict[str, Any] = {
            'attempt': attempt,
            'stage': stage,
            'work_passed': work_passed,
            'work_failed': work_failed,
            'holdout_passed': holdout_passed,
            'holdout_failed': holdout_failed,
            'duration_ms': duration_ms,
            'ok': ok,
        }
        if error is not None:
            payload['error'] = error
        return EventEnvelope(
            event_type='synthesis.attempt',
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
