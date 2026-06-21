"""Run-scoped context for observability events."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, List, Optional

if TYPE_CHECKING:
    from cat_agent.observability.handlers.base import BaseHandler


def new_id(prefix: str = '') -> str:
    value = uuid.uuid4().hex[:12]
    return f'{prefix}{value}' if prefix else value


@dataclass
class RedactConfig:
    redact_tool_args: bool = False
    redact_messages: bool = True
    max_result_chars: int = 2000


@dataclass
class RunContext:
    trace_id: str
    run_id: str
    span_id: str
    parent_span_id: Optional[str]
    agent_name: Optional[str]
    agent_class: str
    handlers: List['BaseHandler'] = field(default_factory=list)
    redact: RedactConfig = field(default_factory=RedactConfig)
    emit_stream_chunks: bool = False


_current_run: ContextVar[Optional[RunContext]] = ContextVar('cat_agent_run_context', default=None)


def get_run_context() -> Optional[RunContext]:
    return _current_run.get()


@contextmanager
def run_context(
    *,
    agent_name: Optional[str],
    agent_class: str,
    handlers: List['BaseHandler'],
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    redact: Optional[RedactConfig] = None,
    emit_stream_chunks: bool = False,
) -> Iterator[RunContext]:
    parent = _current_run.get()
    ctx = RunContext(
        trace_id=trace_id or (parent.trace_id if parent else new_id('trace-')),
        run_id=new_id('run-'),
        span_id=new_id('span-'),
        parent_span_id=parent_span_id or (parent.span_id if parent else None),
        agent_name=agent_name,
        agent_class=agent_class,
        handlers=handlers,
        redact=redact or RedactConfig(),
        emit_stream_chunks=emit_stream_chunks,
    )
    token: Token = _current_run.set(ctx)
    try:
        yield ctx
    finally:
        _current_run.reset(token)


@contextmanager
def child_span() -> Iterator[str]:
    """Open a child span for LLM/tool calls within the current run."""
    parent = _current_run.get()
    if parent is None:
        yield ''
        return
    span_id = new_id('span-')
    previous_span_id = parent.span_id
    parent.span_id = span_id
    previous_parent_span_id = parent.parent_span_id
    parent.parent_span_id = previous_span_id
    try:
        yield span_id
    finally:
        parent.span_id = previous_span_id
        parent.parent_span_id = previous_parent_span_id
