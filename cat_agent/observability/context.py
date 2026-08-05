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
    # False so Langfuse / OTel UIs can show Input/Output by default.
    # Set True to replace message bodies with ``<redacted>``.
    redact_messages: bool = False
    max_result_chars: int = 2000


@dataclass
class RunMetrics:
    llm_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_ms: float = 0.0
    tool_ms: float = 0.0
    truncation_events: int = 0
    max_context_ratio: float = 0.0  # highest fill ratio observed (0.0-1.0+)
    usage_available: bool = False  # did any real usage ever arrive
    silent_truncation_warned: bool = False  # once-per-run server-side truncation warn

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_sec(self) -> Optional[float]:
        if self.llm_ms <= 0 or not self.completion_tokens:
            return None
        return self.completion_tokens / (self.llm_ms / 1000.0)

    def __iadd__(self, other: 'RunMetrics') -> 'RunMetrics':
        self.llm_calls += other.llm_calls
        self.tool_calls += other.tool_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.llm_ms += other.llm_ms
        self.tool_ms += other.tool_ms
        self.truncation_events += other.truncation_events
        self.max_context_ratio = max(self.max_context_ratio, other.max_context_ratio)
        self.usage_available = self.usage_available or other.usage_available
        self.silent_truncation_warned = (
            self.silent_truncation_warned or other.silent_truncation_warned
        )
        return self


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
    metrics: RunMetrics = field(default_factory=RunMetrics)


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
        if parent is not None:
            parent.metrics += ctx.metrics
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
