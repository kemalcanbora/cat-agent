"""Observability hooks for Cat-Agent runs."""

from __future__ import annotations

import os

from cat_agent.observability.context import RedactConfig, RunContext, run_context
from cat_agent.observability.emitter import clear_handlers, emit, register_handler, resolve_handlers
from cat_agent.observability.events import AgentEvent, EventEnvelope
from cat_agent.observability.handlers import (
    AuditTrailHandler,
    BaseHandler,
    CallbackHandler,
    LoggingHandler,
    MermaidExporter,
    OpenTelemetryHandler,
    PrintHandler,
)

__all__ = [
    'AgentEvent',
    'AuditTrailHandler',
    'BaseHandler',
    'CallbackHandler',
    'EventEnvelope',
    'LoggingHandler',
    'MermaidExporter',
    'OpenTelemetryHandler',
    'PrintHandler',
    'RedactConfig',
    'RunContext',
    'clear_handlers',
    'emit',
    'register_handler',
    'resolve_handlers',
    'run_context',
]

if os.environ.get('CAT_AGENT_TRACE'):
    register_handler(LoggingHandler(level=os.environ.get('CAT_AGENT_TRACE_LEVEL', 'INFO')))

if os.environ.get('CAT_AGENT_AUDIT', '').strip().lower() in {'1', 'true', 'yes', 'on'}:
    register_handler(AuditTrailHandler())
