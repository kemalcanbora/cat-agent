"""Dispatch observability events to registered handlers."""

from __future__ import annotations

from typing import List

from cat_agent.observability.context import RunContext, get_run_context
from cat_agent.observability.events import EventEnvelope
from cat_agent.observability.handlers.base import BaseHandler

_default_handlers: List[BaseHandler] = []


def register_handler(handler: BaseHandler) -> None:
    _default_handlers.append(handler)


def clear_handlers() -> None:
    _default_handlers.clear()


def get_default_handlers() -> List[BaseHandler]:
    return list(_default_handlers)


def resolve_handlers(
    agent_handlers: List[BaseHandler] | None,
    run_handlers: List[BaseHandler] | None = None,
) -> List[BaseHandler]:
    if run_handlers is not None:
        handlers = list(run_handlers)
    elif agent_handlers:
        handlers = list(agent_handlers)
    else:
        handlers = get_default_handlers()

    from cat_agent.security.audit import is_audit_enabled

    if is_audit_enabled():
        from cat_agent.observability.handlers.audit_trail import AuditTrailHandler

        audit_handler = AuditTrailHandler()
        if not any(isinstance(handler, AuditTrailHandler) for handler in handlers):
            handlers.append(audit_handler)
    return handlers


def emit(event: EventEnvelope) -> None:
    ctx = get_run_context()
    if ctx is None or not ctx.handlers:
        return
    for handler in ctx.handlers:
        handler.on_event(event)
