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
        return list(run_handlers)
    if agent_handlers:
        return list(agent_handlers)
    return get_default_handlers()


def emit(event: EventEnvelope) -> None:
    ctx = get_run_context()
    if ctx is None or not ctx.handlers:
        return
    for handler in ctx.handlers:
        handler.on_event(event)
