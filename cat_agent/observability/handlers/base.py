"""Handler protocol for observability events."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cat_agent.observability.events import EventEnvelope


@runtime_checkable
class BaseHandler(Protocol):
    def on_event(self, event: EventEnvelope) -> None:
        ...
