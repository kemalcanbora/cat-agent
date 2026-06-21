"""Callback-based observability handler."""

from __future__ import annotations

from typing import Callable

from cat_agent.observability.events import EventEnvelope


class CallbackHandler:
    """Invoke a user callback for every observability event."""

    def __init__(self, callback: Callable[[EventEnvelope], None]) -> None:
        self.callback = callback

    def on_event(self, event: EventEnvelope) -> None:
        self.callback(event)
