"""Print observability events to stdout."""

from __future__ import annotations

from cat_agent.observability.events import EventEnvelope


class PrintHandler:
    """Print each event using its built-in summary line."""

    def on_event(self, event: EventEnvelope) -> None:
        print(event.summary())
