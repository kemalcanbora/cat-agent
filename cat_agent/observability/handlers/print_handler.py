"""Print observability events to stdout."""

from __future__ import annotations

from cat_agent.observability.events import EventEnvelope


class PrintHandler:
    """Print each event using its built-in summary line."""

    def on_event(self, event: EventEnvelope) -> None:
        if event.event_type == 'run.end':
            metrics = (event.payload or {}).get('metrics')
            if isinstance(metrics, dict):
                print(self._run_end_summary(event, metrics))
                return
        print(event.summary())

    @staticmethod
    def _run_end_summary(event: EventEnvelope, metrics: dict) -> str:
        agent = event.agent_name or event.agent_class
        meta = f'trace={event.trace_id} run={event.run_id} agent={agent}'
        duration_ms = event.payload.get('duration_ms')
        yield_count = event.payload.get('yield_count')
        llm_ms = float(metrics.get('llm_ms') or 0)
        completion = int(metrics.get('completion_tokens') or 0)
        tps = completion / (llm_ms / 1000.0) if llm_ms > 0 and completion else None
        tps_s = f' {tps:.1f} tok/s' if tps is not None else ''
        return (
            f'run.end {meta} duration_ms={duration_ms} yield_count={yield_count} '
            f"llm={metrics.get('llm_calls', 0)} tools={metrics.get('tool_calls', 0)} "
            f"tok={metrics.get('prompt_tokens', 0)}/{completion} "
            f"ctx={float(metrics.get('max_context_ratio') or 0):.2f}{tps_s}"
        )
