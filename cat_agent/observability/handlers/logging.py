"""Bridge observability events to loguru."""

from __future__ import annotations

from cat_agent.log import logger
from cat_agent.observability.events import EventEnvelope


class LoggingHandler:
    """Emit concise observability summaries via loguru."""

    def __init__(self, level: str = 'INFO') -> None:
        self.level = level.upper()

    def on_event(self, event: EventEnvelope) -> None:
        payload = event.payload
        agent = event.agent_name or event.agent_class
        trace = event.trace_id
        run = event.run_id

        if event.event_type == 'run.start':
            logger.log(
                self.level,
                'run.start trace={} run={} agent={} messages={} lang={}',
                trace,
                run,
                agent,
                payload.get('message_count'),
                payload.get('lang'),
            )
        elif event.event_type == 'run.end':
            logger.log(
                self.level,
                'run.end trace={} run={} agent={} duration_ms={} yields={}',
                trace,
                run,
                agent,
                payload.get('duration_ms'),
                payload.get('yield_count'),
            )
        elif event.event_type == 'run.error':
            logger.log(
                self.level,
                'run.error trace={} run={} agent={} error={}: {}',
                trace,
                run,
                agent,
                payload.get('error_type'),
                payload.get('error_message'),
            )
        elif event.event_type == 'llm.start':
            logger.log(
                self.level,
                'llm.start trace={} run={} agent={} model={} messages={} tools={}',
                trace,
                run,
                agent,
                payload.get('model'),
                payload.get('message_count'),
                payload.get('tool_count'),
            )
        elif event.event_type == 'llm.end':
            logger.log(
                self.level,
                'llm.end trace={} run={} agent={} model={} duration_ms={} tool_call={} chunks={}',
                trace,
                run,
                agent,
                payload.get('model'),
                payload.get('duration_ms'),
                payload.get('has_tool_call'),
                payload.get('chunk_count'),
            )
        elif event.event_type == 'tool.start':
            logger.log(
                self.level,
                'tool.start trace={} run={} agent={} tool={}',
                trace,
                run,
                agent,
                payload.get('tool_name'),
            )
        elif event.event_type == 'tool.end':
            logger.log(
                self.level,
                'tool.end trace={} run={} agent={} tool={} duration_ms={} success={} result_chars={}',
                trace,
                run,
                agent,
                payload.get('tool_name'),
                payload.get('duration_ms'),
                payload.get('success'),
                payload.get('result_chars'),
            )
        elif event.event_type == 'tool.error':
            logger.log(
                self.level,
                'tool.error trace={} run={} agent={} tool={} error={}: {}',
                trace,
                run,
                agent,
                payload.get('tool_name'),
                payload.get('error_type'),
                payload.get('error_message'),
            )
        elif event.event_type == 'tool.retry':
            logger.log(
                self.level,
                'tool.retry trace={} run={} agent={} tool={} attempt={}/{} error={}: {} delay={}',
                trace,
                run,
                agent,
                payload.get('tool_name'),
                payload.get('attempt'),
                payload.get('max_attempts'),
                payload.get('error_type'),
                payload.get('error_message'),
                payload.get('delay_seconds'),
            )
        elif event.event_type == 'llm.chunk':
            logger.debug(
                'llm.chunk trace={} run={} agent={} chunk={} messages={}',
                trace,
                run,
                agent,
                payload.get('chunk_index'),
                payload.get('message_count'),
            )
