"""Export observability events as OpenTelemetry spans.

Maps Cat-Agent's `run.* / node.* / llm.* / tool.*` event pairs onto OTel spans,
preserving the parent/child nesting carried by each event's `span_id` and
`parent_span_id`. With a configured OTel provider/exporter the result renders as
a trace tree in standard viewers (Jaeger, Grafana Tempo) and as an agent graph
in OpenInference-aware UIs such as Arize Phoenix.

This follows the OTel GenAI semantic conventions where practical
(`gen_ai.*` attributes) and adds `cat_agent.*` attributes for graph nodes.

Requires the OpenTelemetry SDK::

    pip install cat-agent[otel]

Typical setup (exporter configuration is left to the application)::

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)

    from cat_agent.observability import OpenTelemetryHandler
    agent = MyGraph.compile(handlers=[OpenTelemetryHandler()])
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cat_agent.observability.events import EventEnvelope

_GENAI_SYSTEM = 'cat_agent'


class OpenTelemetryHandler:
    """Bridge Cat-Agent observability events to OpenTelemetry spans.

    Args:
        tracer: An OTel `Tracer`. If omitted, one is obtained from the globally
          configured tracer provider via `opentelemetry.trace.get_tracer`.
    """

    def __init__(self, tracer: Optional[Any] = None) -> None:
        try:
            from opentelemetry import trace
        except ImportError as e:
            raise ImportError(
                'OpenTelemetryHandler requires the OpenTelemetry SDK. '
                'Install it with `pip install cat-agent[otel]`.') from e

        self._trace = trace
        self._tracer = tracer or trace.get_tracer('cat_agent')
        # Live spans + the context token used to detach them, keyed by our span_id.
        self._spans: Dict[str, Any] = {}
        self._tokens: Dict[str, Any] = {}

    def on_event(self, event: EventEnvelope) -> None:
        et = event.event_type
        if et.endswith('.start'):
            self._start_span(event)
        elif et.endswith('.end') or et.endswith('.error'):
            self._end_span(event)
        # '*.chunk' events are intentionally ignored to avoid span spam.

    def _start_span(self, event: EventEnvelope) -> None:
        from opentelemetry.trace import set_span_in_context

        parent_span = self._spans.get(event.parent_span_id) if event.parent_span_id else None
        parent_ctx = set_span_in_context(parent_span) if parent_span is not None else None

        span = self._tracer.start_span(self._span_name(event), context=parent_ctx)
        for key, value in self._attributes(event).items():
            if value is not None:
                span.set_attribute(key, value)
        # Keep this span current for any nested spans started under it.
        token = self._trace.use_span(span, end_on_exit=False).__enter__()
        self._spans[event.span_id] = span
        self._tokens[event.span_id] = token

    def _end_span(self, event: EventEnvelope) -> None:
        span = self._spans.pop(event.span_id, None)
        token = self._tokens.pop(event.span_id, None)
        if span is None:
            return
        for key, value in self._attributes(event).items():
            if value is not None:
                span.set_attribute(key, value)
        if event.event_type.endswith('.error'):
            from opentelemetry.trace import Status, StatusCode
            span.set_status(Status(StatusCode.ERROR, event.payload.get('error_message')))
        if token is not None:
            try:
                token.__exit__(None, None, None)
            except Exception:
                pass
        span.end()

    @staticmethod
    def _span_name(event: EventEnvelope) -> str:
        p = event.payload
        agent = event.agent_name or event.agent_class
        kind = event.event_type.split('.')[0]
        if kind == 'run':
            return f'agent.run {agent}'
        if kind == 'node':
            return f'node {p.get("node")}'
        if kind == 'llm':
            return f'llm {p.get("model") or agent}'
        if kind == 'tool':
            return f'tool {p.get("tool_name")}'
        return event.event_type

    @staticmethod
    def _attributes(event: EventEnvelope) -> Dict[str, Any]:
        p = event.payload
        kind = event.event_type.split('.')[0]
        attrs: Dict[str, Any] = {
            'gen_ai.system': _GENAI_SYSTEM,
            'cat_agent.event_type': event.event_type,
            'cat_agent.agent_name': event.agent_name,
            'cat_agent.agent_class': event.agent_class,
        }
        if 'duration_ms' in p:
            attrs['cat_agent.duration_ms'] = p['duration_ms']
        if kind == 'node':
            attrs['gen_ai.operation.name'] = 'chain'
            attrs['cat_agent.node'] = p.get('node')
            attrs['cat_agent.node_type'] = p.get('node_type')
            attrs['cat_agent.graph.next'] = p.get('next')
            attrs['cat_agent.graph.step'] = p.get('step')
        elif kind == 'llm':
            attrs['gen_ai.operation.name'] = 'chat'
            attrs['gen_ai.request.model'] = p.get('model')
            usage = p.get('usage') or {}
            if isinstance(usage, dict):
                if usage.get('prompt_tokens') is not None:
                    attrs['gen_ai.usage.input_tokens'] = usage.get('prompt_tokens')
                if usage.get('completion_tokens') is not None:
                    attrs['gen_ai.usage.output_tokens'] = usage.get('completion_tokens')
        elif kind == 'tool':
            attrs['gen_ai.operation.name'] = 'execute_tool'
            attrs['gen_ai.tool.name'] = p.get('tool_name')
            if 'success' in p:
                attrs['cat_agent.tool.success'] = p.get('success')
        return attrs
