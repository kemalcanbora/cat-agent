"""Export observability events as OpenTelemetry spans.

Maps Cat-Agent's `run.* / node.* / llm.* / tool.*` event pairs onto OTel spans,
preserving the parent/child nesting carried by each event's `span_id` and
`parent_span_id`. With a configured OTel provider/exporter the result renders as
a trace tree in standard viewers (Jaeger, Grafana Tempo), as an agent graph in
OpenInference-aware UIs such as Arize Phoenix, and with Input/Output/model in
Langfuse (via ``langfuse.*`` + ``gen_ai.*`` attributes).

Requires the OpenTelemetry SDK::

    pip install cat-agent[otel]

Typical setup (exporter configuration is left to the application)::

    from cat_agent.observability import OpenTelemetryHandler, with_langfuse

    @with_langfuse
    def main():
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

        # Langfuse maps these attribute names to UI Input / Output fields.
        if p.get('input') is not None:
            attrs['langfuse.observation.input'] = p['input']
            attrs['gen_ai.prompt'] = p['input']
            attrs['input.value'] = p['input']
        if p.get('output') is not None:
            attrs['langfuse.observation.output'] = p['output']
            attrs['gen_ai.completion'] = p['output']
            attrs['output.value'] = p['output']

        if kind == 'run':
            attrs['gen_ai.operation.name'] = 'invoke_agent'
            # Root span → Langfuse trace-level Input/Output.
            if p.get('input') is not None:
                attrs['langfuse.trace.input'] = p['input']
            if p.get('output') is not None:
                attrs['langfuse.trace.output'] = p['output']
        elif kind == 'node':
            attrs['gen_ai.operation.name'] = 'chain'
            attrs['cat_agent.node'] = p.get('node')
            attrs['cat_agent.node_type'] = p.get('node_type')
            attrs['cat_agent.graph.next'] = p.get('next')
            attrs['cat_agent.graph.step'] = p.get('step')
        elif kind == 'llm':
            attrs['gen_ai.operation.name'] = 'chat'
            model = p.get('model') or None
            if model:
                attrs['gen_ai.request.model'] = model
                attrs['gen_ai.response.model'] = model
                attrs['langfuse.observation.model.name'] = model
            usage = p.get('usage') or {}
            if isinstance(usage, dict):
                if usage.get('prompt_tokens') is not None:
                    attrs['gen_ai.usage.input_tokens'] = usage.get('prompt_tokens')
                if usage.get('completion_tokens') is not None:
                    attrs['gen_ai.usage.output_tokens'] = usage.get('completion_tokens')
        elif kind == 'tool':
            attrs['gen_ai.operation.name'] = 'execute_tool'
            attrs['gen_ai.tool.name'] = p.get('tool_name')
            if p.get('tool_args') is not None and p.get('input') is None:
                # tool.start carries args; map them as observation input.
                attrs['langfuse.observation.input'] = p['tool_args']
                attrs['input.value'] = p['tool_args']
            if 'success' in p:
                attrs['cat_agent.tool.success'] = p.get('success')
        return attrs
