"""Tests for graph visualization handlers (Mermaid + OpenTelemetry)."""

from typing import Iterator, List

import pytest

from cat_agent.agent import Agent
from cat_agent.graph import END, AgentNode, FunctionNode, GraphState, StateGraph
from cat_agent.llm.schema import ASSISTANT, USER, Message
from cat_agent.observability import MermaidExporter, OpenTelemetryHandler


class EchoAgent(Agent):
    def __init__(self, name: str, reply: str):
        super().__init__(name=name, system_message="")
        self._reply = reply

    def _run(self, messages: List[Message], lang: str = "en", **kwargs) -> Iterator[List[Message]]:
        yield [Message(role=ASSISTANT, content=self._reply, name=self.name)]


def _branch_graph(handlers):
    def classify(state: GraphState) -> GraphState:
        state.scratch["math"] = any(c.isdigit() for c in (state.last_message.content or ""))
        return state

    return (StateGraph()
            .add_node(FunctionNode("classify", classify))
            .add_node(AgentNode("math_guy", EchoAgent("math_guy", "42")))
            .add_node(AgentNode("chat", EchoAgent("chat", "hi")))
            .set_entry("classify")
            .add_conditional_edges("classify", lambda s: "math_guy" if s.scratch["math"] else "chat")
            .add_edge("math_guy", END)
            .add_edge("chat", END)
            ).compile(name="MathGuyGraph", handlers=handlers)


def _user(text: str) -> List[Message]:
    return [Message(role=USER, content=text)]


class TestMermaidExporter:

    def test_renders_path_taken(self):
        exp = MermaidExporter()
        app = _branch_graph([exp])
        list(app.run(_user("what is 6 * 7?")))
        out = exp.to_mermaid()
        assert out.startswith("flowchart TD")
        assert "__start__([start]) --> " in out
        assert "classify" in out and "math_guy" in out
        assert "__end__([end])" in out
        # The skipped branch must not appear as an edge target.
        assert "--> chat" not in out

    def test_writes_file(self, tmp_path):
        path = tmp_path / "dag.mmd"
        exp = MermaidExporter(path=str(path))
        app = _branch_graph([exp])
        list(app.run(_user("hello")))  # non-numeric -> chat branch, writes on run.end
        content = path.read_text()
        assert "flowchart TD" in content
        assert "chat" in content

    def test_reset_on_new_run(self):
        exp = MermaidExporter()
        app = _branch_graph([exp])
        list(app.run(_user("123")))
        list(app.run(_user("hello")))
        # After the second (non-math) run, only the chat path should remain.
        out = exp.to_mermaid()
        assert "--> chat" in out
        assert "--> math_guy" not in out


class TestOpenTelemetryHandler:

    def _provider_with_memory_exporter(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        provider = TracerProvider()
        exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return provider, exporter

    def test_emits_nested_spans(self):
        pytest.importorskip("opentelemetry.sdk")
        provider, exporter = self._provider_with_memory_exporter()
        tracer = provider.get_tracer("test")
        app = _branch_graph([OpenTelemetryHandler(tracer=tracer)])
        list(app.run(_user("what is 6 * 7?")))

        spans = exporter.get_finished_spans()
        names = [s.name for s in spans]
        assert any(n.startswith("agent.run") for n in names)
        assert "node classify" in names
        assert "node math_guy" in names

        # node spans carry the edge that was taken
        classify = next(s for s in spans if s.name == "node classify")
        assert classify.attributes.get("cat_agent.graph.next") == "math_guy"

        # node spans are nested under the graph run span (same trace)
        run_span = next(s for s in spans if s.name.startswith("agent.run"))
        node_span = next(s for s in spans if s.name == "node classify")
        assert node_span.context.trace_id == run_span.context.trace_id

    def test_requires_opentelemetry(self, monkeypatch):
        # Simulate the package being absent.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "opentelemetry" or name.startswith("opentelemetry."):
                raise ImportError("no otel")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError):
            OpenTelemetryHandler()
