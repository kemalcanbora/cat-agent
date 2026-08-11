"""Tests for cat_agent.graph (graph/DAG workflow engine)."""

from typing import Iterator, List

import pytest

from cat_agent.agent import Agent
from cat_agent.graph import (
    END,
    AgentNode,
    FunctionNode,
    GraphAgent,
    GraphState,
    StateGraph,
)
from cat_agent.llm.schema import ASSISTANT, USER, Message
from cat_agent.observability.events import EventEnvelope


class _CollectingHandler:
    def __init__(self):
        self.events = []

    def on_event(self, event: EventEnvelope) -> None:
        self.events.append(event)


class EchoAgent(Agent):
    """A minimal agent that replies with a fixed tag, no LLM required."""

    def __init__(self, name: str, reply: str):
        super().__init__(name=name, system_message="")
        self._reply = reply

    def _run(self, messages: List[Message], lang: str = "en", **kwargs) -> Iterator[List[Message]]:
        yield [Message(role=ASSISTANT, content=self._reply, name=self.name)]


def _user(text: str) -> List[Message]:
    return [Message(role=USER, content=text)]


class TestState:

    def test_copy_is_deep(self):
        s = GraphState(messages=[Message(role=USER, content="hi")], scratch={"k": [1]})
        c = s.copy()
        c.scratch["k"].append(2)
        c.messages.append(Message(role=ASSISTANT, content="x"))
        assert s.scratch["k"] == [1]
        assert len(s.messages) == 1

    def test_last_message_empty_raises(self):
        with pytest.raises(IndexError):
            _ = GraphState().last_message


class TestBuilderValidation:

    def test_requires_entry(self):
        g = StateGraph().add_node(AgentNode("a", EchoAgent("a", "A")))
        with pytest.raises(ValueError):
            g.compile()

    def test_unknown_entry(self):
        g = StateGraph().add_node(AgentNode("a", EchoAgent("a", "A"))).set_entry("missing")
        with pytest.raises(ValueError):
            g.compile()

    def test_unknown_edge_target(self):
        g = (StateGraph()
             .add_node(AgentNode("a", EchoAgent("a", "A")))
             .set_entry("a")
             .add_edge("a", "ghost"))
        with pytest.raises(ValueError):
            g.compile()

    def test_duplicate_node(self):
        g = StateGraph().add_node(AgentNode("a", EchoAgent("a", "A")))
        with pytest.raises(ValueError):
            g.add_node(AgentNode("a", EchoAgent("a", "B")))

    def test_reserved_name(self):
        with pytest.raises(ValueError):
            StateGraph().add_node(AgentNode(END, EchoAgent("x", "X")))

    def test_static_and_conditional_conflict(self):
        g = (StateGraph()
             .add_node(AgentNode("a", EchoAgent("a", "A")))
             .add_edge("a", END))
        with pytest.raises(ValueError):
            g.add_conditional_edges("a", lambda s: END)


class TestExecution:

    def test_linear_flow(self):
        g = (StateGraph()
             .add_node(AgentNode("a", EchoAgent("a", "first")))
             .add_node(AgentNode("b", EchoAgent("b", "second")))
             .set_entry("a")
             .add_edge("a", "b")
             .add_edge("b", END))
        app = g.compile(name="g")
        out = app.run_nonstream(_user("hello"))
        assert out[-1]["content"] == "second"

    def test_function_node_sets_flag_and_branches(self):
        def flag(state: GraphState) -> GraphState:
            state.scratch["go_left"] = "left" in (state.last_message.content or "")
            return state

        g = (StateGraph()
             .add_node(AgentNode("start", EchoAgent("start", "go left please")))
             .add_node(FunctionNode("decide", flag))
             .add_node(AgentNode("left", EchoAgent("left", "LEFT")))
             .add_node(AgentNode("right", EchoAgent("right", "RIGHT")))
             .set_entry("start")
             .add_edge("start", "decide")
             .add_conditional_edges("decide", lambda s: "left" if s.scratch["go_left"] else "right")
             .add_edge("left", END)
             .add_edge("right", END))
        out = g.compile(name="g").run_nonstream(_user("hi"))
        assert out[-1]["content"] == "LEFT"

    def test_loop_with_max_steps_guard(self):
        # 'a' -> 'a' forever; guard must raise.
        g = (StateGraph(max_steps=5)
             .add_node(AgentNode("a", EchoAgent("a", "loop")))
             .set_entry("a")
             .add_edge("a", "a"))
        app = g.compile(name="g")
        with pytest.raises(RuntimeError):
            list(app.run(_user("hi")))

    def test_counter_loop_terminates(self):
        def inc(state: GraphState) -> GraphState:
            state.scratch["n"] = state.scratch.get("n", 0) + 1
            return state

        g = (StateGraph(max_steps=50)
             .add_node(FunctionNode("inc", inc))
             .set_entry("inc")
             .add_conditional_edges("inc", lambda s: END if s.scratch["n"] >= 3 else "inc"))
        app = g.compile(name="g")
        list(app.run(_user("go")))
        # Reach via internal run to confirm it stopped at 3 (no exception).

    def test_streaming_yields_multiple_chunks(self):
        g = (StateGraph()
             .add_node(AgentNode("a", EchoAgent("a", "A")))
             .add_node(AgentNode("b", EchoAgent("b", "B")))
             .set_entry("a")
             .add_edge("a", "b")
             .add_edge("b", END))
        app = g.compile(name="g")
        chunks = list(app.run(_user("hi")))
        contents = [c[-1]["content"] for c in chunks]
        assert "A" in contents and "B" in contents

    def test_compiled_graph_is_agent(self):
        g = (StateGraph()
             .add_node(AgentNode("a", EchoAgent("a", "A")))
             .set_entry("a")
             .add_edge("a", END))
        app = g.compile(name="g")
        assert isinstance(app, Agent)
        assert app.name == "g"


class TestNodeObservability:

    def _branch_graph(self):
        def flag(state: GraphState) -> GraphState:
            state.scratch["math"] = True
            return state

        return (StateGraph()
                .add_node(FunctionNode("classify", flag))
                .add_node(AgentNode("math", EchoAgent("math", "42")))
                .add_node(AgentNode("chat", EchoAgent("chat", "hi")))
                .set_entry("classify")
                .add_conditional_edges("classify", lambda s: "math" if s.scratch["math"] else "chat")
                .add_edge("math", END)
                .add_edge("chat", END))

    def test_node_events_emitted_with_handler(self):
        handler = _CollectingHandler()
        app = self._branch_graph().compile(name="g", handlers=[handler])
        list(app.run(_user("what is 6 * 7?")))
        types = [e.event_type for e in handler.events]
        assert "node.start" in types
        assert "node.end" in types
        # The DAG path taken: classify -> math -> END
        nodes_started = [e.payload["node"] for e in handler.events if e.event_type == "node.start"]
        assert nodes_started == ["classify", "math"]

    def test_node_end_records_next_edge(self):
        handler = _CollectingHandler()
        app = self._branch_graph().compile(name="g", handlers=[handler])
        list(app.run(_user("hello")))
        ends = {e.payload["node"]: e.payload["next"] for e in handler.events if e.event_type == "node.end"}
        assert ends["classify"] == "math"
        assert ends["math"] == END

    def test_no_node_events_without_handler(self):
        handler = _CollectingHandler()
        app = self._branch_graph().compile(name="g")
        list(app.run(_user("hello")))
        assert handler.events == []
