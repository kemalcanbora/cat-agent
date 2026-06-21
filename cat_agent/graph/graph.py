# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import Callable, Dict, Iterator, List, Optional

from cat_agent.agent import Agent
from cat_agent.llm.schema import Message
from cat_agent.log import logger
from cat_agent.observability.context import child_span, get_run_context
from cat_agent.observability.emitter import emit
from cat_agent.observability.events import AgentEvent
from cat_agent.graph.nodes import Node
from cat_agent.graph.state import GraphState

# Reserved node names marking the graph's virtual start and terminal states.
START = '__start__'
END = '__end__'


class StateGraph:
    """A declarative graph of nodes and edges.

    Build a workflow by adding nodes and wiring them with static edges
    (`add_edge`) or conditional edges (`add_conditional_edges`). Pointing an
    edge back to an earlier node creates a loop; `max_steps` bounds execution so
    cycles cannot run forever. Call `compile()` to obtain a runnable
    `GraphAgent`.
    """

    def __init__(self, max_steps: int = 25):
        if max_steps < 1:
            raise ValueError('max_steps must be >= 1.')
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, str] = {}
        self.branches: Dict[str, Callable[[GraphState], str]] = {}
        self.entry: Optional[str] = None
        self.max_steps = max_steps

    def add_node(self, node: Node) -> 'StateGraph':
        if node.name in (START, END):
            raise ValueError(f"'{node.name}' is a reserved node name.")
        if node.name in self.nodes:
            raise ValueError(f"Node '{node.name}' already exists.")
        self.nodes[node.name] = node
        return self

    def set_entry(self, name: str) -> 'StateGraph':
        self.entry = name
        return self

    def add_edge(self, src: str, dst: str) -> 'StateGraph':
        if src in self.branches:
            raise ValueError(f"Node '{src}' already has conditional edges; cannot add a static edge.")
        self.edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: Callable[[GraphState], str]) -> 'StateGraph':
        if src in self.edges:
            raise ValueError(f"Node '{src}' already has a static edge; cannot add conditional edges.")
        self.branches[src] = router
        return self

    def validate(self) -> None:
        """Check structural integrity before execution."""
        if not self.entry:
            raise ValueError('No entry node set; call set_entry().')
        if self.entry not in self.nodes:
            raise ValueError(f"Entry node '{self.entry}' is not defined.")
        for src, dst in self.edges.items():
            if src not in self.nodes:
                raise ValueError(f"Edge source '{src}' is not a defined node.")
            if dst != END and dst not in self.nodes:
                raise ValueError(f"Edge target '{dst}' is not a defined node.")
        for src in self.branches:
            if src not in self.nodes:
                raise ValueError(f"Conditional edge source '{src}' is not a defined node.")

    def next_node(self, current: str, state: GraphState) -> str:
        """Resolve the next node name after `current` given the current state."""
        if current in self.branches:
            nxt = self.branches[current](state)
        else:
            nxt = self.edges.get(current, END)
        if nxt != END and nxt not in self.nodes:
            raise ValueError(f"Routing from '{current}' produced unknown node '{nxt}'.")
        return nxt

    def compile(self, **kwargs) -> 'GraphAgent':
        self.validate()
        return GraphAgent(self, **kwargs)


class GraphAgent(Agent):
    """A compiled `StateGraph` that is itself an `Agent`.

    Because it implements the standard `Agent` interface, a compiled graph
    composes with `Router`, `GroupChat`, observability handlers, and anything
    else that consumes `agent.run(messages)`.

    By default no system message is injected (each wrapped agent carries its
    own); pass `system_message=...` to override.
    """

    def __init__(self, graph: StateGraph, **kwargs):
        kwargs.setdefault('system_message', '')
        super().__init__(**kwargs)
        self.graph = graph

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        state = GraphState(messages=list(messages))
        current = self.graph.entry
        while current != END:
            if state.step >= self.graph.max_steps:
                raise RuntimeError(
                    f'Graph exceeded max_steps={self.graph.max_steps}; possible infinite loop.')
            node = self.graph.nodes[current]
            logger.debug(f'[graph] step={state.step} entering node={current}')
            # `child_span()` and `emit()` are both no-ops when no observability
            # handlers are active, so this stays cheap in the untraced path.
            with child_span():
                state, next_node = yield from self._exec_node(node, current, state, lang, kwargs)
            current = next_node
        logger.debug(f'[graph] finished after {state.step} step(s)')

    def _exec_node(self, node: Node, current: str, state: GraphState, lang: str, kwargs: dict):
        ctx = get_run_context()
        node_type = type(node).__name__
        started_at = time.monotonic()
        if ctx is not None:
            emit(AgentEvent.node_start(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                node=current,
                node_type=node_type,
                step=state.step,
            ))
        for chunk, new_state in node.run(state, host=self, lang=lang, **kwargs):
            if chunk is not None:
                yield chunk
            if new_state is not None:
                state = new_state
        state.step += 1
        next_node = self.graph.next_node(current, state)
        if ctx is not None:
            emit(AgentEvent.node_end(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                node=current,
                node_type=node_type,
                step=state.step - 1,
                duration_ms=(time.monotonic() - started_at) * 1000,
                next_node=next_node,
            ))
        return state, next_node
