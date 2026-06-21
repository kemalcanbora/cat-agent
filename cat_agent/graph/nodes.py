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

from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

from cat_agent.agent import Agent
from cat_agent.llm.schema import FUNCTION, Message
from cat_agent.graph.state import GraphState

# A node yields a stream of (chunk, new_state) tuples:
#   * chunk: a partial list of messages to stream upward (or None to stream nothing).
#   * new_state: the committed GraphState (or None while still streaming).
# A node should yield the committed state exactly once, as its final item.
NodeStream = Iterator[Tuple[Optional[List[Message]], Optional[GraphState]]]


class Node(ABC):
    """A unit of work in a graph.

    Subclasses implement `run`, which receives the current `GraphState` and the
    host `GraphAgent` (so tool nodes can reuse the agent's tool/observability
    machinery), and yields streaming chunks plus the updated state.
    """

    def __init__(self, name: str):
        if not name:
            raise ValueError('Node name must be a non-empty string.')
        self.name = name

    @abstractmethod
    def run(self, state: GraphState, host: 'Agent', **kwargs) -> NodeStream:
        raise NotImplementedError


class AgentNode(Node):
    """Wrap any Cat-Agent agent (Assistant, ReActChat, a sub-graph, ...) as a node.

    The wrapped agent is run on the current `state.messages`; its generated
    messages are streamed upward and then appended to the state.
    """

    def __init__(self, name: str, agent: Agent):
        super().__init__(name)
        self.agent = agent

    def run(self, state: GraphState, host: 'Agent', **kwargs) -> NodeStream:
        kwargs.pop('host', None)
        last: List[Message] = []
        for chunk in self.agent.run(state.messages, **kwargs):
            last = [Message(**m) if isinstance(m, dict) else m for m in chunk]
            yield last, None
        state.messages = state.messages + last
        yield None, state


class FunctionNode(Node):
    """Run arbitrary Python against the state.

    The supplied callable receives the `GraphState` and must return a
    `GraphState` (typically the same instance, mutated). Use this to set routing
    flags in `state.scratch`, post-process messages, or call non-LLM logic.
    """

    def __init__(self, name: str, fn: Callable[[GraphState], GraphState]):
        super().__init__(name)
        self.fn = fn

    def run(self, state: GraphState, host: 'Agent', **kwargs) -> NodeStream:
        new_state = self.fn(state)
        if not isinstance(new_state, GraphState):
            raise TypeError(
                f"FunctionNode '{self.name}' must return a GraphState, got {type(new_state).__name__}.")
        yield None, new_state


class ToolNode(Node):
    """Invoke a registered tool directly and append its result as a function message.

    `args_from` maps the current state to the tool's argument dict. The tool is
    executed via the host agent's `_call_tool`, so it participates in the
    existing observability spans.
    """

    def __init__(self,
                 name: str,
                 tool_name: str,
                 args_from: Callable[[GraphState], Union[Dict, str]]):
        super().__init__(name)
        self.tool_name = tool_name
        self.args_from = args_from

    def run(self, state: GraphState, host: 'Agent', **kwargs) -> NodeStream:
        kwargs.pop('host', None)
        tool_args = self.args_from(state)
        result = host._call_tool(self.tool_name, tool_args, **kwargs)
        msg = Message(role=FUNCTION, name=self.tool_name, content=result)
        state.messages = state.messages + [msg]
        yield [msg], state
