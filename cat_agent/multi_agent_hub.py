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

from abc import ABC
from typing import Dict, List, Optional

from cat_agent.agent import Agent
from cat_agent.log import logger
from cat_agent.llm.schema import USER, Message
from cat_agent.multi_agent.blackboard import Blackboard
from cat_agent.multi_agent.events import EventCallback, HubEvent, noop_event
from cat_agent.multi_agent.handoff import Handoff
from cat_agent.multi_agent.tools import AskAgentTool, HandoffTool, ReadArtifactTool, WriteArtifactTool


class MultiAgentHub(ABC):
    """Base for orchestrator-mediated multi-agent patterns.

    Subclasses should call ``_init_hub(...)`` during ``__init__`` to enable
    blackboard, ask_agent / handoff tools, and event callbacks.
    """

    def _init_hub(
        self,
        *,
        on_event: Optional[EventCallback] = None,
        blackboard: Optional[Blackboard] = None,
        max_ask_depth: int = 3,
        max_ask_calls: int = 10,
        allow_list: Optional[Dict[str, List[str]]] = None,
        inject_ask_agent: bool = True,
        inject_handoff: bool = True,
        inject_blackboard_tools: bool = True,
        auto_artifact_chars: int = 2000,
    ) -> None:
        self._on_event: EventCallback = on_event or noop_event
        self.blackboard: Blackboard = blackboard or Blackboard()
        self.max_ask_depth = max_ask_depth
        self.max_ask_calls = max_ask_calls
        self.allow_list = allow_list
        self.auto_artifact_chars = auto_artifact_chars
        self._inject_ask_agent = inject_ask_agent
        self._inject_handoff = inject_handoff
        self._inject_blackboard_tools = inject_blackboard_tools

        self._call_stack: List[str] = []
        self._ask_calls: int = 0
        self._pending_handoff: Optional[Handoff] = None
        self._hub_turn: int = 0

    @property
    def agents(self) -> List[Agent]:
        try:
            agent_list = self._agents
            assert isinstance(agent_list, list)
            assert all(isinstance(a, Agent) for a in agent_list)
            assert len(agent_list) > 0
            assert all(a.name for a in agent_list), 'All agents must have a name.'
            assert len(set(a.name for a in agent_list)) == len(agent_list), 'Agents must have unique names.'
        except (AttributeError, AssertionError) as e:
            logger.error(
                f'Class {self.__class__.__name__} inherits from MultiAgentHub. '
                'However, the following constraints are violated: '
                "1) A class that inherits from MultiAgentHub must have an '_agents' attribute of type 'List[Agent]'. "
                "2) The '_agents' must be a non-empty list containing at least one agent. "
                "3) All agents in '_agents' must have non-empty, non-duplicate string names.")
            raise e
        return agent_list

    @property
    def agent_names(self) -> List[str]:
        return [x.name for x in self.agents]

    @property
    def nonuser_agents(self):
        from cat_agent.agents.user_agent import UserAgent  # put here to avoid cyclic import
        return [a for a in self.agents if not isinstance(a, UserAgent)]

    # ------------------------------------------------------------------ events

    def emit_event(
        self,
        event_type: str,
        agent: str,
        payload: Optional[dict] = None,
        *,
        turn: Optional[int] = None,
        depth: Optional[int] = None,
    ) -> None:
        event = HubEvent(
            type=event_type,  # type: ignore[arg-type]
            agent=agent,
            payload=payload or {},
            turn=self._hub_turn if turn is None else turn,
            depth=len(self._call_stack) if depth is None else depth,
        )
        self._on_event(event)

    # ------------------------------------------------------------------ tools

    def _inject_hub_tools(self, agents: Optional[List[Agent]] = None) -> None:
        """Inject hub-mediated tools into each member agent's function_map."""
        if not hasattr(self, 'blackboard'):
            self._init_hub()
        agents = agents if agents is not None else self._agents
        for agent in agents:
            if not agent.name:
                continue
            # Skip pure UserAgent (no function_map tools expected)
            from cat_agent.agents.user_agent import UserAgent
            if isinstance(agent, UserAgent):
                continue
            if not hasattr(agent, 'function_map'):
                continue
            if self._inject_ask_agent:
                agent._init_tool(AskAgentTool(self, agent.name))
            if self._inject_handoff:
                agent._init_tool(HandoffTool(self, agent.name))
            if self._inject_blackboard_tools:
                agent._init_tool(WriteArtifactTool(self, agent.name))
                agent._init_tool(ReadArtifactTool(self, agent.name))

    def set_pending_handoff(self, handoff: Handoff, *, caller: str) -> None:
        self._pending_handoff = handoff
        self.emit_event('handoff', caller, {'to': handoff.to, 'context': handoff.context})

    def consume_pending_handoff(self) -> Optional[Handoff]:
        handoff = self._pending_handoff
        self._pending_handoff = None
        return handoff

    def handle_ask(self, caller: str, target_name: str, question: str, **kwargs) -> str:
        """Run ``target_name`` on an isolated question and return its answer."""
        if not hasattr(self, '_call_stack'):
            self._init_hub()

        if self.allow_list is not None:
            allowed = self.allow_list.get(caller, [])
            if target_name not in allowed:
                return (f'Error: {caller} is not allowed to ask {target_name}. '
                        f'Allowed targets: {", ".join(allowed) or "(none)"}')

        if target_name not in self.agent_names:
            return (f'Error: unknown agent "{target_name}". '
                    f'Available: {", ".join(self.agent_names)}')

        if target_name in self._call_stack:
            return (f'Error: {target_name} is already handling a request in this chain '
                    f'(cycle rejected).')

        if len(self._call_stack) >= self.max_ask_depth:
            return f'Error: maximum delegation depth ({self.max_ask_depth}) reached.'

        if self._ask_calls >= self.max_ask_calls:
            return f'Error: maximum ask_agent calls ({self.max_ask_calls}) reached for this run.'

        self._ask_calls += 1
        self._call_stack.append(caller)
        self.emit_event('ask', caller, {'target': target_name, 'question': question})

        try:
            target = self.agents[self.agent_names.index(target_name)]
            self.emit_event('agent_start', target_name, {'via': 'ask_agent'})
            # Caller tool kwargs often include messages=... — must not forward that
            # into the nested run (duplicate keyword).
            nested_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ('messages', 'function_list', 'files')
            }
            messages = [Message(role=USER, content=question)]
            result = None
            for result in target.run(messages=messages, **nested_kwargs):
                pass
            content = _extract_last_content(result)
            self.emit_event('agent_end', target_name, {'via': 'ask_agent', 'chars': len(content)})
            return content
        except Exception as exc:
            self.emit_event('error', target_name, {'error': str(exc)})
            return f'Error asking {target_name}: {exc}'
        finally:
            if self._call_stack and self._call_stack[-1] == caller:
                self._call_stack.pop()

    def maybe_offload_to_blackboard(self, content: str, *, author: str, key_hint: str = 'output') -> str:
        """If content exceeds the auto-artifact threshold, store and return a ref."""
        if not content or self.auto_artifact_chars <= 0:
            return content
        if len(content) < self.auto_artifact_chars:
            return content
        key = f'{key_hint}_{author}_{len(self.blackboard.keys()) + 1}'
        ref = self.blackboard.write(key, content, author=author)
        return (f'[Large output auto-stored as {ref} (~{len(content)} chars). '
                f'Use read_artifact to retrieve it.]')


def _extract_last_content(result) -> str:
    if not result:
        return ''
    msg = result[-1]
    content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
    if isinstance(content, list):
        return '\n'.join(x.text if getattr(x, 'text', None) else '' for x in content).strip()
    return (content or '').strip()
