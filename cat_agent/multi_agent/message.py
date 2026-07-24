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

"""Structured hub-internal message envelope for multi-agent communication."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

from cat_agent.llm.schema import ASSISTANT, USER, Message

MessageKind = Literal['inform', 'ask', 'answer', 'handoff', 'system']


@dataclass
class AgentMessage:
    """Hub-internal message. Rendered to ``Message`` at the LLM boundary."""

    sender: str
    content: str
    recipients: Optional[List[str]] = None  # None = broadcast
    kind: MessageKind = 'inform'
    reply_to: Optional[str] = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: Dict = field(default_factory=dict)

    def visible_to(self, agent_name: str) -> bool:
        if self.recipients is None:
            return True
        return agent_name in self.recipients or self.sender == agent_name

    def to_message(self, perspective: str) -> Message:
        """Render for the agent named ``perspective``.

        Own messages appear as assistant turns; others as user turns labelled
        with the sender name (matching existing GroupChat history format).
        """
        if self.sender == perspective:
            return Message(role=ASSISTANT, content=self.content, name=self.sender)
        return Message(role=USER, content=f'{self.sender}: {self.content}', name=self.sender)

    @classmethod
    def from_message(cls, msg: Message, *, known_names: Sequence[str] = ()) -> 'AgentMessage':
        """Lift a wire ``Message`` into an ``AgentMessage``.

        ``@Name`` mentions against ``known_names`` become ``recipients``.
        """
        content = _message_text(msg)
        sender = msg.name or msg.role
        recipients = parse_mentions(content, known_names) or None
        return cls(sender=sender, content=content, recipients=recipients, kind='inform')


def parse_mentions(content: str, agent_names: Sequence[str]) -> List[str]:
    """Resolve ``@Name`` mentions with word boundaries against known agent names.

    Avoids false positives from email addresses or ``@`` inside code.
    """
    if not content or not agent_names:
        return []
    # Longest names first so "BobSmith" wins over "Bob"
    names = sorted({n for n in agent_names if n}, key=len, reverse=True)
    if not names:
        return []
    # Require start-of-string or whitespace before '@' so emails (user@x.com)
    # and quoted/code '@Name' do not count as mentions.
    pattern = re.compile(r'(?:^|(?<=\s))@(' + '|'.join(map(re.escape, names)) + r')\b')
    seen: List[str] = []
    for match in pattern.finditer(content):
        name = match.group(1)
        if name not in seen:
            seen.append(name)
    return seen


def filter_visible(messages: Sequence[AgentMessage], agent_name: str) -> List[AgentMessage]:
    return [m for m in messages if m.visible_to(agent_name)]


def render_for_agent(messages: Sequence[AgentMessage], agent_name: str) -> List[Message]:
    """Apply visibility rules and render down to wire ``Message`` objects."""
    return [m.to_message(agent_name) for m in filter_visible(messages, agent_name)]


def _message_text(msg: Message) -> str:
    if isinstance(msg.content, list):
        return '\n'.join(x.text if x.text else '' for x in msg.content).strip()
    return (msg.content or '').strip()
