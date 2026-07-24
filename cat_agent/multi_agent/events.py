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

"""Lightweight multi-agent hub event callbacks.

Named ``HubEvent`` to avoid colliding with
``cat_agent.observability.events.AgentEvent``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional

HubEventType = Literal[
    'agent_start',
    'agent_end',
    'tool_call',
    'tool_result',
    'message',
    'handoff',
    'ask',
    'error',
]


@dataclass
class HubEvent:
    type: HubEventType
    agent: str
    timestamp: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)
    turn: int = 0
    depth: int = 0


EventCallback = Callable[[HubEvent], None]


def noop_event(_event: HubEvent) -> None:
    pass
