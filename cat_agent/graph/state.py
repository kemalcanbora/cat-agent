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

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

from cat_agent.llm.schema import Message


@dataclass
class GraphState:
    """The state object that flows through a graph.

    Attributes:
        messages: The conversation so far. Nodes typically append to this.
        scratch: Free-form per-run data used by nodes and conditional edges
          (counters, routing flags, intermediate values, ...).
        step: Global step counter incremented once per executed node. Used by
          the engine as a loop guard against runaway cycles.
    """

    messages: List[Message] = field(default_factory=list)
    scratch: Dict[str, Any] = field(default_factory=dict)
    step: int = 0

    def copy(self) -> 'GraphState':
        """Return a deep copy of this state.

        Useful for checkpointing or for parallel branches that must not share
        mutable references.
        """
        return GraphState(
            messages=copy.deepcopy(self.messages),
            scratch=copy.deepcopy(self.scratch),
            step=self.step,
        )

    @property
    def last_message(self) -> Message:
        """Return the most recent message, or raise if there is none."""
        if not self.messages:
            raise IndexError('GraphState has no messages yet.')
        return self.messages[-1]
