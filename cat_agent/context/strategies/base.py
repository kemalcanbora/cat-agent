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

"""Context strategy protocol and shared helpers.

References:
- Lindenbauer et al. (2025) arXiv:2508.21433 — observation masking
- Sun et al. (2025) arXiv:2510.11967 — context folding
- Mei et al. (2025) arXiv:2507.13334 — context engineering survey
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from cat_agent.context.budget import ContextBudget, ContextResult
from cat_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, TOOL, USER, Message


@runtime_checkable
class ContextStrategy(Protocol):
    name: str

    def should_apply(self, messages: List[Message], budget: ContextBudget) -> bool: ...

    def apply(self, messages: List[Message], budget: ContextBudget) -> ContextResult: ...


def find_system_and_task_indices(messages: List[Message]) -> tuple[int | None, int | None]:
    """Return (system_idx, first_user_task_idx)."""
    system_idx = None
    task_idx = None
    for i, msg in enumerate(messages):
        if system_idx is None and msg.role == SYSTEM:
            system_idx = i
        if task_idx is None and msg.role == USER:
            task_idx = i
            break
    return system_idx, task_idx


def protected_indices(messages: List[Message], keep_recent_exchanges: int = 1) -> set[int]:
    """Indices that strategies must never drop or mask."""
    protected: set[int] = set()
    sys_i, task_i = find_system_and_task_indices(messages)
    if sys_i is not None:
        protected.add(sys_i)
    if task_i is not None:
        protected.add(task_i)
    # Protect the most recent assistant/tool exchange.
    n = len(messages)
    if n == 0:
        return protected
    protected.add(n - 1)
    # Walk back over trailing tool/function results + their assistant.
    i = n - 1
    toolish = 0
    while i >= 0 and messages[i].role in (FUNCTION, TOOL):
        protected.add(i)
        toolish += 1
        i -= 1
    if i >= 0 and messages[i].role == ASSISTANT:
        protected.add(i)
    return protected


def roles_are_legal(messages: List[Message]) -> bool:
    """Lightweight check that tool results are preceded by an assistant tool call."""
    pending_calls = 0
    for msg in messages:
        if msg.role == ASSISTANT and (msg.tool_calls or msg.function_call):
            pending_calls = len(msg.tool_calls) if msg.tool_calls else 1
        elif msg.role in (FUNCTION, TOOL):
            if pending_calls <= 0:
                return False
            pending_calls -= 1
    return True
