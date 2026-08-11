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

"""Streaming tool-call delta merger (ToolCallChunk pattern).

OpenAI-compatible streams emit partial ``tool_calls`` entries keyed by ``index``.
A single logical call may arrive as:

* first chunk: ``index=0``, ``id=…``, ``function.name=…``, empty/partial arguments
* later chunks: ``index=0``, no id/name, ``function.arguments`` fragments only
* interleaved with ``index=1`` for a concurrent second call

This module merges those deltas into complete :class:`~cat_agent.llm.schema.ToolCall`
objects. It is also the seam where future token-level streaming of tool arguments
would land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from cat_agent.llm.schema import FunctionCall, ToolCall, generate_tool_call_id


@dataclass
class _Acc:
    id: Optional[str] = None
    name: str = ''
    arguments: str = ''


@dataclass
class ToolCallStreamMerger:
    """Accumulate tool-call stream deltas by ``index``."""

    _by_index: Dict[int, _Acc] = field(default_factory=dict)

    def push(self, delta: Any) -> None:
        """Merge one tool-call delta (object or dict)."""
        index = _field(delta, 'index')
        if index is None:
            # Providers sometimes omit index for a single call; use 0 or next slot.
            index = 0 if not self._by_index else max(self._by_index) + 1
        index = int(index)

        acc = self._by_index.setdefault(index, _Acc())

        tc_id = _field(delta, 'id')
        if tc_id:
            acc.id = str(tc_id)

        fn = _field(delta, 'function')
        if fn is not None:
            name = _field(fn, 'name')
            if name:
                # Name fragments are rare but legal; append like arguments.
                acc.name += str(name)
            arguments = _field(fn, 'arguments')
            if arguments:
                acc.arguments += str(arguments)

    def push_many(self, deltas: Optional[Iterable[Any]]) -> None:
        if not deltas:
            return
        for delta in deltas:
            self.push(delta)

    def tool_calls(self) -> List[ToolCall]:
        """Return merged tool calls in index order."""
        out: List[ToolCall] = []
        for index in sorted(self._by_index):
            acc = self._by_index[index]
            out.append(ToolCall(
                id=acc.id or generate_tool_call_id(),
                function=FunctionCall(name=acc.name or '', arguments=acc.arguments or ''),
            ))
        return out

    def clear(self) -> None:
        self._by_index.clear()


def _field(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
