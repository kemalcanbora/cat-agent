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

"""Explicit context folding API (arXiv:2510.11967).

Not an automatic strategy — callers opt in via ``with ctx.fold(task=...)``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

from cat_agent.context.budget import ContextBudget, ContextResult, ContextStats
from cat_agent.llm.schema import SYSTEM, USER, Message


@dataclass
class FoldSession:
    task: str
    scratch: List[Message] = field(default_factory=list)
    result: Optional[str] = None

    def add(self, message: Message) -> None:
        self.scratch.append(message)

    def set_result(self, text: str) -> None:
        self.result = text


class ContextFoldingStrategy:
    """Marker strategy — automatic ``should_apply`` is always False.

    Use :meth:`ContextManager.fold` for the explicit API.
    """

    name = 'context_folding'

    def should_apply(self, messages: List[Message], budget: ContextBudget) -> bool:
        return False

    def apply(self, messages: List[Message], budget: ContextBudget) -> ContextResult:
        return ContextResult(
            messages=list(messages),
            stats=ContextStats(
                budget.current_token_count,
                budget.current_token_count,
                len(messages),
                len(messages),
            ),
            strategy_name=self.name,
            operation='fold',
        )


@contextmanager
def fold_context(task: str) -> Iterator[FoldSession]:
    """Run a sub-task in an isolated scratch list; fold only the result back.

    Example::

        with mgr.fold(task='enumerate pods') as sub:
            sub.add(Message(USER, 'list pods'))
            # ... agent work on sub.scratch ...
            sub.set_result('3 pods running')
        # caller appends a single folded note into the main thread
    """
    session = FoldSession(task=task, scratch=[
        Message(role=SYSTEM, content=f'[Folded sub-task] {task}'),
        Message(role=USER, content=task),
    ])
    yield session


def fold_result_message(session: FoldSession) -> Message:
    body = session.result or '(no result)'
    return Message(
        role=SYSTEM,
        content=f'[Folded result for: {session.task}]\n{body}',
    )
