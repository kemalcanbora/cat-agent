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

"""ContextManager — ordered strategy composition for the agent context window."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Sequence

from cat_agent.context.budget import (
    BackendTokenCounter,
    ContextBudget,
    ContextOverflowError,
    ContextResult,
    ContextStats,
    HeuristicTokenCounter,
    TokenCounter,
    resolve_max_context_tokens,
)
from cat_agent.context.strategies.base import ContextStrategy, roles_are_legal
from cat_agent.context.strategies.compaction import SummaryCompactionStrategy
from cat_agent.context.strategies.folding import (
    ContextFoldingStrategy,
    FoldSession,
    fold_context,
    fold_result_message,
)
from cat_agent.context.strategies.masking import ObservationMaskingStrategy
from cat_agent.llm.schema import Message

try:
    from cat_agent.trace.schema import ContextOpPayload
except Exception:  # pragma: no cover — soft dependency
    ContextOpPayload = None  # type: ignore


class ContextManager:
    """Apply strategies until the budget is satisfied or raise overflow."""

    def __init__(
        self,
        strategies: Optional[Sequence[ContextStrategy]] = None,
        *,
        counter: Optional[TokenCounter] = None,
        max_context_tokens: Optional[int] = None,
        reserved_output_tokens: int = 1024,
        trigger_ratio: float = 0.70,
    ) -> None:
        self.strategies: List[ContextStrategy] = list(strategies) if strategies is not None else [
            ObservationMaskingStrategy(),
        ]
        self.counter = counter or HeuristicTokenCounter()
        self.max_context_tokens = max_context_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.trigger_ratio = trigger_ratio

    def prepare(self, messages: List[Message], *, llm: Any = None) -> ContextResult:
        msgs = list(messages)
        counter = self.counter
        if llm is not None and not isinstance(counter, BackendTokenCounter):
            counter = BackendTokenCounter(llm, fallback=self.counter)

        # Wire counters into strategies that accept them.
        for strat in self.strategies:
            if hasattr(strat, 'counter') and getattr(strat, 'counter') is None:
                strat.counter = counter  # type: ignore[attr-defined]
            if isinstance(strat, SummaryCompactionStrategy) and strat.llm is None and llm is not None:
                # Prefer a cheaper override if the caller set one; else agent's LLM.
                pass

        max_ctx = self.max_context_tokens or resolve_max_context_tokens(llm)
        tokens = counter.count_messages(msgs)
        budget = ContextBudget(
            max_context_tokens=max_ctx,
            reserved_output_tokens=self.reserved_output_tokens,
            current_token_count=tokens,
            trigger_ratio=self.trigger_ratio,
        )

        if not budget.over_threshold:
            return ContextResult(
                messages=msgs,
                stats=ContextStats(tokens, tokens, len(msgs), len(msgs)),
                operations=[],
            )

        operations: list = []
        evicted_all: List[str] = []
        for strat in self.strategies:
            budget.current_token_count = counter.count_messages(msgs)
            if not budget.over_threshold and not budget.over_budget:
                break
            if not strat.should_apply(msgs, budget):
                continue
            result = strat.apply(msgs, budget)
            if result.stats and result.stats.tokens_after > result.stats.tokens_before:
                continue  # invariant: never increase
            if not roles_are_legal(result.messages):
                continue
            msgs = result.messages
            evicted_all.extend(result.evicted_message_ids)
            op = _to_context_op(result)
            if op is not None:
                operations.append(op)
            budget.current_token_count = (
                result.stats.tokens_after if result.stats else counter.count_messages(msgs)
            )

        budget.current_token_count = counter.count_messages(msgs)
        if budget.over_budget:
            raise ContextOverflowError(
                f'Context still over budget after strategies '
                f'({budget.current_token_count} > {budget.usable_tokens} usable tokens). '
                f'Messages={len(msgs)}. Strategies tried: '
                f'{[getattr(s, "name", type(s).__name__) for s in self.strategies]}'
            )

        return ContextResult(
            messages=msgs,
            evicted_message_ids=evicted_all,
            stats=ContextStats(
                tokens_before=tokens,
                tokens_after=budget.current_token_count,
                messages_before=len(messages),
                messages_after=len(msgs),
            ),
            operations=operations,
        )

    @contextmanager
    def fold(self, task: str) -> Iterator[FoldSession]:
        with fold_context(task) as session:
            yield session

    def fold_into(self, messages: List[Message], session: FoldSession) -> List[Message]:
        return list(messages) + [fold_result_message(session)]


def _to_context_op(result: ContextResult):
    if ContextOpPayload is None or result.stats is None:
        return {
            'operation': result.operation,
            'messages_before': result.stats.messages_before if result.stats else 0,
            'messages_after': result.stats.messages_after if result.stats else 0,
            'tokens_before': result.stats.tokens_before if result.stats else 0,
            'tokens_after': result.stats.tokens_after if result.stats else 0,
            'strategy_name': result.strategy_name,
            'evicted_message_ids': list(result.evicted_message_ids),
        }
    return ContextOpPayload(
        operation=result.operation,  # type: ignore[arg-type]
        messages_before=result.stats.messages_before,
        messages_after=result.stats.messages_after,
        tokens_before=result.stats.tokens_before,
        tokens_after=result.stats.tokens_after,
        strategy_name=result.strategy_name,
        evicted_message_ids=list(result.evicted_message_ids),
    )


_DEFAULT_MGR: Optional[ContextManager] = None


def get_default_context_manager(agent: Any = None) -> Optional[ContextManager]:
    """Return ObservationMasking manager when the agent wants the default.

    - ``agent.context_manager is False`` → disabled
    - ``agent.context_manager is a ContextManager`` → use it
    - ``None`` → shared default (no-op when under threshold)
    - ``CAT_AGENT_CONTEXT=0`` → disabled globally
    """
    import os
    flag = os.getenv('CAT_AGENT_CONTEXT', '').strip().lower()
    if flag in {'0', 'false', 'no', 'off'}:
        return None
    attached = getattr(agent, 'context_manager', None)
    if attached is False:
        return None
    if attached is not None:
        return attached
    global _DEFAULT_MGR
    if _DEFAULT_MGR is None:
        _DEFAULT_MGR = ContextManager(
            strategies=[ObservationMaskingStrategy(keep_recent=3)],
        )
    return _DEFAULT_MGR


def default_context_manager(
    *,
    llm: Any = None,
    enable_summary: bool = False,
    persist_dir: Optional[str] = None,
) -> ContextManager:
    strategies: List[ContextStrategy] = [ObservationMaskingStrategy()]
    if enable_summary:
        strategies.append(SummaryCompactionStrategy(llm=llm, persist_dir=persist_dir))
    strategies.append(ContextFoldingStrategy())
    return ContextManager(strategies=strategies)
