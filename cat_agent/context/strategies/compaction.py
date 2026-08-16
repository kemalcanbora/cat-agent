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

"""LLM summary compaction of oldest contiguous message blocks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

from cat_agent.context.budget import ContextBudget, ContextResult, ContextStats, TokenCounter
from cat_agent.context.strategies.base import (
    protected_indices,
    roles_are_legal,
)
from cat_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, TOOL, USER, Message
from cat_agent.utils.message_utils import extract_text_from_message

_SUMMARY_PROMPT = (
    'Summarise the following agent transcript into a compact system note. '
    'Preserve: the original task statement, all decisions made, all unresolved '
    'sub-goals, and any facts the agent asserted. Omit raw tool payloads.'
)


class SummaryCompactionStrategy:
    name = 'summary_compaction'

    def __init__(
        self,
        *,
        llm: Any = None,
        counter: Optional[TokenCounter] = None,
        persist_dir: Optional[str] = None,
        min_block: int = 4,
    ) -> None:
        self.llm = llm
        self.counter = counter
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.min_block = min_block

    def should_apply(self, messages: List[Message], budget: ContextBudget) -> bool:
        if not budget.over_threshold:
            return False
        return len(messages) >= self.min_block + 2

    def apply(self, messages: List[Message], budget: ContextBudget) -> ContextResult:
        before_tokens = budget.current_token_count
        protected = protected_indices(messages)

        # Oldest contiguous unprotected block (after system+task), aligned so we
        # never split an assistant tool-call from its function/tool results.
        start = 0
        while start < len(messages) and start in protected:
            start += 1
        end = start
        while end < len(messages) - 2 and end not in protected:
            end += 1
        start, end = _align_tool_boundaries(messages, start, end)
        if end - start < self.min_block:
            return ContextResult(
                messages=list(messages),
                stats=ContextStats(before_tokens, before_tokens, len(messages), len(messages)),
                strategy_name=self.name,
                operation='compact',
            )

        block = messages[start:end]
        self._persist(block)
        summary_text = self._summarise(block)
        summary_msg = Message(role=SYSTEM, content=f'[Context summary]\n{summary_text}')

        new_messages = list(messages[:start]) + [summary_msg] + list(messages[end:])
        if not roles_are_legal(new_messages):
            return ContextResult(
                messages=list(messages),
                stats=ContextStats(before_tokens, before_tokens, len(messages), len(messages)),
                strategy_name=self.name,
                operation='compact',
            )
        after_tokens = (
            self.counter.count_messages(new_messages)
            if self.counter is not None
            else max(0, before_tokens // 2)
        )
        if after_tokens > before_tokens:
            return ContextResult(
                messages=list(messages),
                stats=ContextStats(before_tokens, before_tokens, len(messages), len(messages)),
                strategy_name=self.name,
                operation='compact',
            )
        return ContextResult(
            messages=new_messages,
            evicted_message_ids=[m.id for m in block],
            stats=ContextStats(
                tokens_before=before_tokens,
                tokens_after=after_tokens,
                messages_before=len(messages),
                messages_after=len(new_messages),
            ),
            strategy_name=self.name,
            operation='compact',
        )

    def _persist(self, block: List[Message]) -> None:
        if self.persist_dir is None:
            return
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        path = self.persist_dir / f'pre_compaction_{block[0].id[:8]}.jsonl'
        with path.open('a', encoding='utf-8') as fh:
            for msg in block:
                fh.write(json.dumps(msg.model_dump(mode='json'), ensure_ascii=False) + '\n')

    def _summarise(self, block: List[Message]) -> str:
        transcript = []
        for msg in block:
            text = extract_text_from_message(msg, add_upload_info=False)
            transcript.append(f'{msg.role}: {text[:2000]}')
        body = '\n'.join(transcript)
        if self.llm is None:
            # Deterministic extractive fallback — never invent facts.
            lines = []
            for msg in block:
                if msg.role in (USER, ASSISTANT, SYSTEM):
                    t = extract_text_from_message(msg, add_upload_info=False).strip()
                    if t:
                        lines.append(f'- ({msg.role}) {t[:300]}')
            return 'Extractive summary:\n' + '\n'.join(lines[:40])

        prompt = [
            Message(role=SYSTEM, content=_SUMMARY_PROMPT),
            Message(role=USER, content=body[:12000]),
        ]
        final: List[Message] = []
        for out in self.llm.chat(messages=prompt, stream=False):
            if out:
                final = out
        if not final:
            return body[:500]
        return extract_text_from_message(final[-1], add_upload_info=False)


def _align_tool_boundaries(messages: List[Message], start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) so it does not split tool-call / tool-result pairs."""
    if start >= end:
        return start, end
    # Do not start on a bare function/tool result (missing its call).
    while start < end and messages[start].role in (FUNCTION, TOOL):
        start += 1
    # If we start on an assistant tool-call, include all following tool results
    # that belong to it when the cut would leave them orphaned outside the block.
    # Conversely: if end falls on tool results whose call is inside the block,
    # extend end to consume those results; if the call is outside, shrink end.
    while end > start and messages[end - 1].role in (FUNCTION, TOOL):
        # Keep results only if their preceding assistant tool-call is inside.
        j = end - 1
        while j >= start and messages[j].role in (FUNCTION, TOOL):
            j -= 1
        if j >= start and messages[j].role == ASSISTANT and (
            messages[j].tool_calls or messages[j].function_call
        ):
            break  # pair fully inside
        end -= 1
    # Do not end immediately after an assistant tool-call with no results in block.
    if end > start:
        last = messages[end - 1]
        if last.role == ASSISTANT and (last.tool_calls or last.function_call):
            # Either extend to include following results or drop the call.
            k = end
            while k < len(messages) and messages[k].role in (FUNCTION, TOOL):
                k += 1
            if k > end and k <= len(messages):
                # Prefer dropping the incomplete call from the block.
                end -= 1
    return start, end
