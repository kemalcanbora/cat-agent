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

"""Observation masking — default context strategy (arXiv:2508.21433).

Placeholders are lossy-but-structured: bulk text is dropped while a pluggable
:class:`~cat_agent.context.residue.ResidueExtractor` keeps compact factual
residue (identifiers, repeated status tokens, head/tail lines, and
low-frequency salient mid-lines scored by within-output token IDF).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Union

from cat_agent.context.budget import ContextBudget, ContextResult, ContextStats, TokenCounter
from cat_agent.context.residue import (
    DEFAULT_RESIDUE_REGISTRY,
    ResidueExtractor,
    ResidueRegistry,
    generic_residue_extractor,
)
from cat_agent.context.strategies.base import protected_indices
from cat_agent.llm.schema import FUNCTION, TOOL, ContentItem, Message


class ObservationMaskingStrategy:
    """Replace old tool-result bodies with a structured elision placeholder.

    Keeps the N most recent observations intact. Never masks the system prompt,
    the original user task, or the most recent assistant/tool exchange.
    """

    name = 'observation_masking'

    def __init__(
        self,
        *,
        keep_recent: int = 3,
        head_chars: int = 80,
        tail_chars: int = 80,
        counter: Optional[TokenCounter] = None,
        residue_registry: Optional[ResidueRegistry] = None,
        residue_extractors: Optional[Dict[str, ResidueExtractor]] = None,
        residue_extractor: Optional[ResidueExtractor] = None,
    ) -> None:
        self.keep_recent = keep_recent
        self.head_chars = head_chars
        self.tail_chars = tail_chars
        self.counter = counter
        if residue_registry is not None:
            self.residue_registry = residue_registry
        else:
            self.residue_registry = ResidueRegistry(
                default=residue_extractor or generic_residue_extractor,
                extractors=residue_extractors,
            )

    def should_apply(self, messages: List[Message], budget: ContextBudget) -> bool:
        if not budget.over_threshold:
            return False
        return any(m.role in (FUNCTION, TOOL) for m in messages)

    def apply(self, messages: List[Message], budget: ContextBudget) -> ContextResult:
        before_tokens = budget.current_token_count
        protected = protected_indices(messages)
        obs_indices = [i for i, m in enumerate(messages) if m.role in (FUNCTION, TOOL)]
        keep = set(obs_indices[-self.keep_recent:]) if self.keep_recent > 0 else set()
        keep |= protected

        new_messages: List[Message] = []
        evicted: List[str] = []
        for i, msg in enumerate(messages):
            if msg.role not in (FUNCTION, TOOL) or i in keep:
                new_messages.append(msg)
                continue
            masked = self._mask_message(msg, step_hint=i)
            if masked.id != msg.id:
                masked.id = msg.id
            new_messages.append(masked)
            evicted.append(msg.id)

        after_tokens = (
            self.counter.count_messages(new_messages)
            if self.counter is not None
            else before_tokens
        )
        if after_tokens > before_tokens:
            return ContextResult(
                messages=list(messages),
                evicted_message_ids=[],
                stats=ContextStats(
                    tokens_before=before_tokens,
                    tokens_after=before_tokens,
                    messages_before=len(messages),
                    messages_after=len(messages),
                ),
                strategy_name=self.name,
                operation='mask',
            )
        return ContextResult(
            messages=new_messages,
            evicted_message_ids=evicted,
            stats=ContextStats(
                tokens_before=before_tokens,
                tokens_after=after_tokens,
                messages_before=len(messages),
                messages_after=len(new_messages),
            ),
            strategy_name=self.name,
            operation='mask',
        )

    def _mask_message(self, msg: Message, step_hint: int) -> Message:
        clone = copy.deepcopy(msg)
        tool = msg.name or 'tool'
        raw = msg.content
        if isinstance(raw, list):
            text_parts = [p for p in raw if getattr(p, 'type', None) == 'text' or getattr(p, 'text', None)]
            other = [p for p in raw if p not in text_parts]
            text = ' '.join(getattr(p, 'text', '') or '' for p in text_parts)
            placeholder = self._placeholder(tool, text, step_hint)
            clone.content = [ContentItem(text=placeholder), *other]
            return clone

        text = raw if isinstance(raw, str) else str(raw)
        clone.content = self._placeholder(tool, text, step_hint)
        return clone

    def _placeholder(self, tool: str, text: str, step_hint: int) -> str:
        nbytes = len(text.encode('utf-8'))
        header = (
            f'[tool output elided: {nbytes / 1024:.1f} KB from {tool}, step {step_hint}]'
        )
        residue = self.residue_registry.extract(tool, text)
        if residue:
            return f'{header}\n{residue}'
        # Fallback excerpt if extractor returns empty.
        excerpt = _head_tail(text, self.head_chars, self.tail_chars)
        return f'{header}\n{excerpt}' if excerpt else header


def _head_tail(text: str, head: int, tail: int) -> str:
    if not text:
        return ''
    if len(text) <= head + tail + 5:
        return text
    return f'{text[:head]} … {text[-tail:]}'
