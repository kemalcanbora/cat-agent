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

"""Token budgeting for conversation context management.

Context management is separate from RAG (:mod:`cat_agent.memory`). Memory
retrieves documents; this module decides what stays in the model window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from cat_agent.llm.schema import Message
from cat_agent.utils.message_utils import extract_text_from_message


@dataclass
class ContextBudget:
    max_context_tokens: int
    reserved_output_tokens: int = 1024
    current_token_count: int = 0
    trigger_ratio: float = 0.70

    @property
    def usable_tokens(self) -> int:
        return max(0, self.max_context_tokens - self.reserved_output_tokens)

    @property
    def over_threshold(self) -> bool:
        if self.usable_tokens <= 0:
            return True
        return self.current_token_count >= int(self.usable_tokens * self.trigger_ratio)

    @property
    def over_budget(self) -> bool:
        return self.current_token_count > self.usable_tokens


@dataclass
class ContextStats:
    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int


@dataclass
class ContextResult:
    messages: list
    evicted_message_ids: list = field(default_factory=list)
    stats: Optional[ContextStats] = None
    strategy_name: str = ''
    operation: str = 'mask'
    operations: list = field(default_factory=list)


class ContextOverflowError(RuntimeError):
    """Raised when no strategy can bring the history under budget."""


@runtime_checkable
class TokenCounter(Protocol):
    def count_message(self, message: Message) -> int: ...

    def count_messages(self, messages: list) -> int: ...


class HeuristicTokenCounter:
    """Approximate counter using the bundled ``o200k_base`` tiktoken helper."""

    def __init__(self) -> None:
        self._cache: Dict[str, int] = {}

    def count_message(self, message: Message) -> int:
        mid = getattr(message, 'id', None) or id(message)
        key = str(mid)
        if key in self._cache:
            # Invalidate if content length changed (cheap check).
            content = message.content
            sig = f'{key}:{_content_sig(content)}'
            cached_sig_key = f'{key}:sig'
            if self._cache.get(cached_sig_key) == hash(sig):  # type: ignore[arg-type]
                return self._cache[key]
        from cat_agent.utils.tokenization_qwen import count_tokens
        text = extract_text_from_message(message, add_upload_info=False)
        # Multimodal: count a flat cost per non-text part without corrupting it.
        extra = _multimodal_extra_tokens(message)
        n = count_tokens(text or '') + extra
        self._cache[key] = n
        self._cache[f'{key}:sig'] = hash(f'{key}:{_content_sig(message.content)}')  # type: ignore[assignment]
        return n

    def count_messages(self, messages: list) -> int:
        return sum(self.count_message(m) for m in messages)

    def invalidate(self, message_id: Optional[str] = None) -> None:
        if message_id is None:
            self._cache.clear()
            return
        self._cache.pop(str(message_id), None)
        self._cache.pop(f'{message_id}:sig', None)


class BackendTokenCounter:
    """Prefer an exact backend tokenizer when available."""

    def __init__(self, llm: Any, fallback: Optional[TokenCounter] = None) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicTokenCounter()

    def count_message(self, message: Message) -> int:
        fn = getattr(self.llm, 'count_tokens', None)
        if callable(fn):
            try:
                return int(fn([message]))
            except Exception:
                pass
        return self.fallback.count_message(message)

    def count_messages(self, messages: list) -> int:
        fn = getattr(self.llm, 'count_tokens', None)
        if callable(fn):
            try:
                return int(fn(messages))
            except Exception:
                pass
        return self.fallback.count_messages(messages)


def _content_sig(content: Any) -> str:
    if isinstance(content, str):
        return f's:{len(content)}'
    if isinstance(content, list):
        return f'l:{len(content)}:{sum(len(str(x)) for x in content)}'
    return f'o:{type(content).__name__}'


def _multimodal_extra_tokens(message: Message) -> int:
    content = message.content
    if not isinstance(content, list):
        return 0
    extra = 0
    for item in content:
        kind = getattr(item, 'type', None)
        if kind and kind != 'text':
            # Fixed placeholder cost; never stringify image bytes into the counter.
            extra += 256
    return extra


def resolve_max_context_tokens(llm: Any, default: int = 8192) -> int:
    if llm is None:
        return default
    for attr in ('max_input_tokens', 'max_context_tokens', 'n_ctx', 'context_length'):
        val = getattr(llm, attr, None)
        if isinstance(val, int) and val > 0:
            return val
    cfg = getattr(llm, 'model_cfg', None) or getattr(llm, 'cfg', None) or {}
    if isinstance(cfg, dict):
        for key in ('max_input_tokens', 'max_tokens', 'n_ctx', 'num_ctx'):
            val = cfg.get(key)
            if isinstance(val, int) and val > 0:
                return val
    try:
        from cat_agent.settings import DEFAULT_MAX_INPUT_TOKENS
        return int(DEFAULT_MAX_INPUT_TOKENS)
    except Exception:
        return default
