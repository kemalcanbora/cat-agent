"""Collect final text from a streaming or non-streaming chat() call."""

from __future__ import annotations

import re
from typing import Any, List, Union

from cat_agent.llm.schema import Message
from cat_agent.log import logger

_THINK_BLOCK = re.compile(
    r'<think>.*?</think>',
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_TAG = re.compile(r'</?think>', flags=re.IGNORECASE)


def strip_thinking_markup(text: str) -> str:
    """Remove model think/reasoning XML wrappers that can wrap the answer."""
    if not text:
        return ''
    cleaned = _THINK_BLOCK.sub('', text)
    cleaned = _THINK_TAG.sub('', cleaned)
    return cleaned.strip()


def collect_chat_text(output: Union[List[Message], Any]) -> str:
    """Collapse a ``llm.chat(...)`` result into one assistant text blob.

    With ``stream=True, delta_stream=False`` each yield carries the *full*
    accumulated content so far — we keep the last non-empty ``content``.

    Reasoning models (e.g. Nemotron) often put useful text only in
    ``reasoning_content`` when the output budget is spent on thinking.
    Fall back to that when ``content`` is empty.
    """
    if isinstance(output, list) and (not output or isinstance(output[0], Message)):
        batches: List[Any] = [output]
    else:
        batches = list(output)

    last_content = ''
    last_reasoning = ''
    for batch in batches:
        if not batch:
            continue
        for msg in batch:
            content = getattr(msg, 'content', None)
            if content is not None and str(content).strip():
                last_content = str(content)
            reasoning = getattr(msg, 'reasoning_content', None)
            if reasoning is not None and str(reasoning).strip():
                last_reasoning = str(reasoning)

    if last_content.strip():
        return strip_thinking_markup(last_content)
    if last_reasoning.strip():
        logger.warning(
            'LLM returned empty content; using reasoning_content '
            '({} chars). Raise max_tokens if code extraction fails.',
            len(last_reasoning),
        )
        return strip_thinking_markup(last_reasoning)
    return ''
