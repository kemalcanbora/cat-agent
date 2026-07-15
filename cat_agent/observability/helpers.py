"""Helpers for observability instrumentation."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Union

from cat_agent.llm.schema import Message
from cat_agent.observability.context import RunContext


def agent_model_name(llm: Any) -> Optional[str]:
    if llm is None:
        return None
    return getattr(llm, 'model', None)


def format_tool_args(tool_args: Union[str, dict], ctx: Optional[RunContext]) -> str:
    if isinstance(tool_args, dict):
        text = json.dumps(tool_args, ensure_ascii=False)
    else:
        text = str(tool_args)
    if ctx and ctx.redact.redact_tool_args:
        return '<redacted>'
    return text


def result_char_count(result: Any, ctx: Optional[RunContext]) -> int:
    if isinstance(result, str):
        text = result
    elif isinstance(result, list):
        text = str(result)
    else:
        text = json.dumps(result, ensure_ascii=False)
    if ctx:
        return min(len(text), ctx.redact.max_result_chars)
    return len(text)


def truncate_result_preview(result: Any, ctx: Optional[RunContext]) -> str:
    if isinstance(result, str):
        text = result
    elif isinstance(result, list):
        text = str(result)
    else:
        text = json.dumps(result, ensure_ascii=False)
    limit = ctx.redact.max_result_chars if ctx else 2000
    if len(text) <= limit:
        return text
    return text[:limit] + '...'


def messages_have_tool_call(messages: List[Message]) -> bool:
    for msg in messages or []:
        if getattr(msg, 'function_call', None):
            return True
    return False


def extract_usage(messages: List[Message]) -> Optional[dict]:
    for msg in reversed(messages or []):
        extra = getattr(msg, 'extra', None) or {}
        usage = extra.get('usage')
        if usage:
            return usage
    return None


def messages_to_payload(messages: List[Message]) -> list:
    return [message.model_dump() for message in (messages or [])]
