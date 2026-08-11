"""Helpers for observability instrumentation."""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional, Union

from cat_agent.llm.schema import Message
from cat_agent.observability.context import RunContext


def agent_model_name(llm: Any) -> Optional[str]:
    """Best-effort model id for traces (OpenAI id, GGUF filename, HF repo, …)."""
    if llm is None:
        return None
    model = getattr(llm, 'model', None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    for attr in ('model_id', 'model_path', 'repo_id', 'filename'):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value.strip():
            if attr == 'model_path':
                return os.path.basename(value.strip())
            return value.strip()
    return None


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


def format_obs_io(value: Any, ctx: Optional[RunContext]) -> Optional[str]:
    """Serialize messages/content for Langfuse Input/Output (and similar UIs)."""
    if value is None:
        return None
    if ctx and ctx.redact.redact_messages:
        return '<redacted>'

    if isinstance(value, list):
        simplified = []
        for item in value:
            if isinstance(item, Message):
                entry: dict = {'role': item.role, 'content': item.content}
                if item.name:
                    entry['name'] = item.name
                if getattr(item, 'tool_calls', None):
                    entry['tool_calls'] = [tc.model_dump() for tc in item.tool_calls]
                if getattr(item, 'reasoning_content', None):
                    entry['reasoning_content'] = item.reasoning_content
                simplified.append(entry)
            elif isinstance(item, dict):
                entry = {'role': item.get('role'), 'content': item.get('content')}
                if item.get('name'):
                    entry['name'] = item['name']
                if item.get('tool_calls'):
                    entry['tool_calls'] = item['tool_calls']
                elif item.get('function_call'):
                    # Legacy dump / audit payload: normalise display shape only.
                    fc = item['function_call']
                    entry['tool_calls'] = [{
                        'id': (item.get('extra') or {}).get('function_id') or '1',
                        'type': 'function',
                        'function': fc if isinstance(fc, dict) else {
                            'name': getattr(fc, 'name', ''),
                            'arguments': getattr(fc, 'arguments', ''),
                        },
                    }]
                if item.get('reasoning_content'):
                    entry['reasoning_content'] = item['reasoning_content']
                simplified.append(entry)
            else:
                simplified.append(item)
        text = json.dumps(simplified, ensure_ascii=False, default=str)
    elif isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)

    limit = ctx.redact.max_result_chars if ctx else 2000
    if len(text) > limit:
        return text[:limit] + '...'
    return text


def _message_field(msg: Any, key: str, default: Any = None) -> Any:
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _text_content(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get('text'):
                parts.append(str(item['text']))
            elif hasattr(item, 'text') and item.text:
                parts.append(str(item.text))
        return '\n'.join(parts).strip()
    return str(content).strip()


def format_llm_obs_output(messages: Any, ctx: Optional[RunContext] = None) -> Optional[str]:
    """Plain-text LLM output for Langfuse (tool calls + reasoning, not just content)."""
    if not messages:
        return None
    if ctx and ctx.redact.redact_messages:
        return '<redacted>'

    parts: list[str] = []
    items = messages if isinstance(messages, list) else [messages]
    for msg in items:
        role = _message_field(msg, 'role')
        if role != 'assistant':
            continue
        reasoning = _text_content(_message_field(msg, 'reasoning_content'))
        if reasoning:
            parts.append(reasoning)
        text = _text_content(_message_field(msg, 'content'))
        if text:
            parts.append(text)
        fc = _message_field(msg, 'function_call')
        tcs = _message_field(msg, 'tool_calls')
        if tcs:
            for tc in tcs:
                if isinstance(tc, dict):
                    fn = tc.get('function') or {}
                    name = fn.get('name', '')
                    args = fn.get('arguments', '')
                else:
                    name = getattr(getattr(tc, 'function', None), 'name', '') or ''
                    args = getattr(getattr(tc, 'function', None), 'arguments', '') or ''
                parts.append(f'[tool_call {name}({args})]')
        elif fc is not None:
            if isinstance(fc, dict):
                name = fc.get('name', '')
                args = fc.get('arguments', '')
            else:
                name = getattr(fc, 'name', '')
                args = getattr(fc, 'arguments', '')
            parts.append(f'[tool_call {name}({args})]')

    if parts:
        text = '\n'.join(parts)
        limit = ctx.redact.max_result_chars if ctx else 2000
        if len(text) > limit:
            return text[:limit] + '...'
        return text
    return format_obs_io(messages, ctx)


def format_run_obs_output(messages: Any, ctx: Optional[RunContext] = None) -> Optional[str]:
    """Summarize an agent run's new messages for Langfuse Output."""
    if not messages:
        return None
    if ctx and ctx.redact.redact_messages:
        return '<redacted>'

    parts: list[str] = []
    items = messages if isinstance(messages, list) else [messages]
    for msg in items:
        role = _message_field(msg, 'role')
        name = _message_field(msg, 'name') or role
        if role == 'assistant':
            block = format_llm_obs_output([msg], ctx)
            if block:
                parts.append(f'{name}: {block}')
        elif role == 'function':
            preview = _text_content(_message_field(msg, 'content'))
            if preview:
                tool = _message_field(msg, 'name') or 'tool'
                parts.append(f'{tool}: {preview}')

    if parts:
        text = '\n\n'.join(parts)
        limit = ctx.redact.max_result_chars if ctx else 2000
        if len(text) > limit:
            return text[:limit] + '...'
        return text
    return format_obs_io(messages, ctx)


def messages_have_tool_call(messages: List[Message]) -> bool:
    for msg in messages or []:
        if getattr(msg, 'tool_calls', None) or getattr(msg, 'function_call', None):
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
