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

"""Input message truncation to fit within a token budget.

The main entry point is :func:`truncate_input_messages_roughly`.

Structured fields (``tool_calls``, ``tool_call_id``, ``extra``, ``content``,
``reasoning_content``) are carried through the native layer alongside a separate
counting ``text``. The counting text is never used as the recovered message body
unless the Rust path mutated it (omit / truncate).
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

from cat_agent.llm.schema import (
    ASSISTANT,
    FUNCTION,
    SYSTEM,
    TOOL,
    ContentItem,
    FunctionCall,
    Message,
    ToolCall,
)
from cat_agent.log import logger
from cat_agent.utils.message_utils import extract_text_from_message
from cat_agent.utils.tokenization_qwen import ensure_qwen_tokenizer


def truncate_input_messages_roughly(messages: List[Message], max_tokens: int) -> List[Message]:
    """Truncate *messages* so the total token count fits within *max_tokens*."""
    from cat_agent.llm.base.model import ModelServiceError  # deferred

    if len([m for m in messages if m.role == SYSTEM]) >= 2:
        raise ModelServiceError(
            code='400',
            message='The input messages must contain no more than one system message. '
            ' And the system message, if exists, must be the first message.',
        )
    if not messages:
        return messages

    ensure_qwen_tokenizer()
    native = _native()
    native_messages = [_message_to_native(msg) for msg in messages]
    try:
        truncated = native.truncate_messages(native_messages, max_tokens)
    except ValueError as error:
        raise ModelServiceError(code='400', message=str(error)) from error

    logger.info('Truncated messages via native Rust path')
    return [_native_to_message(item) for item in truncated]


def _native():
    from importlib import import_module

    try:
        return import_module('cat_agent._native')
    except ImportError as error:
        raise ImportError(
            'Message truncation requires the cat_agent native Rust extension. '
            'Install a platform wheel or build it with: '
            '`maturin develop --manifest-path native/Cargo.toml`'
        ) from error


def _counting_text(msg: Message) -> str:
    """Token-estimation text. Never used as recovered content when structure is intact."""
    body = extract_text_from_message(msg, add_upload_info=True)
    if msg.role == ASSISTANT and msg.tool_calls:
        parts = [body] if body else []
        for tc in msg.tool_calls:
            parts.append(f'{tc.function.name}\n{tc.function.arguments}')
        return '\n'.join(parts)
    return body


def _serialize_content(content: Union[str, List[ContentItem], None]) -> Any:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    return [item.model_dump() for item in content]


def _deserialize_content(content: Any) -> Union[str, List[ContentItem]]:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        items: List[ContentItem] = []
        for item in content:
            if isinstance(item, ContentItem):
                items.append(item)
            elif isinstance(item, dict):
                items.append(ContentItem(**{k: v for k, v in item.items() if v is not None}))
            else:
                items.append(ContentItem(text=str(item)))
        return items
    return str(content)


def _message_to_native(msg: Message) -> dict:
    text = _counting_text(msg)
    data: dict = {
        'role': msg.role,
        'text': text,
        'text_baseline': text,
        'content': _serialize_content(msg.content),
    }
    if msg.name is not None:
        data['name'] = msg.name
    if msg.tool_calls is not None:
        data['tool_calls'] = [tc.model_dump() for tc in msg.tool_calls]
    if msg.tool_call_id is not None:
        data['tool_call_id'] = msg.tool_call_id
    if msg.extra is not None:
        data['extra'] = dict(msg.extra)
    if msg.reasoning_content is not None:
        data['reasoning_content'] = _serialize_content(msg.reasoning_content)
    return data


def _native_to_message(item: dict) -> Message:
    role = item['role']
    text = item.get('text', '')
    baseline = item.get('text_baseline', text)
    name: Optional[str] = item.get('name')
    extra = item.get('extra')
    tool_call_id = item.get('tool_call_id')
    tool_calls = None
    if item.get('tool_calls'):
        tool_calls = [ToolCall(**tc) if isinstance(tc, dict) else tc for tc in item['tool_calls']]
    elif item.get('function_call'):
        fc_data = item['function_call']
        tc_id = (extra or {}).get('function_id') if isinstance(extra, dict) else None
        tool_calls = [
            ToolCall(
                id=tc_id or '1',
                function=FunctionCall(**fc_data) if isinstance(fc_data, dict) else fc_data,
            )
        ]
    reasoning = None
    if 'reasoning_content' in item and item['reasoning_content'] is not None:
        reasoning = _deserialize_content(item['reasoning_content'])

    body_mutated = text != baseline

    if body_mutated:
        if role in (FUNCTION, TOOL):
            return Message(
                role=role,
                content=[ContentItem(text=text)],
                name=name if name is not None else '',
                tool_call_id=tool_call_id,
                extra=extra,
            )
        return Message(
            role=role,
            content=text,
            name=name,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            extra=extra,
            reasoning_content=reasoning,
        )

    content = _deserialize_content(item['content']) if 'content' in item else text
    if role in (FUNCTION, TOOL):
        if isinstance(content, str):
            content = [ContentItem(text=content)]
        return Message(
            role=role,
            content=content,
            name=name if name is not None else '',
            tool_call_id=tool_call_id,
            extra=extra,
        )
    return Message(
        role=role,
        content=content,
        name=name,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        extra=extra,
        reasoning_content=reasoning,
    )
