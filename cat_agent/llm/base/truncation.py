"""Input message truncation to fit within a token budget.

The main entry point is :func:`truncate_input_messages_roughly`.
"""

from typing import List

from cat_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, ContentItem, Message
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


def _message_to_native(msg: Message) -> dict:
    if msg.role == ASSISTANT and msg.function_call:
        text = f'{msg.function_call}'
    else:
        text = extract_text_from_message(msg, add_upload_info=True)
    data = {'role': msg.role, 'text': text}
    if msg.name:
        data['name'] = msg.name
    return data


def _native_to_message(item: dict) -> Message:
    role = item['role']
    text = item['text']
    if role == FUNCTION:
        return Message(
            role=FUNCTION,
            content=[ContentItem(text=text)],
            name=item.get('name', ''),
        )
    return Message(role=role, content=text)
