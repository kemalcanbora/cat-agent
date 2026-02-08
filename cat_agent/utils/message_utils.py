"""Helpers for converting, formatting and extracting data from Message objects."""

import json
import re
from typing import List, Literal, Tuple, Union

from cat_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, FUNCTION, SYSTEM, USER, ContentItem, Message
from cat_agent.log import logger
from cat_agent.utils.file_utils import get_basename_from_url
from cat_agent.utils.misc import has_chinese_chars


# ---------------------------------------------------------------------------
# Message ↔ multimodal / text conversion
# ---------------------------------------------------------------------------


def format_as_multimodal_message(
    msg: Message,
    add_upload_info: bool,
    add_multimodel_upload_info: bool,
    add_audio_upload_info: bool,
    lang: Literal['auto', 'en', 'zh'] = 'auto',
) -> Message:
    assert msg.role in (USER, ASSISTANT, SYSTEM, FUNCTION)
    content: List[ContentItem] = []

    if isinstance(msg.content, str):
        content = [ContentItem(text=msg.content)]
    elif isinstance(msg.content, list):
        files = []
        for item in msg.content:
            k, v = item.get_type_and_value()
            if k in ('text', 'image', 'audio', 'video'):
                content.append(item)
            if k == 'file':
                files.append((v, k))
            if add_multimodel_upload_info and k in ('image', 'video'):
                if isinstance(v, str):
                    files.append((v, k))
                elif isinstance(v, list):
                    for _v in v:
                        files.append((_v, k))
                else:
                    raise TypeError
            if add_audio_upload_info and k == 'audio':
                if isinstance(v, str):
                    files.append((v, k))
                elif isinstance(v, dict):
                    files.append((v['data'], k))
                else:
                    raise TypeError

        if add_upload_info and files and (msg.role in (SYSTEM, USER)):
            if lang == 'auto':
                has_zh = has_chinese_chars(msg)
            else:
                has_zh = (lang == 'zh')

            # Build upload-info tags -- format is the same regardless of language
            _kind_prefix = {'image': '![image]', 'video': '![video]', 'audio': '![audio]'}
            upload = []
            for f, k in [(get_basename_from_url(f), k) for f, k in files]:
                prefix = _kind_prefix.get(k, '[file]')
                upload.append(f'{prefix}({f})')

            if upload:
                upload_str = ' '.join(upload)
                if msg.role in (SYSTEM, USER):
                    upload_str = f'(Uploaded {upload_str}) ' if not has_zh else f'(Uploaded {upload_str})'

                # Avoid duplicate upload info
                if not any(item.text and upload_str in item.text for item in content):
                    if msg.role in (ASSISTANT, FUNCTION):
                        content = [ContentItem(text=upload_str)]
                    else:
                        content = [ContentItem(text=upload_str)] + content
    else:
        raise TypeError

    return Message(
        role=msg.role,
        content=content,
        reasoning_content=msg.reasoning_content,
        name=msg.name if msg.role == FUNCTION else None,
        function_call=msg.function_call,
        extra=msg.extra,
    )


def format_as_text_message(
    msg: Message,
    add_upload_info: bool,
    lang: Literal['auto', 'en', 'zh'] = 'auto',
) -> Message:
    msg = format_as_multimodal_message(
        msg,
        add_upload_info=add_upload_info,
        add_multimodel_upload_info=add_upload_info,
        add_audio_upload_info=add_upload_info,
        lang=lang,
    )
    text = ''
    for item in msg.content:
        if item.type == 'text':
            text += item.value
    msg.content = text
    return msg


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def extract_text_from_message(
    msg: Message,
    add_upload_info: bool,
    lang: Literal['auto', 'en', 'zh'] = 'auto',
) -> str:
    if isinstance(msg.content, list):
        text = format_as_text_message(msg, add_upload_info=add_upload_info, lang=lang).content
    elif isinstance(msg.content, str):
        text = msg.content
    else:
        raise TypeError(f'List of str or str expected, but received {type(msg.content).__name__}.')
    return text.strip()


def extract_files_from_messages(messages: List[Message], include_images: bool) -> List[str]:
    files = []
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if item.file and item.file not in files:
                    files.append(item.file)
                if include_images and item.image and item.image not in files:
                    files.append(item.image)
    return files


def extract_images_from_messages(messages: List[Message]) -> List[str]:
    files = []
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                if item.image and item.image not in files:
                    files.append(item.image)
    return files


def extract_urls(text: str) -> List[str]:
    return re.findall(re.compile(r'https?://\S+'), text)


def extract_markdown_urls(md_text: str) -> List[str]:
    return re.findall(r'!?\[[^\]]*\]\(([^\)]+)\)', md_text)


# ---------------------------------------------------------------------------
# Message-list queries
# ---------------------------------------------------------------------------


def has_chinese_messages(messages: List[Union[Message, dict]], check_roles: Tuple[str] = (SYSTEM, USER)) -> bool:
    for m in messages:
        if m['role'] in check_roles:
            if has_chinese_chars(m['content']):
                return True
    return False


def get_last_usr_msg_idx(messages: List[Union[dict, Message]]) -> int:
    i = len(messages) - 1
    while (i >= 0) and (messages[i]['role'] != 'user'):
        i -= 1
    assert i >= 0, messages
    assert messages[i]['role'] == 'user'
    return i


def rm_default_system(messages: List[Message]) -> List[Message]:
    if len(messages) > 1 and messages[0].role == SYSTEM:
        if isinstance(messages[0].content, str):
            if messages[0].content.strip() == DEFAULT_SYSTEM_MESSAGE:
                return messages[1:]
            else:
                return messages
        elif isinstance(messages[0].content, list):
            if len(messages[0].content) == 1 and messages[0].content[0].text.strip() == DEFAULT_SYSTEM_MESSAGE:
                return messages[1:]
            else:
                return messages
        else:
            raise TypeError
    else:
        return messages


# ---------------------------------------------------------------------------
# Deprecated -- kept for backward compatibility
# ---------------------------------------------------------------------------


def build_text_completion_prompt(
    messages: List[Message],
    allow_special: bool = False,
    default_system: str = DEFAULT_SYSTEM_MESSAGE,
) -> str:
    logger.warning('Support for `build_text_completion_prompt` is deprecated. '
                   'Please use `tokenizer.apply_chat_template(...)` instead to construct the prompt from messages.')

    im_start = '<|im_start|>'
    im_end = '<|im_end|>'

    if messages and messages[0].role == SYSTEM:
        sys = messages[0].content
        assert isinstance(sys, str)
        prompt = f'{im_start}{SYSTEM}\n{sys}{im_end}'
        messages = messages[1:]
    elif default_system:
        prompt = f'{im_start}{SYSTEM}\n{default_system}{im_end}'
    else:
        prompt = ''

    if messages[-1].role != ASSISTANT:
        messages = messages + [Message(ASSISTANT, '')]

    for msg in messages:
        assert isinstance(msg.content, str)
        content = msg.content
        if allow_special:
            assert msg.role in (USER, ASSISTANT, SYSTEM, FUNCTION)
            if msg.function_call:
                assert msg.role == ASSISTANT
                tool_call = msg.function_call.arguments
                try:
                    tool_call = {'name': msg.function_call.name, 'arguments': json.loads(tool_call)}
                    tool_call = json.dumps(tool_call, ensure_ascii=False, indent=2)
                except json.decoder.JSONDecodeError:
                    tool_call = '{"name": "' + msg.function_call.name + '", "arguments": ' + tool_call + '}'
                if content:
                    content += '\n'
                content += f'<tool_call>\n{tool_call}\n</tool_call>'
        else:
            assert msg.role in (USER, ASSISTANT)
            assert msg.function_call is None
        if prompt:
            prompt += '\n'
        prompt += f'{im_start}{msg.role}\n{content}{im_end}'

    assert prompt.endswith(im_end)
    prompt = prompt[:-len(im_end)]
    return prompt
