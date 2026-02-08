"""Post-processing of LLM output: stop-word truncation, text normalisation."""

import copy
from typing import List

from cat_agent.llm.schema import Message
from cat_agent.utils.message_utils import format_as_text_message
from cat_agent.utils.tokenization_qwen import tokenizer


def format_as_text_messages(messages: List[Message]) -> List[Message]:
    """Ensure every message has string-only content (no multimodal items)."""
    for msg in messages:
        if isinstance(msg.content, list):
            for item in msg.content:
                assert item.type == 'text'
        else:
            assert isinstance(msg.content, str)
    return [format_as_text_message(msg, add_upload_info=False) for msg in messages]


def postprocess_stop_words(messages: List[Message], stop: List[str]) -> List[Message]:
    """Truncate output at stop words and remove trailing partial stop words."""
    messages = copy.deepcopy(messages)
    if not messages:
        return messages

    # Truncate at first occurrence of any stop word
    trunc_messages = []
    for msg in messages:
        truncated = False
        trunc_content = []
        for item in msg.content:
            item_type, item_text = item.get_type_and_value()
            if item_type == 'text':
                truncated, item.text = _truncate_at_stop_word(text=item_text, stop=stop)
            trunc_content.append(item)
            if truncated:
                break
        msg.content = trunc_content
        trunc_messages.append(msg)
        if truncated:
            break
    messages = trunc_messages

    # Remove partial stop words at the end (e.g. "Observation" when stop word is "Observation:")
    partial_stop = []
    for s in stop:
        tokens = tokenizer.tokenize(s)[:-1]
        if tokens:
            partial_stop.append(tokenizer.convert_tokens_to_string(tokens))
    partial_stop = sorted(set(partial_stop))

    if messages:
        last_msg = messages[-1].content
        for i in range(len(last_msg) - 1, -1, -1):
            item_type, item_text = last_msg[i].get_type_and_value()
            if item_type == 'text':
                for s in partial_stop:
                    if item_text.endswith(s):
                        last_msg[i].text = item_text[:-len(s)]
                break

    return messages


def _truncate_at_stop_word(text: str, stop: List[str]):
    truncated = False
    for s in stop:
        k = text.find(s)
        if k >= 0:
            truncated = True
            text = text[:k]
    return truncated, text


def rm_think(text: str) -> str:
    """Strip ``<think>...</think>`` reasoning blocks from model output."""
    if '</think>' in text:
        return text.split('</think>')[-1].lstrip('\n')
    return text
