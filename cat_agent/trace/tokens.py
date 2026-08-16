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

"""Heuristic / backend token counting for traces when usage is missing."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Union

from cat_agent.llm.schema import Message
from cat_agent.utils.message_utils import extract_text_from_message


def estimate_message_tokens(messages: Sequence[Union[Message, dict]]) -> int:
    """Approximate token count via the bundled ``o200k_base`` tiktoken helper."""
    from cat_agent.utils.tokenization_qwen import count_tokens

    total = 0
    for msg in messages:
        if isinstance(msg, Message):
            text = extract_text_from_message(msg, add_upload_info=False)
        elif isinstance(msg, dict):
            content = msg.get('content', '')
            if isinstance(content, str):
                text = content
            else:
                text = str(content)
        else:
            text = str(msg)
        total += count_tokens(text or '')
    return total


def resolve_usage(
    messages_out: Optional[List[Message]],
    messages_in: Optional[List[Message]] = None,
    llm: Any = None,
) -> tuple[int, int, bool]:
    """Return ``(prompt_tokens, completion_tokens, estimated)``.

    Prefer backend-reported usage on message ``extra['usage']``. Fall back to
    tokenizer estimates and mark ``estimated=True``.
    """
    from cat_agent.observability.helpers import extract_usage

    usage = extract_usage(messages_out or [])
    if usage:
        prompt = int(usage.get('prompt_tokens') or usage.get('input_tokens') or 0)
        completion = int(usage.get('completion_tokens') or usage.get('output_tokens') or 0)
        if prompt or completion:
            return prompt, completion, False

    # Exact tokenizer when the backend exposes one.
    counter = getattr(llm, 'count_tokens', None) or getattr(llm, 'tokenize', None)
    if callable(counter) and messages_in is not None:
        try:
            if hasattr(llm, 'count_tokens'):
                prompt = int(llm.count_tokens(messages_in))
            else:
                prompt = estimate_message_tokens(messages_in)
        except Exception:
            prompt = estimate_message_tokens(messages_in or [])
    else:
        prompt = estimate_message_tokens(messages_in or [])

    completion = estimate_message_tokens(messages_out or [])
    return prompt, completion, True
