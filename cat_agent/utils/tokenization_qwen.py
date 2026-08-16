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

"""Heuristic tokenization via OpenAI ``o200k_base`` (tiktoken).

Module name and helpers keep the historical ``qwen`` naming for import
stability. Counts are approximate for non-OpenAI backends; prefer
``llm.count_tokens`` / ``BackendTokenCounter`` when an exact tokenizer exists.
"""

from __future__ import annotations

import os
import unicodedata
from typing import Collection, List, Set, Union

import tiktoken

DEFAULT_ENCODING = os.getenv('CAT_AGENT_TIKTOKEN_ENCODING', 'o200k_base')

# Historical names kept for tests / callers that import them.
ENDOFTEXT = '<|endoftext|>'
IMSTART = '<|im_start|>'
IMEND = '<|im_end|>'
SPECIAL_TOKENS_SET = {ENDOFTEXT, IMSTART, IMEND}


class HeuristicTokenizer:
    """Thin wrapper around a built-in tiktoken encoding."""

    def __init__(self, encoding_name: str | None = None, errors: str = 'replace'):
        self.encoding_name = encoding_name or DEFAULT_ENCODING
        self.errors = errors
        self.tokenizer = tiktoken.get_encoding(self.encoding_name)

    def __len__(self) -> int:
        return self.tokenizer.n_vocab

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def encode(self, text: str) -> List[int]:
        text = unicodedata.normalize('NFC', text or '')
        return self.tokenizer.encode(text, allowed_special='all', disallowed_special=())

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = 'all',
        disallowed_special: Union[Collection, str] = (),
    ) -> List[Union[bytes, str]]:
        text = unicodedata.normalize('NFC', text or '')
        ids = self.tokenizer.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )
        return [self.tokenizer.decode_single_token_bytes(i) for i in ids]

    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        text = ''
        temp = b''
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode('utf-8', errors=self.errors)
                    temp = b''
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError('token should only be of type bytes or str')
        if temp:
            text += temp.decode('utf-8', errors=self.errors)
        return text

    def count_tokens(self, text: str) -> int:
        return count_tokens(text)

    def truncate(
        self,
        text: str,
        max_token: int,
        start_token: int = 0,
        keep_both_sides: bool = False,
    ) -> str:
        if start_token:
            ids = self.encode(text)[start_token:]
            text = self.tokenizer.decode(ids)
        return truncate_tokens(text, max_token, keep_both_sides=keep_both_sides)


# Back-compat alias for older imports / type names.
QWenTokenizer = HeuristicTokenizer

tokenizer = HeuristicTokenizer()


def ensure_qwen_tokenizer() -> None:
    """Ensure the native o200k_base counter is ready (no vocab file)."""
    from cat_agent._native import init_qwen_tokenizer

    if not getattr(ensure_qwen_tokenizer, '_initialized', False):
        init_qwen_tokenizer('')
        ensure_qwen_tokenizer._initialized = True


def count_tokens(text: str) -> int:
    from cat_agent._native import count_qwen_tokens

    ensure_qwen_tokenizer()
    return count_qwen_tokens(text)


def truncate_tokens(text: str, max_token: int, keep_both_sides: bool = False) -> str:
    from cat_agent._native import truncate_qwen_text

    ensure_qwen_tokenizer()
    return truncate_qwen_text(text, max_token, keep_both_sides)
