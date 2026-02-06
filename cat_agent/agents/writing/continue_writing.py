# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
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

import copy
from typing import Iterator, List

from cat_agent import Agent
from cat_agent.llm.schema import CONTENT, Message

PROMPT_TEMPLATE_ZH = """You are a writing assistant. Please continue writing appropriate content based on the reference materials and the given preceding text.
# References:
{ref_doc}

# Preceding Text:
{user_request}

Ensure that the continued content remains consistent with the preceding text. Please start continuing:"""

PROMPT_TEMPLATE_EN = """You are a writing assistant, please follow the reference materials and continue to write appropriate content based on the given previous text.

# References:
{ref_doc}

# Previous text:
{user_request}

Please start writing directly, output only the continued text, do not repeat the previous text, do not say irrelevant words, and ensure that the continued content and the previous text remain consistent."""

PROMPT_TEMPLATE = {
    'zh': PROMPT_TEMPLATE_ZH,
    'en': PROMPT_TEMPLATE_EN,
}


class ContinueWriting(Agent):

    def _run(self, messages: List[Message], knowledge: str = '', lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        messages[-1][CONTENT] = PROMPT_TEMPLATE[lang].format(
            ref_doc=knowledge,
            user_request=messages[-1][CONTENT],
        )
        return self._call_llm(messages)
