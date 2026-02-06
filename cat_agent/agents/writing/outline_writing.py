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

PROMPT_TEMPLATE_ZH = """
You are a writing assistant. Your task is to fully understand the reference materials and complete the writing.
# References:
{ref_doc}

The writing title is: {user_request}

In order to complete the above writing task, please first list the outline. The reply should only contain the outline. All first-level headings of the outline should be numbered with Roman numerals. Write only based on the given reference materials and do not introduce other knowledge.
"""

PROMPT_TEMPLATE_EN = """
You are a writing assistant. Your task is to complete writing article based on reference materials.

# References:
{ref_doc}

The title is: {user_request}

In order to complete the above writing tasks, please provide an outline first. The reply only needs to include an outline. The first level titles of the outline are all counted in Roman numerals. Write only based on the given reference materials and do not introduce other knowledge.
"""

PROMPT_TEMPLATE = {
    'zh': PROMPT_TEMPLATE_ZH,
    'en': PROMPT_TEMPLATE_EN,
}


class OutlineWriting(Agent):

    def _run(self, messages: List[Message], knowledge: str = '', lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        messages[-1][CONTENT] = PROMPT_TEMPLATE[lang].format(
            ref_doc=knowledge,
            user_request=messages[-1][CONTENT],
        )
        return self._call_llm(messages)
