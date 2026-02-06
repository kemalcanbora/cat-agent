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
from typing import Dict, Iterator, List, Optional, Union

from cat_agent import Agent
from cat_agent.llm.base import BaseChatModel
from cat_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, Message
from cat_agent.tools import BaseTool
from cat_agent.utils.utils import merge_generate_cfgs


class GenKeyword(Agent):
    PROMPT_TEMPLATE_ZH = """Please extract keywords from the question, both in Chinese and English, and supplement them appropriately with relevant keywords that are not in the question. Try to divide keywords into verbs, nouns, or adjectives as individual words, and avoid long phrases (the aim is to better match and retrieve semantically related but differently phrased relevant information). Keywords are provided in JSON format, such as {{"keywords_zh": ["keyword 1", "keyword 2"], "keywords_en": ["keyword 1", "keyword 2"]}}

Question: Who is the author of this article?
Keywords: {{"keywords_zh": ["Author"], "keywords_en": ["author"]}}
Observation: ...

Question: Explain Figure 1
Keywords: {{"keywords_zh": ["Figure 1", "Fig 1"], "keywords_en": ["Figure 1"]}}
Observation: ...

Question: Core formula
Keywords: {{"keywords_zh": ["Core formula", "formula"], "keywords_en": ["core formula", "formula", "equation"]}}
Observation: ...

Question: {user_request}
Keywords:
"""

    PROMPT_TEMPLATE_EN = """Please extract keywords from the question, both in Chinese and English, and supplement them appropriately with relevant keywords that are not in the question.
Try to divide keywords into verb, noun, or adjective types and avoid long phrases (The aim is to better match and retrieve semantically related but differently phrased relevant information).
Keywords are provided in JSON format, such as {{"keywords_zh": ["Keyword 1", "Keyword 2"], "keywords_en": ["keyword 1", "keyword 2"]}}

Question: Who are the authors of this article?
Keywords: {{"keywords_zh": ["Author"], "keywords_en": ["author"]}}
Observation: ...

Question: Explain Figure 1
Keywords: {{"keywords_zh": ["Figure 1", "Fig 1"], "keywords_en": ["Figure 1"]}}
Observation: ...

Question: core formula
Keywords: {{"keywords_zh": ["Core formula", "formula"], "keywords_en": ["core formula", "formula", "equation"]}}
Observation: ...

Question: {user_request}
Keywords:
"""

    PROMPT_TEMPLATE = {
        'zh': PROMPT_TEMPLATE_ZH,
        'en': PROMPT_TEMPLATE_EN,
    }

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 **kwargs):
        super().__init__(function_list, llm, system_message, **kwargs)
        self.extra_generate_cfg = merge_generate_cfgs(
            base_generate_cfg=self.extra_generate_cfg,
            new_generate_cfg={'stop': ['Observation:']},
        )

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        messages[-1][CONTENT] = self.PROMPT_TEMPLATE[lang].format(user_request=messages[-1].content)
        return self._call_llm(messages=messages)
