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

from cat_agent.agents.keygen_strategies.gen_keyword import GenKeyword
from cat_agent.llm import BaseChatModel
from cat_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, Message
from cat_agent.settings import DEFAULT_MAX_INPUT_TOKENS
from cat_agent.tools import BaseTool
from cat_agent.utils.tokenization_qwen import count_tokens, tokenizer


class GenKeywordWithKnowledge(GenKeyword):
    PROMPT_TEMPLATE_ZH = """Extract Chinese or English keywords based on the question, no more than 10. You can appropriately supplement relevant keywords that are not in the question.
Please generate keywords based on the language style of the given reference material (the purpose is to facilitate matching keywords with the reference material).
Try to split keywords into phrases or single words of types such as verb/noun/adjective, and avoid long phrases.

Begin!
<Refs> Vocabulary list for this scenario:
{ref_doc}
...
Question: {user_request}
Thought: I should extract keywords from the question based on the vocabulary style in the reference material. Keywords are provided in JSON format: {{"keywords_zh": ["keyword 1", "keyword 2"], "keywords_en": ["keyword 1", "keyword 2"]}}
Keywords:
"""

    PROMPT_TEMPLATE_EN = """Extract keywords from questions in both Chinese and English. Additional relevant keywords that may not be directly present in the question can be supplemented appropriately.
Please generate the keywords in a style consistent with the language of the provided reference material, to facilitate matching with the reference content using these keywords.
Try to divide keywords into verb/noun/adjective types and avoid long phrases.

Begin！
<Refs> List of words for this scene:
{ref_doc}
...
Question: {user_request}
Thought: I should use the vocabulary in Resources to extract the key words of the question. Keywords are provided in JSON format: {{"keywords_zh": ["Keyword 1", "Keyword 2"], "keywords_en": ["keyword 1", "keyword 2"]}}.
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
        super().__init__(['extract_doc_vocabulary'] + (function_list or []), llm, system_message, **kwargs)

    def _run(self,
             messages: List[Message],
             files: Optional[List[str]] = None,
             lang: str = 'en',
             **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        files = files or []
        available_token = DEFAULT_MAX_INPUT_TOKENS - count_tokens(f'{self.PROMPT_TEMPLATE[lang]}') - count_tokens(
            messages[-1][CONTENT]) - 100

        voc = self._call_tool(
            'extract_doc_vocabulary',
            {'files': files},
        )
        available_voc = tokenizer.truncate(voc, max_token=available_token)

        messages[-1][CONTENT] = self.PROMPT_TEMPLATE[lang].format(ref_doc=available_voc,
                                                                  user_request=messages[-1][CONTENT])
        return self._call_llm(messages)
