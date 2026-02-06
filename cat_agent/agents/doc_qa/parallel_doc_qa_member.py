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
from cat_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, ROLE, SYSTEM, USER, ContentItem, Message
from cat_agent.tools import BaseTool
from cat_agent.utils.utils import merge_generate_cfgs

NO_RESPONSE = '<None>'
SYSTEM_PROMPT_TEMPLATE_ZH = """You are an expert at document-based QA, capable of answering user questions based on document content.


# Task Description:
Please read the provided document excerpt carefully and answer the question based on its content.
You need to determine whether the document content can answer the question; do not force an answer. If it can be answered, the answer must strictly follow the document content, even if it deviates from the facts.
If the answer deviates from the facts, provide the answer directly without explanation.


# Answering Rules:
- Please answer the question in the same language as the document materials.
- Assess whether the document contains enough information to answer the question. Do not answer if it's irrelevant.
- If the question can be answered, your answer must strictly follow the document content, even if it deviates from the facts. Never provide extra explanations.
- If the question can be answered, directly quote relevant information from the document to ensure the answer is accurate, complete, and strive for conciseness.
- When only a small amount of information in the document is related to the question, focus on that part of the information; be sure to answer in this case.


# Answer Format:
Please provide the answer in JSON format.


## Examples:
When the document content is irrelevant:
{{"res": "none", "content": "{no_response}"}},
Observation: ...

When the document content is answerable and the document is in Chinese:
{{"res": "ans", "content": "Your Answer"}}
Observation: ...

When the document content is answerable and the document is in English:
{{"res": "ans", "content": "[Your Answer]"}}
Observation: ..."""

SYSTEM_PROMPT_TEMPLATE_EN = """You are an expert in document-based question answering, capable of answering user questions based on document content.


# Task Description:
Please read the provided document excerpt carefully and answer questions based on its content.
You must assess whether the document content allows for the questions to be answered, without forcing a response.
If the answer does not align with the facts, provide it directly without explanation.


# Answering Rules:
- Reply in the same language as the source material.
- Evaluate whether the document contains sufficient information to answer the question. Do not respond if it's irrelevant.
- If the question can be answered, your answer must strictly follow the document content, even if it does not align with the facts.
- If the question can be answered, directly quote the relevant information from the document to ensure the answer is accurate, complete, and strive for conciseness.
- When the document contains only minimal information related to the question, focus on this information and be sure to answer.


# Answer Format:
Please provide answers in the form of JSON.


## Examples
When the document content is irrelevant:
{{"res": "none", "content": "{no_response}"}},
Observation: ...

When the document content can provide an answer:
{{"res": "ans", "content": "[Your Answer]"}}
Observation: ..."""

SYSTEM_PROMPT_TEMPLATE = {
    'zh': SYSTEM_PROMPT_TEMPLATE_ZH,
    'en': SYSTEM_PROMPT_TEMPLATE_EN,
}

PROMPT_TEMPLATE_ZH = """# Document:
{ref_doc}

# Question:
{instruction}

Please provide your answer according to the answering rules:"""

PROMPT_TEMPLATE_EN = """# Document:
{ref_doc}

# Question:
{instruction}

Please provide your answer according to the answering rules:"""

PROMPT_TEMPLATE = {
    'zh': PROMPT_TEMPLATE_ZH,
    'en': PROMPT_TEMPLATE_EN,
}


class ParallelDocQAMember(Agent):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 **kwargs):
        super().__init__(function_list, llm, system_message, **kwargs)
        self.extra_generate_cfg = merge_generate_cfgs(
            base_generate_cfg=self.extra_generate_cfg,
            new_generate_cfg={'stop': ['Observation:', 'Observation:\n']},
        )

    def _run(self,
             messages: List[Message],
             knowledge: str = '',
             lang: str = 'en',
             instruction: str = None,
             **kwargs) -> Iterator[List[Message]]:

        messages = copy.deepcopy(messages)

        system_prompt = SYSTEM_PROMPT_TEMPLATE[lang].format(no_response=NO_RESPONSE)
        if messages and messages[0][ROLE] == SYSTEM:
            if isinstance(messages[0][CONTENT], str):
                messages[0][CONTENT] += '\n\n' + system_prompt
            else:
                assert isinstance(messages[0][CONTENT], list)
                messages[0][CONTENT] += [ContentItem(text='\n\n' + system_prompt)]
        else:
            messages.insert(0, Message(SYSTEM, system_prompt))

        assert len(messages) > 0, messages
        assert messages[-1][ROLE] == USER, messages
        prompt = PROMPT_TEMPLATE[lang].format(ref_doc=knowledge, instruction=instruction)

        messages[-1] = Message(USER, prompt)
        return self._call_llm(messages=messages)
