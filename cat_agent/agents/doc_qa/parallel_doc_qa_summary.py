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

from cat_agent.agent import Agent
from cat_agent.llm.schema import CONTENT, ROLE, SYSTEM, USER, ContentItem, Message
from cat_agent.utils.utils import extract_text_from_message

PROMPT_TEMPLATE_ZH = """Note: Your answer must strictly adhere to the content of the provided Knowledge Base, even if it deviates from the facts.
If the majority of the knowledge base is irrelevant to the question, but there are a few sentences directly related, please focus on these sentences and ensure a response.

# Knowledge Base

{ref_doc}"""

PROMPT_TEMPLATE_EN = """Please respond solely based on the content of the provided Knowledge Base.
Note: Your answer must strictly adhere to the content of the provided Knowledge Base, even if it deviates from the facts.
If the materials mainly contains content irrelevant to the question, with only a few sentences directly related, please focus on these sentences and ensure a response.

# Knowledge Base

{ref_doc}"""

PROMPT_TEMPLATE = {
    'zh': PROMPT_TEMPLATE_ZH,
    'en': PROMPT_TEMPLATE_EN,
}

PROMPT_END_TEMPLATE_ZH = """# Question
{question}


# Answering Guidelines
- Please respond solely based on the content of the provided Knowledge Base. Note: Your answer must strictly adhere to the content of the provided Knowledge Base, even if it deviates from the facts.
- If the majority of the knowledge base is irrelevant to the question, with only a few sentences directly related, please focus on these sentences and ensure a response.

Please follow the answering guidelines and answer the question based on the knowledge base:"""

PROMPT_END_TEMPLATE_EN = """# Question
{question}


# Answering Guidelines
- Please respond solely based on the content of the provided Knowledge Base.
- Note: Your answer must strictly adhere to the content of the provided Knowledge Base, even if it deviates from the facts.
- If the materials mainly contains content irrelevant to the question, with only a few sentences directly related, please focus on these sentences and ensure a response.

Please give your answer:"""

PROMPT_END_TEMPLATE = {
    'zh': PROMPT_END_TEMPLATE_ZH,
    'en': PROMPT_END_TEMPLATE_EN,
}


class ParallelDocQASummary(Agent):

    def _run(self, messages: List[Message], knowledge: str = '', lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)

        system_prompt = PROMPT_TEMPLATE[lang].format(ref_doc=knowledge)

        if messages and messages[0][ROLE] == SYSTEM:
            if isinstance(messages[0][CONTENT], str):
                messages[0][CONTENT] += '\n\n' + system_prompt
            else:
                assert isinstance(messages[0][CONTENT], list)
                messages[0][CONTENT] += [ContentItem(text='\n\n' + system_prompt)]
        else:
            messages.insert(0, Message(SYSTEM, system_prompt))

        assert messages[-1][ROLE] == USER, messages
        user_question = extract_text_from_message(messages[-1], add_upload_info=False)
        messages[-1] = Message(USER, PROMPT_END_TEMPLATE[lang].format(question=user_question))

        return self._call_llm(messages=messages)
