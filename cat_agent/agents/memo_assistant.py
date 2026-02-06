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

import json5

from cat_agent.agents import Assistant
from cat_agent.llm import BaseChatModel
from cat_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER, ContentItem, Message
from cat_agent.tools import BaseTool

MEMORY_PROMPT = """
During the conversation, you can use the storage tool at any time to store information you think needs to be remembered, and you can also read historical information that may have been stored at any time.
This will help you remember certain important information during long conversations with users, such as user preferences, special information, or major events.
Regarding data storage and retrieval, here are two suggestions:
1. Keep the key for storing data concise and easy to understand, and you can use keywords of the recorded content;
2. If you forget what data has been stored, you can use scan to view what data has been recorded;

Here are all the information you have stored, so you can skip the operation of reading data specifically:
<info>
{storage_info}
</info>

Your memory is short-lived, please frequently call the tool to store or read important conversation content.
"""


class MemoAssistant(Assistant):

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None):
        function_list = function_list or []
        super().__init__(function_list=['storage'] + function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         files=files)

    def _run(self, messages: List[Message], lang: str = 'zh', knowledge: str = '', **kwargs) -> Iterator[List[Message]]:
        new_message = self._prepend_storage_info_to_sys(messages)
        new_message = self._truncate_dialogue_history(new_message)

        for rsp in super()._run(new_message, lang=lang, knowledge=knowledge, **kwargs):
            yield rsp

    def _prepend_storage_info_to_sys(self, messages: List[Message]) -> List[Message]:
        messages = copy.deepcopy(messages)
        all_kv = {}
        # Obtained from message, with the purpose of facilitating control of information volume
        for msg in messages:
            if msg.function_call and msg.function_call.name == 'storage':
                try:
                    param = json5.loads(msg.function_call.arguments)
                except Exception:
                    continue
                if param['operate'] in ['put', 'update']:
                    all_kv[param['key']] = param['value']
                elif param['operate'] == 'delete' and param['key'] in all_kv:
                    all_kv.pop(param['key'])
                else:
                    pass
        all_kv_str = '\n'.join([f'{k}: {v}' for k, v in all_kv.items()])
        sys_memory_prompt = MEMORY_PROMPT.format(storage_info=all_kv_str)
        if messages and messages[0].role == SYSTEM:
            if isinstance(messages[0].content, str):
                messages[0].content += '\n\n' + sys_memory_prompt
            else:
                assert isinstance(messages[0].content, list)
                messages[0].content += [ContentItem(text='\n\n' + sys_memory_prompt)]
        else:
            messages = [Message(role=SYSTEM, content=sys_memory_prompt)] + messages
        return messages

    def _truncate_dialogue_history(self, messages: List[Message]) -> List[Message]:
        # This simulates a very small window, retaining only the most recent three rounds of conversation
        new_messages = []
        available_turn = 400
        k = len(messages) - 1
        while k > -1:
            msg = messages[k]
            if available_turn == 0:
                break
            if msg.role == USER:
                available_turn -= 1
            new_messages = [msg] + new_messages
            k -= 1

        if k > -1 and messages and messages[0].role == SYSTEM:
            new_messages = [messages[0]] + new_messages

        return new_messages
