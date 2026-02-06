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
from cat_agent.llm import BaseChatModel
from cat_agent.llm.schema import Message, SYSTEM
from cat_agent.tools import BaseTool
from cat_agent.utils.utils import has_chinese_chars


class GroupChatAutoRouter(Agent):
    PROMPT_TEMPLATE_ZH = '''You are the game master of a role-play game, your task is to choose the appropriate role to speak. The roles are as follows:
{agent_descs}

The format of the dialogue history between roles is as follows, with newer dialogues being more important:
Role Name: Speech Content

Please read the dialogue history and choose the next suitable role to speak from [{agent_names}]. When the real user has recently indicated they want to stop chatting, or when the topic should be terminated, please return '[STOP]'. Users are lazy, do not choose the real user unless necessary.
Only return the role name or '[STOP]', do not return any other content.'''

    PROMPT_TEMPLATE_EN = '''You are in a role play game. The following roles are available:
{agent_descs}

The format of dialogue history between roles is as follows:
Role Name: Speech Content

Please read the dialogue history and choose the next suitable role to speak.
When the user indicates to stop chatting or when the topic should be terminated, please return '[STOP]'.
Only return the role name from [{agent_names}] or '[STOP]'. Do not reply any other content.'''

    PROMPT_TEMPLATE = {
        'zh': PROMPT_TEMPLATE_ZH,
        'en': PROMPT_TEMPLATE_EN,
    }

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 agents: List[Agent] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs):
        # This agent need prepend special system message according to inputted agents
        agent_descs = '\n'.join([f'{x.name}: {x.description}' for x in agents])
        lang = 'en'
        if has_chinese_chars(agent_descs):
            lang = 'zh'
        system_prompt = self.PROMPT_TEMPLATE[lang].format(agent_descs=agent_descs,
                                                          agent_names=', '.join([x.name for x in agents]))

        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_prompt,
                         name=name,
                         description=description,
                         **kwargs)

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        dialogue = [] # convert existing messages into a prompt
        for msg in messages:
            if msg.role == SYSTEM:
                continue
            if msg.role == 'function' or not msg.content:
                continue
            if isinstance(msg.content, list):
                content = '\n'.join([x.text if x.text else '' for x in msg.content]).strip()
            else:
                content = msg.content.strip()
            display_name = msg.role
            if msg.name:
                display_name = msg.name
            if dialogue and dialogue[-1].startswith(display_name):
                dialogue[-1] += f'\n{content}'
            else:
                dialogue.append(f'{display_name}: {content}')

        if not dialogue:
            dialogue.append('The conversation has just started, please choose any speaker, do not choose the real user')
        assert messages[0].role == SYSTEM
        new_messages = [copy.deepcopy(messages[0]), Message('user', '\n'.join(dialogue))]
        return self._call_llm(messages=new_messages)
