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

"""Tests for ParallelDocQASummary prompt assembly (no live LLM)."""

from unittest.mock import MagicMock

from cat_agent.agents.doc_qa.parallel_doc_qa_summary import (
    PROMPT_END_TEMPLATE,
    PROMPT_TEMPLATE,
    ParallelDocQASummary,
)
from cat_agent.llm.schema import SYSTEM, USER, Message


def test_prompt_templates_contain_placeholders():
    assert '{ref_doc}' in PROMPT_TEMPLATE['en']
    assert '{question}' in PROMPT_END_TEMPLATE['en']
    assert '{ref_doc}' in PROMPT_TEMPLATE['zh']


def test_summary_agent_injects_knowledge_into_system():
    llm = MagicMock()
    llm.chat.return_value = iter([[Message('assistant', 'ok')]])
    agent = ParallelDocQASummary(llm=llm, name='sum')
    msgs = [Message(SYSTEM, 'base'), Message(USER, 'What happened?')]
    out = list(agent._run(msgs, knowledge='KB FACT', lang='en'))
    assert out
    # Original list is deep-copied; system content gained knowledge block
    # Inspect what was sent to llm.chat
    assert llm.chat.called
    sent = llm.chat.call_args.kwargs.get('messages') or llm.chat.call_args.args[0]
    sys_text = sent[0].content if hasattr(sent[0], 'content') else sent[0]['content']
    assert 'KB FACT' in sys_text
    assert 'Knowledge Base' in sys_text or 'Knowledge' in sys_text
