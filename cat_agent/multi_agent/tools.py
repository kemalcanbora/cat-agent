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

"""Hub-bound tools injected into member agents (not globally registered)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Union

from cat_agent.multi_agent.handoff import Handoff
from cat_agent.tools.base import BaseTool

if TYPE_CHECKING:
    from cat_agent.multi_agent_hub import MultiAgentHub


class AskAgentTool(BaseTool):
    """Ask another agent in this group a self-contained question."""

    name = 'ask_agent'
    description = (
        'Ask another agent in this group a specific question and wait for its answer. '
        'Use this when you need information or a capability that another member has. '
        'The other agent only sees your question, not this conversation.'
    )
    parameters = [
        {
            'name': 'name',
            'type': 'string',
            'required': True,
            'description': 'Name of the agent to ask.',
        },
        {
            'name': 'question',
            'type': 'string',
            'required': True,
            'description': 'A self-contained question. Include all necessary context.',
        },
    ]

    def __init__(self, hub: 'MultiAgentHub', caller_name: str):
        self._hub = hub
        self._caller = caller_name
        super().__init__()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        return self._hub.handle_ask(
            caller=self._caller,
            target_name=params['name'],
            question=params['question'],
            **kwargs,
        )


class HandoffTool(BaseTool):
    """Transfer control to another agent; the caller is not resumed."""

    name = 'handoff'
    description = (
        'Hand the rest of this conversation to another agent. Use when you have '
        'finished your part (e.g. triage) and a specialist should talk to the user. '
        'You will not continue after this call.'
    )
    parameters = [
        {
            'name': 'to',
            'type': 'string',
            'required': True,
            'description': 'Name of the agent that should take over.',
        },
        {
            'name': 'context',
            'type': 'string',
            'required': False,
            'description': 'Optional briefing for the receiving agent.',
        },
    ]

    def __init__(self, hub: 'MultiAgentHub', caller_name: str):
        self._hub = hub
        self._caller = caller_name
        super().__init__()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        target = params['to']
        context = params.get('context')
        self._hub.set_pending_handoff(Handoff(to=target, context=context), caller=self._caller)
        return f'Handed off to {target}.'


class WriteArtifactTool(BaseTool):
    """Write a large artifact to the shared blackboard."""

    name = 'write_artifact'
    description = (
        'Store a large artifact (code, document, retrieved chunks) on the shared '
        'blackboard. Mention only the returned key in conversation so other agents '
        'can read it explicitly.'
    )
    parameters = [
        {
            'name': 'key',
            'type': 'string',
            'required': True,
            'description': 'Short identifier for the artifact (e.g. parser_v1).',
        },
        {
            'name': 'content',
            'type': 'string',
            'required': True,
            'description': 'The full artifact content to store.',
        },
        {
            'name': 'summary',
            'type': 'string',
            'required': False,
            'description': 'One-line summary shown in blackboard listings.',
        },
    ]

    def __init__(self, hub: 'MultiAgentHub', caller_name: str):
        self._hub = hub
        self._caller = caller_name
        super().__init__()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
        except ValueError:
            # LLMs often emit almost-JSON for long content; recover key/content.
            params = _parse_key_content_args(params)
        if 'key' not in params or 'content' not in params:
            return 'Error: write_artifact requires key and content.'
        ref = self._hub.blackboard.write(
            params['key'],
            params['content'],
            author=self._caller,
            summary=params.get('summary') or '',
        )
        return f'Wrote {ref}'


def _parse_key_content_args(params: Union[str, dict]) -> dict:
    if isinstance(params, dict):
        return params
    text = (params or '').strip()
    key_m = re.search(r'"key"\s*:\s*"([^"]+)"', text)
    # content may be truncated / unclosed — take remainder after "content":
    content_m = re.search(r'"content"\s*:\s*"(.*)', text, re.DOTALL)
    summary_m = re.search(r'"summary"\s*:\s*"([^"]*)"', text)
    if not key_m or not content_m:
        raise ValueError('Parameters must be formatted as a valid JSON!')
    content = content_m.group(1)
    # Trim trailing JSON junk: ","summary"... or "}...
    content = re.sub(r'"\s*,\s*"summary".*$', '', content, flags=re.DOTALL)
    content = re.sub(r'"\s*\}\s*$', '', content, flags=re.DOTALL)
    content = content.rstrip('"').replace('\\n', '\n').replace('\\"', '"')
    out = {'key': key_m.group(1), 'content': content}
    if summary_m:
        out['summary'] = summary_m.group(1)
    return out


class ReadArtifactTool(BaseTool):
    """Read an artifact previously stored on the shared blackboard by key."""

    name = 'read_artifact'
    description = 'Read an artifact previously stored on the shared blackboard by key.'
    parameters = [
        {
            'name': 'key',
            'type': 'string',
            'required': True,
            'description': 'Artifact key (with or without the artifact: prefix).',
        },
    ]

    def __init__(self, hub: 'MultiAgentHub', caller_name: str):
        self._hub = hub
        self._caller = caller_name
        super().__init__()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        try:
            value = self._hub.blackboard.read(params['key'])
        except KeyError as exc:
            return f'Error: {exc}'
        if isinstance(value, str):
            return value
        return str(value)
