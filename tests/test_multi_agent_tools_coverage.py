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

"""Coverage for multi-agent hub tools + keygen strategy."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cat_agent.agents.keygen_strategies.split_query_then_gen_keyword import SplitQueryThenGenKeyword
from cat_agent.llm.schema import ASSISTANT, USER, Message
from cat_agent.multi_agent.tools import (
    AskAgentTool,
    HandoffTool,
    ReadArtifactTool,
    WriteArtifactTool,
    _parse_key_content_args,
)


def test_ask_and_handoff_tools():
    hub = MagicMock()
    hub.handle_ask.return_value = 'answer'
    ask = AskAgentTool(hub, 'alice')
    assert ask.call({'name': 'bob', 'question': 'q?'}) == 'answer'
    hub.handle_ask.assert_called_once()

    handoff = HandoffTool(hub, 'alice')
    out = handoff.call({'to': 'bob', 'context': 'brief'})
    assert 'bob' in out
    hub.set_pending_handoff.assert_called_once()


def test_write_read_artifact_and_parse_recovery():
    hub = MagicMock()
    hub.blackboard.write.return_value = 'artifact:k'
    hub.blackboard.read.side_effect = ['hello', KeyError('missing'), {'a': 1}]

    write = WriteArtifactTool(hub, 'alice')
    assert 'artifact:k' in write.call({'key': 'k', 'content': 'body', 'summary': 's'})
    assert write.call({'key': 'only'}) == 'Error: write_artifact requires key and content.'

    # Recover almost-JSON via _verify failure path
    messy = '{"key": "k2", "content": "line1\\nline2", "summary": "sum"}'
    with patch.object(WriteArtifactTool, '_verify_json_format_args', side_effect=ValueError('bad')):
        assert 'artifact:k' in write.call(messy)

    read = ReadArtifactTool(hub, 'alice')
    assert read.call({'key': 'k'}) == 'hello'
    assert read.call({'key': 'nope'}).startswith('Error:')
    assert read.call({'key': 'obj'}) == "{'a': 1}"

    parsed = _parse_key_content_args(
        '{"key": "x", "content": "hello\\"world\\n", "summary": "s"}'
    )
    assert parsed['key'] == 'x'
    assert 'hello' in parsed['content']
    assert parsed['summary'] == 's'
    assert _parse_key_content_args({'key': 'a', 'content': 'b'}) == {'key': 'a', 'content': 'b'}
    with pytest.raises(ValueError):
        _parse_key_content_args('not-json')


def test_split_query_then_gen_keyword_paths():
    agent = SplitQueryThenGenKeyword.__new__(SplitQueryThenGenKeyword)
    agent.llm = MagicMock()

    split = MagicMock()
    split.run.return_value = iter([
        [Message(ASSISTANT, '```json\n{"information": ["short"]}\n```')],
    ])
    keygen = MagicMock()
    keygen.run.return_value = iter([
        [Message(ASSISTANT, '```json\n{"kw": ["a"]}\n```')],
    ])
    agent.split_query = split
    agent.keygen = keygen

    out = list(agent._run([Message(USER, 'long query text here')]))
    assert out
    assert '"text"' in out[-1][0].content

    # Invalid split JSON → keep original query; invalid keyword → no final yield
    split.run.return_value = iter([[Message(ASSISTANT, 'not-json')]])
    keygen.run.return_value = iter([[Message(ASSISTANT, 'not-json')]])
    out2 = list(agent._run([Message(USER, 'q')]))
    assert len(out2) == 1
