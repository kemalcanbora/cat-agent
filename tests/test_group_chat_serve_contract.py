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

"""GroupChat satisfies the Agent surface expected by AgentRegistry."""

from __future__ import annotations

import inspect

from cat_agent.agents.group_chat import GroupChat
from cat_agent.agent import Agent


def test_group_chat_has_arun_nonstream_and_aclose():
    assert issubclass(GroupChat, Agent)
    assert hasattr(GroupChat, 'arun_nonstream')
    assert hasattr(GroupChat, 'aclose')
    assert inspect.iscoroutinefunction(GroupChat.arun_nonstream)
    assert inspect.iscoroutinefunction(GroupChat.aclose)
