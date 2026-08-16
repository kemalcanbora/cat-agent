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

"""Tests for cat_agent.llm.base.postprocessing."""

from cat_agent.llm.base.postprocessing import (
    postprocess_stop_words,
    rm_think,
)
from cat_agent.llm.schema import ASSISTANT, ContentItem, Message


def test_rm_think_strips_block():
    text = '<think>secret</think>\nVisible answer'
    assert rm_think(text) == 'Visible answer'


def test_rm_think_passthrough():
    assert rm_think('plain') == 'plain'


def test_postprocess_stop_words_truncates():
    msg = Message(ASSISTANT, [ContentItem(text='hello Observation: more')])
    out = postprocess_stop_words([msg], stop=['Observation:'])
    text = out[0].content[0].text
    assert text == 'hello '
    assert 'more' not in text


def test_postprocess_stop_words_empty():
    assert postprocess_stop_words([], stop=['x']) == []
