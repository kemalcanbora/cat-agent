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

"""Tests for cat_agent.utils.json_utils repair / encode helpers."""

from pydantic import BaseModel

from cat_agent.utils.json_utils import (
    extract_code,
    json_dumps_compact,
    json_dumps_pretty,
    json_loads,
)


class _Msg(BaseModel):
    role: str
    content: str


def test_json_loads_repairs_trailing_paren():
    assert json_loads('{"a": 1)') == {'a': 1}


def test_json_loads_repairs_extra_braces():
    assert json_loads('{"a": "x"}}') == {'a': 'x'}


def test_json_loads_repairs_trailing_bracket():
    assert json_loads('{"a": 1]') == {'a': 1}


def test_json_loads_escapes_newlines_in_strings():
    raw = '{\n  "content": "line1\nline2"\n}'
    assert json_loads(raw)['content'] == 'line1\nline2'


def test_json_loads_fenced_block():
    text = '```json\n{"k": 2}\n```'
    assert json_loads(text) == {'k': 2}


def test_json_loads_double_encoded_string():
    assert json_loads('"{\\"n\\": 3}"') == {'n': 3}


def test_json_dumps_pretty_and_compact_pydantic():
    obj = {'m': _Msg(role='user', content='hi')}
    pretty = json_dumps_pretty(obj)
    compact = json_dumps_compact(obj)
    assert '"role": "user"' in pretty
    assert '"content": "hi"' in compact
    assert '\n' in pretty
    assert '\n' not in compact


def test_extract_code_from_fence_and_json():
    assert extract_code('```python\nx = 1\n```').strip() == 'x = 1'
    assert extract_code('{"code": "y = 2"}') == 'y = 2'
