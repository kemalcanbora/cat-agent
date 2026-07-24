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

"""Shared scratch / artifact store for multi-agent hubs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Artifact:
    key: str
    value: Any
    author: str
    summary: str = ''


@dataclass
class Blackboard:
    """Hub-level shared scratch available to all orchestration patterns."""

    _store: Dict[str, Artifact] = field(default_factory=dict)

    def write(self, key: str, value: Any, *, author: str, summary: str = '') -> str:
        if not summary:
            summary = _default_summary(value)
        self._store[key] = Artifact(key=key, value=value, author=author, summary=summary)
        return f'artifact:{key}'

    def read(self, key: str) -> Any:
        key = _strip_prefix(key)
        if key not in self._store:
            raise KeyError(f'No artifact named "{key}" on the blackboard.')
        return self._store[key].value

    def get(self, key: str, default: Any = None) -> Any:
        key = _strip_prefix(key)
        art = self._store.get(key)
        return default if art is None else art.value

    def keys(self) -> List[str]:
        return list(self._store.keys())

    def describe(self) -> str:
        if not self._store:
            return '(blackboard empty)'
        lines = []
        for key, art in self._store.items():
            lines.append(f'- artifact:{key} (by {art.author}): {art.summary}')
        return '\n'.join(lines)

    def clear(self) -> None:
        self._store.clear()


def _strip_prefix(key: str) -> str:
    if key.startswith('artifact:'):
        return key[len('artifact:'):]
    return key


def _default_summary(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = text.replace('\n', ' ').strip()
    if len(text) > 80:
        return text[:77] + '...'
    return text or '(empty)'
