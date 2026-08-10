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

"""Helpers to turn agent ``arun`` turns into JSON / SSE payloads."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union


def messages_to_dicts(messages: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            out.append(msg)
        elif hasattr(msg, 'model_dump'):
            out.append(msg.model_dump())
        else:
            out.append(dict(msg))
    return out


def final_content(messages: List[Union[Dict[str, Any], Any]]) -> Optional[str]:
    if not messages:
        return None
    last = messages[-1]
    content = last.get('content') if isinstance(last, dict) else getattr(last, 'content', None)
    if content is None:
        return None
    if isinstance(content, str):
        return content
    return str(content)


def sse_event(payload: Dict[str, Any]) -> str:
    return f'data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n'
