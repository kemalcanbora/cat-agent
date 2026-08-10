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

"""Fakes for platform Nomad / docker tests."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: Any = None, text: str = ''):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else (
            json.dumps(payload) if payload is not None else ''
        )
        self.content = self.text.encode() if self.text else b''

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.routes: Dict[str, Any] = {}

    def request(self, method, url, **kwargs):
        self.calls.append({'method': method, 'url': url, **kwargs})
        key = (method.upper(), url.rstrip('/').split('/v1', 1)[-1])
        # normalize path after /v1
        path = '/' + url.split('/v1/', 1)[-1] if '/v1/' in url else url
        handler = self.routes.get((method.upper(), path))
        if handler is None:
            # prefix match for job paths
            for (m, p), h in self.routes.items():
                if m == method.upper() and path.startswith(p.rstrip('*')):
                    handler = h
                    break
        if handler is None:
            return FakeResponse(404, text=f'no route {method} {path}')
        return handler(method, path, kwargs)
