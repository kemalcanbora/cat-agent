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

"""Request-id middleware and structured access logging for serve."""

from __future__ import annotations

import uuid
from typing import Any, Callable, Optional

from cat_agent.log import logger

REQUEST_ID_HEADER = 'x-request-id'


def new_request_id() -> str:
    return str(uuid.uuid4())


def resolve_request_id(incoming: Optional[str]) -> str:
    value = (incoming or '').strip()
    return value or new_request_id()


def log_access(
    *,
    request_id: str,
    agent: str,
    outcome: str,
    status: int,
    duration_ms: float,
    queue_wait_ms: float,
    stream: bool,
) -> None:
    """One structured line per ``/agents/{name}/run`` — never log prompt/response."""
    logger.info(
        'serve_access request_id={} agent={} outcome={} status={} '
        'duration_ms={:.1f} queue_wait_ms={:.1f} stream={}',
        request_id,
        agent,
        outcome,
        status,
        duration_ms,
        queue_wait_ms,
        stream,
    )


def install_request_id_middleware(app: Any) -> None:
    """Honour inbound ``x-request-id`` (or generate), store on ``request.state``, echo."""
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cat-agent serve requires starlette/fastapi. Install with: pip install 'cat-agent[serve]'"
        ) from exc

    class RequestIdMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next: Callable):
            rid = resolve_request_id(request.headers.get(REQUEST_ID_HEADER))
            request.state.request_id = rid
            response = await call_next(request)
            response.headers[REQUEST_ID_HEADER] = rid
            return response

    app.add_middleware(RequestIdMiddleware)
