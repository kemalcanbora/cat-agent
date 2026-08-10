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

"""Run a FastAPI serve app with uvicorn."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

# Sentinel: argument omitted (use env / settings). Distinct from explicit None.
_UNSET: Any = object()


def _resolve_host(host: object) -> str:
    if host is _UNSET:
        return os.getenv('CAT_AGENT_SERVE_HOST') or '127.0.0.1'
    if host is None:
        raise ValueError(
            'host=None is invalid; omit the argument to use CAT_AGENT_SERVE_HOST / default'
        )
    return str(host)


def _resolve_port(port: object) -> int:
    if port is _UNSET:
        if 'CAT_AGENT_SERVE_PORT' in os.environ and os.environ['CAT_AGENT_SERVE_PORT'].strip():
            return int(os.environ['CAT_AGENT_SERVE_PORT'])
        # Orchestrators (Heroku, Nomad alloc, many Docker templates) set PORT.
        if 'PORT' in os.environ and os.environ['PORT'].strip():
            return int(os.environ['PORT'])
        from cat_agent.settings import SERVE_PORT
        return int(SERVE_PORT)
    if port is None:
        raise ValueError(
            'port=None is invalid; omit the argument to use CAT_AGENT_SERVE_PORT / PORT / default'
        )
    return int(port)


def _resolve_shutdown_timeout(timeout_graceful_shutdown: object) -> int:
    if timeout_graceful_shutdown is _UNSET:
        from cat_agent.settings import SERVE_SHUTDOWN_TIMEOUT
        return int(SERVE_SHUTDOWN_TIMEOUT)
    if timeout_graceful_shutdown is None:
        raise ValueError(
            'timeout_graceful_shutdown=None is invalid; omit the argument to use '
            'CAT_AGENT_SERVE_SHUTDOWN_TIMEOUT'
        )
    return int(timeout_graceful_shutdown)


def run_app(
    app: Any,
    *,
    host: Union[str, None, object] = _UNSET,
    port: Union[int, None, object] = _UNSET,
    workers: int = 1,
    log_level: str = 'info',
    timeout_graceful_shutdown: Union[int, float, None, object] = _UNSET,
) -> None:
    """Block and serve *app* (typically from :func:`create_app`).

    When *host* / *port* / *timeout_graceful_shutdown* are omitted, values come
    from environment / settings (``CAT_AGENT_SERVE_*``, with ``PORT`` as a port
    fallback). Explicit ``None`` is rejected so it cannot silently wipe defaults.

    ``workers > 1`` is refused when the app's registry used
    :meth:`~cat_agent.serve.registry.AgentRegistry.register_factory` — uvicorn
    forks before lifespan, so deferred builds must not be multi-worker.
    """
    resolved_host = _resolve_host(host)
    resolved_port = _resolve_port(port)
    resolved_timeout = _resolve_shutdown_timeout(timeout_graceful_shutdown)
    if workers < 1:
        raise ValueError('workers must be >= 1')

    registry = getattr(getattr(app, 'state', None), 'registry', None)
    if workers > 1 and registry is not None and getattr(registry, 'has_deferred_factories', False):
        raise RuntimeError(
            'workers>1 is not supported when AgentRegistry.register_factory() was used: '
            'uvicorn forks workers before lifespan runs, so deferred agents would not be '
            'built safely. Use workers=1, or register agents eagerly with register().'
        )

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "cat-agent serve requires uvicorn. Install with: pip install 'cat-agent[serve]'"
        ) from exc

    uvicorn.run(
        app,
        host=resolved_host,
        port=resolved_port,
        workers=workers,
        log_level=log_level,
        timeout_graceful_shutdown=resolved_timeout,
        access_log=False,
    )
