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

"""Safe client-facing error bodies for the serve API."""

from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

from cat_agent.log import logger
from cat_agent.security.pii import redact_text

GENERIC_ERROR = 'agent run failed'


def verbose_errors_enabled() -> bool:
    from cat_agent.settings import SERVE_VERBOSE_ERRORS
    return bool(SERVE_VERBOSE_ERRORS)


def wire_error_message(exc: BaseException, *, verbose: Optional[bool] = None) -> str:
    """Message safe to return to clients.

    Default: generic text. Verbose: ``str(exc)`` after :func:`redact_text`.
    """
    if verbose is None:
        verbose = verbose_errors_enabled()
    if not verbose:
        return GENERIC_ERROR
    return redact_text(str(exc))


def error_body(
    agent: str,
    exc: BaseException,
    *,
    request_id: Optional[str] = None,
    verbose: Optional[bool] = None,
) -> Dict[str, Any]:
    """Keep the public shape ``{agent, error_type, error}`` (+ ``request_id``)."""
    body: Dict[str, Any] = {
        'agent': agent,
        'error_type': type(exc).__name__,
        'error': wire_error_message(exc, verbose=verbose),
    }
    if request_id is not None:
        body['request_id'] = request_id
    return body


def log_run_exception(
    exc: BaseException,
    *,
    agent: str,
    request_id: str,
) -> None:
    """Always log full detail server-side (independent of verbose wire mode)."""
    logger.opt(exception=exc).error(
        'serve: agent={} request_id={} error_type={}: {}',
        agent,
        request_id,
        type(exc).__name__,
        str(exc),
    )
    logger.debug(
        'serve: agent={} request_id={} traceback:\n{}',
        agent,
        request_id,
        ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
