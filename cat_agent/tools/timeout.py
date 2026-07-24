"""Per-tool and whole-run timeout helpers.

Two distinct timeout concepts
-----------------------------
``timeout`` (tool cfg / ``call(..., timeout=)``)
    Tool-owned. For ``code_interpreter`` this arms the in-kernel
    ``_M6CountdownTimer`` (SIGALRM). The agent layer does not interpret this
    key as an agent-level deadline.

``attempt_timeout`` (tool cfg)
    Agent-owned, per-attempt wall clock around ``tool.acall`` on the **async**
    path (``asyncio.wait_for``). Worst-case with retry ≈
    ``max_attempts * attempt_timeout + backoff``.

    On the **sync** path the agent layer cannot interrupt a blocking
    ``tool.call`` without spawning a thread (explicitly rejected). A configured
    ``attempt_timeout`` therefore emits a warning and is not enforced as an
    agent wait; if the tool accepts a ``timeout`` parameter it is still
    forwarded so tool-owned timers (e.g. code_interpreter) remain effective.

Clock alignment for code_interpreter
------------------------------------
When ``attempt_timeout`` is set, the agent also passes ``timeout=attempt_timeout``
into tools whose ``call`` signature accepts ``timeout``. That keeps the kernel
timer and ``asyncio.wait_for`` on the same budget instead of racing two
independent clocks. Prefer the kernel error when it fires first; ``wait_for``
is the backstop if the kernel hang does not surface.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Dict, Mapping, Optional


_SYNC_ATTEMPT_TIMEOUT_WARNED: set = set()


def attempt_timeout_for_tool(tool: Any) -> Optional[float]:
    """Return agent-layer per-attempt timeout in seconds, or ``None``."""
    cfg = getattr(tool, 'cfg', None) or {}
    if not isinstance(cfg, Mapping):
        return None
    raw = cfg.get('attempt_timeout')
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def warn_sync_attempt_timeout(tool_name: str, attempt_timeout: float) -> None:
    """Warn once per tool name that sync agent-layer timeout is a no-op."""
    if tool_name in _SYNC_ATTEMPT_TIMEOUT_WARNED:
        return
    _SYNC_ATTEMPT_TIMEOUT_WARNED.add(tool_name)
    warnings.warn(
        f'attempt_timeout={attempt_timeout} for tool {tool_name!r} is not enforceable '
        f'on the sync tool path (blocking calls cannot be interrupted without a worker '
        f'thread). The agent-layer wait is ignored; if the tool accepts a timeout= '
        f'parameter it will still be forwarded for the tool\'s own timer.',
        UserWarning,
        stacklevel=3,
    )


def prepare_tool_call_kwargs(
    tool: Any,
    kwargs: Dict[str, Any],
    attempt_timeout: Optional[float],
) -> Dict[str, Any]:
    """Copy kwargs and, when appropriate, forward ``timeout=`` for clock alignment."""
    call_kwargs = dict(kwargs)
    if attempt_timeout is None or 'timeout' in call_kwargs:
        return call_kwargs
    try:
        sig = inspect.signature(tool.call)
    except (TypeError, ValueError):
        return call_kwargs
    if 'timeout' not in sig.parameters:
        return call_kwargs
    # Prefer int seconds when the value is integral (code_interpreter expects int).
    timeout_val: Any = int(attempt_timeout) if float(attempt_timeout).is_integer() else attempt_timeout
    call_kwargs['timeout'] = timeout_val
    return call_kwargs


def format_tool_timeout_error(tool_name: str, attempt_timeout: float) -> str:
    return (
        f'An error occurred when calling tool `{tool_name}`:\n'
        f'TimeoutError: Tool timed out after {attempt_timeout}s '
        f'(attempt_timeout)'
    )
