"""Opt-in per-tool retry configuration.

Retry is **off by default**. Enable only for tools that are safe to re-invoke
after a failure. Tools that perform non-idempotent side effects (payments,
writes, sending mail) must not enable retry: a failed attempt may have
partially succeeded, and cat-agent has no protocol for detecting that.
"""

from __future__ import annotations

import asyncio
import builtins
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Type

from cat_agent.tools.base import ToolExecutionError, ToolNotFoundError, ToolServiceError


def _resolve_exception_types(names: Sequence[Any]) -> Tuple[Type[BaseException], ...]:
    resolved = []
    for item in names:
        if isinstance(item, type) and issubclass(item, BaseException):
            resolved.append(item)
            continue
        if not isinstance(item, str):
            raise TypeError(f'retryable_exceptions entries must be types or names, got {item!r}')
        cand = getattr(builtins, item, None)
        if cand is None or not (isinstance(cand, type) and issubclass(cand, BaseException)):
            raise ValueError(f'Unknown exception type name for retry config: {item!r}')
        resolved.append(cand)
    return tuple(resolved)


@dataclass(frozen=True)
class ToolRetryConfig:
    """Per-tool retry policy. Construct via :meth:`from_cfg` or explicitly.

    Attributes:
        max_attempts: Total attempts including the first. ``1`` means no retry.
        retryable_exceptions: Exception types (or underlying ``__cause__`` types
            for ``ToolExecutionError``) that trigger another attempt.
        initial_delay: Backoff delay before the second attempt (seconds).
        exponential_base: Multiplier applied after each failed attempt.
        max_delay: Cap on backoff delay (seconds).
    """

    max_attempts: int = 1
    retryable_exceptions: Tuple[Type[BaseException], ...] = (Exception,)
    initial_delay: float = 1.0
    exponential_base: float = 2.0
    max_delay: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError('max_attempts must be >= 1')

    @classmethod
    def from_cfg(cls, raw: Optional[Dict[str, Any]]) -> Optional['ToolRetryConfig']:
        """Parse ``tool.cfg['retry']``. Returns ``None`` when retry is unset/off."""
        if raw is None or raw is False:
            return None
        if raw is True:
            return cls(max_attempts=3)
        if not isinstance(raw, dict):
            # Defensive: mocks or unexpected types mean "no retry" rather than crash.
            return None
        max_attempts = int(raw.get('max_attempts', 3))
        if max_attempts <= 1:
            return None
        retryable = raw.get('retryable_exceptions')
        if retryable is None:
            types: Tuple[Type[BaseException], ...] = (Exception,)
        else:
            types = _resolve_exception_types(list(retryable))
        return cls(
            max_attempts=max_attempts,
            retryable_exceptions=types,
            initial_delay=float(raw.get('initial_delay', 1.0)),
            exponential_base=float(raw.get('exponential_base', 2.0)),
            max_delay=float(raw.get('max_delay', 60.0)),
        )

    def is_retryable(self, exc: BaseException) -> bool:
        """Return whether *exc* should trigger another attempt."""
        if isinstance(exc, asyncio.CancelledError):
            return False
        if isinstance(exc, ToolNotFoundError):
            return False
        # Bare ToolServiceError (hard) / DocParserError are not retryable;
        # ToolExecutionError is soft and may be retried based on its cause.
        if isinstance(exc, ToolServiceError) and not isinstance(exc, ToolExecutionError):
            return False
        # DocParserError is imported lazily to avoid a heavy import cycle.
        from cat_agent.tools.simple_doc_parser import DocParserError
        if isinstance(exc, DocParserError):
            return False
        probe: BaseException = exc
        if isinstance(exc, ToolExecutionError) and exc.__cause__ is not None:
            probe = exc.__cause__
        return isinstance(probe, self.retryable_exceptions)


def retry_config_for_tool(tool: Any) -> Optional[ToolRetryConfig]:
    """Read retry config from a tool instance; ``None`` means no retry."""
    cfg = getattr(tool, 'cfg', None) or {}
    return ToolRetryConfig.from_cfg(cfg.get('retry'))
