"""Shareable rate limiter (token bucket + optional concurrency cap).

Ownership
---------
Construct explicitly and pass into agents / tool configs. There is **no**
process-wide global. Lifetime is whatever owns the instance (typically the
application or a shared client wrapper). Multiple agents may share one
limiter when they hit the same upstream API.

Unconfigured means no ``RateLimiter`` object is attached — call sites skip
the limiter entirely (zero overhead).

Sync ``acquire`` / ``limit()`` may block the calling thread. Async
``acquire_async`` / ``limit_async()`` wait with ``asyncio.sleep`` and never
block the event loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, Mapping, Optional, Union


@dataclass
class _LimiterStats:
    waits: int = 0
    wait_seconds: float = 0.0


class RateLimiter:
    """Token-bucket rate limiter with optional max in-flight concurrency."""

    def __init__(
        self,
        *,
        requests_per_interval: Optional[float] = None,
        interval_seconds: float = 1.0,
        max_concurrency: Optional[int] = None,
    ) -> None:
        if requests_per_interval is None and max_concurrency is None:
            raise ValueError('RateLimiter requires requests_per_interval and/or max_concurrency')
        if interval_seconds <= 0:
            raise ValueError('interval_seconds must be > 0')
        if requests_per_interval is not None and requests_per_interval <= 0:
            raise ValueError('requests_per_interval must be > 0')
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError('max_concurrency must be >= 1')

        self.requests_per_interval = requests_per_interval
        self.interval_seconds = float(interval_seconds)
        self.max_concurrency = max_concurrency

        self._lock = threading.Lock()
        self._tokens = float(requests_per_interval) if requests_per_interval else 0.0
        self._capacity = float(requests_per_interval) if requests_per_interval else 0.0
        self._refill_rate = (
            float(requests_per_interval) / self.interval_seconds
            if requests_per_interval
            else 0.0
        )
        self._last_refill = time.monotonic()

        self._sync_sem = threading.Semaphore(max_concurrency) if max_concurrency else None
        self._async_sem: Optional[asyncio.Semaphore] = None
        self._async_sem_loop: Optional[asyncio.AbstractEventLoop] = None
        self.stats = _LimiterStats()

    @classmethod
    def from_cfg(cls, raw: Union['RateLimiter', Mapping[str, Any], None]) -> Optional['RateLimiter']:
        if raw is None:
            return None
        if isinstance(raw, RateLimiter):
            return raw
        if not isinstance(raw, Mapping):
            return None
        return cls(
            requests_per_interval=raw.get('requests_per_interval'),
            interval_seconds=float(raw.get('interval_seconds', 1.0)),
            max_concurrency=raw.get('max_concurrency'),
        )

    def _refill_unlocked(self) -> None:
        if not self._refill_rate:
            return
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def _claim_token_or_wait(self) -> float:
        """Return 0 if a token was taken; otherwise seconds until one is available."""
        if not self._refill_rate:
            return 0.0
        with self._lock:
            self._refill_unlocked()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            needed = 1.0 - self._tokens
            return needed / self._refill_rate

    def _ensure_async_sem(self) -> Optional[asyncio.Semaphore]:
        if self.max_concurrency is None:
            return None
        loop = asyncio.get_running_loop()
        if self._async_sem is None or self._async_sem_loop is not loop:
            self._async_sem = asyncio.Semaphore(self.max_concurrency)
            self._async_sem_loop = loop
        return self._async_sem

    def acquire(self) -> float:
        """Block until a slot+token are available. Returns seconds spent waiting."""
        waited = 0.0
        if self._sync_sem is not None:
            self._sync_sem.acquire()
        try:
            while True:
                delay = self._claim_token_or_wait()
                if delay <= 0:
                    break
                time.sleep(delay)
                waited += delay
        except BaseException:
            if self._sync_sem is not None:
                self._sync_sem.release()
            raise
        if waited:
            self.stats.waits += 1
            self.stats.wait_seconds += waited
        return waited

    def release(self) -> None:
        if self._sync_sem is not None:
            self._sync_sem.release()

    async def acquire_async(self) -> float:
        """Await until a slot+token are available without blocking the event loop."""
        waited = 0.0
        sem = self._ensure_async_sem()
        if sem is not None:
            await sem.acquire()
        try:
            while True:
                delay = self._claim_token_or_wait()
                if delay <= 0:
                    break
                await asyncio.sleep(delay)
                waited += delay
        except BaseException:
            if sem is not None:
                sem.release()
            raise
        if waited:
            self.stats.waits += 1
            self.stats.wait_seconds += waited
        return waited

    async def release_async(self) -> None:
        sem = self._ensure_async_sem()
        if sem is not None:
            sem.release()

    @contextmanager
    def limit(self) -> Iterator[float]:
        waited = self.acquire()
        try:
            yield waited
        finally:
            self.release()

    @asynccontextmanager
    async def limit_async(self) -> AsyncIterator[float]:
        waited = await self.acquire_async()
        try:
            yield waited
        finally:
            await self.release_async()


def rate_limiter_for_tool(tool: Any) -> Optional[RateLimiter]:
    cfg = getattr(tool, 'cfg', None) or {}
    if not isinstance(cfg, Mapping):
        return None
    if 'rate_limiter' in cfg:
        return RateLimiter.from_cfg(cfg.get('rate_limiter'))
    if 'rate_limit' in cfg:
        return RateLimiter.from_cfg(cfg.get('rate_limit'))
    return None
