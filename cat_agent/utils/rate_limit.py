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

Concurrency cap and event loops
-------------------------------
The token bucket is guarded by a ``threading.Lock`` and is therefore correct
across threads and across event loops.

``max_concurrency`` is enforced by a semaphore. Async waiters use an
``asyncio.Semaphore``, which is bound to a single event loop. A limiter that
enforces ``max_concurrency`` therefore refuses to serve async waiters from a
second event loop: silently creating a per-loop semaphore would let
``max_concurrency=1`` admit one caller *per loop*, quietly breaking the cap.

If you need one concurrency cap shared across several event loops, pass
``cross_loop_concurrency=True``. The cap is then enforced by a
``threading.Semaphore`` acquired off-thread, which is loop-agnostic at the
cost of a thread-pool hop per acquisition.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterator, Mapping, Optional, Union

# Waits shorter than this are scheduling noise, not real contention.
_WAIT_EPSILON_SECONDS = 0.001


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
        cross_loop_concurrency: bool = False,
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
        self.cross_loop_concurrency = bool(cross_loop_concurrency)

        self._lock = threading.Lock()
        self._capacity = float(requests_per_interval) if requests_per_interval else 0.0
        self._tokens = self._capacity
        self._refill_rate = (
            float(requests_per_interval) / self.interval_seconds
            if requests_per_interval
            else 0.0
        )
        self._last_refill = time.monotonic()

        self._sync_sem = threading.Semaphore(max_concurrency) if max_concurrency else None
        # Async semaphore is created lazily on first use and pinned to that loop.
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
            cross_loop_concurrency=bool(raw.get('cross_loop_concurrency', False)),
        )

    # ------------------------------------------------------------------
    # Token bucket
    # ------------------------------------------------------------------

    def _refill_unlocked(self) -> None:
        if not self._refill_rate:
            return
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

    def _reserve_token(self) -> float:
        """Reserve exactly one token; return seconds to wait before using it.

        The token is deducted immediately, so the balance may go negative. Each
        waiter therefore holds a distinct place in line and sleeps only for the
        debt it created. This avoids the thundering herd (and possible
        starvation) of many waiters racing for the same token on wake-up.
        """
        if not self._refill_rate:
            return 0.0
        with self._lock:
            self._refill_unlocked()
            self._tokens -= 1.0
            if self._tokens >= 0.0:
                return 0.0
            return -self._tokens / self._refill_rate

    def _return_token(self) -> None:
        """Give back a reserved token when the acquisition is aborted."""
        if not self._refill_rate:
            return
        with self._lock:
            self._tokens = min(self._capacity, self._tokens + 1.0)

    # ------------------------------------------------------------------
    # Concurrency semaphores
    # ------------------------------------------------------------------

    def _async_semaphore(self) -> Optional[asyncio.Semaphore]:
        """Return this limiter's asyncio semaphore, pinned to one event loop.

        Raises:
            RuntimeError: if called from a different event loop than the one the
              semaphore was created on and ``cross_loop_concurrency`` is False.
        """
        if self.max_concurrency is None or self.cross_loop_concurrency:
            return None
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._async_sem is None:
                self._async_sem = asyncio.Semaphore(self.max_concurrency)
                self._async_sem_loop = loop
            elif self._async_sem_loop is not loop:
                raise RuntimeError(
                    'This RateLimiter enforces max_concurrency and is already bound to a '
                    'different event loop. Sharing it across loops would enforce the cap '
                    'once per loop instead of globally. Use a separate limiter per loop, '
                    'or construct it with cross_loop_concurrency=True.'
                )
            return self._async_sem

    def _record_wait(self, waited: float) -> float:
        if waited < _WAIT_EPSILON_SECONDS:
            return 0.0
        with self._lock:
            self.stats.waits += 1
            self.stats.wait_seconds += waited
        return waited

    # ------------------------------------------------------------------
    # Sync API
    # ------------------------------------------------------------------

    def acquire(self) -> float:
        """Block until a slot and token are available. Returns seconds waited."""
        waited = 0.0
        if self._sync_sem is not None:
            t0 = time.monotonic()
            self._sync_sem.acquire()
            waited += time.monotonic() - t0
        try:
            delay = self._reserve_token()
            if delay > 0:
                time.sleep(delay)
                waited += delay
        except BaseException:
            if self._sync_sem is not None:
                self._sync_sem.release()
            raise
        return self._record_wait(waited)

    def release(self) -> None:
        if self._sync_sem is not None:
            self._sync_sem.release()

    @contextmanager
    def limit(self) -> Iterator[float]:
        waited = self.acquire()
        try:
            yield waited
        finally:
            self.release()

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def _acquire_async_locked(self) -> tuple[float, Optional[asyncio.Semaphore], bool]:
        """Acquire slot + token.

        Returns:
            (seconds_waited, asyncio_semaphore_held, sync_semaphore_held)
        """
        waited = 0.0
        sem: Optional[asyncio.Semaphore] = None
        sync_held = False

        if self.max_concurrency is not None:
            t0 = time.monotonic()
            if self.cross_loop_concurrency:
                # Loop-agnostic cap: block a worker thread, not the event loop.
                assert self._sync_sem is not None
                await asyncio.to_thread(self._sync_sem.acquire)
                sync_held = True
            else:
                sem = self._async_semaphore()
                assert sem is not None
                await sem.acquire()
            waited += time.monotonic() - t0

        try:
            delay = self._reserve_token()
            if delay > 0:
                await asyncio.sleep(delay)
                waited += delay
        except BaseException:
            # Cancelled or errored while waiting on the token: hand back both the
            # reserved token and the concurrency slot.
            self._return_token()
            if sem is not None:
                sem.release()
            if sync_held and self._sync_sem is not None:
                self._sync_sem.release()
            raise

        return waited, sem, sync_held

    async def acquire_async(self) -> float:
        """Await a slot and token without blocking the event loop.

        Prefer :meth:`limit_async`, which releases the concurrency slot for you.
        This method releases nothing; the caller must call :meth:`release` (for
        ``cross_loop_concurrency``) or hold the returned semaphore itself.
        """
        waited, _sem, _sync_held = await self._acquire_async_locked()
        return self._record_wait(waited)

    async def release_async(self) -> None:
        """Release a slot taken by :meth:`acquire_async`.

        Only meaningful when ``cross_loop_concurrency`` is True; the loop-bound
        ``asyncio.Semaphore`` path is released by :meth:`limit_async`, which
        holds the exact semaphore object it acquired.
        """
        if self.max_concurrency is None:
            return
        if self.cross_loop_concurrency and self._sync_sem is not None:
            self._sync_sem.release()
            return
        sem = self._async_semaphore()
        if sem is not None:
            sem.release()

    @asynccontextmanager
    async def limit_async(self) -> AsyncIterator[float]:
        waited, sem, sync_held = await self._acquire_async_locked()
        recorded = self._record_wait(waited)
        try:
            yield recorded
        finally:
            # Release the exact semaphore acquired above — never re-resolve it,
            # which could release a semaphore that was never acquired.
            if sem is not None:
                sem.release()
            if sync_held and self._sync_sem is not None:
                self._sync_sem.release()


def rate_limiter_for_tool(tool: Any) -> Optional[RateLimiter]:
    cfg = getattr(tool, 'cfg', None) or {}
    if not isinstance(cfg, Mapping):
        return None
    if 'rate_limiter' in cfg:
        return RateLimiter.from_cfg(cfg.get('rate_limiter'))
    if 'rate_limit' in cfg:
        return RateLimiter.from_cfg(cfg.get('rate_limit'))
    return None