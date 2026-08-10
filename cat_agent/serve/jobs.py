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

"""In-memory inline job table for async agent runs."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

JobState = Literal['queued', 'running', 'succeeded', 'failed', 'cancelled']

TERMINAL_STATES = frozenset({'succeeded', 'failed', 'cancelled'})


class JobTableFull(Exception):
    """Raised when the in-memory job table cannot accept another job."""

    def __init__(self, *, max_jobs: int):
        self.max_jobs = max_jobs
        super().__init__(f'job table full (max_jobs={max_jobs})')


class JobNotFound(KeyError):
    """Unknown job id for an agent."""


@dataclass
class JobRecord:
    job_id: str
    agent: str
    state: JobState = 'queued'
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    task: Optional[asyncio.Task] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            'job_id': self.job_id,
            'agent': self.agent,
            'state': self.state,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }
        if self.finished_at is not None:
            out['finished_at'] = self.finished_at
        if self.state == 'succeeded' and self.result is not None:
            out['result'] = self.result
        if self.state in ('failed', 'cancelled'):
            if self.error_type:
                out['error_type'] = self.error_type
            if self.error:
                out['error'] = self.error
        return out


RunCallable = Callable[[], Awaitable[Any]]


class InlineJobTable:
    """Bounded in-memory jobs with TTL eviction of finished records."""

    def __init__(
        self,
        *,
        max_jobs: int = 256,
        finished_ttl_seconds: float = 600.0,
    ):
        self.max_jobs = max(1, int(max_jobs))
        self.finished_ttl_seconds = max(0.0, float(finished_ttl_seconds))
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    def _evict_locked(self, now: float) -> None:
        expired = [
            jid
            for jid, rec in self._jobs.items()
            if rec.state in TERMINAL_STATES
            and rec.finished_at is not None
            and (now - rec.finished_at) >= self.finished_ttl_seconds
        ]
        for jid in expired:
            self._jobs.pop(jid, None)

    async def submit(self, agent: str, runner: RunCallable) -> str:
        async with self._lock:
            now = time.time()
            self._evict_locked(now)
            if len(self._jobs) >= self.max_jobs:
                raise JobTableFull(max_jobs=self.max_jobs)
            job_id = uuid.uuid4().hex
            rec = JobRecord(job_id=job_id, agent=agent, state='queued', created_at=now, updated_at=now)
            self._jobs[job_id] = rec

        task = asyncio.create_task(self._run(job_id, runner), name=f'inline-job-{job_id}')
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].task = task
        return job_id

    async def _run(self, job_id: str, runner: RunCallable) -> None:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None or rec.state == 'cancelled':
                return
            rec.state = 'running'
            rec.updated_at = time.time()
        try:
            result = await runner()
        except asyncio.CancelledError:
            async with self._lock:
                rec = self._jobs.get(job_id)
                if rec is not None:
                    now = time.time()
                    rec.state = 'cancelled'
                    rec.error_type = 'Cancelled'
                    rec.error = 'job cancelled'
                    rec.finished_at = now
                    rec.updated_at = now
            raise
        except Exception as exc:
            async with self._lock:
                rec = self._jobs.get(job_id)
                if rec is not None and rec.state not in TERMINAL_STATES:
                    now = time.time()
                    rec.state = 'failed'
                    rec.error_type = type(exc).__name__
                    rec.error = str(exc)
                    rec.finished_at = now
                    rec.updated_at = now
            return
        async with self._lock:
            rec = self._jobs.get(job_id)
            if rec is not None and rec.state not in TERMINAL_STATES:
                now = time.time()
                rec.state = 'succeeded'
                rec.result = result
                rec.finished_at = now
                rec.updated_at = now

    async def get(self, agent: str, job_id: str) -> JobRecord:
        async with self._lock:
            self._evict_locked(time.time())
            rec = self._jobs.get(job_id)
            if rec is None or rec.agent != agent:
                raise JobNotFound(job_id)
            return rec

    async def cancel(self, agent: str, job_id: str) -> JobRecord:
        async with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None or rec.agent != agent:
                raise JobNotFound(job_id)
            task = rec.task
            if rec.state in TERMINAL_STATES:
                return rec
            now = time.time()
            rec.state = 'cancelled'
            rec.error_type = 'Cancelled'
            rec.error = 'job cancelled'
            rec.finished_at = now
            rec.updated_at = now
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        return rec

    async def shutdown(self) -> None:
        """Cancel active jobs and mark them cancelled before process teardown."""
        async with self._lock:
            active = [
                (jid, rec)
                for jid, rec in self._jobs.items()
                if rec.state not in TERMINAL_STATES
            ]
        for _jid, rec in active:
            await self.cancel(rec.agent, rec.job_id)

    def snapshot(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._jobs.values()]
