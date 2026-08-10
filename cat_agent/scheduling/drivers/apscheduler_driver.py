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

"""In-process APScheduler driver (dev / single-node deploys)."""

from __future__ import annotations

import asyncio
import signal
import socket
import time
from typing import Optional, Set

from cat_agent.scheduling.runner import execute_job
from cat_agent.scheduling.store import JobStore, default_scheduler_dsn
from cat_agent.settings import SCHEDULER_LEASE_SECONDS
from cat_agent.tools.base import enable_optional_tools


def _require_apscheduler():
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "APScheduler driver requires the 'scheduler' extra. "
            "Install with: pip install 'cat-agent[scheduler]'"
        ) from exc
    return AsyncIOScheduler, SQLAlchemyJobStore, CronTrigger, IntervalTrigger


class APSchedulerDriver:
    """Long-lived AsyncIOScheduler wrapping the shared ``execute_job`` core."""

    def __init__(
        self,
        store: JobStore,
        *,
        dsn: Optional[str] = None,
        sync_interval_seconds: int = 60,
        lease_seconds: int = SCHEDULER_LEASE_SECONDS,
        owner: Optional[str] = None,
    ):
        self.store = store
        self.dsn = dsn or store.dsn or default_scheduler_dsn()
        self.sync_interval_seconds = sync_interval_seconds
        self.lease_seconds = lease_seconds
        self.owner = owner or f'apsched-{socket.gethostname()}'
        self._scheduler = None
        self._known: Set[str] = set()
        self._stopping = False

    async def start(self) -> None:
        AsyncIOScheduler, SQLAlchemyJobStore, _, _ = _require_apscheduler()
        enable_optional_tools('web_search', 'web_extractor')
        jobstores = {
            'default': SQLAlchemyJobStore(url=self.dsn),
        }
        self._scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            timezone='UTC',
            job_defaults={
                'coalesce': True,
                'max_instances': 1,
                'misfire_grace_time': 1800,
            },
        )
        self._scheduler.start()
        await self.sync_jobs()
        self._scheduler.add_job(
            self.sync_jobs,
            'interval',
            seconds=self.sync_interval_seconds,
            id='__sched_sync__',
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.stop()))
            except NotImplementedError:
                # Windows / restricted loops
                signal.signal(sig, lambda *a: asyncio.create_task(self.stop()))

    async def sync_jobs(self) -> None:
        """Reconcile DB jobs → APScheduler jobs."""
        if self._scheduler is None:
            return
        _, _, CronTrigger, IntervalTrigger = _require_apscheduler()
        jobs = self.store.list_jobs(enabled_only=True)
        seen: Set[str] = set()
        for job in jobs:
            ap_id = f'sched:{job.id}'
            seen.add(ap_id)
            if job.cron_expr:
                trigger = CronTrigger.from_crontab(job.cron_expr, timezone=job.timezone or 'UTC')
            else:
                seconds = int(job.interval_seconds or 3600)
                trigger = IntervalTrigger(seconds=seconds, timezone=job.timezone or 'UTC')
            self._scheduler.add_job(
                self._run_job,
                trigger=trigger,
                id=ap_id,
                args=[job.id],
                replace_existing=True,
                coalesce=True,
                max_instances=1,
                misfire_grace_time=1800,
            )
        # Remove APScheduler jobs for deleted/disabled DB rows.
        for ap_id in list(self._known - seen):
            if ap_id == '__sched_sync__':
                continue
            try:
                self._scheduler.remove_job(ap_id)
            except Exception:
                pass
        self._known = seen | {'__sched_sync__'}

    async def _run_job(self, job_id: str) -> None:
        """Scheduled callable — always takes a lease, then execute_job."""
        if self._stopping:
            return
        now = time.time()
        claimed = self.store.claim_due_jobs(
            limit=1,
            lease_seconds=self.lease_seconds,
            owner=self.owner,
            now=now,
        )
        # Manual force-lease if the job was synced but next_run_at not yet due
        # (APScheduler fired on its own trigger).
        if not any(j.id == job_id for j in claimed):
            job = self.store.get_job(job_id)
            if job is None or not job.enabled:
                return
            if job.lease_until and job.lease_until > now and job.lease_owner != self.owner:
                return
            self.store.force_lease(
                job_id,
                owner=self.owner,
                lease_until=now + self.lease_seconds,
                now=now,
            )
        try:
            await execute_job(
                job_id,
                store=self.store,
                owner=self.owner,
                lease_seconds=self.lease_seconds,
            )
        except Exception:
            # execute_job already persisted failure + advanced next_run_at
            pass

    async def stop(self, *, wait_timeout: float = 30.0) -> None:
        self._stopping = True
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
        # Await in-flight briefly by sleeping; leases released below.
        await asyncio.sleep(min(1.0, wait_timeout))
        self.store.release_all_leases(self.owner)
