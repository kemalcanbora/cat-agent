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

"""Shared job execution core — used by APScheduler and the k8s oneshot driver."""

from __future__ import annotations

import re
import time
import uuid
from typing import Awaitable, Callable, List, Optional, Sequence

from cat_agent.observability.context import run_context
from cat_agent.observability.emitter import emit, resolve_handlers
from cat_agent.observability.events import AgentEvent
from cat_agent.scheduling.models import Job, JobKind, JobRun, JobRunStatus, Source
from cat_agent.scheduling.store import JobStore, new_run_id
from cat_agent.settings import (
    SCHEDULER_BACKOFF_CAP_MULTIPLIER,
    SCHEDULER_JOB_LIMIT,
    SCHEDULER_LEASE_SECONDS,
    SCHEDULER_MAX_REPORT_ITEMS,
)

_SECRET_RE = re.compile(
    r'(?i)(password|passwd|secret|token|api[_-]?key|authorization)\s*[=:]\s*\S+'
)
_BEARER_RE = re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._\-+=/]+')
_DEFAULT_INTERVAL = 3600.0
_ERROR_MAX_CHARS = 500


CollectFn = Callable[[Job, JobStore], Awaitable[int]]
ReportFn = Callable[[Job, Sequence[Source], JobStore], Awaitable[str]]
DeliverFn = Callable[[Job, str, Sequence[Source], JobStore], Awaitable[None]]


def scrub_error(message: str, *, max_chars: int = _ERROR_MAX_CHARS) -> str:
    """Strip secrets from error strings before persisting to ``job_runs``."""
    text = _SECRET_RE.sub(r'\1=[REDACTED]', message or '')
    text = _BEARER_RE.sub('Bearer [REDACTED]', text)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + '...'
    return text


def compute_next_run_at(
    job: Job,
    *,
    now: float,
    consecutive_failures: int,
    backoff_cap_multiplier: int = SCHEDULER_BACKOFF_CAP_MULTIPLIER,
) -> float:
    """Advance the scheduling watermark; apply capped exponential backoff on failure."""
    base = float(job.interval_seconds) if job.interval_seconds else _DEFAULT_INTERVAL
    if consecutive_failures <= 0:
        return now + base
    factor = min(2 ** (consecutive_failures - 1), max(1, backoff_cap_multiplier))
    return now + base * factor


def claim_due_jobs(
    store: JobStore,
    *,
    limit: int,
    lease_seconds: int,
    owner: str,
    now: Optional[float] = None,
) -> List[Job]:
    """Atomically lease due jobs. Multi-process safe (delegates to JobStore)."""
    return store.claim_due_jobs(
        limit=limit,
        lease_seconds=lease_seconds,
        owner=owner,
        now=now if now is not None else time.time(),
    )


async def _default_collect(job: Job, store: JobStore) -> int:
    from cat_agent.scheduling.graph import run_collector

    return await run_collector(job, store)


async def _default_report(job: Job, sources: Sequence[Source], store: JobStore) -> str:
    from cat_agent.scheduling.graph import build_report_markdown

    return await build_report_markdown(job, sources)


async def _default_deliver(
    job: Job,
    body_markdown: str,
    sources: Sequence[Source],
    store: JobStore,
) -> None:
    from cat_agent.scheduling.graph import deliver_report

    await deliver_report(job, body_markdown, sources, store)


async def execute_job(
    job_id: str,
    *,
    store: JobStore,
    now: Optional[float] = None,
    owner: Optional[str] = None,
    lease_seconds: int = SCHEDULER_LEASE_SECONDS,
    max_items: int = SCHEDULER_MAX_REPORT_ITEMS,
    collect_fn: Optional[CollectFn] = None,
    report_fn: Optional[ReportFn] = None,
    deliver_fn: Optional[DeliverFn] = None,
    dry_run: bool = False,
    handlers: Optional[list] = None,
) -> JobRun:
    """Run one job end-to-end. Driver-agnostic.

    Must be safe to call concurrently from different processes — the caller
    should hold a lease, or pass ``owner`` so this function acquires one.
    """
    started = time.time() if now is None else now
    job = store.get_job(job_id)
    run_id = new_run_id()

    if job is None or not job.enabled:
        run = JobRun(
            id=run_id,
            job_id=job_id,
            started_at=started,
            finished_at=started,
            status=JobRunStatus.SKIPPED.value,
            sources_count=0,
            error='job missing or disabled',
        )
        store.insert_run(run)
        return run

    lease_owner = owner or job.lease_owner or f'exec-{uuid.uuid4().hex[:8]}'
    if job.lease_owner != lease_owner:
        store.force_lease(
            job_id,
            owner=lease_owner,
            lease_until=started + lease_seconds,
            now=started,
        )

    run = JobRun(
        id=run_id,
        job_id=job_id,
        started_at=started,
        status=JobRunStatus.RUNNING.value,
    )
    store.insert_run(run)

    collect = collect_fn or _default_collect
    report = report_fn or _default_report
    deliver = deliver_fn or _default_deliver
    failures = job.consecutive_failures
    sources_count = 0
    status = JobRunStatus.OK.value
    error: Optional[str] = None
    dry_run_body: Optional[str] = None
    raised: Optional[BaseException] = None

    resolved = resolve_handlers(handlers)

    try:
        with run_context(
            agent_name=f'schedule:{job_id}',
            agent_class='SchedulerJob',
            handlers=resolved,
        ) as ctx:
            with store.engine.begin() as conn:
                conn.execute(
                    store._sa['update'](store.job_runs)
                    .where(store.job_runs.c.id == run_id)
                    .values(trace_id=ctx.trace_id)
                )

            emit(AgentEvent.run_start(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                message_count=0,
                lang='en',
                input=job.topic,
            ))
            store.renew_lease(
                job_id, owner=lease_owner, lease_until=time.time() + lease_seconds,
            )

            kind = JobKind(job.kind)
            if kind.includes_collection():
                await collect(job, store)

            if kind.includes_report():
                undelivered = store.list_undelivered(
                    job.user_id, max_items=max_items,
                )
                sources_count = len(undelivered)
                if not undelivered:
                    status = JobRunStatus.SKIPPED_EMPTY.value
                else:
                    body = await report(job, undelivered, store)
                    if dry_run:
                        dry_run_body = body
                        status = JobRunStatus.OK.value
                    else:
                        await deliver(job, body, undelivered, store)
                        store.mark_delivered([s.id for s in undelivered])
                        status = JobRunStatus.OK.value
            else:
                sources_count = len(
                    store.list_undelivered(job.user_id, max_items=max_items)
                )
                status = JobRunStatus.OK.value

            failures = 0
            emit(AgentEvent.run_end(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                duration_ms=(time.time() - started) * 1000.0,
                yield_count=1,
                output=status,
            ))
    except BaseException as exc:
        raised = exc
        failures = job.consecutive_failures + 1
        status = JobRunStatus.FAILED.value
        error = scrub_error(f'{type(exc).__name__}: {exc}')
        try:
            with run_context(
                agent_name=f'schedule:{job_id}',
                agent_class='SchedulerJob',
                handlers=resolved,
            ) as ctx:
                emit(AgentEvent.run_error(
                    trace_id=ctx.trace_id,
                    run_id=ctx.run_id,
                    span_id=ctx.span_id,
                    parent_span_id=ctx.parent_span_id,
                    agent_name=ctx.agent_name,
                    agent_class=ctx.agent_class,
                    duration_ms=(time.time() - started) * 1000.0,
                    error_type=type(exc).__name__,
                    error_message=error,
                ))
        except Exception:
            pass

    finished = time.time()
    try:
        store.finish_run(
            run_id,
            status=status,
            finished_at=finished,
            sources_count=sources_count,
            error=dry_run_body[:_ERROR_MAX_CHARS] if dry_run and dry_run_body else error,
        )
    finally:
        next_at = compute_next_run_at(
            job, now=finished, consecutive_failures=failures,
        )
        store.update_schedule_state(
            job_id,
            next_run_at=next_at,
            last_run_at=finished,
            consecutive_failures=failures,
            clear_lease=True,
        )
        try:
            store.release_lease(job_id, owner=lease_owner)
        except Exception:
            pass

    finished_run = store.get_run(run_id)
    if finished_run is None:
        finished_run = JobRun(
            id=run_id,
            job_id=job_id,
            started_at=started,
            finished_at=finished,
            status=status,
            sources_count=sources_count,
            error=error,
        )

    if raised is not None:
        raise raised
    return finished_run


async def run_due_once(
    store: JobStore,
    *,
    owner: str,
    limit: int = SCHEDULER_JOB_LIMIT,
    lease_seconds: int = SCHEDULER_LEASE_SECONDS,
    now: Optional[float] = None,
    collect_fn: Optional[CollectFn] = None,
    report_fn: Optional[ReportFn] = None,
    deliver_fn: Optional[DeliverFn] = None,
) -> List[JobRun]:
    """Claim due jobs and execute them. Kubernetes CronJob entry point."""
    when = time.time() if now is None else now
    claimed = claim_due_jobs(
        store,
        limit=limit,
        lease_seconds=lease_seconds,
        owner=owner,
        now=when,
    )
    results: List[JobRun] = []
    for job in claimed:
        try:
            result = await execute_job(
                job.id,
                store=store,
                now=when,
                owner=owner,
                lease_seconds=lease_seconds,
                collect_fn=collect_fn,
                report_fn=report_fn,
                deliver_fn=deliver_fn,
            )
            results.append(result)
        except Exception:
            runs = store.list_runs(job.id, limit=1)
            if runs:
                results.append(runs[0])
            else:
                results.append(JobRun(
                    id=new_run_id(),
                    job_id=job.id,
                    started_at=when,
                    finished_at=time.time(),
                    status=JobRunStatus.FAILED.value,
                    error='execute_job raised without persisting a run',
                ))
    return results
