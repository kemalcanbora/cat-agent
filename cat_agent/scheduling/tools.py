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

"""LLM-facing scheduling tools (save_source, create_schedule, …)."""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional
from urllib.parse import urlsplit

from cat_agent.scheduling.models import KNOWN_CHANNELS, Job
from cat_agent.scheduling.store import JobStore, default_scheduler_dsn, make_job_id
from cat_agent.settings import SCHEDULER_MAX_JOBS_PER_USER
from cat_agent.tools.decorator import tool

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

_store_ctx: ContextVar[Optional[JobStore]] = ContextVar('sched_store', default=None)
_job_id_ctx: ContextVar[Optional[str]] = ContextVar('sched_job_id', default=None)


@contextmanager
def scheduling_context(
    store: JobStore,
    *,
    job_id: Optional[str] = None,
) -> Iterator[None]:
    """Bind the active JobStore (and optional job id) for tool calls."""
    token_s = _store_ctx.set(store)
    token_j = _job_id_ctx.set(job_id)
    try:
        yield
    finally:
        _store_ctx.reset(token_s)
        _job_id_ctx.reset(token_j)


def _get_store() -> JobStore:
    store = _store_ctx.get()
    if store is not None:
        return store
    return JobStore(dsn=default_scheduler_dsn())


def _validate_target(channel: str, target: str) -> None:
    channel = channel.lower().strip()
    target = (target or '').strip()
    if channel not in KNOWN_CHANNELS:
        raise ValueError(
            f'channel must be one of {sorted(KNOWN_CHANNELS)}, got {channel!r}'
        )
    if channel in ('smtp', 'resend'):
        if not _EMAIL_RE.match(target):
            raise ValueError(f'target must be an email address for channel={channel}')
    elif channel == 'webhook':
        parts = urlsplit(target)
        if parts.scheme not in ('http', 'https') or not parts.netloc:
            raise ValueError('target must be an http(s) URL for channel=webhook')


@tool(name='save_source')
def save_source(
    user_id: str,
    url: str,
    title: str,
    summary: str,
    tags: str = '',
) -> str:
    """Persist a collected source for later report delivery.

    Call once per distinct article or page. Duplicate normalized URLs for the
    same user are ignored. Do not summarize sources in chat — only call this tool.

    Args:
        user_id: Tenant / scope key that owns the source (e.g. ``alice``).
        url: Canonical URL of the source document.
        title: Short human-readable title.
        summary: One-paragraph factual summary of the source.
        tags: Optional comma-separated tags (e.g. ``ai,regulation``).

    Returns:
        A JSON string with ``id``, ``created``, and ``url``.
    """
    store = _get_store()
    source, created = store.save_source(
        user_id=user_id,
        url=url,
        title=title,
        summary=summary,
        tags=tags,
        job_id=_job_id_ctx.get(),
    )
    return json.dumps({
        'id': source.id,
        'created': created,
        'url': source.url,
    })


@tool(name='create_schedule')
def create_schedule(
    user_id: str,
    topic: str,
    every_hours: float,
    channel: str,
    target: str,
) -> str:
    """Create a recurring collect-and-report job for a user.

    Example: every 5 hours, email a report of sources collected on a topic.
    The job is persisted immediately and picked up by the APScheduler driver
    or the next Kubernetes CronJob poll.

    Args:
        user_id: Tenant / scope key that owns the schedule.
        topic: Free-text topic fed to the collector agent (e.g. ``AI regulation``).
        every_hours: Cadence in hours. Must be ``>= 0.25`` (15 minutes).
        channel: Delivery channel: ``smtp``, ``resend``, or ``webhook``.
        target: Email address (smtp/resend) or HTTPS webhook URL.

    Returns:
        A JSON string containing the created ``job_id`` and schedule metadata.
    """
    if every_hours < 0.25:
        raise ValueError('every_hours must be >= 0.25')
    channel = (channel or '').strip().lower()
    _validate_target(channel, target)
    store = _get_store()
    if store.count_jobs_for_user(user_id) >= SCHEDULER_MAX_JOBS_PER_USER:
        raise ValueError(
            f'user {user_id!r} already has {SCHEDULER_MAX_JOBS_PER_USER} jobs '
            '(CAT_AGENT_SCHEDULER_MAX_JOBS_PER_USER)'
        )
    now = time.time()
    interval_seconds = int(every_hours * 3600)
    job_id = make_job_id(user_id, topic)
    # Disambiguate collisions.
    base = job_id
    n = 2
    while store.get_job(job_id) is not None:
        job_id = f'{base}-{n}'
        n += 1
    job = Job(
        id=job_id,
        user_id=user_id,
        kind='collect_and_report',
        topic=topic,
        interval_seconds=interval_seconds,
        channel=channel,
        target=target.strip(),
        enabled=True,
        next_run_at=now + interval_seconds,
        created_at=now,
        updated_at=now,
    )
    store.upsert_job(job)
    return json.dumps({
        'job_id': job.id,
        'user_id': job.user_id,
        'topic': job.topic,
        'every_hours': every_hours,
        'interval_seconds': interval_seconds,
        'channel': job.channel,
        'target': job.target,
        'next_run_at': job.next_run_at,
    })


@tool(name='list_schedules')
def list_schedules(user_id: str) -> str:
    """List persisted schedule jobs for a user.

    Args:
        user_id: Tenant / scope key whose jobs should be listed.

    Returns:
        A JSON array of job summaries (id, topic, cadence, next_run_at, enabled).
    """
    store = _get_store()
    jobs = store.list_jobs(user_id=user_id)
    payload = [
        {
            'job_id': j.id,
            'topic': j.topic,
            'kind': j.kind,
            'interval_seconds': j.interval_seconds,
            'cron_expr': j.cron_expr,
            'channel': j.channel,
            'target': j.target,
            'enabled': j.enabled,
            'next_run_at': j.next_run_at,
            'consecutive_failures': j.consecutive_failures,
        }
        for j in jobs
    ]
    return json.dumps(payload)


@tool(name='cancel_schedule')
def cancel_schedule(job_id: str) -> str:
    """Disable and delete a schedule job.

    Args:
        job_id: The job identifier returned by ``create_schedule``
            (e.g. ``report:alice:ai-news``).

    Returns:
        A JSON string with ``job_id`` and ``deleted`` boolean.
    """
    store = _get_store()
    deleted = store.delete_job(job_id)
    return json.dumps({'job_id': job_id, 'deleted': deleted})
