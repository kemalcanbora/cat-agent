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

"""Tests for execute_job / run_due_once shared runner."""

from __future__ import annotations

import time
from typing import List

import pytest

from cat_agent.scheduling.models import Job, JobRunStatus
from cat_agent.scheduling.runner import execute_job, run_due_once, scrub_error
from cat_agent.scheduling.store import JobStore


@pytest.fixture()
def store(tmp_path):
    return JobStore(dsn=f'sqlite:///{tmp_path / "runner.sqlite"}')


def _job(**overrides) -> Job:
    now = time.time()
    base = dict(
        id='report:alice:ai-news',
        user_id='alice',
        kind='report',
        topic='AI news',
        interval_seconds=3600,
        channel='smtp',
        target='alice@example.com',
        enabled=True,
        next_run_at=now - 1,
        created_at=now,
        updated_at=now,
    )
    base.update(overrides)
    return Job(**base)


class _FakeChannel:
    def __init__(self):
        self.calls: List[dict] = []
        self.fail = False

    async def send(self, *, target, subject, body_markdown, body_html=None):
        if self.fail:
            raise RuntimeError('smtp boom password=supersecret')
        self.calls.append({
            'target': target,
            'subject': subject,
            'body_markdown': body_markdown,
        })
        return {'ok': True}


@pytest.mark.asyncio
async def test_empty_window_skipped_empty_no_delivery(store):
    store.upsert_job(_job())
    channel = _FakeChannel()

    async def report(job, sources, store_):
        return '# report'

    async def deliver(job, body, sources, store_):
        await channel.send(
            target=job.target, subject='r', body_markdown=body,
        )

    run = await execute_job(
        'report:alice:ai-news',
        store=store,
        owner='test',
        collect_fn=None,
        report_fn=report,
        deliver_fn=deliver,
    )
    # kind=report with no sources → skipped_empty; deliver never called.
    assert run.status == JobRunStatus.SKIPPED_EMPTY.value
    assert channel.calls == []
    job = store.get_job('report:alice:ai-news')
    assert job is not None
    assert job.next_run_at > time.time() - 5
    assert job.consecutive_failures == 0


@pytest.mark.asyncio
async def test_delivery_failure_leaves_delivered_at_null(store):
    store.upsert_job(_job())
    store.save_source(
        user_id='alice',
        url='https://example.com/a',
        title='A',
        summary='alpha',
    )
    channel = _FakeChannel()
    channel.fail = True

    async def report(job, sources, store_):
        return '# report'

    async def deliver(job, body, sources, store_):
        await channel.send(
            target=job.target, subject='r', body_markdown=body,
        )

    with pytest.raises(RuntimeError):
        await execute_job(
            'report:alice:ai-news',
            store=store,
            owner='test',
            report_fn=report,
            deliver_fn=deliver,
        )

    undelivered = store.list_undelivered('alice')
    assert len(undelivered) == 1
    assert undelivered[0].delivered_at is None
    runs = store.list_runs('report:alice:ai-news', limit=1)
    assert runs[0].status == JobRunStatus.FAILED.value
    assert 'supersecret' not in (runs[0].error or '')
    assert '[REDACTED]' in (runs[0].error or '')


@pytest.mark.asyncio
async def test_next_run_advances_even_when_run_raises(store):
    now = time.time()
    store.upsert_job(_job(next_run_at=now - 1, interval_seconds=100))

    async def report(job, sources, store_):
        raise ValueError('boom')

    # Seed a source so we enter the report path.
    store.save_source(
        user_id='alice', url='https://ex.com/1', title='t', summary='s',
    )

    async def noop_deliver(job, body, sources, store_):
        return None

    with pytest.raises(ValueError):
        await execute_job(
            'report:alice:ai-news',
            store=store,
            owner='test',
            now=now,
            report_fn=report,
            deliver_fn=noop_deliver,
        )

    job = store.get_job('report:alice:ai-news')
    assert job is not None
    # failures=1 → next = now_finished + 100 * 2^0
    assert job.consecutive_failures == 1
    assert job.next_run_at >= now + 100
    assert job.lease_owner is None


@pytest.mark.asyncio
async def test_successful_delivery_marks_watermark(store):
    store.upsert_job(_job())
    store.save_source(
        user_id='alice', url='https://ex.com/ok', title='t', summary='s',
    )
    channel = _FakeChannel()

    async def report(job, sources, store_):
        assert len(sources) == 1
        return '# ok'

    async def deliver(job, body, sources, store_):
        await channel.send(target=job.target, subject='s', body_markdown=body)

    run = await execute_job(
        'report:alice:ai-news',
        store=store,
        owner='test',
        report_fn=report,
        deliver_fn=deliver,
    )
    assert run.status == JobRunStatus.OK.value
    assert store.list_undelivered('alice') == []
    assert len(channel.calls) == 1


@pytest.mark.asyncio
async def test_run_due_once_claims_and_runs(store):
    now = time.time()
    store.upsert_job(_job(id='j1', next_run_at=now - 5))
    store.upsert_job(_job(id='j2', next_run_at=now - 5, user_id='bob', target='b@x.com'))

    async def report(job, sources, store_):
        return 'empty-path'

    async def noop_deliver(job, body, sources, store_):
        return None

    results = await run_due_once(
        store,
        owner='cron-pod',
        now=now,
        report_fn=report,
        deliver_fn=noop_deliver,
    )
    assert len(results) == 2
    assert {r.status for r in results} == {JobRunStatus.SKIPPED_EMPTY.value}


def test_scrub_error_redacts_secrets():
    assert 'sekrit' not in scrub_error('auth failed password=sekrit')
    assert '[REDACTED]' in scrub_error('token: abc123')
