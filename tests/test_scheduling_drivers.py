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

"""Driver parity: APScheduler path and run_due_once share execute_job outcomes."""

from __future__ import annotations

import time

import pytest

from cat_agent.scheduling.models import Job, JobRunStatus
from cat_agent.scheduling.runner import execute_job, run_due_once
from cat_agent.scheduling.store import JobStore


@pytest.fixture()
def store(tmp_path):
    return JobStore(dsn=f'sqlite:///{tmp_path / "drivers.sqlite"}')


def _job(job_id: str, *, now: float, user_id: str = 'alice') -> Job:
    return Job(
        id=job_id,
        user_id=user_id,
        kind='report',
        topic='parity',
        interval_seconds=3600,
        channel='smtp',
        target=f'{user_id}@example.com',
        enabled=True,
        next_run_at=now - 10,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_apscheduler_callable_and_run_due_once_same_outcome(store):
    """Both drivers must produce identical JobRun status for the same fixture.

    APScheduler's scheduled callable is execute_job-after-lease; run_due_once
    claims then execute_job. With the same injectables, outcomes match.
    """
    now = time.time()
    store.upsert_job(_job('job-parity', now=now))
    store.save_source(
        user_id='alice',
        url='https://example.com/parity',
        title='Parity',
        summary='same fixture',
        collected_at=now - 100,
    )

    bodies = []

    async def report(job, sources, store_):
        bodies.append(len(sources))
        return '# parity report'

    delivered = []

    async def deliver(job, body, sources, store_):
        delivered.append(job.id)

    # Path A: oneshot / k8s
    store_a_job = store.get_job('job-parity')
    assert store_a_job is not None
    results_a = await run_due_once(
        store,
        owner='oneshot-pod',
        now=now,
        report_fn=report,
        deliver_fn=deliver,
    )
    assert len(results_a) == 1
    assert results_a[0].status == JobRunStatus.OK.value
    assert store.list_undelivered('alice') == []

    # Reset watermark + schedule for path B (simulate second identical fixture).
    store.mark_delivered([])  # no-op
    # Re-insert as undelivered by saving a new source and resetting job.
    store.save_source(
        user_id='alice',
        url='https://example.com/parity-2',
        title='Parity2',
        summary='same fixture 2',
        collected_at=now - 50,
    )
    store.upsert_job(_job('job-parity-b', now=now, user_id='alice'))
    # Clear lease / failures from path A job; use a fresh job id for path B.
    job_b = store.get_job('job-parity-b')
    assert job_b is not None

    # Path B: APScheduler-style — claim/force lease then execute_job
    store.force_lease(
        'job-parity-b',
        owner='apsched-host',
        lease_until=now + 60,
        now=now,
    )
    run_b = await execute_job(
        'job-parity-b',
        store=store,
        owner='apsched-host',
        now=now,
        report_fn=report,
        deliver_fn=deliver,
    )
    assert run_b.status == results_a[0].status
    assert run_b.sources_count >= 1
    assert 'job-parity-b' in delivered
    assert bodies[0] == 1
    assert bodies[1] >= 1


@pytest.mark.asyncio
async def test_get_driver_factory(store):
    from cat_agent.scheduling.drivers import get_driver

    driver = get_driver('apscheduler', store, owner='test')
    assert driver.owner == 'test'
    oneshot = get_driver('oneshot', store)
    assert hasattr(oneshot, 'main')
