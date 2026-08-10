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

"""Backoff / consecutive_failures behaviour for scheduled jobs."""

from __future__ import annotations

import time

import pytest

from cat_agent.scheduling.models import Job
from cat_agent.scheduling.runner import compute_next_run_at, execute_job
from cat_agent.scheduling.store import JobStore


@pytest.fixture()
def store(tmp_path):
    return JobStore(dsn=f'sqlite:///{tmp_path / "backoff.sqlite"}')


def _job(**overrides) -> Job:
    now = time.time()
    base = dict(
        id='report:alice:ai-news',
        user_id='alice',
        kind='report',
        topic='AI news',
        interval_seconds=100,
        channel='smtp',
        target='alice@example.com',
        enabled=True,
        next_run_at=now - 1,
        created_at=now,
        updated_at=now,
        consecutive_failures=0,
    )
    base.update(overrides)
    return Job(**base)


class TestComputeNextRunAt:
    def test_success_resets_to_base_interval(self):
        job = _job()
        now = 1_000_000.0
        assert compute_next_run_at(job, now=now, consecutive_failures=0) == now + 100

    def test_failures_grow_exponentially(self):
        job = _job()
        now = 1_000_000.0
        assert compute_next_run_at(job, now=now, consecutive_failures=1) == now + 100
        assert compute_next_run_at(job, now=now, consecutive_failures=2) == now + 200
        assert compute_next_run_at(job, now=now, consecutive_failures=3) == now + 400

    def test_backoff_is_capped(self):
        job = _job()
        now = 1_000_000.0
        # Cap multiplier default 8 → max delay 800.
        assert compute_next_run_at(
            job, now=now, consecutive_failures=10, backoff_cap_multiplier=8,
        ) == now + 800


@pytest.mark.asyncio
async def test_failures_increment_and_reset_on_success(store):
    store.upsert_job(_job())
    store.save_source(
        user_id='alice', url='https://ex.com/1', title='t', summary='s',
    )

    async def boom(job, sources, store_):
        raise RuntimeError('fail')

    async def noop_deliver(job, body, sources, store_):
        return None

    with pytest.raises(RuntimeError):
        await execute_job(
            'report:alice:ai-news',
            store=store,
            owner='t',
            report_fn=boom,
            deliver_fn=noop_deliver,
        )
    job = store.get_job('report:alice:ai-news')
    assert job is not None
    assert job.consecutive_failures == 1

    with pytest.raises(RuntimeError):
        await execute_job(
            'report:alice:ai-news',
            store=store,
            owner='t',
            report_fn=boom,
            deliver_fn=noop_deliver,
        )
    job = store.get_job('report:alice:ai-news')
    assert job is not None
    assert job.consecutive_failures == 2

    async def ok_report(job, sources, store_):
        return '# ok'

    async def ok_deliver(job, body, sources, store_):
        return None

    run = await execute_job(
        'report:alice:ai-news',
        store=store,
        owner='t',
        report_fn=ok_report,
        deliver_fn=ok_deliver,
    )
    assert run.status == 'ok'
    job = store.get_job('report:alice:ai-news')
    assert job is not None
    assert job.consecutive_failures == 0
