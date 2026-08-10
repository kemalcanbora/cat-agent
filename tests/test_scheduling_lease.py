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

"""Concurrent lease claim tests for scheduling JobStore."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from cat_agent.scheduling.models import Job
from cat_agent.scheduling.runner import claim_due_jobs
from cat_agent.scheduling.store import JobStore


@pytest.fixture()
def store(tmp_path):
    return JobStore(dsn=f'sqlite:///{tmp_path / "lease.sqlite"}')


def _due_job(job_id: str, *, now: float) -> Job:
    return Job(
        id=job_id,
        user_id='alice',
        kind='report',
        topic='t',
        interval_seconds=3600,
        channel='smtp',
        target='a@example.com',
        enabled=True,
        next_run_at=now - 10,
        created_at=now,
        updated_at=now,
    )


class TestClaimDueJobs:
    def test_two_threads_never_claim_same_job(self, store):
        now = time.time()
        for i in range(8):
            store.upsert_job(_due_job(f'job-{i}', now=now))

        def claim(owner: str):
            return claim_due_jobs(
                store, limit=50, lease_seconds=60, owner=owner, now=now,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(claim, 'owner-a'),
                pool.submit(claim, 'owner-b'),
            ]
            results = [f.result() for f in as_completed(futures)]

        ids_a = {j.id for j in results[0]}
        ids_b = {j.id for j in results[1]}
        assert ids_a.isdisjoint(ids_b)
        assert ids_a | ids_b == {f'job-{i}' for i in range(8)}

    def test_expired_lease_is_reclaimable(self, store):
        now = time.time()
        store.upsert_job(_due_job('job-x', now=now))
        first = claim_due_jobs(
            store, limit=1, lease_seconds=1, owner='pod-1', now=now,
        )
        assert [j.id for j in first] == ['job-x']
        # Still leased — second claim gets nothing.
        assert claim_due_jobs(
            store, limit=1, lease_seconds=1, owner='pod-2', now=now + 0.5,
        ) == []
        # After expiry, reclaim succeeds.
        reclaimed = claim_due_jobs(
            store, limit=1, lease_seconds=60, owner='pod-2', now=now + 2,
        )
        assert [j.id for j in reclaimed] == ['job-x']
        assert reclaimed[0].lease_owner == 'pod-2'
