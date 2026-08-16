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

"""Coverage tests for APScheduler driver (mocked APScheduler)."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cat_agent.scheduling.drivers import apscheduler_driver as ad
from cat_agent.scheduling.models import Job


def _job(**kwargs):
    base = dict(
        id='job-1',
        user_id='u',
        kind='collect',
        topic='news',
        channel='webhook',
        target='http://x',
        interval_seconds=60,
        cron_expr=None,
        timezone='UTC',
        enabled=True,
    )
    base.update(kwargs)
    return Job(**base)


def _install_fake_apscheduler(monkeypatch):
    class CronTrigger:
        @staticmethod
        def from_crontab(expr, timezone='UTC'):
            return SimpleNamespace(kind='cron', expr=expr, timezone=timezone)

    class IntervalTrigger:
        def __init__(self, seconds, timezone='UTC'):
            self.kind = 'interval'
            self.seconds = seconds
            self.timezone = timezone

    class SQLAlchemyJobStore:
        def __init__(self, url):
            self.url = url

    class AsyncIOScheduler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.jobs = {}
            self.started = False

        def start(self):
            self.started = True

        def add_job(self, fn, *a, **k):
            jid = k.get('id') or 'anon'
            self.jobs[jid] = {'fn': fn, 'args': a, 'kwargs': k}

        def remove_job(self, jid):
            if jid not in self.jobs:
                raise KeyError(jid)
            del self.jobs[jid]

        def shutdown(self, wait=False):
            self.started = False

    sched_mod = ModuleType('apscheduler')
    asyncio_mod = ModuleType('apscheduler.schedulers.asyncio')
    asyncio_mod.AsyncIOScheduler = AsyncIOScheduler
    jobstores_mod = ModuleType('apscheduler.jobstores.sqlalchemy')
    jobstores_mod.SQLAlchemyJobStore = SQLAlchemyJobStore
    triggers_cron = ModuleType('apscheduler.triggers.cron')
    triggers_cron.CronTrigger = CronTrigger
    triggers_interval = ModuleType('apscheduler.triggers.interval')
    triggers_interval.IntervalTrigger = IntervalTrigger
    schedulers = ModuleType('apscheduler.schedulers')
    triggers = ModuleType('apscheduler.triggers')
    jobstores = ModuleType('apscheduler.jobstores')

    for name, mod in {
        'apscheduler': sched_mod,
        'apscheduler.schedulers': schedulers,
        'apscheduler.schedulers.asyncio': asyncio_mod,
        'apscheduler.jobstores': jobstores,
        'apscheduler.jobstores.sqlalchemy': jobstores_mod,
        'apscheduler.triggers': triggers,
        'apscheduler.triggers.cron': triggers_cron,
        'apscheduler.triggers.interval': triggers_interval,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    return AsyncIOScheduler, CronTrigger, IntervalTrigger


def test_require_apscheduler_import_error(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith('apscheduler'):
            raise ImportError('missing')
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    with pytest.raises(ImportError, match='scheduler'):
        ad._require_apscheduler()


def test_start_sync_stop(monkeypatch):
    _install_fake_apscheduler(monkeypatch)
    store = MagicMock()
    store.dsn = 'sqlite:///:memory:'
    store.list_jobs.return_value = [
        _job(id='a', cron_expr='0 * * * *'),
        _job(id='b', cron_expr=None, interval_seconds=30),
    ]
    store.release_all_leases.return_value = 0

    driver = ad.APSchedulerDriver(store, sync_interval_seconds=10, owner='test-owner')

    with patch.object(ad, 'enable_optional_tools'):
        async def run():
            await driver.start()
            assert driver._scheduler is not None
            assert driver._scheduler.started
            assert 'sched:a' in driver._scheduler.jobs
            assert 'sched:b' in driver._scheduler.jobs
            assert '__sched_sync__' in driver._scheduler.jobs

            # Second sync removes stale known job
            driver._known.add('sched:stale')
            driver._scheduler.jobs['sched:stale'] = {'fn': None}
            store.list_jobs.return_value = [_job(id='a', cron_expr='0 * * * *')]
            await driver.sync_jobs()
            assert 'sched:stale' not in driver._scheduler.jobs

            # remove_job exception is swallowed
            driver._known.add('sched:gone')
            await driver.sync_jobs()

            await driver.stop(wait_timeout=0.01)
            assert driver._scheduler is None
            store.release_all_leases.assert_called_with('test-owner')

        asyncio.run(run())


def test_sync_jobs_noop_without_scheduler():
    store = MagicMock()
    driver = ad.APSchedulerDriver(store)
    asyncio.run(driver.sync_jobs())
    store.list_jobs.assert_not_called()


def test_run_job_stopping_and_claim_paths(monkeypatch):
    _install_fake_apscheduler(monkeypatch)
    store = MagicMock()
    driver = ad.APSchedulerDriver(store, owner='owner-1', lease_seconds=30)
    driver._stopping = True
    asyncio.run(driver._run_job('job-1'))
    store.claim_due_jobs.assert_not_called()

    driver._stopping = False
    store.claim_due_jobs.return_value = [_job(id='job-1')]
    with patch.object(ad, 'execute_job', new=AsyncMock()) as ex:
        asyncio.run(driver._run_job('job-1'))
        ex.assert_awaited()

    # Force-lease path when claim misses
    store.claim_due_jobs.return_value = []
    store.get_job.return_value = _job(id='job-1', lease_until=None)
    with patch.object(ad, 'execute_job', new=AsyncMock()) as ex:
        asyncio.run(driver._run_job('job-1'))
        store.force_lease.assert_called()
        ex.assert_awaited()

    # Disabled / missing job
    store.get_job.return_value = None
    store.force_lease.reset_mock()
    with patch.object(ad, 'execute_job', new=AsyncMock()) as ex:
        asyncio.run(driver._run_job('job-1'))
        store.force_lease.assert_not_called()
        ex.assert_not_awaited()

    store.get_job.return_value = _job(id='job-1', enabled=False)
    with patch.object(ad, 'execute_job', new=AsyncMock()) as ex:
        asyncio.run(driver._run_job('job-1'))
        ex.assert_not_awaited()

    # Lease held by other owner
    store.get_job.return_value = _job(
        id='job-1', enabled=True, lease_until=9e12, lease_owner='other',
    )
    with patch.object(ad, 'execute_job', new=AsyncMock()) as ex:
        asyncio.run(driver._run_job('job-1'))
        ex.assert_not_awaited()

    # execute_job exception swallowed
    store.claim_due_jobs.return_value = [_job(id='job-1')]
    with patch.object(ad, 'execute_job', new=AsyncMock(side_effect=RuntimeError('x'))):
        asyncio.run(driver._run_job('job-1'))


def test_start_signal_handler_fallback(monkeypatch):
    _install_fake_apscheduler(monkeypatch)
    store = MagicMock()
    store.dsn = 'sqlite:///:memory:'
    store.list_jobs.return_value = []
    store.release_all_leases.return_value = 0
    driver = ad.APSchedulerDriver(store)

    loop = MagicMock()
    loop.add_signal_handler.side_effect = NotImplementedError()

    with patch.object(ad, 'enable_optional_tools'), \
            patch('asyncio.get_running_loop', return_value=loop), \
            patch('signal.signal') as sig:
        async def run():
            await driver.start()
            await driver.stop(wait_timeout=0.01)

        asyncio.run(run())
        assert sig.called
