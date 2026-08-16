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

"""Coverage for Resend channel + oneshot scheduler driver."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.scheduling.channels.base import PermanentDeliveryError, TransientDeliveryError
from cat_agent.scheduling.channels.resend import ResendChannel


def test_resend_from_env(monkeypatch):
    monkeypatch.delenv('RESEND_API_KEY', raising=False)
    with pytest.raises(ValueError, match='RESEND_API_KEY'):
        ResendChannel.from_env()

    monkeypatch.setenv('RESEND_API_KEY', 'rk')
    monkeypatch.delenv('RESEND_FROM', raising=False)
    monkeypatch.delenv('SMTP_FROM', raising=False)
    with pytest.raises(ValueError, match='RESEND_FROM'):
        ResendChannel.from_env()

    monkeypatch.setenv('SMTP_FROM', 'a@b.com')
    ch = ResendChannel.from_env()
    assert ch.api_key == 'rk'
    assert ch.from_addr == 'a@b.com'


@pytest.mark.asyncio
async def test_resend_send_success_and_errors(monkeypatch):
    resend = ModuleType('resend')
    resend.api_key = None
    resend.Emails = MagicMock()
    resend.Emails.send.return_value = {'id': 'msg-1'}
    monkeypatch.setitem(sys.modules, 'resend', resend)

    ch = ResendChannel(api_key='k', from_addr='from@x.com')
    result = await ch.send(target='t@x.com', subject='s', body_markdown='**hi**')
    assert result.ok and result.provider_id == 'msg-1'

    resend.Emails.send.return_value = SimpleNamespace(id='msg-2')
    result2 = await ch.send(
        target='t@x.com', subject='s', body_markdown='x', body_html='<p>x</p>'
    )
    assert result2.provider_id == 'msg-2'

    resend.Emails.send.side_effect = RuntimeError('timeout while sending')
    with pytest.raises(TransientDeliveryError):
        await ch.send(target='t@x.com', subject='s', body_markdown='x')

    resend.Emails.send.side_effect = RuntimeError('invalid recipient')
    with pytest.raises(PermanentDeliveryError):
        await ch.send(target='t@x.com', subject='s', body_markdown='x')


def test_oneshot_owner_and_main(monkeypatch):
    from cat_agent.scheduling.drivers import oneshot as oneshot_mod

    monkeypatch.delenv('POD_NAME', raising=False)
    monkeypatch.setenv('HOSTNAME', 'host-a')
    assert oneshot_mod._owner_name() == 'host-a'

    monkeypatch.setenv('POD_NAME', 'pod-1')
    assert oneshot_mod._owner_name() == 'pod-1'

    run_ok = SimpleNamespace(
        job_id='j', id='r1', status='ok', sources_count=0, error=None, trace_id=None
    )
    run_fail = SimpleNamespace(
        job_id='j2', id='r2', status='failed', sources_count=1, error='e', trace_id='t'
    )
    store = MagicMock()

    async def fake_run_due(*a, **k):
        return [run_ok, run_fail]

    with patch.object(oneshot_mod, 'JobStore', return_value=store, create=True), \
            patch('cat_agent.scheduling.store.JobStore', return_value=store), \
            patch('cat_agent.scheduling.runner.run_due_once', side_effect=fake_run_due), \
            patch('cat_agent.tools.base.enable_optional_tools'), \
            patch('cat_agent.scheduling.drivers.oneshot.default_scheduler_dsn', create=True), \
            patch('builtins.print'):
        # Import path uses local imports inside main — patch at source modules
        with patch('cat_agent.scheduling.store.JobStore', return_value=store), \
                patch('cat_agent.scheduling.store.default_scheduler_dsn', return_value='dsn'), \
                patch('cat_agent.scheduling.runner.run_due_once', side_effect=fake_run_due), \
                patch('cat_agent.tools.base.enable_optional_tools'):
            rc = oneshot_mod.main([])
            assert rc == 1  # one failed
            store.release_all_leases.assert_called()

    async def empty_run(*a, **k):
        return []

    with patch('cat_agent.scheduling.store.JobStore', return_value=store), \
            patch('cat_agent.scheduling.store.default_scheduler_dsn', return_value='dsn'), \
            patch('cat_agent.scheduling.runner.run_due_once', side_effect=empty_run), \
            patch('cat_agent.tools.base.enable_optional_tools'), \
            patch('builtins.print'):
        assert oneshot_mod.main([]) == 0
