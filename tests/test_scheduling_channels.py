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

"""SMTP channel + retry classification tests (no network)."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage

import pytest

from cat_agent.scheduling.channels.base import (
    PermanentDeliveryError,
    TransientDeliveryError,
    markdown_to_html,
    send_with_retry,
)
from cat_agent.scheduling.channels.smtp import SMTPChannel
from cat_agent.scheduling.tools import (
    create_schedule,
    list_schedules,
    save_source,
    scheduling_context,
)
from cat_agent.scheduling.store import JobStore


class _FakeSMTP:
    instances = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.messages = []
        self._starttls = False
        self._logged_in = False
        _FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self._starttls = True

    def login(self, user, password):
        self._logged_in = True
        self.user = user

    def send_message(self, msg: EmailMessage):
        self.messages.append(msg)


class _FailSMTP(_FakeSMTP):
    def send_message(self, msg):
        raise smtplib.SMTPResponseException(450, b'try later')


class _PermanentFailSMTP(_FakeSMTP):
    def send_message(self, msg):
        raise smtplib.SMTPResponseException(550, b'user unknown')


@pytest.fixture(autouse=True)
def _clear_fake():
    _FakeSMTP.instances.clear()
    yield
    _FakeSMTP.instances.clear()


@pytest.mark.asyncio
async def test_smtp_channel_sends_via_fake(monkeypatch):
    monkeypatch.setattr(
        'cat_agent.scheduling.channels.smtp.smtplib.SMTP',
        _FakeSMTP,
    )
    ch = SMTPChannel(
        host='localhost', port=25, user='u', password='p',
        from_addr='from@example.com', starttls=True,
    )
    result = await ch.send(
        target='to@example.com',
        subject='Hello',
        body_markdown='# Hi\n\nBody',
    )
    assert result.ok
    assert len(_FakeSMTP.instances) == 1
    smtp = _FakeSMTP.instances[0]
    assert smtp._starttls is True
    assert smtp._logged_in is True
    assert len(smtp.messages) == 1
    assert smtp.messages[0]['To'] == 'to@example.com'


@pytest.mark.asyncio
async def test_smtp_450_is_transient(monkeypatch):
    monkeypatch.setattr(
        'cat_agent.scheduling.channels.smtp.smtplib.SMTP',
        _FailSMTP,
    )
    ch = SMTPChannel(host='localhost', port=25, from_addr='a@b.c', starttls=False)
    with pytest.raises(TransientDeliveryError):
        await ch.send(target='t@e.com', subject='s', body_markdown='b')


@pytest.mark.asyncio
async def test_smtp_550_is_permanent(monkeypatch):
    monkeypatch.setattr(
        'cat_agent.scheduling.channels.smtp.smtplib.SMTP',
        _PermanentFailSMTP,
    )
    ch = SMTPChannel(host='localhost', port=25, from_addr='a@b.c', starttls=False)
    with pytest.raises(PermanentDeliveryError):
        await ch.send(target='t@e.com', subject='s', body_markdown='b')


@pytest.mark.asyncio
async def test_send_with_retry_retries_transient():
    class Flaky:
        name = 'flaky'

        def __init__(self):
            self.n = 0

        async def send(self, **kwargs):
            self.n += 1
            if self.n < 3:
                raise TransientDeliveryError('temp')
            from cat_agent.scheduling.channels.base import DeliveryResult
            return DeliveryResult(ok=True)

    ch = Flaky()
    result = await send_with_retry(
        ch, target='t', subject='s', body_markdown='b', initial_delay=0.01,
    )
    assert result.ok
    assert result.attempts == 3
    assert ch.n == 3


def test_markdown_to_html_basic():
    html = markdown_to_html('# Title\n\nHello **world**')
    assert '<h1>' in html
    assert '<strong>world</strong>' in html


def test_create_schedule_and_save_source(tmp_path):
    store = JobStore(dsn=f'sqlite:///{tmp_path / "t.sqlite"}')
    with scheduling_context(store):
        out = create_schedule(
            user_id='alice',
            topic='AI news',
            every_hours=5,
            channel='smtp',
            target='alice@example.com',
        )
        assert 'report:alice:ai-news' in out
        listed = list_schedules('alice')
        assert 'ai-news' in listed
        saved = save_source(
            user_id='alice',
            url='https://Example.com/a/?utm_source=x',
            title='A',
            summary='summary',
        )
        assert '"created": true' in saved.lower() or '"created":true' in saved.replace(' ', '')


def test_create_schedule_rejects_bad_email(tmp_path):
    store = JobStore(dsn=f'sqlite:///{tmp_path / "t.sqlite"}')
    with scheduling_context(store):
        with pytest.raises(ValueError, match='email'):
            create_schedule(
                user_id='alice',
                topic='t',
                every_hours=1,
                channel='smtp',
                target='not-an-email',
            )


def test_create_schedule_rejects_short_interval(tmp_path):
    store = JobStore(dsn=f'sqlite:///{tmp_path / "t.sqlite"}')
    with scheduling_context(store):
        with pytest.raises(ValueError, match='0.25'):
            create_schedule(
                user_id='alice',
                topic='t',
                every_hours=0.1,
                channel='smtp',
                target='a@b.co',
            )
