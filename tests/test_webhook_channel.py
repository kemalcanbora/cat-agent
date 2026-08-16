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

"""Tests for webhook channel delivery (mocked HTTP)."""

from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib import error

import pytest

from cat_agent.scheduling.channels.base import (
    PermanentDeliveryError,
    TransientDeliveryError,
)
from cat_agent.scheduling.channels.webhook import WebhookChannel


def test_webhook_success():
    ch = WebhookChannel()
    resp = MagicMock()
    resp.status = 200
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    with patch('cat_agent.scheduling.channels.webhook.request.urlopen', return_value=resp):
        result = ch._send_sync(
            target='https://example.com/hook',
            subject='hi',
            body_markdown='body',
        )
    assert result.ok is True


def test_webhook_http_500_transient():
    ch = WebhookChannel()
    with patch(
        'cat_agent.scheduling.channels.webhook.request.urlopen',
        side_effect=error.HTTPError('u', 503, 'err', hdrs=None, fp=BytesIO()),
    ):
        with pytest.raises(TransientDeliveryError):
            ch._send_sync(target='https://example.com/hook', subject='s', body_markdown='b')


def test_webhook_http_400_permanent():
    ch = WebhookChannel()
    with patch(
        'cat_agent.scheduling.channels.webhook.request.urlopen',
        side_effect=error.HTTPError('u', 400, 'bad', hdrs=None, fp=BytesIO()),
    ):
        with pytest.raises(PermanentDeliveryError):
            ch._send_sync(target='https://example.com/hook', subject='s', body_markdown='b')
