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

"""Generic JSON webhook delivery (Slack-incoming-webhook compatible)."""

from __future__ import annotations

import asyncio
import json
from typing import Optional
from urllib import error, request

from cat_agent.scheduling.channels.base import (
    DeliveryResult,
    PermanentDeliveryError,
    TransientDeliveryError,
)


class WebhookChannel:
    name = 'webhook'

    def _send_sync(
        self,
        *,
        target: str,
        subject: str,
        body_markdown: str,
    ) -> DeliveryResult:
        # Slack-compatible: text field; also include subject for generic consumers.
        payload = {
            'text': f'*{subject}*\n\n{body_markdown}',
            'subject': subject,
            'markdown': body_markdown,
        }
        data = json.dumps(payload).encode('utf-8')
        req = request.Request(
            target,
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with request.urlopen(req, timeout=30) as resp:
                status = getattr(resp, 'status', 200)
                if 200 <= int(status) < 300:
                    return DeliveryResult(ok=True, provider_id='webhook')
                if int(status) >= 500:
                    raise TransientDeliveryError(f'HTTP {status}')
                raise PermanentDeliveryError(f'HTTP {status}')
        except error.HTTPError as exc:
            code = int(exc.code or 0)
            if code >= 500 or code == 429:
                raise TransientDeliveryError(f'HTTP {code}') from exc
            raise PermanentDeliveryError(f'HTTP {code}') from exc
        except error.URLError as exc:
            raise TransientDeliveryError(str(exc.reason)) from exc

    async def send(
        self,
        *,
        target: str,
        subject: str,
        body_markdown: str,
        body_html: Optional[str] = None,
    ) -> DeliveryResult:
        del body_html  # unused for webhooks
        return await asyncio.to_thread(
            self._send_sync,
            target=target,
            subject=subject,
            body_markdown=body_markdown,
        )
