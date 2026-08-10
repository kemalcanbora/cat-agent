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

"""Resend HTTP delivery channel (optional ``email`` extra)."""

from __future__ import annotations

import os
from typing import Optional

from cat_agent.scheduling.channels.base import (
    DeliveryResult,
    PermanentDeliveryError,
    TransientDeliveryError,
    markdown_to_html,
)


class ResendChannel:
    name = 'resend'

    def __init__(self, *, api_key: str, from_addr: str):
        self.api_key = api_key
        self.from_addr = from_addr

    @classmethod
    def from_env(cls) -> 'ResendChannel':
        api_key = os.getenv('RESEND_API_KEY', '').strip()
        from_addr = os.getenv('RESEND_FROM', '').strip() or os.getenv('SMTP_FROM', '').strip()
        if not api_key:
            raise ValueError('RESEND_API_KEY is required for the resend channel')
        if not from_addr:
            raise ValueError('RESEND_FROM (or SMTP_FROM) is required for the resend channel')
        return cls(api_key=api_key, from_addr=from_addr)

    async def send(
        self,
        *,
        target: str,
        subject: str,
        body_markdown: str,
        body_html: Optional[str] = None,
    ) -> DeliveryResult:
        try:
            import resend  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Resend channel requires the 'email' extra. "
                "Install with: pip install 'cat-agent[email]'"
            ) from exc

        resend.api_key = self.api_key
        html_body = body_html if body_html is not None else markdown_to_html(body_markdown)
        try:
            response = resend.Emails.send({
                'from': self.from_addr,
                'to': [target],
                'subject': subject,
                'text': body_markdown or '',
                'html': html_body,
            })
        except Exception as exc:
            msg = str(exc)
            if any(x in msg for x in ('429', '500', '502', '503', '504', 'timeout')):
                raise TransientDeliveryError(msg) from exc
            raise PermanentDeliveryError(msg) from exc
        provider_id = None
        if isinstance(response, dict):
            provider_id = response.get('id')
        else:
            provider_id = getattr(response, 'id', None)
        return DeliveryResult(ok=True, provider_id=provider_id)
