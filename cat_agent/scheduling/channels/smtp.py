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

"""SMTP delivery channel (stdlib smtplib — default for on-prem / air-gapped)."""

from __future__ import annotations

import asyncio
import os
import smtplib
from email.message import EmailMessage
from typing import Optional

from cat_agent.scheduling.channels.base import (
    DeliveryResult,
    PermanentDeliveryError,
    TransientDeliveryError,
    markdown_to_html,
)


class SMTPChannel:
    name = 'smtp'

    def __init__(
        self,
        *,
        host: str,
        port: int = 587,
        user: str = '',
        password: str = '',
        from_addr: str = '',
        starttls: bool = True,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.from_addr = from_addr or user
        self.starttls = starttls

    @classmethod
    def from_env(cls) -> 'SMTPChannel':
        host = os.getenv('SMTP_HOST', '').strip()
        if not host:
            raise ValueError('SMTP_HOST is required for the smtp delivery channel')
        return cls(
            host=host,
            port=int(os.getenv('SMTP_PORT', '587')),
            user=os.getenv('SMTP_USER', ''),
            password=os.getenv('SMTP_PASSWORD', ''),
            from_addr=os.getenv('SMTP_FROM', '') or os.getenv('SMTP_USER', ''),
            starttls=os.getenv('SMTP_STARTTLS', '1').strip().lower()
            not in {'0', 'false', 'no', 'off'},
        )

    def _send_sync(
        self,
        *,
        target: str,
        subject: str,
        body_markdown: str,
        body_html: Optional[str],
    ) -> DeliveryResult:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = self.from_addr
        msg['To'] = target
        msg.set_content(body_markdown or '')
        html_body = body_html if body_html is not None else markdown_to_html(body_markdown)
        msg.add_alternative(html_body, subtype='html')
        try:
            with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
                if self.starttls:
                    smtp.starttls()
                if self.user:
                    smtp.login(self.user, self.password)
                smtp.send_message(msg)
        except smtplib.SMTPResponseException as exc:
            code = int(exc.smtp_code or 0)
            if 400 <= code < 500:
                raise TransientDeliveryError(f'SMTP {code}') from exc
            raise PermanentDeliveryError(f'SMTP {code}') from exc
        except (smtplib.SMTPServerDisconnected, TimeoutError, OSError) as exc:
            raise TransientDeliveryError(str(exc)) from exc
        except smtplib.SMTPException as exc:
            raise PermanentDeliveryError(str(exc)) from exc
        return DeliveryResult(ok=True, provider_id='smtp')

    async def send(
        self,
        *,
        target: str,
        subject: str,
        body_markdown: str,
        body_html: Optional[str] = None,
    ) -> DeliveryResult:
        return await asyncio.to_thread(
            self._send_sync,
            target=target,
            subject=subject,
            body_markdown=body_markdown,
            body_html=body_html,
        )
