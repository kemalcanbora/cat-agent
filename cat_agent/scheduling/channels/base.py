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

"""Delivery channel protocol and shared helpers."""

from __future__ import annotations

import asyncio
import html
import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from cat_agent.utils.backoff import compute_backoff_delay


@dataclass
class DeliveryResult:
    ok: bool
    provider_id: Optional[str] = None
    error: Optional[str] = None
    attempts: int = 1


@runtime_checkable
class DeliveryChannel(Protocol):
    name: str

    async def send(
        self,
        *,
        target: str,
        subject: str,
        body_markdown: str,
        body_html: Optional[str] = None,
    ) -> DeliveryResult:
        ...


def markdown_to_html(markdown: str) -> str:
    """Minimal Markdown → HTML for email bodies (no extra deps)."""
    text = html.escape(markdown or '')
    # Fenced code blocks
    text = re.sub(
        r'```[\w]*\n(.*?)```',
        lambda m: f'<pre><code>{m.group(1)}</code></pre>',
        text,
        flags=re.DOTALL,
    )
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(
        r'\[([^\]]+)\]\((https?://[^)]+)\)',
        r'<a href="\2">\1</a>',
        text,
    )
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'(?:<li>.*?</li>\n?)+', lambda m: f'<ul>\n{m.group(0)}</ul>\n', text)
    paragraphs = []
    for block in re.split(r'\n\n+', text):
        block = block.strip()
        if not block:
            continue
        if block.startswith('<'):
            paragraphs.append(block)
        else:
            paragraphs.append(f'<p>{block.replace(chr(10), "<br/>")}</p>')
    return '\n'.join(paragraphs)


class TransientDeliveryError(Exception):
    """Retryable delivery failure (timeouts, 5xx, SMTP 4xx)."""


class PermanentDeliveryError(Exception):
    """Non-retryable delivery failure (SMTP 5xx, HTTP 4xx)."""


async def send_with_retry(
    channel: DeliveryChannel,
    *,
    target: str,
    subject: str,
    body_markdown: str,
    body_html: Optional[str] = None,
    max_attempts: int = 4,
    initial_delay: float = 0.5,
) -> DeliveryResult:
    delay = initial_delay
    last_error: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = await channel.send(
                target=target,
                subject=subject,
                body_markdown=body_markdown,
                body_html=body_html,
            )
            if result.ok:
                result.attempts = attempt
                return result
            raise TransientDeliveryError(result.error or 'delivery failed')
        except PermanentDeliveryError:
            raise
        except TransientDeliveryError as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            await asyncio.sleep(delay)
            delay = compute_backoff_delay(delay, max_delay=30.0, jitter=False)
        except Exception as exc:
            # Unknown errors: retry once class of timeout/connection, else permanent.
            name = type(exc).__name__.lower()
            msg = str(exc).lower()
            transient = any(
                k in name or k in msg
                for k in ('timeout', 'temporarily', 'connection', '421', '450', '451', '452')
            )
            if not transient or attempt >= max_attempts:
                raise PermanentDeliveryError(str(exc)) from exc
            last_error = exc
            await asyncio.sleep(delay)
            delay = compute_backoff_delay(delay, max_delay=30.0, jitter=False)
    raise TransientDeliveryError(str(last_error) if last_error else 'delivery failed')


def get_channel(name: str) -> DeliveryChannel:
    """Factory for configured delivery channels."""
    key = (name or '').strip().lower()
    if key == 'smtp':
        from cat_agent.scheduling.channels.smtp import SMTPChannel

        return SMTPChannel.from_env()
    if key == 'resend':
        from cat_agent.scheduling.channels.resend import ResendChannel

        return ResendChannel.from_env()
    if key == 'webhook':
        from cat_agent.scheduling.channels.webhook import WebhookChannel

        return WebhookChannel()
    raise ValueError(f'Unknown delivery channel: {name!r}')
