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

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JobKind(str, Enum):
    COLLECT = 'collect'
    REPORT = 'report'
    COLLECT_AND_REPORT = 'collect_and_report'

    def includes_collection(self) -> bool:
        return self in (JobKind.COLLECT, JobKind.COLLECT_AND_REPORT)

    def includes_report(self) -> bool:
        return self in (JobKind.REPORT, JobKind.COLLECT_AND_REPORT)


class JobRunStatus(str, Enum):
    RUNNING = 'running'
    OK = 'ok'
    FAILED = 'failed'
    SKIPPED = 'skipped'
    SKIPPED_EMPTY = 'skipped_empty'


class DeliveryChannelName(str, Enum):
    SMTP = 'smtp'
    RESEND = 'resend'
    WEBHOOK = 'webhook'


KNOWN_CHANNELS = frozenset(c.value for c in DeliveryChannelName)


@dataclass
class Job:
    id: str
    user_id: str
    kind: str
    topic: str
    channel: str
    target: str
    interval_seconds: Optional[int] = None
    cron_expr: Optional[str] = None
    timezone: str = 'UTC'
    enabled: bool = True
    next_run_at: float = 0.0
    last_run_at: Optional[float] = None
    lease_owner: Optional[str] = None
    lease_until: Optional[float] = None
    consecutive_failures: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def kind_enum(self) -> JobKind:
        return JobKind(self.kind)


@dataclass
class JobRun:
    id: str
    job_id: str
    started_at: float
    status: str
    finished_at: Optional[float] = None
    sources_count: int = 0
    error: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class Source:
    id: str
    user_id: str
    url: str
    title: str
    summary: str
    collected_at: float
    job_id: Optional[str] = None
    tags: str = ''
    delivered_at: Optional[float] = None
    content_hash: Optional[str] = None
    metadata: dict = field(default_factory=dict)
