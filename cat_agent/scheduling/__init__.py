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

"""Scheduled source collection and report delivery."""

from __future__ import annotations

from cat_agent.scheduling.models import (
    KNOWN_CHANNELS,
    DeliveryChannelName,
    Job,
    JobKind,
    JobRun,
    JobRunStatus,
    Source,
)
from cat_agent.scheduling.runner import (
    claim_due_jobs,
    compute_next_run_at,
    execute_job,
    run_due_once,
    scrub_error,
)
from cat_agent.scheduling.store import (
    JobStore,
    SourceStore,
    content_hash_for,
    default_scheduler_dsn,
    make_job_id,
    normalize_url,
    source_id_for,
)

# Register LLM-facing tools on import.
from cat_agent.scheduling import tools as _tools  # noqa: F401

__all__ = [
    'KNOWN_CHANNELS',
    'DeliveryChannelName',
    'Job',
    'JobKind',
    'JobRun',
    'JobRunStatus',
    'JobStore',
    'Source',
    'SourceStore',
    'claim_due_jobs',
    'compute_next_run_at',
    'content_hash_for',
    'default_scheduler_dsn',
    'execute_job',
    'make_job_id',
    'normalize_url',
    'run_due_once',
    'scrub_error',
    'source_id_for',
]
