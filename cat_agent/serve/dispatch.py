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

"""Nomad parameterized job dispatch for hybrid agent jobs."""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, Mapping, Optional, Protocol

from cat_agent.settings import SERVE_DISPATCH_MAX_PAYLOAD_BYTES


class PayloadTooLarge(ValueError):
    """Dispatch payload exceeds Nomad's 16KiB hard cap (or configured max)."""

    def __init__(self, size: int, limit: int):
        self.size = size
        self.limit = limit
        super().__init__(
            f'dispatch payload is {size} bytes; Nomad rejects payloads over '
            f'{limit} bytes — refuse rather than truncate'
        )


class NomadDispatcher(Protocol):
    def dispatch(
        self,
        job_id: str,
        *,
        payload: str,
        meta: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        ...

    def get_job(self, job_id: str) -> Dict[str, Any]:
        ...

    def allocations(self, job_id: str) -> list:
        ...


def encode_payload(data: Mapping[str, Any] | bytes | str) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode('utf-8')
    return json.dumps(data, separators=(',', ':'), sort_keys=True).encode('utf-8')


def validate_payload_size(
    raw: bytes,
    *,
    limit: int | None = None,
) -> None:
    lim = SERVE_DISPATCH_MAX_PAYLOAD_BYTES if limit is None else int(limit)
    if len(raw) > lim:
        raise PayloadTooLarge(len(raw), lim)


def map_nomad_status(job: Mapping[str, Any], allocs: list) -> str:
    """Map Nomad job/alloc state onto the shared job status vocabulary."""
    status = (job.get('Status') or '').lower()
    if status == 'dead':
        failed = any(a.get('ClientStatus') == 'failed' for a in allocs)
        return 'failed' if failed else 'succeeded'
    if status == 'running':
        return 'running'
    if status in ('pending',):
        return 'queued'
    return status or 'queued'


class DispatchClient:
    """Submit and poll parameterized Nomad jobs for an agent."""

    def __init__(
        self,
        nomad: NomadDispatcher,
        *,
        dispatch_job_id: str | None = None,
        max_payload_bytes: int | None = None,
    ):
        self.nomad = nomad
        self.dispatch_job_id = (
            dispatch_job_id
            or os.environ.get('CAT_AGENT_DISPATCH_JOB_ID', '').strip()
            or ''
        )
        self.max_payload_bytes = (
            SERVE_DISPATCH_MAX_PAYLOAD_BYTES
            if max_payload_bytes is None
            else int(max_payload_bytes)
        )

    def submit(
        self,
        *,
        payload: Mapping[str, Any] | bytes | str,
        job_id: str,
        requested_by: str = 'api',
    ) -> str:
        if not self.dispatch_job_id:
            raise RuntimeError(
                'CAT_AGENT_DISPATCH_JOB_ID is not set; cannot dispatch jobs'
            )
        raw = encode_payload(payload)
        validate_payload_size(raw, limit=self.max_payload_bytes)
        # Nomad expects base64-encoded Payload in the JSON API.
        b64 = base64.b64encode(raw).decode('ascii')
        result = self.nomad.dispatch(
            self.dispatch_job_id,
            payload=b64,
            meta={'job_id': job_id, 'requested_by': requested_by},
        )
        # Dispatched instance ID is the job_id we expose to callers.
        dispatched = (
            result.get('DispatchedJobID')
            or result.get('EvalID')
            or job_id
        )
        return str(dispatched)

    def status(self, dispatched_job_id: str) -> Dict[str, Any]:
        job = self.nomad.get_job(dispatched_job_id)
        allocs = self.nomad.allocations(dispatched_job_id)
        state = map_nomad_status(job, allocs)
        return {
            'job_id': dispatched_job_id,
            'state': state,
            'nomad_status': job.get('Status'),
            'allocations': [
                {
                    'id': a.get('ID'),
                    'client_status': a.get('ClientStatus'),
                    'desired_status': a.get('DesiredStatus'),
                }
                for a in allocs
            ],
        }
