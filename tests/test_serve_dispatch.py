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

"""Tests for Nomad dispatch client."""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

import pytest

from cat_agent.serve.dispatch import (
    DispatchClient,
    PayloadTooLarge,
    encode_payload,
    map_nomad_status,
    validate_payload_size,
)


class FakeNomad:
    def __init__(self):
        self.calls: List[dict] = []
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.allocs: Dict[str, list] = {}

    def dispatch(
        self,
        job_id: str,
        *,
        payload: str,
        meta: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        self.calls.append({'job_id': job_id, 'payload': payload, 'meta': meta})
        dispatched = f'{job_id}/{meta["job_id"]}' if meta else f'{job_id}/x'
        self.jobs[dispatched] = {'Status': 'running', 'ID': dispatched}
        self.allocs[dispatched] = [{'ID': 'a1', 'ClientStatus': 'running', 'DesiredStatus': 'run'}]
        return {'DispatchedJobID': dispatched, 'EvalID': 'e1'}

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return dict(self.jobs.get(job_id) or {'Status': 'pending', 'ID': job_id})

    def allocations(self, job_id: str) -> list:
        return list(self.allocs.get(job_id) or [])


class TestDispatchPayload:

    def test_reject_over_16kib(self):
        raw = b'x' * (16 * 1024 + 1)
        with pytest.raises(PayloadTooLarge) as ei:
            validate_payload_size(raw, limit=16 * 1024)
        assert '16' in str(ei.value) or '16384' in str(ei.value)
        assert 'truncate' in str(ei.value).lower() or 'over' in str(ei.value).lower()

    def test_dispatch_id_is_returned_job_id(self):
        nomad = FakeNomad()
        client = DispatchClient(
            nomad, dispatch_job_id='agent-demo-calculator-task', max_payload_bytes=16 * 1024,
        )
        jid = client.submit(
            payload={'messages': [{'role': 'user', 'content': 'hi'}]},
            job_id='job-abc',
            requested_by='tester',
        )
        assert jid == 'agent-demo-calculator-task/job-abc'
        assert nomad.calls[0]['meta']['job_id'] == 'job-abc'
        # Payload is base64 and decodes to JSON — never truncated
        decoded = base64.b64decode(nomad.calls[0]['payload'])
        assert b'hi' in decoded

    def test_submit_rejects_large_payload(self):
        nomad = FakeNomad()
        client = DispatchClient(
            nomad, dispatch_job_id='task', max_payload_bytes=100,
        )
        with pytest.raises(PayloadTooLarge):
            client.submit(payload={'messages': [{'role': 'user', 'content': 'z' * 200}]}, job_id='j')
        assert nomad.calls == []

    def test_status_maps_nomad_states(self):
        assert map_nomad_status({'Status': 'pending'}, []) == 'queued'
        assert map_nomad_status({'Status': 'running'}, []) == 'running'
        assert map_nomad_status({'Status': 'dead'}, [{'ClientStatus': 'complete'}]) == 'succeeded'
        assert map_nomad_status({'Status': 'dead'}, [{'ClientStatus': 'failed'}]) == 'failed'

        nomad = FakeNomad()
        client = DispatchClient(nomad, dispatch_job_id='task')
        jid = client.submit(payload={'messages': [{'role': 'user', 'content': 'x'}]}, job_id='j1')
        st = client.status(jid)
        assert st['state'] == 'running'
        assert st['job_id'] == jid

    def test_encode_sorted_stable(self):
        a = encode_payload({'b': 1, 'a': 2})
        b = encode_payload({'a': 2, 'b': 1})
        assert a == b
