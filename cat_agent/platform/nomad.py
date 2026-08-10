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

"""Nomad HTTP API wrapper."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional

from cat_agent.platform.config import PlatformConfig


class NomadError(Exception):
    """Base Nomad client error."""


class NomadUnreachable(NomadError):
    """Nomad address could not be reached."""


class NomadRejected(NomadError):
    """Nomad rejected the request (4xx/5xx with body)."""


class NomadNotFound(NomadError):
    """Job or allocation was not found."""


class NomadClient:
    def __init__(self, config: PlatformConfig, *, session: Any = None) -> None:
        import requests

        self.config = config
        self._session = session or requests.Session()
        self._base = config.nomad_addr.rstrip('/')

    def _headers(self) -> Dict[str, str]:
        h = {'Content-Type': 'application/json'}
        if self.config.nomad_token:
            h['X-Nomad-Token'] = self.config.nomad_token
        return h

    def _params(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        p: Dict[str, Any] = {}
        if self.config.namespace:
            p['namespace'] = self.config.namespace
        if extra:
            p.update(extra)
        return p

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f'{self._base}{path}'
        try:
            resp = self._session.request(
                method,
                url,
                headers=self._headers(),
                params=self._params(kwargs.pop('params', None)),
                timeout=kwargs.pop('timeout', 30),
                **kwargs,
            )
        except Exception as exc:  # requests.RequestException
            raise NomadUnreachable(
                f'Nomad unreachable at {self._base}: {exc}'
            ) from exc
        if resp.status_code == 404:
            raise NomadNotFound(resp.text or f'not found: {path}')
        if resp.status_code == 403:
            raise NomadRejected(
                'Nomad rejected the request (403). Check the ACL token scopes.'
            )
        if resp.status_code >= 400:
            raise NomadRejected(
                f'Nomad rejected the request ({resp.status_code}): {resp.text[:500]}'
            )
        if not resp.content:
            return None
        try:
            return resp.json()
        except Exception:
            return resp.text

    def status_leader(self) -> str:
        return str(self._request('GET', '/v1/status/leader'))

    def nodes(self) -> List[Dict[str, Any]]:
        data = self._request('GET', '/v1/nodes')
        return list(data or [])

    def node(self, node_id: str) -> Dict[str, Any]:
        return dict(self._request('GET', f'/v1/node/{node_id}') or {})

    def parse_job(self, hcl: str) -> Dict[str, Any]:
        return dict(
            self._request(
                'POST',
                '/v1/jobs/parse',
                json={'JobHCL': hcl, 'Canonicalize': True},
            )
            or {}
        )

    def submit(self, job: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self._request('POST', '/v1/jobs', json={'Job': job}) or {})

    def submit_hcl(self, hcl: str) -> Dict[str, Any]:
        job = self.parse_job(hcl)
        return self.submit(job)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return dict(self._request('GET', f'/v1/job/{job_id}') or {})

    def list_jobs(self) -> List[Dict[str, Any]]:
        return list(self._request('GET', '/v1/jobs') or [])

    def list_agents(self, team: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for stub in self.list_jobs():
            jid = stub.get('ID') or stub.get('Name')
            if not jid:
                continue
            try:
                job = self.get_job(jid)
            except NomadNotFound:
                continue
            meta = job.get('Meta') or {}
            if meta.get('managed_by') != 'cat-agent':
                continue
            if team and meta.get('team') != team:
                continue
            out.append(job)
        return out

    def allocations(self, job_id: str) -> List[Dict[str, Any]]:
        return list(self._request('GET', f'/v1/job/{job_id}/allocations') or [])

    def deployment_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        deps = self._request('GET', f'/v1/job/{job_id}/deployments') or []
        return deps[0] if deps else None

    def job_versions(self, job_id: str) -> List[Dict[str, Any]]:
        data = self._request('GET', f'/v1/job/{job_id}/versions') or {}
        if isinstance(data, dict):
            return list(data.get('Versions') or [])
        return list(data)

    def stop(self, job_id: str, *, purge: bool = True) -> Dict[str, Any]:
        return dict(
            self._request(
                'DELETE',
                f'/v1/job/{job_id}',
                params={'purge': 'true' if purge else 'false'},
            )
            or {}
        )

    def dispatch(
        self,
        job_id: str,
        *,
        payload: str,
        meta: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {'Payload': payload}
        if meta:
            body['Meta'] = meta
        return dict(self._request('POST', f'/v1/job/{job_id}/dispatch', json=body) or {})

    def logs(
        self,
        alloc_id: str,
        task: str,
        *,
        stderr: bool = False,
        follow: bool = False,
    ) -> str:
        params = {
            'task': task,
            'type': 'stderr' if stderr else 'stdout',
            'plain': 'true',
        }
        # follow streaming is left to CLI via subprocess nomad; plain one-shot here
        data = self._request('GET', f'/v1/client/fs/logs/{alloc_id}', params=params)
        return data if isinstance(data, str) else str(data or '')

    def wait_healthy(self, job_id: str, *, timeout: float = 180.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            allocs = self.allocations(job_id)
            running = [
                a
                for a in allocs
                if a.get('ClientStatus') == 'running'
                and (a.get('DesiredStatus') or 'run') == 'run'
            ]
            if running:
                return
            failed = [a for a in allocs if a.get('ClientStatus') == 'failed']
            if failed and not running:
                raise NomadRejected(
                    f'job {job_id} has failed allocations and is not healthy'
                )
            time.sleep(2)
        raise NomadRejected(f'timed out waiting for job {job_id} to become healthy')

    def watch_deployment(self, job_id: str, *, timeout: float = 300.0) -> Iterator[str]:
        deadline = time.time() + timeout
        last = ''
        while time.time() < deadline:
            dep = self.deployment_status(job_id)
            status = (dep or {}).get('Status') or 'pending'
            line = f'deployment {status}'
            if line != last:
                yield line
                last = line
            if status in ('successful', 'cancelled', 'failed'):
                if status != 'successful':
                    raise NomadRejected(f'deployment ended with status {status}')
                return
            # batch / no deployment: fall back to alloc health
            if dep is None:
                allocs = self.allocations(job_id)
                if any(a.get('ClientStatus') == 'running' for a in allocs):
                    yield 'allocation running'
                    return
                if any(a.get('ClientStatus') == 'complete' for a in allocs):
                    yield 'allocation complete'
                    return
            time.sleep(2)
        raise NomadRejected(f'timed out watching deployment for {job_id}')
