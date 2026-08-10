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

"""Kubernetes CronJob / one-shot scheduler entry point."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
from typing import Optional

from cat_agent.env import load_env_file

load_env_file()


def _owner_name() -> str:
    return (
        os.getenv('POD_NAME')
        or os.getenv('HOSTNAME')
        or socket.gethostname()
        or 'oneshot'
    )


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry: claim due jobs, execute, exit 0/1 for CronJob backoffLimit."""
    del argv  # reserved for future flags
    from cat_agent.scheduling.store import JobStore, default_scheduler_dsn
    from cat_agent.scheduling.runner import run_due_once
    from cat_agent.settings import SCHEDULER_JOB_LIMIT, SCHEDULER_LEASE_SECONDS
    from cat_agent.tools.base import enable_optional_tools

    dsn = os.getenv('CAT_AGENT_SCHEDULER_DSN') or default_scheduler_dsn()
    owner = _owner_name()
    limit = int(os.getenv('CAT_AGENT_JOB_LIMIT', str(SCHEDULER_JOB_LIMIT)))
    lease_seconds = int(os.getenv('CAT_AGENT_LEASE_SECONDS', str(SCHEDULER_LEASE_SECONDS)))

    store = JobStore(dsn=dsn)
    enable_optional_tools('web_search', 'web_extractor')

    stopping = False

    def _on_signal(signum, frame):
        nonlocal stopping
        stopping = True
        try:
            store.release_all_leases(owner)
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    async def _run():
        return await run_due_once(
            store,
            owner=owner,
            limit=limit,
            lease_seconds=lease_seconds,
        )

    try:
        results = asyncio.run(_run())
    except KeyboardInterrupt:
        store.release_all_leases(owner)
        return 1
    finally:
        store.release_all_leases(owner)

    failed = 0
    for run in results:
        line = {
            'job_id': run.job_id,
            'run_id': run.id,
            'status': run.status,
            'sources_count': run.sources_count,
            'error': run.error,
            'trace_id': run.trace_id,
            'owner': owner,
        }
        print(json.dumps(line, ensure_ascii=False), flush=True)
        if run.status == 'failed':
            failed += 1

    if stopping:
        return 1
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
