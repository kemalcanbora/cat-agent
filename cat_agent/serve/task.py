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

"""Entrypoint for Nomad-dispatched one-shot agent tasks.

Usage (container)::

    python -m cat_agent.serve.task

Reads JSON payload from ``CAT_AGENT_PAYLOAD`` (file path), resolves the registry
from ``CAT_AGENT_ENTRYPOINT``, runs one ``arun_nonstream``, writes the result
next to the payload (or stdout), exits 0 on success and 1 on failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _load_payload(path: str) -> Dict[str, Any]:
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode('utf-8'))
    if not isinstance(data, dict):
        raise ValueError('payload must be a JSON object')
    return data


def _write_result(payload_path: str, result: Dict[str, Any]) -> None:
    out = Path(payload_path).with_suffix('.result.json')
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')


async def _run_once(
    *,
    entrypoint: str,
    payload: Dict[str, Any],
    agent_name: Optional[str] = None,
) -> Dict[str, Any]:
    from cat_agent.serve.factory import load_registry
    from cat_agent.serve.stream import final_content, messages_to_dicts

    registry = load_registry(entrypoint)
    await registry.build_deferred()
    try:
        if agent_name:
            key = agent_name
        else:
            names = [info.name for info in registry.list_info()]
            if len(names) != 1:
                raise RuntimeError(
                    'payload must set agent when registry has != 1 agent; '
                    f'got {names!r}'
                )
            key = names[0]
        agent = registry.get(key)
        messages = payload.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('payload.messages must be a non-empty list')
        kwargs = {}
        if payload.get('run_timeout') is not None:
            kwargs['run_timeout'] = float(payload['run_timeout'])
        last = await agent.arun_nonstream(messages, **kwargs)
        dicts = messages_to_dicts(last)
        return {
            'agent': key,
            'messages': dicts,
            'content': final_content(dicts),
            'job_id': os.environ.get('CAT_AGENT_JOB_ID', ''),
        }
    finally:
        for a in list(registry.ready_agents()):
            aclose = getattr(a, 'aclose', None)
            if aclose is not None:
                await aclose()


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    entrypoint = os.environ.get('CAT_AGENT_ENTRYPOINT', '').strip()
    payload_path = os.environ.get('CAT_AGENT_PAYLOAD', '').strip()
    if not entrypoint:
        print('CAT_AGENT_ENTRYPOINT is required', file=sys.stderr)
        return 1
    if not payload_path:
        print('CAT_AGENT_PAYLOAD is required (path to JSON payload file)', file=sys.stderr)
        return 1
    try:
        payload = _load_payload(payload_path)
        agent_name = payload.get('agent')
        if agent_name is not None:
            agent_name = str(agent_name)
        result = asyncio.run(
            _run_once(
                entrypoint=entrypoint,
                payload=payload,
                agent_name=agent_name,
            )
        )
        _write_result(payload_path, result)
        print(json.dumps({'ok': True, 'job_id': result.get('job_id')}))
        return 0
    except Exception as exc:
        err = {'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)}
        print(json.dumps(err), file=sys.stderr)
        try:
            _write_result(payload_path, err)
        except Exception:
            pass
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
