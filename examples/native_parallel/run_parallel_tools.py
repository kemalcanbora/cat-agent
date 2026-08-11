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

"""Live parallel native tool-call check against any OpenAI-compatible endpoint.

Does **not** assume cat-agent-stack is running. Configure via env:

  CAT_AGENT_GATEWAY_URL or OPENAI_BASE_URL   — API base URL (required to run)
  OPENAI_API_KEY or CAT_AGENT_GATEWAY_KEY    — API key (default: EMPTY)
  CAT_AGENT_GATEWAY_MODEL or OPENAI_MODEL    — model id (default: gpt-4o-mini)

If the URL is unset or the endpoint is unreachable, exit 0 with a skip message.
On success, prints the outbound request (tools + messages) and the returned
``tool_calls`` array so the Phase 2 gate can be verified by inspection.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a city.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {'type': 'string', 'description': 'City name'},
                },
                'required': ['city'],
            },
        },
    },
]

MESSAGES = [
    {
        'role': 'user',
        'content': (
            'Call get_weather twice in one response (parallel tool calls): '
            'once for Paris and once for Berlin. Do not answer in prose first.'
        ),
    },
]


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return default


def _skip(reason: str) -> int:
    print(f'SKIP: {reason}')
    return 0


def _probe(base_url: str, api_key: str) -> str | None:
    """Return None if reachable enough to attempt chat; else a skip reason."""
    models_url = base_url.rstrip('/') + '/models'
    req = urllib.request.Request(
        models_url,
        headers={'Authorization': f'Bearer {api_key}'},
        method='GET',
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status >= 500:
                return f'gateway returned HTTP {resp.status} for GET /models'
            return None
    except urllib.error.HTTPError as exc:
        # Many gateways require auth for /models or return 404; still try chat.
        if exc.code in (401, 403, 404):
            return None
        return f'gateway GET /models failed: HTTP {exc.code}'
    except Exception as exc:  # noqa: BLE001 — any network failure → skip
        return f'gateway unreachable at {base_url!r}: {exc}'


def main() -> int:
    base_url = _env('CAT_AGENT_GATEWAY_URL', 'OPENAI_BASE_URL')
    if not base_url:
        return _skip(
            'set CAT_AGENT_GATEWAY_URL or OPENAI_BASE_URL to run the live check '
            '(OpenAI-compatible /v1 base, e.g. http://127.0.0.1:4000/v1).'
        )

    api_key = _env('OPENAI_API_KEY', 'CAT_AGENT_GATEWAY_KEY', default='EMPTY') or 'EMPTY'
    model = _env('CAT_AGENT_GATEWAY_MODEL', 'OPENAI_MODEL', default='gpt-4o-mini') or 'gpt-4o-mini'

    probe_reason = _probe(base_url, api_key)
    if probe_reason:
        return _skip(probe_reason)

    payload = {
        'model': model,
        'messages': MESSAGES,
        'tools': TOOLS,
        'tool_choice': 'auto',
        'parallel_tool_calls': True,
        'stream': False,
    }

    print('=== OUTBOUND REQUEST ===')
    print(json.dumps({
        'url': base_url.rstrip('/') + '/chat/completions',
        'model': model,
        'tools': TOOLS,
        'messages': MESSAGES,
        'parallel_tool_calls': True,
    }, indent=2))

    body = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        base_url.rstrip('/') + '/chat/completions',
        data=body,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode('utf-8')
            data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return _skip(f'chat/completions failed: {exc}')

    message = (data.get('choices') or [{}])[0].get('message') or {}
    tool_calls = message.get('tool_calls') or []

    print('=== INBOUND RESPONSE (tool_calls) ===')
    print(json.dumps(tool_calls, indent=2))

    # Also exercise cat_agent oai path so wire conversion is covered live.
    try:
        from cat_agent.llm.oai import TextChatAtOAI
        from cat_agent.llm.schema import Message

        llm = TextChatAtOAI({
            'model': model,
            'api_key': api_key,
            'api_base': base_url,
        })
        assert llm.use_raw_api is True
        functions = [t['function'] for t in TOOLS]
        out = llm.chat(
            messages=[Message(role='user', content=MESSAGES[0]['content'])],
            functions=functions,
            stream=False,
        )
        internal = [
            {
                'tool_calls': [tc.model_dump() for tc in (m.tool_calls or [])],
            }
            for m in out
            if m.tool_calls
        ]
        print('=== CAT_AGENT INTERNAL (from _chat_no_stream) ===')
        print(json.dumps(internal, indent=2))

        # Round-trip result messages through the converter and show wire shape.
        history = [m.model_dump() for m in out if m.tool_calls]
        for m in out:
            if m.tool_calls:
                for tc in m.tool_calls:
                    history.append({
                        'role': 'function',
                        'name': tc.function.name,
                        'content': f'result for {tc.function.arguments}',
                        'tool_call_id': tc.id,
                        'extra': {'function_id': tc.id},
                    })
        wire = TextChatAtOAI._conv_cat_agent_messages_to_oai(history)
        print('=== CAT_AGENT OUTBOUND WIRE (after fake results) ===')
        print(json.dumps(wire, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f'NOTE: cat_agent path not exercised: {exc}')

    if len(tool_calls) < 2:
        print(
            f'FAIL: expected >=2 parallel tool_calls, got {len(tool_calls)}. '
            'Model may have refused parallel calls; try another model.',
            file=sys.stderr,
        )
        return 1

    ids = [tc.get('id') for tc in tool_calls]
    if len(set(ids)) != len(ids) or any(not i for i in ids):
        print(f'FAIL: tool_calls ids not distinct/non-empty: {ids}', file=sys.stderr)
        return 1

    print(f'OK: {len(tool_calls)} parallel tool_calls with ids {ids}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
