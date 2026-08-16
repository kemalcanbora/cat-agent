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

"""LLM-as-judge annotator for Tier-2 MAST modes (opt-in only)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from cat_agent.analysis.detectors import Finding
from cat_agent.analysis.taxonomy import MAST_MODES, PAPER_CITATION, modes_for_tier
from cat_agent.llm.schema import SYSTEM, USER, Message
from cat_agent.trace.redact import redact_obj
from cat_agent.trace.schema import Run

_JUDGE_SCHEMA_HINT = """
Return STRICT JSON only:
{
  "findings": [
    {
      "mode_id": "1.1",
      "present": true,
      "confidence": 0.0-1.0,
      "evidence_steps": [0, 3],
      "explanation": "one line"
    }
  ]
}
Cover every Tier-2 mode id exactly once. Do not invent mode ids.
""".strip()


def render_trace_compact(run: Run, *, max_chars: int = 12000) -> str:
    lines = [
        f'run_id={run.run_id} agent={run.agent_name}/{run.agent_class}',
        f'status={run.status} termination_reason={run.termination_reason}',
        f'final_output={(run.final_output or "")[:500]}',
        'steps:',
    ]
    for step in run.steps:
        payload = {k: v for k, v in step.payload.items() if k not in ('messages_in',)}
        if 'message_out' in payload:
            mo = payload['message_out']
            if isinstance(mo, dict):
                payload['message_out'] = {
                    'role': mo.get('role'),
                    'content': str(mo.get('content', ''))[:300],
                }
        if 'result_preview' in payload:
            payload['result_preview'] = str(payload['result_preview'])[:300]
        lines.append(
            f'  [{step.step_index}] kind={step.kind} parent={step.parent_step_id} '
            f'{json.dumps(payload, ensure_ascii=False, default=str)[:800]}'
        )
    text = '\n'.join(lines)
    if len(text) > max_chars:
        return text[:max_chars] + '\n...[truncated]'
    return text


def _mode_catalog_for_judge() -> str:
    rows = []
    for mode in modes_for_tier('judge'):
        rows.append(f'{mode.id} {mode.name}: {mode.definition}')
    return '\n'.join(rows)


def judge_trace(
    run: Run,
    judge_llm: Any,
    *,
    opt_in: bool = False,
) -> List[Finding]:
    if not opt_in:
        raise PermissionError(
            'Judge analysis requires explicit opt-in (pass judge_llm with tiers including '
            '"judge", or --judge on the CLI). Trace content is never sent otherwise.'
        )
    if judge_llm is None:
        raise ValueError('judge_llm is required for Tier-2 analysis')

    redacted = redact_obj(run.model_dump(mode='json'))
    compact = render_trace_compact(Run.model_validate(redacted))
    prompt = [
        Message(
            role=SYSTEM,
            content=(
                f'You are a MAST failure annotator. {PAPER_CITATION}\n'
                f'Tier-2 modes:\n{_mode_catalog_for_judge()}\n{_JUDGE_SCHEMA_HINT}'
            ),
        ),
        Message(role=USER, content=f'Trace:\n{compact}'),
    ]

    raw = _call_judge(judge_llm, prompt)
    try:
        data = _parse_judge_json(raw)
    except (json.JSONDecodeError, ValueError):
        raw = _call_judge(judge_llm, prompt + [
            Message(role=USER, content='Your previous reply was invalid JSON. Reply with JSON only.'),
        ])
        data = _parse_judge_json(raw)

    findings: List[Finding] = []
    by_id = {f.get('mode_id'): f for f in data.get('findings') or [] if isinstance(f, dict)}
    for mode in modes_for_tier('judge'):
        item = by_id.get(mode.id) or {}
        findings.append(Finding(
            mode_id=mode.id,
            present=bool(item.get('present')),
            confidence=float(item.get('confidence') or 0.0),
            evidence_steps=[int(x) for x in (item.get('evidence_steps') or [])],
            explanation=str(item.get('explanation') or ''),
            deterministic=False,
        ))
    return findings


def _call_judge(llm: Any, messages: List[Message]) -> str:
    final = []
    chat = getattr(llm, 'chat', None)
    if chat is None:
        raise TypeError('judge_llm must provide chat()')
    for out in chat(messages=messages, stream=False):
        if out:
            final = out
    if not final:
        return '{}'
    content = final[-1].content
    return content if isinstance(content, str) else str(content)


def _parse_judge_json(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith('```'):
        text = text.strip('`')
        if text.startswith('json'):
            text = text[4:]
    data = json.loads(text)
    if not isinstance(data, dict) or 'findings' not in data:
        raise ValueError('Judge output missing findings')
    return data
