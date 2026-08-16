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

"""Tests for cat_agent.analysis (MAST)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cat_agent.analysis import analyze_trace, render_text_report
from cat_agent.analysis.detectors import (
    detect_loss_of_history,
    detect_step_repetition,
    detect_unaware_termination,
    resolve_evicted_messages,
)
from cat_agent.analysis.taxonomy import MAST_MODES
from cat_agent.llm.schema import ASSISTANT, FUNCTION, USER, Message
from cat_agent.trace.schema import LLMCallPayload, Run, Step, utc_now_iso
from cat_agent.trace.store import JSONLTraceStore


GOLDEN = Path(__file__).parent / 'golden' / 'mast_report.txt'


def _run(**kwargs) -> Run:
    base = dict(
        agent_name='a',
        agent_class='Assistant',
        status='completed',
        initial_messages=[Message(USER, 'task')],
        final_output='answer',
    )
    base.update(kwargs)
    return Run(**base)


def test_taxonomy_has_14_modes():
    assert len(MAST_MODES) == 14
    ids = {m.id for m in MAST_MODES}
    assert ids == {
        '1.1', '1.2', '1.3', '1.4', '1.5',
        '2.1', '2.2', '2.3', '2.4', '2.5', '2.6',
        '3.1', '3.2', '3.3',
    }


def test_clean_run_no_findings():
    run = _run(steps=[
        Step.from_payload(
            step_index=0, kind='llm_call',
            payload={
                'model': 'm',
                'messages_in': [Message(USER, 'hi').model_dump()],
                'message_out': Message(ASSISTANT, 'hello').model_dump(),
                'prompt_tokens': 1,
                'completion_tokens': 1,
            },
        ),
    ])
    result = analyze_trace(run, tiers=('deterministic',))
    assert result.present_findings() == []
    assert 'No MAST failure modes detected' in result.summary


def test_step_repetition_positive():
    steps = []
    for i in range(3):
        steps.append(Step.from_payload(
            step_index=i, kind='tool_call',
            payload={
                'tool_name': 'search',
                'arguments': {'q': 'same'},
                'result_preview': 'x',
                'result_bytes': 1,
                'succeeded': True,
            },
        ))
    run = _run(steps=steps)
    finding = detect_step_repetition(run)
    assert finding.present is True
    assert finding.mode_id == '1.3'


def test_step_repetition_ignores_retry_after_failure():
    steps = [
        Step.from_payload(
            step_index=0, kind='tool_call',
            payload={
                'tool_name': 'fetch', 'arguments': {'url': 'https://x'},
                'result_preview': 'transient', 'result_bytes': 9,
                'succeeded': False, 'error': 'timeout',
            },
        ),
        Step.from_payload(
            step_index=1, kind='tool_call',
            payload={
                'tool_name': 'fetch', 'arguments': {'url': 'https://x'},
                'result_preview': 'ok', 'result_bytes': 2, 'succeeded': True,
            },
        ),
        Step.from_payload(
            step_index=2, kind='tool_call',
            payload={
                'tool_name': 'fetch', 'arguments': {'url': 'https://x'},
                'result_preview': 'ok', 'result_bytes': 2, 'succeeded': True,
            },
        ),
    ]
    # Only two successful calls after a failure → treated as retries / refresh, not stuck loop of 3
    assert detect_step_repetition(_run(steps=steps)).present is False


def test_step_repetition_ignores_pagination():
    steps = [
        Step.from_payload(
            step_index=i, kind='tool_call',
            payload={
                'tool_name': 'list_items',
                'arguments': {'q': 'pods', 'page': i + 1, 'page_size': 10},
                'result_preview': f'page{i}', 'result_bytes': 5, 'succeeded': True,
            },
        )
        for i in range(5)
    ]
    assert detect_step_repetition(_run(steps=steps)).present is False


def test_step_repetition_ignores_spaced_identical_reads():
    steps = [
        Step.from_payload(
            step_index=0, kind='tool_call',
            payload={
                'tool_name': 'lookup', 'arguments': {'key': 'config'},
                'result_preview': 'v1', 'result_bytes': 2, 'succeeded': True,
            },
        ),
        Step.from_payload(
            step_index=1, kind='tool_call',
            payload={
                'tool_name': 'add', 'arguments': {'a': 1, 'b': 2},
                'result_preview': '3', 'result_bytes': 1, 'succeeded': True,
            },
        ),
        Step.from_payload(
            step_index=2, kind='tool_call',
            payload={
                'tool_name': 'lookup', 'arguments': {'key': 'config'},
                'result_preview': 'v1', 'result_bytes': 2, 'succeeded': True,
            },
        ),
        Step.from_payload(
            step_index=3, kind='tool_call',
            payload={
                'tool_name': 'lookup', 'arguments': {'key': 'config'},
                'result_preview': 'v1', 'result_bytes': 2, 'succeeded': True,
            },
        ),
    ]
    # Spaced by other work → no contiguous streak of 3
    assert detect_step_repetition(_run(steps=steps)).present is False


CLEAN_FIXTURES = Path(__file__).parent / 'fixtures' / 'clean_traces'


def test_clean_fixture_traces_have_no_tier1_findings():
    from cat_agent.trace.store import load_runs_from_jsonl

    files = sorted(CLEAN_FIXTURES.glob('clean_*.jsonl'))
    assert len(files) >= 20, f'expected ≥20 clean fixtures, found {len(files)}'
    for path in files:
        runs = load_runs_from_jsonl(path)
        assert runs, path
        for run in runs.values():
            assert run.status == 'completed', (path, run.status)
            assert run.final_output and str(run.final_output).strip(), path.name
            result = analyze_trace(run, tiers=('deterministic',))
            present = result.present_findings()
            assert present == [], (
                f'{path.name} / {run.agent_name}: unexpected findings '
                f'{[(f.mode_id, f.evidence_steps, f.explanation) for f in present]}'
            )
            for mode_id in ('1.3', '1.4', '1.5'):
                f = next(x for x in result.findings if x.mode_id == mode_id)
                assert f.present is False, (path.name, mode_id, f.evidence_steps)


def test_unaware_termination_positive():
    run = _run(status='terminated', termination_reason='max_steps', final_output=None, steps=[])
    finding = detect_unaware_termination(run)
    assert finding.present is True


def test_unaware_termination_negative_with_answer():
    run = _run(status='terminated', termination_reason='max_steps', final_output='done')
    assert detect_unaware_termination(run).present is False


def test_loss_of_history_positive():
    steps = [
        Step.from_payload(
            step_index=0, kind='tool_call',
            payload={
                'tool_name': 'search', 'arguments': {'q': 'pods'},
                'result_preview': 'cluster has 12 pods running now',
                'result_bytes': 10, 'succeeded': True,
            },
        ),
        Step.from_payload(
            step_index=1, kind='context_op',
            payload={
                'operation': 'mask',
                'messages_before': 5, 'messages_after': 5,
                'tokens_before': 100, 'tokens_after': 40,
                'strategy_name': 'observation_masking',
                'evicted_message_ids': ['abc'],
            },
        ),
        Step.from_payload(
            step_index=2, kind='tool_call',
            payload={
                'tool_name': 'search', 'arguments': {'q': 'pods'},
                'result_preview': 'again',
                'result_bytes': 5, 'succeeded': True,
            },
        ),
    ]
    run = _run(steps=steps)
    assert detect_loss_of_history(run).present is True


def test_loss_of_history_resolves_ids_after_jsonl_roundtrip(tmp_path):
    """Message.id is exclude=True; traces must still re-inject ids for 1.4."""
    body = (
        'cluster inventory: pod-0 Ready, pod-1 CrashLoopBackOff, '
        'pod-2 OOMKilled — full dump for correlation'
    )
    tool_msg = Message(FUNCTION, body, name='kubectl', id='evict-me-001')
    user_msg = Message(USER, 'diagnose the cluster', id='keep-me-user')

    steps = [
        Step.from_payload(
            step_index=0,
            kind='llm_call',
            payload=LLMCallPayload(
                model='stub',
                messages_in=[user_msg, tool_msg],
                message_out=Message(ASSISTANT, 'looking'),
                prompt_tokens=10,
                completion_tokens=2,
            ),
        ),
        Step.from_payload(
            step_index=1,
            kind='tool_call',
            payload={
                'tool_name': 'kubectl',
                'arguments': {'cmd': 'get pods'},
                'result_preview': body[:80],
                'result_bytes': len(body),
                'succeeded': True,
            },
        ),
        Step.from_payload(
            step_index=2,
            kind='context_op',
            payload={
                'operation': 'mask',
                'messages_before': 4,
                'messages_after': 4,
                'tokens_before': 500,
                'tokens_after': 120,
                'strategy_name': 'observation_masking',
                'evicted_message_ids': [tool_msg.id],
            },
        ),
        Step.from_payload(
            step_index=3,
            kind='tool_call',
            payload={
                'tool_name': 'kubectl',
                'arguments': {'cmd': 'get pods'},
                'result_preview': 'again',
                'result_bytes': 5,
                'succeeded': True,
            },
        ),
    ]
    run = _run(steps=steps, initial_messages=[user_msg])

    # In-process: ids present on dumped llm_call payload.
    assert steps[0].payload['messages_in'][1]['id'] == 'evict-me-001'
    assert tool_msg.id in resolve_evicted_messages(run)

    path = tmp_path / 'loh.jsonl'
    store = JSONLTraceStore(path)
    store.write_run_header(run)
    for step in run.steps:
        store.append_step(run.run_id, step)
    store.finalize_run(run)

    loaded = store.load_run(run.run_id)
    assert loaded is not None

    # Public Message.model_dump still omits id.
    assert 'id' not in Message(USER, 'x', id='nope').model_dump()

    # After JSONL reload, ids survive inside the trace payload.
    reloaded_in = loaded.steps[0].payload['messages_in']
    assert reloaded_in[1]['id'] == 'evict-me-001'
    assert loaded.steps[2].payload['evicted_message_ids'] == ['evict-me-001']

    resolved = resolve_evicted_messages(loaded)
    assert set(resolved) == {'evict-me-001'}
    assert body[:40].lower() in str(resolved['evict-me-001'].get('content', '')).lower()

    finding = detect_loss_of_history(loaded)
    assert finding.present is True
    assert 'unresolved' not in finding.explanation


def test_judge_mocked():
    run = _run(steps=[])
    llm = MagicMock()
    payload = {
        'findings': [
            {
                'mode_id': m.id,
                'present': m.id == '1.1',
                'confidence': 0.8 if m.id == '1.1' else 0.0,
                'evidence_steps': [0] if m.id == '1.1' else [],
                'explanation': 'x',
            }
            for m in MAST_MODES if m.tier == 'judge'
        ]
    }
    llm.chat.return_value = [[Message(ASSISTANT, json.dumps(payload))]]
    # chat is used as iterator
    def _chat(**kwargs):
        yield [Message(ASSISTANT, json.dumps(payload))]
    llm.chat = _chat
    result = analyze_trace(run, judge_llm=llm, tiers=('deterministic', 'judge'))
    present = {f.mode_id for f in result.present_findings()}
    assert '1.1' in present


def test_deterministic_not_overridden_by_judge():
    steps = [
        Step.from_payload(
            step_index=i, kind='tool_call',
            payload={
                'tool_name': 't', 'arguments': {'a': 1},
                'result_preview': 'r', 'result_bytes': 1, 'succeeded': True,
            },
        )
        for i in range(3)
    ]
    run = _run(steps=steps)

    def _chat(**kwargs):
        # Judge tries to clear 1.3 — must not win
        findings = [
            {
                'mode_id': m.id,
                'present': False,
                'confidence': 0.0,
                'evidence_steps': [],
                'explanation': 'no',
            }
            for m in MAST_MODES if m.tier == 'judge'
        ]
        findings.append({
            'mode_id': '1.3', 'present': False, 'confidence': 0.0,
            'evidence_steps': [], 'explanation': 'nope',
        })
        yield [Message(ASSISTANT, json.dumps({'findings': findings}))]

    llm = MagicMock()
    llm.chat = _chat
    result = analyze_trace(run, judge_llm=llm, tiers=('deterministic', 'judge'))
    f13 = next(f for f in result.findings if f.mode_id == '1.3')
    assert f13.present is True
    assert f13.deterministic is True


def test_golden_report(tmp_path):
    run = _run(
        run_id='golden-run',
        status='terminated',
        termination_reason='max_steps',
        final_output=None,
        steps=[
            Step.from_payload(
                step_index=i, kind='tool_call',
                payload={
                    'tool_name': 't', 'arguments': {'a': 1},
                    'result_preview': 'r', 'result_bytes': 1, 'succeeded': True,
                },
            )
            for i in range(3)
        ],
    )
    result = analyze_trace(run, tiers=('deterministic',))
    text = render_text_report(result)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    # Stable enough: key phrases
    assert '1.3' in text and '1.5' in text
    assert 'YES' in text
    assert 'Cemri' in text or '2503.13657' in text
