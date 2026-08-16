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

"""Deterministic (Tier 1) MAST detectors — zero LLM cost.

Step Repetition (1.3) operationalisation notes
----------------------------------------------
MAST defines 1.3 as *unnecessary* reiteration of *previously completed* steps.
We therefore do **not** flag:

- retries after a failed tool call (same name+args, earlier ``succeeded=False``)
- pagination / cursor walks (args differ only on page/offset/cursor keys)
- the same tool with genuinely different arguments
- identical successful read-only queries spaced apart by other successful work
  (intentional re-reads / refresh)

We **do** flag tight stuck loops: a contiguous streak of the same successful
``(tool_name, normalised_args)`` with no other successful tool activity between,
length ≥ ``min_repetitions`` (default 3). LLM-side repetition requires the same
``messages_in`` hash streak (agent not advancing the transcript).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from cat_agent.analysis.taxonomy import MODES_BY_ID
from cat_agent.trace.schema import Run, Step

# Argument keys that distinguish pagination / cursors — differing only here
# is not step repetition.
_PAGINATION_KEYS = frozenset({
    'page', 'page_number', 'page_num', 'pageno',
    'offset', 'skip', 'limit', 'page_size', 'pagesize', 'per_page', 'pageSize',
    'cursor', 'next_cursor', 'nextCursor', 'continuation', 'continuation_token',
    'next_token', 'nextToken', 'start', 'from', 'after', 'before',
    'index', 'start_index', 'end_index',
})

# Volatile keys stripped before comparing tool args.
_VOLATILE_KEYS = frozenset({
    'request_id', 'requestId', 'nonce', 'timestamp', 'ts', 'uuid', 'idempotency_key',
    'trace_id', 'span_id', 'client_request_id',
})


@dataclass
class Finding:
    mode_id: str
    present: bool
    confidence: float
    evidence_steps: List[int] = field(default_factory=list)
    explanation: str = ''
    deterministic: bool = True


@dataclass
class DetectorConfig:
    """Config for Tier-1 detectors.

    ``min_repetitions`` is the minimum *contiguous successful* streak length
    for 1.3 (default 3). Raising this blindly is discouraged — prefer the
    retry / pagination / spacing rules in :func:`detect_step_repetition`.
    """

    repetition_similarity: float = 0.97  # only for residual near-dup after normalisation
    min_repetitions: int = 3


def _norm_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    except TypeError:
        return str(obj)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def _parse_args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return {'_raw': raw}
    return raw


def normalize_tool_arguments(arguments: Any) -> Any:
    """Canonicalise tool args for repetition comparison.

    - parse JSON strings
    - drop volatile keys (request_id, timestamp, …)
    - recursively sort dict keys
    """
    obj = _parse_args(arguments)

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if k in _VOLATILE_KEYS:
                    continue
                out[str(k)] = walk(v)
            return {k: out[k] for k in sorted(out)}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return value
        return value

    return walk(obj)


def _strip_pagination(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    return {k: v for k, v in obj.items() if k not in _PAGINATION_KEYS}


def _is_pagination_variant(a: Any, b: Any) -> bool:
    """True when args differ only on pagination/cursor keys."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    core_a = _strip_pagination(a)
    core_b = _strip_pagination(b)
    if _norm_json(core_a) != _norm_json(core_b):
        return False
    # Must actually differ on at least one pagination key (else they're identical).
    pag_a = {k: a.get(k) for k in _PAGINATION_KEYS if k in a}
    pag_b = {k: b.get(k) for k in _PAGINATION_KEYS if k in b}
    return pag_a != pag_b


def _tool_fingerprint(step: Step) -> str:
    name = step.payload.get('tool_name', '')
    args = normalize_tool_arguments(step.payload.get('arguments'))
    return _hash_text(f'{name}|{_norm_json(args)}')


def _tool_succeeded(step: Step) -> bool:
    return bool(step.payload.get('succeeded', True))


def detect_step_repetition(run: Run, cfg: Optional[DetectorConfig] = None) -> Finding:
    cfg = cfg or DetectorConfig()
    mode = MODES_BY_ID['1.3']
    evidence: List[int] = []

    tool_steps = [s for s in run.steps if s.kind == 'tool_call']

    def _is_retry_completion(step: Step) -> bool:
        """True for a success that immediately recovers a prior same-key failure.

        Only the first success after a failure is treated as a retry. Further
        identical successes can still form a stuck-loop streak.
        """
        if not _tool_succeeded(step):
            return True
        fp = _tool_fingerprint(step)
        priors = [
            s for s in tool_steps
            if s.step_index < step.step_index and _tool_fingerprint(s) == fp
        ]
        if not priors:
            return False
        return not _tool_succeeded(priors[-1])

    # Contiguous successful streaks of the same fingerprint, skipping retries
    # and ignoring pagination variants when comparing neighbours.
    streak_fp: Optional[str] = None
    streak_indices: List[int] = []
    streak_args: Any = None

    def _flush_streak() -> None:
        nonlocal streak_fp, streak_indices, streak_args
        if streak_fp is not None and len(streak_indices) >= cfg.min_repetitions:
            evidence.extend(streak_indices)
        streak_fp = None
        streak_indices = []
        streak_args = None

    for step in tool_steps:
        if not _tool_succeeded(step) or _is_retry_completion(step):
            _flush_streak()
            continue
        fp = _tool_fingerprint(step)
        args = normalize_tool_arguments(step.payload.get('arguments'))
        if streak_fp is None:
            streak_fp = fp
            streak_indices = [step.step_index]
            streak_args = args
            continue
        if fp == streak_fp:
            streak_indices.append(step.step_index)
            continue
        # Different fingerprint — check pagination variant of same tool.
        if (
            step.payload.get('tool_name') == _tool_name_at(tool_steps, streak_indices[0])
            and _is_pagination_variant(streak_args, args)
        ):
            # Pagination walk breaks a repetition streak without starting a new
            # identical streak on the previous key.
            _flush_streak()
            streak_fp = fp
            streak_indices = [step.step_index]
            streak_args = args
            continue
        # Different successful tool activity — break streak.
        _flush_streak()
        streak_fp = fp
        streak_indices = [step.step_index]
        streak_args = args
    _flush_streak()

    # LLM stuck-loop: identical messages_in hash in a contiguous streak.
    llm_steps = [s for s in run.steps if s.kind == 'llm_call']
    llm_streak_key: Optional[str] = None
    llm_streak: List[int] = []

    def _flush_llm() -> None:
        nonlocal llm_streak_key, llm_streak
        if llm_streak_key is not None and len(llm_streak) >= cfg.min_repetitions:
            evidence.extend(llm_streak)
        llm_streak_key = None
        llm_streak = []

    for step in llm_steps:
        key = _hash_text(_norm_json(step.payload.get('messages_in') or []))
        if llm_streak_key is None or key != llm_streak_key:
            _flush_llm()
            llm_streak_key = key
            llm_streak = [step.step_index]
        else:
            llm_streak.append(step.step_index)
    _flush_llm()

    evidence = sorted(set(evidence))
    present = len(evidence) >= cfg.min_repetitions
    return Finding(
        mode_id=mode.id,
        present=present,
        confidence=1.0 if present else 0.0,
        evidence_steps=evidence,
        explanation=(
            f'Stuck-loop repetition at steps {evidence}'
            if present else 'No unnecessary stuck-loop step repetition detected'
        ),
        deterministic=True,
    )


def _tool_name_at(tool_steps: Sequence[Step], step_index: int) -> Optional[str]:
    for s in tool_steps:
        if s.step_index == step_index:
            return s.payload.get('tool_name')
    return None


def detect_unaware_termination(run: Run) -> Finding:
    mode = MODES_BY_ID['1.5']
    reasons = {'max_steps', 'max_tokens', 'wall_clock', 'max_tool_calls'}
    no_answer = not (run.final_output and str(run.final_output).strip())
    present = (
        run.status == 'terminated'
        and (run.termination_reason or '') in reasons
        and no_answer
    )
    evidence = [s.step_index for s in run.steps[-3:]] if present else []
    return Finding(
        mode_id=mode.id,
        present=present,
        confidence=1.0 if present else 0.0,
        evidence_steps=evidence,
        explanation=(
            f'Terminated for {run.termination_reason} without a final answer'
            if present else 'Termination conditions handled or final answer present'
        ),
        deterministic=True,
    )


def _message_id(msg: Any) -> Optional[str]:
    if msg is None:
        return None
    if isinstance(msg, dict):
        mid = msg.get('id')
        return str(mid) if mid else None
    mid = getattr(msg, 'id', None)
    return str(mid) if mid else None


def _message_content_text(msg: Any) -> str:
    if msg is None:
        return ''
    if isinstance(msg, dict):
        content = msg.get('content', '')
    else:
        content = getattr(msg, 'content', '')
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get('text') or ''))
            else:
                parts.append(str(getattr(item, 'text', '') or ''))
        return ' '.join(parts)
    return str(content or '')


def iter_persisted_messages(run: Run) -> List[Any]:
    """Messages stored on the run (initial + llm_call payloads), Message or dict."""
    out: List[Any] = list(run.initial_messages or [])
    for step in run.steps:
        if step.kind != 'llm_call':
            continue
        out.extend(step.payload.get('messages_in') or [])
        mout = step.payload.get('message_out')
        if mout is not None:
            out.append(mout)
    return out


def resolve_evicted_messages(run: Run) -> Dict[str, Any]:
    """Map ``context_op.evicted_message_ids`` → persisted message bodies.

    Trace dumps re-inject ``Message.id`` (which is ``exclude=True`` on the
    schema) so JSONL reload still lets 1.4 correlate eviction ids with the
    messages that carried that content.
    """
    wanted: set = set()
    for step in run.steps:
        if step.kind != 'context_op':
            continue
        for mid in step.payload.get('evicted_message_ids') or []:
            if mid:
                wanted.add(str(mid))
    if not wanted:
        return {}
    found: Dict[str, Any] = {}
    for msg in iter_persisted_messages(run):
        mid = _message_id(msg)
        if mid in wanted and mid not in found:
            found[mid] = msg
    return found


def detect_loss_of_history(run: Run) -> Finding:
    """Weaker checkable version: re-request of content from an evicted message."""
    mode = MODES_BY_ID['1.4']
    eviction_steps: List[int] = []
    evidence: List[int] = []

    for step in run.steps:
        if step.kind == 'context_op':
            eviction_steps.append(step.step_index)

    resolved = resolve_evicted_messages(run)
    masked_bodies: List[str] = []
    for msg in resolved.values():
        text = _message_content_text(msg).lower().strip()
        if text and 'elided' not in text:
            masked_bodies.append(text[:200])
    for step in run.steps:
        if step.kind == 'tool_call':
            preview = str(step.payload.get('result_preview') or '')
            if preview and 'elided' not in preview.lower():
                masked_bodies.append(preview[:200].lower())

    for step in run.steps:
        if not eviction_steps or step.step_index <= min(eviction_steps):
            continue
        blob = _norm_json(step.payload).lower()
        re_request = bool(re.search(
            r'\b(again|remind|what was|repeat|you (already |previously )?told|earlier)\b',
            blob,
        ))
        overlap = any(body[:40] in blob for body in masked_bodies if len(body) >= 40)
        if re_request and (overlap or step.kind == 'tool_call'):
            evidence.append(step.step_index)

    # Identical successful tool call after a context_op that evicted messages —
    # but only when there was no intervening failed attempt (retry) and the
    # earlier call already succeeded (true re-fetch of lost context).
    prior_success: Dict[str, int] = {}
    seen_ctx = False
    for step in run.steps:
        if step.kind == 'context_op' and step.payload.get('evicted_message_ids'):
            seen_ctx = True
        if step.kind != 'tool_call':
            continue
        fp = _tool_fingerprint(step)
        if not _tool_succeeded(step):
            continue
        if seen_ctx and fp in prior_success:
            evidence.extend([prior_success[fp], step.step_index])
        prior_success.setdefault(fp, step.step_index)

    evidence = sorted(set(evidence))
    present = bool(evidence)
    # When eviction ids were recorded but none resolve after reload, the
    # structural tool-fingerprint signal may still fire; surface that gap in
    # the explanation so operators know Message.id injection is missing.
    unresolved = []
    for step in run.steps:
        if step.kind != 'context_op':
            continue
        for mid in step.payload.get('evicted_message_ids') or []:
            if mid and str(mid) not in resolved:
                unresolved.append(str(mid))
    if present:
        explanation = (
            f'Re-request of previously available information after context_op '
            f'(steps {evidence})'
        )
        if unresolved:
            explanation += (
                f'; {len(unresolved)} evicted id(s) unresolved against '
                f'persisted messages'
            )
    else:
        explanation = 'No structural loss-of-history signal'

    return Finding(
        mode_id=mode.id,
        present=present,
        confidence=0.9 if present else 0.0,
        evidence_steps=evidence,
        explanation=explanation,
        deterministic=True,
    )


def run_deterministic_detectors(
    run: Run,
    cfg: Optional[DetectorConfig] = None,
) -> List[Finding]:
    return [
        detect_step_repetition(run, cfg),
        detect_unaware_termination(run),
        detect_loss_of_history(run),
    ]
