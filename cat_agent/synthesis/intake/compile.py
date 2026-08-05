"""Compile a confirmed draft + interview into a :class:`ToolSpec`."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from cat_agent.llm import get_chat_model
from cat_agent.llm.base import BaseChatModel
from cat_agent.llm.schema import SYSTEM, USER, Message
from cat_agent.log import logger
from cat_agent.synthesis.intake.draft import Draft
from cat_agent.synthesis.intake.interview import (
    InterviewState,
    ResolvedDecision,
    sanitize_messages_for_llm,
)
from cat_agent.synthesis.artifacts import coerce_spec_type
from cat_agent.synthesis.llm_text import collect_chat_text
from cat_agent.synthesis.spec import Example, tool_spec_from_dict
from cat_agent.utils.utils import extract_code

_CHAR_MAP = str.maketrans({
    'ı': 'i', 'İ': 'I', 'ş': 's', 'Ş': 'S', 'ğ': 'g', 'Ğ': 'G',
    'ü': 'u', 'Ü': 'U', 'ö': 'o', 'Ö': 'O', 'ç': 'c', 'Ç': 'C',
    'ä': 'a', 'Ä': 'A', 'ß': 'ss',
    'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a', 'å': 'a',
    'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
    'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
    'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o',
    'ú': 'u', 'ù': 'u', 'û': 'u',
    'ñ': 'n', 'Ñ': 'N',
    'ý': 'y', 'ÿ': 'y',
})

# Forbidden in compiled descriptions / rules.
_BAD_AMBIGUITY = re.compile(
    r'\b(unspecified|not decided|not specified|left unspecified|no specific rule)\b',
    re.I,
)


@dataclass
class CompileResult:
    spec: Optional[Any]
    ok: bool
    name_changed: bool = False
    original_name: str = ''
    sanitised_name: str = ''
    error: Optional[str] = None
    needs_reinterview: bool = False
    reinterview_question: Optional[str] = None
    failed_field: Optional[str] = None
    model_added_examples: List[Example] = field(default_factory=list)
    used_draft_fallback: bool = False


def sanitise_name(raw: str) -> Tuple[str, bool]:
    original = (raw or '').strip()
    text = original.translate(_CHAR_MAP)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    if not text:
        text = 'generated_tool'
    if text[0].isdigit():
        text = 't_' + text
    changed = text != re.sub(r'[^a-z0-9_]+', '_', original.lower()).strip('_')
    if any(ord(c) > 127 for c in original):
        changed = True
    return text, changed


def compile_to_spec(
    draft: Draft,
    history: Sequence[Message],
    confirmation: str,
    *,
    llm: Union[dict, BaseChatModel, None] = None,
    state: Optional[InterviewState] = None,
    extra_examples: Optional[Sequence[Example]] = None,
    decisions: Optional[Sequence[ResolvedDecision]] = None,
    draft_only: bool = False,
) -> CompileResult:
    """Build a ToolSpec. Examples come from the draft (never model-rewritten).

    When *draft_only* is True, skip the model and build fields from the draft
    plus resolved decisions — used as the recovery path when history is corrupt.
    """
    state = state or InterviewState()
    decisions = list(decisions or state.decisions or [])

    if draft_only:
        payload = _fallback_payload_from_draft(draft, confirmation, decisions)
        used_fallback = True
        logger.info(
            'compile_to_spec draft_only=True; fields from draft columns={}',
            list(payload.get('parameters') or {}),
        )
    else:
        if isinstance(llm, dict) or llm is None:
            model = get_chat_model(llm or {})
        else:
            model = llm
        payload = _ask_model_for_fields(
            model, draft, history, confirmation, decisions)
        used_fallback = False
        if payload is None:
            logger.warning(
                'compile_to_spec: model returned no JSON '
                '(likely non-JSON prose or empty response); '
                'falling back to draft-derived fields. '
                'history_turns={} confirmation_len={}',
                len(history),
                len(confirmation or ''),
            )
            payload = _fallback_payload_from_draft(draft, confirmation, decisions)
            used_fallback = True

    # Scrub ambiguity language from model output.
    if isinstance(payload.get('description'), str):
        payload['description'] = _BAD_AMBIGUITY.sub(
            'settled by default', payload['description'])

    # Fill gaps from the draft when the model omits required fields.
    # (Root cause of live-run compile failure: model returned incomplete /
    # non-JSON payload while the draft already had inputs, output, and examples.)
    draft_fallback = _fallback_payload_from_draft(draft, confirmation, decisions)
    if not (payload.get('parameters') or {}):
        logger.warning(
            'compile_to_spec: model omitted parameters; '
            'filling from draft columns={}',
            list(draft_fallback.get('parameters') or {}),
        )
        payload['parameters'] = draft_fallback['parameters']
    if not (payload.get('description') or '').strip():
        payload['description'] = draft_fallback['description']
    if not payload.get('returns'):
        payload['returns'] = draft_fallback['returns']
    if not payload.get('name'):
        payload['name'] = draft_fallback['name']

    # Prefer the draft heading when present — models invent names like
    # "VATSplitTool" → vatsplittool; the H1 "VAT split" → vat_split.
    draft_name_raw = _guess_name(draft)
    draft_name, draft_name_changed = sanitise_name(draft_name_raw)
    model_name_raw = str(payload.get('name') or draft_name_raw)
    model_name, model_changed = sanitise_name(model_name_raw)
    if draft_name and draft_name != 'generated_tool':
        name_raw = draft_name_raw
        name = draft_name
        name_changed = draft_name_changed
        if model_name != draft_name:
            logger.info(
                'compile_to_spec: preferring draft name {!r} over model name {!r}',
                draft_name,
                model_name,
            )
    else:
        name_raw = model_name_raw
        name, name_changed = model_name, model_changed

    examples = list(draft.examples)
    model_added: List[Example] = []
    for added in list(state.added_examples) + list(extra_examples or []):
        examples.append(added)
        model_added.append(added)

    for item in payload.get('new_examples') or []:
        traced = _traceable_example(item, history, state)
        if traced is None:
            logger.warning('Dropping untraced model example: {}', item)
            continue
        examples.append(traced)
        model_added.append(traced)

    parameters = {}
    for key, value in (payload.get('parameters') or {}).items():
        if isinstance(value, dict):
            parameters[str(key)] = {
                'type': coerce_spec_type(value.get('type') or 'Any'),
                'description': value.get('description') or '',
            }
        else:
            parameters[str(key)] = {
                'type': coerce_spec_type(str(value)),
                'description': '',
            }

    # Merge decisions into description so synthesis sees concrete rules.
    description = str(payload.get('description') or '').strip()
    if decisions:
        rules = '\n'.join(f'- {d.rule}' for d in decisions)
        if 'Resolved decisions:' not in description:
            description = f'{description}\n\nResolved decisions:\n{rules}'.strip()

    data = {
        'name': name,
        'description': description,
        'parameters': parameters,
        'returns': coerce_spec_type(str(payload.get('returns') or 'Any')),
        'examples': [
            {'inputs': ex.inputs, 'expected': ex.expected, 'note': ex.note}
            for ex in examples
        ],
        'holdout_ratio': float(payload.get('holdout_ratio') or 0.3),
        'requires_network': bool(payload.get('requires_network') or False),
        'deps': list(payload.get('deps') or []),
    }

    failed_field = _diagnose_payload(data, draft)
    if failed_field and used_fallback:
        # Draft itself is incomplete — targeted question naming the gap.
        return CompileResult(
            spec=None,
            ok=False,
            error=f'Draft is missing {failed_field}',
            name_changed=name_changed,
            original_name=name_raw,
            sanitised_name=name,
            needs_reinterview=True,
            reinterview_question=_targeted_question(failed_field, draft),
            failed_field=failed_field,
            model_added_examples=model_added,
            used_draft_fallback=True,
        )

    try:
        spec = tool_spec_from_dict(data)
    except ValueError as exc:
        failed_field = failed_field or _diagnose_validation_error(str(exc))
        logger.error(
            'compile_to_spec validation failed field={!r}: {} | '
            'payload_keys={} param_keys={} n_examples={}',
            failed_field,
            exc,
            list(payload.keys()),
            list(parameters.keys()),
            len(examples),
        )
        # Prefer draft-only recovery if we haven't tried it yet.
        if not draft_only and not used_fallback:
            return CompileResult(
                spec=None,
                ok=False,
                error=str(exc),
                name_changed=name_changed,
                original_name=name_raw,
                sanitised_name=name,
                needs_reinterview=False,  # pipeline will retry draft_only
                failed_field=failed_field,
                model_added_examples=model_added,
            )
        return CompileResult(
            spec=None,
            ok=False,
            error=str(exc),
            name_changed=name_changed,
            original_name=name_raw,
            sanitised_name=name,
            needs_reinterview=True,
            reinterview_question=_targeted_question(failed_field or 'details', draft),
            failed_field=failed_field,
            model_added_examples=model_added,
            used_draft_fallback=used_fallback,
        )

    return CompileResult(
        spec=spec,
        ok=True,
        name_changed=name_changed,
        original_name=name_raw,
        sanitised_name=name,
        model_added_examples=model_added,
        used_draft_fallback=used_fallback,
    )


def _diagnose_payload(data: Dict[str, Any], draft: Draft) -> Optional[str]:
    if not data.get('name'):
        return 'name'
    if not (data.get('description') or '').strip():
        return 'description'
    params = data.get('parameters') or {}
    if not params:
        return 'parameters'
    for pname, pspec in params.items():
        if isinstance(pspec, dict) and not (pspec.get('type') or '').strip():
            return f'parameters.{pname}.type'
    if not data.get('returns'):
        return 'returns'
    if len(data.get('examples') or []) < 3:
        return 'examples'
    return None


def _diagnose_validation_error(error: str) -> str:
    low = error.lower()
    if 'example' in low:
        return 'examples'
    if 'parameter' in low:
        return 'parameters'
    if 'description' in low:
        return 'description'
    if 'name' in low or 'identifier' in low:
        return 'name'
    if 'return' in low:
        return 'returns'
    return 'unknown'


def _targeted_question(failed_field: str, draft: Draft) -> str:
    """One specific question — never a generic restate-everything prompt."""
    if failed_field == 'parameters' or failed_field.startswith('parameters.'):
        cols = ', '.join(draft.example_columns[:-1]) if draft.example_columns else 'the inputs'
        return (
            f'What should each of these inputs mean in one short phrase: {cols}?'
        )
    if failed_field == 'returns':
        return 'What should the tool return — a single number, or several named values?'
    if failed_field == 'examples':
        return (
            'Please add at least one more concrete example row '
            '(inputs and the expected result).'
        )
    if failed_field == 'name':
        return 'What short name should we give this tool (letters and numbers only)?'
    if failed_field == 'description':
        return 'In one sentence, what should this tool do?'
    return (
        f'One detail is still unclear ({failed_field}). '
        f'Could you clarify just that piece?'
    )


def _ask_model_for_fields(
    llm: BaseChatModel,
    draft: Draft,
    history: Sequence[Message],
    confirmation: str,
    decisions: Sequence[ResolvedDecision],
) -> Optional[Dict[str, Any]]:
    system = (
        'You compile a ToolSpec for an internal code synthesizer.\n'
        'Return ONE JSON object only — raw JSON, no markdown fences, no commentary.\n'
        'Required keys: name, description, parameters, returns.\n'
        'parameters: each value is '
        '{"type": "<json-schema-ish type>", "description": "<English>"}.\n'
        'returns must be ONE short type token only: '
        'string, integer, number, boolean, object, array '
        '(never a sentence like "object with fields...").\n'
        'Write name/description/parameter descriptions in English.\n'
        'Do NOT include examples — they come from the draft table.\n'
        'Never write "unspecified" or "not decided" — use concrete rules.\n'
        'Include resolved decisions in the description as settled facts.'
    )
    decisions_txt = '\n'.join(
        f'- [{d.topic}/{d.source}] {d.rule}' for d in decisions
    ) or '(none)'
    user = (
        f'Confirmation text:\n{confirmation}\n\n'
        f'Resolved decisions:\n{decisions_txt}\n\n'
        f'Draft:\n{draft.raw_markdown}\n\n'
        f'Interview transcript:\n{_history_text(history)}\n'
    )
    messages = sanitize_messages_for_llm([
        Message(role=SYSTEM, content=system),
        Message(role=USER, content=user),
    ])
    raw = _call_llm(llm, messages)
    return _parse_json_object(raw)


def _traceable_example(
    item: Any,
    history: Sequence[Message],
    state: InterviewState,
) -> Optional[Example]:
    if not isinstance(item, dict):
        return None
    excerpt = str(item.get('source_answer_excerpt') or '').strip()
    if len(excerpt) < 3:
        return None
    blob = _history_text(history) + '\n' + '\n'.join(state.corrections)
    if excerpt not in blob:
        tokens = [t for t in re.split(r'\s+', excerpt.lower()) if len(t) > 2]
        user_blob = ' '.join(
            str(m.content).lower()
            for m in history
            if m.role == USER and m.content
        )
        if not tokens or not all(t in user_blob for t in tokens[:4]):
            return None
    inputs = item.get('inputs')
    if not isinstance(inputs, dict):
        return None
    return Example(
        inputs=dict(inputs),
        expected=item.get('expected'),
        note=f'from interview: {excerpt[:80]}',
    )


def _guess_name(draft: Draft) -> str:
    for line in draft.raw_markdown.splitlines():
        if line.startswith('#') and not line.startswith('##'):
            return line.lstrip('#').strip() or 'generated_tool'
    return 'generated_tool'


def _infer_param_type(draft: Draft, column: str) -> str:
    for example in draft.examples:
        value = example.inputs.get(column)
        if value is None:
            continue
        if isinstance(value, bool):
            return 'boolean'
        if isinstance(value, int) and not isinstance(value, bool):
            return 'integer'
        if isinstance(value, float):
            return 'number'
        if isinstance(value, dict):
            return 'object'
        if isinstance(value, list):
            return 'array'
        return 'string'
    return 'number'


def _infer_returns(draft: Draft) -> str:
    if not draft.examples:
        return 'Any'
    sample = draft.examples[0].expected
    if isinstance(sample, dict):
        return 'object'
    if isinstance(sample, list):
        return 'array'
    if isinstance(sample, bool):
        return 'boolean'
    if isinstance(sample, int) and not isinstance(sample, bool):
        return 'integer'
    if isinstance(sample, float):
        return 'number'
    if isinstance(sample, str):
        return 'string'
    return 'Any'


def _fallback_payload_from_draft(
    draft: Draft,
    confirmation: str,
    decisions: Sequence[ResolvedDecision] = (),
) -> Dict[str, Any]:
    columns = list(draft.example_columns or [])
    input_cols = columns[:-1] if len(columns) >= 2 else columns
    if not input_cols and draft.examples:
        input_cols = list(draft.examples[0].inputs.keys())
    parameters = {
        col: {
            'type': _infer_param_type(draft, col),
            'description': f'{col} (from draft examples)',
        }
        for col in input_cols
    }
    description = (confirmation or '').strip()
    description = _BAD_AMBIGUITY.sub('settled by default', description)
    if len(description) < 20:
        for line in draft.raw_markdown.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('|'):
                description = stripped
                break
    if not description:
        description = f'Tool generated from draft {_guess_name(draft)!r}.'
    if decisions:
        rules = '; '.join(d.rule for d in decisions)
        description = f'{description} Decisions: {rules}'
    return {
        'name': _guess_name(draft),
        'description': description[:800],
        'parameters': parameters,
        'returns': _infer_returns(draft),
    }


def _history_text(history: Sequence[Message]) -> str:
    parts = []
    for msg in history:
        if msg.content is None or (
            isinstance(msg.content, str) and not msg.content.strip()
        ):
            continue
        parts.append(f'{msg.role or "?"}: {msg.content}')
    return '\n'.join(parts)


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or '').strip()
    if not text:
        return None
    fenced = extract_code(text)
    for candidate in (fenced, text):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        start = candidate.find('{')
        end = candidate.rfind('}')
        if start >= 0 and end > start:
            try:
                data = json.loads(candidate[start:end + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
    return None


def _call_llm(llm: BaseChatModel, messages: List[Message]) -> str:
    messages = sanitize_messages_for_llm(messages)
    try:
        output = llm.chat(messages=messages, stream=True)
    except TypeError:
        output = llm.chat(messages=messages)
    return collect_chat_text(output)
