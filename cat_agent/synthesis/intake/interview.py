"""Interview the business user to resolve draft ambiguities (D2–D5).

Flow is driven by an explicit state machine in
:mod:`cat_agent.synthesis.intake.pipeline`::

    INTERVIEW → CONFIRM → COMPILE → DONE
                  ↑          │
                  └──────────┘  (correction only)
    HOLDOUT_REOPEN is the only other backward edge.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from cat_agent.agents.user_agent import PENDING_USER_INPUT
from cat_agent.llm import get_chat_model
from cat_agent.llm.base import BaseChatModel
from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.log import logger
from cat_agent.synthesis.intake.draft import Draft, OpenQuestion
from cat_agent.synthesis.intake.lang import detect_lang
from cat_agent.synthesis.llm_text import collect_chat_text
from cat_agent.synthesis.spec import Example

if TYPE_CHECKING:
    from cat_agent.observability.handlers.base import BaseHandler

JARGON_BLOCKLIST = (
    'json', 'schema', 'parameter', 'parameters', 'dtype', 'toolspec',
    'typescript', 'pythonic', 'kwargs', 'api endpoint',
)

# Soft default when the user defers a decision.
DEFAULT_ROUNDING_RULE = (
    'When a value lands exactly on a half-cent, round half up '
    '(away from zero for positive amounts).'
)

_AFFIRMATIVE = frozenset({
    'y', 'yes', 'yep', 'yeah', 'ok', 'okay', 'correct', 'right', 'sure',
    'ja', 'oui', 'si', 'sí', 'sim', 'evet', 'tak', 'ano', 'da',
    'richtig', 'genau', 'exact', 'doğru', 'dogru', 'klopt', 'juist',
})
_DEFER = frozenset({
    'skip', 'whatever', 'either', 'either way', 'you decide', 'your call',
    'doesnt matter', "doesn't matter", 'does not matter', 'no preference',
    'egal', 'wie du willst', 'peu importe', 'me da igual', 'fark etmez',
    'bana göre fark etmez', 'maakt niet uit',
})
_UNSPECIFIED_MARKERS = (
    'unspecified', 'not decided', 'not specified', 'left open',
    'left unspecified', 'no specific rule', 'still left',
)

QUESTION_HINT = '(answer, or type "skip" if you don\'t mind either way)'
CLOSED_YES_NO = 'Please answer with yes or no only.'


class Phase(str, Enum):
    INTERVIEW = 'interview'
    CONFIRM = 'confirm'
    COMPILE = 'compile'
    DONE = 'done'
    HOLDOUT_REOPEN = 'holdout_reopen'


@dataclass
class ResolvedDecision:
    topic: str
    rule: str
    source: str = 'assistant_default'  # or 'user'
    raw_answer: str = ''


@dataclass
class Question:
    text: str
    kind: str = 'general'
    priority: int = 50
    meta: Dict[str, Any] = field(default_factory=dict)
    pending_sentinel: str = PENDING_USER_INPUT

    def with_hint(self) -> 'Question':
        if self.kind == 'confirm':
            return self
        if QUESTION_HINT in self.text:
            return self
        return Question(
            text=f'{self.text.rstrip()} {QUESTION_HINT}',
            kind=self.kind,
            priority=self.priority,
            meta=dict(self.meta),
            pending_sentinel=self.pending_sentinel,
        )


@dataclass
class InterviewState:
    history: List[Message] = field(default_factory=list)
    questions_asked: int = 0
    confirmation: Optional[str] = None
    confirmation_generated: bool = False
    confirmed: bool = False
    phase: Phase = Phase.INTERVIEW
    added_examples: List[Example] = field(default_factory=list)
    example_traces: List[Dict[str, Any]] = field(default_factory=list)
    corrections: List[str] = field(default_factory=list)
    decisions: List[ResolvedDecision] = field(default_factory=list)
    asked_question_keys: List[str] = field(default_factory=list)
    consecutive_empty: int = 0
    user_turns: int = 0
    llm_calls: int = 0


def is_blank(answer: Optional[str]) -> bool:
    return not (answer or '').strip()


def is_affirmative(answer: Optional[str]) -> bool:
    text = (answer or '').strip().lower().rstrip('.!')
    if not text:
        return False
    if text in _AFFIRMATIVE:
        return True
    # "yes please", "ja gerne"
    first = text.split()[0]
    return first in _AFFIRMATIVE


def is_deferral(answer: Optional[str]) -> bool:
    text = (answer or '').strip().lower().rstrip('.!')
    if not text:
        return False
    if text in _DEFER:
        return True
    # Fuzzy: "doesn't matter to me", "you can decide"
    compact = re.sub(r'\s+', ' ', text)
    for phrase in (
        'doesnt matter', "doesn't matter", 'does not matter',
        'you decide', 'your call', 'no preference', 'peu importe',
        'me da igual', 'fark etmez', 'maakt niet uit', 'egal',
    ):
        if phrase in compact:
            return True
    return False


def question_key(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '').strip().lower())[:160]


class SpecInterviewer:
    """Derive clarifying questions from a draft; hard-capped (D2)."""

    def __init__(
        self,
        llm: Union[dict, BaseChatModel, None],
        max_questions: int = 3,
        lang: Optional[str] = None,
        handlers: Optional[List['BaseHandler']] = None,
    ):
        if isinstance(llm, dict) or llm is None:
            self.llm = get_chat_model(llm or {})
        else:
            self.llm = llm
        self.max_questions = max(1, int(max_questions))
        self.lang_override = lang
        self._handlers = handlers or []

    def working_lang(self, draft: Draft) -> str:
        return detect_lang(
            draft.raw_markdown, override=self.lang_override or None
        ) or (draft.detected_lang or 'en')

    def next_interview_question(
        self,
        draft: Draft,
        state: InterviewState,
    ) -> Optional[Question]:
        """Return the next INTERVIEW question, or ``None`` when interview is done.

        Never emits a confirmation — that is the pipeline's CONFIRM phase.
        """
        if state.phase != Phase.INTERVIEW:
            return None

        deterministic = self._deterministic_questions(draft, state)
        if deterministic and state.questions_asked < self.max_questions:
            q = deterministic[0].with_hint()
            return self._dedupe_or_none(state, q)

        if state.questions_asked >= self.max_questions:
            return None

        if self._interview_gaps_resolved(draft, state):
            return None

        q = self._ask_model(draft, state)
        if q is None:
            return None
        return self._dedupe_or_none(state, q.with_hint())

    def enter_confirm(self, draft: Draft, state: InterviewState) -> Question:
        """Generate the confirmation **once** on entry to CONFIRM."""
        if state.confirmation_generated and state.confirmation:
            return Question(text=state.confirmation, kind='confirm', priority=0)
        confirm = self._build_confirmation(draft, state)
        state.confirmation = confirm.text
        state.confirmation_generated = True
        state.phase = Phase.CONFIRM
        return confirm

    def closed_yes_no_prompt(self) -> Question:
        return Question(text=CLOSED_YES_NO, kind='confirm_closed', priority=0)

    def is_interview_complete(self, draft: Draft, state: InterviewState) -> bool:
        if state.questions_asked >= self.max_questions:
            return True
        if self._deterministic_questions(draft, state):
            return False
        return self._interview_gaps_resolved(draft, state)

    def record_interview_answer(
        self,
        state: InterviewState,
        question: Question,
        answer: str,
    ) -> InterviewState:
        """Record a non-blank interview answer. Handles deferrals as decisions."""
        assert not is_blank(answer)
        text = answer.strip()
        self._append_turn(state, question.text, text)
        state.questions_asked += 1
        state.asked_question_keys.append(question_key(question.text))
        state.consecutive_empty = 0
        state.user_turns += 1

        if is_deferral(text):
            decision = self._resolve_deferral(question, text)
            state.decisions.append(decision)
            # Mark open parse questions of matching kind as handled.
            logger.info(
                'Deferred answer on {!r} → assistant_default: {}',
                question.kind,
                decision.rule,
            )
        return state

    def record_confirm_answer(
        self,
        state: InterviewState,
        answer: str,
    ) -> str:
        """Return ``'yes'``, ``'no'``, or ``'ambiguous'``."""
        assert not is_blank(answer)
        text = answer.strip()
        self._append_turn(state, state.confirmation or '(confirm)', text)
        state.user_turns += 1
        state.consecutive_empty = 0

        if is_affirmative(text):
            state.confirmed = True
            state.phase = Phase.COMPILE
            return 'yes'
        if is_deferral(text):
            # Deferral on confirm = accept with defaults already recorded.
            state.confirmed = True
            state.phase = Phase.COMPILE
            return 'yes'
        # Short negatives
        low = text.lower().rstrip('.!')
        if low in {'n', 'no', 'nein', 'non', 'hayır', 'hayir', 'nee', 'não', 'nao'}:
            state.confirmed = False
            state.confirmation = None
            state.confirmation_generated = False
            state.corrections.append(text)
            state.phase = Phase.INTERVIEW
            return 'no'
        # Longer text = correction
        if len(text.split()) >= 3 or any(
            w in low for w in ('not', 'wrong', 'should', 'instead', 'actually')
        ):
            state.confirmed = False
            state.confirmation = None
            state.confirmation_generated = False
            state.corrections.append(text)
            state.phase = Phase.INTERVIEW
            return 'no'
        return 'ambiguous'

    def note_empty_input(self, state: InterviewState) -> str:
        """Return a local re-prompt string; no LLM call."""
        state.consecutive_empty += 1
        if state.consecutive_empty >= 2:
            return (
                'Please type an answer, or "skip" / "you decide" if either way is fine, '
                'or "yes" to confirm.'
            )
        return 'Please type an answer (empty input is ignored).'

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dedupe_or_none(
        self, state: InterviewState, question: Question
    ) -> Optional[Question]:
        key = question_key(question.text)
        if key in state.asked_question_keys:
            logger.warning('Skipping repeat question: {}', key[:80])
            return None
        return question

    def _append_turn(self, state: InterviewState, question: str, answer: str) -> None:
        q = (question or '').strip()
        a = (answer or '').strip()
        if not q or not a:
            raise ValueError('Refusing to append empty-content message to history')
        state.history.append(Message(role=ASSISTANT, content=q))
        state.history.append(Message(role=USER, content=a))

    def _interview_gaps_resolved(self, draft: Draft, state: InterviewState) -> bool:
        # Unresolved parse ambiguities?
        asked = set(state.asked_question_keys)
        for oq in draft.open_questions:
            if oq.kind in {'number', 'date'} and question_key(oq.message) not in asked:
                # Unless deferred via a decision covering numbers
                if not any(d.topic == 'locale_number' for d in state.decisions):
                    return False
        return self._model_complete_check(draft, state)

    def _deterministic_questions(
        self,
        draft: Draft,
        state: InterviewState,
    ) -> List[Question]:
        asked = set(state.asked_question_keys)
        out: List[Question] = []
        for oq in draft.open_questions:
            if oq.kind in {'number', 'date'} and question_key(oq.message) not in asked:
                out.append(
                    Question(
                        text=oq.message,
                        kind=oq.kind,
                        priority=0,
                        meta={'raw': oq.raw, 'column': oq.column, 'row': oq.row},
                    )
                )
        contradiction = _find_contradiction(draft.examples)
        if contradiction and question_key(contradiction) not in asked:
            out.append(Question(text=contradiction, kind='contradiction', priority=1))
        # Draft explicitly flags rounding as open → ask before model freestyle.
        if (
            _draft_flags_open_rounding(draft.raw_markdown)
            and not any(d.topic == 'rounding' for d in state.decisions)
            and not any('round' in k for k in asked)
        ):
            out.append(
                Question(
                    text=(
                        'When a calculated amount lands exactly on a half-cent, '
                        'how should we round?'
                    ),
                    kind='rounding',
                    priority=2,
                )
            )
        out.sort(key=lambda q: q.priority)
        return out

    def _resolve_deferral(self, question: Question, answer: str) -> ResolvedDecision:
        kind = question.kind
        if kind == 'rounding' or any(
            w in question.text.lower() for w in ('round', 'half-cent', 'half cent', 'rund')
        ):
            return ResolvedDecision(
                topic='rounding',
                rule=DEFAULT_ROUNDING_RULE,
                source='assistant_default',
                raw_answer=answer,
            )
        if kind in {'number', 'date'}:
            return ResolvedDecision(
                topic='locale_number',
                rule='Interpret example numbers using the draft Locale when set; '
                     'otherwise prefer the unambiguous reading.',
                source='assistant_default',
                raw_answer=answer,
            )
        if kind == 'error' or any(
            w in question.text.lower() for w in ('error', 'invalid', 'negative', 'refuse')
        ):
            return ResolvedDecision(
                topic='error_behaviour',
                rule='On invalid input (negative amounts or rates outside a sensible range), '
                     'refuse and report an error.',
                source='assistant_default',
                raw_answer=answer,
            )
        return ResolvedDecision(
            topic=kind or 'general',
            rule='Use the most common business default for this case.',
            source='assistant_default',
            raw_answer=answer,
        )

    def _ask_model(self, draft: Draft, state: InterviewState) -> Optional[Question]:
        lang = self.working_lang(draft)
        remaining = self.max_questions - state.questions_asked
        decisions = '\n'.join(
            f'- [{d.topic}] {d.rule} (source={d.source})' for d in state.decisions
        ) or '(none)'
        user = (
            f'Working language: {lang}\n'
            f'Questions remaining (budget): {remaining}\n'
            f'Open issues from parsing:\n{_fmt_open(draft.open_questions)}\n'
            f'Already resolved decisions (do NOT re-ask):\n{decisions}\n'
            f'Corrections so far: {state.corrections!r}\n\n'
            f'DRAFT:\n{draft.raw_markdown}\n\n'
            'Ask ONLY about gaps not already resolved. Prefer rounding, then error '
            'behaviour, then units — but skip any the draft already settles.\n'
            'If nothing consequential remains, reply with exactly: NO_QUESTION\n'
            'Otherwise one plain-language question, no preamble.'
        )
        text = self._call(lang, user, state, confirm=False).strip()
        if not text or text.upper().startswith('NO_QUESTION'):
            return None
        text = _strip_jargon(text)
        kind = 'general'
        low = text.lower()
        if any(w in low for w in ('round', 'rund', 'arrond', 'yuvarla', 'half-cent')):
            kind = 'rounding'
        elif any(w in low for w in ('error', 'invalid', 'fehler', 'erreur', 'negative')):
            kind = 'error'
        return Question(text=text, kind=kind, priority=40)

    def _model_complete_check(self, draft: Draft, state: InterviewState) -> bool:
        asked = set(state.asked_question_keys)
        for oq in draft.open_questions:
            if oq.kind in {'number', 'date'} and question_key(oq.message) not in asked:
                if not any(d.topic == 'locale_number' for d in state.decisions):
                    return False
        # Open rounding flag (or no rounding rule at all) still unresolved.
        if (
            not any(d.topic == 'rounding' for d in state.decisions)
            and state.questions_asked < self.max_questions
            and not any('round' in k for k in asked)
            and (
                _draft_flags_open_rounding(draft.raw_markdown)
                or not _draft_settles_rounding(draft.raw_markdown)
            )
        ):
            low = draft.raw_markdown.lower()
            if _draft_flags_open_rounding(draft.raw_markdown) or any(
                w in low for w in ('vat', 'tax', 'rate', 'euro', 'currency', 'round')
            ):
                return False
        lang = self.working_lang(draft)
        decisions = '\n'.join(
            f'- [{d.topic}] {d.rule}' for d in state.decisions
        ) or '(none)'
        user = (
            f'Draft:\n{draft.raw_markdown}\n\n'
            f'Resolved decisions:\n{decisions}\n\n'
            'Have all consequential ambiguities been resolved? '
            'Reply YES or NO only.'
        )
        text = self._call(lang, user, state, confirm=False).strip().upper()
        return text.startswith('YES')

    def _build_confirmation(self, draft: Draft, state: InterviewState) -> Question:
        lang = self.working_lang(draft)
        decisions = '\n'.join(
            f'- {d.rule}' for d in state.decisions
        ) or '(none beyond the draft)'
        user = (
            f'Draft:\n{draft.raw_markdown}\n\n'
            f'Resolved decisions (include these as settled facts, never as open):\n'
            f'{decisions}\n\n'
            f'Corrections: {state.corrections!r}\n\n'
            'Write a confirmation of AT MOST 3 sentences. Cover: inputs and output, '
            'the core rule, and error behaviour if any. End by asking if that is right. '
            'No JSON, no field names, no type names. Never say "unspecified" or '
            '"not decided".'
        )
        text = _strip_jargon(self._call(lang, user, state, confirm=True).strip())
        text = _enforce_max_sentences(text, 3)
        # Strip forbidden ambiguity language if the model slips.
        for marker in _UNSPECIFIED_MARKERS:
            if marker in text.lower():
                text = re.sub(re.escape(marker), 'settled by default', text, flags=re.I)
        return Question(text=text, kind='confirm', priority=0)

    def _call(
        self,
        lang: str,
        user: str,
        state: InterviewState,
        *,
        confirm: bool,
    ) -> str:
        system = (
            _confirm_system_prompt(lang) if confirm else _interview_system_prompt(lang)
        )
        messages = sanitize_messages_for_llm(
            _llm_messages(system, user, state.history)
        )
        state.llm_calls += 1
        return _call_llm(self.llm, messages)


def sanitize_messages_for_llm(messages: Sequence[Message]) -> List[Message]:
    """Drop empty/None content messages before dispatch (defensive host guard)."""
    cleaned: List[Message] = []
    for msg in messages:
        content = msg.content
        if content is None:
            logger.warning('Dropping message with None content (role={})', msg.role)
            continue
        if isinstance(content, str) and not content.strip():
            logger.warning('Dropping message with blank content (role={})', msg.role)
            continue
        if isinstance(content, list) and not content:
            logger.warning('Dropping message with empty content list (role={})', msg.role)
            continue
        cleaned.append(msg)
    if not cleaned:
        raise ValueError('No messages left after sanitising empty content')
    return cleaned


def holdout_question(failures: Sequence[Dict[str, Any]], lang: str = 'en') -> Question:
    case = failures[0] if failures else {}
    inputs = case.get('inputs')
    expected = case.get('expected')
    actual = case.get('returned')
    templates = {
        'en': (
            f'One situation was not covered by your examples. '
            f'For inputs {inputs!r}, the tool produced {actual!r}, '
            f'but we expected {expected!r}. What should happen in this case? '
            f'{QUESTION_HINT}'
        ),
        'de': (
            f'Ein Fall war in Ihren Beispielen nicht abgedeckt. '
            f'Bei Eingaben {inputs!r} ergab sich {actual!r}, '
            f'erwartet war {expected!r}. Was soll in diesem Fall passieren? '
            f'{QUESTION_HINT}'
        ),
        'fr': (
            f'Un cas n\'était pas couvert par vos exemples. '
            f'Pour les entrées {inputs!r}, le résultat a été {actual!r}, '
            f'alors que {expected!r} était attendu. Que doit-il se passer ? '
            f'{QUESTION_HINT}'
        ),
        'tr': (
            f'Örneklerinizde kapsanmayan bir durum var. '
            f'Girdiler {inputs!r} için sonuç {actual!r} oldu, '
            f'beklenen {expected!r}. Bu durumda ne olmalı? {QUESTION_HINT}'
        ),
    }
    text = templates.get(lang, templates['en'])
    return Question(text=text, kind='holdout', priority=0, meta={'failures': list(failures)})


def insensitivity_question(
    finding: Any,
    lang: str = 'en',
) -> Question:
    """Ask whether input-space insensitivity is intentional (A4-class gaps)."""
    tried = getattr(finding, 'variants_tried', 0)
    samples = list(getattr(finding, 'sample_unchanged', []) or [])[:3]
    param = getattr(finding, 'param', '?')
    templates = {
        'en': (
            f'Changed one character of a valid example {tried} times; '
            f'the output never changed (parameter {param!r}). '
            f'Is that intended? If not, add one of these as a negative example: '
            f'{samples!r}. {QUESTION_HINT}'
        ),
        'de': (
            f'Ein Zeichen eines gültigen Beispiels wurde {tried}-mal geändert; '
            f'die Ausgabe blieb gleich (Parameter {param!r}). '
            f'Ist das beabsichtigt? Falls nicht, fügen Sie eines davon als '
            f'Negativbeispiel hinzu: {samples!r}. {QUESTION_HINT}'
        ),
        'fr': (
            f'Un caractère d\'un exemple valide a été modifié {tried} fois ; '
            f'la sortie n\'a jamais changé (paramètre {param!r}). '
            f'Est-ce voulu ? Sinon, ajoutez l\'un de ceux-ci en exemple négatif : '
            f'{samples!r}. {QUESTION_HINT}'
        ),
        'tr': (
            f'Geçerli bir örneğin bir karakteri {tried} kez değiştirildi; '
            f'çıktı hiç değişmedi (parametre {param!r}). '
            f'Bu istenen davranış mı? Değilse şunlardan birini negatif örnek '
            f'olarak ekleyin: {samples!r}. {QUESTION_HINT}'
        ),
    }
    text = templates.get(lang, templates['en'])
    return Question(
        text=text,
        kind='insensitivity',
        priority=0,
        meta={
            'param': param,
            'variants_tried': tried,
            'sample_unchanged': samples,
            'base_inputs': dict(getattr(finding, 'base_inputs', {}) or {}),
        },
    )


def _draft_settles_rounding(text: str) -> bool:
    """True when the draft already chooses a rounding rule (not merely mentions it)."""
    low = (text or '').lower()
    # Explicit open flags → not settled.
    if any(
        m in low for m in (
            'rounding', 'round', 'half-cent', 'half cent',
        )
    ) and any(
        m in low for m in (
            'not specified', 'unspecified', 'expect to be asked',
            'left open', 'to be decided', 'tbd',
        )
    ):
        return False
    return any(
        w in low for w in (
            'half-up', 'half up', 'half-down', 'half down',
            'banker', 'bankers', "banker's",
            'round half', 'round up', 'round down',
            'kaufmännisch', 'arrondi', 'yuvarla',
        )
    )


def _draft_flags_open_rounding(text: str) -> bool:
    low = (text or '').lower()
    return any(
        m in low for m in (
            'not specified', 'unspecified', 'expect to be asked',
            'left open', 'to be decided',
        )
    ) and any(m in low for m in ('round', 'half-cent', 'half cent'))


def _find_contradiction(examples: Sequence[Example]) -> Optional[str]:
    seen: Dict[str, Any] = {}
    for ex in examples:
        key = json.dumps(ex.inputs, sort_keys=True, default=str)
        if key in seen and seen[key] != ex.expected:
            return (
                f'Two of your examples use the same inputs {ex.inputs!r} but expect '
                f'different results ({seen[key]!r} vs {ex.expected!r}). '
                f'Which one is correct?'
            )
        seen[key] = ex.expected
    return None


def _fmt_open(questions: Sequence[OpenQuestion]) -> str:
    if not questions:
        return '(none)'
    return '\n'.join(f'- [{q.kind}] {q.message}' for q in questions)


def _strip_jargon(text: str) -> str:
    out = text
    for term in JARGON_BLOCKLIST:
        out = re.sub(re.escape(term), '…', out, flags=re.I)
    return out


def _enforce_max_sentences(text: str, max_sentences: int) -> str:
    # Split on sentence terminators; keep up to N.
    parts = re.split(r'(?<=[.!?])\s+', (text or '').strip())
    parts = [p for p in parts if p.strip()]
    if len(parts) <= max_sentences:
        return text.strip()
    # Prefer keeping a trailing yes/no ask within the budget.
    tail = parts[-1]
    if '?' in tail:
        head = parts[: max(0, max_sentences - 1)]
        return ' '.join(head + [tail]).strip()
    return ' '.join(parts[:max_sentences]).strip()


def _interview_system_prompt(lang: str) -> str:
    return (
        f'You help a non-technical business user clarify a tool draft.\n'
        f'Write every question in language code `{lang}`.\n'
        'Ask about the most consequential unresolved gap only.\n'
        'Never re-ask something already answered or deferred.\n'
        'No JSON, schema, parameters, or type names.'
    )


def _confirm_system_prompt(lang: str) -> str:
    return (
        f'You restate a business rule for confirmation in language `{lang}`.\n'
        'At most 3 sentences. Plain language. No JSON or type names.\n'
        'Treat deferred answers as settled defaults — never call them unspecified.\n'
        'End by asking if that understanding is correct.'
    )


def _history_transcript(history: Sequence[Message]) -> str:
    parts = []
    for msg in history:
        content = msg.content
        if content is None or (isinstance(content, str) and not content.strip()):
            continue
        role = (msg.role or '?').upper()
        parts.append(f'{role}: {content}')
    return '\n'.join(parts)


def _llm_messages(
    system: str,
    user: str,
    history: Sequence[Message] = (),
) -> List[Message]:
    transcript = _history_transcript(history)
    body = user
    if transcript:
        body = f'Interview so far:\n{transcript}\n\n{body}'
    return [
        Message(role=SYSTEM, content=system),
        Message(role=USER, content=body),
    ]


def _call_llm(llm: BaseChatModel, messages: List[Message]) -> str:
    messages = sanitize_messages_for_llm(messages)
    try:
        output = llm.chat(messages=messages, stream=True)
    except TypeError:
        output = llm.chat(messages=messages)
    return collect_chat_text(output)
