"""Tests for SpecInterviewer + intake state machine (live-run fixes 1.1–1.6)."""

from __future__ import annotations

import re
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.agents.user_agent import PENDING_USER_INPUT
from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.synthesis.intake.draft import Draft, OpenQuestion
from cat_agent.synthesis.intake.interview import (
    DEFAULT_ROUNDING_RULE,
    Phase,
    InterviewState,
    Question,
    SpecInterviewer,
    is_affirmative,
    is_blank,
    is_deferral,
    question_key,
    sanitize_messages_for_llm,
)
from cat_agent.synthesis.intake.pipeline import synthesize_from_draft
from cat_agent.synthesis.spec import Example


def _mock_llm(responses: List[str]) -> MagicMock:
    llm = MagicMock()
    llm.model = 'mock-intake'
    queue = list(responses)

    def chat(**kwargs):
        text = queue.pop(0) if queue else 'NO_QUESTION'
        return iter([[Message(role=ASSISTANT, content=text)]])

    llm.chat = MagicMock(side_effect=chat)
    return llm


def _draft_complete() -> Draft:
    md = """\
# add_one
Add one to x.
| x | result |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
"""
    return Draft.from_markdown(md, locale='en-IE')


def _draft_ambiguous() -> Draft:
    return Draft(
        raw_markdown='# t\n| a | r |\n|---|---|\n| 1,500 | 2 |\n| 2 | 3 |\n| 3 | 4 |\n',
        source_path=None,
        examples=[
            Example(inputs={'a': '1,500'}, expected=2),
            Example(inputs={'a': 2}, expected=3),
            Example(inputs={'a': 3}, expected=4),
        ],
        example_columns=['a', 'r'],
        open_questions=[
            OpenQuestion(
                kind='number',
                message='In your examples, does `1,500` mean one thousand five hundred, or one point five?',
                raw='1,500',
            )
        ],
        detected_lang='en',
    )


def _vat_like_draft(*, rounding_specified: bool) -> Draft:
    rounding_line = (
        'Round half-up at the half-cent boundary.'
        if rounding_specified
        else '(Rounding at the half-cent boundary is not specified — expect to be asked.)'
    )
    error_line = (
        ''
        if rounding_specified
        else ''  # leave error open when rounding is specified for prioritisation test
    )
    md = f"""\
# VAT split
Split a VAT-inclusive gross into net and tax.
## Rules
- Round both to two decimal places
{rounding_line}
{error_line}
## Examples
| gross | rate | result |
|---|---|---|
| 120 | 0.20 | {{"net": 100.0, "tax": 20.0}} |
| 100 | 0.0 | {{"net": 100.0, "tax": 0.0}} |
| 1.00 | 0.20 | {{"net": 0.83, "tax": 0.17}} |
"""
    return Draft.from_markdown(md, locale='en-IE')


class TestInterviewBasics:

    def test_question_cap_default_is_three(self):
        llm = _mock_llm([
            'How should we round?',
            'What about negatives?',
            'Empty string?',
            'Should never ask this',
        ])
        interviewer = SpecInterviewer(llm=llm, max_questions=3)
        interviewer._model_complete_check = lambda *a, **k: False  # type: ignore
        draft = _draft_complete()
        draft.open_questions = []
        state = InterviewState(phase=Phase.INTERVIEW)
        asked = 0
        for _ in range(10):
            if interviewer.is_interview_complete(draft, state):
                break
            q = interviewer.next_interview_question(draft, state)
            if q is None:
                break
            state = interviewer.record_interview_answer(state, q, f'answer-{asked}')
            asked += 1
        assert asked <= 3
        assert state.questions_asked <= 3

    def test_ambiguous_numbers_asked_first(self):
        llm = _mock_llm(['How do you round?'])
        interviewer = SpecInterviewer(llm=llm, max_questions=3)
        draft = _draft_ambiguous()
        state = InterviewState(phase=Phase.INTERVIEW)
        q = interviewer.next_interview_question(draft, state)
        assert q is not None
        assert q.kind == 'number'
        assert '1,500' in q.text
        assert 'skip' in q.text.lower()

    def test_contradictory_examples(self):
        draft = Draft(
            raw_markdown='# t\n| a | r |\n|---|---|\n| 1 | 2 |\n| 1 | 9 |\n| 2 | 3 |\n',
            source_path=None,
            examples=[
                Example(inputs={'a': 1}, expected=2),
                Example(inputs={'a': 1}, expected=9),
                Example(inputs={'a': 2}, expected=3),
            ],
            example_columns=['a', 'r'],
            detected_lang='en',
        )
        interviewer = SpecInterviewer(llm=_mock_llm([]), max_questions=3)
        state = InterviewState(phase=Phase.INTERVIEW)
        q = interviewer.next_interview_question(draft, state)
        assert q is not None
        assert q.kind == 'contradiction'

    def test_open_rounding_flagged_by_draft(self):
        interviewer = SpecInterviewer(llm=_mock_llm([]), max_questions=3)
        draft = _vat_like_draft(rounding_specified=False)
        state = InterviewState(phase=Phase.INTERVIEW)
        q = interviewer.next_interview_question(draft, state)
        assert q is not None
        assert q.kind == 'rounding'
        assert 'half-cent' in q.text.lower() or 'round' in q.text.lower()

    def test_rounding_specified_skips_rounding_question(self):
        """Prioritisation is gap-driven: settled rounding → ask error instead."""
        llm = _mock_llm([
            'What should happen for a negative gross or a rate above 1?',
        ])
        interviewer = SpecInterviewer(llm=llm, max_questions=3)
        interviewer._model_complete_check = lambda *a, **k: False  # type: ignore
        draft = _vat_like_draft(rounding_specified=True)
        state = InterviewState(phase=Phase.INTERVIEW)
        q = interviewer.next_interview_question(draft, state)
        assert q is not None
        assert q.kind != 'rounding'
        assert 'round' not in q.text.lower() or 'negative' in q.text.lower()

    def test_pending_user_input_sentinel(self):
        q = Question(text='hello?', kind='general')
        assert q.pending_sentinel == PENDING_USER_INPUT

    def test_questions_in_draft_language(self):
        cases = [
            ('de', 'Wie sollen wir runden?'),
            ('fr', 'Comment devons-nous arrondir ?'),
            ('tr', 'Nasıl yuvarlamalıyız?'),
        ]
        for lang, question_text in cases:
            llm = _mock_llm([question_text])
            interviewer = SpecInterviewer(llm=llm, max_questions=3, lang=lang)
            draft = _draft_complete()
            draft.detected_lang = lang
            draft.open_questions = []
            interviewer._model_complete_check = lambda *a, **k: False  # type: ignore
            state = InterviewState(phase=Phase.INTERVIEW)
            q = interviewer.next_interview_question(draft, state)
            assert q is not None
            assert question_text in q.text

    def test_no_jargon_in_questions(self):
        dirty = 'What should the JSON schema parameters look like for the ToolSpec?'
        llm = _mock_llm([dirty])
        interviewer = SpecInterviewer(llm=llm, max_questions=3)
        interviewer._model_complete_check = lambda *a, **k: False  # type: ignore
        draft = _draft_complete()
        draft.open_questions = []
        state = InterviewState(phase=Phase.INTERVIEW)
        q = interviewer.next_interview_question(draft, state)
        assert q is not None
        low = q.text.lower()
        for term in ('json', 'schema', 'parameter', 'toolspec'):
            assert term not in low, f'jargon leaked: {term} in {q.text!r}'

    def test_llm_messages_start_with_user_after_history(self):
        from cat_agent.llm.base.truncation import truncate_input_messages_roughly
        from cat_agent.synthesis.intake.interview import _llm_messages

        history = [
            Message(role=ASSISTANT, content='How should we round?'),
            Message(role=USER, content='doesnt matter'),
        ]
        messages = _llm_messages('system prompt', 'Are we done?', history)
        assert messages[0].role == SYSTEM
        assert messages[1].role == USER
        assert 'How should we round?' in messages[1].content
        assert 'doesnt matter' in messages[1].content
        out = truncate_input_messages_roughly(messages, max_tokens=1_000_000)
        assert out[1].role == USER


class TestStateMachine:
    """1.1 — CONFIRM once; affirm → COMPILE; no confirm regeneration."""

    def test_scripted_yes_produces_exactly_one_confirmation(self, tmp_path):
        draft_path = tmp_path / 'draft.md'
        draft_path.write_text(
            """\
# add_one
Add one to x.
| x | result |
|---|---|
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
""",
            encoding='utf-8',
        )
        confirm_text = 'You add one to a number and return the result. Is that right?'
        compile_payload = '''\
{
  "name": "add_one",
  "description": "Add one to x.",
  "parameters": {"x": {"type": "integer", "description": "value"}},
  "returns": "integer"
}
'''
        llm = _mock_llm(['YES', confirm_text, compile_payload])
        confirmations: List[str] = []

        def ask(q: Question) -> str:
            if q.kind == 'confirm':
                confirmations.append(q.text)
                return 'yes'
            return 'half up'

        with patch(
            'cat_agent.synthesis.intake.pipeline.ToolSmith.synthesize',
            return_value=MagicMock(ok=True, status='success', error=None,
                                   holdout_failures=[], spec=None),
        ):
            with patch(
                'cat_agent.synthesis.intake.pipeline.get_executor',
                return_value=MagicMock(name='wasm'),
            ):
                result = synthesize_from_draft(
                    draft_path, llm=llm, ask=ask, output_dir=str(tmp_path),
                )

        assert len(confirmations) == 1
        assert result.provenance.get('confirmation_generations') == 1
        assert result.interview.confirmation_generated is True
        assert result.interview.phase == Phase.DONE or result.ok


class TestAffirmativeDetection:
    """1.2 — Lexicon before model."""

    def test_affirmative_tokens_advance(self):
        interviewer = SpecInterviewer(llm=_mock_llm([]), max_questions=3)
        for token in (
            'yes', 'y', 'ok', 'okay', 'correct', 'right', 'yep', 'sure',
            'evet', 'ja', 'oui', 'sí', 'sim', 'tak', 'ano',
        ):
            state = InterviewState(
                phase=Phase.CONFIRM,
                confirmation='Is that right?',
                confirmation_generated=True,
            )
            assert is_affirmative(token), token
            verdict = interviewer.record_confirm_answer(state, token)
            assert verdict == 'yes', token
            assert state.phase == Phase.COMPILE

    def test_correction_returns_to_interview(self):
        interviewer = SpecInterviewer(llm=_mock_llm([]), max_questions=3)
        state = InterviewState(
            phase=Phase.CONFIRM,
            confirmation='Is that right?',
            confirmation_generated=True,
        )
        verdict = interviewer.record_confirm_answer(
            state, 'No, tax should be in cents'
        )
        assert verdict == 'no'
        assert state.phase == Phase.INTERVIEW
        assert state.questions_asked == 0  # corrections do not consume budget
        assert state.corrections
        assert state.confirmation_generated is False


class TestEmptyInput:
    """1.3 — Empty never reaches the LLM."""

    def test_blank_helpers(self):
        assert is_blank('')
        assert is_blank('   ')
        assert not is_blank('yes')

    def test_collect_answer_reprompts_without_llm(self, tmp_path):
        from cat_agent.synthesis.intake.pipeline import _collect_answer

        calls = {'n': 0, 'empties': 0}
        state = InterviewState()
        llm_calls_before = state.llm_calls

        def ask(q: Question) -> str:
            calls['n'] += 1
            if calls['empties'] < 2:
                calls['empties'] += 1
                return '  '
            return 'half up'

        answer = _collect_answer(ask, Question(text='Round how?', kind='rounding'), state)
        assert answer == 'half up'
        assert calls['n'] == 3
        assert state.llm_calls == llm_calls_before

    def test_sanitize_drops_empty_before_dispatch(self):
        messages = [
            Message(role=SYSTEM, content='sys'),
            Message(role=USER, content=''),
            Message(role=USER, content='real answer'),
            Message(role=ASSISTANT, content=None),  # type: ignore[arg-type]
        ]
        cleaned = sanitize_messages_for_llm(messages)
        assert all(
            m.content not in (None, '') and str(m.content).strip()
            for m in cleaned
        )
        assert len(cleaned) == 2

    def test_append_refuses_empty(self):
        interviewer = SpecInterviewer(llm=_mock_llm([]))
        state = InterviewState()
        with pytest.raises(ValueError, match='empty-content'):
            interviewer._append_turn(state, 'Q?', '')


class TestDeferral:
    """1.4 — doesn't matter → concrete assistant_default."""

    def test_deferral_yields_rounding_default(self):
        interviewer = SpecInterviewer(llm=_mock_llm([]), max_questions=3)
        draft = _vat_like_draft(rounding_specified=False)
        state = InterviewState(phase=Phase.INTERVIEW)
        q = interviewer.next_interview_question(draft, state)
        assert q is not None and q.kind == 'rounding'
        assert is_deferral("doesn't matter")
        state = interviewer.record_interview_answer(state, q, "doesn't matter")
        assert len(state.decisions) == 1
        d = state.decisions[0]
        assert d.source == 'assistant_default'
        assert d.topic == 'rounding'
        assert 'half' in d.rule.lower()
        assert DEFAULT_ROUNDING_RULE in d.rule or 'half up' in d.rule.lower()

    def test_confirmation_never_says_unspecified_after_deferral(self):
        confirm = (
            'Gross and rate go in; net and tax come out. '
            'Half-cent amounts round half up. Invalid rates are refused. Right?'
        )
        llm = _mock_llm([confirm])
        interviewer = SpecInterviewer(llm=llm, max_questions=3)
        draft = _vat_like_draft(rounding_specified=False)
        state = InterviewState(
            phase=Phase.INTERVIEW,
            decisions=[],
        )
        # Pretend we deferred rounding
        from cat_agent.synthesis.intake.interview import ResolvedDecision
        state.decisions.append(
            ResolvedDecision(
                topic='rounding',
                rule=DEFAULT_ROUNDING_RULE,
                source='assistant_default',
                raw_answer='doesnt matter',
            )
        )
        state.asked_question_keys.append(question_key(
            'When a calculated amount lands exactly on a half-cent, how should we round?'
        ))
        state.questions_asked = 1
        q = interviewer.enter_confirm(draft, state)
        low = q.text.lower()
        for marker in ('unspecified', 'not decided', 'not specified', 'left unspecified'):
            assert marker not in low
        assert len(re.findall(r'[.!?]', q.text)) <= 4  # ≤3 sentences + maybe trailing?


class TestConfirmationLength:
    """1.6 — ≤3 sentences."""

    def test_confirmation_capped_at_three_sentences(self):
        long = (
            'First sentence about inputs. '
            'Second about the rule. '
            'Third about errors. '
            'Fourth must be dropped. '
            'Fifth also gone.'
        )
        llm = _mock_llm([long])
        interviewer = SpecInterviewer(llm=llm, max_questions=3)
        draft = _draft_complete()
        state = InterviewState(phase=Phase.INTERVIEW)
        q = interviewer.enter_confirm(draft, state)
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', q.text.strip()) if s.strip()]
        assert len(sentences) <= 3


class TestNoRepeatQuestion:

    def test_same_question_never_asked_twice(self):
        interviewer = SpecInterviewer(llm=_mock_llm([]), max_questions=3)
        draft = _vat_like_draft(rounding_specified=False)
        state = InterviewState(phase=Phase.INTERVIEW)
        q1 = interviewer.next_interview_question(draft, state)
        assert q1 is not None
        state = interviewer.record_interview_answer(state, q1, 'skip')
        q2 = interviewer.next_interview_question(draft, state)
        if q2 is not None:
            assert question_key(q2.text) != question_key(q1.text)
