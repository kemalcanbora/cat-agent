"""End-to-end intake → interview → confirm → compile → synthesise.

Explicit phases::

    INTERVIEW → CONFIRM → COMPILE → (ToolSmith) → DONE
                  ↑                        │
                  └── correction ──────────┘
    HOLDOUT_REOPEN is the only other backward edge (Task 8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

from cat_agent.llm.base import BaseChatModel
from cat_agent.log import logger
from cat_agent.synthesis.executors import SandboxExecutor, get_executor
from cat_agent.synthesis.artifacts import update_manifest_verification
from cat_agent.synthesis.intake.compile import CompileResult, compile_to_spec
from cat_agent.synthesis.intake.draft import Draft
from cat_agent.synthesis.intake.interview import (
    Phase,
    InterviewState,
    Question,
    SpecInterviewer,
    holdout_question,
    insensitivity_question,
    is_affirmative,
    is_blank,
    question_key,
)
from cat_agent.synthesis.mutation import probe_input_sensitivity
from cat_agent.synthesis.smith import Status, SynthesisResult, ToolSmith
from cat_agent.synthesis.spec import Example
from cat_agent.synthesis.spec_quality import SpecWarning

if TYPE_CHECKING:
    from cat_agent.observability.handlers.base import BaseHandler
    from cat_agent.security.principal import Principal

AskFn = Callable[[Question], str]

DEFAULT_MAX_QUESTIONS = 3
DEFAULT_MAX_USER_TURNS = 8


@dataclass
class IntakeResult:
    ok: bool
    draft: Draft
    spec: Optional[Any]
    synthesis: Optional[SynthesisResult]
    interview: InterviewState
    confirmation: Optional[str] = None
    compile: Optional[CompileResult] = None
    holdout_rounds: int = 0
    error: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    phase: Phase = Phase.DONE
    spec_warnings: List[SpecWarning] = field(default_factory=list)


def synthesize_from_draft(
    path: Union[str, Path],
    llm: Union[dict, BaseChatModel, None],
    *,
    intake_llm: Union[dict, BaseChatModel, None] = None,
    executor: Optional[SandboxExecutor] = None,
    locale: Optional[str] = None,
    ask: Optional[AskFn] = None,
    max_holdout_rounds: int = 2,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    max_user_turns: int = DEFAULT_MAX_USER_TURNS,
    output_dir: Optional[str] = None,
    handlers: Optional[List['BaseHandler']] = None,
    lang: Optional[str] = None,
    allow_weak_spec: bool = True,
    principal: Optional['Principal'] = None,
) -> IntakeResult:
    """One-call API: draft → interview → confirm → ToolSpec → ToolSmith."""
    ask = ask or _default_ask
    draft = Draft.from_path(path, locale=locale)
    synth_llm = llm
    interview_llm = intake_llm if intake_llm is not None else llm

    interviewer = SpecInterviewer(
        llm=interview_llm,
        max_questions=max_questions,
        lang=lang,
        handlers=handlers,
    )
    state = InterviewState(phase=Phase.INTERVIEW)
    working_lang = interviewer.working_lang(draft)

    # -------------------- INTERVIEW --------------------
    while state.phase == Phase.INTERVIEW:
        if state.user_turns >= max_user_turns:
            logger.warning('Hit max_user_turns={}; forcing CONFIRM', max_user_turns)
            break

        if interviewer.is_interview_complete(draft, state):
            break

        question = interviewer.next_interview_question(draft, state)
        if question is None:
            break

        answer = _collect_answer(ask, question, state)
        if answer is None:
            # User abandoned after repeated empties — stop cleanly.
            return IntakeResult(
                ok=False,
                draft=draft,
                spec=None,
                synthesis=None,
                interview=state,
                error='Abandoned after repeated empty inputs.',
                phase=state.phase,
            )
        state = interviewer.record_interview_answer(state, question, answer)

    # -------------------- CONFIRM (once) --------------------
    confirm_q = interviewer.enter_confirm(draft, state)
    confirmation_generations = 1

    while state.phase == Phase.CONFIRM:
        if state.user_turns >= max_user_turns:
            return IntakeResult(
                ok=False,
                draft=draft,
                spec=None,
                synthesis=None,
                interview=state,
                confirmation=state.confirmation,
                error='Hit turn cap before confirmation was accepted.',
                phase=state.phase,
            )

        answer = _collect_answer(ask, confirm_q, state)
        if answer is None:
            return IntakeResult(
                ok=False,
                draft=draft,
                spec=None,
                synthesis=None,
                interview=state,
                confirmation=state.confirmation,
                error='Abandoned during confirmation.',
                phase=state.phase,
            )

        verdict = interviewer.record_confirm_answer(state, answer)
        if verdict == 'yes':
            break
        if verdict == 'ambiguous':
            # Closed yes/no — do NOT regenerate the summary.
            closed = interviewer.closed_yes_no_prompt()
            answer2 = _collect_answer(ask, closed, state)
            if answer2 is None:
                return IntakeResult(
                    ok=False,
                    draft=draft,
                    spec=None,
                    synthesis=None,
                    interview=state,
                    confirmation=state.confirmation,
                    error='Abandoned during yes/no confirmation.',
                    phase=state.phase,
                )
            verdict2 = interviewer.record_confirm_answer(state, answer2)
            if verdict2 == 'yes':
                break
            if verdict2 == 'ambiguous':
                # Treat second ambiguity as correction path.
                state.confirmed = False
                state.confirmation = None
                state.confirmation_generated = False
                state.corrections.append(answer2.strip())
                state.phase = Phase.INTERVIEW

        if state.phase == Phase.INTERVIEW:
            # Correction: finish any remaining interview then regenerate confirm once.
            while (
                state.phase == Phase.INTERVIEW
                and state.user_turns < max_user_turns
                and not interviewer.is_interview_complete(draft, state)
            ):
                q = interviewer.next_interview_question(draft, state)
                if q is None:
                    break
                ans = _collect_answer(ask, q, state)
                if ans is None:
                    break
                state = interviewer.record_interview_answer(state, q, ans)
            confirm_q = interviewer.enter_confirm(draft, state)
            confirmation_generations += 1

    if not state.confirmed:
        return IntakeResult(
            ok=False,
            draft=draft,
            spec=None,
            synthesis=None,
            interview=state,
            confirmation=state.confirmation,
            error='Confirmation was not accepted.',
            phase=state.phase,
            provenance={'confirmation_generations': confirmation_generations},
        )

    confirmation = state.confirmation or ''
    state.phase = Phase.COMPILE

    # -------------------- COMPILE --------------------
    compile_result = compile_to_spec(
        draft,
        state.history,
        confirmation,
        llm=interview_llm,
        state=state,
        decisions=state.decisions,
    )

    # On failure: retry from draft alone (ignore corrupted history).
    if not compile_result.ok:
        logger.error(
            'compile_to_spec failed with history (field={}): {}',
            compile_result.failed_field,
            compile_result.error,
        )
        compile_result = compile_to_spec(
            draft,
            history=[],
            confirmation=confirmation,
            llm=interview_llm,
            state=InterviewState(decisions=list(state.decisions)),
            decisions=state.decisions,
            draft_only=True,
        )

    # One targeted question max, never a generic restate, never a repeat.
    if (
        not compile_result.ok
        and compile_result.reinterview_question
        and question_key(compile_result.reinterview_question) not in state.asked_question_keys
    ):
        q = Question(
            text=compile_result.reinterview_question,
            kind='compile_gap',
            priority=0,
        )
        state.asked_question_keys.append(question_key(q.text))
        answer = _collect_answer(ask, q, state)
        if answer is not None:
            state = interviewer.record_interview_answer(state, q, answer)
            compile_result = compile_to_spec(
                draft,
                history=[],
                confirmation=confirmation,
                llm=interview_llm,
                state=state,
                decisions=state.decisions,
                draft_only=True,
            )

    if not compile_result.ok or compile_result.spec is None:
        return IntakeResult(
            ok=False,
            draft=draft,
            spec=None,
            synthesis=None,
            interview=state,
            confirmation=confirmation,
            compile=compile_result,
            error=(
                f'Compilation failed'
                f'{f" ({compile_result.failed_field})" if compile_result.failed_field else ""}: '
                f'{compile_result.error}. Draft and interview are preserved for review.'
            ),
            phase=Phase.COMPILE,
            provenance={
                'confirmation_generations': confirmation_generations,
                'interview': _interview_payload(state, confirmation),
            },
        )

    if compile_result.name_changed:
        logger.info(
            'Tool name sanitised: {!r} → {!r}',
            compile_result.original_name,
            compile_result.sanitised_name,
        )

    spec = compile_result.spec
    spec_warnings = list(compile_result.warnings or [])
    warn_codes = [w.code for w in spec_warnings if w.severity == 'warn']
    if warn_codes and not allow_weak_spec:
        return IntakeResult(
            ok=False,
            draft=draft,
            spec=spec,
            synthesis=None,
            interview=state,
            confirmation=confirmation,
            compile=compile_result,
            error=(
                'Weak spec rejected (allow_weak_spec=False): '
                + ', '.join(warn_codes)
            ),
            phase=Phase.COMPILE,
            provenance={
                'confirmation_generations': confirmation_generations,
                'interview': _interview_payload(state, confirmation),
                'spec_warnings': [_warning_dict(w) for w in spec_warnings],
            },
            spec_warnings=spec_warnings,
        )

    exec_backend = executor or get_executor('wasm')
    smith = ToolSmith(
        llm=synth_llm,
        executor=exec_backend,
        handlers=handlers,
        output_dir=output_dir,
        intake_llm=interview_llm,
    )

    provenance = {
        'draft_markdown': draft.raw_markdown,
        'draft_lang': working_lang,
        'locale': draft.locale,
        'interview': _interview_payload(state, confirmation),
        'source_path': draft.source_path,
        'confirmation_generations': confirmation_generations,
        'spec_warnings': [_warning_dict(w) for w in spec_warnings],
    }
    if principal is not None:
        provenance['synthesized_by'] = principal.user_id
        provenance['group_id'] = principal.group_id

    # -------------------- SYNTHESISE (+ holdout reopen) --------------------
    holdout_rounds = 0
    synthesis = smith.synthesize(spec, provenance=provenance, principal=principal)

    while (
        synthesis.status == Status.HOLDOUT_FAILED
        and holdout_rounds < max_holdout_rounds
    ):
        state.phase = Phase.HOLDOUT_REOPEN
        holdout_rounds += 1
        q = holdout_question(synthesis.holdout_failures, lang=working_lang)
        answer = _collect_answer(ask, q, state)
        if answer is None:
            break
        state = interviewer.record_interview_answer(state, q, answer)
        new_ex = _example_from_holdout_answer(synthesis.holdout_failures, answer)
        if new_ex is None:
            break
        state.added_examples.append(new_ex)
        state.example_traces.append({
            'source': 'holdout',
            'answer': answer,
            'example': {'inputs': new_ex.inputs, 'expected': new_ex.expected},
        })
        from cat_agent.synthesis.spec import tool_spec_from_dict

        data = spec.to_dict()
        data['examples'] = [
            {'inputs': ex.inputs, 'expected': ex.expected, 'note': ex.note}
            for ex in list(spec.examples) + [new_ex]
        ]
        try:
            spec = tool_spec_from_dict(data)
        except ValueError as exc:
            if 'already registered' in str(exc):
                from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, TOOL_REGISTRY
                TOOL_REGISTRY.pop(spec.registered_name, None)
                OPTIONAL_TOOL_REGISTRY.pop(spec.registered_name, None)
                TOOL_REGISTRY.pop(spec.name, None)
                OPTIONAL_TOOL_REGISTRY.pop(spec.name, None)
                spec = tool_spec_from_dict(data)
            else:
                raise
        provenance['interview'] = _interview_payload(state, confirmation)
        synthesis = smith.synthesize(spec, provenance=provenance, principal=principal)

    state.phase = Phase.DONE

    if synthesis.status == Status.HOLDOUT_FAILED and holdout_rounds >= max_holdout_rounds:
        return IntakeResult(
            ok=False,
            draft=draft,
            spec=spec,
            synthesis=synthesis,
            interview=state,
            confirmation=confirmation,
            compile=compile_result,
            holdout_rounds=holdout_rounds,
            error=(
                f'Holdout still failing after {max_holdout_rounds} clarification rounds.'
            ),
            provenance=provenance,
            phase=Phase.DONE,
            spec_warnings=spec_warnings,
        )

    # Input-space sensitivity: advisory question, not a rejection.
    if synthesis.ok and synthesis.code:
        synthesis, spec, provenance = _maybe_resolve_insensitivity(
            ask=ask,
            interviewer=interviewer,
            state=state,
            smith=smith,
            spec=spec,
            synthesis=synthesis,
            provenance=provenance,
            confirmation=confirmation,
            working_lang=working_lang,
            executor=exec_backend,
            principal=principal,
        )

    return IntakeResult(
        ok=bool(synthesis.ok),
        draft=draft,
        spec=spec,
        synthesis=synthesis,
        interview=state,
        confirmation=confirmation,
        compile=compile_result,
        holdout_rounds=holdout_rounds,
        error=None if synthesis.ok else synthesis.error,
        provenance=provenance,
        phase=Phase.DONE,
        spec_warnings=spec_warnings,
    )


def _warning_dict(warning: SpecWarning) -> Dict[str, str]:
    return {
        'code': warning.code,
        'message': warning.message,
        'severity': warning.severity,
    }


def _maybe_resolve_insensitivity(
    *,
    ask: AskFn,
    interviewer: SpecInterviewer,
    state: InterviewState,
    smith: ToolSmith,
    spec: Any,
    synthesis: SynthesisResult,
    provenance: Dict[str, Any],
    confirmation: str,
    working_lang: str,
    executor: SandboxExecutor,
    principal: Optional['Principal'] = None,
) -> tuple:
    """Ask about input insensitivity; optionally append a negative and resynthesise."""

    def runner(code: str, inputs: Dict[str, Any]) -> Any:
        result = executor.run(
            code,
            inputs,
            deps=spec.deps or None,
            function_name=spec.function_name,
        )
        if not result.ok:
            raise RuntimeError(result.error or 'exec failed')
        return result.returned

    findings = probe_input_sensitivity(
        synthesis.code or '',
        spec,
        list(spec.examples),
        runner,
        limit=64,
    )
    if not findings:
        return synthesis, spec, provenance

    finding = findings[0]
    q = insensitivity_question(finding, lang=working_lang)
    answer = _collect_answer(ask, q, state)
    if answer is None:
        return synthesis, spec, provenance
    state = interviewer.record_interview_answer(state, q, answer)
    # Affirmative = "yes, intended" → leave alone. Anything else → add negative.
    if is_affirmative(answer):
        decision = {
            'kind': 'insensitivity',
            'param': finding.param,
            'question': q.text,
            'answer': answer,
            'user_said_intended': True,
            'variants_tried': finding.variants_tried,
            'variants_per_example': dict(finding.variants_per_example),
        }
        provenance['interview'] = _interview_payload(state, confirmation)
        provenance['insensitivity'] = decision
        provenance['warnings_overridden'] = True
        provenance['override_decision'] = decision
        verification = dict(synthesis.verification or {})
        verification['warnings_overridden'] = True
        verification['override_decision'] = decision
        synthesis.verification = verification
        if synthesis.artifact_dir:
            try:
                update_manifest_verification(synthesis.artifact_dir, verification)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'Could not update manifest verification after override: {}',
                    exc,
                )
        return synthesis, spec, provenance

    sample = finding.sample_unchanged[0] if finding.sample_unchanged else None
    if sample is None:
        return synthesis, spec, provenance
    new_inputs = dict(finding.base_inputs)
    new_inputs[finding.param] = sample
    new_ex = Example(
        inputs=new_inputs,
        expected=False,
        note='insensitivity clarification: near-miss negative',
    )
    state.added_examples.append(new_ex)
    state.example_traces.append({
        'source': 'insensitivity',
        'answer': answer,
        'example': {'inputs': new_ex.inputs, 'expected': new_ex.expected},
    })
    from cat_agent.synthesis.spec import tool_spec_from_dict

    data = spec.to_dict()
    data['examples'] = [
        {'inputs': ex.inputs, 'expected': ex.expected, 'note': ex.note}
        for ex in list(spec.examples) + [new_ex]
    ]
    try:
        spec = tool_spec_from_dict(data)
    except ValueError as exc:
        if 'already registered' in str(exc):
            from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, TOOL_REGISTRY
            TOOL_REGISTRY.pop(spec.registered_name, None)
            OPTIONAL_TOOL_REGISTRY.pop(spec.registered_name, None)
            TOOL_REGISTRY.pop(spec.name, None)
            OPTIONAL_TOOL_REGISTRY.pop(spec.name, None)
            spec = tool_spec_from_dict(data)
        else:
            raise
    decision = {
        'kind': 'insensitivity',
        'param': finding.param,
        'question': q.text,
        'answer': answer,
        'user_said_intended': False,
        'variants_tried': finding.variants_tried,
        'variants_per_example': dict(finding.variants_per_example),
        'added_example': {'inputs': new_ex.inputs, 'expected': new_ex.expected},
    }
    provenance['interview'] = _interview_payload(state, confirmation)
    provenance['insensitivity'] = decision
    provenance['warnings_overridden'] = False
    provenance['override_decision'] = decision
    synthesis = smith.synthesize(spec, provenance=provenance, principal=principal)
    return synthesis, spec, provenance


def _collect_answer(
    ask: AskFn,
    question: Question,
    state: InterviewState,
    *,
    max_empty: int = 5,
) -> Optional[str]:
    """Collect a non-blank answer locally — never sends empty content to the LLM."""
    empties = 0
    while empties < max_empty:
        raw = ask(question)
        if not is_blank(raw):
            state.consecutive_empty = 0
            return raw.strip()
        empties += 1
        hint = (
            'Please type an answer, or "skip" / "you decide" if either way is fine, '
            'or "yes" to confirm.'
            if empties >= 2
            else 'Please type an answer (empty input is ignored).'
        )
        print(hint)
        state.consecutive_empty = empties
    return None


def _default_ask(question: Question) -> str:
    print(question.text)
    print(f'[{question.pending_sentinel}]')
    return input('> ')


def _interview_payload(state: InterviewState, confirmation: str) -> Dict[str, Any]:
    turns = []
    pending_q = None
    for msg in state.history:
        if msg.role == 'assistant':
            pending_q = msg.content
        elif msg.role == 'user' and pending_q is not None:
            turns.append({'question': pending_q, 'answer': msg.content})
            pending_q = None
        elif msg.role == 'user':
            turns.append({'question': None, 'answer': msg.content})
    return {
        'turns': turns,
        'confirmation': confirmation,
        'confirmed': state.confirmed,
        'questions_asked': state.questions_asked,
        'corrections': list(state.corrections),
        'decisions': [
            {
                'topic': d.topic,
                'rule': d.rule,
                'source': d.source,
                'raw_answer': d.raw_answer,
            }
            for d in state.decisions
        ],
        'added_examples': [
            {'inputs': ex.inputs, 'expected': ex.expected, 'note': ex.note}
            for ex in state.added_examples
        ],
        'example_traces': list(state.example_traces),
        'phase': state.phase.value,
        'user_turns': state.user_turns,
        'llm_calls': state.llm_calls,
    }


def _example_from_holdout_answer(
    failures: Sequence[Dict[str, Any]],
    answer: str,
) -> Optional[Example]:
    if not failures:
        return None
    case = failures[0]
    inputs = dict(case.get('inputs') or {})
    expected: Any = answer.strip()
    try:
        if expected.startswith('{') or expected.startswith('['):
            expected = json.loads(expected)
        else:
            from cat_agent.synthesis.intake.numbers import parse_cell
            parsed = parse_cell(expected)
            if parsed.ok and not parsed.ambiguous:
                expected = parsed.value
    except Exception:
        pass
    return Example(
        inputs=inputs,
        expected=expected,
        note=f'holdout clarification: {answer[:80]}',
    )
