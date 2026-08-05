"""Synthesise a sandboxed ``@tool`` from a :class:`ToolSpec`."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from cat_agent.llm import get_chat_model
from cat_agent.llm.base import BaseChatModel
from cat_agent.llm.schema import ASSISTANT, SYSTEM, USER, Message
from cat_agent.log import logger
from cat_agent.observability.context import get_run_context, run_context
from cat_agent.observability.emitter import emit, resolve_handlers
from cat_agent.observability.events import AgentEvent
from cat_agent.observability.helpers import agent_model_name
from cat_agent.synthesis.artifacts import write_artifacts
from cat_agent.synthesis.entry_point import (
    ensure_entry_point,
    extract_impl_code,
    simplify_name_error,
)
from cat_agent.synthesis.executors.base import ExecResult, SandboxExecutor
from cat_agent.synthesis.harness import assert_json_serializable
from cat_agent.synthesis.llm_text import collect_chat_text
from cat_agent.synthesis.overfit import check_overfit
from cat_agent.synthesis.spec import Example, ToolSpec

if TYPE_CHECKING:
    from cat_agent.observability.handlers.base import BaseHandler


class Status(str, Enum):
    SUCCESS = 'success'
    EXHAUSTED = 'exhausted'
    HOLDOUT_FAILED = 'holdout_failed'


HOLDOUT_FAILED_MESSAGE = (
    'The generated code passed all supplied examples but failed an unseen case. '
    'Add this case to your spec and run again.'
)


@dataclass
class AttemptRecord:
    attempt: int
    code: str
    work_passed: int = 0
    work_failed: int = 0
    holdout_passed: int = 0
    holdout_failed: int = 0
    error: Optional[str] = None
    duration_ms: float = 0.0
    stage: str = 'work'
    example_results: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SynthesisResult:
    ok: bool
    status: Status
    spec: ToolSpec
    code: Optional[str]
    artifact_dir: Optional[str]
    attempts: List[AttemptRecord]
    registered_name: str
    error: Optional[str] = None
    holdout_failures: List[Dict[str, Any]] = field(default_factory=list)


class ToolSmith:
    """LLM-driven synthesizer that never executes model code in-process.

    *intake_llm* is optional and unused by the code loop itself; the intake
    pipeline stores it for provenance. Prefer a larger multilingual model for
    intake than for synthesis (D4).
    """

    def __init__(
        self,
        llm: Union[dict, BaseChatModel, None],
        executor: SandboxExecutor,
        max_attempts: int = 5,
        handlers: Optional[List['BaseHandler']] = None,
        *,
        output_dir: Optional[str] = None,
        intake_llm: Union[dict, BaseChatModel, None] = None,
    ):
        if isinstance(llm, dict) or llm is None:
            self.llm = get_chat_model(llm or {})
        else:
            self.llm = llm
        if isinstance(intake_llm, dict):
            self.intake_llm = get_chat_model(intake_llm)
        else:
            self.intake_llm = intake_llm  # may be None → callers fall back to llm
        self.executor = executor
        self.max_attempts = max(1, int(max_attempts))
        self._handlers = handlers or []
        self.output_dir = output_dir

    def synthesize(
        self,
        spec: ToolSpec,
        *,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> SynthesisResult:
        handlers = resolve_handlers(self._handlers, None)
        with run_context(
            agent_name=f'ToolSmith:{spec.function_name}',
            agent_class='ToolSmith',
            handlers=handlers,
        ) as ctx:
            return self._synthesize_inner(spec, ctx, provenance=provenance)

    def _synthesize_inner(
        self,
        spec: ToolSpec,
        ctx,
        *,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> SynthesisResult:
        work, holdout = spec.split_examples()
        for example in list(work) + list(holdout):
            assert_json_serializable(example.expected, label='example.expected')

        history: List[Message] = [
            Message(role=SYSTEM, content=_system_prompt(spec)),
        ]
        feedback = ''
        attempts: List[AttemptRecord] = []
        last_code: Optional[str] = None

        for attempt_no in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            user_prompt = _user_prompt(spec, work, feedback=feedback, attempt=attempt_no)
            history.append(Message(role=USER, content=user_prompt))
            raw = self._call_llm(history)
            history.append(Message(role=ASSISTANT, content=raw))
            code = extract_impl_code(raw, spec.function_name)
            code, name_err = ensure_entry_point(code, spec.function_name)
            last_code = code

            if name_err:
                record = AttemptRecord(
                    attempt=attempt_no,
                    code=code,
                    error=name_err,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    stage='entry_point',
                )
                attempts.append(record)
                self._emit_attempt(ctx, record)
                feedback = name_err
                continue

            # Overfit guards use work examples only — never holdout values in feedback.
            overfit_msg = check_overfit(code, work)
            if overfit_msg:
                record = AttemptRecord(
                    attempt=attempt_no,
                    code=code,
                    error=overfit_msg,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    stage='overfit',
                )
                attempts.append(record)
                self._emit_attempt(ctx, record)
                feedback = overfit_msg
                continue

            work_results, work_error = self._run_examples(spec, code, work, label='work')
            work_passed = sum(1 for r in work_results if r['ok'])
            work_failed = len(work_results) - work_passed
            if work_error:
                record = AttemptRecord(
                    attempt=attempt_no,
                    code=code,
                    work_passed=work_passed,
                    work_failed=work_failed,
                    error=work_error,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    stage='work',
                    example_results=work_results,
                )
                attempts.append(record)
                self._emit_attempt(ctx, record)
                feedback = work_error
                continue

            holdout_results, holdout_error = self._run_examples(
                spec, code, holdout, label='holdout')
            holdout_passed = sum(1 for r in holdout_results if r['ok'])
            holdout_failed = len(holdout_results) - holdout_passed
            if holdout_error:
                failing = [r for r in holdout_results if not r['ok']]
                record = AttemptRecord(
                    attempt=attempt_no,
                    code=code,
                    work_passed=work_passed,
                    work_failed=0,
                    holdout_passed=holdout_passed,
                    holdout_failed=holdout_failed,
                    error=HOLDOUT_FAILED_MESSAGE,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    stage='holdout',
                    example_results=work_results + holdout_results,
                )
                attempts.append(record)
                self._emit_attempt(ctx, record)
                return SynthesisResult(
                    ok=False,
                    status=Status.HOLDOUT_FAILED,
                    spec=spec,
                    code=code,
                    artifact_dir=None,
                    attempts=attempts,
                    registered_name=spec.registered_name,
                    error=HOLDOUT_FAILED_MESSAGE,
                    holdout_failures=failing,
                )

            duration_ms = (time.perf_counter() - started) * 1000
            record = AttemptRecord(
                attempt=attempt_no,
                code=code,
                work_passed=work_passed,
                holdout_passed=holdout_passed,
                duration_ms=duration_ms,
                stage='success',
                example_results=work_results + holdout_results,
            )
            attempts.append(record)
            self._emit_attempt(ctx, record)

            artifact_dir = write_artifacts(
                spec=spec,
                code=code,
                executor_name=getattr(self.executor, 'name', 'wasm'),
                model_name=agent_model_name(self.llm),
                attempt_count=attempt_no,
                example_results=work_results + holdout_results,
                work=work,
                holdout=holdout,
                base=self.output_dir,
                provenance=provenance,
            )
            self._audit_created(spec, artifact_dir, attempt_no)
            return SynthesisResult(
                ok=True,
                status=Status.SUCCESS,
                spec=spec,
                code=code,
                artifact_dir=str(artifact_dir),
                attempts=attempts,
                registered_name=spec.registered_name,
            )

        err = (
            f'ToolSmith gave up after {self.max_attempts} attempts for {spec.name!r}.'
        )
        if attempts and attempts[-1].error:
            err = f'{err} Last error: {attempts[-1].error}'
        return SynthesisResult(
            ok=False,
            status=Status.EXHAUSTED,
            spec=spec,
            code=last_code,
            artifact_dir=None,
            attempts=attempts,
            registered_name=spec.registered_name,
            error=err,
        )

    def _run_examples(
        self,
        spec: ToolSpec,
        code: str,
        examples: Sequence[Example],
        *,
        label: str,
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        for index, example in enumerate(examples):
            result: ExecResult = self.executor.run(
                code,
                example.inputs,
                deps=spec.deps or None,
                function_name=spec.function_name,
            )
            entry: Dict[str, Any] = {
                'set': label,
                'index': index,
                'inputs': example.inputs,
                'expected': example.expected,
                'ok': False,
                'returned': result.returned,
                'error': result.error,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration_ms': result.duration_ms,
            }
            if not result.ok:
                entry['ok'] = False
                concise = simplify_name_error(
                    function_name=spec.function_name,
                    error=result.error,
                    stderr=result.stderr,
                )
                if concise:
                    errors.append(
                        f'[{label}#{index}] inputs={example.inputs!r} '
                        f'expected={example.expected!r} error={concise!r}'
                    )
                else:
                    errors.append(
                        f'[{label}#{index}] inputs={example.inputs!r} '
                        f'expected={example.expected!r} error={result.error!r} '
                        f'stderr={result.stderr!r}'
                    )
            elif result.returned != example.expected:
                entry['ok'] = False
                errors.append(
                    f'[{label}#{index}] inputs={example.inputs!r} '
                    f'expected={example.expected!r} actual={result.returned!r}'
                )
            else:
                entry['ok'] = True
            results.append(entry)
        if errors:
            return results, '\n'.join(errors)
        return results, None

    def _call_llm(self, messages: List[Message]) -> str:
        try:
            output = self.llm.chat(messages=messages, stream=True)
        except TypeError:
            output = self.llm.chat(messages=messages)
        return collect_chat_text(output)

    def _emit_attempt(self, ctx, record: AttemptRecord) -> None:
        if ctx is None or not ctx.handlers:
            return
        emit(AgentEvent.synthesis_attempt(
            trace_id=ctx.trace_id,
            run_id=ctx.run_id,
            span_id=ctx.span_id,
            parent_span_id=ctx.parent_span_id,
            agent_name=ctx.agent_name,
            agent_class=ctx.agent_class,
            attempt=record.attempt,
            stage=record.stage,
            work_passed=record.work_passed,
            work_failed=record.work_failed,
            holdout_passed=record.holdout_passed,
            holdout_failed=record.holdout_failed,
            duration_ms=record.duration_ms,
            ok=record.error is None and record.stage == 'success',
            error=record.error,
        ))

    def _audit_created(self, spec: ToolSpec, artifact_dir, attempt_no: int) -> None:
        from cat_agent.security.audit import append_audit_record, is_audit_enabled

        if not is_audit_enabled():
            return
        ctx = get_run_context()
        append_audit_record(
            'synthesis.tool_created',
            {
                'name': spec.function_name,
                'registered_name': spec.registered_name,
                'artifact_dir': str(artifact_dir),
                'attempt_count': attempt_no,
                'executor': getattr(self.executor, 'name', None),
                'model': agent_model_name(self.llm),
            },
            trace_id=ctx.trace_id if ctx else None,
            run_id=ctx.run_id if ctx else None,
            agent_name=ctx.agent_name if ctx else 'ToolSmith',
            agent_class='ToolSmith',
        )
        logger.info('Audit: synthesis.tool_created {}', spec.registered_name)


def _system_prompt(spec: ToolSpec) -> str:
    deps_note = (
        'You may use only the Python standard library.'
        if not spec.deps
        else f'Allowed third-party packages: {spec.deps!r}.'
    )
    return (
        'You are a careful Python engineer. Write ONE function that solves the task.\n'
        'Constraints:\n'
        '- Python 3.10-compatible syntax only (no match/case reliance on 3.11+, '
        'no ExceptionGroup, no tomllib, no typing.Self).\n'
        '- Full type hints on every parameter and the return type.\n'
        '- Google-style docstring with an Args: section for every parameter.\n'
        f'- Function name must be exactly `{spec.function_name}` '
        f'(spelling matters — the harness calls `{spec.function_name}(**inputs)`).\n'
        f'- {deps_note}\n'
        '- Do not hardcode example outputs; derive results from inputs.\n'
        '- Return only JSON-serialisable values '
        '(dict/list/str/int/float/bool/None).\n'
        '- Return only a single markdown fenced python code block.\n'
    )


def _user_prompt(
    spec: ToolSpec,
    work: Sequence[Example],
    *,
    feedback: str,
    attempt: int,
) -> str:
    params = '\n'.join(
        f'- {k}: {p.type}'
        + (f' — {p.description}' if (p.description or '').strip() else '')
        for k, p in spec.parameters.items()
    )
    examples = '\n'.join(
        f'- inputs={ex.inputs!r} → expected={ex.expected!r}'
        + (f'  ({ex.note})' if ex.note else '')
        for ex in work
    )
    parts = [
        f'Attempt {attempt}. Implement `{spec.function_name}`.',
        f'Description: {spec.description}',
        f'Returns: {spec.returns}',
        f'Parameters:\n{params}',
        f'Examples (must pass):\n{examples}',
    ]
    if feedback:
        parts.append(f'Previous attempt feedback:\n{feedback}')
    return '\n\n'.join(parts)
