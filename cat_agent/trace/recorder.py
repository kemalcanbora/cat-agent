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

"""TraceRecorder — agents write steps here; stores flush append-only."""

from __future__ import annotations

import json
import os
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional, Union

from cat_agent.llm.schema import Message
from cat_agent.trace.cost import estimate_cost_usd, load_price_table
from cat_agent.trace.redact import redact_llm_config, redact_obj
from cat_agent.trace.schema import (
    ContextOpPayload,
    ErrorPayload,
    HandoffPayload,
    LLMCallPayload,
    Run,
    RunLimits,
    Step,
    ToolCallPayload,
    new_id,
    utc_now_iso,
)
from cat_agent.trace.store import InMemoryTraceStore, JSONLTraceStore, TraceStore
from cat_agent.trace.tokens import resolve_usage

_current_recorder: ContextVar[Optional['TraceRecorder']] = ContextVar(
    'cat_agent_trace_recorder', default=None,
)


def get_trace_recorder() -> Optional['TraceRecorder']:
    return _current_recorder.get()


def is_trace_enabled() -> bool:
    raw = os.getenv('CAT_AGENT_TRACE', '').strip().lower()
    return raw in {'1', 'true', 'yes', 'on'}


def default_trace_store() -> Optional[TraceStore]:
    if not is_trace_enabled():
        return None
    path = os.getenv('CAT_AGENT_TRACE_FILE', '').strip()
    if path:
        return JSONLTraceStore(path)
    return InMemoryTraceStore()


def _parse_tool_args(tool_args: Union[str, dict]) -> dict:
    if isinstance(tool_args, dict):
        return tool_args
    try:
        parsed = json.loads(tool_args or '{}')
        return parsed if isinstance(parsed, dict) else {'_raw': tool_args}
    except (TypeError, json.JSONDecodeError):
        return {'_raw': str(tool_args)}


def _preview(result: Any, limit: int = 2000) -> tuple[str, int]:
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(result, ensure_ascii=False, default=str)
        except TypeError:
            text = str(result)
    raw_bytes = len(text.encode('utf-8'))
    if len(text) > limit:
        text = text[:limit] + '...'
    return text, raw_bytes


class TraceRecorder:
    """Records a single :class:`Run` with append-only step flushes."""

    def __init__(
        self,
        *,
        store: Optional[TraceStore] = None,
        agent_name: str = '',
        agent_class: str = '',
        llm_config: Optional[dict] = None,
        initial_messages: Optional[List[Message]] = None,
        metadata: Optional[dict] = None,
        limits: Optional[RunLimits] = None,
        parent_step_id: Optional[str] = None,
        price_table: Optional[dict] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.store = store
        self.limits = limits or RunLimits()
        self.parent_step_id = parent_step_id
        self._price_table = price_table if price_table is not None else load_price_table()
        self._started_mono = time.monotonic()
        self._token: Optional[Any] = None
        self._open_step_id: Optional[str] = None  # current nesting anchor for children
        self.run = Run(
            run_id=run_id or new_id(),
            agent_name=agent_name or '',
            agent_class=agent_class or '',
            llm_config=redact_llm_config(llm_config or {}),
            initial_messages=list(initial_messages or []),
            metadata=dict(metadata or {}),
            status='running',
        )
        if self.store is not None:
            self.store.write_run_header(self.run)

    # ------------------------------------------------------------------ nesting

    @property
    def current_step_id(self) -> Optional[str]:
        """Step id that nested agent runs should use as ``parent_step_id``."""
        return self._open_step_id or (
            self.run.steps[-1].step_id if self.run.steps else None
        )

    def set_nesting_anchor(self, step_id: Optional[str]) -> None:
        self._open_step_id = step_id

    # ------------------------------------------------------------------ limits

    def check_limits(self) -> Optional[str]:
        """Return a termination_reason if a budget is exhausted, else ``None``."""
        elapsed = time.monotonic() - self._started_mono
        lim = self.limits
        if lim.max_wall_clock_seconds is not None and elapsed >= lim.max_wall_clock_seconds:
            return 'wall_clock'
        totals = self.run.totals
        # Include in-progress step counts from recomputed totals.
        self.run.recompute_totals(wall_clock_seconds=elapsed)
        totals = self.run.totals
        if lim.max_steps is not None and totals.steps >= lim.max_steps:
            return 'max_steps'
        if lim.max_total_tokens is not None and totals.total_tokens >= lim.max_total_tokens:
            return 'max_tokens'
        if lim.max_tool_calls is not None and totals.tool_calls >= lim.max_tool_calls:
            return 'max_tool_calls'
        return None

    def should_stop(self) -> bool:
        return self.check_limits() is not None

    # ------------------------------------------------------------------ steps

    def _flush_step(self, step: Step) -> Step:
        self.run.steps.append(step)
        elapsed = time.monotonic() - self._started_mono
        self.run.recompute_totals(wall_clock_seconds=elapsed)
        cost = estimate_cost_usd(
            model=self._model_hint(),
            prompt_tokens=self.run.totals.prompt_tokens,
            completion_tokens=self.run.totals.completion_tokens,
            price_table=self._price_table,
        )
        self.run.totals.estimated_cost_usd = cost
        if self.store is not None:
            self.store.append_step(self.run.run_id, step)
        return step

    def _model_hint(self) -> Optional[str]:
        cfg = self.run.llm_config or {}
        model = cfg.get('model')
        return model if isinstance(model, str) else None

    def record_llm_call(
        self,
        *,
        model: Optional[str] = None,
        model_type: Optional[str] = None,
        gen_ai_system: Optional[str] = None,
        messages_in: Optional[List[Message]] = None,
        message_out: Optional[Message] = None,
        messages_out: Optional[List[Message]] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        tokens_estimated: bool = False,
        finish_reason: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        sampling_params: Optional[dict] = None,
        llm: Any = None,
        parent_step_id: Optional[str] = None,
        started_at: Optional[str] = None,
        duration_ms: int = 0,
    ) -> Step:
        outs = messages_out or ([message_out] if message_out else [])
        if prompt_tokens is None or completion_tokens is None:
            p, c, est = resolve_usage(outs, messages_in, llm=llm)
            prompt_tokens = p if prompt_tokens is None else prompt_tokens
            completion_tokens = c if completion_tokens is None else completion_tokens
            tokens_estimated = tokens_estimated or est
        out_msg = message_out or (outs[-1] if outs else None)
        payload = LLMCallPayload(
            model=model,
            model_type=model_type,
            gen_ai_system=gen_ai_system,
            messages_in=list(messages_in or []),
            message_out=out_msg,
            prompt_tokens=int(prompt_tokens or 0),
            completion_tokens=int(completion_tokens or 0),
            tokens_estimated=tokens_estimated,
            finish_reason=finish_reason,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            sampling_params=dict(sampling_params or {}),
        )
        step = Step.from_payload(
            step_index=len(self.run.steps),
            kind='llm_call',
            payload=payload,
            parent_step_id=parent_step_id if parent_step_id is not None else self.parent_step_id,
            started_at=started_at,
            ended_at=utc_now_iso(),
            duration_ms=duration_ms,
        )
        return self._flush_step(step)

    def record_tool_call(
        self,
        *,
        tool_name: str,
        arguments: Union[str, dict],
        result: Any = None,
        succeeded: bool = True,
        error: Optional[str] = None,
        parent_step_id: Optional[str] = None,
        started_at: Optional[str] = None,
        duration_ms: int = 0,
    ) -> Step:
        preview, nbytes = _preview(result if result is not None else (error or ''))
        payload = ToolCallPayload(
            tool_name=tool_name,
            arguments=redact_obj(_parse_tool_args(arguments)),
            result_preview=redact_obj(preview) if isinstance(preview, str) else preview,
            result_bytes=nbytes,
            succeeded=succeeded,
            error=error,
        )
        step = Step.from_payload(
            step_index=len(self.run.steps),
            kind='tool_call',
            payload=payload,
            parent_step_id=parent_step_id if parent_step_id is not None else self.parent_step_id,
            started_at=started_at,
            ended_at=utc_now_iso(),
            duration_ms=duration_ms,
        )
        return self._flush_step(step)

    def record_handoff(
        self,
        *,
        from_agent: str,
        to_agent: str,
        reason: Optional[str] = None,
        parent_step_id: Optional[str] = None,
    ) -> Step:
        payload = HandoffPayload(from_agent=from_agent, to_agent=to_agent, reason=reason)
        step = Step.from_payload(
            step_index=len(self.run.steps),
            kind='handoff',
            payload=payload,
            parent_step_id=parent_step_id if parent_step_id is not None else self.parent_step_id,
            ended_at=utc_now_iso(),
        )
        self._open_step_id = step.step_id
        return self._flush_step(step)

    def record_context_op(self, payload: ContextOpPayload | dict, **kwargs) -> Step:
        if isinstance(payload, dict):
            payload = ContextOpPayload.model_validate(payload)
        step = Step.from_payload(
            step_index=len(self.run.steps),
            kind='context_op',
            payload=payload,
            parent_step_id=kwargs.get('parent_step_id', self.parent_step_id),
            ended_at=utc_now_iso(),
            duration_ms=int(kwargs.get('duration_ms') or 0),
        )
        return self._flush_step(step)

    def record_error(
        self,
        exc: BaseException,
        *,
        recoverable: bool = False,
        parent_step_id: Optional[str] = None,
    ) -> Step:
        payload = ErrorPayload(
            error_type=type(exc).__name__,
            message=str(exc),
            traceback=''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            recoverable=recoverable,
        )
        step = Step.from_payload(
            step_index=len(self.run.steps),
            kind='error',
            payload=payload,
            parent_step_id=parent_step_id if parent_step_id is not None else self.parent_step_id,
            ended_at=utc_now_iso(),
        )
        return self._flush_step(step)

    # ------------------------------------------------------------------ finish

    def finish(
        self,
        *,
        status: str = 'completed',
        termination_reason: Optional[str] = None,
        final_output: Optional[str] = None,
    ) -> Run:
        self.run.status = status  # type: ignore[assignment]
        self.run.termination_reason = termination_reason
        self.run.ended_at = utc_now_iso()
        if final_output is not None:
            self.run.final_output = final_output
        elapsed = time.monotonic() - self._started_mono
        self.run.recompute_totals(wall_clock_seconds=elapsed)
        cost = estimate_cost_usd(
            model=self._model_hint(),
            prompt_tokens=self.run.totals.prompt_tokens,
            completion_tokens=self.run.totals.completion_tokens,
            price_table=self._price_table,
        )
        self.run.totals.estimated_cost_usd = cost
        if self.store is not None:
            self.store.finalize_run(self.run)
        return self.run

    def finish_for_limit(self, reason: str, final_output: Optional[str] = None) -> Run:
        return self.finish(
            status='terminated',
            termination_reason=reason,
            final_output=final_output,
        )


@contextmanager
def trace_run(
    *,
    store: Optional[TraceStore] = None,
    agent_name: str = '',
    agent_class: str = '',
    llm_config: Optional[dict] = None,
    initial_messages: Optional[List[Message]] = None,
    metadata: Optional[dict] = None,
    limits: Optional[RunLimits] = None,
    parent_step_id: Optional[str] = None,
    price_table: Optional[dict] = None,
) -> Iterator[TraceRecorder]:
    """Context manager that binds a recorder to the current context."""
    resolved_store = store if store is not None else default_trace_store()
    # Nested runs: inherit parent nesting anchor when not explicit.
    parent = get_trace_recorder()
    if parent_step_id is None and parent is not None:
        parent_step_id = parent.current_step_id
    recorder = TraceRecorder(
        store=resolved_store,
        agent_name=agent_name,
        agent_class=agent_class,
        llm_config=llm_config,
        initial_messages=initial_messages,
        metadata=metadata,
        limits=limits,
        parent_step_id=parent_step_id,
        price_table=price_table,
    )
    token = _current_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _current_recorder.reset(token)
