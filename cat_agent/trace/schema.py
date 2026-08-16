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

"""Structured execution-trace schema (version 1.0).

Field names align with OpenTelemetry GenAI semantic conventions where a direct
equivalent exists (``gen_ai.request.model``, ``gen_ai.usage.input_tokens``, …)
so an OTel exporter can be added later without a schema migration. This module
does not depend on OpenTelemetry.

See Yehudai et al. (2025/2026) arXiv:2503.16416 and
https://opentelemetry.io/docs/specs/semconv/gen-ai/
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from cat_agent.llm.schema import Message

SCHEMA_VERSION = '1.0'


def message_dump_with_id(msg: Message) -> Dict[str, Any]:
    """Serialize a Message for traces, re-injecting ``id``.

    ``Message.id`` is ``exclude=True`` so public dump↔Message round-trips stay
    byte-identical, but eviction correlation (MAST 1.4) needs stable ids in
    persisted LLM-call payloads and ``initial_messages``.
    """
    data = msg.model_dump(mode='json')
    data['id'] = msg.id
    return data


def ensure_message_ids_in_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Re-inject Message.id into an llm_call payload dict (in place + return)."""
    msgs = payload.get('messages_in')
    if isinstance(msgs, list):
        payload['messages_in'] = [
            message_dump_with_id(m) if isinstance(m, Message)
            else m
            for m in msgs
        ]
    out = payload.get('message_out')
    if isinstance(out, Message):
        payload['message_out'] = message_dump_with_id(out)
    return payload

RunStatus = Literal['running', 'completed', 'failed', 'terminated']
StepKind = Literal['llm_call', 'tool_call', 'handoff', 'context_op', 'user_input', 'error']
ContextOpName = Literal['mask', 'compact', 'fold', 'evict']
TerminationReason = Literal[
    'goal_reached',
    'max_steps',
    'max_tokens',
    'wall_clock',
    'max_tool_calls',
    'error',
    'user_cancelled',
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def new_id() -> str:
    return str(uuid.uuid4())


class RunTotals(BaseModel):
    model_config = ConfigDict(extra='forbid')

    steps: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Optional[float] = None
    wall_clock_seconds: float = 0.0
    tool_calls: int = 0
    errors: int = 0
    tokens_estimated: bool = False


class LLMCallPayload(BaseModel):
    """Payload for an LLM call step.

    OTel-compatible aliases are included in :meth:`model_dump_otel`.
    """

    model_config = ConfigDict(extra='allow')

    model: Optional[str] = None
    model_type: Optional[str] = None
    gen_ai_system: Optional[str] = Field(
        default=None,
        description='OTel gen_ai.system (e.g. openai, anthropic, llama.cpp)',
    )
    messages_in: List[Message] = Field(default_factory=list)
    message_out: Optional[Message] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    finish_reason: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    sampling_params: Dict[str, Any] = Field(default_factory=dict)

    @field_serializer('messages_in')
    def _ser_messages_in(self, messages: List[Message]) -> List[Dict[str, Any]]:
        return [message_dump_with_id(m) for m in messages]

    @field_serializer('message_out')
    def _ser_message_out(self, message: Optional[Message]) -> Optional[Dict[str, Any]]:
        return message_dump_with_id(message) if message is not None else None

    def model_dump_otel(self, **kwargs) -> Dict[str, Any]:
        data = self.model_dump(**kwargs)
        if self.model is not None:
            data['gen_ai.request.model'] = self.model
        if self.gen_ai_system is not None:
            data['gen_ai.system'] = self.gen_ai_system
        data['gen_ai.usage.input_tokens'] = self.prompt_tokens
        data['gen_ai.usage.output_tokens'] = self.completion_tokens
        return data


class ToolCallPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result_preview: str = ''
    result_bytes: int = 0
    succeeded: bool = True
    error: Optional[str] = None


class HandoffPayload(BaseModel):
    model_config = ConfigDict(extra='forbid')

    from_agent: str
    to_agent: str
    reason: Optional[str] = None


class ContextOpPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    operation: ContextOpName
    messages_before: int
    messages_after: int
    tokens_before: int
    tokens_after: int
    strategy_name: str
    evicted_message_ids: List[str] = Field(default_factory=list)


class ErrorPayload(BaseModel):
    model_config = ConfigDict(extra='allow')

    error_type: str
    message: str
    traceback: str = ''
    recoverable: bool = False


StepPayload = Union[
    LLMCallPayload,
    ToolCallPayload,
    HandoffPayload,
    ContextOpPayload,
    ErrorPayload,
    Dict[str, Any],
]


class Step(BaseModel):
    model_config = ConfigDict(extra='allow')

    step_index: int
    step_id: str = Field(default_factory=new_id)
    parent_step_id: Optional[str] = None
    kind: StepKind
    started_at: str = Field(default_factory=utc_now_iso)
    ended_at: Optional[str] = None
    duration_ms: int = 0
    payload: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        *,
        step_index: int,
        kind: StepKind,
        payload: BaseModel | Dict[str, Any],
        parent_step_id: Optional[str] = None,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        duration_ms: int = 0,
        step_id: Optional[str] = None,
    ) -> 'Step':
        if isinstance(payload, LLMCallPayload):
            body = payload.model_dump_otel(mode='json')
        elif isinstance(payload, BaseModel):
            body = payload.model_dump(mode='json')
        else:
            body = dict(payload)
            if kind == 'llm_call':
                body = ensure_message_ids_in_payload(body)
        return cls(
            step_index=step_index,
            step_id=step_id or new_id(),
            parent_step_id=parent_step_id,
            kind=kind,
            started_at=started_at or utc_now_iso(),
            ended_at=ended_at,
            duration_ms=duration_ms,
            payload=body,
        )


class Run(BaseModel):
    """Top-level execution trace for one agent ``run()`` invocation."""

    model_config = ConfigDict(extra='allow')

    schema_version: str = SCHEMA_VERSION
    run_id: str = Field(default_factory=new_id)
    agent_name: str = ''
    agent_class: str = ''
    started_at: str = Field(default_factory=utc_now_iso)
    ended_at: Optional[str] = None
    status: RunStatus = 'running'
    termination_reason: Optional[str] = None
    llm_config: Dict[str, Any] = Field(default_factory=dict)
    initial_messages: List[Message] = Field(default_factory=list)
    final_output: Optional[str] = None
    totals: RunTotals = Field(default_factory=RunTotals)
    steps: List[Step] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_serializer('initial_messages')
    def _ser_initial_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        return [message_dump_with_id(m) for m in messages]

    def recompute_totals(self, *, wall_clock_seconds: Optional[float] = None) -> None:
        prompt = 0
        completion = 0
        tool_calls = 0
        errors = 0
        tokens_estimated = False
        for step in self.steps:
            if step.kind == 'llm_call':
                prompt += int(step.payload.get('prompt_tokens') or 0)
                completion += int(step.payload.get('completion_tokens') or 0)
                if step.payload.get('tokens_estimated'):
                    tokens_estimated = True
            elif step.kind == 'tool_call':
                tool_calls += 1
                if not step.payload.get('succeeded', True):
                    errors += 1
            elif step.kind == 'error':
                errors += 1
        self.totals = RunTotals(
            steps=len(self.steps),
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            estimated_cost_usd=self.totals.estimated_cost_usd,
            wall_clock_seconds=(
                wall_clock_seconds
                if wall_clock_seconds is not None
                else self.totals.wall_clock_seconds
            ),
            tool_calls=tool_calls,
            errors=errors,
            tokens_estimated=tokens_estimated,
        )


class RunLimits(BaseModel):
    """Hard budgets for a single agent run. ``None`` means unlimited."""

    model_config = ConfigDict(extra='forbid')

    max_steps: Optional[int] = None
    max_total_tokens: Optional[int] = None
    max_wall_clock_seconds: Optional[float] = None
    max_tool_calls: Optional[int] = None


def parse_step_payload(kind: StepKind, payload: Dict[str, Any]) -> StepPayload:
    if kind == 'llm_call':
        return LLMCallPayload.model_validate(payload)
    if kind == 'tool_call':
        return ToolCallPayload.model_validate(payload)
    if kind == 'handoff':
        return HandoffPayload.model_validate(payload)
    if kind == 'context_op':
        return ContextOpPayload.model_validate(payload)
    if kind == 'error':
        return ErrorPayload.model_validate(payload)
    return payload
