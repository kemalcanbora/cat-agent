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

"""Helpers used by Agent to emit traces without coupling to Loguru."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional, Union

from cat_agent.llm.schema import Message
from cat_agent.trace.recorder import get_trace_recorder, is_trace_enabled, trace_run
from cat_agent.trace.schema import RunLimits


def llm_cfg_snapshot(llm: Any) -> dict:
    if llm is None:
        return {}
    cfg = getattr(llm, 'model_cfg', None) or getattr(llm, 'cfg', None)
    if isinstance(cfg, dict):
        return dict(cfg)
    out: dict = {}
    for key in ('model', 'model_type', 'model_server', 'generate_cfg'):
        val = getattr(llm, key, None)
        if val is not None:
            out[key] = val
    return out


def gen_ai_system(model_type: Any) -> Optional[str]:
    if not model_type:
        return None
    mt = str(model_type).lower()
    return {
        'oai': 'openai',
        'openai': 'openai',
        'transformers': 'huggingface',
        'llama_cpp': 'llama.cpp',
        'llama_cpp_vision': 'llama.cpp',
        'mlx_lm': 'mlx',
        'openvino': 'openvino',
    }.get(mt, mt)


def final_output_text(messages: List[Union[Message, Dict]]) -> Optional[str]:
    if not messages:
        return None
    last = messages[-1]
    if isinstance(last, dict):
        content = last.get('content')
    else:
        content = getattr(last, 'content', None)
    if isinstance(content, str):
        return content
    if content is None:
        return None
    return str(content)


def should_trace_run(agent: Any, kwargs: dict) -> bool:
    if kwargs.get('trace') is False:
        return False
    if kwargs.get('trace') is True or kwargs.get('trace_store') is not None:
        return True
    if getattr(agent, 'run_limits', None) is not None:
        return True
    if get_trace_recorder() is not None:
        return True
    return is_trace_enabled()


def wrap_run_with_trace(
    agent: Any,
    *,
    new_messages: List[Message],
    kwargs: dict,
    core: Iterator[List[Union[Message, Dict]]],
) -> Iterator[List[Union[Message, Dict]]]:
    if not should_trace_run(agent, kwargs):
        yield from core
        return

    limits = kwargs.get('run_limits', getattr(agent, 'run_limits', None))
    if limits is not None and not isinstance(limits, RunLimits):
        limits = RunLimits.model_validate(limits)
    store = kwargs.get('trace_store')
    parent = get_trace_recorder()
    parent_step_id = kwargs.get('parent_step_id')
    if parent_step_id is None and parent is not None:
        parent_step_id = parent.current_step_id

    with trace_run(
        store=store,
        agent_name=getattr(agent, 'name', None) or '',
        agent_class=type(agent).__name__,
        llm_config=llm_cfg_snapshot(getattr(agent, 'llm', None)),
        initial_messages=new_messages,
        metadata=dict(kwargs.get('trace_metadata') or {}),
        limits=limits,
        parent_step_id=parent_step_id,
    ) as recorder:
        last_rsp: List[Union[Message, Dict]] = []
        try:
            for rsp in core:
                last_rsp = rsp
                reason = recorder.check_limits()
                if reason:
                    recorder.finish_for_limit(
                        reason,
                        final_output=final_output_text(last_rsp),
                    )
                    return
                yield rsp
            if recorder.run.status == 'running':
                recorder.finish(
                    status='completed',
                    termination_reason='goal_reached',
                    final_output=final_output_text(last_rsp),
                )
        except Exception as ex:
            recorder.record_error(ex, recoverable=False)
            recorder.finish(
                status='failed',
                termination_reason='error',
                final_output=final_output_text(last_rsp),
            )
            raise


async def awrap_run_with_trace(
    agent: Any,
    *,
    new_messages: List[Message],
    kwargs: dict,
    core,
):
    if not should_trace_run(agent, kwargs):
        async for rsp in core:
            yield rsp
        return

    limits = kwargs.get('run_limits', getattr(agent, 'run_limits', None))
    if limits is not None and not isinstance(limits, RunLimits):
        limits = RunLimits.model_validate(limits)
    store = kwargs.get('trace_store')
    parent = get_trace_recorder()
    parent_step_id = kwargs.get('parent_step_id')
    if parent_step_id is None and parent is not None:
        parent_step_id = parent.current_step_id

    with trace_run(
        store=store,
        agent_name=getattr(agent, 'name', None) or '',
        agent_class=type(agent).__name__,
        llm_config=llm_cfg_snapshot(getattr(agent, 'llm', None)),
        initial_messages=new_messages,
        metadata=dict(kwargs.get('trace_metadata') or {}),
        limits=limits,
        parent_step_id=parent_step_id,
    ) as recorder:
        last_rsp: List[Union[Message, Dict]] = []
        try:
            async for rsp in core:
                last_rsp = rsp
                reason = recorder.check_limits()
                if reason:
                    recorder.finish_for_limit(
                        reason,
                        final_output=final_output_text(last_rsp),
                    )
                    return
                yield rsp
            if recorder.run.status == 'running':
                recorder.finish(
                    status='completed',
                    termination_reason='goal_reached',
                    final_output=final_output_text(last_rsp),
                )
        except Exception as ex:
            recorder.record_error(ex, recoverable=False)
            recorder.finish(
                status='failed',
                termination_reason='error',
                final_output=final_output_text(last_rsp),
            )
            raise


def record_llm_call(
    agent: Any,
    *,
    messages_for_llm: List[Message],
    final_output: List[Message],
    started_at: float,
    extra_cfg: Optional[dict] = None,
) -> None:
    recorder = get_trace_recorder()
    if recorder is None:
        return
    from cat_agent.observability.helpers import agent_model_name

    model = agent_model_name(getattr(agent, 'llm', None))
    llm = getattr(agent, 'llm', None)
    model_type = getattr(llm, 'model_type', None) or (
        (getattr(llm, 'model_cfg', None) or {}).get('model_type')
    )
    gen_cfg = extra_cfg or {}
    recorder.record_llm_call(
        model=model,
        model_type=model_type if isinstance(model_type, str) else None,
        gen_ai_system=gen_ai_system(model_type),
        messages_in=messages_for_llm,
        messages_out=final_output,
        temperature=gen_cfg.get('temperature'),
        top_p=gen_cfg.get('top_p'),
        max_tokens=gen_cfg.get('max_tokens') or gen_cfg.get('max_new_tokens'),
        sampling_params={
            k: v for k, v in gen_cfg.items()
            if k in ('temperature', 'top_p', 'top_k', 'seed', 'stop')
        },
        llm=llm,
        duration_ms=int((time.monotonic() - started_at) * 1000),
    )


def record_tool_call(
    *,
    tool_name: str,
    tool_args: Union[str, dict],
    result: Any = None,
    succeeded: bool = True,
    error: Optional[str] = None,
    started_at: float,
) -> None:
    recorder = get_trace_recorder()
    if recorder is None:
        return
    recorder.record_tool_call(
        tool_name=tool_name,
        arguments=tool_args,
        result=result,
        succeeded=succeeded,
        error=error,
        duration_ms=int((time.monotonic() - started_at) * 1000),
    )


def apply_context_manager(agent: Any, messages: List[Message]) -> List[Message]:
    mgr = getattr(agent, 'context_manager', None)
    if mgr is False:
        return messages
    if mgr is None:
        try:
            from cat_agent.context import get_default_context_manager
            mgr = get_default_context_manager(agent)
        except Exception:
            return messages
    if mgr is None or mgr is False:
        return messages
    result = mgr.prepare(messages, llm=getattr(agent, 'llm', None))
    recorder = get_trace_recorder()
    if recorder is not None:
        for op in getattr(result, 'operations', None) or []:
            recorder.record_context_op(op)
    return getattr(result, 'messages', messages)


def check_run_limit_stop() -> Optional[str]:
    """Return termination reason if the active recorder hit a budget."""
    recorder = get_trace_recorder()
    if recorder is None:
        return None
    return recorder.check_limits()
