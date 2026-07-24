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

import asyncio
import copy
import json
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING, Union

from cat_agent.llm import get_chat_model
from cat_agent.llm.base import BaseChatModel
from cat_agent.llm.schema import CONTENT, DEFAULT_SYSTEM_MESSAGE, ROLE, SYSTEM, ContentItem, Message
from cat_agent.log import logger
from cat_agent.observability.context import child_span, get_run_context, run_context
from cat_agent.observability.emitter import emit, resolve_handlers
from cat_agent.observability.events import AgentEvent
from cat_agent.observability.helpers import (
    agent_model_name,
    extract_usage,
    format_tool_args,
    messages_have_tool_call,
    messages_to_payload,
    result_char_count,
    truncate_result_preview,
)
from cat_agent.tools import TOOL_REGISTRY, BaseTool, MCPManager
from cat_agent.tools.base import OPTIONAL_TOOL_REGISTRY, is_tool_allowed_for_agent
from cat_agent.tools.base import ToolExecutionError, ToolNotFoundError, ToolServiceError
from cat_agent.tools.simple_doc_parser import DocParserError
from cat_agent.utils.utils import has_chinese_messages, merge_generate_cfgs

if TYPE_CHECKING:
    from cat_agent.observability.handlers.base import BaseHandler

# One-time warning when sync run() is invoked under a running event loop.
_RUN_IN_LOOP_WARNED = False


def _chain_hard_tool_errors(errors: List[BaseException]) -> BaseException:
    """Raise the earliest-in-order hard error; link the rest via ``__context__``."""
    assert errors
    for i in range(len(errors) - 1):
        errors[i].__context__ = errors[i + 1]
    return errors[0]


class Agent(ABC):
    """A base class for Agent.

    An agent can receive messages and provide response by LLM or Tools.
    Different agents have distinct workflows for processing messages and generating responses in the `_run` method.
    """

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 handlers: Optional[List['BaseHandler']] = None,
                 **kwargs):
        """Initialization the agent.

        Args:
            function_list: One list of tool name, tool configuration or Tool object,
              such as 'code_interpreter',
              {'name': 'code_interpreter', 'timeout': 10},  # tool-owned kernel timer
              {'name': 'web_search', 'attempt_timeout': 15, 'retry': {'max_attempts': 3}},
              or CodeInterpreter().
              ``timeout`` is tool-owned (honored by tools that implement it, e.g.
              code_interpreter). ``attempt_timeout`` is the agent-layer per-attempt
              deadline (async only; sync path warns and does not enforce a wait).
            llm: The LLM model configuration or LLM model object.
              Set the configuration as {'model': '', 'api_key': '', 'model_server': ''}.
            system_message: The specified system message for LLM chat.
            name: The name of this agent.
            description: The description of this agent, which will be used for multi_agent.
            handlers: Optional observability handlers for run, LLM, and tool events.
        """
        if handlers is None:
            handlers = kwargs.pop('handlers', None)
        if isinstance(llm, dict):
            self.llm = get_chat_model(llm)
        else:
            self.llm = llm
        self.extra_generate_cfg: dict = {}

        self.function_map = {}
        if function_list:
            for tool in function_list:
                self._init_tool(tool)

        self.system_message = system_message
        self.name = name
        self.description = description
        self._handlers = handlers or []
        self._async_closed = False
        self._async_inflight = 0

    def run_nonstream(self, messages: List[Union[Dict, Message]], **kwargs) -> Union[List[Message], List[Dict]]:
        """Same as self.run, but with stream=False,
        meaning it returns the complete response directly
        instead of streaming the response incrementally."""
        *_, last_responses = self.run(messages, **kwargs)
        return last_responses

    def run(self, messages: List[Union[Dict, Message]],
            **kwargs) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        """Return one response generator based on the received messages.

        This method performs a uniform type conversion for the inputted messages,
        and calls the _run method to generate a reply.

        Args:
            messages: A list of messages.

        Yields:
            The response generator.
        """
        global _RUN_IN_LOOP_WARNED
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            if not _RUN_IN_LOOP_WARNED:
                warnings.warn(
                    'run() called from a running event loop; this blocks it. Use arun().',
                    RuntimeWarning,
                    stacklevel=2,
                )
                _RUN_IN_LOOP_WARNED = True

        messages = list(messages)
        _return_message_type = 'dict'
        new_messages = []
        # Only return dict when all input messages are dict
        if not messages:
            _return_message_type = 'message'
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'

        if kwargs.get('run_timeout') is not None:
            warnings.warn(
                'run_timeout is not enforceable on the sync run() path; ignoring. Use arun().',
                UserWarning,
                stacklevel=2,
            )
            kwargs.pop('run_timeout', None)

        if 'lang' not in kwargs:
            if has_chinese_messages(new_messages):
                kwargs['lang'] = 'zh'
            else:
                kwargs['lang'] = 'en'

        if self.system_message:
            if not new_messages or new_messages[0][ROLE] != SYSTEM:
                new_messages.insert(0, Message(role=SYSTEM, content=self.system_message))
            else:
                sys_msg = copy.deepcopy(new_messages[0])
                if isinstance(sys_msg[CONTENT], str):
                    sys_msg[CONTENT] = self.system_message + '\n\n' + sys_msg[CONTENT]
                else:
                    assert isinstance(sys_msg[CONTENT], list)
                    assert sys_msg[CONTENT][0].text
                    sys_msg[CONTENT] = [ContentItem(text=self.system_message + '\n\n')] + sys_msg[CONTENT]
                new_messages[0] = sys_msg

        handlers = resolve_handlers(self._handlers, kwargs.get('handlers'))
        emit_stream_chunks = bool(kwargs.get('emit_stream_chunks'))
        if handlers:
            with run_context(
                agent_name=self.name,
                agent_class=type(self).__name__,
                handlers=handlers,
                trace_id=kwargs.get('trace_id'),
                emit_stream_chunks=emit_stream_chunks,
            ) as ctx:
                yield from self._run_with_observability(
                    new_messages=new_messages,
                    return_message_type=_return_message_type,
                    lang=kwargs.get('lang', 'en'),
                    run_kwargs=kwargs,
                    ctx=ctx,
                )
        else:
            yield from self._yield_run_responses(
                self._run(messages=new_messages, **kwargs),
                _return_message_type,
            )

    def _run_with_observability(
        self,
        *,
        new_messages: List[Message],
        return_message_type: str,
        lang: str,
        run_kwargs: dict,
        ctx,
    ) -> Iterator[List[Union[Message, Dict]]]:
        started_at = time.monotonic()
        yield_count = 0
        emit(AgentEvent.run_start(
            trace_id=ctx.trace_id,
            run_id=ctx.run_id,
            span_id=ctx.span_id,
            parent_span_id=ctx.parent_span_id,
            agent_name=ctx.agent_name,
            agent_class=ctx.agent_class,
            message_count=len(new_messages),
            lang=lang,
        ))
        try:
            for rsp in self._yield_run_responses(self._run(messages=new_messages, **run_kwargs), return_message_type):
                yield_count += 1
                yield rsp
            emit(AgentEvent.run_end(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                duration_ms=(time.monotonic() - started_at) * 1000,
                yield_count=yield_count,
            ))
        except Exception as ex:
            emit(AgentEvent.run_error(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                duration_ms=(time.monotonic() - started_at) * 1000,
                error_type=type(ex).__name__,
                error_message=str(ex),
            ))
            raise

    def _yield_run_responses(
        self,
        response_iter: Iterator[List[Message]],
        return_message_type: str,
    ) -> Iterator[List[Union[Message, Dict]]]:
        for rsp in response_iter:
            for i in range(len(rsp)):
                if not rsp[i].name and self.name:
                    rsp[i].name = self.name
            if return_message_type == 'message':
                yield [Message(**x) if isinstance(x, dict) else x for x in rsp]
            else:
                yield [x.model_dump() if not isinstance(x, dict) else x for x in rsp]

    async def arun_nonstream(
        self,
        messages: List[Union[Dict, Message]],
        **kwargs,
    ) -> Union[List[Message], List[Dict]]:
        """Same as :meth:`arun`, but returns only the final collected response.

        The async path does not stream tokens; this method collects and returns
        the last complete message list from :meth:`arun`.
        """
        last_responses: Union[List[Message], List[Dict]] = []
        async for rsp in self.arun(messages, **kwargs):
            last_responses = rsp
        return last_responses

    async def arun(
        self,
        messages: List[Union[Dict, Message]],
        **kwargs,
    ) -> AsyncIterator[Union[List[Message], List[Dict]]]:
        """Async agent entry point.

        The async path does not stream tokens; ``arun`` collects each model turn
        fully and yields complete message lists (not token deltas). Prefer
        :meth:`arun_nonstream` when you only need the final response.

        Cancellation: cancelling the task that awaits ``arun`` cancels waiting on
        in-flight tool gathers. Sync tools running via ``asyncio.to_thread`` cannot
        be aborted mid-call — their worker threads run to completion. MCP tools
        bridged with ``wrap_future`` cancel the cross-loop future when possible.

        Concurrent :meth:`aclose` waits for in-flight ``arun`` calls to finish
        before closing resources. Calling ``arun`` after ``aclose`` raises
        ``RuntimeError``.
        """
        if self._async_closed:
            raise RuntimeError('Agent has been closed via aclose(); cannot arun.')
        self._async_inflight += 1
        try:
            new_messages, return_message_type, kwargs = self._normalize_run_inputs(messages, **kwargs)
            run_timeout = kwargs.pop('run_timeout', None)
            if run_timeout is not None:
                run_timeout = float(run_timeout)
                if run_timeout <= 0:
                    run_timeout = None
            if run_timeout is not None:
                kwargs['_run_deadline'] = time.monotonic() + run_timeout
            handlers = resolve_handlers(self._handlers, kwargs.get('handlers'))
            emit_stream_chunks = bool(kwargs.get('emit_stream_chunks'))
            if handlers:
                with run_context(
                    agent_name=self.name,
                    agent_class=type(self).__name__,
                    handlers=handlers,
                    trace_id=kwargs.get('trace_id'),
                    emit_stream_chunks=emit_stream_chunks,
                ) as ctx:
                    async for rsp in self._arun_with_observability(
                        new_messages=new_messages,
                        return_message_type=return_message_type,
                        lang=kwargs.get('lang', 'en'),
                        run_kwargs=kwargs,
                        ctx=ctx,
                    ):
                        yield rsp
            else:
                async for rsp in self._ayield_run_responses(
                    self._arun(messages=new_messages, **kwargs),
                    return_message_type,
                ):
                    yield rsp
        finally:
            self._async_inflight -= 1

    def _normalize_run_inputs(
        self,
        messages: List[Union[Dict, Message]],
        **kwargs,
    ) -> Tuple[List[Message], str, dict]:
        messages = list(messages)
        _return_message_type = 'dict'
        new_messages = []
        if not messages:
            _return_message_type = 'message'
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'

        if 'lang' not in kwargs:
            if has_chinese_messages(new_messages):
                kwargs['lang'] = 'zh'
            else:
                kwargs['lang'] = 'en'

        if self.system_message:
            if not new_messages or new_messages[0][ROLE] != SYSTEM:
                new_messages.insert(0, Message(role=SYSTEM, content=self.system_message))
            else:
                sys_msg = copy.deepcopy(new_messages[0])
                if isinstance(sys_msg[CONTENT], str):
                    sys_msg[CONTENT] = self.system_message + '\n\n' + sys_msg[CONTENT]
                else:
                    assert isinstance(sys_msg[CONTENT], list)
                    assert sys_msg[CONTENT][0].text
                    sys_msg[CONTENT] = [ContentItem(text=self.system_message + '\n\n')] + sys_msg[CONTENT]
                new_messages[0] = sys_msg
        return new_messages, _return_message_type, kwargs

    async def _arun_with_observability(
        self,
        *,
        new_messages: List[Message],
        return_message_type: str,
        lang: str,
        run_kwargs: dict,
        ctx,
    ) -> AsyncIterator[List[Union[Message, Dict]]]:
        started_at = time.monotonic()
        yield_count = 0
        emit(AgentEvent.run_start(
            trace_id=ctx.trace_id,
            run_id=ctx.run_id,
            span_id=ctx.span_id,
            parent_span_id=ctx.parent_span_id,
            agent_name=ctx.agent_name,
            agent_class=ctx.agent_class,
            message_count=len(new_messages),
            lang=lang,
        ))
        try:
            async for rsp in self._ayield_run_responses(
                self._arun(messages=new_messages, **run_kwargs),
                return_message_type,
            ):
                yield_count += 1
                yield rsp
            emit(AgentEvent.run_end(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                duration_ms=(time.monotonic() - started_at) * 1000,
                yield_count=yield_count,
            ))
        except Exception as ex:
            emit(AgentEvent.run_error(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=ctx.span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                duration_ms=(time.monotonic() - started_at) * 1000,
                error_type=type(ex).__name__,
                error_message=str(ex),
            ))
            raise

    async def _ayield_run_responses(
        self,
        response_iter: AsyncIterator[List[Message]],
        return_message_type: str,
    ) -> AsyncIterator[List[Union[Message, Dict]]]:
        async for rsp in response_iter:
            for i in range(len(rsp)):
                if not rsp[i].name and self.name:
                    rsp[i].name = self.name
            if return_message_type == 'message':
                yield [Message(**x) if isinstance(x, dict) else x for x in rsp]
            else:
                yield [x.model_dump() if not isinstance(x, dict) else x for x in rsp]

    async def aclose(self) -> None:
        """Close async resources (e.g. reused ``AsyncOpenAI`` client).

        Waits for any in-flight :meth:`arun` calls to finish, then closes the LLM
        async client if present. Concurrent ``aclose`` during ``arun`` is defined:
        ``aclose`` waits; a later ``arun`` raises ``RuntimeError``.
        """
        while self._async_inflight > 0:
            await asyncio.sleep(0.01)
        self._async_closed = True
        aclose_fn = getattr(self.llm, 'aclose', None)
        if aclose_fn is not None:
            result = aclose_fn()
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result

    async def __aenter__(self) -> 'Agent':
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    @abstractmethod
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        """Return one response generator based on the received messages.

        The workflow for an agent to generate a reply.
        Each agent subclass needs to implement this method.

        Args:
            messages: A list of messages.
            lang: Language, which will be used to select the language of the prompt
              during the agent's execution process.

        Yields:
            The response generator.
        """
        raise NotImplementedError

    async def _arun(self, messages: List[Message], lang: str = 'en', **kwargs) -> AsyncIterator[List[Message]]:
        """Async workflow. Default collects the sync :meth:`_run` iterator off-thread.

        Subclasses that need concurrent tool execution should override this.
        The async path does not stream tokens; each yield is a complete message list.
        """
        def _collect() -> List[List[Message]]:
            return list(self._run(messages=messages, lang=lang, **kwargs))

        chunks = await asyncio.to_thread(_collect)
        for chunk in chunks:
            yield chunk

    def _audit_context(self) -> dict:
        ctx = get_run_context()
        if ctx is None:
            return {
                'trace_id': None,
                'run_id': None,
                'agent_name': self.name,
                'agent_class': type(self).__name__,
            }
        return {
            'trace_id': ctx.trace_id,
            'run_id': ctx.run_id,
            'agent_name': ctx.agent_name,
            'agent_class': ctx.agent_class,
        }

    def _call_llm(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        extra_generate_cfg: Optional[dict] = None,
    ) -> Iterator[List[Message]]:
        """The interface of calling LLM for the agent.

        We prepend the system_message of this agent to the messages, and call LLM.

        Args:
            messages: A list of messages.
            functions: The list of functions provided to LLM.
            stream: LLM streaming output or non-streaming output.
              For consistency, we default to using streaming output across all agents.

        Yields:
            The response generator of LLM.
        """
        from cat_agent.security.audit import append_audit_record, is_audit_enabled
        from cat_agent.security.pii import maybe_redact_messages_for_prompt

        ctx = get_run_context()
        messages_for_llm = maybe_redact_messages_for_prompt(messages)
        audit_meta = self._audit_context()
        extra_cfg = merge_generate_cfgs(
            base_generate_cfg=self.extra_generate_cfg,
            new_generate_cfg=extra_generate_cfg,
        )

        if is_audit_enabled():
            append_audit_record(
                'audit.prompt',
                {'messages': messages_to_payload(messages_for_llm)},
                **audit_meta,
            )

        if ctx is None or not ctx.handlers:
            final_output: List[Message] = []
            for output in self.llm.chat(
                messages=messages_for_llm,
                functions=functions,
                stream=stream,
                extra_generate_cfg=extra_cfg,
            ):
                if output:
                    final_output = output
                yield output
            if is_audit_enabled() and final_output:
                append_audit_record(
                    'audit.model_output',
                    {'messages': messages_to_payload(final_output)},
                    **audit_meta,
                )
            return

        model = agent_model_name(self.llm)
        tool_count = len(functions or [])
        with child_span() as span_id:
            emit(AgentEvent.llm_start(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                model=model,
                message_count=len(messages_for_llm),
                tool_count=tool_count,
            ))
            started_at = time.monotonic()
            chunk_count = 0
            final_output = []
            for output in self.llm.chat(
                messages=messages_for_llm,
                functions=functions,
                stream=stream,
                extra_generate_cfg=extra_cfg,
            ):
                chunk_count += 1
                if output:
                    final_output = output
                if ctx.emit_stream_chunks:
                    emit(AgentEvent.llm_chunk(
                        trace_id=ctx.trace_id,
                        run_id=ctx.run_id,
                        span_id=span_id,
                        parent_span_id=ctx.parent_span_id,
                        agent_name=ctx.agent_name,
                        agent_class=ctx.agent_class,
                        chunk_index=chunk_count,
                        message_count=len(output or []),
                    ))
                yield output
            emit(AgentEvent.llm_end(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                duration_ms=(time.monotonic() - started_at) * 1000,
                model=model,
                has_tool_call=messages_have_tool_call(final_output),
                usage=extract_usage(final_output),
                chunk_count=chunk_count,
            ))
            if is_audit_enabled() and final_output:
                append_audit_record(
                    'audit.model_output',
                    {'messages': messages_to_payload(final_output)},
                    **audit_meta,
                )

    async def _acall_llm(
        self,
        messages: List[Message],
        functions: Optional[List[Dict]] = None,
        extra_generate_cfg: Optional[dict] = None,
    ) -> List[Message]:
        """Async LLM call. Does not stream tokens; collects and returns the full message list."""
        from cat_agent.security.audit import append_audit_record, is_audit_enabled
        from cat_agent.security.pii import maybe_redact_messages_for_prompt

        ctx = get_run_context()
        messages_for_llm = maybe_redact_messages_for_prompt(messages)
        audit_meta = self._audit_context()
        extra_cfg = merge_generate_cfgs(
            base_generate_cfg=self.extra_generate_cfg,
            new_generate_cfg=extra_generate_cfg,
        )

        if is_audit_enabled():
            append_audit_record(
                'audit.prompt',
                {'messages': messages_to_payload(messages_for_llm)},
                **audit_meta,
            )

        async def _invoke() -> List[Message]:
            achat = getattr(self.llm, 'achat', None)
            if achat is not None:
                result = await achat(
                    messages=messages_for_llm,
                    functions=functions,
                    extra_generate_cfg=extra_cfg,
                )
                return list(result) if result else []

            def _collect() -> List[Message]:
                final: List[Message] = []
                for output in self.llm.chat(
                    messages=messages_for_llm,
                    functions=functions,
                    stream=True,
                    extra_generate_cfg=extra_cfg,
                ):
                    if output:
                        final = output
                return final

            return await asyncio.to_thread(_collect)

        if ctx is None or not ctx.handlers:
            final_output = await _invoke()
            if is_audit_enabled() and final_output:
                append_audit_record(
                    'audit.model_output',
                    {'messages': messages_to_payload(final_output)},
                    **audit_meta,
                )
            return final_output

        model = agent_model_name(self.llm)
        tool_count = len(functions or [])
        with child_span() as span_id:
            emit(AgentEvent.llm_start(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                model=model,
                message_count=len(messages_for_llm),
                tool_count=tool_count,
            ))
            started_at = time.monotonic()
            final_output = await _invoke()
            emit(AgentEvent.llm_end(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                duration_ms=(time.monotonic() - started_at) * 1000,
                model=model,
                has_tool_call=messages_have_tool_call(final_output),
                usage=extract_usage(final_output),
                chunk_count=1,
            ))
            if is_audit_enabled() and final_output:
                append_audit_record(
                    'audit.model_output',
                    {'messages': messages_to_payload(final_output)},
                    **audit_meta,
                )
            return final_output

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> Union[str, List[ContentItem]]:
        """The interface of calling tools for the agent.

        Args:
            tool_name: The name of one tool.
            tool_args: Model generated or user given tool parameters.

        Returns:
            The output of tools.
        """
        from cat_agent.security.audit import append_audit_record, is_audit_enabled

        ctx = get_run_context()
        audit_meta = self._audit_context()
        if is_audit_enabled():
            append_audit_record(
                'audit.tool_call',
                {
                    'tool_name': tool_name,
                    'tool_args': format_tool_args(tool_args, ctx),
                },
                **audit_meta,
            )

        if ctx is None or not ctx.handlers:
            try:
                tool_result, _attempts = self._execute_tool(tool_name, tool_args, **kwargs)
            except (ToolNotFoundError, ToolExecutionError) as ex:
                if is_audit_enabled():
                    append_audit_record(
                        'audit.tool_result',
                        {
                            'tool_name': tool_name,
                            'success': False,
                            'result': str(ex.message or ex),
                        },
                        **audit_meta,
                    )
                return ex.message or str(ex)
            if is_audit_enabled():
                append_audit_record(
                    'audit.tool_result',
                    {
                        'tool_name': tool_name,
                        'success': True,
                        'result': truncate_result_preview(tool_result, ctx),
                    },
                    **audit_meta,
                )
            return tool_result

        with child_span() as span_id:
            emit(AgentEvent.tool_start(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                tool_name=tool_name,
                tool_args=format_tool_args(tool_args, ctx),
            ))
            started_at = time.monotonic()
            attempts = 1
            try:
                tool_result, attempts = self._execute_tool(
                    tool_name, tool_args, _obs_span_id=span_id, **kwargs)
            except (ToolNotFoundError, ToolExecutionError) as ex:
                attempts = getattr(ex, 'attempts', attempts)
                emit(AgentEvent.tool_error(
                    trace_id=ctx.trace_id,
                    run_id=ctx.run_id,
                    span_id=span_id,
                    parent_span_id=ctx.parent_span_id,
                    agent_name=ctx.agent_name,
                    agent_class=ctx.agent_class,
                    tool_name=tool_name,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                    error_type=type(ex).__name__,
                    error_message=str(ex.message or ex),
                    attempts=attempts,
                ))
                emit(AgentEvent.tool_end(
                    trace_id=ctx.trace_id,
                    run_id=ctx.run_id,
                    span_id=span_id,
                    parent_span_id=ctx.parent_span_id,
                    agent_name=ctx.agent_name,
                    agent_class=ctx.agent_class,
                    tool_name=tool_name,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                    success=False,
                    result_chars=len(str(ex.message or ex)),
                    attempts=attempts,
                ))
                if is_audit_enabled():
                    append_audit_record(
                        'audit.tool_result',
                        {
                            'tool_name': tool_name,
                            'success': False,
                            'result': str(ex.message or ex),
                        },
                        **audit_meta,
                    )
                return ex.message or str(ex)
            except (ToolServiceError, DocParserError) as ex:
                emit(AgentEvent.tool_error(
                    trace_id=ctx.trace_id,
                    run_id=ctx.run_id,
                    span_id=span_id,
                    parent_span_id=ctx.parent_span_id,
                    agent_name=ctx.agent_name,
                    agent_class=ctx.agent_class,
                    tool_name=tool_name,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                    error_type=type(ex).__name__,
                    error_message=str(ex),
                    attempts=getattr(ex, 'attempts', attempts),
                ))
                raise
            emit(AgentEvent.tool_end(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                tool_name=tool_name,
                duration_ms=(time.monotonic() - started_at) * 1000,
                success=True,
                result_chars=result_char_count(tool_result, ctx),
                attempts=attempts,
            ))
            if is_audit_enabled():
                append_audit_record(
                    'audit.tool_result',
                    {
                        'tool_name': tool_name,
                        'success': True,
                        'result': truncate_result_preview(tool_result, ctx),
                    },
                    **audit_meta,
                )
            return tool_result

    def _normalize_tool_result(
        self,
        tool_result: Union[str, list, dict, List[ContentItem]],
    ) -> Union[str, List[ContentItem]]:
        if isinstance(tool_result, str):
            return tool_result
        elif isinstance(tool_result, list) and all(isinstance(item, ContentItem) for item in tool_result):
            return tool_result  # multimodal tool results
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

    def _emit_tool_retry(
        self,
        *,
        tool_name: str,
        attempt: int,
        max_attempts: int,
        exc: BaseException,
        delay_seconds: float,
        span_id: Optional[str],
    ) -> None:
        ctx = get_run_context()
        if ctx is None or not ctx.handlers:
            return
        emit(AgentEvent.tool_retry(
            trace_id=ctx.trace_id,
            run_id=ctx.run_id,
            span_id=span_id or ctx.span_id,
            parent_span_id=ctx.parent_span_id,
            agent_name=ctx.agent_name,
            agent_class=ctx.agent_class,
            tool_name=tool_name,
            attempt=attempt,
            max_attempts=max_attempts,
            error_type=type(exc).__name__,
            error_message=str(getattr(exc, 'message', None) or exc),
            delay_seconds=delay_seconds,
        ))

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: Union[str, dict],
        **kwargs,
    ) -> Tuple[Union[str, List[ContentItem]], int]:
        obs_span_id = kwargs.pop('_obs_span_id', None)
        if tool_name not in self.function_map:
            raise ToolNotFoundError(tool_name)
        tool = self.function_map[tool_name]
        from cat_agent.tools.retry import retry_config_for_tool
        from cat_agent.tools.timeout import (
            attempt_timeout_for_tool,
            prepare_tool_call_kwargs,
            warn_sync_attempt_timeout,
        )
        from cat_agent.utils.backoff import compute_backoff_delay

        retry_cfg = retry_config_for_tool(tool)
        max_attempts = retry_cfg.max_attempts if retry_cfg else 1
        delay = retry_cfg.initial_delay if retry_cfg else 1.0
        attempt_timeout = attempt_timeout_for_tool(tool)
        if attempt_timeout is not None:
            warn_sync_attempt_timeout(tool_name, attempt_timeout)
        call_kwargs = prepare_tool_call_kwargs(tool, kwargs, attempt_timeout)
        last_exc: Optional[BaseException] = None

        for attempt in range(1, max_attempts + 1):
            try:
                tool_result = tool.call(tool_args, **call_kwargs)
            except (ToolServiceError, DocParserError) as ex:
                last_exc = ex
                if retry_cfg and attempt < max_attempts and retry_cfg.is_retryable(ex):
                    self._emit_tool_retry(
                        tool_name=tool_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        exc=ex,
                        delay_seconds=delay,
                        span_id=obs_span_id,
                    )
                    time.sleep(delay)
                    delay = compute_backoff_delay(
                        delay,
                        exponential_base=retry_cfg.exponential_base,
                        max_delay=retry_cfg.max_delay,
                    )
                    continue
                if isinstance(ex, ToolExecutionError):
                    ex.attempts = attempt  # type: ignore[attr-defined]
                raise
            except Exception as ex:
                exception_type = type(ex).__name__
                exception_message = str(ex)
                traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
                error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                                f'{exception_type}: {exception_message}\n' \
                                f'Traceback:\n{traceback_info}'
                logger.warning(error_message)
                wrapped = ToolExecutionError(tool_name, error_message)
                wrapped.__cause__ = ex
                last_exc = wrapped
                if retry_cfg and attempt < max_attempts and retry_cfg.is_retryable(wrapped):
                    self._emit_tool_retry(
                        tool_name=tool_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        exc=wrapped,
                        delay_seconds=delay,
                        span_id=obs_span_id,
                    )
                    time.sleep(delay)
                    delay = compute_backoff_delay(
                        delay,
                        exponential_base=retry_cfg.exponential_base,
                        max_delay=retry_cfg.max_delay,
                    )
                    continue
                wrapped.attempts = attempt  # type: ignore[attr-defined]
                raise wrapped from ex

            return self._normalize_tool_result(tool_result), attempt

        assert last_exc is not None
        raise last_exc

    async def _acall_tool(
        self,
        tool_name: str,
        tool_args: Union[str, dict] = '{}',
        **kwargs,
    ) -> Union[str, List[ContentItem]]:
        """Async tool call with the same error semantics as :meth:`_call_tool`."""
        from cat_agent.security.audit import append_audit_record, is_audit_enabled

        ctx = get_run_context()
        audit_meta = self._audit_context()
        if is_audit_enabled():
            append_audit_record(
                'audit.tool_call',
                {
                    'tool_name': tool_name,
                    'tool_args': format_tool_args(tool_args, ctx),
                },
                **audit_meta,
            )

        if ctx is None or not ctx.handlers:
            try:
                tool_result, _attempts = await self._aexecute_tool(tool_name, tool_args, **kwargs)
            except (ToolNotFoundError, ToolExecutionError) as ex:
                if is_audit_enabled():
                    append_audit_record(
                        'audit.tool_result',
                        {
                            'tool_name': tool_name,
                            'success': False,
                            'result': str(ex.message or ex),
                        },
                        **audit_meta,
                    )
                return ex.message or str(ex)
            if is_audit_enabled():
                append_audit_record(
                    'audit.tool_result',
                    {
                        'tool_name': tool_name,
                        'success': True,
                        'result': truncate_result_preview(tool_result, ctx),
                    },
                    **audit_meta,
                )
            return tool_result

        with child_span() as span_id:
            emit(AgentEvent.tool_start(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                tool_name=tool_name,
                tool_args=format_tool_args(tool_args, ctx),
            ))
            started_at = time.monotonic()
            attempts = 1
            try:
                tool_result, attempts = await self._aexecute_tool(
                    tool_name, tool_args, _obs_span_id=span_id, **kwargs)
            except (ToolNotFoundError, ToolExecutionError) as ex:
                attempts = getattr(ex, 'attempts', attempts)
                emit(AgentEvent.tool_error(
                    trace_id=ctx.trace_id,
                    run_id=ctx.run_id,
                    span_id=span_id,
                    parent_span_id=ctx.parent_span_id,
                    agent_name=ctx.agent_name,
                    agent_class=ctx.agent_class,
                    tool_name=tool_name,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                    error_type=type(ex).__name__,
                    error_message=str(ex.message or ex),
                    attempts=attempts,
                ))
                emit(AgentEvent.tool_end(
                    trace_id=ctx.trace_id,
                    run_id=ctx.run_id,
                    span_id=span_id,
                    parent_span_id=ctx.parent_span_id,
                    agent_name=ctx.agent_name,
                    agent_class=ctx.agent_class,
                    tool_name=tool_name,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                    success=False,
                    result_chars=len(str(ex.message or ex)),
                    attempts=attempts,
                ))
                if is_audit_enabled():
                    append_audit_record(
                        'audit.tool_result',
                        {
                            'tool_name': tool_name,
                            'success': False,
                            'result': str(ex.message or ex),
                        },
                        **audit_meta,
                    )
                return ex.message or str(ex)
            except (ToolServiceError, DocParserError) as ex:
                emit(AgentEvent.tool_error(
                    trace_id=ctx.trace_id,
                    run_id=ctx.run_id,
                    span_id=span_id,
                    parent_span_id=ctx.parent_span_id,
                    agent_name=ctx.agent_name,
                    agent_class=ctx.agent_class,
                    tool_name=tool_name,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                    error_type=type(ex).__name__,
                    error_message=str(ex),
                    attempts=getattr(ex, 'attempts', attempts),
                ))
                raise
            emit(AgentEvent.tool_end(
                trace_id=ctx.trace_id,
                run_id=ctx.run_id,
                span_id=span_id,
                parent_span_id=ctx.parent_span_id,
                agent_name=ctx.agent_name,
                agent_class=ctx.agent_class,
                tool_name=tool_name,
                duration_ms=(time.monotonic() - started_at) * 1000,
                success=True,
                result_chars=result_char_count(tool_result, ctx),
                attempts=attempts,
            ))
            if is_audit_enabled():
                append_audit_record(
                    'audit.tool_result',
                    {
                        'tool_name': tool_name,
                        'success': True,
                        'result': truncate_result_preview(tool_result, ctx),
                    },
                    **audit_meta,
                )
            return tool_result

    async def _aexecute_tool(
        self,
        tool_name: str,
        tool_args: Union[str, dict],
        **kwargs,
    ) -> Tuple[Union[str, List[ContentItem]], int]:
        obs_span_id = kwargs.pop('_obs_span_id', None)
        if tool_name not in self.function_map:
            raise ToolNotFoundError(tool_name)
        tool = self.function_map[tool_name]
        from cat_agent.tools.retry import retry_config_for_tool
        from cat_agent.tools.timeout import (
            attempt_timeout_for_tool,
            format_tool_timeout_error,
            prepare_tool_call_kwargs,
        )
        from cat_agent.utils.backoff import compute_backoff_delay

        retry_cfg = retry_config_for_tool(tool)
        max_attempts = retry_cfg.max_attempts if retry_cfg else 1
        delay = retry_cfg.initial_delay if retry_cfg else 1.0
        attempt_timeout = attempt_timeout_for_tool(tool)
        # Whole-run deadline may further tighten this attempt.
        run_deadline = kwargs.pop('_run_deadline', None)
        call_kwargs = prepare_tool_call_kwargs(tool, kwargs, attempt_timeout)
        last_exc: Optional[BaseException] = None

        for attempt in range(1, max_attempts + 1):
            effective_timeout = attempt_timeout
            if run_deadline is not None:
                remaining = run_deadline - time.monotonic()
                if remaining <= 0:
                    wrapped = ToolExecutionError(
                        tool_name,
                        format_tool_timeout_error(tool_name, attempt_timeout or 0.0),
                    )
                    wrapped.attempts = attempt  # type: ignore[attr-defined]
                    raise wrapped
                if effective_timeout is None:
                    effective_timeout = remaining
                else:
                    effective_timeout = min(effective_timeout, remaining)

            try:
                coro = tool.acall(tool_args, **call_kwargs)
                if effective_timeout is not None:
                    try:
                        tool_result = await asyncio.wait_for(coro, timeout=effective_timeout)
                    except asyncio.TimeoutError as ex:
                        # Abandon the wait. Sync work inside to_thread keeps running but
                        # cannot write into agent message history / observability: those
                        # updates happen only after this method returns.
                        error_message = format_tool_timeout_error(
                            tool_name, float(effective_timeout))
                        logger.warning(error_message)
                        wrapped = ToolExecutionError(tool_name, error_message)
                        wrapped.__cause__ = ex
                        last_exc = wrapped
                        if retry_cfg and attempt < max_attempts and retry_cfg.is_retryable(wrapped):
                            self._emit_tool_retry(
                                tool_name=tool_name,
                                attempt=attempt,
                                max_attempts=max_attempts,
                                exc=wrapped,
                                delay_seconds=delay,
                                span_id=obs_span_id,
                            )
                            await asyncio.sleep(delay)
                            delay = compute_backoff_delay(
                                delay,
                                exponential_base=retry_cfg.exponential_base,
                                max_delay=retry_cfg.max_delay,
                            )
                            continue
                        wrapped.attempts = attempt  # type: ignore[attr-defined]
                        raise wrapped from ex
                else:
                    tool_result = await coro
            except asyncio.CancelledError:
                raise
            except (ToolServiceError, DocParserError) as ex:
                last_exc = ex
                if retry_cfg and attempt < max_attempts and retry_cfg.is_retryable(ex):
                    self._emit_tool_retry(
                        tool_name=tool_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        exc=ex,
                        delay_seconds=delay,
                        span_id=obs_span_id,
                    )
                    await asyncio.sleep(delay)
                    delay = compute_backoff_delay(
                        delay,
                        exponential_base=retry_cfg.exponential_base,
                        max_delay=retry_cfg.max_delay,
                    )
                    continue
                if isinstance(ex, ToolExecutionError):
                    ex.attempts = attempt  # type: ignore[attr-defined]
                raise
            except Exception as ex:
                exception_type = type(ex).__name__
                exception_message = str(ex)
                traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
                error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                                f'{exception_type}: {exception_message}\n' \
                                f'Traceback:\n{traceback_info}'
                logger.warning(error_message)
                wrapped = ToolExecutionError(tool_name, error_message)
                wrapped.__cause__ = ex
                last_exc = wrapped
                if retry_cfg and attempt < max_attempts and retry_cfg.is_retryable(wrapped):
                    self._emit_tool_retry(
                        tool_name=tool_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        exc=wrapped,
                        delay_seconds=delay,
                        span_id=obs_span_id,
                    )
                    await asyncio.sleep(delay)
                    delay = compute_backoff_delay(
                        delay,
                        exponential_base=retry_cfg.exponential_base,
                        max_delay=retry_cfg.max_delay,
                    )
                    continue
                wrapped.attempts = attempt  # type: ignore[attr-defined]
                raise wrapped from ex

            return self._normalize_tool_result(tool_result), attempt

        assert last_exc is not None
        raise last_exc

    def _init_tool(self, tool: Union[str, Dict, BaseTool]):
        if isinstance(tool, BaseTool):
            tool_name = tool.name
            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = tool
        elif isinstance(tool, dict) and 'mcpServers' in tool:
            from cat_agent.security.offline import is_offline_mode
            if is_offline_mode():
                raise ValueError(
                    'MCP servers require network access and are disabled when CAT_AGENT_OFFLINE=1.'
                )
            tools = MCPManager().initConfig(tool)
            for tool in tools:
                tool_name = tool.name
                if tool_name in self.function_map:
                    logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
                self.function_map[tool_name] = tool
        else:
            if isinstance(tool, dict):
                tool_name = tool['name']
                tool_cfg = tool
            else:
                tool_name = tool
                tool_cfg = None
            if tool_name not in TOOL_REGISTRY:
                if tool_name in OPTIONAL_TOOL_REGISTRY:
                    raise ValueError(
                        f'Tool {tool_name} is opt-in (network/cloud). '
                        f'Call cat_agent.tools.enable_optional_tools("{tool_name}") before use.'
                    )
                raise ValueError(f'Tool {tool_name} is not registered.')

            tool_cls = TOOL_REGISTRY[tool_name]
            if not is_tool_allowed_for_agent(tool_name, tool_cls):
                from cat_agent.security.offline import is_offline_mode
                if is_offline_mode():
                    logger.warning('Skipping tool {} because CAT_AGENT_OFFLINE=1', tool_name)
                    return
                raise ValueError(
                    f'Tool {tool_name} requires network access and is disabled in offline mode.'
                )

            if tool_name in self.function_map:
                logger.warning(f'Repeatedly adding tool {tool_name}, will use the newest tool in function list')
            self.function_map[tool_name] = TOOL_REGISTRY[tool_name](tool_cfg)

    def _detect_tool(self, message: Message) -> Tuple[bool, str, str, str]:
        """A built-in tool call detection for func_call format message.

        Args:
            message: one message generated by LLM.

        Returns:
            Need to call tool or not, tool name, tool args, text replies.
        """
        func_name = None
        func_args = None

        if message.function_call:
            func_call = message.function_call
            func_name = func_call.name
            func_args = func_call.arguments
        text = message.content
        if not text:
            text = ''

        return (func_name is not None), func_name, func_args, text


# The most basic form of an agent is just a LLM, not augmented with any tool or workflow.
class BasicAgent(Agent):

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        extra_generate_cfg = {'lang': lang}
        if kwargs.get('seed') is not None:
            extra_generate_cfg['seed'] = kwargs['seed']
        return self._call_llm(messages, extra_generate_cfg=extra_generate_cfg)
