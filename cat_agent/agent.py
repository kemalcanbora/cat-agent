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

import copy
import json
import time
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING, Union

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
              such as 'code_interpreter', {'name': 'code_interpreter', 'timeout': 10}, or CodeInterpreter().
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
                tool_result = self._execute_tool(tool_name, tool_args, **kwargs)
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
            try:
                tool_result = self._execute_tool(tool_name, tool_args, **kwargs)
            except (ToolNotFoundError, ToolExecutionError) as ex:
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

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: Union[str, dict],
        **kwargs,
    ) -> Union[str, List[ContentItem]]:
        if tool_name not in self.function_map:
            raise ToolNotFoundError(tool_name)
        tool = self.function_map[tool_name]
        try:
            tool_result = tool.call(tool_args, **kwargs)
        except (ToolServiceError, DocParserError) as ex:
            raise ex
        except Exception as ex:
            exception_type = type(ex).__name__
            exception_message = str(ex)
            traceback_info = ''.join(traceback.format_tb(ex.__traceback__))
            error_message = f'An error occurred when calling tool `{tool_name}`:\n' \
                            f'{exception_type}: {exception_message}\n' \
                            f'Traceback:\n{traceback_info}'
            logger.warning(error_message)
            raise ToolExecutionError(tool_name, error_message) from ex

        if isinstance(tool_result, str):
            return tool_result
        elif isinstance(tool_result, list) and all(isinstance(item, ContentItem) for item in tool_result):
            return tool_result  # multimodal tool results
        else:
            return json.dumps(tool_result, ensure_ascii=False, indent=4)

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
