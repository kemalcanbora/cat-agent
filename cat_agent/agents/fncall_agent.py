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
from typing import AsyncIterator, Dict, Iterator, List, Literal, Optional, Union

from cat_agent import Agent
from cat_agent.agent import _chain_hard_tool_errors
from cat_agent.llm import BaseChatModel
from cat_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, FUNCTION, Message
from cat_agent.memory import Memory
from cat_agent.settings import MAX_LLM_CALL_PER_RUN
from cat_agent.tools import BaseTool
from cat_agent.tools.base import ToolServiceError
from cat_agent.tools.simple_doc_parser import DocParserError
from cat_agent.utils.utils import extract_files_from_messages


class FnCallAgent(Agent):
    """This is a widely applicable function call agent integrated with llm and tool use ability."""

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 handlers: Optional[List] = None,
                 **kwargs):
        """Initialization the agent.

        Args:
            function_list: One list of tool name, tool configuration or Tool object,
              such as 'code_interpreter',
              {'name': 'code_interpreter', 'timeout': 10},  # tool-owned kernel timer
              {'name': 'web_search', 'attempt_timeout': 15},
              or CodeInterpreter().
              See Agent.__init__ for timeout vs attempt_timeout.
            llm: The LLM model configuration or LLM model object.
              Set the configuration as {'model': '', 'api_key': '', 'model_server': ''}.
            system_message: The specified system message for LLM chat.
            name: The name of this agent.
            description: The description of this agent, which will be used for multi_agent.
            files: A file url list. The initialized files for the agent.
            handlers: Optional observability handlers for run, LLM, and tool events.
        """
        if handlers is None:
            handlers = kwargs.pop('handlers', None)
        rate_limiter = kwargs.pop('rate_limiter', None)
        principal = kwargs.pop('principal', None)
        workspace = kwargs.pop('workspace', None)
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description,
                         handlers=handlers,
                         rate_limiter=rate_limiter,
                         principal=principal,
                         workspace=workspace)

        if not hasattr(self, 'mem'):
            # Default to use Memory to manage files
            if 'qwq' in self.llm.model.lower() or 'qvq' in self.llm.model.lower() or 'qwen3' in self.llm.model.lower():
                mem_llm = None
            else:
                mem_llm = self.llm
            self.mem = Memory(llm=mem_llm, files=files, **kwargs)

        self._function_schemas_cache: Optional[List[Dict]] = None

    @property
    def function_schemas(self) -> List[Dict]:
        if self._function_schemas_cache is None:
            self._function_schemas_cache = [func.function for func in self.function_map.values()]
        return self._function_schemas_cache

    def _init_tool(self, tool: Union[str, Dict, BaseTool]):
        super()._init_tool(tool)
        self._function_schemas_cache = None

    def _run(self, messages: List[Message], lang: Literal['en', 'zh'] = 'en', **kwargs) -> Iterator[List[Message]]:
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response = []
        while True and num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            extra_generate_cfg = {'lang': lang}
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']
            output_stream = self._call_llm(messages=messages,
                                           functions=self.function_schemas,
                                           extra_generate_cfg=extra_generate_cfg)
            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output
            if output:
                response.extend(output)
                messages.extend(output)
                used_any_tool = False
                for out in output:
                    use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                    if use_tool:
                        tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                        fn_msg = Message(role=FUNCTION,
                                         name=tool_name,
                                         content=tool_result,
                                         extra={'function_id': out.extra.get('function_id', '1')})
                        messages.append(fn_msg)
                        response.append(fn_msg)
                        yield response
                        used_any_tool = True
                if not used_any_tool:
                    break
        yield response

    async def _arun(self, messages: List[Message], lang: Literal['en', 'zh'] = 'en', **kwargs) -> AsyncIterator[List[Message]]:
        """Async FnCall loop with concurrent tool execution for a single model turn.

        The async path does not stream tokens; each yield is a complete message list.
        Multiple tool calls in one turn are run via ``asyncio.gather``.
        """
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response: List[Message] = []
        while True and num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            extra_generate_cfg = {'lang': lang}
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']
            output = await self._acall_llm(
                messages=messages,
                functions=self.function_schemas,
                extra_generate_cfg=extra_generate_cfg,
            )
            if output:
                response.extend(output)
                messages.extend(output)
                yield list(response)

                tool_jobs = []
                for out in output:
                    use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                    if use_tool:
                        tool_jobs.append((out, tool_name, tool_args))

                if not tool_jobs:
                    break

                tasks = [
                    asyncio.create_task(
                        self._acall_tool(tool_name, tool_args, messages=messages, **kwargs)
                    )
                    for _, tool_name, tool_args in tool_jobs
                ]
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                except asyncio.CancelledError:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise

                hard_errors: List[BaseException] = []
                for (out, tool_name, _), result in zip(tool_jobs, results):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    if isinstance(result, (ToolServiceError, DocParserError)):
                        hard_errors.append(result)
                        continue
                    if isinstance(result, BaseException):
                        exception_type = type(result).__name__
                        exception_message = str(result)
                        error_message = (
                            f'An error occurred when calling tool `{tool_name}`:\n'
                            f'{exception_type}: {exception_message}'
                        )
                        result = error_message
                    fn_msg = Message(
                        role=FUNCTION,
                        name=tool_name,
                        content=result,
                        extra={'function_id': out.extra.get('function_id', '1') if out.extra else '1'},
                    )
                    messages.append(fn_msg)
                    response.append(fn_msg)
                    yield list(response)

                if hard_errors:
                    raise _chain_hard_tool_errors(hard_errors)
        yield response

    def _call_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs) -> str:
        # Temporary plan: Check if it is necessary to transfer files to the tool
        # Todo: This should be changed to parameter passing, and the file URL should be determined by the model
        if self.function_map[tool_name].file_access:
            assert 'messages' in kwargs
            files = extract_files_from_messages(kwargs['messages'], include_images=True) + self.mem.system_files
            return super()._call_tool(tool_name, tool_args, files=files, **kwargs)
        else:
            return super()._call_tool(tool_name, tool_args, **kwargs)

    async def _acall_tool(self, tool_name: str, tool_args: Union[str, dict] = '{}', **kwargs):
        if self.function_map[tool_name].file_access:
            assert 'messages' in kwargs
            files = extract_files_from_messages(kwargs['messages'], include_images=True) + self.mem.system_files
            return await super()._acall_tool(tool_name, tool_args, files=files, **kwargs)
        return await super()._acall_tool(tool_name, tool_args, **kwargs)
