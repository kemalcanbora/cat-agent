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
import os
import threading
from pprint import pformat
from typing import Dict, Iterator, List, Optional

import openai

from cat_agent.utils.utils import format_as_text_message

if openai.__version__.startswith('0.'):
    from openai.error import OpenAIError  # noqa
    BadRequestError = OpenAIError  # type: ignore[misc, assignment]
else:
    from openai import OpenAIError
    try:
        from openai import BadRequestError
    except ImportError:  # pragma: no cover
        BadRequestError = OpenAIError  # type: ignore[misc, assignment]

from cat_agent.llm.base import ModelServiceError, register_llm
from cat_agent.llm.function_calling import BaseFnCallModel
from cat_agent.llm.schema import ASSISTANT, FunctionCall, Message, ToolCall, generate_tool_call_id
from cat_agent.llm.tool_call_stream import ToolCallStreamMerger
from cat_agent.log import logger


def _merge_usage(msg: Message, usage) -> None:
    if usage is None:
        return
    if isinstance(usage, dict):
        prompt = usage.get('prompt_tokens', 0) or 0
        completion = usage.get('completion_tokens', 0) or 0
        total = usage.get('total_tokens', 0) or 0
    else:
        prompt = getattr(usage, 'prompt_tokens', 0) or 0
        completion = getattr(usage, 'completion_tokens', 0) or 0
        total = getattr(usage, 'total_tokens', 0) or 0
    extra = dict(msg.extra or {})
    extra['usage'] = {
        'prompt_tokens': prompt,
        'completion_tokens': completion,
        'total_tokens': total,
    }
    msg.extra = extra


def _messages_from_completion_message(msg) -> List[Message]:
    """Map a non-streaming OpenAI chat message to internal Message list.

    Tool calls become a **single** assistant message with ``tool_calls`` populated.
    """
    out: List[Message] = []
    reasoning = getattr(msg, 'reasoning_content', None)
    content = getattr(msg, 'content', None) or ''
    raw_tool_calls = getattr(msg, 'tool_calls', None) or []

    if reasoning:
        out.append(Message(role=ASSISTANT, content='', reasoning_content=reasoning))

    tool_calls: List[ToolCall] = []
    for i, tc in enumerate(raw_tool_calls):
        fn = getattr(tc, 'function', None)
        if fn is None and isinstance(tc, dict):
            fn = tc.get('function') or {}
            name = fn.get('name') or ''
            arguments = fn.get('arguments') or ''
            tc_id = tc.get('id') or generate_tool_call_id()
        else:
            name = getattr(fn, 'name', None) or ''
            arguments = getattr(fn, 'arguments', None) or ''
            tc_id = getattr(tc, 'id', None) or generate_tool_call_id()
        tool_calls.append(ToolCall(
            id=tc_id,
            function=FunctionCall(name=name or '', arguments=arguments or ''),
        ))

    if content and tool_calls:
        out.append(Message(role=ASSISTANT, content=content, tool_calls=tool_calls))
    elif tool_calls:
        out.append(Message(role=ASSISTANT, content='', tool_calls=tool_calls))
    elif content:
        out.append(Message(role=ASSISTANT, content=content))

    if not out:
        out = [Message(role=ASSISTANT, content=content)]
    return out


@register_llm('oai')
class TextChatAtOAI(BaseFnCallModel):

    @property
    def supports_native_tools(self) -> bool:
        return True

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.model = self.model or 'gpt-4o-mini'
        cfg = cfg or {}

        api_base = cfg.get('api_base')
        api_base = api_base or cfg.get('base_url')
        api_base = api_base or cfg.get('model_server')
        api_base = (api_base or '').strip()

        api_key = cfg.get('api_key')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        api_key = (api_key or 'EMPTY').strip()

        self._async_client = None
        self._async_client_lock = threading.Lock()
        self._api_kwargs: Dict = {}
        self._thread_local = threading.local()
        # None = unknown; True/False after first probe. Avoids double requests once known.
        self._supports_stream_options: Optional[bool] = None

        if openai.__version__.startswith('0.'):
            if api_base:
                openai.api_base = api_base
            if api_key:
                openai.api_key = api_key
            self._complete_create = openai.Completion.create
            self._chat_complete_create = openai.ChatCompletion.create
            self._sync_chat_complete_create = self._chat_complete_create
        else:
            api_kwargs = {}
            if api_base:
                api_kwargs['base_url'] = api_base
            if api_key:
                api_kwargs['api_key'] = api_key
            self._api_kwargs = api_kwargs

            def _chat_complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                bridged = getattr(self._thread_local, 'bridged_create', None)
                if bridged is not None:
                    return bridged(*args, **kwargs)

                client = openai.OpenAI(**api_kwargs)
                return client.chat.completions.create(*args, **kwargs)

            def _complete_create(*args, **kwargs):
                # OpenAI API v1 does not allow the following args, must pass by extra_body
                extra_params = ['top_k', 'repetition_penalty']
                if any((k in kwargs) for k in extra_params):
                    kwargs['extra_body'] = copy.deepcopy(kwargs.get('extra_body', {}))
                    for k in extra_params:
                        if k in kwargs:
                            kwargs['extra_body'][k] = kwargs.pop(k)
                if 'request_timeout' in kwargs:
                    kwargs['timeout'] = kwargs.pop('request_timeout')

                client = openai.OpenAI(**api_kwargs)
                return client.completions.create(*args, **kwargs)

            self._complete_create = _complete_create
            self._chat_complete_create = _chat_complete_create
            self._sync_chat_complete_create = _chat_complete_create

    def _ensure_async_client(self):
        if openai.__version__.startswith('0.'):
            raise ModelServiceError(message='AsyncOpenAI requires openai>=1.0')
        with self._async_client_lock:
            if self._async_client is None:
                self._async_client = openai.AsyncOpenAI(**self._api_kwargs)
            return self._async_client

    async def aclose(self) -> None:
        with self._async_client_lock:
            client = self._async_client
            self._async_client = None
        if client is not None:
            await client.close()

    async def achat(
        self,
        messages: List,
        functions: Optional[List[Dict]] = None,
        extra_generate_cfg: Optional[Dict] = None,
        **kwargs,
    ) -> List[Message]:
        """Async chat via reused ``AsyncOpenAI``. Does not stream tokens; collects the full result."""
        del kwargs
        import asyncio

        loop = asyncio.get_running_loop()
        client = self._ensure_async_client()

        def _bridged_create(*args, **kw):
            # Normalize kwargs the same way as the sync create wrapper.
            extra_params = ['top_k', 'repetition_penalty']
            if any((k in kw) for k in extra_params):
                kw = dict(kw)
                kw['extra_body'] = copy.deepcopy(kw.get('extra_body', {}))
                for k in extra_params:
                    if k in kw:
                        kw['extra_body'][k] = kw.pop(k)
            if 'request_timeout' in kw:
                kw = dict(kw)
                kw['timeout'] = kw.pop('request_timeout')
            # Non-streaming only on the async path (arun collects full turns).
            kw = dict(kw)
            kw['stream'] = False
            fut = asyncio.run_coroutine_threadsafe(
                client.chat.completions.create(*args, **kw),
                loop,
            )
            return fut.result()

        def _collect() -> List[Message]:
            self._thread_local.bridged_create = _bridged_create
            try:
                # stream=False avoids needing an async-stream sync adapter.
                result = self.chat(
                    messages=messages,
                    functions=functions,
                    stream=False,
                    delta_stream=False,
                    extra_generate_cfg=extra_generate_cfg,
                )
                return list(result) if result else []
            finally:
                self._thread_local.bridged_create = None

        return await asyncio.to_thread(_collect)

    def _create_chat_stream(self, messages: List[dict], generate_cfg: dict):
        """Create a streaming chat completion, with optional stream_options + fallback."""
        cfg = dict(generate_cfg)
        include_usage = cfg.pop('include_usage', True)
        want_stream_options = bool(include_usage) and self._supports_stream_options is not False

        def _create(*, with_stream_options: bool):
            kwargs = dict(cfg)
            if with_stream_options:
                kwargs['stream_options'] = {'include_usage': True}
            return self._chat_complete_create(
                model=self.model, messages=messages, stream=True, **kwargs)

        if not want_stream_options:
            return _create(with_stream_options=False)

        try:
            response = _create(with_stream_options=True)
            self._supports_stream_options = True
            return response
        except (BadRequestError, TypeError) as ex:
            self._supports_stream_options = False
            logger.debug(
                'stream_options not supported by server ({}); retrying without it.',
                ex,
            )
            return _create(with_stream_options=False)

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        messages = self.convert_messages_to_dicts(messages)
        logger.debug(f'LLM Input generate_cfg: \n{generate_cfg}')
        try:
            response = self._create_chat_stream(messages, generate_cfg)
            if delta_stream:
                for chunk in response:
                    if chunk.choices:
                        if hasattr(chunk.choices[0].delta,
                                   'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                            yield [
                                Message(role=ASSISTANT,
                                        content='',
                                        reasoning_content=chunk.choices[0].delta.reasoning_content)
                            ]
                        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                            yield [Message(role=ASSISTANT, content=chunk.choices[0].delta.content)]
            else:
                full_response = ''
                full_reasoning_content = ''
                merger = ToolCallStreamMerger()
                last_res: Optional[List[Message]] = None
                captured_usage = None
                for chunk in response:
                    if getattr(chunk, 'usage', None):
                        captured_usage = chunk.usage
                    if not chunk.choices:
                        continue
                    if hasattr(chunk.choices[0].delta,
                               'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                        full_reasoning_content += chunk.choices[0].delta.reasoning_content
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                    if hasattr(chunk.choices[0].delta, 'tool_calls') and chunk.choices[0].delta.tool_calls:
                        merger.push_many(chunk.choices[0].delta.tool_calls)

                    res = []
                    if full_reasoning_content:
                        res.append(Message(role=ASSISTANT, content='', reasoning_content=full_reasoning_content))
                    merged_calls = merger.tool_calls()
                    if full_response and merged_calls:
                        res.append(Message(role=ASSISTANT, content=full_response, tool_calls=merged_calls))
                    elif merged_calls:
                        res.append(Message(role=ASSISTANT, content='', tool_calls=merged_calls))
                    elif full_response:
                        res.append(Message(role=ASSISTANT, content=full_response))
                    last_res = res
                    yield res
                if captured_usage and last_res:
                    _merge_usage(last_res[-1], captured_usage)
                    yield last_res
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        messages = self.convert_messages_to_dicts(messages)
        try:
            response = self._chat_complete_create(model=self.model, messages=messages, stream=False, **generate_cfg)
            msg = response.choices[0].message
            out = _messages_from_completion_message(msg)
            if out:
                _merge_usage(out[-1], getattr(response, 'usage', None))
            return out
        except OpenAIError as ex:
            raise ModelServiceError(exception=ex)

    def convert_messages_to_dicts(self, messages: List[Message]) -> List[dict]:
        # TODO: Change when the VLLM deployed model needs to pass reasoning_complete.
        #  At this time, in order to be compatible with lower versions of vLLM,
        #  and reasoning content is currently not useful
        messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]
        messages = [msg.model_dump() for msg in messages]
        messages = self._conv_cat_agent_messages_to_oai(messages)

        # `cat_agent.log.logger` is Loguru; use lazy logging to avoid
        # expensive formatting when DEBUG is disabled.
        logger.opt(lazy=True).debug("LLM Input:\n{}", lambda: pformat(messages, indent=2))
        return messages
