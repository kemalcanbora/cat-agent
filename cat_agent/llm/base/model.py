"""Core LLM abstractions: BaseChatModel, ModelServiceError, and the LLM registry."""

import copy
import json
import os
import random
from abc import ABC, abstractmethod
from pprint import pformat
from typing import Dict, Iterator, List, Literal, Optional, Union

from cat_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, FUNCTION, SYSTEM, USER, Message
from cat_agent.log import logger
from cat_agent.settings import DEFAULT_MAX_INPUT_TOKENS
from cat_agent.utils.json_utils import json_dumps_compact
from cat_agent.utils.message_utils import (format_as_multimodal_message, format_as_text_message, has_chinese_messages)
from cat_agent.utils.misc import merge_generate_cfgs, print_traceback

from cat_agent.llm.base.postprocessing import format_as_text_messages, postprocess_stop_words
from cat_agent.llm.base.retry import retry_model_service, retry_model_service_iterator
from cat_agent.llm.base.truncation import truncate_input_messages_roughly

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LLM_REGISTRY = {}


def register_llm(model_type):

    def decorator(cls):
        LLM_REGISTRY[model_type] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ModelServiceError(Exception):

    def __init__(self,
                 exception: Optional[Exception] = None,
                 code: Optional[str] = None,
                 message: Optional[str] = None,
                 extra: Optional[dict] = None):
        if exception is not None:
            super().__init__(exception)
        else:
            super().__init__(f'\nError code: {code}. Error message: {message}')
        self.exception = exception
        self.code = code
        self.message = message
        self.extra = extra


# ---------------------------------------------------------------------------
# BaseChatModel
# ---------------------------------------------------------------------------


class BaseChatModel(ABC):
    """The base class of LLM."""

    @property
    def support_multimodal_input(self) -> bool:
        return False

    @property
    def support_multimodal_output(self) -> bool:
        return False

    @property
    def support_audio_input(self) -> bool:
        return False

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.model = cfg.get('model', '').strip()
        generate_cfg = copy.deepcopy(cfg.get('generate_cfg', {}))
        cache_dir = cfg.get('cache_dir', generate_cfg.pop('cache_dir', None))
        self.max_retries = generate_cfg.pop('max_retries', 0)
        self.generate_cfg = generate_cfg
        self.model_type = cfg.get('model_type', '')

        self.use_raw_api = os.getenv('CAT_AGENT_USE_RAW_API', 'false').lower() == 'true'
        if 'use_raw_api' in generate_cfg:
            self.use_raw_api = generate_cfg.pop('use_raw_api')

        if cache_dir:
            try:
                import diskcache
            except ImportError:
                print_traceback(is_error=False)
                logger.warning('Caching disabled because diskcache is not installed. Please `pip install diskcache`.')
                cache_dir = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = diskcache.Cache(directory=cache_dir)
        else:
            self.cache = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quick_chat(self, prompt: str) -> str:
        *_, responses = self.chat(messages=[Message(role=USER, content=prompt)])
        assert len(responses) == 1
        assert not responses[0].function_call
        assert isinstance(responses[0].content, str)
        return responses[0].content

    def chat(
        self,
        messages: List[Union[Message, Dict]],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        delta_stream: bool = False,
        extra_generate_cfg: Optional[Dict] = None,
    ) -> Union[List[Message], List[Dict], Iterator[List[Message]], Iterator[List[Dict]]]:
        """LLM chat interface.

        Args:
            messages: Inputted messages.
            functions: Inputted functions for function calling. OpenAI format supported.
            stream: Whether to use streaming generation.
            delta_stream: Whether to stream the response incrementally.
            extra_generate_cfg: Extra LLM generation hyper-parameters.

        Returns:
            The generated message list response by LLM.
        """

        # Unify input to List[Message]
        messages = copy.deepcopy(messages)
        _return_message_type = 'dict'
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'
        messages = new_messages

        if not messages:
            raise ValueError('Messages can not be empty.')

        # Cache lookup
        if self.cache is not None:
            cache_key = dict(messages=messages, functions=functions, extra_generate_cfg=extra_generate_cfg)
            cache_key: str = json_dumps_compact(cache_key, sort_keys=True)
            cache_value: str = self.cache.get(cache_key)
            if cache_value:
                cache_value: List[dict] = json.loads(cache_value)
                if _return_message_type == 'message':
                    cache_value: List[Message] = [Message(**m) for m in cache_value]
                if stream:
                    cache_value: Iterator[List[Union[Message, dict]]] = iter([cache_value])
                return cache_value

        if stream and delta_stream:
            logger.warning(
                'Support for `delta_stream=True` is deprecated. '
                'Please use `stream=True and delta_stream=False` or `stream=False` instead. '
                'Using `delta_stream=True` makes it difficult to implement advanced postprocessing and retry mechanisms.'
            )

        generate_cfg = merge_generate_cfgs(base_generate_cfg=self.generate_cfg, new_generate_cfg=extra_generate_cfg)
        if 'seed' not in generate_cfg:
            generate_cfg['seed'] = random.randint(a=0, b=2**30)
        if 'lang' in generate_cfg:
            lang: Literal['en', 'zh'] = generate_cfg.pop('lang')
        else:
            lang: Literal['en', 'zh'] = 'zh' if has_chinese_messages(messages) else 'en'
        if not stream and 'incremental_output' in generate_cfg:
            generate_cfg.pop('incremental_output')

        if DEFAULT_SYSTEM_MESSAGE and messages[0].role != SYSTEM:
            messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)] + messages

        # Truncate if necessary
        max_input_tokens = generate_cfg.pop('max_input_tokens', DEFAULT_MAX_INPUT_TOKENS)
        if max_input_tokens > 0:
            messages = truncate_input_messages_roughly(messages=messages, max_tokens=max_input_tokens)

        # Determine function-calling mode
        fncall_mode = bool(functions)
        if 'function_choice' in generate_cfg:
            fn_choice = generate_cfg['function_choice']
            valid_fn_choices = [f.get('name', f.get('name_for_model', None)) for f in (functions or [])]
            valid_fn_choices = ['auto', 'none'] + [f for f in valid_fn_choices if f]
            if fn_choice not in valid_fn_choices:
                raise ValueError(f'The value of function_choice must be one of the following: {valid_fn_choices}. '
                                 f'But function_choice="{fn_choice}" is received.')
            if fn_choice == 'none':
                fncall_mode = False

        # Preprocessing
        messages = self._preprocess_messages(messages,
                                             lang=lang,
                                             generate_cfg=generate_cfg,
                                             functions=functions,
                                             use_raw_api=self.use_raw_api)
        if not self.support_multimodal_input:
            messages = [format_as_text_message(msg, add_upload_info=False) for msg in messages]

        if self.use_raw_api:
            logger.debug('`use_raw_api` takes effect.')
            assert stream and (not delta_stream), '`use_raw_api` only support full stream!!!'
            return self.raw_chat(messages=messages, functions=functions, stream=stream, generate_cfg=generate_cfg)

        if not fncall_mode:
            for k in ('parallel_function_calls', 'function_choice', 'thought_in_content'):
                generate_cfg.pop(k, None)

        # Dispatch
        def _call_model_service():
            if fncall_mode:
                return self._chat_with_functions(
                    messages=messages, functions=functions,
                    stream=stream, delta_stream=delta_stream,
                    generate_cfg=generate_cfg, lang=lang,
                )
            else:
                if messages[-1].role == ASSISTANT:
                    assert not delta_stream, 'Continuation mode does not currently support `delta_stream`'
                    return self._continue_assistant_response(messages, generate_cfg=generate_cfg, stream=stream)
                else:
                    return self._chat(messages, stream=stream, delta_stream=delta_stream, generate_cfg=generate_cfg)

        if stream and delta_stream:
            output = _call_model_service()
        elif stream and (not delta_stream):
            output = retry_model_service_iterator(_call_model_service, max_retries=self.max_retries)
        else:
            output = retry_model_service(_call_model_service, max_retries=self.max_retries)

        # Post-processing
        if isinstance(output, list):
            assert not stream
            logger.debug(f'LLM Output: \n{pformat([_.model_dump() for _ in output], indent=2)}')
            output = self._postprocess_messages(output, fncall_mode=fncall_mode, generate_cfg=generate_cfg)
            if not self.support_multimodal_output:
                output = format_as_text_messages(messages=output)
            if self.cache:
                self.cache.set(cache_key, json_dumps_compact(output))
            return self._convert_messages_to_target_type(output, _return_message_type)
        else:
            assert stream
            if delta_stream:
                generate_cfg = copy.deepcopy(generate_cfg)
                assert 'skip_stopword_postproc' not in generate_cfg
                generate_cfg['skip_stopword_postproc'] = True
            output = self._postprocess_messages_iterator(output, fncall_mode=fncall_mode, generate_cfg=generate_cfg)

            def _format_and_cache() -> Iterator[List[Message]]:
                o = []
                for o in output:
                    if o:
                        if not self.support_multimodal_output:
                            o = format_as_text_messages(messages=o)
                        yield o
                if o and (self.cache is not None):
                    self.cache.set(cache_key, json_dumps_compact(o))

            return self._convert_messages_iterator_to_target_type(_format_and_cache(), _return_message_type)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _chat(self, messages, stream, delta_stream, generate_cfg):
        if stream:
            return self._chat_stream(messages, delta_stream=delta_stream, generate_cfg=generate_cfg)
        else:
            return self._chat_no_stream(messages, generate_cfg=generate_cfg)

    @abstractmethod
    def _chat_with_functions(self, messages, functions, stream, delta_stream, generate_cfg, lang):
        raise NotImplementedError

    def _continue_assistant_response(self, messages, generate_cfg, stream):
        raise NotImplementedError

    @abstractmethod
    def _chat_stream(self, messages, delta_stream, generate_cfg):
        raise NotImplementedError

    @abstractmethod
    def _chat_no_stream(self, messages, generate_cfg):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Pre/post processing
    # ------------------------------------------------------------------

    def _preprocess_messages(self, messages, lang, generate_cfg, functions=None, use_raw_api=False):
        add_multimodel_upload_info = bool(functions) or (not self.support_multimodal_input)
        add_audio_upload_info = bool(functions) or (not self.support_audio_input)
        return [
            format_as_multimodal_message(msg,
                                         add_upload_info=True,
                                         add_multimodel_upload_info=add_multimodel_upload_info,
                                         add_audio_upload_info=add_audio_upload_info,
                                         lang=lang) for msg in messages
        ]

    def _postprocess_messages(self, messages, fncall_mode, generate_cfg):
        messages = [
            format_as_multimodal_message(msg, add_upload_info=False,
                                         add_multimodel_upload_info=False,
                                         add_audio_upload_info=False)
            for msg in messages
        ]
        if not generate_cfg.get('skip_stopword_postproc', False):
            stop = generate_cfg.get('stop', [])
            messages = postprocess_stop_words(messages, stop=stop)
        return messages

    def _postprocess_messages_iterator(self, messages, fncall_mode, generate_cfg):
        pre_msg = []
        for pre_msg in messages:
            yield self._postprocess_messages(pre_msg, fncall_mode=fncall_mode, generate_cfg=generate_cfg)
        logger.debug(f'LLM Output: \n{pformat([_.model_dump() for _ in pre_msg], indent=2)}')

    # ------------------------------------------------------------------
    # Output type conversion
    # ------------------------------------------------------------------

    def _convert_messages_to_target_type(self, messages, target_type):
        if target_type == 'message':
            return [Message(**x) if isinstance(x, dict) else x for x in messages]
        elif target_type == 'dict':
            return [x.model_dump() if not isinstance(x, dict) else x for x in messages]
        else:
            raise NotImplementedError

    def _convert_messages_iterator_to_target_type(self, messages_iter, target_type):
        for messages in messages_iter:
            yield self._convert_messages_to_target_type(messages, target_type)

    # ------------------------------------------------------------------
    # Raw / OAI-compat helpers
    # ------------------------------------------------------------------

    def raw_chat(self, messages, functions=None, stream=True, generate_cfg=None):
        if functions and functions[0].get('type') != 'function':
            functions = [{'type': 'function', 'function': f} for f in functions]
        if functions:
            generate_cfg['tools'] = functions
        if stream:
            return self._chat_stream(messages=messages, delta_stream=False, generate_cfg=generate_cfg)

    @staticmethod
    def _conv_cat_agent_messages_to_oai(messages):
        new_messages = []
        for msg in messages:
            if msg['role'] == ASSISTANT:
                if new_messages[-1]['role'] != ASSISTANT:
                    new_messages.append({'role': ASSISTANT})
                if msg.get('content'):
                    new_messages[-1]['content'] = msg['content']
                if msg.get('reasoning_content'):
                    new_messages[-1]['reasoning_content'] = msg['reasoning_content']
                if msg.get('function_call'):
                    if not new_messages[-1].get('tool_calls'):
                        new_messages[-1]['tool_calls'] = []
                    new_messages[-1]['tool_calls'].append({
                        'id': msg.get('extra', {}).get('function_id', '1'),
                        'type': 'function',
                        'function': {
                            'name': msg['function_call']['name'],
                            'arguments': msg['function_call']['arguments'],
                        }
                    })
            elif msg['role'] == FUNCTION:
                new_msg = copy.deepcopy(msg)
                new_msg['role'] = 'tool'
                new_msg['id'] = msg.get('extra', {}).get('function_id', '1')
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        return new_messages

    def quick_chat_oai(self, messages: List[dict], tools: Optional[list] = None) -> dict:
        """Temporary OpenAI-compatible streaming interface (may change at any time)."""

        def _to_internal(msgs):
            out = []
            for msg in msgs:
                if msg['role'] in ('system', 'user'):
                    out.append(msg)
                elif msg['role'] == 'tool':
                    m = copy.deepcopy(msg)
                    m['role'] = 'function'
                    out.append(m)
                elif msg['role'] == 'assistant':
                    if msg['content']:
                        out.append({'role': 'assistant', 'content': msg['content']})
                    if msg.get('reasoning_content', ''):
                        out.append({'role': 'assistant', 'content': '', 'reasoning_content': msg['reasoning_content']})
                    if msg.get('tool_calls'):
                        for tc in msg['tool_calls']:
                            out.append({
                                'role': 'assistant', 'content': '',
                                'function_call': {'name': tc['function']['name'],
                                                  'arguments': tc['function']['arguments']},
                            })
            return out

        def _to_oai(data):
            message = {'role': 'assistant', 'content': '', 'reasoning_content': '', 'tool_calls': []}
            for item in data:
                if item.get('reasoning_content'):
                    message['reasoning_content'] += item['reasoning_content']
                if item.get('content'):
                    message['content'] += item['content']
                if item.get('function_call'):
                    message['tool_calls'].append({
                        'id': f"{len(message['tool_calls']) + 1}",
                        'type': 'function',
                        'function': {'name': item['function_call']['name'],
                                     'arguments': item['function_call']['arguments']},
                    })
            return {
                'choices': [{'message': message}],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            }

        functions = [t['function'] for t in tools] if tools else None
        for rsp in self.chat(messages=_to_internal(messages), functions=functions, stream=True):
            yield _to_oai(rsp)
