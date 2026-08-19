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
import re
from threading import Thread
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

from cat_agent.llm.base import register_llm
from cat_agent.llm.function_calling import BaseFnCallModel
from cat_agent.llm.schema import (
    ASSISTANT, FUNCTION, Message, FunctionCall, ToolCall, generate_tool_call_id,
)
from cat_agent.llm.schema import IMAGE, AUDIO, VIDEO
from cat_agent.log import logger


# ---------------------------------------------------------------------------
# Native HF tool-call parsing
# ---------------------------------------------------------------------------

# FunctionGemma 270m: <start_function_call>call:name{...}<end_function_call>
# Some models emit "call name" (space) instead of "call:name" (colon).
_RE_FUNCTIONGEMMA = re.compile(
    r'<start_function_call>\s*call[:\s]+(\w+)\s*\{(.*?)\}\s*<end_function_call>',
    re.DOTALL,
)
# Gemma 4: <|tool_call>call:name{...}<tool_call|>
_RE_GEMMA4 = re.compile(
    r'<\|tool_call>call[:\s]+(\w+)\s*\{(.*?)\}<tool_call\|>',
    re.DOTALL,
)


def _parse_gemma_kv_args(raw: str) -> Dict[str, Any]:
    """Parse Gemma's custom key:value argument format into a Python dict.

    Handles bare values, <escape>-delimited strings (FunctionGemma 270m),
    <|"|>-delimited strings (Gemma 4), arrays, and nested objects.
    """
    raw = raw.strip()
    if not raw:
        return {}
    # Try standard JSON first (some models may emit JSON)
    try:
        parsed = json.loads('{' + raw + '}')
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Replace Gemma escape markers with JSON quotes
    cleaned = raw.replace('<escape>', '"').replace('<|"|>', '"')
    try:
        parsed = json.loads('{' + cleaned + '}')
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Manual key:value parsing as last resort
    result: Dict[str, Any] = {}
    for match in re.finditer(r'(\w+)\s*:\s*', cleaned):
        key = match.group(1)
        rest = cleaned[match.end():]
        val, _ = _parse_value(rest)
        result[key] = val
    return result


def _parse_value(s: str) -> tuple:
    """Parse a single value from the start of *s*, return (value, remaining)."""
    s = s.lstrip()
    if not s:
        return '', s
    if s[0] == '"':
        end = s.index('"', 1)
        return s[1:end], s[end + 1:]
    if s[0] == '[':
        depth, i = 1, 1
        while i < len(s) and depth > 0:
            if s[i] == '[':
                depth += 1
            elif s[i] == ']':
                depth -= 1
            i += 1
        try:
            return json.loads(s[:i]), s[i:]
        except (json.JSONDecodeError, ValueError):
            return s[1:i - 1], s[i:]
    if s[0] == '{':
        depth, i = 1, 1
        while i < len(s) and depth > 0:
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
            i += 1
        try:
            return json.loads(s[:i]), s[i:]
        except (json.JSONDecodeError, ValueError):
            return s[1:i - 1], s[i:]
    # Bare value: read until comma or end
    end = len(s)
    for delim in (',', '}'):
        pos = s.find(delim)
        if pos != -1 and pos < end:
            end = pos
    token = s[:end].strip()
    # Try numeric / bool coercion
    if token.lower() == 'true':
        return True, s[end:]
    if token.lower() == 'false':
        return False, s[end:]
    try:
        return int(token), s[end:]
    except ValueError:
        pass
    try:
        return float(token), s[end:]
    except ValueError:
        pass
    return token, s[end:]


def parse_native_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from HF model output supporting FunctionGemma and Gemma 4."""
    calls: List[Dict[str, Any]] = []
    for pattern in (_RE_FUNCTIONGEMMA, _RE_GEMMA4):
        for m in pattern.finditer(text):
            name = m.group(1)
            args = _parse_gemma_kv_args(m.group(2))
            calls.append({'name': name, 'arguments': args})
        if calls:
            break
    # Deduplicate: small models sometimes emit the same call multiple times
    seen = set()
    unique: List[Dict[str, Any]] = []
    for c in calls:
        key = (c['name'], json.dumps(c['arguments'], sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _convert_messages_for_use_chat_template_tools(
    messages: List[Message],
) -> List[Dict[str, Any]]:
    """Convert cat-agent internal messages to the HF chat template format.

    Transforms Message(role=ASSISTANT, tool_calls=[...]) into
    {"role": "assistant", "tool_calls": [{"type":"function","function":{...}}]}
    and Message(role=FUNCTION, name=..., content=...) into
    {"role": "tool", "content": [{"name": ..., "response": ...}]}.
    """
    converted: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.tool_calls:
            hf_calls = []
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        pass
                hf_calls.append({
                    'type': 'function',
                    'function': {'name': tc.function.name, 'arguments': args},
                })
            entry: Dict[str, Any] = {'role': 'assistant', 'tool_calls': hf_calls}
            if msg.content:
                entry['content'] = msg.content if isinstance(msg.content, str) else ''
            converted.append(entry)
        elif msg.role == FUNCTION:
            content_text = ''
            if isinstance(msg.content, str):
                content_text = msg.content
            elif isinstance(msg.content, list) and msg.content:
                content_text = msg.content[0].text or ''
            # The HF template (FunctionGemma) calls dictsort on the response,
            # so it MUST be a dict — wrap plain strings.
            try:
                response_val = json.loads(content_text)
            except (json.JSONDecodeError, ValueError, TypeError):
                response_val = {'result': content_text}
            if not isinstance(response_val, dict):
                response_val = {'result': response_val}
            converted.append({
                'role': 'tool',
                'content': [{'name': msg.name or '', 'response': response_val}],
            })
        else:
            dumped = msg.model_dump()
            converted.append({'role': dumped.get('role', 'user'), 'content': dumped.get('content', '')})
    return converted


def _format_transformers_import_error(err: BaseException) -> str:
    root = err
    while root.__cause__ is not None:
        root = root.__cause__
    return (
        'Could not import HuggingFace Transformers.\n'
        f'Root cause: {type(root).__name__}: {root}\n'
        'This usually means the active Python environment has incompatible packages '
        '(often pyOpenSSL/cryptography breaking accelerate).\n'
        'Try: pip install "cat-agent[transformers]"\n'
        'Or use a fresh venv: python -m venv .venv && source .venv/bin/activate '
        '&& pip install "cat-agent[transformers]"'
    )


@register_llm('transformers')
class Transformers(BaseFnCallModel):
    """
    Transformers class supports loading models from `transformers` library.

    Example of creating an assistant:
        llm_cfg = {
            'model': 'Qwen/Qwen3-4B',
            'model_type': 'transformers',
            'device': 'cuda'
        }
        bot = Assistant(llm=llm_cfg, ...)
    """

    @property
    def supports_use_chat_template_tools(self) -> bool:
        return False

    def __init__(self, cfg: Optional[Dict] = None):
        self._use_chat_template_tools = bool((cfg or {}).get('use_chat_template_tools', False))
        super().__init__(cfg)

        if 'model' not in cfg:
            raise ValueError('Please provide the model id or directory through `model` in cfg.')

        try:
            import transformers
            from transformers import AutoConfig, AutoProcessor
            from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
        except ImportError as e:
            raise ImportError(_format_transformers_import_error(e)) from e
        
        self.hf_config = AutoConfig.from_pretrained(cfg['model'])
        arch = self.hf_config.architectures[0]
        if len(self.hf_config.architectures) > 1:
            logger.warning(f'The config for the transformers model type contains more than one architecture, choosing the first: {arch}')

        # try loading a processor, if got a tokenizer, regarding the model as text-only
        processor = AutoProcessor.from_pretrained(cfg['model'])
        if isinstance(processor, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            logger.info('Regarding the transformers model as text-only since its processor is a tokenizer.')
            self.tokenizer = processor
            self._support_multimodal_input = False
        else:
            self.processor = processor
            self.tokenizer = self.processor.tokenizer
            self._support_multimodal_input = True

        model_cls = getattr(transformers, arch)
        self.hf_model = model_cls.from_pretrained(cfg['model'], config=self.hf_config, torch_dtype='auto').to(cfg.get('device', 'cpu'))

    @property
    def support_multimodal_input(self) -> bool:
        return self._support_multimodal_input
    
    @property
    def support_audio_input(self) -> bool:
        return self._support_multimodal_input

    def _get_streamer(self):
        from transformers import TextIteratorStreamer

        return TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

    def _get_inputs(self, messages: List[Message]):
        import torch

        messages_plain = [message.model_dump() for message in messages]
        if not self.support_multimodal_input:
            # For text-only models, apply_chat_template returns a BatchEncoding
            # when return_tensors='pt'. We must extract the underlying tensor.
            encodings = self.tokenizer.apply_chat_template(
                messages_plain,
                add_generation_prompt=True,
                return_tensors='pt',
            )
            if hasattr(encodings, "input_ids"):
                input_ids = encodings.input_ids
            else:
                input_ids = encodings
            inputs = dict(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            )
        else:
            for message in messages_plain:
                for content_item in message['content']:
                    content_item['type'] = [type_ for type_ in ('text', IMAGE, AUDIO, VIDEO) if type_ in content_item][0]
            
            has_vision = False
            audio_paths = []
            for message in messages_plain:
                for content_item in message['content']:
                    if content_item['type'] in (IMAGE, VIDEO):
                        has_vision = True
                    if content_item['type'] in (AUDIO,):
                        audio_paths.append(content_item[AUDIO])
            
            prompt = self.processor.apply_chat_template(messages_plain, add_generation_prompt=True, tokenize=False)
            processor_kwargs = {'text': prompt}
            
            if has_vision:
                from qwen_vl_utils import process_vision_info
                
                images, videos = process_vision_info(messages_plain)
                processor_kwargs['images'] = images
                processor_kwargs['videos'] = videos
            
            if audio_paths:
                import librosa

                audios = []
                for path in audio_paths:
                    if path.startswith("file://"):
                        audios.append(librosa.load(path[len("file://") :], sr=self.processor.feature_extractor.sampling_rate)[0])
                    else:
                        audios.append(librosa.load(path, sr=self.processor.feature_extractor.sampling_rate)[0])
                processor_kwargs['audios'] = audios
            
            inputs = self.processor(**processor_kwargs, return_tensors="pt")

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.hf_model.device)
        return inputs

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        generate_cfg = copy.deepcopy(generate_cfg)
        inputs = self._get_inputs(messages)
        streamer = self._get_streamer()

        generate_cfg.update(inputs)
        generate_cfg.update(dict(
            streamer=streamer,
            max_new_tokens=generate_cfg.get('max_new_tokens', 2048)
        ))

        # Handle special keys that are not accepted by `generate`
        if 'seed' in generate_cfg:
            from transformers import set_seed
            set_seed(generate_cfg['seed'])
            del generate_cfg['seed']
        # `stop` is used by some backends but not by transformers' generate()
        if 'stop' in generate_cfg:
            logger.debug(f"Removing unsupported `stop` from generate_cfg for transformers backend: {generate_cfg['stop']}")
            del generate_cfg['stop']

        def generate_and_signal_complete():
            self.hf_model.generate(**generate_cfg)

        t1 = Thread(target=generate_and_signal_complete)
        t1.start()
        partial_text = ''
        for new_text in streamer:
            partial_text += new_text
            if delta_stream:
                yield [Message(ASSISTANT, new_text)]
            else:
                yield [Message(ASSISTANT, partial_text)]

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        generate_cfg = copy.deepcopy(generate_cfg)

        inputs = self._get_inputs(messages)
        generate_cfg.update(inputs)
        generate_cfg.update(dict(
            max_new_tokens=generate_cfg.get('max_new_tokens', 2048)
        ))

        # Handle special keys that are not accepted by `generate`
        if 'seed' in generate_cfg:
            from transformers import set_seed
            set_seed(generate_cfg['seed'])
            del generate_cfg['seed']
        if 'stop' in generate_cfg:
            logger.debug(f"Removing unsupported `stop` from generate_cfg for transformers backend: {generate_cfg['stop']}")
            del generate_cfg['stop']

        response = self.hf_model.generate(**generate_cfg)
        response = response[:, inputs['input_ids'].size(-1):]
        answer = self.tokenizer.batch_decode(response, skip_special_tokens=True)[0]
        return [Message(ASSISTANT, answer)]

    # ------------------------------------------------------------------
    # Native HF tool calling (use_chat_template_tools=True)
    # ------------------------------------------------------------------

    def _get_inputs_with_tools(
        self,
        messages: List[Message],
        functions: List[Dict],
    ) -> dict:
        """Tokenize messages with tool schemas via the HF chat template."""
        import torch

        hf_messages = _convert_messages_for_use_chat_template_tools(messages)
        tools = []
        for fn in functions:
            if fn.get('type') == 'function':
                tools.append(fn)
            else:
                tools.append({'type': 'function', 'function': fn})

        template_target = self.processor if self.support_multimodal_input else self.tokenizer
        encodings = template_target.apply_chat_template(
            hf_messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors='pt',
        )

        if isinstance(encodings, dict):
            inputs = dict(encodings)
        elif hasattr(encodings, 'input_ids'):
            inputs = {k: getattr(encodings, k) for k in ('input_ids', 'attention_mask')
                      if hasattr(encodings, k)}
        else:
            inputs = dict(
                input_ids=encodings,
                attention_mask=torch.ones_like(encodings),
            )
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(self.hf_model.device)
        return inputs

    def _chat_with_use_chat_template_tools(
        self,
        messages: List[Message],
        functions: List[Dict],
        stream: bool,
        generate_cfg: dict,
    ) -> Union[List[Message], Iterator[List[Message]]]:
        """Native HF tool-calling path: pass tools to apply_chat_template."""
        generate_cfg = copy.deepcopy(generate_cfg)
        for k in ('function_choice', 'thought_in_content', 'seed', 'stop'):
            if k in generate_cfg:
                if k == 'seed':
                    from transformers import set_seed
                    set_seed(generate_cfg['seed'])
                del generate_cfg[k]

        inputs = self._get_inputs_with_tools(messages, functions)
        prompt_len = inputs['input_ids'].shape[-1]

        generate_cfg.update(inputs)
        generate_cfg['max_new_tokens'] = generate_cfg.get('max_new_tokens', 2048)

        response = self.hf_model.generate(**generate_cfg)
        new_tokens = response[:, prompt_len:]

        # Decode WITH special tokens to detect function call markers
        raw_output = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)[0]
        clean_output = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        logger.debug(f'Native tools raw output: {raw_output!r}')

        calls = parse_native_tool_calls(raw_output)

        # If there's already a tool result in the conversation, the model should
        # produce a final answer — not call the same tool again. This prevents
        # infinite loops where the model re-emits the tool call after seeing the result.
        has_prior_tool_result = any(m.role == FUNCTION for m in messages)
        if calls and has_prior_tool_result:
            prior_tool_names = {m.name for m in messages if m.role == FUNCTION}
            new_calls = [c for c in calls if c['name'] not in prior_tool_names]
            if not new_calls:
                logger.debug('Suppressing repeated tool call(s) after tool result; using text answer.')
                calls = []

        if calls and not has_prior_tool_result:
            # First pass: return only the tool call, no content
            call = calls[0]
            tc_id = generate_tool_call_id()
            result = [Message(
                role=ASSISTANT, content='',
                tool_calls=[ToolCall(
                    id=tc_id,
                    function=FunctionCall(
                        name=call['name'],
                        arguments=json.dumps(call['arguments']),
                    ),
                )],
            )]
        elif calls:
            # Subsequent passes with new tools
            tool_calls = []
            for call in calls:
                tc_id = generate_tool_call_id()
                tool_calls.append(ToolCall(
                    id=tc_id,
                    function=FunctionCall(
                        name=call['name'],
                        arguments=json.dumps(call['arguments']),
                    ),
                ))
            result = [Message(role=ASSISTANT, content='', tool_calls=tool_calls)]
        else:
            # Final answer: strip any function call markers from clean text
            answer = clean_output.strip()
            # Remove any residual function call text the model may have appended
            for marker in ('<start_function_call>', '<end_function_call>',
                           '<|tool_call>', '<tool_call|>'):
                answer = answer.replace(marker, '')
            # Take only text before any "call:" pattern (model hallucinating calls)
            call_pos = re.search(r'\bcall[:\s]+\w+\s*\{', answer)
            if call_pos:
                answer = answer[:call_pos.start()].strip()
            result = [Message(role=ASSISTANT, content=answer)]

        if stream:
            return iter([result])
        return result
