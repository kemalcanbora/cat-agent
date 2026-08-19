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

"""Tests for native HF tool calling (FunctionGemma / Gemma 4 support).

All tests mock the model and tokenizer — no GPU or model download required.
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from cat_agent.llm.schema import ASSISTANT, FUNCTION, USER, FunctionCall, Message, ToolCall
from cat_agent.llm.transformers_llm import (
    _convert_messages_for_use_chat_template_tools,
    _parse_gemma_kv_args,
    parse_native_tool_calls,
)


# ---------------------------------------------------------------------------
# Fixtures: mock transformers / torch
# ---------------------------------------------------------------------------

def _install_fake_transformers(monkeypatch, *, use_chat_template_tools=False):
    """Install fake transformers/torch modules and return a Transformers LLM instance."""

    class PreTrainedTokenizer:
        pass

    class PreTrainedTokenizerFast:
        pass

    class FakeTokenizer(PreTrainedTokenizer):
        _apply_chat_template_calls = []

        def apply_chat_template(self, messages, *, add_generation_prompt=True,
                                return_tensors=None, tools=None, tokenize=True,
                                return_dict=False):
            self._apply_chat_template_calls.append({
                'messages': messages,
                'tools': tools,
                'add_generation_prompt': add_generation_prompt,
                'return_dict': return_dict,
            })
            tensor = MagicMock()
            tensor.to = MagicMock(return_value=tensor)
            tensor.shape = (1, 10)  # batch=1, seq_len=10
            enc = MagicMock()
            enc.input_ids = tensor
            enc.attention_mask = tensor
            enc.__contains__ = lambda self, k: k in ('input_ids', 'attention_mask')
            enc.__getitem__ = lambda self, k: tensor
            enc.items = lambda: [('input_ids', tensor), ('attention_mask', tensor)]
            enc.keys = lambda: ['input_ids', 'attention_mask']
            return enc

        def batch_decode(self, response, skip_special_tokens=True):
            return ['decoded-answer']

    generate_call_count = {'n': 0}

    class FakeModel:
        def __init__(self):
            self.device = 'cpu'

        def to(self, device):
            self.device = device
            return self

        def generate(self, **kwargs):
            generate_call_count['n'] += 1
            out = MagicMock()
            sliced = MagicMock()
            out.__getitem__ = MagicMock(return_value=sliced)
            out.shape = (1, 20)
            return out

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    class FakeConfig:
        architectures = ['FakeModel']

    class AutoConfig:
        @staticmethod
        def from_pretrained(model):
            return FakeConfig()

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model):
            return FakeTokenizer()

    class TextIteratorStreamer:
        def __init__(self, *a, **k):
            self._items = ['chunk']

        def __iter__(self):
            return iter(self._items)

    def set_seed(s):
        return None

    transformers = ModuleType('transformers')
    transformers.AutoConfig = AutoConfig
    transformers.AutoProcessor = AutoProcessor
    transformers.PreTrainedTokenizer = PreTrainedTokenizer
    transformers.PreTrainedTokenizerFast = PreTrainedTokenizerFast
    transformers.TextIteratorStreamer = TextIteratorStreamer
    transformers.set_seed = set_seed
    transformers.FakeModel = FakeModel

    torch = ModuleType('torch')
    torch.ones_like = lambda x: MagicMock(name='mask')
    torch.is_tensor = lambda v: True

    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.setitem(sys.modules, 'torch', torch)

    from cat_agent.llm.transformers_llm import Transformers

    cfg = {'model': 'test/model', 'device': 'cpu'}
    if use_chat_template_tools:
        cfg['use_chat_template_tools'] = True
    llm = Transformers(cfg)
    return llm, FakeTokenizer, generate_call_count


SUM_NUMBERS_SCHEMA = {
    'name': 'sum_numbers',
    'description': 'Sum a list of numbers.',
    'parameters': {
        'type': 'object',
        'properties': {'numbers': {'type': 'array', 'items': {'type': 'number'}}},
        'required': ['numbers'],
    },
}


# ---------------------------------------------------------------------------
# Test A: Prompt-based function calling still works
# ---------------------------------------------------------------------------

class TestPromptBasedStillWorks:
    def test_default_no_use_chat_template_tools(self, monkeypatch):
        llm, _, _ = _install_fake_transformers(monkeypatch, use_chat_template_tools=False)
        assert llm._use_chat_template_tools is False

    def test_fncall_prompt_used_when_use_chat_template_tools_false(self, monkeypatch):
        llm, _, _ = _install_fake_transformers(monkeypatch, use_chat_template_tools=False)
        from cat_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt
        assert isinstance(llm.fncall_prompt, NousFnCallPrompt)


# ---------------------------------------------------------------------------
# Test B: use_chat_template_tools=True passes tools to apply_chat_template
# ---------------------------------------------------------------------------

class TestNativeToolsPassesTools:
    def test_apply_chat_template_receives_tools(self, monkeypatch):
        llm, FakeTokenizer, _ = _install_fake_transformers(monkeypatch, use_chat_template_tools=True)
        FakeTokenizer._apply_chat_template_calls.clear()

        msgs = [Message(USER, 'Sum 1,2,3')]
        functions = [SUM_NUMBERS_SCHEMA]
        llm._get_inputs_with_tools(msgs, functions)

        assert len(FakeTokenizer._apply_chat_template_calls) == 1
        call_args = FakeTokenizer._apply_chat_template_calls[0]
        assert call_args['tools'] is not None
        assert len(call_args['tools']) == 1
        assert call_args['tools'][0]['function']['name'] == 'sum_numbers'


# ---------------------------------------------------------------------------
# Test C: Parser correctly extracts FunctionGemma call
# ---------------------------------------------------------------------------

class TestParseNativeToolCalls:
    def test_functiongemma_format(self):
        text = '<start_function_call>call:sum_numbers{numbers:[1,2,3,4,5]}<end_function_call>'
        calls = parse_native_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]['name'] == 'sum_numbers'
        assert calls[0]['arguments'] == {'numbers': [1, 2, 3, 4, 5]}

    def test_gemma4_format(self):
        text = '<|tool_call>call:get_weather{location:<|"|>Tokyo<|"|>}<tool_call|>'
        calls = parse_native_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]['name'] == 'get_weather'
        assert calls[0]['arguments']['location'] == 'Tokyo'

    def test_functiongemma_with_escape_strings(self):
        text = '<start_function_call>call:search{query:<escape>hello world<escape>}<end_function_call>'
        calls = parse_native_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]['arguments']['query'] == 'hello world'

    def test_functiongemma_space_separator(self):
        """Some FunctionGemma outputs use space instead of colon after 'call'."""
        text = '<start_function_call>call sum_numbers{numbers:[1,2,3,4,5]}<end_function_call>'
        calls = parse_native_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]['name'] == 'sum_numbers'
        assert calls[0]['arguments'] == {'numbers': [1, 2, 3, 4, 5]}

    def test_functiongemma_with_wrapping_tokens(self):
        """Output may contain extra tokens like [ANSWER] wrapping the call."""
        text = '[ANSWER]\n<start_function_call>call sum_numbers{numbers:[1,2,3,4,5]}<end_function_call>'
        calls = parse_native_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]['name'] == 'sum_numbers'

    def test_no_match_returns_empty(self):
        assert parse_native_tool_calls('The sum is 15') == []
        assert parse_native_tool_calls('') == []

    def test_multiple_calls(self):
        text = (
            '<start_function_call>call:a{x:1}<end_function_call>'
            '<start_function_call>call:b{y:2}<end_function_call>'
        )
        calls = parse_native_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]['name'] == 'a'
        assert calls[1]['name'] == 'b'


# ---------------------------------------------------------------------------
# Test D: Parser edge cases
# ---------------------------------------------------------------------------

class TestParseEdgeCases:
    def test_nested_object(self):
        args = _parse_gemma_kv_args('config:{"theme":"dark","size":12}')
        assert args['config'] == {'theme': 'dark', 'size': 12}

    def test_boolean_and_numeric(self):
        args = _parse_gemma_kv_args('count:42,active:true,rate:3.14')
        assert args['count'] == 42
        assert args['active'] is True
        assert abs(args['rate'] - 3.14) < 0.001

    def test_empty_args(self):
        assert _parse_gemma_kv_args('') == {}

    def test_standard_json_passthrough(self):
        args = _parse_gemma_kv_args('"numbers": [1, 2, 3]')
        assert args == {'numbers': [1, 2, 3]}


# ---------------------------------------------------------------------------
# Test E: Message conversion (FUNCTION -> native tool format)
# ---------------------------------------------------------------------------

class TestConvertMessages:
    def test_function_message_becomes_tool_role(self):
        msgs = [
            Message(USER, 'Sum 1,2,3'),
            Message(
                role=ASSISTANT, content='',
                tool_calls=[ToolCall(
                    id='call_1',
                    function=FunctionCall(name='sum_numbers', arguments='{"numbers":[1,2,3]}'),
                )],
            ),
            Message(role=FUNCTION, name='sum_numbers', content='The sum is 6'),
        ]
        converted = _convert_messages_for_use_chat_template_tools(msgs)

        assert converted[0]['role'] == 'user'
        assert converted[1]['role'] == 'assistant'
        assert 'tool_calls' in converted[1]
        assert converted[1]['tool_calls'][0]['function']['name'] == 'sum_numbers'
        # Arguments should be deserialized dict, not string
        assert converted[1]['tool_calls'][0]['function']['arguments'] == {'numbers': [1, 2, 3]}

        assert converted[2]['role'] == 'tool'
        assert isinstance(converted[2]['content'], list)
        assert converted[2]['content'][0]['name'] == 'sum_numbers'
        # Plain string responses are wrapped in {"result": ...} for HF template compat
        assert converted[2]['content'][0]['response'] == {'result': 'The sum is 6'}

    def test_function_message_not_converted_to_user(self):
        """Regression: tool result MUST be role=tool, NOT role=user."""
        msgs = [
            Message(USER, 'hi'),
            Message(
                role=ASSISTANT, content='',
                tool_calls=[ToolCall(
                    id='c1',
                    function=FunctionCall(name='fn', arguments='{}'),
                )],
            ),
            Message(role=FUNCTION, name='fn', content='result'),
        ]
        converted = _convert_messages_for_use_chat_template_tools(msgs)
        tool_msgs = [m for m in converted if m['role'] == 'tool']
        user_msgs_after_first = [m for m in converted[1:] if m['role'] == 'user']
        assert len(tool_msgs) == 1, 'Tool result must use role=tool'
        assert len(user_msgs_after_first) == 0, 'Tool result must NOT be role=user'


# ---------------------------------------------------------------------------
# Test F: Full-cycle acceptance test (USER -> TOOL_CALL -> TOOL_RESULT -> FINAL)
# ---------------------------------------------------------------------------

class TestFullCycleAcceptance:
    def test_user_to_tool_call_to_result_to_final(self, monkeypatch):
        """Complete cycle: exactly 1 tool call, result fed back natively, final answer produced."""
        llm, FakeTokenizer, gen_count = _install_fake_transformers(monkeypatch, use_chat_template_tools=True)

        # First generate: model emits a tool call
        tool_call_output = '<start_function_call>call:sum_numbers{numbers:[1,2,3,4,5]}<end_function_call>'
        # Second generate: model emits final answer
        final_answer = 'The sum is 15'

        decode_calls = {'n': 0}
        original_batch_decode = llm.tokenizer.batch_decode

        def mock_batch_decode(response, skip_special_tokens=True):
            decode_calls['n'] += 1
            # First call pair (skip=False then skip=True) is for the tool call
            # Second call pair is for the final answer
            call_num = decode_calls['n']
            if call_num <= 2:
                if not skip_special_tokens:
                    return [tool_call_output]
                return [tool_call_output]
            else:
                if not skip_special_tokens:
                    return [final_answer]
                return [final_answer]

        llm.tokenizer.batch_decode = mock_batch_decode
        FakeTokenizer._apply_chat_template_calls.clear()

        # --- Pass 1: User message -> Tool call ---
        messages_pass1 = [Message(USER, 'Sum 1, 2, 3, 4, and 5.')]
        functions = [SUM_NUMBERS_SCHEMA]

        result1 = llm._chat_with_use_chat_template_tools(
            messages_pass1, functions, stream=False, generate_cfg={},
        )

        assert len(result1) == 1
        msg1 = result1[0]
        assert msg1.role == ASSISTANT
        assert msg1.tool_calls is not None
        assert len(msg1.tool_calls) == 1
        assert msg1.tool_calls[0].function.name == 'sum_numbers'
        parsed_args = json.loads(msg1.tool_calls[0].function.arguments)
        assert parsed_args == {'numbers': [1, 2, 3, 4, 5]}

        # --- Simulate tool execution ---
        tool_result_msg = Message(
            role=FUNCTION, name='sum_numbers',
            content='The sum of [1, 2, 3, 4, 5] is 15.',
            tool_call_id=msg1.tool_calls[0].id,
        )

        # --- Pass 2: Tool result -> Final answer ---
        messages_pass2 = messages_pass1 + [msg1, tool_result_msg]
        result2 = llm._chat_with_use_chat_template_tools(
            messages_pass2, functions, stream=False, generate_cfg={},
        )

        assert len(result2) == 1
        msg2 = result2[0]
        assert msg2.role == ASSISTANT
        assert msg2.tool_calls is None or len(msg2.tool_calls) == 0
        assert msg2.content == final_answer

        # --- Verify apply_chat_template was called correctly on pass 2 ---
        assert len(FakeTokenizer._apply_chat_template_calls) == 2
        pass2_call = FakeTokenizer._apply_chat_template_calls[1]
        pass2_msgs = pass2_call['messages']

        # Find the tool result message in what was passed to apply_chat_template
        tool_msgs = [m for m in pass2_msgs if m.get('role') == 'tool']
        assert len(tool_msgs) == 1, (
            'Tool result must be passed as role=tool to apply_chat_template'
        )
        assert tool_msgs[0]['content'][0]['name'] == 'sum_numbers'

        # Find the assistant message with tool_calls
        assistant_with_calls = [
            m for m in pass2_msgs
            if m.get('role') == 'assistant' and 'tool_calls' in m
        ]
        assert len(assistant_with_calls) == 1, (
            'Assistant tool_calls message must be present in second pass'
        )


# ---------------------------------------------------------------------------
# Test G: Regression — tool result must NOT be a generic user message
# ---------------------------------------------------------------------------

class TestRegressionToolResultFormat:
    def test_tool_result_not_passed_as_user_message(self, monkeypatch):
        """If tool result were passed as role=user, this test MUST fail."""
        llm, FakeTokenizer, _ = _install_fake_transformers(monkeypatch, use_chat_template_tools=True)
        FakeTokenizer._apply_chat_template_calls.clear()

        # Simulate a conversation with a completed tool call
        messages = [
            Message(USER, 'Sum 1,2,3'),
            Message(
                role=ASSISTANT, content='',
                tool_calls=[ToolCall(
                    id='c1',
                    function=FunctionCall(name='sum_numbers', arguments='{"numbers":[1,2,3]}'),
                )],
            ),
            Message(role=FUNCTION, name='sum_numbers', content='The sum is 6'),
        ]

        llm.tokenizer.batch_decode = lambda r, skip_special_tokens=True: ['The sum is 6']
        llm._chat_with_use_chat_template_tools(messages, [SUM_NUMBERS_SCHEMA], stream=False, generate_cfg={})

        assert len(FakeTokenizer._apply_chat_template_calls) == 1
        passed_msgs = FakeTokenizer._apply_chat_template_calls[0]['messages']

        # The FUNCTION message MUST become role=tool, NOT role=user
        roles_after_user = [m['role'] for m in passed_msgs[1:]]
        assert 'tool' in roles_after_user, (
            f'Expected role=tool in messages but got roles: {roles_after_user}'
        )
        # No user message should appear after the first one
        user_count = sum(1 for m in passed_msgs if m['role'] == 'user')
        assert user_count == 1, (
            f'Tool result was converted to a user message! Roles: {[m["role"] for m in passed_msgs]}'
        )

    def test_no_regression_without_use_chat_template_tools(self, monkeypatch):
        """use_chat_template_tools=False must use the existing NousFnCallPrompt path."""
        llm, FakeTokenizer, _ = _install_fake_transformers(monkeypatch, use_chat_template_tools=False)
        assert llm._use_chat_template_tools is False
        # _chat_with_use_chat_template_tools should NOT be called via _chat_with_functions
        # when use_chat_template_tools is False — it falls through to prompt path
        assert not llm._use_chat_template_tools


# ---------------------------------------------------------------------------
# Test: stream=True wraps result in iterator
# ---------------------------------------------------------------------------

class TestStreamMode:
    def test_stream_returns_iterator(self, monkeypatch):
        llm, _, _ = _install_fake_transformers(monkeypatch, use_chat_template_tools=True)
        llm.tokenizer.batch_decode = lambda r, skip_special_tokens=True: ['hello']

        result = llm._chat_with_use_chat_template_tools(
            [Message(USER, 'hi')], [SUM_NUMBERS_SCHEMA],
            stream=True, generate_cfg={},
        )
        chunks = list(result)
        assert len(chunks) == 1
        assert chunks[0][0].role == ASSISTANT
