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

"""Coverage for mlx_lm backend (mocked mlx-lm)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import cat_agent.llm.mlx_lm_llm as mlx_mod
from cat_agent.llm.base import ModelServiceError
from cat_agent.llm.schema import ASSISTANT, USER, Message


def test_mlx_missing_dependency(monkeypatch):
    monkeypatch.setattr(mlx_mod, '_HAS_MLX_LM', False)
    with pytest.raises(ModelServiceError, match='mlx-lm'):
        mlx_mod.MLXLm({'model': 'x'})


def test_mlx_requires_model_id(monkeypatch):
    monkeypatch.setattr(mlx_mod, '_HAS_MLX_LM', True)
    monkeypatch.setattr(mlx_mod, 'load', MagicMock())
    with pytest.raises(ValueError, match='requires `model`'):
        mlx_mod.MLXLm({})


def _make_llm(monkeypatch, *, with_chat_template=True):
    monkeypatch.setattr(mlx_mod, '_HAS_MLX_LM', True)

    tok = MagicMock()
    if with_chat_template:
        tok.apply_chat_template = MagicMock(return_value='PROMPT')
    else:
        del tok.apply_chat_template

    model = MagicMock()
    monkeypatch.setattr(mlx_mod, 'load', MagicMock(return_value=(model, tok)))
    monkeypatch.setattr(mlx_mod, 'make_sampler', MagicMock(return_value='SAMPLER'))
    llm = mlx_mod.MLXLm({'model': 'org/m', 'generate_cfg': {'max_new_tokens': 8}})
    return llm, model, tok


def test_mlx_convert_prepare_build_and_chat(monkeypatch):
    llm, model, tok = _make_llm(monkeypatch)

    assert llm.support_multimodal_input is False
    assert llm.support_audio_input is False
    assert llm.supports_native_tools is False

    converted = llm._convert_messages([
        Message(USER, 'hi'),
        {'role': 'assistant', 'content': [{'text': 'a'}, {'text': 'b'}]},
        {'role': 'user', 'content': 123},
    ])
    assert converted[0]['content'] == 'hi'
    assert converted[1]['content'] == 'ab'
    assert converted[2]['content'] == '123'

    kwargs = llm._prepare_generate_kwargs({
        'seed': 1,
        'max_new_tokens': 16,
        'temperature': 0.5,
        'top_p': 0.9,
        'stop': ['END'],
        'top_k': 5,
        'extra': 1,
    })
    assert kwargs['max_tokens'] == 16
    assert kwargs['stop'] == ['END']
    assert kwargs['sampler'] == 'SAMPLER'
    assert 'seed' not in kwargs

    prompt = llm._build_prompt([Message(USER, 'q')])
    assert prompt == 'PROMPT'

    monkeypatch.setattr(
        mlx_mod,
        'stream_generate',
        MagicMock(return_value=[SimpleNamespace(text='Hel'), 'lo', SimpleNamespace(text='')]),
    )
    deltas = list(llm._chat_stream([Message(USER, 'q')], delta_stream=True, generate_cfg={}))
    assert [m[0].content for m in deltas] == ['Hel', 'lo']

    monkeypatch.setattr(
        mlx_mod,
        'stream_generate',
        MagicMock(return_value=[SimpleNamespace(text='A'), SimpleNamespace(text='B')]),
    )
    full = list(llm._chat_stream([Message(USER, 'q')], delta_stream=False, generate_cfg={}))
    assert full[-1][0].content == 'AB'

    monkeypatch.setattr(mlx_mod, 'generate', MagicMock(return_value='DONE'))
    assert llm._chat_no_stream([Message(USER, 'q')])[0].content == 'DONE'

    monkeypatch.setattr(mlx_mod, 'generate', MagicMock(return_value=SimpleNamespace(text='OBJ')))
    assert llm._chat_no_stream([Message(USER, 'q')])[0].content == 'OBJ'


def test_mlx_prompt_fallback_without_chat_template(monkeypatch):
    llm, _, _ = _make_llm(monkeypatch, with_chat_template=False)
    prompt = llm._build_prompt([Message(USER, 'hello')])
    assert 'USER: hello' in prompt
    assert prompt.endswith('ASSISTANT:')
