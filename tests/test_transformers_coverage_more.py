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

"""Extra coverage for transformers LLM (mocked transformers/torch)."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.llm.schema import ASSISTANT, USER, Message
from cat_agent.llm.transformers_llm import Transformers, _format_transformers_import_error


def _install_fake_transformers(monkeypatch, *, multimodal=False, multi_arch=False):
    class PreTrainedTokenizer:
        pass

    class PreTrainedTokenizerFast:
        pass

    class FakeTokenizer(PreTrainedTokenizer):
        def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None):
            tensor = MagicMock()
            tensor.to = MagicMock(return_value=tensor)
            enc = MagicMock()
            enc.input_ids = tensor
            return enc

        def batch_decode(self, response, skip_special_tokens=True):
            return ['decoded-answer']

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()
            self.feature_extractor = SimpleNamespace(sampling_rate=16000)

        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
            return 'PROMPT'

        def __call__(self, **kwargs):
            t = MagicMock()
            return {'input_ids': t, 'pixel_values': t}

    class FakeModel:
        def __init__(self):
            self.device = 'cpu'

        def to(self, device):
            self.device = device
            return self

        def generate(self, **kwargs):
            out = MagicMock()
            sliced = MagicMock()
            out.__getitem__ = MagicMock(return_value=sliced)
            return out

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

    class FakeConfig:
        architectures = ['FakeModel', 'Other'] if multi_arch else ['FakeModel']

    class AutoConfig:
        @staticmethod
        def from_pretrained(model):
            return FakeConfig()

    class AutoProcessor:
        @staticmethod
        def from_pretrained(model):
            if multimodal:
                return FakeProcessor()
            return FakeTokenizer()

    class TextIteratorStreamer:
        def __init__(self, *a, **k):
            self._items = ['Hel', 'lo']

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
    return transformers, torch, FakeTokenizer, FakeProcessor


def test_format_import_error_walks_cause():
    root = AttributeError('root-cause')
    mid = ImportError('mid')
    mid.__cause__ = root
    top = ImportError('top')
    top.__cause__ = mid
    msg = _format_transformers_import_error(top)
    assert 'root-cause' in msg
    assert 'transformers' in msg.lower()


def test_init_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, 'transformers', None)
    # Force ImportError from `import transformers`
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == 'transformers' or name.startswith('transformers.'):
            raise ImportError('no tf')
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', fake)
    with pytest.raises(ImportError, match='HuggingFace Transformers'):
        Transformers({'model': 'x'})


def test_init_text_only_and_properties(monkeypatch):
    _install_fake_transformers(monkeypatch, multimodal=False, multi_arch=True)
    llm = Transformers({'model': 'org/m', 'device': 'cpu'})
    assert llm.support_multimodal_input is False
    assert llm.support_audio_input is False
    assert llm.supports_native_tools is False
    assert llm.tokenizer is not None
    assert llm.hf_model is not None


def test_init_multimodal(monkeypatch):
    _install_fake_transformers(monkeypatch, multimodal=True)
    llm = Transformers({'model': 'org/vl', 'device': 'cpu'})
    assert llm.support_multimodal_input is True
    assert hasattr(llm, 'processor')


def test_get_inputs_text_only_tensor_and_encoding(monkeypatch):
    _install_fake_transformers(monkeypatch, multimodal=False)
    llm = Transformers({'model': 'org/m'})

    # Encoding with .input_ids
    inputs = llm._get_inputs([Message(USER, 'hi')])
    assert 'input_ids' in inputs
    assert 'attention_mask' in inputs

    # Bare tensor (no .input_ids) — else branch
    bare = MagicMock(spec=['to'])
    bare.to = MagicMock(return_value=bare)
    llm.tokenizer.apply_chat_template = MagicMock(return_value=bare)
    inputs2 = llm._get_inputs([Message(USER, 'hi')])
    assert inputs2['input_ids'] is bare


def test_get_inputs_multimodal_vision_and_audio(monkeypatch):
    _install_fake_transformers(monkeypatch, multimodal=True)
    llm = Transformers({'model': 'org/vl'})

    qwen = ModuleType('qwen_vl_utils')
    qwen.process_vision_info = MagicMock(return_value=(['img'], ['vid']))
    monkeypatch.setitem(sys.modules, 'qwen_vl_utils', qwen)

    librosa = ModuleType('librosa')
    librosa.load = MagicMock(return_value=([0.1, 0.2], 16000))
    monkeypatch.setitem(sys.modules, 'librosa', librosa)

    # Bypass Message.model_dump None-key noise: feed dicts via patched path
    plain = [{
        'role': 'user',
        'content': [
            {'text': 'describe'},
            {'image': 'http://img'},
            {'audio': 'file:///tmp/a.wav'},
            {'video': 'http://v'},
        ],
    }]
    with patch.object(Message, 'model_dump', side_effect=lambda *a, **k: plain[0]):
        # model_dump called per message — return same structure
        msgs = [Message(USER, 'placeholder')]
        with patch.object(msgs[0], 'model_dump', return_value=plain[0]):
            inputs = llm._get_inputs(msgs)
    assert 'input_ids' in inputs
    qwen.process_vision_info.assert_called()
    assert librosa.load.called

    plain2 = [{
        'role': 'user',
        'content': [
            {'text': 'x'},
            {'audio': '/tmp/b.wav'},
        ],
    }]
    msgs2 = [Message(USER, 'x')]
    with patch.object(msgs2[0], 'model_dump', return_value=plain2[0]):
        llm._get_inputs(msgs2)


def test_chat_stream_and_no_stream(monkeypatch):
    _install_fake_transformers(monkeypatch, multimodal=False)
    llm = Transformers({'model': 'org/m'})

    # Make generate a no-op for the background thread
    llm.hf_model.generate = MagicMock(return_value=MagicMock())

    streamer_items = ['A', 'B']

    class Streamer:
        def __iter__(self):
            return iter(streamer_items)

    with patch.object(llm, '_get_streamer', return_value=Streamer()), \
            patch.object(llm, '_get_inputs', return_value={'input_ids': MagicMock()}):
        delta = list(llm._chat_stream(
            [Message(USER, 'q')],
            delta_stream=True,
            generate_cfg={'seed': 1, 'stop': ['\n'], 'max_new_tokens': 8},
        ))
        assert [d[0].content for d in delta] == ['A', 'B']

        full = list(llm._chat_stream(
            [Message(USER, 'q')],
            delta_stream=False,
            generate_cfg={'seed': 2, 'stop': ['x']},
        ))
        assert full[-1][0].content == 'AB'

    # no-stream path
    fake_ids = MagicMock()
    fake_ids.size = MagicMock(return_value=3)
    response = MagicMock()
    response.__getitem__ = MagicMock(return_value=MagicMock())
    llm.hf_model.generate = MagicMock(return_value=response)
    llm.tokenizer.batch_decode = MagicMock(return_value=['final'])
    with patch.object(llm, '_get_inputs', return_value={'input_ids': fake_ids}):
        out = llm._chat_no_stream(
            [Message(USER, 'q')],
            generate_cfg={'seed': 3, 'stop': ['end']},
        )
    assert out[0].role == ASSISTANT
    assert out[0].content == 'final'


def test_get_streamer(monkeypatch):
    _install_fake_transformers(monkeypatch, multimodal=False)
    llm = Transformers({'model': 'org/m'})
    s = llm._get_streamer()
    assert list(s) == ['Hel', 'lo']
