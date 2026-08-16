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

"""Coverage tests for cat_agent.tools.search_tools.embedding (mocked backends)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cat_agent.tools.search_tools import embedding as emb


def test_native_hash_embed_missing_extension():
    with patch.object(emb, 'import_module', side_effect=ImportError('no native')):
        with pytest.raises(ImportError, match='native Rust extension'):
            emb._native_hash_embed(['a'], 8)


def test_native_hash_embed_delegates():
    native = MagicMock()
    native.hash_embed.return_value = [[0.1, 0.2]]
    with patch.object(emb, 'import_module', return_value=native):
        out = emb._native_hash_embed(['hi'], 2)
    assert out == [[0.1, 0.2]]
    native.hash_embed.assert_called_once_with(['hi'], 2)


def test_hash_embedder_uses_native():
    with patch.object(emb, '_native_hash_embed', return_value=[[1.0, 0.0]]) as native:
        vectors = emb.HashEmbedder(dimensions=2).embed(['x'])
    assert vectors == [[1.0, 0.0]]
    native.assert_called_once_with(['x'], 2)


def test_build_embedder_hash_default():
    e = emb.build_embedder(None)
    assert isinstance(e, emb.HashEmbedder)
    assert e.dimensions == 384

    e2 = emb.build_embedder({'embedding_backend': 'hash', 'embedding_dimensions': 16})
    assert e2.dimensions == 16


def test_build_embedder_onnx_requires_path():
    with pytest.raises(ValueError, match='embedding_model_path'):
        emb.build_embedder({'embedding_backend': 'onnx'})


def test_onnx_embedder_import_error():
    with patch.dict('sys.modules', {'onnxruntime': None}):
        with pytest.raises(ImportError, match='onnxruntime'):
            emb.OnnxEmbedder('/tmp/model.onnx')


def test_onnx_embedder_embed_paths():
    np = MagicMock()
    arr = MagicMock()
    np.array.return_value = arr
    np.ones_like.return_value = arr
    np.zeros_like.return_value = arr
    np.int64 = 'int64'
    np.linalg.norm.return_value = 0.0

    pooled = MagicMock()
    pooled.shape = (4,)
    pooled.astype.return_value.tolist.return_value = [0.25, 0.25, 0.25, 0.25]

    output_3d = MagicMock()
    output_3d.ndim = 3
    output_3d.mean.return_value = [pooled]

    session = MagicMock()
    session.get_inputs.return_value = [
        SimpleNamespace(name='input_ids'),
        SimpleNamespace(name='attention_mask'),
        SimpleNamespace(name='token_type_ids'),
    ]
    session.get_outputs.return_value = [SimpleNamespace(name='last', shape=['batch', 4])]
    session.run.return_value = [output_3d]

    embedder = emb.OnnxEmbedder.__new__(emb.OnnxEmbedder)
    embedder._np = np
    embedder._session = session
    embedder._input_names = ['input_ids', 'attention_mask', 'token_type_ids']
    embedder._output_name = 'last'
    embedder.dimensions = 4

    with patch('cat_agent.utils.tokenization_qwen.tokenizer') as tok:
        tok.encode.side_effect = [[1, 2, 3], []]
        vectors = embedder.embed(['hello', ''])

    assert vectors[0] == [0.25, 0.25, 0.25, 0.25]
    assert vectors[1] == [0.0, 0.0, 0.0, 0.0]


def test_onnx_embedder_2d_output_and_dim_mismatch():
    np = MagicMock()
    arr = MagicMock()
    np.array.return_value = arr
    np.ones_like.return_value = arr
    np.int64 = 'int64'
    np.linalg.norm.return_value = 0.0

    pooled = MagicMock()
    pooled.shape = (8,)

    output_2d = MagicMock()
    output_2d.ndim = 2
    output_2d.__getitem__.return_value = pooled

    session = MagicMock()
    session.get_inputs.return_value = [SimpleNamespace(name='input_ids')]
    session.get_outputs.return_value = [SimpleNamespace(name='out', shape=[None, 8])]
    session.run.return_value = [output_2d]

    embedder = emb.OnnxEmbedder.__new__(emb.OnnxEmbedder)
    embedder._np = np
    embedder._session = session
    embedder._input_names = ['input_ids']
    embedder._output_name = 'out'
    embedder.dimensions = 4

    with patch('cat_agent.utils.tokenization_qwen.tokenizer') as tok:
        tok.encode.return_value = [1]
        with pytest.raises(ValueError, match='does not match configured dimensions'):
            embedder.embed(['x'])


def test_onnx_embedder_unsupported_inputs():
    np = MagicMock()
    session = MagicMock()
    session.get_inputs.return_value = [SimpleNamespace(name='weird')]
    session.get_outputs.return_value = [SimpleNamespace(name='out', shape=[4])]

    embedder = emb.OnnxEmbedder.__new__(emb.OnnxEmbedder)
    embedder._np = np
    embedder._session = session
    embedder._input_names = ['weird']
    embedder._output_name = 'out'
    embedder.dimensions = 4

    with patch('cat_agent.utils.tokenization_qwen.tokenizer') as tok:
        tok.encode.return_value = [1]
        with pytest.raises(ValueError, match='Unsupported ONNX'):
            embedder.embed(['x'])


def test_build_embedder_onnx_wires_constructor():
    fake = MagicMock()
    with patch.object(emb, 'OnnxEmbedder', return_value=fake) as ctor:
        out = emb.build_embedder({
            'embedding_backend': 'onnx',
            'embedding_model_path': '/m.onnx',
            'embedding_dimensions': 32,
        })
    assert out is fake
    ctor.assert_called_once_with(model_path='/m.onnx', dimensions=32)


def test_onnx_embedder_init_infers_dimensions():
    session = MagicMock()
    session.get_inputs.return_value = [SimpleNamespace(name='input_ids')]
    session.get_outputs.return_value = [SimpleNamespace(name='out', shape=[None, 64])]

    ort = MagicMock()
    ort.InferenceSession.return_value = session

    # Stub only onnxruntime. Replacing a live ``numpy`` entry in ``sys.modules``
    # (then restoring/removing it) can leave NumPy's C extension initialized
    # while the package is gone from ``sys.modules``. NumPy 2.4+ then raises
    # ``ImportError: cannot load module more than once per process`` on the
    # next real import (e.g. llama_cpp / rank_bm25 under pytest-cov).
    with patch.dict('sys.modules', {'onnxruntime': ort}):
        embedder = emb.OnnxEmbedder('/tmp/m.onnx')
    assert embedder.dimensions == 64

    session.get_outputs.return_value = [SimpleNamespace(name='out', shape=['batch', 'dim'])]
    with patch.dict('sys.modules', {'onnxruntime': ort}):
        embedder2 = emb.OnnxEmbedder('/tmp/m.onnx', dimensions=None)
    assert embedder2.dimensions == 384
