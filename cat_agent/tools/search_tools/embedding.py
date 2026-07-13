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

"""Local embedding backends for native vector search."""

from __future__ import annotations

import hashlib
import math
from typing import List, Protocol


class Embedder(Protocol):
    dimensions: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class HashEmbedder:
    """Deterministic bag-of-keywords embedding (no external model required)."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def embed(self, texts: List[str]) -> List[List[float]]:
        from cat_agent.tools.search_tools.keyword_search import split_text_into_keywords

        vectors: List[List[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for token in split_text_into_keywords(text[:2000]):
                index = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16) % self.dimensions
                vector[index] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 0:
                vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors


class OnnxEmbedder:
    """Optional ONNX Runtime embedder for semantic vectors."""

    def __init__(self, model_path: str, dimensions: int | None = None):
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError(
                'ONNX embeddings require onnxruntime. Install with: pip install onnxruntime'
            ) from error

        self._np = np
        self._session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self._input_names = [item.name for item in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name
        inferred = self._session.get_outputs()[0].shape
        if dimensions is not None:
            self.dimensions = dimensions
        elif isinstance(inferred[-1], int):
            self.dimensions = inferred[-1]
        else:
            self.dimensions = 384

    def embed(self, texts: List[str]) -> List[List[float]]:
        from cat_agent.utils.tokenization_qwen import tokenizer

        vectors: List[List[float]] = []
        for text in texts:
            token_ids = tokenizer.encode(text[:2000])
            if not token_ids:
                vectors.append([0.0] * self.dimensions)
                continue
            feed = {}
            if 'input_ids' in self._input_names:
                input_ids = self._np.array([token_ids], dtype=self._np.int64)
                feed['input_ids'] = input_ids
            if 'attention_mask' in self._input_names:
                feed['attention_mask'] = self._np.ones_like(feed.get('input_ids', input_ids))
            if 'token_type_ids' in self._input_names:
                feed['token_type_ids'] = self._np.zeros_like(feed['input_ids'])
            if not feed:
                raise ValueError(
                    f'Unsupported ONNX embedding model inputs: {self._input_names}. '
                    'Expected at least input_ids.'
                )
            output = self._session.run([self._output_name], feed)[0]
            if output.ndim == 3:
                pooled = output.mean(axis=1)[0]
            else:
                pooled = output[0]
            if pooled.shape[-1] != self.dimensions:
                raise ValueError(
                    f'ONNX model output dimension {pooled.shape[-1]} '
                    f'does not match configured dimensions {self.dimensions}'
                )
            norm = self._np.linalg.norm(pooled)
            if norm > 0:
                pooled = pooled / norm
            vectors.append(pooled.astype(float).tolist())
        return vectors


def build_embedder(cfg: dict | None = None) -> Embedder:
    cfg = cfg or {}
    backend = cfg.get('embedding_backend', 'hash')
    if backend == 'onnx':
        model_path = cfg.get('embedding_model_path')
        if not model_path:
            raise ValueError('embedding_model_path is required when embedding_backend="onnx"')
        return OnnxEmbedder(
            model_path=model_path,
            dimensions=cfg.get('embedding_dimensions'),
        )
    return HashEmbedder(dimensions=int(cfg.get('embedding_dimensions', 384)))
