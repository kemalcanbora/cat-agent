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

import json
import os
import tempfile
from collections import OrderedDict
from hashlib import sha256
from importlib import import_module
from typing import List, Tuple

from cat_agent.log import logger
from cat_agent.security.encrypted_files import file_exists, read_bytes, read_json, write_bytes, write_json
from cat_agent.settings import DEFAULT_WORKSPACE
from cat_agent.tools.base import register_tool
from cat_agent.tools.doc_parser import Record
from cat_agent.tools.search_tools.base_search import BaseSearch
from cat_agent.tools.search_tools.embedding import build_embedder


def _native():
    try:
        return import_module('cat_agent._native')
    except ImportError as error:
        raise ImportError(
            'VectorSearch requires the cat_agent native Rust extension. '
            'Install a platform wheel or build it with: '
            '`maturin develop --manifest-path native/Cargo.toml`'
        ) from error


@register_tool('vector_search')
class VectorSearch(BaseSearch):
    """Semantic retrieval backed by a native HNSW index (usearch)."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._index_cache = OrderedDict()
        self._index_cache_size = max(1, int(self.cfg.get('index_cache_size', 2)))
        self.rebuild_rag = self.cfg.get('rebuild_rag')
        self.embedder = build_embedder(self.cfg)
        self.index_path = self.cfg.get(
            'vector_index_path',
            os.path.join(DEFAULT_WORKSPACE, 'storage', 'vector_indexes', 'vector_index.usearch'),
        )
        self.meta_path = self.cfg.get('vector_meta_path', f'{self.index_path}.meta.json')
        self.metric = self.cfg.get('vector_metric', 'cos')

    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, int, float]]:
        try:
            query_json = json.loads(query)
            if 'text' in query_json:
                query = query_json['text']
        except json.decoder.JSONDecodeError:
            pass

        cache_key = _corpus_fingerprint(docs, self.embedder.dimensions, self.metric)
        cached = self._index_cache.get(cache_key)
        if cached is None:
            all_chunks = [chunk for doc in docs for chunk in doc.raw]
            texts = [chunk.content[:2000] for chunk in all_chunks]
            vectors = self.embedder.embed(texts)
            index = self._load_or_build_index(vectors, all_chunks, cache_key)
            cached = (all_chunks, index)
            self._index_cache[cache_key] = cached
            self._index_cache.move_to_end(cache_key)
            while len(self._index_cache) > self._index_cache_size:
                self._index_cache.popitem(last=False)
        else:
            self._index_cache.move_to_end(cache_key)

        all_chunks, index = cached
        query_vector = self.embedder.embed([query])[0]
        matches = index.search(query_vector, len(all_chunks))
        key_to_chunk = {
            idx: (chunk.metadata['source'], chunk.metadata['chunk_id'])
            for idx, chunk in enumerate(all_chunks)
        }
        chunk_and_score = [
            (*key_to_chunk[key], _distance_to_score(float(score), self.metric))
            for key, score in matches
            if key in key_to_chunk
        ]
        chunk_and_score.sort(key=lambda item: item[2], reverse=True)
        return chunk_and_score

    def _load_or_build_index(self, vectors, all_chunks, cache_key):
        vector_index = _native().VectorIndex
        metadata = self._load_metadata()
        can_reuse = (
            self.rebuild_rag is not True
            and metadata.get('fingerprint') == cache_key
            and metadata.get('chunk_count') == len(all_chunks)
            and metadata.get('dimensions') == self.embedder.dimensions
            and file_exists(self.index_path)
        )
        if can_reuse:
            try:
                logger.info('[VectorSearch] Reusing native HNSW index at {}', self.index_path)
                return self._load_index_from_disk()
            except (OSError, RuntimeError, ValueError) as error:
                logger.warning('[VectorSearch] Failed to load persisted index; rebuilding: {}', error)

        logger.info('[VectorSearch] Building native HNSW index at {}', self.index_path)
        index = vector_index(self.embedder.dimensions, self.metric)
        keys = list(range(len(vectors)))
        index.add(keys, vectors)
        self._save_index(index, all_chunks, cache_key)
        return index

    def _load_index_from_disk(self):
        vector_index = _native().VectorIndex
        data = read_bytes(self.index_path)
        if data is None:
            raise FileNotFoundError(self.index_path)
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(data)
            temp_path = handle.name
        try:
            return vector_index.load(temp_path, self.embedder.dimensions, self.metric)
        finally:
            os.unlink(temp_path)

    def _save_index(self, index, all_chunks, cache_key) -> None:
        os.makedirs(os.path.dirname(self.index_path) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.meta_path) or '.', exist_ok=True)
        index_tmp = f'{self.index_path}.tmp'
        metadata = {
            'fingerprint': cache_key,
            'chunk_count': len(all_chunks),
            'dimensions': self.embedder.dimensions,
            'metric': self.metric,
            'chunks': [
                [chunk.metadata.get('source', ''), chunk.metadata.get('chunk_id')]
                for chunk in all_chunks
            ],
        }
        try:
            index.save(index_tmp)
            with open(index_tmp, 'rb') as handle:
                write_bytes(self.index_path, handle.read())
            write_json(self.meta_path, metadata)
        except (OSError, TypeError) as error:
            logger.warning('[VectorSearch] Failed to persist native HNSW index: {}', error)
        finally:
            if os.path.isfile(index_tmp):
                os.remove(index_tmp)

    def _load_metadata(self) -> dict:
        return read_json(self.meta_path)


def _corpus_fingerprint(docs: List[Record], dimensions: int, metric: str) -> str:
    digest = sha256(f'{dimensions}:{metric}'.encode('utf-8'))
    for doc in docs:
        digest.update(doc.url.encode('utf-8', errors='surrogatepass'))
        digest.update(b'\0')
        for chunk in doc.raw:
            digest.update(str(chunk.metadata.get('source', '')).encode('utf-8', errors='surrogatepass'))
            digest.update(b'\0')
            digest.update(str(chunk.metadata.get('chunk_id', '')).encode('ascii', errors='replace'))
            digest.update(b'\0')
            digest.update(chunk.content[:2000].encode('utf-8', errors='surrogatepass'))
            digest.update(b'\xff')
    return digest.hexdigest()


def _distance_to_score(distance: float, metric: str) -> float:
    if metric.lower() in ('cos', 'cosine'):
        return 1.0 - distance
    return -distance
