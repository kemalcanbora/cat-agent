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
from collections import OrderedDict
from hashlib import sha256
from importlib import import_module
from typing import List, Tuple

import json5

from cat_agent.log import logger
from cat_agent.settings import DEFAULT_MAX_REF_TOKEN, DEFAULT_WORKSPACE
from cat_agent.tools.base import register_tool
from cat_agent.tools.doc_parser import Record
from cat_agent.tools.search_tools.base_search import BaseSearch


def _native():
    try:
        return import_module('cat_agent._native')
    except ImportError as error:
        raise ImportError(
            'KeywordSearch requires the cat_agent native Rust extension. '
            'Install a platform wheel or build it with: '
            '`maturin develop --manifest-path native/Cargo.toml`'
        ) from error


@register_tool('keyword_search')
class KeywordSearch(BaseSearch):
    """BM25 retrieval backed by the mandatory Rust index and tokenizer."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._index_cache = OrderedDict()
        self._index_cache_size = max(1, int(self.cfg.get('index_cache_size', 4)))
        self.rebuild_rag = self.cfg.get('rebuild_rag')
        self.index_path = self.cfg.get(
            'keyword_index_path',
            os.path.join(DEFAULT_WORKSPACE, 'storage', 'keyword_indexes', 'rag_index.json'),
        )
        self.meta_path = self.cfg.get('keyword_meta_path', f'{self.index_path}.meta.json')

    def search(self, query: str, docs: List[Record], max_ref_token: int = DEFAULT_MAX_REF_TOKEN) -> list:
        chunk_and_score = self.sort_by_scores(query=query, docs=docs)
        if not chunk_and_score:
            return self._get_the_front_part(docs, max_ref_token)

        max_sims = chunk_and_score[0][-1]

        if max_sims != 0:
            return super().get_topk(chunk_and_score=chunk_and_score, docs=docs, max_ref_token=max_ref_token)
        else:
            return self._get_the_front_part(docs, max_ref_token)

    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, int, float]]:
        wordlist = parse_keyword(query)
        logger.debug('wordlist: ' + ','.join(wordlist))
        if not wordlist:
            return []

        cache_key = _corpus_fingerprint(docs)
        cached = self._index_cache.get(cache_key)
        if cached is None:
            all_chunks = [chunk for doc in docs for chunk in doc.raw]
            tokenized_corpus = [split_text_into_keywords(chunk.content) for chunk in all_chunks]
            index = self._load_or_build_index(tokenized_corpus, all_chunks, cache_key)
            cached = (all_chunks, index)
            self._index_cache[cache_key] = cached
            self._index_cache.move_to_end(cache_key)
            while len(self._index_cache) > self._index_cache_size:
                self._index_cache.popitem(last=False)
        else:
            self._index_cache.move_to_end(cache_key)

        all_chunks, index = cached
        doc_scores = index.scores(wordlist)
        chunk_and_score = [
            (chk.metadata['source'], chk.metadata['chunk_id'], score) for chk, score in zip(all_chunks, doc_scores)
        ]
        chunk_and_score.sort(key=lambda item: item[2], reverse=True)
        assert len(chunk_and_score) > 0

        return chunk_and_score

    def _load_or_build_index(self, tokenized_corpus, all_chunks, cache_key):
        rag_index = _native().RagIndex
        metadata = self._load_metadata()
        can_reuse = (
            self.rebuild_rag is not True
            and metadata.get('fingerprint') == cache_key
            and metadata.get('chunk_count') == len(all_chunks)
            and os.path.isfile(self.index_path)
        )
        if can_reuse:
            try:
                logger.info(f'[KeywordSearch] Reusing Rust BM25 index at {self.index_path}')
                return rag_index.load(self.index_path)
            except (OSError, RuntimeError, ValueError) as error:
                logger.warning(f'[KeywordSearch] Failed to load persisted index; rebuilding: {error}')

        logger.info(f'[KeywordSearch] Building Rust BM25 index at {self.index_path}')
        index = rag_index(tokenized_corpus)
        self._save_index(index, all_chunks, cache_key)
        return index

    def _save_index(self, index, all_chunks, cache_key) -> None:
        os.makedirs(os.path.dirname(self.index_path) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.meta_path) or '.', exist_ok=True)
        index_tmp = f'{self.index_path}.tmp'
        meta_tmp = f'{self.meta_path}.tmp'
        metadata = {
            'fingerprint': cache_key,
            'chunk_count': len(all_chunks),
            'chunks': [
                [chunk.metadata.get('source', ''), chunk.metadata.get('chunk_id')]
                for chunk in all_chunks
            ],
        }
        try:
            index.save(index_tmp)
            os.replace(index_tmp, self.index_path)
            with open(meta_tmp, 'w', encoding='utf-8') as file:
                json.dump(metadata, file, ensure_ascii=False)
            os.replace(meta_tmp, self.meta_path)
        except (OSError, TypeError) as error:
            logger.warning(f'[KeywordSearch] Failed to persist Rust BM25 index: {error}')
            for path in (index_tmp, meta_tmp):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _load_metadata(self) -> dict:
        try:
            with open(self.meta_path, encoding='utf-8') as file:
                metadata = json.load(file)
        except (OSError, ValueError, TypeError):
            return {}
        return metadata if isinstance(metadata, dict) else {}


def _corpus_fingerprint(docs: List[Record]) -> str:
    digest = sha256()
    for doc in docs:
        digest.update(doc.url.encode('utf-8', errors='surrogatepass'))
        digest.update(b'\0')
        for chunk in doc.raw:
            digest.update(str(chunk.metadata.get('source', '')).encode('utf-8', errors='surrogatepass'))
            digest.update(b'\0')
            digest.update(str(chunk.metadata.get('chunk_id', '')).encode('ascii', errors='replace'))
            digest.update(b'\0')
            digest.update(chunk.content.encode('utf-8', errors='surrogatepass'))
            digest.update(b'\xff')
    return digest.hexdigest()


def __getattr__(name: str):
    if name == 'WORDS_TO_IGNORE':
        return _native().WORDS_TO_IGNORE
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def clean_en_token(token: str) -> str:
    return _native().clean_en_token(token)


def tokenize_and_filter(input_text: str) -> List[str]:
    return _native().tokenize_and_filter(input_text)


def split_text_into_keywords(text: str) -> List[str]:
    return _native().split_text_into_keywords(text)


def string_tokenizer(text: str) -> List[str]:
    return _native().string_tokenizer(text)


def parse_keyword(text):
    try:
        res = json5.loads(text)
    except Exception:
        return split_text_into_keywords(text)

    try:
        native = _native()
        _wordlist = []
        if 'keywords_zh' in res and isinstance(res['keywords_zh'], list):
            _wordlist.extend([kw.lower() for kw in res['keywords_zh']])
        if 'keywords_en' in res and isinstance(res['keywords_en'], list):
            _wordlist.extend([kw.lower() for kw in res['keywords_en']])
        _wordlist = native.stem_words(_wordlist)
        wordlist = [word for word in _wordlist if word not in __getattr__('WORDS_TO_IGNORE')]
        wordlist += split_text_into_keywords(res.get('text', ''))
        return wordlist
    except Exception:
        return split_text_into_keywords(text)
