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
from typing import List, Tuple

from cat_agent.tools.base import register_tool
from cat_agent.tools.doc_parser import Record
from cat_agent.tools.search_tools.base_search import BaseSearch


@register_tool('vector_search')
class VectorSearch(BaseSearch):
    # TODO: Optimize the accuracy of the embedding retriever.

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._index_cache = OrderedDict()
        self._index_cache_size = max(1, int(self.cfg.get('index_cache_size', 2)))

    def sort_by_scores(self, query: str, docs: List[Record], **kwargs) -> List[Tuple[str, int, float]]:
        # TODO: More types of embedding can be configured
        try:
            from langchain.schema import Document
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Please install langchain by: `pip install langchain`')
        try:
            from langchain_community.embeddings import OpenAIEmbeddings
            from langchain_community.vectorstores import FAISS
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Please install langchain_community by: `pip install langchain_community`, '
                'and install faiss by: `pip install faiss-cpu` or `pip install faiss-gpu` (for CUDA supported GPU)')
        # Extract raw query
        try:
            query_json = json.loads(query)
            # This assumes that the user's input will not contain json str with the 'text' attribute
            if 'text' in query_json:
                query = query_json['text']
        except json.decoder.JSONDecodeError:
            pass

        api_key = os.getenv('OPENAI_API_KEY', '')
        cache_key = _corpus_fingerprint(docs, api_key)
        cached = self._index_cache.get(cache_key)
        if cached is None:
            all_chunks = [
                Document(page_content=chk.content[:2000], metadata=chk.metadata)
                for doc in docs
                for chk in doc.raw
            ]
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            db = FAISS.from_documents(all_chunks, embeddings)
            cached = (db, len(all_chunks))
            self._index_cache[cache_key] = cached
            self._index_cache.move_to_end(cache_key)
            while len(self._index_cache) > self._index_cache_size:
                self._index_cache.popitem(last=False)
        else:
            self._index_cache.move_to_end(cache_key)

        db, chunk_count = cached
        chunk_and_score = db.similarity_search_with_score(query, k=chunk_count)

        return [(chk.metadata['source'], chk.metadata['chunk_id'], score) for chk, score in chunk_and_score]


def _corpus_fingerprint(docs: List[Record], api_key: str) -> str:
    digest = sha256(api_key.encode('utf-8'))
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
