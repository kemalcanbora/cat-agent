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
import time
from typing import Dict, List, Optional, Union

from pydantic import BaseModel

from cat_agent.log import logger
from cat_agent.settings import DEFAULT_MAX_REF_TOKEN, DEFAULT_PARSER_PAGE_SIZE, DEFAULT_WORKSPACE
from cat_agent.tools.base import BaseTool, register_tool
from cat_agent.tools.simple_doc_parser import PARAGRAPH_SPLIT_SYMBOL, SimpleDocParser, get_plain_doc
from cat_agent.tools.storage import KeyNotExistsError, Storage
from cat_agent.utils.tokenization_qwen import count_tokens, ensure_qwen_tokenizer
from cat_agent.utils.utils import get_basename_from_url, hash_sha256


def _native():
    from importlib import import_module

    try:
        return import_module('cat_agent._native')
    except ImportError as error:
        raise ImportError(
            'DocParser chunking requires the cat_agent native Rust extension. '
            'Install a platform wheel or build it with: '
            '`maturin develop --manifest-path native/Cargo.toml`'
        ) from error


class Chunk(BaseModel):
    content: str
    metadata: dict
    token: int

    def __init__(self, content: str, metadata: dict, token: int):
        super().__init__(content=content, metadata=metadata, token=token)

    def to_dict(self) -> dict:
        return {'content': self.content, 'metadata': self.metadata, 'token': self.token}


class Record(BaseModel):
    url: str
    raw: List[Chunk]
    title: str

    def __init__(self, url: str, raw: List[Chunk], title: str):
        super().__init__(url=url, raw=raw, title=title)

    def to_dict(self) -> dict:
        return {'url': self.url, 'raw': [x.to_dict() for x in self.raw], 'title': self.title}


@register_tool('doc_parser')
class DocParser(BaseTool):
    description = 'Extract content from a file and divide it into blocks, returning the chunked content.'
    parameters = {
        'type': 'object',
        'properties': {
            'url': {
                'description': 'The path of the file to be parsed, which can be a local path or a downloadable http(s) link.',
                'type': 'string',
            }
        },
        'required': ['url'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.max_ref_token: int = self.cfg.get('max_ref_token', DEFAULT_MAX_REF_TOKEN)
        self.parser_page_size: int = self.cfg.get('parser_page_size', DEFAULT_PARSER_PAGE_SIZE)

        self.data_root = self.cfg.get('path', os.path.join(DEFAULT_WORKSPACE, 'tools', self.name))
        self.db = Storage({'storage_root_path': self.data_root})

        self.doc_extractor = SimpleDocParser({'structured_doc': True})

    def call(self, params: Union[str, dict], **kwargs) -> dict:
        """Extracting and blocking

        Returns:
            Parse doc as the following chunks:
              {
                'url': 'This is the url of this file',
                'title': 'This is the extracted title of this file',
                'raw': [
                        {
                            'content': 'This is one chunk',
                            'token': 'The token number',
                            'metadata': {}  # some information of this chunk
                        },
                        ...,
                      ]
             }
        """

        params = self._verify_json_format_args(params)
        # Compatible with the parameter passing of the qwen-agent version <= 0.0.3
        max_ref_token = kwargs.get('max_ref_token', self.max_ref_token)
        parser_page_size = kwargs.get('parser_page_size', self.parser_page_size)

        url = params['url']

        cached_name_chunking = f'{hash_sha256(url)}_{str(parser_page_size)}'
        try:
            # Directly load the chunked doc
            record = self.db.get(cached_name_chunking)
            record = json.loads(record)
            logger.info(f'Read chunked {url} from cache.')
            return record
        except KeyNotExistsError:
            doc = self.doc_extractor.call({'url': url})

        total_token = 0
        for page in doc:
            for para in page['content']:
                total_token += para['token']

        if doc and 'title' in doc[0]:
            title = doc[0]['title']
        else:
            title = get_basename_from_url(url)

        logger.info(f'Start chunking {url} ({title})...')
        time1 = time.time()
        if total_token <= max_ref_token:
            # The whole doc is one chunk
            content = [
                Chunk(content=get_plain_doc(doc),
                      metadata={
                          'source': url,
                          'title': title,
                          'chunk_id': 0
                      },
                      token=total_token)
            ]
            cached_name_chunking = f'{hash_sha256(url)}_without_chunking'
        else:
            content = self.split_doc_to_chunk(doc, url, title=title, parser_page_size=parser_page_size)

        time2 = time.time()
        logger.info(f'Finished chunking {url} ({title}). Time spent: {time2 - time1} seconds.')

        # save the document data
        new_record = Record(url=url, raw=content, title=title).to_dict()
        new_record_str = json.dumps(new_record, ensure_ascii=False)
        self.db.put(cached_name_chunking, new_record_str)
        return new_record

    def split_doc_to_chunk(self,
                           doc: List[dict],
                           path: str,
                           title: str = '',
                           parser_page_size: int = DEFAULT_PARSER_PAGE_SIZE) -> List[Chunk]:
        ensure_qwen_tokenizer()
        native_chunks = _native().split_doc_to_chunks(
            doc,
            path,
            title,
            parser_page_size,
            PARAGRAPH_SPLIT_SYMBOL,
        )
        return [
            Chunk(
                content=item['content'],
                metadata=item['metadata'],
                token=item['token'],
            )
            for item in native_chunks
        ]
