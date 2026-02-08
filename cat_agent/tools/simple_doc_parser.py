# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
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

"""SimpleDocParser tool -- thin wrapper around the ``parsers`` package.

Parsing logic lives in ``cat_agent.tools.parsers.*``.
This module provides the registered tool and backward-compatible re-exports.
"""

import json
import os
import re
import time
from typing import Dict, Optional, Union

from cat_agent.log import logger
from cat_agent.settings import DEFAULT_WORKSPACE
from cat_agent.tools.base import BaseTool, register_tool
from cat_agent.tools.storage import KeyNotExistsError, Storage
from cat_agent.utils.tokenization_qwen import count_tokens
from cat_agent.utils.file_utils import get_file_type, is_http_url, sanitize_chrome_file_path, save_url_to_local_work_dir
from cat_agent.utils.misc import hash_sha256

# ---------------------------------------------------------------------------
# Backward-compatible re-exports (other modules import these from here)
# ---------------------------------------------------------------------------
from cat_agent.tools.parsers import (  # noqa: F401
    PARSER_SUPPORTED_FILE_TYPES,
    parse_document,
)
from cat_agent.tools.parsers.base import (  # noqa: F401
    DocParserError,
    PARAGRAPH_SPLIT_SYMBOL,
    clean_paragraph,
    get_plain_doc,
)
# Backward-compatible re-exports (tests and other code may import parsers from here)
from cat_agent.tools.parsers.pdf_parser import parse_pdf  # noqa: F401
from cat_agent.tools.parsers.word_parser import parse_word  # noqa: F401
from cat_agent.tools.parsers.ppt_parser import parse_ppt  # noqa: F401
from cat_agent.tools.parsers.txt_parser import parse_txt  # noqa: F401
from cat_agent.tools.parsers.html_parser import parse_html_bs  # noqa: F401
from cat_agent.tools.parsers.excel_parser import parse_excel, parse_csv, parse_tsv, df_to_md  # noqa: F401


@register_tool('simple_doc_parser')
class SimpleDocParser(BaseTool):
    description = f"Extract the content of a document. Supported types include: {' / '.join(PARSER_SUPPORTED_FILE_TYPES)}"
    parameters = {
        'type': 'object',
        'properties': {
            'url': {
                'description': 'The path of the file to be extracted, which can be a local path or a downloadable http(s) link.',
                'type': 'string',
            }
        },
        'required': ['url'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.data_root = self.cfg.get('path', os.path.join(DEFAULT_WORKSPACE, 'tools', self.name))
        self.extract_image = self.cfg.get('extract_image', False)
        self.structured_doc = self.cfg.get('structured_doc', False)
        self.db = Storage({'storage_root_path': self.data_root})

    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list]:
        """Parse a document by URL/path and return formatted content.

        Returns:
            Extracted doc as plain text or structured page list.
        """
        params = self._verify_json_format_args(params)
        path = params['url']
        cached_name_ori = f'{hash_sha256(path)}_ori'

        try:
            parsed_file = json.loads(self.db.get(cached_name_ori))
            logger.info(f'Read parsed {path} from cache.')
        except KeyNotExistsError:
            logger.info(f'Start parsing {path}...')
            time1 = time.time()

            # Resolve the file path
            path = self._resolve_path(path)

            os.makedirs(self.data_root, exist_ok=True)
            if is_http_url(path):
                tmp_file_root = os.path.join(self.data_root, hash_sha256(path))
                os.makedirs(tmp_file_root, exist_ok=True)
                path = save_url_to_local_work_dir(path, tmp_file_root)

            f_type = get_file_type(path)
            try:
                parsed_file = parse_document(path, extract_image=self.extract_image, file_type=f_type)
            except Exception as ex:
                raise DocParserError(code=type(ex).__name__, message=str(ex))

            # Annotate token counts
            for page in parsed_file:
                for para in page['content']:
                    para['token'] = count_tokens(para.get('text', para.get('table')))

            time2 = time.time()
            logger.info(f'Finished parsing {path}. Time spent: {time2 - time1} seconds.')
            self.db.put(cached_name_ori, json.dumps(parsed_file, ensure_ascii=False, indent=2))

        if not self.structured_doc:
            return get_plain_doc(parsed_file)
        return parsed_file

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Normalise a user-supplied path (handle Chrome file:// URIs, Windows paths, etc.)."""
        f_type = get_file_type(path)
        if f_type in PARSER_SUPPORTED_FILE_TYPES:
            if path.startswith('https://') or path.startswith('http://') or re.match(
                    r'^[A-Za-z]:\\', path) or re.match(r'^[A-Za-z]:/', path):
                return path
            return sanitize_chrome_file_path(path)
        return path
