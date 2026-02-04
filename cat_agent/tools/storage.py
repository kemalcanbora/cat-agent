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

import dbm
import os
from typing import Dict, Optional, Union

from cat_agent.settings import DEFAULT_WORKSPACE
from cat_agent.tools.base import BaseTool, register_tool


class KeyNotExistsError(ValueError):
    pass


def _norm_key(key: str) -> str:
    """Strip leading slash so keys are stored consistently."""
    return key[1:] if key.startswith('/') else key


@register_tool('storage')
class Storage(BaseTool):
    """
    This is a special tool for data storage (backed by stdlib dbm).
    """
    description = '存储和读取数据的工具'
    parameters = {
        'type': 'object',
        'properties': {
            'operate': {
                'description': '数据操作类型，可选项为["put", "get", "delete", "scan"]之一，分别为存数据、取数据、删除数据、遍历数据',
                'type': 'string',
            },
            'key': {
                'description': '数据的路径，类似于文件路径，是一份数据的唯一标识，不能为空，默认根目录为`/`。存数据时，应该合理的设计路径，保证路径含义清晰且唯一。',
                'type': 'string',
                'default': '/'
            },
            'value': {
                'description': '数据的内容，仅存数据时需要',
                'type': 'string',
            },
        },
        'required': ['operate'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        root = self.cfg.get('storage_root_path', os.path.join(DEFAULT_WORKSPACE, 'tools', self.name))
        os.makedirs(root, exist_ok=True)
        # dbm stores in a single file (e.g. storage.db); 'c' = create if missing
        self._db_path = os.path.join(root, 'storage.db')

    def _open_db(self, mode: str = 'c'):
        return dbm.open(self._db_path, mode)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        operate = params['operate']
        key = _norm_key(params.get('key', '/'))

        if operate == 'put':
            assert 'value' in params
            return self.put(key, params['value'])
        elif operate == 'get':
            return self.get(key)
        elif operate == 'delete':
            return self.delete(key)
        else:
            return self.scan(key)

    def put(self, key: str, value: str, path: Optional[str] = None) -> str:
        if path is not None:
            # Legacy: when path is passed, use that dir for the db (e.g. tests)
            db_path = os.path.join(path, 'storage.db')
            os.makedirs(path, exist_ok=True)
            with dbm.open(db_path, 'c') as db:
                db[key.encode('utf-8')] = value.encode('utf-8')
        else:
            with self._open_db('c') as db:
                db[key.encode('utf-8')] = value.encode('utf-8')
        return f'Successfully saved {key}.'

    def get(self, key: str, path: Optional[str] = None) -> str:
        db_path = self._db_path if path is None else os.path.join(path, 'storage.db')
        with dbm.open(db_path, 'c') as db:
            kb = key.encode('utf-8')
            if kb not in db:
                raise KeyNotExistsError(f'Get Failed: {key} does not exist')
            return db[kb].decode('utf-8')

    def delete(self, key: str, path: Optional[str] = None) -> str:
        db_path = self._db_path if path is None else os.path.join(path, 'storage.db')
        with dbm.open(db_path, 'c') as db:
            kb = key.encode('utf-8')
            if kb not in db:
                return f'Delete Failed: {key} does not exist'
            del db[kb]
        return f'Successfully deleted {key}'

    def scan(self, key: str, path: Optional[str] = None) -> str:
        db_path = self._db_path if path is None else os.path.join(path, 'storage.db')
        prefix = (key.rstrip('/') + '/') if key else ''
        with dbm.open(db_path, 'c') as db:
            kvs = {}
            for k in db.keys():
                k_str = k.decode('utf-8')
                if k_str == key or (prefix and k_str.startswith(prefix)):
                    rel = '/' + k_str[len(key):].lstrip('/') if key and k_str.startswith(key) else '/' + k_str
                    kvs[rel] = db[k].decode('utf-8')
            if not kvs:
                return f'Scan Failed: {key} does not exist.'
            return '\n'.join([f'{k}: {v}' for k, v in sorted(kvs.items())])
