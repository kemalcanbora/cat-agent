import dbm
import os
import sqlite3
from typing import Dict, Optional, Union

from cat_agent.settings import DEFAULT_WORKSPACE
from cat_agent.tools.base import BaseTool, register_tool


class KeyNotExistsError(ValueError):
    pass


def _norm_key(key: str) -> str:
    """Strip leading slash so keys are stored consistently."""
    return key[1:] if key.startswith('/') else key


def _sqlite_path(root: str) -> str:
    return os.path.join(root, 'storage.sqlite')


def _legacy_dbm_path(root: str) -> str:
    return os.path.join(root, 'storage.db')


class _SqliteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._ensure_schema()
        self._maybe_migrate_from_dbm()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                'CREATE TABLE IF NOT EXISTS kv ('
                'key TEXT PRIMARY KEY, '
                'value TEXT NOT NULL)'
            )

    def _maybe_migrate_from_dbm(self) -> None:
        legacy_path = os.path.join(os.path.dirname(self.db_path), 'storage.db')
        if not os.path.exists(legacy_path):
            return
        with self._connect() as conn:
            row = conn.execute('SELECT COUNT(*) FROM kv').fetchone()
            if row and row[0] > 0:
                return
        try:
            with dbm.open(legacy_path, 'c') as legacy_db:
                with self._connect() as conn:
                    for key in legacy_db.keys():
                        conn.execute(
                            'INSERT OR REPLACE INTO kv(key, value) VALUES (?, ?)',
                            (key.decode('utf-8'), legacy_db[key].decode('utf-8')),
                        )
                    conn.commit()
        except dbm.error:
            return

    def put(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO kv(key, value) VALUES (?, ?)',
                (key, value),
            )
            conn.commit()

    def get(self, key: str) -> str:
        with self._connect() as conn:
            row = conn.execute('SELECT value FROM kv WHERE key = ?', (key,)).fetchone()
        if row is None:
            raise KeyNotExistsError(f'Get Failed: {key} does not exist')
        return row[0]

    def delete(self, key: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute('DELETE FROM kv WHERE key = ?', (key,))
            conn.commit()
            return cursor.rowcount > 0

    def scan(self, key: str) -> Dict[str, str]:
        prefix = (key.rstrip('/') + '/') if key else ''
        with self._connect() as conn:
            rows = conn.execute('SELECT key, value FROM kv ORDER BY key').fetchall()
        kvs: Dict[str, str] = {}
        for stored_key, value in rows:
            if stored_key == key or (prefix and stored_key.startswith(prefix)):
                rel = '/' + stored_key[len(key):].lstrip('/') if key and stored_key.startswith(key) else '/' + stored_key
                kvs[rel] = value
        return kvs


@register_tool('storage')
class Storage(BaseTool):
    """Key/value storage backed by SQLite for cross-platform portability."""

    description = 'Tool for storing and reading data.'
    parameters = {
        'type': 'object',
        'properties': {
            'operate': {
                'description': 'Type of data operation, options are ["put", "get", "delete", "scan"], which respectively represent saving data, retrieving data, deleting data, and scanning data.',
                'type': 'string',
            },
            'key': {
                'description': 'Data path, similar to a file path, serves as a unique identifier for a piece of data. It cannot be empty and defaults to "/" as the root directory. When saving data, the path should be reasonably designed to ensure clarity and uniqueness.',
                'type': 'string',
                'default': '/'
            },
            'value': {
                'description': 'The content of the data, needed only when saving data.',
                'type': 'string',
            },
        },
        'required': ['operate'],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        root = self.cfg.get('storage_root_path', os.path.join(DEFAULT_WORKSPACE, 'tools', self.name))
        os.makedirs(root, exist_ok=True)
        self._root = root
        self._store = _SqliteStore(_sqlite_path(root))

    def _store_for_path(self, path: Optional[str]) -> _SqliteStore:
        if path is None:
            return self._store
        return _SqliteStore(_sqlite_path(path))

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        operate = params['operate']
        key = _norm_key(params.get('key', '/'))

        if operate == 'put':
            assert 'value' in params
            return self.put(key, params['value'])
        if operate == 'get':
            return self.get(key)
        if operate == 'delete':
            return self.delete(key)
        return self.scan(key)

    def put(self, key: str, value: str, path: Optional[str] = None) -> str:
        self._store_for_path(path).put(key, value)
        return f'Successfully saved {key}.'

    def get(self, key: str, path: Optional[str] = None) -> str:
        return self._store_for_path(path).get(key)

    def delete(self, key: str, path: Optional[str] = None) -> str:
        deleted = self._store_for_path(path).delete(key)
        if not deleted:
            return f'Delete Failed: {key} does not exist'
        return f'Successfully deleted {key}'

    def scan(self, key: str, path: Optional[str] = None) -> str:
        kvs = self._store_for_path(path).scan(key)
        if not kvs:
            return f'Scan Failed: {key} does not exist.'
        return '\n'.join([f'{k}: {v}' for k, v in sorted(kvs.items())])
