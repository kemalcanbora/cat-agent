"""Encrypted long-term memory store with vector recall."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from typing import Dict, List, Optional

from cat_agent.log import logger
from cat_agent.security.at_rest import is_encrypt_at_rest_enabled, is_encrypted_at_rest_required
from cat_agent.security.crypto import (
    EncryptionKeyError,
    decrypt_value,
    encrypt_value,
    is_encrypted_value,
    resolve_encryption_key,
)
from cat_agent.settings import DEFAULT_WORKSPACE

MEMORY_KINDS = ('fact', 'summary', 'episode')


@dataclass
class MemoryRecord:
    memory_id: str
    scope: str
    kind: str
    text: str
    created_at: str
    metadata: Dict = field(default_factory=dict)
    score: Optional[float] = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_key(*, create_if_missing: bool) -> Optional[bytes]:
    if not is_encrypt_at_rest_enabled():
        return None
    try:
        return resolve_encryption_key(create_if_missing=create_if_missing)
    except EncryptionKeyError:
        if is_encrypted_at_rest_required():
            raise
        return None


class MemoryStore:
    """SQLite-backed long-term memory with AES-GCM at rest and semantic recall.

    Recall uses the native HNSW vector index with hash embeddings when the
    Rust extension is available, and falls back to keyword-overlap scoring
    otherwise. Payloads are encrypted with the same key management as the
    rest of Cat-Agent storage.
    """

    def __init__(self, path: Optional[str] = None, embedding_cfg: Optional[Dict] = None):
        root = path or os.path.join(DEFAULT_WORKSPACE, 'memory')
        os.makedirs(root, exist_ok=True)
        self.db_path = os.path.join(root, 'memories.sqlite')
        self._embedding_cfg = embedding_cfg or {}
        self._embedder = None
        self._index_cache: Dict[str, tuple] = {}
        self._ensure_schema()
        key = _resolve_key(create_if_missing=True)
        if is_encrypt_at_rest_enabled() and key is None:
            logger.warning(
                'Encryption is enabled but no key is available; memory store at {} will use plaintext.',
                self.db_path,
            )
        self._key = key

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                'CREATE TABLE IF NOT EXISTS memories ('
                'memory_id TEXT PRIMARY KEY, '
                'scope TEXT NOT NULL, '
                'created_at TEXT NOT NULL, '
                'payload TEXT NOT NULL)'
            )
            conn.execute('CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)')

    def _encode_payload(self, kind: str, text: str, metadata: Dict) -> str:
        payload = json.dumps({'kind': kind, 'text': text, 'metadata': metadata}, ensure_ascii=False)
        if self._key is not None:
            return encrypt_value(payload, self._key)
        return payload

    def _decode_payload(self, payload: str) -> Dict:
        if is_encrypted_value(payload):
            if self._key is None:
                raise EncryptionKeyError(
                    'Encrypted memory record found but no decryption key is available.'
                )
            payload = decrypt_value(payload, self._key)
        return json.loads(payload)

    def add(self, text: str, *, scope: str = 'default', kind: str = 'fact',
            metadata: Optional[Dict] = None) -> MemoryRecord:
        if kind not in MEMORY_KINDS:
            raise ValueError(f'kind must be one of {MEMORY_KINDS}, got {kind!r}')
        record = MemoryRecord(
            memory_id=uuid.uuid4().hex,
            scope=scope,
            kind=kind,
            text=text,
            created_at=_now_iso(),
            metadata=metadata or {},
        )
        with self._connect() as conn:
            conn.execute(
                'INSERT INTO memories(memory_id, scope, created_at, payload) VALUES (?, ?, ?, ?)',
                (record.memory_id, scope, record.created_at,
                 self._encode_payload(kind, text, record.metadata)),
            )
            conn.commit()
        self._index_cache.pop(scope, None)
        return record

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT memory_id, scope, created_at, payload FROM memories WHERE memory_id = ?',
                (memory_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list(self, scope: str = 'default') -> List[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT memory_id, scope, created_at, payload FROM memories '
                'WHERE scope = ? ORDER BY created_at',
                (scope,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def delete(self, memory_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT scope FROM memories WHERE memory_id = ?', (memory_id,)
            ).fetchone()
            cursor = conn.execute('DELETE FROM memories WHERE memory_id = ?', (memory_id,))
            conn.commit()
        if row:
            self._index_cache.pop(row[0], None)
        return cursor.rowcount > 0

    def clear(self, scope: str = 'default') -> int:
        with self._connect() as conn:
            cursor = conn.execute('DELETE FROM memories WHERE scope = ?', (scope,))
            conn.commit()
        self._index_cache.pop(scope, None)
        return cursor.rowcount

    def search(self, query: str, *, scope: str = 'default', top_k: int = 5) -> List[MemoryRecord]:
        records = self.list(scope)
        if not records or not query.strip():
            return []
        try:
            ranked = self._vector_rank(query, scope, records)
        except ImportError:
            logger.debug('[MemoryStore] Native extension unavailable; using keyword recall.')
            ranked = self._keyword_rank(query, records)
        return [record for record in ranked if record.score and record.score > 0][:top_k]

    def _row_to_record(self, row) -> MemoryRecord:
        memory_id, scope, created_at, payload = row
        data = self._decode_payload(payload)
        return MemoryRecord(
            memory_id=memory_id,
            scope=scope,
            kind=data.get('kind', 'fact'),
            text=data.get('text', ''),
            created_at=created_at,
            metadata=data.get('metadata', {}),
        )

    def _get_embedder(self):
        if self._embedder is None:
            from cat_agent.tools.search_tools.embedding import build_embedder

            self._embedder = build_embedder(self._embedding_cfg)
        return self._embedder

    def _vector_rank(self, query: str, scope: str, records: List[MemoryRecord]) -> List[MemoryRecord]:
        native = import_module('cat_agent._native')
        embedder = self._get_embedder()

        ids = tuple(record.memory_id for record in records)
        cached = self._index_cache.get(scope)
        if cached is None or cached[0] != ids:
            vectors = embedder.embed([record.text[:2000] for record in records])
            index = native.VectorIndex(embedder.dimensions, 'cos')
            index.add(list(range(len(records))), vectors)
            self._index_cache[scope] = (ids, index)
        index = self._index_cache[scope][1]

        query_vector = embedder.embed([query[:2000]])[0]
        matches = index.search(query_vector, len(records))
        ranked = []
        for key, distance in matches:
            record = records[key]
            record.score = 1.0 - float(distance)
            ranked.append(record)
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    @staticmethod
    def _keyword_rank(query: str, records: List[MemoryRecord]) -> List[MemoryRecord]:
        query_words = {word for word in query.lower().split() if len(word) > 1}
        for record in records:
            text_words = set(record.text.lower().split())
            overlap = len(query_words & text_words)
            record.score = overlap / max(1, len(query_words))
        return sorted(records, key=lambda item: item.score, reverse=True)
