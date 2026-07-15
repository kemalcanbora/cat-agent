"""Migrate plaintext local storage to encrypted at-rest format."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from cat_agent.security.encrypted_cache import migrate_plaintext_cache
from cat_agent.security.encrypted_files import ENCRYPTED_SUFFIX, migrate_plaintext_file
from cat_agent.settings import DEFAULT_WORKSPACE


def _default_storage_dirs(workspace: str) -> List[str]:
    return [
        os.path.join(workspace, 'tools', 'doc_parser'),
        os.path.join(workspace, 'tools', 'simple_doc_parser'),
        os.path.join(workspace, 'tools', 'storage'),
    ]


def _index_files(workspace: str) -> List[str]:
    paths: List[str] = []
    for subdir in ('keyword_indexes', 'vector_indexes'):
        base = Path(workspace) / 'storage' / subdir
        if not base.is_dir():
            continue
        for item in base.iterdir():
            if item.is_file() and not item.name.endswith(ENCRYPTED_SUFFIX):
                paths.append(str(item))
    return paths


def migrate_workspace_storage(workspace: str | None = None) -> Dict[str, int]:
    """Encrypt SQLite caches and RAG index files under the workspace."""
    root = workspace or DEFAULT_WORKSPACE
    report = {
        'sqlite_records': 0,
        'index_files': 0,
    }

    for storage_dir in _default_storage_dirs(root):
        report['sqlite_records'] += migrate_plaintext_cache(storage_dir)

    for path in _index_files(root):
        if migrate_plaintext_file(path):
            report['index_files'] += 1

    return report


def count_plaintext_index_files(workspace: str | None = None) -> int:
    root = workspace or DEFAULT_WORKSPACE
    return sum(1 for path in _index_files(root) if Path(path).is_file())
