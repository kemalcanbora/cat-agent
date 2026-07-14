"""Download or copy the WASI CPython runtime on first use.

Wheels no longer bundle the large ``python*.wasm`` / ``python311.zip`` assets.
They are cached under the workspace (or a configured ``runtime_dir``) after a
verified download from the cat-agent GitHub release tag.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import requests

from cat_agent import __version__
from cat_agent.log import logger
from cat_agent.settings import DEFAULT_WORKSPACE

BUNDLED_RUNTIME_DIR = Path(__file__).resolve().parent / 'wasm_runtime'
GITHUB_RAW_BASE = 'https://github.com/kemalcanbora/cat-agent/raw'

_RUNTIME_ASSETS = {
    'bin/python-3.11.1.wasm': '88b0f02cc42b389ab14e8c1b9a57e7ea5ab75397c0859baf9265c5eac58d3437',
    'usr/local/lib/python311.zip': '4593d0c62a1b4cb7de17c591578eb85de3b4037828b18a6764bbd304435da605',
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_if_valid(source: Path, dest: Path, expected_sha256: str) -> bool:
    if not source.is_file():
        return False
    if _sha256(source) != expected_sha256:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return True


def _download_asset(relative_path: str, dest: Path, expected_sha256: str) -> None:
    url = f'{GITHUB_RAW_BASE}/v{__version__}/cat_agent/tools/resource/wasm_runtime/{relative_path}'
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Downloading WASM runtime asset {}', relative_path)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    dest.write_bytes(response.content)
    if _sha256(dest) != expected_sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f'Checksum mismatch for downloaded WASM asset {relative_path}. '
            f'Expected sha256 {expected_sha256}.'
        )


def ensure_wasm_runtime(runtime_dir: str | None = None) -> str:
    """Return a directory containing the WASI CPython runtime, downloading if needed."""
    target = Path(runtime_dir or os.path.join(DEFAULT_WORKSPACE, 'storage', 'wasm_runtime'))
    target.mkdir(parents=True, exist_ok=True)

    for relative_path, expected_sha256 in _RUNTIME_ASSETS.items():
        dest = target / relative_path
        if dest.is_file() and _sha256(dest) == expected_sha256:
            continue

        bundled = BUNDLED_RUNTIME_DIR / relative_path
        if _copy_if_valid(bundled, dest, expected_sha256):
            continue

        _download_asset(relative_path, dest, expected_sha256)

    return str(target)
