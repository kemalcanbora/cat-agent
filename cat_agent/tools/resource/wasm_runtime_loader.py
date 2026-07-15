"""Download or copy the WASI CPython runtime for offline sandboxed execution."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from cat_agent.log import logger
from cat_agent.security.offline import guard_outbound_request, is_offline_mode
from cat_agent.settings import DEFAULT_WORKSPACE

BUNDLED_RUNTIME_DIR = Path(__file__).resolve().parent / 'wasm_runtime'

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


def runtime_assets_present(runtime_dir: str | Path | None = None) -> bool:
    base = Path(runtime_dir) if runtime_dir else BUNDLED_RUNTIME_DIR
    for relative_path, expected_sha256 in _RUNTIME_ASSETS.items():
        asset = base / relative_path
        if not asset.is_file() or _sha256(asset) != expected_sha256:
            return False
    return True


def fetch_runtime_assets(output_dir: str | Path) -> str:
    """Copy bundled WASM assets into *output_dir* for offline transfer."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    if not runtime_assets_present(BUNDLED_RUNTIME_DIR):
        raise FileNotFoundError(
            f'Bundled WASM runtime is missing under {BUNDLED_RUNTIME_DIR}. '
            'Install cat-agent[wasm-bundled] or build from source.'
        )
    for relative_path, expected_sha256 in _RUNTIME_ASSETS.items():
        dest = target / relative_path
        if dest.is_file() and _sha256(dest) == expected_sha256:
            continue
        _copy_if_valid(BUNDLED_RUNTIME_DIR / relative_path, dest, expected_sha256)
    return str(target)


def ensure_wasm_runtime(runtime_dir: str | None = None) -> str:
    """Return a directory containing the WASI CPython runtime."""
    target = Path(runtime_dir or os.path.join(DEFAULT_WORKSPACE, 'storage', 'wasm_runtime'))
    target.mkdir(parents=True, exist_ok=True)

    if runtime_assets_present(target):
        return str(target)

    for relative_path, expected_sha256 in _RUNTIME_ASSETS.items():
        dest = target / relative_path
        if dest.is_file() and _sha256(dest) == expected_sha256:
            continue
        bundled = BUNDLED_RUNTIME_DIR / relative_path
        if _copy_if_valid(bundled, dest, expected_sha256):
            continue
        if is_offline_mode():
            raise FileNotFoundError(
                'WASM runtime assets are not available locally and downloads are disabled '
                'in offline mode (CAT_AGENT_OFFLINE=1). '
                'Install cat-agent[wasm-bundled], run '
                '`python -m cat_agent.cli fetch-runtime --output <dir>`, '
                'or set runtime_dir to a pre-provisioned copy.'
            )
        _download_asset(relative_path, dest, expected_sha256)

    return str(target)


def _download_asset(relative_path: str, dest: Path, expected_sha256: str) -> None:
    import requests

    from cat_agent import __version__

    url = (
        'https://github.com/kemalcanbora/cat-agent/raw/'
        f'v{__version__}/cat_agent/tools/resource/wasm_runtime/{relative_path}'
    )
    guard_outbound_request(purpose=f'WASM runtime download from {url}', url=url)
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
