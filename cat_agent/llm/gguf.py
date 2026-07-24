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

"""Resolve GGUF model paths from local disk / HuggingFace cache before downloading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from cat_agent.log import logger


def resolve_gguf_path(
    *,
    model_path: Optional[str] = None,
    repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Return a filesystem path to a GGUF weights file.

    Resolution order:
      1. Explicit ``model_path`` (returned as-is)
      2. HuggingFace hub cache (``try_to_load_from_cache``) when ``repo_id`` +
         ``filename`` are set — no network
      3. ``~/models/<filename>`` if that file exists
      4. Download via ``hf_hub_download`` (network)
    """
    if model_path:
        return model_path

    if not (repo_id and filename):
        raise ValueError(
            "GGUF resolve requires either 'model_path' or both 'repo_id' and 'filename'"
        )

    cached = _hf_cache_path(repo_id, filename, cache_dir=cache_dir)
    if cached:
        logger.info(f"Using HuggingFace cache for {repo_id}/{filename}: {cached}")
        return cached

    home_models = Path.home() / 'models' / filename
    if home_models.is_file():
        logger.info(f"Using local GGUF at {home_models}")
        return str(home_models)

    from huggingface_hub import hf_hub_download

    logger.info(f"Downloading GGUF from HuggingFace: {repo_id} / {filename}")
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def _hf_cache_path(
    repo_id: str,
    filename: str,
    *,
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST
    except ImportError:
        return None

    try:
        cached = try_to_load_from_cache(
            repo_id,
            filename,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        logger.debug(f"HuggingFace cache lookup failed for {repo_id}/{filename}: {exc}")
        return None

    if cached is None or cached is _CACHED_NO_EXIST:
        return None
    if isinstance(cached, (str, os.PathLike)) and os.path.isfile(cached):
        return str(cached)
    return None
