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

"""Build LLM config dicts from environment (platform-owned key names)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]


def _get_env(*keys: str) -> str | None:
    for k in keys:
        v = os.environ.get(k)
        if v is not None and str(v).strip() != '':
            return str(v).strip()
    return None


def _resolve_agent_yaml(explicit: PathLike | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_file() else None
    env = os.environ.get('CAT_AGENT_AGENT_YAML', '').strip()
    if env:
        path = Path(env)
        if path.is_file():
            return path
    cwd = Path.cwd() / 'agent.yaml'
    return cwd if cwd.is_file() else None


def model_alias_from_agent_yaml(path: PathLike) -> Optional[str]:
    """Return ``model.alias`` from an agent.yaml (None if missing / unreadable)."""
    p = Path(path)
    if not p.is_file():
        return None
    text = p.read_text(encoding='utf-8')
    try:
        import yaml

        data = yaml.safe_load(text) or {}
        if isinstance(data, dict):
            model = data.get('model') or {}
            if isinstance(model, dict):
                alias = model.get('alias')
                if alias is not None and str(alias).strip():
                    return str(alias).strip()
    except Exception:  # noqa: BLE001 — fall through to line scan
        pass

    in_model = False
    for line in text.splitlines():
        if re.match(r'^model:\s*$', line):
            in_model = True
            continue
        if in_model:
            if re.match(r'^\S', line):
                break
            m = re.match(r'^\s+alias:\s*[\'"]?([^\'"#\n]+)', line)
            if m:
                return m.group(1).strip().strip("'\"")
    return None


def apply_agent_yaml_env(
    path: PathLike | None = None,
    *,
    override: bool = False,
) -> Dict[str, str]:
    """Copy ``agent.yaml`` ``env:`` into ``os.environ`` for local script runs.

    Under Nomad (``CAT_AGENT_MANAGED=1``) the platform already injects these;
    this is a no-op then. Returns the keys that were applied.
    """
    if (os.environ.get('CAT_AGENT_MANAGED') or '').strip() == '1':
        return {}
    yaml_path = _resolve_agent_yaml(path)
    if yaml_path is None:
        return {}
    try:
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding='utf-8')) or {}
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(data, dict):
        return {}
    env_block = data.get('env') or {}
    if not isinstance(env_block, dict):
        return {}
    applied: Dict[str, str] = {}
    for raw_key, raw_val in env_block.items():
        key = str(raw_key).strip()
        if not key or raw_val is None:
            continue
        val = str(raw_val).strip()
        if not val:
            continue
        if not override and (os.environ.get(key) or '').strip():
            continue
        os.environ[key] = val
        applied[key] = val
    return applied


def llm_config_from_env(
    *,
    agent_yaml: PathLike | None = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Resolve chat-model config with platform-safe precedence.

    Order for ``model`` (first hit wins):
    ``overrides['model']`` > ``CAT_AGENT_LLM_MODEL`` / ``OPENAI_MODEL`` /
    ``LLM_MODEL`` > ``agent.yaml`` ``model.alias`` > ``default``.

    ``agent_yaml`` defaults to ``CAT_AGENT_AGENT_YAML`` or ``./agent.yaml``.

    Other fields: ``overrides`` > ``CAT_AGENT_LLM_*`` > ``OPENAI_*`` >
    ``OLLAMA_API_BASE`` > defaults.

    Under Nomad, ``CAT_AGENT_MANAGED=1`` skips dotenv so a stray ``.env`` cannot
    replace the gateway ``OPENAI_BASE_URL`` / ``CAT_AGENT_LLM_BASE_URL``.
    """
    cfg: Dict[str, Any] = {
        'model_type': 'oai',
        'model': 'default',
    }

    base = _get_env(
        'CAT_AGENT_LLM_BASE_URL',
        'OPENAI_BASE_URL',
        'OLLAMA_API_BASE',
        'OLLAMA_BASE_URL',  # legacy alias
    )
    if base:
        cfg['base_url'] = base.rstrip('/')

    key = _get_env(
        'CAT_AGENT_LLM_API_KEY',
        'OPENAI_API_KEY',
        'OLLAMA_API_KEY',
    )
    if key:
        cfg['api_key'] = key

    model = _get_env('CAT_AGENT_LLM_MODEL', 'OPENAI_MODEL', 'LLM_MODEL')
    if not model:
        yaml_path = _resolve_agent_yaml(agent_yaml)
        if yaml_path is not None:
            model = model_alias_from_agent_yaml(yaml_path)
    if model:
        cfg['model'] = model

    model_type = _get_env('CAT_AGENT_LLM_MODEL_TYPE')
    if model_type:
        cfg['model_type'] = model_type

    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
