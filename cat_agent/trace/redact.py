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

"""Secret redaction for serialized traces (independent of Loguru / PII paths)."""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Iterable, List, Optional, Pattern, Sequence

SECRET_PLACEHOLDER = '[REDACTED]'

_DEFAULT_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    (re.compile(r'(?i)(authorization\s*[:=]\s*bearer\s+)\S+'), r'\1' + SECRET_PLACEHOLDER),
    (re.compile(r'\bsk-[A-Za-z0-9_\-]{8,}\b'), SECRET_PLACEHOLDER),
    (re.compile(r'\b(?:api[_-]?key|secret|password|token)\s*[:=]\s*\S+', re.I), SECRET_PLACEHOLDER),
    (
        re.compile(
            r'([?&](?:token|api[_-]?key|access[_-]?token|key|secret)=)[^&\s"\']+',
            re.I,
        ),
        r'\1' + SECRET_PLACEHOLDER,
    ),
)

_SENSITIVE_KEYS = frozenset({
    'api_key', 'apikey', 'api-key', 'authorization', 'auth', 'password',
    'secret', 'token', 'access_token', 'refresh_token', 'private_key',
})


def _extra_patterns_from_env() -> List[Pattern[str]]:
    raw = os.getenv('CAT_AGENT_TRACE_REDACT_PATTERNS', '').strip()
    if not raw:
        return []
    out: List[Pattern[str]] = []
    for part in raw.split('||'):
        part = part.strip()
        if part:
            out.append(re.compile(part))
    return out


def redact_string(
    text: str,
    *,
    extra_patterns: Optional[Sequence[Pattern[str]]] = None,
) -> str:
    if not text:
        return text
    result = text
    for pattern, repl in _DEFAULT_PATTERNS:
        result = pattern.sub(repl, result)
    for pattern in list(extra_patterns or []) + _extra_patterns_from_env():
        result = pattern.sub(SECRET_PLACEHOLDER, result)
    return result


def redact_llm_config(cfg: Optional[dict]) -> dict:
    """Drop credentials from an LLM config dict before persistence."""
    if not cfg:
        return {}
    out: dict = {}
    for key, value in cfg.items():
        lk = str(key).lower().replace('-', '_')
        if lk in _SENSITIVE_KEYS or 'api_key' in lk or lk.endswith('_key'):
            out[key] = SECRET_PLACEHOLDER
        elif isinstance(value, dict):
            out[key] = redact_llm_config(value)
        elif isinstance(value, str):
            out[key] = redact_string(value)
        else:
            out[key] = value
    return out


def redact_obj(
    obj: Any,
    *,
    extra_patterns: Optional[Sequence[Pattern[str]]] = None,
) -> Any:
    """Deep-copy *obj* and redact secrets in strings and sensitive dict keys."""
    if isinstance(obj, str):
        return redact_string(obj, extra_patterns=extra_patterns)
    if isinstance(obj, dict):
        cleaned: dict = {}
        for key, value in obj.items():
            lk = str(key).lower().replace('-', '_')
            if lk in _SENSITIVE_KEYS or 'api_key' in lk:
                cleaned[key] = SECRET_PLACEHOLDER
            else:
                cleaned[key] = redact_obj(value, extra_patterns=extra_patterns)
        return cleaned
    if isinstance(obj, list):
        return [redact_obj(item, extra_patterns=extra_patterns) for item in obj]
    if isinstance(obj, tuple):
        return tuple(redact_obj(item, extra_patterns=extra_patterns) for item in obj)
    return copy.deepcopy(obj) if hasattr(obj, '__dict__') else obj
