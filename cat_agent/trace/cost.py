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

"""Token cost accounting from a user-supplied price table. Never guess prices."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional


def load_price_table(path_or_json: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Load ``{model: {input_per_1m: float, output_per_1m: float}}``.

    Reads ``CAT_AGENT_TRACE_PRICE_TABLE`` (path or inline JSON) when *path_or_json*
    is omitted. Returns ``{}`` when unset — cost stays ``None``.
    """
    raw = path_or_json if path_or_json is not None else os.getenv('CAT_AGENT_TRACE_PRICE_TABLE', '')
    raw = (raw or '').strip()
    if not raw:
        return {}
    if raw.startswith('{'):
        data = json.loads(raw)
    else:
        with open(raw, encoding='utf-8') as fh:
            data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError('Price table must be a JSON object keyed by model name')
    return data  # type: ignore[return-value]


def estimate_cost_usd(
    *,
    model: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    price_table: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Optional[float]:
    """Return USD cost or ``None`` when the model has no configured price."""
    table = price_table if price_table is not None else load_price_table()
    if not model or not table:
        return None
    entry = table.get(model) or table.get(model.lower())
    if not entry:
        return None
    inp = float(entry.get('input_per_1m') or entry.get('prompt_per_1m') or 0.0)
    out = float(entry.get('output_per_1m') or entry.get('completion_per_1m') or 0.0)
    if inp == 0.0 and out == 0.0 and 'input_per_1m' not in entry and 'prompt_per_1m' not in entry:
        return None
    return (prompt_tokens / 1_000_000.0) * inp + (completion_tokens / 1_000_000.0) * out
