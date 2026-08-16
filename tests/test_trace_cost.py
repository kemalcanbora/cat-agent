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

"""Tests for cat_agent.trace.cost."""

import json

from cat_agent.trace.cost import estimate_cost_usd, load_price_table


def test_load_price_table_empty_when_unset(monkeypatch):
    monkeypatch.delenv('CAT_AGENT_TRACE_PRICE_TABLE', raising=False)
    assert load_price_table() == {}


def test_load_price_table_inline_json(monkeypatch):
    monkeypatch.setenv(
        'CAT_AGENT_TRACE_PRICE_TABLE',
        '{"m": {"input_per_1m": 1.0, "output_per_1m": 2.0}}',
    )
    table = load_price_table()
    assert table['m']['input_per_1m'] == 1.0


def test_load_price_table_from_file(tmp_path, monkeypatch):
    path = tmp_path / 'prices.json'
    path.write_text(json.dumps({
        'gpt': {'prompt_per_1m': 0.5, 'completion_per_1m': 1.5},
    }), encoding='utf-8')
    monkeypatch.delenv('CAT_AGENT_TRACE_PRICE_TABLE', raising=False)
    table = load_price_table(str(path))
    assert 'gpt' in table


def test_estimate_cost_none_without_model_or_table():
    assert estimate_cost_usd(model=None, prompt_tokens=100, completion_tokens=10) is None
    assert estimate_cost_usd(
        model='x', prompt_tokens=100, completion_tokens=10, price_table={},
    ) is None


def test_estimate_cost_usd_math():
    table = {'stub': {'input_per_1m': 1.0, 'output_per_1m': 2.0}}
    cost = estimate_cost_usd(
        model='stub',
        prompt_tokens=1_000_000,
        completion_tokens=500_000,
        price_table=table,
    )
    assert cost == 1.0 + 1.0


def test_estimate_cost_case_insensitive_model():
    table = {'Stub-Model': {'input_per_1m': 1.0, 'output_per_1m': 0.0}}
    # exact key first; lower lookup when missing
    assert estimate_cost_usd(
        model='Stub-Model', prompt_tokens=1_000_000, completion_tokens=0, price_table=table,
    ) == 1.0
    table2 = {'stub-model': {'input_per_1m': 3.0, 'output_per_1m': 0.0}}
    assert estimate_cost_usd(
        model='STUB-MODEL', prompt_tokens=1_000_000, completion_tokens=0, price_table=table2,
    ) == 3.0
