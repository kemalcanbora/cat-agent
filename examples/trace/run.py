#!/usr/bin/env python3
"""Structured execution traces (cat_agent.trace).

Writes a JSONL run (schema v1.0) with RunLimits, redacted llm_config,
and token totals — the same path as ``CAT_AGENT_TRACE=1``.

    export OLLAMA_API_KEY=...
    export OLLAMA_BASE_URL=https://ollama.com/v1
    python examples/trace/run.py
"""

from __future__ import annotations

import os
from pathlib import Path

from cat_agent.agents import Assistant
from cat_agent.trace import JSONLTraceStore, RunLimits, load_runs_from_jsonl


def llm_cfg() -> dict:
    base = (
        os.getenv('OLLAMA_BASE_URL')
        or os.getenv('OLLAMA_API_BASE')
        or 'https://ollama.com/v1'
    ).rstrip('/')
    if not base.endswith('/v1'):
        base = base + '/v1'
    return {
        'model': os.getenv('LLM_MODEL', 'minimax-m2.7:cloud'),
        'model_type': 'oai',
        'model_server': base,
        'api_key': os.getenv('OLLAMA_API_KEY') or os.getenv('OPENAI_API_KEY') or 'EMPTY',
        'generate_cfg': {'temperature': 0.2, 'max_tokens': 256},
    }


def main():
    if not (os.getenv('OLLAMA_API_KEY') or os.getenv('OPENAI_API_KEY')):
        print('Set OLLAMA_API_KEY (and optionally OLLAMA_BASE_URL / LLM_MODEL) first.')
        return

    out = Path('workspace') / 'examples' / 'trace.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    store = JSONLTraceStore(out)
    bot = Assistant(
        llm=llm_cfg(),
        name='trace-demo',
        system_message='Answer in one short sentence.',
        run_limits=RunLimits(max_steps=8, max_total_tokens=20_000),
    )

    messages = [{'role': 'user', 'content': 'What is 2 + 2? Reply with just the number.'}]
    for rsp in bot.run(messages, trace=True, trace_store=store):
        pass

    run = next(iter(load_runs_from_jsonl(out).values()))
    print(f'status={run.status} reason={run.termination_reason}')
    print(f'steps={run.totals.steps} prompt={run.totals.prompt_tokens} '
          f'completion={run.totals.completion_tokens} '
          f'estimated={run.totals.tokens_estimated}')
    print(f'llm_config keys (redacted): {sorted(run.llm_config.keys())}')
    assert run.llm_config.get('api_key') in (None, '', '[REDACTED]')
    print(f'trace → {out}')
    if rsp:
        last = rsp[-1]
        print('reply:', last.get('content') if isinstance(last, dict) else last.content)


if __name__ == '__main__':
    main()
