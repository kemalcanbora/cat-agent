#!/usr/bin/env python3
"""MAST trajectory failure analysis (cat_agent.analysis).

Tier-1 detectors always run (no LLM). Optional Tier-2 judge uses Ollama when
``OLLAMA_API_KEY`` is set.

    export OLLAMA_API_KEY=...          # optional, for judge
    export OLLAMA_BASE_URL=https://ollama.com/v1
    python examples/analysis/run.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List

from cat_agent.agent import Agent
from cat_agent.analysis import analyze_trace, render_text_report
from cat_agent.llm import get_chat_model
from cat_agent.llm.schema import ASSISTANT, FunctionCall, Message
from cat_agent.tools import tool
from cat_agent.trace import JSONLTraceStore, RunLimits, load_runs_from_jsonl


@tool(allow_overwrite=True)
def search(q: str) -> str:
    """Return a fixed search stub (used only to exercise the tool loop)."""
    return f'results for {q}'


class StuckStubLLM:
    """Always emits the same successful tool call — triggers MAST 1.3."""

    model = 'stuck-stub'
    model_type = 'oai'
    model_cfg = {'model': 'stuck-stub', 'model_type': 'oai'}

    def chat(self, messages, functions=None, stream=True, extra_generate_cfg=None):
        out = [Message(
            ASSISTANT, '',
            function_call=FunctionCall('search', '{"q":"same"}'),
        )]
        if stream:
            yield out
        else:
            return out


class StuckAgent(Agent):
    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        for _ in range(12):
            for out in self._call_llm(messages):
                yield out
                if out and out[-1].function_call:
                    result = self._call_tool(
                        out[-1].function_call.name,
                        out[-1].function_call.arguments,
                    )
                    fn = Message(role='function', name='search', content=str(result))
                    messages.append(out[-1])
                    messages.append(fn)
                    yield [fn]


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
        'generate_cfg': {'temperature': 0.1, 'max_tokens': 1024},
    }


def main():
    out = Path('workspace') / 'examples' / 'analysis.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    store = JSONLTraceStore(out)
    agent = StuckAgent(
        llm=StuckStubLLM(),
        name='stuck-loop',
        function_list=['search'],
        context_manager=False,
        run_limits=RunLimits(max_steps=8, max_tool_calls=5),
    )
    list(agent.run(
        [{'role': 'user', 'content': 'Find the bug'}],
        trace=True,
        trace_store=store,
    ))

    run = next(iter(load_runs_from_jsonl(out).values()))
    print('=== Tier-1 (deterministic) ===')
    result = analyze_trace(run, tiers=('deterministic',))
    print(render_text_report(result))
    print(f'trace → {out}')

    key = os.getenv('OLLAMA_API_KEY') or os.getenv('OPENAI_API_KEY')
    if key:
        print('\n=== Tier-2 (opt-in LLM judge) ===')
        judge = get_chat_model(llm_cfg())
        judged = analyze_trace(run, judge_llm=judge, tiers=('deterministic', 'judge'))
        print(render_text_report(judged))
    else:
        print('\n(skip judge — set OLLAMA_API_KEY to enable Tier-2)')


if __name__ == '__main__':
    main()
