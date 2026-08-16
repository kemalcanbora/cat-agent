#!/usr/bin/env python3
"""Run a deliberately failing loop, write a trace, then analyse with MAST Tier-1."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List

from cat_agent.agent import Agent
from cat_agent.analysis import analyze_trace, render_text_report
from cat_agent.llm.schema import ASSISTANT, USER, FunctionCall, Message
from cat_agent.trace import JSONLTraceStore, RunLimits, load_runs_from_jsonl


class RepeatingStubLLM:
    model = 'fail-model'
    model_type = 'oai'
    model_cfg = {'model': 'fail-model', 'model_type': 'oai'}

    def chat(self, messages, functions=None, stream=True, extra_generate_cfg=None):
        out = [Message(
            ASSISTANT, '',
            function_call=FunctionCall('search', '{"q":"same"}'),
            extra={'usage': {'prompt_tokens': 20, 'completion_tokens': 5}},
        )]
        if stream:
            yield out
        else:
            return out


class FailLoopAgent(Agent):
    """Keeps emitting the same tool call until RunLimits stop it."""

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        for _ in range(20):
            for out in self._call_llm(messages):
                yield out
                if out and out[-1].function_call:
                    # Soft-fail missing tool → string, continue
                    result = self._call_tool(out[-1].function_call.name, out[-1].function_call.arguments)
                    yield [Message(role='function', name='search', content=result)]


def main():
    out = Path('workspace') / 'failure_analysis' / 'fail.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    os.environ['CAT_AGENT_TRACE'] = '1'
    store = JSONLTraceStore(out)
    agent = FailLoopAgent(
        llm=RepeatingStubLLM(),
        name='fail-loop',
        context_manager=False,
        run_limits=RunLimits(max_steps=6, max_tool_calls=4),
    )
    list(agent.run([{'role': 'user', 'content': 'Find the bug'}], trace_store=store, trace=True))
    runs = load_runs_from_jsonl(out)
    run = next(iter(runs.values()))
    result = analyze_trace(run, tiers=('deterministic',))
    print(render_text_report(result))
    print(f'\nTrace written to {out}')


if __name__ == '__main__':
    main()
