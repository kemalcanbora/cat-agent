#!/usr/bin/env python3
"""Generate ≥20 clean successful traces for MAST Tier-1 negative tests.

Each run completes with status=completed, a final_answer, and patterns that
must NOT trip 1.3 / 1.4 / 1.5:

- transient tool retry then success
- paginated list calls (same endpoint, different page)
- same tool with different arguments
- identical read-only query at two distant points (other work between)
- ordinary single-shot and multi-tool happy paths

Writes JSONL under tests/fixtures/clean_traces/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from cat_agent.agent import Agent
from cat_agent.llm.schema import ASSISTANT, FUNCTION, USER, FunctionCall, Message
from cat_agent.tools import tool
from cat_agent.trace import JSONLTraceStore, load_runs_from_jsonl
from cat_agent.trace.schema import Run, Step, utc_now_iso

OUT_DIR = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'clean_traces'


class ScriptedLLM:
    """Yields a fixed sequence of assistant messages, then a final answer."""

    model = 'fixture-model'
    model_type = 'oai'
    model_cfg = {'model': 'fixture-model', 'model_type': 'oai'}

    def __init__(self, turns: List[List[Message]], final: str = 'Done.'):
        self._turns = list(turns)
        self._final = final
        self._i = 0

    def chat(self, messages, functions=None, stream=True, extra_generate_cfg=None):
        if self._i < len(self._turns):
            out = self._turns[self._i]
            self._i += 1
        else:
            out = [Message(
                ASSISTANT, self._final,
                extra={'usage': {'prompt_tokens': 8, 'completion_tokens': 4}},
            )]
        if stream:
            yield out
        else:
            return out


def _fc(name: str, args: Dict[str, Any]) -> Message:
    return Message(
        ASSISTANT, '',
        function_call=FunctionCall(name=name, arguments=json.dumps(args)),
        extra={'usage': {'prompt_tokens': 12, 'completion_tokens': 6}},
    )


def _answer(text: str) -> Message:
    return Message(
        ASSISTANT, text,
        extra={'usage': {'prompt_tokens': 10, 'completion_tokens': 5}},
    )


# --- real tools (registered via decorator for Agent.function_map) ------------

@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def lookup(key: str) -> str:
    """Read-only key lookup."""
    return f'value-for-{key}'


@tool
def list_items(
    page: int = 1,
    page_size: int = 10,
    q: str = '',
    offset: int = 0,
    cursor: str = '',
) -> dict:
    """Paginated list endpoint."""
    return {
        'page': page,
        'page_size': page_size,
        'offset': offset,
        'cursor': cursor,
        'q': q,
        'items': [f'item-{page}-{i}' for i in range(3)],
    }


@tool
def flaky_fetch(url: str) -> str:
    """Fails once per url then succeeds (module-level attempt counter)."""
    key = url
    n = _FLAKY.setdefault(key, 0)
    _FLAKY[key] = n + 1
    if n == 0:
        raise RuntimeError(f'transient error fetching {url}')
    return f'body:{url}'


_FLAKY: Dict[str, int] = {}


class FnScriptAgent(Agent):
    """Minimal tool loop driven by ScriptedLLM (like FnCallAgent)."""

    def _run(self, messages: List[Message], lang: str = 'en', **kwargs) -> Iterator[List[Message]]:
        response: List[Message] = []
        for _ in range(30):
            output: List[Message] = []
            for output in self._call_llm(messages, functions=self._schemas()):
                if output:
                    yield response + output
            if not output:
                break
            response.extend(output)
            messages.extend(output)
            used = False
            for _src, tc_id, tool_name, tool_args in self._iter_tool_call_jobs(output):
                result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                fn = Message(
                    role=FUNCTION, name=tool_name, content=result,
                    tool_call_id=tc_id, extra={'function_id': tc_id},
                )
                messages.append(fn)
                response.append(fn)
                yield response
                used = True
            if not used:
                break
        yield response

    def _schemas(self):
        return [f.function for f in self.function_map.values()]


def _run_agent(name: str, turns: List[List[Message]], final: str, user: str, tools) -> Run:
    _FLAKY.clear()
    store = JSONLTraceStore(OUT_DIR / f'{name}.jsonl')
    path = OUT_DIR / f'{name}.jsonl'
    if path.exists():
        path.unlink()
    store = JSONLTraceStore(path)
    llm = ScriptedLLM(turns, final=final)
    agent = FnScriptAgent(
        llm=llm,
        name=name,
        function_list=tools,
        context_manager=False,
    )
    list(agent.run([{'role': 'user', 'content': user}], trace_store=store, trace=True))
    runs = load_runs_from_jsonl(path)
    run = next(iter(runs.values()))
    assert run.status == 'completed', (name, run.status, run.termination_reason)
    assert run.final_output and str(run.final_output).strip(), name
    return run


def _handcrafted(name: str, steps: List[Step], final: str) -> Run:
    """Build a completed Run directly and write JSONL (for patterns hard to drive via tools)."""
    path = OUT_DIR / f'{name}.jsonl'
    if path.exists():
        path.unlink()
    store = JSONLTraceStore(path)
    run = Run(
        agent_name=name,
        agent_class='Assistant',
        status='completed',
        termination_reason='goal_reached',
        started_at=utc_now_iso(),
        ended_at=utc_now_iso(),
        initial_messages=[Message(USER, f'task:{name}')],
        final_output=final,
        steps=steps,
        llm_config={'model': 'fixture-model'},
    )
    run.recompute_totals(wall_clock_seconds=0.1)
    store.write_run_header(run)
    for step in steps:
        store.append_step(run.run_id, step)
    store.finalize_run(run)
    return run


def _tool_step(idx: int, name: str, args: dict, *, ok: bool = True, preview: str = 'ok') -> Step:
    return Step.from_payload(
        step_index=idx,
        kind='tool_call',
        payload={
            'tool_name': name,
            'arguments': args,
            'result_preview': preview if ok else f'error:{preview}',
            'result_bytes': len(preview),
            'succeeded': ok,
            'error': None if ok else preview,
        },
    )


def _llm_step(idx: int, content: str = 'thinking') -> Step:
    return Step.from_payload(
        step_index=idx,
        kind='llm_call',
        payload={
            'model': 'fixture-model',
            'messages_in': [Message(USER, f'msg-{idx}').model_dump()],
            'message_out': Message(ASSISTANT, content).model_dump(),
            'prompt_tokens': 5 + idx,
            'completion_tokens': 3,
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs: List[Run] = []

    # 01–05: simple math / single tool
    for i, (a, b) in enumerate([(1, 2), (3, 4), (10, 5), (7, 8), (0, 1)], start=1):
        runs.append(_run_agent(
            f'clean_math_{i:02d}',
            turns=[[_fc('add', {'a': a, 'b': b})], [_answer(f'Sum is {a + b}')]],
            final=f'Sum is {a + b}',
            user=f'What is {a}+{b}?',
            tools=[add],
        ))

    # 06–08: multi-tool different args
    runs.append(_run_agent(
        'clean_multi_01',
        turns=[
            [_fc('add', {'a': 2, 'b': 3})],
            [_fc('multiply', {'a': 5, 'b': 6})],
            [_answer('Product of sum path done')],
        ],
        final='Product of sum path done',
        user='add then multiply',
        tools=[add, multiply],
    ))
    runs.append(_run_agent(
        'clean_multi_02',
        turns=[
            [_fc('lookup', {'key': 'alpha'})],
            [_fc('lookup', {'key': 'beta'})],
            [_fc('lookup', {'key': 'gamma'})],
            [_answer('looked up three keys')],
        ],
        final='looked up three keys',
        user='lookup alpha beta gamma',
        tools=[lookup],
    ))
    runs.append(_run_agent(
        'clean_multi_03',
        turns=[
            [_fc('add', {'a': 1, 'b': 1})],
            [_fc('add', {'a': 2, 'b': 2})],
            [_fc('add', {'a': 3, 'b': 3})],
            [_answer('three different adds')],
        ],
        final='three different adds',
        user='add pairs',
        tools=[add],
    ))

    # 09–11: pagination (same tool, different page) — must NOT be 1.3
    runs.append(_run_agent(
        'clean_paginate_01',
        turns=[
            [_fc('list_items', {'page': 1, 'page_size': 10, 'q': 'pods'})],
            [_fc('list_items', {'page': 2, 'page_size': 10, 'q': 'pods'})],
            [_fc('list_items', {'page': 3, 'page_size': 10, 'q': 'pods'})],
            [_answer('listed three pages')],
        ],
        final='listed three pages',
        user='list all pods pages',
        tools=[list_items],
    ))
    runs.append(_run_agent(
        'clean_paginate_02',
        turns=[
            [_fc('list_items', {'page': 1, 'offset': 0, 'q': 'svc'})],
            [_fc('list_items', {'page': 1, 'offset': 10, 'q': 'svc'})],
            [_answer('offset walk done')],
        ],
        final='offset walk done',
        user='page with offsets',
        tools=[list_items],
    ))
    runs.append(_run_agent(
        'clean_paginate_03',
        turns=[
            [_fc('list_items', {'cursor': 'c0', 'q': 'x'})],
            [_fc('list_items', {'cursor': 'c1', 'q': 'x'})],
            [_fc('list_items', {'cursor': 'c2', 'q': 'x'})],
            [_answer('cursor walk done')],
        ],
        final='cursor walk done',
        user='cursor pagination',
        tools=[list_items],
    ))

    # 12–14: transient retry then success — must NOT be 1.3
    for i, url in enumerate(['https://a.example', 'https://b.example', 'https://c.example'], start=1):
        runs.append(_run_agent(
            f'clean_retry_{i:02d}',
            turns=[
                [_fc('flaky_fetch', {'url': url})],  # fails → soft error string
                [_fc('flaky_fetch', {'url': url})],  # succeeds
                [_answer(f'fetched {url}')],
            ],
            final=f'fetched {url}',
            user=f'fetch {url}',
            tools=[flaky_fetch],
        ))

    # 15–17: identical read-only at distant points (other tool work between)
    runs.append(_run_agent(
        'clean_reread_01',
        turns=[
            [_fc('lookup', {'key': 'config'})],
            [_fc('add', {'a': 1, 'b': 2})],
            [_fc('multiply', {'a': 3, 'b': 4})],
            [_fc('lookup', {'key': 'config'})],  # intentional refresh
            [_answer('refreshed config after work')],
        ],
        final='refreshed config after work',
        user='check config, compute, recheck config',
        tools=[lookup, add, multiply],
    ))
    runs.append(_run_agent(
        'clean_reread_02',
        turns=[
            [_fc('lookup', {'key': 'status'})],
            [_fc('list_items', {'page': 1, 'q': 'jobs'})],
            [_fc('lookup', {'key': 'status'})],
            [_answer('status re-checked')],
        ],
        final='status re-checked',
        user='status then list then status',
        tools=[lookup, list_items],
    ))
    # Handcrafted: two identical reads far apart with LLM-only gaps but only 2 total
    runs.append(_handcrafted(
        'clean_reread_03',
        steps=[
            _llm_step(0, 'call lookup'),
            _tool_step(1, 'lookup', {'key': 'now'}, preview='t0'),
            _llm_step(2, 'other thoughts'),
            _llm_step(3, 'more thoughts'),
            _tool_step(4, 'add', {'a': 9, 'b': 1}, preview='10'),
            _llm_step(5, 'refresh'),
            _tool_step(6, 'lookup', {'key': 'now'}, preview='t1'),
            _llm_step(7, 'done'),
        ],
        final='two spaced lookups ok',
    ))

    # 18–20: plain LLM-only / short happy paths
    for i in range(1, 4):
        runs.append(_run_agent(
            f'clean_chat_{i:02d}',
            turns=[[_answer(f'Hello world {i}')]],
            final=f'Hello world {i}',
            user=f'say hello {i}',
            tools=[],
        ))

    # 21–22: mixed pagination + different tools
    runs.append(_run_agent(
        'clean_mix_01',
        turns=[
            [_fc('list_items', {'page': 1, 'q': 'ns'})],
            [_fc('add', {'a': 1, 'b': 1})],
            [_fc('list_items', {'page': 2, 'q': 'ns'})],
            [_answer('mix pagination and math')],
        ],
        final='mix pagination and math',
        user='list and add',
        tools=[list_items, add],
    ))
    runs.append(_handcrafted(
        'clean_mix_02',
        steps=[
            _llm_step(0),
            _tool_step(1, 'search', {'q': 'alpha', 'request_id': 'r1'}, preview='A'),
            _llm_step(2),
            _tool_step(3, 'search', {'q': 'beta', 'request_id': 'r2'}, preview='B'),
            _llm_step(4),
            _tool_step(5, 'search', {'q': 'gamma', 'nonce': 'n3'}, preview='C'),
            _llm_step(6, 'All distinct queries done'),
        ],
        final='All distinct queries done',
    ))

    # Combined index for CLI batch analysis
    index = OUT_DIR / 'all_clean.jsonl'
    if index.exists():
        index.unlink()
    # Concatenate individual files
    with index.open('w', encoding='utf-8') as out:
        for p in sorted(OUT_DIR.glob('clean_*.jsonl')):
            out.write(p.read_text(encoding='utf-8'))
            if not out.tell() or True:
                pass

    print(f'Wrote {len(runs)} clean traces to {OUT_DIR}')
    for r in runs:
        print(f'  {r.agent_name}: status={r.status} steps={len(r.steps)} final={r.final_output!r:.40}')


if __name__ == '__main__':
    main()
