"""Tool retry, attempt_timeout, and rate limiting with a local GGUF model.

REQUIRES a model download (or an existing Hugging Face hub / ~/models cache).
Do not run this from CI.

Demonstrates:
  - opt-in per-tool retry (flaky network-style tool) — LLM-driven
  - agent-layer attempt_timeout on the async path — LLM-driven
  - shareable RateLimiter under parallel tool calls — deterministic gather
    (small FC models often emit one tool call instead of three; Demo 3 does
    not rely on the model for parallelism)
  - observability events (tool.retry, rate_limit.wait, tool.error)

    PYTHONPATH=. /usr/local/bin/python3.10 examples/tool_resilience/resilience_example.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cat_agent.agents import Assistant
from cat_agent.observability import CallbackHandler
from cat_agent.observability.context import run_context
from cat_agent.tools import tool
from cat_agent.utils.rate_limit import RateLimiter

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

_flaky_calls = 0
_ping_inflight = 0
_ping_max_inflight = 0


@tool(allow_overwrite=True)
async def flaky_lookup(query: str) -> str:
    """Look up a short fact. May fail once with a transient error (demo retry).

    Args:
        query: What to look up
    """
    global _flaky_calls
    _flaky_calls += 1
    # Fail the first attempt so agent-layer retry kicks in.
    if _flaky_calls == 1:
        raise ConnectionError('transient upstream glitch (demo)')
    return f'Lookup[{query}]=ok (attempt {_flaky_calls})'


@tool(allow_overwrite=True)
async def slow_digest(text: str) -> str:
    """Slowly summarize text. Used to exercise attempt_timeout.

    Args:
        text: Text to digest
    """
    await asyncio.sleep(8.0)  # longer than attempt_timeout below
    return f'digest[{text[:40]}]=done'


@tool(allow_overwrite=True)
async def paced_ping(label: str) -> str:
    """Cheap ping used to show the shared rate limiter under parallel calls.

    Args:
        label: Label for this ping
    """
    global _ping_inflight, _ping_max_inflight
    _ping_inflight += 1
    _ping_max_inflight = max(_ping_max_inflight, _ping_inflight)
    try:
        # Long enough that siblings queue behind max_concurrency=1.
        await asyncio.sleep(0.35)
        return f'pong:{label}'
    finally:
        _ping_inflight -= 1


# ---------------------------------------------------------------------------
# LLM + agent
# ---------------------------------------------------------------------------

def main_llm_cfg() -> dict:
    # repo_id/filename: uses HF hub cache (or ~/models/<filename>) before downloading.
    # max_tokens here is the PER-REQUEST output cap — not a run budget.
    return {
        'model_type': 'llama_cpp',
        'repo_id': 'Salesforce/xLAM-2-3b-fc-r-gguf',
        'filename': 'xLAM-2-3B-fc-r-F16.gguf',
        'n_ctx': 4096,
        'n_gpu_layers': -1,
        'n_threads': 6,
        'temperature': 0.6,
        'max_tokens': 1024,
        'verbose': False,
    }


def on_event(event) -> None:
    # Surface resilience-related events; keep the rest quiet.
    if event.event_type in (
        'tool.retry',
        'tool.error',
        'tool.end',
        'rate_limit.wait',
        'run.start',
        'run.end',
    ):
        print(event.summary())


async def main() -> None:
    llm_cfg = main_llm_cfg()
    print(
        'REQUIRES model download unless already cached.\n'
        f'LLM: llama_cpp {llm_cfg["repo_id"]} / {llm_cfg["filename"]}\n'
    )

    # Cap at 1 in-flight paced_ping so 3 parallel calls must queue (visible waits).
    tool_limiter = RateLimiter(max_concurrency=1)
    # Separate limiter for LLM turns — constructed here, not a process global.
    llm_limiter = RateLimiter(requests_per_interval=5, interval_seconds=1.0)
    handler = CallbackHandler(on_event)

    bot = Assistant(
        llm=llm_cfg,
        name='ResilienceBot',
        description='Demos retry, attempt_timeout, and rate limiting.',
        system_message=(
            'You are ResilienceBot. Prefer tools over guessing. '
            'Use flaky_lookup for lookups. Use slow_digest only when asked to digest.'
        ),
        function_list=[
            {
                'name': 'flaky_lookup',
                'retry': {
                    'max_attempts': 3,
                    'initial_delay': 0.2,
                    'retryable_exceptions': ['ConnectionError'],
                },
            },
            {
                'name': 'slow_digest',
                'attempt_timeout': 1.0,  # async wait_for → error observation, not crash
            },
            {
                'name': 'paced_ping',
                'rate_limiter': tool_limiter,
            },
        ],
        rate_limiter=llm_limiter,
        handlers=[handler],
    )

    # --- Demo 1: retry (flaky tool succeeds on attempt 2) ---
    print('=== Demo 1: opt-in retry (flaky_lookup) ===\n')
    global _flaky_calls, _ping_max_inflight
    _flaky_calls = 0
    async with bot:
        result = await bot.arun_nonstream([
            {'role': 'user', 'content': 'Look up the capital of France with flaky_lookup.'},
        ])
        _print_result(result)

        # --- Demo 2: attempt_timeout (slow tool → error observation, not crash) ---
        print('\n=== Demo 2: attempt_timeout (slow_digest times out) ===\n')
        result = await bot.arun_nonstream([
            {'role': 'user', 'content': 'Please slow_digest this text: hello world from cat-agent.'},
        ])
        _print_result(result)

        # --- Demo 3: rate limit under parallel tool calls (deterministic) ---
        # Small FC models often emit one tool call (or tool-call-as-text) instead of
        # three parallel calls. Drive gather directly so the limiter demo is reliable.
        print('\n=== Demo 3: rate limiter (max_concurrency=1, 3 parallel paced_ping) ===')
        print('(deterministic asyncio.gather — not dependent on the model)\n')
        _ping_max_inflight = 0
        labels = ['A', 'B', 'C']
        t0 = time.monotonic()
        with run_context(
            agent_name=bot.name,
            agent_class=type(bot).__name__,
            handlers=[handler],
        ):
            results = await asyncio.gather(*[
                bot._acall_tool('paced_ping', json.dumps({'label': label}))
                for label in labels
            ])
        elapsed = time.monotonic() - t0
        for label, out in zip(labels, results):
            print(f'[tool:paced_ping] {out}')
        print(f'max_inflight={_ping_max_inflight} (expected 1 under the limiter)')
        print(f'elapsed={elapsed:.2f}s (≈ 3 × 0.35s when serialized)')

    print(
        f'\nRateLimiter stats: waits={tool_limiter.stats.waits} '
        f'wait_seconds={tool_limiter.stats.wait_seconds:.3f}'
    )


def _print_result(result) -> None:
    for msg in result or []:
        role = msg.get('role') if isinstance(msg, dict) else msg.role
        name = msg.get('name') if isinstance(msg, dict) else msg.name
        content = msg.get('content') if isinstance(msg, dict) else msg.content
        if isinstance(msg, dict):
            tcs = msg.get('tool_calls')
            fc = None
            if tcs:
                fc = [
                    (tc.get('function') if isinstance(tc, dict) else None)
                    for tc in tcs
                ]
            elif msg.get('function_call'):
                fc = [msg.get('function_call')]
        else:
            fc = (
                [tc.function.model_dump() for tc in msg.tool_calls]
                if getattr(msg, 'tool_calls', None) else None
            )
        if fc:
            print(f'[{role}] tool_call → {fc}')
        elif role == 'function':
            print(f'[tool:{name}] {content}')
        elif content:
            print(f'[{role}] {content}')


if __name__ == '__main__':
    asyncio.run(main())
